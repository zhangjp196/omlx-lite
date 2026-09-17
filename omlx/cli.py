#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for oMLX.

Commands:
    omlx serve --model-dir /path/to/models    Start multi-model server

Usage:
    # Multi-model serving
    omlx serve --model-dir /path/to/models

    # With pinned models
    omlx serve --model-dir /path/to/models --pin llama-3b,qwen-7b
"""

import argparse
import faulthandler
import math
import sys

from ._version import __version__


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than 0")
    return parsed


def _has_cli_overrides(args) -> bool:
    """Check if CLI args contain non-default values that should be saved.

    All argparse defaults are None, so `is not None` means the user
    explicitly passed the flag on the command line.
    """
    persisted_fields = (
        "model_dir",
        "port",
        "host",
        "log_level",
        "sse_keepalive_mode",
        "max_audio_upload_size",
        "max_concurrent_requests",
        "embedding_batch_size",
        "memory_guard",
        "memory_guard_gb",
        "paged_ssd_cache_dir",
        "paged_ssd_cache_max_size",
        "hot_cache_max_size",
        "hot_cache_write_through",
        "initial_cache_blocks",
        "hf_endpoint",
        "hf_cache_enabled",
        "ms_endpoint",
    )
    if any(getattr(args, field, None) is not None for field in persisted_fields):
        return True

    # --no-cache is the only persistable boolean flag with a False default.
    return bool(getattr(args, "no_cache", False))


def serve_command(args):
    """Start the OpenAI-compatible multi-model server."""
    import logging
    import os
    import uvicorn

    from ._version import __version__
    from . import process_title
    from .settings import burst_decode_env, init_settings
    from .logging_config import configure_file_logging, AdminStatsAccessFilter

    process_title.set_process_title()

    try:
        from ._build_info import build_number
    except ImportError:
        build_number = None

    # Print version banner
    print(f"\033[33moMLX Lite - LLM inference, optimized for your Mac\033[0m")
    print(f"\033[33m├─ https://github.com/jundot/omlx\033[0m")
    if build_number:
        print(f"\033[33m├─ Version: {__version__}\033[0m")
        print(f"\033[33m└─ Build: {build_number}\033[0m")
    else:
        print(f"\033[33m└─ Version: {__version__}\033[0m")
    print()

    # Initialize global settings first (to get log_level from file if not specified)
    settings = init_settings(base_path=args.base_path, cli_args=args)

    # The native ANE compile-cache gate reads this env var once, at the first
    # compile, so it must be exported before any engine loads. setdefault
    # keeps an explicit env override authoritative.
    if settings.cache.ane_compile_cache:
        os.environ.setdefault("OMLX_QWEN35_ANE_COMPILE_CACHE", "1")

    # Register TRACE level (5) — includes full message content
    TRACE = 5
    logging.addLevelName(TRACE, "TRACE")

    # Configure logging (use settings value which has proper priority)
    level_name = settings.server.log_level.upper()
    log_level = (
        TRACE if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Set omlx loggers
    for name in [
        "omlx",
        "omlx.scheduler",
        "omlx.paged_ssd_cache",
        "omlx.memory_monitor",
        "omlx.paged_cache",
        "omlx.prefix_cache",
        "omlx.engine_pool",
        "omlx.model_discovery",
    ]:
        logging.getLogger(name).setLevel(log_level)

    # Suppress repetitive admin stats access logs
    logging.getLogger("uvicorn.access").addFilter(AdminStatsAccessFilter())

    # Suppress noisy third-party loggers unless trace level
    if log_level > TRACE:
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)

    # Ensure required directories exist
    settings.ensure_directories()

    # Apply HuggingFace endpoint if configured
    if settings.huggingface.endpoint:
        os.environ["HF_ENDPOINT"] = settings.huggingface.endpoint

    # Apply ModelScope endpoint if configured
    if settings.modelscope.endpoint:
        os.environ["MODELSCOPE_DOMAIN"] = settings.modelscope.endpoint

    # Seed Burst Decode env vars so EngineConfig picks up the saved mode at
    # engine construction (no restart needed when the mode changes later).
    for _key, _value in burst_decode_env(settings.server.burst_decode_mode).items():
        os.environ[_key] = _value

    # Validate before persisting CLI overrides, so invalid flags never poison
    # settings.json.
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        sys.exit(1)

    # Save CLI args to settings.json if non-default values provided
    if _has_cli_overrides(args):
        try:
            settings.save_cli_overrides(args)
            print("Saved CLI arguments to settings.json")
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    # Configure file logging (writes to {base_path}/logs/server.log)
    log_dir = settings.logging.get_log_dir(settings.base_path)
    configure_file_logging(
        log_dir=log_dir,
        level=settings.server.log_level,
        include_request_id=True,
        retention_days=settings.logging.retention_days,
    )
    print(f"Log directory: {log_dir}")

    # Enable native crash diagnostics (SIGABRT, SIGSEGV, SIGFPE, SIGBUS).
    # On Metal/MLX crashes (#511, #520), this dumps all Python thread
    # tracebacks to the server log before the process terminates.
    crash_log_path = log_dir / "crash.log"
    _crash_file = open(crash_log_path, "a")
    faulthandler.enable(file=_crash_file, all_threads=True)

    # Bind the socket before importing/initializing the server. Uvicorn's
    # normal startup runs ASGI lifespan before binding host/port, which means
    # pinned models can be preloaded before a port conflict is detected.
    bind_hosts = [h.strip() for h in settings.server.host.split(",") if h.strip()]
    for h in bind_hosts:
        print(f"Binding server at http://{h}:{settings.server.port}")
    # uvicorn does not support "trace" — map to "debug" for its internal logging
    uvicorn_level = (
        "debug" if settings.server.log_level == "trace" else settings.server.log_level
    )
    # Only show access logs at trace level
    show_access_log = settings.server.log_level == "trace"
    uvicorn_config = uvicorn.Config(
        "omlx.server:app",
        host=bind_hosts[0],
        port=settings.server.port,
        log_level=uvicorn_level,
        access_log=show_access_log,
    )
    # Bind a socket per host so an occupied port fails fast before model preload.
    # uvicorn.Server.run(sockets=[...]) accepts a list and listens on all of them.
    serve_sockets = [uvicorn_config.bind_socket()]
    for h in bind_hosts[1:]:
        extra_cfg = uvicorn.Config(
            "omlx.server:app",
            host=h,
            port=settings.server.port,
            log_level=uvicorn_level,
            access_log=show_access_log,
        )
        serve_sockets.append(extra_cfg.bind_socket())

    try:
        # Import server and config after the port is known to be available.
        from .server import init_server
        from .config import parse_size

        model_dirs = settings.get_effective_model_dirs()
        print(f"Base path: {settings.base_path}")
        print(f"Model directories: {', '.join(str(d) for d in model_dirs)}")
        # State first: a bare tier line reads as "this is enforced" even when
        # the guard is off, and with it off the tier governs nothing.
        if settings.memory.prefill_memory_guard:
            print(f"Memory guard: on (tier: {settings.memory.memory_guard_tier})")
        else:
            print("Memory guard: off")

        # Determine paged SSD cache directory
        # Priority: --no-cache > CLI arg > settings file
        if args.no_cache:
            paged_ssd_cache_dir = None
        elif args.paged_ssd_cache_dir:
            # CLI argument takes precedence
            paged_ssd_cache_dir = args.paged_ssd_cache_dir
        elif settings.cache.enabled:
            # Use settings file value (resolved path or default)
            paged_ssd_cache_dir = str(
                settings.cache.get_ssd_cache_dir(settings.base_path)
            )
        else:
            # Cache explicitly disabled in settings
            paged_ssd_cache_dir = None

        # Build scheduler config for BatchedEngine
        scheduler_config = settings.to_scheduler_config()
        # Set paged SSD cache options
        scheduler_config.paged_ssd_cache_dir = paged_ssd_cache_dir
        # Determine cache max size: CLI arg > settings (with auto resolution)
        if paged_ssd_cache_dir:
            if args.paged_ssd_cache_max_size:
                # CLI argument specified explicitly
                cache_max_size_bytes = parse_size(args.paged_ssd_cache_max_size)
            else:
                # Use settings value (handles "auto" -> 10% of SSD capacity)
                cache_max_size_bytes = settings.cache.get_ssd_cache_max_size_bytes(
                    settings.base_path
                )
            scheduler_config.paged_ssd_cache_max_size = cache_max_size_bytes
        else:
            scheduler_config.paged_ssd_cache_max_size = 0
            cache_max_size_bytes = 0

        # Hot cache: CLI arg > settings
        if paged_ssd_cache_dir:
            if args.hot_cache_max_size:
                hot_cache_max_bytes = parse_size(args.hot_cache_max_size)
            else:
                hot_cache_max_bytes = settings.cache.get_hot_cache_max_size_bytes()
            scheduler_config.hot_cache_max_size = hot_cache_max_bytes
        else:
            scheduler_config.hot_cache_max_size = 0

        # Write-through: explicit CLI flag > settings file (already mapped by
        # settings.to_scheduler_config()).
        if getattr(args, "hot_cache_write_through", None) is not None:
            scheduler_config.hot_cache_write_through = bool(
                args.hot_cache_write_through
            )

        if args.no_cache:
            print(
                "Mode: Multi-model serving (no oMLX Lite cache, mlx-lm BatchGenerator only)"
            )
        elif paged_ssd_cache_dir:
            print("Mode: Multi-model serving (continuous batching + paged SSD cache)")
            # Format cache size for display
            cache_max_size_display = f"{cache_max_size_bytes / (1024**3):.1f}GB"
            print(
                f"paged SSD cache: {paged_ssd_cache_dir} (max: {cache_max_size_display})"
            )
            if scheduler_config.hot_cache_max_size > 0:
                hot_display = f"{scheduler_config.hot_cache_max_size / (1024**3):.1f}GB"
                print(f"Hot cache: {hot_display} (in-memory)")
        else:
            print("Mode: Multi-model serving (continuous batching, no cache)")

        # Set MLX buffer cache limit high to prevent the allocator from
        # immediately releasing Metal buffers when the cache is full.
        # Without this, allocator::free() can call buf->release() while the
        # GPU is still using the buffer, causing kernel panics on M4.
        # With a large cache limit, freed buffers always stay in the pool
        # and are only released via mx.clear_cache() (which we protect
        # with mx.synchronize()). See issue #300.
        import mlx.core as mx

        total_mem = mx.device_info().get("memory_size", 0)
        if total_mem > 0:
            mx.set_cache_limit(total_mem)

        # Initialize server
        # Note: pinned_models and default_model are managed via admin page (model_settings.json)
        # Sampling parameters (max_tokens, temperature, etc.) are per-model settings
        init_server(
            model_dirs=[str(d) for d in model_dirs],
            scheduler_config=scheduler_config,
            api_key=settings.auth.api_key,
            global_settings=settings,
        )

        for h in bind_hosts:
            print(f"Starting server at http://{h}:{settings.server.port}")
        try:
            uvicorn.Server(uvicorn_config).run(sockets=serve_sockets)
        except KeyboardInterrupt:
            pass
    finally:
        # Uvicorn closes sockets during normal shutdown; this covers failures
        # after bind succeeds but before the server takes ownership.
        for sock in serve_sockets:
            sock.close()


def _app_control_socket_path():
    from pathlib import Path

    return Path.home() / "Library" / "Application Support" / "oMLX" / "control.sock"


def _app_bundle_path():
    from pathlib import Path

    from .utils.install import get_app_bundle_cli_path

    cli_path = get_app_bundle_cli_path()
    try:
        return cli_path.parents[2]
    except IndexError:
        return Path("/Applications/oMLX Lite.app")


def _open_macos_app() -> None:
    import subprocess

    app_path = _app_bundle_path()
    subprocess.run(
        ["/usr/bin/open", "-gj", str(app_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _send_app_control(command: str, timeout: float = 2.0) -> dict:
    import json
    import socket

    sock_path = _app_control_socket_path()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(sock_path))
        sock.sendall(json.dumps({"command": command}).encode("utf-8") + b"\n")
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    raw = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(raw.decode("utf-8"))


def _send_app_control_with_launch(command: str, timeout: float) -> dict:
    import time

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    _open_macos_app()
    while time.monotonic() < deadline:
        try:
            return _send_app_control(command)
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Could not reach oMLX Lite.app control socket: {last_error}")


def _wait_app_control_state(states: set[str], timeout: float) -> dict:
    import time

    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = _send_app_control("status")
        if last.get("state") in states:
            return last
        time.sleep(0.5)
    return last


def _run_brew_services(command: str) -> int:
    import shutil
    import subprocess

    brew = shutil.which("brew")
    if not brew:
        print("Homebrew is not available on PATH.")
        return 1
    result = subprocess.run([brew, "services", command, "omlx"])
    return result.returncode


def lifecycle_command(args) -> int:
    """Run background lifecycle commands for the current installation."""
    from .utils.install import is_app_bundle, is_homebrew

    command = args.command
    timeout = getattr(args, "timeout", 60.0)
    no_wait = getattr(args, "no_wait", False)

    if is_app_bundle():
        try:
            if command == "stop":
                try:
                    response = _send_app_control(command)
                except OSError:
                    print("oMLX Lite stopped")
                    return 0
            else:
                response = _send_app_control_with_launch(command, timeout=timeout)
            if not response.get("ok"):
                print(response.get("message") or f"oMLX Lite {command} failed")
                return 1

            if command in {"start", "restart"} and not no_wait:
                response = _wait_app_control_state({"running", "unresponsive"}, timeout)
                if response.get("state") not in {"running", "unresponsive"}:
                    print(
                        f"oMLX Lite server is {response.get('state', 'unknown')} "
                        f"after {int(timeout)}s."
                    )
                    return 1

            if command == "stop":
                print("oMLX Lite stopped")
            elif command == "start":
                print(
                    f"oMLX Lite server {response.get('state')} on port {response.get('port')}"
                )
            elif command == "restart":
                print(f"oMLX Lite server restarted on port {response.get('port')}")
            return 0
        except Exception as exc:
            print(f"Failed to control oMLX Lite.app: {exc}")
            return 1

    if is_homebrew():
        mapping = {"start": "start", "stop": "stop", "restart": "restart"}
        return _run_brew_services(mapping[command])

    if command == "start":
        print("Background start is available for the macOS app and Homebrew installs.")
        print("For this install, run foreground server mode with: omlx serve")
    else:
        print("Background stop/restart requires the macOS app or Homebrew service.")
    return 1


def diagnose_menubar() -> int:
    """Diagnose why the oMLX menubar icon might be missing.

    Reports macOS version, app install path, running menubar process, and the
    most recent visibility warning from the log. Prints manual recovery steps
    since Tahoe's ControlCenter doesn't expose a public API to re-enable a
    hidden status item.
    """
    import platform
    import subprocess
    from pathlib import Path

    print("oMLX Lite menubar diagnostics")
    print("=" * 40)

    mac_ver = platform.mac_ver()[0] or "unknown"
    print(f"macOS:          {mac_ver}")
    print(f"Bundle ID:      app.omlx")

    app_path = Path("/Applications/oMLX Lite.app")
    print(f"App installed:  {'yes' if app_path.exists() else 'NO (install DMG first)'}")

    try:
        res = subprocess.run(
            ["pgrep", "-af", "oMLX"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        running = bool(res.stdout.strip())
        print(f"Menubar app:    {'running' if running else 'NOT running'}")
        if running:
            first_line = res.stdout.strip().splitlines()[0]
            pid = first_line.split()[0] if first_line else "?"
            print(f"PID:            {pid}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Menubar app:    check failed ({e})")

    # `menubar.log` is the Swift app's own visibility-probe log — every line
    # in it is relevant. `server.log` is the Python child's stdout/stderr, so
    # only lines that mention the menubar are worth pulling out of it.
    log_dir = Path.home() / "Library" / "Application Support" / "oMLX" / "logs"
    log_candidates = [(log_dir / "menubar.log", False), (log_dir / "server.log", True)]
    print(f"Log dir:        {log_dir}")

    hits: list[tuple[str, str]] = []
    for path, needs_filter in log_candidates:
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 131072))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError as e:
            print(f"Could not read {path.name}: {e}")
            continue
        for ln in tail.splitlines():
            if not ln.strip():
                continue
            if needs_filter and not (
                "menubar visibility probe" in ln
                or "NSStatusItem" in ln
                or "ControlCenter" in ln
                or "Menu Bar" in ln
            ):
                continue
            hits.append((path.name, ln))

    if hits:
        print("\nRecent visibility log entries (last 10):")
        for src, ln in hits[-10:]:
            print(f"  [{src}] {ln}")
    else:
        print("\nNo visibility log entries found (app may not have probed yet).")

    print()
    print("If the icon is missing on macOS Tahoe (26.x):")
    print("  1. In the oMLX Lite app: Settings > Appearance > Menu Bar Icon > Restore")
    print("  2. Or turn it back on in System Settings > Menu Bar")
    print(
        "     open 'x-apple.systempreferences:com.apple.ControlCenter-Settings.extension?MenuBar'"
    )
    print("  3. If oMLX Lite isn't in the list, quit the app and relaunch oMLX Lite.app")
    print()
    print("Note: Restore edits ControlCenter's own StatusKit approval, which")
    print("needs Full Disk Access. Without it, use the System Settings toggle.")
    return 0


def diagnose_command(args) -> int:
    """Dispatch 'omlx diagnose <target>' to the appropriate subcommand."""
    target = getattr(args, "target", None)
    if target == "menubar":
        return diagnose_menubar()
    print(f"Unknown diagnose target: {target}")
    print("Available: menubar")
    return 1


def cluster_command(args) -> int:
    """Run cluster diagnostics, collective checks, and shard planning."""
    import json

    action = getattr(args, "cluster_action", None)
    if action == "status":
        from .cluster.probe import collect_cluster_status, format_cluster_status

        try:
            status = collect_cluster_status(route_to=args.route_to)
        except ValueError as exc:
            print(f"Cluster status error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_cluster_status(status))
        return 0

    if action == "worker-smoke":
        from .cluster.supervisor import run_worker_smoke

        try:
            result = run_worker_smoke(timeout=args.timeout)
        except (OSError, RuntimeError, TimeoutError) as exc:
            print(f"Cluster worker smoke failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("oMLX Lite cluster worker smoke passed")
            print(f"Worker PID:  {result['worker_pid']}")
            print(f"Protocol:    {result['protocol_version']}")
            print(f"Round trip:  {result['elapsed_seconds']:.3f}s")
        return 0

    if action == "collective-smoke":
        from .cluster.collective import (
            CollectiveSmokeError,
            run_local_collective_smoke,
        )

        try:
            result = run_local_collective_smoke(timeout=args.timeout)
        except (CollectiveSmokeError, OSError, RuntimeError, ValueError) as exc:
            print(f"Cluster collective smoke failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("oMLX Lite local MLX collective smoke passed")
            print(f"Backend:     {result['backend']} (loopback only)")
            print(f"Ranks:       {result['rank_count']}")
            print(f"All-sum:     {result['expected_sum']}")
            print(f"MLX:         {result['mlx_version']}")
            print(f"Elapsed:     {result['elapsed_seconds']:.3f}s")
        return 0

    if action == "pipeline-smoke":
        from .cluster.collective import (
            CollectiveSmokeError,
            run_local_pipeline_smoke,
        )

        try:
            result = run_local_pipeline_smoke(timeout=args.timeout)
        except (CollectiveSmokeError, OSError, RuntimeError, ValueError) as exc:
            print(f"Cluster pipeline smoke failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("oMLX Lite unequal Nemotron-H pipeline smoke passed")
            print(f"Backend:     {result['backend']} (loopback only)")
            print(f"Ranks:       {result['rank_count']}")
            print(f"Checksum:    {result['ranks'][0]['checksum']}")
            print(f"Elapsed:     {result['elapsed_seconds']:.3f}s")
        return 0

    if action == "plan":
        import socket

        from .cluster.planner import (
            NodeBudget,
            PlanningError,
            format_shard_plan,
            locate_model_layout,
            plan_unequal_pipeline,
            synthetic_model_layout,
        )
        from .config import parse_size
        from .utils import hardware

        def parse_cluster_size(value: str) -> int:
            normalized = (
                value.strip()
                .upper()
                .replace("KIB", "KB")
                .replace("MIB", "MB")
                .replace("GIB", "GB")
                .replace("TIB", "TB")
            )
            size = parse_size(normalized)
            if size < 0:
                raise ValueError("sizes must be non-negative")
            return size

        try:
            reserve_bytes = parse_cluster_size(args.reserve)
            nodes = []
            for rank, definition in enumerate(args.node or []):
                node_id, separator, raw_size = definition.rpartition("=")
                if not separator or not node_id.strip() or not raw_size.strip():
                    raise ValueError("--node must use NAME=SIZE (for example studio=256GB)")
                nodes.append(
                    NodeBudget(
                        node_id=node_id.strip(),
                        capacity_bytes=parse_cluster_size(raw_size),
                        reserve_bytes=reserve_bytes,
                        rank=rank,
                    )
                )
            if not nodes:
                detected = hardware.detect_hardware()
                nodes.append(
                    NodeBudget(
                        node_id=socket.gethostname(),
                        capacity_bytes=detected.max_working_set_bytes,
                        reserve_bytes=reserve_bytes,
                        rank=0,
                    )
                )

            holder = None
            if args.model:
                # The Mac being planned for is often the one holding a single
                # stage, so ask each peer rather than assume this node can
                # read the whole model.
                holder = locate_model_layout(args.model, args.peer or [])
                model = holder.layout
            else:
                model = synthetic_model_layout(
                    total_weight_bytes=parse_cluster_size(args.model_size),
                    layer_count=args.layers,
                )
            plan = plan_unequal_pipeline(model, nodes)
        except (OSError, PlanningError, ValueError) as exc:
            print(f"Cluster planning failed: {exc}", file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_shard_plan(plan))
            if holder is not None and not holder.is_local:
                print(f"Measured:    {holder.node} (the node holding the model)")
        return 0

    print(
        "Unknown cluster action. Available: status, worker-smoke, "
        "collective-smoke, pipeline-smoke, plan",
        file=sys.stderr,
    )
    return 2


def main():
    parser = argparse.ArgumentParser(
        description="omlx: Production-ready LLM server for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  omlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
        """,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the oMLX Lite version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    for name, help_text in (
        ("start", "Start oMLX Lite as a managed background server"),
        ("stop", "Stop the managed background oMLX Lite server"),
        ("restart", "Restart the managed background oMLX Lite server"),
    ):
        lifecycle_parser = subparsers.add_parser(
            name,
            help=help_text,
            description=help_text,
        )
        lifecycle_parser.add_argument(
            "--timeout",
            type=float,
            default=60.0,
            help="Seconds to wait for the macOS app/server to reach the requested state",
        )
        if name in {"start", "restart"}:
            lifecycle_parser.add_argument(
                "--no-wait",
                action="store_true",
                help="Return after sending the request without waiting for server health",
            )

    # Serve command (multi-model)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start multi-model OpenAI-compatible server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Start a multi-model inference server with LRU-based memory management.

Models are discovered from subdirectories of --model-dir. Each subdirectory
should contain a valid model with config.json and *.safetensors files.

Example directory structure:
  /path/to/models/
  ├── llama-3b/           → model_id: "llama-3b"
  │   ├── config.json
  │   └── model.safetensors
  ├── qwen-7b/            → model_id: "qwen-7b"
  └── mistral-7b/         → model_id: "mistral-7b"
""",
    )

    # Required arguments
    serve_parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing model subdirectories (default: ~/.omlx/models)",
    )
    # Server options
    serve_parser.add_argument(
        "--host", type=str, default=None, help="Host to bind (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=None, help="Port to bind (default: 8000)"
    )
    serve_parser.add_argument(
        "--log-level",
        type=str,
        choices=["trace", "debug", "info", "warning", "error"],
        default=None,
        help="Log level (default: info). trace includes full message content",
    )
    serve_parser.add_argument(
        "--sse-keepalive-mode",
        type=str,
        choices=["chunk", "comment", "off"],
        default=None,
        help="SSE keepalive emission mode (default: chunk). 'chunk' emits "
        "protocol-aware no-op events compatible with strict clients like "
        "OpenClaw / WorkBuddy; 'comment' emits the legacy ': keep-alive' SSE "
        "comment; 'off' disables keepalive entirely",
    )
    serve_parser.add_argument(
        "--max-audio-upload-size",
        type=str,
        default=None,
        help="Maximum audio upload size for /v1/audio/transcriptions and "
        "/v1/audio/process (e.g. '100MB', '500MB'). Overrides the value "
        "in settings.json (built-in default: 100MB). Uploads are buffered "
        "in memory, so this is also a per-request RAM cap",
    )

    # Scheduler options (for BatchedEngine)
    serve_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="Max requests processed simultaneously. Higher values increase throughput but use more memory. (default: 8)",
    )
    serve_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Max embedding inputs processed in one forward pass. Higher values increase throughput but use more memory. (default: 32)",
    )

    # Memory guard options
    serve_parser.add_argument(
        "--memory-guard",
        type=str,
        choices=["off", "safe", "balanced", "aggressive"],
        default=None,
        help="Memory guard tier, or 'off' to disable the guard. safe reserves more system memory; aggressive allows more oMLX Lite memory use. Passing a tier also turns the guard on. (default: balanced)",
    )
    serve_parser.add_argument(
        "--memory-guard-gb",
        type=_positive_float,
        default=None,
        help="Custom memory guard ceiling in GB. Sets memory guard tier to custom and turns the guard on.",
    )

    # paged SSD cache options
    serve_parser.add_argument(
        "--paged-ssd-cache-dir",
        type=str,
        default=None,
        help="Directory for paged SSD cache storage (enables oMLX Lite prefix cache)",
    )
    serve_parser.add_argument(
        "--paged-ssd-cache-max-size",
        type=str,
        default=None,
        help="Maximum paged SSD cache size (e.g., '100GB', '50GB'). Default: 100GB",
    )
    serve_parser.add_argument(
        "--hot-cache-max-size",
        type=str,
        default=None,
        help="Maximum in-memory hot cache size (e.g., '8GB', '4GB'). Default: 0 (disabled)",
    )
    serve_parser.add_argument(
        "--hot-cache-write-through",
        action="store_true",
        default=None,
        help="Persist every hot-cache block to SSD immediately (write-through). "
        "Keeps RAM-speed resume while retaining SSD durability for all sessions.",
    )
    serve_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable oMLX Lite paged SSD cache. mlx-lm BatchGenerator still manages KV states internally.",
    )
    serve_parser.add_argument(
        "--initial-cache-blocks",
        type=int,
        default=None,
        help="Number of cache blocks to pre-allocate at startup (default: 256). "
        "Higher values reduce dynamic allocation overhead for large contexts.",
    )

    # HuggingFace options
    serve_parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help="Custom HuggingFace Hub endpoint URL (e.g., https://hf-mirror.com)",
    )
    serve_parser.add_argument(
        "--hf-cache",
        dest="hf_cache_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Discover models from the standard HuggingFace Hub local cache (default: enabled)",
    )

    # ModelScope options
    serve_parser.add_argument(
        "--ms-endpoint",
        type=str,
        default=None,
        help="Custom ModelScope Hub endpoint URL",
    )

    # Base path and auth
    serve_parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base directory for oMLX Lite data (default: ~/.omlx)",
    )
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (required for non-loopback binds)",
    )

    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Diagnose installation or runtime issues",
        description="Run diagnostic checks and print recovery steps.",
    )
    diagnose_parser.add_argument(
        "target",
        type=str,
        choices=["menubar"],
        help="What to diagnose. 'menubar' checks Tahoe ControlCenter visibility.",
    )

    # Cluster diagnostics and planning for the first implementation slice.
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Inspect distributed-node readiness and exercise a local worker",
        description=(
            "Distributed-cluster diagnostics and unequal-memory planning. "
            "This command does not configure interfaces or initialize JACCL."
        ),
    )
    cluster_subparsers = cluster_parser.add_subparsers(
        dest="cluster_action",
        required=True,
        help="Cluster diagnostic command",
    )
    cluster_status_parser = cluster_subparsers.add_parser(
        "status",
        help="Report local memory, runtime, RDMA, and Thunderbolt readiness",
    )
    cluster_status_parser.add_argument(
        "--route-to",
        metavar="IP",
        default=None,
        help="Also inspect the active route to an IPv4 or IPv6 peer address",
    )
    cluster_status_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_smoke_parser = cluster_subparsers.add_parser(
        "worker-smoke",
        help="Run a real isolated worker ready/ping/shutdown round trip",
    )
    cluster_smoke_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=5.0,
        help="Per-operation worker deadline in seconds (default: 5)",
    )
    cluster_smoke_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_collective_parser = cluster_subparsers.add_parser(
        "collective-smoke",
        help="Run two local MLX ranks and verify a ring all-sum",
    )
    cluster_collective_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=20.0,
        help="Overall collective deadline in seconds (default: 20)",
    )
    cluster_collective_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_pipeline_parser = cluster_subparsers.add_parser(
        "pipeline-smoke",
        help="Run an unequal two-rank hybrid Nemotron-H graph",
    )
    cluster_pipeline_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=30.0,
        help="Overall pipeline deadline in seconds (default: 30)",
    )
    cluster_pipeline_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_plan_parser = cluster_subparsers.add_parser(
        "plan",
        help="Plan contiguous layers across unequal node memory budgets",
    )
    cluster_plan_source = cluster_plan_parser.add_mutually_exclusive_group(
        required=True
    )
    cluster_plan_source.add_argument(
        "--model",
        metavar="PATH",
        help="Inspect safetensors headers from a downloaded model directory",
    )
    cluster_plan_source.add_argument(
        "--model-size",
        metavar="SIZE",
        help="Plan an estimated model before download (for example 300GB)",
    )
    cluster_plan_parser.add_argument(
        "--layers",
        type=_positive_int,
        default=80,
        help="Layer count used with --model-size (default: 80)",
    )
    cluster_plan_parser.add_argument(
        "--node",
        action="append",
        metavar="NAME=SIZE",
        help=(
            "Node memory budget in rank order; repeat for each node. "
            "Defaults to this Mac's recommended working set."
        ),
    )
    cluster_plan_parser.add_argument(
        "--reserve",
        default="0",
        metavar="SIZE",
        help="Memory to reserve on every node for KV/activations (default: 0)",
    )
    cluster_plan_parser.add_argument(
        "--peer",
        action="append",
        metavar="SSH_HOST",
        help=(
            "SSH host that may hold the model; repeat for each. Used with "
            "--model when this Mac holds only its own stage: the first peer "
            "that can read a complete model is the one that measures it."
        ),
    )
    cluster_plan_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )

    args, extra_args = parser.parse_known_args(sys.argv[1:])

    if extra_args:
        parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
    if args.command == "serve":
        if (
            getattr(args, "memory_guard", None) == "off"
            and getattr(args, "memory_guard_gb", None) is not None
        ):
            parser.error(
                "--memory-guard off cannot be combined with "
                "--memory-guard-gb (a custom ceiling needs the guard on)"
            )
        serve_command(args)
    elif args.command in {"start", "stop", "restart"}:
        sys.exit(lifecycle_command(args))
    elif args.command == "diagnose":
        sys.exit(diagnose_command(args))
    elif args.command == "cluster":
        sys.exit(cluster_command(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
