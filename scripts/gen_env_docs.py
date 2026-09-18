# SPDX-License-Identifier: Apache-2.0
"""Generate docs/ENV_VARS.md from OMLX_* environment-variable usage in omlx/.

#10 (todo/performance-adjustments.md): the OMLX_* knobs are scattered across
the tree with no single index. This is the additive first step -- an
auto-generated inventory plus a drift guard -- before any call-site migration.

Usage::

    python scripts/gen_env_docs.py            # (re)write docs/ENV_VARS.md
    python scripts/gen_env_docs.py --check     # exit 1 if the doc is stale
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "omlx"
OUT = ROOT / "docs" / "ENV_VARS.md"

# Static OMLX_* string literals. Dynamic constructions (f-strings, concat)
# still surface their literal prefix, which is enough for an inventory.
_PAT = re.compile(r'["\'](OMLX_[A-Z0-9_]+)["\']')


def scan() -> dict[str, list[tuple[str, int]]]:
    hits: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for path in sorted(SRC.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = str(path.relative_to(ROOT))
        for lineno, line in enumerate(text.splitlines(), 1):
            for match in _PAT.finditer(line):
                hits[match.group(1)].append((rel, lineno))
    return hits


def render(hits: dict[str, list[tuple[str, int]]]) -> str:
    rows: list[str] = []
    for name in sorted(hits):
        refs = hits[name]
        shown = ", ".join(f"`{f}:{n}`" for f, n in refs[:4])
        if len(refs) > 4:
            shown += f" (+{len(refs) - 4})"
        rows.append(f"| `{name}` | {len(refs)} | {shown} |")
    return (
        "# OMLX_* Environment Variables\n\n"
        f"Auto-generated from `omlx/` by `scripts/gen_env_docs.py` — "
        f"{len(hits)} variables across "
        f"{len({f for refs in hits.values() for f, _ in refs})} files. "
        "Do not edit by hand; run the generator and commit the result.\n\n"
        "| Variable | Occurrences | Locations |\n"
        "|---|---|---|\n" + "\n".join(rows) + "\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if docs/ENV_VARS.md is out of date",
    )
    args = ap.parse_args()

    content = render(scan())
    if args.check:
        existing = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if existing != content:
            print(
                "docs/ENV_VARS.md is stale; run: python scripts/gen_env_docs.py",
                file=sys.stderr,
            )
            return 1
        print("docs/ENV_VARS.md is up to date")
        return 0

    OUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
