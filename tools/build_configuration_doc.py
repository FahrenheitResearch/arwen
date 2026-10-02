"""Regenerate the configuration reference's per-domain override list.

Usage: python -m tools.build_configuration_doc [--check]

Only the field count and ordered block are generated. The surrounding
configuration instructions remain authored text.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/public/CONFIGURATION.md"
BLOCK = re.compile(
    r"\*\*Which keys a `\[\[domain\]\]` table may override\.\*\* Exactly these \d+,\n"
    r"and no others \(`gpuwm/experiment\.py`'s `_DOMAIN_RUN_OVERRIDES`\):\n\n"
    r"(?:    [^\n]+\n)+")


def render(text: str) -> str:
    from gpuwm.experiment import _DOMAIN_RUN_OVERRIDES

    keys = tuple(_DOMAIN_RUN_OVERRIDES)
    lines = []
    current = "    "
    for key in keys:
        addition = ("  " if current.strip() else "") + key
        if len(current + addition) > 84 and current.strip():
            lines.append(current)
            current = "    " + key
        else:
            current += addition
    if current.strip():
        lines.append(current)
    block = ("**Which keys a `[[domain]]` table may override.** Exactly these "
             f"{len(keys)},\nand no others (`gpuwm/experiment.py`'s "
             "`_DOMAIN_RUN_OVERRIDES`):\n\n" + "\n".join(lines) + "\n")
    if len(BLOCK.findall(text)) != 1:
        raise ValueError("configuration override block is missing or ambiguous")
    return BLOCK.sub(lambda match: block, text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    before = DOC.read_text(encoding="utf-8")
    after = render(before)
    if args.check:
        if before != after:
            print("configuration override order or count differs; regenerate")
            return 1
    else:
        DOC.write_text(after, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
