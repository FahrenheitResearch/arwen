#!/usr/bin/env python3
"""Fail if an oracle namelist contains a token unknown to WRF nssl_mp_params."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_nssl_namelist.py MODULE.F NAMELIST.INPUT")
    source_path = Path(sys.argv[1])
    namelist_path = Path(sys.argv[2])
    source = source_path.read_text(encoding="ascii", errors="strict")
    start = source.index("NAMELIST /nssl_mp_params/")
    end = source.index("! #####################################################################", start)
    declaration = "\n".join(line.split("!", 1)[0] for line in source[start:end].splitlines())
    allowed = {name.lower() for name in re.findall(r"[A-Za-z][A-Za-z0-9_]*", declaration)}
    allowed.discard("namelist")
    allowed.discard("nssl_mp_params")

    text = namelist_path.read_text(encoding="ascii", errors="strict")
    supplied = {name.lower() for name in re.findall(r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)\s*=", text)}
    unknown = sorted(supplied - allowed)
    if unknown:
        raise SystemExit("ERROR: unknown nssl_mp_params token(s): " + ", ".join(unknown))
    print(f"source={source_path}")
    print(f"namelist={namelist_path}")
    print(f"allowed_token_count={len(allowed)}")
    print(f"supplied_token_count={len(supplied)}")
    print("supplied_tokens=" + ",".join(sorted(supplied)))
    print("unknown_tokens=none")


if __name__ == "__main__":
    main()
