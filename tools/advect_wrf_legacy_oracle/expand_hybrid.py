#!/usr/bin/env python3
"""Expand the HRRR fork's variadic hybrid-mass macros in module_advect_em.F.

NOAA-EMC/HRRR 40ee6058c (WRFV3.9) guards its hybrid vertical coordinate
behind ``#if ( HYBRID_COORD==1 )`` with variadic function-like macros
(``mut(...)`` -> ``(c1(k)*mut(...)+c2(k))`` and so on) that WRF's own
build expands with a C preprocessor.  gfortran's Fortran preprocessor runs
in traditional mode and rejects ``...`` parameter lists, so this tool does
the same textual expansion the macros define, on executable text only
(a ``!`` comment tail is left alone), and writes the result beside a
receipt that pins the input, the output and every expansion count.  The
``#if ( HYBRID_COORD==1 )`` block then compiles away (HYBRID_COORD is not
defined to the compiler), as it would after the macros had been consumed.

Exactly the five macro definitions of the source head are applied:
  mub(...)    -> (c1(k)*mub(...)+c2(k))
  muu(...)    -> (c1(k)*muu(...)+c2(k))
  muv(...)    -> (c1(k)*muv(...)+c2(k))
  mut(...)    -> (c1(k)*mut(...)+c2(k))
  mu_old(...) -> (c1(k)*mu_old(...))
No other text changes; tools/ieva_wrf_oracle/legacy_build.py applies the
same expansion to the implicit routines it extracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MACROS = (
    ("mub", "(c1(k)*mub({args})+c2(k))"),
    ("muu", "(c1(k)*muu({args})+c2(k))"),
    ("muv", "(c1(k)*muv({args})+c2(k))"),
    ("mut", "(c1(k)*mut({args})+c2(k))"),
    ("mu_old", "(c1(k)*mu_old({args}))"),
)


def expand(source: str) -> tuple[str, dict[str, int]]:
    counts = {name: 0 for name, _ in MACROS}
    out = []
    for line in source.splitlines(keepends=True):
        if line.lstrip().startswith("#"):
            out.append(line)
            continue
        code, bang, comment = line.partition("!")
        for name, form in MACROS:
            pattern = re.compile(r"\b" + name + r"\(([^()]*)\)")

            def sub(match, form=form, name=name):
                counts[name] += 1
                return form.format(args=match.group(1))
            code = pattern.sub(sub, code)
        out.append(code + bang + comment)
    return "".join(out), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")
    expanded, counts = expand(text)
    args.output.write_text(expanded, encoding="utf-8", newline="\n")
    receipt = {
        "source": args.source.name,
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output": args.output.name,
        "output_sha256": hashlib.sha256(expanded.encode("utf-8")).hexdigest(),
        "expansions": counts,
        "macros": {name: form for name, form in MACROS},
    }
    if args.receipt is not None:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="ascii")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
