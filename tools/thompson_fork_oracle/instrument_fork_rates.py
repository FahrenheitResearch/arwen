#!/usr/bin/env python3
"""Write an INSTRUMENTED copy of the HRRR v4.1.21 fork's
module_mp_thompson.F that publishes every process rate and the running
tendencies of mp_thompson at five points of the call (cp1..cp4, cpx), as raw
float64 streams.

The instrumentation is tools/thompson_real_column_parity/
instrument_wrf_rates.py's (WRITE/OPEN statements and declarations only,
anchored by exact text that must occur once); the fork's anchor text is the
same.  Only the field lists differ: the fork has no graupel number, so the
four png_* rates, ngten and ng1d are dropped.

USAGE
    python3 instrument_fork_rates.py PRISTINE.F OUTPUT.F [SCHEMA.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "thompson_real_column_parity"))

import instrument_wrf_rates as I  # noqa: E402

RATES = tuple(r for r in I.RATES if not r.startswith("png_"))
LATE_RATES = I.LATE_RATES
TENDENCIES = tuple(t for t in I.TENDENCIES if t != "ngten")
EXIT_FIELDS = tuple(e for e in I.EXIT_FIELDS if e != "ng1d")
SCHEMA = {
    "cp1": ["ii", "k", *RATES, "rho", *TENDENCIES, *EXIT_FIELDS],
    "cp2": ["ii", "k", *LATE_RATES, "rho", "temp", "qv", "ssatw",
            *TENDENCIES],
    "cp3": ["ii", "k", *TENDENCIES, "pptrain", "pptsnow", "pptgraul",
            "pptice"],
    "cp4": ["ii", "k", *TENDENCIES],
    "cpx": ["ii", "k", *EXIT_FIELDS],
}


def instrument(text: str) -> str:
    I.RATES = RATES
    I.SCHEMA = SCHEMA
    return I.instrument(text)


def main(argv):
    if len(argv) not in (3, 4):
        print(__doc__)
        return 2
    src = Path(argv[1]).read_text(encoding="latin-1")
    Path(argv[2]).write_text(instrument(src), encoding="latin-1",
                             newline="\n")
    if len(argv) == 4:
        Path(argv[3]).write_text(json.dumps(SCHEMA, indent=1) + "\n",
                                 encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
