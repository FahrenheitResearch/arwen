"""Freeze the words the WSM7 column oracle dumped into the engine's table.

usage: python -m tools.wsm7_wrf461_column_oracle.freeze_words \
           ORACLE_DIR/wsm7-init-words.txt gpuwm/data/wsm_family/wsm7-wrf461-words.json

``dump_wsm7_init`` (build.sh) prints every float32 word wsm7init leaves in
module_mp_wsm7 and every radar_init word refl10cm_wsm7 reads, as hex bit
patterns.  This writes them, unchanged, as the JSON the WSM-family runner
(gpuwm/core/wsm_family.py) binds to the kernel.  The words are WRF's own
float32 results, rgmma Gamma products included -- never recomputed on the
host -- and tests/test_wsm7_wrf461_column_exact.py holds the JSON to the
committed oracle dump.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

INDEXED = ("xcre", "xcse", "xcge", "xcrg", "xcsg", "xcgg")
BINS = ("xxds", "xdts", "xxdg", "xdtg", "simpson")
F64_SCALARS = ("k_w", "pi5", "lamda4", "m_w_0_re", "m_w_0_im",
               "m_i_0_re", "m_i_0_im")
RADAR_F32 = ("xam_r", "xam_s", "xam_g", "xbm_s", "xbm_g", "xmu_s", "xmu_g",
             "xocms", "xobms", "xocmg", "xobmg")


def parse(text: str) -> dict:
    init, radar32, radar64 = {}, {}, {}
    seen = {name: 0 for name in INDEXED}
    bins = {name: [] for name in BINS}
    for line in text.splitlines():
        name, bits = line.split()[:2]
        if name in INDEXED:
            seen[name] += 1
            radar32[f"{name}{seen[name]}"] = bits.upper()
        elif name in BINS:
            bins[name].append(bits.upper())
        elif name in F64_SCALARS:
            radar64[name] = bits.upper()
        elif name in RADAR_F32:
            radar32[name] = bits.upper()
        else:
            init[name] = bits.upper()
    radar64.update(bins)
    return {"init_f32": init, "radar_f32": radar32, "radar_f64": radar64}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    words = parse(Path(argv[1]).read_text(encoding="ascii"))
    doc = {
        "schema": "gpuwm.wsm-family.wrf-words.v1",
        "scheme": "wsm7",
        "source": ("WRF v4.6.1 (d66e442f) phys/module_mp_wsm7.F wsm7init and "
                   "phys/module_mp_radar.F radar_init, gfortran 13.3.0 -O0, "
                   "dumped by tools/wsm7_wrf461_column_oracle/dump_wsm7_init.F90"),
        **words,
    }
    Path(argv[2]).write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n",
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
