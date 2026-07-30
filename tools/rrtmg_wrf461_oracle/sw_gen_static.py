"""Generate the embedded static-table block of gpuwm/core/rrtmg_sw.py.

The RRTMG SW cloud-optics (swcldpr), reference-atmosphere (swatmref),
aerosol (swaerpr) and band-wavenumber tables are DATA statements in the WRF
Fortran source - they are not part of RRTMG_SW_DATA and therefore not
covered by the ingest coefficient loader.  This tool converts the verified
oracle dump of those arrays (sw_tables.npz, dumped post-init from the
unmodified modules) into base64-encoded little-endian buffers embedded as
Python source, so the production module is self-contained.  A gate test
compares the embedded copies bit-for-bit against the dump.

usage: sw_gen_static.py sw_tables.npz out_fragment.py
"""

import base64
import sys

import numpy as np

ARRAYS = [
    ("cld/extliq1", "f4"), ("cld/ssaliq1", "f4"), ("cld/asyliq1", "f4"),
    ("cld/extice2", "f4"), ("cld/ssaice2", "f4"), ("cld/asyice2", "f4"),
    ("cld/extice3", "f4"), ("cld/ssaice3", "f4"), ("cld/asyice3", "f4"),
    ("cld/fdlice3", "f4"),
    ("cld/abari", "f4"), ("cld/bbari", "f4"), ("cld/cbari", "f4"),
    ("cld/dbari", "f4"), ("cld/ebari", "f4"), ("cld/fbari", "f4"),
    ("ref/pref", "f4"), ("ref/preflog", "f4"), ("ref/tref", "f4"),
    ("aer/rsrtaua", "f4"), ("aer/rsrpiza", "f4"), ("aer/rsrasya", "f4"),
    ("wvn/wavenum1", "f4"), ("wvn/wavenum2", "f4"), ("wvn/ngb", "i4"),
]


def main(npz_path, out_path):
    d = np.load(npz_path)
    lines = [
        "# --- BEGIN GENERATED STATIC TABLES"
        " (tools/rrtmg_wrf461_oracle/sw_gen_static.py) ---",
        "# Source: sw_tables.npz oracle dump of the unmodified WRF modules"
        " post rrtmg_sw_ini.",
        "_STATIC_TABLES = {",
    ]
    for name, dt in ARRAYS:
        a = np.ascontiguousarray(d[name].astype(dt), dtype=np.dtype("<" + dt))
        b64 = base64.b64encode(a.tobytes(order="F")).decode()
        chunks = [b64[i:i + 64] for i in range(0, len(b64), 64)]
        payload = "\n".join(f'        "{c}"' for c in chunks)
        lines.append(f'    "{name}": ("{dt}", {tuple(a.shape)},')
        lines.append(payload + "),")
    lines.append("}")
    lines.append("# --- END GENERATED STATIC TABLES ---")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}: {len(ARRAYS)} arrays")


if __name__ == "__main__":
    main(*sys.argv[1:3])
