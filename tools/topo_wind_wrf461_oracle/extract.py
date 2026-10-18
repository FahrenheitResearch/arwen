"""Cut WRF v4.6.1's topo_wind static coefficients out of start_em.F verbatim.

    python extract.py <WRF_SOURCE_ROOT> <OUT_DIR>

WRF forms LAP_HGT, CTOPO and CTOPO2 inline in dyn_em/start_em.F, inside
start_domain_em, which cannot be compiled on its own.  This copies the two
blocks into one small Fortran module, after checking the sha256 of the whole
file and of each block, so the oracle compiles WRF's own statements and
nothing else:

* start_em.F:1539-1577, the terrain loop that forms LAP_HGT (and the slope
  fields, which topo_wind does not read);
* start_em.F:1579-1626, CTOPO and CTOPO2 for topo_wind = 1 and 2.

The wrapper (dummy arguments under WRF's own grid%/config_flags% names) is
the one tools/terrain_drag_wrf471_oracle/extract.py uses; the two blocks are
byte-identical between v4.6.1 and v4.7.1 (same per-block sha256), the file
around them is not.

WRF v4.6.1 is tag v4.6.1, commit d66e442fccc04111067e29274c9f9eaccc3cef28,
public domain.  The file hash is of the LF checkout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
START_EM = ("dyn_em/start_em.F",
            "ff3a590a5e5228d3def988b9e50aa60fe1544c0bcf9fabce2c10a3cd9b6d33b0")


def _wrf471_extract():
    path = (Path(__file__).resolve().parents[1] / "terrain_drag_wrf471_oracle"
            / "extract.py")
    spec = importlib.util.spec_from_file_location("_td471_extract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit("usage: extract.py WRF_SOURCE_ROOT OUT_DIR")
    root, out = Path(args[0]), Path(args[1])
    base = _wrf471_extract()
    path, digest = START_EM
    raw = (root / path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != digest:
        raise SystemExit(f"{path} is not WRF v4.6.1 ({WRF_COMMIT}): sha256 "
                         f"{got}, expected {digest}")
    lines = raw.split(b"\n")
    blocks = {name: base._block(lines, name, first, last, sha)
              for name, first, last, sha in base.BLOCKS}
    out.mkdir(parents=True, exist_ok=True)
    header = (f"! Extracted verbatim from WRF v4.6.1 ({WRF_COMMIT}) by "
              "tools/topo_wind_wrf461_oracle/extract.py.\n"
              "! Do not edit: regenerate.\n")
    (out / "wrf_topo_wind_inline.F90").write_text(
        header + base._MODULE.format(**blocks), encoding="ascii",
        newline="\n")
    print(f"extracted {len(blocks)} blocks into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
