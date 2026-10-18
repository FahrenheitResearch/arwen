"""Cut WRF v4.6.1's slope-radiation and terrain-shading code out verbatim.

    python extract.py <WRF_SOURCE_ROOT> <OUT_DIR>

The same cut as tools/wrf_topo_radiation_v471_oracle/extract.py, pinned to
WRF tag v4.6.1 (commit d66e442fccc04111067e29274c9f9eaccc3cef28) instead.

What differs between the two tags, measured on the sources:

* phys/module_radiation_driver.F, share/module_model_constants.F and
  dyn_em/nest_init_utils.F are byte-identical (same sha256).
* phys/module_surface_driver.F differs only in the urban LCZ table names
  and one call argument (mminlu); the TOPO_RAD_ADJ_DRVR and TOPO_RAD_ADJ
  blocks hash the same and sit one line earlier in v4.6.1 (6976, 7046).
* dyn_em/start_em.F differs by one DEALLOCATE far from the slope block;
  the slope block hashes the same at the same lines.

So every block below carries the v4.7.1 block digest unchanged: the
statements WRF v4.6.1 compiles are the ones v4.7.1 compiles.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_V471 = (Path(__file__).resolve().parent.parent
         / "wrf_topo_radiation_v471_oracle" / "extract.py")
_spec = importlib.util.spec_from_file_location("_topo_extract_v471", _V471)
v471 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v471)

WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"

FILES = {
    "phys/module_radiation_driver.F":
        "779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa",
    "phys/module_surface_driver.F":
        "fde0533233b2b11524ef78da1e48178c867bde0de08a4de03d97d410b103b4fb",
    "dyn_em/start_em.F":
        "ff3a590a5e5228d3def988b9e50aa60fe1544c0bcf9fabce2c10a3cd9b6d33b0",
    "share/module_model_constants.F":
        "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
    "dyn_em/nest_init_utils.F":
        "07db287108ab031f89e8bb80d830871daf8ba962f60d394e1ff31c845c1b4a14",
}

#: v4.7.1's blocks at their v4.6.1 lines; the digests are v4.7.1's.
_LINES_V461 = {"topo_rad_adj_drvr": (6976, 7043),
               "topo_rad_adj": (7046, 7113)}
BLOCKS = tuple(
    (name, path, *_LINES_V461.get(name, (first, last)), digest)
    for name, path, first, last, digest in v471.BLOCKS)


def main(argv=None) -> int:
    v471.WRF_COMMIT = WRF_COMMIT
    v471.FILES = FILES
    v471.BLOCKS = BLOCKS
    rc = v471.main(argv)
    out = Path((sys.argv[1:] if argv is None else argv)[1])
    for name in ("wrf_topo_blocks.F90", "wrf_topo_inline.F90",
                 "wrf_blend_terrain.F90"):
        path = out / name
        path.write_text(path.read_text(encoding="ascii").replace(
            "WRF v4.7.1", "WRF v4.6.1").replace(
            "tools/wrf_topo_radiation_v471_oracle/extract.py",
            "tools/wrf_topo_radiation_v461_oracle/extract.py"),
            encoding="ascii", newline="\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
