#!/usr/bin/env python3
"""Rewrite the MYNN surface layer's evidence warning in the physics registry.

The registry is generated: ``tools/build_registry.py`` must still reproduce it
byte for byte, so the edit goes through the builder's own ``render``.  Only
the one warning of ``components.surface_layer.options.mynn`` that states the
oracle evidence is replaced; the maturity value is not changed.  Idempotent,
so an integrator can re-run it over a merged registry.

    python tools/mynn_sfclay_wrf461_column_oracle/patch_registry_warning.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

REGISTRY = REPO_ROOT / "gpuwm" / "physics_registry_v2.json"

OLD_PREFIX = ("UNVERIFIED against a WRF forecast. The column solver is "
              "oracle-matched over land and water at max relative error")

NEW = (
    "UNVERIFIED against a WRF forecast, but BITWISE against WRF v4.6.1 at "
    "column level (mynn_sfclay_variant = wrf_461). "
    "tools/mynn_sfclay_wrf461_column_oracle drives the unmodified "
    "SFCLAY1D_mynn (gfortran 13.3.0 -O2, glibc 2.39, x86-64) over 110 columns "
    "(convective and stable land, snow and sea ice, open water from calm to "
    "hurricane force, high terrain, thin and coarse first layers, the zolrib "
    "non-convergent fallback (6 columns) and 40 further edge probes) "
    "through 11 option sets "
    "(isftcflx 0-3, isfflx 0 and 1, three grid lengths, seeded first steps "
    "and restarts, and three with the SPP stochastic roughness, spp_pbl=1) "
    "and 2-3 successive steps: all 35 outputs, 244,860 words, "
    "equal WRF's under the strict build (GPUWM_WRF_EXACT=1) and under default "
    "arithmetic, replayed and free-running "
    "(tests/test_mynn_sfclay_wrf461_column_oracle_gpu.py, RTX PRO 6000, "
    "NVRTC 12.9). That took WOOF's own float32 exp, log, pow and atan, the "
    "unit compiled without multiply-add contraction, and WRF's association "
    "expression by expression (including zolrib's ri*psix2**2/psit2 and the "
    "unfused psi-table interpolation). WRF's stock GNU flags vectorize one "
    "Exner loop through libmvec's SIMD powf, which gives 4 of the 110 columns "
    "a different answer inside a 110-column tile than alone; the kernel "
    "follows the scalar build, whose answer is a function of the column. "
    "It has run a 300-step coupled forecast, but no gpuwm/WRF trajectory "
    "comparison exists.")


def patch(registry: dict) -> int:
    warnings = registry["components"]["surface_layer"]["options"]["mynn"][
        "warnings"]
    hits = [i for i, w in enumerate(warnings)
            if w.startswith(OLD_PREFIX) or w.startswith(NEW[:80])]
    if len(hits) != 1:
        raise SystemExit(f"expected one MYNN evidence warning, found {hits}")
    changed = warnings[hits[0]] != NEW
    warnings[hits[0]] = NEW
    return int(changed)


def main() -> int:
    from build_registry import render
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    changed = patch(registry)
    REGISTRY.write_bytes(render(registry))
    print("patched" if changed else "already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
