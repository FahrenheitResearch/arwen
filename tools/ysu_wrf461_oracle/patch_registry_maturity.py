#!/usr/bin/env python3
"""Stop YSU and Noah claiming ``supported`` on no WRF evidence.

Written as a script rather than applied by hand for two reasons.  The registry
is generated -- ``tools/build_registry.py`` must still reproduce it byte for
byte -- so the edit has to go through the builder's own ``canonical_json``.
And the tracked file carries another lane's uncommitted work, so the same edit
must be applied twice: once to the working tree, and once to the blob at HEAD
so that the commit contains this lane's hunks and nothing else.

    python tools/ysu_wrf461_oracle/patch_registry_maturity.py IN.json OUT.json

Idempotent: re-running over an already-patched registry changes nothing, so the
byte-for-byte builder gate cannot be tripped by applying it twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_registry import canonical_json  # noqa: E402

YSU_WARNINGS = [
    "YSU RUNS and its distance from WRF is now MEASURED rather than assumed. "
    "Until 2026-07-26 this option was maturity 'supported' with no warnings "
    "and no WRF number of any kind: the CUDA kernel had only ever been "
    "compared to gpuwm.verify.npref.np_ysu_column, a float64 mirror of the "
    "same transcription, so a misread line agreed with itself. "
    "tools/ysu_wrf461_oracle now drives bl_ysu_run in the byte-unmodified "
    "phys/physics_mmm/bl_ysu.F90 at the pinned commit and dumps 24 columns x "
    "40 levels of inputs and outputs. Measured on two RTX 5090s with "
    "identical maxima (tests/test_ysu_wrf461_parity.py): potential-temperature "
    "tendency 1 ULP, EXCH_H/EXCH_M 7 ULP (2.4e-04 m2/s), PBLH/WSTAR/DELTA 1 "
    "ULP, KPBL exact, and the momentum and moisture tendencies 1457 and 23302 "
    "ULP -- which is 4.2e-08 m/s2 and 3.1e-11 kg/kg/s in absolute terms, "
    "because those tendencies are near-total cancellations that amplify a "
    "7-ULP diffusivity. What remains is CUDA's expf/powf against glibc's; "
    "respelling ust**3, cbrtf and WRF's exact association for fric, xkzm, "
    "prnumfac and entfac was measured and changes none of it.",

    "CORRECTNESS BUG, not a parity gap: a POSITIVE SUBNORMAL BR selects the "
    "WRONG STABILITY REGIME. WRF's test is br > 0.0 (bl_ysu.F90:613); the "
    "kernel's is br <= 0.0f, and CuPy appends -ftz=true unconditionally, so a "
    "br of 1.4e-45 flushes to zero and the kernel takes the CONVECTIVE arm "
    "where WRF takes the STABLE one. Measured on that column: WSTAR 0.236 "
    "against WRF's 0, DELTA 18.49 against WRF's 0, PBLH 1525 ULP out, and "
    "every tendency 1e7 to 1e8 ULP out. The smallest NORMAL br (1.17549435e-38) "
    "agrees exactly, which is how the cause is known to be the flush and not "
    "the operator. This is the FOURTH time -ftz has produced a real bug in "
    "this tree. Closing it needs a compare that cannot be flushed; --ftz=false "
    "is not available through CuPy.",

    "gpuwm SHORT CIRCUITS a case WRF computes. kernels/ysu.cu:94 returns zero "
    "tendencies, PBLH = dz(1) and KPBL = 1 whenever UST, HFX and QFX are all "
    "zero. bl_ysu_run has no such branch: measured on that column it produces "
    "a 703 m PBL with nine levels in it, EXCH_M of 0.143 m2/s and nonzero "
    "momentum tendencies. -ftz makes the same branch fire on a SUBNORMAL ust, "
    "i.e. on a nonzero input. Not fixed here because removing the branch "
    "exposes WRF's own 0/0 (next warning) and is not a one-line change.",

    "DELIBERATE DIVERGENCE, WRF is undefined: with ust**3 underflowed to zero, "
    "wstar3 zero and sfcflg true, bl_ysu.F90:963 computes prfac2 = "
    "15.9*(wstar3+wstar3_2)/ust3/(1.+4.*karman*(...)/ust3) = 0/0 and fills the "
    "column with NaN -- pinned in the oracle fixture. gpuwm returns finite "
    "numbers. Separately, at kpbl == kte with cloud at kpbl-1, bl_ysu.F90:846 "
    "reads thlix(i,k+2) one element past an array declared kts:kte; "
    "kernels/ysu.cu:252 guards that access with kpbl < nz and skips the "
    "top-down block instead.",

    "THE ARM WRF ACTUALLY USES IS NOT THE ARM THIS PORT IMPLEMENTS. "
    "module_bl_ysu.F:404 always passes ctopo and ctopo2, and the Registry "
    "default (topo_wind=0) fills both with 1.0, so every default WRF column "
    "takes bl_ysu.F90:1308 -- ad(i,1) = 1+fric*vconvlim+ctopo*fric*(1-vconvlim) "
    "-- which needs the paj TKE block, GET_PBLH and the Beljaars vconv, none of "
    "which is ported. kernels/ysu.cu implements :1315, ad(i,1) = 1+fric, and "
    "has no ctopo argument. Measured WRF against WRF on the same fixture: the "
    "two arms differ by up to 182 ULP (1.1e-08 m/s2) on 6 of 960 lanes, "
    "wherever vconvlim < 1. Small, but it means the port cannot be bitwise "
    "against a default WRF run even with everything else closed.",

    "-ftz=true also FLUSHES SUBNORMAL TENDENCIES to exactly zero. Eleven lanes "
    "of the oracle fixture carry a subnormal qv/qc/qi tendency from WRF and the "
    "kernel writes zero in all eleven, which reads as up to 207470 ULP but is "
    "3.3e-13 kg/kg/s. Physically nil; it is recorded because it is the same "
    "mechanism as the br bug above, where the consequence was not nil.",

    "UNVERIFIED against a WRF forecast. The oracle compares one YSU call at a "
    "time on synthetic columns. No gpuwm/WRF trajectory comparison exists for "
    "this scheme, and no gpuwm run has ever been checked against WRF with "
    "bl_pbl_physics=1. That, plus the open items above, is why this is no "
    "longer 'supported'.",
]

NOAH_WARNINGS = [
    "NOAH IS UNVERIFIED AGAINST WRF. Until 2026-07-26 this option was maturity "
    "'supported' -- the highest grade in this registry -- with no warnings, no "
    "oracle and no ULP gate. What tests/test_noah.py actually establishes is "
    "self-consistency and physics plausibility: 37 tests, of which the CUDA "
    "sweeps compare kernels/noah.cu to gpuwm.verify.npref.np_noah_column (a "
    "float64 mirror of the SAME transcription), plus energy-closure under "
    "1 W/m2, a moisture budget, and REDPRM's derived parameters. None of it "
    "compares a single number to the unmodified phys/module_sf_noahlsm.F. A "
    "transcription error present in both the kernel and its mirror passes "
    "every one of those tests.",

    "The only WRF-derived evidence Noah carries is its PARAMETER TABLES: "
    "gpuwm/data/noah_tables VEGPARM.TBL, SOILPARM.TBL and GENPARM.TBL are "
    "byte-identical to WRF v4.6.1's run/ directory and are parsed by "
    "transcribing SOIL_VEG_GEN_PARM's list-directed READ sequence "
    "(tests/test_noah.py::test_shipped_tables_byte_identical_to_wrf_run_dir). "
    "That pins the inputs, not the physics.",

    "WHAT CLOSING THIS TAKES, in the order it should be done. SFLX is public "
    "in module_sf_noahlsm.F, so the harness is the Noah-MP one with a smaller "
    "stub set: (1) FRH2O -- the frozen-water Newton solve, already mirrored at "
    "gpuwm/core/noah.py:289 and run for every subfreezing soil layer of every "
    "cold start, so it is both the smallest leaf and the one whose error would "
    "be widest; (2) REDPRM, whose derived parameters test_noah.py already "
    "asserts against hand values; (3) SFLX whole-column, against which "
    "kernels/noah.cu can be measured the way NOAHMP_SFLX was. Until at least "
    "(1) and (3) exist, this option's numbers have never been outside gpuwm.",
]


def patch(registry: dict) -> dict:
    ysu = registry["components"]["pbl"]["options"]["ysu"]
    ysu["maturity"] = "implemented-unverified"
    ysu["warnings"] = list(YSU_WARNINGS)
    noah = registry["components"]["land_surface"]["options"]["noah"]
    noah["maturity"] = "implemented-unverified"
    noah["warnings"] = list(NOAH_WARNINGS)
    return registry


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    source, destination = Path(argv[1]), Path(argv[2])
    registry = json.loads(source.read_text(encoding="utf-8"))
    destination.write_bytes(
        (canonical_json(patch(registry)) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
