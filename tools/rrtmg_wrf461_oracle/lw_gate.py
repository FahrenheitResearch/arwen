"""Shared gate plumbing for the RRTMG LW port: fixtures -> port inputs.

Used by tests/test_rrtmg_lw*.py and by the per-band gate runner.  All
comparisons are bitwise (max_ulp 0 means identical bit patterns; NaN would
fail by design).
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from lw_fixtures import read_fixture  # noqa: E402

#: Default fixture location (oracle scratch); override with env var.
DEFAULT_FIXDIR = os.environ.get(
    "GPUWM_RRTMG_LW_FIXTURES",
    os.path.expanduser("~/.claude/jobs/fea7a141/tmp/rrtmg_lw/fixtures"),
)

#: Raw variable names per band module, as read from RRTMG_LW_DATA by
#: lw_kgb01..16 (the frozen loader contract's per-module content).
RAW_VARS = {
    1: ["fracrefao", "fracrefbo", "kao", "kbo", "kao_mn2", "kbo_mn2",
        "selfrefo", "forrefo"],
    2: ["fracrefao", "fracrefbo", "kao", "kbo", "selfrefo", "forrefo"],
    3: ["fracrefao", "fracrefbo", "kao", "kbo", "kao_mn2o", "kbo_mn2o",
        "selfrefo", "forrefo"],
    4: ["fracrefao", "fracrefbo", "kao", "kbo", "selfrefo", "forrefo"],
    5: ["fracrefao", "fracrefbo", "kao", "kbo", "kao_mo3", "selfrefo",
        "forrefo", "ccl4o"],
    6: ["fracrefao", "kao", "kao_mco2", "selfrefo", "forrefo",
        "cfc11adjo", "cfc12o"],
    7: ["fracrefao", "fracrefbo", "kao", "kbo", "kao_mco2", "kbo_mco2",
        "selfrefo", "forrefo"],
    8: ["fracrefao", "fracrefbo", "kao", "kbo", "kao_mco2", "kbo_mco2",
        "kao_mn2o", "kbo_mn2o", "kao_mo3", "selfrefo", "forrefo",
        "cfc12o", "cfc22adjo"],
    9: ["fracrefao", "fracrefbo", "kao", "kbo", "kao_mn2o", "kbo_mn2o",
        "selfrefo", "forrefo"],
    10: ["fracrefao", "fracrefbo", "kao", "kbo", "selfrefo", "forrefo"],
    11: ["fracrefao", "fracrefbo", "kao", "kbo", "kao_mo2", "kbo_mo2",
         "selfrefo", "forrefo"],
    12: ["fracrefao", "kao", "selfrefo", "forrefo"],
    13: ["fracrefao", "fracrefbo", "kao", "kao_mco2", "kao_mco",
         "kbo_mo3", "selfrefo", "forrefo"],
    14: ["fracrefao", "fracrefbo", "kao", "kbo", "selfrefo", "forrefo"],
    15: ["fracrefao", "kao", "kao_mn2", "selfrefo", "forrefo"],
    16: ["fracrefao", "fracrefbo", "kao", "kbo", "selfrefo", "forrefo"],
}


def load_coeffs_fixture(fixdir=DEFAULT_FIXDIR):
    return read_fixture(os.path.join(fixdir, "lw_coeffs.bin"))


def case_paths(fixdir=DEFAULT_FIXDIR):
    paths = sorted(glob.glob(os.path.join(fixdir, "lw_case_*.bin")))
    if not paths:
        raise RuntimeError(
            f"no RRTMG LW fixtures under {fixdir}; run "
            "tools/rrtmg_wrf461_oracle/lw_build.sh (see its header) or set "
            "GPUWM_RRTMG_LW_FIXTURES")
    return paths


def raw_from_coeffs_fixture(cfx):
    """Rebuild the frozen loader contract's dict from the oracle dump."""
    raw = {}
    for band in range(1, 17):
        mod = f"rrlw_kg{band:02d}"
        raw[mod] = {v: cfx[f"{mod}/{v}"] for v in RAW_VARS[band]
                    if f"{mod}/{v}" in cfx}
        missing = [v for v in RAW_VARS[band] if f"{mod}/{v}" not in cfx]
        if missing:
            raise KeyError(f"{mod}: fixture missing raw arrays {missing}")
    return raw


def port_coeffs_from_fixture(cfx):
    """Coefficient dict for the compute-chain ports, taken directly from
    the post-init Fortran module dump (so routine gates are independent of
    the init-chain port)."""
    C = {}
    for band in range(1, 17):
        mod = f"rrlw_kg{band:02d}"
        for key, val in cfx.items():
            if key.startswith(f"{mod}/"):
                name = key.split("/", 1)[1]
                C[f"kg{band:02d}/{name}"] = val
    for name in ("totplnk", "totplk16", "rwgt", "delwave", "ngb", "ngs"):
        C[f"wvn/{name}"] = cfx[f"rrlw_wvn/{name}"]
    for name in ("pref", "preflog", "tref", "chi_mls"):
        C[f"ref/{name}"] = cfx[f"rrlw_ref/{name}"]
    for name in ("tau_tbl", "exp_tbl", "tfn_tbl", "bpade"):
        C[f"tbl/{name}"] = cfx[f"rrlw_tbl/{name}"]
    for name in ("fluxfac", "heatfac", "oneminus", "pi", "grav", "avogad"):
        C[f"con/{name}"] = cfx[f"rrlw_con/{name}"]
    for name in ("abscld1", "absice0", "absice1", "absice2", "absice3",
                 "absliq0", "absliq1"):
        C[f"cld/{name}"] = cfx[f"rrlw_cld/{name}"]
    return C


def state_from_fixture(fx):
    """taumol/rtrnmc input state assembled from the inatm+setcoef stage
    dumps of one case fixture (NOT from the port's own setcoef)."""
    st = {}
    for key, val in fx.items():
        if key.startswith("setcoef/") or key.startswith("inatm/"):
            st[key.split("/", 1)[1]] = val
    st["laytrop"] = int(fx["setcoef/laytrop"])
    st["wbroad"] = fx["inatm/wbrodl"]
    return st


def ulp_report(got, want, what):
    got = np.atleast_1d(np.asarray(got, np.float32))
    want = np.atleast_1d(np.asarray(want, np.float32))
    if got.shape != want.shape:
        return f"{what}: SHAPE {got.shape} vs {want.shape}"
    neq = got.view(np.uint32) != want.view(np.uint32)
    if not neq.any():
        return None
    gi = np.asarray(got.view(np.int32), np.int64)
    wi = np.asarray(want.view(np.int32), np.int64)
    gi = np.where(gi < 0, -0x80000000 - gi, gi)
    wi = np.where(wi < 0, -0x80000000 - wi, wi)
    ulp = int(np.abs(gi - wi)[neq].max())
    idx = tuple(int(v) for v in np.argwhere(neq)[0])
    return (f"{what}: {int(neq.sum())}/{got.size} differ, max_ulp={ulp}, "
            f"first at {idx}: got {got[idx]!r} want {want[idx]!r}")


def assert_ulp0(got, want, what):
    msg = ulp_report(got, want, what)
    if msg is not None:
        raise AssertionError(msg)


def assert_ulp0_int(got, want, what):
    got = np.atleast_1d(np.asarray(got, np.int64))
    want = np.atleast_1d(np.asarray(want, np.int64))
    if got.shape != want.shape or (got != want).any():
        bad = np.argwhere(np.atleast_1d(got != want))
        raise AssertionError(f"{what}: int mismatch at {bad[:5].tolist()}")


def band_slice(band):
    """Python slice of taug/fracs columns for one band (1-based band)."""
    lo = 0 if band == 1 else int(np.cumsum([10, 12, 16, 14, 16, 8, 12, 8,
                                            12, 6, 8, 8, 4, 2, 2, 2])[band - 2])
    hi = int(np.cumsum([10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2,
                        2, 2])[band - 1])
    return slice(lo, hi)
