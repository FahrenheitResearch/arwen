"""Bitwise gates for the legacy RRTMG option-4 driver-side prep port.

Every comparison is max_ulp 0 (identical FP32 bit patterns) against
oracle fixtures recorded from the UNMODIFIED WRF v4.6.1 wrappers:

* LW: tools/rrtmg_wrf461_oracle/lw_extract.F90 dumps the raw RRTMG_LWRAD
  inputs (wrfin/*), the exact rrtmg_lw arguments the wrapper builds
  (in/*), the rrtmg_lw outputs (out/*) and the untouched wrapper's
  WRF-level outputs (wrap/*).  lwrad_prep is gated wrfin/* -> in/*,
  lwrad_outputs out/* -> wrap/*.  Fixture location comes from
  GPUWM_RRTMG_LW_FIXTURES (lw_gate.DEFAULT_FIXDIR); the LW gates skip
  cleanly when the directory is absent, mirroring test_rrtmg_lw_numpy.
* SW: tools/rrtmg_wrf461_oracle/sw_fixtures/*.npz (committed, always
  present).  swrad_prep is gated in/* -> mcin/* + entry/*; night columns
  gate the swrad_night_outputs contract against wrf/* and require
  swrad_prep to fail closed.

The wrapper flags the LW oracle does not dump (warm_rain, F_QI/F_QS/
F_QG) are reconstructed from the committed fixture generator
tools/rrtmg_wrf461_oracle/lw_make_inputs.py: every case uses all-true
flags and warm_rain=.false. except the two MP-option-3 cases
(mp_physics == 3, F_QI=F_QS=F_QG=.false.), whose warm_rain variant is
marked by the tsk the generator wrote (260.0 for the cold, swap-active
variant 0; the synthetic default otherwise).
"""

from __future__ import annotations

import glob
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(_REPO, "tools", "rrtmg_wrf461_oracle")
sys.path.insert(0, _TOOLS)

from lw_fixtures import read_fixture          # noqa: E402
from lw_gate import DEFAULT_FIXDIR, ulp_report  # noqa: E402

from gpuwm.core import rrtmg_legacy_prep as prep  # noqa: E402

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _assert(msg):
    assert msg is None, msg


def assert_bits(name, got, want):
    _assert(ulp_report(got, want, name))


_G = np.float32(9.81)      # the g dummy WRF passes to both wrappers

# ---------------------------------------------------------------------------
# LW: RRTMG_LWRAD prep and output mapping (skip cleanly without fixtures)
# ---------------------------------------------------------------------------

_LW_PRESENT = os.path.isdir(DEFAULT_FIXDIR)
LW_CASES = (sorted(glob.glob(os.path.join(DEFAULT_FIXDIR, "lw_case_*.bin")))
            if _LW_PRESENT else [])

lw_skip = pytest.mark.skipif(
    not LW_CASES,
    reason="RRTMG LW oracle fixtures not present "
           "(run tools/rrtmg_wrf461_oracle/lw_build.sh or set "
           "GPUWM_RRTMG_LW_FIXTURES)")


@lru_cache(maxsize=4)
def _lw_case(path):
    return read_fixture(path)


def _lw_flags(fx):
    """Reconstruct (warm_rain, f_qi, f_qs, f_qg) per the committed
    generator lw_make_inputs.py (see module docstring)."""
    if int(fx["meta/mp_physics"]) == 3:
        warm_rain = float(fx["wrfin/tsk"]) != 260.0
        return warm_rain, False, False, False
    return False, True, True, True


def _lw_prep_kwargs(fx):
    """The scalar lwrad_prep argument dict for one oracle case."""
    warm_rain, f_qi, f_qs, f_qg = _lw_flags(fx)
    nlayers = prep.compute_lw_nlayers(int(fx["meta/nz"]) + 1,
                                      fx["meta/p_top"])
    assert nlayers == int(fx["meta/nlayers"]), "nlayers formula drifted"
    return dict(
        p3d=fx["wrfin/p3d"], p8w=fx["wrfin/p8w"],
        t3d=fx["wrfin/t3d"], t8w=fx["wrfin/t8w"], dz8w=fx["wrfin/dz8w"],
        qv3d=fx["wrfin/qv"], qc3d=fx["wrfin/qc"], qr3d=fx["wrfin/qr"],
        qi3d=fx["wrfin/qi"], qs3d=fx["wrfin/qs"], qg3d=fx["wrfin/qg"],
        cldfra3d=fx["wrfin/cldfra"], o33d=fx["wrfin/o33d"],
        re_cloud=fx["wrfin/re_cloud"], re_ice=fx["wrfin/re_ice"],
        re_snow=fx["wrfin/re_snow"],
        tsk=fx["wrfin/tsk"], emiss=fx["wrfin/emiss"],
        xland=fx["wrfin/xland"], xice=fx["wrfin/xice"],
        snow=fx["wrfin/snow"], xlat=fx["meta/xlat"],
        icloud=int(fx["meta/icloud"]), warm_rain=warm_rain,
        cldovrlp=int(fx["meta/cldovrlp"]), idcor=int(fx["meta/idcor"]),
        o3input=int(fx["meta/o3input"]),
        has_reqc=int(fx["meta/has_reqc"]), has_reqi=int(fx["meta/has_reqi"]),
        has_reqs=int(fx["meta/has_reqs"]),
        f_qc=True, f_qr=True, f_qi=f_qi, f_qs=f_qs, f_qg=f_qg,
        yr=int(fx["meta/yr"]), julian=fx["meta/julian"],
        nlayers=nlayers, mp_physics=int(fx["meta/mp_physics"]), g=_G)


def _run_lw_prep(fx, overrides=None):
    kwargs = _lw_prep_kwargs(fx)
    if overrides:
        kwargs.update(overrides)
    return prep.lwrad_prep(**kwargs)


_LW_INT_ENTRIES = ("ncol", "nlay", "icld", "inflglw", "iceflglw", "liqflglw")
_LW_LAYER_ENTRIES = (
    "play", "tlay", "h2ovmr", "o3vmr", "co2vmr", "ch4vmr", "n2ovmr",
    "o2vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr", "ccl4vmr",
    "reicmcl", "relqmcl", "resnmcl")
_LW_LEVEL_ENTRIES = ("plev", "tlev")
_LW_MCL_ENTRIES = ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl")


@lw_skip
@pytest.mark.parametrize(
    "path", LW_CASES or ["missing"],
    ids=[Path(p).stem for p in LW_CASES] or ["missing"])
def test_lwrad_prep_bitwise(path):
    """lwrad_prep == in/* on every rrtmg_lw entry array and scalar."""
    fx = _lw_case(path)
    got = _run_lw_prep(fx)
    for name in _LW_INT_ENTRIES:
        assert int(got[name]) == int(fx[f"in/{name}"]), (
            f"in/{name}: got {got[name]} want {fx[f'in/{name}']}")
    for name in _LW_LAYER_ENTRIES + _LW_LEVEL_ENTRIES:
        assert_bits(f"in/{name}", got[name], fx[f"in/{name}"][0])
    assert_bits("in/tsfc", got["tsfc"], fx["in/tsfc"][0])
    assert_bits("in/emis", got["emis"], fx["in/emis"][0])
    for name in _LW_MCL_ENTRIES:
        assert_bits(f"in/{name}", got[name], fx[f"in/{name}"][:, 0, :])
    assert_bits("in/tauaer", got["tauaer"], fx["in/tauaer"][0])


@lw_skip
@pytest.mark.parametrize(
    "path", LW_CASES or ["missing"],
    ids=[Path(p).stem for p in LW_CASES] or ["missing"])
def test_lwrad_outputs_bitwise(path):
    """lwrad_outputs == wrap/* (the untouched wrapper's WRF-level LW
    outputs), fed with the oracle's rrtmg_lw fluxes (out/*)."""
    fx = _lw_case(path)
    got = prep.lwrad_outputs(
        uflx=fx["out/uflx"][0], dflx=fx["out/dflx"][0],
        hr=fx["out/hr"][0], uflxc=fx["out/uflxc"][0],
        dflxc=fx["out/dflxc"][0], hrc=fx["out/hrc"][0],
        pi3d=fx["wrap/pi3d"])
    for name in ("glw", "olr", "lwcf", "lwupt", "lwuptc", "lwdnt",
                 "lwdntc", "lwupb", "lwupbc", "lwdnb", "lwdnbc"):
        assert_bits(f"wrap/{name}", got[name], fx[f"wrap/{name}"])
    assert_bits("wrap/rthratenlw", got["rthratenlw"],
                fx["wrap/rthratenlw"])
    assert_bits("wrap/rthratenlwc", got["rthratenlwc"],
                fx["wrap/rthratenlwc"])


# ---------------------------------------------------------------------------
# SW: RRTMG_SWRAD prep and the night contract (fixtures committed)
# ---------------------------------------------------------------------------

_SW_FIXDIR = Path(_TOOLS) / "sw_fixtures"
_SW_FILES = ("fixtures_real.npz", "fixtures_synth.npz")

_sw_cache = None


def _sw_fixtures():
    global _sw_cache
    if _sw_cache is None:
        _sw_cache = {}
        for name in _SW_FILES:
            f = np.load(_SW_FIXDIR / name)
            _sw_cache.update({k: f[k] for k in f.files})
    return _sw_cache


def _sw_case_ids():
    return sorted({k.split("/")[0] for k in _sw_fixtures()})


SW_ALL = _sw_case_ids()
SW_DAY = [c for c in SW_ALL if int(_sw_fixtures()[f"{c}/night"]) == 0]
SW_NIGHT = [c for c in SW_ALL if int(_sw_fixtures()[f"{c}/night"]) == 1]


def _sw_prep_kwargs(d, c):
    # The SW oracle pins warm_rain=.false. and all six F_Q* true
    # (sw_fixture_driver.F90 step-1 call).
    return dict(
        p3d=d[f"{c}/in/p3d"], p8w=d[f"{c}/in/p8w"],
        t3d=d[f"{c}/in/t3d"], t8w=d[f"{c}/in/t8w"],
        dz8w=d[f"{c}/in/dz8w"],
        qv3d=d[f"{c}/in/qv3d"], qc3d=d[f"{c}/in/qc3d"],
        qr3d=d[f"{c}/in/qr3d"], qi3d=d[f"{c}/in/qi3d"],
        qs3d=d[f"{c}/in/qs3d"], qg3d=d[f"{c}/in/qg3d"],
        cldfra3d=d[f"{c}/in/cldfra3d"], o33d=d[f"{c}/in/o33d"],
        re_cloud=d[f"{c}/in/re_cloud"], re_ice=d[f"{c}/in/re_ice"],
        re_snow=d[f"{c}/in/re_snow"],
        tsk=d[f"{c}/in/tsk"], albedo=d[f"{c}/in/albedo"],
        xland=d[f"{c}/in/xland"], xice=d[f"{c}/in/xice"],
        snow=d[f"{c}/in/snow"], xlat=d[f"{c}/in/xlat"],
        xcoszen=d[f"{c}/in/xcoszen"], solcon=d[f"{c}/in/solcon"],
        obscur=d[f"{c}/in/obscur"],
        icloud=int(d[f"{c}/in/icloud"]), warm_rain=False,
        cldovrlp=int(d[f"{c}/in/cldovrlp"]), idcor=int(d[f"{c}/in/idcor"]),
        o3input=int(d[f"{c}/in/o3input"]),
        has_reqc=int(d[f"{c}/in/has_reqc"]),
        has_reqi=int(d[f"{c}/in/has_reqi"]),
        has_reqs=int(d[f"{c}/in/has_reqs"]),
        f_qc=True, f_qr=True, f_qi=True, f_qs=True, f_qg=True,
        yr=int(d[f"{c}/in/yr"]), julian=d[f"{c}/in/julian"],
        mp_physics=int(d[f"{c}/in/mp_physics"]), g=_G,
        sf_surface_physics=int(d[f"{c}/in/sf_surface_physics"]))


def _run_sw_prep(d, c, overrides=None):
    kwargs = _sw_prep_kwargs(d, c)
    if overrides:
        kwargs.update(overrides)
    return prep.swrad_prep(**kwargs)


_SW_MCIN_ARRAYS = ("play", "cldfrac", "ciwpth", "clwpth", "cswpth",
                   "rei", "rel", "res", "hgt")
_SW_MCIN_INTS = ("icld", "idcor", "juldat", "permuteseed", "irng")
_SW_ENTRY_INTS = ("dyofyr", "icld", "inflgsw", "iceflgsw", "liqflgsw",
                  "nlay")
_SW_ENTRY_SCALARS = ("tsfc", "asdir", "asdif", "aldir", "aldif",
                     "coszen", "adjes", "scon")
_SW_ENTRY_LAYERS = ("play", "tlay", "h2ovmr", "o3vmr", "co2vmr",
                    "ch4vmr", "n2ovmr", "o2vmr",
                    "reicmcl", "relqmcl", "resnmcl")
_SW_ENTRY_LEVELS = ("plev", "tlev")
_SW_ENTRY_MCL = ("cldfmcl", "taucmcl", "ssacmcl", "asmcmcl", "fsfcmcl",
                 "ciwpmcl", "clwpmcl", "cswpmcl")


@pytest.mark.parametrize("case", SW_DAY)
def test_swrad_prep_bitwise(case):
    """swrad_prep == mcin/* (the mcica_subcol_sw call record) and
    entry/* (the exact rrtmg_sw arguments) on every day column."""
    d = _sw_fixtures()
    got = _run_sw_prep(d, case)
    mcin = got["mcica_inputs"]
    for name in _SW_MCIN_ARRAYS:
        assert_bits(f"{case} mcin/{name}", mcin[name],
                    d[f"{case}/mcin/{name}"])
    for name in _SW_MCIN_INTS:
        assert int(mcin[name]) == int(d[f"{case}/mcin/{name}"]), (
            f"{case} mcin/{name}")
    assert_bits(f"{case} mcin/lat", mcin["lat"], d[f"{case}/mcin/lat"])

    for name in _SW_ENTRY_INTS:
        assert int(got[name]) == int(d[f"{case}/entry/{name}"]), (
            f"{case} entry/{name}: got {got[name]} "
            f"want {d[f'{case}/entry/{name}']}")
    for name in _SW_ENTRY_SCALARS:
        assert_bits(f"{case} entry/{name}", got[name],
                    d[f"{case}/entry/{name}"])
    for name in _SW_ENTRY_LAYERS + _SW_ENTRY_LEVELS:
        assert_bits(f"{case} entry/{name}", got[name],
                    d[f"{case}/entry/{name}"])
    for name in _SW_ENTRY_MCL:
        assert_bits(f"{case} entry/{name}", got[name],
                    d[f"{case}/entry/{name}"])
    # COSZR is written before the dorrsw gate.
    assert_bits(f"{case} wrf/coszr", got["coszr"], d[f"{case}/wrf/coszr"])


@pytest.mark.parametrize("case", SW_NIGHT)
def test_swrad_night_contract(case):
    """coszen <= 0: rrtmg_sw is never called (swrad_prep fails closed),
    COSZR is still set, the fixed diagnostic list is zeroed, and
    RTHRATENSW/GSW/flux profiles are untouched (harness zeros)."""
    d = _sw_fixtures()
    assert float(d[f"{case}/in/xcoszen"]) <= 0.0
    with pytest.raises(ValueError, match="night column"):
        _run_sw_prep(d, case)

    night = prep.swrad_night_outputs(d[f"{case}/in/xcoszen"])
    assert_bits(f"{case} wrf/coszr", night["coszr"],
                d[f"{case}/wrf/coszr"])
    for name in prep.SW_NIGHT_ZEROED:
        assert_bits(f"{case} wrf/{name}", night[name],
                    d[f"{case}/wrf/{name}"])
    # The untouched set is exactly that -- the night mapping must not
    # invent values for fields WRF leaves stale.
    for name in prep.SW_NIGHT_UNTOUCHED:
        assert name not in night
    # The oracle harness zero-filled them before the call, so the
    # recorded values pin "untouched" as "still the caller's zeros".
    for name in ("gsw", "rthratensw", "rthratenswc",
                 "swupflx", "swupflxc", "swdnflx", "swdnflxc"):
        assert not np.asarray(d[f"{case}/wrf/{name}"]).any(), (
            f"{case} wrf/{name}: night fixture expected caller zeros")


def test_night_gate_boundary():
    """The dorrsw gate is coszrs <= 0.0: exactly 0 is night, the smallest
    positive float32 is day (prep proceeds far enough to reject nothing)."""
    with pytest.raises(ValueError, match="night column"):
        prep.swrad_prep(
            p3d=np.full(4, 5.0e4, np.float32),
            p8w=np.linspace(1.0e5, 1.0e4, 5).astype(np.float32),
            t3d=np.full(4, 260.0, np.float32),
            t8w=np.full(5, 260.0, np.float32),
            dz8w=np.full(4, 500.0, np.float32),
            qv3d=np.full(4, 1e-3, np.float32),
            tsk=280.0, albedo=0.2, xland=1.0, xice=0.0, snow=0.0,
            xlat=30.0, xcoszen=0.0, solcon=1361.0, obscur=0.0,
            icloud=1, warm_rain=False, cldovrlp=2, idcor=0, o3input=0,
            has_reqc=0, has_reqi=0, has_reqs=0, yr=2000, julian=100.0)


# ---------------------------------------------------------------------------
# Batch twins: the column-vectorized entries must equal the scalar entries
# bitwise, per column, over the full fixture decks (packed as flag-tuple
# groups per the shared-flag contract), at synthetic width (pure tiling),
# through the McICA generators (batch call == per-column calls), and on
# the cupy path when a GPU is available and not already loaded.
# ---------------------------------------------------------------------------

_LW_BATCH_INT_KEYS = ("nlay", "icld", "inflglw", "iceflglw", "liqflglw",
                      "juldat")
_LW_BATCH_COL_KEYS = (
    "play", "plev", "tlay", "tlev", "h2ovmr", "o3vmr", "co2vmr", "ch4vmr",
    "n2ovmr", "o2vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr", "ccl4vmr",
    "emis", "reicmcl", "relqmcl", "resnmcl", "tauaer", "o31d", "hgt",
    "cldfrac", "pdel")
_LW_BATCH_MCL_KEYS = ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl")

_SW_BATCH_INT_KEYS = ("nlay", "icld", "inflgsw", "iceflgsw", "liqflgsw",
                      "dyofyr", "juldat")
_SW_BATCH_SURF_KEYS = ("tsfc", "asdir", "asdif", "aldir", "aldif",
                       "coszen", "scon", "coszr")
_SW_BATCH_COL_KEYS = (
    "play", "plev", "tlay", "tlev", "h2ovmr", "o3vmr", "co2vmr", "ch4vmr",
    "n2ovmr", "o2vmr", "reicmcl", "relqmcl", "resnmcl", "o31d", "pdel")
_SW_BATCH_MCL_KEYS = ("cldfmcl", "taucmcl", "ssacmcl", "asmcmcl",
                      "fsfcmcl", "ciwpmcl", "clwpmcl", "cswpmcl")


def _lw_batch_key(fx):
    """Everything lwrad_prep_batch shares across a batch, per case."""
    warm_rain, f_qi, f_qs, f_qg = _lw_flags(fx)
    return (int(fx["meta/nz"]), int(fx["meta/nlayers"]),
            int(fx["meta/icloud"]), bool(warm_rain),
            int(fx["meta/cldovrlp"]), int(fx["meta/idcor"]),
            int(fx["meta/o3input"]), int(fx["meta/has_reqc"]),
            int(fx["meta/has_reqi"]), int(fx["meta/has_reqs"]),
            bool(f_qi), bool(f_qs), bool(f_qg), int(fx["meta/yr"]),
            float(np.float32(fx["meta/julian"])),
            int(fx["meta/mp_physics"]))


def _lw_case_groups():
    groups = {}
    for path in LW_CASES:
        fx = read_fixture(path)
        groups.setdefault(_lw_batch_key(fx), []).append(
            (Path(path).stem, fx))
    assert sum(len(v) for v in groups.values()) == len(LW_CASES)
    return groups


_LW_STACK_PROFILES = (
    ("p3d", "p3d"), ("p8w", "p8w"), ("t3d", "t3d"), ("t8w", "t8w"),
    ("dz8w", "dz8w"), ("qv3d", "qv"), ("qc3d", "qc"), ("qr3d", "qr"),
    ("qi3d", "qi"), ("qs3d", "qs"), ("qg3d", "qg"),
    ("cldfra3d", "cldfra"), ("o33d", "o33d"), ("re_cloud", "re_cloud"),
    ("re_ice", "re_ice"), ("re_snow", "re_snow"))
_LW_STACK_SURFACE = (("tsk", "tsk"), ("emiss", "emiss"),
                     ("xland", "xland"), ("xice", "xice"),
                     ("snow", "snow"))


def _lw_batch_kwargs(fxs, overrides=None):
    """Stack one flag group's cases into lwrad_prep_batch arguments."""
    kw = {arg: np.stack([fx[f"wrfin/{src}"] for fx in fxs])
          for arg, src in _LW_STACK_PROFILES + _LW_STACK_SURFACE}
    kw["xlat"] = np.stack([fx["meta/xlat"] for fx in fxs])
    scalar = _lw_prep_kwargs(fxs[0])
    for name in ("icloud", "warm_rain", "cldovrlp", "idcor", "o3input",
                 "has_reqc", "has_reqi", "has_reqs", "f_qc", "f_qr",
                 "f_qi", "f_qs", "f_qg", "yr", "julian", "nlayers",
                 "mp_physics", "g"):
        kw[name] = scalar[name]
    if overrides:
        kw.update(overrides)
    return kw


def _check_lw_batch_column(batch, j, ref, label):
    assert set(batch) == set(ref), f"{label}: key set drifted"
    for name in _LW_BATCH_INT_KEYS:
        assert int(batch[name]) == int(ref[name]), f"{label} {name}"
    assert_bits(f"{label} tsfc", np.asarray(batch["tsfc"])[j], ref["tsfc"])
    for name in _LW_BATCH_COL_KEYS:
        assert_bits(f"{label} {name}", np.asarray(batch[name])[j],
                    ref[name])
    for name in _LW_BATCH_MCL_KEYS:
        assert_bits(f"{label} {name}", np.asarray(batch[name])[:, j, :],
                    ref[name])


@lw_skip
def test_lwrad_prep_batch_bitwise_deck():
    """lwrad_prep_batch == lwrad_prep bitwise, per column, all LW cases
    packed as flag-tuple groups."""
    for cases in _lw_case_groups().values():
        fxs = [fx for _, fx in cases]
        batch = prep.lwrad_prep_batch(**_lw_batch_kwargs(fxs))
        assert int(batch["ncol"]) == len(fxs)
        for j, (stem, fx) in enumerate(cases):
            _check_lw_batch_column(batch, j, _run_lw_prep(fx), stem)


@lw_skip
def test_lwrad_outputs_batch_bitwise_deck():
    """lwrad_outputs_batch == lwrad_outputs bitwise, per column, over the
    whole deck's rrtmg_lw fluxes; the clean-sky quartet is exercised with
    stand-in clean fluxes (real flux data re-labelled)."""
    recs = []
    for path in LW_CASES:
        fx = read_fixture(path)
        recs.append((Path(path).stem, {
            "uflx": fx["out/uflx"][0], "dflx": fx["out/dflx"][0],
            "hr": fx["out/hr"][0], "uflxc": fx["out/uflxc"][0],
            "dflxc": fx["out/dflxc"][0], "hrc": fx["out/hrc"][0],
            "pi3d": fx["wrap/pi3d"]}))
    groups = {}
    for stem, r in recs:
        groups.setdefault((r["uflx"].shape, r["pi3d"].shape),
                          []).append((stem, r))
    for members in groups.values():
        kw = {name: np.stack([r[name] for _, r in members])
              for name in ("uflx", "dflx", "hr", "uflxc", "dflxc", "hrc",
                           "pi3d")}
        batch = prep.lwrad_outputs_batch(
            **kw, uflxcln=kw["uflxc"], dflxcln=kw["dflx"],
            calc_clean_atm_diag=1)
        for j, (stem, r) in enumerate(members):
            ref = prep.lwrad_outputs(**r, uflxcln=r["uflxc"],
                                     dflxcln=r["dflx"],
                                     calc_clean_atm_diag=1)
            assert set(batch) == set(ref), f"{stem}: key set drifted"
            for k, v in ref.items():
                assert_bits(f"{stem} {k}", np.asarray(batch[k])[j], v)


def _sw_batch_key(d, c):
    """Everything swrad_prep_batch shares across a batch, per case."""
    return ((int(d[f"{c}/in/p3d"].shape[0]),)
            + tuple(int(d[f"{c}/in/{k}"]) for k in (
                "icloud", "cldovrlp", "idcor", "o3input", "has_reqc",
                "has_reqi", "has_reqs", "mp_physics",
                "sf_surface_physics", "yr"))
            + (float(np.float32(d[f"{c}/in/julian"])),))


def _sw_day_groups():
    d = _sw_fixtures()
    groups = {}
    for c in SW_DAY:
        groups.setdefault(_sw_batch_key(d, c), []).append(c)
    return groups


_SW_STACK_PROFILES = ("p3d", "p8w", "t3d", "t8w", "dz8w", "qv3d", "qc3d",
                      "qr3d", "qi3d", "qs3d", "qg3d", "cldfra3d", "o33d",
                      "re_cloud", "re_ice", "re_snow")
_SW_STACK_SURFACE = ("tsk", "albedo", "xland", "xice", "snow", "xlat",
                     "xcoszen", "solcon", "obscur")


def _sw_batch_kwargs(d, cs, overrides=None):
    kw = {name: np.stack([d[f"{c}/in/{name}"] for c in cs])
          for name in _SW_STACK_PROFILES + _SW_STACK_SURFACE}
    scalar = _sw_prep_kwargs(d, cs[0])
    for name in ("icloud", "warm_rain", "cldovrlp", "idcor", "o3input",
                 "has_reqc", "has_reqi", "has_reqs", "f_qc", "f_qr",
                 "f_qi", "f_qs", "f_qg", "yr", "julian", "mp_physics",
                 "g", "sf_surface_physics"):
        kw[name] = scalar[name]
    if overrides:
        kw.update(overrides)
    return kw


def _check_sw_batch_column(batch, j, ref, label):
    assert set(batch) == set(ref), f"{label}: key set drifted"
    for name in _SW_BATCH_INT_KEYS:
        assert int(batch[name]) == int(ref[name]), f"{label} {name}"
    assert_bits(f"{label} adjes", batch["adjes"], ref["adjes"])
    for name in _SW_BATCH_SURF_KEYS:
        assert_bits(f"{label} {name}", np.asarray(batch[name])[j],
                    ref[name])
    for name in _SW_BATCH_COL_KEYS:
        assert_bits(f"{label} {name}", np.asarray(batch[name])[j],
                    ref[name])
    for name in _SW_BATCH_MCL_KEYS:
        assert_bits(f"{label} {name}", np.asarray(batch[name])[:, j, :],
                    ref[name])
    bm, rm = batch["mcica_inputs"], ref["mcica_inputs"]
    assert set(bm) == set(rm), f"{label}: mcica_inputs keys drifted"
    for name in _SW_MCIN_ARRAYS:
        assert_bits(f"{label} mcin/{name}", np.asarray(bm[name])[j],
                    rm[name])
    for name in _SW_MCIN_INTS:
        assert int(bm[name]) == int(rm[name]), f"{label} mcin/{name}"
    assert_bits(f"{label} mcin/lat", np.asarray(bm["lat"])[j], rm["lat"])


def test_swrad_prep_batch_bitwise_deck():
    """swrad_prep_batch == swrad_prep bitwise, per column, all SW day
    cases packed as flag-tuple groups (mcin record and entry args)."""
    d = _sw_fixtures()
    for cs in _sw_day_groups().values():
        batch = prep.swrad_prep_batch(**_sw_batch_kwargs(d, cs))
        assert int(batch["ncol"]) == len(cs)
        for j, c in enumerate(cs):
            _check_sw_batch_column(batch, j, _run_sw_prep(d, c), c)


def test_swrad_prep_batch_rejects_mixed_day_night():
    """The pre-gathered-day-columns contract: one night column anywhere
    in the batch fails closed, exactly like the scalar entry."""
    d = _sw_fixtures()
    day, night = SW_DAY[0], SW_NIGHT[0]
    with pytest.raises(ValueError, match="night column"):
        prep.swrad_prep_batch(**_sw_batch_kwargs(d, [day, night]))


def test_swrad_prep_batch_rejects_empty():
    d = _sw_fixtures()
    kw = _sw_batch_kwargs(d, SW_DAY[:1])
    kw = {k: (v[:0] if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
          for k, v in kw.items()}
    with pytest.raises(ValueError, match="empty batch"):
        prep.swrad_prep_batch(**kw)


def test_swrad_night_outputs_batch_matches_scalar():
    d = _sw_fixtures()
    cz = np.stack([d[f"{c}/in/xcoszen"] for c in SW_NIGHT])
    batch = prep.swrad_night_outputs_batch(cz, calc_clean_atm_diag=1)
    for j, c in enumerate(SW_NIGHT):
        ref = prep.swrad_night_outputs(d[f"{c}/in/xcoszen"],
                                       calc_clean_atm_diag=1)
        assert set(batch) == set(ref)
        for k, v in ref.items():
            assert_bits(f"{c} night/{k}", np.asarray(batch[k])[j], v)
    for name in prep.SW_NIGHT_UNTOUCHED:
        assert name not in batch


# ---- synthetic width: pure tiling, batch column == scalar result ----

def _tile_kwargs(kw, n):
    return {k: (np.broadcast_to(v, (n,) + v.shape[1:])
                if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
            for k, v in kw.items()}


def _assert_tiled(name, got, want, n, col_axis):
    """Every batch lane bitwise equal to the (broadcast) scalar result;
    col_axis 0 for (ncol, ...) outputs, 1 for (ngpt, ncol, nlay)."""
    got = np.asarray(got)
    want = np.asarray(want, np.float32)
    if col_axis == 0:
        assert got.shape == (n,) + want.shape, (
            f"{name}: shape {got.shape} vs (n,)+{want.shape}")
        wantu = want.view(np.uint32).reshape((1,) + want.shape)
    else:
        assert got.shape == (want.shape[0], n, want.shape[1]), (
            f"{name}: shape {got.shape} vs tiled {want.shape}")
        wantu = want.view(np.uint32)[:, None, :]
    neq = got.view(np.uint32) != wantu
    bad = np.argwhere(neq)
    assert not bad.size, (
        f"{name}: {bad.shape[0]} tiled elements differ, first at "
        f"{tuple(bad[0])}")


@lw_skip
def test_lwrad_prep_batch_width_5000():
    """One fixture column tiled to 5000 batch lanes (no perturbations):
    every lane's outputs equal the scalar result bitwise (shape/index
    bugs at width cannot hide)."""
    fx = _lw_case(LW_CASES[0])
    ref = _run_lw_prep(fx)
    n = 5000
    batch = prep.lwrad_prep_batch(**_tile_kwargs(_lw_batch_kwargs([fx]), n))
    assert int(batch["ncol"]) == n
    for name in _LW_BATCH_INT_KEYS:
        assert int(batch[name]) == int(ref[name]), name
    _assert_tiled("tsfc", batch["tsfc"], ref["tsfc"], n, 0)
    for name in _LW_BATCH_COL_KEYS:
        _assert_tiled(name, batch[name], ref[name], n, 0)
    for name in _LW_BATCH_MCL_KEYS:
        _assert_tiled(name, batch[name], ref[name], n, 1)


def test_swrad_prep_batch_width_5000():
    d = _sw_fixtures()
    c = SW_DAY[0]
    ref = _run_sw_prep(d, c)
    n = 5000
    batch = prep.swrad_prep_batch(**_tile_kwargs(_sw_batch_kwargs(d, [c]),
                                                 n))
    assert int(batch["ncol"]) == n
    for name in _SW_BATCH_INT_KEYS:
        assert int(batch[name]) == int(ref[name]), name
    for name in _SW_BATCH_SURF_KEYS:
        _assert_tiled(name, batch[name], ref[name], n, 0)
    for name in _SW_BATCH_COL_KEYS:
        _assert_tiled(name, batch[name], ref[name], n, 0)
    for name in _SW_BATCH_MCL_KEYS:
        _assert_tiled(name, batch[name], ref[name], n, 1)


# ---- McICA: batch call == per-column calls (generator purity) ----

def test_mcica_generators_batch_equals_per_column():
    """The (ncol, ...)-native generators produce identical per-column
    results at ncol=N vs N calls at ncol=1 when the scalar dummies
    (icld, idcor, juldat, lat) are held fixed: the kissvec seeds are
    per-column and no cross-column state exists.  This is the property
    the batch entries' single whole-batch McICA call relies on."""
    from gpuwm.core.rrtmg_mcica import (
        generate_lw_subcolumns as gen_lw,
        generate_sw_subcolumns as gen_sw,
        NBNDLW as nbl, NBNDSW as nbs,
        LW_PERMUTESEED as pslw, SW_PERMUTESEED as pssw)
    d = _sw_fixtures()
    cs = max(_sw_day_groups().values(), key=len)[:6]
    assert len(cs) >= 2, "need at least two same-flag day columns"
    mi = [_run_sw_prep(d, c)["mcica_inputs"] for c in cs]
    icld, idcor, juldat = mi[0]["icld"], mi[0]["idcor"], mi[0]["juldat"]
    lat0 = np.float32(mi[0]["lat"])
    n = len(cs)
    nlay = mi[0]["play"].shape[0]
    profs = tuple(np.stack([m[k] for m in mi])
                  for k in ("play", "cldfrac", "ciwpth", "clwpth",
                            "cswpth", "rei", "rel", "res"))
    hgt = np.stack([m["hgt"] for m in mi])
    tauc_sw = np.zeros((nbs, n, nlay), np.float32)
    ssac = np.ones((nbs, n, nlay), np.float32)
    asmc = np.zeros((nbs, n, nlay), np.float32)
    fsfc = np.zeros((nbs, n, nlay), np.float32)
    batch_sw = gen_sw(1, n, nlay, icld, pssw, 0, *profs, tauc_sw, ssac,
                      asmc, fsfc, hgt, idcor, juldat, lat0)
    tauc_lw = np.zeros((nbl, n, nlay), np.float32)
    batch_lw = gen_lw(1, n, nlay, icld, pslw, 0, *profs, tauc_lw, hgt,
                      idcor, juldat, lat0)
    for j in range(n):
        one = tuple(a[j:j + 1] for a in profs)
        sw1 = gen_sw(1, 1, nlay, icld, pssw, 0, *one,
                     tauc_sw[:, j:j + 1], ssac[:, j:j + 1],
                     asmc[:, j:j + 1], fsfc[:, j:j + 1], hgt[j:j + 1],
                     idcor, juldat, lat0)
        for k, v in sw1.items():
            got = (batch_sw[k][:, j:j + 1, :] if v.ndim == 3
                   else batch_sw[k][j:j + 1])
            assert_bits(f"sw {k} col{j}", got, v)
        lw1 = gen_lw(1, 1, nlay, icld, pslw, 0, *one, tauc_lw[:, j:j + 1],
                     hgt[j:j + 1], idcor, juldat, lat0)
        for k, v in lw1.items():
            got = (batch_lw[k][:, j:j + 1, :] if v.ndim == 3
                   else batch_lw[k][j:j + 1])
            assert_bits(f"lw {k} col{j}", got, v)


@pytest.mark.parametrize("overlap", [4, 5])
def test_swrad_prep_batch_exponential_overlap_distinct_lats(overlap):
    """The one regime where the generators' scalar lat dummy reaches the
    outputs (idcor=1 with exponential overlaps): the batch entry groups
    columns by xlat bit pattern, and per-column results still equal the
    scalar entry bitwise."""
    d = _sw_fixtures()
    cs = []
    seen = set()
    for c in max(_sw_day_groups().values(), key=len):
        bits = np.float32(d[f"{c}/in/xlat"]).view(np.uint32).item()
        if bits not in seen:
            seen.add(bits)
            cs.append(c)
        if len(cs) == 3:
            break
    assert len(cs) == 3, "need three distinct-latitude day columns"
    over = dict(cldovrlp=overlap, idcor=1)
    batch = prep.swrad_prep_batch(**_sw_batch_kwargs(d, cs, over))
    for j, c in enumerate(cs):
        _check_sw_batch_column(batch, j, _run_sw_prep(d, c, over),
                               f"{c} icld{overlap}")


@lw_skip
@pytest.mark.parametrize("overlap", [4, 5])
def test_lwrad_prep_batch_exponential_overlap_distinct_lats(overlap):
    groups = _lw_case_groups()
    cases = max(groups.values(), key=len)
    picked = []
    seen = set()
    for stem, fx in cases:
        bits = np.float32(fx["meta/xlat"]).view(np.uint32).item()
        if bits not in seen:
            seen.add(bits)
            picked.append((stem, fx))
        if len(picked) == 3:
            break
    assert len(picked) == 3, "need three distinct-latitude LW columns"
    over = dict(cldovrlp=overlap, idcor=1)
    fxs = [fx for _, fx in picked]
    batch = prep.lwrad_prep_batch(**_lw_batch_kwargs(fxs, over))
    for j, (stem, fx) in enumerate(picked):
        _check_lw_batch_column(batch, j, _run_lw_prep(fx, over),
                               f"{stem} icld{overlap}")


# ---- cupy path: device batch == numpy batch, bitwise ----

def _cupy_or_skip():
    """cupy module, or skip with the reason (unavailable / GPU already
    loaded beyond the machine-wide budget -- the integration lead runs
    the gates then)."""
    if os.environ.get("GPUWM_LEGACY_PREP_SKIP_CUPY"):
        pytest.skip("cupy gates disabled via GPUWM_LEGACY_PREP_SKIP_CUPY")
    try:
        import cupy
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"cupy unavailable: {exc}")
    try:
        ndev = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if ndev < 1:                                  # pragma: no cover
        pytest.skip("no CUDA device")
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60)
        used_mib = sum(float(tok) for tok in r.stdout.split())
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"nvidia-smi precheck unavailable: {exc}")
    if used_mib > 24 * 1024:                      # pragma: no cover
        pytest.skip(f"GPU loaded machine-wide ({used_mib:.0f} MiB used "
                    "> 24 GiB budget); cupy gates deferred")
    return cupy


def _to_device(cupy, kw):
    return {k: (cupy.asarray(v)
                if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
            for k, v in kw.items()}


def _assert_device_dict(got, want, label):
    """Every array bitwise identical after device->host transfer; ints
    equal; nested dicts recursed."""
    assert set(got) == set(want), f"{label}: key set drifted"
    for k, w in want.items():
        g = got[k]
        if isinstance(w, dict):
            _assert_device_dict(g, w, f"{label}/{k}")
            continue
        if isinstance(w, (int, np.integer)):
            assert int(g) == int(w), f"{label}/{k}"
            continue
        if not isinstance(g, np.ndarray) and hasattr(g, "get"):
            g = g.get()
        assert_bits(f"{label}/{k}", g, w)


def test_swrad_prep_batch_cupy_bitwise_deck():
    """cupy == numpy bitwise for swrad_prep_batch over the SW day deck.
    A mismatch here is a real finding (cupy-compiled device arithmetic
    flushes FP32 subnormal results): report it, never widen it."""
    cupy = _cupy_or_skip()
    d = _sw_fixtures()
    for key, cs in _sw_day_groups().items():
        kw = _sw_batch_kwargs(d, cs)
        want = prep.swrad_prep_batch(**kw)
        got = prep.swrad_prep_batch(**_to_device(cupy, kw))
        _assert_device_dict(got, want, f"cupy sw group {key[:4]}")


@lw_skip
def test_lwrad_batch_cupy_bitwise_deck():
    """cupy == numpy bitwise for lwrad_prep_batch (all LW flag groups)
    and lwrad_outputs_batch (whole deck)."""
    cupy = _cupy_or_skip()
    for key, cases in _lw_case_groups().items():
        fxs = [fx for _, fx in cases]
        kw = _lw_batch_kwargs(fxs)
        want = prep.lwrad_prep_batch(**kw)
        got = prep.lwrad_prep_batch(**_to_device(cupy, kw))
        _assert_device_dict(got, want, f"cupy lw group {key[:5]}")
    fx = _lw_case(LW_CASES[0])
    kw = {"uflx": fx["out/uflx"], "dflx": fx["out/dflx"],
          "hr": fx["out/hr"], "uflxc": fx["out/uflxc"],
          "dflxc": fx["out/dflxc"], "hrc": fx["out/hrc"],
          "pi3d": fx["wrap/pi3d"][None, :]}
    want = prep.lwrad_outputs_batch(**kw)
    got = prep.lwrad_outputs_batch(**_to_device(cupy, kw))
    _assert_device_dict(got, want, "cupy lw outputs")


def test_has_req_flags_require_radii_arrays():
    """Audit hardening: has_req* != 0 with a missing radius array fails
    closed in every prep entry instead of silently radiating zeros."""
    import numpy as np
    kte = 4
    prof = np.linspace(900.0, 500.0, kte).astype(np.float32)
    common = dict(
        p3d=prof * 100.0, p8w=np.linspace(1000.0, 450.0, kte + 1
                                          ).astype(np.float32) * 100.0,
        t3d=np.full(kte, 270.0, np.float32),
        t8w=np.full(kte + 1, 270.0, np.float32),
        dz8w=np.full(kte, 300.0, np.float32),
        qv3d=np.full(kte, 1e-3, np.float32),
        tsk=np.float32(285.0), emiss=np.float32(0.95),
        xland=np.float32(1.0), xice=np.float32(0.0),
        snow=np.float32(0.0), xlat=np.float32(35.0),
        icloud=1, warm_rain=False, cldovrlp=2, idcor=0, o3input=2,
        yr=1990, julian=100.0)
    for flags in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        with pytest.raises(ValueError, match="fail-closed"):
            prep.lwrad_prep(has_reqc=flags[0], has_reqi=flags[1],
                            has_reqs=flags[2], nlayers=kte + 26,
                            o33d=np.full(kte, 5e-8, np.float32), **common)
