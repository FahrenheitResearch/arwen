"""mp=28's own snow and graupel fallout, surface totals and 10 cm echo.

The breakage this file prevents.  mp=28 used to run thompson.cu's classic
snow and graupel fallout, its classic graupel-number entry and exit, and
refl.cu's classic calc_refl10cm.  Those units are byte-frozen for mp=8 and
their arithmetic is not WRF v4.6.1's: CUDA's powf/pow/log10f, a graupel
slope taken from the intercept with a size clamp WRF does not apply, the
substep factor folded as DT*onstep, crg(4) = 720 where thompson_init leaves
720.000061, binary64 slopes and radar constants where WRF forms REAL(4) ones,
and the four surface totals summed in kernel order rather than
mp_gt_driver's.  The 0 ULP column oracle (tools/thompson_aerosol_column_
oracle) measured REFL_10CM off WRF at 3,492 of 7,497 cells and a 1.6e-4
relative miss on a high-terrain column's RAINNC.

CPU tests pin the transcription's constants to WRF's own thompson_init and
radar_init words and keep the adapter off the frozen kernels.  The GPU tests
hold the echo to WRF's calc_refl10cm word for word on the committed fixture
and the surface totals to mp_gt_driver's order of addition.
"""

from __future__ import annotations

import pathlib
import re
import struct

import numpy as np
import pytest

from conftest import requires_gpu

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_KERNELS = _ROOT / "gpuwm" / "core" / "kernels"
_DATA = _ROOT / "tests" / "data"
_CONSTANTS = _DATA / "thompson_wrf461_init_constants.txt"
_REFL_FIXTURE = _DATA / "thompson_aerosol_refl_wrf461.npz"


def _wrf_words():
    """name, index -> the IEEE word thompson_init/radar_init leave."""
    words = {}
    for line in _CONSTANTS.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, index, bits = line.split()[:3]
        if len(bits) == 8:
            value = np.uint32(int(bits, 16)).view(np.float32)
        else:
            value = struct.unpack(">d", bytes.fromhex(bits))[0]
        words[(name, int(index))] = value
    return words


def _macros():
    """THOMPSON_AA_* numeric macros of the mp=28 sources, as their words."""
    text = "".join((_KERNELS / name).read_text(encoding="utf-8") for name in (
        "thompson_aerosol_common.cuh", "thompson_aerosol_state.cu",
        "thompson_aerosol_sed.cu"))
    found = {}
    for m in re.finditer(
            r"^#define\s+(THOMPSON_AA_\w+)\s+\(?(-?[0-9][0-9a-fA-FxXpP.+-]*)"
            r"\)?\s*(?://.*)?$", text, flags=re.M):
        name, literal = m.groups()
        if "x" in literal.lower():
            found[name] = float.fromhex(literal)
        elif literal.endswith(("f", "F")):
            found[name] = np.float32(literal[:-1])
        else:
            found[name] = float(literal)
    return found


#: Header constant -> (thompson_init / radar_init name, index).
_FLOAT_WORDS = {
    "THOMPSON_AA_CSE1": ("cse", 1), "THOMPSON_AA_CSG1": ("csg", 1),
    "THOMPSON_AA_CSE3": ("cse", 3),
    "THOMPSON_AA_CSE4": ("cse", 4), "THOMPSON_AA_CSG4": ("csg", 4),
    "THOMPSON_AA_CSE7": ("cse", 7), "THOMPSON_AA_CSG7": ("csg", 7),
    "THOMPSON_AA_CSE10": ("cse", 10), "THOMPSON_AA_CSG10": ("csg", 10),
    "THOMPSON_AA_CGG1": ("cgg", 1), "THOMPSON_AA_CGG2": ("cgg", 2),
    "THOMPSON_AA_CGG3": ("cgg", 3), "THOMPSON_AA_CGG4": ("cgg", 4),
    "THOMPSON_AA_CGG6": ("cgg", 6), "THOMPSON_AA_CGG7": ("cgg", 7),
    "THOMPSON_AA_CGG12": ("cgg", 12), "THOMPSON_AA_CGE2": ("cge", 2),
    "THOMPSON_AA_CGE4": ("cge", 4), "THOMPSON_AA_AM_G5": ("am_g", 5),
    "THOMPSON_AA_OGG1": ("ogg1", 0), "THOMPSON_AA_OGG2": ("ogg2", 0),
    "THOMPSON_AA_OGG3": ("ogg3", 0), "THOMPSON_AA_OGE1": ("oge1", 0),
    "THOMPSON_AA_OBMG": ("obmg", 0), "THOMPSON_AA_CRG3": ("crg", 3),
    "THOMPSON_AA_CRG4": ("crg", 4), "THOMPSON_AA_CRE2": ("cre", 2),
    "THOMPSON_AA_CRE4": ("cre", 4), "THOMPSON_AA_ORG2": ("org2", 0),
    "THOMPSON_AA_OCMS": ("ocms", 0), "THOMPSON_AA_OBMS": ("obms", 0),
    "THOMPSON_AA_OAMS": ("oams", 0), "THOMPSON_AA_OBMR": ("obmr", 0),
    "THOMPSON_AA_RHO_NOT": ("RHO_NOT", 0), "THOMPSON_AA_AM_S": ("am_s", 0),
    "THOMPSON_AA_AM_R": ("am_r", 0), "THOMPSON_AA_MU_S": ("mu_s", 0),
    "THOMPSON_AA_KAP0": ("Kap0", 0), "THOMPSON_AA_KAP1": ("Kap1", 0),
    "THOMPSON_AA_LAM0": ("Lam0", 0), "THOMPSON_AA_LAM1": ("Lam1", 0),
    "THOMPSON_AA_AV_S": ("av_s", 0), "THOMPSON_AA_FV_S": ("fv_s", 0),
    "THOMPSON_AA_AV_G": ("av_g_old", 0), "THOMPSON_AA_BV_G": ("bv_g_old", 0),
    "THOMPSON_AA_PI": ("PI", 0),
}
_DOUBLE_WORDS = {
    "THOMPSON_AA_RADAR_PI5": ("PI5", 0),
    "THOMPSON_AA_RADAR_LAMDA4": ("lamda4", 0),
    "THOMPSON_AA_RADAR_K_W": ("K_w", 0),
    "THOMPSON_AA_RADAR_MW_RE": ("m_w_0re", 0),
    "THOMPSON_AA_RADAR_MW_IM": ("m_w_0im", 0),
    "THOMPSON_AA_RADAR_MI_RE": ("m_i_0re", 0),
    "THOMPSON_AA_RADAR_MI_IM": ("m_i_0im", 0),
}


@pytest.mark.parametrize("macro", sorted(_FLOAT_WORDS))
def test_every_real4_constant_is_thompson_inits_own_word(macro):
    """crg(4) = cgg(4,1) = 720.000061, not 720: thompson_init forms the
    gamma moments at run time as WGAMMA = EXP(GAMMLN(x)) in REAL(4)."""
    want = _wrf_words()[_FLOAT_WORDS[macro]]
    got = np.float32(_macros()[macro])
    assert got.view(np.uint32) == np.float32(want).view(np.uint32), (
        macro, got, want)


@pytest.mark.parametrize("macro", sorted(_DOUBLE_WORDS))
def test_every_radar_constant_is_radar_inits_own_word(macro):
    """PI5 and lamda4 are formed from REAL(4) literals and m_w_0, m_i_0, K_w
    from COMPLEX(4) ones (module_mp_radar.F :76-80); a binary64 evaluation
    of the same formulas is a different word."""
    want = _wrf_words()[_DOUBLE_WORDS[macro]]
    assert _macros()[macro] == want, (macro, _macros()[macro], want)


def test_the_size_bins_and_simpson_weights_are_radar_inits_own_words():
    """The echo reads xxDs, xdts and the Simpson weights from
    gpuwm.core.refl's device table; they must be WRF's words."""
    from gpuwm.core.refl import radar_init
    words = _wrf_words()
    rc = radar_init()
    for name, table in (("xxDs", rc.xxds), ("xdts", rc.xdts),
                        ("simpson", rc.simpson)):
        want = np.array([words[(name, n + 1)] for n in range(50)])
        assert np.array_equal(np.asarray(table)[:50].view(np.int64),
                              want.view(np.int64)), name


def test_the_v461_adapter_reaches_none_of_the_frozen_mp8_kernels():
    """The classic fallout, graupel-number entry and exit and echo are mp=8's
    and stay frozen; the v4.6.1 path must not call them (the fork
    generation keeps its own passes)."""
    src = (_ROOT / "gpuwm" / "core" / "microphysics_aerosol.py").read_text(
        encoding="utf-8")
    for name in ("launch_snow_sedimentation(", "launch_graupel_sedimentation(",
                 "launch_classic_graupel_number_init(",
                 "launch_classic_graupel_number_finalize(",
                 "launch_ice_sedimentation("):
        assert not re.search(r"(?<![A-Za-z_])" + re.escape(name), src), name
    for name in ("launch_aa_snow_sedimentation(",
                 "launch_aa_graupel_sedimentation(",
                 "launch_aa_surface_precipitation(",
                 "launch_aa_graupel_number_init(",
                 "launch_aa_graupel_number_finalize("):
        assert name in src, name
    refl = (_ROOT / "gpuwm" / "core" / "refl.py").read_text(encoding="utf-8")
    assert "launch_aa_refl10cm(" in refl


# ---------------------------------------------------------------------------
# GPU.
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_echo_is_wrfs_calc_refl10cm_word_for_word():
    """72 columns of the column oracle's WRF end states (some 2 K and 5 K
    warmer), half of them with melting snow below a melting level, run
    through WRF v4.6.1's own calc_refl10cm and through WOOF's kernel on the
    same inputs (tools/thompson_aerosol_column_oracle/refl_check.py)."""
    import cupy as cp
    from gpuwm.core.thompson_aerosol_state import launch_aa_refl10cm
    x = dict(np.load(_REFL_FIXTURE))
    want = x.pop("refl_wrf")
    ncol, nz = x["t"].shape
    dev = {k: cp.asarray(np.ascontiguousarray(v.T).reshape(nz, 1, ncol))
           for k, v in x.items()}
    refl = cp.zeros((nz, 1, ncol), dtype=cp.float32)
    launch_aa_refl10cm(dev["qv"], dev["qr"], dev["nr"], dev["qs"], dev["qg"],
                       dev["ng"], dev["t"], dev["p"], refl)
    got = cp.asnumpy(refl).reshape(nz, ncol).T
    bad = np.argwhere(got.view(np.int32) != want.view(np.int32))
    assert bad.size == 0, (len(bad), bad[:5].tolist(),
                           got[tuple(bad[0])], want[tuple(bad[0])])


@requires_gpu
def test_the_melting_term_is_live_on_the_fixture():
    """Without the melting-snow term the same columns are a different echo:
    the fixture's melting half exercises the complex backscatter."""
    import cupy as cp
    from gpuwm.core.thompson_aerosol_state import launch_aa_refl10cm
    x = dict(np.load(_REFL_FIXTURE))
    want = x.pop("refl_wrf")
    ncol, nz = x["t"].shape
    dev = {k: cp.asarray(np.ascontiguousarray(v.T).reshape(nz, 1, ncol))
           for k, v in x.items()}
    refl = cp.zeros((nz, 1, ncol), dtype=cp.float32)
    launch_aa_refl10cm(dev["qv"], dev["qr"], dev["nr"], dev["qs"], dev["qg"],
                       dev["ng"], dev["t"], dev["p"], refl, melting=False)
    dry = cp.asnumpy(refl).reshape(nz, ncol).T
    assert np.count_nonzero(dry.view(np.int32) != want.view(np.int32)) > 50


@requires_gpu
def test_the_surface_totals_add_in_mp_gt_drivers_order():
    """RAINNCV = pptrain + pptsnow + pptgraul + pptice, left to right, and
    likewise the accumulations and SR (mp_gt_driver :1294-1308).  The
    values are chosen so another order rounds differently."""
    import cupy as cp
    from gpuwm.core.thompson_aerosol_sed import launch_aa_surface_precipitation
    f32 = np.float32
    rng = np.random.default_rng(1308)
    n = 4096
    ppt = {k: (rng.uniform(0, 1, n)
               * 10.0 ** rng.integers(-9, 1, n)).astype(f32)
           for k in ("rain", "snow", "graupel", "ice")}
    acc = {k: (rng.uniform(0, 50, n)).astype(f32)
           for k in ("rainnc", "snownc", "graupelnc")}
    dev = {"rainnc": cp.asarray(acc["rainnc"]),
           "rainncv": cp.asarray(ppt["snow"]),
           "snownc": cp.asarray(acc["snownc"]),
           "snowncv": cp.asarray(ppt["ice"]),
           "graupelnc": cp.asarray(acc["graupelnc"]),
           "graupelncv": cp.asarray(ppt["graupel"]),
           "sr": cp.asarray(ppt["rain"])}
    launch_aa_surface_precipitation(
        dev["rainnc"], dev["rainncv"], dev["snownc"], dev["snowncv"],
        dev["graupelnc"], dev["graupelncv"], dev["sr"])
    r, s, g, i = ppt["rain"], ppt["snow"], ppt["graupel"], ppt["ice"]
    rainncv = ((r + s) + g) + i
    want = {
        "rainncv": rainncv,
        "rainnc": (((acc["rainnc"] + r) + s) + g) + i,
        "snowncv": s + i,
        "snownc": (acc["snownc"] + s) + i,
        "graupelncv": g,
        "graupelnc": acc["graupelnc"] + g,
        "sr": ((s + g) + i) / (rainncv + f32(1.0e-12)),
    }
    for name, value in want.items():
        got = cp.asnumpy(dev[name])
        assert np.array_equal(got.view(np.int32),
                              value.astype(f32).view(np.int32)), name
    # The negative control: the kernels' old order, ice first, is another
    # number on these values.
    old = (((acc["rainnc"] + i) + s) + g) + r
    assert not np.array_equal(old.view(np.int32),
                              want["rainnc"].view(np.int32))
