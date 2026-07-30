"""Pinned mass-chain parity gate against retained real.exe wrfinput files.

Environmental, CPU-only: it consumes the read-only reference bundle's WPS
output (surface slots and full met columns), the repo-local reference-run
SOILHGT artifacts, and locally retained copies of the reference real.exe
``wrfinput_d0N`` files.  Any missing input skips, like the suite's other
environmental gates.  The gate pins the derivation this lane closed: WRF
computes ``psfc = met PSFC * exp(g*(SOILHGT-HGT)/(Rd*Tv1))`` (sfcprs2,
``module_initialize_real.F:1416-1425``, ``:8443-8501``; the id>=3 bypass in
the group source never fired in the retained artifacts) and the dry column
mass ``mu0 = psfc - intq - p_top`` (``p_dts``, ``:1482``/``:6758-6787``)
with ``intq`` integrated on the ORIGINAL met surface (``integ_moist``,
``:1457``/``:6961-7153``).  gpuwm's float64 host chain must reproduce the
FP32 reference to rounding-floor tolerances; regressions of the
source-orography data (for example a SOILHGT byte-equal to HGT_M) blow
these bounds by four orders of magnitude.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gpuwm.case_data import load_experiment_case  # noqa: E402
from gpuwm.ingest.real import (  # noqa: E402
    _integrate_moisture,
    _saturation_mixing_ratio,
    surface_pressure_from_surface,
)

REPO = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
CONFIG = REPO / "configs" / "real74_4dom.toml"

_WRFINPUT_CANDIDATES = tuple(
    Path(root) for root in (
        os.environ.get("GPUWM_REAL74_WRFINPUT_DIR", ""),
        str(BUNDLE / "wrfinput_reference"),
    ) if root
)


def _wrfinput_dir() -> Path | None:
    for candidate in _WRFINPUT_CANDIDATES:
        if (candidate / "wrfinput_d01").is_file():
            return candidate
    return None


WRFINPUT_DIR = _wrfinput_dir()

_INPUTS_MISSING = (
    WRFINPUT_DIR is None
    or not (BUNDLE / "met_em" / "met_em.d01.1974-04-03_12_00_00.nc").is_file()
    or not (REPO / "configs" / "real74" / "soilhgt.d01.nc").is_file()
)
# A gate that can silently vanish is not a gate: assembly/ratification runs
# set GPUWM_REQUIRE_CASE_GATES=1, which converts the environmental skip
# into a hard failure when the pinned inputs are absent.
_REQUIRE_GATES = os.environ.get("GPUWM_REQUIRE_CASE_GATES") == "1"

requires_inputs = pytest.mark.skipif(
    _INPUTS_MISSING and not _REQUIRE_GATES,
    reason="retained real.exe wrfinput copies / reference bundle absent",
)


@pytest.fixture(autouse=True)
def _fail_when_gates_are_required():
    if _REQUIRE_GATES and _INPUTS_MISSING:
        pytest.fail(
            "GPUWM_REQUIRE_CASE_GATES=1 but the retained wrfinput copies / "
            "reference bundle / repo source-orography artifacts are absent; "
            "this gate must run in ratification environments")

# FP32 rounding floor of the reference chain (measured 2026-07-27:
# PSFC MAE 0.0028-0.0029 Pa, max <= 0.0099; MU MAE 0.0034, max <= 0.0135).
PSFC_MAE_PA = 0.005
PSFC_MAX_PA = 0.02
MU_MAE_PA = 0.007
MU_MAX_PA = 0.03


@requires_inputs
@pytest.mark.parametrize("grid_id", (1, 2, 3))
def test_mass_chain_matches_retained_wrfinput_to_fp32_floor(grid_id):
    met_em = (BUNDLE / "met_em" /
              f"met_em.d{grid_id:02d}.1974-04-03_12_00_00.nc")
    with netCDF4.Dataset(met_em) as source:
        tt = np.asarray(source["TT"][0], dtype=np.float64)
        rh = np.asarray(source["RH"][0], dtype=np.float64)
        ght = np.asarray(source["GHT"][0], dtype=np.float64)
        pres = np.asarray(source["PRES"][0], dtype=np.float64)
        psfc = np.asarray(source["PSFC"][0], dtype=np.float64)
        hgt = np.asarray(source["HGT_M"][0], dtype=np.float64)

    _, data = load_experiment_case(CONFIG)
    artifact = data.source_orography_for_domain(grid_id)
    with netCDF4.Dataset(artifact.path) as declared:
        soilhgt = np.asarray(
            declared[artifact.variable][0], dtype=np.float64)
    assert soilhgt.shape == hgt.shape
    assert not np.array_equal(soilhgt, hgt), (
        "source orography degenerated to the model terrain; the "
        "sfcp_to_sfcp remap would silently vanish")

    qsfc = _saturation_mixing_ratio(tt[0], psfc, rh[0])
    qv_upper = _saturation_mixing_ratio(tt[1:], pres[1:], rh[1:])
    psfc_adjusted = surface_pressure_from_surface(
        psfc, soilhgt, hgt, tt[0], qsfc)
    _, intq, _ = _integrate_moisture(
        qv_upper, pres[1:], tt[1:], ght[1:], psfc, tt[0], qsfc, soilhgt)

    with netCDF4.Dataset(WRFINPUT_DIR / f"wrfinput_d{grid_id:02d}") as ref:
        psfc_ref = np.asarray(ref["PSFC"][0], dtype=np.float64)
        dry_ref = (np.asarray(ref["MU"][0], dtype=np.float64)
                   + np.asarray(ref["MUB"][0], dtype=np.float64))
        p_top = float(np.asarray(ref["P_TOP"][:]).ravel()[0])

    psfc_err = np.abs(psfc_adjusted - psfc_ref)
    assert psfc_err.mean() <= PSFC_MAE_PA
    assert psfc_err.max() <= PSFC_MAX_PA

    dry_err = np.abs((psfc_adjusted - intq - p_top) - dry_ref)
    assert dry_err.mean() <= MU_MAE_PA
    assert dry_err.max() <= MU_MAX_PA


# Soil/skin transcription floor (measured 2026-07-27, d01 land: TSK MAE
# 1e-5 K, TSLB MAE 2e-5 K max 1e-4, SMOIS exact).
TSK_MAE_K = 1.0e-3
TSLB_MAE_K = 2.0e-3
TSLB_MAX_K = 2.0e-2
SMOIS_MAX = 1.0e-5


@requires_inputs
@pytest.mark.parametrize("grid_id", (1,))
def test_soil_chain_matches_retained_wrfinput_to_transcription_floor(grid_id):
    from gpuwm.ingest.soil import (
        _interp_nodes,
        _soil_temperature_elevation_delta,
        ERA5_LAYER_MIDPOINTS_M,
        NOAH_LAYER_MIDPOINTS_M,
    )

    met_em = (BUNDLE / "met_em" /
              f"met_em.d{grid_id:02d}.1974-04-03_12_00_00.nc")
    layers = ("000007", "007028", "028100", "100289")
    with netCDF4.Dataset(met_em) as source:
        st = [np.asarray(source[f"ST{n}"][0], dtype=np.float64)
              for n in layers]
        sm = [np.asarray(source[f"SM{n}"][0], dtype=np.float64)
              for n in layers]
        skin = np.asarray(source["SKINTEMP"][0], dtype=np.float64)
        hgt = np.asarray(source["HGT_M"][0], dtype=np.float64)

    _, data = load_experiment_case(CONFIG)
    artifact = data.source_orography_for_domain(grid_id)
    with netCDF4.Dataset(artifact.path) as declared:
        soilhgt = np.asarray(
            declared[artifact.variable][0], dtype=np.float64)

    with netCDF4.Dataset(WRFINPUT_DIR / f"wrfinput_d{grid_id:02d}") as ref:
        tslb_ref = np.asarray(ref["TSLB"][0], dtype=np.float64)
        smois_ref = np.asarray(ref["SMOIS"][0], dtype=np.float64)
        tsk_ref = np.asarray(ref["TSK"][0], dtype=np.float64)
        tmn_ref = np.asarray(ref["TMN"][0], dtype=np.float64)
        land = np.asarray(ref["LANDMASK"][0], dtype=np.float64) > 0.5

    delta = _soil_temperature_elevation_delta(hgt, soilhgt, land)
    skin_adjusted = skin + delta
    tsk_err = np.abs(skin_adjusted - tsk_ref)[land]
    assert tsk_err.mean() <= TSK_MAE_K

    nodes_z = np.concatenate(([0.0], ERA5_LAYER_MIDPOINTS_M, [3.0]))
    temp_nodes = np.stack(
        [skin_adjusted, *(value + delta for value in st), tmn_ref])
    moist_nodes = np.stack([sm[0], *sm, sm[-1]])
    tslb = _interp_nodes(temp_nodes, nodes_z, NOAH_LAYER_MIDPOINTS_M)
    smois = _interp_nodes(moist_nodes, nodes_z, NOAH_LAYER_MIDPOINTS_M)
    tslb_err = np.abs(tslb - tslb_ref)[:, land]
    smois_err = np.abs(smois - smois_ref)[:, land]
    assert tslb_err.mean() <= TSLB_MAE_K
    assert tslb_err.max() <= TSLB_MAX_K
    assert smois_err.max() <= SMOIS_MAX
