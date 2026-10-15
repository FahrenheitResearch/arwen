"""Noah sea ice against the byte-unmodified WRF v4.6.1 ``seaice_noah``.

WRF's surface driver calls ``seaice_noah`` (phys/module_sf_noah_seaice_drv.F)
after ``lsm`` / ``lsm_mosaic`` on every Noah run
(phys/module_surface_driver.F:2917-2929).  ``lsm`` only sets SH2O = 1 and
LAI = 0.01 on a sea-ice column and returns; until 2.8.6 gpuwm had nothing
after it, so a sea-ice column kept its initial SKINTEMP for the whole
forecast while the surface layer kept reading that frozen skin (PROVENANCE
D7).  ``noah_seaice_column`` in ``gpuwm/core/kernels/noah.cu`` now ports the
driver and ``SFLX_SEAICE``'s subtree on WRF's default namelist path
(seaice_albedo_opt, seaice_thickness_opt and seaice_snowdepth_opt all 0,
fractional_seaice 0), with gfortran -O0's rounding and glibc's expf/powf.

``tools/noah_seaice_wrf461_oracle/build.sh`` drives ``seaice_noah`` over 24
columns, one per branch of the scheme, in three fixtures: the default, frpcpn
on, and seaice_albedo_default 0.75.  This module measures the kernel against
those words and pins what it measures, for equality.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.noah_seaice_oracle import (
    SEAICE_ORACLE_DIR,
    SEAICE_ORACLE_FILES,
    SEAICE_STATE,
    load_seaice_oracle,
    seaice_port_outputs,
)

#: Worst ULP distance from ``noah_seaice_column`` to the word ``seaice_noah``
#: wrote, over every column of the three fixtures.  MEASURED 2026-10-05 on an
#: RTX PRO 6000 (sm_120).
BASELINE_MAX_ULP = {name: 0 for name in (*SEAICE_STATE, "tslb")}

#: Column 13 is open water by the threshold (XICE 0.4); 14 sits exactly on it.
BELOW_THRESHOLD_CASE = 13
AT_THRESHOLD_CASE = 14


def test_fixture_is_the_pinned_wrf_driver_and_its_own_receipts():
    receipts = (SEAICE_ORACLE_DIR / "oracle-sha256sums.txt").read_text(
        encoding="ascii").splitlines()
    recorded = {}
    for line in receipts:
        digest, path = line.split(maxsplit=1)
        recorded[path.strip().rsplit("/", 1)[-1]] = digest
    for name in SEAICE_ORACLE_FILES:
        blob = (SEAICE_ORACLE_DIR / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == recorded[name], name
    joined = " ".join(receipts)
    for expected in ("module_sf_noah_seaice.F", "module_sf_noah_seaice_drv.F",
                     "module_sf_noahlsm.F", "module_model_constants.F",
                     "module_wrf_error.F"):
        assert expected in joined, expected


def test_the_switch_fixtures_discriminate_their_switch():
    base = load_seaice_oracle()
    for name in ("noah-seaice-frpcpn.csv", "noah-seaice-alb075.csv"):
        other = load_seaice_oracle(name)
        moved = [k for k in base.reference
                 if base.reference[k].tobytes() != other.reference[k].tobytes()]
        assert moved, f"{name} answers exactly as the default fixture"
    alb = load_seaice_oracle("noah-seaice-alb075.csv").reference["albedo"]
    ice = load_seaice_oracle().inputs["xice"] >= 0.5
    assert np.all(alb[ice] == np.float32(0.75))


def test_fixture_covers_the_branches_that_matter():
    fx = load_seaice_oracle()
    ref, inp = fx.reference, fx.inputs
    smallest = np.float32(np.finfo(np.float32).smallest_subnormal)
    assert fx.cases == tuple(range(1, 25))
    # cold (T12 <= TFREEZ) and melting (T1 pinned at TFREEZ) columns
    assert (ref["tsk"] < np.float32(273.15)).any()
    assert (ref["tsk"] == np.float32(273.15)).any()
    # melt reaching the snow-out reset to 0.01 m SWE / 0.05 m depth
    assert (ref["acsnom"] > 0).any()
    # subnormal SNOW and SNOWC probes
    assert (inp["snow"] == smallest).any() and (inp["snowc"] == smallest).any()
    # precipitation: snow, freezing rain, rain
    assert (inp["rainbl"] > 0).sum() >= 6
    # the threshold column is untouched, the one on the threshold is not
    i13 = fx.cases.index(BELOW_THRESHOLD_CASE)
    i14 = fx.cases.index(AT_THRESHOLD_CASE)
    for k in SEAICE_STATE:
        assert ref[k][0, i13].tobytes() == inp[k][0, i13].tobytes(), k
    assert ref["tsk"][0, i14] != inp["tsk"][0, i14]


@pytest.mark.gpu
@requires_gpu
def test_noah_seaice_column_holds_its_measured_distance_from_wrf():
    import cupy  # noqa: F401

    measured: dict[str, int] = {}
    for name in SEAICE_ORACLE_FILES:
        fx = load_seaice_oracle(name)
        port = seaice_port_outputs(fx)
        for k in (*SEAICE_STATE, "tslb"):
            d = fp32_ulp_distance(np.ascontiguousarray(port[k], np.float32),
                                  np.ascontiguousarray(fx.reference[k]))
            measured[k] = max(measured.get(k, 0), int(d.max()))
    assert measured == BASELINE_MAX_ULP, (
        "noah_seaice_column moved from WRF's seaice_noah.\n"
        f"  measured {measured}")


@pytest.mark.gpu
@requires_gpu
def test_an_impossible_initial_snowpack_is_refused_with_wrfs_words():
    """SFLX_SEAICE's one fatal: SNOW/1000 > SNOWH after the floor."""
    import cupy as cp

    from gpuwm.core.noah import launch_noah_seaice

    fx = load_seaice_oracle()
    device = {k: cp.asarray(v.copy()) for k, v in fx.inputs.items()}
    device["snow"][0, 0] = np.float32(200.0)       # 0.2 m SWE
    device["snowh"][0, 0] = np.float32(0.1)        # in 0.1 m of snow
    with pytest.raises(ValueError, match="less than snow water equiv"):
        launch_noah_seaice(device, 60.0, frpcpn=False,
                           check_snow_density=True)
