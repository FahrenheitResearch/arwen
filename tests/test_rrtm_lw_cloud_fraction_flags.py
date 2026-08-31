"""The RRTM longwave adapter resolves cal_cldfra1's flags per scheme.

WHAT THIS PREVENTS.  ``gpuwm/core/rrtm_lw.py`` (the ``ra_lw_physics=1``
adapter) used to hardcode ``f_qi=True, f_qs=True`` into ``cal_cldfra1``
for EVERY microphysics scheme, zero-filling absent species.  For a scheme
whose Registry package declares both ``qi`` and ``qs`` that spelling is
WRF's own arm, and for P3 (``qi`` and no ``qs``) the zero-filled ``qs``
collapses the Morrison-arm weights to the P3 arm exactly -- but for a
Kessler-class package (``qc`` only) it is a real divergence: WRF's driver
hands cal_cldfra1 the Registry flags (module_radiation_driver.F:
1326-1332), and the qc-only arm at :3891-3899 weights a sub-freezing
liquid cloud with the ICE saturation (``weight = 1`` at T <= 273.15 K),
where the hardcoded spelling reached the Morrison arm at :3870-3877 whose
weight collapses to 0 (liquid saturation).  A supercooled Kessler cloud
sitting between ice and water saturation therefore radiated as PARTIAL
cloud where WRF says overcast.

The remedy is the same split the legacy-RRTMG sibling landed: resolve
``(F_QI, F_QS)`` from ``legacy_cloud_fraction_flags`` -- the Registry
package membership helper in ``gpuwm/core/rrtmg_legacy.py`` -- never a
re-spelled table and never literals.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.core.rrtmg_legacy as rrtmg_legacy
from gpuwm.core import rrtm_lw
from gpuwm.core.rrtm_lw import RRTMLongwaveRadiation
from gpuwm.core.rrtmg_legacy import legacy_cloud_fraction_flags
from gpuwm.verify.npref import np_cal_cldfra1

F = np.float32
NZ = 40
_REAL_RRTM_LONGWAVE_COLUMNS = rrtm_lw.rrtm_longwave_columns
KESSLER_MP_PHYSICS = 1
P3_MP_PHYSICS = 50

#: Bottom-up model levels seeded with SUPERCOOLED cloud water in the
#: Kessler fixture (T ~ 264.1, 261.5, 258.8 K -- all below 273.15).
_SUPERCOOLED_LEVELS = (9, 10, 11)
#: One WARM cloud-water level (T ~ 282.8 K): both spellings give
#: weight = 0 there, proving the divergence is confined below freezing.
_WARM_LEVEL = 2
#: Bottom-up levels seeded with P3 ice (T ~ 240 K band).
_P3_ICE_LEVELS = (20, 21, 22)


def _wrf_qc_only_cal_cldfra1_oracle(qv, qc, t, p):
    """Hand transcription of module_radiation_driver.F:3891-3899 + 3945-3979.

    Deliberately independent of gpuwm's mirrors: the
    ``F_QC .and. .not. F_QI .and. .not. F_QS`` arm written straight from
    the Fortran -- ``QCLD = QC`` and the freezing-point weight -- so it
    cannot agree with the port by sharing its code.
    """

    qv, qc, t, p = (np.asarray(a, dtype=np.float64)
                    for a in (qv, qc, t, p))
    alpha0, gamma, qcldmin, pexp, rhgrid = 100.0, 0.49, 1.0e-12, 0.25, 1.0
    svp1, svp2, svpi2 = 0.61078, 17.2693882, 21.8745584
    svp3, svpi3, svpt0 = 35.86, 7.66, 273.15
    ep_2 = 287.0 / 461.6

    out = np.zeros(qv.shape, dtype=np.float64)
    for index in np.ndindex(qv.shape):
        tc = t[index] - svpt0
        esw = 1000.0 * svp1 * np.exp(svp2 * tc / (t[index] - svp3))
        esi = 1000.0 * svp1 * np.exp(svpi2 * tc / (t[index] - svpi3))
        qvsw = ep_2 * esw / (p[index] - esw)
        qvsi = ep_2 * esi / (p[index] - esi)
        # :3892-3898 -- QCLD is cloud water alone; the weight is the
        # 273.15 K phase threshold, not a mixing-ratio blend.
        qcld = qc[index]
        if qcld < qcldmin:
            weight = 0.0
        else:
            weight = 0.0 if t[index] > svpt0 else 1.0
        qvs_weight = (1.0 - weight) * qvsw + weight * qvsi
        rhum = qv[index] / qvs_weight
        if qcld < qcldmin:
            out[index] = 0.0
        elif rhum >= rhgrid:
            out[index] = 1.0
        else:
            subsat = max(1.0e-10, rhgrid * qvs_weight - qv[index])
            arg = max(-6.9, -alpha0 * qcld / subsat ** gamma)
            rhum = max(1.0e-10, rhum)
            fraction = (rhum / rhgrid) ** pexp * (1.0 - np.exp(arg))
            out[index] = 0.0 if fraction < 0.01 else fraction
    return out


def _ice_saturation_qv(t, p):
    """Float64 ``qvsi`` from the driver's Murray constants, for seeding."""

    tc = t - 273.15
    esi = 1000.0 * 0.61078 * np.exp(21.8745584 * tc / (t - 7.66))
    return (287.0 / 461.6) * esi / (p - esi)


def _atmosphere_and_state(*, mp_physics, ny=2, nx=3):
    """A real adapter fixture (the test_rrtm_longwave.py construction)."""

    heights = np.linspace(0.0, 16000.0, NZ, dtype=F)
    pw = np.linspace(1013.0, 50.0, NZ + 1, dtype=F)
    p = (F(0.5) * (pw[:-1] + pw[1:])).astype(F)
    t = np.maximum((F(288.15) - F(6.5e-3) * heights).astype(F), F(210.0))
    qv = (F(0.012) * np.exp(-heights / F(2500.0))).astype(F)

    def field3d(column):
        column = np.asarray(column, F)
        return np.ascontiguousarray(
            np.broadcast_to(column[:, None, None],
                            (column.shape[0], ny, nx)).copy())

    atmosphere = {
        "temperature": field3d(t),
        "pressure": field3d(p * F(100.0)),
        "p_interface": field3d(pw * F(100.0)),
        "z_interface": field3d(np.concatenate(
            [[F(0.0)], np.cumsum(np.full(NZ, 400.0, F))])),
        "dz": field3d(np.full(NZ, 400.0, F)),
        "exner": field3d((p / F(1000.0)) ** F(287.0 / 1004.0)),
        "qv": field3d(qv),
        "qc": np.zeros((NZ, ny, nx), F),
        "qi": np.zeros((NZ, ny, nx), F),
    }
    if mp_physics == KESSLER_MP_PHYSICS:
        for level in _SUPERCOOLED_LEVELS:
            atmosphere["qc"][level] = F(2.0e-4)
            # Between the two saturations: 2% ABOVE ice saturation (WRF's
            # qc-only arm reads overcast) and, at these temperatures,
            # ~9% BELOW water saturation (the old hardcoded spelling
            # reads a partial Xu-Randall fraction).
            atmosphere["qv"][level] = F(1.02 * _ice_saturation_qv(
                float(t[level]), float(p[level]) * 100.0))
        atmosphere["qc"][_WARM_LEVEL] = F(3.0e-4)
    elif mp_physics == P3_MP_PHYSICS:
        for level in _P3_ICE_LEVELS:
            atmosphere["qi"][level] = F(3.0e-5)
        atmosphere["qc"][_P3_ICE_LEVELS[1]] = F(2.0e-6)  # mixed phase
    state = SimpleNamespace(elapsed_seconds=0.0,
                            fnm=np.full(NZ, 0.5, F),
                            fnp=np.full(NZ, 0.5, F),
                            p_top=float(pw[-1] * 100.0))
    fields = {"tsk": np.full((ny, nx), 288.15, F),
              "emiss": np.full((ny, nx), 0.95, F)}
    cfg = SimpleNamespace(bl_pbl_physics=1, icloud_bl=0, dt=60.0,
                          mp_physics=mp_physics)
    return atmosphere, fields, state, cfg


def _pack_top_down(field3d):
    """The adapter's column packing: (nz,ny,nx) -> (ny*nx, nz) top-down."""

    nz, ny, nx = field3d.shape
    return np.ascontiguousarray(
        np.asarray(field3d).transpose(1, 2, 0).reshape(ny * nx, nz)[:, ::-1])


def _run_adapter_capturing_solver_inputs(monkeypatch, *, mp_physics):
    """Run the real adapter on the NumPy shim; capture the solver call.

    The spy records every kwarg handed to ``rrtm_longwave_columns`` --
    the complete input surface of the transfer solve, ``cldfra``
    included -- then forwards to the REAL solver, so the returned result
    is the shipped computation, not a mock's.
    """

    monkeypatch.setitem(sys.modules, "cupy", np)
    atmosphere, fields, state, cfg = _atmosphere_and_state(
        mp_physics=mp_physics)
    ny, nx = fields["tsk"].shape

    captured = {}
    # The PRISTINE solver, bound at import time: a second helper call in
    # one test must not chain through the first call's still-patched spy
    # (which would overwrite the first call's captured arrays).
    real_columns = _REAL_RRTM_LONGWAVE_COLUMNS

    def spy(**kwargs):
        for name, value in kwargs.items():
            if isinstance(value, np.ndarray):
                captured[name] = value.copy()
        return real_columns(**kwargs)

    monkeypatch.setattr(rrtm_lw, "rrtm_longwave_columns", spy)
    adapter = RRTMLongwaveRadiation(
        datetime(2011, 4, 27, 18), np.full((ny, nx), 39.0, F),
        np.full((ny, nx), -87.0, F), p_top=state.p_top, icloud=1)
    rthratenlw, glw, olr = adapter.longwave(
        atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    outputs = {"rthratenlw": np.asarray(rthratenlw),
               "glw": np.asarray(glw), "olr": np.asarray(olr)}
    return atmosphere, captured, outputs


def test_the_adapter_wires_the_registry_pair_not_hardcoded_booleans():
    """Source-level: the call site cannot regress to literal flags."""

    source = inspect.getsource(RRTMLongwaveRadiation.longwave)
    assert "legacy_cloud_fraction_flags" in source, (
        "the RRTM longwave adapter no longer resolves cal_cldfra1's flags "
        "from the Registry package membership helper in "
        "gpuwm/core/rrtmg_legacy.py")
    assert "f_qc=True, f_qi=f_qi, f_qs=f_qs" in source
    assert "f_qi=True, f_qs=True" not in source, (
        "the hardcoded fused flags are back; they send a Kessler-class "
        "supercooled cloud through the Morrison arm (liquid saturation, "
        "weight=0) where WRF's qc-only arm at "
        "module_radiation_driver.F:3891-3899 gives weight=1 below "
        "273.15 K")


def test_a_kessler_supercooled_cloud_gets_wrfs_qc_only_weighting(
        monkeypatch):
    """mp=1 through the real adapter matches the Fortran qc-only arm.

    The seeded levels sit between ice and water saturation at ~260 K, so
    the two arms disagree by construction: WRF's freezing-point weight
    reads them overcast, the old hardcoded spelling read a partial
    fraction off the liquid saturation.
    """

    atmosphere, captured, _outputs = _run_adapter_capturing_solver_inputs(
        monkeypatch, mp_physics=KESSLER_MP_PHYSICS)

    qv = _pack_top_down(atmosphere["qv"])
    qc = _pack_top_down(atmosphere["qc"])
    t = _pack_top_down(atmosphere["temperature"])
    p = _pack_top_down(atmosphere["pressure"])
    oracle = _wrf_qc_only_cal_cldfra1_oracle(qv, qc, t, p)
    np.testing.assert_allclose(captured["cldfra"], oracle,
                               rtol=2.0e-6, atol=2.0e-7)

    # The physics the numbers encode, asserted by name: every seeded
    # supercooled level is OVERCAST (ice-saturation weighting)...
    for level in _SUPERCOOLED_LEVELS:
        row = NZ - 1 - level
        np.testing.assert_allclose(captured["cldfra"][:, row], 1.0)
    # ...and the warm cloud level is where both arms agree (weight=0).
    warm = _wrf_qc_only_cal_cldfra1_oracle(
        qv[:, NZ - 1 - _WARM_LEVEL], qc[:, NZ - 1 - _WARM_LEVEL],
        t[:, NZ - 1 - _WARM_LEVEL], p[:, NZ - 1 - _WARM_LEVEL])
    np.testing.assert_allclose(
        captured["cldfra"][:, NZ - 1 - _WARM_LEVEL], warm,
        rtol=2.0e-6, atol=2.0e-7)


def test_the_old_hardcoded_flags_diverge_on_the_supercooled_levels(
        monkeypatch):
    """The measured consequence, kept as the reason this gate exists.

    Force the pre-fix spelling -- ``(True, True)`` with zero-filled
    qi/qs -- through the same adapter and the same column: the seeded
    supercooled levels drop from overcast to a partial fraction.
    """

    monkeypatch.setattr(rrtmg_legacy, "legacy_cloud_fraction_flags",
                        lambda mp: (True, True))
    _atmo, captured, _outputs = _run_adapter_capturing_solver_inputs(
        monkeypatch, mp_physics=KESSLER_MP_PHYSICS)
    for level in _SUPERCOOLED_LEVELS:
        row = NZ - 1 - level
        old = captured["cldfra"][:, row]
        assert np.all(old < 1.0), (
            "forcing the old fused flags should reproduce the divergence "
            "this fix removed; if this fails the fixture no longer sits "
            "between the two saturations")
        assert np.all(old > 0.0)


def test_a_p3_ice_column_is_byte_identical_to_the_collapsed_spelling(
        monkeypatch):
    """mp=50: the resolved (True, False) equals the old (True, True) bytes.

    The collapse argument, MEASURED rather than asserted: with ``qs``
    zero-filled, the Morrison arm's ``QCLD = QI+QC+0`` and ``weight =
    (QI+0)/QCLD`` are the P3 arm's expressions exactly, so the flag split
    may not move a single byte of the solve's inputs or outputs for P3.
    """

    assert legacy_cloud_fraction_flags(P3_MP_PHYSICS) == (True, False)
    _atmo, resolved, resolved_out = _run_adapter_capturing_solver_inputs(
        monkeypatch, mp_physics=P3_MP_PHYSICS)

    monkeypatch.setattr(rrtmg_legacy, "legacy_cloud_fraction_flags",
                        lambda mp: (True, True))
    _atmo, hardcoded, hardcoded_out = _run_adapter_capturing_solver_inputs(
        monkeypatch, mp_physics=P3_MP_PHYSICS)

    assert set(resolved) == set(hardcoded)
    for name in sorted(resolved):
        assert resolved[name].tobytes() == hardcoded[name].tobytes(), (
            f"solver input {name!r} moved under the flag split for P3")
    for name in sorted(resolved_out):
        assert resolved_out[name].tobytes() == \
            hardcoded_out[name].tobytes(), (
                f"output {name!r} moved under the flag split for P3")
    # Not vacuous: the seeded ice band is cloudy in the solve.
    assert float(resolved["cldfra"].max()) == pytest.approx(1.0)


def test_the_cpu_mirror_takes_the_same_qc_only_arm_for_kessler():
    """The float64 reference with mp=1's resolved flags, vs the Fortran."""

    f_qi, f_qs = legacy_cloud_fraction_flags(KESSLER_MP_PHYSICS)
    assert (f_qi, f_qs) == (False, False)
    qv = np.array([[0.0023, 0.0009, 0.004]])
    qc = np.array([[2.0e-4, 3.0e-4, 0.0]])
    t = np.array([[263.0, 255.0, 281.0]])
    p = np.array([[70000.0, 55000.0, 90000.0]])
    got = np_cal_cldfra1(qv, qc, np.zeros_like(qc), np.zeros_like(qc),
                         t, p, f_qc=True, f_qi=f_qi, f_qs=f_qs)
    oracle = _wrf_qc_only_cal_cldfra1_oracle(qv, qc, t, p)
    np.testing.assert_allclose(got, oracle, rtol=1.0e-12, atol=0.0)
    # The first column is the divergence case: overcast off the ice
    # saturation, partial off the liquid one.
    assert oracle[0, 0] == pytest.approx(1.0)
    old = np_cal_cldfra1(qv, qc, np.zeros_like(qc), np.zeros_like(qc),
                         t, p, f_qc=True, f_qi=True, f_qs=True)
    assert 0.0 < old[0, 0] < 1.0


@pytest.mark.parametrize("stock_tree", [
    # Machine-local staging paths are deliberately not part of the
    # contract: name the tree, let the machine say where.
    pathlib.Path(os.environ.get("GPUWM_WRF461_STOCK_TREE",
                                "wrf-stock-v461-gate-20260721")),
])
def test_the_qc_only_arm_citation_is_checkable_against_stock_wrf(stock_tree):
    """The line numbers, verified rather than asserted, when the tree is
    present.  A line-numbered claim nobody re-reads is a claim about the
    past; this one is cheap to re-check, so it is checked.
    """

    source = stock_tree / "phys" / "module_radiation_driver.F"
    if not source.is_file():                                # pragma: no cover
        pytest.skip(f"stock WRF v4.6.1 tree not present at {stock_tree}")
    lines = source.read_text(encoding="utf-8",
                             errors="replace").splitlines()
    # 1-based citations in the code comments; 0-based here.
    assert "F_QC .and. .not. F_QI .and. .not. F_QS" in lines[3890]
    assert "QCLD = QC(i,k,j)" in lines[3891]
    assert "if (t_phy(i,k,j) .gt. 273.15) weight = 0." in lines[3895]
    assert "if (t_phy(i,k,j) .le. 273.15) weight = 1." in lines[3896]
    # The driver hands cal_cldfra1 the Registry flags, not constants.
    assert "CALL cal_cldfra1(CLDFRA,qv,qc,qi,qs," in lines[1325]
    assert "F_QV,F_QC,F_QI,F_QS,t,p," in lines[1326]
