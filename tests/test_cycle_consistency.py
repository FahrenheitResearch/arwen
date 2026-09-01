"""The hydrostatic consistency instrument, validated in both directions.

Every pre-existing gate on the restart path is an IDENTITY gate, and a DA
increment must break identity by construction, so none of them can see a
state whose carried diagnostics describe the pre-analysis atmosphere.
This instrument can -- but only if it is proved to read zero on a clean
state and to read loud on a dirty one, with its own noise floor printed
so nobody later tunes the threshold below what it can resolve.

The both-directions discipline is why this file carries a NEGATIVE
CONTROL for every correction made to the instrument.  The correction
that added the vertical metric ``zz`` to the equation of state removed a
false alarm; a correction that removes a false alarm is exactly the kind
that can install a blind spot in its place, so each of these clean-state
tests is paired with a dirty state built on the SAME metric that the
instrument must still reject.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.cycle.consistency import (UNIT_VERTICAL_METRIC, derived_is_stale,
                                     hydrostatic_residual, rebuild_exner,
                                     require_consistent)
from gpuwm.cycle.contracts import CONSISTENCY_SCHEMA, CycleRefusal

P0 = 1.0e5
RD = 287.0
CP = 1004.5
KAPPA = RD / (CP - RD)


def _self_consistent(nz: int = 6, ncells: int = 40):
    """A unit-metric state: the synthetic replay path's precision."""
    rng = np.random.default_rng(20260814)
    rho_theta = 250.0 + 40.0 * rng.random((nz, ncells))
    exner = (RD * rho_theta / P0) ** KAPPA
    return {"rho_theta": rho_theta}, {"exner": exner}


def _port_consistent(nz: int = 6, ncells: int = 40, dtype=np.float32):
    """A state built through the PORT's equation of state.

    ``mpas_port.cuda_backend.recovery::pressure_point`` is the authority:

        argument = zz * (rgas / p0) * rtheta
        exner    = pow(argument, rgas / (cp - rgas))

    ``zz`` here spans 0.83..2.35, the measured range of the x1.40962
    mesh's vertical metric, and the fields are stored float32 because
    that is what comes off the device.
    """
    rng = np.random.default_rng(20260814)
    rho_theta = (250.0 + 40.0 * rng.random((nz, ncells))).astype(dtype)
    zz = (0.83 + 1.52 * rng.random((nz, ncells))).astype(dtype)
    exner = ((zz.astype(np.float64) * (RD / P0)
              * rho_theta.astype(np.float64)) ** KAPPA).astype(dtype)
    return {"rho_theta": rho_theta}, {"exner": exner}, zz


def test_residual_floor_is_reported_and_tiny(capsys):
    prognostic, derived = _self_consistent()
    result = hydrostatic_residual(prognostic, derived,
                                  vertical_metric=UNIT_VERTICAL_METRIC)
    assert result["schema"] == CONSISTENCY_SCHEMA
    assert result["n_points"] == prognostic["rho_theta"].size
    assert result["vertical_metric"] == UNIT_VERTICAL_METRIC
    floor = result["resolution_floor"]
    with capsys.disabled():
        print(f"\nhydrostatic instrument resolution floor = {floor:.3e}; "
              f"clean-state max relative residual = "
              f"{result['max_relative_residual']:.3e}")
    assert result["max_relative_residual"] <= max(10.0 * floor, 1.0e-15)


def test_residual_catches_unrebuilt_diagnostics():
    prognostic, derived = _self_consistent()
    dirty = {"rho_theta": prognostic["rho_theta"].copy()}
    dirty["rho_theta"][3, 11] *= 1.01          # a 1 % analysis increment
    result = hydrostatic_residual(dirty, derived,
                                  vertical_metric=UNIT_VERTICAL_METRIC)
    assert result["max_relative_residual"] > 1.0e-6
    assert result["argmax_index"] == [3, 11]

    with pytest.raises(CycleRefusal) as excinfo:
        require_consistent(dirty, derived, threshold=1.0e-6,
                           vertical_metric=UNIT_VERTICAL_METRIC,
                           label="cycle=2 parent=mpas-cuda")
    message = str(excinfo.value)
    assert "diagnostics do not agree" in message
    assert "argmax_index=[3, 11]" in message
    assert "threshold=1e-06" in message
    assert "resolution_floor" in message

    # ...and the clean state passes the same call, so the gate is not
    # simply refusing everything handed to it.
    clean = require_consistent(prognostic, derived, threshold=1.0e-6,
                               vertical_metric=UNIT_VERTICAL_METRIC,
                               label="cycle=2 parent=mpas-cuda")
    assert clean["max_relative_residual"] < 1.0e-6


# --------------------------------------------------------------------------
# the port's equation of state: zz is IN the argument


def test_port_eos_state_reads_at_its_floor(capsys):
    """The regression this module halted a correct closed loop on.

    A state built exactly as ``recovery.pressure_point`` builds one must
    read at the instrument's floor.  Recomputing without ``zz`` reads
    ``|zz**(rd/cv) - 1|`` instead, which for this metric range is ~0.4 --
    five orders above the gate, on a state that is correct.
    """
    prognostic, derived, zz = _port_consistent()
    result = hydrostatic_residual(prognostic, derived, vertical_metric=zz)
    with capsys.disabled():
        print(f"\nport-EOS float32 boundary: residual = "
              f"{result['max_relative_residual']:.3e}, floor = "
              f"{result['resolution_floor']:.3e}, "
              f"gate = 1.000e-06")
    assert result["vertical_metric"] == "supplied"
    assert result["carried_dtype"] == "float32"
    assert result["max_relative_residual"] <= result["resolution_floor"]
    assert result["max_relative_residual"] < 1.0e-6

    # And the omission this test exists to forbid, measured rather than
    # asserted: grading the same correct state on a unit metric.
    blind = hydrostatic_residual(prognostic, derived,
                                 vertical_metric=UNIT_VERTICAL_METRIC)
    expected = float(np.max(np.abs(zz.astype(np.float64) ** KAPPA - 1.0)))
    assert blind["max_relative_residual"] > 0.1
    assert blind["max_relative_residual"] == pytest.approx(expected, rel=1e-5)


def test_port_eos_floor_reflects_float32_not_float64():
    """The floor must describe the STATE's precision, not numpy's.

    A float64 recomputation differenced against itself always reports
    ~1e-16.  That is true about this module's arithmetic and false about
    what the instrument can resolve on a float32 device boundary, and a
    floor that flatters itself is how a threshold gets tuned into
    decoration.
    """
    _, _, _ = _port_consistent()
    f32 = hydrostatic_residual(*_port_consistent(dtype=np.float32)[:2],
                               vertical_metric=_port_consistent(
                                   dtype=np.float32)[2])
    f64 = hydrostatic_residual(*_port_consistent(dtype=np.float64)[:2],
                               vertical_metric=_port_consistent(
                                   dtype=np.float64)[2])
    assert f32["resolution_floor"] > 1.0e-7
    assert f32["resolution_floor"] < 1.0e-6      # still below the gate
    assert f64["resolution_floor"] < 1.0e-14


def test_port_eos_still_rejects_a_genuinely_inconsistent_state():
    """THE NEGATIVE CONTROL for the ``zz`` correction.

    The correction removed a false alarm.  The way a correction like that
    goes wrong is by removing the alarm altogether, so the same
    port-metric state is dirtied by a 1 % increment -- the smallest thing
    anybody would call an analysis -- and the instrument must still fire,
    on the right cell, with the metric supplied.
    """
    prognostic, derived, zz = _port_consistent()
    dirty = {"rho_theta": prognostic["rho_theta"].copy()}
    dirty["rho_theta"][4, 17] = np.float32(dirty["rho_theta"][4, 17] * 1.01)

    result = hydrostatic_residual(dirty, derived, vertical_metric=zz)
    assert result["max_relative_residual"] > 1.0e-3
    assert result["argmax_index"] == [4, 17]
    assert result["max_relative_residual"] > 1000.0 * \
        result["resolution_floor"]

    with pytest.raises(CycleRefusal) as excinfo:
        require_consistent(dirty, derived, vertical_metric=zz,
                           threshold=1.0e-6, label="negative control")
    assert "argmax_index=[4, 17]" in str(excinfo.value)
    assert "vertical_metric='supplied'" in str(excinfo.value)


def test_a_wrong_metric_is_still_caught():
    """A blind spot check on the metric itself.

    Supplying ``zz`` is not a licence to supply any ``zz``: a state
    graded against the wrong mesh's metric must fail, or the correction
    would have turned the metric into an unchecked free parameter that
    can be tuned until any state passes.
    """
    prognostic, derived, zz = _port_consistent()
    wrong = (zz * np.float32(1.05)).astype(np.float32)
    result = hydrostatic_residual(prognostic, derived, vertical_metric=wrong)
    assert result["max_relative_residual"] > 1.0e-2


def test_metric_is_required_and_refuses_a_bad_one():
    prognostic, derived, zz = _port_consistent()
    with pytest.raises(TypeError):
        hydrostatic_residual(prognostic, derived)      # no default, ever

    with pytest.raises(CycleRefusal) as excinfo:
        hydrostatic_residual(prognostic, derived, vertical_metric=None)
    assert "will not assume a vertical metric" in str(excinfo.value)

    with pytest.raises(CycleRefusal) as excinfo:
        hydrostatic_residual(prognostic, derived,
                             vertical_metric=zz[:, :3])
    assert "does not have the shape of rho_theta" in str(excinfo.value)

    with pytest.raises(CycleRefusal) as excinfo:
        hydrostatic_residual(prognostic, derived,
                             vertical_metric=np.zeros_like(zz))
    assert "finite and positive" in str(excinfo.value)


def test_rebuild_exner_and_the_grader_share_one_equation():
    """The rebuild and the grade were two transcriptions; now they are one.

    ``engine._rebuild_derived`` used to carry its own copy of the EOS and
    it omitted ``zz`` in the same way this module did.  A rebuild that
    disagrees with the grader is worse than either bug alone: it writes a
    wrong exner into the anchor and then passes it.
    """
    prognostic, derived, zz = _port_consistent()
    rebuilt = rebuild_exner(prognostic["rho_theta"], zz)
    graded = hydrostatic_residual(prognostic, {"exner": rebuilt},
                                  vertical_metric=zz)
    assert graded["max_relative_residual"] == 0.0


def test_derived_is_stale_detects_sha_drift():
    manifest = {"parent": {"prognostic_sha256": "ab" * 32},
                "derived": {"derived_from_sha256": "ab" * 32}}
    assert derived_is_stale(manifest) is False
    manifest["parent"]["prognostic_sha256"] = "cd" * 32
    assert derived_is_stale(manifest) is True
