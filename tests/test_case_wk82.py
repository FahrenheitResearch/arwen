# tests/test_case_wk82.py  (Phase 2 Task 11: WK82 quarter-circle supercell)
import pytest

# Task 11's benchmark is a GPU test and carried a module-level
# ``pytestmark = pytest.mark.gpu``; the Phase 3 Task 2 (T11-1) sampling-
# cadence test is CPU-only float64 setup code, so the module mark became a
# per-test ``@pytest.mark.gpu`` decorator (marker-equivalent for the
# pre-existing test).


def test_mass_drift_sampling_cadence():
    """T11-1 (Phase 2 final review): ``run()`` samples the dry-mass drift
    for the ``mass_drift_max`` gate every 10 steps -- the previous 100-step
    cadence could miss short-lived boundary-exchange spikes between
    samples.  The cadence is single-sourced in ``mass_sample_steps`` (the
    exact step set ``run()`` samples at, final step included via the
    post-loop drift fold-in)."""
    from gpuwm.verify.cases import wk82
    assert wk82.MASS_SAMPLE_EVERY == 10
    assert list(wk82.mass_sample_steps(35)) == [10, 20, 30]
    assert list(wk82.mass_sample_steps(9)) == []
    # the default 2 h / 6 s benchmark: 1200 steps -> 120 samples, the last
    # one landing exactly on the final step
    cfg = wk82.default_config()
    n_total = int(round(cfg.run_seconds / cfg.dt))
    steps = list(wk82.mass_sample_steps(n_total))
    assert len(steps) == n_total // 10 == 120
    assert steps[-1] == n_total


def test_sustained_metric_rejects_one_spike_per_window():
    """One favorable sample in each 15-minute window is not sustained."""
    import numpy as np
    from gpuwm.verify.cases.wk82 import sustained_updraft_fraction

    times = np.arange(3600.0, 7200.0 + 6.0, 6.0)
    speeds = np.full(times.shape, 20.0)
    speeds[::150] = 50.0
    history = np.column_stack((times, speeds))
    assert sustained_updraft_fraction(history, threshold=25.0) < 0.01
    history[:, 1] = 50.0
    assert sustained_updraft_fraction(history, threshold=25.0) == 1.0


def test_growth_uses_both_analysis_intervals_and_motion_uses_shear():
    """An endpoint recovery cannot hide shrinking at the middle checkpoint,
    and right-of-shear is a vector projection rather than a y-only proxy."""
    from gpuwm.verify.cases.wk82 import (
        right_of_shear_distance,
        separation_growth_intervals,
    )

    growth = separation_growth_intervals({3600.0: 10.0,
                                          5400.0: 14.0,
                                          7200.0: 12.0})
    assert growth == (4.0, -2.0)
    assert right_of_shear_distance((0.0, 0.0), (1000.0, -2000.0),
                                   (10.0, 0.0)) == pytest.approx(2000.0)


def test_wk82_exports_boundary_flux_closure_gate():
    from gpuwm.verify.cases import wk82

    assert wk82.GATES["mass_closure_residual_max"] == (None, 1.0e-5)


def test_wk82_supercell_benchmark_rejects_incomplete_no_pbl_operator():
    """The retired WK82 gate may not bless horizontal-only km_opt=4.

    With PBL disabled, WRF v4.6.1 also runs vertical_diffusion_2.  Until
    gpuwm implements those u/v/w stresses and their surface-flux policy, the
    former two-hour benchmark is an explicit rejection contract.  Shipped
    real74 TOMLs and the phase3 case remain supported because PBL is on.
    """
    from gpuwm.verify.cases.wk82 import run

    with pytest.raises(NotImplementedError,
                       match=r"km_opt=4.*bl_pbl_physics=0.*vertical"):
        run()
