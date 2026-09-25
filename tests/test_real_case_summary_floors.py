"""The single-domain run route hands its floor receipts on the summary.

``run_experiment``'s single-domain branch returns
``dataclass_replace(summary, frame_records=..., moisture_floor_receipts=...)``
on the :class:`RealCaseRunSummary` that ``integrate_prepared_case`` built.
Without the field that replace raised TypeError after the forecast had
written every frame, so the run failed before rendering (reproduced by the
proving ground on an ERA5 single-domain run at the 2.8 tip).
"""

from dataclasses import fields, replace
from datetime import datetime

from gpuwm.runtime import ExperimentRunSummary, RealCaseRunSummary


def test_real_case_summary_takes_the_floor_receipts_the_single_domain_route_sets():
    summary = RealCaseRunSummary(
        wrfout_paths=(), nan_free=True, w_max_ms=0.0, boundary_w_max_ms=0.0,
        interior_w_max_ms=0.0, w_max_boundary_row=None, boundary_zone_blowup=False,
        dynamics_substeps=0, ysu_nan_guard_fires=0, surface_forcing_updates=0,
        swdown_peak_wm2=0.0, swdown_peak_time=datetime(2020, 1, 1), completed_seconds=0.0)
    floors = {"moisture_floors_by_domain": {"d01": {"fired": False}}}
    updated = replace(summary, frame_records=(), moisture_floor_receipts=floors)
    assert updated.moisture_floor_receipts == floors


def test_both_summaries_carry_the_field_the_supervisor_reads():
    for cls in (RealCaseRunSummary, ExperimentRunSummary):
        assert "moisture_floor_receipts" in {f.name for f in fields(cls)}
