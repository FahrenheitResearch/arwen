"""A163: the forecast memory gate counts each headroom once.

The breakage the gate prevents is a CUDA out-of-memory mid-run, so every
row of the measured battery must stay under the envelope, and the margin
must be the battery's measured worst ratio plus the stated safety term.
The defect it fixes: the envelope multiplied the itemized subtotal by
1.15 and then, on the legacy-RRTMG lane, added 20% of that product as
pool slack, so HRRR physics on HRRR's full grid was priced at 119.09 GiB
and the BEP+BEM city trees above a 24 GiB card they ran on.
"""

from __future__ import annotations

import math

import pytest

from gpuwm.core import preflight as pf

GIB = pf.GIB

#: The battery rows the itemization prices short (pool used 1.12-1.16x
#: the subtotal): the configuration ``FORECAST_ITEMIZATION_GAP_ROWS``
#: carries.  Every other row's itemization is complete (1.02-1.06x).
_GAP_LABEL = "HRRR physics, RTE-RRTMGP"


def _headroom(row):
    if row[0].startswith(_GAP_LABEL):
        ratio = max(r["ratio"] for r in pf.FORECAST_ITEMIZATION_GAP_ROWS)
        return round(ratio + pf.FORECAST_PEAK_RATIO_SAFETY, 2)
    return pf.FORECAST_POOL_HEADROOM


def _ratio(row):
    """``(peak - itemized non-pool - unmodelled residue - held arrays) /
    (subtotal - held arrays)``: the margin is measured on what the pool
    turns over, net of every term the envelope adds once beside it, and
    the arrays held at their allocated size (urban) taken out of both
    sides."""
    _l, _c, _lane, _d, subtotal, non_pool, peak, _pool, held = row
    return ((peak - non_pool - pf.ENVELOPE_UNMODELLED_BYTES - held)
            / (subtotal - held))


@pytest.mark.parametrize("row", pf.FORECAST_PEAK_BATTERY,
                         ids=lambda r: f"{r[0]} on {r[1]}")
def test_every_measured_peak_is_under_the_envelope(row):
    """The gate's job, row by row: no measured forecast peak lands above
    the envelope its own itemization prices."""
    _label, _card, lane, domains, subtotal, non_pool, peak, _pool, held = row
    envelope = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=pf.forecast_pool_estimate_bytes(
            subtotal, held_exact_bytes=held, headroom=_headroom(row)),
        non_pool_bytes=non_pool, domains=domains, family="linux",
        legacy_radiation=lane == "legacy-rrtmg", held_exact_bytes=held)
    assert envelope >= peak, (envelope / GIB, peak / GIB)


def test_the_margin_is_the_measured_worst_ratio_plus_the_safety_term():
    """Each ratio constant IS its battery rows' worst :func:`_ratio`,
    rounded up to the next hundredth -- not a taste, and not a second
    allowance beside a first."""
    complete = [_ratio(r) for r in pf.FORECAST_PEAK_BATTERY
                if not r[0].startswith(_GAP_LABEL)]
    short = [_ratio(r) for r in pf.FORECAST_PEAK_BATTERY
             if r[0].startswith(_GAP_LABEL)]
    assert complete and short
    assert 0 <= pf.FORECAST_PEAK_RATIO_MEASURED - max(complete) < 0.01
    gap = max(r["ratio"] for r in pf.FORECAST_ITEMIZATION_GAP_ROWS)
    assert 0 <= gap - max(short) < 0.01
    assert pf.FORECAST_POOL_HEADROOM == pytest.approx(
        pf.FORECAST_PEAK_RATIO_MEASURED + pf.FORECAST_PEAK_RATIO_SAFETY)
    # The safety term is the card-to-card swing of the pool-held ratio on
    # the configuration the itemization prices most exactly, rounded up.
    lean = [row for row in pf.FORECAST_PEAK_BATTERY
            if row[0] == "lean 528x528x49"]
    assert len(lean) == 2
    swing = abs(lean[0][7] / lean[0][4] - lean[1][7] / lean[1][4])
    assert 0 <= pf.FORECAST_PEAK_RATIO_SAFETY - swing < 0.01


def test_the_short_itemization_row_selects_its_own_margin():
    """HRRR physics under RTE-RRTMGP pays its measured ratio; the same
    schemes under legacy RRTMG, and RTE-RRTMGP under the lean schemes,
    pay the common one."""
    from types import SimpleNamespace

    def run(lane, **schemes):
        rte = lane == "rte"
        base = dict(ra_physics=0 if rte else 4, ra_lw_physics=4 if rte else -1,
                    ra_sw_physics=4 if rte else -1,
                    ra_rrtmg_variant="rte-rrtmgp" if rte else "rrtmg_legacy",
                    wrf_rrtmg_compatibility=(
                        "wrf-rrtmg-4-4-to-rte-rrtmgp-v1" if rte
                        else "wrf-rrtmg-4-4-legacy-v1"),
                    mp_physics=8, bl_pbl_physics=1, sf_sfclay_physics=91,
                    sf_surface_physics=2)
        base.update(schemes)
        return SimpleNamespace(**base)

    gap = round(max(r["ratio"] for r in pf.FORECAST_ITEMIZATION_GAP_ROWS)
                + pf.FORECAST_PEAK_RATIO_SAFETY, 2)
    hrrr = dict(mp_physics=28, bl_pbl_physics=5, sf_sfclay_physics=5,
                sf_surface_physics=3)
    assert pf.forecast_pool_headroom([run("rte", **hrrr)]) == gap
    assert pf.forecast_pool_headroom([run("legacy", **hrrr)]) == (
        pf.FORECAST_POOL_HEADROOM)
    assert pf.forecast_pool_headroom([run("rte")]) == (
        pf.FORECAST_POOL_HEADROOM)
    # One pool per forecast: a tree with one matching domain pays it.
    assert pf.forecast_pool_headroom(
        [run("rte"), run("rte", bl_pbl_physics=5)]) == gap


def test_the_legacy_lane_no_longer_pays_a_second_margin():
    """Same estimate, same envelope, whichever radiation lane runs it:
    the 20% slack priced the headroom the estimate's margin prices."""
    common = dict(alloc_estimate_bytes=8 * GIB, non_pool_bytes=GIB,
                  domains=1)
    for family in ("linux", "windows"):
        assert (pf.machine_peak_envelope_bytes(
                    **common, family=family, legacy_radiation=True)
                == pf.machine_peak_envelope_bytes(
                    **common, family=family, legacy_radiation=False)
                == 9 * GIB + pf.ENVELOPE_UNMODELLED_BYTES)


def test_hrrr_full_grid_evidence_old_and_new():
    """The queue row's own numbers: 83.35 GiB itemized, 3.56 GiB
    non-pool, legacy RRTMG.  The old arithmetic reproduces the 119.09 GiB
    it printed; the new one prices the same itemization once, still above
    what the PRO 6000 probe's measured ratio puts it at, and still over
    that card's 93.93 GiB budget."""
    subtotal, non_pool = int(83.35 * GIB), int(3.56 * GIB)
    estimate_old = math.ceil(1.15 * subtotal)
    old = (estimate_old + non_pool + pf.ENVELOPE_UNMODELLED_BYTES
           + math.ceil(0.20 * estimate_old))
    assert old / GIB == pytest.approx(119.09, abs=0.02)
    new = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=math.ceil(pf.FORECAST_POOL_HEADROOM * subtotal),
        non_pool_bytes=non_pool, family="linux", legacy_radiation=True)
    assert new / GIB == pytest.approx(
        pf.FORECAST_POOL_HEADROOM * 83.35 + 3.56 + 0.5, abs=0.02)
    assert new < old - 15 * GIB
    probe = [row for row in pf.FORECAST_PEAK_BATTERY
             if row[0].startswith("CONUS HRRR physics")][0]
    assert new >= _ratio(probe) * subtotal + non_pool
    assert new > 93.93 * GIB


def test_every_forecast_door_prices_with_the_one_margin():
    """The estimate, the [tiles] window price and the mid-run spawn check
    read the same proportional margin; the 1.15 plan factor is left to
    the preparation pool it still describes."""
    import inspect

    from gpuwm.core import prepared_tile_memory
    from gpuwm.ingest import nest_spawn_init

    assert pf.ExperimentMemoryEstimate.__dataclass_fields__[
        "headroom"].default == pf.FORECAST_POOL_HEADROOM
    assert "forecast_pool_headroom(" in inspect.getsource(
        pf.estimate_experiment)
    tile_source = inspect.getsource(prepared_tile_memory)
    assert "ALLOCATOR_HEADROOM" not in tile_source
    assert "forecast_pool_headroom(" in tile_source
    spawn_source = inspect.getsource(nest_spawn_init.admit_spawned_child)
    assert "forecast_pool_headroom(" in spawn_source


# ---------------------------------------------------------------------------
# The urban arrays are priced at their allocated size
# ---------------------------------------------------------------------------

#: The two 750 m BEP+BEM city trees the urban lane ran to the end on a
#: 24 GiB RTX 4090 (2.25 km parent 288 x 288, 750 m nest, 59 levels,
#: Thompson, YSU, Noah-MP, legacy RRTMG), by their grid shapes only.
_CITY_TREES = {"tree-a": ((288, 288), (216, 216), (122, 98)),
               "tree-b": ((288, 288), (216, 186), (114, 112))}

def _rtx_4090():
    """The RTX 4090's own profile, the card those runs used: 128 SMs x
    1,536 threads, priced from its recorded compile platform (sm_89, NVRTC
    13.4.92), which is what `gpuwm check` reads on a live 4090."""
    from gpuwm.core.device_inventory import DeviceLocalMemoryProfile
    return DeviceLocalMemoryProfile(
        name="NVIDIA GeForce RTX 4090", multiprocessor_count=128,
        max_threads_per_multiprocessor=1536,
        compile_platform=("89", "13.4.92"))


def _city_tree(tmp_path, name):
    from gpuwm.experiment import load_experiment

    (nx1, ny1), (nx2, ny2), (i0, j0) = _CITY_TREES[name]
    text = f"""
[experiment]
name = "{name}"
start_time = 2026-09-30T00:00:00
run_seconds = 1200.0
feedback = 1
smooth_option = 0
blend_width = 5
spec_bdy_width = 5
restart_interval_s = 0.0

[projection]
map_proj = "lambert"
ref_lat = 36.0
ref_lon = -120.0
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -120.0

[shared]
nz = 59
ztop = 20000.0
p_top = 5000.0
moist = true
moist_cq = true
mp_physics = 8
ra_physics = 0
ra_lw_physics = 4
ra_sw_physics = 4
wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-legacy-v1"
ra_rrtmg_variant = "rrtmg_legacy"
sf_sfclay_physics = 1
sf_surface_physics = 4
bl_pbl_physics = 1
num_soil_layers = 4
sf_urban_physics = 3
use_wudapt_lcz = 1
km_opt = 4
bldt = 0.0

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = {nx1}
ny = {ny1}
time_step = 10
dx = 2250.0
specified = true
nested = false
history_interval_s = 600.0
radt = 10.0
cu_physics = 0

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = {i0}
j_parent_start = {j0}
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = {nx2}
ny = {ny2}
specified = false
nested = true
history_interval_s = 600.0
radt = 3.0
cu_physics = 0
"""
    path = tmp_path / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return load_experiment(path)


def test_the_urban_arrays_are_priced_at_their_allocated_size(tmp_path):
    """Every urban array held for the run is a held item at its own
    shape; the margin multiplies the rest of the subtotal only, and the
    per-nest term is a fraction of the rest."""
    from gpuwm.core.urban_state import (URBAN_PER_CALL_ARRAYS,
                                        urban_array_shapes,
                                        urban_held_array_shapes)

    exp = _city_tree(tmp_path, "tree-a")
    estimate = pf.estimate_experiment(exp, profile=_rtx_4090())
    for dc, dom in zip(exp.domains, estimate.domains):
        held = {item.name: item for item in dom.items if item.held_exact}
        shapes = urban_held_array_shapes(dc.run)
        assert set(held) == {f"fields/{name}" for name in shapes}
        assert held["fields/tw1_urb4d"].shape == (5400, dc.run.ny, dc.run.nx)
        assert held["fields/tflev_urb3d"].shape == (5100, dc.run.ny,
                                                    dc.run.nx)
        assert set(urban_array_shapes(dc.run)) - set(shapes) <= set(
            URBAN_PER_CALL_ARRAYS)
        assert dom.held_exact_bytes == pf.urban_held_bytes(dc.run)
    held = estimate.held_exact_bytes
    assert held == sum(pf.urban_held_bytes(dc.run) for dc in exp.domains)
    assert estimate.alloc_estimate_bytes == (
        math.ceil(estimate.headroom * (estimate.subtotal_bytes - held))
        + held)
    assert estimate.peak_envelope_bytes == (
        estimate.alloc_estimate_bytes + estimate.envelope_intercept_bytes
        + pf.ENVELOPE_UNMODELLED_BYTES
        + math.ceil(pf.ENVELOPE_PER_NEST_FRACTION
                    * (estimate.alloc_estimate_bytes - held)))
    # A configuration with no urban model holds nothing at exact size.
    assert all(not item.held_exact for dom in pf.estimate_experiment(
        _city_tree_without_urban(tmp_path)).domains for item in dom.items)


def _city_tree_without_urban(tmp_path):
    import dataclasses

    exp = _city_tree(tmp_path, "tree-b")
    return dataclasses.replace(exp, domains=tuple(
        dataclasses.replace(dc, run=dataclasses.replace(
            dc.run, sf_urban_physics=0, use_wudapt_lcz=0))
        for dc in exp.domains))


@pytest.mark.parametrize("name", sorted(_CITY_TREES))
def test_the_city_trees_fit_the_24_gib_card_they_ran_on(tmp_path, name):
    """The queue row's urban half.  Both 750 m BEP+BEM trees ran to the
    end on a 24 GiB RTX 4090; priced on that card's own profile, the
    envelope now fits the whole-process budget `gpuwm check` gives a
    24 GiB card with 23.5 GiB free (23.00 GiB: free less the 0.5 GiB
    other-process margin), and the arithmetic before A163 did not."""
    exp = _city_tree(tmp_path, name)
    estimate = pf.estimate_experiment(exp, profile=_rtx_4090())
    budget = int(23.5 * GIB) - pf.EXTERNAL_MARGIN_BYTES
    assert estimate.peak_envelope_bytes <= budget, (
        estimate.peak_envelope_bytes / GIB)
    subtotal = estimate.subtotal_bytes
    old_estimate = math.ceil(1.15 * subtotal)
    old = (old_estimate + estimate.envelope_intercept_bytes
           + pf.ENVELOPE_UNMODELLED_BYTES
           + math.ceil(0.05 * old_estimate) + math.ceil(0.20 * old_estimate))
    assert old > budget
    # The held arrays are most of such a tree: the margin on the whole
    # subtotal priced headroom the pool never holds for them.
    assert estimate.held_exact_bytes > subtotal // 2


def test_the_measured_urban_rows_hold_the_pricing():
    """The urban battery rows: the peak less the held arrays stays under
    the measured margin on the rest, and under the envelope, on both
    cards."""
    urban = [row for row in pf.FORECAST_PEAK_BATTERY if row[8]]
    assert {row[1] for row in urban} >= {"RTX 5070 Ti", "RTX 5090"}
    for row in urban:
        assert _ratio(row) <= pf.FORECAST_PEAK_RATIO_MEASURED, row[:2]
