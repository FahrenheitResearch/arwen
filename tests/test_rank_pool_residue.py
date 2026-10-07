"""A13: a device rank's pool margin is bounded in bytes, and full HRRR
fits four 32 GB cards.

The breakage the rank gate prevents is a CUDA out-of-memory on one card
mid-run, so every rank whose pool-held peak was recorded must stay under
its price.  The defect this fixes: the A163 ratio (1.09 measured + 0.04
safety), measured on 3.5-11 GiB single-card subtotals where the pool's
residue over the itemization reads as 9-17%, priced a 3.3 GiB margin on
each 25-26 GiB rank of full HRRR (1797 x 1057 x 50, HRRR physics) split
2x2 over four RTX 5090, and refused a run whose four cards measured
27.6-29.5 GiB at their peaks against a 30.86 GiB budget.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import preflight as pf
from gpuwm.core.devices import DeviceOptions
from gpuwm.core.devices_memory import (devices_gate, estimate_devices,
                                       rank_lake_column_bounds)
from gpuwm.core.device_inventory import DeviceLocalMemoryProfile
from gpuwm.core.streaming import ranked_halo, ranked_specs

GIB = pf.GIB

#: The card the A13 receipts were measured on: 170 SMs x 1,536 threads,
#: the default 1 KiB stack, frames read at sm_120 / NVRTC 12.9.86.
RTX_5090 = DeviceLocalMemoryProfile(
    name="NVIDIA GeForce RTX 5090", multiprocessor_count=170,
    max_threads_per_multiprocessor=1536, default_stack_limit_bytes=1024,
    compile_platform=("120", "12.9.86"))

#: The budget box E's four cards presented (device free bytes at start;
#: the run report read 30.76-30.81 GiB, the CPU receipts priced 30.86).
BUDGET = int(30.86 * GIB)

#: The full HRRR door experiment the A13 kit measured, as shipped in the
#: tree.  Its 24 h and start time are irrelevant to the rank price: the
#: ranked road retains one forcing interval on the card.
FULL_HRRR_TOML = (Path(__file__).resolve().parents[1]
                  / "tools" / "meeting_perf_a13" / "hrrr-door-c24s.toml")

#: Each rank's lake-eligible columns as the prepared statics bounded
#: them on box E (price22.json of the receipts), per layout.
LAKE_BOUNDS = {"2x2": (1042, 1959, 4803, 35177),
               "1x4": (2071, 4038, 31730, 9004)}


def _full_hrrr():
    from gpuwm.experiment import load_experiment
    return load_experiment(str(FULL_HRRR_TOML))


def _layout(exp, grid, lake_bounds):
    """The experiment under ``[devices]`` ``grid`` and a lake mask that
    reproduces the receipts' per-rank bounds (cells deep inside each
    rank's interior, outside every neighbour's window)."""
    gy, gx = (int(v) for v in grid.split("x"))
    n = gy * gx
    cfg = exp.root.run
    opts = DeviceOptions(count=n, grid=(gy, gx), ids=tuple(range(n)),
                         transport="auto")
    halo = ranked_halo(cfg)
    specs = ranked_specs(cfg, opts, halo=halo)
    mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    for spec, count in zip(specs, lake_bounds):
        rows = np.arange(spec.cj0 + 2 * halo, spec.cj0 + spec.cny - 2 * halo)
        cols = np.arange(spec.ci0 + 2 * halo, spec.ci0 + spec.cnx - 2 * halo)
        block = np.zeros((len(rows), len(cols)), dtype=bool)
        block.flat[:count] = True
        mask[np.ix_(rows, cols)] |= block
    assert tuple(rank_lake_column_bounds(specs, mask)) == tuple(lake_bounds)
    return replace(exp, devices=opts), opts, mask


def _price(grid):
    exp, opts, mask = _layout(_full_hrrr(), grid, LAKE_BOUNDS[grid])
    n = opts.count
    return estimate_devices(
        exp, options=opts, forcing_intervals=2, source="rap-native",
        budgets={i: BUDGET for i in range(n)}, lake_mask=mask,
        profiles={i: RTX_5090 for i in range(n)})


def test_the_bound_is_the_widest_measured_residue_plus_the_swing():
    """The rank bound IS the receipts' worst pool residue plus the
    card-to-card swing, rounded up to the quarter gibibyte -- not a
    taste.  Every receipt with a pool-held peak counts, single-card and
    ranked alike, so the bound is the most conservative the data
    supports."""
    receipts = pf._pool_residue_receipts()
    assert len(receipts) >= 18
    worst = max(held - itemized for _l, itemized, held in receipts)
    assert pf.POOL_RESIDUE_MEASURED_BYTES == worst
    assert 1.8 * GIB < worst < 1.85 * GIB
    assert 0.33 * GIB < pf.POOL_RESIDUE_SWING_BYTES < 0.34 * GIB
    quarter = GIB // 4
    assert pf.RANK_POOL_RESIDUE_BOUND_BYTES % quarter == 0
    assert 0 <= pf.RANK_POOL_RESIDUE_BOUND_BYTES - (
        worst + pf.POOL_RESIDUE_SWING_BYTES) < quarter
    assert pf.RANK_POOL_RESIDUE_BOUND_BYTES == int(2.25 * GIB)


@pytest.mark.parametrize("row", pf.RANK_POOL_RESIDUE_BATTERY,
                         ids=lambda r: f"{r[2]} rank {r[3]}")
def test_every_measured_rank_is_under_its_bounded_envelope(row):
    """The gate's job on the ranked road, rank by rank: the measured
    pool-held peak stays under the itemized pool plus the bound, and the
    measured device peak under the whole envelope (bounded pool plus the
    itemized non-pool plus the unmodelled residue)."""
    _label, _card, _layout, _rank, itemized, held, peak, non_pool = row
    assert held - itemized <= pf.RANK_POOL_RESIDUE_BOUND_BYTES - (
        pf.POOL_RESIDUE_SWING_BYTES)
    envelope = (itemized + pf.RANK_POOL_RESIDUE_BOUND_BYTES + non_pool
                + pf.ENVELOPE_UNMODELLED_BYTES)
    assert envelope >= peak, (envelope / GIB, peak / GIB)
    # And the residue the ratio priced on these ranks is what the fix
    # retires: at least 1 GiB per rank above what the pool held.
    ratio_margin = math.ceil(pf.FORECAST_POOL_HEADROOM * itemized) - itemized
    assert ratio_margin - (held - itemized) > GIB


def test_the_cap_leaves_small_subtotals_on_the_ratio():
    """Below the crossover the price is the A163 price to the byte; above
    it the margin is the bound."""
    cap = pf.RANK_POOL_RESIDUE_BOUND_BYTES
    h = pf.FORECAST_POOL_HEADROOM
    crossover = cap / (h - 1)
    assert 17 * GIB < crossover < 18 * GIB
    for subtotal in (2 * GIB, 8 * GIB, 11 * GIB, int(crossover) - 1):
        assert pf.forecast_pool_estimate_bytes(
            subtotal, headroom=h, residue_cap_bytes=cap) == (
            pf.forecast_pool_estimate_bytes(subtotal, headroom=h))
    for subtotal in (18 * GIB, 25 * GIB, 64 * GIB):
        assert pf.forecast_pool_estimate_bytes(
            subtotal, headroom=h, residue_cap_bytes=cap) == subtotal + cap
    # Held arrays (urban) stay outside the margin either way.
    assert pf.forecast_pool_estimate_bytes(
        30 * GIB, held_exact_bytes=20 * GIB, headroom=h,
        residue_cap_bytes=cap) == 30 * GIB + math.ceil(
            (h - 1) * 10 * GIB)


def test_full_hrrr_2x2_is_admitted_on_four_32gb_cards():
    """The queue row itself: full HRRR, HRRR physics, [devices] 2x2 on
    four RTX 5090 at box E's budget, priced with the prepared statics'
    lake bounds, is ADMITTED on every card by default, and each card's
    price still bounds the widest peak any rank of this grid measured
    (29.54 GiB, 1x4 rank 2 on box E)."""
    estimate = _price("2x2")
    gate = devices_gate(estimate, budgets={i: BUDGET for i in range(4)})
    assert not gate["refuse"], gate["verdict"]
    assert gate["verdict"].count("ADMITTED") == 4
    assert "pool margin per rank is min(" in gate["verdict"]
    widest_measured = max(row[6] for row in pf.RANK_POOL_RESIDUE_BATTERY)
    for card in estimate["cards"]:
        assert card["total_bytes"] <= BUDGET
        assert card["total_bytes"] >= widest_measured - 0.6 * GIB
    # The price fell from 31.23-31.66 GiB (the A13 receipts, refused) by
    # the margin the ratio priced beyond the bound; the admission then
    # widens the MYNN column chunk into the room that fits (A3c), so the
    # admitted price sits under the budget, not far under it.
    totals = sorted(card["total_bytes"] / GIB for card in estimate["cards"])
    assert 29.9 < totals[0], totals
    # At the narrowest MYNN chunk (the width the receipts measured) the
    # same four ranks price 30.4-30.8 GiB: the bound alone, no widening.
    narrow = max(rank["mynn_column_chunk"] for rank in estimate["rank_shapes"])
    assert narrow >= 8192


def test_full_hrrr_1x4_prices_bound_the_measured_cards():
    """The layout the receipts measured: with the bound, every card's
    price covers the peak that card measured, and the two end ranks are
    admitted; the two middle ranks (two seams each) price 31.45 and
    31.76 GiB, 0.6 and 0.9 GiB over the CPU receipts' 30.86 GiB budget,
    and stay refused, 2.0 to 2.2 GiB above what they measured (29.44
    and 29.54 GiB)."""
    estimate = _price("1x4")
    measured = {row[3]: row[6] for row in pf.RANK_POOL_RESIDUE_BATTERY}
    for card in estimate["cards"]:
        assert card["total_bytes"] >= measured[card["card"]], card
    gate = devices_gate(estimate, budgets={i: BUDGET for i in range(4)})
    admitted = [card["total_bytes"] <= BUDGET for card in estimate["cards"]]
    assert admitted == [True, False, False, True], gate["verdict"]


def test_the_loader_and_template_keep_the_ratio():
    """Only device ranks carry the bound; the loader slab and the
    retained template are small resident domains priced on the ratio."""
    estimate = _price("2x2")
    assert estimate["rank_pool_residue_bound_bytes"] == (
        pf.RANK_POOL_RESIDUE_BOUND_BYTES)
    assert "receipt" in estimate["rank_pool_basis"]
    # A template of the full width and the slab-row height is far under
    # the crossover, so its price is the ratio's whether capped or not.
    exp = _full_hrrr()
    cfg = exp.root.run
    from gpuwm.ingest.prepared_store import default_slab_rows
    from tilestream.harness import tile_config
    rows = default_slab_rows(cfg.nx, cfg.ny)
    small = replace(exp, domains=(replace(
        exp.root, run=tile_config(cfg, cfg.nx, rows)),))
    plain = pf.estimate_experiment(small, forcing_intervals=1, profile=RTX_5090)
    capped = pf.estimate_experiment(
        small, forcing_intervals=1, profile=RTX_5090,
        pool_residue_cap_bytes=pf.RANK_POOL_RESIDUE_BOUND_BYTES)
    assert plain.peak_envelope_bytes == capped.peak_envelope_bytes
    assert not capped.pool_residue_cap_active


def test_a_gap_row_suite_keeps_the_ratio_on_the_ranked_road():
    """The bound covers the pool's residue over a COMPLETE itemization.
    A suite an itemization-gap row raised the headroom for (HRRR physics
    under RTE-RRTMGP: pool USED 1.16x the subtotal, arrays the forecast
    allocates and the itemization does not list, a shortfall that grows
    with the rank) keeps the ratio its row measured: on a 25 GiB rank
    the unlisted arrays alone take about 4.1 GiB, which a 2.25 GiB bound
    would under-price by 1.8 GiB per card, and the gate would admit a
    card that then allocates past its price."""
    exp = _full_hrrr()
    cfg = exp.root.run
    gap = replace(cfg, ra_rrtmg_variant="rte-rrtmgp")
    assert pf.forecast_pool_headroom([cfg]) == pf.FORECAST_POOL_HEADROOM
    assert pf.forecast_pool_headroom([gap]) > pf.FORECAST_POOL_HEADROOM
    assert pf.rank_pool_residue_cap_bytes([cfg]) == (
        pf.RANK_POOL_RESIDUE_BOUND_BYTES)
    assert pf.rank_pool_residue_cap_bytes([gap]) is None
    # estimate_experiment drops the cap for the gap-row suite whatever
    # a caller passes: the rank prices at the ratio to the byte.
    from tilestream.harness import tile_config
    halo = ranked_halo(gap)
    gy, gx = 2, 2
    opts = DeviceOptions(count=4, grid=(gy, gx), ids=(0, 1, 2, 3),
                         transport="auto")
    spec = ranked_specs(gap, opts, halo=halo)[0]
    rank = replace(exp, domains=(replace(
        exp.root, run=tile_config(gap, spec.cnx, spec.cny)),))
    plain = pf.estimate_experiment(rank, forcing_intervals=1,
                                   tile_buffer=True, profile=RTX_5090)
    capped = pf.estimate_experiment(
        rank, forcing_intervals=1, tile_buffer=True, profile=RTX_5090,
        pool_residue_cap_bytes=pf.RANK_POOL_RESIDUE_BOUND_BYTES)
    assert capped.pool_residue_cap_bytes is None
    assert not capped.pool_residue_cap_active
    assert capped.alloc_estimate_bytes == plain.alloc_estimate_bytes
    assert plain.subtotal_bytes > 17 * GIB
    assert plain.alloc_estimate_bytes - plain.subtotal_bytes > (
        pf.RANK_POOL_RESIDUE_BOUND_BYTES + GIB)
    # The ranked estimate carries no bound for this suite and the gate
    # says which margin applied and why.
    gap_exp, gap_opts, mask = _layout(
        replace(exp, domains=(replace(exp.root, run=gap),)), "2x2",
        LAKE_BOUNDS["2x2"])
    estimate = estimate_devices(
        gap_exp, options=gap_opts, forcing_intervals=2, source="rap-native",
        budgets={i: BUDGET for i in range(4)}, lake_mask=mask,
        profiles={i: RTX_5090 for i in range(4)})
    assert estimate["rank_pool_residue_bound_bytes"] is None
    assert estimate["rank_pool_headroom"] == pf.forecast_pool_headroom([gap])
    verdict = devices_gate(estimate, budgets={i: BUDGET for i in range(4)})["verdict"]
    assert "no byte bound" in verdict
    assert "itemization-gap row" in verdict
    assert "min(" not in verdict
    # And the legacy-RRTMG door experiment (the validated case) still
    # carries the bound and the gate still prints it.
    legacy = _price("2x2")
    assert legacy["rank_pool_residue_bound_bytes"] == (
        pf.RANK_POOL_RESIDUE_BOUND_BYTES)
    assert legacy["rank_pool_headroom"] == pf.FORECAST_POOL_HEADROOM
    assert "min(" in devices_gate(legacy, budgets={i: BUDGET for i in range(4)})["verdict"]


def test_the_bound_applies_only_to_suites_with_a_rank_receipt():
    """A13 review: the rank receipts are HRRR physics under legacy RRTMG
    only, so only that suite (RANK_POOL_RESIDUE_SUITES) carries the
    bound.  The single-card battery shows the default suite's pool-held
    residue growing with the subtotal (0.73 GiB at 3.53 GiB, 1.82 GiB at
    8.43 GiB, about 21%), so on a 27 GiB rank it would exceed the 2.25
    GiB bound by up to 1.5 GiB and the gate would admit a card that then
    allocates past its price.  Such a suite keeps the ratio on the ranked
    road until a ranked run of it records its pool-held peak."""
    exp = _full_hrrr()
    cfg = exp.root.run
    # The receipts' suite matches the table; the door experiment is it.
    assert pf.rank_pool_residue_suite([cfg]) is pf.RANK_POOL_RESIDUE_SUITES[0]
    assert pf.rank_pool_residue_cap_bytes([cfg]) == (
        pf.RANK_POOL_RESIDUE_BOUND_BYTES)
    assert pf.rank_pool_residue_basis([cfg]) == pf.RANK_POOL_RESIDUE_BASIS
    # The battery's own numbers: the default suite's residue is
    # proportional to its subtotal in the only two receipts it has.
    rows = {(row[0], row[1]): row for row in pf.FORECAST_PEAK_BATTERY}
    small = rows["default suite, legacy RRTMG 300x300x49", "RTX 5070 Ti"]
    large = rows["default suite, legacy RRTMG 500x500x49", "RTX 5070 Ti"]
    for row in (small, large):
        assert 0.19 < (row[7] - row[4]) / row[4] < 0.23, row
    assert (large[7] - large[4]) * 27 * GIB / large[4] > (
        pf.RANK_POOL_RESIDUE_BOUND_BYTES + 3 * GIB)
    # The default suite (A163 battery's), legacy RRTMG, on the ranked
    # road: no bound, the ratio's price to the byte, and the gate says
    # why.
    default = replace(cfg, mp_physics=8, bl_pbl_physics=1,
                      sf_sfclay_physics=91, sf_surface_physics=2,
                      num_soil_layers=4)
    assert pf.forecast_pool_headroom([default]) == pf.FORECAST_POOL_HEADROOM
    assert pf.rank_pool_residue_suite([default]) is None
    assert pf.rank_pool_residue_cap_bytes([default]) is None
    assert pf.rank_pool_residue_basis([default]) == (
        pf.RANK_POOL_RESIDUE_UNMEASURED_BASIS)
    # One domain off the table takes the forecast off it.
    assert pf.rank_pool_residue_cap_bytes([cfg, default]) is None
    # A cumulus scheme on the receipts' suite is a suite without a
    # receipt.
    assert pf.rank_pool_residue_cap_bytes([replace(cfg, cu_physics=3)]) is None
    # So is RRTMG on one radiation stream beside another scheme on the
    # other: every rank receipt ran RRTMG on both (A13 review 3).
    mixed = replace(cfg, ra_physics=0, ra_lw_physics=4, ra_sw_physics=1)
    assert pf.radiation_scheme_ids(mixed) == (4, 1)
    assert pf.rank_pool_residue_suite([mixed]) is None
    assert pf.rank_pool_residue_cap_bytes([mixed]) is None
    assert pf.rank_pool_residue_basis([mixed]) == (
        pf.RANK_POOL_RESIDUE_UNMEASURED_BASIS)
    from tilestream.harness import tile_config
    halo = ranked_halo(default)
    opts = DeviceOptions(count=4, grid=(2, 2), ids=(0, 1, 2, 3),
                         transport="auto")
    spec = ranked_specs(default, opts, halo=halo)[0]
    rank = replace(exp, domains=(replace(
        exp.root, run=tile_config(default, spec.cnx, spec.cny)),))
    plain = pf.estimate_experiment(rank, forcing_intervals=1,
                                   tile_buffer=True, profile=RTX_5090)
    capped = pf.estimate_experiment(
        rank, forcing_intervals=1, tile_buffer=True, profile=RTX_5090,
        pool_residue_cap_bytes=pf.RANK_POOL_RESIDUE_BOUND_BYTES)
    assert capped.pool_residue_cap_bytes is None
    assert not capped.pool_residue_cap_active
    assert capped.alloc_estimate_bytes == plain.alloc_estimate_bytes
    assert plain.subtotal_bytes > 17 * GIB
    assert plain.alloc_estimate_bytes - plain.subtotal_bytes > (
        pf.RANK_POOL_RESIDUE_BOUND_BYTES)
    default_exp, default_opts, mask = _layout(
        replace(exp, domains=(replace(exp.root, run=default),)), "2x2",
        LAKE_BOUNDS["2x2"])
    estimate = estimate_devices(
        default_exp, options=default_opts, forcing_intervals=2,
        source="rap-native", budgets={i: BUDGET for i in range(4)},
        lake_mask=mask, profiles={i: RTX_5090 for i in range(4)})
    assert estimate["rank_pool_residue_bound_bytes"] is None
    assert estimate["rank_pool_headroom"] == pf.FORECAST_POOL_HEADROOM
    verdict = devices_gate(estimate, budgets={i: BUDGET for i in range(4)})["verdict"]
    assert "no byte bound" in verdict
    assert "RANK_POOL_RESIDUE_SUITES" in verdict
    assert "min(" not in verdict
