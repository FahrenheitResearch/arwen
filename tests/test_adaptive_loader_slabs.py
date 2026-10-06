"""Loader batching spends an existing memory price and retains one row."""
from dataclasses import replace

import pytest

from gpuwm.core import prepared_tile_memory as ptm, preflight as pf
from gpuwm.ingest import prepared_store as store
from test_prepared_tile_memory import experiment, priced, profile


@pytest.mark.parametrize("nx,ny", [(12, 1), (563, 326), (1797, 1057), (8192, 137)])
@pytest.mark.parametrize("rows", [1, 2, 16, 32, 64])
def test_partition_covers_each_mass_row_once_and_retains_one(nx, ny, rows):
    plan = store._plan_loader_slabs(ny, rows, 1)
    assert plan[-1] == (ny - 1, 1)
    assert [j for start, height in plan for j in range(start, start + height)] == list(range(ny))
    assert all(1 <= height <= rows for _, height in plan)


def test_explicit_partition_keeps_its_existing_tail():
    assert store._plan_loader_slabs(1057, 64, 0) == store._plan_slabs(1057, 64)


def test_retained_template_cannot_exceed_the_priced_slab():
    with pytest.raises(store.PreparedStoreError, match="unpriced device state"):
        store._plan_loader_slabs(1057, 4, 64)


def test_unknown_budget_retains_original_column_policy():
    exp = experiment(2174, 1352, root_dx_m=5000.)
    model = ptm.PreparedTileMemory(exp, profile(), 2, domain=exp.root, options=exp.tiles)
    fixed = model.fixed_terms()
    rows = store.default_slab_rows(2174, 1352)
    assert fixed["loader_rows"] == rows == 1
    assert fixed["template_rows"] == 1


@pytest.mark.parametrize("free_gib", [6.41, 6.54, 16.0, 31.4])
def test_known_budget_charges_full_loader_envelope_and_its_next_row(free_gib):
    exp = experiment(2174, 1352, root_dx_m=5000.)
    machine, _, footprint = priced(exp, free_gib)
    model = footprint.prepared_memory
    fixed = model.fixed_terms()
    rows = fixed["loader_rows"]
    assert fixed["template_rows"] == 1
    assert 1 <= rows <= 64
    assert model.vram_bytes(37 * 37 * 49, 1) <= model.loader_budget_bytes
    if rows < 64:
        changed = replace(model, _cache={})
        larger = changed._domain(exp.root.run.nx, rows + 1)
        candidate = dict(fixed, loader_rows=rows + 1,
                         loader_pool_peak_bytes=larger.resident_bytes + larger.transient_bytes)
        changed._cache["fixed"] = candidate
        assert changed.vram_bytes(37 * 37 * 49, 1) > model.loader_budget_bytes


def test_known_large_card_uses_existing_64_row_ceiling():
    exp = experiment(563, 326)
    _, _, footprint = priced(exp, 31.4)
    assert footprint.prepared_memory.fixed_terms()["loader_rows"] == 64
