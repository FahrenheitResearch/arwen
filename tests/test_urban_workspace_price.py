"""A176: the prepared doors price BEP+BEM's column workspace from the land
cover they restore.

The configuration door (``gpuwm check CONFIG``, ``gpuwm go``'s gate before
the fetch) runs before the land cover exists, so it prices the column
workspace at every column urban: the plan's allocation once a domain holds
4,792 urban columns at 59 levels, and up to 1 GiB per domain above it below
that.  The prepared single-domain and tree runners and the mid-run spawn
check hold the land cover, so they count its urban columns with the rule
``urban_var_init`` uses and price the plan ``urban_bem.ColumnPlan`` builds
from exactly those columns.

The breakages these tests keep away: a prepared price BELOW the plan the
run allocates (a CUDA out-of-memory the admission was there to prevent),
and a prepared run refused at the configuration's bound although it fits
(the defect).  The configuration door keeps the bound and says so.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import preflight as pf
from gpuwm.core import urban_bem
from gpuwm.core.urban_state import (_column_workspace_shapes, option_spec,
                                    prepared_urban_columns,
                                    resolve_dimensions,
                                    urban_categories_for,
                                    urban_column_count, urban_columns_line,
                                    urban_var_init_host)
from gpuwm.core.urban_tables import load_urban_params

GIB = pf.GIB
MODIS = "MODIFIED_IGBP_MODIS_NOAH"
NZ = 59
#: The plan's column cap at 59 levels: FORECAST_WORKSPACE_BYTES over one
#: column's workspace.
CAP = urban_bem.FORECAST_WORKSPACE_BYTES // urban_bem.workspace_bytes_per_column(NZ)
#: What one urban column below the cap adds to the plan: its workspace and
#: its int32 error word.
PER_COLUMN = urban_bem.workspace_bytes_per_column(NZ) + 4


def _city_tree(tmp_path, *, sf_urban_physics=3):
    """The scaled BEP+BEM city tree measured on the RTX 5070 Ti (A163):
    a 2.25 km parent 216 x 216, a 750 m nest 162 x 162, 59 levels."""
    from gpuwm.experiment import load_experiment

    text = f"""
[experiment]
name = "urban-price"
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
nz = {NZ}
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
sf_urban_physics = {sf_urban_physics}
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
nx = 216
ny = 216
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
i_parent_start = 95
j_parent_start = 71
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 162
ny = 162
specified = false
nested = true
history_interval_s = 600.0
radt = 3.0
cu_physics = 0
"""
    path = tmp_path / "urban-price.toml"
    path.write_text(text, encoding="utf-8")
    return load_experiment(path)


def _land_use(shape, urban, *, lcz_class=55, seed=0):
    """A land use with exactly ``urban`` LCZ columns among grassland (10)
    and water (17), placed at random."""
    rng = np.random.default_rng(seed)
    lu = rng.choice(np.array([10, 17], dtype=np.int32), size=shape)
    flat = lu.reshape(-1)
    flat[rng.choice(flat.size, size=urban, replace=False)] = lcz_class
    return lu


def _bem_bytes(estimate):
    return sum(item.nbytes for domain in estimate.domains
               for item in domain.items
               if item.name == "fields/bem_column_workspace")


@pytest.mark.parametrize("urban", [0, 1, 37, CAP - 1, CAP, CAP + 1, 46_656])
def test_the_count_prices_what_the_column_plan_allocates(monkeypatch, urban):
    """The price at a known count IS ``ColumnPlan``'s allocation: its
    workspace (none on a domain with no urban column) and its error words,
    built by the class itself on the host.  Below the price would admit a
    run that cannot allocate; above it is the defect."""
    ny, nx = 216, 216
    frc = np.zeros((ny, nx), np.float32)
    frc.reshape(-1)[:urban] = 0.85
    monkeypatch.setitem(sys.modules, "cupy", np)
    plan = urban_bem.ColumnPlan(frc, NZ, urban_bem.FORECAST_WORKSPACE_BYTES)
    monkeypatch.delitem(sys.modules, "cupy")
    shapes = _column_workspace_shapes(3, ny * nx, NZ, urban_columns=urban)
    workspace = 0 if plan.ws is None else plan.ws.nbytes
    assert 4 * shapes["bem_column_workspace"][0] == workspace
    assert 4 * shapes["bem_column_errors"][0] == plan.err.nbytes
    # Without a count (the configuration door) the price is the bound:
    # every column urban, which is the allocation from the cap upward.
    bound = _column_workspace_shapes(3, ny * nx, NZ)
    assert bound["bem_column_workspace"] >= shapes["bem_column_workspace"]
    if urban >= CAP:
        assert bound == shapes


def test_the_count_is_the_cold_starts_urban_columns():
    """``urban_column_count`` reads the columns ``urban_var_init`` leaves
    with an urban fraction above 0, which are the columns ``ColumnPlan``
    lists: LCZ classes and the legacy urban class, an input fraction kept
    or replaced from the table, water and grass left out."""
    shape = (24, 30)
    lu = _land_use(shape, 80, lcz_class=52, seed=3)
    lu.reshape(-1)[:5] = 61                       # more LCZ classes
    frc_in = np.zeros(shape, np.float32)
    frc_in.reshape(-1)[:200] = np.linspace(-0.5, 1.5, 200)
    cfg = SimpleNamespace(sf_surface_physics=4)
    categories = urban_categories_for(cfg, MODIS)
    params = load_urban_params(3, 1)
    one = np.ones(shape, np.float32)
    soil = np.ones((4,) + shape, np.float32)
    for given in (None, frc_in):
        out = urban_var_init_host(
            option=3, use_wudapt_lcz=1, params=params,
            categories=categories, ivgtyp=lu, tsk=one * 290,
            tslb=soil * 285, tmn=one * 280, smois=soil * 0.3,
            frc_urb2d=given, nz=NZ,
            dims=resolve_dimensions(3, urban_bem),
            spec=option_spec(3, urban_bem))
        want = int(np.count_nonzero(out["frc_urb2d"] > 0))
        assert want == int(np.count_nonzero(np.isin(lu, range(51, 62))))
        assert urban_column_count(
            lu, categories=categories, use_wudapt_lcz=1,
            frc_urb_tbl=params.FRC_URB_TBL, frc_urb2d=given) == want


@pytest.mark.parametrize("lsm", [2, 4])
def test_the_categories_are_the_ones_the_runtime_attaches(lsm):
    """``gpuwm.core.physics._attach_urban`` reads Noah-MP's land-use
    identity under Noah-MP and VEGPARM's urban rows under Noah; the count
    must read the same categories, or it counts other columns than the
    plan."""
    from gpuwm.core.noahmp import load_noahmp_parameters
    from gpuwm.core.urban_tables import UrbanCategories, urban_category_set

    got = urban_categories_for(SimpleNamespace(sf_surface_physics=lsm), MODIS)
    _, veg = load_noahmp_parameters().vegetation_groups(MODIS)
    if lsm == 4:
        from gpuwm.core.noahmp_runtime import NoahmpRuntimeParameters

        identity = NoahmpRuntimeParameters(
            dataset_identifier=MODIS).land_use
        want = UrbanCategories(isurban=int(identity.isurban),
                               natural=int(identity.natural),
                               lcz=tuple(int(v) for v in identity.lcz))
    else:
        want = urban_category_set(MODIS, isurban=int(veg.scalar("ISURBAN")))
    assert got == want


def test_the_prepared_price_moves_by_the_workspace_alone(tmp_path):
    """Priced at a small city's count, the estimate and the envelope fall
    by exactly the workspace the bound charged for rural columns, and by
    nothing else; at a city of the cap or more nothing moves."""
    exp = _city_tree(tmp_path)
    bound = pf.estimate_experiment(exp)
    small = {1: 2_000, 2: 3_000}
    priced = pf.estimate_experiment(exp, urban_columns=small)
    saved = sum((CAP - count) * PER_COLUMN for count in small.values())
    assert _bem_bytes(bound) - _bem_bytes(priced) == sum(
        (CAP - count) * (PER_COLUMN - 4) for count in small.values())
    assert bound.held_exact_bytes - priced.held_exact_bytes == saved
    assert bound.alloc_estimate_bytes - priced.alloc_estimate_bytes == saved
    assert bound.peak_envelope_bytes - priced.peak_envelope_bytes == saved
    big = pf.estimate_experiment(exp, urban_columns={1: CAP, 2: 10 * CAP})
    assert big.peak_envelope_bytes == bound.peak_envelope_bytes
    # A domain the map does not name keeps the bound.
    one = pf.estimate_experiment(exp, urban_columns={2: 3_000})
    assert (bound.peak_envelope_bytes - one.peak_envelope_bytes
            == (CAP - 3_000) * PER_COLUMN)
    # The admission estimate a door weighs takes the same reading.
    assert (pf.admission_estimate(exp, urban_columns=small)
            .peak_envelope_bytes == priced.peak_envelope_bytes)


def test_a_prepared_run_whose_real_workspace_does_not_fit_is_refused(
        tmp_path, monkeypatch):
    """The card between the two prices: a prepared door that reads a small
    city admits the run the configuration's bound refused, and one that
    reads a big city on the same card is refused before anything is
    allocated, naming the envelope it read."""
    from gpuwm.core import streaming as st
    from gpuwm.core.resident_admission import (MEMORY_GATE_OVERRIDE_ENV,
                                               ResidentMemoryRefused)

    monkeypatch.delenv(MEMORY_GATE_OVERRIDE_ENV, raising=False)
    exp = _city_tree(tmp_path)
    small = {1: 500, 2: 500}
    need_small = pf.admission_estimate(
        exp, urban_columns=small).peak_envelope_bytes
    need_bound = pf.admission_estimate(exp).peak_envelope_bytes
    assert need_bound - need_small > GIB
    card = SimpleNamespace(vram_bytes=(need_small + need_bound) // 2,
                           device_profile=None)
    record = st.admit_resident_road(exp, None, machine=card,
                                    urban_columns=small)
    assert record["fits"] is True and record["need_bytes"] == need_small
    with pytest.raises(ResidentMemoryRefused) as refused:
        st.admit_resident_road(exp, None, machine=card)
    assert refused.value.need_bytes == need_bound
    with pytest.raises(ResidentMemoryRefused) as refused:
        st.admit_resident_road(exp, None, machine=card,
                               urban_columns={1: CAP, 2: 162 * 162})
    assert refused.value.need_bytes == need_bound
    assert "refused before anything was allocated" in str(refused.value)


def test_the_tree_door_reads_each_domains_land_cover(tmp_path):
    """The tree door counts each BEP+BEM domain off the static it restores
    (a prepared cache's LU_INDEX on the native land-use identity its
    physics attaches with; a wrfinput's own LU_INDEX and FRC_URB2D on the
    file's dataset), and says so in one line."""
    from gpuwm.prepared_domain_tree_forecast import tree_urban_columns

    exp = _city_tree(tmp_path)
    lu1 = _land_use((216, 216), 1_234, seed=1)
    lu2 = _land_use((162, 162), 5_678, lcz_class=52, seed=2)
    inputs = SimpleNamespace(experiment=exp, domains=(
        SimpleNamespace(grid_id=1, static_fields={"LU_INDEX": lu1}),
        SimpleNamespace(grid_id=2, static_fields={"LU_INDEX": lu2})))
    counts = tree_urban_columns(inputs)
    assert counts == {1: 1_234, 2: 5_678}
    line = urban_columns_line(counts)
    assert "d01 1,234" in line and "d02 5,678" in line
    assert "not at every column urban" in line
    # A wrfinput bundle: the file's own land use, fraction and dataset.
    frc = np.zeros((162, 162), np.float32)
    wrf = SimpleNamespace(
        grid_id=2, static_fields={},
        restored=SimpleNamespace(raw={"LU_INDEX": lu2, "FRC_URB2D": frc}),
        geog_selection=SimpleNamespace(
            landuse_global_attrs=lambda: {"MMINLU": MODIS}))
    inputs = SimpleNamespace(experiment=exp, domains=(inputs.domains[0], wrf))
    assert tree_urban_columns(inputs) == {1: 1_234, 2: 5_678}
    # No BEP+BEM, no reading: the price never depended on the count, and
    # a door whose bundles name no land-use dataset (the wrfinput and
    # met_em doors' fixtures) is not asked for one.
    plain = _city_tree(tmp_path, sf_urban_physics=2)
    assert tree_urban_columns(SimpleNamespace(
        experiment=plain, domains=inputs.domains)) is None
    bare = SimpleNamespace(
        grid_id=2, static_fields={}, restored=SimpleNamespace(raw={}),
        geog_selection=SimpleNamespace(landuse_global_attrs=lambda: {}))
    assert tree_urban_columns(SimpleNamespace(
        experiment=plain, domains=(inputs.domains[0], bare))) is None
    # A BEP+BEM bundle that names no dataset keeps the bound for itself.
    assert tree_urban_columns(SimpleNamespace(
        experiment=exp, domains=(inputs.domains[0], bare))) == {1: 1_234}


@pytest.mark.parametrize("moving", ["follow", "spawn"])
def test_a_domain_whose_ground_can_change_keeps_the_bound(tmp_path, moving):
    """A mover and a spawn nest, and every domain under them, hold a land
    cover other than the one restored at the start, so the tree door does
    not count them and they keep the configuration's bound."""
    from gpuwm.prepared_domain_tree_forecast import tree_urban_columns

    exp = _city_tree(tmp_path)
    run = exp.domains[0].run
    domains = (SimpleNamespace(grid_id=1, parent_id=0, run=run),
               SimpleNamespace(grid_id=2, parent_id=1, run=run,
                               **{moving: object()}),
               SimpleNamespace(grid_id=3, parent_id=2, run=run))
    fake = SimpleNamespace(domains=domains, relocation=None)
    lu = _land_use((216, 216), 900, seed=4)
    bundles = tuple(SimpleNamespace(grid_id=g, static_fields={"LU_INDEX": lu})
                    for g in (1, 2, 3))
    assert tree_urban_columns(
        SimpleNamespace(experiment=fake, domains=bundles)) == {1: 900}


def test_the_spawn_check_reads_the_newborns_statics(tmp_path):
    """The mid-run spawn check prices a newborn from the statics it is
    about to adopt; a follower, and statics that name no land-use
    dataset, keep the bound."""
    from gpuwm.ingest.nest_spawn_init import spawn_urban_columns

    exp = _city_tree(tmp_path)
    child = exp.domains[1]
    lu = _land_use((162, 162), 2_500, seed=5)
    statics = {"static_fields": {"LU_INDEX": lu},
               "landuse_attrs": {"MMINLU": MODIS}, "receipt": {}}
    assert spawn_urban_columns(child, statics) == 2_500
    assert spawn_urban_columns(child, None) is None
    assert spawn_urban_columns(
        child, dict(statics, landuse_attrs={})) is None
    assert spawn_urban_columns(
        SimpleNamespace(follow=object(), run=child.run), statics) is None
    from gpuwm.core.preflight import urban_held_bytes
    assert (urban_held_bytes(child.run)
            - urban_held_bytes(child.run, urban_columns=2_500)
            == (CAP - 2_500) * PER_COLUMN)


def test_prepared_urban_columns_is_none_where_the_count_moves_no_price():
    """Only BEP+BEM's workspace is priced by the count, and a door with no
    land use to read keeps the bound."""
    lu = _land_use((8, 8), 10)
    for option in (0, 1, 2):
        cfg = SimpleNamespace(sf_urban_physics=option, sf_surface_physics=4,
                              use_wudapt_lcz=1)
        assert prepared_urban_columns(cfg, lu, landuse_dataset=MODIS) is None
    cfg = SimpleNamespace(sf_urban_physics=3, sf_surface_physics=4,
                          use_wudapt_lcz=1)
    assert prepared_urban_columns(cfg, None, landuse_dataset=MODIS) is None
    assert prepared_urban_columns(cfg, lu, landuse_dataset=MODIS) == 10


def test_the_configuration_door_says_its_price_is_the_bound(tmp_path):
    """``gpuwm check`` and ``gpuwm go``'s gate price before the land cover
    exists; beside a BEP+BEM figure they say it is the every-column bound
    and that a prepared run prices tighter, so a refusal there is read as
    the bound it is.  No BEP+BEM, no sentence."""
    from gpuwm.go_cli import memory_refusal_text

    exp = _city_tree(tmp_path)
    estimate = pf.estimate_experiment(exp)
    note = pf.bem_workspace_bound_note(estimate)
    assert note is not None
    assert f"{_bem_bytes(estimate) / GIB:.2f} GiB" in note
    assert "every column urban" in note and "tighter" in note
    assert "--prepared-root" in note
    assert pf.bem_workspace_bound_note(pf.estimate_experiment(
        _city_tree(tmp_path, sf_urban_physics=2))) is None
    # go's refusal before the fetch carries it under the verdict.
    gate = {"verdict": "the forecast needs 30.00 GiB", "refuse": True,
            "phases": SimpleNamespace(forecast=estimate)}
    assert f"\n  note: {note}\n" in memory_refusal_text(gate)
    plain = {"verdict": "the forecast needs 30.00 GiB", "refuse": True,
             "phases": SimpleNamespace(forecast=pf.estimate_experiment(
                 _city_tree(tmp_path, sf_urban_physics=2)))}
    assert "note:" not in memory_refusal_text(plain)
