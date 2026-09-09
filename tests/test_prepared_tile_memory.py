"""Current prepared tiles pay unfused radiation and per-stream retained bytes."""
from dataclasses import asdict, replace
from datetime import datetime
import math
from types import SimpleNamespace

import pytest

from gpuwm import domain_wizard as dw
from gpuwm.core import preflight as pf, streaming as st
from gpuwm.core import prepared_tile_memory as ptm
from gpuwm.core.prepared_tile_memory import standalone_rte_storage_bytes
from gpuwm.physics_compat import MORRISON_PROFILE_ID
from tilestream import autoplan as ap


def experiment(nx=628, ny=462, *, mode="auto", root_dx_m=12000.0, **tiles):
    text = dw.render_config(
        name="prepared-tile-memory", start_time=datetime(2026, 9, 8, 18),
        hours=6, projection=dw._projection_entries(21.85418, -73.51348, "lambert"),
        dims=[(nx, ny)], ratios=(), fetch_hints={}, case_data=None,
        root_dx_m=root_dx_m, profile=MORRISON_PROFILE_ID, tiles="auto")
    exp = dw.experiment_from_text(text, source="<prepared-tile-memory>")
    return replace(exp, tiles=st.StreamingOptions(mode=mode, **tiles))


def profile():
    # Committed Aug20 measured context/profile, also used by the bounded
    # Sep08 prepared-store probes; no live GPU query in any test.
    return pf.DeviceLocalMemoryProfile(
        "NVIDIA GeForce RTX 3080", 68, 1536, bare_context_bytes=182452224)


def priced(exp, free_gib=6.41):
    card = profile()
    machine = ap.Machine(int(free_gib * ap.GIB), 64 * ap.GIB,
                         device_profile=card)
    estimate = pf.estimate_experiment(exp, profile=card)
    fp = st.radiation_footprint(exp.root.run, exp.tiles,
                              resident_estimate=estimate, machine=machine)
    return machine, estimate, fp


@pytest.mark.parametrize("shape", [(563, 326), (628, 462)])
@pytest.mark.parametrize("free_gib", [6.41, 6.54])
def test_user_geometries_stream_with_the_same_candidate_and_admission(shape, free_gib):
    exp = experiment(*shape)
    before = asdict(exp.root.run)
    machine, estimate, fp = priced(exp, free_gib)
    assert fp.prepared_memory is not None
    assert estimate.peak_envelope_bytes > machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    decision = st.decide(exp.root.run, exp.tiles, machine=machine,
                         resident_estimate=estimate)
    assert decision.stream
    env = st.streamed_envelope(exp.root.run, exp.tiles, decision=decision,
                              machine=machine, resident_estimate=estimate)
    assert env.peak_vram_bytes == fp.vram_bytes(
        env.window_nx * env.window_ny * exp.root.run.nz, env.nbuffers)
    assert env.peak_vram_bytes == decision.resident_bytes
    assert env.peak_vram_bytes <= decision.budget_bytes
    assert env.peak_vram_bytes <= machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert env.radiation_transient_bytes == 0  # included, never dropped
    assert asdict(exp.root.run) == before
    phases = pf.estimate_phases(exp, source=None, profile=profile(), machine=machine)
    assert phases.forecast_envelope_bytes == env.peak_vram_bytes
    assert "full unfused radiation peak" in decision.detail["plan"]


def test_explicit_tiles_use_the_same_inventory_without_changing_the_config():
    exp = experiment(mode="on", tile_nx=1, tile_ny=1, nbuffers=2)
    machine, estimate, fp = priced(exp)
    env = st.streamed_envelope(exp.root.run, exp.tiles, machine=machine,
                              resident_estimate=estimate)
    assert exp.tiles.resident_context is not None
    assert (env.tile_nx, env.tile_ny, env.nbuffers, env.halo) == (1, 1, 2, 18)
    assert env.peak_vram_bytes == fp.vram_bytes(37 * 37 * 49, 2)
    assert env.peak_vram_bytes < machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES


def test_insufficient_vram_remains_a_real_refusal():
    exp = experiment()
    machine, estimate, _ = priced(exp, 2.0)
    with pytest.raises(ap.CannotPlan):
        st.decide(exp.root.run, exp.tiles, machine=machine, resident_estimate=estimate)


def test_explicit_oversized_tile_is_priced_above_the_budget():
    exp = experiment(mode="on", tile_nx=92, tile_ny=92, nbuffers=2)
    machine, estimate, _ = priced(exp)
    env = st.streamed_envelope(exp.root.run, exp.tiles, machine=machine,
                              resident_estimate=estimate)
    assert env.peak_vram_bytes > machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES


def _with_run(exp, **changes):
    run = replace(exp.root.run, **changes)
    return replace(exp, domains=(replace(exp.root, run=run),))


@pytest.mark.parametrize("lw,sw,expected", [
    (0, 0, False), (1, 1, False), (1, 0, False), (0, 1, False),
    (4, 4, True), (4, 0, True), (0, 4, True), (4, 1, True), (1, 4, True),
])
def test_itemized_rte_storage_requires_an_active_rrtmg_spectrum(lw, sw, expected):
    from gpuwm.config import RunConfig

    cfg = RunConfig(nx=50, ny=50, nz=49, dx=12000., dy=12000.,
                    ztop=18000., dt=30., run_seconds=600., specified=True,
                    ra_lw_physics=lw, ra_sw_physics=sw)
    root = SimpleNamespace(run=cfg)
    exp = SimpleNamespace(root=root, domains=(root,))
    options = SimpleNamespace(enabled=True, store="host",
                              follower_context=None, radiation_context=None)
    assert ptm.supported(exp, cfg, options) is expected


@pytest.mark.parametrize("changes", [dict(mp_physics=8), dict(cu_physics=0),
                                       dict(bl_pbl_physics=5),
                                       dict(sf_sfclay_physics=1, bl_pbl_physics=1),
                                       dict(use_adaptive_time_step=True)])
def test_other_physics_on_the_route_is_priced_by_the_itemized_model(changes):
    """The model prices the ROUTE; a twelve-value fingerprint gated it (ENG-013).

    Any prepared single-root host-store RTE+RRTMGP run pays the unfused
    per-buffer LW+SW storage the module's first paragraph describes.  With
    the fingerprint as the gate, moving one physics selector off the
    Morrison/KF/YSU default fell back to the fused-inventory price, measured
    23 % low on this route -- `check`/`go` said "fits" and the run met the
    unfused allocation at the first radiation call.
    """
    exp = _with_run(experiment(), **changes)
    run = exp.root.run
    assert not ptm.measured_anchor(exp, run, exp.tiles)
    machine, estimate, fp = priced(exp)
    assert fp.prepared_memory is not None
    cells = 37 * 37 * run.nz
    assert fp.vram_bytes(cells, 2) > fp.vram_bytes(cells, 1)
    # At the review's window (980,000 cells, two buffers: 9.304 GiB itemized
    # against 7.580 GiB fused on the default config) the itemized obligation
    # exceeds what the fused fallback quoted for the same window: this is
    # the under-sizing the fingerprint let through.
    window = (980_000 // run.nz) * run.nz
    fused = ap.footprint_for(run)
    assert fp.vram_bytes(window, 2) > fused.vram_bytes(window, 2) + fused.radiation_transient_bytes
    assert fp.radiation_transient_bytes == 0


def test_a_different_column_chunk_is_priced_by_the_itemized_model():
    exp = experiment()
    exp = replace(exp, column_chunk=2048)
    assert not ptm.measured_anchor(exp, exp.root.run, exp.tiles)
    _, _, fp = priced(exp)
    assert fp.prepared_memory is not None
    named = standalone_rte_storage_bytes(49, 4096, 2048, 10000.0)
    assert named["lw/finalized_tau"] == 2048 * 74 * 256 * 4


def test_the_default_configuration_is_the_measured_anchor_and_is_priced_the_same_way():
    exp = experiment()
    assert ptm.measured_anchor(exp, exp.root.run, exp.tiles)
    assert ptm.supported(exp, exp.root.run, exp.tiles)
    _, _, fp = priced(exp)
    assert fp.prepared_memory is not None


def test_legacy_rrtmg_is_the_named_limit_of_the_itemization():
    """The standalone storage is the RTE+RRTMGP solver's own; legacy RRTMG keeps
    the fused price until its prepared factory is itemized (guard cites ENG-013)."""
    exp = _with_run(experiment(), ra_rrtmg_variant="rrtmg_legacy")
    run = exp.root.run
    assert not ptm.supported(exp, run, exp.tiles)
    fp = st.radiation_footprint(run, exp.tiles)
    assert fp.prepared_memory is None
    assert fp.radiation_transient_bytes == ap.RADIATION_TRANSIENT_BYTES[fp.rung]


def test_unfused_named_storage_contains_planck_optics_and_does_not_alias_streams():
    exp = experiment()
    _, _, fp = priced(exp)
    memory = fp.prepared_memory
    named = standalone_rte_storage_bytes(49, 4096, 3125, 10000.0)
    assert named["lw/finalized_tau"] == 3125 * 74 * 256 * 4
    assert named["sw/finalized_tau_ssa_g"] == 3 * 3125 * 50 * 224 * 4
    assert named["lw/planck_current"] == 3125 * 256 * 150 * 4
    assert named["lw/planck_previous"] == named["lw/planck_current"]
    one, two = fp.vram_bytes(4096 * 49, 1), fp.vram_bytes(4096 * 49, 2)
    assert two - one >= math.floor(pf.ALLOCATOR_HEADROOM * memory.buffer_bytes(4096 * 49))
    # Independently retained unfused arrays cannot be replaced by the much
    # smaller, fused explicit-workspace phase maximum.
    fused = sum(math.prod(s) * size for s, size in
                pf.rrtmgp_workspace_shapes(49, 3125, 10000.0).values())
    assert sum(named.values()) > 4 * fused


def test_cell_only_bound_covers_rectangular_window_inventories():
    exp = experiment()
    _, _, fp = priced(exp)
    memory = fp.prepared_memory
    for nx, ny in [(37, 37), (37, 128), (128, 37), (64, 96), (256, 44)]:
        exact = memory._domain(nx, ny)
        upper = memory.buffer_terms(nx * ny * 49)
        assert exact.resident_bytes <= upper["resident_bytes"]
        assert exact.transient_bytes <= upper["step_transient_bytes"]


def test_monotone_cost_and_binary_inversion_keep_the_first_rejected_column():
    exp = experiment()
    _, _, fp = priced(exp)
    previous = 0
    for cols in (1, 1369, 2176, 2177, 3124, 3125, 3126, 4096, 10000):
        cost = fp.vram_bytes(cols * 49, 2)
        assert cost >= previous
        previous = cost
    for buffers in (1, 2):
        budget = int(5.4 * ap.GIB)
        cells = ap._max_window_cells(fp, buffers, budget)
        assert cells > 0 and cells % 49 == 0
        assert fp.vram_bytes(cells, buffers) <= budget
        assert fp.vram_bytes(cells + 49, buffers) > budget


def test_default_loader_template_and_measured_allocator_peak_are_inside_the_bound():
    from gpuwm.ingest.prepared_store import default_slab_rows
    exp = experiment()
    _, _, fp = priced(exp)
    terms = fp.prepared_memory.fixed_terms()
    rows = default_slab_rows(628, 462)
    assert terms["loader_rows"] == rows == 4
    assert terms["template_resident_bytes"] == fp.prepared_memory._domain(628, 462 % rows).resident_bytes
    # Sep08 ordinary64-row loader + two distinct nonblocking compute streams.
    # Pool requests and retained bytes are compared to POOL inventory; KF/YSU
    # workspaces belong there even though the older helper calls them nonpool.
    pool = (2 * fp.buffer_bytes(37 * 37 * 49)
            + terms["template_resident_bytes"] + terms["k_tables_bytes"])
    assert math.ceil(pf.ALLOCATOR_HEADROOM * pool) > 978856448
    assert fp.vram_bytes(128 * 128 * 49, 1) > 2212101120 + terms["cuda_context_bytes"]


def test_wide_requested_grid_uses_the_same_column_bounded_loader_policy():
    from gpuwm.config import DEFAULT_COLUMN_CHUNK
    from gpuwm.ingest.prepared_store import default_slab_rows, _plan_slabs
    for nx, ny in ((12, 130), (563, 326), (628, 462), (2174, 1352), (4096, 17)):
        rows = default_slab_rows(nx, ny)
        assert 1 <= rows <= min(64, ny)
        assert rows * nx <= max(DEFAULT_COLUMN_CHUNK, nx)
        slabs = _plan_slabs(ny, rows)
        assert sum(height for _, height in slabs) == ny
        assert slabs[-1][0] + slabs[-1][1] == ny
    exp = experiment(2174, 1352, root_dx_m=5000.0)
    before = asdict(exp.root.run)
    _, _, fp = priced(exp, 6.47)
    fixed = fp.prepared_memory.fixed_terms()
    assert fixed["loader_rows"] == 1
    old_slab = fp.prepared_memory._domain(2174, 64)
    assert fixed["loader_pool_peak_bytes"] < old_slab.resident_bytes
    # The loader no longer refuses this grid before any compute window can
    # be priced. Host store and source-crop admission remain separate gates.
    assert fp.vram_bytes(37 * 37 * 49, 1) < 5.49 * ap.GIB
    assert asdict(exp.root.run) == before


def test_runtime_uses_its_supplied_device_profile_without_another_probe(monkeypatch):
    exp = experiment()
    machine, estimate, _ = priced(exp)
    monkeypatch.setattr(ap.Machine, "detect", lambda **kw: pytest.fail("unexpected GPU probe"))
    runtime = st.decide(exp.root.run, exp.tiles, machine=machine)
    admitted = st.decide(exp.root.run, exp.tiles, machine=machine, resident_estimate=estimate)
    assert (runtime.tile_nx, runtime.tile_ny, runtime.nbuffers, runtime.resident_bytes) == (
        admitted.tile_nx, admitted.tile_ny, admitted.nbuffers, admitted.resident_bytes)


def test_final_single_root_stepper_decision_does_not_reapply_old_tree_reservations():
    exp = experiment(563, 326)
    machine, estimate, _ = priced(exp)
    direct = st.decide(exp.root.run, exp.tiles, machine=machine, resident_estimate=estimate)
    rows = {}
    tree = st.decide_tree([SimpleNamespace(cfg=exp.root, parent=None)], exp.tiles,
                          machine=machine, resident_estimate=estimate, decisions=rows)
    selected = tree.decided[0][-1]
    assert (selected.tile_nx, selected.tile_ny, selected.nbuffers) == (
        direct.tile_nx, direct.tile_ny, direct.nbuffers)
    assert tree.process_overhead_bytes == 0 and tree.radiation_transient_bytes == 0
    assert tree.vram_spent_bytes == direct.resident_bytes
    assert rows[1] is selected


def test_omitted_estimate_and_explicit_estimate_produce_the_same_envelope():
    exp = experiment(563, 326)
    machine, estimate, _ = priced(exp)
    implicit = st.streamed_envelope(exp.root.run, exp.tiles, machine=machine)
    explicit = st.streamed_envelope(exp.root.run, exp.tiles, machine=machine,
                                   resident_estimate=estimate)
    assert implicit == explicit


def test_retained_forcing_inventory_prices_eager_factory_tables_at_actual_count():
    exp = experiment()
    machine, short, _ = priced(exp)
    wide = pf.estimate_experiment(exp, profile=profile(), forcing_intervals=47)
    a = st.radiation_footprint(exp.root.run, exp.tiles, resident_estimate=short, machine=machine)
    b = st.radiation_footprint(exp.root.run, exp.tiles, resident_estimate=wide, machine=machine)
    assert b.prepared_memory.forcing_intervals == 47
    assert b.prepared_memory.buffer_terms(37 * 37 * 49)["resident_bytes"] > a.prepared_memory.buffer_terms(37 * 37 * 49)["resident_bytes"]
    assert b.vram_bytes(37 * 37 * 49, 2) > a.vram_bytes(37 * 37 * 49, 2)
