"""Resident ranks walk the MYNN width their shared card budget admitted."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace

import pytest

from gpuwm.core import mynn_pbl_scratch as mynn
from gpuwm.core.devices import DeviceOptions
from test_devices_door import exp, rank_api  # noqa: F401


@pytest.fixture(autouse=True)
def isolated_width(monkeypatch):
    monkeypatch.delenv(mynn.MYNN_PBL_COLUMN_CHUNK_ENV, raising=False)
    monkeypatch.setattr(mynn, "_PINNED", None)
    monkeypatch.setattr(mynn, "_RESOLVED", {})
    monkeypatch.setattr(mynn, "_TILE_WALKED", {})
    monkeypatch.setattr(mynn, "probe_mynn_card", lambda: None)


def _split(exp, ids):
    cfg = replace(exp.root.run, nx=800, ny=600, nz=50, moist=True,
                  mp_physics=6, bl_pbl_physics=5, sf_sfclay_physics=5)
    return replace(exp, domains=(replace(exp.root, run=cfg),),
                   devices=DeviceOptions(count=len(ids), ids=ids))


def _required(price):
    return {row["card"]: sum(row[key] for key in (
        "resident_bytes", "seam_bytes", "template_bytes"))
        for row in price["cards"]}


def _fixed_price(monkeypatch, split, width):
    from gpuwm.core.devices_memory import estimate_devices

    with monkeypatch.context() as patch:
        patch.setattr(mynn, "mynn_rank_chunk_candidates", lambda nz: (width,))
        return estimate_devices(split, vram_gib=96)


def test_pricing_width_is_local_to_its_fit_and_never_changes_streamed_policy():
    rendezvous = Barrier(2)

    def fit(width):
        with mynn.mynn_pricing_rank_chunk(width):
            rendezvous.wait()
            return mynn.resolve_mynn_tile_column_chunk(50)

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(fit, 32768)
        two = pool.submit(fit, 98304)
        assert (one.result(), two.result()) == (32768, 98304)
    assert mynn._RESOLVED == {}
    assert mynn.resolve_mynn_tile_column_chunk(50) == 8192
    with pytest.raises(RuntimeError):
        with mynn.mynn_pricing_rank_chunk(32768):
            raise RuntimeError("failed fit")
    assert mynn.resolve_mynn_tile_column_chunk(50) == 8192


@pytest.mark.parametrize("ids", [(0, 1), (0, 0)])
@pytest.mark.parametrize("width", [16384, 24576, 32768])
def test_width_and_snapshot_allowance_share_one_physical_card_budget(
        exp, rank_api, monkeypatch, ids, width):
    from gpuwm.core.devices_memory import estimate_devices, devices_gate

    split = _split(exp, ids)
    priced = _fixed_price(monkeypatch, split, width)
    budgets = {dev: size + 1001 for dev, size in _required(priced).items()}
    fitted = estimate_devices(split, budgets=budgets, vram_gib=96)
    assert not devices_gate(fitted, budgets=budgets)["refuse"]
    assert [rank["mynn_column_chunk"] for rank in fitted["rank_shapes"]] == [width] * 2
    for card in fitted["cards"]:
        assert card["total_bytes"] == budgets[card["card"]]
        assert card["frame_snapshot_bytes"] == 1001
        assert card["resident_bytes"] == sum(
            fitted["rank_shapes"][rank]["resident_bytes"] for rank in card["ranks"])
    # Raising the physical budget to the widest price changes workspace, not
    # the mandatory state or seam allocation. Each repeated-ID rank still
    # has its own whole workspace charged to that card.
    priced96 = _fixed_price(monkeypatch, split, 98304)
    wide = estimate_devices(split, budgets=_required(priced96), vram_gib=96)
    assert [rank["mynn_column_chunk"] for rank in wide["rank_shapes"]] == [98304] * 2
    assert not devices_gate(wide, budgets=_required(priced96))["refuse"]


def test_minimum_fits_without_snapshots_and_a_real_shortfall_still_refuses(
        exp, rank_api, monkeypatch):
    from gpuwm.core.devices_memory import estimate_devices, devices_gate

    split = _split(exp, (0, 0))
    minimum = _fixed_price(monkeypatch, split, 8192)
    budgets = _required(minimum)
    fitted = estimate_devices(split, budgets=budgets, vram_gib=96)
    assert all(rank["mynn_column_chunk"] == 8192 for rank in fitted["rank_shapes"])
    assert all(rank["frame_snapshot_bytes"] == 0 for rank in fitted["rank_shapes"])
    assert not devices_gate(fitted, budgets=budgets)["refuse"]
    budgets[0] -= 1
    refused = estimate_devices(split, budgets=budgets, vram_gib=96)
    assert devices_gate(refused, budgets=budgets)["refuse"]


@pytest.mark.parametrize("ids", [(0, 1), (0, 0)])
def test_the_required_footprint_is_the_minimum_width_not_the_widened_one(
        exp, rank_api, monkeypatch, ids):
    """Open-defects ledger A3c: the admitted price fills each card with
    widened MYNN workspace, so it tracks the card's size, not the
    forecast's need (105 KB per column on 8 x RTX 5090 against 75.5 on
    3 x RTX PRO 6000 for one 1 km grid).  The required price is the
    narrowest width, the one that refuses, and is stated beside it."""
    from gpuwm.core.devices_memory import estimate_devices, devices_gate

    split = _split(exp, ids)
    minimum = estimate_devices(split, vram_gib=96)
    priced96 = _fixed_price(monkeypatch, split, 98304)
    budgets = _required(priced96)
    wide = estimate_devices(split, budgets=budgets, vram_gib=96)
    assert [rank["mynn_column_chunk"] for rank in wide["rank_shapes"]] == [98304] * 2
    footprint = wide["footprint"]
    assert footprint["required_bytes"] == minimum["footprint"]["required_bytes"]
    # Required is state + seams + template at the narrowest width (the
    # loader's envelope on the first card if larger); no frame snapshots.
    narrowest = _required(minimum)
    for row in wide["cards"]:
        assert row["required_bytes"] >= narrowest[row["card"]]
        if row["card"] != ids[0]:
            assert row["required_bytes"] == narrowest[row["card"]]
    assert footprint["mynn_widening_bytes"] == sum(
        row["resident_bytes"] for row in wide["cards"]) - sum(
        row["resident_bytes"] for row in minimum["cards"]) > 0
    assert footprint["required_kb_per_column"] < footprint["admitted_kb_per_column"]
    assert footprint["columns"] == 800 * 600
    verdict = devices_gate(wide, budgets=budgets)["verdict"]
    assert "MYNN workspace widened into free memory" in verdict
    assert f"{footprint['required_kb_per_column']:.1f} KB per column required" in verdict
    # Unwidened, the two figures agree and the line names no widening.
    assert minimum["footprint"]["mynn_widening_bytes"] == 0
    assert "widened into free memory)" not in devices_gate(minimum)["verdict"]


@pytest.mark.parametrize("selection", ["environment", "pin"])
def test_explicit_width_is_not_narrowed_to_make_admission_pass(
        exp, rank_api, monkeypatch, selection):
    from gpuwm.core.devices_memory import estimate_devices, devices_gate

    split = _split(exp, (0, 1))
    budgets = _required(_fixed_price(monkeypatch, split, 8192))
    if selection == "environment":
        monkeypatch.setenv(mynn.MYNN_PBL_COLUMN_CHUNK_ENV, "24576")
    else:
        monkeypatch.setattr(mynn, "_PINNED", 24576)
    assert mynn.mynn_rank_chunk_candidates(50) == (24576,)
    priced = estimate_devices(split, budgets=budgets, vram_gib=96)
    assert all(rank["mynn_column_chunk"] == 24576 for rank in priced["rank_shapes"])
    assert devices_gate(priced, budgets=budgets)["refuse"]


def test_rank_binding_is_immutable_and_refuses_existing_wrong_scratch():
    from gpuwm.io.restart import classify_state_attr

    cfg = SimpleNamespace(nx=800, ny=600, nz=50, bl_mynn_version="wrf_461")
    state = SimpleNamespace(_scratch={})
    assert mynn.bind_mynn_rank_chunk(state, cfg, 32768) == 32768
    assert mynn.bind_mynn_rank_chunk(state, cfg, 32768) == 32768
    with pytest.raises(ValueError, match="invalidate scratch"):
        mynn.bind_mynn_rank_chunk(state, cfg, 98304)
    old = next(iter(mynn.mynn_pbl_scratch_shapes(8192, cfg.nz).items()))
    state = SimpleNamespace(_scratch={old[0]: SimpleNamespace(shape=old[1])})
    with pytest.raises(ValueError, match="already has shape"):
        mynn.bind_mynn_rank_chunk(state, cfg, 32768)
    assert not hasattr(state, "_mynn_rank_column_chunk")
    assert classify_state_attr("_mynn_rank_column_chunk") == "infra"


def test_binding_prices_and_walks_only_columns_present_in_a_small_rank():
    cfg = SimpleNamespace(nx=32, ny=16, nz=50, bl_mynn_version="wrf_461")
    state = SimpleNamespace(_scratch={})
    assert mynn.bind_mynn_rank_chunk(state, cfg, 32768) == 512
    assert state._mynn_rank_column_chunk == 512


@pytest.mark.parametrize("rank_width,tile,expected", [
    (32768, True, 32768), (None, True, 8192), (None, False, None)])
def test_driver_uses_rank_width_without_changing_other_state_policies(
        exp, monkeypatch, rank_width, tile, expected):
    from gpuwm.core import physics

    cfg = _split(exp, (0, 1)).root.run
    driver = physics.PhysicsDriver.__new__(physics.PhysicsDriver)
    driver.state = SimpleNamespace(elapsed_seconds=0.0, w=object(), _tile_buffer=tile)
    if rank_width is not None:
        mynn.bind_mynn_rank_chunk(driver.state, cfg, rank_width)
    driver.fields = {}
    driver.bldt_seconds = cfg.dt

    class ReachedSolver(Exception):
        pass

    def step(*args, **kwargs):
        assert kwargs["column_chunk"] == expected
        raise ReachedSolver

    monkeypatch.setattr(physics, "mynn_pbl_step", step)
    with pytest.raises(ReachedSolver):
        driver._run_mynn_pbl({}, cfg)


def test_admitted_widths_reach_the_rank_builder(exp, monkeypatch):
    from gpuwm import prepared_single_domain_forecast as forecast
    from gpuwm.core import streaming

    split = _split(exp, (0, 1))
    observed = []
    monkeypatch.setattr(streaming, "ranked_domain_builder", lambda bundle, **kw:
                        observed.append(kw) or object())
    monkeypatch.setattr(streaming, "ranked_decision", lambda *a, **kw:
                        SimpleNamespace(halo=1))
    monkeypatch.setattr(streaming, "make_stepper", lambda *a, **kw:
                        SimpleNamespace(tiled_run=SimpleNamespace(
                            transport_report={"path": "host"})))
    node = SimpleNamespace(cfg=split.root, state=SimpleNamespace(), clock=object())
    forecast._devices_stepper(SimpleNamespace(geography={}), node, split, {},
                             admission={"rank_shapes": [
                                 {"frame_snapshot_bytes": 100, "mynn_column_chunk": 32768},
                                 {"frame_snapshot_bytes": 200, "mynn_column_chunk": 98304}]})
    assert observed[0]["mynn_column_chunks"] == (32768, 98304)
    assert observed[0]["snapshot_limits"] == (100, 200)
