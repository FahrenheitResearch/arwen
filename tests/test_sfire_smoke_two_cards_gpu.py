"""Coupled SFIRE with bulk smoke on two physical cards.

The breakage this prevents (MULTICARD-CHECK 2026-10-08, reported on 2.8.7
and the 2.8.8 tip, still in 2.8.9 cd336e62d6): a ranked fire-smoke run on
two physical cards built rank 1's chem history diagnostics (``dz8w`` and the
column sums) while card 0 was current.  On cards with no peer path the
build failed with "Peer access is unavailable between these devices"; on
cards with a peer path the kernels ran on card 0 reading card 1 across the
bus.  The existing fire GPU tests put every rank on card 0, so neither
showed.

These tests place the two ranks on cards 0 and 1, so they need a host with
two cards and are skipped, by name, on a one-card host.  They fail on the
old code on ANY two-card host: the device check below sees card 0 current
while rank 1's diagnostics run, whether or not the pair has a peer path.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


def _two_cards():
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("fire smoke on two physical cards needs two CUDA devices; this host has one")


def _ranked(state, cfg, ids):
    from gpuwm.core import streaming as st
    from gpuwm.core.devices import DeviceOptions
    from tilestream import driver, gather, physics_inventory
    take = st.streamed_store_inventory()
    bundle = SimpleNamespace(store={name: gather.pinned_copy(value)
        for name, value in take(state, None).items()},
        scalars=physics_inventory.carrier_scalars(state),
        geography=driver.geography_store(state, host=True),
        boundaries=None, template=state)
    options = DeviceOptions(count=len(ids), grid=(1, len(ids)), ids=tuple(ids))
    return st.ranked_domain_builder(bundle, options=options)(
        None, cfg, st.ranked_decision(cfg, options))


def _initial(cp, smoke=True):
    from test_restart import _physics_state
    from gpuwm.core.sfire_coupler import FireCoupler
    from gpuwm.core.streaming import prime_lazy_carriers
    state, cfg, driver = _physics_state(cp, nx=96, ny=72, nz=16,
        dx=90., dy=120., dt=.25, mp_physics=1, cu_physics=0, ra_physics=0,
        ifire=2, sr_x=4, sr_y=3, fire_smoke=smoke, fire_fuel_read=0,
        fire_fuel_cat=1, fire_fmc_read=0, fire_boundary_guard=2,
        fmoist_run=True, fmoist_interp=True, fmoist_freq=1,
        fire_num_ignitions=1, fire_ignition_start_x1=2880.,
        fire_ignition_start_y1=2880., fire_ignition_radius1=160.,
        fire_ignition_ros1=280.)
    prime_lazy_carriers(state, cfg)
    driver.fire = FireCoupler(state, cfg, driver.fields, {
        "LFN_HIST": cp.full(driver.fire.fine_shape, -17.5, cp.float32),
        "LFN_TIME": cp.asarray([-125.5], cp.float32)})
    return state, cfg


@pytest.fixture
def card_spy(monkeypatch):
    """Record, at every chem layer-thickness evaluation, which card is
    current and which card holds the state it reads."""
    import cupy as cp
    from gpuwm.core import chem_history
    seen = []
    original = chem_history.dz8w

    def spy(state):
        seen.append((int(cp.cuda.Device().id), int(state.php.device.id)))
        return original(state)

    monkeypatch.setattr(chem_history, "dz8w", spy)
    return seen


@requires_gpu
@pytest.mark.gpu
def test_fire_smoke_builds_with_each_ranks_diagnostics_on_its_own_card(card_spy):
    import cupy as cp
    _two_cards()
    home, cfg = _initial(cp)
    stream = _ranked(home, cfg, (0, 1))
    try:
        run = stream.tiled_run
        assert "state/chemdiag_sfire_source_lasttime" in run.store_keys()
        assert {held for _current, held in card_spy} == {0, 1}, card_spy
        mismatched = [pair for pair in card_spy if pair[0] != pair[1]]
        assert not mismatched, (
            "chem history ran with a card current that does not hold its state "
            f"(current, holder): {mismatched}")
        for rank, arrays in enumerate(run.arrays):
            cards = {int(a.device.id) for a in arrays.values()}
            assert cards == {run.devices[rank]}, (rank, cards)
    finally:
        stream.tiled_run.close()


@requires_gpu
@pytest.mark.gpu
def test_fire_smoke_on_two_cards_matches_one_domain_word_for_word(record_property):
    """Eight coupled steps with bulk smoke: the two-card run's whole store
    equals the one-card single-domain run's to the word, and the smoke ledger
    closes."""
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.core.streaming import streamed_store_inventory
    from test_sfire_tilestream_gpu import _assert_words
    _two_cards()
    reference, cfg = _initial(cp)
    home, ranked_cfg = _initial(cp)
    stream = _ranked(home, ranked_cfg, (0, 1))
    words = 0
    try:
        for number in range(8):
            due = (number + 1) % 4 == 0
            step(reference, cfg, fire_history_due=due)
            stream(None, ranked_cfg, fire_history_due=due)
            arrays = streamed_store_inventory()(reference, None)
            try:
                _assert_words(cp, arrays, stream.store)
            except AssertionError as exc:
                differences = {name: int(np.count_nonzero(
                    cp.asnumpy(value).view(np.uint32)
                    != np.asarray(stream.store[name]).view(np.uint32)))
                    for name, value in arrays.items()
                    if cp.asnumpy(value).tobytes() != np.asarray(stream.store[name]).tobytes()}
                raise AssertionError(("completed_step", number + 1, differences)) from exc
            words += sum(value.nbytes // 4 for value in arrays.values())
        from gpuwm.core.chem_driver import chem_ledger_receipt
        ledger = chem_ledger_receipt(reference)["rows"]["fire_smoke"]
        assert ledger["emitted_kg"] > 0
        assert abs(ledger["closure_kg"]) < 1e-8
        record_property("graded_32bit_words", words)
        record_property("completed_steps", 8)
        record_property("different_words", 0)
        record_property("emitted_kg", ledger["emitted_kg"])
    finally:
        stream.tiled_run.close()
