"""A finished ordinary member leaves the card before the next member starts.

Every time-lagged and multi-model member takes the ordinary path: one
original forecast after another on a card. The state and the physics driver
of a finished forecast reference each other, so its arrays stay allocated
until a collector pass. Nothing ran one between members: measured on this
150x150x49 case, the default pool held 620 MB more at each later member's
entry, and the ordinary runner's own admission reads those bytes as taken.
Once that was collected, two process-wide device caches still grew by
27,648 bytes a member: the microphysics ring descriptor tables, kept for the
process although they address one state's arrays, and the Noah tables,
keyed by the parameter object each forecast loads anew.

The gate runs the real session, the real waves and the real prepared-tree
runner on the card, and reads the default pool at each member's entry.
"""
from dataclasses import replace
import gc
import os
from pathlib import Path
import re

import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

_MEMBERS = 4
_SEED = 2026100300


@pytest.fixture(scope="module")
def release_real(tmp_path_factory):
    directory = os.environ.get("GPUWM_TEST_WRF_REAL_DIRECTORY")
    if not directory:
        pytest.skip("set GPUWM_TEST_WRF_REAL_DIRECTORY to retained real WRF inputs")
    from gpuwm.wrfinput_door import resolve_wrfinput_run
    from gpuwm.wrfinput_forecast import prepare_wrf_run
    from gpuwm.prepared_domain_tree_forecast import _with_terrain_acoustics
    run = resolve_wrfinput_run(Path(directory))
    text, count = re.subn(r"(?m)^(\s*history_interval_s)\s*=.*$",
                         r"\1 = 12.0", run.toml_text)
    assert count, "the imported configuration must contain history authority"
    inputs = prepare_wrf_run(replace(run, toml_text=text),
                             tmp_path_factory.mktemp("release-ensemble-source"), run_seconds=24)
    return _with_terrain_acoustics(inputs)


def _renderer():
    from gpuwm import rustwx
    renderer = (Path(os.environ["GPUWM_ENSEMBLE_RENDERER"])
                if "GPUWM_ENSEMBLE_RENDERER" in os.environ else rustwx.find_renderer())
    if renderer is None:
        pytest.fail("the production gate requires the built native ensemble renderer")
    return renderer


def _member_initialization(inputs, member):
    """The wrfinput door's own initialization, plus this member's seeded winds.

    Each member is its own forecast, as a time-lagged or multi-model member
    is: the seed gate the engine's identity tests use moves every member's
    initial winds by its own draw.
    """
    from gpuwm.ensemble.batch_perturbation import initialize_member_winds
    from gpuwm.forecast_initialization import DomainInitialization
    from gpuwm.wrfinput_forecast import WrfInitialization

    class SeededInitialization(WrfInitialization):
        def restore_domain(self, domain, grid, bundle, **options):
            initialized = super().restore_domain(domain, grid, bundle, **options)

            def initialize():
                result = initialized.initialize_physics()
                initialize_member_winds(state=initialized.state, cfg=domain.run,
                    member_indices=(member,), seeds=(_SEED + member,),
                    phase="after_physics_before_step")
                return result
            return DomainInitialization(initialized.initial_result, initialize)
    return SeededInitialization(inputs)


def test_default_pool_at_each_member_entry_equals_the_second_member_entry(release_real, tmp_path):
    import cupy as cp
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.runtime_context import current_capture
    pool = cp.get_default_memory_pool()
    entries, exits, provided = [], [], []

    def megabytes(values):
        return [round(value / 1e6, 1) for value in values]

    # The original runner with one reading in front of it and one behind.
    # The signature is the runner's own, so the session hands it the same
    # progress observer and the same handoff options a door's runner gets.
    def runner(inputs, *, output_directory, observer=None, **options):
        member = current_capture().member_id
        cp.cuda.get_current_stream().synchronize()
        entries.append((member, int(pool.used_bytes())))
        try:
            return run_prepared_tree(inputs, output_directory=output_directory, observer=observer,
                                     initialization=_member_initialization(inputs, member), **options)
        finally:
            cp.cuda.get_current_stream().synchronize()
            exits.append((member, int(pool.used_bytes())))

    # Each member is handed its inputs by a provider, as the recipe doors
    # hand a time-lagged or multi-model member its own prepared source.
    def provider(*, shared_inputs, member_id, request):
        provided.append(member_id)
        return shared_inputs

    out = tmp_path / "run"
    # Kept member files make every member an ordinary forecast, which is the
    # path a time-lagged or multi-model member always takes.
    session = PreparedEnsembleSession({"members": _MEMBERS, "keep_member_files": True},
                                      output_directory=out, renderer=_renderer(),
                                      input_provider=provider)
    report = session.run_prepared(runner, release_real, io_mode="history")
    assert report["status"] == "PASS"
    assert report["members_completed"] == list(range(_MEMBERS))
    assert sorted(provided) == list(range(_MEMBERS))
    assert all(batch["execution_mode"] == "ordinary_member" for batch in report["packing"]["batches"])
    assert [member for member, _ in entries] == list(range(_MEMBERS))
    used = [bytes_ for _, bytes_ in entries]
    # What each member still held as its runner returned, before the
    # session released it. Reported beside the entries, not asserted: an
    # interpreter collection of its own may already have taken some of it.
    left = [bytes_ for _, bytes_ in exits]
    # Member 0 starts from whatever the process held. Member 1 starts from
    # that plus the tables the first forecast leaves for every later one.
    # From there on nothing accumulates: member k starts where member 1 did.
    assert used[1] >= used[0], megabytes(used)
    for member in range(2, _MEMBERS):
        assert used[member] == used[1], (
            f"default pool used bytes at each member's entry, in MB: {megabytes(used)} "
            f"(at each member's exit: {megabytes(left)}); member {member} started with "
            f"{round((used[member] - used[1]) / 1e6, 1)} MB of an earlier member's state "
            "still allocated")
    # The first member is released too: once the whole run has ended and
    # been collected, the pool holds what member 1 started from and no less.
    del report
    gc.collect()
    cp.cuda.get_current_stream().synchronize()
    settled = int(pool.used_bytes())
    assert used[1] == settled, (
        f"default pool used bytes at each member's entry, in MB: {megabytes(used)}; after the "
        f"run ended and was collected: {megabytes([settled])[0]}. Member 1 started with "
        f"{round((used[1] - settled) / 1e6, 1)} MB that the finished run no longer holds")
    print(f"member release: entry MB {megabytes(used)}, exit MB {megabytes(left)}, "
          f"settled MB {megabytes([settled])[0]}")


def test_noah_device_tables_are_shared_by_value_and_keep_no_forecast_alive():
    """Each forecast loads its own Noah parameters; equal tables share one upload.

    The device tables were keyed by the parameter object's identity and held
    it, so every member of an ensemble left one more copy of the same tables
    on the card, and its parameter object in memory, for the rest of the run.
    """
    import weakref

    import numpy as np

    from gpuwm.core import noah

    dzs = np.asarray([0.1, 0.3, 0.6, 1.0], np.float32)
    first = noah.pack_params(noah.load_tables())
    tables = noah._device_tables(first, dzs)
    held = len(noah._DEVICE_TABLES)
    loaded = weakref.ref(first)
    del first
    gc.collect()
    assert loaded() is None, "the cache keeps a finished forecast's parameter object alive"
    # A later forecast loads the same tables again: the same upload serves it.
    again = noah.pack_params(noah.load_tables())
    assert all(a is b for a, b in zip(noah._device_tables(again, dzs), tables))
    assert len(noah._DEVICE_TABLES) == held
    # Other values are other tables.
    other = noah.pack_params(noah.load_tables())
    other.veg[0, 0] += 1.0
    assert noah._device_tables(other, dzs)[0] is not tables[0]
    assert noah._device_tables(again, dzs[::-1].copy())[3] is not tables[3]
