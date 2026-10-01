"""A checkpoint inside a forcing interval can retain the full old prefix."""
import numpy as np
import pytest

from gpuwm.io import restart
from test_restart import _cfg, _extension_state, _external_mirror


def _live(cfg, monkeypatch, **kwargs):
    state = _extension_state(cfg, monkeypatch, **kwargs)
    state._lateral_boundary_device = _external_mirror(bound=True)
    return state


@pytest.mark.parametrize('count', [2, 3])
def test_preserved_prefix_restores_full_state_before_or_after_append(tmp_path, monkeypatch, count):
    cfg = _cfg(moist=True, mp_physics=1, specified=True, spec_bdy_width=3, spec_zone=1, relax_zone=2)
    original = _live(cfg, monkeypatch, count=2)
    original.elapsed_seconds = 60.
    original.thp.fill(17.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg, preserved_forcing_prefix=True)
    live = _live(cfg, monkeypatch, count=count)
    live.thp.fill(-2.)
    info = restart.restore_restart(path, live, cfg, preserved_forcing_prefix=True)
    assert info.elapsed_seconds == 60. and live.elapsed_seconds == 60.
    np.testing.assert_array_equal(live.thp, original.thp)


@pytest.mark.parametrize('mutation', ['future-in-old-prefix', 'base', 'shorter', 'clock', 'ordinary'])
def test_preserved_prefix_rejects_reinterpretation_before_state_mutation(tmp_path, monkeypatch, mutation):
    cfg = _cfg(moist=True, mp_physics=1, specified=True, spec_bdy_width=3, spec_zone=1, relax_zone=2)
    original = _live(cfg, monkeypatch, count=3)
    original.elapsed_seconds = 60.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg,
                                 preserved_forcing_prefix=mutation != 'ordinary')
    live = _live(cfg, monkeypatch, count=2 if mutation == 'shorter' else 4,
                 seed=12 if mutation == 'future-in-old-prefix' else 11)
    if mutation == 'base':
        live.thb += 1
    if mutation == 'clock':
        # A real writer must reject a clock beyond the available forcing.
        original.elapsed_seconds = 10000.
        with pytest.raises(restart.RestartMismatchError, match='ends before'):
            restart.write_restart(tmp_path / 'bad-clock.npz', original, cfg, preserved_forcing_prefix=True)
        return
    before = live.thp.copy()
    with pytest.raises(restart.RestartMismatchError):
        restart.restore_restart(path, live, cfg, preserved_forcing_prefix=True)
    np.testing.assert_array_equal(live.thp, before)


# ---------------------------------------------------------------------------
# Chained preparation: a checkpoint written before the preparation seals
# ---------------------------------------------------------------------------

from collections.abc import Sequence
from dataclasses import replace


class _Streaming(Sequence):
    """A streamed boundary series with ``ready`` intervals prepared.

    Asking for an interval that is not prepared yet fails loudly here,
    which is how these tests prove a pre-seal checkpoint never waits.
    """

    def __init__(self, intervals, *, ready, head="a" * 64, sealed=False):
        self._intervals = tuple(intervals)
        self.ready = ready
        self.head_sha256 = head
        self._sealed = sealed
        self.bounds = tuple((i.start_seconds, i.end_seconds)
                            for i in self._intervals)

    def ready_prefix(self):
        return len(self._intervals) if self._sealed else self.ready

    def sealed(self):
        return self._sealed

    def __len__(self):
        return len(self._intervals)

    def __getitem__(self, index):
        if index >= len(self._intervals):
            raise IndexError(index)
        if index >= self.ready_prefix():
            raise AssertionError(f"waited on unprepared interval {index}")
        return self._intervals[index]


def _streamed(state, **kwargs):
    boundaries = state.lateral_boundaries
    state.lateral_boundaries = replace(
        boundaries, intervals=_Streaming(boundaries.intervals, **kwargs))
    return state


def _specified():
    return _cfg(moist=True, mp_physics=1, specified=True, spec_bdy_width=3,
                spec_zone=1, relax_zone=2)


def test_a_checkpoint_before_the_seal_waits_for_nothing(tmp_path,
                                                        monkeypatch):
    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=4), ready=1)
    original.elapsed_seconds = 60.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    header, _ = restart._load_restart(path, with_arrays=False)
    assert header['forcing_extension_mode'] == \
        restart.PRESERVED_FORCING_PREFIX_MODE
    assert header['boundary_stream'] == {'head_sha256': 'a' * 64}
    assert len(header['lateral_boundary_prefix']['intervals']) == 1


def test_a_pre_seal_checkpoint_resumes_on_the_sealed_series(tmp_path,
                                                             monkeypatch):
    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=4), ready=2)
    original.elapsed_seconds = 60.
    original.thp.fill(23.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    live = _streamed(_live(cfg, monkeypatch, count=4), ready=4, sealed=True)
    live.thp.fill(-1.)
    info = restart.restore_restart(path, live, cfg)
    assert info.elapsed_seconds == 60.
    np.testing.assert_array_equal(live.thp, original.thp)


@pytest.mark.parametrize('ready', [2, 3])
def test_a_pre_seal_checkpoint_resumes_on_the_series_still_streaming(
        tmp_path, monkeypatch, ready):
    """A forecast killed before the seal resumes beside the same
    preparation: only the recorded intervals are compared, so the resume
    never waits for an interval the preparation has not built yet."""

    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=4), ready=2)
    original.elapsed_seconds = 60.
    original.thp.fill(29.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    live = _streamed(_live(cfg, monkeypatch, count=4), ready=ready)
    live.thp.fill(-1.)
    info = restart.restore_restart(path, live, cfg)
    assert info.elapsed_seconds == 60.
    np.testing.assert_array_equal(live.thp, original.thp)


def test_a_pre_seal_checkpoint_refuses_changed_bytes_while_streaming(
        tmp_path, monkeypatch):
    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=4), ready=2)
    original.elapsed_seconds = 60.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    live = _streamed(_live(cfg, monkeypatch, count=4, seed=12), ready=2)
    before = live.thp.copy()
    with pytest.raises(restart.RestartMismatchError):
        restart.restore_restart(path, live, cfg)
    np.testing.assert_array_equal(live.thp, before)


def test_a_pre_seal_checkpoint_refuses_another_head(tmp_path, monkeypatch):
    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=3), ready=1)
    original.elapsed_seconds = 60.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    live = _streamed(_live(cfg, monkeypatch, count=3), ready=3, sealed=True,
                     head='b' * 64)
    before = live.thp.copy()
    with pytest.raises(restart.RestartMismatchError, match='prepared head'):
        restart.restore_restart(path, live, cfg)
    np.testing.assert_array_equal(live.thp, before)


@pytest.mark.parametrize('seed', [11, 12])
def test_a_pre_seal_checkpoint_resumes_under_the_sealed_binding_only_on_the_same_bytes(
        tmp_path, monkeypatch, seed):
    """Resumed on the sealed tree bound by its proof (an eager series), the
    recorded intervals must still be byte-identical: a different
    preparation (another seed) is refused before any state is touched."""

    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=3), ready=2)
    original.elapsed_seconds = 60.
    original.thp.fill(5.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    live = _live(cfg, monkeypatch, count=3, seed=seed)
    live.thp.fill(-3.)
    if seed == 11:
        restart.restore_restart(path, live, cfg)
        np.testing.assert_array_equal(live.thp, original.thp)
    else:
        before = live.thp.copy()
        with pytest.raises(restart.RestartMismatchError):
            restart.restore_restart(path, live, cfg)
        np.testing.assert_array_equal(live.thp, before)


def test_a_checkpoint_after_the_seal_is_the_ordinary_checkpoint(
        tmp_path, monkeypatch):
    cfg = _specified()
    eager = _live(cfg, monkeypatch, count=3)
    eager.elapsed_seconds = 60.
    streamed = _streamed(_live(cfg, monkeypatch, count=3), ready=3,
                         sealed=True)
    streamed.elapsed_seconds = 60.
    one = restart.write_restart(tmp_path / 'eager.npz', eager, cfg)
    two = restart.write_restart(tmp_path / 'streamed.npz', streamed, cfg)
    first, a = restart._load_restart(one, with_arrays=True)
    second, b = restart._load_restart(two, with_arrays=True)
    for key in ('setup_fingerprint', 'config', 'array_manifest'):
        assert first[key] == second[key]
    assert 'boundary_stream' not in second
    assert 'forcing_extension_mode' not in second
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


# ---------------------------------------------------------------------------
# A boundary value that clears out between two forcing times (A140, A140b)
# ---------------------------------------------------------------------------

from gpuwm.io.restart import lateral_boundary_prefix_identity
from gpuwm.state_serialization_contract import (
    BUILT_END_FRAME_PREFIX_SCHEMA, LATERAL_BOUNDARY_PREFIX_SCHEMA)


def _clear_out_state(cfg, monkeypatch, **kwargs):
    """A root whose ``qc`` boundary falls to exactly zero at forcing time 1."""

    from test_restart import _fill_setup, _shim_state

    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)
    state.lateral_boundaries = _clear_out_boundaries(**kwargs)
    state._lateral_boundary_device = _external_mirror(bound=True)
    return state


def _clear_out_boundaries(*, count=4, jump=0.0, cleared=(1,), seed=3,
                          unbuilt=False):
    """Hourly ``qc`` intervals whose frames at ``cleared`` times are all zero.

    The frames are FP32 values, as a prepared state's are, and the
    intervals come from the same builder a chained preparation uses, which
    records the frame each tendency was built toward.  Rebuilt as
    ``value + tendency * 3600`` instead, the end frame of an interval that
    clears out leaves residuals of about 1e-20 where the next start is 0.0.
    ``jump`` moves interval 1's start away from the frame interval 0 was
    built toward, the splice the continuity refusal exists to catch.
    ``unbuilt`` drops every recorded end frame, as a series read from a
    prepared cache written before 2.8.1 carries none.  The frames drawn do
    not depend on ``count``, so a longer series renews a shorter one.
    """

    from gpuwm.ingest.lateral_bc import (
        LateralBoundaries, build_lateral_interval_from_sides,
        extract_lateral_side)

    rng = np.random.default_rng(seed)
    snapshots = []
    for index in range(count):
        values = rng.uniform(1e-5, 1e-3, (1, 8, 8)).astype(np.float32)
        if index in cleared:
            values[...] = 0.0
        snapshots.append({"qc": values.astype(np.float64)})
    frames = [{side: extract_lateral_side(snapshot, side, 3)
               for side in ("west", "east", "south", "north")}
              for snapshot in snapshots]
    intervals = [build_lateral_interval_from_sides(
        frames[k], frames[k + 1], start_seconds=3600.0 * k,
        end_seconds=3600.0 * (k + 1)) for k in range(count - 1)]
    if jump:
        moved = {side: {"qc": np.asarray(frames[1][side]["qc"]) + jump}
                 for side in frames[1]}
        intervals[1] = build_lateral_interval_from_sides(
            moved, frames[2], start_seconds=3600.0, end_seconds=7200.0)
    if unbuilt:
        intervals = [replace(interval) for interval in intervals]
        assert all(interval.end_frame_sha256 is None
                   for interval in intervals)
    return LateralBoundaries(tuple(intervals), 3, 1, 2)


def test_the_clear_out_fixture_shares_its_built_frame_and_breaks_the_rebuilt_one(
        monkeypatch):
    """The precondition of the tests below, and the defect itself: the
    frame intervals 0 and 1 share has one digest as its builder recorded it
    and two as the rebuilt identity reconstructs it, with no content
    between them."""

    state = _clear_out_state(_specified(), monkeypatch)
    built = lateral_boundary_prefix_identity(state)
    rebuilt = lateral_boundary_prefix_identity(state,
                                               rebuilt_end_frames=True)
    assert built['schema'] == BUILT_END_FRAME_PREFIX_SCHEMA
    assert rebuilt['schema'] == LATERAL_BOUNDARY_PREFIX_SCHEMA
    rows, old = built['intervals'], rebuilt['intervals']
    assert rows[0]['end_frame_sha256'] == rows[1]['start_frame_sha256']
    assert old[0]['end_frame_sha256'] != old[1]['start_frame_sha256']
    for new, before in zip(rows, old):
        # The row bytes and start frames are the rebuilt identity's.
        assert {key: value for key, value in new.items()
                if key != 'end_frame_sha256'} == {
            key: value for key, value in before.items()
            if key != 'end_frame_sha256'}
    first, second = state.lateral_boundaries.intervals[:2]
    for side in ('west', 'east', 'south', 'north'):
        before = getattr(first.fields['qc'], side)
        after = getattr(second.fields['qc'], side)
        # Interval 1 starts from exactly the frame interval 0 was built
        # toward: its tendency is recomputed from interval 1's start bytes.
        np.testing.assert_array_equal(
            (np.asarray(after.value) - np.asarray(before.value)) / 3600.0,
            np.asarray(before.tendency))


@pytest.mark.parametrize('ready', [2, 3])
def test_a_pre_seal_checkpoint_over_a_cleared_out_boundary_is_written_and_resumes(
        tmp_path, monkeypatch, ready):
    """A140: the chained forecast's first checkpoint before the seal was
    refused as a discontinuous preserved forcing frame at interval 1.
    A140b: it is written with the continuity check applied."""

    cfg = _specified()
    original = _streamed(_clear_out_state(cfg, monkeypatch), ready=ready)
    original.elapsed_seconds = 3600.
    original.thp.fill(31.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    header, _ = restart._load_restart(path, with_arrays=False)
    assert header['boundary_stream'] == {'head_sha256': 'a' * 64}
    assert header['lateral_boundary_prefix']['schema'] == \
        BUILT_END_FRAME_PREFIX_SCHEMA
    assert len(header['lateral_boundary_prefix']['intervals']) == ready

    streaming = _streamed(_clear_out_state(cfg, monkeypatch), ready=ready)
    streaming.thp.fill(-1.)
    restart.restore_restart(path, streaming, cfg)
    np.testing.assert_array_equal(streaming.thp, original.thp)

    sealed = _streamed(_clear_out_state(cfg, monkeypatch), ready=3,
                       sealed=True)
    sealed.thp.fill(-2.)
    info = restart.restore_restart(path, sealed, cfg)
    assert info.elapsed_seconds == 3600.
    np.testing.assert_array_equal(sealed.thp, original.thp)

    eager = _clear_out_state(cfg, monkeypatch)
    eager.thp.fill(-3.)
    restart.restore_restart(path, eager, cfg)
    np.testing.assert_array_equal(eager.thp, original.thp)


@pytest.mark.parametrize('live, reason', [
    # The frame continuity refusal applies to a head-bound checkpoint
    # again (A140b): interval 1 does not start where interval 0 was built
    # to go.
    ({'jump': 1e-4}, 'discontinuous preserved forcing frame at interval 1'),
    # Another preparation, continuous in itself, whose bytes differ.
    ({'seed': 4}, 'changed a previously declared interval'),
])
def test_a_pre_seal_checkpoint_still_refuses_changed_bytes_after_a_clear_out(
        tmp_path, monkeypatch, live, reason):
    cfg = _specified()
    original = _streamed(_clear_out_state(cfg, monkeypatch), ready=2)
    original.elapsed_seconds = 3600.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    state = _streamed(_clear_out_state(cfg, monkeypatch, **live), ready=3,
                      sealed=True)
    before = state.thp.copy()
    with pytest.raises(restart.RestartMismatchError, match=reason):
        restart.restore_restart(path, state, cfg)
    np.testing.assert_array_equal(state.thp, before)


@pytest.mark.parametrize('head_bound', [False, True])
def test_a_preserved_checkpoint_refuses_a_spliced_frame(
        tmp_path, monkeypatch, head_bound):
    """A series whose interval 1 does not start where interval 0 was built
    to go is refused when the checkpoint is written, bound to a prepared
    head or not: A140 had switched the check off for a head-bound one."""

    cfg = _specified()
    spliced = _clear_out_state(cfg, monkeypatch, jump=1.0)
    spliced.elapsed_seconds = 3600.
    if head_bound:
        _streamed(spliced, ready=3)
    with pytest.raises(restart.RestartMismatchError,
                       match='discontinuous preserved forcing frame at '
                             'interval 1'):
        restart.write_restart(tmp_path / 'state.npz', spliced, cfg,
                              preserved_forcing_prefix=not head_bound)
    assert not (tmp_path / 'state.npz').exists()


def test_a_preserved_checkpoint_bound_to_no_head_over_a_clear_out_is_written_and_resumes(
        tmp_path, monkeypatch):
    """A continuous case's checkpoints carry the preserved-prefix contract
    with no prepared head; over a hydrometeor that clears out, the rebuilt
    identity refused them as a discontinuous frame when written."""

    cfg = _specified()
    original = _clear_out_state(cfg, monkeypatch)
    original.elapsed_seconds = 5400.
    original.thp.fill(13.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg,
                                 preserved_forcing_prefix=True)
    header, _ = restart._load_restart(path, with_arrays=False)
    assert 'boundary_stream' not in header
    assert header['lateral_boundary_prefix']['schema'] == \
        BUILT_END_FRAME_PREFIX_SCHEMA
    live = _clear_out_state(cfg, monkeypatch)
    live.thp.fill(-1.)
    restart.restore_restart(path, live, cfg, preserved_forcing_prefix=True)
    np.testing.assert_array_equal(live.thp, original.thp)


@pytest.mark.parametrize('jump', [0.0, 1e-4])
def test_a_preserved_prefix_renewal_joins_a_cleared_out_frame(
        tmp_path, monkeypatch, jump):
    """The renewal appends past the stored prefix, whose last interval
    ends on the frame that clears out.  Its end frame, as built, is the
    renewal's next start frame, so the renewal is admitted; a renewal whose
    next interval starts elsewhere is refused (A140b)."""

    cfg = _specified()
    original = _clear_out_state(cfg, monkeypatch, count=2)
    original.elapsed_seconds = 1800.
    original.thp.fill(7.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg,
                                 preserved_forcing_prefix=True)
    renewed = _clear_out_state(cfg, monkeypatch, count=4, jump=jump)
    renewed.thp.fill(-1.)
    before = renewed.thp.copy()
    if jump:
        with pytest.raises(restart.RestartMismatchError,
                           match='discontinuous preserved forcing frame at '
                                 'interval 1'):
            restart.restore_restart(path, renewed, cfg,
                                    preserved_forcing_prefix=True)
        np.testing.assert_array_equal(renewed.thp, before)
        return
    info = restart.restore_restart(path, renewed, cfg,
                                   preserved_forcing_prefix=True)
    assert info.elapsed_seconds == 1800.
    np.testing.assert_array_equal(renewed.thp, original.thp)
    # The rebuilt identity reads the same join as a jump.
    rows = lateral_boundary_prefix_identity(
        renewed, rebuilt_end_frames=True)['intervals']
    assert rows[0]['end_frame_sha256'] != rows[1]['start_frame_sha256']


@pytest.mark.parametrize('jump', [0.0, 1e-4])
def test_a_sealed_extension_joins_a_cleared_out_frame(
        tmp_path, monkeypatch, jump):
    """A checkpoint sealed at the end of its forcing, where a hydrometeor
    has just cleared out, resumes on the extended forcing (A140b); an
    extension whose first interval starts from another frame is refused."""

    cfg = _specified()
    source = _clear_out_state(cfg, monkeypatch, count=2)
    source.elapsed_seconds = 3600.
    path = restart.write_restart(tmp_path / 'sealed.npz', source, cfg,
                                 sealed_forcing_extension=True)
    extended = _clear_out_state(cfg, monkeypatch, count=3, jump=jump)
    if jump:
        with pytest.raises(restart.RestartMismatchError,
                           match='discontinuous live forcing frame at '
                                 'interval 1'):
            restart._validate_restart(path, extended, cfg,
                                      sealed_forcing_extension=True)
        return
    restart._validate_restart(path, extended, cfg,
                              sealed_forcing_extension=True)


def test_a_checkpoint_in_the_rebuilt_identity_still_reads(
        tmp_path, monkeypatch):
    """A checkpoint written before 2.8.1 records its forcing in the rebuilt
    identity.  The live series, whose builder recorded its end frames, is
    hashed the same way for the comparison, so the checkpoint resumes."""

    cfg = _specified()
    original = _clear_out_state(cfg, monkeypatch, cleared=(), unbuilt=True)
    original.elapsed_seconds = 5400.
    original.thp.fill(19.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg,
                                 preserved_forcing_prefix=True)
    header, _ = restart._load_restart(path, with_arrays=False)
    assert header['lateral_boundary_prefix']['schema'] == \
        LATERAL_BOUNDARY_PREFIX_SCHEMA
    live = _clear_out_state(cfg, monkeypatch, cleared=())
    assert lateral_boundary_prefix_identity(live)['schema'] == \
        BUILT_END_FRAME_PREFIX_SCHEMA
    live.thp.fill(-1.)
    restart.restore_restart(path, live, cfg, preserved_forcing_prefix=True)
    np.testing.assert_array_equal(live.thp, original.thp)


def test_a_rebuilt_identity_checkpoint_over_a_clear_out_is_refused_by_name(
        tmp_path, monkeypatch):
    """The pre-seal checkpoints A140's build wrote record a clear-out in
    the rebuilt identity, which reads as a discontinuity.  The continuity
    check applies to them again, and the refusal names the identity."""

    from test_restart import _rewrite_header

    cfg = _specified()
    original = _streamed(_clear_out_state(cfg, monkeypatch), ready=2)
    original.elapsed_seconds = 3600.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    rebuilt = lateral_boundary_prefix_identity(
        restart._ReadyPrefixState(
            original, original.lateral_boundaries.intervals),
        rebuilt_end_frames=True)
    legacy = _rewrite_header(
        path, tmp_path / 'legacy.npz',
        lambda header: header.__setitem__('lateral_boundary_prefix',
                                          rebuilt))
    live = _streamed(_clear_out_state(cfg, monkeypatch), ready=3,
                     sealed=True)
    before = live.thp.copy()
    with pytest.raises(restart.RestartMismatchError,
                       match='rebuilt end-frame identity'):
        restart.restore_restart(legacy, live, cfg)
    np.testing.assert_array_equal(live.thp, before)


def test_a_built_identity_checkpoint_on_forcing_that_records_no_end_frames_is_refused_by_name(
        tmp_path, monkeypatch):
    cfg = _specified()
    original = _clear_out_state(cfg, monkeypatch)
    original.elapsed_seconds = 5400.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg,
                                 preserved_forcing_prefix=True)
    live = _clear_out_state(cfg, monkeypatch, unbuilt=True)
    before = live.thp.copy()
    with pytest.raises(restart.RestartMismatchError,
                       match='records none'):
        restart.restore_restart(path, live, cfg,
                                preserved_forcing_prefix=True)
    np.testing.assert_array_equal(live.thp, before)


def _clear_out_snapshots(count=4, seed=7, shape=(3, 12, 14)):
    """FP32 frames, as a prepared state's are, whose ``qc`` is zero at 1 h."""

    rng = np.random.default_rng(seed)
    snapshots = []
    for index in range(count):
        qc = rng.uniform(1e-5, 1e-3, shape).astype(np.float32)
        if index == 1:
            qc[...] = 0.0
        snapshots.append({
            'u': rng.standard_normal(shape).astype(np.float32),
            'qc': qc})
    return snapshots


def _chained_preparation(tmp_path, snapshots):
    """A chained preparation driven one forcing time at a time.

    The same writer, accumulator and marker files every chained route
    uses (:class:`gpuwm.ingest.boundary_stream.PreparedTreeWriter`), so the
    forecast side reads its intervals back from real segment files.
    """

    from gpuwm.ingest.boundary_stream import PreparedTreeWriter
    from gpuwm.ingest.lateral_bc import StateBoundaryFrames
    from test_boundary_stream import PROOF_HEAD, _initial, _met, _times

    times = _times(len(snapshots))
    staging = tmp_path / '.tmp-run'
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / 'run',
        identity={'source': 'checkpoint-before-seal-test'}, chained=True)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    seconds = [(t - times[0]).total_seconds() for t in times]
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            'spec_bdy_width': 5, 'spec_zone': 1, 'relax_zone': 4,
            'schedule': [[seconds[k], seconds[k + 1]]
                         for k in range(len(times) - 1)],
            'fields': frames.inventory},
        proof_head=PROOF_HEAD, input_manifest_sha256='0' * 64,
        forcing=frames)

    def build(index):
        frames.add_snapshot(snapshots[index], index=index)
        writer.write_segment(index - 1, frames.interval(index - 1, times))
        frames.release(index - 1)

    def seal():
        receipt = writer.seal_cache()
        writer.publish({**PROOF_HEAD,
                        'prepared_cache': {
                            'content_sha256': receipt['content_sha256']},
                        'boundary_stream': writer.boundary_stream_proof()})

    return writer, tmp_path / 'run', build, seal, times


def test_a_chained_preparation_that_clears_out_checkpoints_before_its_seal_and_resumes(
        tmp_path, monkeypatch):
    """A140 end to end at the restart layer: the intervals are read back
    from a chained preparation's own segment files, the checkpoint at 1 h
    is written with two intervals prepared and no seal, and it resumes
    beside the unsealed preparation, on the sealed one, and on the
    one-shot series of the same frames.  A140b: each segment marker and
    the sealed cache's interval rows record the end frame the builder
    built toward, and the continuity check holds on it."""

    import json

    from gpuwm.ingest.boundary_stream import (
        read_head, segment_marker_path, streamed_boundaries)
    from gpuwm.state_serialization_contract import (
        lateral_boundary_prefix_row)
    from test_boundary_stream import _frames
    from test_restart import _fill_setup, _shim_state

    cfg = _specified()
    snapshots = _clear_out_snapshots()
    writer, root, build, seal, times = _chained_preparation(
        tmp_path, snapshots)
    try:
        build(1)
        build(2)
        head = read_head(root)

        def attached(boundaries):
            state = _shim_state(cfg, monkeypatch)
            _fill_setup(state)
            state.lateral_boundaries = boundaries
            state._lateral_boundary_device = _external_mirror(bound=True)
            return state

        original = attached(streamed_boundaries(root, head=head))
        first, second = (original.lateral_boundaries.intervals[k]
                         for k in (0, 1))
        markers = [json.loads(segment_marker_path(root, k).read_text())
                   for k in (0, 1)]
        assert [marker['end_frame_sha256'] for marker in markers] == [
            first.end_frame_sha256, second.end_frame_sha256]
        built = [lateral_boundary_prefix_row(interval)
                 for interval in (first, second)]
        rebuilt = [lateral_boundary_prefix_row(interval,
                                               rebuilt_end_frame=True)
                   for interval in (first, second)]
        assert built[0]['end_frame_sha256'] == built[1]['start_frame_sha256']
        assert rebuilt[0]['end_frame_sha256'] != \
            rebuilt[1]['start_frame_sha256']
        original.elapsed_seconds = 3600.
        original.thp.fill(41.)
        path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
        header, _ = restart._load_restart(path, with_arrays=False)
        assert header['boundary_stream'] == {
            'head_sha256': head['head_sha256']}
        assert header['lateral_boundary_prefix']['intervals'] == built
        assert not (root / 'proof.json').exists()

        streaming = attached(streamed_boundaries(root, head=head))
        streaming.thp.fill(-1.)
        restart.restore_restart(path, streaming, cfg)
        np.testing.assert_array_equal(streaming.thp, original.thp)

        build(3)
        seal()
        sealed_header = json.loads(
            (root / 'prepared-cache' / 'header.json').read_text())
        rows = sealed_header['metadata']['lbc']['intervals']
        assert [row['end_frame_sha256'] for row in rows[:2]] == [
            marker['end_frame_sha256'] for marker in markers]
        sealed = attached(streamed_boundaries(root, head=head))
        assert sealed.lateral_boundaries.intervals.sealed()
        sealed.thp.fill(-2.)
        info = restart.restore_restart(path, sealed, cfg)
        assert info.elapsed_seconds == 3600.
        np.testing.assert_array_equal(sealed.thp, original.thp)

        eager = attached(_frames(snapshots).build(times))
        assert lateral_boundary_prefix_identity(eager) == \
            lateral_boundary_prefix_identity(sealed)
        eager.thp.fill(-3.)
        restart.restore_restart(path, eager, cfg)
        np.testing.assert_array_equal(eager.thp, original.thp)
    finally:
        writer._stop_heartbeat(final='producing')


def test_a_domain_tree_checkpoints_its_streamed_root_before_the_seal_and_resumes(
        tmp_path, monkeypatch):
    """The tree runner's checkpoint (``write_tree_restart``) over a root
    still streaming from its chained preparation, whose boundary clears
    out at 1 h: written with two intervals prepared, and restored with its
    rebuilt child on the sealed series."""

    from datetime import timedelta
    from test_restart import _sealed_tree_fixture

    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=4, run_seconds=10800.0, payload_seed=31)
    source.root.state.lateral_boundaries = _clear_out_boundaries()
    _streamed(source.root.state, ready=2)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    header = restart.read_restart_header(root_path)
    assert header['boundary_stream'] == {'head_sha256': 'a' * 64}
    assert len(header['lateral_boundary_prefix']['intervals']) == 2

    resumed, _ = _sealed_tree_fixture(
        monkeypatch, forcing_count=4, run_seconds=10800.0, payload_seed=91)
    resumed.root.state.lateral_boundaries = _clear_out_boundaries()
    _streamed(resumed.root.state, ready=3, sealed=True)
    info = restart.restore_tree_restart(root_path, resumed)
    assert info.elapsed_ticks == 3600
    for got, want in zip(resumed.walk_parent_first(),
                         source.walk_parent_first()):
        assert np.array_equal(got.state.u, want.state.u, equal_nan=True)
