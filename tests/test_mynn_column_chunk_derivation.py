"""The MYNN column width is the swept optimum, and the card's answer is kept.

``mynn_dmp_mf_columns`` is 52% of all GPU time in a quiet root step of the
299x299x59 + 282x129 pair, and it launches 128 blocks of 128 threads -- on a
170-SM card that is 42 SMs given nothing, one block on every SM that gets
any, and 4.7% of the card's resident thread slots, against a register file
that admits three blocks per SM.  So the width was derived from the card
instead: the largest chunk whose blocks still fit in one wave, clipped by
what the card's memory admits, never below the 16,384 columns 2.7.4 shipped.

**Then it was measured, and the derivation lost its own sweep.**  Ten arms on
an RTX 5090 at nz = 59 on 2026-09-15, sole tenant, 200 root steps of
the pair's prepared control leg per arm, two passes with the second in
reverse order (``tools/mynn_chunk_sweep/summary.txt`` and
``sweep.tsv`` beside it, written by the harness in that directory):
the quiet root cycle rose with every widening, 0.5263 s at 16,384 columns
against 0.6252 s at 65,536, with no knee anywhere between.  The width the
derivation picks on that card, 65,280, is 18.8% slower and costs 2.9 GiB
more scratch.

**And that sweep had only looked one way.**  It started at the width 2.7.4
shipped and only ever widened, so all it could find was a direction, and it
ranked its own narrowest arm first.  Ten more arms the same evening, same
card, same prepared leg, same protocol, running downward: 0.4450 s at 8,192
columns against 0.5268 s at 16,384 and 0.5039 s at 4,096.  The optimum is
interior, it is 8,192 columns, and it is what ships.  The two sweeps overlap
at 16,384 and agree there to 0.0005 s, which is what lets the nine widths be
read as one curve.

So the width that SHIPS is the measured one and the derivation is a receipt
term.  Both halves are tested here: the arithmetic still has to be right,
because the receipt states it and the next card's sweep will be argued
against it, and the shipped default has to be the number the sweeps
measured, because a default that quietly re-derives itself is how the
regression got in.  Every test injects its card rather than reading one, so
the CPU tier proves the arithmetic on the 5090, on a 10 GiB card and on no
card at all from the same run.

The width is workspace shape only -- one thread owns one whole column, reads
no neighbour, and the kernels hold no shared memory, no atomic and no
per-chunk seed -- so nothing here is a claim about the forecast.  That claim
is ``test_the_column_chunk_is_not_a_seam`` in
``tests/test_mynn_pbl_scratch.py``, which needs a card, the sweeps' own
twenty arms, which wrote one digest between them, and the 1,500-step pair at
8,192 columns whose 244 frames were byte-identical to the baseline's.
"""

import inspect

import pytest

from gpuwm.core import mynn_pbl_scratch as scratch


#: The 5090 the pair runs on: 170 SMs, 2,048 resident threads per SM, a
#: 65,536-register file per SM.  Free/total are a quiet card.
CARD_5090 = dict(sm_count=170, max_threads_per_sm=2048,
                 registers_per_sm=65536, warp_size=32,
                 free_bytes=31 * 1024 ** 3, total_bytes=32 * 1024 ** 3,
                 device_name="NVIDIA GeForce RTX 5090")

#: A 10 GiB card: 68 SMs, 1,536 threads per SM, the same register file.
CARD_10GIB = dict(sm_count=68, max_threads_per_sm=1536,
                  registers_per_sm=65536, warp_size=32,
                  free_bytes=9538895872, total_bytes=10736893952,
                  device_name="10 GiB card")

#: A card with room to fill but almost none to spare.
CARD_TIGHT = dict(sm_count=68, max_threads_per_sm=1536,
                  registers_per_sm=65536, warp_size=32,
                  free_bytes=3 * 1024 ** 3, total_bytes=6 * 1024 ** 3,
                  device_name="tight card")

#: The sweeps of record, as the shipped constant's own docstring states
#: them: columns to median quiet root cycle in seconds, two passes per
#: sweep, an RTX 5090, nz = 59, 2026-09-15.  The tests below read the
#: ranking out of these rather than restating a winner, so the citation and
#: the assertion cannot drift.
#:
#: They are kept apart because they are two measurements five hours apart
#: that overlap at one width, and that overlap is the only reason the nine
#: others can be read as one curve.
SWEEP_UPWARD = {16384: 0.5263, 32768: 0.5819, 45056: 0.6170,
                65536: 0.6252, 90112: 0.6395}
SWEEP_DOWNWARD = {4096: 0.5039, 8192: 0.4450, 12288: 0.4819,
                  16384: 0.5268, 24576: 0.5353}

#: The nine widths as one curve.  The overlap takes the downward sweep's
#: figure, the slower of the two readings at that width, so nothing in the
#: ranking below is bought by picking the kinder number.
SWEEP_2026_09_15 = {**SWEEP_UPWARD, **SWEEP_DOWNWARD}

#: The worst pass-to-pass spread seen in either sweep, at any width.
WORST_PASS_SPREAD = 0.0037

#: Where that sweep's record lives, quoted in the constant's
#: documentation so the number is traceable from the source alone.
SWEEP_RECORD = "tools/mynn_chunk_sweep/summary.txt"


def _shipped_width_documentation() -> str:
    """The ``#:`` block above ``MYNN_PBL_COLUMN_CHUNK_DEFAULT``.

    Read out of the module source because a ``#:`` comment is documentation
    for a reader and for Sphinx but is not an attribute at run time, and the
    citation has to be checked where it actually lives.
    """
    source = inspect.getsource(scratch)
    head, sep, _ = source.partition("MYNN_PBL_COLUMN_CHUNK_DEFAULT = ")
    assert sep, "the shipped default is no longer declared by that name"
    return head.rpartition("MYNN_PBL_COLUMN_CHUNK_FLOOR = 16384")[2]


@pytest.fixture(autouse=True)
def _no_ambient_state(monkeypatch):
    """No inherited override, no memo, no pin, and no card by accident.

    Every test injects its own card.  Without this a box that happens to
    have a GPU would answer from its own device and the arithmetic under
    test would never run.
    """
    monkeypatch.delenv(scratch.MYNN_PBL_COLUMN_CHUNK_ENV, raising=False)
    monkeypatch.setattr(scratch, "_RESOLVED", {})
    monkeypatch.setattr(scratch, "_PINNED", None)
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: None)
    monkeypatch.setattr(scratch, "MYNN_PBL_COLUMN_CHUNK",
                        scratch.MYNN_PBL_COLUMN_CHUNK_DEFAULT)


def test_the_shipped_width_is_the_fastest_arm_of_the_sweep():
    """Fixed means default: the default is the number that was measured.

    The sweeps of record are the twenty arms of 2026-09-15 on an RTX 5090,
    ``tools/mynn_chunk_sweep/summary.txt`` (restated in the documentation
    of ``MYNN_PBL_COLUMN_CHUNK_DEFAULT``).  Their fastest arm is the shipped
    width, and it stays the shipped width on a card that would gladly hold
    eight times as much.

    This test is the guard on the regression that produced it: the first
    package of these levers ran the derived width and came out 0.5% SLOWER
    as a pair, entirely because this one number was taken from the card
    instead of from a measurement.

    It is also the guard on the second way this number goes wrong.  The
    default and the derivation's floor were one constant while the upward
    sweep was the whole record; they are two now, and the shipped width is
    BELOW the floor, so anything that re-couples them puts the default back
    to a width that was measured slower.
    """
    fastest = min(SWEEP_2026_09_15, key=SWEEP_2026_09_15.get)
    assert fastest == 8192
    assert scratch.MYNN_PBL_COLUMN_CHUNK_DEFAULT == fastest
    assert scratch.MYNN_PBL_COLUMN_CHUNK == fastest, "the module-level knob"
    assert fastest < scratch.MYNN_PBL_COLUMN_CHUNK_FLOOR == 16384
    for card in (CARD_5090, CARD_10GIB, CARD_TIGHT, None):
        choice = scratch.choose_mynn_column_chunk(59, card=card, environ={})
        assert choice.chunk == fastest, card
        assert choice.source == "pinned-measured", card
    # The citation travels with the constant, or the next reader has a bare
    # magic number and no way to argue with it -- and BOTH sweeps travel
    # with it, because the upward one on its own endorses 16,384 and a
    # reader who finds only it will re-derive the width this default left
    # behind.
    documentation = _shipped_width_documentation()
    for token in ("2026-09-15", SWEEP_RECORD, "0.5263", "65,280",
                  "0.4450", "8,192", "2,390.5"):
        assert token in documentation, token


def test_the_shipped_width_beats_every_other_arm_by_more_than_the_spread():
    """The ranking is a finding, not a preference: the gaps are real.

    The worst pass-to-pass spread at any width in either sweep was 0.0037 s;
    the gap from the winner to the runner-up is 0.0369 s, nearly ten times
    that.  A sweep whose arms sat inside their own noise would not settle a
    default, and these do not.
    """
    ranked = sorted(SWEEP_2026_09_15.items(), key=lambda item: item[1])
    (best, best_s), (_second, second_s) = ranked[0], ranked[1]
    assert best == scratch.MYNN_PBL_COLUMN_CHUNK_DEFAULT
    assert second_s - best_s > 9 * WORST_PASS_SPREAD


def test_the_optimum_is_interior_and_not_the_end_of_a_ladder():
    """A minimum with a slower arm on each side is a measurement.

    A winner at the end of a ladder is only a direction, and that is how the
    width came to be pinned at 16,384: the upward sweep started there, only
    ever widened, and ranked its own narrowest arm first.  The curve falls
    to the shipped width and turns back up below it, so no further narrowing
    is on offer, and it rises on the other side everywhere, which is why no
    ceiling term -- card or VRAM -- can be the binding term on this card.
    """
    widths = sorted(SWEEP_2026_09_15)
    best = scratch.MYNN_PBL_COLUMN_CHUNK_DEFAULT
    assert widths[0] < best < widths[-1], "the winner is not an endpoint"
    seconds = [SWEEP_2026_09_15[width] for width in widths]
    turn = widths.index(best)
    assert seconds[:turn + 1] == sorted(seconds[:turn + 1], reverse=True)
    assert seconds[turn:] == sorted(seconds[turn:])
    # Below the optimum the launch count starts to dominate: 4,096 columns
    # cost 13.2% more per cycle than 8,192 for a quarter of the workspace.
    assert SWEEP_DOWNWARD[4096] - SWEEP_DOWNWARD[8192] > 9 * WORST_PASS_SPREAD
    # Widening was monotonic in its own right, which is the finding that
    # retired the derivation.
    upward = [SWEEP_UPWARD[width] for width in sorted(SWEEP_UPWARD)]
    assert upward == sorted(upward)
    # And the two sweeps are one curve only because they overlap and agree
    # there: five hours apart, same width, 0.0005 s between them.
    overlap = abs(SWEEP_UPWARD[16384] - SWEEP_DOWNWARD[16384])
    assert round(overlap, 4) == 0.0005


def test_this_card_would_be_filled_in_one_wave_by_the_derived_width():
    """170 SMs x 3 blocks x 128 threads = 65,280 columns, 510 blocks.

    Three blocks per SM is the register-file ceiling: 65,536 registers per
    SM at 168 registers per thread is 390 threads, 384 after rounding down
    to a warp, which is three 128-thread blocks.  The card's 2,048 resident
    threads per SM are not the binding term and must not be used as one --
    16 blocks per SM would be a width no card's memory could hold.

    The arithmetic still has to be right, because the receipt states it and
    the next card's sweep will be argued against it.  It is no longer the
    width that runs: this is the arm the 2026-09-15 sweeps measured at
    0.6252 s per quiet root cycle, against 0.5263 s at 16,384 columns and
    0.4450 s at the width that ships.
    """
    choice = scratch.derive_mynn_column_chunk(59, card=CARD_5090, environ={})
    assert choice.chunk == 170 * 3 * 128 == 65280
    assert choice.blocks == 510
    assert choice.source == "card-ceiling"
    assert choice.card_ceiling == 65280
    assert choice.vram_ceiling >= choice.card_ceiling
    # 3,919.1 MiB, against 491.8 MiB at the shipped width: the whole cost of
    # taking this road, stated rather than discovered at allocation time.
    assert choice.workspace_bytes == scratch.mynn_pbl_scratch_bytes(65280, 59)
    assert 3900 < choice.workspace_bytes / 2 ** 20 < 3930
    # And it is not what runs.
    assert scratch.choose_mynn_column_chunk(
        59, card=CARD_5090, environ={}).chunk == 8192


def test_a_smaller_card_derives_a_smaller_width_from_its_own_terms():
    """68 SMs would fill at 26,112 columns; 10 GiB only admits 20,480.

    Both ceilings stay live in the receipt because the question the next
    card's sweep asks is which of them, if either, predicts throughput
    there.  On the one card swept so far neither does.
    """
    choice = scratch.derive_mynn_column_chunk(59, card=CARD_10GIB, environ={})
    assert choice.card_ceiling == 68 * 3 * 128 == 26112
    assert choice.source == "vram-ceiling"
    assert choice.chunk == 20480 < choice.card_ceiling
    assert choice.workspace_bytes < 1.25 * 1024 ** 3


def test_a_card_with_no_room_derives_the_width_2_7_4_shipped():
    """The floor is what 2.7.4 ran, so the derivation never crosses it.

    Floor and default are two constants and are no longer the same number:
    the floor bounds this derivation, which is a receipt term, while the
    default is the swept optimum a run walks and is narrower.  A sweep on
    the next card may move the default again without moving what the
    derivation may never go below, which is the case these two constants
    were separated for.
    """
    choice = scratch.derive_mynn_column_chunk(59, card=CARD_TIGHT, environ={})
    assert choice.vram_ceiling < scratch.MYNN_PBL_COLUMN_CHUNK_FLOOR
    assert choice.chunk == scratch.MYNN_PBL_COLUMN_CHUNK_FLOOR == 16384
    assert choice.source == "floor"
    assert scratch.MYNN_PBL_COLUMN_CHUNK_DEFAULT < choice.chunk
    # The floor binds the derivation only.  What this card RUNS is the
    # shipped width, which its memory holds with room to spare.
    assert scratch.choose_mynn_column_chunk(
        59, card=CARD_TIGHT, environ={}).chunk == 8192


def test_no_card_keeps_the_floor_and_touches_no_runtime():
    """``gpuwm domain`` on a CPU-only box prices what the card will run.

    ``probe_mynn_card`` returns None off a GPU, and the derivation must not
    try to make one up.  Since the default is pinned, the CPU-only estimate
    and the width a 5090 runs are the same 8,192 columns, which is what
    makes a preflight on a laptop worth reading.
    """
    choice = scratch.derive_mynn_column_chunk(59, card=None, environ={})
    assert choice.chunk == scratch.MYNN_PBL_COLUMN_CHUNK_FLOOR
    assert choice.source == "floor-no-card"
    assert choice.card_ceiling is None and choice.free_bytes is None
    assert scratch.choose_mynn_column_chunk(
        59, card=None, environ={}).chunk == 8192


def test_the_operator_override_is_honoured_verbatim_and_named():
    """The sweep's handle: the value it asks for is the value that runs.

    It is deliberately not clipped to either ceiling, and it outranks the
    shipped default as well as the derivation.  A sweep that asked for
    90,112 columns and silently got 16,384 would report a timing and a width
    that do not belong to each other -- which is the whole instrument.
    """
    choice = scratch.choose_mynn_column_chunk(
        59, card=CARD_5090,
        environ={scratch.MYNN_PBL_COLUMN_CHUNK_ENV: " 90112 "})
    assert choice.chunk == 90112
    assert choice.source == "override"
    entry = choice.receipt()
    assert entry["override_env"] == "GPUWM_MYNN_COLUMN_CHUNK"
    assert entry["chunk"] == 90112 and entry["blocks"] == 704
    assert "GPUWM_MYNN_COLUMN_CHUNK" in choice.line()
    # The narrower widths the downward sweep asks for are honoured the same
    # way: nothing clips an override up to the shipped default either, and
    # an override that names the shipped width still reports itself as an
    # override rather than as the default.
    for width in (4096, 8192, 12288, 24576):
        asked = scratch.choose_mynn_column_chunk(
            59, card=CARD_5090,
            environ={scratch.MYNN_PBL_COLUMN_CHUNK_ENV: str(width)})
        assert asked.chunk == width and asked.source == "override"
        assert asked.would_have_derived.chunk == 65280


def test_an_override_that_is_not_a_column_count_is_refused():
    """It sizes the arena, so it is refused rather than quietly dropped.

    An ignored typo would allocate the default's workspace and report the
    sweep's intended width beside a timing taken at neither.
    """
    with pytest.raises(ValueError) as excinfo:
        scratch.choose_mynn_column_chunk(
            59, card=CARD_5090,
            environ={scratch.MYNN_PBL_COLUMN_CHUNK_ENV: "45k"})
    message = str(excinfo.value)
    assert "GPUWM_MYNN_COLUMN_CHUNK" in message
    assert "positive integer number of columns" in message
    for bad in ("0", "-1", "nonsense"):
        with pytest.raises(ValueError):
            scratch.mynn_column_chunk_override(
                {scratch.MYNN_PBL_COLUMN_CHUNK_ENV: bad})


def test_the_vram_ceiling_does_not_move_with_a_neighbour_s_allocation():
    """Quantised, so an ordinary few hundred MiB of drift changes nothing.

    The receipt states this ceiling, and a receipt term that moved with
    whatever else happened to be resident would make two runs of the same
    case on the same card two different runs on paper.
    """
    widths = {
        scratch.mynn_pbl_vram_chunk_ceiling(
            59, free_bytes=CARD_10GIB["free_bytes"] - delta,
            total_bytes=CARD_10GIB["total_bytes"])
        for delta in (0, 64 * 2 ** 20, 200 * 2 ** 20, 400 * 2 ** 20)
    }
    assert widths == {20480}
    # And on a quiet card the term that binds is the card's own size, not
    # what is free, so a co-tenant holding 18 GiB of a 32 GiB card derives
    # the same width as an empty one.
    quiet = scratch.derive_mynn_column_chunk(59, card=CARD_5090, environ={})
    shared = scratch.derive_mynn_column_chunk(
        59, card=dict(CARD_5090, free_bytes=13 * 1024 ** 3), environ={})
    assert quiet.chunk == shared.chunk == 65280


def test_the_ceiling_is_whole_blocks_of_the_launch_width():
    """The launch is ``ceil(columns / 128)`` blocks, so the width is blocks.

    A ceiling that was not a multiple of the block width would put a ragged
    block on the card in the one place the derivation exists to fill.
    """
    for card in (CARD_5090, CARD_10GIB):
        ceiling = scratch.mynn_pbl_card_chunk_ceiling(
            sm_count=card["sm_count"],
            max_threads_per_sm=card["max_threads_per_sm"],
            registers_per_sm=card["registers_per_sm"])
        assert ceiling % scratch.MYNN_PBL_COLUMN_TPB == 0
    # A kernel that spilled to twice the registers fits half the blocks.
    assert scratch.mynn_pbl_card_chunk_ceiling(
        sm_count=170, max_threads_per_sm=2048, registers_per_sm=65536,
        registers_per_thread=336) == 170 * 1 * 128
    # A cheap kernel is capped by the card's resident threads, never above.
    assert scratch.mynn_pbl_card_chunk_ceiling(
        sm_count=170, max_threads_per_sm=2048, registers_per_sm=65536,
        registers_per_thread=16) == 170 * 2048


def test_the_registry_and_the_solver_are_handed_one_width():
    """One resolution per process, because the arena is allocated once.

    ``DomainState.scratch`` keeps the shape a slot was first requested
    with, and the run checks the arena it built against the ledger it
    priced.  A second answer between the two would kill the run at the first
    MYNN call with a capacity error, which is exactly the failure the memo
    exists to prevent.
    """
    from gpuwm.config import RunConfig
    from gpuwm.core import preflight as pf

    cards = iter((CARD_5090, CARD_TIGHT))
    scratch.probe_mynn_card = lambda device=None: next(cards)
    first = scratch.resolve_mynn_column_chunk(59)
    second = scratch.resolve_mynn_column_chunk(59)
    assert first == second == 8192

    cfg = RunConfig(nx=299, ny=299, nz=59, dx=2700.0, dy=2700.0,
                    ztop=20000.0, dt=12.0, run_seconds=60.0, moist=True,
                    mp_physics=28, sf_sfclay_physics=5,
                    sf_surface_physics=3, bl_pbl_physics=5)
    assert pf.mynn_pbl_column_chunk(cfg) == 8192 < 299 * 299
    nest = RunConfig(nx=282, ny=129, nz=59, dx=900.0, dy=900.0,
                     ztop=20000.0, dt=3.0, run_seconds=60.0, moist=True,
                     mp_physics=28, sf_sfclay_physics=5,
                     sf_surface_physics=3, bl_pbl_physics=5)
    # The nest has 36,378 columns, so it walks five chunks of the shared
    # width and declares the same workspace the parent did -- which is the
    # bounded-workspace property this module exists for.
    assert pf.mynn_pbl_column_chunk(nest) == 8192 < 282 * 129
    # A domain narrower than the width asks only for its own columns.
    small = RunConfig(nx=50, ny=20, nz=59, dx=900.0, dy=900.0,
                      ztop=20000.0, dt=3.0, run_seconds=60.0, moist=True,
                      mp_physics=28, sf_sfclay_physics=5,
                      sf_surface_physics=3, bl_pbl_physics=5)
    assert pf.mynn_pbl_column_chunk(small) == 1000


def test_the_width_in_force_is_published_where_the_knob_reads_it(monkeypatch):
    """``tilestream`` reports the width the run uses, not a stale constant.

    The capacity knob reads ``MYNN_PBL_COLUMN_CHUNK`` out of three modules;
    all three have to move together or a tile is sized against one width and
    walked at another.  That was measured once as an unchanged 480^2 ceiling
    at two different chunks.  With the default pinned, the override is what
    moves it, so the override is what this checks.
    """
    import sys

    scratch.probe_mynn_card = lambda device=None: CARD_5090
    assert scratch.resolve_mynn_column_chunk(59) == 8192
    scratch._RESOLVED.clear()
    monkeypatch.setenv(scratch.MYNN_PBL_COLUMN_CHUNK_ENV, "24576")
    assert scratch.resolve_mynn_column_chunk(59) == 24576
    for name in scratch._CHUNK_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        assert module.MYNN_PBL_COLUMN_CHUNK == 24576, name


def test_a_pin_outranks_the_default_and_can_be_released():
    """The in-process handle a capacity measurement drives.

    ``tilestream.shared_workspace.set_mynn_column_chunk`` narrows the width
    on purpose to find a tile ceiling; nothing must widen it back underneath
    the measurement.
    """
    scratch.probe_mynn_card = lambda device=None: CARD_5090
    assert scratch.resolve_mynn_column_chunk(59) == 8192
    scratch.pin_mynn_column_chunk(4096)
    assert scratch.resolve_mynn_column_chunk(59) == 4096
    assert scratch.mynn_column_chunk_receipt()["source"] == "pinned"
    scratch.pin_mynn_column_chunk(None)
    assert scratch.resolve_mynn_column_chunk(59) == 8192
    with pytest.raises(ValueError):
        scratch.pin_mynn_column_chunk(0)


def test_the_receipt_states_the_width_and_what_the_card_would_have_asked():
    """A width nobody can reconstruct is a number, not a measurement.

    The receipt has to say which width a run used and where it came from, or
    a comparison between two runs is between two things that cannot be
    named.  And it has to carry the derivation it did NOT use, or the next
    card's sweep has nothing to disagree with: the whole reason this default
    is pinned is that a sweep beat a derivation, and a sweep on another card
    can only be read against what that card would have asked for.
    """
    scratch.probe_mynn_card = lambda device=None: CARD_5090
    assert scratch.mynn_column_chunk_receipt() is None, "absent when MYNN is off"
    scratch.resolve_mynn_column_chunk(59)
    entry = scratch.mynn_column_chunk_receipt()
    for field in ("chunk", "source", "nz", "floor", "blocks",
                  "threads_per_block", "column_bytes", "workspace_bytes",
                  "override_env", "would_have_derived"):
        assert field in entry, field
    assert entry["chunk"] == 8192
    assert entry["source"] == "pinned-measured"
    # The floor rides the receipt as the DERIVATION's bound, which is the
    # only reason the width above it is allowed to be narrower than it.
    assert entry["floor"] == scratch.MYNN_PBL_COLUMN_CHUNK_FLOOR == 16384
    assert entry["column_bytes"] == scratch.mynn_pbl_column_bytes(59) == 62952
    assert (entry["workspace_bytes"]
            == entry["chunk"] * entry["column_bytes"] + 256)
    # The card's own answer, whole, one level down -- and NOT at the top
    # level, so no reader can mistake the receipt for saying the card chose.
    derived = entry["would_have_derived"]
    for field in ("chunk", "source", "card_ceiling", "vram_ceiling",
                  "sm_count", "max_threads_per_sm", "registers_per_sm",
                  "registers_per_thread", "free_bytes", "total_bytes",
                  "device_name", "blocks", "workspace_bytes"):
        assert field in derived, field
    assert derived["chunk"] == 65280 and derived["source"] == "card-ceiling"
    assert derived["card_ceiling"] == 65280
    assert derived["device_name"] == "NVIDIA GeForce RTX 5090"
    assert "card_ceiling" not in entry and "sm_count" not in entry
    # 3.3 GiB is what the derivation would have cost over the shipped width,
    # and it is the figure lever 1 -- both legs resident on one card -- has
    # to be priced against.
    assert (derived["workspace_bytes"]
            - entry["workspace_bytes"]) > 3.3 * 1024 ** 3


def test_the_per_column_cost_is_the_scheme_s_own_accounting():
    """The VRAM ceiling divides a budget by this, so it has to be exact.

    A per-column figure taken as ``bytes(chunk) / chunk`` would smear the
    flag words -- which do not scale with the width -- across the columns
    and price a wide chunk slightly high and a narrow one slightly low.
    """
    for nz in (49, 59, 79):
        per_column = scratch.mynn_pbl_column_bytes(nz)
        for chunk in (1, 8192, 16384, 65280):
            assert (scratch.mynn_pbl_scratch_bytes(chunk, nz)
                    == chunk * per_column + 256)
    assert scratch.mynn_pbl_column_bytes(49) == 52352
    assert scratch.mynn_pbl_column_bytes(59) == 62952
