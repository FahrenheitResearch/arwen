"""``--stream-init``: the flag, and the pricing rule ``auto`` decides on.

Everything here runs with no card.  That is not a convenience -- it is the
point of the split between :func:`_stream_init_pricing` (arithmetic over
three integers) and :func:`_free_device_bytes` (the only part that needs a
device).  The rule that decides whether a 16 GB card can hold a prepared
domain must be testable on a machine that does not have one, or it is a rule
nobody can check until it has already refused a forecast.

The instrument is tested in BOTH directions, per the house rule: a domain
that fits must choose resident, a domain that does not must choose store, and
the boundary between them must sit where the constants say it does rather
than where an off-by-one puts it.
"""

from __future__ import annotations

import pytest

import gpuwm.prepared_single_domain_forecast as runner


GIB = 1 << 30


class _Reader:
    """A ``PreparedCacheReader``'s only surface the pricing reads.

    ``arrays`` is the header's own manifest -- ``{key: {"nbytes": ...}}`` --
    already validated against each array's shape and dtype when the real
    reader opened the bundle.  The double is deliberately that narrow: if the
    pricing ever starts reading something else off a reader, this stops
    compiling rather than silently pricing a different thing.
    """

    def __init__(self, arrays, content_sha256="cafe"):
        self.arrays = arrays
        self.content_sha256 = content_sha256


class _Decision:
    def __init__(self, stream):
        self.stream = stream

    def explain(self):
        return "[tiles] OFF: test double"


class _Cfg:
    """The three extents the pricing reads off a run config."""

    def __init__(self, n, nz=49):
        self.nx = self.ny = int(n)
        self.nz = int(nz)


#: THE GATE (b) CASE, to the byte, from the receipts in
#: ``Downloads/node1-streaming/STREAM-ATTACH.md``.  1024 x 1024 x 49 on a
#: 5070 Ti: ``state/*`` in the cache manifest, the free memory the decision was
#: actually taken against, and the allocation the resident road then died on.
OC1024_STATE_MANIFEST_BYTES = 4_945_485_824
OC1024_FREE_BYTES = 11_053_301_760
OC1024_TOTAL_BYTES = 16_609_378_304
OC1024_RESIDENT_DEATH_BYTES = 16_276_726_272 + 205_520_896


def _manifest(state_bytes, other_bytes=0):
    return _Reader({
        "state/u": {"nbytes": state_bytes // 2},
        "state/w": {"nbytes": state_bytes - state_bytes // 2},
        # Everything the restore does NOT put on the card at state shape.
        "met/TT": {"nbytes": other_bytes},
        "lbc/0/u/west/value": {"nbytes": other_bytes},
        "coord/znw": {"nbytes": 8},
    })


# ---------------------------------------------------------------------------
# what the resident road actually costs
# ---------------------------------------------------------------------------

def test_priced_from_state_arrays_only():
    """``restore`` copies ``state/*`` onto the card and nothing else.

    The met fields, the LBC tables and the coordinate arrays are read to the
    HOST; pricing them as device bytes would refuse a domain that fits.
    """
    reader = _manifest(10 * GIB, other_bytes=4 * GIB)
    assert runner._prepared_state_manifest_bytes(reader) == 10 * GIB


def test_headroom_is_the_measured_ratio_not_a_guess():
    """1.40, and derived from the two per-column measurements it cites."""
    assert runner._PREPARED_RESIDENT_HEADROOM == pytest.approx(
        15780.0 / 11276.5)
    assert runner._PREPARED_RESIDENT_HEADROOM == pytest.approx(1.40, abs=0.01)


def test_pricing_applies_headroom_and_the_fit_fraction():
    priced = runner._stream_init_pricing(10 * GIB, 16 * GIB, 16 * GIB)
    assert priced["state_manifest_bytes"] == 10 * GIB
    assert priced["priced_resident_bytes"] == int(round(
        10 * GIB * runner._PREPARED_RESIDENT_HEADROOM))
    assert priced["fit_allowance_bytes"] == int(
        16 * GIB * runner._AUTO_RESIDENT_FIT_FRACTION)
    # 10 GiB of state prices at 14.0 GiB against an allowance of 12.8 GiB.
    assert priced["fits"] is False


def test_the_manifest_alone_underprices_the_case_that_actually_refused():
    """THE DEFECT, pinned as a number rather than as a story.

    ``manifest x headroom`` is what ``auto`` used to decide on, and on the one
    case where the answer was checked against a real card it was 2.38x low.
    Kept as a live assertion so that a future edit which quietly reinstates
    the manifest-only rule fails here with the ratio in the message.
    """
    manifest_only = runner._stream_init_pricing(
        OC1024_STATE_MANIFEST_BYTES, OC1024_FREE_BYTES, OC1024_TOTAL_BYTES)
    assert manifest_only["fits"] is True
    ratio = (OC1024_RESIDENT_DEATH_BYTES
             / manifest_only["priced_resident_bytes"])
    assert ratio == pytest.approx(2.38, abs=0.02)


def test_the_measured_per_column_term_predicts_the_refusal():
    """Within 1% of the allocation the resident road actually died on.

    The instrument is checked against a known answer, not against itself: the
    number on the right is a CuPy ``OutOfMemoryError``'s "allocated so far"
    plus the block it could not add, read off ``OC1024.R2.forecast.log``.
    """
    priced = runner._stream_init_pricing(
        OC1024_STATE_MANIFEST_BYTES, OC1024_FREE_BYTES, OC1024_TOTAL_BYTES,
        columns=1024 * 1024, nz=49)
    assert priced["priced_from"] == "measured-per-column"
    assert priced["priced_resident_bytes"] == pytest.approx(
        OC1024_RESIDENT_DEATH_BYTES, rel=0.01)


def test_the_larger_of_the_two_terms_wins_in_both_directions():
    """A cache far bigger than the measured configuration raises the price."""
    huge = runner._stream_init_pricing(
        14 * GIB, 16 * GIB, 16 * GIB, columns=64 * 64, nz=49)
    assert huge["priced_from"] == "manifest"
    assert huge["priced_resident_bytes"] == huge["manifest_priced_bytes"]
    wide = runner._stream_init_pricing(
        1 * GIB, 16 * GIB, 16 * GIB, columns=1024 * 1024, nz=49)
    assert wide["priced_from"] == "measured-per-column"
    assert wide["priced_resident_bytes"] == wide["measured_priced_bytes"]


def test_the_measured_term_scales_with_nz():
    """Half the levels, half the price: one measurement, stated as a rate."""
    tall = runner._stream_init_pricing(0, 16 * GIB, 16 * GIB,
                                       columns=512 * 512, nz=49)
    short = runner._stream_init_pricing(0, 16 * GIB, 16 * GIB,
                                        columns=512 * 512, nz=98)
    assert short["measured_priced_bytes"] == pytest.approx(
        2 * tall["measured_priced_bytes"], rel=1e-9)


def test_the_fit_boundary_sits_where_the_constants_put_it():
    """Both directions across one boundary, to the byte.

    A rule tested only on the side that passes is a rule whose threshold is
    unmeasured.
    """
    free = 16 * GIB
    allowance = int(free * runner._AUTO_RESIDENT_FIT_FRACTION)
    exact = int(allowance / runner._PREPARED_RESIDENT_HEADROOM)
    assert runner._stream_init_pricing(exact, free, free)["fits"] is True
    generous = runner._stream_init_pricing(exact + GIB // 8, free, free)
    assert generous["fits"] is False


# ---------------------------------------------------------------------------
# what the STREAMED run costs BESIDE the resident state
# ---------------------------------------------------------------------------
#
# The resident road does not end at the resident state.  A streamed forecast
# that took it then allocates the tile buffers on top of it and meets the
# radiation transient on top of THAT, and the peak of the three is what the
# card has to hold.  Pricing only the first term against a FRACTION of free
# VRAM made the margin scale the wrong way: 20% of a 16 GiB card is 3.2 GiB
# and covers the tiles, 20% of a 10 GiB card is 1.9 GiB and does not come
# close -- so the rule got safer exactly where the card had more room and
# thinner exactly where it had less.

#: A 10 GiB card as Windows/WDDM presents one: ~9.5 GiB free of 10.0.  WDDM
#: does not refuse an over-commit, it PAGES -- so a rule that over-spends
#: here produces no ``OutOfMemoryError`` at all, just a forecast that stops
#: completing steps.  That is why this has to be caught in the pricing.
CARD10_FREE_BYTES = int(9.5 * GIB)
CARD10_TOTAL_BYTES = int(10.0 * GIB)


def _real_cfg(n, nz=49):
    """A real :class:`RunConfig`, because the streamed term reads a real one.

    ``footprint_for`` asks ``gpuwm.config``/``gpuwm.core.physics`` whether
    radiation is on, and those read a config rather than a namespace.  Still
    no card: everything below is arithmetic over the measured footprints.
    """
    from gpuwm.config import RunConfig

    return RunConfig(nx=n, ny=n, nz=nz, dx=3000.0, dy=3000.0, ztop=20000.0,
                     dt=15.0, run_seconds=3600.0,
                     ra_sw_physics=4, ra_lw_physics=4)


def _pinned_decision(cfg, tile=128):
    from gpuwm.core import streaming

    return streaming.decide(
        cfg, streaming.StreamingOptions(mode="on", tile_nx=tile, tile_ny=tile))


def test_the_streamed_term_is_the_tiles_plus_the_radiation_transient():
    """Both halves, each traceable to the constant it came from."""
    from tilestream import autoplan
    from gpuwm.core import streaming

    cfg = _real_cfg(424)
    decision = _pinned_decision(cfg)
    priced = runner._streamed_peak_bytes(cfg, decision)
    envelope = streaming.streamed_envelope(cfg, None, decision=decision)
    rad = autoplan.footprint_for(cfg).radiation_transient_bytes
    assert rad == autoplan.RADIATION_TRANSIENT_BYTES["full"]
    assert priced == int(envelope.vram_bytes) + int(rad)
    # The tiles alone are 5.13 GiB and radiation adds 2.74 on top.
    assert priced / GIB == pytest.approx(7.87, abs=0.05)


def test_a_run_that_does_not_stream_prices_no_streamed_term():
    cfg = _real_cfg(424)

    class _NoStream:
        stream = False

    assert runner._streamed_peak_bytes(cfg, _NoStream()) is None


def test_a_decision_that_cannot_be_priced_is_none_not_a_guess():
    """An unpriceable streamed term must not silently become zero.

    Zero is the answer that reinstates the defect, so the failure has its own
    value and the caller decides what to do with it.
    """
    assert runner._streamed_peak_bytes(None, _Decision(True)) is None
    assert runner._streamed_peak_bytes(_real_cfg(424), _Decision(True)) is None


def test_pricing_adds_the_streamed_term_to_the_peak():
    priced = runner._stream_init_pricing(
        0, CARD10_FREE_BYTES, CARD10_TOTAL_BYTES,
        columns=424 * 424, nz=49, streamed_bytes=int(7.87 * GIB))
    assert priced["streamed_peak_bytes"] == int(7.87 * GIB)
    assert priced["resident_peak_bytes"] == (
        priced["priced_resident_bytes"] + priced["streamed_peak_bytes"])
    assert priced["fits"] is False


def test_an_unpriced_streamed_term_is_reported_as_absent():
    """``None`` is carried into the receipt rather than folded away as 0."""
    priced = runner._stream_init_pricing(0, 16 * GIB, 16 * GIB,
                                         columns=384 * 384, nz=49)
    assert priced["streamed_peak_bytes"] is None
    assert priced["resident_peak_bytes"] == priced["priced_resident_bytes"]


# ---------------------------------------------------------------------------
# the road ``auto`` chooses, and the two that force one
# ---------------------------------------------------------------------------

def test_a_run_that_does_not_stream_is_untouched():
    """No road, no receipt, no device query, nothing printed.

    ``decision=None`` is what an experiment with no ``[tiles]`` produces, and
    the resident route must be reached through exactly the code it was
    reached through before this flag existed.
    """
    for mode in ("auto", "resident"):
        road, receipt = runner._choose_stream_init_road(
            mode, decision=None, reader=None)
        assert (road, receipt) == ("resident", None)


def test_store_on_a_run_that_does_not_stream_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        runner._choose_stream_init_road(
            "store", decision=_Decision(False), reader=None)
    message = str(excinfo.value)
    assert "does not stream" in message
    assert "[tiles]" in message


def test_auto_takes_the_resident_road_when_it_comfortably_fits():
    road, receipt = runner._choose_stream_init_road(
        "auto", decision=_Decision(True), reader=_manifest(4 * GIB),
        device_memory=(15 * GIB, 16 * GIB))
    assert road == "resident"
    assert receipt["fits"] is True
    assert receipt["requested"] == "auto"
    # The sentence carries the numbers, not an adjective.
    assert "GiB" in receipt["why"] and "fits" in receipt["why"]


def test_auto_takes_the_store_road_when_the_domain_does_not_fit():
    road, receipt = runner._choose_stream_init_road(
        "auto", decision=_Decision(True), reader=_manifest(11 * GIB),
        device_memory=(15 * GIB, 16 * GIB))
    assert road == "store"
    assert receipt["fits"] is False
    assert "slab by slab" in receipt["why"]


def test_bare_auto_takes_the_store_road_on_the_case_that_used_to_crash():
    """THE FIXED-MEANS-DEFAULT GATE, at the exact numbers of gate (b).

    No flag: ``auto``, the 1024 x 1024 x 49 cache, the free memory the decision
    was taken against on the card that then refused.  The road must be the
    store, and the receipt must SAY WHY in numbers rather than merely record
    an outcome.
    """
    road, receipt = runner._choose_stream_init_road(
        "auto", decision=_Decision(True),
        reader=_manifest(OC1024_STATE_MANIFEST_BYTES),
        cfg=_Cfg(1024), device_memory=(OC1024_FREE_BYTES, OC1024_TOTAL_BYTES))
    assert road == "store"
    assert receipt["fits"] is False
    assert receipt["priced_from"] == "measured-per-column"
    assert receipt["columns"] == 1024 * 1024 and receipt["nz"] == 49
    assert "does NOT fit" in receipt["why"]
    assert "slab by slab" in receipt["why"]
    assert "1048576 columns" in receipt["why"]


def test_auto_refuses_the_resident_road_when_the_tiles_will_not_also_fit():
    """THE 10 GiB CARD, and the reason it stalled instead of refusing.

    424 x 424 x 49 prices its resident state at 2.64 GiB, which sits well
    inside 80% of a 10 GiB card's free memory -- so ``auto`` took the resident
    road.  The tiles then cost 5.13 GiB and the radiation transient 2.74 on
    top, for a 10.51 GiB peak against 9.50 GiB free.  On Windows/WDDM that
    over-commit does not raise: the driver pages the excess to system RAM and
    the forecast simply stops completing steps, at step one and again at every
    radiation call afterwards.

    Every number here comes from a measured constant this repository already
    carried; the pricing just never added the second and third to the first.
    """
    cfg = _real_cfg(424)
    decision = _pinned_decision(cfg)
    road, receipt = runner._choose_stream_init_road(
        "auto", decision=decision, reader=_manifest(1 * GIB), cfg=cfg,
        device_memory=(CARD10_FREE_BYTES, CARD10_TOTAL_BYTES))
    assert road == "store"
    assert receipt["fits"] is False
    # The resident state ALONE would have fitted; that is the whole defect.
    assert receipt["priced_resident_bytes"] < receipt["fit_allowance_bytes"]
    assert receipt["resident_peak_bytes"] > receipt["fit_allowance_bytes"]
    # And the sentence says which term broke it, in bytes.
    assert "tile buffers" in receipt["why"]
    assert "radiation" in receipt["why"]


def test_the_parity_venue_keeps_the_resident_road_on_the_card_it_fits():
    """384 x 384 on a 16 GiB card must not be pushed onto the store.

    The streamed term is a real cost and adding it moves the crossover, but a
    domain with genuine room for all three terms is still the resident road's
    -- it is the faster one and the one every pre-existing receipt was written
    by.  A fix that swept the parity venue onto the store would be a
    regression dressed as a repair.
    """
    cfg = _real_cfg(384)
    decision = _pinned_decision(cfg)
    road, receipt = runner._choose_stream_init_road(
        "auto", decision=decision, reader=_manifest(1 * GIB), cfg=cfg,
        device_memory=(15 * GIB, 16 * GIB))
    assert road == "resident"
    assert receipt["fits"] is True
    assert receipt["streamed_peak_bytes"] > 0


def test_the_same_domain_flips_road_on_the_smaller_card():
    """One domain, two cards, two roads -- the instrument in both directions.

    The old rule priced the resident state against a FRACTION of free VRAM, so
    it answered the same way on both cards up to a scale factor and could not
    see that the tiles are an ABSOLUTE cost.  384 x 384 fits all three terms
    on 16 GiB and cannot on 10 GiB, and the rule must now say so.
    """
    cfg = _real_cfg(384)
    decision = _pinned_decision(cfg)
    big, _ = runner._choose_stream_init_road(
        "auto", decision=decision, reader=_manifest(1 * GIB), cfg=cfg,
        device_memory=(15 * GIB, 16 * GIB))
    small, _ = runner._choose_stream_init_road(
        "auto", decision=decision, reader=_manifest(1 * GIB), cfg=cfg,
        device_memory=(CARD10_FREE_BYTES, CARD10_TOTAL_BYTES))
    assert (big, small) == ("resident", "store")


def test_a_fitting_domain_still_takes_the_resident_road():
    """The fastest road keeps winning where it really fits.

    384 x 384 x 49 is the parity venue: both roads run it, and the resident one
    is the road every pre-existing receipt was written by.  A pricing fix that
    pushed it onto the store would be a regression dressed as a repair.
    """
    road, receipt = runner._choose_stream_init_road(
        "auto", decision=_Decision(True),
        reader=_manifest(OC1024_STATE_MANIFEST_BYTES * 384 * 384
                         // (1024 * 1024)),
        cfg=_Cfg(384), device_memory=(15 * GIB, 16 * GIB))
    assert road == "resident"
    assert receipt["fits"] is True
    assert "faster one" in receipt["why"]


def test_the_road_flips_at_the_width_the_measurement_puts_it_at():
    """Both sides of one boundary, walked in widths rather than in bytes.

    A rule tested only where it passes is a rule whose threshold is unmeasured;
    this walks 3 km domains up until ``auto`` changes its mind, and pins that
    the change happens once and stays.
    """
    roads = {
        n: runner._choose_stream_init_road(
            "auto", decision=_Decision(True), reader=_manifest(1 * GIB),
            cfg=_Cfg(n), device_memory=(OC1024_FREE_BYTES,
                                        OC1024_TOTAL_BYTES))[0]
        for n in (256, 512, 704, 768, 896, 1024, 1152, 1280)}
    assert roads[256] == roads[512] == "resident"
    assert roads[1024] == roads[1152] == roads[1280] == "store"
    flips = [n for n, road in sorted(roads.items()) if road == "store"]
    # Monotone: once the store road is chosen it is never given back.
    assert flips == sorted(n for n in roads if n >= flips[0])


def test_store_is_forced_even_on_a_domain_that_would_have_fitted():
    """The bit-parity gate's road: both roads over one small domain."""
    road, receipt = runner._choose_stream_init_road(
        "store", decision=_Decision(True), reader=_manifest(1 * GIB),
        device_memory=(15 * GIB, 16 * GIB))
    assert road == "store"
    # The pricing is still recorded, so the receipt shows the run COULD have
    # gone resident and was told not to.
    assert receipt["fits"] is True
    assert receipt["requested"] == "store"


def test_resident_is_forced_even_on_a_domain_that_cannot_fit():
    """Not silently rescued: it fails where it fails today, in the alloc."""
    road, receipt = runner._choose_stream_init_road(
        "resident", decision=_Decision(True), reader=_manifest(14 * GIB),
        device_memory=(15 * GIB, 16 * GIB))
    assert road == "resident"
    assert receipt["fits"] is False


def test_an_unknown_road_is_refused_before_anything_is_read():
    with pytest.raises(ValueError, match="--stream-init must be one of"):
        runner._choose_stream_init_road(
            "hybrid", decision=_Decision(True), reader=None)


# ---------------------------------------------------------------------------
# the flag itself
# ---------------------------------------------------------------------------

_REQUIRED = [
    "--source", "gfs", "--prepared-root", ".", "--proof-sha256", "x",
    "--source-manifest-sha256", "x", "--prepared-content-sha256", "x",
    "--experiment-config", "e.toml", "--wps-namelist", "n.input",
    "--io-mode", "history", "--outdir", "out",
]


def test_stream_init_defaults_to_auto():
    assert runner._parse_args(_REQUIRED).stream_init == "auto"


@pytest.mark.parametrize("value", runner.STREAM_INIT_CHOICES)
def test_every_documented_road_parses(value):
    args = runner._parse_args(_REQUIRED + ["--stream-init", value])
    assert args.stream_init == value


def test_a_bad_road_is_a_usage_error_not_a_traceback(capsys):
    """argparse's own refusal: exit 2 and a sentence listing the choices."""
    with pytest.raises(SystemExit) as excinfo:
        runner._parse_args(_REQUIRED + ["--stream-init", "sideways"])
    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "--stream-init" in stderr
    assert "'sideways'" in stderr
    for choice in runner.STREAM_INIT_CHOICES:
        assert choice in stderr


# ---------------------------------------------------------------------------
# the store bundle's receipt is pinned exactly as the restored cache's is
# ---------------------------------------------------------------------------

def test_store_receipt_must_carry_the_caller_pinned_digest():
    good = {"schema": runner._PREPARED_STORE_SCHEMA, "status": "LOADED",
            "content_sha256": "beef"}
    runner._validate_store_bundle_receipt(good, "beef")
    for broken in ({**good, "content_sha256": "other"},
                   {**good, "status": "PARTIAL"},
                   {**good, "schema": "something-else"},
                   None):
        with pytest.raises(ValueError, match="caller-pinned"):
            runner._validate_store_bundle_receipt(broken, "beef")


def test_the_unchecked_block_names_the_setup_fingerprint():
    """The one check the store road cannot make says so, in the receipt."""
    gaps = runner._store_direct_gaps(_Reader({}, content_sha256="beef"))
    assert gaps["setup_fingerprint"]["checked"] is False
    assert gaps["setup_fingerprint"]["cache_content_sha256"] == "beef"
    assert "domain-shaped state" in gaps["setup_fingerprint"]["why"]
