"""``tools/dual_run_screen.py`` classifies, and it still bites.

The screen is the standing corruption detector on a card with no ECC, so
it has three ways to be worthless and this file pins all three.

It can cry wolf.  It did: a completed dual run whose 151 history files
were byte-identical and whose 5092 final-state digests agreed was
reported as a CORRUPTION FINDING because six allocator high-water
numbers -- maxima over a 50 ms background poll, not model state --
disagreed between the arms.  The chain stopped and no deliverables were
produced from a run that was fine.

It can also go blind, which is worse and is the failure mode a fix for
the first one invites: widen the exclusion to "the memory block" and the
screen stops comparing a whole block of real run properties.  So the
tests below check the exclusion is SCOPED -- an observed high-water is
excluded, a computed estimate sitting beside it in the same block is
not -- and that an excluded key which disagrees is still PRINTED.  An
exclusion nobody can see is indistinguishable from a screen that never
looked.

And it can say PASS on nothing, which is the worst of the three because
it looks exactly like the good outcome.  Seven separate arrangements --
no receipts, unreadable receipts, a receipt only one arm wrote,
receipts under different names, zero-byte history files, a receipt with
no digest in it, and the same directory handed in twice -- each printed
``DUAL-RUN SCREEN: PASS (bit-identical)`` with ``final-state digests
compared: 0``.  A corruption detector that greens on an empty
comparison is not a weak detector, it is a false one, and every one of
those seven is pinned below as a REFUSAL.  The positive control sits
beside them: a genuinely bit-identical pair still passes, and now says
how much it compared to get there.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def screen_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("tools.dual_run_screen")


@pytest.fixture()
def environmental():
    from gpuwm.verify.cases.convective_boundary_layer import (
        ENVIRONMENTAL_FIELDS)

    return frozenset(ENVIRONMENTAL_FIELDS)


#: The six keys of the false corruption finding, verbatim from the
#: receipt that produced it.  Pinned as data so a future refactor of the
#: predicate has to keep answering for these exact names.
SAMPLED_HIGH_WATER_KEYS = (
    "memory.cupy_pool_peak_total_bytes_observed",
    "memory.cupy_pool_peak_used_bytes_observed",
    "memory.gpu_peak_sampling.probes.cuda_device_used.peak_bytes",
    "memory.gpu_peak_sampling.probes.cupy_pool_total.peak_bytes",
    "memory.gpu_peak_sampling.probes.cupy_pool_used.peak_bytes",
    "memory.gpu_peak_used_bytes_observed",
)

#: Keys inside the SAME memory block that are properties of the run and
#: must stay on the comparison surface.  Without this half, "exclude the
#: memory telemetry" degrades to "stop screening memory".
SCREENED_MEMORY_KEYS = (
    "memory.preflight_alloc_estimate_bytes",
    "memory.gpu_peak_sampling.interval_seconds",
    "memory.gpu_peak_sampling.mechanism",
)


def _receipt() -> dict:
    """A receipt shaped like the real one, small enough to read."""

    return {
        "status": "PASS",
        "wall_seconds": 4184.9,
        "timing_seconds": {"total": 4184.9, "forecast_execution": 4109.8},
        "memory": {
            "cupy_pool_peak_total_bytes_observed": 13176113152,
            "cupy_pool_peak_used_bytes_observed": 12862861312,
            "gpu_peak_used_bytes_observed": 17344036864,
            "preflight_alloc_estimate_bytes": 13988989984,
            "gpu_peak_sampling": {
                "interval_seconds": 0.05,
                "mechanism": "background-polling-thread+boundary-samples",
                "probes": {
                    "cuda_device_used": {"peak_bytes": 17344036864,
                                         "samples": 67569, "error": None},
                    "cupy_pool_total": {"peak_bytes": 13176113152,
                                        "samples": 67569, "error": None},
                    "cupy_pool_used": {"peak_bytes": 12862861312,
                                       "samples": 67569, "error": None},
                },
            },
        },
        "final_state_digest": {"d01:U": "a" * 64, "d01:V": "b" * 64},
        "output": {"frame_count": 2,
                   "files": [{"path": "/a/wrfout_d01_x", "bytes": 10}]},
    }


def _pair(tmp_path: Path, *, mutate_b=None, corrupt_a_bytes=False):
    """Two run directories that differ only as the caller asks."""

    roots = []
    for arm, name in ((0, "run-a"), (1, "run-b")):
        root = tmp_path / name
        (root / "wrfout").mkdir(parents=True)
        (root / "evidence").mkdir(parents=True)
        payload = b"history-bytes" * 8
        if arm == 0 and corrupt_a_bytes:
            payload = bytearray(payload)
            payload[3] ^= 0x01
            payload = bytes(payload)
        (root / "wrfout" / "wrfout_d01_2021-12-11_00_00_00").write_bytes(
            payload)
        doc = _receipt()
        # Every arm's own paths differ by run directory; that is not a
        # difference in the forecast and the fixture carries it so the
        # test exercises the run-local rule too.
        doc["output"]["files"][0]["path"] = str(root / "wrfout" / "f")
        if arm == 1 and mutate_b is not None:
            mutate_b(doc)
        (root / "evidence" / "run-receipt.json").write_text(
            json.dumps(doc, indent=1), encoding="utf-8")
        roots.append(root)
    return roots[0], roots[1]


def _set(doc: dict, dotted: str, value) -> None:
    node = doc
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _get(doc: dict, dotted: str):
    node = doc
    for part in dotted.split("."):
        node = node[part]
    return node


# --------------------------------------------------------------------------
# the classifier
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", SAMPLED_HIGH_WATER_KEYS)
def test_sampled_high_water_is_environmental(screen_module, environmental,
                                             key):
    """The six keys of the false finding, each named."""

    assert screen_module.exclusion_reason(key, environmental) == (
        "sampled memory high-water")


@pytest.mark.parametrize("key", SCREENED_MEMORY_KEYS)
def test_the_rest_of_the_memory_block_is_still_screened(
        screen_module, environmental, key):
    """The fix must not turn into "stop comparing memory"."""

    assert screen_module.exclusion_reason(key, environmental) is None


def test_the_rule_is_scoped_to_the_memory_block(screen_module,
                                                environmental):
    """An ``_observed`` leaf elsewhere in a receipt keeps being compared.

    The predicate keys on the block, not on the spelling of the leaf.
    A suffix rule loose in the whole document would let any future field
    opt itself out of the corruption screen by being named ``_observed``.
    """

    assert screen_module.exclusion_reason(
        "physics_mode.microphysics_scheme_observed", environmental) is None
    assert screen_module.exclusion_reason(
        "output.peak_bytes", environmental) is None


def test_the_registered_partition_is_still_honoured(screen_module,
                                                    environmental):
    """``ENVIRONMENTAL_FIELDS`` is the tree's list; the screen defers."""

    assert screen_module.exclusion_reason("wall_seconds", environmental) == (
        "registered ENVIRONMENTAL_FIELDS")


def _arm(root: Path, *, frame: bytes | None = b"history-bytes" * 8,
         receipt: str | None = None,
         receipt_name: str = "evidence/run-receipt.json") -> Path:
    """One run directory, built to order.

    Deliberately lower-level than :func:`_pair`: the refusal battery
    needs arms that are individually broken -- a missing receipt, an
    unreadable one, no history at all -- which is precisely what
    :func:`_pair` is built never to produce.
    """

    root.mkdir(parents=True, exist_ok=True)
    if frame is not None:
        (root / "wrfout").mkdir(parents=True, exist_ok=True)
        (root / "wrfout" / "wrfout_d01_2021-12-11_00_00_00").write_bytes(frame)
    if receipt is not None:
        path = root / receipt_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(receipt, encoding="utf-8")
    return root


def _good_receipt() -> str:
    return json.dumps(_receipt(), indent=1)


# --------------------------------------------------------------------------
# the verdict is derived from evidence, not set
# --------------------------------------------------------------------------


def test_pass_is_unreachable_without_evidence(screen_module):
    """The structural claim, tested directly on the verdict function.

    Everything below builds directories to prove the screen refuses in
    practice.  This proves it cannot do otherwise in principle: an
    :class:`Evidence` that counted nothing has no ``PASS`` available to
    it, because ``PASS`` is derived from the counters rather than
    assigned by whichever branch ran last.
    """

    empty = screen_module.Evidence()
    assert empty.verdict == screen_module.REFUSED
    assert empty.blocking_reasons

    # ... and one counter at zero is enough to keep it unreachable.
    for missing in ("frames_compared", "frame_bytes_hashed",
                    "receipt_digests_compared"):
        counts = {"frames_compared": 1, "frame_bytes_hashed": 1,
                  "receipt_digests_compared": 1}
        counts[missing] = 0
        assert screen_module.Evidence(**counts).verdict == (
            screen_module.REFUSED), missing

    assert screen_module.Evidence(
        frames_compared=1, frame_bytes_hashed=1,
        receipt_digests_compared=1).verdict == screen_module.PASS


def test_a_real_finding_outranks_a_thin_comparison(screen_module):
    """A mismatch is a finding even when the rest of the screen is thin.

    Downgrading an observed byte difference to "could not evaluate"
    because some OTHER leg of the screen was empty would lose the one
    thing the screen exists to report.
    """

    assert screen_module.Evidence(
        mismatches=("wrfout/x",)).verdict == screen_module.MISMATCH


# --------------------------------------------------------------------------
# the refusal battery: seven arrangements that used to print PASS
# --------------------------------------------------------------------------


def test_no_receipts_at_all_refuses(screen_module, tmp_path):
    """Zero final-state digests compared is not a bit-identical run."""

    verdict, lines = screen_module.screen(
        _arm(tmp_path / "a"), _arm(tmp_path / "b"), REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.REFUSED, text
    assert "DUAL-RUN SCREEN: REFUSED -- NOT EVALUATED" in text
    assert "PASS (bit-identical)" not in text
    assert "final-state digest" in text


def test_an_unreadable_receipt_refuses(screen_module, tmp_path):
    """A receipt that will not parse is a symptom, not a thing to skip.

    The screen used to swallow ``JSONDecodeError`` and carry on, so a
    truncated or corrupted receipt -- exactly what a bad card produces --
    removed itself from the comparison and the screen greened.
    """

    verdict, lines = screen_module.screen(
        _arm(tmp_path / "a", receipt="{not json"),
        _arm(tmp_path / "b", receipt="{not json"), REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.REFUSED, text
    assert "could not be read" in text
    assert "evidence/run-receipt.json" in text


def test_a_receipt_only_one_arm_wrote_refuses(screen_module, tmp_path):
    """A crashed arm must suppress the verdict, not supply one.

    Standing project law on A/B arms: absent evidence of work suppresses
    every verdict field.  An arm that never wrote its receipt did not
    finish, and "the receipts we could pair all agreed" is vacuously
    true when zero receipts paired.
    """

    verdict, lines = screen_module.screen(
        _arm(tmp_path / "a", receipt=_good_receipt()),
        _arm(tmp_path / "b"), REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.REFUSED, text
    assert "only in A" in text


def test_receipts_under_different_names_refuse(screen_module, tmp_path):
    """Pairing is by relative path; nothing paired means nothing compared."""

    verdict, lines = screen_module.screen(
        _arm(tmp_path / "a", receipt=_good_receipt(),
             receipt_name="evidence/run-receipt.json"),
        _arm(tmp_path / "b", receipt=_good_receipt(),
             receipt_name="evidence/rerun-receipt.json"), REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.REFUSED, text


def test_zero_byte_history_files_refuse(screen_module, tmp_path):
    """Two empty files are byte-identical and prove nothing whatever."""

    verdict, lines = screen_module.screen(
        _arm(tmp_path / "a", frame=b"", receipt=_good_receipt()),
        _arm(tmp_path / "b", frame=b"", receipt=_good_receipt()), REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.REFUSED, text
    assert "history bytes" in text


def test_a_receipt_carrying_no_digest_refuses(screen_module, tmp_path):
    """Receipts that paired but carried no digest still compared none."""

    thin = json.dumps({"status": "PASS", "wall_seconds": 1.0})
    verdict, lines = screen_module.screen(
        _arm(tmp_path / "a", receipt=thin),
        _arm(tmp_path / "b", receipt=thin), REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.REFUSED, text
    assert "final-state digests compared: 0" in text


def test_the_same_directory_twice_refuses(screen_module, tmp_path):
    """One run compared with itself agrees with itself.

    The A/B law in its purest form: an exactly-zero delta between an arm
    and itself is not evidence that the card is healthy, it is evidence
    that the experiment never ran.  The screen could not previously tell
    it had been handed one directory twice.
    """

    arm = _arm(tmp_path / "a", receipt=_good_receipt())
    verdict, lines = screen_module.screen(arm, arm, REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.REFUSED, text
    assert "same directory" in text


def test_an_arm_nested_inside_the_other_refuses(screen_module, tmp_path):
    """``rglob`` from A would walk B's files as if they were A's."""

    outer = _arm(tmp_path / "a", receipt=_good_receipt())
    inner = _arm(outer / "arm-b", receipt=_good_receipt())
    verdict, lines = screen_module.screen(outer, inner, REPO_ROOT)

    assert verdict == screen_module.REFUSED, "\n".join(lines)


def test_an_arm_with_no_history_refuses_rather_than_crying_corruption(
        screen_module, tmp_path):
    """"Nothing to compare" is not "the card corrupted memory".

    This path already declined to pass, but it reported MISMATCH, whose
    documented meaning is a CORRUPTION FINDING and whose exit status
    stops a chain with the wrong reason.  The module docstring has always
    promised a distinct "could not be evaluated" outcome; this is it.
    """

    verdict, lines = screen_module.screen(
        _arm(tmp_path / "a", frame=None, receipt=_good_receipt()),
        _arm(tmp_path / "b", receipt=_good_receipt()), REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.REFUSED, text
    assert "CORRUPTION FINDING" not in text


# --------------------------------------------------------------------------
# the screen, end to end
# --------------------------------------------------------------------------


def test_healthy_pair_passes_and_shows_what_it_excluded(
        screen_module, tmp_path):
    """The positive control, and the false finding reproduced.

    A genuinely bit-identical pair must still PASS -- a refusal that
    fires on healthy work is just a differently-shaped lie -- and the
    six allocator high-water numbers must still be excluded out loud.
    """

    run_a, run_b = _pair(tmp_path, mutate_b=lambda doc: [
        _set(doc, key, _get(doc, key) + 4096)
        for key in SAMPLED_HIGH_WATER_KEYS])
    verdict, lines = screen_module.screen(run_a, run_b, REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.PASS, text
    assert "DUAL-RUN SCREEN: PASS (bit-identical)" in text
    assert "non-environmental scalar blocks differing: 0" in text
    # Excluded, and SHOWN.  Every one of the six, by name, with both
    # values -- a silent exclusion is the failure this half exists for.
    for key in SAMPLED_HIGH_WATER_KEYS:
        assert f"EXCLUDED [sampled memory high-water]: " \
               f"evidence/run-receipt.json:{key}" in text, key
    assert "sampled memory high-water 6" in text


def test_a_pass_states_how_much_it_compared(screen_module, tmp_path):
    """"Identical" is a claim with a size attached to it.

    The counts are the half of the closure nobody has to guess a floor
    for: a reader handed ``2 final-state digests`` can see a thin screen
    for themselves even when it cleared the minimum.
    """

    run_a, run_b = _pair(tmp_path)
    verdict, lines = screen_module.screen(run_a, run_b, REPO_ROOT)
    text = "\n".join(lines)
    assert verdict == screen_module.PASS, text

    assert "history bytes hashed: 104" in text
    assert "final-state digests compared: 2" in text
    verdict_line = [line for line in lines if line.startswith("DUAL-RUN")][0]
    assert "1 history file" in verdict_line
    assert "2 final-state digests" in verdict_line


def test_a_flipped_history_bit_still_fails(screen_module, tmp_path):
    """The negative control.  A screen that only passes proves nothing."""

    run_a, run_b = _pair(tmp_path, corrupt_a_bytes=True)
    verdict, lines = screen_module.screen(run_a, run_b, REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.MISMATCH
    assert "DUAL-RUN SCREEN: MISMATCH -- CORRUPTION FINDING" in text
    assert "DIFFERS: wrfout/wrfout_d01_2021-12-11_00_00_00" in text


def test_a_screened_memory_key_still_fails(screen_module, tmp_path):
    """The other direction: the block did not fall out of the screen."""

    run_a, run_b = _pair(tmp_path, mutate_b=lambda doc: _set(
        doc, "memory.preflight_alloc_estimate_bytes",
        doc["memory"]["preflight_alloc_estimate_bytes"] + 4096))
    verdict, lines = screen_module.screen(run_a, run_b, REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.MISMATCH
    assert "SCALAR DIFFERS: evidence/run-receipt.json:" \
           "memory.preflight_alloc_estimate_bytes" in text


def test_a_changed_final_state_digest_still_fails(screen_module, tmp_path):
    """Digests cover arrays the history files do not carry."""

    run_a, run_b = _pair(tmp_path, mutate_b=lambda doc: _set(
        doc, "final_state_digest.d01:U", "c" * 64))
    verdict, lines = screen_module.screen(run_a, run_b, REPO_ROOT)
    text = "\n".join(lines)

    assert verdict == screen_module.MISMATCH
    assert "DIGEST DIFFERS: evidence/run-receipt.json:" \
           "final_state_digest.d01:U" in text


# --------------------------------------------------------------------------
# the exit statuses the docstring promises
# --------------------------------------------------------------------------


def test_exit_statuses_are_the_documented_three(screen_module, tmp_path,
                                                capsys):
    """0 PASS, 1 MISMATCH, 2 the screen could not be evaluated.

    The third was unreachable: every non-PASS outcome, including "there
    was nothing to compare", left through the same ``return 1`` and
    presented as a corruption finding.
    """

    def run(run_a, run_b):
        return screen_module.main(["--run-a", str(run_a), "--run-b",
                                   str(run_b), "--repo", str(REPO_ROOT)])

    passing_a, passing_b = _pair(tmp_path / "ok")
    assert run(passing_a, passing_b) == 0

    bad_a, bad_b = _pair(tmp_path / "bad", corrupt_a_bytes=True)
    assert run(bad_a, bad_b) == 1

    assert run(_arm(tmp_path / "thin-a"), _arm(tmp_path / "thin-b")) == 2
    capsys.readouterr()


def test_a_refusal_is_written_to_the_out_file_too(screen_module, tmp_path,
                                                  capsys):
    """The artifact a reader finds later must carry the refusal.

    A receipt file that exists only on the passing path is how a refusal
    becomes invisible to everyone reading the campaign directory
    afterwards.
    """

    out = tmp_path / "receipts" / "dual_run_screen.txt"
    status = screen_module.main([
        "--run-a", str(_arm(tmp_path / "a")),
        "--run-b", str(_arm(tmp_path / "b")),
        "--repo", str(REPO_ROOT), "--out", str(out)])
    capsys.readouterr()

    assert status == 2
    assert "REFUSED -- NOT EVALUATED" in out.read_text(encoding="utf-8")
