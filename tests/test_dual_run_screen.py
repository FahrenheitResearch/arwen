"""``tools/dual_run_screen.py`` classifies, and it still bites.

The screen is the standing corruption detector on a card with no ECC, so
it has two ways to be worthless and this file pins both.

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


# --------------------------------------------------------------------------
# the screen, end to end
# --------------------------------------------------------------------------


def test_healthy_pair_passes_and_shows_what_it_excluded(
        screen_module, tmp_path):
    """The false finding, reproduced: it must now PASS -- out loud."""

    run_a, run_b = _pair(tmp_path, mutate_b=lambda doc: [
        _set(doc, key, _get(doc, key) + 4096)
        for key in SAMPLED_HIGH_WATER_KEYS])
    ok, lines = screen_module.screen(run_a, run_b, REPO_ROOT)
    text = "\n".join(lines)

    assert ok, text
    assert "DUAL-RUN SCREEN: PASS (bit-identical)" in text
    assert "non-environmental scalar blocks differing: 0" in text
    # Excluded, and SHOWN.  Every one of the six, by name, with both
    # values -- a silent exclusion is the failure this half exists for.
    for key in SAMPLED_HIGH_WATER_KEYS:
        assert f"EXCLUDED [sampled memory high-water]: " \
               f"evidence/run-receipt.json:{key}" in text, key
    assert "sampled memory high-water 6" in text


def test_a_flipped_history_bit_still_fails(screen_module, tmp_path):
    """The negative control.  A screen that only passes proves nothing."""

    run_a, run_b = _pair(tmp_path, corrupt_a_bytes=True)
    ok, lines = screen_module.screen(run_a, run_b, REPO_ROOT)
    text = "\n".join(lines)

    assert not ok
    assert "DUAL-RUN SCREEN: MISMATCH -- CORRUPTION FINDING" in text
    assert "DIFFERS: wrfout/wrfout_d01_2021-12-11_00_00_00" in text


def test_a_screened_memory_key_still_fails(screen_module, tmp_path):
    """The other direction: the block did not fall out of the screen."""

    run_a, run_b = _pair(tmp_path, mutate_b=lambda doc: _set(
        doc, "memory.preflight_alloc_estimate_bytes",
        doc["memory"]["preflight_alloc_estimate_bytes"] + 4096))
    ok, lines = screen_module.screen(run_a, run_b, REPO_ROOT)
    text = "\n".join(lines)

    assert not ok
    assert "SCALAR DIFFERS: evidence/run-receipt.json:" \
           "memory.preflight_alloc_estimate_bytes" in text


def test_a_changed_final_state_digest_still_fails(screen_module, tmp_path):
    """Digests cover arrays the history files do not carry."""

    run_a, run_b = _pair(tmp_path, mutate_b=lambda doc: _set(
        doc, "final_state_digest.d01:U", "c" * 64))
    ok, lines = screen_module.screen(run_a, run_b, REPO_ROOT)
    text = "\n".join(lines)

    assert not ok
    assert "DIGEST DIFFERS: evidence/run-receipt.json:" \
           "final_state_digest.d01:U" in text
