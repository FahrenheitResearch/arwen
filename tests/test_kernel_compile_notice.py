"""The first-run kernel-compile notice fires cold and stays quiet warm.

The first GPU forecast on a machine pays ~100 s of NVRTC compilation
(measured on the first cross-architecture field run of the published
wheel) under a progress status that never mentioned it.  The notice
module answers "is that about to happen?" from CuPy's on-disk kernel
cache alone -- no CUDA import -- so these tests run everywhere.

The architecture half of this file pins the defect the 2026-08-16
pre-sim profiling audit measured: this box's cache held 7,164 sm_120
entries when its 5090 was swapped for a 3080, so the cache was not
empty, the notice never fired, and 51 s of sm_86 recompilation ran
silently inside step 1's wall clock.
"""

import json
import struct
import time
from pathlib import Path

from gpuwm.kernel_compile_notice import (
    ARCHITECTURE_MISSING,
    COLD_CACHE,
    COMPILING_STATUS,
    CUPY_CACHE_ENV,
    cupy_kernel_cache_dir,
    kernel_cache_is_cold,
    kernel_cache_state,
    kernel_compile_notice,
    scan_kernel_cache,
)


def _cubin(architecture: int) -> bytes:
    """One cache entry shaped exactly the way CuPy writes them.

    CuPy prefixes the compiled blob with the 40-character SHA1 of the
    blob (``cupy/cuda/compiler.py``), so the ELF starts at byte 40; the
    architecture lives in the second byte of the CUDA ELF's ``e_flags``
    (verified against a real 7,445-entry cache on the reference box:
    7,164 entries decoded sm_120 and 281 sm_86).
    """

    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = 2               # ELFCLASS64
    header[5] = 1               # ELFDATA2LSB
    header[6] = 1               # EV_CURRENT
    struct.pack_into("<H", header, 16, 2)        # e_type
    struct.pack_into("<H", header, 18, 190)      # EM_CUDA
    struct.pack_into("<I", header, 48, (architecture << 8) | 2)
    return b"0" * 40 + bytes(header)


def test_missing_cache_directory_is_cold_and_speaks(tmp_path):
    absent = tmp_path / "never-created"
    assert kernel_cache_is_cold(absent)
    notice = kernel_compile_notice(absent)
    assert notice is not None
    # The line must say what is happening and what to expect: this is
    # the sentence that stops "it hung" at minute two of a first run.
    assert "compiling GPU kernels" in notice
    assert "minute" in notice
    assert "cached" in notice


def test_empty_cache_directory_is_still_cold(tmp_path):
    empty = tmp_path / "kernel_cache"
    empty.mkdir()
    assert kernel_cache_is_cold(empty)
    assert kernel_compile_notice(empty) is not None


def test_directory_with_only_subdirectories_is_cold(tmp_path):
    nested = tmp_path / "kernel_cache"
    (nested / "not-a-cubin").mkdir(parents=True)
    assert kernel_cache_is_cold(nested)


def test_any_cached_entry_means_warm_and_silent(tmp_path):
    warm = tmp_path / "kernel_cache"
    warm.mkdir()
    # Content-hashed blob names are CuPy's business, not this check's:
    # any regular file counts.
    (warm / "0a1b2c3d4e5f.cubin").write_bytes(b"\x00")
    assert not kernel_cache_is_cold(warm)
    assert kernel_compile_notice(warm) is None


def test_cache_dir_resolution_honours_cupy_env(monkeypatch, tmp_path):
    override = tmp_path / "elsewhere"
    monkeypatch.setenv(CUPY_CACHE_ENV, str(override))
    assert cupy_kernel_cache_dir() == override
    # multi_run points every arm at its own cache; the default answer
    # must be CuPy's documented one when the variable is absent.
    monkeypatch.delenv(CUPY_CACHE_ENV, raising=False)
    resolved = cupy_kernel_cache_dir()
    assert resolved.name == "kernel_cache"
    assert resolved.parent.name == ".cupy"


def test_default_resolution_feeds_the_no_argument_call(monkeypatch,
                                                       tmp_path):
    cache = tmp_path / "cupy-cache"
    monkeypatch.setenv(CUPY_CACHE_ENV, str(cache))
    assert kernel_compile_notice() is not None
    cache.mkdir()
    (cache / "kernel.cubin").write_bytes(b"\x00")
    assert kernel_compile_notice() is None


def test_the_status_token_is_a_single_progress_word():
    # gpuwm go prints the token verbatim in heartbeat lines between
    # RESTORING_PREPARED_CACHE and RUNNING; it must look like its
    # neighbours (one SHOUTING_SNAKE word, no spaces to wrap).
    assert COMPILING_STATUS == "COMPILING_GPU_KERNELS"


# ---------------------------------------------------------------------------
# The measured defect: a warm cache for the WRONG card
# ---------------------------------------------------------------------------


def test_a_cache_full_of_another_cards_kernels_still_announces(tmp_path):
    """THE AUDIT'S CASE.  7,164 sm_120 entries, an sm_86 card, 51 s of
    silence inside step 1.

    The cache is keyed by architecture, so not one of those entries can
    be loaded by the new card and every kernel is compiled again.  The
    old predicate saw files and said "warm"."""

    cache = tmp_path / "kernel_cache"
    cache.mkdir()
    for index in range(4):
        (cache / f"{index:040x}.cubin").write_bytes(_cubin(120))

    state = kernel_cache_state(cache, compute_capability="86")
    assert state.reason == ARCHITECTURE_MISSING
    assert state.entries == 4
    assert state.entries_for_capability == 0
    notice = kernel_compile_notice(cache, compute_capability="86")
    assert notice is not None
    # It must name WHY, or a reader who has seen the notice before on a
    # first run reads a second one as a bug.
    assert "sm_86" in notice
    assert "compiling GPU kernels" in notice


def test_an_entry_for_this_card_is_warmth(tmp_path):
    cache = tmp_path / "kernel_cache"
    cache.mkdir()
    (cache / ("a" * 40 + ".cubin")).write_bytes(_cubin(120))
    (cache / ("b" * 40 + ".cubin")).write_bytes(_cubin(86))

    state = kernel_cache_state(cache, compute_capability="86")
    assert state.reason is None
    assert state.entries_for_capability == 1
    assert kernel_compile_notice(cache, compute_capability="86") is None


def test_an_entry_that_cannot_be_decoded_is_never_used_to_announce(tmp_path):
    """Unknown is not absent.

    A PTX entry, a truncated blob, a future CuPy layout: none of those
    is evidence that this card's kernels are missing, and announcing a
    two-minute compile on the strength of a file we could not read is
    the false positive that would teach a reader to ignore the line."""

    cache = tmp_path / "kernel_cache"
    cache.mkdir()
    (cache / ("c" * 40 + ".cubin")).write_bytes(_cubin(120))
    (cache / ("d" * 40 + ".cubin")).write_bytes(b"\x00" * 8)

    state = kernel_cache_state(cache, compute_capability="86")
    assert state.reason is None
    assert state.undecodable == 1
    assert kernel_compile_notice(cache, compute_capability="86") is None


def test_an_unknown_capability_falls_back_to_the_cold_test(tmp_path):
    cache = tmp_path / "kernel_cache"
    cache.mkdir()
    (cache / ("e" * 40 + ".cubin")).write_bytes(_cubin(120))
    # No card to ask: the honest answer is the one this module could
    # always give, and it must not become a guess.
    assert kernel_cache_state(cache, compute_capability=None).reason is None
    assert kernel_compile_notice(cache) is None


def test_an_empty_cache_is_cold_whatever_the_card_is(tmp_path):
    empty = tmp_path / "kernel_cache"
    empty.mkdir()
    state = kernel_cache_state(empty, compute_capability="86")
    assert state.reason == COLD_CACHE
    assert kernel_cache_is_cold(empty)


def test_the_reading_is_taken_before_the_run_fills_the_cache(tmp_path):
    """THE SECOND HALF OF THE SAME DEFECT, and it was measured too.

    With the architecture test in place but asked at the announcement,
    the reference box STILL said nothing on a staged card swap: 200
    sm_120 entries against an sm_86 card, and by the time the runner
    asked, 160 fresh sm_86 entries of its own were already in the
    directory.  The question is "was this cache usable when the run
    started", so the census has to come from when the run started.
    """

    cache = tmp_path / "kernel_cache"
    cache.mkdir()
    for index in range(4):
        (cache / f"{index:040x}.cubin").write_bytes(_cubin(120))

    # Taken at launch: unusable, and it says so.
    census = scan_kernel_cache(cache)
    assert kernel_cache_state(
        cache, compute_capability="86", census=census).reason == \
        ARCHITECTURE_MISSING

    # The run then compiles its own kernels into the same directory.
    for index in range(4, 12):
        (cache / f"{index:040x}.cubin").write_bytes(_cubin(86))

    # Asked NOW, the directory is warm -- which is true and is the
    # wrong question.  Asked against the launch census, the answer is
    # still the one the reader needed.
    assert kernel_cache_state(cache, compute_capability="86").reason is None
    assert kernel_cache_state(
        cache, compute_capability="86", census=census).reason == \
        ARCHITECTURE_MISSING


def test_a_stalled_first_step_announces_itself_and_a_fast_one_does_not(
        tmp_path, capsys):
    """THE THIRD AND LAST HALF.  Prediction is not enough, so the run
    also MEASURES.

    Predicting from the cache fails in the real chain for a reason that
    is nobody's mistake: `gpuwm go`'s preprocessing stage uses the GPU
    too, so by the time the forecast runner starts, the cache already
    holds entries for this card and reads as warm -- while every kernel
    the FORECAST needs is still missing.  Reproduced end to end on the
    reference box: 200 sm_120 entries staged against an sm_86 card, the
    census clean, and 52 s of compilation inside model step 1.

    So the last line of defence is the stall itself, which cannot be
    wrong about whether a reader is waiting.
    """

    import types

    from gpuwm.prepared_single_domain_forecast import _FirstStepStallWatch

    class _Log:
        def __init__(self):
            self.announced = None

        def announce_kernel_compile(self, **fields):
            self.announced = fields

    def _watch(log, census):
        return _FirstStepStallWatch(
            progress_path=tmp_path / "progress.json",
            inputs=types.SimpleNamespace(source="gfs"),
            exp=types.SimpleNamespace(run_seconds=21600.0),
            step_log=log, census=census, delay=0.05)

    # A stall: it says so, names what the cache looked like at launch,
    # and publishes the status `gpuwm go`'s heartbeat relays verbatim.
    log = _Log()
    watch = _watch(log, (200, 0, {"120": 200}))
    watch.arm()
    time.sleep(0.4)
    printed = capsys.readouterr().out
    assert "model step 1 has been running" in printed
    assert "compile" in printed
    assert log.announced is not None
    assert log.announced["cached_entries"] == 200
    published = json.loads(
        (tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert published["status"] == COMPILING_STATUS

    # A fast first step: the observer disarms it and nothing is said.
    quiet = _Log()
    fast = _watch(quiet, (200, 0, {"120": 200}))
    fast.arm()
    fast.observe(grid_id=1, step_count=1, model_seconds=6.0,
                 step_wall_seconds=1.2)
    time.sleep(0.3)
    assert quiet.announced is None
    assert "model step 1 has been running" not in capsys.readouterr().out


def test_the_stall_watch_still_calls_the_observer_it_wraps():
    """It chains in front of the step log; it must not replace it."""

    import types

    from gpuwm.prepared_single_domain_forecast import _FirstStepStallWatch

    seen = []
    watch = _FirstStepStallWatch(
        progress_path=Path("unused"),
        inputs=types.SimpleNamespace(source="gfs"),
        exp=types.SimpleNamespace(run_seconds=1.0),
        step_log=None, census=(0, 0, {}), delay=999.0)
    wrapped = watch.wrap(lambda **event: seen.append(event))
    wrapped(grid_id=1, step_count=1, model_seconds=6.0, step_wall_seconds=0.1)
    assert seen == [{"grid_id": 1, "step_count": 1, "model_seconds": 6.0,
                     "step_wall_seconds": 0.1}]
    # ... and a run with the log switched off still gets a disarm.
    assert watch.wrap(None) == watch.observe


def test_the_state_names_the_capability_it_judged_against(tmp_path):
    cache = tmp_path / "kernel_cache"
    cache.mkdir()
    (cache / ("f" * 40 + ".cubin")).write_bytes(_cubin(120))
    state = kernel_cache_state(cache, compute_capability="86")
    # The receipt has to carry it: "the compile was announced" is not a
    # fact a later reader can check without knowing which card asked.
    assert state.compute_capability == "86"
    assert state.architectures == {"120": 1}
