"""The first-run kernel-compile notice fires cold and stays quiet warm.

The first GPU forecast on a machine pays ~100 s of NVRTC compilation
(measured on the first cross-architecture field run of the published
wheel) under a progress status that never mentioned it.  The notice
module answers "is that about to happen?" from CuPy's on-disk kernel
cache alone -- no CUDA import -- so these tests run everywhere.
"""

from gpuwm.kernel_compile_notice import (
    COMPILING_STATUS,
    CUPY_CACHE_ENV,
    cupy_kernel_cache_dir,
    kernel_cache_is_cold,
    kernel_compile_notice,
)


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
