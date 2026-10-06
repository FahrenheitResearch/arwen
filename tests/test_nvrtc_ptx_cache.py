"""Cache reuse cannot substitute another source, compiler or option set."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from conftest import requires_gpu
from gpuwm import nvrtc_ptx_cache as cache


class Compiler:
    def __init__(self, directory):
        self.directory = directory
        self.calls = []

    def get_cache_dir(self):
        return str(self.directory)

    def compile_using_nvrtc(self, source, options, arch, filename, **kwargs):
        self.calls.append((source, options, arch, filename, kwargs))
        return (source + repr(options)).encode(), None


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    compiler = Compiler(tmp_path / "cache")
    platform = {"arch": "120", "nvrtc_banner": "V13.4.92 CL-367"}
    monkeypatch.setattr(cache, "_compiler_runtime", lambda: (None, compiler))

    def identity(cupy, owner, source, options, arch, filename):
        return {"source": source, "options": list(options), "filename": filename,
                **platform, "arch": platform["arch"] if arch is None else arch}

    monkeypatch.setattr(cache, "_cache_identity", identity)
    # Runtime, compiler and identity are all host-only test doubles.
    # Keep the actual process ban and CUDA visibility unchanged.
    monkeypatch.setattr("gpuwm.local_gpu.no_local_gpu", lambda: False)
    monkeypatch.delenv("CUPY_CACHE_IN_MEMORY", raising=False)
    cache.cache_events(clear=True)
    return compiler, platform


def compile_unit():
    return cache.compile_using_nvrtc("unit", ("-std=c++17", "--ftz=false"),
                                     None, "unit.cu")


def test_repeat_reuses_exact_image_and_original_arguments(runtime):
    compiler, _ = runtime
    cold = compile_unit()
    warm = compile_unit()
    assert cold == warm
    assert compiler.calls == [("unit", ("-std=c++17", "--ftz=false"),
                               None, "unit.cu", {})]
    events = cache.cache_events()
    assert [row["status"] for row in events] == ["miss", "hit"]
    assert events[0]["cache_written"]
    assert events[1]["compile_seconds"] == 0.0


@pytest.mark.parametrize("options", [(), ("-Iexternal",)])
@pytest.mark.parametrize("arch", [None, "120"])
def test_local_gpu_ban_refuses_before_runtime_probe_or_cache(
        options, arch, monkeypatch):
    from gpuwm.local_gpu import NO_LOCAL_GPU_ENV

    monkeypatch.setenv(NO_LOCAL_GPU_ENV, "1")
    before = cache.cache_events()
    def forbidden(*args, **kwargs):
        raise AssertionError("local NVRTC runtime or cache boundary was reached")
    for name in ("_compiler_runtime", "_cache_identity", "_compile_environment",
                 "_cache_directory", "_read_entry", "_write_entry",
                 "_external_dependencies"):
        monkeypatch.setattr(cache, name, forbidden)
    with pytest.raises(RuntimeError, match=NO_LOCAL_GPU_ENV):
        cache.compile_using_nvrtc("unit", options, arch)
    assert cache.cache_events() == before


def test_private_getter_api_reuses_exact_image(runtime, monkeypatch):
    compiler, _ = runtime
    monkeypatch.setattr(compiler, "get_cache_dir", None)
    monkeypatch.setattr(compiler, "_get_cache_dir", lambda: str(compiler.directory), raising=False)
    assert compile_unit() == compile_unit()
    assert len(compiler.calls) == 1
    assert [row["status"] for row in cache.cache_events()] == ["miss", "hit"]


def test_cupy_14_active_disk_backend_reuses_exact_image(runtime, monkeypatch, tmp_path):
    compiler, _ = runtime
    class DiskBackend:
        def __init__(self, directory):
            self._cache_dir = str(directory)
    monkeypatch.setattr(compiler, "get_cache_dir", None)
    monkeypatch.setattr(compiler, "_DiskKernelCacheBackend", DiskBackend, raising=False)
    monkeypatch.setattr(compiler, "_kernel_cache_backend", DiskBackend(compiler.directory), raising=False)
    # An already-selected backend retains its programmatic directory when
    # the environment later changes, exactly as CuPy 14.2 does.
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "later-environment"))
    assert cache._cache_directory(compiler) == compiler.directory
    assert compile_unit() == compile_unit()
    assert len(compiler.calls) == 1
    assert [row["status"] for row in cache.cache_events()] == ["miss", "hit"]
    assert next((compiler.directory / "gpuwm-direct-nvrtc").glob("*.image")).is_file()
    assert not (tmp_path / "later-environment").exists()


def test_unknown_storage_backend_bypasses_without_a_default(runtime, monkeypatch):
    compiler, _ = runtime
    class DiskBackend:
        pass
    monkeypatch.setattr(compiler, "get_cache_dir", None)
    monkeypatch.setattr(compiler, "_DiskKernelCacheBackend", DiskBackend, raising=False)
    monkeypatch.setattr(compiler, "_kernel_cache_backend", SimpleNamespace(_cache_dir=str(compiler.directory)), raising=False)
    assert compile_unit() == compile_unit()
    assert len(compiler.calls) == 2
    assert all(row["status"] == "bypass" for row in cache.cache_events())
    assert not compiler.directory.exists()


@pytest.mark.parametrize("change", ["source", "options", "arch", "compiler",
                                    "filename"])
def test_changed_compile_identity_is_a_miss(runtime, change):
    compiler, platform = runtime
    compile_unit()
    source, options, arch, filename = "unit", ("-std=c++17", "--ftz=false"), None, "unit.cu"
    if change == "source":
        source += " changed"
    elif change == "options":
        options = ("-std=c++17", "--ftz=true")
    elif change == "arch":
        platform["arch"] = "89"
    elif change == "compiler":
        platform["nvrtc_banner"] = "V13.4.93 CL-368"
    elif change == "filename":
        filename = "other.cu"
    cache.compile_using_nvrtc(source, options, arch, filename)
    assert len(compiler.calls) == 2
    assert cache.cache_events()[-1]["status"] == "miss"


@pytest.mark.parametrize("corruption", ["payload", "truncated", "magic", "header"])
def test_corrupt_entry_compiles_and_replaces_it(runtime, corruption):
    compiler, _ = runtime
    expected = compile_unit()
    entry = next((compiler.directory / "gpuwm-direct-nvrtc").glob("*.image"))
    blob = entry.read_bytes()
    if corruption == "payload":
        broken = blob[:-1] + bytes([blob[-1] ^ 1])
    elif corruption == "truncated":
        broken = blob[:len(cache._MAGIC) + 2]
    elif corruption == "magic":
        broken = b"x" + blob[1:]
    else:
        broken = blob.replace(b'"identity"', b'"bad_____"', 1)
    entry.write_bytes(broken)
    assert compile_unit() == expected
    assert len(compiler.calls) == 2
    assert compile_unit() == expected
    assert len(compiler.calls) == 2


def test_unwritable_directory_preserves_original_compile(runtime):
    compiler, _ = runtime
    compiler.directory.write_text("a file blocks the cache directory")
    assert compile_unit() == compile_unit()
    assert len(compiler.calls) == 2
    assert not any(row["cache_written"] for row in cache.cache_events())


def test_unresolved_identity_preserves_original_compile(runtime, monkeypatch):
    compiler, _ = runtime
    def unresolved(*args):
        raise RuntimeError("unavailable compiler identity")
    monkeypatch.setattr(cache, "_cache_identity", unresolved)
    assert compile_unit() == compile_unit()
    assert len(compiler.calls) == 2
    assert all(row["status"] == "bypass" for row in cache.cache_events())


@pytest.mark.parametrize("source", ["#include <external.h>",
                                     "# include <external.h>",
                                     "\t#\tinclude \"external.h\"",
                                     "# \\\n include <external.h>",
                                     "#\n\tinclude <external.h>",
                                     "# /* comment */ include <external.h>",
                                     "%: include <external.h>",
                                     "#include_next <external.h>",
                                     "__FILE__", "__DATE__", "__TIME__"])
def test_external_or_time_dependent_source_bypasses(runtime, source):
    compiler, _ = runtime
    cache.compile_using_nvrtc(source)
    cache.compile_using_nvrtc(source)
    assert len(compiler.calls) == 2
    assert all(row["status"] == "bypass" for row in cache.cache_events())


@pytest.mark.parametrize("options", [("--pre-include=external.h",),
                                      ("--pre-include", "external.h"),
                                      ("-include", "external.h"),
                                      ("-includeexternal.h",),
                                      ("--include-path=headers",),
                                      ("--include-path", "headers"),
                                      ("-Iheaders",), ("-I", "headers"),
                                      ("-imacros", "external.h"),
                                      ("-isystem", "headers")])
def test_external_dependency_options_bypass_cache(runtime, options):
    compiler, _ = runtime
    cache.compile_using_nvrtc("unit", options)
    cache.compile_using_nvrtc("unit", options)
    assert len(compiler.calls) == 2
    assert compiler.calls[0][1] == options
    assert not compiler.directory.exists()
    assert all(row["status"] == "bypass" and row["key"] is None
               for row in cache.cache_events())


def test_explicit_memory_only_cache_is_respected(runtime, monkeypatch):
    compiler, _ = runtime
    monkeypatch.setenv("CUPY_CACHE_IN_MEMORY", "1")
    compile_unit()
    assert not compiler.directory.exists()
    assert cache.cache_events()[-1]["status"] == "bypass"


def test_concurrent_atomic_writes_remain_readable(runtime):
    compiler, _ = runtime
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda _: compile_unit(), range(16)))
    assert all(output == outputs[0] for output in outputs)
    calls = len(compiler.calls)
    assert compile_unit() == outputs[0]
    assert len(compiler.calls) == calls
    directory = compiler.directory / "gpuwm-direct-nvrtc"
    assert len(list(directory.iterdir())) == 1


def test_full_toolchain_identity_keys_patch_build_and_effective_arch(tmp_path,
                                                                  monkeypatch):
    compiler_file = tmp_path / "compiler.py"
    compiler_file.write_text("compiler implementation")
    compiler = SimpleNamespace(
        __file__=str(compiler_file), _get_arch=lambda: "120",
        _get_nvrtc_version=lambda: (13, 4),
        _get_arch_for_options_for_nvrtc=lambda arch: (f"-arch=sm_{arch}", "cubin"),
        _get_cupy_cache_key=lambda: "cupy-key",
        _preprocess=lambda *args: "// Compiler Build ID: CL-367\n// V13.4.92")
    monkeypatch.setattr(cache, "_compile_environment", cache._compile_environment.__wrapped__)
    cupy = SimpleNamespace(__version__="14.2.0", cuda=SimpleNamespace(
        runtime=SimpleNamespace(driverGetVersion=lambda: 13020)))
    first = cache._cache_identity(cupy, compiler, "unit", ("--ftz=false",), None, "unit.cu")
    compiler._preprocess = lambda *args: "// Compiler Build ID: CL-368\n// V13.4.93"
    second = cache._cache_identity(cupy, compiler, "unit", ("--ftz=false",), None, "unit.cu")
    assert first["target"] == ["-arch=sm_120", "cubin"]
    assert first["options"] == ["--ftz=false"]
    assert cache._entry_key(first) != cache._entry_key(second)


@pytest.mark.parametrize("image", [b"\x7fELF\x00\xff\x01", "// PTX text\n"])
def test_image_artifact_preserves_type_bytes_and_symbol_mapping(tmp_path, image):
    entry = tmp_path / "unit.image"
    identity = {"source_sha256": "a" * 64, "target": ["sm_120", "cubin"]}
    mapping = {"function<float>": "_Z8functionIfEvv"}
    assert cache._write_entry(entry, identity, image, mapping)
    assert cache._read_entry(entry, identity) == (image, mapping)
    assert cache._read_entry(entry, {**identity, "target": ["sm_89", "cubin"]}) is None


def test_warm_report_counts_direct_image_artifacts_recursively(runtime):
    from gpuwm.warm_kernels import _direct_cache_inventory, _direct_cache_counts
    compiler, _ = runtime
    compile_unit()
    compile_unit()
    inventory = _direct_cache_inventory(compiler.directory)
    assert inventory["image_files"] == 1
    assert inventory["bytes"] > 0
    nested = compiler.directory / "gpuwm-direct-nvrtc" / "nested"
    nested.mkdir()
    (nested / "another.image").write_bytes(b"artifact")
    assert _direct_cache_inventory(compiler.directory)["image_files"] == 2
    counts = _direct_cache_counts(cache.cache_events())
    assert counts["direct_nvrtc_images_compiled"] == 1
    assert counts["direct_nvrtc_cache_hits"] == 1


@requires_gpu
def test_native_image_cache_retains_subnormal_words_and_recovers_corruption(
        tmp_path, monkeypatch):
    import cupy as cp
    from cupy.cuda import compiler
    import numpy as np
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path))
    if hasattr(compiler, "_DiskKernelCacheBackend"):
        monkeypatch.setattr(compiler, "_kernel_cache_backend",
                            compiler._DiskKernelCacheBackend(str(tmp_path)))
    monkeypatch.delenv("CUPY_CACHE_IN_MEMORY", raising=False)
    cache.cache_events(clear=True)
    source = '''extern "C" __global__ void product(const float* x, float* y) {
        y[0] = __fmul_rn(x[0], x[1]);
    }'''
    operands = np.asarray([1.e-30, 1.e-10], dtype=np.float32)
    expected = np.asarray([operands[0] * operands[1]], dtype=np.float32)

    def compiled_result():
        image, mapping = cache.compile_using_nvrtc(
            source, ("-std=c++17", "--ftz=false"), None, "product.cu")
        assert mapping is None
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
        result = cp.empty(1, dtype=cp.float32)
        module.get_function("product")((1,), (1,), (cp.asarray(operands), result))
        return image, cp.asnumpy(result).view(np.uint32)

    cold_image, cold = compiled_result()
    warm_image, warm = compiled_result()
    assert cold_image == warm_image
    assert np.array_equal(cold, expected.view(np.uint32))
    assert np.array_equal(cold, warm)
    entry = next((tmp_path / "gpuwm-direct-nvrtc").glob("*.image"))
    blob = entry.read_bytes()
    entry.write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
    recovered_image, recovered = compiled_result()
    assert recovered_image == cold_image
    assert np.array_equal(recovered, cold)
    events = cache.cache_events()
    assert [row["status"] for row in events] == ["miss", "hit", "miss"]
    assert all(isinstance(row["key"], str) and len(row["key"]) == 64 for row in events)
    assert events[1]["compile_seconds"] == 0.0
