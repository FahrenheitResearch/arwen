"""Real cubin load, template alias preservation, and source-change fallback."""
import pytest

pytestmark = pytest.mark.gpu


def test_real_cupy_bridge_preserves_template_name_mapping(tmp_path, monkeypatch):
    import cupy as cp
    import numpy as np
    from cupy.cuda import compiler
    write = pytest.importorskip("gpuwm.kernel_pack").write
    from gpuwm.kernel_aot import Pack, capture_requests, compiler_identity, install, recipe
    from gpuwm.kernel_aot_build import compile_entry

    source = 'typedef unsigned int word; template<typename T> __global__ void aot_copy_template(const T* a, T* b) { b[threadIdx.x] = a[threadIdx.x]; }'
    names = ["aot_copy_template<unsigned int>"]
    options = ("--std=c++17", "--fmad=false")
    with capture_requests() as requests:
        cp.RawModule(code=source, options=options, name_expressions=names).get_function(names[0])
        cp.RawModule(code=source, options=options, name_expressions=["aot_copy_template<word>"]).get_function("aot_copy_template<word>")
    captured = list(requests.values())
    assert len(captured) == 2
    cp.clear_memo()
    row = recipe(captured[0])
    entry = compile_entry(row, "sm_" + cp.cuda.Device().compute_capability)
    alias_names = ["aot_copy_template<word>"]
    alias = compile_entry(recipe(captured[1]),
                          "sm_" + cp.cuda.Device().compute_capability)
    path = tmp_path / "template-pack.json"
    write(path, compiler_identity(), [entry, alias])
    original = compiler._compile_with_cache_cuda
    monkeypatch.setattr(compiler, "_compile_with_cache_cuda", original)
    monkeypatch.setattr(compiler, "compile_using_nvrtc", compiler.compile_using_nvrtc)
    install(cp, Pack(path))
    calls = []
    compile_original = compiler._compile_using_nvrtc_no_warning
    def count(text, *a, **kw):
        if text.startswith(source):
            calls.append(text)
        return compile_original(text, *a, **kw)
    monkeypatch.setattr(compiler, "_compile_using_nvrtc_no_warning", count)
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "empty-cupy"))
    results = []
    words = np.array([1, 0x80000001, 0x7fc00001], dtype=np.uint32)
    first = cp.RawModule(code=source, options=options, name_expressions=names)
    first.get_function(names[0])
    second = cp.RawModule(code=source, options=options, name_expressions=alias_names)
    second.get_function(alias_names[0])
    first.get_function(names[0])
    assert calls == []
    for text in (source, source + "\n// changed template source"):
        module = cp.RawModule(code=text, options=options, name_expressions=names)
        kernel = module.get_function(names[0])
        out = cp.empty(words.shape, dtype=cp.uint32)
        kernel((1,), (len(words),), (cp.asarray(words), out))
        results.append(out.get().tobytes())
        if text == source:
            assert calls == []
    assert calls == [source + "\n// changed template source"]
    assert results == [words.tobytes()] * 2


def test_direct_pack_keeps_both_numerical_modes_when_builder_defaults_fast(tmp_path, monkeypatch):
    import cupy as cp
    import numpy as np
    from cupy.cuda import compiler
    from gpuwm import kernel_aot as aot, numerics
    from gpuwm.kernel_aot_build import compile_entry
    from gpuwm.kernel_pack import write

    source = ('extern "C" __global__ void arithmetic(const float* a, float* b) '
              '{ b[0] = a[0]*a[1] + a[2]; }')
    monkeypatch.setattr(numerics, "_mode", "fast")
    rows = []
    for options in (("--std=c++17", "--fmad=false"), ("--std=c++17",)):
        request = dict(source=source, options=options, route="direct-nvrtc", filename="arithmetic.cu")
        rows.append(compile_entry(aot.recipe(request), "sm_" + cp.cuda.Device().compute_capability))
    path = tmp_path / "modes.json"
    write(path, aot.compiler_identity(), rows)
    selected = aot.Pack(path)
    original = compiler.compile_using_nvrtc
    monkeypatch.setattr(compiler, "compile_using_nvrtc", original)
    monkeypatch.setattr(compiler, "_compile_with_cache_cuda", compiler._compile_with_cache_cuda)
    aot.install(cp, selected)

    words = []
    a = cp.asarray(np.array([1 + 2**-23, 1 + 2**-23, -(1 + 2**-22)], dtype=np.float32))
    output = cp.empty((1,), dtype=cp.float32)
    for mode in ("strict", "fast"):
        monkeypatch.setattr(numerics, "_mode", mode)
        image, _ = compiler.compile_using_nvrtc(
            source, options=("--std=c++17", "--fmad=false"), filename="arithmetic.cu")
        module = cp.cuda.function.Module()
        module.load(image)
        kernel = module.get_function("arithmetic")
        kernel((1,), (1,), (a, output))
        words.append(int(output.get().view(np.uint32)[0]))
    assert words == [0, 0x28800000]  # strict rounds the product first; fast contracts it.
