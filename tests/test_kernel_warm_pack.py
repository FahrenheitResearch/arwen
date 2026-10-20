"""Persisted images must be exact, recoverable and safe across shape rebinding."""
import json
import pytest
from gpuwm.kernel_warm_pack import WarmImages


def test_exact_request_survives_process_and_changed_requests_miss(tmp_path):
    args = dict(source="s", options=["--fmad=false"], filename="a.cu", name_expressions=["k<float>"])
    result = b"\x7fELFactual-compiler-result", {"k<float>": "symbol"}
    pack = WarmImages(tmp_path, {"compiler": "cu12"}, "shape/options/strict", "120")
    pack.put(args, result)
    pack.close()
    second = WarmImages(tmp_path, {"compiler": "cu12"}, "shape/options/strict", "120")
    assert second.get(args) == result
    assert second.get(dict(args, options=["--fmad=true"])) is None
    assert second.get(dict(args, filename="b.cu")) is None
    assert WarmImages(tmp_path, {"compiler": "cu13"}, "shape/options/strict", "120").get(args) is None
    assert WarmImages(tmp_path, {"compiler": "cu12"}, "other-shape", "120").get(args) is None
    assert WarmImages(tmp_path, {"compiler": "cu12"}, "shape/options/strict", "89").get(args) is None
    # Returned mappings cannot mutate the retained symbol aliases.
    second.get(args)[1]["k<float>"] = "wrong"
    assert second.get(args) == result


def test_damaged_image_and_mapping_fall_back(tmp_path):
    args = dict(source="s")
    pack = WarmImages(tmp_path, {}, "shape", "120")
    pack.put(args, (b"\x7fELFresult", {"alias": "symbol"}))
    pack.close()
    document = json.loads(pack.index.read_text())
    row = next(iter(document["entries"].values()))
    (pack.root / (row["image"] + ".cubin")).write_bytes(b"wrong")
    with pytest.warns(RuntimeWarning):
        assert WarmImages(tmp_path, {}, "shape", "120").get(args) is None
    # Rebuild repairs the bad blob; a corrupted mapping must still miss.
    pack = WarmImages(tmp_path, {}, "shape", "120")
    pack.put(args, (b"\x7fELFresult", {"alias": "symbol"}))
    pack.close()
    row["mapping"]["alias"] = "wrong"
    pack.index.write_text(json.dumps(document))
    with pytest.warns(RuntimeWarning):
        assert WarmImages(tmp_path, {}, "shape", "120").get(args) is None


def test_parallel_publish_keeps_both_members(tmp_path):
    one = WarmImages(tmp_path, {}, "shape", "120")
    two = WarmImages(tmp_path, {}, "shape", "120")
    one.put({"source": "one"}, (b"\x7fELFone", None))
    two.put({"source": "two"}, (b"\x7fELFtwo", None))
    one.close()
    two.close()
    loaded = WarmImages(tmp_path, {}, "shape", "120")
    assert loaded.get({"source": "one"}) == (b"\x7fELFone", None)
    assert loaded.get({"source": "two"}) == (b"\x7fELFtwo", None)


def test_metadata_second_process_identity_options_and_corruption(tmp_path):
    from gpuwm.compiler_metadata import CompilerMetadata
    args = dict(kind="empty-preprocess", options=["--ftz=false"], arch="120")
    measured = []
    def measure():
        measured.append(1)
        return "// measured compiler build"
    cache = CompilerMetadata(tmp_path, {"library": "cu12-bytes"})
    assert cache.resolve(args, measure) == "// measured compiler build"
    second = CompilerMetadata(tmp_path, {"library": "cu12-bytes"})
    assert second.resolve(args, measure) == "// measured compiler build"
    assert len(measured) == 1
    CompilerMetadata(tmp_path, {"library": "cu13-bytes"}).resolve(args, measure)
    second.resolve(dict(args, options=["--ftz=true"]), measure)
    second.resolve(dict(args, arch="89"), measure)
    assert len(measured) == 4
    path = next(cache.root.glob("*.json"))
    row = json.loads(path.read_text());row['value'] = 'corrupted'
    path.write_text(json.dumps(row))
    with pytest.warns(RuntimeWarning):
        CompilerMetadata(tmp_path, {"library": "cu12-bytes"}).resolve(row['arguments'], measure)
    assert len(measured) == 5


def test_metadata_single_flight_and_retry(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from gpuwm.compiler_metadata import CompilerMetadata
    cache = CompilerMetadata(tmp_path, {})
    entered,release = threading.Event(),threading.Event()
    calls = []
    def measure():
        calls.append(1);entered.set();assert release.wait(5)
        return "// actual output"
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(cache.resolve, {}, measure)
        assert entered.wait(5)
        second = pool.submit(cache.resolve, {}, measure)
        release.set()
        assert first.result() == second.result() == "// actual output"
    assert calls == [1]
    def fail(): raise RuntimeError('compiler refused')
    with pytest.raises(RuntimeError, match='compiler refused'):
        cache.resolve({'retry': True}, fail)
    assert cache.resolve({'retry': True}, lambda: '// recovered') == '// recovered'


def test_metadata_bridge_leaves_nonempty_and_other_backend_compilation_intact(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gpuwm.kernel_warm_pack import install_metadata
    from gpuwm.certify import compile_platform
    calls = []
    banners = []
    def preprocess(*args):
        calls.append(args)
        return '// measured preprocess'
    def banner():
        banners.append(1)
        return '// measured banner'
    compiler = SimpleNamespace(_preprocess=preprocess)
    monkeypatch.setattr(compile_platform, 'nvrtc_banner', banner)
    monkeypatch.setattr('gpuwm.kernel_aot.effective_options', tuple)
    install_metadata(compiler, tmp_path, {'library': 'actual-digest'})
    assert compiler._preprocess('', ('-O3',), '120', 'nvrtc') == '// measured preprocess'
    assert compiler._preprocess('', ('-O3',), '120', 'nvrtc') == '// measured preprocess'
    compiler._preprocess('real source', ('-O3',), '120', 'nvrtc')
    compiler._preprocess('', ('-O3',), '120', 'nvcc')
    assert len(calls) == 3
    # install_metadata only wraps CuPy's empty preprocess route.
    assert compile_platform.nvrtc_banner is banner


def test_banner_persists_measured_bytes_and_refuses_unresolved_identity(tmp_path):
    from types import SimpleNamespace
    from gpuwm.compiler_metadata import nvrtc_banner
    calls = []
    nvrtc = SimpleNamespace(createProgram=lambda *a: 1,
        compileProgram=lambda *a: calls.append(a), destroyProgram=lambda *a: None,
        getPTX=lambda *a: b'// Compiler Build ID: CL-123\n// Cuda compilation tools, release 13.4, V13.4.92\n.version 8.8\n')
    library = dict(status='resolved', sha256='loaded-compiler-bytes')
    first = nvrtc_banner(nvrtc, library, tmp_path)
    assert first == nvrtc_banner(nvrtc, library, tmp_path)
    assert len(calls) == 1 and '.version' not in first
    nvrtc_banner(nvrtc, dict(library, sha256='changed-compiler'), tmp_path)
    nvrtc_banner(nvrtc, dict(status='unavailable'), tmp_path)
    nvrtc_banner(nvrtc, dict(status='unavailable'), tmp_path)
    assert len(calls) == 4


def test_literal_and_nested_headers_bind_actual_bytes(tmp_path):
    header = tmp_path / 'one.h'
    nested = tmp_path / 'two.h'
    header.write_text('#include "two.h"\n')
    nested.write_text('#define VALUE 1\n')
    args = dict(source='#include "one.h"\n', options=['-I'+str(tmp_path)])
    one = WarmImages(tmp_path / 'packs', {}, 'shape', '120')
    one.put(args, (b'\x7fELFheader-one', None));one.close()
    assert WarmImages(tmp_path / 'packs', {}, 'shape', '120').get(args)[0] == b'\x7fELFheader-one'
    nested.write_text('#define VALUE 2\n')
    assert WarmImages(tmp_path / 'packs', {}, 'shape', '120').get(args) is None


def test_macro_header_refuses_old_image_and_storing_unbound_result(tmp_path):
    args = dict(source='#define H "owned.h"\n#include H\n', options=['-I'+str(tmp_path)])
    pack = WarmImages(tmp_path, {}, 'shape', '120')
    with pytest.warns(RuntimeWarning, match='dependency is unbound'):
        pack.put(args, (b'\x7fELFstale-header', None))
    assert pack.get(args) is None and not pack.rows


def test_header_parser_does_not_hide_directive_after_comment_like_string(tmp_path):
    from gpuwm.kernel_dependencies import header_dependencies
    header = tmp_path / 'owned.h';header.write_text('#define VALUE 1\n')
    source = 'const char* a="/*";\n/* comment */ #include "owned.h"\nconst char* b="*/";'
    assert header_dependencies(source, ['-I'+str(tmp_path)])[0]['path'] == str(header)


def test_forced_include_metadata_uses_original_probe(tmp_path):
    from types import SimpleNamespace
    from gpuwm.kernel_warm_pack import install_metadata
    calls = []
    compiler = SimpleNamespace(_preprocess=lambda *args: calls.append(args) or 'actual')
    install_metadata(compiler, tmp_path, {})
    for _ in range(2):
        with pytest.warns(RuntimeWarning, match='forced-include'):
            compiler._preprocess('', ('--pre-include=owned.h',), '120', 'nvrtc')
    assert len(calls) == 2


def test_header_comment_replacement_preserves_directive_boundaries(tmp_path):
    from gpuwm.kernel_dependencies import header_dependencies
    header = tmp_path / 'owned.h'
    header.write_text('#define VALUE 1\n')
    options = ['-I' + str(tmp_path)]
    source = 'const char* s="// /* \\"quoted\\" */";\n/*before\n comment*/ #include /*between*/ "owned.h" //after\n'
    assert header_dependencies(source, options)[0]['path'] == str(header)
    # A null directive cannot consume the following non-directive line.
    assert header_dependencies('#\ninclude "missing.h"\n', options) == []
    with pytest.raises(ValueError, match='macro or nonliteral'):
        header_dependencies('/* a\n b */ #include NAME\n', options)
