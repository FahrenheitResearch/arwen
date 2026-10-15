"""A new compiler site joins the argument audit without a count edit."""
import pytest

import test_kernel_manifest as manifest
import test_kernel_loader_inert as loader_identity


@pytest.mark.parametrize("compile_call", [
    "cp.RawModule(code=source, options=options)",
    "compile_using_nvrtc(source, options=options)",
])
def test_new_recorded_compile_site_is_discovered(tmp_path, monkeypatch, compile_call):
    root = tmp_path / "gpuwm/core"
    root.mkdir(parents=True)
    path = root / "future.py"
    path.write_text("def compile():\n    source = 'source'\n    options = ()\n"
                    f"    module = {compile_call}\n"
                    "    record_module(module, source=source, options=options)\n")
    monkeypatch.setattr(manifest, "REPO", tmp_path)
    assert manifest._discover_manifest_site_files(()) == ("gpuwm/core/future.py",)
    assert manifest.audit_source("gpuwm/core/future.py", path.read_text()) == []
    path.write_text(path.read_text().replace("source=source", "source='wrong source'"))
    assert manifest.audit_source("gpuwm/core/future.py", path.read_text())


def test_new_raw_constructor_without_a_record_still_fails(tmp_path, monkeypatch):
    root = tmp_path / "gpuwm/core"
    root.mkdir(parents=True)
    path = root / "future.py"
    path.write_text("def compile():\n    cp.RawModule(code='source', options=())\n")
    monkeypatch.setattr(manifest, "REPO", tmp_path)
    assert manifest._discover_manifest_site_files(()) == ("gpuwm/core/future.py",)
    assert manifest.audit_source("gpuwm/core/future.py", path.read_text())


def test_named_literal_options_match_the_actual_compiler_tuple():
    source = (
        "OPTIONS = ('-std=c++17', '--fmad=false')\n"
        "def compile():\n"
        "    source = 'code'\n"
        "    module = cp.RawModule(code=source, options=('-std=c++17', '--fmad=false'))\n"
        "    record_module(module, source=source, options=OPTIONS)\n")
    assert manifest.audit_source("future.py", source) == []
    assert manifest.audit_source("future.py", source.replace(
        "OPTIONS = ('-std=c++17', '--fmad=false')",
        "OPTIONS = ('-std=c++17', '--fmad=true')"))


def test_additive_loader_header_declaration_needs_no_second_literal_list(monkeypatch):
    additions = {"future": ["future.cuh"]}
    monkeypatch.setattr(loader_identity, "_declared_header_extensions", lambda: additions)
    monkeypatch.setattr(loader_identity.kernel_loader, "EXTRA_HEADERS", {
        **loader_identity._EXPECTED_HEADERS, "future": ("future.cuh",)})
    loader_identity.test_allow_list_is_a_closed_literal_mapping()
    loader_identity.kernel_loader.EXTRA_HEADERS["future"] = ("different.cuh",)
    with pytest.raises(AssertionError):
        loader_identity.test_allow_list_is_a_closed_literal_mapping()
