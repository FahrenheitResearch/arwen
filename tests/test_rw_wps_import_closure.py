"""New scheme dependencies must follow the preprocessing doors into staging."""
import os
import subprocess
import sys

import pytest

from tools import build_rw_wps_release as builder


def _tree(tmp_path, monkeypatch, config):
    source, staged = tmp_path / "source", tmp_path / "staged"
    for root in (source, staged):
        (root / "gpuwm").mkdir(parents=True)
        (root / "gpuwm/__init__.py").write_text("")
        (root / "gpuwm/config.py").write_text(config)
    (source / "gpuwm/core").mkdir()
    (source / "gpuwm/core/__init__.py").write_text("")
    monkeypatch.setattr(builder, "REPO", source)
    # The synthetic fixture is not a checkout. Real staging still requires
    # tracked source files through the unchanged _copy_source boundary.
    monkeypatch.setattr(builder, "_require_tracked", lambda path: None)
    return source, staged


@pytest.mark.parametrize("statement", [
    "from gpuwm.core.future_scheme import VALUE",
    "from gpuwm.core import future_scheme\nVALUE = future_scheme.VALUE",
    "from .core.future_scheme import VALUE",
    "import importlib\nVALUE = importlib.import_module('gpuwm.core.future_scheme').VALUE",
])
def test_new_scheme_is_staged_transitively_and_imports_in_isolation(
        tmp_path, monkeypatch, statement):
    source, staged = _tree(tmp_path, monkeypatch, statement + "\n")
    (source / "gpuwm/core/future_scheme.py").write_text(
        "from .scheme_constants import VALUE\n")
    (source / "gpuwm/core/scheme_constants.py").write_text("VALUE = 47\n")
    files = builder._complete_staged_import_closure(staged)
    assert set(files) == {"gpuwm/core/__init__.py",
                          "gpuwm/core/future_scheme.py",
                          "gpuwm/core/scheme_constants.py"}
    assert builder._complete_staged_import_closure(staged) == []
    completed = subprocess.run([
        sys.executable, "-I", "-S", "-c",
        f"import sys; sys.path.insert(0, {str(staged)!r}); "
        "from gpuwm.config import VALUE; assert VALUE == 47",
    ], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("module", ["gpuwm.core.model", "gpuwm.verify.future"])
def test_required_excluded_code_still_refuses(tmp_path, monkeypatch, module):
    source, staged = _tree(tmp_path, monkeypatch, f"import {module}\n")
    target = source.joinpath(*module.split(".")).with_suffix(".py")
    target.parent.mkdir(exist_ok=True)
    (target.parent / "__init__.py").write_text("")
    target.write_text("VALUE = 1\n")
    with pytest.raises(RuntimeError, match="excluded executor or verification"):
        builder._complete_staged_import_closure(staged)
    assert not staged.joinpath(*module.split(".")).with_suffix(".py").exists()


def test_missing_required_source_still_refuses(tmp_path, monkeypatch):
    _, staged = _tree(tmp_path, monkeypatch, "import gpuwm.core.absent\n")
    with pytest.raises(RuntimeError, match="no package source"):
        builder._complete_staged_import_closure(staged)


def test_type_only_and_declared_optional_imports_stay_out(tmp_path, monkeypatch):
    _, staged = _tree(tmp_path, monkeypatch,
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import gpuwm.core.type_only\n"
        "def optional():\n    import gpuwm.core.optional\n")
    monkeypatch.setattr(builder, "_OPTIONAL_STAGED_IMPORTS", {
        ("gpuwm/config.py", "gpuwm.core.optional"): "optional forecast boundary"})
    assert builder._complete_staged_import_closure(staged) == []
    assert not (staged / "gpuwm/core").exists()


def test_named_metadata_import_keeps_unreachable_forecast_dependencies_out(tmp_path, monkeypatch):
    source, staged = _tree(tmp_path, monkeypatch,
        "from gpuwm.core.future_scheme import VALUE\n")
    (source / "gpuwm/core/future_scheme.py").write_text(
        "def value():\n    return 47\n"
        "VALUE = value()\n"
        "def forecast():\n    import gpuwm.core.model\n")
    (source / "gpuwm/core/model.py").write_text("raise RuntimeError('executor')\n")
    derived = set()
    builder._complete_staged_import_closure(staged, derived)
    assert not (staged / "gpuwm/core/model.py").exists()
    assert builder._staged_internal_imports(staged, source_root=source,
                                            derived_modules=derived) == []
    assert builder._staged_internal_imports(staged, source_root=source,
                                            derived_modules=derived, optional=True)
    (staged / "gpuwm/config.py").write_text(
        "from gpuwm.core.future_scheme import forecast\n")
    with pytest.raises(RuntimeError, match="excluded executor"):
        builder._complete_staged_import_closure(staged, derived)


def test_optional_forecast_module_import_does_not_expand_named_metadata(tmp_path, monkeypatch):
    source, staged = _tree(tmp_path, monkeypatch,
        "from gpuwm.core.future_scheme import VALUE\n"
        "def forecast():\n    from gpuwm.core import future_scheme\n")
    (source / "gpuwm/core/future_scheme.py").write_text(
        "VALUE = 47\ndef forecast():\n    import gpuwm.core.model\n")
    (source / "gpuwm/core/model.py").write_text("raise RuntimeError('executor')\n")
    monkeypatch.setattr(builder, "_OPTIONAL_STAGED_IMPORTS", {
        ("gpuwm/config.py", "gpuwm.core.future_scheme"): "forecast-only module access"})
    # This annotation applies to both imports in the synthetic config, so
    # use a separate metadata door to keep its required named import explicit.
    (staged / "gpuwm/metadata.py").write_text(
        "from gpuwm.core.future_scheme import VALUE\n")
    builder._complete_staged_import_closure(staged)
    assert (staged / "gpuwm/core/future_scheme.py").is_file()
    assert not (staged / "gpuwm/core/model.py").exists()


def test_unimported_driver_methods_are_outside_metadata_symbol_closure(tmp_path, monkeypatch):
    source, staged = _tree(tmp_path, monkeypatch,
        "from gpuwm.core.future_scheme import VALUE\n")
    (source / "gpuwm/core/future_scheme.py").write_text(
        "VALUE = 47\nclass Driver:\n"
        "    def run(self):\n        import gpuwm.core.model\n")
    (source / "gpuwm/core/model.py").write_text("raise RuntimeError('executor')\n")
    derived = set()
    builder._complete_staged_import_closure(staged, derived)
    assert not (staged / "gpuwm/core/model.py").exists()
    (staged / "gpuwm/config.py").write_text("from gpuwm.core.future_scheme import Driver\n")
    with pytest.raises(RuntimeError, match="excluded executor"):
        builder._complete_staged_import_closure(staged, derived)


def test_class_method_defaults_are_required_module_initialization(tmp_path, monkeypatch):
    source, staged = _tree(tmp_path, monkeypatch,
        "from gpuwm.core.future_scheme import VALUE\n")
    (source / "gpuwm/core/future_scheme.py").write_text(
        "VALUE = 47\ndef initialize():\n    import gpuwm.core.model\n"
        "class Driver:\n    def run(self, value=initialize()):\n        pass\n")
    (source / "gpuwm/core/model.py").write_text("raise RuntimeError('executor')\n")
    with pytest.raises(RuntimeError, match="excluded executor"):
        builder._complete_staged_import_closure(staged)


def test_new_importer_expands_an_already_parsed_modules_symbol_closure(tmp_path, monkeypatch):
    source, staged = _tree(tmp_path, monkeypatch,
        "from gpuwm.core.future_scheme import VALUE\n"
        "from gpuwm.core.later import VALUE as LATER\n")
    (source / "gpuwm/core/future_scheme.py").write_text(
        "VALUE = 47\ndef forecast():\n    import gpuwm.core.model\n")
    # This importer arrives in a later fixed-point iteration. Its named
    # import makes the previously unreachable function required.
    (source / "gpuwm/core/later.py").write_text(
        "from gpuwm.core.new_importer import VALUE\n")
    (source / "gpuwm/core/new_importer.py").write_text(
        "from gpuwm.core.future_scheme import forecast\nVALUE = forecast\n")
    (source / "gpuwm/core/model.py").write_text("raise RuntimeError('executor')\n")
    with pytest.raises(RuntimeError, match="excluded executor"):
        builder._complete_staged_import_closure(staged)


def test_scan_observes_same_size_edits_and_removed_importers(tmp_path, monkeypatch):
    source, staged = _tree(tmp_path, monkeypatch,
        "from gpuwm.core.future_scheme import VALUE\n")
    core = staged / "gpuwm/core"
    core.mkdir()
    (core / "__init__.py").write_text("")
    scheme = core / "future_scheme.py"
    scheme.write_text("VALUE = 47\ndef forecast():\n    import gpuwm.core.model\n")
    derived = {"gpuwm.core.future_scheme"}
    config = staged / "gpuwm/config.py"
    with builder._StagedImportScan() as scan:
        assert builder._staged_internal_imports(
            staged, source_root=source, derived_modules=derived, scan=scan) == []
        original_stat = scheme.stat()
        scheme.write_text(scheme.read_text().replace("gpuwm.core.model", "gpuwm.core.other"))
        os.utime(scheme, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        # Same size and timestamps still require the new content to be read.
        assert builder._staged_internal_imports(
            staged, source_root=source, derived_modules=derived, scan=scan) == []
        importer = staged / "gpuwm/late.py"
        importer.write_text("import gpuwm.core.future_scheme\n")
        records = builder._staged_internal_imports(
            staged, source_root=source, derived_modules=derived, scan=scan)
        assert [record['module'] for record in records] == ['gpuwm.core.other']
        importer.unlink()
        assert builder._staged_internal_imports(
            staged, source_root=source, derived_modules=derived, scan=scan) == []
        config.write_text("from gpuwm.core.future_scheme import forecast\n")
        records = builder._staged_internal_imports(
            staged, source_root=source, derived_modules=derived, scan=scan)
        assert [record['module'] for record in records] == ['gpuwm.core.other']


def test_both_builders_still_refuse_untracked_source(tmp_path, monkeypatch):
    from tools import build_native_wrf_distribution as native

    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    tracked = tmp_path / 'tracked.py'
    tracked.write_text('VALUE = 47\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'tracked.py'], check=True)
    untracked = tmp_path / 'untracked.py'
    untracked.write_text('VALUE = 48\n')
    monkeypatch.setattr(builder, '_tree_is_a_git_checkout', lambda: True)
    for module in (builder, native):
        monkeypatch.setattr(module, 'REPO', tmp_path)
        module._require_tracked(tracked)
        with pytest.raises(subprocess.CalledProcessError) as refused:
            module._require_tracked(untracked)
        assert refused.value.returncode == 1
        assert refused.value.cmd[-2:] == ['--error-unmatch', 'untracked.py']
