"""An editable install of a source tree with no .git binds its content.

The README's source-archive route (extract, then `bash install.sh`, which
runs `pip install -e .`) produced an install that could name no identity:
doctor reported run provenance MISSING and recommended the very
`pip install -e .` that had just run, stage reuse degraded to "unknown
engine" and the HRRR hierarchy route refused.  The tree has no commit to
name, but setuptools leaves its file list in gpuwm.egg-info/SOURCES.txt,
and the identity is the digest of those files' current bytes.
"""

from __future__ import annotations

import builtins
import importlib.metadata
import os
import time
from pathlib import Path

import pytest

from gpuwm import provenance as install_provenance
from gpuwm import runtime_manifest as runtime

LISTED = ("pyproject.toml", "gpuwm/__init__.py", "gpuwm/data/authority.json",
          "gpuwm.egg-info/PKG-INFO", "gpuwm.egg-info/SOURCES.txt")


def _archive(tmp_path: Path, *, extra: tuple[str, ...] = ()) -> Path:
    root = tmp_path / "gpuwm-2.8.0"
    (root / "gpuwm" / "data").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "gpuwm"\nversion = "2.8.0"\n', encoding="utf-8")
    (root / "gpuwm" / "__init__.py").write_text('VERSION = "2.8.0"\n', encoding="utf-8")
    (root / "gpuwm" / "data" / "authority.json").write_text('{"value": 1}\n', encoding="utf-8")
    info = root / "gpuwm.egg-info"
    info.mkdir()
    (info / "PKG-INFO").write_text(
        "Metadata-Version: 2.4\nName: gpuwm\nVersion: 2.8.0\n", encoding="utf-8")
    (info / "SOURCES.txt").write_text("\n".join(LISTED + extra) + "\n", encoding="utf-8")
    return root


def _age(root: Path, seconds: float = 60.0) -> None:
    """Back-date every file, as an archive extracted a while ago is."""
    past = time.time() - seconds
    for path in root.rglob("*"):
        if path.is_file():
            os.utime(path, (past, past))


@pytest.fixture
def source_only(monkeypatch, tmp_path):
    """Nothing but the source tree can answer, and the digest cache is private."""
    monkeypatch.delenv(runtime.MANIFEST_ENV, raising=False)
    monkeypatch.setattr(runtime, "git_checkout_root", lambda root: None)
    monkeypatch.setattr(runtime, "installed_distribution", lambda *args: None)
    monkeypatch.setattr(runtime, "source_tree_identity", lambda **kwargs: None)
    cache = tmp_path / "home" / "source-content.json"
    monkeypatch.setattr(runtime, "_content_cache_path", lambda root: cache)
    return cache


def test_an_archive_binds_its_content_and_claims_no_commit(tmp_path, source_only):
    root = _archive(tmp_path)
    first = runtime.provenance(root)
    assert first["identity_source"] == "installed-source-content"
    assert first["git_commit"] is None and first["git_tree"] is None
    assert first["git_status_short"] is None
    content = first["installed_source_content"]
    assert content["source_file_count"] == len(LISTED)
    assert content["missing_file_count"] == 0
    assert content["source_version"] == "2.8.0"
    assert runtime.provenance(root) == first
    (root / "gpuwm" / "data" / "authority.json").write_text('{"value": 2}\n', encoding="utf-8")
    assert (runtime.provenance(root)["installed_source_content"]["content_sha256"]
            != content["content_sha256"])


def test_the_same_bytes_elsewhere_are_the_same_identity(tmp_path, source_only):
    one = runtime.provenance(_archive(tmp_path / "a"))
    other = runtime.provenance(_archive(tmp_path / "b"))
    assert one == other


def test_remembered_digests_are_not_read_again(tmp_path, source_only, monkeypatch):
    root = _archive(tmp_path)
    _age(root)
    first = runtime.provenance(root)
    assert source_only.is_file()
    opened = []
    real_open = builtins.open

    def counting_open(file, *args, **kwargs):
        opened.append(Path(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(runtime, "open", counting_open, raising=False)
    assert runtime.provenance(root) == first
    assert not [path for path in opened if root in path.parents]


def test_an_edit_that_keeps_the_size_is_still_seen(tmp_path, source_only):
    root = _archive(tmp_path)
    _age(root)
    first = runtime.provenance(root)["installed_source_content"]["content_sha256"]
    (root / "gpuwm" / "data" / "authority.json").write_text('{"value": 7}\n', encoding="utf-8")
    assert runtime.provenance(root)["installed_source_content"]["content_sha256"] != first


def test_a_freshly_written_file_is_never_remembered(tmp_path, source_only):
    import json

    root = _archive(tmp_path)
    _age(root)
    (root / "gpuwm" / "__init__.py").write_text('VERSION = "2.8.1"\n', encoding="utf-8")
    runtime.provenance(root)
    remembered = json.loads(source_only.read_text(encoding="utf-8"))["entries"]
    assert "gpuwm/data/authority.json" in remembered
    assert "gpuwm/__init__.py" not in remembered


def test_a_missing_listed_file_is_recorded_not_refused(tmp_path, source_only):
    root = _archive(tmp_path)
    whole = runtime.provenance(root)["installed_source_content"]
    (root / "gpuwm" / "data" / "authority.json").unlink()
    content = runtime.provenance(root)["installed_source_content"]
    assert content["missing_file_count"] == 1
    assert content["content_sha256"] != whole["content_sha256"]


def test_listed_bytecode_is_not_part_of_the_identity(tmp_path, source_only):
    root = _archive(tmp_path, extra=("gpuwm/__pycache__/x.cpython-312.pyc",))
    before = runtime.provenance(root)
    (root / "gpuwm" / "__pycache__").mkdir()
    (root / "gpuwm" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"derived")
    assert runtime.provenance(root) == before


@pytest.mark.parametrize("entry", ("../outside.py", "/etc/passwd", "C:/outside.py"))
def test_a_listed_file_outside_the_tree_is_refused(tmp_path, source_only, entry):
    root = _archive(tmp_path, extra=(entry,))
    with pytest.raises(runtime.IdentityError, match="outside that folder"):
        runtime.provenance(root)


def test_a_tree_with_no_file_list_still_refuses_and_names_the_step(tmp_path, source_only):
    root = _archive(tmp_path)
    for path in (root / "gpuwm.egg-info").iterdir():
        path.unlink()
    (root / "gpuwm.egg-info").rmdir()
    with pytest.raises(runtime.IdentityError, match="SOURCES.txt") as refusal:
        runtime.provenance(root)
    assert "pip install -e ." in str(refusal.value)


def test_a_checkout_is_never_bound_by_content(tmp_path, source_only):
    root = _archive(tmp_path)
    (root / ".git").mkdir()
    assert runtime.source_content_identity(root) is None


def test_a_wheel_beside_a_pyproject_keeps_its_wheel_identity(tmp_path, source_only, monkeypatch):
    site = tmp_path / "site-packages"
    (site / "gpuwm").mkdir(parents=True)
    (site / "gpuwm" / "__init__.py").write_text("", encoding="utf-8")
    (site / "pyproject.toml").write_text(
        '[project]\nname = "gpuwm"\nversion = "9.9.9"\n', encoding="utf-8")
    info = site / "gpuwm-2.8.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: gpuwm\nVersion: 2.8.0\n", encoding="utf-8")
    (info / "RECORD").write_text("gpuwm/__init__.py,sha256=AAAA,0\n", encoding="utf-8")
    record = {"record_aggregate_sha256": "a" * 64}
    monkeypatch.setattr(runtime, "wheel_record_identity", lambda: record)
    identity = runtime.provenance(site)
    assert identity["identity_source"] == "installed-wheel-record"
    assert "installed_source_content" not in identity
    distribution = next(importlib.metadata.distributions(path=[str(site)]))
    described = install_provenance.describe_provenance(
        site / "gpuwm", distribution, reported_version="2.8.0", probe_git=lambda root: None)
    assert described.install_kind == "wheel"


def test_a_source_local_egg_info_is_described_as_editable(tmp_path):
    root = _archive(tmp_path)
    distribution = next(importlib.metadata.distributions(path=[str(root)]))
    described = install_provenance.describe_provenance(
        root / "gpuwm", distribution, reported_version="2.8.0", probe_git=lambda root: None)
    assert described.install_kind == "editable"
    assert described.code_version == "2.8.0"


def test_the_forecast_recheck_sees_an_edit_to_the_archive(tmp_path, source_only):
    from gpuwm import prepared_single_domain_forecast as runner

    root = _archive(tmp_path)
    first = runtime.provenance(root)
    assert runner._runtime_source_identity_change(first, runtime.provenance(root)) is None
    (root / "gpuwm" / "data" / "authority.json").write_text('{"value": 2}\n', encoding="utf-8")
    moved = runner._runtime_source_identity_change(first, runtime.provenance(root))
    assert moved is not None and "installed_source_content" in moved


def test_stage_reuse_compares_the_archive_content(tmp_path, source_only, monkeypatch):
    from gpuwm import stage_reuse

    identity = runtime.provenance(_archive(tmp_path))
    monkeypatch.setattr(runtime, "provenance", lambda root: identity)
    engine = stage_reuse.engine_source_identity()
    assert engine["identity_source"] == "installed-source-content"
    assert engine["installed_source_content"] == identity["installed_source_content"]


def test_doctor_reports_the_archive_identity_as_verified(tmp_path, source_only, monkeypatch):
    from gpuwm import doctor

    identity = runtime.provenance(_archive(tmp_path))
    monkeypatch.setattr(runtime, "provenance", lambda root: identity)
    check = doctor._install_identity_check()
    assert check.status == "verified"
    assert "installed-source-content" in check.detail
    assert "no .git" in check.detail
