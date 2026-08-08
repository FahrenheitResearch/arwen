"""`gpuwm version`: which code is running, and can pip move it.

A field user upgraded repeatedly and nothing changed: `pip install -e
~/gpuwm` at 1.6.2 shadows every wheel pip writes to site-packages, so
each upgrade succeeded, printed a new version, and left the old source
tree serving every import.  The number he could see -- gpuwm.__version__
-- is read from distribution METADATA, so it reported what pip had just
written rather than what was running.  These tests pin the four things
that make the difference: the version comes from the distribution that
actually provides the imported package, the location is reported, an
editable install is named as one, and the network is optional.

The install shapes are built out of REAL files -- a real package
directory and a real importlib.metadata distribution reading a real
.dist-info -- rather than mocks, because the shape that broke the first
draft was one no fixture would have invented: the reference box carries
a stale gpuwm whose __file__ is None.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from importlib.metadata import PathDistribution

import pytest

from gpuwm import version_cli


def _dist_info(site: Path, *, name="gpuwm", version="1.6.2",
               direct_url: dict | None = None) -> PathDistribution:
    """A real .dist-info on disk, read by a real PathDistribution."""

    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8")
    if direct_url is not None:
        (info / "direct_url.json").write_text(
            json.dumps(direct_url), encoding="utf-8")
    return PathDistribution(info)


def _package(root: Path) -> Path:
    package = root / "gpuwm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    return package


# --- the wheel shape ----------------------------------------------------

def test_a_wheel_install_reports_its_version_and_location(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    package = _package(site)
    shape = version_cli.describe_install(
        package, _dist_info(site, version="1.8.0"))
    assert shape["version"] == "1.8.0"
    assert shape["distribution"] == "gpuwm"
    assert shape["editable"] is False
    assert shape["package_root"] == package.resolve()
    line = version_cli.headline(shape)
    assert line.startswith("gpuwm 1.8.0 (installed wheel at ")
    assert "editable" not in line


def test_a_wheel_install_is_told_to_upgrade_with_pip(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    shape = version_cli.describe_install(
        _package(site), _dist_info(site, version="1.8.0"))
    report = "\n".join(version_cli.local_report(shape))
    assert "upgrade with:" in report and "-m pip install --upgrade" in report
    assert "NOT" not in report


# --- the editable shape -------------------------------------------------

def test_a_pep610_editable_install_is_named_as_one(tmp_path):
    """dir_info.editable is the modern, authoritative marker."""
    site = tmp_path / "site-packages"
    site.mkdir()
    source = tmp_path / "home" / "user" / "gpuwm"
    package = _package(source)
    distribution = _dist_info(
        site, version="1.6.2",
        direct_url={"url": source.as_uri(), "dir_info": {"editable": True}})
    shape = version_cli.describe_install(package, distribution)
    assert shape["version"] == "1.6.2"
    assert shape["editable"] is True
    assert shape["source_root"] == source.resolve()
    assert version_cli.headline(shape).startswith(
        f"gpuwm 1.6.2 (editable source at {source.resolve()}")


def test_an_editable_install_says_pip_upgrade_will_change_nothing(tmp_path):
    """THE sentence the field user needed and did not get."""
    site = tmp_path / "site-packages"
    site.mkdir()
    source = tmp_path / "home" / "user" / "gpuwm"
    shape = version_cli.describe_install(_package(source), _dist_info(
        site, direct_url={"url": source.as_uri(),
                          "dir_info": {"editable": True}}))
    report = "\n".join(version_cli.local_report(shape))
    assert "EDITABLE install" in report
    assert "change NOTHING about the code that runs" in report
    assert "shadows any wheel" in report
    # Not a checkout, so no `git pull` is offered -- a remedy naming a
    # command that would fail is worse than naming none.
    assert "git -C" not in report
    assert "the way you obtained it" in report


def test_an_editable_checkout_is_told_to_git_pull(tmp_path):
    """The reported case verbatim: an editable install of a real clone."""
    import subprocess

    site = tmp_path / "site-packages"
    site.mkdir()
    source = tmp_path / "home" / "user" / "gpuwm"
    package = _package(source)
    init = subprocess.run(["git", "init", "-q", str(source)],
                          capture_output=True, text=True)
    if init.returncode != 0:
        pytest.skip("no usable git on this machine")
    subprocess.run(["git", "-C", str(source), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-q", "--allow-empty",
                    "-m", "fixture"], capture_output=True, check=False)
    shape = version_cli.describe_install(package, _dist_info(
        site, direct_url={"url": source.as_uri(),
                          "dir_info": {"editable": True}}))
    assert shape["git"].get("commit")
    report = "\n".join(version_cli.local_report(shape))
    assert f"git -C {source.resolve()} pull" in report
    assert f"git {shape['git']['commit']}" in version_cli.headline(shape)


def test_a_legacy_develop_install_is_editable_without_direct_url(tmp_path):
    """setup.py develop and hand-written .pth files leave no PEP 610.

    The structural signal catches them: a distribution whose metadata
    lives in site-packages while the package it claims does not.
    """
    site = tmp_path / "site-packages"
    site.mkdir()
    source = tmp_path / "src" / "gpuwm-checkout"
    shape = version_cli.describe_install(
        _package(source), _dist_info(site, direct_url=None))
    assert shape["editable"] is True


def test_a_dist_info_that_is_unreadable_does_not_break_the_shape(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    distribution = _dist_info(site)
    (distribution._path / "direct_url.json").write_text(
        "{not json", encoding="utf-8")
    shape = version_cli.describe_install(_package(site), distribution)
    assert shape["editable"] is False          # structural signal only


# --- no distribution at all --------------------------------------------

def test_a_bare_source_tree_does_not_borrow_another_version(tmp_path):
    """The trap: gpuwm.__version__ would report SOME distribution's number.

    With nothing installed that provides this code, there is no version
    to report, and reporting one would be the exact confusion this
    command exists to end.
    """
    shape = version_cli.describe_install(_package(tmp_path), None)
    assert shape["version"] is None and shape["distribution"] is None
    report = "\n".join(version_cli.local_report(shape))
    assert "SECOND copy" in report
    assert version_cli.headline(shape).startswith("gpuwm (source tree at ")


# --- git identity, guarded ---------------------------------------------

def test_a_directory_that_is_not_a_repository_reports_no_git(tmp_path):
    assert version_cli._git_identity(tmp_path) == {}


def test_an_absent_git_binary_is_not_an_error(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("git: not found")

    monkeypatch.setattr(version_cli.subprocess, "run", no_git)
    assert version_cli._git_identity(tmp_path) == {}


def test_this_checkout_reports_its_own_sha_and_branch():
    """Against the real repository this suite runs from."""
    root = Path(__file__).resolve().parents[1]
    git = version_cli._git_identity(root)
    if not git:
        pytest.skip("the suite is not running from a git checkout")
    assert len(git["commit"]) >= 7
    assert git["branch"]
    assert version_cli.headline(
        version_cli.describe_install(root / "gpuwm", None))


# --- the network is optional and last -----------------------------------

def _shape(tmp_path, version="1.6.2", editable=False):
    site = tmp_path / "site-packages"
    site.mkdir(exist_ok=True)
    root = (tmp_path / "src") if editable else site
    direct = ({"url": root.as_uri(), "dir_info": {"editable": True}}
              if editable else None)
    return version_cli.describe_install(
        _package(root), _dist_info(site, version=version,
                                   direct_url=direct))


def test_an_unreachable_index_prints_nothing_at_all(tmp_path, monkeypatch):
    """Offline is silence, not an error and not a stack trace."""
    def refuse(*args, **kwargs):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(version_cli.urllib.request, "urlopen", refuse)
    assert version_cli.pypi_latest("gpuwm") is None
    assert version_cli.pypi_report(_shape(tmp_path)) == []


@pytest.mark.parametrize("boom", [
    TimeoutError("timed out"),
    OSError("connection reset"),
    ValueError("not json"),
    KeyError("info"),
])
def test_every_index_failure_is_silence(monkeypatch, boom):
    def refuse(*args, **kwargs):
        raise boom

    monkeypatch.setattr(version_cli.urllib.request, "urlopen", refuse)
    assert version_cli.pypi_latest("gpuwm") is None


def test_the_local_lines_are_printed_before_the_index_is_touched(
        tmp_path, monkeypatch, capsys):
    """A version command that cannot answer on a plane is broken.

    A short timeout is not enough on its own: the local answer must
    already be on the terminal when the socket is opened, so a hung
    index costs the reader the staleness line and nothing else.
    """
    import builtins

    order: list[str] = []
    monkeypatch.setattr(version_cli, "install_shape",
                        lambda: _shape(tmp_path))
    real_print = builtins.print

    def watched(*args, **kwargs):
        order.append("print")
        real_print(*args, **kwargs)

    def slow(*args, **kwargs):
        order.append("network")
        raise urllib.error.URLError("still offline")

    monkeypatch.setattr(builtins, "print", watched)
    monkeypatch.setattr(version_cli.urllib.request, "urlopen", slow)
    assert version_cli.version_main(None) == 0
    monkeypatch.undo()

    assert "network" in order, "the index was never asked at all"
    assert order[0] == "print"
    assert order.index("print") < order.index("network")
    assert capsys.readouterr().out.startswith("gpuwm 1.6.2")


def test_offline_skips_the_index_entirely(tmp_path, monkeypatch):
    monkeypatch.setattr(version_cli, "install_shape",
                        lambda: _shape(tmp_path))

    def never(*args, **kwargs):
        raise AssertionError("--offline still reached the network")

    monkeypatch.setattr(version_cli.urllib.request, "urlopen", never)
    from types import SimpleNamespace
    assert version_cli.version_main(SimpleNamespace(offline=True)) == 0


def test_a_stale_editable_install_gets_the_whole_sentence(tmp_path,
                                                          monkeypatch):
    """The reported case, end to end: 1.6.2 editable against a 1.8.0 index."""
    monkeypatch.setattr(version_cli, "pypi_latest", lambda *a, **k: "1.8.0")
    shape = _shape(tmp_path, version="1.6.2", editable=True)
    lines = version_cli.local_report(shape) + version_cli.pypi_report(shape)
    report = "\n".join(lines)
    assert "gpuwm 1.6.2" in report
    assert "editable source at" in report
    assert "PyPI latest is 1.8.0" in report
    assert "will NOT get there via pip" in report


def test_a_current_wheel_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(version_cli, "pypi_latest", lambda *a, **k: "1.8.0")
    shape = _shape(tmp_path, version="1.8.0")
    assert "current" in "\n".join(version_cli.pypi_report(shape))


def test_a_behind_wheel_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(version_cli, "pypi_latest", lambda *a, **k: "1.9.0")
    shape = _shape(tmp_path, version="1.8.0")
    assert "behind" in "\n".join(version_cli.pypi_report(shape))


def test_a_source_tree_never_asks_the_index(tmp_path, monkeypatch):
    """There is no distribution to compare, so there is no question."""
    def never(*args, **kwargs):
        raise AssertionError("asked the index about a bare source tree")

    monkeypatch.setattr(version_cli, "pypi_latest", never)
    shape = version_cli.describe_install(_package(tmp_path), None)
    assert version_cli.pypi_report(shape) == []


@pytest.mark.parametrize("installed,latest,expected", [
    ("1.6.2", "1.8.0", True),
    ("1.8.0", "1.8.0", False),
    ("1.9.0", "1.8.0", False),
    (None, "1.8.0", None),
    ("1.8.0", None, None),
    ("not-a-version", "1.8.0", None),
])
def test_version_comparison(installed, latest, expected):
    assert version_cli._is_behind(installed, latest) is expected


# --- the surfaces -------------------------------------------------------

def test_the_cli_registers_version_and_it_exits_zero(capsys):
    from gpuwm import cli

    assert cli.main(["version", "--offline"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("gpuwm")


@pytest.mark.parametrize("argv", [["doctor"], ["doctor", "--explain"]])
def test_doctor_reports_which_copy_of_gpuwm_produced_it(argv, capsys):
    """A report that diagnoses an estate has to say whose.

    First line, in both widths: it reframes everything under it.
    """
    from gpuwm import cli

    cli.main(argv)
    out = capsys.readouterr().out
    assert out.splitlines()[0] == version_cli.headline()


def test_the_json_report_is_still_only_json(capsys):
    """A front end parses this; a header line would break it."""
    from gpuwm import cli

    cli.main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and payload


def test_the_doctor_header_never_fails_the_report(monkeypatch):
    from gpuwm import doctor

    def broken():
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(version_cli, "headline", broken)
    assert "unavailable" in doctor._install_headline()
