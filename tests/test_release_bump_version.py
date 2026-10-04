"""Every declaration of the release version moves together and agrees.

THE BREAKAGE THIS PREVENTS: opening 2.7.4 moved ``pyproject.toml`` and the
data package's ``VERSION`` and missed the terminal crate, so the release
contract set refused the candidate on the terminal/engine version test and
the cut needed a second commit. ``tools/release/bump_version.py`` rewrites
all four declaring files and opens the changelog section in one step; the
tests here run it on a copy of the real files and hold the real tree to
``--check``. The tool is a script under tools/release, so it is loaded by
path here, like the launcher template.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "release" / "bump_version.py"


def _load():
    spec = importlib.util.spec_from_file_location("bump_version", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_tree(tmp_path: Path) -> Path:
    module = _load()
    for relative in {r for r, _, _ in module.DECLARATIONS} | {module.CHANGELOG}:
        source = REPO_ROOT / relative
        if not source.is_file():
            pytest.skip(f"the bump test needs the source tree ({relative})")
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return tmp_path


def test_the_real_tree_declares_one_version_everywhere():
    module = _load()
    if not (REPO_ROOT / "pyproject.toml").is_file():
        pytest.skip("the bump check needs the source tree")
    assert module.disagreements(REPO_ROOT) == []


def test_bumping_moves_every_declaration_and_opens_the_changelog(tmp_path):
    module = _load()
    root = _copy_tree(tmp_path)
    before = module.declared(root)[0][1]
    to = "99.0.1"
    assert to != before
    touched = module.bump(root, to)
    assert module.disagreements(root) == []
    assert {version for _, version in module.declared(root)} == {to}
    assert set(touched) == {r for r, _, _ in module.DECLARATIONS} | {module.CHANGELOG}
    changelog = (root / module.CHANGELOG).read_text(encoding="utf-8")
    assert changelog.startswith("# Changelog")
    assert changelog.splitlines()[2] == f"## {to} (unreleased)"
    assert f"## {before}" in changelog  # the previous section stays
    # The pin the engine places on its data package moved with it.
    assert f'"gpuwm-data=={to}"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    # The terminal's lockfile entry moved, and only that entry.
    lock = (root / "tools/arwen-tui/Cargo.lock").read_text(encoding="utf-8")
    assert f'name = "arwen-tui"\nversion = "{to}"' in lock
    assert lock.count(to) == 1


def test_bumping_repins_the_vendor_manifest_digest_of_the_terminal_lock(tmp_path):
    """Opening 2.7.5, 2.7.6 and 2.8.0 each left the manifest naming the old lock.

    The shared UI vendor manifest pins the SHA-256 of the terminal crate's
    Cargo.lock, and the bump rewrites that lock's version line, so
    tests/test_vendored_cargo_registry.py refused the new line until a
    second commit re-pinned the digest by hand.
    """
    import hashlib

    module = _load()
    root = _copy_tree(tmp_path)
    source = REPO_ROOT / module.VENDOR_MANIFEST
    if not source.is_file():
        pytest.skip("the bump test needs the vendor manifest")
    manifest = root / module.VENDOR_MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, manifest)
    before = manifest.read_bytes().decode("utf-8")
    touched = module.bump(root, "99.0.4")
    assert module.VENDOR_MANIFEST in touched
    digest = hashlib.sha256((root / module.TERMINAL_LOCK).read_bytes()).hexdigest()
    after = manifest.read_bytes().decode("utf-8")
    assert f'"arwen-tui": "{digest}"' in after
    # Only that digest moved: every other line of the manifest is as it was.
    changed = [pair for pair in zip(before.splitlines(), after.splitlines()) if pair[0] != pair[1]]
    assert len(changed) == 1 and '"arwen-tui"' in changed[0][0]
    assert len(before.splitlines()) == len(after.splitlines())
    # A second pass at the same number has nothing to re-pin.
    assert module.VENDOR_MANIFEST not in module.bump(root, "99.0.4")


def test_bumping_regenerates_the_binary_notice_and_then_the_zarr_notice(tmp_path):
    """The Zarr reader's notice reproduces the binary notice whole.

    The bump moves the terminal lock, the binary notice pins that lock's
    digest, and the Zarr notice carries a copy of the binary notice. Run the
    other way round, or not at all, the Zarr notice ships the copy from before
    the bump: it went out through 2.8.4 with the binary notice of 2026-09-08.
    The generators are stand-ins here; they need cargo and the vendored trees.
    """
    module = _load()
    root = _copy_tree(tmp_path)
    stub = ("import pathlib, sys\n"
            "with open('generators.log', 'a', encoding='utf-8') as stream:\n"
            "    stream.write(pathlib.Path(sys.argv[0]).name + '\\n')\n")
    for relative in (module.NOTICE_TOOL, module.ZARR_NOTICE_TOOL):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(stub.encode("utf-8"))
    for relative in (*module.NOTICES, *module.ZARR_NOTICES, module.ZARR_LOCK):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(b"stand-in\n")
    log = root / "generators.log"

    touched = module.bump(root, "99.0.5")
    assert log.read_text(encoding="utf-8").split() == [
        Path(module.NOTICE_TOOL).name, Path(module.ZARR_NOTICE_TOOL).name]
    assert touched.index(module.NOTICES[0]) < touched.index(module.ZARR_NOTICES[0])
    assert set(module.NOTICES) | set(module.ZARR_NOTICES) <= set(touched)

    # A partial copy without the Zarr workspace has no Zarr notice to keep in step.
    (root / module.ZARR_LOCK).unlink()
    log.unlink()
    touched = module.bump(root, "99.0.6")
    assert log.read_text(encoding="utf-8").split() == [Path(module.NOTICE_TOOL).name]
    assert set(module.NOTICES) <= set(touched)
    assert not set(module.ZARR_NOTICES) & set(touched)


def test_bumping_again_does_not_open_a_second_changelog_section(tmp_path):
    module = _load()
    root = _copy_tree(tmp_path)
    module.bump(root, "99.0.2")
    touched_again = module.bump(root, "99.0.3")
    assert module.CHANGELOG in touched_again
    # A second pass at the same number rewrites the declarations and leaves the section alone.
    assert module.CHANGELOG not in module.bump(root, "99.0.3")
    changelog = (root / module.CHANGELOG).read_text(encoding="utf-8")
    assert changelog.count("## 99.0.3 (unreleased)") == 1
    assert changelog.count("## 99.0.2 (unreleased)") == 1


def test_check_names_the_file_that_disagrees(tmp_path):
    module = _load()
    root = _copy_tree(tmp_path)
    cargo = root / "tools/arwen-tui/Cargo.toml"
    current = module.declared(root)[0][1]
    text = cargo.read_text(encoding="utf-8")
    cargo.write_text(text.replace(f'version = "{current}"', 'version = "0.0.1"', 1), encoding="utf-8")
    assert module.disagreements(root) == [
        f"tools/arwen-tui/Cargo.toml says 0.0.1, pyproject.toml says {current}"
    ]
    assert module.main(["--check", "--root", str(root)]) == 1


def test_a_malformed_target_is_refused():
    module = _load()
    with pytest.raises(SystemExit):
        module.bump(REPO_ROOT, "2.7")
