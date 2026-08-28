"""The WIF dataset's pin, precedence and fail-closed resolver.

The ingest is measured elsewhere (``tests/test_wif_climatology.py``).
This file only asks whether a user can REACH it: whether the resolver
finds a staged copy, whether it refuses in a way that names the route,
whether an operator's override is honoured rather than quietly stepped
over, and whether the staging command verifies before it installs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gpuwm.ingest import wif_dataset as wd


def _fake_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / wd.WIF_DATASET_FILE
    path.write_bytes(b"")
    return path


def test_the_pin_is_a_size_and_a_sha256_and_the_file_is_not_shipped():
    """225 MB cannot ride in a wheel, and the constant says so.

    ``WIF_DATASET_REDISTRIBUTED`` exists so the packaging exclusion, the
    fetch route and the refusal text cannot drift apart: flipping it is
    the single edit that would claim the file ships, and it must stay
    False while the file is over the channel caps.
    """

    assert wd.WIF_DATASET_REDISTRIBUTED is False
    assert wd.WIF_DATASET_BYTES == 225_443_520
    assert len(wd.WIF_DATASET_SHA256) == 64
    assert int(wd.WIF_DATASET_SHA256, 16) >= 0
    asset = wd.WIF_DATASET_ASSET
    assert (asset.filename, asset.bytes, asset.sha256) == (
        wd.WIF_DATASET_FILE, wd.WIF_DATASET_BYTES, wd.WIF_DATASET_SHA256)
    # Over PyPI's 100 MB per-file cap and GitHub's 100 MiB blob limit --
    # which is the whole reason this file is externalized rather than
    # committed, and the reason the refusal may not say "reinstall".
    assert wd.WIF_DATASET_BYTES > 100 * 1000 * 1000
    assert wd.WIF_DATASET_BYTES > 100 * 1024 * 1024


def test_the_precedence_is_explicit_then_file_env_then_root_then_user(
        tmp_path):
    """Four rungs, highest first, and each one wins over the next."""

    explicit = _fake_dataset(tmp_path / "explicit")
    file_env = _fake_dataset(tmp_path / "file-env")
    root_env = _fake_dataset(tmp_path / "root-env")
    env = {wd.WIF_DATASET_PATH_ENV: str(file_env),
           wd.WIF_DATASET_ROOT_ENV: str(root_env.parent)}

    assert wd.resolve_wif_climatology_path(explicit, env=env) == explicit
    assert wd.resolve_wif_climatology_path(None, env=env) == file_env
    env.pop(wd.WIF_DATASET_PATH_ENV)
    assert wd.resolve_wif_climatology_path(None, env=env) == root_env
    env.pop(wd.WIF_DATASET_ROOT_ENV)
    assert wd.resolve_wif_data_root(None, env=env) == wd.user_wif_data_root()


def test_a_directory_is_accepted_wherever_a_path_is(tmp_path):
    """Pointing at the WRF tree rather than the file inside it is right.

    Refusing it would be pedantry with no failure behind it: there is
    exactly one filename, and the operator named the directory holding
    it.
    """

    staged = _fake_dataset(tmp_path / "wrf-run")
    assert wd.resolve_wif_climatology_path(tmp_path / "wrf-run") == staged


def test_a_named_path_that_does_not_exist_is_an_error_not_a_fallthrough(
        tmp_path):
    """The override is honoured even when honouring it means refusing.

    Silently ignoring an operator's override and reading the staged copy
    instead is how a run ends up using a dataset nobody chose -- the same
    rule ``resolve_ccn_activation_path`` states.
    """

    _fake_dataset(tmp_path / "staged")
    env = {wd.WIF_DATASET_ROOT_ENV: str(tmp_path / "staged")}
    with pytest.raises(wd.MissingWifClimatologyDataset) as caught:
        wd.resolve_wif_climatology_path(tmp_path / "typo.dat", env=env)
    assert "was NAMED" in str(caught.value)
    assert str(tmp_path / "typo.dat") in str(caught.value)


def test_the_refusal_names_what_where_how_and_why(tmp_path):
    """A refusal a reader can act on without opening any source."""

    env = {wd.WIF_DATASET_ROOT_ENV: str(tmp_path / "absent")}
    with pytest.raises(wd.MissingWifClimatologyDataset) as caught:
        wd.resolve_wif_climatology_path(None, env=env)
    message = str(caught.value)
    # what is missing, and where it was looked for
    assert wd.WIF_DATASET_FILE in message
    assert str(tmp_path / "absent") in message
    # what it is, and the pin an operator can check a candidate against
    assert "WPS intermediate format" in message
    assert wd.WIF_DATASET_SHA256 in message
    assert f"{wd.WIF_DATASET_BYTES}" in message
    # where to get it
    assert wd.WIF_DATASET_SOURCE.split(".")[0] in message
    # how to supply it -- every rung of the precedence, and the command
    assert "gpuwm fetch-tables --wif" in message
    assert wd.WIF_DATASET_PATH_ENV in message
    assert wd.WIF_DATASET_ROOT_ENV in message
    assert "wif_climatology_path" in message
    # why it is fatal rather than defaulted: the named breakage
    assert "aer_init_opt=0" in message
    assert "synthetic" in message


def test_validate_refuses_the_wrong_size_before_it_hashes(tmp_path):
    """Size first: a stat rejects the common mistakes without 215 MiB."""

    path = _fake_dataset(tmp_path / "staged")
    path.write_bytes(b"not the dataset")
    with pytest.raises(ValueError) as caught:
        wd.validate_wif_dataset(path)
    assert "15 bytes" in str(caught.value)
    assert str(wd.WIF_DATASET_BYTES) in str(caught.value)
    assert wd.WIF_DATASET_SHA256 not in str(caught.value).split("expected")[0]


def test_validate_refuses_a_right_sized_impostor(tmp_path, monkeypatch):
    """Same size, different bytes: caught by the SHA-256, not the stat.

    Exercised at a small pinned size rather than by writing 215 MiB --
    the code path under test is the digest comparison, and paying a
    quarter-gigabyte of IO to reach it would make this test the slowest
    in the file for no additional coverage.
    """

    payload = b"\x00" * 4096
    path = _fake_dataset(tmp_path / "staged")
    path.write_bytes(payload)
    monkeypatch.setattr(wd, "WIF_DATASET_BYTES", len(payload))
    monkeypatch.setattr(wd, "WIF_DATASET_SHA256", "0" * 64)
    with pytest.raises(ValueError) as caught:
        wd.validate_wif_dataset(path)
    assert hashlib.sha256(payload).hexdigest() in str(caught.value)


def test_the_staging_command_verifies_before_it_installs(tmp_path,
                                                         monkeypatch):
    """``gpuwm fetch-tables --wif --from DIR`` is the table contract.

    Wrong bytes are refused and deleted rather than installed, and the
    destination is left untouched -- which is what makes the pin a
    guarantee about what a run reads rather than a report about what it
    downloaded.
    """

    from gpuwm import table_assets

    source = tmp_path / "wrf-run"
    source.mkdir()
    (source / wd.WIF_DATASET_FILE).write_bytes(b"wrong bytes")
    destination = tmp_path / "root"

    code = table_assets.stage_wif_dataset(source_dir=source,
                                          root=destination)
    assert code == 2
    assert not (destination / wd.WIF_DATASET_FILE).exists()
    assert list(destination.glob("*")) == []


def test_presence_is_answered_without_hashing(tmp_path):
    """The preflight question costs a stat, not 215 MiB.

    ``wif_dataset_is_staged`` deliberately does not compete with
    ``validate_wif_dataset``: absence is one sentence, drift is another,
    and making every launch hash the dataset to say "yes" is not a
    service.
    """

    assert wd.wif_dataset_is_staged(
        tmp_path / "absent", env={}) is False
    _fake_dataset(tmp_path / "present")
    assert wd.wif_dataset_is_staged(
        tmp_path / "present", env={}) is True
