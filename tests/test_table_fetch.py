"""``gpuwm fetch-tables``: staging, verification, and refusal contract.

The externalized Thompson asset (freezeH2O.dat) ships as a release
asset, not in the wheel or repository, so the fetch path IS the install
path for it: every byte must be verified against the thompson_contract
pins before an atomic install, mismatches must be refused and deleted,
and an existing wrong file must never be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pytest

from gpuwm import table_assets
from gpuwm.core.thompson_contract import (
    CLASSIC_TABLE_ASSETS,
    TableAsset,
    validate_table_assets,
)
from gpuwm.physics_compat import packaged_thompson_table_root

REPO_ROOT = Path(__file__).resolve().parents[1]


def _asset_for(payload: bytes, name: str = "synthetic.dat") -> TableAsset:
    return TableAsset(name, len(payload),
                      hashlib.sha256(payload).hexdigest())


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**{"from_dir": None, **kwargs})


# ---------------------------------------------------------------------------
# Unit: staging semantics with synthetic assets
# ---------------------------------------------------------------------------

def test_fetch_from_dir_verifies_then_installs_atomically(tmp_path):
    payload = b"\x00\x01" * 4096
    asset = _asset_for(payload)
    source = tmp_path / "src"
    root = tmp_path / "root"
    source.mkdir(), root.mkdir()
    (source / asset.filename).write_bytes(payload)

    final = table_assets.fetch_asset_from_dir(root, asset, source)
    assert final == root / asset.filename
    assert final.read_bytes() == payload
    # no partial droppings
    assert list(root.glob(".*fetch-partial")) == []


def test_fetch_from_dir_refuses_and_deletes_wrong_bytes(tmp_path):
    payload = b"\x00\x01" * 4096
    asset = _asset_for(payload)
    source = tmp_path / "src"
    root = tmp_path / "root"
    source.mkdir(), root.mkdir()
    (source / asset.filename).write_bytes(b"corrupted" + payload)

    with pytest.raises(table_assets.TableAssetError, match="refused"):
        table_assets.fetch_asset_from_dir(root, asset, source)
    assert not (root / asset.filename).exists()
    assert list(root.glob(".*fetch-partial")) == []


def test_fetch_from_url_verifies_sha256_not_just_size(tmp_path):
    payload = os.urandom(8192)
    asset = _asset_for(payload)
    served = tmp_path / "served"
    root = tmp_path / "root"
    served.mkdir(), root.mkdir()

    # same size, different bytes -> refusal
    evil = bytearray(payload)
    evil[0] ^= 0xFF
    (served / asset.filename).write_bytes(bytes(evil))
    url = (served / asset.filename).as_uri()
    with pytest.raises(table_assets.TableAssetError, match="SHA-256"):
        table_assets.fetch_asset_from_url(root, asset, url)
    assert not (root / asset.filename).exists()

    # correct bytes -> installed
    (served / asset.filename).write_bytes(payload)
    final = table_assets.fetch_asset_from_url(root, asset, url)
    assert final.read_bytes() == payload


def test_classify_assets_partitions_valid_invalid_absent(tmp_path):
    good = b"good" * 1000
    asset_ok = _asset_for(good, "ok.dat")
    asset_bad = _asset_for(b"expected", "bad.dat")
    asset_gone = _asset_for(b"absent", "gone.dat")
    (tmp_path / "ok.dat").write_bytes(good)
    (tmp_path / "bad.dat").write_bytes(b"not-expected")

    valid, invalid, absent = table_assets.classify_assets(
        tmp_path, (asset_ok, asset_bad, asset_gone))
    assert [a.filename for a in valid] == ["ok.dat"]
    assert len(invalid) == 1 and "bad.dat" in invalid[0]
    assert [a.filename for a in absent] == ["gone.dat"]


# ---------------------------------------------------------------------------
# Integration: the real pins, the real packaged root
# ---------------------------------------------------------------------------

def _staged_root_without_externalized(tmp_path, monkeypatch) -> Path:
    """A table root holding the packaged (wheel-shipped) assets only.

    This is exactly what a wheel install looks like before
    ``gpuwm fetch-tables`` runs.  Hardlinks keep the copies free.
    """

    packaged = packaged_thompson_table_root()
    if not packaged.is_dir():
        pytest.skip("packaged table root absent (wheel-only checkout)")
    root = tmp_path / "tables"
    root.mkdir()
    for asset in CLASSIC_TABLE_ASSETS:
        if asset.filename in table_assets.EXTERNALIZED_TABLE_FILENAMES:
            continue
        source = packaged / asset.filename
        if not source.is_file():
            pytest.skip(f"packaged asset absent: {asset.filename}")
        try:
            os.link(source, root / asset.filename)
        except OSError:
            import shutil
            shutil.copyfile(source, root / asset.filename)
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(root))
    return root


def test_doctor_names_the_fetch_remedy_for_the_externalized_gap(
        tmp_path, monkeypatch):
    from gpuwm import doctor

    _staged_root_without_externalized(tmp_path, monkeypatch)
    check = doctor._thompson_tables_check()
    assert check.status == "missing"
    assert "freezeH2O.dat" in check.detail
    assert "externalized" in check.detail
    assert check.remedy is not None
    assert "gpuwm fetch-tables" in check.remedy
    # the reinstall hint is the wrong remedy for a fetchable gap
    assert "reinstall" not in check.remedy


def test_fetch_tables_main_stages_offline_and_is_idempotent(
        tmp_path, monkeypatch, capsys):
    packaged = packaged_thompson_table_root()
    externalized = [packaged / name
                    for name in table_assets.EXTERNALIZED_TABLE_FILENAMES]
    if not all(path.is_file() for path in externalized):
        pytest.skip("externalized asset bytes not present in this checkout")

    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    assert table_assets.fetch_tables_main(
        _args(from_dir=str(packaged))) == 0
    validate_table_assets(root)  # the exact load-time validation
    out = capsys.readouterr().out
    assert "verified and installed" in out

    # second run: nothing to fetch, still 0, no network, no rewrite
    before = (root / "freezeH2O.dat").stat().st_mtime_ns
    assert table_assets.fetch_tables_main(_args()) == 0
    assert (root / "freezeH2O.dat").stat().st_mtime_ns == before
    assert "nothing to fetch" in capsys.readouterr().out


def test_fetch_tables_main_refuses_existing_wrong_bytes(
        tmp_path, monkeypatch, capsys):
    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    (root / "freezeH2O.dat").write_bytes(b"tampered")
    assert table_assets.fetch_tables_main(_args()) == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and "never overwritten" in out
    # refusal must not delete or replace the operator's file
    assert (root / "freezeH2O.dat").read_bytes() == b"tampered"


def test_fetch_tables_main_treats_packaged_gap_as_reinstall(
        tmp_path, monkeypatch, capsys):
    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    (root / "thompson_aux_tables.dat").unlink()
    assert table_assets.fetch_tables_main(_args()) == 2
    assert "reinstall" in capsys.readouterr().out


def test_cli_dispatches_fetch_tables_for_real(monkeypatch, capsys):
    """Through gpuwm.cli.main, not just --help: the dispatch table must
    route fetch-tables to its handler (a --help-only probe once passed
    while the real dispatch raised AttributeError)."""

    from gpuwm.cli import main

    packaged = packaged_thompson_table_root()
    valid, invalid, absent = table_assets.classify_assets(packaged)
    if invalid or absent:
        pytest.skip("packaged table root incomplete in this checkout")
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(packaged))
    assert main(["fetch-tables"]) == 0
    assert "nothing to fetch" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The staging location: outside the install, or a wheel upgrade deletes it
# ---------------------------------------------------------------------------

def _wheel_shaped_install(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """(packaged root missing the externalized pair, user-level root).

    What a fresh ``pip install gpuwm`` looks like: the two assets under
    PyPI's per-file cap are inside site-packages, the two over it are
    not anywhere yet.  ``~/.gpuwm`` is redirected into ``tmp_path`` so
    the test never touches the real home directory.
    """

    real_packaged = packaged_thompson_table_root()
    if not real_packaged.is_dir():
        pytest.skip("packaged table root absent (wheel-only checkout)")
    packaged = tmp_path / "site-packages" / "gpuwm" / "data" / "thompson"
    packaged = packaged / "tables"
    packaged.mkdir(parents=True)
    for asset in CLASSIC_TABLE_ASSETS:
        if asset.filename in table_assets.EXTERNALIZED_TABLE_FILENAMES:
            continue
        source = real_packaged / asset.filename
        if not source.is_file():
            pytest.skip(f"packaged asset absent: {asset.filename}")
        try:
            os.link(source, packaged / asset.filename)
        except OSError:
            import shutil
            shutil.copyfile(source, packaged / asset.filename)

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("GPUWM_THOMPSON_TABLE_ROOT", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        "gpuwm.physics_compat.packaged_thompson_table_root",
        lambda: packaged)
    return packaged, home / ".gpuwm" / "tables" / "thompson"


def test_a_wheel_stages_outside_site_packages_and_resolves_there(
        tmp_path, monkeypatch, capsys):
    """`FileNotFoundError: missing Thompson table asset
    .../site-packages/gpuwm/data/thompson/tables/qr_acr_qg_V4.dat`

    A wheel user paid for a 315 MiB `gpuwm fetch-tables`, then a wheel
    upgrade deleted every byte of it, because staging went INSIDE
    site-packages.  Staging now lands beside ~/.gpuwm/bridges, and the
    resolver reads it there.
    """

    from gpuwm.physics_compat import thompson_table_root

    real_packaged = packaged_thompson_table_root()
    # Same guard as the offline staging test above, and for the same
    # reason: this stages `--from` the packaged root, and a published
    # clone deliberately does not carry the externalized assets --
    # `gpuwm fetch-tables` downloads them.  Without this the release
    # snapshot fails its own suite for doing exactly what it should.
    if not all((real_packaged / name).is_file()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES):
        pytest.skip("externalized asset bytes not present in this checkout")

    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)

    # Before staging: the resolver can only answer with the packaged
    # root, and it is short the two externalized assets.
    assert Path(thompson_table_root()) == packaged
    assert {a.filename for a in table_assets.unstaged_table_assets()} == set(
        table_assets.EXTERNALIZED_TABLE_FILENAMES)

    assert table_assets.staging_root() == user_root
    assert table_assets.fetch_tables_main(
        _args(from_dir=str(real_packaged))) == 0
    printed = capsys.readouterr().out
    assert "so a wheel upgrade cannot delete it" in printed

    # The staged root is COMPLETE -- half of it in site-packages would
    # resolve to nothing -- and it is what a run now reads.
    validate_table_assets(user_root)
    assert Path(thompson_table_root()) == user_root
    assert table_assets.unstaged_table_assets() == []

    # The wheel upgrade that used to erase the work: site-packages is
    # emptied, and the staged set still answers.
    for path in packaged.iterdir():
        path.unlink()
    assert Path(thompson_table_root()) == user_root
    assert table_assets.unstaged_table_assets() == []


def test_a_complete_packaged_root_is_left_alone(tmp_path, monkeypatch,
                                                capsys):
    """Negative control: a clone, and a wheel staged before this change.

    Both have all four assets in the packaged root, and neither should
    be told to re-download 362 MiB into a new location.
    """

    from gpuwm.physics_compat import thompson_table_root

    packaged = packaged_thompson_table_root()
    valid, invalid, absent = table_assets.classify_assets(packaged)
    if invalid or absent:
        pytest.skip("packaged table root incomplete in this checkout")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("GPUWM_THOMPSON_TABLE_ROOT", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    assert Path(thompson_table_root()) == packaged
    assert table_assets.staging_root() == packaged
    assert table_assets.fetch_tables_main(_args()) == 0
    assert "nothing to fetch" in capsys.readouterr().out
    assert not (home / ".gpuwm").exists()


def test_a_named_root_is_never_filled_from_the_package(tmp_path,
                                                       monkeypatch):
    """GPUWM_THOMPSON_TABLE_ROOT stays the operator's to populate.

    The completion-from-package step exists so a root THIS command
    chose is whole.  A mirror an operator named is not that, and
    quietly copying the package into it would make their explicit
    configuration mean something else.
    """

    root = tmp_path / "mirror"
    root.mkdir()
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(root))
    assert table_assets.staging_root() == root
    assert table_assets.staging_is_self_chosen(root) is False


# ---------------------------------------------------------------------------
# The refusal: one sentence at preflight, never a traceback mid-forecast
# ---------------------------------------------------------------------------

def test_a_missing_table_is_one_sentence_naming_the_file_and_the_command(
        tmp_path, monkeypatch):
    root = tmp_path / "empty"
    root.mkdir()
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(root))
    with pytest.raises(table_assets.MissingTableAssets) as raised:
        table_assets.require_thompson_tables()
    message = str(raised.value)
    # One sentence: no line breaks, and no full stop that starts
    # another one.  (Filenames carry dots; sentences carry ". ".)
    assert "\n" not in message
    assert ". " not in message
    assert "mp_physics=8" in message
    assert "qr_acr_qg_V4.dat" in message
    assert "gpuwm fetch-tables" in message
    # A FileNotFoundError subclass, so every stage that already turns
    # OSError into a sentence prints it instead of a traceback.
    assert isinstance(raised.value, FileNotFoundError)


def test_drift_stays_the_byte_validator_s_sentence_not_the_preflight_s(
        tmp_path, monkeypatch):
    """Negative control: the preflight answers absence, and only absence.

    A table that is PRESENT but wrong is a different failure with a
    better sentence available -- ``validate_table_assets`` can name the
    expected and actual byte counts.  The cheap gate must not shadow it
    with "not staged, run fetch-tables", which would send an operator
    with a drifted mirror to download a file they already have.
    """

    packaged = packaged_thompson_table_root()
    root = tmp_path / "drifted"
    root.mkdir()
    for asset in CLASSIC_TABLE_ASSETS:
        (root / asset.filename).write_bytes(b"drift")
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(root))
    assert table_assets.unstaged_table_assets() == []
    assert table_assets.require_thompson_tables() == root
    with pytest.raises(ValueError, match="has 5 bytes; expected"):
        validate_table_assets(root)

    # ... and a genuinely absent set is the preflight's to name.
    (root / CLASSIC_TABLE_ASSETS[0].filename).unlink()
    with pytest.raises(table_assets.MissingTableAssets):
        table_assets.require_thompson_tables()

    valid, invalid, absent = table_assets.classify_assets(packaged)
    if invalid or absent:
        pytest.skip("packaged table root incomplete in this checkout")
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(packaged))
    assert table_assets.unstaged_table_assets() == []


def test_gpuwm_check_warns_about_unstaged_tables_and_still_passes(
        tmp_path, monkeypatch, capsys):
    """Warn, never block: `gpuwm check` is the memory preflight.

    Sizing a domain whose tables live elsewhere is legitimate, so this
    is one line and the exit code is untouched -- the run doors are
    where the same condition is refused.
    """

    from gpuwm.core.preflight import _warn_unstaged_physics_tables

    empty = tmp_path / "no-tables"
    empty.mkdir()
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(empty))

    class _Domain:
        def __init__(self, mp):
            self.run = type("R", (), {"mp_physics": mp})()

    exp = type("E", (), {"domains": (_Domain(8),)})()
    assert _warn_unstaged_physics_tables(exp) is None
    warned = capsys.readouterr().err
    assert warned.startswith("warning: ")
    assert "qr_acr_qg_V4.dat" in warned
    assert "gpuwm fetch-tables" in warned

    # Negative control: a non-mp8 config says nothing at all.
    other = type("E", (), {"domains": (_Domain(10),)})()
    assert _warn_unstaged_physics_tables(other) is None
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# The staged root has to be complete for mp=28 too, and the two legs of
# the command are independent
# ---------------------------------------------------------------------------

def _packaged_ccn_asset():
    """The one aerosol asset, or a skip when this checkout lacks it."""

    from gpuwm.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS

    return AEROSOL_TABLE_ASSETS[0]


def test_staging_carries_the_mp28_activation_table_into_the_staged_root(
        tmp_path, monkeypatch, capsys):
    """`MissingAerosolTableAsset: CCN_ACTIVATE.BIN ... was not found`

    Reproduced on the published 2.6.1 wheel: `gpuwm fetch-tables` staged
    the four classic tables into ~/.gpuwm/tables/thompson, printed "table
    root ... complete and byte-valid", exited 0 -- and the next
    mp_physics=28 forecast died at its first microphysics step, because
    the aerosol contract resolves CCN_ACTIVATE.BIN against the SAME root
    as the classic tables and the copy was still sitting in
    site-packages.  "Complete" has to mean complete for every scheme the
    root serves, not just for mp=8.
    """

    asset = _packaged_ccn_asset()
    real_packaged = packaged_thompson_table_root()
    if not (real_packaged / asset.filename).is_file():
        pytest.skip(f"packaged asset absent: {asset.filename}")
    if not all((real_packaged / name).is_file()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES):
        pytest.skip("externalized asset bytes not present in this checkout")

    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)
    # A real wheel carries the activation table; the helper above only
    # lays down the classic set, so put it where a wheel has it.
    import shutil

    shutil.copyfile(real_packaged / asset.filename,
                    packaged / asset.filename)

    assert table_assets.fetch_tables_main(
        _args(from_dir=str(real_packaged))) == 0
    printed = capsys.readouterr().out
    assert asset.filename in printed
    assert "the mp=28 activation table" in printed

    staged = user_root / asset.filename
    assert staged.is_file(), "the staged root is still fatal for mp=28"
    assert staged.stat().st_size == asset.bytes

    # The check the first microphysics step makes, against the root a
    # run on this install actually resolves.
    from gpuwm.core.thompson_aerosol_contract import (
        resolve_ccn_activation_path,
        validate_ccn_activation_asset,
    )

    assert resolve_ccn_activation_path(None, user_root) == staged
    assert validate_ccn_activation_asset(staged) == asset


def test_a_present_activation_table_is_left_alone(tmp_path, monkeypatch,
                                                  capsys):
    """Idempotent: the copy is a gap-filler, never an overwrite."""

    asset = _packaged_ccn_asset()
    real_packaged = packaged_thompson_table_root()
    if not (real_packaged / asset.filename).is_file():
        pytest.skip(f"packaged asset absent: {asset.filename}")
    if not all((real_packaged / name).is_file()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES):
        pytest.skip("externalized asset bytes not present in this checkout")

    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)
    import shutil

    shutil.copyfile(real_packaged / asset.filename,
                    packaged / asset.filename)
    assert table_assets.fetch_tables_main(
        _args(from_dir=str(real_packaged))) == 0
    first = (user_root / asset.filename).stat().st_mtime_ns
    capsys.readouterr()

    assert table_assets.fetch_tables_main(
        _args(from_dir=str(real_packaged))) == 0
    printed = capsys.readouterr().out
    assert "the mp=28 activation table" not in printed
    assert (user_root / asset.filename).stat().st_mtime_ns == first


def test_a_refused_wif_leg_still_stages_the_mandatory_tables(
        tmp_path, monkeypatch, capsys):
    """`gpuwm fetch-tables --wif` used to stage NOTHING when --wif failed.

    Reproduced on the published 2.6.1 wheel against its own default URL:
    the WIF leg ran first, 404'd, returned 2, and the classic tables --
    which no flag makes optional and every forecast reads -- were never
    attempted.  The user asked for both and got neither.  The legs are
    independent; only --wif-only skips the mandatory one.
    """

    real_packaged = packaged_thompson_table_root()
    if not all((real_packaged / name).is_file()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES):
        pytest.skip("externalized asset bytes not present in this checkout")

    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)
    wif_root = tmp_path / "wif"

    # --from carries the classic tables but not the WIF dataset, so the
    # optional leg refuses for a real reason while the mandatory one can
    # be satisfied.
    code = table_assets.fetch_tables_main(_args(
        from_dir=str(real_packaged), wif=True, wif_root=str(wif_root)))
    printed = capsys.readouterr().out

    assert code == 2, "a refused optional leg still fails the command"
    assert "REFUSED" in printed
    # ...and the mandatory leg ran anyway.
    validate_table_assets(user_root)
    assert table_assets.unstaged_table_assets() == []
    # The summary says which leg did what, so the exit code is not the
    # only thing the reader has.
    assert "classic coefficient tables: staged" in printed
    assert "aerosol climatology dataset: REFUSED" in printed


def test_wif_only_is_the_one_flag_that_skips_the_mandatory_leg(
        tmp_path, monkeypatch, capsys):
    """The operator saying so in as many words is still honoured."""

    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)
    wif_root = tmp_path / "wif"
    code = table_assets.fetch_tables_main(_args(
        from_dir=str(tmp_path / "empty"), wif=True, wif_only=True,
        wif_root=str(wif_root)))
    assert code == 2
    assert not user_root.exists(), "--wif-only touched the classic root"


def test_the_wif_dataset_resolves_from_the_versioned_release_base(
        monkeypatch):
    """404: the dataset was fetched from the fixed v1.0.0 table release.

    Verified against the live endpoint on 2026-09-01: the coefficient
    tables answer 200 there and QNWFA_QNIFA_SIGMA_MONTHLY.dat answers
    404, because it has never been published under that tag.  It is
    carried as an asset of the release that pins it, beside the bridge
    bundles, so its base has to move with the release.
    """

    from gpuwm import bridge_assets

    monkeypatch.delenv(table_assets.ASSET_URL_BASE_ENV, raising=False)
    monkeypatch.delenv("GPUWM_BRIDGE_ASSET_URL_BASE", raising=False)

    class _Pins:
        release = "v9.9.9"

    monkeypatch.setattr(bridge_assets, "load_pins", lambda: _Pins())
    base = table_assets.wif_asset_url_base()
    assert base.endswith("/releases/download/v9.9.9")
    assert base == bridge_assets.asset_url_base(_Pins())
    assert "v1.0.0" not in base


def test_an_unreachable_wif_dataset_names_the_file_url_and_remedy(
        monkeypatch):
    """A refusal that does not say what to do is a traceback with manners."""

    from gpuwm import bridge_assets
    from gpuwm.ingest.wif_dataset import WIF_DATASET_FILE

    monkeypatch.delenv(table_assets.ASSET_URL_BASE_ENV, raising=False)

    def _no_release():
        raise bridge_assets.BridgeAssetError("the packaged pins declare "
                                             "no release")

    monkeypatch.setattr(bridge_assets, "load_pins", _no_release)
    with pytest.raises(table_assets.TableAssetError) as excinfo:
        table_assets.wif_asset_url_base()
    message = str(excinfo.value)
    assert WIF_DATASET_FILE in message
    assert "releases/download" in message
    assert "--from DIR" in message
    assert table_assets.ASSET_URL_BASE_ENV in message


def test_the_wif_leg_stages_from_a_local_file_url_base(tmp_path,
                                                       monkeypatch, capsys):
    """The whole route, over a file:// base, with the real pin enforced."""

    from gpuwm.ingest.wif_dataset import (
        WIF_DATASET_ASSET, WIF_DATASET_FILE)

    payload = tmp_path / "published"
    payload.mkdir()
    source = _real_wif_bytes()
    if source is None:
        pytest.skip("the pinned WIF dataset is not present on this host")
    import shutil

    shutil.copyfile(source, payload / WIF_DATASET_FILE)

    base = payload.resolve().as_uri()
    monkeypatch.setenv(table_assets.ASSET_URL_BASE_ENV, base)
    wif_root = tmp_path / "wif"
    assert table_assets.stage_wif_dataset(None, str(wif_root)) == 0
    landed = wif_root / WIF_DATASET_FILE
    assert landed.is_file()
    assert landed.stat().st_size == WIF_DATASET_ASSET.bytes
    assert "verified and installed" in capsys.readouterr().out


def _real_wif_bytes():
    """The pinned dataset wherever this host keeps it, else None."""

    from gpuwm.ingest.wif_dataset import (
        WIF_DATASET_ASSET, WIF_DATASET_FILE, resolve_wif_data_root)

    candidates = [resolve_wif_data_root(None) / WIF_DATASET_FILE]
    for candidate in candidates:
        if (candidate.is_file()
                and candidate.stat().st_size == WIF_DATASET_ASSET.bytes):
            return candidate
    return None
