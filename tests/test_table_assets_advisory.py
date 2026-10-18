"""The Thompson domain advisory describes the next run's staging route."""

from pathlib import Path

import pytest

from gpuwm import physics_compat, table_assets
from gpuwm.cli import main as cli_main


@pytest.mark.parametrize("named_root", [False, True], ids=["first-use", "inspection-only"])
def test_thompson_domain_advisory_matches_staging_route(
        tmp_path, monkeypatch, capsys, named_root):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    root = tmp_path / "tables"
    root.mkdir()
    # Presence is all this advisory checks. Leave precisely the two
    # externalized tables absent, as in a fresh wheel installation.
    for asset in table_assets.CLASSIC_TABLE_ASSETS:
        if asset.filename not in table_assets.EXTERNALIZED_TABLE_FILENAMES:
            (root / asset.filename).touch()
    monkeypatch.setattr(physics_compat, "packaged_thompson_table_root", lambda: root)
    if named_root:
        monkeypatch.setenv(physics_compat.THOMPSON_TABLE_ROOT_ENV, str(root))
    else:
        monkeypatch.delenv(physics_compat.THOMPSON_TABLE_ROOT_ENV, raising=False)

    def no_download(*args, **kwargs):
        pytest.fail("domain must only inspect tables, never stage them")

    monkeypatch.setattr(table_assets, "_stage_on_first_use", no_download)
    out = tmp_path / "thompson.toml"
    assert cli_main([
        "domain", "--point", "37.5,-90", "--point-extent-km", "1500",
        "--root-dx", "12", "--card", "32gb", "--source", "gfs",
        "--cycle", "2026-10-17T00", "--hours", "1", "--physics-profile",
        "thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1", "--out", str(out),
    ]) == 0
    captured = capsys.readouterr()
    advisory = next(line for line in captured.err.splitlines()
                    if "mp_physics=8 (Thompson) needs" in line)
    assert "qr_acr_qg_V4.dat" in advisory
    assert "freezeH2O.dat" in advisory
    assert "315 MiB" in advisory
    assert "gpuwm fetch-tables" in advisory
    assert "--from DIR" in advisory
    if named_root:
        assert "before preparing this forecast" in advisory
        assert "first `gpuwm go` run" not in advisory
    else:
        assert "first `gpuwm go` run downloads and checks the two tables" in advisory
        assert "from the pinned release assets" in advisory
        assert "ahead of time or offline" in advisory
        assert "before preparing this forecast" not in advisory
    assert out.is_file()
    assert not (home / ".gpuwm" / "tables").exists()
    assert all(not (root / name).exists()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES)
