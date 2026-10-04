"""Public preparation CLI preserves the optional native physical-store binding."""
from pathlib import Path
import pytest
from gpuwm import source_cli


def _gfs(extra=()):
    return ["--source", "gfs", "--gfs-series", "/source/series.tsv",
            "--cycle", "2024-01-12_00:00:00", "--bridge", "/bin/gfs_grib2_bridge",
            "--wps-namelist", "/case/namelist.wps", "--experiment-config", "/case/experiment.toml",
            "--static-input", "/case/static.npz", "--static-receipt", "/case/static.json",
            "--source-manifest", "/source/inputs.json", "--source-manifest-sha256", "a" * 64,
            "--output-root", "/prepared", "--preprocess-backend", "cpu",
            "--preprocess-workers", "2", *extra]


def test_gfs_default_command_does_not_select_a_physical_store():
    command = source_cli._gfs_command(source_cli._parser().parse_args(_gfs()))
    assert "--physical-input-store" not in command
    assert "--physical-output-store" not in command


@pytest.mark.parametrize("flag", ["--physical-input-store", "--physical-output-store"])
def test_gfs_public_dry_run_reaches_the_native_store_argument(flag, capsys):
    assert source_cli.main(_gfs([flag, "/member/physical", "--dry-run"])) == 0
    output = capsys.readouterr().out
    assert "-m gpuwm.gfs_direct" in output
    assert flag + " /member/physical" in output.replace("\\", "/")


def test_gfs_sealed_input_refuses_as_posted_before_launch():
    args = source_cli._parser().parse_args(_gfs(["--physical-input-store", "/member/physical", "--as-posted", "/posting"]))
    args.source_sha256s = None
    args.source_sha256s_sha256 = None
    assert any("sealed physical input" in message for message in source_cli._required_gfs_args(args))


def test_gfs_posted_capture_reaches_the_native_door():
    args = source_cli._parser().parse_args(_gfs(["--physical-output-store", "/source/physical", "--as-posted", "/posting"]))
    args.source_sha256s = None
    args.source_sha256s_sha256 = None
    assert source_cli._required_gfs_args(args) == []


def test_gfs_posted_member_does_not_resolve_an_installed_decoder(monkeypatch, capsys):
    from gpuwm import bridges
    arguments = _gfs(["--physical-input-provider", "/provider", "--physical-member-index", "29",
                      "--as-posted", "/posting", "--dry-run"])
    for flag in ("--bridge", "--source-manifest", "--source-manifest-sha256"):
        index = arguments.index(flag)
        del arguments[index:index+2]
    def forbidden(*_args, **_kwargs):
        raise AssertionError("posted member resolved an unused raw decoder")
    monkeypatch.setattr(source_cli, "_distribution_decoder", forbidden)
    monkeypatch.setattr(bridges, "resolve_source_decoder", forbidden)
    assert source_cli.main(arguments) == 0
    output = capsys.readouterr().out
    assert "-m gpuwm.gfs_direct" in output
    assert "--bridge" not in output


@pytest.mark.parametrize("source, command", [
    ("hrrr", source_cli._hrrr_command), ("gfs", source_cli._gfs_command),
    ("mapped", source_cli._mapped_command)])
def test_posted_provider_forwarding_preserves_original_member_index(source, command):
    args = source_cli._parser().parse_args([
        "--source", source, "--physical-input-provider", "/provider",
        "--physical-member-index", "29", "--as-posted", "/posting",
        "--output-root", "/prepared", "--input", "/input.grib2"])
    args.supplement = []
    args.provenance = []
    actual = command(args)
    assert Path(actual[actual.index("--physical-input-provider") + 1]) == Path("/provider")
    assert actual[actual.index("--physical-member-index") + 1] == "29"


@pytest.mark.parametrize("extra", [
    ["--physical-input-provider", "/provider"],
    ["--physical-member-index", "29"],
    ["--physical-input-provider", "/provider", "--physical-member-index", "29"]])
def test_posted_provider_partial_or_unposted_request_is_refused(extra, capsys):
    with pytest.raises(SystemExit) as stopped:
        source_cli.main(_gfs([*extra, "--dry-run"]))
    assert stopped.value.code == 2
    assert "physical-input-provider" in capsys.readouterr().err


def test_hrrr_base_bridge_reuse_stays_hrrr_only(capsys):
    with pytest.raises(SystemExit) as stopped:
        source_cli.main(_gfs(["--physical-base-prepared", "/hrrr/base", "--dry-run"]))
    assert stopped.value.code == 2
    assert "physical-base-prepared is used only by the native HRRR route" in capsys.readouterr().err
