"""Launch relays explicit donor bytes to the existing preparation contracts."""

import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from gpuwm import go_cli, runplan
from gpuwm.cli import build_parser, main
from gpuwm.launch_supplements import hrrr_source_manifest
from test_go_native_launch import allow_launch_resources, emit


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_hrrr_go_dry_run_preserves_repeated_donors(tmp_path):
    config, _ = emit(tmp_path, source="hrrr", subprocess_door=True)
    data = tmp_path / "source files"
    data.mkdir()
    donors = [data / "donor file.grib2", data / "second donor.grib2"]
    args = ["go", str(config), "--data-dir", str(data), "--dry-run"]
    for donor in donors:
        donor.write_bytes(b"donor")
        args += ["--supplement", f"PMSL={donor}"]
    result = subprocess.run([sys.executable, "-m", "gpuwm.cli", *args],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(line.removeprefix("Run: ") for line in result.stdout.splitlines()
                   if line.startswith("Run: "))
    tokens = shlex.split(command)[1:]
    parsed = build_parser().parse_args(tokens)
    assert [Path(value.removeprefix("PMSL=")) for value in parsed.supplement] == donors
    replay = subprocess.run([sys.executable, "-m", "gpuwm.cli", *tokens, "--dry-run"],
                            capture_output=True, text=True, timeout=60)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert not (tmp_path / "weather area-go").exists()


@pytest.mark.parametrize("bad", ["missing", "outside", "wrong_role", "duplicate"])
def test_bad_hrrr_donor_refuses_before_output_or_fetch(tmp_path, monkeypatch, capsys, bad):
    config = emit(tmp_path, source="hrrr")
    data = tmp_path / "source"
    data.mkdir()
    donor = (tmp_path if bad == "outside" else data) / "donor.grib2"
    if bad != "missing":
        donor.write_bytes(b"explicit donor")
    role = "surface" if bad == "wrong_role" else "PMSL"
    extra = ["--supplement", f"{role}={donor}"]
    if bad == "duplicate":
        extra *= 2
    out = tmp_path / "uncreated"
    monkeypatch.setattr(runplan, "_run_fetch", lambda *a, **k: pytest.fail("fetched"))
    assert main(["go", str(config), "--data-dir", str(data), "--outdir", str(out),
                 "--dry-run", *extra]) == 2
    error = capsys.readouterr().err
    assert {"missing": "is missing", "outside": "must be inside", "wrong_role": "requires PMSL",
            "duplicate": "duplicate supplement"}[bad] in error
    assert not out.exists()


def test_unused_gfs_supplement_refuses_at_the_human_door(tmp_path, capsys):
    config = emit(tmp_path, source="gfs")
    donor = tmp_path / "donor"
    donor.write_bytes(b"donor")
    assert main(["go", str(config), "--supplement", f"PMSL={donor}", "--dry-run"]) == 2
    assert "not used by the prepared:go route" in capsys.readouterr().err


def test_hrrr_go_binds_and_forwards_donors_without_rewriting_fetch_manifest(
        tmp_path, monkeypatch, capsys):
    from tools.hrrr_pipeline import verify_source_tree
    from tools.prepare_hrrr_wrf import _parser

    config = emit(tmp_path, source="hrrr")
    allow_launch_resources(monkeypatch)
    data = tmp_path / "source"
    data.mkdir()
    donors = [data / "donor one.grib2", data / "donor two.grib2"]
    for index, donor in enumerate(donors):
        donor.write_bytes(f"explicit donor {index}".encode())
    primary = data / "primary.grib2"
    primary.write_bytes(b"primary fixture; no GRIB decoding claimed")
    fetched = data / "SHA256SUMS"
    expected = f"{_digest(primary)}  {primary.name}\n"
    seen = []
    def fetch(*args, **kwargs):
        fetched.write_text(expected)
        return {}
    def prepare(root, *, arguments, **kwargs):
        parsed = _parser().parse_args(arguments[3:])
        assert parsed.supplement == [f"PMSL={path}" for path in donors]
        assert parsed.source_manifest != fetched
        assert _digest(parsed.source_manifest) == parsed.source_manifest_sha256
        series = tmp_path / "series.tsv"
        series.write_text("".join(
            f"{hour}\t{primary}\t{primary}"
            + "".join(f"\tPMSL={donor}" for donor in donors) + "\n" for hour in (0, 1)))
        receipt = verify_source_tree(source_root=data, manifest=parsed.source_manifest,
            expected_manifest_sha256=parsed.source_manifest_sha256, series=series, workers=1)
        assert len(receipt["supplement_bindings"]) == 4
        assert fetched.read_text() == expected
        donors[0].write_bytes(b"changed after binding")
        with pytest.raises(ValueError, match="payload hash mismatch"):
            verify_source_tree(source_root=data, manifest=parsed.source_manifest,
                expected_manifest_sha256=parsed.source_manifest_sha256, series=series, workers=1)
        seen.append(parsed)
        raise runplan.PlanError("Contract witness complete; stopping before GRIB decoding.")
    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    monkeypatch.setattr(runplan, "_prepare_stage", prepare)
    args = ["go", str(config), "--data-dir", str(data), "--outdir", str(tmp_path / "run"),
            "--run-stamp", "off", "--products", "none"]
    for path in donors:
        args += ["--supplement", f"PMSL={path}"]
    assert main(args) == 1
    assert len(seen) == 1
    assert "Contract witness complete" in capsys.readouterr().err


def test_donor_cannot_replace_a_changed_fetched_payload(tmp_path):
    donor = tmp_path / "donor.grib2"
    donor.write_bytes(b"old bytes")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{_digest(donor)}  {donor.name}\n")
    donor.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="differs from the fetched source receipt"):
        hrrr_source_manifest([f"PMSL={donor}"], source_root=tmp_path,
                             fetched_manifest=manifest, output=tmp_path / "run" / "manifest")
    assert not (tmp_path / "run").exists()


def test_run_plan_resolves_donors_relative_to_the_plan(tmp_path):
    config = emit(tmp_path, source="hrrr")
    data = tmp_path / "source"
    data.mkdir()
    (data / "donor.grib2").write_bytes(b"explicit donor")
    raw = {"schema": runplan.PLAN_SCHEMA, "name": "donor", "route": "prepared",
           "config": {"path": config.name},
           "run_options": {"data_dir": "source", "supplement": ["PMSL=source/donor.grib2"]}}
    plan = runplan.build_plan(raw, source="relative donor plan", base_dir=tmp_path, sha256="0" * 64)
    assert plan.run_options["supplement"] == [f"PMSL={data / 'donor.grib2'}"]
    runplan.resolve_plan(plan, require_inputs=False)
    raw["run_options"]["supplement"] = "PMSL=source/donor.grib2"
    with pytest.raises(runplan.PlanError, match="must be a list"):
        runplan.build_plan(raw, source="invalid donor plan", base_dir=tmp_path, sha256="0" * 64)
