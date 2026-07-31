"""``gpuwm go``: the documented GFS chain, run without a human courier.

Five commands in a fixed order, three of them taking digests the
previous one printed.  ``go`` carries those values between stages by
reading the ARTIFACTS -- the manifest file, ``proof.json`` -- rather
than the printed prose, and runs the same commands a person would so
the provenance in the artifacts is the same either way.

The load-bearing tests here are the two equivalence gates.  ``go``'s
composed ``rw-wps`` line must equal the one ``gpuwm fetch
--author-front-door-manifest`` prints, and its composed forecast line
must equal the one the front door prints, because those printed
commands ARE the documented manual chain.  If either drifts, ``go``
stops being an automation of the documented route and becomes a second
route that happens to look like it.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from gpuwm import go_cli
from gpuwm.cli import main as cli_main


# ---------------------------------------------------------------------------
# Fixtures: a wizard-authored single-domain GFS config, made once
# ---------------------------------------------------------------------------

PROFILE = "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"


def _emit(tmp_path, name, *extra, source="gfs", ladder="12", profile=PROFILE):
    out = tmp_path / f"{name}.toml"
    argv = ["domain", "--point=35.3,-97.5", "--card", "24gb",
            "--ladder", ladder, "--source", source,
            "--cycle", "2026-07-29T18", "--hours", "6",
            "--out", str(out), *extra]
    if profile is not None:
        argv += ["--physics-profile", profile]
    assert cli_main(argv) == 0
    return out


@pytest.fixture(scope="module")
def gfs_config(tmp_path_factory):
    return _emit(tmp_path_factory.mktemp("gfs"), "myarea")


# ---------------------------------------------------------------------------
# Refusals: never half-orchestrate
# ---------------------------------------------------------------------------

def test_an_era5_case_data_config_is_refused_toward_gpuwm_run(tmp_path,
                                                              capsys):
    config = _emit(tmp_path, "era5", source="era5")
    assert cli_main(["go", str(config), "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "[case_data]" in error
    assert "gpuwm run" in error


def test_a_domain_tree_is_refused_toward_its_own_runner(tmp_path, capsys):
    config = _emit(tmp_path, "tree", ladder="12-3")
    assert cli_main(["go", str(config), "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "2 domains" in error
    assert "prepared_domain_tree_forecast.py" in error
    assert go_cli.MANUAL_CHAIN in error


def test_a_config_with_no_shipped_profile_is_refused_before_anything_runs(
        tmp_path, capsys):
    """The chain's LAST stage would refuse this; say so at the first."""

    config = _emit(tmp_path, "default_suite", profile=None)
    assert cli_main(["go", str(config), "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "matches none of the profiles" in error
    assert "--physics-profile" in error


def test_a_missing_config_is_a_refusal_not_a_traceback(tmp_path, capsys):
    assert cli_main(["go", str(tmp_path / "nope.toml"), "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "does not exist" in error
    assert "Traceback" not in error


def test_a_config_without_a_fetch_table_is_refused(tmp_path, capsys,
                                                   gfs_config):
    stripped = tmp_path / "nofetch.toml"
    text = gfs_config.read_text(encoding="utf-8")
    stripped.write_text(text.split("[fetch]")[0], encoding="utf-8")
    assert cli_main(["go", str(stripped), "--dry-run"]) == 2
    assert "no [fetch] table" in capsys.readouterr().err


@pytest.mark.parametrize("source", ["hrrr", "gdas", "era5"])
def test_only_the_documented_source_is_orchestrated(source):
    """Scope stated as a property of the other routes, not a preference."""

    assert go_cli.ORCHESTRATED_SOURCES == ("gfs",)
    assert source not in go_cli.ORCHESTRATED_SOURCES


# ---------------------------------------------------------------------------
# THE equivalence gates
# ---------------------------------------------------------------------------

def _flags(command: list[str]) -> dict:
    """``--flag -> value`` for one composed command."""

    out, index = {}, 0
    while index < len(command):
        token = command[index]
        if token.startswith("--"):
            following = (command[index + 1]
                         if index + 1 < len(command) else None)
            if following is not None and not following.startswith("--"):
                out[token] = following
                index += 2
                continue
            out[token] = True
        index += 1
    return out


def _printed_flags(lines) -> dict:
    """``--flag -> value`` out of a printed, pasteable command.

    Parenthetical notes are dropped before parsing.  The composer ends
    with "(--run-seconds and --history-interval-seconds default to the
    hash-bound experiment...)", which is prose ABOUT flags rather than
    flags -- and reading it as a flag is exactly the prose-scraping
    mistake ``go`` itself refuses to make.
    """

    command_lines = []
    for line in lines:
        text = str(line).rstrip()
        if not text.strip().startswith(("python", "rw-wps", "--", "gpuwm")):
            continue
        # Strip only a TRAILING continuation backslash; a backslash
        # inside a value is part of that value.
        command_lines.append(text[:-1] if text.endswith("\\") else text)
    return _flags(shlex.split(" ".join(command_lines)))


def _stage_a_fetched_directory(out: Path, gfs_config: Path, authority: Path):
    """The minimum on-disk state ``author_gfs_front_door_manifest`` needs.

    A real fetch manifest, a real series file, and real role files, so
    the composer under test runs its true path rather than a mocked one.
    """

    out.mkdir(parents=True, exist_ok=True)
    authority.mkdir(parents=True, exist_ok=True)
    (authority / "namelist.wps").write_bytes(
        gfs_config.with_suffix(".namelist.wps").read_bytes())
    (authority / "experiment.toml").write_bytes(gfs_config.read_bytes())
    (out / "gfs-series.tsv").write_text("0\tgfs.f000.grib2\t81\n",
                                        encoding="utf-8")
    (out / "gfs.f000.grib2").write_bytes(b"GRIB-stub")
    (out / "fetch-manifest.json").write_text(json.dumps({
        "schema": "gpuwm-fetch-manifest-v1", "source": "gfs",
        "cycle": "2026-07-29T18:00:00Z", "forecast_hours": [0],
        "files": [{"name": "gfs.f000.grib2", "role": "gfs-subset",
                   "forecast_hour": 0, "sha256": "0" * 64}],
    }), encoding="utf-8")


def test_the_prepare_stage_matches_what_fetch_prints(tmp_path, gfs_config):
    """go's rw-wps line IS the line the manual chain tells you to paste.

    Compared against the real ``author_gfs_front_door_manifest``, driven
    over real files, because a hand-written expectation here is a second
    opinion that can agree with neither.  An earlier draft of this test
    DID hand-write it, matched, and hid the fact that both were missing
    ``--physics-profile`` -- which the end-to-end run then found.

    go composes its command from the same inputs rather than parsing
    that printed line, so ``--explain`` moving the text cannot break the
    chain; this is what keeps the two from drifting apart anyway.
    """

    from gpuwm.fetch import author_gfs_front_door_manifest

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go")
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stub")
    _stage_a_fetched_directory(plan["data"], gfs_config, plan["authority"])

    printed: list[str] = []
    manifest, digest = author_gfs_front_door_manifest(
        out=plan["data"], bridge=bridge,
        wps_namelist=plan["authority"] / "namelist.wps",
        experiment_config=plan["authority"] / "experiment.toml",
        progress=printed.append)

    theirs = _printed_flags(
        [line for block in printed for line in str(block).splitlines()])
    mine = _flags(go_cli.prepare_command(
        plan, bridge, manifest=manifest, manifest_sha256=digest,
        cycle_stamp="2026-07-29_18:00:00",
        geog_root=Path(theirs["--geog-root"])))

    assert set(mine) == set(theirs), (
        f"go-only: {set(mine) - set(theirs)}, "
        f"printed-only: {set(theirs) - set(mine)}")
    # --output-root is the one deliberate difference: the printed line
    # suggests a child of the download directory, and go keeps its
    # authority/prepared/run trees together under its own root.  Every
    # other flag -- including the manifest digest and the profile -- has
    # to be identical, because those are the relay.
    for flag, value in theirs.items():
        if flag == "--output-root":
            continue
        assert str(mine[flag]).replace("\\", "/") == str(value), flag
    assert Path(mine["--output-root"]).name == "prepared"
    # The flag whose absence broke the documented chain.
    assert theirs["--physics-profile"] == PROFILE
    assert mine["--physics-profile"] == PROFILE


def test_the_printed_rw_wps_command_carries_the_physics_profile(tmp_path,
                                                                gfs_config):
    """The bug a real end-to-end found, pinned so it cannot come back.

    ``rw-wps``'s ``--physics-profile`` is spelled optional and is not:
    absent, `gpuwm/source_cli.py` substitutes ``WSM6_PROFILE_ID`` and
    compares the experiment's physics against THAT, so the pasted
    command refused every config except a wsm6-no-radiation one --
    including the Morrison profile FIRST-LIGHT.md's own worked example
    uses.  Observed live: "selected physics differs from profile
    'wsm6-ysu-mm5-noah-no-radiation-v1'".
    """

    from gpuwm.fetch import author_gfs_front_door_manifest

    data = tmp_path / "data"
    authority = tmp_path / "authority"
    _stage_a_fetched_directory(data, gfs_config, authority)
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stub")

    printed: list[str] = []
    author_gfs_front_door_manifest(
        out=data, bridge=bridge,
        wps_namelist=authority / "namelist.wps",
        experiment_config=authority / "experiment.toml",
        progress=printed.append)
    text = "\n".join(str(block) for block in printed)
    assert f"--physics-profile {PROFILE}" in text


def test_a_config_matching_no_profile_prints_no_profile_flag(tmp_path):
    """Silence beats inventing a flag the runner would reject.

    A config matching no shipped profile cannot be rescued by naming
    one here; the front door explains that case itself after
    preparation, and a guessed flag would only move the refusal earlier
    without making it truer.
    """

    from gpuwm.fetch import author_gfs_front_door_manifest

    config = _emit(tmp_path, "default_suite", profile=None)
    data = tmp_path / "data"
    authority = tmp_path / "authority"
    _stage_a_fetched_directory(data, config, authority)
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stub")

    printed: list[str] = []
    author_gfs_front_door_manifest(
        out=data, bridge=bridge,
        wps_namelist=authority / "namelist.wps",
        experiment_config=authority / "experiment.toml",
        progress=printed.append)
    text = "\n".join(str(block) for block in printed)
    assert "--physics-profile" not in text
    # ...and the rest of the command is still printed in full.
    assert "--source-manifest-sha256" in text and "--output-root" in text


def test_the_forecast_stage_matches_what_the_front_door_prints(tmp_path,
                                                               gfs_config):
    """go's forecast line IS the line rw-wps tells you to copy.

    Compared against ``gfs_direct.prepared_forecast_next_command`` -- the
    function that composes the printed command -- driven by a proof
    document of the shape the front door writes.  Same proof, same
    command, or ``go`` has drifted off the documented route.
    """

    from gpuwm.gfs_direct import prepared_forecast_next_command

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go")
    plan["prepared"].mkdir(parents=True)
    proof = {
        "input_manifest_sha256": "a" * 64,
        "prepared_cache": {"content_sha256": "b" * 64},
        "physics": {"profile": PROFILE},
    }
    proof_path = plan["prepared"] / "proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    digests = go_cli.proof_digests(plan["prepared"])
    assert digests["source_manifest"] == "a" * 64
    assert digests["prepared_content"] == "b" * 64

    mine = _flags(go_cli.forecast_command(plan, digests))
    theirs = _printed_flags(prepared_forecast_next_command(
        proof, output_root=plan["prepared"],
        experiment_config=plan["authority"] / "experiment.toml",
        wps_namelist=plan["authority"] / "namelist.wps", source="gfs"))

    # The composer prints --outdir as a sibling of the preparation; go
    # owns its own tree.  Every OTHER flag must match exactly, including
    # all three digests, which is the whole point.
    for flag in ("--source", "--prepared-root", "--proof-sha256",
                 "--source-manifest-sha256", "--prepared-content-sha256",
                 "--experiment-config", "--wps-namelist",
                 "--physics-profile", "--io-mode"):
        assert flag in mine and flag in theirs, flag
        assert str(mine[flag]).replace("\\", "/") == \
            str(theirs[flag]).replace("\\", "/"), flag
    assert set(mine) == set(theirs)


def test_the_proof_digest_is_read_not_recomputed(tmp_path, gfs_config):
    """go transports the digests; it must never invent one.

    ``--proof-sha256`` is a hash OF the proof file, and the other two
    are values carried INSIDE it.  If go computed the inner two itself,
    the runner's comparison would be go checking its own arithmetic
    instead of checking the front door's claim.
    """

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    proof = {"input_manifest_sha256": "a" * 64,
             "prepared_cache": {"content_sha256": "b" * 64}}
    path = prepared / "proof.json"
    path.write_text(json.dumps(proof), encoding="utf-8")

    from gpuwm.fetch import sha256_file

    digests = go_cli.proof_digests(prepared)
    assert digests["proof"] == sha256_file(path)
    assert digests["source_manifest"] == "a" * 64
    assert digests["prepared_content"] == "b" * 64


def test_a_hierarchy_proof_is_refused_toward_the_tree_runner(tmp_path):
    """A proof with no single prepared-cache identity is a tree product."""

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "proof.json").write_text(
        json.dumps({"input_manifest_sha256": "a" * 64}), encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal, match="prepared_domain_tree"):
        go_cli.proof_digests(prepared)


def test_a_missing_proof_is_refused_rather_than_guessed(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with pytest.raises(go_cli.GoRefusal, match="nothing to bind against"):
        go_cli.proof_digests(prepared)


# ---------------------------------------------------------------------------
# Stage failure: replay, and stop
# ---------------------------------------------------------------------------

class _FakeCompleted:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.mark.parametrize("failing_index, failing_label", [
    (0, "authority"),
    (1, "fetch"),
    (2, "manifest"),
])
def test_a_failing_stage_replays_its_output_and_stops(
        tmp_path, capsys, monkeypatch, gfs_config, failing_index,
        failing_label):
    """Tested at more than one stage, because "stops" is the contract.

    Every later stage consumes the previous one's output, so unlike
    ``gpuwm setup`` -- whose steps are independent and all run -- a
    failure here must end the chain.  The failing stage's whole output
    is replayed with or without ``--explain``: the reason a stage
    refused is the one thing an orchestrator must never summarize.
    """

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) - 1 == failing_index:
            return _FakeCompleted(
                3, stdout="line one of the real diagnosis\n",
                stderr="line two, naming the actual problem\n")
        return _FakeCompleted(0, stdout="chatty success\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    rc = cli_main(["go", str(gfs_config), "--outdir", str(tmp_path / "go")])
    printed = capsys.readouterr().out
    assert rc == 3
    assert f"FAILED  {failing_label}" in printed
    # Verbatim, without --explain, because this is the reason.
    assert "line one of the real diagnosis" in printed
    assert "line two, naming the actual problem" in printed
    assert "nothing after it ran" in printed
    # And nothing after it ran.
    assert len(calls) == failing_index + 1
    # A succeeding stage stays quiet.
    assert "chatty success" not in printed


def test_a_succeeding_chain_reports_one_line_per_stage(tmp_path, capsys,
                                                       monkeypatch,
                                                       gfs_config):
    plan_root = tmp_path / "go"

    def fake_run(command, **kwargs):
        # Materialize the artifacts the relay reads back.
        if "--author-front-door-manifest" in command:
            manifest = plan_root / "data" / "gfs-input-manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}", encoding="utf-8")
        if "--output-root" in command:
            prepared = plan_root / "prepared"
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "proof.json").write_text(json.dumps({
                "input_manifest_sha256": "a" * 64,
                "prepared_cache": {"content_sha256": "b" * 64}}),
                encoding="utf-8")
        if "--io-mode" in command:
            # The forecast stage publishes history frames, and the render
            # stage enumerates them: `gpuwm render` takes FILES, and
            # handing it the directory is what a real run died on after
            # the forecast had already succeeded.  A forecast that
            # published nothing is a skipped render, not a rendered
            # nothing, so this fake has to publish one.
            frames = plan_root / "run" / "wrfout"
            frames.mkdir(parents=True, exist_ok=True)
            (frames / "wrfout_d01_2026-07-29_18_00_00").write_text(
                "", encoding="utf-8")
        return _FakeCompleted(0, stdout="detail nobody asked for\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)

    rc = cli_main(["go", str(gfs_config), "--outdir", str(plan_root)])
    printed = capsys.readouterr().out
    assert rc == 0
    for label in ("authority", "fetch", "manifest", "prepare", "forecast",
                  "render"):
        assert f"  ok      {label}" in printed
    assert "detail nobody asked for" not in printed
    # No "next:" block, and that absence is the point.
    #
    # This chain used to end by printing `gpuwm render ...` for the
    # reader to paste, which is the same baton pass `go` exists to
    # remove: a command printed instead of run reads as "it stopped".
    # Rendering is the sixth stage now, so a finished `go` leaves
    # pictures, not homework.
    assert "next:" not in printed
    assert "go: rendered " in printed


def test_explain_replays_every_stage(tmp_path, capsys, monkeypatch,
                                     gfs_config):
    plan_root = tmp_path / "go"

    def fake_run(command, **kwargs):
        if "--author-front-door-manifest" in command:
            manifest = plan_root / "data" / "gfs-input-manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}", encoding="utf-8")
        if "--output-root" in command:
            prepared = plan_root / "prepared"
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "proof.json").write_text(json.dumps({
                "input_manifest_sha256": "a" * 64,
                "prepared_cache": {"content_sha256": "b" * 64}}),
                encoding="utf-8")
        return _FakeCompleted(0, stdout="the full receipt\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    assert cli_main(["go", str(gfs_config), "--outdir", str(plan_root),
                     "--explain"]) == 0
    assert "the full receipt" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------

def test_dry_run_prints_five_filled_in_commands_and_runs_nothing(
        tmp_path, capsys, monkeypatch, gfs_config):
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not run anything")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    assert cli_main(["go", str(gfs_config), "--dry-run",
                     "--outdir", str(tmp_path / "go")]) == 0
    printed = capsys.readouterr().out
    for step in ("1. authority", "2. fetch", "3. manifest", "4. prepare",
                 "5. forecast"):
        assert step in printed
    # Filled in from the config, not left as placeholders.
    assert "--physics-profile " + PROFILE in printed
    assert "--cycle 2026-07-29T18" in printed
    assert "--hours 6" in printed
    # The two values that cannot exist yet name the file they come from
    # rather than showing a plausible-looking hash.
    assert "after step 3" in printed
    assert "proof.json" in printed


def test_the_fetch_area_keeps_its_equals_form_for_a_negative_box(
        tmp_path, capsys, monkeypatch, gfs_config):
    """A leading-minus area is an option token unless it is joined.

    The same rule the wizard's printed command follows; getting it wrong
    here would make the fetch stage fail with "expected one argument"
    on every western-hemisphere domain.
    """

    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    assert cli_main(["go", str(gfs_config), "--dry-run",
                     "--outdir", str(tmp_path / "go")]) == 0
    printed = capsys.readouterr().out
    assert "--area=-" in printed
    assert "--area -" not in printed


def test_the_cwd_relative_fetch_out_key_is_not_trusted(tmp_path, gfs_config):
    """`[fetch].out` is written relative to the wizard's cwd, not the file.

    A config emitted from one directory records a download path that
    means something else read from another -- six `..` hops walked past
    the drive root and clamped at `C:/AppData/...` in the run that found
    this.  The table calls itself advisory; go honours the values the
    domain was SIZED against and owns where the bytes land.
    """

    import tomllib

    recorded = tomllib.loads(
        gfs_config.read_text(encoding="utf-8"))["fetch"]["out"]
    assert recorded.startswith("..") or Path(recorded).is_absolute()

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go")
    assert plan["data"] == tmp_path / "go" / "data"

    override = tmp_path / "already-fetched"
    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go",
                                   data_dir=override)
    assert plan["data"] == override


def test_a_second_go_into_the_same_tree_is_refused_in_its_own_words(
        tmp_path, capsys, gfs_config, monkeypatch):
    """Re-running the command is the second thing anyone does.

    Every stage is create-only, so the first one refuses a tree an
    earlier run owns -- correctly, since merging two runs would publish
    receipts describing neither.  But its message names
    ``--output-directory``, a flag nobody typed to get here, and it
    arrives after a stage has already been spent reaching it.  `go`
    answers first, in the vocabulary of the command that was actually
    run.
    """

    plan_root = tmp_path / "go"
    (plan_root / "authority").mkdir(parents=True)

    def fail_if_called(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("a stage ran after an up-front refusal")

    monkeypatch.setattr(subprocess, "run", fail_if_called)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    assert cli_main(["go", str(gfs_config), "--outdir", str(plan_root)]) == 2
    printed = capsys.readouterr()
    message = printed.out + printed.err
    assert "already exists" in message
    assert "--outdir" in message
    assert "Traceback" not in message
