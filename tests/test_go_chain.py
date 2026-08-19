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
import itertools
import shutil
import subprocess
from pathlib import Path

import pytest

from gpuwm import go_cli, run_stamp
from gpuwm.cli import main as cli_main


#: Flags whose value is a path INSIDE this run's tree, in the order the
#: stages are composed.  A stage fake reads the tree off its own command
#: rather than assuming one: every run claims its own timestamped folder
#: under ``--outdir`` (``gpuwm.run_stamp``), so a hard-coded
#: ``<outdir>/prepared`` is a directory no stage writes to.
_STAGE_TREE_FLAGS = ("--output-directory", "--output-root", "--outdir",
                     "--prepared-root", "--render-dir")


def _stage_root(command) -> Path:
    """The run root a composed stage command points into."""

    for flag in _STAGE_TREE_FLAGS:
        if flag in command:
            return Path(command[command.index(flag) + 1]).parent
    raise AssertionError(
        f"no run-tree flag in {command!r}; the fake cannot tell which "
        "run folder this stage was composed for")


# ---------------------------------------------------------------------------
# Fixtures: a wizard-authored single-domain GFS config, made once
# ---------------------------------------------------------------------------

PROFILE = "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"


def _emit(tmp_path, name, *extra, source="gfs", ladder="12", profile=PROFILE,
          point="35.3,-97.5"):
    out = tmp_path / f"{name}.toml"
    argv = ["domain", f"--point={point}", "--card", "24gb",
            "--ladder", ladder, "--source", source,
            "--cycle", "2026-07-29T18", "--hours", "6",
            "--out", str(out), *extra]
    if profile is not None:
        argv += ["--physics-profile", profile]
    assert cli_main(argv) == 0
    return out


def _emit_unnamed_suite(tmp_path, name, *extra, **kwargs):
    """A config whose physics matches NO shipped profile.

    The chain still has an unnamed-suite branch -- ``plan["profile"]``
    is ``None``, ``--physics-profile`` is omitted from every composed
    command, and the verification status is stated instead -- and the
    two tests below are what hold it.  Until 2026-08-06 the wizard's own
    ``--physics-profile``-less emission produced such a config, so they
    built their fixture that way; 1.7.1 bound the gfs/era5 default to
    the certified Morrison profile (the nocturnal-radiation directive),
    which is a shipped profile, so that emission no longer reaches the
    branch and no door emits the unnamed suite any more.

    The branch itself is untouched and still reachable -- a hand-written
    config, an imported namelist, or the unshipped
    ``DEFAULT_SUITE_PHYSICS`` suite, which
    :data:`gpuwm.domain_wizard.DEFAULT_PHYSICS_PROFILE`'s own docstring
    records as "reachable programmatically" -- so the fixture asks the
    wizard for that suite directly rather than hand-writing a stand-in
    that could agree with neither the emitter nor the loader.  Its
    ``[shared]`` block is the suite the pre-1.7.1 default emitted, key
    for key; only the emitted HEADER differs, because 1.7.1 states
    nocturnal validity on every file it writes.

    The assertion below is the fixture's own proof: a helper that
    quietly stopped producing an unmatched suite would leave these two
    tests passing against the branch they were written to leave.
    """

    from gpuwm import domain_wizard

    bound = domain_wizard.DEFAULT_PHYSICS_PROFILE
    domain_wizard.DEFAULT_PHYSICS_PROFILE = None
    try:
        config = _emit(tmp_path, name, *extra, profile=None, **kwargs)
    finally:
        domain_wizard.DEFAULT_PHYSICS_PROFILE = bound
    # The fixture is only a fixture if it really matches nothing.
    from gpuwm.experiment import load_experiment
    from gpuwm.physics_compat import identify_single_domain_profile
    assert identify_single_domain_profile(
        load_experiment(config).root.run) is None
    return config


@pytest.fixture(scope="module")
def gfs_config(tmp_path_factory):
    return _emit(tmp_path_factory.mktemp("gfs"), "myarea")


#: What the pinned card below reports free.  Comfortably above this
#: file's fixture config (a 24 GiB-card ladder, ~19.94 GiB forecast peak
#: envelope) so the gate's verdict here is "fits", deterministically.
_PINNED_FREE_BYTES = 30 * 1024 ** 3


@pytest.fixture(autouse=True)
def _a_card_whose_free_vram_this_file_decides(monkeypatch):
    """Pin the ONE number the outside world moves in the memory gate.

    ``memory_gate`` refuses when the binding phase exceeds the free VRAM
    it measures *right now*, which is correct and is the whole point of
    gating before the fetch.  It also means that on a shared card every
    test below that drives the real chain -- the stage-failure replay,
    the one-line-per-stage report, ``--explain`` -- turns red whenever
    another run happens to hold the card, because the chain stops at a
    genuine refusal instead of reaching the stage behaviour under test.
    Proven: with 8 GiB reported free, five tests in this file fail; with
    the card idle they pass.  A test suite must not have that reading.

    So the card's free VRAM is pinned here and everything else in the
    gate stays real -- the phase estimates, the reserve policy, the
    verdict sentence, the refuse/warn arithmetic all run as shipped.
    The gate's own tests below substitute ``memory_gate`` wholesale from
    inside the test body, after this fixture, so they still choose their
    own numbers and are unaffected.

    The pin sits on the gate's subprocess probe seam: the gate asks the
    card nothing in-process (that stood up a CUDA context the go process
    then held for the whole chain as its progress printer), so the ONE
    place the outside world enters is ``device_memory_probe_subprocess``.
    Pinning it also makes this file identical on every box -- with or
    without a card, busy or idle -- where the old memGetInfo pin still
    left the no-device path machine-dependent.  ``profile: None`` prices
    the non-pool terms against the reference profile, deterministically.
    """

    from gpuwm.core import preflight

    monkeypatch.setattr(
        preflight, "device_memory_probe_subprocess",
        lambda **_kwargs: {"free_bytes": _PINNED_FREE_BYTES,
                           "total_bytes": 32 * 1024 ** 3,
                           "profile": None})


def test_the_pinned_card_is_what_the_gate_reads(gfs_config, tmp_path):
    """The fixture above is load-bearing; prove it reaches the gate."""

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "out")
    gate = go_cli.memory_gate(plan)
    assert gate["free_bytes"] == _PINNED_FREE_BYTES
    assert not gate["refuse"]


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
    # The runner is named by its INSTALLED spelling: a wheel has no
    # tools/ directory, so a tools/-path pointer names a file the
    # reader provably does not have.
    assert "gpuwm-prepared-tree-forecast" in error
    assert go_cli.MANUAL_CHAIN in error
    # The one-command re-emit remedy is the wizard's own default now,
    # so the remedy says so instead of trailing off in "...".
    assert "without --ladder" in error
    assert "..." not in error.split("remedy:")[1]


def test_the_default_emission_is_what_the_default_runner_accepts(
        tmp_path, capsys):
    """Default wizard output piped to the default runner composes.

    The 4090 user-zero stress run (2026-08-03) followed the obvious
    path: `gpuwm domain --point ... --card 24gb --source gfs` with no
    --ladder, then `gpuwm go` on the file it wrote -- and go refused
    it, because the flags door's --ladder default was `auto`, the
    deepest tree that fits.  The interactive door had already ruled on
    this exact seam (domain_interactive.DEFAULT_LADDER = "12": "two
    features that do not compose is not a feature"); this test pins
    the same ruling onto the flags door.

    Real emission, real plan reader, no profile flag: the default
    suite runs as written (owner ruling 2026-07-31), so nothing here
    needs one.
    """

    out = tmp_path / "default.toml"
    assert cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                     "--source", "gfs", "--cycle", "2026-07-29T18",
                     "--hours", "6", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    # The wizard's own closing block names the runner its default
    # emission is for -- the one-command chain, not the tree route.
    assert "gpuwm go " in printed

    plan = go_cli.plan_from_config(out)
    assert plan["source"] == "gfs"
    assert cli_main(["go", str(out), "--dry-run"]) == 0


def test_a_config_with_no_shipped_profile_runs_with_status_stated(
        tmp_path, capsys):
    """Converted (owner ruling 2026-07-31): the chain's last stage runs
    the config's own suite as written, so the first stage plans it
    instead of refusing it -- with the verification status stated in one
    sentence and no --physics-profile invented anywhere."""

    config = _emit_unnamed_suite(tmp_path, "default_suite")
    assert cli_main(["go", str(config), "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "supported, not yet WRF-verified" in captured.out
    assert "--physics-profile" not in captured.out

    plan = go_cli.plan_from_config(config, outdir=tmp_path / "go")
    assert plan["profile"] is None
    for command in (
            go_cli.authority_command(plan),
            go_cli.forecast_command(plan, {
                "proof": "a" * 64, "source_manifest": "b" * 64,
                "prepared_content": "c" * 64}),
    ):
        assert "--physics-profile" not in command


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

    config = _emit_unnamed_suite(tmp_path, "default_suite")
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


def test_go_forwards_a_profile_only_when_the_whole_config_is_it(tmp_path):
    """The derivation behind ``--physics-profile``, fixed 2026-08-09.

    ``plan_from_config`` used to derive the forwarded profile from the
    ROOT domain alone, while stage 1's drift refusal reads every
    ``[[domain]]`` table -- so on the wizard's own ``--ladder`` trees
    (root = the profile, nests deliberately departing: ``cu_physics =
    0`` below the gray zone, tighter ``radt``, the ``diff_6th_factor``
    ladder) the chain composed a stage-1 command guaranteed to refuse
    its own config.  `gpuwm run-plan`'s prepared route dispatches
    exactly this shape (``go_main(..., allow_tree=True)``).  Before the
    stage-1 refusal existed the same derivation was WORSE, not fine: the
    materializer silently flattened those nests onto the profile, which
    is the ledger #90 defect itself.

    The derivation now asks the materializer's own conflict predicate:
    a config the profile contradicts nowhere carries the assertion end
    to end, and one that deliberately says more runs as its own suite,
    unnamed (owner ruling 2026-07-31), with the verification status
    stated in the receipts.
    """

    tree = _emit(tmp_path, "tree", ladder="12-3")
    plan = go_cli.plan_from_config(tree, outdir=tmp_path / "go",
                                   allow_tree=True)
    assert plan["profile"] is None
    assert "--physics-profile" not in go_cli.authority_command(plan)
    # And stage 1 ACCEPTS what go now composes, publishing the config's
    # nest physics unchanged -- the whole point of omitting the flag.
    from gpuwm.prepared_single_domain_forecast import (
        _render_materialized_experiment)
    _rendered, exp, _receipt = _render_materialized_experiment(
        tree.read_text(encoding="utf-8"), source="gfs", profile=None)
    assert int(exp.domains[1].run.cu_physics) == 0

    # Agreement-driven, not tree-driven: the same tree with its nest
    # brought onto the profile's values forwards the assertion again.
    agreeing = tmp_path / "agreeing.toml"
    agreeing.write_text(
        tree.read_text(encoding="utf-8")
        .replace("radt = 3.0", "radt = 12.0")
        .replace("cu_physics = 0", "cu_physics = 1")
        .replace("diff_6th_factor = 0.1\n", "diff_6th_factor = 0.12\n"),
        encoding="utf-8")
    shutil.copy(tree.with_suffix(".namelist.wps"),
                agreeing.with_suffix(".namelist.wps"))
    agreeing_plan = go_cli.plan_from_config(
        agreeing, outdir=tmp_path / "go-agree", allow_tree=True)
    assert agreeing_plan["profile"] == PROFILE

    # The single-domain emission was never affected and still binds.
    single = _emit(tmp_path, "single")
    assert go_cli.plan_from_config(
        single, outdir=tmp_path / "go-single")["profile"] == PROFILE


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

    # `--progress-format` is the second allowed difference, and the last.
    # The printed line is for a PERSON at a terminal, so it leaves the
    # runner's default in place and the WRF-shaped `Timing for main:`
    # lines land on their screen -- that is the reason to run the stage
    # by hand.  `go` owns the runner's stdout instead (its subprocess arm
    # would buffer tens of megabytes of discarded per-step lines) so it
    # asks for jsonl.  Subtracted rather than excused: the set equality
    # below still has to hold exactly, so no THIRD flag can drift in
    # behind this one.
    assert "--progress-format" in mine, (
        "go no longer sets the progress transport; drop this subtraction "
        "rather than leaving it to hide a real difference")
    assert "--progress-format" not in theirs, (
        "the printed line now sets a progress transport too, so these two "
        "should simply be compared directly")
    assert set(mine) - {"--progress-format"} == set(theirs)


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

class _FakePopen:
    """A ``subprocess.Popen`` double over this file's ``fake_run`` doubles.

    ``_run_stage`` spells ``subprocess.run`` out as ``Popen`` +
    ``communicate`` -- byte for byte what ``run`` does internally --
    because the interrupt path has to be able to NAME the pid of the
    stage gpuwm was waiting on without signalling it.  The seam these
    tests patch moved with it; nothing else about them changed.
    """

    _pids = itertools.count(424242)

    def __init__(self, completed):
        self._completed = completed
        self.pid = next(self._pids)
        self.returncode = None

    def communicate(self):
        self.returncode = self._completed.returncode
        return self._completed.stdout, self._completed.stderr


def _popen_double(fake_run):
    """Adapt a ``(command, **kwargs) -> CompletedProcess`` double."""

    def factory(command, **kwargs):
        return _FakePopen(fake_run(command, **kwargs))

    return factory


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
        tmp_path, capsys, monkeypatch, gfs_config, staged_geog, failing_index,
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

    monkeypatch.setattr(subprocess, "Popen", _popen_double(fake_run))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    rc = cli_main(["go", str(gfs_config), "--outdir", str(tmp_path / "go"),
                   "--geog-root", str(staged_geog)])
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


def test_a_failed_chain_says_what_it_left_on_disk(
        tmp_path, capsys, monkeypatch, gfs_config, staged_geog):
    """D-05.  A failed prepare leaves a scratch tree of a few hundred MB
    and said nothing about it anywhere.

    The interrupted arm of this same try/except has always told the
    reader what is on disk; the failure arm returned the code in
    silence.  Nothing is deleted -- the tree is the evidence of what
    failed -- but it is now named, measured, and declared safe to remove.
    """

    root = tmp_path / "go"

    def fake_run(command, **kwargs):
        # Into THIS RUN's tree, read off the stage's own command rather
        # than assumed: every run claims its own timestamped folder under
        # --outdir, and a fake that wrote to the case root would be
        # measuring a directory no stage uses.
        run_root = _stage_root(command)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "scratch.bin").write_bytes(b"x" * 3_000_000)
        return _FakeCompleted(3, stdout="the real diagnosis\n")

    monkeypatch.setattr(subprocess, "Popen", _popen_double(fake_run))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    rc = cli_main(["go", str(gfs_config), "--outdir", str(root),
                   "--geog-root", str(staged_geog)])
    captured = capsys.readouterr()
    assert rc == 3
    assert str(root) in captured.err
    assert "partial tree with no certification capsule" in captured.err
    assert "MiB" in captured.err
    assert "safe to remove" in captured.err
    # and it really is still there: the note must not have tidied away
    # the evidence it is describing
    run_root = run_stamp.latest(root)
    assert run_root is not None
    assert (run_root / "scratch.bin").exists()


def test_outdir_and_data_dir_naming_one_directory_is_refused(
        tmp_path, capsys, gfs_config):
    """E-07.  The default puts the download inside the run root, so only
    an explicit --data-dir can make the two equal -- and when it does,
    the create-only run directory and the meant-to-be-reused download
    cache become the same directory, so the next run refuses against the
    reader's own cache."""

    shared = tmp_path / "both"
    shared.mkdir()
    rc = cli_main(["go", str(gfs_config), "--outdir", str(shared),
                   "--data-dir", str(shared)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "cannot be both" in err
    assert "Traceback" not in err

    # the negative control: different directories still plan fine
    plan = go_cli.plan_from_config(
        gfs_config, outdir=tmp_path / "run", data_dir=tmp_path / "dl")
    assert plan["root"] != plan["data"]


def test_a_succeeding_chain_reports_one_line_per_stage(tmp_path, capsys,
                                                       monkeypatch,
                                                       staged_geog,
                                                       gfs_config):
    plan_root = tmp_path / "go"

    def fake_run(command, **kwargs):
        # Materialize the artifacts the relay reads back.
        if "--author-front-door-manifest" in command:
            manifest = plan_root / "data" / "gfs-input-manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}", encoding="utf-8")
        if "--output-root" in command:
            prepared = Path(command[command.index("--output-root") + 1])
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
            frames = Path(
                command[command.index("--outdir") + 1]) / "wrfout"
            frames.mkdir(parents=True, exist_ok=True)
            (frames / "wrfout_d01_2026-07-29_18_00_00").write_text(
                "", encoding="utf-8")
        return _FakeCompleted(0, stdout="detail nobody asked for\n")

    monkeypatch.setattr(subprocess, "Popen", _popen_double(fake_run))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)

    rc = cli_main(["go", str(gfs_config), "--outdir", str(plan_root),
                   "--geog-root", str(staged_geog)])
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


def test_a_passing_stages_note_survives_the_output_capture(
        tmp_path, capsys, monkeypatch, staged_geog, gfs_config):
    """A skipped product must not become a silent success one level up.

    `gpuwm render` says, in one sentence, when a frame's declared inputs
    are absent and a product was therefore not drawn -- it exists so
    that skip is never silent.  `go` captures every stage's output and
    prints `ok render`, so without this the sentence is produced and
    swallowed, and the chain re-creates the silence the render change
    removed.

    `warning:` already survived; `note:` is this tree's word for "true,
    worth knowing, not a fault", and it survives on the same terms.  The
    third assertion is the boundary: ordinary chatter still does not.
    """
    plan_root = tmp_path / "go"

    def fake_run(command, **kwargs):
        if "--author-front-door-manifest" in command:
            manifest = plan_root / "data" / "gfs-input-manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}", encoding="utf-8")
        if "--output-root" in command:
            prepared = Path(command[command.index("--output-root") + 1])
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "proof.json").write_text(json.dumps({
                "input_manifest_sha256": "a" * 64,
                "prepared_cache": {"content_sha256": "b" * 64}}),
                encoding="utf-8")
        return _FakeCompleted(0, stdout=(
            "note: render skipped 1 product render(s) (refl)\n"
            "warning: something to know\n"
            "render: /png/t2.png\n"))

    # Popen, not run: `_run_stage` spells subprocess.run out as Popen +
    # communicate so the interrupt path can NAME the child's pid.  This
    # test was written against the older seam and merged forward
    # unchanged -- a patch on `run` is simply inert now, so the stages
    # ran for real, fetched over the network, and died on the bridge
    # stub.  Same double as every sibling in this file.
    monkeypatch.setattr(subprocess, "Popen", _popen_double(fake_run))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    assert cli_main(["go", str(gfs_config), "--outdir", str(plan_root),
                     "--geog-root", str(staged_geog)]) == 0
    printed = capsys.readouterr().out
    assert "note: render skipped 1 product render(s) (refl)" in printed
    assert "warning: something to know" in printed
    assert "/png/t2.png" not in printed, \
        "a passing stage's ordinary output still stays behind --explain"


def test_explain_replays_every_stage(tmp_path, capsys, monkeypatch,
                                     staged_geog,
                                     gfs_config):
    plan_root = tmp_path / "go"

    def fake_run(command, **kwargs):
        if "--author-front-door-manifest" in command:
            manifest = plan_root / "data" / "gfs-input-manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}", encoding="utf-8")
        if "--output-root" in command:
            prepared = Path(command[command.index("--output-root") + 1])
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "proof.json").write_text(json.dumps({
                "input_manifest_sha256": "a" * 64,
                "prepared_cache": {"content_sha256": "b" * 64}}),
                encoding="utf-8")
        return _FakeCompleted(0, stdout="the full receipt\n")

    monkeypatch.setattr(subprocess, "Popen", _popen_double(fake_run))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    assert cli_main(["go", str(gfs_config), "--outdir", str(plan_root),
                     "--geog-root", str(staged_geog),
                     "--explain"]) == 0
    assert "the full receipt" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------

def test_dry_run_prints_five_filled_in_commands_and_runs_nothing(
        tmp_path, capsys, monkeypatch, gfs_config):
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not run anything")

    monkeypatch.setattr(subprocess, "Popen", explode)
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
    # A point low enough that the forcing box's first corner is a
    # southern latitude whatever the sizing model of the day emits: this
    # test is about the "=" joining rule, not about how many cells fit.
    southern = _emit(tmp_path, "southern", point="12.0,-97.5")
    assert cli_main(["go", str(southern), "--dry-run",
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

    Every stage is create-only, so a chain refuses a tree an earlier run
    owns -- correctly, since merging two runs would publish receipts
    describing neither.  The runner's own message names
    ``--output-directory``, a flag nobody typed to get here, and it
    arrives after a stage has already been spent reaching it, so `go`
    answers first in the vocabulary of the command that was run.

    What CHANGED in 2.5.0 is which command reaches it.  A bare re-run no
    longer does: each run claims its own timestamped folder under
    ``--outdir``, which is what the collaborator running this in a loop
    asked for.  The refusal now guards the two ways a caller can still
    put two runs in one tree -- naming an existing run folder, and
    ``--run-stamp off`` -- and this pins the second, with its wording.
    """

    plan_root = tmp_path / "go"
    (plan_root / "authority").mkdir(parents=True)

    def fail_if_called(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("a stage ran after an up-front refusal")

    monkeypatch.setattr(subprocess, "Popen", fail_if_called)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    assert cli_main(["go", str(gfs_config), "--outdir", str(plan_root),
                     "--run-stamp", "off"]) == 2
    printed = capsys.readouterr()
    message = printed.out + printed.err
    assert "already exists" in message
    assert "--outdir" in message
    assert "Traceback" not in message


def test_a_bare_second_go_claims_its_own_folder_rather_than_refusing(
        tmp_path, gfs_config):
    """The complaint, answered: two runs of one config, two trees.

    The stages stay create-only; what changed is that a run no longer
    walks into the last one's directory to find out.
    """

    plan_root = tmp_path / "go"
    (plan_root / "authority").mkdir(parents=True)   # a previous flat run
    first = go_cli.claim_run_root(
        go_cli.plan_from_config(gfs_config, outdir=plan_root))
    second = go_cli.claim_run_root(
        go_cli.plan_from_config(gfs_config, outdir=plan_root))
    assert first["root"] != second["root"]
    for plan in (first, second):
        assert plan["root"].parent == plan_root
        assert not plan["authority"].exists(), (
            "a freshly claimed run folder already holds an authority "
            "tree, so the create-only refusal would fire on it")


# ---------------------------------------------------------------------------
# The memory gate: before the download, never after it
# ---------------------------------------------------------------------------

def test_the_memory_gate_prices_both_phases_and_names_the_binding_one(
        gfs_config, tmp_path):
    """`go` must know what this run costs BEFORE `gpuwm fetch` runs.

    The bug this closes: the only estimate anyone computed described the
    forecast, so a domain sized to a 12 GB card downloaded 81 GFS files
    and then died in preprocessing at 15.82 GB.  Both phases are priced
    here, from the config alone, with no device and no download.
    """
    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "out")
    gate = go_cli.memory_gate(plan)
    phases = gate["phases"]
    assert phases.ingest_priced
    assert phases.ingest.n_forcing_times >= 2
    # ONE resident forcing time: the adapters build the start time last
    # (gpuwm/ingest/lateral_bc.py:start_last_forcing_order) so nothing is
    # held across the loop.  The gate has to price what the adapters do.
    assert phases.ingest.resident_times == 1
    assert phases.binding_phase in ("forecast", "ingest")
    assert phases.binding_phase in gate["verdict"]
    assert "forecast" in gate["verdict"] and "ingest" in gate["verdict"]


def test_the_memory_gate_refuses_ahead_of_the_fetch_stage(gfs_config,
                                                          tmp_path,
                                                          monkeypatch,
                                                          capsys):
    """A refusal must land with the download still un-started.

    Every stage command is replaced by a recorder, so if `fetch` appears
    in the record at all the gate ran too late.
    """
    ran: list[str] = []

    def _record(label, command, **kwargs):
        ran.append(label)

    monkeypatch.setattr(go_cli, "_run_stage", _record)
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    monkeypatch.setattr(
        go_cli, "memory_gate",
        lambda plan, **kw: {
            "verdict": "preprocessing (ingest) is the memory-binding phase "
                       "at 40.00 GiB peak envelope",
            "refuse": True, "warn": True, "free_bytes": 8 * 1024 ** 3,
        })
    args = _args(gfs_config, tmp_path / "out")
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.go_main(args)
    assert ran == []
    message = str(refusal.value)
    assert "BEFORE the fetch stage" in message
    assert "memory-binding phase" in message
    # The measured free-VRAM honesty stays; the remedy must be
    # REACHABLE: the 3080 walk followed `gpuwm domain --vram-gib <free>`
    # verbatim and was refused at every grid size, because the flag
    # names a card and the number fed to it was a free-VRAM figure.
    assert "8.00 GiB free right now" in message
    assert "gpuwm domain" in message
    assert "--vram-gib" not in message
    assert "--no-memory-gate" in message


def test_unstaged_geography_is_refused_ahead_of_the_fetch_stage(
        gfs_config, tmp_path, monkeypatch):
    """The first-run wall, measured on the 1.4.0 wheel and closed here.

    `gpuwm doctor` prints `MISSING WPS_GEOG ... -> gpuwm fetch-geog` and
    exits 0, which is right: a ~16 GB download nobody opted into is not
    a broken install.  `gpuwm go` then ran three stages, downloaded the
    forcing, and died in the fourth with the whole of:

        FAILED  prepare (exit 2)
          rw-wps --source gfs: /.../WPS_GEOG/topo_gmted2010_30s/index.

    No verb, no remedy, and the answer had been on screen a minute
    earlier from a different command.  `go` asks the same check now, on
    the same side of the download as the memory gate.
    """
    ran: list[str] = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: ran.append(label))
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))

    absent = tmp_path / "never-fetched" / "WPS_GEOG"
    args = _args(gfs_config, tmp_path / "out", geog_root=absent)
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.go_main(args)
    assert ran == [], "a stage ran before the geography check"

    message = str(refusal.value)
    assert "gpuwm fetch-geog" in message
    assert str(absent) in message
    assert "--geog-root" in message
    # The layered half carries the why, including the honest account of
    # doctor's exit 0 on the same gap.
    assert "before the fetch stage" in message
    assert "exits 0" in message


def test_a_partial_geography_tree_names_what_is_wrong_with_it(tmp_path):
    """A dataset present but unindexed is a partial download, not a
    missing opt-in, and the refusal has to distinguish them by name."""

    geog = _staged_geog_tree(tmp_path)
    victim = sorted(p for p in geog.iterdir() if p.is_dir())[0]
    (victim / "index").unlink()

    message = go_cli.geography_refusal(geog)
    assert message is not None
    assert victim.name in message
    assert "gpuwm fetch-geog" in message


def test_a_staged_geography_tree_passes_the_check(staged_geog):
    """Non-vacuity: the check says yes to a tree shaped like a real one."""

    assert go_cli.geography_refusal(staged_geog) is None


def test_the_memory_gate_warns_without_blocking_and_can_be_skipped(
        gfs_config, tmp_path, monkeypatch, capsys):
    """Over budget but inside free VRAM is an advisory, not a refusal."""
    ran: list[str] = []

    def _stage(label, command, **kwargs):
        ran.append(label)
        if label == "manifest":  # stop the chain where the real work starts
            raise go_cli.GoStageFailed(9)

    monkeypatch.setattr(go_cli, "_run_stage", _stage)
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    monkeypatch.setattr(
        go_cli, "memory_gate",
        lambda plan, **kw: {"verdict": "the forecast is the memory-binding "
                                       "phase at 9.00 GiB peak envelope",
                            "refuse": False, "warn": True,
                            "free_bytes": 12 * 1024 ** 3})
    assert go_cli.go_main(_args(gfs_config, tmp_path / "warn")) == 9
    assert ran == ["authority", "fetch", "manifest"]
    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert "memory-binding phase" in printed

    called: list[str] = []
    ran.clear()
    monkeypatch.setattr(
        go_cli, "memory_gate",
        lambda plan, **kw: called.append("gate") or {})
    args = _args(gfs_config, tmp_path / "skipped")
    args.no_memory_gate = True
    assert go_cli.go_main(args) == 9
    assert called == []
    assert ran == ["authority", "fetch", "manifest"]


# ---------------------------------------------------------------------------
# The memory gate must not stand up a CUDA context in the go process
# ---------------------------------------------------------------------------

#: A card as the subprocess probe reports one: the two device questions
#: the gate prices against, answered together.
_PROBE_PROFILE = {
    "name": "pinned probe card",
    "multiprocessor_count": 170,
    "max_threads_per_multiprocessor": 1536,
    "default_stack_limit_bytes": 1024,
}


@pytest.fixture
def _in_process_cupy_poisoned(monkeypatch):
    """Any in-process touch of cupy's CUDA half is the defect, said loudly.

    ``memGetInfo`` and ``deviceGetLimit`` cannot be asked without
    standing up a CUDA primary context, and the ``gpuwm go`` process
    outlives its own gate as nothing but the stage orchestrator and
    progress printer -- so a context stood up there sits on the card for
    the entire chain (measured 0.486 GiB on the RTX 5090) as a consumer
    no term of the budget the gate just computed names.  The poison
    replaces cupy in ``sys.modules``: importing it stays legal (imports
    allocate nothing), touching any attribute raises.
    """

    import sys as _sys
    import types

    poison = types.ModuleType("cupy")

    def _refuse(name):
        raise AssertionError(
            f"the go process touched in-process cupy.{name}: that stands "
            "up a CUDA primary context that then sits on the card for the "
            "whole chain while this process does nothing but print "
            "progress")

    poison.__getattr__ = _refuse
    monkeypatch.setitem(_sys.modules, "cupy", poison)


def test_the_gate_asks_the_card_in_a_subprocess_and_prices_its_answer(
        gfs_config, tmp_path, monkeypatch, _in_process_cupy_poisoned):
    """The gate's device questions run in a short-lived subprocess.

    The card's answers must be the ones the gate prices -- free VRAM
    from the probe, the reserve from the probe's device profile -- and
    in-process cupy (poisoned above) must never be touched.  On code
    that still asks in-process, the poison sends the gate down its
    no-device path and the first assertion states the defect: the probe
    seam was ignored.
    """

    from gpuwm.core import preflight

    payload = {"free_bytes": 30 * 1024 ** 3, "total_bytes": 32 * 1024 ** 3,
               "profile": dict(_PROBE_PROFILE)}
    # raising=False so unfixed code FAILS the assertion below (the
    # defect, stated) instead of erroring on a not-yet-existing seam.
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess",
                        lambda **_kwargs: dict(payload), raising=False)
    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "probe")
    gate = go_cli.memory_gate(plan)
    assert gate["free_bytes"] == payload["free_bytes"]

    from gpuwm.core.preflight import (ReservePolicy, _load_experiment_any,
                                      profile_from_device_probe)

    profile = profile_from_device_probe(payload)
    assert profile is not None
    assert profile.name == _PROBE_PROFILE["name"]
    exp = _load_experiment_any(plan["config"])
    reserve = ReservePolicy.n0_alloc(
        exp, profile=profile,
        estimate_bytes=gate["phases"].forecast.alloc_estimate_bytes)
    assert gate["budget_bytes"] == (payload["free_bytes"]
                                    - reserve.reserve_bytes)
    peak = gate["phases"].peak_envelope_bytes
    assert gate["refuse"] == (peak > payload["free_bytes"])
    assert gate["warn"] == (peak > gate["budget_bytes"])


def test_a_card_the_probe_cannot_see_never_refuses(
        gfs_config, tmp_path, monkeypatch, _in_process_cupy_poisoned):
    """No readable device (a planning box, a CI runner): the phases are
    still priced and the verdict still prints, but nothing refuses on a
    card nobody measured -- through the probe seam, and still without
    touching in-process cupy."""

    from gpuwm.core import preflight

    monkeypatch.setattr(preflight, "device_memory_probe_subprocess",
                        lambda **_kwargs: None, raising=False)
    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "nodev")
    gate = go_cli.memory_gate(plan)
    assert gate["free_bytes"] is None
    assert gate["refuse"] is False
    assert gate["warn"] is False
    assert gate["verdict"]
    assert "phases" in gate


def _staged_geog_tree(root: Path) -> Path:
    """A WPS_GEOG tree shaped the way `gpuwm fetch-geog` leaves one.

    Directory names and the `index` file are the whole of what the
    prepare stage's precondition reads, so the nine empty-but-indexed
    directories are a faithful stand-in for 16 GB of terrain.  Names
    come from `geog_assets`, the module that stages the real one -- the
    same source doctor reads -- so this fixture cannot drift from the
    check under test.
    """

    from gpuwm.geog_assets import geog_datasets

    geog = root / "WPS_GEOG"
    for name in geog_datasets():
        (geog / name).mkdir(parents=True, exist_ok=True)
        (geog / name / "index").write_text("", encoding="utf-8")
    return geog


@pytest.fixture(scope="module")
def staged_geog(tmp_path_factory):
    return _staged_geog_tree(tmp_path_factory.mktemp("geog"))


_DEFAULT_GEOG: list[Path] = []


def _args(config, outdir, geog_root=None):
    import argparse
    import tempfile

    if geog_root is None:
        # Every chain test past the memory gate now needs a usable
        # geography tree, because `go` checks for one there (it used to
        # find out in the prepare stage, after the download).  One
        # stand-in for the whole file rather than a fixture threaded
        # through thirty call sites.
        if not _DEFAULT_GEOG:
            _DEFAULT_GEOG.append(
                _staged_geog_tree(Path(tempfile.mkdtemp(prefix="gowps-"))))
        geog_root = _DEFAULT_GEOG[0]
    return argparse.Namespace(
        config=config, outdir=outdir, data_dir=None, geog_root=geog_root,
        dry_run=False, no_memory_gate=False, explain=False)


# ---------------------------------------------------------------------------
# The forecast lead: a start of cycle + K, carried through the chain
# ---------------------------------------------------------------------------

def test_go_carries_the_configs_forecast_lead_into_its_fetch(tmp_path):
    """A lead in [fetch] is load-bearing, exactly like cycle/hours/area.

    ``gpuwm go`` is step 3 of what the wizard itself prints, so a config
    whose start_time is cycle + K has to reach a fetch that downloads
    f{K}..  Ignoring the lead here would download f000.. and then hand
    the front door a series that does not carry the lead the experiment
    starts from -- a refusal produced by the orchestrator, from a config
    that is entirely correct.
    """

    config = _emit(tmp_path, "lead", "--forecast-start-hour", "174")
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "go")
    assert plan["forecast_start_hour"] == 174
    # The cycle stays the CYCLE; the lead is carried beside it.
    assert plan["cycle"] == "2026-07-29T18"
    fetch = go_cli.fetch_command(plan)
    assert "--forecast-start-hour" in fetch
    assert fetch[fetch.index("--forecast-start-hour") + 1] == "174"

    # And an analysis-start config still prints the command it always did.
    plain = go_cli.plan_from_config(
        _emit(tmp_path, "plain"), outdir=tmp_path / "go-plain")
    assert plain["forecast_start_hour"] == 0
    assert "--forecast-start-hour" not in go_cli.fetch_command(plain)


def test_go_derives_the_statics_corridor_from_a_follow_config(tmp_path,
                                                              gfs_config):
    """A config that declares a [relocation] follow source gets
    --statics-corridor on the prepare stage; every other config's
    prepare line is byte-for-byte what it always was."""

    two_domain = tmp_path / "follow.toml"
    two_domain.write_text(_write_follow_tree_config(gfs_config),
                          encoding="utf-8")
    plan = go_cli.plan_from_config(two_domain, outdir=tmp_path / "go",
                                   allow_tree=True)
    assert plan["statics_corridor"] is True
    command = go_cli.prepare_command(
        plan, tmp_path / "bridge", manifest=tmp_path / "m.json",
        manifest_sha256="a" * 64, cycle_stamp="2026-07-29_18:00:00",
        geog_root=tmp_path / "geog")
    assert "--statics-corridor" in command

    plain = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go2")
    assert plain["statics_corridor"] is False
    unchanged = go_cli.prepare_command(
        plain, tmp_path / "bridge", manifest=tmp_path / "m.json",
        manifest_sha256="a" * 64, cycle_stamp="2026-07-29_18:00:00",
        geog_root=tmp_path / "geog")
    assert "--statics-corridor" not in unchanged


def _write_follow_tree_config(gfs_config) -> str:
    """The wizard's own single-domain emission, grown into a two-domain
    follow tree: a child plus a [relocation] itinerary on it."""

    from gpuwm.experiment import load_experiment

    base = load_experiment(gfs_config)
    dt = float(base.root.run.dt)
    text = gfs_config.read_text(encoding="utf-8")
    return text + f"""
[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 30
j_parent_start = 30
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 45
ny = 45
history_interval_s = 3600.0

[relocation]
enabled = true
grid_id = 2

[[relocation.move]]
at_seconds = {dt * 2:.1f}
di_parent_cells = 1
dj_parent_cells = 0
"""


def test_the_printed_rw_wps_line_and_go_agree_on_the_corridor(tmp_path,
                                                              gfs_config):
    """The pasted manual line and go's driven line derive the corridor
    flag from the same config predicate, so neither can drift: both
    carry --statics-corridor for a follow config."""

    from gpuwm.fetch import author_gfs_front_door_manifest

    config = tmp_path / "follow.toml"
    config.write_text(_write_follow_tree_config(gfs_config),
                      encoding="utf-8")
    config.with_suffix(".namelist.wps").write_bytes(
        gfs_config.with_suffix(".namelist.wps").read_bytes())
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "go",
                                   allow_tree=True)
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stub")
    _stage_a_fetched_directory(plan["data"], config, plan["authority"])

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
    assert theirs.get("--statics-corridor") is True
    assert mine.get("--statics-corridor") is True
    assert set(mine) == set(theirs)


# ---------------------------------------------------------------------------
# UX finding N18 (2026-08-18 upgrader walk): the run-folder line prints on
# the real path BEFORE the gates, so a refused `gpuwm go` still teaches the
# 2.5.0 layout.  Both of the walk's real attempts refused (memory gate,
# then WPS_GEOG) and the announcement -- which --dry-run prints second --
# never appeared.
# ---------------------------------------------------------------------------

def test_a_refused_real_go_still_announces_the_run_folder(
        gfs_config, tmp_path, monkeypatch, capsys):
    ran: list[str] = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: ran.append(label))
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    monkeypatch.setattr(
        go_cli, "memory_gate",
        lambda plan, **kw: {
            "verdict": "the forecast is the memory-binding phase at "
                       "40.00 GiB peak envelope",
            "refuse": True, "warn": True, "free_bytes": 8 * 1024 ** 3,
        })
    args = _args(gfs_config, tmp_path / "out")
    with pytest.raises(go_cli.GoRefusal):
        go_cli.go_main(args)
    assert ran == [], "the gate still fires before any stage"
    printed = capsys.readouterr().out
    assert "go: run folder" in printed, (
        "a refused real run must still say where a run WOULD land")
    # The same line the dry run prints second: folder name, case root,
    # and the subtree inventory.
    assert "authority/, prepared/, run/ and png/" in printed


def test_the_real_run_folder_line_agrees_with_the_dry_run(
        gfs_config, tmp_path, monkeypatch, capsys):
    """One line, one function, both paths -- the path a reader plans
    against is the path they get."""

    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    args = _args(gfs_config, tmp_path / "out")
    args.dry_run = True
    assert go_cli.go_main(args) == 0
    dry = [line for line in capsys.readouterr().out.splitlines()
           if line.startswith("go: run folder")]
    assert len(dry) == 1

    monkeypatch.setattr(
        go_cli, "memory_gate",
        lambda plan, **kw: {
            "verdict": "the forecast is the memory-binding phase",
            "refuse": True, "warn": True, "free_bytes": 8 * 1024 ** 3,
        })
    args = _args(gfs_config, tmp_path / "out")
    with pytest.raises(go_cli.GoRefusal):
        go_cli.go_main(args)
    real = [line for line in capsys.readouterr().out.splitlines()
            if line.startswith("go: run folder")]
    assert len(real) == 1
    # Same shape up to the stamp (the two invocations claim different
    # stamped names on a shared case root).
    assert dry[0].split("run-", 1)[0] == real[0].split("run-", 1)[0]
