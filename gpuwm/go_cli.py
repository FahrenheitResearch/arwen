"""``gpuwm go``: run the documented GFS chain without hand-relaying hashes.

Getting from a wizard-authored config to a forecast means five commands
in a fixed order, and three of them take values the previous one
printed: the front-door manifest's digest, then the preparation proof's
digest, its bound input-manifest digest, and its prepared-cache content
digest.  ``docs/public/FIRST-LIGHT.md`` section 3a spells the relay out
and says "copy it rather than retyping".

That relay is integrity plumbing.  Every one of those digests exists so
a later stage can refuse inputs an earlier stage did not produce, and
none of them is a decision a person makes -- so asking a person to
carry them is asking them to be a checksum courier, which is how a
first run ends in a transcription error three stages deep.

**What this does not do is weaken a single check.**  ``go`` runs the
same commands, as subprocesses, in the documented order, and every
artifact lands on disk exactly where the manual chain leaves it.  The
relayed values are read back from those artifacts -- ``sha256`` of the
manifest file, and the three digests carried inside ``proof.json`` --
never scraped from a stage's printed prose.  Prose is a presentation
layer that ``--explain`` is free to move; the artifacts are the
contract.  Each stage still verifies what it always verified, against
values ``go`` transported but did not compute.

**Ordering is not negotiable and there is no continue-on-failure.**
The authority materialization has to precede preprocessing, because the
runner binds the experiment config into the prepared cache and doing it
afterwards means preprocessing twice.  Every later stage consumes the
previous one's output, so a failed stage replays its whole output and
stops -- unlike ``gpuwm setup``, whose steps are independent.

**Scope is the route that is actually documented.**  GFS single-domain
with a shipped ``--physics-profile``.  Anything else -- an ERA5
``[case_data]`` config, a multi-domain tree, a config whose physics
matches no shipped profile -- is refused up front, naming the manual
chain, rather than half-orchestrated into a state nobody can finish by
hand either.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from gpuwm.explain import explain_enabled

#: The runner that owns both the authority materialization and the
#: forecast, as an importable module rather than a script path.
#:
#: It used to be ``tools/prepared_single_domain_forecast.py`` and it
#: cannot be any more: ``go``'s whole job is that a person who has
#: ``pip install gpuwm`` can get from a config to a forecast, and a
#: script path under ``tools/`` is a promise only a git checkout keeps.
#: The runner's substance now lives in the package (``tools/`` keeps a
#: thin delegating entry point for the spelling the docs use), so every
#: stage below is ``python -m <module>`` -- resolvable from a wheel, a
#: checkout, or a venv with neither on the working directory.
RUNNER_MODULE = "gpuwm.prepared_single_domain_forecast"

#: The checkout spelling of the same runner, kept for the docs and for
#: receipts that name it.  Not used to locate anything.
RUNNER_RELATIVE = "tools/prepared_single_domain_forecast.py"

#: Seconds between "still working" lines while a stage is quiet.
#:
#: rw-wps' static build and the runner's preparation each run for
#: minutes with nothing on the terminal, and a silent terminal is read
#: as a hang -- an owner watching one concluded the tool was broken and
#: stopped it.  Whatever else a long stage does, it must keep saying
#: that it is a long stage.
HEARTBEAT_SECONDS = 20.0

#: The sources whose printed-value relay is documented end to end.
#:
#: GFS only, and for a reason that is a property of the other routes
#: rather than a preference.  HRRR does not merely lack the printed
#: relay -- ``gpuwm fetch --source hrrr`` says so itself, ending with
#: "fetching cannot bind the run's own flags: --wps-namelist,
#: --geog-root, --experiment-config, --valid-time and --output-root are
#: yours to supply" -- it cannot reach this chain's last stage at all:
#: ``tools/prepared_single_domain_forecast.py`` declares
#: ``SUPPORTED_SOURCES = {"gfs", "era5", "20crv3"}`` and refuses
#: ``hrrr`` outright.  Its front door writes no ``proof.json`` and
#: prints no three-digest command, so there is nothing to carry.
#:
#: ERA5 reaches that runner but arrives by the ``[case_data]``
#: config-driven route, which ``gpuwm run`` executes in one command and
#: which needs no relay.  Both are refused above with their own pointer.
ORCHESTRATED_SOURCES = ("gfs",)

#: Where the manual chain points a reader whose config this refuses.
MANUAL_CHAIN = "docs/public/FIRST-LIGHT.md section 3a"


class GoRefusal(ValueError):
    """A config this command will not half-orchestrate."""


def _repo_root() -> Path:
    """The directory containing the ``gpuwm`` package.

    A checkout root in a clone; ``site-packages`` in an install.  Only
    used to notice which of the two this is -- nothing is located
    through it any more.
    """

    return Path(__file__).resolve().parent.parent


def _is_checkout() -> bool:
    """True when this install is a repository clone."""

    return (_repo_root() / RUNNER_RELATIVE).is_file()


def _stage_cwd() -> Path:
    """Where the stage subprocesses run.

    The caller's directory, so a relative ``--out`` or ``--data-dir``
    means what the person who typed it meant.  Every path ``go`` itself
    composes is absolute, so this choice cannot move an artifact; it can
    only stop a relative one from landing somewhere surprising.  (It
    used to be the checkout root, which no wheel install has.)
    """

    return Path.cwd()


def _quote(value) -> str:
    return shlex.quote(str(value).replace("\\", "/"))


def printable(command: list[str]) -> str:
    """One command as a pasteable line."""

    return " ".join(_quote(token) for token in command)


# ---------------------------------------------------------------------------
# Reading the config: what the chain needs, and what it refuses
# ---------------------------------------------------------------------------

def plan_from_config(config: Path, *, outdir: Path | None = None,
                     data_dir: Path | None = None) -> dict:
    """Everything the five stages need, or a refusal saying why not.

    Read from the wizard's own emitted tables rather than asked for
    again: ``[fetch]`` carries the resolved cycle, hours and area the
    wizard sized the domain against, and the physics tables carry the
    profile.  A flag that re-asked for any of them would let the two
    disagree, and the run would then be sized for one box and fetched
    for another.
    """

    import tomllib

    from gpuwm.experiment import load_experiment
    from gpuwm.physics_compat import identify_single_domain_profile

    config = Path(config)
    if not config.is_file():
        raise GoRefusal(f"{config} does not exist")
    payload = tomllib.loads(config.read_text(encoding="utf-8"))

    if "case_data" in payload:
        raise GoRefusal(
            f"{config} declares a [case_data] table, which is the ERA5 "
            "config-driven route -- `gpuwm run` executes that one "
            "directly and needs no chain.\n"
            "  remedy: gpuwm check " + str(config) + " && gpuwm run "
            + str(config))

    domains = payload.get("domain")
    if isinstance(domains, list) and len(domains) > 1:
        raise GoRefusal(
            f"{config} declares {len(domains)} domains, and the "
            "single-domain runner this chain drives takes one.  A domain "
            "tree is prepared the same way but run by "
            "tools/prepared_domain_tree_forecast.py, which binds a "
            "preparation receipt rather than three digests.\n"
            f"  remedy: follow {MANUAL_CHAIN}, or re-emit with a single "
            "domain: gpuwm domain --ladder 12 ...")

    fetch_table = payload.get("fetch")
    if not isinstance(fetch_table, dict) or "source" not in fetch_table:
        raise GoRefusal(
            f"{config} carries no [fetch] table, so the cycle and area "
            "this domain was sized against are not recorded in it.\n"
            "  remedy: re-emit the config with `gpuwm domain`, which "
            "writes that table")

    source = str(fetch_table["source"])
    if source not in ORCHESTRATED_SOURCES:
        raise GoRefusal(
            f"{config} is a --source {source} config, and this chain is "
            f"the {'/'.join(ORCHESTRATED_SOURCES)} one.  "
            + ("hrrr cannot reach its last stage at all: "
               "tools/prepared_single_domain_forecast.py accepts "
               "gfs/era5/20crv3 and refuses hrrr, and the HRRR front "
               "door writes no proof.json, so there are no digests to "
               "carry.\n"
               if source == "hrrr" else
               f"{source} has no printed-value relay to automate.\n")
            + f"  remedy: follow {MANUAL_CHAIN} for this source")

    try:
        experiment = load_experiment(config)
        profile = identify_single_domain_profile(experiment.root.run)
    except Exception as error:  # a config that will not load is the finding
        raise GoRefusal(f"{config} does not load as an experiment: "
                        f"{error}") from error
    if profile is None:
        raise GoRefusal(
            f"the physics in {config} matches none of the profiles the "
            "prepared single-domain runner accepts, so the chain's last "
            "stage would refuse it.  This is the wizard's default suite "
            "unless a profile was asked for.\n"
            "  remedy: re-emit with a profile -- gpuwm domain "
            "--physics-profile <id> ... -- and `gpuwm domain --help` "
            "lists the ids")

    # No "is the runner on disk?" gate any more.  It used to be here
    # because the runner was a script under tools/ that a wheel install
    # does not carry, and the honest answer was "go clone the
    # repository".  The runner is part of the package now, so if this
    # module imported, so did it.
    for key in ("cycle", "hours", "area"):
        if key not in fetch_table:
            raise GoRefusal(
                f"{config}'s [fetch] table has no {key!r}; re-emit the "
                "config with `gpuwm domain`")

    base = config.parent
    root = Path(outdir) if outdir is not None else base / f"{config.stem}-go"
    # `[fetch].out` is deliberately NOT used, and this is not an
    # oversight in either direction.
    #
    # The wizard writes that key relative to the directory IT was run in
    # (`_relative_or_absolute(data_dir, Path.cwd())`), not relative to
    # the config file, so a config emitted from one directory records a
    # download path that means something else -- or nothing -- read from
    # another.  A config emitted from a checkout root and read from
    # /tmp resolved to `C:/AppData/...`, six `..` hops having walked
    # past the drive root and clamped there.
    #
    # The table declares itself "advisory ... validated, not executed",
    # and the values that are load-bearing are the ones the domain was
    # SIZED against -- cycle, hours, area -- which are absolute facts
    # and are honoured exactly.  Where the bytes land is this command's
    # own business, so they land under its own root unless `--data-dir`
    # names an existing download to reuse.
    data = Path(data_dir) if data_dir is not None else root / "data"
    return {
        "config": config,
        "wps_namelist": base / f"{config.stem}.namelist.wps",
        "source": source,
        "cycle": str(fetch_table["cycle"]),
        "hours": int(fetch_table["hours"]),
        "area": str(fetch_table["area"]),
        "data": data,
        "profile": profile,
        "root": Path(root),
        "authority": Path(root) / "authority",
        "prepared": Path(root) / "prepared",
        "run": Path(root) / "run",
        "render": Path(root) / "png",
        "runner": RUNNER_MODULE,
    }


# ---------------------------------------------------------------------------
# The five commands, composed from the plan
# ---------------------------------------------------------------------------

def _area_flag(area: str) -> str:
    """``--area=VALUE`` when the value leads with a minus, else split.

    The same rule the wizard's printed command follows: argparse reads a
    leading-minus token as an option unless it is joined with ``=``.
    """

    return f"--area={area}"


def authority_command(plan: dict) -> list[str]:
    return [sys.executable, "-m", str(plan["runner"]),
            "--materialize-authorities",
            "--source", plan["source"],
            "--base-experiment-config", str(plan["config"]),
            "--base-wps-namelist", str(plan["wps_namelist"]),
            "--physics-profile", plan["profile"],
            "--output-directory", str(plan["authority"])]


def fetch_command(plan: dict) -> list[str]:
    return [sys.executable, "-m", "gpuwm.cli", "fetch",
            "--source", plan["source"], "--cycle", plan["cycle"],
            "--hours", str(plan["hours"]), _area_flag(plan["area"]),
            "--out", str(plan["data"])]


def manifest_command(plan: dict, bridge: Path) -> list[str]:
    return [sys.executable, "-m", "gpuwm.cli", "fetch",
            "--source", plan["source"], "--author-front-door-manifest",
            "--out", str(plan["data"]), "--bridge", str(bridge),
            "--wps-namelist", str(plan["authority"] / "namelist.wps"),
            "--experiment-config", str(plan["authority"] / "experiment.toml")]


def prepare_command(plan: dict, bridge: Path, *, manifest: Path,
                    manifest_sha256: str, cycle_stamp: str,
                    geog_root: Path) -> list[str]:
    """The rw-wps invocation, composed from the artifacts on disk.

    Every value here is either from the plan or read back from a file a
    previous stage wrote -- the manifest's path and its digest.  This is
    the same command ``gpuwm fetch --author-front-door-manifest``
    prints; composing it from the same inputs rather than parsing that
    printed line is what keeps it working when the printed line moves
    behind ``--explain``.
    """

    return [sys.executable, "-m", "gpuwm.source_cli",
            "--source", plan["source"],
            "--gfs-series", str(plan["data"] / f"{plan['source']}-series.tsv"),
            "--cycle", cycle_stamp,
            "--bridge", str(bridge),
            "--wps-namelist", str(plan["authority"] / "namelist.wps"),
            "--experiment-config",
            str(plan["authority"] / "experiment.toml"),
            "--source-manifest", str(manifest),
            "--source-manifest-sha256", manifest_sha256,
            # Not optional in practice, whatever the flag's help says.
            # `gpuwm/source_cli.py` falls back to WSM6_PROFILE_ID when
            # --physics-profile is absent and then compares the config's
            # physics against THAT, so omitting it refuses every config
            # except a wsm6-no-radiation one.  The end-to-end run that
            # found this used the profile FIRST-LIGHT.md's own worked
            # example uses.
            "--physics-profile", plan["profile"],
            "--geog-root", str(geog_root),
            "--output-root", str(plan["prepared"])]


def forecast_command(plan: dict, digests: dict) -> list[str]:
    return [sys.executable, "-m", str(plan["runner"]),
            "--source", plan["source"],
            "--prepared-root", str(plan["prepared"]),
            "--proof-sha256", digests["proof"],
            "--source-manifest-sha256", digests["source_manifest"],
            "--prepared-content-sha256", digests["prepared_content"],
            "--experiment-config",
            str(plan["authority"] / "experiment.toml"),
            "--wps-namelist", str(plan["authority"] / "namelist.wps"),
            "--physics-profile", plan["profile"],
            "--io-mode", "history", "--outdir", str(plan["run"])]


# ---------------------------------------------------------------------------
# Reading relayed values back out of the artifacts
# ---------------------------------------------------------------------------

def proof_digests(prepared_root: Path) -> dict:
    """The three digests the forecast stage binds, from ``proof.json``.

    Exactly the values ``gpuwm.gfs_direct`` composes its printed
    next-command from, read from the same document: the digest OF the
    proof, and the two digests carried INSIDE it.  Reading the file is
    what makes this a relay rather than a re-derivation -- ``go`` must
    not compute a digest the runner will then compare against itself.
    """

    from gpuwm.fetch import sha256_file

    proof_path = Path(prepared_root) / "proof.json"
    if not proof_path.is_file():
        raise GoRefusal(
            f"preparation finished without writing {proof_path}, so the "
            "forecast stage has nothing to bind against")
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    cache = payload.get("prepared_cache")
    content = cache.get("content_sha256") if isinstance(cache, dict) else None
    manifest_digest = payload.get("input_manifest_sha256")
    if not isinstance(content, str) or not isinstance(manifest_digest, str):
        raise GoRefusal(
            f"{proof_path} carries no single prepared-cache identity, "
            "which is what a multi-domain hierarchy product looks like.  "
            f"Its runner is tools/prepared_domain_tree_forecast.py; see "
            f"{MANUAL_CHAIN}")
    return {"proof": sha256_file(proof_path),
            "source_manifest": manifest_digest,
            "prepared_content": content}


def wrfout_frames(plan: dict) -> list[Path]:
    """Every history file the forecast stage published, in time order."""

    return sorted((plan["run"] / "wrfout").glob("wrfout_d*"))


def render_command(plan: dict, frames: list[Path] | None = None) -> list[str]:
    """The sixth stage: turn the forecast into pictures.

    ``go`` used to stop after the forecast and print this line for the
    reader to paste.  That is one more baton pass, and a baton pass is
    where a first run ends -- a command printed instead of run reads as
    "it stopped", not "your turn".  Every other handoff in this chain is
    internal now; this one has no reason to be the exception, and the
    reader who typed ``gpuwm go`` wanted a forecast they can look at.

    ``gpuwm render`` takes FILES.  The documented line spells them
    ``.../wrfout/*`` and relies on a shell to expand that; a subprocess
    has no shell, and handing the directory over produced "unreadable
    wrfout (not a regular file)" on a run that had otherwise finished.
    So the frames are enumerated here.  ``frames=None`` keeps the
    glob spelling, which is what ``--dry-run`` should print: at that
    point the directory is empty and naming its contents would be a
    guess.
    """

    targets = ([str(frame) for frame in frames] if frames is not None
               else [str(plan["run"] / "wrfout" / "*")])
    return [sys.executable, "-m", "gpuwm.cli", "render", *targets,
            "--out", str(plan["render"])]


def render_extra_missing() -> str | None:
    """Why rendering cannot run here, or ``None`` when it can.

    ``wrf`` (pip distribution ``wrf-rust``) is the mandated science core
    for every derived field, and it lives behind the ``render`` extra.
    An install without it can still produce the forecast, so a missing
    extra is not a failed chain -- but it IS the one place the chain
    genuinely stops, and it says so with the command that finishes the
    job rather than falling silent.
    """

    try:
        import wrf  # noqa: F401
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _render_stage(plan: dict, *, explain: bool) -> bool:
    """Run the render stage; return whether anything was rendered."""

    missing = render_extra_missing()
    if missing is not None:
        print("  -- render skipped: this environment has no 'wrf' "
              f"package ({missing}),")
        print("     which every derived field comes from.")
        print("     remedy: pip install 'gpuwm[render]', then:")
        print(f"       {printable(render_command(plan))}")
        return False
    frames = wrfout_frames(plan)
    if not frames:
        print("  -- render skipped: the forecast stage published no "
              f"wrfout frame under {plan['run'] / 'wrfout'}.")
        return False
    _run_stage("render", render_command(plan, frames), explain=explain)
    return True


def resolve_bridge() -> Path:
    """The built ``gfs_grib2_bridge``, through the one resolver.

    :mod:`gpuwm.bridges` is where every other consumer looks, so a
    bridge staged by ``gpuwm setup`` or built in a checkout is found
    the same way here as it is by ``gpuwm doctor``.
    """

    from gpuwm import bridges

    found = bridges.find_bridge("gfs_grib2_bridge")
    if found is None:
        raise GoRefusal(
            "no built gfs_grib2_bridge, which every stage of this chain "
            "decodes through.\n"
            "  remedy: gpuwm setup\n"
            "  # or `gpuwm doctor --explain` for the build route on a "
            "platform with no published bundle")
    return found


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

def _elapsed_words(seconds: float) -> str:
    """``3m 07s`` -- an elapsed time a person reads without converting."""

    whole = int(seconds)
    return f"{whole // 60}m {whole % 60:02d}s"


def _progress_note(progress: Path | None) -> str:
    """What the running stage says about itself, if it says anything.

    The runner publishes ``progress.json`` beside its output and updates
    it as the forecast advances, which is a better answer than a
    stopwatch: "2m 40s elapsed" and "2m 40s elapsed, INTEGRATING, 1200
    model seconds" differ by whether the reader can tell it is moving.
    A missing or half-written file is not an error here -- it is a stage
    that has not got there yet -- so every failure to read one is
    silent.
    """

    if progress is None:
        return ""
    try:
        payload = json.loads(Path(progress).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    status = payload.get("status")
    model = payload.get("model_elapsed_seconds")
    parts = [str(status)] if isinstance(status, str) else []
    if isinstance(model, (int, float)) and model > 0:
        parts.append(f"{float(model):.0f} model seconds")
    return (", " + ", ".join(parts)) if parts else ""


def _run_stage(label: str, command: list[str], *, explain: bool,
               progress: Path | None = None,
               heartbeat_seconds: float = HEARTBEAT_SECONDS) -> None:
    """Run one stage; replay everything it said and stop if it failed.

    Output is captured so the default is one line per stage, and
    released in full the moment a stage fails -- the reason a stage
    refused is never summarized, because a chain that stops without
    saying why leaves the reader with five commands to re-run by hand
    to find out.

    Captured output means a quiet stage is a quiet terminal, and two of
    these stages run for minutes.  An owner watching one of them
    concluded the tool had hung.  So while a stage runs, this prints a
    line every ``heartbeat_seconds`` naming the stage, how long it has
    been going, and -- when the stage publishes a progress file -- what
    it says it is doing.  The heartbeat is a property of waiting, not of
    verbosity: it is not behind ``--explain``, because the reader who
    needs it is the one who did not pass any flags.
    """

    print(f"  .. {label}", flush=True)
    started = time.monotonic()
    # The subprocess still runs through ``subprocess.run`` -- one call,
    # one place, the same capture semantics -- and the waiting moves to
    # a worker thread so this one stays free to say that it is waiting.
    # Doing it the other way round (Popen plus a poll loop) would have
    # split the "how a stage is run" answer across two code paths.
    box: dict[str, object] = {}

    def _wait() -> None:
        try:
            box["completed"] = subprocess.run(
                command, capture_output=True, text=True,
                errors="replace", cwd=str(_stage_cwd()))
        except BaseException as error:  # re-raised on this thread below
            box["error"] = error

    worker = threading.Thread(target=_wait, name=f"go-stage-{label}",
                              daemon=True)
    worker.start()
    next_beat = started + heartbeat_seconds
    while worker.is_alive():
        worker.join(timeout=0.2)
        now = time.monotonic()
        if worker.is_alive() and now >= next_beat:
            print(f"     .. {label}, {_elapsed_words(now - started)}"
                  f"{_progress_note(progress)}", flush=True)
            next_beat = now + heartbeat_seconds
    worker.join()
    if "error" in box:
        raise box["error"]
    completed = box["completed"]
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        print(f"  FAILED  {label} (exit {completed.returncode})")
        for line in output.splitlines():
            print(f"    {line}")
        print(f"go: stopped at {label}; every later stage consumes this "
              "one's output, so nothing after it ran.")
        raise GoStageFailed(completed.returncode)
    print(f"  ok      {label} ({_elapsed_words(time.monotonic() - started)})")
    if explain:
        for line in output.splitlines():
            print(f"    {line}")


class GoStageFailed(Exception):
    """A stage exited nonzero; its output has already been replayed."""

    def __init__(self, code: int):
        super().__init__(f"stage exited {code}")
        self.code = code


def go_main(args) -> int:
    from gpuwm.fetch import sha256_file
    from gpuwm.geog_assets import default_geog_root

    explain = explain_enabled(args)
    plan = plan_from_config(args.config,
                            outdir=getattr(args, "outdir", None),
                            data_dir=getattr(args, "data_dir", None))
    bridge = resolve_bridge()
    geog_root = (Path(args.geog_root) if getattr(args, "geog_root", None)
                 else default_geog_root())

    manifest = plan["data"] / "gfs-input-manifest.json"
    cycle_stamp = _cycle_stamp(plan["cycle"])

    if getattr(args, "dry_run", False):
        print(f"go: {plan['config']} -- source {plan['source']}, cycle "
              f"{plan['cycle']}, {plan['hours']} h, profile "
              f"{plan['profile']}")
        print("")
        for label, command in (
                ("1. authority", authority_command(plan)),
                ("2. fetch", fetch_command(plan)),
                ("3. manifest", manifest_command(plan, bridge)),
        ):
            print(f"{label}\n     {printable(command)}")
        # The last two carry values that do not exist yet.  Naming the
        # FILE each one is read from beats printing a plausible-looking
        # hash: the point of a dry run is to show what will happen, and
        # what will happen is a read from that artifact.
        print("4. prepare\n     " + printable(prepare_command(
            plan, bridge, manifest=manifest,
            manifest_sha256="<sha256 of " + manifest.name + ", after step 3>",
            cycle_stamp=cycle_stamp, geog_root=geog_root)))
        print("5. forecast\n     " + printable(forecast_command(plan, {
            "proof": "<sha256 of prepared/proof.json, after step 4>",
            "source_manifest": "<proof.json input_manifest_sha256>",
            "prepared_content":
                "<proof.json prepared_cache.content_sha256>"})))
        print("6. render\n     " + printable(render_command(plan)))
        return 0

    # Re-running `gpuwm go` is the second thing anyone does, and the
    # first stage is create-only: the runner will not materialize an
    # authority into a directory an earlier run owns, because the
    # receipt it writes has to describe one run.  That refusal is right
    # and its message names --output-directory, a flag this caller never
    # typed.  Answer it here, in the caller's own vocabulary, before
    # spending a stage to reach it.
    if plan["authority"].exists():
        raise GoRefusal(
            f"{plan['authority']} already exists, so this chain has been "
            f"run into {plan['root']} before.  Every stage is create-only "
            "on purpose -- merging two runs into one tree would publish "
            "receipts describing neither.\n"
            f"  remedy: gpuwm go {_quote(plan['config'])} --outdir "
            "<a new directory>\n"
            f"  # or remove {plan['root']} if that run is finished with")

    print(f"go: {plan['config']} -- source {plan['source']}, cycle "
          f"{plan['cycle']}, {plan['hours']} h, profile {plan['profile']}")
    try:
        _run_stage("authority", authority_command(plan), explain=explain)
        _run_stage("fetch", fetch_command(plan), explain=explain)
        _run_stage("manifest", manifest_command(plan, bridge),
                   explain=explain)
        if not manifest.is_file():
            raise GoRefusal(
                f"the manifest stage wrote no {manifest}, so the "
                "preparation stage has nothing to bind against")
        _run_stage("prepare", prepare_command(
            plan, bridge, manifest=manifest,
            manifest_sha256=sha256_file(manifest),
            cycle_stamp=cycle_stamp, geog_root=geog_root), explain=explain,
            progress=plan["prepared"] / "progress.json")
        digests = proof_digests(plan["prepared"])
        _run_stage("forecast", forecast_command(plan, digests),
                   explain=explain,
                   progress=plan["run"] / "progress.json")
        rendered = _render_stage(plan, explain=explain)
    except GoStageFailed as failure:
        return failure.code

    print(f"go: wrote {plan['run']}")
    if rendered:
        print(f"go: rendered {plan['render']}")
    return 0


def _cycle_stamp(cycle: str) -> str:
    """``YYYY-MM-DD_HH:MM:SS`` -- the stamp the front door's --cycle takes.

    The config records the cycle as ``YYYY-MM-DDTHH``; the front door
    wants the WRF-style stamp.  One conversion, in one place, rather
    than a second cycle spelling in the config.
    """

    from gpuwm.fetch import parse_cycle

    return f"{parse_cycle(cycle, 'gfs'):%Y-%m-%d_%H:%M:%S}"


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "go",
        help="run the documented GFS chain for a wizard-authored config "
             "-- authority, fetch, front-door manifest, rw-wps, forecast "
             "-- carrying each stage's digests to the next instead of "
             "asking you to copy them")
    parser.add_argument("config", type=Path, metavar="CONFIG",
                        help="a single-domain GFS experiment TOML emitted "
                             "by `gpuwm domain --physics-profile <id>`")
    parser.add_argument("--outdir", type=Path, default=None, metavar="DIR",
                        help="root for the authority, prepared and run "
                             "trees (default <config-stem>-go beside the "
                             "config)")
    parser.add_argument("--data-dir", type=Path, default=None, metavar="DIR",
                        dest="data_dir",
                        help="reuse an existing `gpuwm fetch` download "
                             "instead of fetching into <outdir>/data")
    parser.add_argument("--geog-root", type=Path, default=None, metavar="DIR",
                        help="staged WPS_GEOG tree (default: the one "
                             "`gpuwm fetch-geog` stages into)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="print the six commands, filled in, and "
                             "exit without running any of them")
    parser.set_defaults(func=go_main)
    return parser


__all__ = [
    "GoRefusal", "GoStageFailed", "HEARTBEAT_SECONDS", "MANUAL_CHAIN",
    "ORCHESTRATED_SOURCES", "RUNNER_MODULE", "RUNNER_RELATIVE",
    "authority_command", "fetch_command", "forecast_command", "go_main",
    "manifest_command", "plan_from_config", "prepare_command", "printable",
    "proof_digests", "register_cli", "render_command",
    "render_extra_missing", "resolve_bridge", "wrfout_frames",
]
