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

**Scope is the route that is actually documented.**  GFS
single-domain.  A config bound to a shipped ``--physics-profile``
carries that assertion through every stage; a config bound to none --
the wizard's default suite, or any hand-authored suite the engine
implements -- runs as written, with its WRF-verification status stated
in the receipts (owner ruling 2026-07-31: the suite choice is the
user's, and status is reported, never gating).  What is still refused
up front -- an ERA5 ``[case_data]`` config, a multi-domain tree, a
source with no printed-value relay -- is refused naming the manual
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

from gpuwm.explain import explain_enabled, layered, render

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
#:
#: A repo-relative path was the whole pointer until 1.4.1, and a pip
#: install contains no docs tree -- so the one instruction offered to
#: the reader named a file they provably did not have, with no URL to
#: reach it by (A-10, and the pointer half of B-03).  The URL is built
#: from the project's own declared Repository metadata so the two cannot
#: drift apart, and the local path stays for readers who do have a
#: checkout.
DOCS_BASE_URL = (
    "https://github.com/FahrenheitResearch/arwen/blob/main/")
MANUAL_CHAIN = (
    f"docs/public/FIRST-LIGHT.md section 3a "
    f"({DOCS_BASE_URL}docs/public/FIRST-LIGHT.md)")


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
        # Named by its INSTALLED spelling: a wheel has no tools/
        # directory, so a tools/-path pointer names a file a pip reader
        # provably does not have.  And the re-emit remedy is complete
        # rather than trailing off in "...": the wizard's default IS the
        # single-domain shape now, so "without --ladder" is the whole
        # instruction -- a tree config is one this reader explicitly
        # opted into.
        raise GoRefusal(
            f"{config} declares {len(domains)} domains, and the "
            "single-domain runner this chain drives takes one.  A domain "
            "tree is prepared the same way but run by "
            "gpuwm-prepared-tree-forecast (module form: python -m "
            "gpuwm.prepared_domain_tree_forecast), which binds a "
            "preparation receipt rather than three digests; the full "
            f"tree sequence is {MANUAL_CHAIN}.\n"
            "  remedy: re-emit the config without --ladder (the wizard's "
            "default is one 12 km domain, the shape this chain runs end "
            "to end), or prepare the tree with rw-wps and run "
            "gpuwm-prepared-tree-forecast on what it wrote")

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
    # ``profile`` may be None: the runner executes the config's own
    # suite as written (owner ruling 2026-07-31), so a config matching
    # no shipped profile is not a refusal any more -- the chain just
    # omits --physics-profile and the receipts report the suite's
    # verification status.  A matched profile is still forwarded, so a
    # bound config keeps its exactness assertion end to end.

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
    # E-07.  The default puts the download INSIDE the run root, so the
    # two are related by construction and only an explicit --data-dir can
    # make them equal.  When it does, the run claims the directory the
    # inputs live in: every stage is create-only, so a re-run then
    # refuses against a directory the reader thinks of as their download
    # cache, and the download and the run share a receipt.  Both flags
    # were the reader's, so neither is dropped -- this refuses and says
    # which one to move.
    if outdir is not None and data_dir is not None:
        try:
            same = Path(outdir).resolve() == Path(data_dir).resolve()
        except (OSError, RuntimeError):
            same = Path(outdir) == Path(data_dir)
        if same:
            raise GoRefusal(
                f"--outdir and --data-dir both name {root}.  The run "
                f"directory is create-only and the download directory is "
                f"meant to be reused, so one directory cannot be both: "
                f"the next run would refuse against your own cache.  "
                f"Point --outdir at a new directory and leave --data-dir "
                f"on the download, or drop --data-dir and let the "
                f"download land under the run root as {root / 'data'}.")
    return {
        "config": config,
        "wps_namelist": base / f"{config.stem}.namelist.wps",
        "source": source,
        "cycle": str(fetch_table["cycle"]),
        "hours": int(fetch_table["hours"]),
        # Load-bearing exactly like cycle/hours/area: the config's
        # start_time IS cycle + this lead, so a fetch that ignored it
        # would download a window the front door then refuses for not
        # carrying the lead the experiment starts from.
        "forecast_start_hour": int(fetch_table.get("forecast_start_hour", 0)),
        # Load-bearing for the same reason, and dropped until 1.4.1.  It
        # sets the LATERAL BOUNDARY interval: a config asking for hourly
        # forcing whose fetch runs without --cadence downloads f000/f003
        # and the run gets 3-hourly boundaries -- three times coarser
        # than the file says, with exit 0, "forecast validity PASS", and
        # nothing anywhere saying the request was not honoured.  A
        # measured case: [fetch] cadence = 1, report.json
        # boundary_interval_seconds = 10800.  The block is advisory in
        # the sense that this command chooses where bytes land; it is
        # not advisory about what is fetched.
        "cadence": (int(fetch_table["cadence"])
                    if fetch_table.get("cadence") is not None else None),
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

def _lead_note(plan: dict) -> str:
    """The forecast lead, said in the plan line, or nothing at lead 0."""

    lead = plan.get("forecast_start_hour") or 0
    if not lead:
        return ""
    return (f", initialized from forecast lead f{lead:03d} "
            f"(a {lead} h forecast, not an analysis)")


def _area_flag(area: str) -> str:
    """``--area=VALUE`` when the value leads with a minus, else split.

    The same rule the wizard's printed command follows: argparse reads a
    leading-minus token as an option unless it is joined with ``=``.
    """

    return f"--area={area}"


def _profile_flags(plan: dict) -> list[str]:
    """``--physics-profile <id>`` when the config is bound to one.

    Empty for a config matching no shipped profile: the runner executes
    the hash-bound experiment's own suite as written, and inventing a
    name here would re-create the WSM6-substitution defect this chain
    already met once.
    """

    profile = plan.get("profile")
    return [] if profile is None else ["--physics-profile", str(profile)]


def authority_command(plan: dict) -> list[str]:
    return [sys.executable, "-m", str(plan["runner"]),
            "--materialize-authorities",
            "--source", plan["source"],
            "--base-experiment-config", str(plan["config"]),
            "--base-wps-namelist", str(plan["wps_namelist"]),
            *_profile_flags(plan),
            "--output-directory", str(plan["authority"])]


def fetch_command(plan: dict) -> list[str]:
    command = [sys.executable, "-m", "gpuwm.cli", "fetch",
               "--source", plan["source"], "--cycle", plan["cycle"],
               "--hours", str(plan["hours"]), _area_flag(plan["area"]),
               "--out", str(plan["data"])]
    if plan.get("cadence") is not None:
        command.extend(("--cadence", str(plan["cadence"])))
    if plan.get("forecast_start_hour"):
        command.extend(
            ("--forecast-start-hour", str(plan["forecast_start_hour"])))
    return command


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
            # Forwarded only when the config is bound to one: the GFS
            # front door no longer substitutes WSM6 for an absent
            # profile (the defect that once made this flag mandatory in
            # practice) -- an unnamed config's own suite is prepared as
            # written and its verification status reported.
            *_profile_flags(plan),
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
            *_profile_flags(plan),
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
            "Its runner is gpuwm-prepared-tree-forecast (module form: "
            "python -m gpuwm.prepared_domain_tree_forecast); see "
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
            # Popen + communicate is exactly what subprocess.run does;
            # spelling it out is what publishes the child's pid, which
            # the interrupt path has to be able to NAME (it does not
            # signal it -- see GoInterrupted).
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, errors="replace", cwd=str(_stage_cwd()))
            box["pid"] = proc.pid
            out, err = proc.communicate()
            box["completed"] = subprocess.CompletedProcess(
                command, proc.returncode, out, err)
        except BaseException as error:  # re-raised on this thread below
            box["error"] = error

    worker = threading.Thread(target=_wait, name=f"go-stage-{label}",
                              daemon=True)
    worker.start()
    next_beat = started + heartbeat_seconds
    try:
        while worker.is_alive():
            worker.join(timeout=0.2)
            now = time.monotonic()
            if worker.is_alive() and now >= next_beat:
                print(f"     .. {label}, {_elapsed_words(now - started)}"
                      f"{_progress_note(progress)}", flush=True)
                next_beat = now + heartbeat_seconds
        worker.join()
    except KeyboardInterrupt:
        # Ctrl-C.  The wait happens on this thread, so the interrupt
        # lands here; the stage subprocess is on the worker.  Hand the
        # stage name and that pid up and let go_main say it in one
        # sentence.  Nothing is signalled from here: in a terminal the
        # child already received its own SIGINT with the rest of the
        # foreground process group, and a `kill` issued by gpuwm at a
        # pid it merely observed is how a tool ends up stopping
        # something that was never its to stop.
        raise GoInterrupted(label, box.get("pid")) from None
    if "error" in box:
        raise box["error"]
    completed = box["completed"]
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        print(f"  FAILED  {label} (exit {completed.returncode})")
        # The stage's own refusal is usually its last few lines; replay
        # a readable tail by default and everything under --explain.
        lines = output.splitlines()
        tail = lines if explain else lines[-_FAILURE_TAIL_LINES:]
        if len(tail) < len(lines):
            print(f"    ... ({len(lines) - len(tail)} earlier line(s); "
                  "re-run with --explain to replay all of them)")
        for line in tail:
            print(f"    {line}")
        print(f"go: stopped at {label}; every later stage consumes this "
              "one's output, so nothing after it ran.")
        raise GoStageFailed(completed.returncode)
    print(f"  ok      {label} ({_elapsed_words(time.monotonic() - started)})")
    if explain:
        for line in output.splitlines():
            print(f"    {line}")
    else:
        # Warnings must not need a flag to be seen: a passing stage's
        # warning lines surface, one line each, nothing else.
        #
        # `note:` joins `warning:` here, and for the same reason rather
        # than for tidiness.  This tree uses `note:` for "true, and worth
        # knowing, and not a fault" -- the render stage says it when a
        # frame's declared inputs are absent and a product was therefore
        # not drawn.  That sentence exists precisely so a skipped product
        # is not a silent success; a chain that captured it and printed
        # `ok render` would have re-created the silence one level up.
        for line in output.splitlines():
            head = line.lstrip().lower()
            if head.startswith("warning:") or head.startswith("note:"):
                print(f"    {line.strip()}")


#: Failure-replay tail length: enough to carry any refusal message this
#: tree prints plus a Python traceback's tail, small enough to read.
_FAILURE_TAIL_LINES = 30


class GoStageFailed(Exception):
    """A stage exited nonzero; its output has already been replayed."""

    def __init__(self, code: int):
        super().__init__(f"stage exited {code}")
        self.code = code


def _tree_bytes(root) -> int:
    """Bytes under ROOT, counting what can be counted.

    Best-effort by construction: a tree a failed stage was mid-write in
    can lose a file between the walk and the stat, and a size is not
    worth an exception on an error path.
    """
    total = 0
    try:
        for path in Path(root).rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _partial_tree_note(root) -> str:
    """What a failed run left on disk, and that removing it is safe."""
    root = Path(root)
    if not root.exists():
        return (f"go: {root} was not created, so there is nothing on disk "
                "from this run.")
    return (f"go: {root} is a partial tree with no certification capsule, "
            f"holding {_human_bytes(_tree_bytes(root))}.  It is kept "
            f"because it is the evidence of what failed; nothing later "
            f"reads it, so it is safe to remove once you are done with "
            f"it.  Every stage is create-only, so a re-run needs a new "
            f"--outdir either way.")
def _experimental_labels(plan: dict) -> tuple[str, ...]:
    """(label, reason) for the experimental options this plan selects.

    Read from the registry through the config, never from a scheme name,
    so registering a second experimental option surfaces it here without
    a further edit.  Any failure to read the config is silent by design:
    the banner is not the place a broken config is diagnosed, and the
    loader that IS that place runs a moment later.
    """
    try:
        from gpuwm.experiment import is_experiment_toml, load_experiment
        from gpuwm.physics_compat import _experimental_components

        path = plan.get("config")
        if path is None:
            return ()
        if is_experiment_toml(path):
            configs = [dc.run for dc in load_experiment(path).domains]
        else:
            from gpuwm.config import load_config
            configs = [load_config(path)]
        clauses: dict[str, str] = {}
        for cfg in configs:
            for label, reason in _experimental_components(cfg):
                clauses.setdefault(label, reason)
        return tuple(sorted(clauses.items()))
    except Exception:
        return ()


#: Ctrl-C's exit code, by the shell convention 128 + SIGINT.
INTERRUPT_EXIT_CODE = 130


class GoInterrupted(Exception):
    """Ctrl-C arrived while ``label``'s subprocess (``pid``) was running.

    Carries the pid so the message can NAME the process gpuwm was
    waiting on, and deliberately carries no authority to kill it.  The
    bounded contract, shared with the multi-GPU orchestration lane:
    Ctrl-C returns 130, kills no child and no unrelated process, and an
    unobserved child pid is printed rather than acted on.
    """

    def __init__(self, label: str, pid: int | None):
        super().__init__(f"interrupted during {label}")
        self.label = label
        self.pid = pid


def _physics_words(plan: dict) -> str:
    """The banner's physics clause: profile id, or one honest sentence."""

    experimental = _experimental_labels(plan)
    if experimental:
        # An experimental option outranks the profile id in the banner.
        # "supported, not yet WRF-verified" would be the wrong sentence
        # for a scheme WRF does not have -- it promises a comparison
        # that is merely outstanding, and there is none to be
        # outstanding.  Warn-not-block: this is the wording, not a gate.
        # The REASON comes from physics_compat so the banner and the two
        # runners cannot say different things; only the shout is local.
        return ", ".join(
            f"{label}: {reason.replace('experimental,', 'EXPERIMENTAL,', 1)}"
            for label, reason in experimental)
    if plan.get("profile") is not None:
        return f"profile {plan['profile']}"
    return ("physics from the config as written (supported, not yet "
            "WRF-verified)")


def memory_gate(plan: dict, *, vram_gib: float | None = None) -> dict:
    """Price every phase of ``plan`` against this card, before the fetch.

    The whole point is WHERE this runs.  A configuration that cannot fit
    used to discover that in preprocessing -- after `gpuwm fetch` had
    pulled 81 GFS files down -- because the only memory estimate anyone
    computed described the forecast.  Both phases are priced here, at
    planning time, and the verdict names the binding phase and its
    number in one sentence.

    Returns a dict with ``verdict`` (always), ``refuse`` (a genuine OOM
    prediction: the binding phase does not fit even the whole card's
    free VRAM) and ``warn`` (it fits free VRAM but not the reserved
    budget -- said out loud, never blocking).
    """

    from gpuwm.core.preflight import (ReservePolicy, _load_experiment_any,
                                      config_forcing_source,
                                      device_memory_probe_subprocess,
                                      estimate_phases,
                                      profile_from_device_probe,
                                      unpriced_ingest_note)

    config = Path(plan["config"])
    source = config_forcing_source(config)
    if source is None:
        return {"verdict": unpriced_ingest_note(config, plan.get("source")),
                "refuse": False, "warn": True, "free_bytes": None}
    exp = _load_experiment_any(config)
    # This gate is about THIS machine -- it is the last thing between the
    # user and a download -- so the non-pool terms are priced against the
    # card that is actually here, not against the 170-SM reference the
    # module was calibrated on.  `gpuwm check` reads the same numbers off
    # the same device; the two must not disagree about one card.
    #
    # Both device questions -- the free VRAM the budget subtracts from
    # and the card's local-memory profile -- are asked in ONE short-lived
    # SUBPROCESS, never in-process.  An in-process memGetInfo or
    # deviceGetLimit stands up a CUDA primary context, and this process
    # outlives its gate as the chain's stage runner and progress printer:
    # that context sat on the card for the entire run -- measured
    # 0.486 GiB on the RTX 5090 -- as a consumer no term of the budget
    # computed below names.  The probe's context dies with the probe, so
    # the printer holds no device memory at all, and the forecast stage
    # (whose own context IS priced, in the reserve's device-overhead
    # term) gets the card the verdict described.
    probe = device_memory_probe_subprocess()
    profile = profile_from_device_probe(probe)
    # The cadence the domain was SIZED against, from the same [fetch]
    # table the fetch stage reads, so the two cannot disagree.
    cadence_h = plan.get("cadence")
    phases = estimate_phases(
        exp, source=source, vram_gib=vram_gib, profile=profile,
        ingest_forcing_interval_seconds=(
            float(cadence_h) * 3600.0 if cadence_h else None))

    if probe is None:
        # No device here (a planning box, a CI runner): price the phases
        # and print the verdict, but never refuse on a card we cannot see.
        return {"verdict": phases.verdict(None), "refuse": False,
                "warn": False, "free_bytes": None, "phases": phases}
    free = int(probe["free_bytes"])
    reserve = ReservePolicy.n0_alloc(
        exp, profile=profile,
        estimate_bytes=phases.forecast.alloc_estimate_bytes)
    budget = reserve.budget_bytes(free)
    peak = phases.peak_envelope_bytes
    return {
        "verdict": phases.verdict(budget),
        "refuse": peak > free,
        "warn": peak > budget,
        "free_bytes": free,
        "budget_bytes": budget,
        "phases": phases,
    }


def geography_refusal(geog_root: Path) -> str | None:
    """Why the prepare stage cannot build static fields, or ``None``.

    The prepare stage builds every domain's terrain, land use, soil and
    greenness out of the staged WPS_GEOG tree.  Without it there is no
    run -- and what a first-time reader used to get for that was three
    successful stages, a download, and then this, verbatim:

        FAILED  prepare (exit 2)
          rw-wps --source gfs: /.../WPS_GEOG/topo_gmted2010_30s/index.

    No verb, no remedy, and a sentence that ends in a path and a full
    stop.  Meanwhile ``gpuwm doctor`` had the whole answer, including
    the command, and ``go`` never asked it.

    Asked here for the same reason the memory gate is: BEFORE the fetch
    stage spends the user's bandwidth on forcing data for a run that
    cannot preprocess.

    This is a hard refusal and stays one.  It is not the warn-not-block
    case -- nothing here is a prediction that might be wrong.  The
    input is absent, the stage that needs it is next, and it was
    measured failing.  ``gpuwm doctor`` correctly exits 0 on the same
    gap, because a ~16 GB download nobody opted into is not a broken
    install; that is a statement about the INSTALL.  This is a
    statement about a RUN, and the two answers differ honestly.
    """

    from gpuwm.doctor import geography_gaps

    gaps = geography_gaps(Path(geog_root))
    if not gaps:
        return None
    return layered(
        f"the staged WPS_GEOG tree is not usable: "
        + "; ".join(gap.brief or gap.detail for gap in gaps) + ".\n"
        "  Every domain's static fields (terrain, land use, soil, "
        "greenness) are built from it by the prepare stage, so no "
        "stage after it can run.\n"
        "  remedy: gpuwm fetch-geog\n"
        f"  # stages the nine required datasets into {geog_root}\n"
        "  # gpuwm go CONFIG --geog-root DIR uses a tree staged elsewhere",
        "Refusing here, before the fetch stage downloads the forcing "
        "data, rather than in preprocessing after it.\n\n"
        "`gpuwm doctor` reports this same gap and still exits 0.  That "
        "is not a disagreement: the download is an explicit opt-in, so "
        "an install that did everything its documentation asked is not "
        "a broken install, and doctor is describing the install.  This "
        "message describes a RUN, which cannot proceed without the "
        "tree.\n\n"
        + "\n".join(f"  {gap.name}: {gap.detail}" for gap in gaps))


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
              f"{plan['cycle']}, {plan['hours']} h, "
              f"{_physics_words(plan)}{_lead_note(plan)}")
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
          f"{plan['cycle']}, {plan['hours']} h, {_physics_words(plan)}"
          f"{_lead_note(plan)}")

    # The interrupt contract covers the gate too: the memory probe
    # is a real subprocess (first-run staging), and a Ctrl-C landing
    # there deserves the same one sentence and rc 130 as any stage.
    try:
        # BEFORE the download, not after it.  `gpuwm fetch` is the stage that
        # costs the user gigabytes and minutes; a configuration that cannot
        # fit has to be told so on this side of it.
        if not getattr(args, "no_memory_gate", False):
            gate = memory_gate(plan)
            print(f"go: memory -- {gate['verdict']}")
            if gate["refuse"]:
                raise GoRefusal(
                    f"this configuration will not fit: {gate['verdict']}, and "
                    f"the card has {gate['free_bytes'] / (1024 ** 3):.2f} GiB "
                    "free right now.  Refusing here, BEFORE the fetch stage "
                    "downloads the forcing data, rather than in preprocessing "
                    "after it.\n"
                    "  remedy: re-size for this card -- gpuwm domain "
                    f"--vram-gib {gate['free_bytes'] // (1024 ** 3)} ... -- or "
                    "free VRAM and re-run\n"
                    "  # gpuwm go CONFIG --no-memory-gate runs it anyway")
            if gate["warn"]:
                print("go: WARNING -- that is above the reserved budget, "
                      "though it fits the free VRAM measured just now.  "
                      "Proceeding; a driver/other-process spike could still "
                      "OOM this run.")

        # Same rule as the memory gate, same side of the download.
        geography = geography_refusal(geog_root)
        if geography is not None:
            raise GoRefusal(geography)

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
        # D-05.  The interrupted arm below has always told the reader what
        # is on disk; this one raised and said nothing, so a failed
        # prepare left a scratch tree of a few hundred MB with no mention
        # anywhere that it existed, was incomplete, or could be removed.
        # Nothing is deleted here: the tree is the only evidence of what
        # went wrong, and a tool that tidies away the failure a person is
        # about to report is worse than one that leaves it.
        print(_partial_tree_note(plan["root"]), file=sys.stderr)
        return failure.code
    except GoInterrupted as stop:
        # One sentence, 130, and the truth about what is on disk.  The
        # partial tree has no certification capsule, which is what makes
        # its incompleteness visible to every receipt reader.
        child = ("" if stop.pid is None else
                 f"  # the {stop.label} process was pid {stop.pid}; gpuwm "
                 "signalled nothing and killed nothing\n")
        print(render(layered(
            f"go: interrupted during {stop.label}; no later stage ran and "
            f"{plan['root']} is a partial tree with no certification "
            "capsule.\n"
            f"{child}"
            f"  remedy: gpuwm go {_quote(plan['config'])} --outdir "
            "<a new directory>   # every stage is create-only",
            "Ctrl-C sends SIGINT to the whole foreground process group, "
            "so the stage subprocess received it directly and gpuwm did "
            "not (and will not) signal any pid itself -- a tool that "
            "kills pids it merely observed is a tool that eventually "
            "kills the wrong one. If the stage was launched into the "
            "background by a shell, SIGINT is SIG_IGN for the whole job "
            "and neither process can see a Ctrl-C at all; send SIGTERM "
            "there instead."),
            explain=explain, command="gpuwm go"), file=sys.stderr)
        return INTERRUPT_EXIT_CODE

    # The forecast writes its own validity verdict (health, stability,
    # input-identity gates) into report.json; say it here so the chain
    # ends with the one line a reader is looking for.
    try:
        report = json.loads(
            (plan["run"] / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    verdict = report.get("status")
    delivered = _boundary_interval_refusal(plan, report)
    if delivered is not None:
        print(f"go: {delivered}", file=sys.stderr)
        return 1
    if verdict:
        print(f"go: forecast validity {verdict}")
    print(f"go: wrote {plan['run']}")
    if rendered:
        print(f"go: rendered {plan['render']}")
    return 0


def _boundary_interval_refusal(plan: dict, report: dict) -> str | None:
    """Refuse a run whose lateral boundaries are not the cadence asked for.

    The forecast records ``input.boundary_interval_seconds`` -- the real
    spacing of the lateral boundary times it integrated against.  When
    the ``[fetch]`` table names a cadence, that number is what the
    config asked for, and the two disagreeing means the run tracked the
    driving model on a different clock from the one the file declares.

    Checked against the RECEIPT rather than against the fetch command,
    because the defect this closes was a fetch command that looked right
    and a run that was three times coarser: exit 0, "forecast validity
    PASS", and nothing anywhere naming the substitution.  A verdict that
    can only be reached by reading report.json by hand is not a verdict.
    """

    cadence = plan.get("cadence")
    if cadence is None:
        return None
    observed = report.get("input", {}).get("boundary_interval_seconds")
    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
        return None
    requested = float(cadence) * 3600.0
    if float(observed) == requested:
        return None
    return (
        f"the run integrated against {float(observed) / 3600.0:g} h lateral "
        f"boundaries, but [fetch] cadence = {cadence} in "
        f"{plan['config'].name} asks for {cadence:g} h.  The boundary clock "
        "is what ties a limited-area run to its driving model, so this is "
        "not a rounding difference and the run above is not the run the "
        "config describes.  Re-run after making the two agree; "
        f"{plan['run'] / 'report.json'} records what was delivered.")


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
                             "by `gpuwm domain --source gfs` -- the "
                             "wizard's default emission is exactly this "
                             "shape (--physics-profile optional: an "
                             "unbound config's own suite runs as written)")
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
    parser.add_argument("--no-memory-gate", action="store_true",
                        dest="no_memory_gate",
                        help="skip the before-the-fetch memory check that "
                             "refuses a configuration whose binding phase "
                             "cannot fit this card's free VRAM")
    parser.set_defaults(func=go_main)
    return parser


__all__ = [
    "GoRefusal", "GoStageFailed", "HEARTBEAT_SECONDS", "MANUAL_CHAIN",
    "ORCHESTRATED_SOURCES", "RUNNER_MODULE", "RUNNER_RELATIVE",
    "authority_command", "fetch_command", "forecast_command", "go_main",
    "manifest_command", "memory_gate", "plan_from_config",
    "prepare_command", "printable",
    "proof_digests", "register_cli", "render_command",
    "render_extra_missing", "resolve_bridge", "wrfout_frames",
]
