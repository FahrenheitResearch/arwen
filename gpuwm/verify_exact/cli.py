"""`gpuwm verify-exact`: the whole combo sweep, one card, one process.

Reads a recording root, replays every selected run through WOOF in this
process, scores each against every recorded WRF build, and prints one line
per run: ``PASS`` or the first diverging step, field and max ULP.  Per-scheme
column fixtures under ``--fixtures`` are scored the same way.  Exit status:
0 when at least one run or fixture passed (PASS, WOOF-ONLY-OK, fixture
PASS) and none failed; 1 when any failed; 2 on a usage error, including a
selection that matches nothing or a --combos id the recording lacks; 3 when
nothing failed and nothing passed, which proves nothing (only WRF refusals,
unsound or unknown referees, informational builds).

``--score-only`` rescoring needs no card: it reads WOOF's stored replays
from ``--out`` and scores them again, for example after a raw-array overlay
arrives (``--raw-overlay``) that lets a digest miss be measured in ULP.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpuwm.verify_exact.compare import FieldMeta, RunRecord, compare_runs
from gpuwm.verify_exact.coverage import describe as describe_coverage, pair_coverage
from gpuwm.verify_exact.recording import (Recording, ReplayRun, load_run_record,
                                          write_run_record)
from gpuwm.verify_exact.scoring import FAIL_OUTCOMES, Verdict, decide

#: Outcomes that count as a run proving something (the exit status is 3 when
#: nothing failed and none of these occurred).
PASS_OUTCOMES = frozenset({"PASS", "WOOF-ONLY-OK"})

#: Exit status when nothing failed and nothing passed.
EXIT_NO_PROOF = 3

#: The combo factors that name a scheme, carried into each result row so a
#: report can group failures by scheme without re-reading the combo list.
SCHEME_FACTORS = ("mp", "pbl", "sfclay", "lsm", "cu", "rad", "sf_lake_physics", "urban",
                  "gwd", "shcu")


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "verify-exact",
        help="replay recorded WRF runs through WOOF and score every step bitwise",
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Recording and fixture formats: docs/dev/verify-exact.md.  Run under "
               "GPUWM_WRF_EXACT=1 (the strict build is the 0 ULP standard); the command "
               "refuses default arithmetic unless --default-arithmetic says so.")
    parser.add_argument("--recordings", type=Path, metavar="DIR",
                        help="recording root (runs.csv, inputs/, <build>/<case>/<combo>/)")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR",
                        help="where WOOF's replays and the results are written")
    parser.add_argument("--fixtures", type=Path, default=None, metavar="DIR",
                        help="per-scheme column fixtures (default: RECORDINGS/fixtures when present)")
    parser.add_argument("--combos", default="", metavar="IDS",
                        help="comma-separated combo ids (default: every combo)")
    parser.add_argument("--cases", default="", metavar="NAMES", help="comma-separated case names")
    parser.add_argument("--strata", default="", metavar="LETTERS",
                        help="comma-separated strata (A, B, C, D, G, S...)")
    parser.add_argument("--shard", default="1/1", metavar="I/N",
                        help="run every N-th selected run starting at the I-th (1-based)")
    parser.add_argument("--referee", choices=("scalar", "strict"), default="scalar",
                        help="the WRF build a PASS/FAIL is decided against (its Thompson-fixed "
                             "variant where recorded); every recorded build is reported")
    parser.add_argument("--combos-file", type=Path, default=None, metavar="JSON",
                        help="combo list (default: RECORDINGS/provenance/sweep/combos.json)")
    parser.add_argument("--run-summary", type=Path, default=None, metavar="CSV",
                        help="per-run-summary.csv with clean_reference (default: RECORDINGS/...)")
    parser.add_argument("--raw-overlay", type=Path, action="append", default=[], metavar="DIR",
                        help="extra root holding reference raw arrays in the recording layout")
    parser.add_argument("--repeat", choices=("auto", "off"), default="auto",
                        help="auto: run WOOF twice for combos whose checks ask for a repeat")
    parser.add_argument("--score-only", action="store_true",
                        help="no card: rescore WOOF replays already stored under --out")
    parser.add_argument("--keep-history", action="store_true",
                        help="keep each replay's wrfout files (default: hash and delete)")
    parser.add_argument("--list", action="store_true", help="print the selection and exit")
    parser.add_argument("--default-arithmetic", action="store_true",
                        help="allow a run without GPUWM_WRF_EXACT=1 (not a 0 ULP proof)")
    parser.add_argument("--no-replay", action="store_true",
                        help="score fixtures only")
    parser.add_argument("--write-needs", type=Path, default=None, metavar="JSON",
                        help="list the referee raw arrays that would turn each digest miss "
                             "into a ULP measurement (input to tools/verify_exact_raw_subset.py)")
    parser.set_defaults(func=main)


def _split(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def _shard(text: str) -> tuple[int, int]:
    try:
        index, count = (int(v) for v in text.split("/"))
    except ValueError:
        raise ValueError(f"--shard {text!r}: expected I/N") from None
    if not 1 <= index <= count:
        raise ValueError(f"--shard {text!r}: I must be 1..N")
    return index, count


def _exact_environment() -> dict[str, str]:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("GPUWM_WRF_EXACT")}


def _exact_controls() -> dict[str, bool]:
    from gpuwm import wrf_exact
    return wrf_exact.effective_controls()


def _engine_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    identity = {"tree_root": str(root)}
    for key, command in (("head", ["rev-parse", "HEAD"]), ("index_tree", ["write-tree"])):
        try:
            identity[key] = subprocess.run(["git", "-C", str(root), *command], check=True,
                                           capture_output=True, text=True,
                                           timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            identity[key] = "unknown"
    return identity


def _start_time(namelist: Path) -> datetime:
    from gpuwm.fortran_namelist import parse_namelist
    tc = parse_namelist(namelist)["time_control"]
    parts = [int((tc.get(f"start_{k}") or [0])[0])
             for k in ("year", "month", "day", "hour", "minute", "second")]
    return datetime(*parts)


def _factors(run: ReplayRun) -> dict:
    factors = run.combo_record.get("factors") or {}
    return {k: factors[k] for k in SCHEME_FACTORS if k in factors}


def _woof_record_dir(out: Path, run: ReplayRun, repeat: bool = False) -> Path:
    return out / ("woof-repeat" if repeat else "woof") / run.case / run.combo


def _plan_steps(run: ReplayRun) -> int:
    """How many steps WOOF integrates: as far as any scorable reference got."""

    refs = run.scorable()
    if not refs:
        return run.steps
    reach = [r.steps_requested if r.status == "OK" else r.last_good_step for r in refs.values()]
    return max(1, max(reach))


def _planned_last_step(run: ReplayRun) -> int:
    """The last history step the plan reaches: frames fall every steps_per_frame steps."""

    per_frame = max(1, run.steps_per_frame)
    return (_plan_steps(run) // per_frame) * per_frame


class Sweep:
    def __init__(self, args, recording: Recording):
        self.args = args
        self.recording = recording
        self.out = Path(args.out)
        self.exclude = tuple(recording.unreliable_fields)

    # -- the WOOF side -------------------------------------------------
    def _replay_once(self, run: ReplayRun, *, repeat: bool, meta, keep_steps, start, dt):
        from gpuwm.verify_exact.replay import replay

        record_dir = _woof_record_dir(self.out, run, repeat)
        work = self.out / "scratch" / ("repeat" if repeat else "first") / run.case / run.combo
        steps = _plan_steps(run)
        woof = replay(run, work, reference_meta=meta, keep_steps=keep_steps,
                      run_seconds=steps * dt, step_seconds=dt, start=start,
                      keep_history=self.args.keep_history)
        return record_dir, woof

    def _store(self, record_dir: Path, woof, refs_records) -> None:
        # Keep WOOF's arrays only where a later ULP measurement can use them:
        # fields that differ from at least one reference at a kept step.
        kept = {}
        for step, arrays in woof.kept.items():
            names = [n for n in arrays
                     if any(step in r.digests and n in r.digests[step]
                            and r.digests[step][n] != woof.digests.get(step, {}).get(n)
                            for r in refs_records)]
            if names:
                kept[step] = {n: arrays[n] for n in names}
        write_run_record(record_dir, fields=woof.fields, frames=woof.digests,
                         steps_per_frame=woof.steps_per_frame, raw=kept,
                         extra={"woof": {"status": woof.status, "error": woof.error,
                                         "wall_s": woof.wall_s,
                                         "nonfinite_step": woof.nonfinite_step,
                                         "phases": woof.phases,
                                         "exact_environment": _exact_environment(),
                                         "exact_controls": _exact_controls()}})

    def _load_woof(self, run: ReplayRun, repeat: bool = False):
        record_dir = _woof_record_dir(self.out, run, repeat)
        if not (record_dir / "hashes.json").is_file():
            return None, None
        document = json.loads((record_dir / "hashes.json").read_text(encoding="utf-8"))
        return load_run_record(record_dir, label="woof"), document.get("woof", {})

    # -- one run -------------------------------------------------------
    def score(self, run: ReplayRun) -> Verdict:
        verdict = Verdict(run.combo, run.case, run.stratum, "", factors=_factors(run),
                          declared_divergences=tuple(run.combo_record.get("declared_divergences") or ()))
        refs = run.scorable()
        referee = run.referee(self.args.referee)
        verdict.referee = referee.build if referee else None
        verdict.referee_soundness = run.referee_soundness(referee) if referee else ""
        verdict.referee_sound = verdict.referee_soundness == "sound" if referee else True
        verdict.designated = referee.designated if referee else False
        strict = run.references.get("strict")
        if referee is None and strict is not None:
            status = strict.status if strict.status.startswith("WRF") else f"WRF {strict.status}"
            verdict.wrf_note = f"{status}: {strict.detail}".strip()
        if run.inputs is None:
            verdict.outcome = "WRF-REFUSED"
            return verdict
        ref_records = {b: self.recording.reference_record(r) for b, r in refs.items()}
        if self.args.score_only:
            woof_record, info = self._load_woof(run)
            if woof_record is None:
                verdict.outcome = "WOOF-CRASH"
                verdict.woof_status, verdict.woof_error = "CRASH", "no stored replay under --out"
                return verdict
            repeat_record, _ = self._load_woof(run, repeat=True)
        else:
            meta: dict[str, FieldMeta] = {}
            for record in ref_records.values():
                for name, field_meta in record.fields.items():
                    meta.setdefault(name, field_meta)
            keep = {s for r in refs.values() for s in r.raw_steps}
            dt = _step_seconds(run)
            start = _start_time(run.namelist)
            record_dir, woof = self._replay_once(run, repeat=False, meta=meta,
                                                 keep_steps=keep, start=start, dt=dt)
            self._store(record_dir, woof, ref_records.values())
            woof_record = RunRecord(woof.fields, woof.digests, woof.kept.get,
                                    tuple(sorted(woof.kept)), "woof")
            info = {"status": woof.status, "error": woof.error, "wall_s": woof.wall_s,
                    "nonfinite_step": woof.nonfinite_step, "phases": woof.phases}
            repeat_record = None
            if (self.args.repeat == "auto" and run.wants_repeat() and woof.status == "OK"):
                repeat_dir, again = self._replay_once(run, repeat=True, meta=meta,
                                                      keep_steps=set(), start=start, dt=dt)
                self._store(repeat_dir, again, [woof_record])
                repeat_record = RunRecord(again.fields, again.digests, label="woof-repeat")
                info["wall_s"] = round(info["wall_s"] + again.wall_s, 2)
        verdict.woof_status, verdict.woof_error = initialisation_refusal(
            str(info.get("status", "")), str(info.get("error", "")), woof_record)
        verdict.woof_nonfinite_step = info.get("nonfinite_step")
        verdict.wall_s = float(info.get("wall_s") or 0.0)
        verdict.phases = dict(info.get("phases") or {})
        for build, record in ref_records.items():
            ref = refs[build]
            verdict.comparisons[build] = compare_runs(
                woof_record, record, exclude=self.exclude,
                through_step=None if ref.status == "OK" else ref.last_good_step)
        if repeat_record is not None:
            verdict.repeat = compare_runs(repeat_record, woof_record, exclude=self.exclude)
        verdict.outcome = decide(
            woof_status=verdict.woof_status, nonfinite_step=verdict.woof_nonfinite_step,
            referee=verdict.referee, comparisons=verdict.comparisons,
            referee_sound=verdict.referee_sound, designated=verdict.designated,
            repeat=verdict.repeat, repeat_required=run.wants_repeat(),
            planned_last_step=_planned_last_step(run),
            woof_last_step=woof_record.last_step if woof_record.digests else None)
        return verdict


def initialisation_refusal(status: str, error: str, record) -> tuple[str, str]:
    """A named refusal raised while the runner initialised is a refusal, not a crash.

    The door's own checks fire before the runner starts; several checks of
    the same kind (a soil value out of range, a terrain-drag input this
    route cannot supply) fire inside the runner before its first step.  A
    ``ValueError`` with no history frame written is that, and is reported
    as WOOF-REFUSED; anything else, or anything after a frame, is a crash.
    """

    if status == "CRASH" and not record.digests and error.startswith("ValueError"):
        return "REFUSED", "at initialisation: " + error
    return status, error


def raw_needs(verdict: Verdict, run: ReplayRun, *, max_fields: int = 4) -> list[dict]:
    """Referee arrays that would measure this verdict's first miss in ULP.

    One step (the first step at or after the miss that the recording kept
    raw), the lead field and up to ``max_fields - 1`` more of the fields
    that differ there.  Empty when the miss is already measured.
    """

    if verdict.referee is None or verdict.referee not in verdict.comparisons:
        return []
    comparison = verdict.comparisons[verdict.referee]
    if comparison.first_step is None or comparison.lead_detail() is not None:
        return []
    ref = run.references[verdict.referee]
    steps = [s for s in ref.raw_steps if s >= comparison.first_step]
    if not steps:
        return []
    return [{"build": ref.build, "case": run.case, "combo": run.combo, "step": steps[0],
             "fields": list(comparison.first_fields[:max_fields])}]


def _step_seconds(run: ReplayRun) -> float:
    from gpuwm.verify_exact.replay import time_step_seconds
    return time_step_seconds(run.namelist)


def main(args) -> int:
    try:
        return _main(args)
    except ValueError as error:
        print(f"gpuwm verify-exact: {error}", file=sys.stderr)
        return 2


def _select(recording: Recording, args) -> list[ReplayRun]:
    """The selected runs, before sharding.  A selection that names a combo,
    case or stratum the recording does not hold, or that selects nothing, is
    a usage error: a typo must not read as a clean run."""

    wanted = {"--combos": (_split(args.combos), {c for _, c in recording.runs}),
              "--cases": (_split(args.cases), {c for c, _ in recording.runs}),
              "--strata": (_split(args.strata), {r.stratum for r in recording.runs.values()})}
    for flag, (names, known) in wanted.items():
        unknown = [n for n in names if n not in known]
        if unknown:
            raise ValueError(f"{flag} names {', '.join(unknown)}, which the recording "
                             f"{recording.root} does not hold")
    runs = recording.select(combos=wanted["--combos"][0], cases=wanted["--cases"][0],
                            strata=wanted["--strata"][0])
    if not runs:
        raise ValueError(f"the selection matches no run in {recording.root}")
    return runs


def _main(args) -> int:
    started = time.perf_counter()
    index, count = _shard(args.shard)
    out = Path(args.out)
    fixtures_root = args.fixtures
    if fixtures_root is None and args.recordings is not None:
        candidate = Path(args.recordings) / "fixtures"
        fixtures_root = candidate if candidate.is_dir() else None
    runs: list[ReplayRun] = []
    recording = None
    if not args.no_replay:
        if args.recordings is None:
            raise ValueError("--recordings is required unless --no-replay")
        recording = Recording(args.recordings, combos_file=args.combos_file,
                              summary_file=args.run_summary, overlays=args.raw_overlay)
        runs = _select(recording, args)[index - 1::count]
    if args.list:
        for run in runs:
            referee = run.referee(args.referee)
            print(f"{run.combo:<16} {run.case:<22} steps {_plan_steps(run):>3}  referee "
                  f"{referee.build if referee else '-'}  builds {','.join(sorted(run.scorable()))}"
                  + ("" if run.inputs else "  (no inputs: WRF refused)"))
        print(f"{len(runs)} runs selected")
        return 0
    needs_card = bool(runs) and not args.score_only
    if needs_card or fixtures_root is not None:
        from gpuwm import wrf_exact
        if not wrf_exact.ENABLED and not args.default_arithmetic:
            raise ValueError("GPUWM_WRF_EXACT=1 is not set, so this process would run default "
                             "arithmetic and no 0 ULP claim could follow; set it before starting "
                             "the process, or pass --default-arithmetic for a non-proof run")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{index}of{count}" + ("-score" if args.score_only else "")
    results_path = out / "results" / f"{tag}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    from gpuwm import wrf_exact
    header = {"started": datetime.now().isoformat(timespec="seconds"),
              "argv": sys.argv, "exact_environment": _exact_environment(),
              "exact_controls": wrf_exact.effective_controls(),
              "engine": _engine_identity(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    (out / "results" / f"{tag}.header.json").write_text(json.dumps(header, indent=1), encoding="utf-8")
    controls = ",".join(k for k, v in header["exact_controls"].items() if v) or "none"
    print(f"gpuwm verify-exact: {len(runs)} runs, shard {index}/{count}, referee {args.referee}, "
          f"exact env {header['exact_environment'] or 'none'}, strict controls {controls}",
          flush=True)
    counts: dict[str, int] = {}
    needs: list[dict] = []
    outcomes: dict[tuple[str, str], str] = {}
    failed = passed = 0
    with results_path.open("w", encoding="utf-8") as results:
        if recording is not None:
            sweep = Sweep(args, recording)
            for run in runs:
                verdict = sweep.score(run)
                print(verdict.line(), flush=True)
                results.write(json.dumps(verdict.as_json()) + "\n")
                results.flush()
                counts[verdict.outcome] = counts.get(verdict.outcome, 0) + 1
                outcomes[(run.combo, run.case)] = verdict.outcome
                failed += verdict.failed
                passed += verdict.outcome in PASS_OUTCOMES
                needs.extend(raw_needs(verdict, run))
        if fixtures_root is not None:
            from gpuwm.verify_exact.fixtures import discover_fixtures, run_fixture
            for fixture in discover_fixtures(fixtures_root):
                result = run_fixture(fixture)
                print(result.line(), flush=True)
                results.write(json.dumps({"fixture": result.as_json()}) + "\n")
                key = "fixture " + result.outcome
                counts[key] = counts.get(key, 0) + 1
                failed += result.outcome != "PASS"
                passed += result.outcome == "PASS"
    if args.write_needs is not None:
        Path(args.write_needs).write_text(json.dumps(needs, indent=1), encoding="utf-8")
    if recording is not None and outcomes:
        cover = pair_coverage(recording.combos, outcomes)
        (out / "results" / f"{tag}.coverage.json").write_text(json.dumps(cover, indent=1),
                                                              encoding="utf-8")
        scope = "" if count == 1 else f" (shard {index}/{count} only)"
        for row in cover:
            print(f"gpuwm verify-exact: {describe_coverage(row)}{scope}", flush=True)
    wall = time.perf_counter() - started
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    print(f"gpuwm verify-exact: {summary or 'nothing selected'}; wall {wall:.1f} s", flush=True)
    if failed:
        return 1
    if not passed:
        print("gpuwm verify-exact: no proof: no run or fixture passed, and none failed "
              f"(exit {EXIT_NO_PROOF})", flush=True)
        return EXIT_NO_PROOF
    return 0


__all__ = ["FAIL_OUTCOMES", "main", "register_cli"]
