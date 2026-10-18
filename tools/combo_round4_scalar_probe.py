"""Recover a one-step reference only after all recorded field hashes agree.

Run reference and score on box CPU; replay must hold a GPU queue slot.
Raw arrays stay on this box. This supplemental one-step probe is not a sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from gpuwm.verify_exact.compare import field_digest, compare_runs
from gpuwm.verify_exact.recording import Recording, load_run_record, write_run_record
from gpuwm.verify_exact.replay import replay, time_step_seconds


def validate_reference(arrays, record, excluded=()):
    expected = record.digests[1]
    mismatches = []
    compared = 0
    for name, digest in expected.items():
        if name in excluded:
            continue
        compared += 1
        meta = record.fields[name]
        if name not in arrays or field_digest(arrays[name], meta.dtype, meta.shape) != digest:
            mismatches.append(name)
    if not compared or mismatches:
        raise ValueError(f"new scalar run does not match the recording: {mismatches}")
    return compared


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("reference", "replay", "score"), required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--combos-file", type=Path, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--combo", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wrf-exe", type=Path)
    ap.add_argument("--wrf-tables", type=Path)
    ap.add_argument("--candidate-hashes", type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rec = Recording(args.recordings, combos_file=args.combos_file)
    run = rec.runs[(args.case, args.combo)]
    ref = run.referee("scalar")
    if ref is None or not ref.designated or not run.referee_sound(ref):
        raise ValueError("probe requires a sound designated scalar referee")
    record = rec.reference_record(ref)
    dt = time_step_seconds(run.namelist)
    reference_dir = args.out / "reference"
    woof_dir = args.out / "woof"
    if args.phase == "reference":
        if not args.wrf_exe or not args.wrf_tables:
            raise ValueError("reference phase requires the existing scalar executable and its tables")
        stage = args.out / "wrf-stage"
        stage.mkdir(exist_ok=False)
        for table in args.wrf_tables.iterdir():
            if table.is_file() and (table.suffix in (".TBL", ".formatted", ".dat") or "DATA" in table.name):
                (stage / table.name).symlink_to(table.resolve())
        for name in ("wrfinput_d01", "wrfbdy_d01"):
            (stage / name).symlink_to((run.inputs / name).resolve())
        text = run.namelist.read_text()
        # Shorten the duration only; preserve every numerical and history setting.
        for key, value in (("run_days", 0), ("run_hours", 0), ("run_minutes", 0), ("run_seconds", int(dt))):
            text, count = re.subn(rf"(?m)^(\s*{key}\s*=).*$", rf"\g<1> {value},", text)
            if count != 1:
                raise ValueError(f"expected exactly one {key}")
        (stage / "namelist.input").write_text(text)
        shutil.copyfile(args.recordings / "provenance/sweep/iofields.txt", stage / "iofields.txt")
        command = [str(args.wrf_exe.resolve())]
        with (args.out / "wrf.log").open("w") as log:
            result = subprocess.run(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            raise ValueError(f"scalar WRF exited {result.returncode}")
        from gpuwm.netcdf_bridge import open_dataset
        paths = list(stage.glob("wrfout_d01_*"))
        if len(paths) != 1:
            raise ValueError(f"expected one two-frame history file, got {len(paths)}")
        dataset = open_dataset(paths[0])
        dataset.set_auto_maskandscale(False)
        names = [n for n in record.fields if n in dataset.variables and n not in rec.unreliable_fields]
        dataset.prefetch(names)
        arrays = {n: dataset.variables[n][:][1] for n in names}
        compared = validate_reference(arrays, record, rec.unreliable_fields)
        write_run_record(reference_dir, fields=record.fields, frames={1: record.digests[1]},
                         steps_per_frame=1, raw={1: {"U": arrays["U"]}})
        receipt = dict(command=command, exit=result.returncode, compared_fields=compared,
                       matched_recording=True, omitted_unreliable=list(rec.unreliable_fields),
                       binary_sha256=hashlib.sha256(args.wrf_exe.read_bytes()).hexdigest(),
                       input_hashes={name: hashlib.sha256((run.inputs / name).read_bytes()).hexdigest()
                                     for name in ("wrfinput_d01", "wrfbdy_d01")},
                       namelist_sha256=hashlib.sha256(text.encode()).hexdigest())
        (args.out / "reference.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # The receipt records the numbers; raw model history is disposable.
        size = sum(p.stat().st_size for p in stage.iterdir() if p.is_file() and not p.is_symlink())
        shutil.rmtree(stage)
        (args.out / "deleted-raw.log").write_text(f"{stage}\t{size}\n")
        print(json.dumps(receipt))
    elif args.phase == "replay":
        from gpuwm.verify_exact.cli import _start_time
        start = _start_time(run.namelist)
        result = replay(run, args.out / "woof-stage", reference_meta=record.fields,
                        keep_steps={1}, run_seconds=dt, step_seconds=dt, start=start)
        write_run_record(woof_dir, fields=result.fields, frames=result.digests,
                         steps_per_frame=result.steps_per_frame, raw=result.kept,
                         extra={"woof": {"status": result.status, "error": result.error,
                                         "wall_s": result.wall_s, "phases": result.phases}})
        print(json.dumps(dict(status=result.status, error=result.error, wall_s=result.wall_s)))
        return 0 if result.status == "OK" else 1
    else:
        reference = load_run_record(reference_dir)
        candidate = load_run_record(woof_dir)
        if args.candidate_hashes:
            recorded = load_run_record(args.candidate_hashes.parent)
            mismatches = [name for name, digest in candidate.digests[1].items()
                          if recorded.digests.get(1, {}).get(name) != digest]
            if mismatches:
                raise ValueError(f"one-step probe differs from the completed sweep replay: {mismatches}")
        comparison = compare_runs(candidate, reference, exclude=rec.unreliable_fields, through_step=1)
        detail = comparison.as_json()
        (args.out / "comparison.json").write_text(json.dumps(detail, indent=2) + "\n")
        print(json.dumps(detail))
        for folder in (reference_dir, woof_dir):
            for blob in folder.glob("step*.bin"):
                size = blob.stat().st_size
                blob.unlink()
                with (args.out / "deleted-raw.log").open("a") as log:
                    log.write(f"{blob}\t{size}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
