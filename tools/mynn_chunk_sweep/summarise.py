#!/usr/bin/env python
"""Turn one sweep arm into a row, and a finished sweep into a verdict.

Two modes, both driven by ``sweep.sh``:

``--arm``      read one arm's own step log and run receipt, append a row to
               the sweep TSV.  A root CYCLE is the parent's step plus the
               nest sub-steps that follow it, which is the unit the leg's
               wall time is actually made of; the step log records each
               domain's step separately.

``--collect``  read the TSV and every digest list, print the table, name the
               fastest width, and state whether the forecast moved.

Nothing here is a tolerance.  The digest comparison is exact: every arm must
write byte-identical frames, because the column chunk is workspace shape and
nothing else.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys


FIELDS = (
    "pass", "chunk", "receipt_chunk", "receipt_source", "cycles",
    "quiet_n", "quiet_median_s", "quiet_mean_s", "quiet_min_s",
    "nest_rad_n", "nest_rad_median_s", "parent_rad_n", "parent_rad_median_s",
    "first_cycle_s", "wall_s", "steps_reported",
    "pool_peak_bytes", "card_peak_bytes", "workspace_bytes",
    "frames", "digest_sha256", "foreign_seen", "own_seen", "outdir",
)


def _step_log(outdir: pathlib.Path) -> list[dict]:
    path = outdir / "progress.jsonl"
    if not path.is_file():
        raise SystemExit(f"no step log at {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _root_cycles(rows: list[dict]) -> list[float]:
    """Wall seconds per root cycle: the parent's step plus its sub-steps.

    The parent is the domain with the smallest id that takes steps.  A cycle
    opens at each of its step records and closes at the next one, so every
    nest sub-step in between is charged to the cycle it belongs to.
    """
    steps = [row for row in rows if row.get("event") == "step"]
    if not steps:
        raise SystemExit("step log carries no step events")
    parent = min(int(row["domain"]) for row in steps)
    cycles: list[float] = []
    current: float | None = None
    for row in steps:
        seconds = float(row.get("step_wall_seconds") or 0.0)
        if int(row["domain"]) == parent:
            if current is not None:
                cycles.append(current)
            current = seconds
        elif current is not None:
            current += seconds
    if current is not None:
        cycles.append(current)
    return cycles


def _populations(cycles: list[float], warm: int) -> dict:
    """Split the cycles the way the profile of record splits them.

    Two populations and no middle: quiet cycles, and the cycles a radiation
    call lands in.  The threshold is relative to the median of the arm's own
    cycles, so it does not carry a number measured on some other card.
    """
    timed = cycles[warm:] or cycles
    median = statistics.median(timed)
    quiet = [value for value in timed if value < 1.5 * median]
    heavy = sorted(value for value in timed if value >= 1.5 * median)
    nest, parent = heavy, []
    if len(heavy) >= 4:
        split = statistics.median(heavy)
        nest = [value for value in heavy if value < 2.0 * split]
        parent = [value for value in heavy if value >= 2.0 * split]
    return {
        "cycles": len(cycles),
        "quiet_n": len(quiet),
        "quiet_median_s": round(statistics.median(quiet), 5) if quiet else "",
        "quiet_mean_s": round(statistics.fmean(quiet), 5) if quiet else "",
        "quiet_min_s": round(min(quiet), 5) if quiet else "",
        "nest_rad_n": len(nest),
        "nest_rad_median_s": round(statistics.median(nest), 5) if nest else "",
        "parent_rad_n": len(parent),
        "parent_rad_median_s": (round(statistics.median(parent), 5)
                                if parent else ""),
        "first_cycle_s": round(cycles[0], 5),
    }


def _card_rows(cardwatch: pathlib.Path | None, start: str, end: str):
    """Each cardwatch sample inside this arm's window, as process names.

    A sample line is ``utc TAB device_used TAB "pid, name, used;pid, ..."``.
    The third column is empty when the card held nothing.
    """
    if cardwatch is None or not cardwatch.is_file():
        return None
    samples = []
    for line in cardwatch.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not (start <= parts[0] <= end):
            continue
        names = []
        for entry in parts[2].split(";"):
            entry = entry.strip()
            if not entry:
                continue
            fields = [field.strip() for field in entry.split(",")]
            names.append(fields[1] if len(fields) > 1 else fields[0])
        samples.append(names)
    return samples


def _occupancy(cardwatch: pathlib.Path | None, start: str, end: str,
               own_prefix: str | None):
    """Samples that saw a FOREIGN process, and samples that saw our own.

    This column used to count every sample whose process list was non-empty,
    which includes the arm's own forecast -- so it read 32 to 42 for arms
    that ran completely alone, and could never read zero while a run was in
    progress.  A column that cannot take the value it is being read for says
    nothing.  It now tests each process name against the engine install this
    sweep runs from: a name outside that prefix is somebody else's job.

    Foreign jobs are recorded, never killed: an arm that ran beside one is
    not thrown away here, it is marked so the collect stage can decline to
    compare it with one that ran alone.  ``own_seen`` is the other half of
    the same check: an arm whose own forecast never appeared in the samples
    means the sampler was idle, not the card.
    """
    samples = _card_rows(cardwatch, start, end)
    if samples is None:
        return "n/a", "n/a"
    if not own_prefix:
        # Say so, rather than emit a number nobody can interpret.
        return "no-prefix", "no-prefix"
    foreign = sum(1 for names in samples
                  if any(not name.startswith(own_prefix) for name in names))
    own = sum(1 for names in samples
              if any(name.startswith(own_prefix) for name in names))
    return foreign, own


def _receipt(outdir: pathlib.Path) -> dict:
    path = outdir / "evidence" / "run-receipt.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    memory = document.get("memory", {})
    chunk = memory.get("mynn_column_chunk") or {}
    return {
        "receipt_chunk": chunk.get("chunk", ""),
        "receipt_source": chunk.get("source", ""),
        "workspace_bytes": chunk.get("workspace_bytes", ""),
        "pool_peak_bytes": memory.get("cupy_pool_peak_used_bytes_observed", ""),
        "card_peak_bytes": memory.get("gpu_peak_used_bytes_observed", ""),
        "wall_s": round(float(document.get("wall_seconds") or 0.0), 3),
    }


def _digest_of_digests(path: pathlib.Path) -> tuple[int, str]:
    import hashlib

    body = path.read_bytes()
    lines = [line for line in body.splitlines() if line.strip()]
    return len(lines), hashlib.sha256(body).hexdigest()


def arm(args) -> int:
    outdir = pathlib.Path(args.outdir)
    rows = _step_log(outdir)
    cycles = _root_cycles(rows)
    record = {field: "" for field in FIELDS}
    record.update(_populations(cycles, args.warm))
    record.update(_receipt(outdir))
    record["pass"] = args.pass_index
    record["chunk"] = args.chunk
    record["outdir"] = str(outdir)
    starts = [row for row in rows if row.get("event") == "run_start"]
    ends = [row for row in rows if row.get("event") == "run_end"]
    record["steps_reported"] = ends[0].get("steps", "") if ends else ""
    if starts and ends:
        import datetime as dt

        def stamp(row):
            return dt.datetime.fromtimestamp(
                int(row["emitted_unix_ms"]) / 1000.0,
                dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        record["foreign_seen"], record["own_seen"] = _occupancy(
            pathlib.Path(args.cardwatch) if args.cardwatch else None,
            stamp(starts[0]), stamp(ends[-1]), args.own_prefix)
    frames, digest = _digest_of_digests(pathlib.Path(args.digests))
    record["frames"] = frames
    record["digest_sha256"] = digest
    tsv = pathlib.Path(args.tsv)
    new = not tsv.exists()
    with tsv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        if new:
            writer.writeheader()
        writer.writerow(record)
    return 0


def collect(args) -> int:
    work = pathlib.Path(args.work)
    tsv = work / "sweep.tsv"
    if not tsv.is_file():
        raise SystemExit(f"no sweep.tsv under {work}")
    with tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise SystemExit("sweep.tsv has no arms")

    print(f"MYNN column-chunk sweep, {len(rows)} arms, "
          f"{len({row['chunk'] for row in rows})} widths")
    print()
    header = ("chunk", "pass", "quiet median s", "quiet n", "wall s",
              "workspace MiB", "pool peak MiB", "receipt", "foreign",
              "own")
    print("%10s %5s %15s %8s %9s %14s %14s %16s %8s %6s" % header)
    for row in sorted(rows, key=lambda r: (int(r["chunk"]), int(r["pass"]))):
        def mib(value):
            return f"{int(value) / 2 ** 20:.0f}" if value else "-"
        print("%10s %5s %15s %8s %9s %14s %14s %16s %8s %6s" % (
            row["chunk"], row["pass"], row["quiet_median_s"], row["quiet_n"],
            row["wall_s"], mib(row["workspace_bytes"]),
            mib(row["pool_peak_bytes"]),
            f"{row['receipt_chunk']}/{row['receipt_source']}",
            row["foreign_seen"], row.get("own_seen", "")))
    print()

    # Every arm must have run the width it was asked for.  An arm whose
    # receipt names a different width is timing something else.
    mismatched = [row for row in rows
                  if str(row["receipt_chunk"]) != str(row["chunk"])]
    if mismatched:
        print("REFUSING to rank: these arms did not run the width they were "
              "given -- the override did not reach the run:")
        for row in mismatched:
            print(f"  asked {row['chunk']}, receipt says "
                  f"{row['receipt_chunk']} ({row['receipt_source']})")
        return 2

    by_width: dict[int, list[float]] = {}
    for row in rows:
        if row["quiet_median_s"]:
            by_width.setdefault(int(row["chunk"]), []).append(
                float(row["quiet_median_s"]))
    ranked = sorted(by_width.items(), key=lambda item: statistics.median(item[1]))
    base = statistics.median(by_width[min(by_width)]) if by_width else 0.0
    print("ranked by the median of each width's passes, against the "
          f"narrowest width ({min(by_width)} columns, {base:.4f} s):")
    for width, values in ranked:
        median = statistics.median(values)
        spread = (max(values) - min(values)) if len(values) > 1 else 0.0
        print(f"  {width:>7} columns  {median:.4f} s per quiet root cycle  "
              f"{base / median if median else 0:.3f}x  "
              f"pass spread {spread:.4f} s")
    if ranked:
        print(f"\nFASTEST: {ranked[0][0]} columns")
        if len(ranked) > 1:
            first, second = ranked[0], ranked[1]
            gap = statistics.median(second[1]) - statistics.median(first[1])
            spread = max((max(v) - min(v)) for _, v in ranked)
            if gap <= spread:
                print(f"  BUT the gap to {second[0]} columns ({gap:.4f} s) is "
                      f"not larger than the worst pass-to-pass spread "
                      f"({spread:.4f} s): these two widths are NOT "
                      "distinguished by this sweep, and the narrower one is "
                      "the cheaper default")

    digests = sorted((work / "digests").glob("*.txt"))
    verdict = "EQUAL"
    detail = ""
    if len(digests) < 2:
        verdict = "UNPROVEN"
        detail = f"only {len(digests)} digest list(s) to compare"
    else:
        reference = digests[0].read_text(encoding="utf-8").splitlines()
        for path in digests[1:]:
            other = path.read_text(encoding="utf-8").splitlines()
            if other == reference:
                continue
            verdict = "MOVED"
            for left, right in zip(reference, other):
                if left != right:
                    detail = (f"{path.stem} differs from {digests[0].stem} "
                              f"at {left.split()[-1]}: "
                              f"{left.split()[0][:16]} vs {right.split()[0][:16]}")
                    break
            else:
                detail = (f"{path.stem} wrote {len(other)} frames, "
                          f"{digests[0].stem} wrote {len(reference)}")
            break
    print(f"\nIDENTITY: {verdict}"
          + (f" -- {detail}" if detail else
             f" -- {len(digests)} arms, "
             f"{len(digests[0].read_text(encoding='utf-8').splitlines())} "
             "frames each, byte for byte"))
    if verdict == "MOVED":
        print("A frame that moves with the column chunk is a DEFECT, not a "
              "tolerance: the chunk is workspace shape, the kernels are "
              "one thread per column with no neighbour read, no shared "
              "memory, no atomic and no per-chunk seed.  Do not widen "
              "anything; find what in the walk is not column-local.")
        return 3
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--outdir")
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--pass-index", type=int, default=1)
    parser.add_argument("--warm", type=int, default=5)
    parser.add_argument("--cardwatch")
    parser.add_argument(
        "--own-prefix",
        help="path prefix of the engine install this sweep runs from "
             "(the venv).  A compute process whose name does not start "
             "with it is somebody else's job.  Without it the two "
             "occupancy columns read 'no-prefix' rather than a number "
             "that counts the arm's own forecast as a co-tenant.")
    parser.add_argument("--digests")
    parser.add_argument("--tsv")
    parser.add_argument("--work")
    args = parser.parse_args(argv)
    if args.arm:
        for required in ("outdir", "chunk", "digests", "tsv"):
            if getattr(args, required) is None:
                parser.error(f"--arm needs --{required.replace('_', '-')}")
        return arm(args)
    if args.work is None:
        parser.error("--collect needs --work")
    return collect(args)


if __name__ == "__main__":
    sys.exit(main())
