"""Retire scored, harness-created histories while keeping one comparison frame."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from gpuwm.tuning import verify_packet, read_json, write_json, digest
from gpuwm.tuning_collect import FRAME, verify_collected_scores


def retire_day(packet, runs, collection, day):
    if os.name == "nt":
        raise ValueError("simulation histories retire on the assigned remote box")
    round_ = verify_packet(packet)
    collection = Path(collection).resolve()
    verify_collected_scores(collection, round_sha256=digest(Path(packet) / "round.json"))
    receipt = read_json(collection / "collection-receipt.json")
    if day not in receipt["days"]:
        raise ValueError("the collection does not contain this completed day")
    runs = Path(runs).resolve()
    own_day = runs / day
    if not own_day.resolve().is_relative_to(runs) or own_day.is_symlink():
        raise ValueError("the run day must stay inside its owned output directory")
    manifest = own_day / "retention.json"
    if manifest.exists():
        raise ValueError("this day already has a retirement manifest")
    deleted, kept = [], []
    for member in round_["members"]:
        directory = own_day / member["id"]
        run_receipt = read_json(directory / "tuning-run.json")
        if run_receipt["status"] != "succeeded" or run_receipt["job_sha256"] != digest(directory / "job.json"):
            raise ValueError("only successfully scored harness-created run histories can be retired")
        run_root = (directory / "run").resolve()
        if not run_root.is_relative_to(own_day):
            raise ValueError("a run directory escapes the owned day")
        for path in sorted(run_root.rglob("wrfout_*")):
            if path.is_symlink() or not path.resolve().is_relative_to(run_root):
                raise ValueError("history links cannot lead deletion outside the run")
            match = FRAME.fullmatch(path.name)
            if not match or not path.is_file():
                continue
            record = {"path": str(path.relative_to(runs)), "bytes": path.stat().st_size}
            if match.groups()[1:] == (day, "22", "00", "00"):
                kept.append({**record, "sha256": digest(path)})
            else:
                deleted.append(record)
    expected = len(round_["members"]) * len(round_["domain_inventory"])
    if len(kept) != expected:
        raise ValueError("all member/domain map frames must survive before other histories retire")
    # Record the exact deletion list before touching a file. The native point
    # values and scores have already been bound by the collection receipt.
    write_json(manifest, {"status": "retiring", "collection_sha256": digest(collection / "collection-receipt.json"),
        "deleted": deleted, "kept_map_frames": kept, "keep_reason": "same 22 UTC best/current Rust comparison after winner freezes"})
    for record in deleted:
        (runs / record["path"]).unlink()
    for member in round_["members"]:
        (own_day / member["id"] / "run" / ".keep").write_text(
            "Only retained 22 UTC frames await the frozen best/current Rust maps. Remove after rendering.\n", encoding="ascii")
    value = read_json(manifest)
    value["status"] = "complete"
    write_json(manifest, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "runs", "collection", "day"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    retire_day(args.packet, args.runs, args.collection, args.day)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
