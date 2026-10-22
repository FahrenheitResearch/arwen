"""Relocate frozen public station observations without changing their values."""
import argparse
from pathlib import Path
import shutil

from gpuwm.tuning import digest, read_json, verify_packet, write_json


def stage(packet, source, output):
    round_ = verify_packet(packet)
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("observation destination already exists; preserve the frozen cohort")
    output.mkdir(parents=True)
    raw_source = source / "raw-utc-days.csv"
    raw_target = output / raw_source.name
    shutil.copy2(raw_source, raw_target)
    results = []
    for day in round_["train_days"] + round_["heldout_days"]:
        path = source / f"surface-{day}.json"
        record = read_json(path)
        if record["provenance"]["sha256"] != digest(raw_source):
            raise ValueError("frozen raw observations differ from the native decoder's provenance")
        record["provenance"]["uri"] = str(raw_target)
        target = output / path.name
        write_json(target, record)
        results.append({"day": day, "source_sha256": digest(path), "staged_sha256": digest(target)})
    write_json(output / "relocation-receipt.json", {"round_sha256": digest(Path(packet) / "round.json"),
        "raw_sha256": digest(raw_target), "only_change": "provenance URI points to the same raw observation bytes", "days": results})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "source", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    stage(args.packet, args.source, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
