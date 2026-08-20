"""Run the REAL engine exe against a golden and diff the identities.

This is the hand-runnable half of the crate goldens: it builds the input
list from the staging tree exactly as the extractor did, runs
`gpuwm_mapped_engine inspect`, and compares the identities the parity
contract cares about — per-field array sha256, shape, axes, missing
count, the grid fingerprint, the frame times — against the golden the
Python engine produced.

Verify against the artifact: this runs the built exe, not a library
binding of it, and prints the exact command it ran.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[5]

sys.path.insert(0, str(HERE))
from extract import CASES, _inputs  # noqa: E402


def _engine() -> pathlib.Path:
    for profile in ("release", "debug"):
        candidate = (
            REPOSITORY / "tools" / "rw_wps" / "target" / profile / "gpuwm_mapped_engine.exe"
        )
        if candidate.is_file():
            return candidate
        candidate = candidate.with_suffix("")
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "no built engine: run `cargo build --release -p mapped-engine` in tools/rw_wps"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    arguments = parser.parse_args(argv)

    golden = json.loads((HERE / f"{arguments.case}.json").read_text(encoding="utf-8"))
    case = CASES[arguments.case]
    files = _inputs(case)
    if [path.name for path in files] != golden["input_names"]:
        raise SystemExit(
            "the staging tree no longer holds the inputs this golden was measured "
            "on; re-run extract.py to re-measure before comparing"
        )

    with tempfile.TemporaryDirectory(prefix="gpuwm-mapped-engine-golden-") as scratch:
        listing = pathlib.Path(scratch) / "inputs.txt"
        listing.write_text(
            "\n".join(str(path) for path in files) + "\n", encoding="utf-8"
        )
        command = [
            str(_engine()),
            "inspect",
            "--mapping",
            str(REPOSITORY / golden["mapping"]),
            "--input-list",
            str(listing),
        ]
        print("$ " + " ".join(command))
        completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        print(completed.stderr.strip().splitlines()[-1] if completed.stderr else "")
        raise SystemExit(f"engine refused with exit {completed.returncode}")
    document = json.loads(completed.stdout.strip().splitlines()[-1])

    differences: list[str] = []

    def compare(label: str, expected: object, observed: object) -> None:
        if expected != observed:
            differences.append(f"{label}: python={expected!r} rust={observed!r}")

    compare("mapping_sha256", golden["mapping_sha256"], document["mapping"]["sha256"])
    compare("source_format", golden["source_format"], document["source_format"])
    for key in ("nx", "ny", "vertical_count", "fingerprint"):
        compare(f"grid.{key}", golden["grid"][key], document["grid"][key])
    compare("status", golden["status"], document["status"])
    compare("frame_count", len(golden["frames"]), len(document["frames"]))

    for index, (expected_frame, observed_frame) in enumerate(
        zip(golden["frames"], document["frames"])
    ):
        for key in (
            "valid_time",
            "source_cycle",
            "member",
            "decoded_direct_fields",
            "unresolved_direct_fields",
        ):
            compare(f"frames[{index}].{key}", expected_frame[key], observed_frame[key])
        for name, expected_field in expected_frame["fields"].items():
            observed_field = observed_frame["fields"].get(name)
            if observed_field is None:
                differences.append(f"frames[{index}].fields.{name}: absent in rust")
                continue
            for key in ("axes", "shape", "missing", "sha256"):
                compare(
                    f"frames[{index}].fields.{name}.{key}",
                    expected_field[key],
                    observed_field[key],
                )
            compare(
                f"frames[{index}].fields.{name}.source_reference_count",
                expected_field["source_reference_count"],
                len(observed_field["source_references"]),
            )
    expected_materialization = golden["materialization"]
    compare(
        "materialization.verdict",
        expected_materialization["verdict"],
        document["materialization"]["verdict"],
    )
    if expected_materialization["verdict"] == "PASS":
        compare(
            "materialization.frame_count",
            expected_materialization["frame_count"],
            document["materialization"]["frame_count"],
        )
        # The PORTABLE digest, not the raw one: the raw header quotes each
        # field's absolute input paths and every derived field's array
        # digest, and both of those are identities of the box rather than
        # of the decode (a derived field's last bits come from the box's
        # libm).  Comparing the raw one made this script fail on any
        # machine but the recording one, with every decoded number right.
        compare(
            "materialization.portable_rule",
            expected_materialization.get("portable_rule"),
            document["materialization"].get("portable_rule"),
        )
        compare(
            "materialization.frame_header_sha256_portable",
            expected_materialization["frame_header_sha256_portable"],
            document["materialization"]["frame_header_sha256_portable"],
        )

    if differences:
        print(f"FAIL: {len(differences)} difference(s)")
        for line in differences[:40]:
            print("  " + line)
        return 1
    fields = sum(len(frame["fields"]) for frame in golden["frames"])
    print(
        f"PASS: {len(files)} inputs, {len(golden['frames'])} frames, {fields} field "
        "identities byte-identical to the Python engine"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
