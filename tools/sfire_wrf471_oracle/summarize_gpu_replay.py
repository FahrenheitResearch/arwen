"""Create a portable word-comparison receipt from the focused replay log."""
from pathlib import Path
import argparse
import hashlib
import json
import re


def summarize(log, source):
    log, source = Path(log), Path(source)
    rows = {}
    for line in log.read_text(errors="replace").splitlines():
        if '{"' not in line:
            continue
        try:
            record, _ = json.JSONDecoder().raw_decode(line[line.index('{"'):])
        except ValueError:
            continue
        if isinstance(record, dict):
            for key, value in record.items():
                if isinstance(value, dict) and "words" in value and "max_ulp" in value:
                    rows[key] = value
    text = log.read_text(errors="replace")
    summary = re.findall(r"=+ (.+? (?:passed|failed).*?) =+", text)
    fingerprint = {}
    for pattern in ("gpuwm/core/sfire*.py", "gpuwm/core/kernels/sfire*", "tests/test_sfire*wrf471_parity.py"):
        for file in sorted(source.glob(pattern)):
            fingerprint[file.relative_to(source).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    return {"test_summary": summary[-1] if summary else "No successful pytest summary",
            "gpu": next((line for line in text.splitlines() if line.startswith("NVIDIA ")), "unrecorded"),
            "runtime": next((line for line in text.splitlines() if line.startswith("CuPy ")), "unrecorded"),
            "word_table": rows, "graded_word_count": sum(row["words"] for row in rows.values()),
            "different_word_count": sum(row["different_words"] for row in rows.values()),
            "max_ulp": max((row["max_ulp"] for row in rows.values()), default=0),
            "source_sha256": fingerprint, "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            "scope": "Printed word table covers routine gates; atmospheric and correction gates are included in pytest summary."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    result = summarize(args.log, args.source)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("test_summary", "graded_word_count", "different_word_count", "max_ulp")}))
