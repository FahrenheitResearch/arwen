"""Wrap the byte-extracted WRF wind routine without editing its body."""
from pathlib import Path
import argparse
import hashlib
import json
import re

SOURCE_SHA256 = "7662f29bb003697d08ccee585e272a3ed7362edc73bcef5ea8dc95d100d9e578"


def extract(source, output):
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Wind oracle needs the source-pinned WRF v4.7.1 driver")
    body = re.search(rb"(?mi)^subroutine interpolate_atm2fire\(.*?^end subroutine interpolate_atm2fire[^\r\n]*", raw, re.DOTALL)
    if body is None:
        raise ValueError("WRF interpolate_atm2fire routine is missing")
    routine = body.group()
    output.write_bytes(b"module module_sfire_wind_oracle\nuse module_fr_fire_util\nuse module_model_constants, only: g\nimplicit none\ncontains\n" + routine + b"\nend module module_sfire_wind_oracle\n")
    receipt = dict(source_file=source.name, source_sha256=hashlib.sha256(raw).hexdigest(),
                   routine_sha256=hashlib.sha256(routine).hexdigest(),
                   wrapper_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   transformation="byte-extracted original body, module services supplied by compiled WRF")
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(extract(args.source, args.output)))
