"""Byte-extract the unmodified WRF rhs_ph into a compilable wrapper module."""
from pathlib import Path
import argparse
import hashlib
import json
import re


def extract(prep, destination):
    raw = Path(prep).read_bytes()
    match = re.search(rb"(?im)^SUBROUTINE rhs_ph\(.*?^  END SUBROUTINE rhs_ph[^\r\n]*", raw, re.DOTALL)
    if match is None:
        raise ValueError("pinned prep extraction does not contain the original rhs_ph")
    native = match.group()
    destination = Path(destination)
    header = (b"module rhs_open_native\nuse module_model_constants\n"
              b"use module_configure, only: grid_config_rec_type\nimplicit none\ncontains\n")
    module = header + native + b"\nend module\n"
    (destination / "rhs_native.F90").write_bytes(module)
    receipt = dict(native_routine_sha256=hashlib.sha256(native).hexdigest(),
                   native_wrapper_sha256=hashlib.sha256(module).hexdigest(),
                   modification="none: the routine body is the pinned WRF v4.7.1 bytes")
    (destination / "rhs-extraction.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prep")
    parser.add_argument("destination")
    args = parser.parse_args()
    extract(args.prep, args.destination)
