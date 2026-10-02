#!/usr/bin/env python3
"""Replay all serialized inputs against native Fortran compiler controls.

This checks the complete binary outputs without an acceptance tolerance.
The signalling-NaN build must preserve every reference word. The optimized
build is measured and reported, while the -O0 binary remains the oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import struct


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare(reference: bytes, other: bytes, indices=None) -> dict:
    if len(reference) != len(other) or len(reference) % 4:
        raise ValueError("Fortran controls wrote inconsistent output sizes")
    ref_words = memoryview(reference).cast("I")
    other_words = memoryview(other).cast("I")
    selected = range(len(ref_words)) if indices is None else indices
    count = 0
    first = None
    for i in selected:
        count += 1
        if ref_words[i] != other_words[i]:
            if first is None:
                first = i
    selected = range(len(ref_words)) if indices is None else indices
    different = sum(ref_words[i] != other_words[i] for i in selected)
    return {
        "words": count,
        "different_words": different,
        "first_different_word": first,
        "reference_sha256": digest(reference),
        "control_sha256": digest(other),
    }


def defined_diagnostic_indices(data: bytes) -> dict:
    header = struct.unpack_from("<27i", data)
    mode = header[1]
    flags, tenddec = header[25:27]
    if mode < 5 or not tenddec:
        return {}
    ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte = header[2:20]
    nxi,nki,nji = ime-ims+1,kme-kms+1,jme-jms+1
    size = nxi*nki*nji
    xstart,xend = its,min(ite,ide-1)
    degrade_xs = not (flags & (1 << 0) or flags & (1 << 8) or its > ids+3)
    degrade_xe = not (flags & (1 << 0) or flags & (1 << 9) or ite < ide-4)
    if degrade_xs:
        xstart = max(its,ids+1)
    if degrade_xe:
        xend = min(ite,ide-2)
    # Native h starts with the x store. When that store is skipped the later
    # y accumulation reads an undefined INTENT(OUT) value, so those rows are
    # represented solely by the complete storage-canary comparison above.
    ht = [size + ((j-jms)*nki+(k-kms))*nxi+(i-ims)
          for j in range(jts,min(jte,jde-1)+1)
          for k in range(kts,min(kte,kde-1)+1)
          for i in range(xstart,xend+1)]
    zt = [2*size + ((j-jms)*nki+(k-kms))*nxi+(i-ims)
          for j in range(jts,min(jte,jde-1)+1)
          for k in range(kts,min(kte,kde-1)+1)
          for i in range(its,min(ite,ide-1)+1)]
    return {"h_tendency_defined": ht, "z_tendency_defined": zt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = sorted(args.inputs.glob("*.in.bin"))
    if not cases:
        parser.error("no *.in.bin inputs found")
    result = {
        "schema_version": 1,
        "reference": "WRF v4.7.1, -O0 -ffp-contract=off -fcheck=all",
        "controls": {
            "o2": "-O2 -ftree-vectorize -funroll-loops -ffp-contract=off -fcheck=all",
            "snan": "reference flags plus -finit-real=snan -finit-integer=-999999 -finit-derived",
        },
        "cases": {},
    }
    for source in cases:
        input_data = source.read_bytes()
        masks = defined_diagnostic_indices(input_data)
        outputs = {}
        for variant in ("reference", "o2", "snan"):
            executable = args.build / ("run_advect" if variant == "reference" else f"{variant}/run_advect")
            target = args.output / f"{source.name[:-7]}-{variant}.out.bin"
            subprocess.run(["nice", "-n", "10", str(executable.resolve()), str(source.resolve()),
                            str(target.resolve())], check=True)
            outputs[variant] = target.read_bytes()
        row = {"input_sha256": digest(input_data),
               **{variant: compare(outputs["reference"], outputs[variant]) for variant in ("o2", "snan")}}
        for variant in ("o2", "snan"):
            row[variant]["defined_diagnostics"] = {
                name: compare(outputs["reference"], outputs[variant], indices)
                for name, indices in masks.items()}
        original = source.with_name(source.name[:-7] + ".out.bin")
        if original.exists():
            row["reference_replay"] = compare(original.read_bytes(), outputs["reference"])
            if row["reference_replay"]["different_words"]:
                raise AssertionError("reference replay changed the existing fixture words")
        result["cases"][source.name] = row
        print(f"{source.name}: O2 {row['o2']['different_words']} words; snan {row['snan']['different_words']} words", flush=True)
    result["input_count"] = len(cases)
    result["o2_different_words"] = sum(row["o2"]["different_words"] for row in result["cases"].values())
    result["snan_different_words"] = sum(row["snan"]["different_words"] for row in result["cases"].values())
    (args.output / "compiler-controls.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    if result["snan_different_words"]:
        raise AssertionError("WRF signalling-NaN initialization changed reference output words")


if __name__ == "__main__":
    main()
