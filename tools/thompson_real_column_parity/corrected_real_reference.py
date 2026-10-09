#!/usr/bin/env python3
"""Cut an independent corrected-RACG oracle on unchanged stock real columns.

Run builds and fixture generation only on the authorized CPU box. The stock
WRF source and existing fixture are read-only. The derived module changes only
the eight table indexes whose non-hail slab exceeds dimNRHG=1.  One build
(build_wrf.sh ... corrected-racg) links both drivers, so the same corrected
module answers mp=28 (``fixture``) and mp=8 (``fixture --mp 8``, from the
stock mp=8 companion fixture's columns).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from tools.thompson_mp8_column_oracle.corrected_mp28 import PINNED, OLD, NEW, corrected_source
sys.path.insert(0, str(HERE))

VARIANT = "wrf461-corrected-racg-eight-index-reads-v1"


class _DeletionReceipt(list):
    def __init__(self, journal):
        super().__init__()
        self.journal = journal

    def append(self, entry):
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
            stream.flush()
        super().append(entry)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_words(fixture):
    keys = sorted(name for name in fixture if name.startswith("col_"))
    keys += ["dt", "labels", "origin"]
    return {name: {"dtype": fixture[name].dtype.str,
                   "shape": list(fixture[name].shape),
                   "sha256": hashlib.sha256(fixture[name].tobytes()).hexdigest()}
            for name in keys}


def stage_source(source, output, receipt):
    source, output, receipt = map(Path, (source, output, receipt))
    if source.resolve() == output.resolve():
        raise ValueError("the pinned shared WRF source is read-only; use a private corrected source path")
    raw = source.read_bytes()
    repaired = corrected_source(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(repaired)
    record = {"schema": VARIANT, "stock_source_sha256": PINNED,
              "corrected_source_sha256": hashlib.sha256(repaired).hexdigest(),
              "old": OLD, "new": NEW, "replacement_count": 8,
              "only_replacements": repaired.decode().replace(NEW, OLD).encode() == raw}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


def cut(build, stock, output, receipt, mp=28):
    # Installing the existing CPU host backend belongs to fixture generation,
    # never metadata/source validation during test collection.
    import make_fixture as M
    build, stock, output, receipt = map(Path, (build, stock, output, receipt))
    if stock.resolve() == output.resolve():
        raise ValueError("preserve the stock fixture; write a separate corrected reference")
    source_record = json.loads((build / "RACG-SOURCE.json").read_text())
    if (source_record.get("schema") != VARIANT or source_record.get("replacement_count") != 8
            or source_record.get("only_replacements") is not True
            or source_record.get("stock_source_sha256") != PINNED):
        raise ValueError("the corrected oracle requires the exact eight-read source repair receipt")
    if _hash(build / "pristine/module_mp_thompson.F") != source_record["corrected_source_sha256"]:
        raise ValueError("the corrected module changed since its source receipt")
    build_receipt = (build / "BUILD-RECEIPT.txt").read_text()
    if "oracle_source_variant = corrected-racg" not in build_receipt:
        raise ValueError("refusing an unnamed stock or unproved oracle variant")
    stock_hash = _hash(stock)
    with np.load(stock, allow_pickle=False) as z:
        stock_arrays = {name: z[name].copy() for name in z.files}
    raw = {name: stock_arrays[f"col_{name}"] for name in M.RAW_KEYS}
    if raw["p"].shape[0] != 42:
        raise ValueError("this reference must retain all 42 original real columns")
    if int(stock_arrays.get("mp_physics", 28)) != mp:
        raise ValueError(f"the stock fixture is not an mp_physics={mp} fixture")
    dt = float(stock_arrays["dt"])
    # The existing helper runs both independent Fortran binaries on the same
    # stream and refuses unless instrumentation preserves every output byte.
    deleted = _DeletionReceipt(build / "RAW-DELETIONS.jsonl")
    arrays = M._answers(build, raw, dt, f"corrected-real-columns-mp{mp}", mp,
                        deleted=deleted)
    for name in ("labels", "origin", "dt", "columns_from"):
        if name in stock_arrays:
            arrays[name] = stock_arrays[name].copy()
    inputs = input_words(stock_arrays)
    if input_words(arrays) != inputs:
        raise ValueError("corrected oracle changed input words or column selection")
    arrays["reference_variant"] = np.array(VARIANT)
    arrays["stock_fixture_sha256"] = np.array(stock_hash)
    arrays["source_repair_receipt"] = np.array(json.dumps(source_record, sort_keys=True))
    arrays["input_words_receipt"] = np.array(json.dumps(inputs, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    if output.stat().st_size >= 1_000_000:
        raise ValueError("corrected real-column test fixture must remain below 1 MB")
    if _hash(stock) != stock_hash:
        raise ValueError("the original stock fixture changed")
    record = {"schema": VARIANT, "source": source_record,
              "stock_fixture_sha256": stock_hash, "corrected_fixture_sha256": _hash(output),
              "fixture_bytes": output.stat().st_size, "columns": 42,
              "mp_physics": mp,
              "input_words": inputs, "inputs_byte_identical": True,
              "instrumentation_output_fidelity": "passed by make_fixture._run_wrf",
              "answer_hashes": {name: hashlib.sha256(value.tobytes()).hexdigest()
                                for name, value in arrays.items()
                                if name.startswith(("wrf_", "rate_", "cp1_", "cp2_", "cpx_"))},
              "deleted": deleted}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    source = actions.add_parser("source")
    for name in ("stock_source", "corrected_source", "receipt"):
        source.add_argument(name, type=Path)
    fixture = actions.add_parser("fixture")
    for name in ("build", "stock_fixture", "output", "receipt"):
        fixture.add_argument(name, type=Path)
    fixture.add_argument("--mp", type=int, choices=(28, 8), default=28)
    args = parser.parse_args()
    record = (stage_source(args.stock_source, args.corrected_source, args.receipt)
              if args.action == "source" else cut(args.build, args.stock_fixture, args.output,
                                                  args.receipt, args.mp))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
