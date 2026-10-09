"""Summarize maintained donor-negative capture without retaining raw arrays."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare_words(actual, native):
    assert actual.dtype == native.dtype == np.float32 and actual.shape == native.shape
    return {"words": int(actual.size),
        "different_words": int(np.count_nonzero(actual.view(np.uint32) != native.view(np.uint32))),
        "control_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "native_sha256": hashlib.sha256(native.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("prefix",type=Path)
    parser.add_argument("positive_proof",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    positive=json.loads(args.positive_proof.read_text())
    assert positive["native_exact"] and positive["native_different_words"] == 0
    capture=json.loads(args.prefix.with_suffix(".json").read_text())
    assert capture["diagnostic"] == "Only the wrf_defor12 evaluation-point donor mapping from 83fde6032 is removed"
    native_path=args.root / "tests/data/wrf471_diff_opt1/wrf471.npz"
    sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(native_path) == capture["native_archive_sha256"] == positive["native_archive_sha256"]
    assert sha(args.prefix.with_suffix(".npz")) == capture["archive_sha256"]
    rows={}
    with np.load(native_path,allow_pickle=False) as native,np.load(args.prefix.with_suffix(".npz"),allow_pickle=False) as control:
        expected={row["name"]+"_"+field: row["name"]+"_expected_"+field
                  for row in capture["cases"] for field in row["fields"]}
        assert set(control.files) == set(expected)
        for name, reference in expected.items():
            rows[name]=compare_words(control[name],native[reference])
            assert rows[name]["native_sha256"] == positive["fields"][name]["current_sha256"]
    proof={"schema":"current-coordinate-donor-negative-control-v1",
        "positive_proof_sha256":sha(args.positive_proof),"capture_receipt_sha256":sha(args.prefix.with_suffix(".json")),
        "native_archive_sha256":sha(native_path),"control_archive_sha256":capture["archive_sha256"],
        "measurement_tool_sha256":sha(__file__),"capture_tool_sha256":sha(args.root/"tools/wrf_diffopt1_oracle/capture.py"),
        "capture_identity":positive["capture_identity"],"control_device":capture["device"],
        "controlled_module_source_sha256":capture["module_source_sha256"],
        "case_count":len(capture["cases"]),"field_count":len(rows),"words":sum(row["words"] for row in rows.values()),
        "changed_native_words":sum(row["different_words"] for row in rows.values()),
        "changed_fields":{name:row["different_words"] for name,row in rows.items() if row["different_words"]},
        "fields":rows,"intervention":capture["diagnostic"],"source_inputs":positive["source_inputs"]}
    args.output.write_text(json.dumps(proof,indent=2)+"\n")
    raw=args.prefix.with_suffix(".npz")
    removed={"path":str(raw),"bytes":raw.stat().st_size,"sha256":sha(raw)}
    raw.unlink()
    args.output.with_suffix(".deleted.json").write_text(json.dumps(removed,indent=2)+"\n")
    print(json.dumps({k:proof[k] for k in ("case_count","field_count","words","changed_native_words","changed_fields")}),flush=True)

if __name__ == "__main__": main()
