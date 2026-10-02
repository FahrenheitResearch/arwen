"""Reproduce the vertical damper's half-pi rounding against compiled WRF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpuwm.core.kernels import module_source
from gpuwm.verify.smallstep_vertical_oracle import measure_vertical_case, vertical_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, raw, metadata = next(c for c in vertical_cases() if c[0] == "implicit_damper")
    runs = {}
    arrays = {}
    for label, causal in (("native", False), ("ordered", True)):
        for corrected in (False, True):
            outputs = {}
            options = dict(no_fma=causal, wrf_phi_order=causal, wrf_top_order=causal,
                           wrf_update_order=causal, preserve_subnormals=causal,
                           correct_pi=corrected, replay_pi_defect=not corrected)
            case_meta = dict(metadata, theta_offset=0.0 if causal else 300.0)
            key = label + ("_corrected" if corrected else "_old")
            runs[key] = measure_vertical_case(raw, case_meta, args.library,
                                              output_arrays=outputs, **options)
            arrays[key] = outputs
    evidence = {}
    for label in ("native", "ordered"):
        old = arrays[label + "_old"]["w_native"]
        new = arrays[label + "_corrected"]["w_native"]
        ref = arrays[label + "_corrected"]["w_wrf"]
        changed = old.view(np.uint32) != new.view(np.uint32)
        repaired = changed & (new.view(np.uint32) == ref.view(np.uint32))
        entries = []
        for k, j, i in np.argwhere(repaired):
            entries.append(dict(index=[int(k), int(j), int(i)],
                                old_word=int(old[k,j,i].view(np.uint32)),
                                corrected_word=int(new[k,j,i].view(np.uint32)),
                                fortran_word=int(ref[k,j,i].view(np.uint32))))
        evidence[label] = dict(changed_words=int(changed.sum()), repaired_words=int(repaired.sum()),
                               repaired=entries, old=runs[label + "_old"]["advance_w"]["w"],
                               corrected=runs[label + "_corrected"]["advance_w"]["w"])
    original_source = module_source("acoustic").replace("1.5707963267948966f", "1.5707963f")
    result = dict(source_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
                  old_half_pi_word=int(np.float32(1.5707963).view(np.uint32)),
                  corrected_half_pi_word=int(np.float32(np.pi / 2).view(np.uint32)), evidence=evidence)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
