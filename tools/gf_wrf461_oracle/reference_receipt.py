"""Save independent WRF corrected outputs as compact hash-bound word deltas."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import gf_regime_columns as gr


def write(directory):
    base = directory / "gf-regime-wrf-crgamma-k0.bin"
    old = gr.read_outputs(base)
    provenance = json.loads((directory / "corrected/provenance.json").read_text())
    for arm in ("safe", "corrected"):
        path = directory / f"gf-regime-wrf-{arm}-k0.bin"
        new = gr.read_outputs(path)
        words = {}
        for key in ("lev", "sca", "isc"):
            a, b = old[key].view(np.uint32).ravel(), new[key].view(np.uint32).ravel()
            words[key] = [[int(i), int(b[i])] for i in np.flatnonzero(a != b)]
        receipt = dict(base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ncol=new["ncol"], nz=new["nz"], words=words, provenance=provenance)
        (directory / f"gf-regime-wrf-{arm}-delta.json").write_text(
            json.dumps(receipt, indent=2) + "\n")


def branch(original, corrected):
    from gpuwm.verify.gf_oracle import load_gf_oracle
    old, new = load_gf_oracle(original), load_gf_oracle(corrected)
    words = {}
    for key, fields, source in (("lev", gr.OUT_LEV, "levels"),
                                ("sca", gr.OUT_SCA, "surface"),
                                ("isc", gr.OUT_INT, "surface")):
        dtype = np.int32 if key == "isc" else np.float32
        a = np.stack([getattr(old, source)[f].astype(dtype) for f in fields], axis=1).view(np.uint32).ravel()
        b = np.stack([getattr(new, source)[f].astype(dtype) for f in fields], axis=1).view(np.uint32).ravel()
        words[key] = [[int(i), int(b[i])] for i in np.flatnonzero(a != b)]
    receipt = dict(base_sha256={name: hashlib.sha256((original/name).read_bytes()).hexdigest()
                    for name in ("gf-levels.csv", "gf-surface.csv")},
                   corrected_sha256={name: hashlib.sha256((corrected/name).read_bytes()).hexdigest()
                    for name in ("gf-levels.csv", "gf-surface.csv")}, words=words,
                   source_sha256=hashlib.sha256((corrected/'corrected.F').read_bytes()).hexdigest())
    (original/'gf-corrected-boundary-delta.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == "__main__":
    if len(sys.argv) == 3:
        branch(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        write(Path(sys.argv[1]))
