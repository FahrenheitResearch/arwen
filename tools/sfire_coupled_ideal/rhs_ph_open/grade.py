"""Grade the actual open geopotential kernel against unmodified compiled rhs_ph."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json

import cupy as cp
import numpy as np

from gpuwm.core.dycore import _launch_open_geopotential
from gpuwm.wrf_exact import ENABLED
from tools.sfire_wrf471_oracle.fixture import words


def grade(corpus, destination):
    corpus = Path(corpus)
    pinned = json.loads((corpus / "receipt.json").read_text())
    cases = {}
    natives = {}
    for name, record in pinned["cases"].items():
        path = corpus / (name + ".npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("open rhs_ph fixture differs from its native receipt")
        with np.load(path, allow_pickle=False) as archive:
            f = dict(archive)
        nz, ny, nx = f["u"].shape[0], f["mut"].shape[0], f["mut"].shape[1]
        arrays = {key: cp.asarray(f[key]) for key in (
            "u", "v", "w", "phb", "rdnw", "fnm", "fnp", "c1f", "c2f", "msft", "msfu", "msfv")}
        state = SimpleNamespace(**arrays, p=cp.empty((nz, ny, nx), cp.float32),
            php=cp.asarray(f["ph"]), rph_t=cp.asarray(f["seed"]),
            mup=cp.zeros((ny, nx), cp.float32), mub2d=cp.asarray(f["mut"]),
            cfn=np.float32(f["cfn"]), cfn1=np.float32(f["cfn1"]), has_msf=True)
        mask = int(f["mask"])
        cfg = SimpleNamespace(dx=90., dy=120., open_x=bool(mask & 1), open_y=bool(mask & 2),
                              h_sca_adv_order=int(f["order"]))
        _launch_open_geopotential(state, cfg, cp.asarray(f["ww"]),
            add_vertical=bool(f["vertical"]), mux=cp.asarray(f["muu"]), muy=cp.asarray(f["muv"]))
        actual = cp.asnumpy(state.rph_t)
        cases[name] = words(actual, f["native"])
        number = int(name.split("case_")[1].split("_")[0])
        control = int(name.rsplit("_", 1)[1])
        natives[number, control] = f["native"]
    # Control 1 holds WRF's zero top U level, control 2 a sentinel there. With
    # WRF's zero FNM/FNP at kde, no native word may depend on that level.
    top_ghost = {str(case): words(natives[case, 1], natives[case, 2]) for case in range(1, 17)}
    summary = dict(cases=len(cases), words=sum(v["words"] for v in cases.values()),
                   different_words=sum(v["different_words"] for v in cases.values()),
                   max_ulp=max(v["max_ulp"] for v in cases.values()),
                   nonfinite_differences=sum(v["nonfinite_differences"] for v in cases.values()),
                   top_ghost_dependent_native_words=sum(v["different_words"] for v in top_ghost.values()))
    receipt = dict(reference="compiled unmodified WRF v4.7.1 rhs_ph, no allowed-difference mask",
                   strict=ENABLED, cupy=cp.__version__, device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                   corpus_sha256=hashlib.sha256((corpus / "receipt.json").read_bytes()).hexdigest(),
                   summary=summary, cases=cases, top_ghost_controls=top_ghost,
                   source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                  for name in ("gpuwm/core/dycore.py", "gpuwm/core/kernels/dycore.cu", "gpuwm/core/kernels/__init__.py")})
    Path(destination).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary))
    if summary["different_words"]:
        raise SystemExit("open geopotential kernel differs from the unmodified native routine")
    if summary["top_ghost_dependent_native_words"]:
        raise SystemExit("a native word depends on the top U level that WRF never assigns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("destination")
    args = parser.parse_args()
    grade(args.corpus, args.destination)
