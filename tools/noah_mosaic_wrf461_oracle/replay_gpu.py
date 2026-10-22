"""Replay every Noah mosaic oracle family on the GPU against one fixture root.

Usage (on a CUDA host, from a checkout whose ``tests`` folder is importable)::

    python tools/noah_mosaic_wrf461_oracle/replay_gpu.py FIXTURE_ROOT OUT.json

``FIXTURE_ROOT`` holds the family folders a build writes under
``BUILD_DIR/fixtures/mosaic`` and ``BUILD_DIR/fixtures/mosaic_ucm`` (raw
case directories) or the committed packed corpus.  Both layouts are read by
:func:`gpuwm.verify.noah_mosaic_oracle.load`.  Run it once with
``GPUWM_WRF_EXACT=1`` (strict build) and once without (default arithmetic).

Unlike the committed gates, nothing is held out here: the column/field
pairs that ``tests/test_noah_mosaic_wrf471_parity.py`` pins by hash are
graded like every other word, so the strict-build number is the plain
0-ULP claim over every output word and the default number shows what FTZ
moves.  Every differing pair is classified (NaN payload only, NaN against
a number, or finite with its ULP and absolute distance).  The D1 grid-soil and D2 accumulator expectations are the gates'
corrected-reduction contract (WRF's defective index and unweighted sums are
documented WRF bugs, not copied); every other expected word is the raw WRF
output.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def _family_dirs(root: Path) -> dict[str, Path]:
    out = {}
    for sub in (root, root / "mosaic", root / "mosaic_ucm"):
        if not sub.is_dir():
            continue
        for child in sorted(sub.iterdir()):
            if child.is_dir() and any(child.iterdir()):
                out.setdefault(child.name, child)
    return out


def _classify(record):
    """One differing column/field pair: how its differing words differ.

    ``nan_payload``: both words NaN, only the bit pattern differs (CUDA
    arithmetic returns the canonical 0x7fffffff, x86 its own payload);
    ``nan_vs_finite``: one side NaN; ``finite``: both finite, with the ULP
    distance and absolute difference.
    """
    wrf = np.array([int(w, 16) for w in record["wrf_words"]], dtype=np.uint32)
    gpu = np.array([int(w, 16) for w in record["gpu_words"]], dtype=np.uint32)
    differ = wrf != gpu
    wf, gf = wrf.view(np.float32), gpu.view(np.float32)
    nan_w, nan_g = np.isnan(wf), np.isnan(gf)
    both = differ & nan_w & nan_g
    one = differ & (nan_w ^ nan_g)
    fin = differ & ~nan_w & ~nan_g
    out = dict(family=record["family"], fixture=record["fixture"], column=record["column"],
               field=record["field"], gpu_sha256=record["gpu_sha256"],
               differing_words=int(differ.sum()),
               nan_payload=int(both.sum()), nan_vs_finite=int(one.sum()), finite=int(fin.sum()),
               wrf_words=[w for w, d in zip(record["wrf_words"], differ) if d][:4],
               gpu_words=[w for w, d in zip(record["gpu_words"], differ) if d][:4])
    if fin.any():
        a, b = wf[fin].astype(np.float64), gf[fin].astype(np.float64)
        ia = wrf[fin].view(np.int32).astype(np.int64)
        ib = gpu[fin].view(np.int32).astype(np.int64)
        ka = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
        kb = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
        out["finite_max_ulp"] = int(np.abs(ka - kb).max())
        out["finite_max_abs"] = float(np.abs(a - b).max())
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root, out = Path(argv[0]), Path(argv[1])
    # The gate modules do ``from conftest import requires_gpu``; with the
    # repo root ahead of tests/ on PYTHONPATH that name resolves to the root
    # conftest.py, which has no such marker.  Put tests/ first.
    tests_dir = Path(__file__).resolve().parents[2] / "tests"
    sys.path.insert(0, str(tests_dir))
    sys.modules.pop("conftest", None)
    import cupy as cp
    import test_noah_mosaic_wrf471_parity as plain
    import test_noah_mosaic_ucm_wrf471_parity as ucm
    families = _family_dirs(root)
    report = {"fixture_root": str(root),
              "strict": os.environ.get("GPUWM_WRF_EXACT", "0") == "1",
              "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
              "compute_capability": cp.cuda.Device(0).compute_capability,
              "families": {}}

    plain_families = [f for f in ("base", "cats", "lcz", "twins", "ftz", "terrain")
                      if f in families]
    for family in plain_families:
        # replay_all reads its module ROOT; point it at this family's parent.
        plain.ROOT = families[family].parent
        # pinned={}: hold nothing out, every column/field pair is graded.
        result = plain.replay_all(families=(family,), pinned={})
        fields = {n: dict(v) for n, v in result["table"].items()}
        words = sum(v["n"] for v in fields.values())
        report["families"][family] = dict(
            fields=fields, words=words,
            differing_pairs=[_classify(r) for r in result["held_records"]],
            max_ulp=max(v["max_ulp"] for v in fields.values()),
            n_nonzero=sum(v["n_nonzero"] for v in fields.values()),
            first_differences=result["first_differences"][:20])

    ucm_sets = (("ucm", "ucm_lcz"), ("ucm_table", "ucm_lcz_table"))
    for pair in ucm_sets:
        pair = tuple(f for f in pair if f in families)
        if not pair:
            continue
        result = ucm.replay(root=families[pair[0]].parent, families=pair)
        for family in pair:
            fields = {k.split("/", 1)[1]: v for k, v in result["table"].items()
                      if k.split("/", 1)[0] == family}
            diffs = [d for d in result["differences"] if d["family"] == family]
            report["families"][family] = dict(
                fields=fields, words=sum(v["n"] for v in fields.values()),
                max_ulp=max(v["max_ulp"] for v in fields.values()),
                n_nonzero=sum(v["n_nonzero"] for v in fields.values()),
                differing_fields=sorted({(d["step"], d["field"]) for d in diffs}),
                first_differences=diffs[:20])
    wrfinit = tuple(f for f in ("ucm_wrfinit", "ucm_lcz_wrfinit") if f in families)
    if wrfinit:
        result = ucm.replay(root=families[wrfinit[0]].parent, families=wrfinit,
                            production_frc=True)
        for family in wrfinit:
            fields = {k.split("/", 1)[1]: v for k, v in result["table"].items()
                      if k.split("/", 1)[0] == family}
            diffs = [d for d in result["differences"] if d["family"] == family]
            report["families"][family] = dict(
                fields=fields, words=sum(v["n"] for v in fields.values()),
                max_ulp=max(v["max_ulp"] for v in fields.values()),
                n_nonzero=sum(v["n_nonzero"] for v in fields.values()),
                differing_fields=sorted({(d["step"], d["field"]) for d in diffs}),
                first_differences=diffs[:20])

    fam = report["families"].values()
    report["total_words"] = sum(f["words"] for f in fam)
    report["total_nonzero"] = sum(f["n_nonzero"] for f in fam)
    report["max_ulp"] = max(f["max_ulp"] for f in fam)
    out.write_text(json.dumps(report, indent=1, default=lambda o: o.tolist()
                              if isinstance(o, np.ndarray) else str(o)) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: (v["words"], v["n_nonzero"], v["max_ulp"])
                      for k, v in report["families"].items()}, indent=1))
    print("TOTAL words", report["total_words"], "nonzero", report["total_nonzero"],
          "max_ulp", report["max_ulp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
