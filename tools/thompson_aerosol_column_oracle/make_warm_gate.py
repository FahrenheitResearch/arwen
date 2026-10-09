#!/usr/bin/env python3
"""Cut the committed mp=28 warm-network gate out of a finished oracle run.

The gate (``tests/data/mp28_warm_network_oracle.npz``, read by
``tests/test_thompson_aerosol_warm_network_oracle_gpu.py``) is the subset of
an oracle column set whose WARM-NETWORK outputs -- vapour, theta, cloud
water and droplet number, rain water and rain number, the two aerosol
numbers and surface rain -- WOOF's strict build (``GPUWM_WRF_EXACT=1``)
gives bit for bit equal to unmodified WRF v4.6.1 at a time step, together
with WRF's own answers there.  Only columns NEW to bit identity relative to
a baseline run are kept, so every committed column is one a warm-network
transcription defect used to move.

The WRF answers are the ``summary-strict-dtN.wrf.npz`` files ``compare.py``
writes (WRF run on the Exner function and layer depths the GPU adapter
formed, which the gate's GPU run forms again identically).  A baseline run
that kept only its GPU outputs is graded against the run's own WRF answers,
column by column under the same label, after its Exner function is checked
to be the same words (the WRF inputs are then identical).

usage: make_warm_gate.py RUN_DIR BASELINE_RUN_DIR OUT.npz [DT ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_io as io  # noqa: E402

f32 = np.float32

#: The outputs the warm network decides.
WARM_FIELDS = ("qv", "th", "qc", "nc", "qr", "nr", "nwfa", "nifa", "rainnc")
#: Column inputs gpu_run.py reads.
INPUTS = ("p", "th", "geop", "w", *io.SPECIES, "nwfa2d", "nifa2d")


def identical(run, dt, reference=None):
    """Per column of ``reference`` (default ``run``), matched by label: every
    warm field of ``run``'s strict GPU output bit-identical to WRF.  WRF's
    answers are ``run``'s own when it kept them, else ``reference``'s, after
    the two runs' Exner functions are checked to be the same words.  A
    column ``run`` lacks reads False."""
    run = Path(run)
    reference = run if reference is None else Path(reference)
    pick, present = _match(run, reference)
    rows = pick[present]
    gpu = np.load(run / f"gpu-strict-dt{dt}.npz")
    own = run / f"summary-strict-dt{dt}.wrf.npz"
    if own.exists():
        wrf, wrf_rows = np.load(own), rows
    else:
        wrf = np.load(reference / f"summary-strict-dt{dt}.wrf.npz")
        wrf_rows = np.flatnonzero(present)
        ref_gpu = np.load(reference / f"gpu-strict-dt{dt}.npz")
        a = np.asarray(gpu["in_pii"], f32)[rows].view(np.int32)
        b = np.asarray(ref_gpu["in_pii"], f32)[wrf_rows].view(np.int32)
        if not (a == b).all():
            raise SystemExit(f"{run}: a different Exner function; its GPU "
                             "outputs cannot be graded with these answers")
    same_all = np.ones(rows.shape, bool)
    for name in WARM_FIELDS:
        a = np.asarray(gpu[name], f32)[rows].view(np.int32)
        b = np.asarray(wrf[name], f32)[wrf_rows].view(np.int32)
        same = a == b
        same_all &= same if same.ndim == 1 else same.all(axis=1)
    ok = np.zeros(pick.shape, bool)
    ok[present] = same_all
    return ok


def _match(run, reference):
    """Indices into ``run``'s columns for each of ``reference``'s labels."""
    have = [str(x) for x in io.load(Path(run) / "columns.npz")["labels"]]
    want = [str(x) for x in io.load(Path(reference) / "columns.npz")["labels"]]
    where = {label: i for i, label in enumerate(have)}
    pick = np.asarray([where.get(label, -1) for label in want])
    return pick, pick >= 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    run, base, out = argv[:3]
    dts = [int(x) for x in argv[3:]] or [20, 5]
    cols = io.load(Path(run) / "columns.npz")
    keep = {}
    for dt in dts:
        keep[dt] = identical(run, dt) & ~identical(base, dt, reference=run)
    union = np.zeros_like(next(iter(keep.values())))
    for mask in keep.values():
        union |= mask
    index = np.flatnonzero(union)
    payload = {k: np.asarray(cols[k])[index] for k in INPUTS}
    payload["labels"] = np.asarray(cols["labels"])[index]
    payload["source_index"] = index.astype(np.int32)
    payload["dts"] = np.asarray(dts, np.int32)
    for dt in dts:
        wrf = np.load(Path(run) / f"summary-strict-dt{dt}.wrf.npz")
        gpu = np.load(Path(run) / f"gpu-strict-dt{dt}.npz")
        payload[f"check_dt{dt}"] = keep[dt][index]
        payload[f"in_pii_dt{dt}"] = np.asarray(gpu["in_pii"], f32)[index]
        for name in WARM_FIELDS:
            payload[f"wrf_{name}_dt{dt}"] = np.asarray(wrf[name], f32)[index]
        print(f"dt={dt}: {int(keep[dt].sum())} columns newly bit-identical "
              "in every warm-network field")
    np.savez_compressed(out, **payload)
    print(f"{out}: {len(index)} columns")


if __name__ == "__main__":
    main()
