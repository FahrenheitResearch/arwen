"""Grade the twin and (with a card) the CUDA kernel against an oracle npz.

Usage: python check_fixture.py FIXTURE.npz RECEIPT.json [--device]

Writes a JSON receipt: per output, the word count and the number of
differing float32 words, for the NumPy twin and, with --device, the
``rrtmg_aer3_typed_sw_optics`` kernel in the SW engine layout, plus the
arithmetic mode (GPUWM_WRF_EXACT) the process ran under.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from gpuwm.core import rrtmg_aerosol_optics as ao

OUTS = ("tauaer", "ssaaer", "asyaer")
INPUTS = ("p", "t", "qv", "dz8w", "nwfa", "nifa")


def _differ(got, want):
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    assert got.shape == want.shape, (got.shape, want.shape)
    return {"words": int(got.size),
            "differ": int((got.view(np.uint32) != want.view(np.uint32)).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("fixture", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--device", action="store_true")
    args = parser.parse_args()
    data = np.load(args.fixture)
    o = {k: data[k] for k in data.files if k != "receipt"}
    ins = [o[f"in/{k}"] for k in INPUTS]
    aer_type = ao.aer_type_from_xland(o["in/xland"])
    result = {
        "fixture": str(args.fixture),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "oracle_receipt": json.loads(str(data["receipt"])),
        "columns": int(ins[0].shape[0]), "layers": int(ins[0].shape[1]),
        "water_columns": int((aer_type == 3).sum()),
        "type_matches_driver": bool(np.array_equal(aer_type, o["out/aer_type"])),
        "maritime_tables_sha256": ao.maritime_tables_sha256(),
        "rural_tables_sha256": ao.tables_sha256(),
    }
    tau, ssa, asy, taod = ao.aer3_sw_optics(*ins, aer_type=aer_type)
    twin = {name: _differ(got, o[f"out/{name}"])
            for name, got in zip(OUTS, (tau, ssa, asy))}
    twin["taod5503d"] = _differ(taod, o["out/taod5503d"])
    result["twin"] = twin
    if args.device:
        import cupy as cp
        dev = [cp.asarray(a) for a in ins]
        nz = ins[0].shape[1]
        t_dev = ao.aer_type_from_xland_device(cp.asarray(o["in/xland"]))
        ztaua, zasya, zomga, taod_d = ao.aer3_sw_optics_device(
            *dev, nz + 1, aer_type=t_dev)
        want = ao.sw_engine_layout(o["out/tauaer"], o["out/ssaaer"],
                                   o["out/asyaer"], nz + 1)
        result["device"] = {
            "ztaua": _differ(cp.asnumpy(ztaua), want[0]),
            "zasya": _differ(cp.asnumpy(zasya), want[1]),
            "zomga": _differ(cp.asnumpy(zomga), want[2]),
            "taod5503d": _differ(cp.asnumpy(taod_d), o["out/taod5503d"]),
        }
        land = cp.asarray(aer_type == 1)
        rural = ao.aer3_sw_optics_device(*dev, nz + 1)
        result["device_land_vs_rural_kernel"] = {
            name: _differ(cp.asnumpy(a[land]), cp.asnumpy(b[land]))
            for name, a, b in zip(("ztaua", "zasya", "zomga"),
                                  (ztaua, zasya, zomga), rural[:3])}
        props = cp.cuda.runtime.getDeviceProperties(0)
        result["gpu"] = props["name"].decode()
        result["nvrtc"] = ".".join(map(str, cp.cuda.nvrtc.getVersion()))
    result["GPUWM_WRF_EXACT"] = os.environ.get("GPUWM_WRF_EXACT", "")
    total = sum(v["differ"] for part in ("twin", "device",
                                         "device_land_vs_rural_kernel")
                for v in result.get(part, {}).values())
    result["verdict"] = "PASS" if total == 0 and result["type_matches_driver"] else "FAIL"
    args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("verdict", "columns", "layers",
                                             "water_columns")}))


if __name__ == "__main__":
    main()
