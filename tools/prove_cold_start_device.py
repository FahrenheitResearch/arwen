"""Replay captured CPU initialization and grade closure arrays and JSON receipts."""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np

from gpuwm.ingest import real
from gpuwm.ingest.closure_device import (
    thompson_cold_start_moment_closure, make_temperature_provider)


def main():
    original = real._thompson_cold_start_moment_closure
    rows = []

    def grade(state, xp, cfg, alt, **kw):
        fields = ("qc", "qr", "qi", "nr", "ni") + (("nc",) if cfg.mp_physics == 28 else ())
        candidate = SimpleNamespace(**{k: cp.asarray(getattr(state, k)) for k in fields})
        dumped = {}
        seed_masks = {}
        if os.environ.get("CLOSURE_DUMP"):
            for name, q, n in (("rain", "qr", "nr"), ("ice", "qi", "ni")):
                mask = (np.asarray(getattr(state, q)) > 0) & (np.asarray(getattr(state, n)) <= 0)
                seed_masks[name] = mask
                dumped[name + ".mass"] = np.asarray(getattr(state, q))[mask]
                dumped[name + ".number"] = np.asarray(getattr(state, n))[mask]
                dumped[name + ".alt"] = np.asarray(alt, dtype=np.float32)[mask]
        built = []
        source_temperature = kw.get("temperature")

        def temperature():
            if not built:
                built.append(source_temperature() if callable(source_temperature) else source_temperature)
            return built[0]

        host_kw = dict(kw, temperature=temperature)
        start = time.perf_counter()
        receipt = original(state, xp, cfg, alt, **host_kw)
        host_seconds = time.perf_counter() - start
        closed = dict(zip(source_temperature.__code__.co_freevars,
                          (cell.cell_contents for cell in source_temperature.__closure__)))
        provider = make_temperature_provider(closed["theta_h"], closed["total_pressure_h"])
        temperature_cells = [0]

        def device_temperature(idx):
            gathered = provider(idx)
            reference = temperature().ravel()[idx.get()]
            assert np.array_equal(gathered.get().view(np.uint32), reference.view(np.uint32))
            temperature_cells[0] += idx.size
            return gathered

        device_kw = dict(kw, temperature=device_temperature)
        cp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        result = thompson_cold_start_moment_closure(candidate, cp, cfg, alt, **device_kw)
        cp.cuda.runtime.deviceSynchronize()
        seconds = time.perf_counter() - start
        for field in ("nr", "ni") + (("nc",) if cfg.mp_physics == 28 else ()):
            a, b = np.asarray(getattr(state, field)), getattr(candidate, field).get()
            diffs = np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))
            print(f"{field} IDENTICAL={diffs == 0} differing_cells={diffs}", flush=True)
            assert diffs == 0
        assert json.dumps(result, sort_keys=True) == json.dumps(receipt, sort_keys=True)
        assert json.dumps(result) == json.dumps(receipt)
        if os.environ.get("CLOSURE_DUMP"):
            # Preserve the exact seed inputs from CPU replay for the second card.
            for name, q, n in (("rain", "qr", "nr"), ("ice", "qi", "ni")):
                mask = seed_masks[name]
                dumped[name + ".temperature"] = np.asarray(temperature(), dtype=np.float32)[mask]
                dumped[name + ".theta"] = closed["theta_h"][mask]
                dumped[name + ".pressure"] = closed["total_pressure_h"][mask]
                dumped[name + ".expected"] = np.asarray(getattr(state, n))[mask]
            dumped["receipt"] = np.array(json.dumps(receipt, sort_keys=True))
            np.savez_compressed(Path(os.environ["CLOSURE_DUMP"]) / f"init-{len(rows)}.npz", **dumped)
        print(f"TEMPERATURE IDENTICAL cells={temperature_cells[0]}", flush=True)
        print(f"RECEIPT IDENTICAL host_seconds={host_seconds:.6f} device_seconds={seconds:.6f}", flush=True)
        rows.append(dict(host_seconds=host_seconds, device_seconds=seconds, receipt=receipt))
        return receipt

    real._thompson_cold_start_moment_closure = grade
    from gpuwm.ingest.horiz import HorizontalSnapshot
    from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend
    for filename in (arg for arg in sys.argv[1:] if not arg.startswith("--")):
        rec = pickle.loads(Path(filename).read_bytes())
        kw = {k: v for k, v in rec["kwargs"].items() if not k.endswith("__name")}
        kw["preprocess_backend"] = resolve_preprocess_backend("cpu")
        kw["state_backend"] = "cpu"
        print(f"CAPTURE {filename}", flush=True)
        real.initialize_real(HorizontalSnapshot(**rec["snapshot"]), rec["cfg"],
                             rec["coord"], rec["terrain"], **kw)
    Path("closure-proof.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
