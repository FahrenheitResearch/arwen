"""Grade gathered CPU replay inputs and receipts on the current CUDA card."""
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from gpuwm.ingest.closure_device import gathered_numbers, _seed_receipt, make_temperature_provider

for filename in sys.argv[1:]:
    with np.load(filename) as dump:
        receipt = json.loads(str(dump["receipt"]))
        for name in ("rain", "ice"):
            m, n, a, t = [dump[name + "." + field] for field in ("mass", "number", "alt", "temperature")]
            empty = np.zeros_like(m)
            if name + ".theta" in dump:
                provider = make_temperature_provider(dump[name + ".theta"], dump[name + ".pressure"])
                temperature = provider(cp.arange(m.size, dtype=cp.int64))
                differences = int(np.count_nonzero(temperature.get().view(np.uint32) != t.view(np.uint32)))
                assert differences == 0
                print(f"{Path(filename).name} {name} TEMPERATURE IDENTICAL {m.size}/{m.size}", flush=True)
            else:
                temperature = t
            start = time.perf_counter()
            out, volume = gathered_numbers(name, m, n, a, temperature, empty, empty)
            cp.cuda.runtime.deviceSynchronize()
            seconds = time.perf_counter() - start
            differences = int(np.count_nonzero(out.get().view(np.uint32) != dump[name + ".expected"].view(np.uint32)))
            observed = _seed_receipt(name, m.size, m, a, volume.get(), t, empty, empty)
            expected = receipt[name + "_number_seed"]
            assert json.dumps(observed, sort_keys=True) == json.dumps(expected, sort_keys=True)
            assert differences == 0
            from gpuwm.core.thompson_entry import R1
            print(f"{Path(filename).name} {name} IDENTICAL {m.size}/{m.size} receipt IDENTICAL "
                  f"seeded={m.size} offenders={np.count_nonzero(m > R1)} seconds={seconds:.6f}", flush=True)
