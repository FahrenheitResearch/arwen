"""De-risking spike driver: determinism + minimum-domain-size probes.

Run as::

    /tmp/arwen-env/bin/python -m tilestream.spike_check determinism 96 96 49 3
    /tmp/arwen-env/bin/python -m tilestream.spike_check sizes
    /tmp/arwen-env/bin/python -m tilestream.spike_check timing 96 96 49
"""

from __future__ import annotations

import json
import sys
import traceback

from tilestream import harness as H


def determinism(nx: int, ny: int, nz: int, steps: int) -> dict:
    state, cfg, digest = H.build_and_run(nx, ny, nz, steps)
    out, _fields = H.hash_outputs(state)
    return {"mode": "determinism", "nx": nx, "ny": ny, "nz": nz,
            "steps": steps, "state_sha256": digest,
            "benchmark_outputs_sha256": out,
            "n_persisted_arrays": len(H.state_arrays(state)),
            "persisted_names": list(H.state_arrays(state))}


def one_size(nx: int, ny: int, nz: int, steps: int) -> dict:
    try:
        _state, _cfg, digest = H.build_and_run(nx, ny, nz, steps)
        return {"nx": nx, "ny": ny, "nz": nz, "steps": steps, "ok": True,
                "sha256": digest}
    except Exception as exc:  # noqa: BLE001 - probing for the failure mode
        return {"nx": nx, "ny": ny, "nz": nz, "steps": steps, "ok": False,
                "error_type": type(exc).__name__, "error": str(exc)[:2000],
                "traceback": traceback.format_exc()[-2500:]}


def sizes(spec: str, nz: int, steps: int) -> dict:
    rows = []
    for token in spec.split(","):
        n = int(token)
        rows.append(one_size(n, n, nz, steps))
    return {"mode": "sizes", "rows": rows}


def timing(nx: int, ny: int, nz: int, steps: int = 20,
           warmup: int = 3) -> dict:
    import cupy as cp

    cfg = H.make_config(nx, ny, nz)
    state = H.make_state(cfg)
    H.run_steps(state, cfg, warmup)
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    H.run_steps(state, cfg, steps, sync=False)
    end.record()
    cp.cuda.runtime.deviceSynchronize()
    ms = float(cp.cuda.get_elapsed_time(start, end))
    return {"mode": "timing", "nx": nx, "ny": ny, "nz": nz, "steps": steps,
            "gpu_ms_per_step": ms / steps,
            "sha256_after": H.hash_state(state)}


def main(argv: list[str]) -> int:
    mode = argv[1]
    if mode == "determinism":
        result = determinism(int(argv[2]), int(argv[3]), int(argv[4]),
                             int(argv[5]))
    elif mode == "sizes":
        result = sizes(argv[2], int(argv[3]), int(argv[4]))
    elif mode == "one":
        result = one_size(int(argv[2]), int(argv[3]), int(argv[4]),
                          int(argv[5]))
    elif mode == "timing":
        result = timing(int(argv[2]), int(argv[3]), int(argv[4]),
                        int(argv[5]) if len(argv) > 5 else 20)
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
