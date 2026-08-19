"""Generate a LARGE randomized numpy f32 kernel sweep (sin/cos/exp/log)
for the optional `npmath_extended_sweep` Rust test.

Not committed as goldens (too big); run ad hoc:

    python tools/static_rust_port/gen_npmath_sweep.py <outdir>
    GPUWM_NPMATH_SWEEP=<outdir> cargo test -p static-fields npmath_extended
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(31415926)

n = 400_000
trig = np.concatenate([
    rng.uniform(-4.0, 4.0, n // 2),
    rng.uniform(-200_000.0, 200_000.0, n // 4),
    rng.uniform(-1e-3, 1e-3, n // 8),
    np.float32(np.pi / 2) * rng.integers(-40, 40, n // 8).astype(np.float64),
]).astype(np.float32)
exp_in = np.concatenate([
    rng.uniform(-120.0, 100.0, n // 2),
    rng.uniform(-1.0, 1.0, n // 2),
]).astype(np.float32)
log_in = np.concatenate([
    np.exp(rng.uniform(-100.0, 88.0, n // 2)).astype(np.float64),
    rng.uniform(1e-38, 4.0, n // 4),
    rng.uniform(0.5, 1.5, n // 4),
]).astype(np.float32)

with np.errstate(all="ignore"):
    (out / "trig_in.f32").write_bytes(trig.tobytes())
    (out / "sin_out.f32").write_bytes(np.sin(trig).tobytes())
    (out / "cos_out.f32").write_bytes(np.cos(trig).tobytes())
    (out / "exp_in.f32").write_bytes(exp_in.tobytes())
    (out / "exp_out.f32").write_bytes(np.exp(exp_in).tobytes())
    (out / "log_in.f32").write_bytes(log_in.tobytes())
    (out / "log_out.f32").write_bytes(np.log(log_in).tobytes())
print(f"wrote sweep ({trig.size} trig, {exp_in.size} exp, {log_in.size} log)"
      f" to {out}")
