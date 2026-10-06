"""Read the per-thread local frame of the lane's two kernel modules.

The same reading tools/vram_reserve_probe.py ``frames`` takes for every
module (the production loader, local_size_bytes of each exported kernel,
the widest per module), restricted to ``swint`` and ``rrtmg_aer3`` so it
runs in seconds.  Prints JSON with the device, the NVRTC build and the
frames.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SYMBOL = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)')


def main() -> None:
    import cupy as cp
    from gpuwm.core.kernels import load_module

    kdir = Path(__file__).resolve().parents[2] / "gpuwm" / "core" / "kernels"
    frames = {}
    for name in ("swint", "rrtmg_aer3"):
        module = load_module(name)
        widest = 0
        text = (kdir / f"{name}.cu").read_text(encoding="utf-8")
        for symbol in sorted(set(_SYMBOL.findall(text))):
            widest = max(widest, int(
                module.get_function(symbol).attributes["local_size_bytes"]))
        frames[name] = widest
    from gpuwm.certify.compile_platform import compile_platform_fingerprint
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(json.dumps({
        "device": props["name"].decode(),
        "platform": compile_platform_fingerprint(),
        "frames": frames}, indent=2))


if __name__ == "__main__":
    main()
