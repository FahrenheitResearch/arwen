"""Focused diagnostics for the cu6 parameter words and thermal helpers."""
from __future__ import annotations
import ctypes as c
import json
from pathlib import Path
import sys
import numpy as np
from check import compare_words, pointer, FP

def main():
    build = Path(sys.argv[1]).resolve()
    libraries = [c.CDLL(str(build / n)) for n in ("libcu6_oracle.so", "libcu6_host.so")]
    values = [np.empty(7, dtype=np.float32) for _ in libraries]
    thermal = []
    t = np.linspace(180., 330., 10001, dtype=np.float32)
    for lib, a in zip(libraries, values, strict=True):
        lib.cu6_constants.argtypes = [FP]
        lib.cu6_constants(pointer(a))
        result = [np.empty_like(t) for _ in range(3)]
        lib.cu6_thermo.argtypes = [c.c_int, FP, FP, FP, FP]
        lib.cu6_thermo(t.size, pointer(t), *(pointer(a) for a in result))
        thermal.append(result)
    output = {"constants": compare_words(*values), "constant_words": [a.view(np.uint32).tolist() for a in values],
              "thermal": [compare_words(a,b) for a,b in zip(*thermal, strict=True)]}
    t = np.r_[np.array([0x3f65c73d], dtype=np.uint32).view(np.float32),
              np.linspace(0.,1.5707,10001,dtype=np.float32)]
    tangent = []
    for lib in libraries:
        answer = np.empty_like(t)
        lib.cu6_tan.argtypes = [c.c_int, FP, FP]
        lib.cu6_tan(t.size,pointer(t),pointer(answer))
        tangent.append(answer)
    output["tangent"] = compare_words(*tangent)
    output["first_tangent_words"] = [int(a.view(np.uint32)[0]) for a in tangent]
    Path(sys.argv[2]).write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output))

if __name__ == "__main__":
    main()
