"""Locate the first real assignment that differs in a single cu6 column.

Builds diagnostic copies inside the specified lane-owned build directory.
The production Fortran and CUDA source files are never changed here.
"""
from __future__ import annotations
import argparse
import ctypes as c
from pathlib import Path
import subprocess
import sys
import numpy as np
from check import fixture, fortran, host
import transcribe as tr

HERE = Path(__file__).resolve().parent


def build(build):
    source = (build / "module_cu_tiedtke.F").read_text()
    routines = tr.read_routines(source)
    by_line = {line: r for r in routines.values() for line, _ in r.body}
    out = []
    for line, s in tr.statements(source):
        routine = by_line.get(line)
        if routine is None:
            out.append(s)
            continue
        tail = s
        condition = None
        if s.startswith("if"):
            begin = s.index("(")
            end = tr.close_paren(s, begin)
            tail = s[end + 1:].strip()
            condition = s[:end + 1]
        parts = tr.split(tail, "=")
        if len(parts) == 2:
            name = parts[0].split("(")[0].strip()
            variable = routine.variables.get(name)
            if (variable is not None and variable.kind == "float") or name == routine.name:
                if condition is not None:
                    out.append(condition + " then")
                out.append(tail)
                out.append(f"write(*,'(A,I0,1X,Z8.8)') 'F ',{line},transfer({parts[0]},0)")
                if condition is not None:
                    out.append("endif")
                continue
        out.append(s)
    (build / "trace.F90").write_text("\n".join(out)+"\n")
    subprocess.run(["gfortran", "-cpp", "-ffree-form", "-ffree-line-length-none", "-O0", "-fno-fast-math",
                    "-ffp-contract=off", "-fPIC", "-shared", "module_model_constants.F",
                    "trace.F90", str(HERE / "oracle.F90"), "-o", "libcu6_trace.so"], cwd=build, check=True)
    cpp = (HERE / "host.cpp").read_text()
    for line in cpp.splitlines():
        if line.startswith('#include "'):
            target = (HERE / line.split('"')[1]).resolve()
            cpp = cpp.replace(line, f'#include "{target}"')
    cpp = '#include <cstdio>\n#define TK_TRACE(line, value) std::printf("C %d %08x\\n",line,bits<unsigned>(float(value)))\n' + cpp
    (build / "trace.cpp").write_text(cpp)
    subprocess.run(["g++", "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                    "-fPIC", "-shared", "trace.cpp", "-o", "libcu6_trace_host.so"], cwd=build, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", type=Path)
    p.add_argument("--column", type=int, default=56)
    p.add_argument("--nz", type=int, default=50)
    p.add_argument("--built", action="store_true")
    args = p.parse_args()
    args.build = args.build.resolve()
    if not args.built:
        build(args.build)
    lib = c.CDLL(str(args.build / "libcu6_oracle.so"))
    inputs, _ = fixture(lib, 72, args.nz)
    col = args.column
    single = {name: a if name == "znu" else np.ascontiguousarray(
        a[col:col+1] if a.ndim == 1 else a[:,col:col+1]) for name,a in inputs.items()}
    cfg = dict(dt=60., itimestep=2, stepcu=1, fqc=1, fqi=1)
    fortran(c.CDLL(str(args.build / "libcu6_trace.so")), single, **cfg)
    host(c.CDLL(str(args.build / "libcu6_trace_host.so")), single, **cfg)


if __name__ == "__main__":
    main()
