#!/usr/bin/env python3
"""Build the Linux CUDA submission recorder without initializing CUDA."""
from pathlib import Path
import argparse
import subprocess

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cc", default="cc")
    args = p.parse_args()
    source = Path(__file__).resolve().parents[1] / "gpuwm/native_step/recorder.c"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.cc, "-shared", "-fPIC", "-O2", "-std=gnu11",
               "-Wall", "-Wextra", "-Werror", "-o", str(args.output),
               str(source), "-ldl", "-pthread"]
    subprocess.run(command, check=True)
    print(args.output.resolve())

if __name__ == "__main__":
    main()
