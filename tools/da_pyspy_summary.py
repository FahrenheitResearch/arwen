"""Seconds per source line and per function from a ``py-spy record --format raw`` file.

Each raw line is ``frame;frame;...;frame count``.  A sample is charged to the
deepest frame inside the project (``gpuwm/`` or ``tools/``, excluding this
profiler's own wrappers), so time spent inside numpy, cupy or the
interpreter lands on the project line that asked for it.  Measurement
tooling only.

    python -m tools.da_pyspy_summary prof.pyspy.txt --rate 20 --top 25 [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

FRAME = re.compile(r"^(?P<func>.*?) \((?P<file>[^():]+(?::[^():]+)*?):(?P<line>\d+)\)$")
SKIP = ("tools/da_phase_profile.py", "tools/da_pyspy_summary.py")


def project_frame(frame: str):
    match = FRAME.match(frame.strip())
    if match is None:
        return None
    path = match["file"].replace("\\", "/")
    if any(path.endswith(skip) for skip in SKIP):
        return None
    for root in ("gpuwm/", "tools/"):
        at = path.find(root)
        if at >= 0 and "site-packages/cupy" not in path \
                and "site-packages/numpy" not in path:
            return match["func"], path[at:], int(match["line"])
    return None


def summarise(lines, rate: float):
    by_line = Counter()
    by_function_self = Counter()
    by_function_inclusive = Counter()
    total = 0
    for raw in lines:
        raw = raw.rstrip("\n")
        if not raw:
            continue
        stack, _, count = raw.rpartition(" ")
        try:
            n = int(count)
        except ValueError:
            continue
        total += n
        frames = [project_frame(f) for f in stack.split(";")]
        frames = [f for f in frames if f is not None]
        if not frames:
            by_line[("<outside project>", "", 0)] += n
            continue
        func, path, line = frames[-1]
        by_line[(func, path, line)] += n
        by_function_self[(func, path)] += n
        for key in {(f, p) for f, p, _ in frames}:
            by_function_inclusive[key] += n
    seconds = 1.0 / rate
    return {
        "total_seconds": total * seconds,
        "lines": [{"function": f, "file": p, "line": ln, "seconds": n * seconds}
                  for (f, p, ln), n in by_line.most_common()],
        "functions_self": [{"function": f, "file": p, "seconds": n * seconds}
                           for (f, p), n in by_function_self.most_common()],
        "functions_inclusive": [{"function": f, "file": p, "seconds": n * seconds}
                                for (f, p), n in by_function_inclusive.most_common()],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("--rate", type=float, default=100.0,
                        help="the --rate py-spy recorded at (samples per second)")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None,
                        help="print each hot line's source text from this tree")
    args = parser.parse_args(argv)
    cache: dict[str, list[str]] = {}

    def text(path: str, line: int) -> str:
        if args.source_root is None:
            return ""
        if path not in cache:
            try:
                cache[path] = (args.source_root / path).read_text(
                    encoding="utf-8").splitlines()
            except OSError:
                cache[path] = []
        lines = cache[path]
        return ("  | " + lines[line - 1].strip()[:90]) if 0 < line <= len(lines) else ""
    summary = summarise(args.raw.read_text().splitlines(), args.rate)
    print(f"total {summary['total_seconds']:.1f} s")
    print("-- by line (deepest project frame) --")
    for row in summary["lines"][:args.top]:
        print(f"{row['seconds']:9.1f} s  {row['file']}:{row['line']}  {row['function']}"
              f"{text(row['file'], row['line'])}")
    print("-- by function, inclusive --")
    for row in summary["functions_inclusive"][:args.top]:
        print(f"{row['seconds']:9.1f} s  {row['file']}  {row['function']}")
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
