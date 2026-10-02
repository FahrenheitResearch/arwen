"""PTX identity of CUDA kernel modules between two source trees.

What it answers: does the compiler emit the same code for a kernel module at
two trees?  It is the reading a kernel re-pin cites when the source bytes
moved and the claim is that the compiled behaviour did not (an opt-in
``#if`` branch, a comment, a guard), or names exactly which kernels did move
(the 2.8.2 openbc merge receipt under tools/advect_wrf471_oracle/receipts/).

Usage, from a checkout, under an interpreter that has CuPy's NVRTC::

    python -m tools.kernel_ptx_identity.compare \\
        --tree <label>=<path> --tree <label>=<path> \\
        --module acoustic [--module ...] \\
        [--arch compute_89 --arch compute_90 --arch compute_120] \\
        [--options="<label>:<options>" ...] \\
        [--python <interpreter>] [--out <receipt.json>]

How it reads each tree.  A child process of ``--python`` (default: this
interpreter) runs with ``PYTHONPATH=<path>``, ``CUDA_VISIBLE_DEVICES`` empty
and every ``GPUWM_WRF_EXACT*`` selector removed, and returns
``gpuwm.core.kernels.module_source(name)``: the exact string the default
loader hands NVRTC.  The child refuses a tree whose ``gpuwm`` resolves
anywhere else, so an installed gpuwm cannot stand in for the tree.

How it compiles.  This process compiles every string to PTX for each
``--arch`` under each ``--options`` set through
:func:`gpuwm.nvrtc_cache_key.compile_program`, the keyed route A160 requires
of a direct NVRTC compile.  Both trees go through the same helper with the
same options, so the appended options constant is identical on both sides.
No device is opened: NVRTC compiles to PTX without one.

The default option sets are ``load_module``, the loader's own caller options
(``-std=c++17``, gpuwm/core/kernels/__init__.py), and ``rawmodule``, that tuple
with the flush flag CuPy's ``_compile_with_cache_cuda`` appends to every
RawModule compile, which is what a forecast actually runs.  Readings are keyed
``<label> | <arch>`` and the receipt lists each label's options once.

What it records, comparing nothing normalised: per module, each tree's file
and assembled-source SHA-256; per option set and architecture, whether the
whole PTX text is identical, and every ``.entry`` and defined ``.func`` by
name with the ones whose text differs listed.  No path appears in the
receipt: trees are named by the labels given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

DEFAULT_ARCHS = ("compute_89", "compute_90", "compute_120")
DEFAULT_OPTION_SETS = {"load_module": ("-std=c++17",),
                       "rawmodule": ("-std=c++17", "-ftz=true")}

_CHILD = r'''
import json, pathlib, sys
tree = pathlib.Path(sys.argv[1]).resolve()
import gpuwm
from gpuwm.core.kernels import module_source
owner = pathlib.Path(gpuwm.__file__).resolve().parent.parent
if owner != tree:
    sys.exit("gpuwm resolved outside the tree under comparison")
print(json.dumps({name: module_source(name) for name in json.loads(sys.argv[2])}))
'''

#: The head of a PTX function: optional linkage, the kind, an optional
#: return parameter list, then the name and its parameter list.
_HEAD = re.compile(r"^(?:\.(?:visible|weak|extern)\s+)*\.(entry|func)\s+"
                   r"(?:\([^)]*\)\s*)?([A-Za-z_$%][\w$]*)\s*\(")


def ptx_functions(ptx: str) -> dict[str, dict[str, str]]:
    """``{"entry": {name: text}, "func": {name: text}}`` for every function
    DEFINED in ``ptx``; a prototype (a head that reaches ``;`` before a
    body) is not a definition and is skipped."""
    found: dict[str, dict[str, str]] = {"entry": {}, "func": {}}
    lines = ptx.split("\n")
    i = 0
    while i < len(lines):
        match = _HEAD.match(lines[i])
        if not match:
            i += 1
            continue
        kind, name = match.groups()
        j, body = i + 1, False
        while j < len(lines):
            text = lines[j].strip()
            if text == "{":
                body = True
            elif body and lines[j] == "}":
                break
            elif not body and text.endswith(";"):
                break
            j += 1
        if body:
            if name in found[kind]:
                raise ValueError(f"PTX defines {kind} {name} twice")
            found[kind][name] = "\n".join(lines[i:j + 1])
        i = j + 1
    return found


def tree_sources(python: str, tree: Path, modules: list[str]) -> dict[str, str]:
    """``module_source(name)`` for each module as ``tree`` assembles it by
    default: no WRF-exact selector, no visible device."""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GPUWM_WRF_EXACT")}
    env.update(PYTHONPATH=str(tree), CUDA_VISIBLE_DEVICES="",
               GPUWM_NO_LOCAL_GPU="1")
    done = subprocess.run([python, "-c", _CHILD, str(tree), json.dumps(modules)],
                          capture_output=True, text=True, cwd=str(tree), env=env)
    if done.returncode:
        raise SystemExit(f"{tree.name}: module_source failed\n{done.stderr[-4000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def compile_ptx(source: str, options: tuple[str, ...], arch: str, name: str) -> str:
    from gpuwm.nvrtc_cache_key import compile_program
    blob = compile_program(source, options + (f"-arch={arch}",),
                           name=f"{name}.cu", target="ptx")
    text = blob.decode("utf-8") if isinstance(blob, bytes) else str(blob)
    return text.rstrip("\x00")


def _sha256(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def compare(trees: dict[str, Path], modules: list[str], archs, option_sets,
            python: str) -> dict:
    (a_label, a_tree), (b_label, b_tree) = trees.items()
    sources = {label: tree_sources(python, tree, modules)
               for label, tree in trees.items()}
    from cupy.cuda import nvrtc
    receipt = {
        "tool": "tools/kernel_ptx_identity/compare.py",
        "trees": list(trees),
        "nvrtc_version": list(nvrtc.getVersion()),
        "assembly": ("gpuwm.core.kernels.module_source(name) in a child "
                     "process per tree; GPUWM_WRF_EXACT* removed, "
                     "CUDA_VISIBLE_DEVICES empty"),
        "compile": ("gpuwm.nvrtc_cache_key.compile_program(source, options "
                    "+ (-arch=<arch>,), target=ptx)"),
        "option_sets": {label: list(options)
                        for label, options in option_sets.items()},
        "archs": list(archs),
        "modules": {},
    }
    for module in modules:
        row = {"file_sha256": {}, "assembled_source_sha256": {}}
        for label, tree in trees.items():
            row["file_sha256"][label] = _sha256(
                (tree / "gpuwm" / "core" / "kernels" / f"{module}.cu").read_bytes())
            row["assembled_source_sha256"][label] = _sha256(sources[label][module])
        names: dict[str, list[str]] | None = None
        readings = {}
        for label_set, options in option_sets.items():
            for arch in archs:
                ptx = {label: compile_ptx(sources[label][module], options, arch, module)
                       for label in trees}
                functions = {label: ptx_functions(text) for label, text in ptx.items()}
                reading = {"ptx_identical": ptx[a_label] == ptx[b_label],
                           "ptx_sha256": {label: _sha256(text) for label, text in ptx.items()}}
                these = {}
                for kind in ("entry", "func"):
                    a, b = functions[a_label][kind], functions[b_label][kind]
                    every = sorted(set(a) | set(b))
                    these[kind] = every
                    reading[f"{kind}_compared"] = len(every)
                    reading[f"{kind}_different"] = [n for n in every if a.get(n) != b.get(n)]
                if names is None:
                    names = these
                elif names != these:
                    reading["function_names"] = these
                readings[f"{label_set} | {arch}"] = reading
        row["entries"] = names["entry"]
        row["funcs"] = names["func"]
        row["readings"] = readings
        receipt["modules"][module] = row
    return receipt


def _tree(value: str) -> tuple[str, Path]:
    label, sep, path = value.partition("=")
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError("--tree takes <label>=<path>")
    root = Path(path).resolve()
    if not (root / "gpuwm" / "core" / "kernels" / "__init__.py").is_file():
        raise argparse.ArgumentTypeError(f"{label}: not a gpuwm source tree")
    return label, root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tree", type=_tree, action="append", required=True,
                        help="<label>=<path>, exactly twice")
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument("--arch", action="append")
    parser.add_argument("--options", action="append",
                        help="one NVRTC option set as <label>:<options>, "
                             "options space separated; spell it "
                             "--options=<label>:<options>")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter for the per-tree assembly children")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    trees = dict(args.tree)
    if len(args.tree) != 2 or len(trees) != 2:
        parser.error("give --tree exactly twice, with two different labels")
    option_sets = dict(DEFAULT_OPTION_SETS)
    if args.options:
        option_sets = {}
        for text in args.options:
            label, sep, options = text.partition(":")
            if not sep or not label or label in option_sets:
                parser.error("--options takes <label>:<options>, one label each")
            option_sets[label] = tuple(options.split())
    receipt = compare(trees, list(dict.fromkeys(args.module)),
                      tuple(args.arch or DEFAULT_ARCHS), option_sets, args.python)
    text = json.dumps(receipt, indent=1) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
