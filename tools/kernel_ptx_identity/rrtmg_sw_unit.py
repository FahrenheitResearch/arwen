"""PTX identity of the legacy-RRTMG shortwave unit between two source trees.

compare.py reads a module as ``module_source(name)`` assembles it, and
rrtmg_sw.cu does not compile that way: CudaSW prepends the table #defines
that ``_pack_cuda_tables`` builds and compiles the result with
``("-std=c++17", "--ftz=false")`` (gpuwm/core/rrtmg_sw.py).  This runs
compare.compare() unchanged, with only the per-tree assembly swapped for
CudaSW's, so a re-pin of rrtmg_sw.cu can cite which entries moved.

Usage, from a checkout, under an interpreter that has CuPy's NVRTC::

    python -m tools.kernel_ptx_identity.rrtmg_sw_unit \\
        --tree <label>=<path> --tree <label>=<path> \\
        [--python <interpreter>] [--out <receipt.json>]
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.kernel_ptx_identity import compare

OPTION_SETS = {"cudasw": ("-std=c++17", "--ftz=false")}

_CHILD = r'''
import json, pathlib, sys
tree = pathlib.Path(sys.argv[1]).resolve()
import gpuwm
owner = pathlib.Path(gpuwm.__file__).resolve().parent.parent
if owner != tree:
    sys.exit("gpuwm resolved outside the tree under comparison")
from gpuwm.core import rrtmg_sw as sw
from gpuwm.core.rrtmg_legacy import _sw_tables
_packed, defines = sw._pack_cuda_tables(_sw_tables())
source = (tree / "gpuwm" / "core" / "kernels" / "rrtmg_sw.cu").read_text(encoding="ascii")
print(json.dumps({"rrtmg_sw": defines + source}))
'''


def cudasw_sources(python: str, tree: Path, modules: list[str]) -> dict[str, str]:
    """The string CudaSW hands NVRTC in ``tree``: no WRF-exact selector, no
    visible device."""
    if list(modules) != ["rrtmg_sw"]:
        raise SystemExit("this assembly is the rrtmg_sw unit's only")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GPUWM_WRF_EXACT")}
    env.update(PYTHONPATH=str(tree), CUDA_VISIBLE_DEVICES="",
               GPUWM_NO_LOCAL_GPU="1")
    done = subprocess.run([python, "-c", _CHILD, str(tree)],
                          capture_output=True, text=True, cwd=str(tree), env=env)
    if done.returncode:
        raise SystemExit(f"{tree.name}: CudaSW assembly failed\n{done.stderr[-4000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tree", type=compare._tree, action="append",
                        required=True, help="<label>=<path>, exactly twice")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter for the per-tree assembly children")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    trees = dict(args.tree)
    if len(args.tree) != 2 or len(trees) != 2:
        parser.error("give --tree exactly twice, with two different labels")
    compare.tree_sources = cudasw_sources
    receipt = compare.compare(trees, ["rrtmg_sw"], compare.DEFAULT_ARCHS,
                              OPTION_SETS, args.python)
    receipt["tool"] = "tools/kernel_ptx_identity/rrtmg_sw_unit.py"
    receipt["assembly"] = (
        "gpuwm.core.rrtmg_sw._pack_cuda_tables(_sw_tables()) defines + "
        "kernels/rrtmg_sw.cu, as CudaSW.__init__ builds it, in a child "
        "process per tree; GPUWM_WRF_EXACT* removed, CUDA_VISIBLE_DEVICES "
        "empty")
    receipt["nvrtc_build"] = next(
        (dist.version for dist in importlib.metadata.distributions()
         if "nvrtc" in (dist.metadata["Name"] or "").lower()), None)
    text = json.dumps(receipt, indent=1) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
