"""The route inventory is regenerable, and it cannot miss a route.

An inventory that silently drops a construction path would let the FTZ
receipt claim coverage it does not have, so the injection test below plants
one site of each supported kind in a throwaway checkout and requires all
three back.  A hand-maintained list would pass the regeneration half and fail
this one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ftz_receipt import route_inventory as ri  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "tools" / "ftz_receipt" / "receipt" / "route_inventory.json"


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(root), check=True,
                   capture_output=True, text=True)


def test_committed_inventory_regenerates_byte_identically():
    assert COMMITTED.exists(), f"{COMMITTED} is not committed"
    regenerated = ri.render(ri.build_inventory(ROOT))
    assert regenerated == COMMITTED.read_text(encoding="utf-8"), (
        "route_inventory.json drifted from the tree; regenerate with "
        "python -m tools.ftz_receipt.route_inventory")


def test_every_record_carries_file_line_constructor_and_options():
    document = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert document["schema"] == ri.SCHEMA_ID
    assert document["sites"], "inventory is empty"
    for site in document["sites"]:
        assert (ROOT / site["file"]).exists(), site["file"]
        assert site["line"] >= 1
        assert site["constructor_kind"] in set(
            ri.CONSTRUCTOR_KINDS.values())
        assert isinstance(site["takes_caller_options"], bool)
        if site["takes_caller_options"]:
            # Either a literal tuple, or the expression text that produced it.
            assert site["options"] is not None or (
                site["options_expression"] is not None
                or not site["options_argument_present"])
        else:
            assert site["options"] is None
        text = (ROOT / site["file"]).read_text(
            encoding="utf-8").splitlines()[site["line"] - 1].strip()
        assert text == site["text"], f"{site['file']}:{site['line']} moved"


def test_option_tuples_are_not_one_global_string():
    """The receipt's per-route claims rest on the tuples differing."""
    document = json.loads(COMMITTED.read_text(encoding="utf-8"))
    distinct = {json.dumps(t)
                for t in document["distinct_literal_option_tuples"]}
    assert len(distinct) > 1, (
        "a single option tuple across all sites would make the per-route "
        "receipt meaningless")


SYNTHETIC = '''\
import cupy as cp
from cupy.cuda.compiler import compile_using_nvrtc


def make_raw_module():
    return cp.RawModule(code="__global__ void k() {}",
                        options=("-std=c++17", "--ftz=false"))


def make_nvrtc():
    return compile_using_nvrtc("__global__ void k() {}",
                               ("-std=c++17", "-arch=compute_90"))


def make_reduction():
    return cp.ReductionKernel("T x", "int32 y", "x != (T)0 ? 1 : 0",
                              "a | b", "y = a", "0", "synthetic_probe")
'''


@pytest.fixture()
def injected_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "synthetic_routes.py").write_text(
        SYNTHETIC, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    return root


def test_injected_sites_of_all_three_kinds_are_reported(injected_checkout):
    document = ri.build_inventory(injected_checkout)
    kinds = {site["constructor_kind"] for site in document["sites"]}
    assert kinds == {"cupy.RawModule",
                     "cupy.cuda.compiler.compile_using_nvrtc",
                     "cupy.ReductionKernel"}, (
        f"inventory missed a route kind; reported {sorted(kinds)}")
    by_kind = {site["constructor_kind"]: site for site in document["sites"]}
    assert by_kind["cupy.RawModule"]["options"] == [
        "-std=c++17", "--ftz=false"]
    assert by_kind["cupy.cuda.compiler.compile_using_nvrtc"]["options"] == [
        "-std=c++17", "-arch=compute_90"]
    assert by_kind["cupy.ReductionKernel"]["takes_caller_options"] is False
    assert by_kind["cupy.ReductionKernel"]["options"] is None


def test_inventory_reports_a_site_it_would_otherwise_skip(injected_checkout):
    """Control: removing the injected file must drop exactly those sites."""
    before = len(ri.build_inventory(injected_checkout)["sites"])
    (injected_checkout / "pkg" / "synthetic_routes.py").write_text(
        "x = 1\n", encoding="utf-8")
    after = len(ri.build_inventory(injected_checkout)["sites"])
    assert before == 3 and after == 0, (before, after)


def test_scan_records_non_literal_option_expressions():
    """A tuple built at run time is recorded as an expression, not dropped."""
    text = ('import cupy as cp\n'
            'def f(options):\n'
            '    return cp.RawModule(code="", options=tuple(options))\n')
    sites = ri.scan_source("scratch.py", text)
    assert len(sites) == 1
    assert sites[0]["options"] is None
    assert sites[0]["options_are_literal"] is False
    assert sites[0]["options_expression"] == "tuple(options)"
