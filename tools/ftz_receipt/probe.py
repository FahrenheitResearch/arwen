"""Measure FP32 subnormal handling on every production compile route.

The probe answers one question per (route, mechanism) cell: does the device
result equal the IEEE-754 binary32 result, or the result you get by treating
subnormal operands and subnormal results as signed zero?  It answers it by
running the arithmetic and recording bits.  No cell's answer is written down
anywhere in this tool; ``classify`` derives every verdict from the bit
columns by the rule in its docstring, and ``tests/test_ftz_receipt.py``
recomputes them independently from ``bitpatterns.csv``.

Routes
------
``R1``            the production loader, ``gpuwm.core.kernels.load_module``.
``R1-ftztrue``    compile-side control: R1's own option tuple plus an
                  explicit ``--ftz=true``.  If this arm does not differ from
                  an arm that asks for ``--ftz=false``, the pipeline is not
                  flag-sensitive and no other cell means anything.
``R2``            ``cp.RawModule`` with the option tuple the shortwave
                  radiation site supplies.
``R3``            ``compile_using_nvrtc`` + ``cuda.function.Module``, the
                  longwave site's shape, which bypasses CuPy's option
                  rewrite entirely.
``R4``            ``cp.ReductionKernel``, which takes no caller options at
                  all.
``R5``            inline PTX written without the ``.ftz`` modifier, compiled
                  through R1's module.

The option tuple for each production arm is READ from
``receipt/route_inventory.json`` rather than typed here, so an arm cannot
drift away from the site it claims to represent.

Usage::

    python -m tools.ftz_receipt.probe --out tools/ftz_receipt/receipt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import os
import platform
import re
import sys
from pathlib import Path

# The in-memory kernel cache makes the compile fire on every session, so the
# effective-option capture below measures rather than reports a cache hit.
os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")

import numpy as np  # noqa: E402

from tools.ftz_receipt import route_inventory as ri  # noqa: E402
from tools.ftz_receipt import sass  # noqa: E402

SCHEMA_ID = "gpuwm.ftz-receipt/v1"

#: Verdict vocabulary.  Consumers import these names; nothing re-types the
#: strings, so a rename cannot leave a stale literal behind in a test.
VERDICT_IEEE = "ieee-agreement"
VERDICT_FLUSH = "flush-to-zero"
VERDICT_DIVERGENT = "divergent"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_NOT_APPLICABLE = "not-applicable"

VERDICTS = (VERDICT_IEEE, VERDICT_FLUSH, VERDICT_DIVERGENT,
            VERDICT_INCONCLUSIVE, VERDICT_NOT_APPLICABLE)

#: Mechanism ids, in the order the plan enumerates them.
MECHANISMS = (
    ("plain-operator arithmetic", "mul"),
    ("__f*_rn intrinsic", "mul"),
    ("fminf/fmaxf", "min"),
    ("float compare", "compare"),
    ("__double2float_rn", "convert"),
    ("(float) cast", "convert"),
)

MODULE_KERNELS = {
    "plain-operator arithmetic": "ftz_probe_mul_op",
    "__f*_rn intrinsic": "ftz_probe_mul_rn",
    "fminf/fmaxf": "ftz_probe_minmax",
    "float compare": "ftz_probe_compare",
    "__double2float_rn": "ftz_probe_d2f_rn",
    "(float) cast": "ftz_probe_cast",
}

PTX_KERNELS = {
    "plain-operator arithmetic": "ftz_probe_ptx_mul",
    "fminf/fmaxf": "ftz_probe_ptx_min",
    "float compare": "ftz_probe_ptx_setp",
    "(float) cast": "ftz_probe_ptx_cvt",
}

PTX_NOT_APPLICABLE = {
    "__f*_rn intrinsic":
        "inline PTX has no intrinsic layer: __fmul_rn lowers to the same "
        "mul.rn.f32 the plain-operator cell of this route writes directly",
    "__double2float_rn":
        "inline PTX has no intrinsic layer: __double2float_rn lowers to the "
        "same cvt.rn.f32.f64 the cast cell of this route writes directly",
}


def _f32_bits(value: np.float32) -> int:
    return int(np.float32(value).view(np.uint32))


def _bits_to_f32(bits: int) -> np.float32:
    return np.uint32(bits).view(np.float32)


TINY_NORMAL = np.float32(2.0) ** np.float32(-126)
TRUE_MIN = _bits_to_f32(0x00000001)

#: Input table.  ``a``/``b`` feed the four float-operand mechanisms; ``d``
#: feeds the two double-to-float conversions.  Rows 1 and 7 are normal
#: throughout: a probe that returned zero everywhere would fail them.
INPUT_ROWS = (
    {"id": "normal_control", "a": np.float32(1.5), "b": np.float32(2.0),
     "d": 3.5},
    {"id": "subnormal_product", "a": np.float32(1e-20),
     "b": np.float32(1e-20), "d": 1e-40},
    {"id": "deep_underflow", "a": np.float32(1e-30), "b": np.float32(1e-15),
     "d": 1e-45},
    {"id": "subnormal_operand", "a": np.float32(1e-40), "b": np.float32(1.0),
     "d": float(np.float32(1e-40))},
    {"id": "min_subnormal_operand", "a": TRUE_MIN, "b": np.float32(1.0),
     "d": float(TRUE_MIN)},
    {"id": "negative_subnormal_operand", "a": -np.float32(1e-40),
     "b": np.float32(1.0), "d": -float(np.float32(1e-40))},
    {"id": "smallest_normal_control", "a": TINY_NORMAL, "b": np.float32(1.0),
     "d": float(TINY_NORMAL)},
    # The compare mechanism is only separable against an exact zero: with two
    # non-zero operands a flushed compare and an IEEE compare agree, and the
    # cell would measure nothing.
    {"id": "subnormal_vs_zero", "a": np.float32(1e-40), "b": np.float32(0.0),
     "d": 0.0},
    {"id": "min_subnormal_vs_zero", "a": TRUE_MIN, "b": np.float32(0.0),
     "d": 0.0},
)


# ---- host references ------------------------------------------------------

def _is_subnormal(value: np.float32) -> bool:
    bits = _f32_bits(value)
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0


def _flush(value: np.float32) -> np.float32:
    """Signed zero for a subnormal, the value itself otherwise."""
    if _is_subnormal(value):
        return _bits_to_f32(_f32_bits(value) & 0x80000000)
    return np.float32(value)


def _reference(op: str, row: dict, *, flushing: bool) -> int:
    """Host binary32 answer for one cell, with or without flushing."""
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        if op == "convert":
            result = np.float32(np.float64(row["d"]))
            return _f32_bits(_flush(result) if flushing else result)
        a = _flush(row["a"]) if flushing else np.float32(row["a"])
        b = _flush(row["b"]) if flushing else np.float32(row["b"])
        if op == "mul":
            result = np.float32(a * b)
        elif op == "min":
            result = np.float32(min(a, b)) if not (
                np.isnan(a) or np.isnan(b)) else np.float32(np.nan)
        elif op == "compare":
            result = np.float32(1.0) if a != b else np.float32(0.0)
        else:
            raise ValueError(f"unknown op {op!r}")
        return _f32_bits(_flush(result) if flushing else result)


# ---- verdict rule ---------------------------------------------------------

def classify(rows: list[dict]) -> str:
    """Derive one cell's verdict from its bit columns.

    ``rows`` are that cell's per-input records, each carrying
    ``device_bits``, ``ieee_reference_bits`` and ``flushed_reference_bits``.
    The rule, in order:

    1. no rows at all -> ``not-applicable`` (the route cannot express it);
    2. every device word equals the IEEE reference, and at least one input
       could have told the two references apart -> ``ieee-agreement``;
    3. every device word equals the IEEE reference but no input separates
       the references -> ``inconclusive`` (the cell measured nothing);
    4. every device word equals the flushed reference -> ``flush-to-zero``;
    5. otherwise -> ``divergent``.

    Rule 4 is checked after rule 2 so a cell whose references coincide
    everywhere is never called a flusher on the strength of rows that could
    not distinguish.
    """
    if not rows:
        return VERDICT_NOT_APPLICABLE
    discriminating = sum(
        1 for row in rows
        if row["ieee_reference_bits"] != row["flushed_reference_bits"])
    matches_ieee = all(
        row["device_bits"] == row["ieee_reference_bits"] for row in rows)
    if matches_ieee:
        return VERDICT_IEEE if discriminating else VERDICT_INCONCLUSIVE
    if all(row["device_bits"] == row["flushed_reference_bits"]
           for row in rows):
        return VERDICT_FLUSH
    return VERDICT_DIVERGENT


# ---- compile-path instrumentation ----------------------------------------

#: PTX float instructions whose ``.ftz`` modifier the parser reports.
PTX_FLOAT_INSTRUCTION = re.compile(
    r"^\s*(?P<op>[a-z][a-z0-9]*)"
    r"(?P<mods>(?:\.[a-z0-9]+)*)"
    r"\.f32\b")


def object_format(blob: bytes) -> str:
    """Name what NVRTC handed back: an ELF cubin, or PTX text."""
    if blob[:4] == b"\x7fELF":
        return "cubin"
    return "ptx"


def ptx_ftz_modifiers(text: str) -> dict:
    """Count ``.f32`` instructions in PTX, split by ``.ftz`` presence.

    Reads the text; it does not know or care which file it came from, so a
    parser that always answered the same way fails the fixture pair in
    ``tests/test_ftz_receipt.py``.
    """
    with_ftz: dict[str, int] = {}
    without_ftz: dict[str, int] = {}
    for line in text.replace("\x00", "").splitlines():
        match = PTX_FLOAT_INSTRUCTION.match(line)
        if match is None:
            continue
        modifiers = match.group("mods").split(".")
        bucket = with_ftz if "ftz" in modifiers else without_ftz
        key = match.group("op")
        bucket[key] = bucket.get(key, 0) + 1
    return {
        "with_ftz": dict(sorted(with_ftz.items())),
        "without_ftz": dict(sorted(without_ftz.items())),
        "f32_instruction_count": (sum(with_ftz.values())
                                  + sum(without_ftz.values())),
    }


class NvrtcCapture:
    """Record the option tuple and PTX NVRTC actually received.

    CuPy rewrites caller options inside ``_compile_with_cache_cuda`` before
    the compiler sees them, so the only place the effective tuple exists is
    the NVRTC entry point itself.  Wrapping it measures; reading the source
    and re-deriving the rewrite would only restate it.
    """

    def __init__(self):
        self.label: str | None = None
        self.records: dict[str, list[dict]] = {}
        self._original = None
        self._module = None

    def __enter__(self):
        from cupy.cuda import compiler
        self._module = compiler
        self._original = compiler._compile_using_nvrtc_no_warning
        capture = self

        def wrapper(source, options=(), *args, **kwargs):
            result = capture._original(source, options, *args, **kwargs)
            if capture.label is not None:
                blob = result[0] if isinstance(result, tuple) else result
                if isinstance(blob, str):
                    blob = blob.encode("utf-8")
                flags = [item for item in options
                         if not str(item).startswith("-I")]
                includes = len(options) - len(flags)
                capture.records.setdefault(capture.label, []).append({
                    "fired": True,
                    "effective_flags": flags,
                    "include_path_count": includes,
                    "include_paths_omitted":
                        "-I entries are this machine's toolkit layout, not "
                        "part of the arithmetic contract",
                    "object": blob,
                    "object_format": object_format(blob),
                    "object_sha256": hashlib.sha256(blob).hexdigest(),
                    "source_sha256": hashlib.sha256(
                        source.encode("utf-8")).hexdigest(),
                })
            return result

        compiler._compile_using_nvrtc_no_warning = wrapper
        return self

    def __exit__(self, *exc):
        self._module._compile_using_nvrtc_no_warning = self._original
        return False


def ftz_append_site() -> dict:
    """Locate CuPy's ``-ftz`` option append at run time, never by coordinate."""
    from cupy.cuda import compiler
    source = inspect.getsource(compiler)
    pattern = re.compile(r"options\s*\+=\s*\(\s*'-ftz=(true|false)'\s*,?\s*\)")
    for number, line in enumerate(source.splitlines(), 1):
        match = pattern.search(line)
        if match:
            return {
                "found": True,
                "module": compiler.__name__,
                "line": number,
                "text": line.strip(),
                "appended": f"-ftz={match.group(1)}",
            }
    return {"found": False, "module": compiler.__name__,
            "reason": "no unconditional -ftz append in "
                      "inspect.getsource(cupy.cuda.compiler)"}


# ---- routes ---------------------------------------------------------------

def _probe_source() -> str:
    from gpuwm.core import kernels as gpuwm_kernels
    return gpuwm_kernels._preamble() + (
        Path(gpuwm_kernels.__file__).parent / "ftz_probe.cu").read_text(
            encoding="utf-8")


def _site_options(inventory: dict, **spec) -> tuple[dict, list]:
    site = ri.find_site(inventory, **spec)
    return site, list(site["options"] or [])


def build_routes(inventory: dict, capture: NvrtcCapture) -> dict:
    """Compile every arm, each through its production constructor."""
    import cupy as cp
    from cupy.cuda import compiler as cc
    from gpuwm.core import kernels as gpuwm_kernels

    routes: dict[str, dict] = {}
    source = _probe_source()

    # R1 -- the production loader.  No option tuple appears here; the loader
    # owns it.
    r1_site, r1_options = _site_options(
        inventory, file="gpuwm/core/kernels/__init__.py",
        constructor_kind="cupy.RawModule", enclosing="load_module")
    capture.label = "R1"
    r1_module = gpuwm_kernels.load_module("ftz_probe")
    routes["R1"] = {
        "id": "R1",
        "label": "loader RawModule",
        "constructor": "gpuwm.core.kernels.load_module",
        "site": f"{r1_site['file']}:{r1_site['line']}",
        "site_options": r1_options,
        "arm_options": r1_options,
        "kind": "module",
        "module": r1_module,
        "kernels": MODULE_KERNELS,
    }

    # R1-ftztrue -- compile-side control.  The one literal option in this
    # tool, and the reason it exists: without it nothing proves the pipeline
    # responds to the flag at all.
    control_options = r1_options + ["--ftz=true"]
    capture.label = "R1-ftztrue"
    control_module = cp.RawModule(code=source, options=tuple(control_options))
    control_module.compile()
    routes["R1-ftztrue"] = {
        "id": "R1-ftztrue",
        "label": "loader RawModule + explicit --ftz=true (control)",
        "constructor": "cupy.RawModule",
        "site": f"{r1_site['file']}:{r1_site['line']} + control flag",
        "site_options": r1_options,
        "arm_options": control_options,
        "kind": "module",
        "module": control_module,
        "kernels": MODULE_KERNELS,
    }

    # R2 -- the shortwave site's constructor and tuple.  The site left
    # cupy.RawModule with 2358b3c06: NVRTC 13 rejects the duplicated
    # --ftz that the RawModule path relied on (CuPy injects -ftz=true and
    # the last-occurrence-wins behaviour is gone), so rrtmg_sw.py now
    # compiles through the same direct-NVRTC bypass the longwave site
    # uses, and this arm rides that constructor.
    r2_site, r2_options = _site_options(
        inventory, file="gpuwm/core/rrtmg_sw.py",
        constructor_kind="cupy.cuda.compiler.compile_using_nvrtc",
        enclosing="__init__")
    capture.label = "R2"
    r2_ptx, _r2_mapping = cc.compile_using_nvrtc(
        source, tuple(r2_options), None, "ftz_probe.cu")
    r2_module = cp.cuda.function.Module()
    r2_module.load(r2_ptx.encode() if isinstance(r2_ptx, str) else r2_ptx)
    routes["R2"] = {
        "id": "R2",
        "label": "direct NVRTC with the shortwave option tuple",
        "constructor": "cupy.cuda.compiler.compile_using_nvrtc",
        "site": f"{r2_site['file']}:{r2_site['line']}",
        "site_options": r2_options,
        "arm_options": r2_options,
        "kind": "module",
        "module": r2_module,
        "kernels": MODULE_KERNELS,
    }

    # R3 -- the longwave site's direct-NVRTC shape.  Its last option element
    # is built at run time from the device arch; the inventory records the
    # element as dynamic and this arm supplies it the same way the site does.
    r3_site = ri.find_site(
        inventory, file="gpuwm/core/rrtmg_lw.py",
        constructor_kind="cupy.cuda.compiler.compile_using_nvrtc")
    arch = cc._get_arch()
    r3_options = [item if item is not None else f"-arch=compute_{arch}"
                  for item in r3_site["options"]]
    capture.label = "R3"
    ptx, _mapping = cc.compile_using_nvrtc(
        source, tuple(r3_options), None, "ftz_probe.cu")
    r3_module = cp.cuda.function.Module()
    r3_module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    routes["R3"] = {
        "id": "R3",
        "label": "direct NVRTC + cuda.function.Module",
        "constructor": "cupy.cuda.compiler.compile_using_nvrtc",
        "site": f"{r3_site['file']}:{r3_site['line']}",
        "site_options": r3_site["options"],
        "arm_options": r3_options,
        "kind": "module",
        "module": r3_module,
        "kernels": MODULE_KERNELS,
    }

    # R4 -- ReductionKernel, which supplies no caller options at all.
    r4_site = ri.find_site(
        inventory, file="gpuwm/core/mynn_pbl_gpu.py",
        constructor_kind="cupy.ReductionKernel", enclosing="_nonpositive")
    capture.label = "R4"
    routes["R4"] = {
        "id": "R4",
        "label": "CuPy-generated ReductionKernel",
        "constructor": "cupy.ReductionKernel",
        "site": f"{r4_site['file']}:{r4_site['line']}",
        "site_options": r4_site["options"],
        "arm_options": [],
        "kind": "reduction",
        "kernels": _reduction_kernels(cp),
    }

    # R5 -- inline PTX, riding R1's module.
    routes["R5"] = {
        "id": "R5",
        "label": "inline PTX without .ftz",
        "constructor": "gpuwm.core.kernels.load_module",
        "site": f"{r1_site['file']}:{r1_site['line']}",
        "site_options": r1_options,
        "arm_options": r1_options,
        "kind": "module",
        "module": r1_module,
        "kernels": PTX_KERNELS,
        "not_applicable": PTX_NOT_APPLICABLE,
    }
    capture.label = None
    return routes


def _reduction_kernels(cp) -> dict:
    """One ReductionKernel per mechanism, in the shipped site's shape."""
    def uint_kernel(expression: str, name: str):
        return cp.ReductionKernel(
            "uint32 x, uint32 y", "uint32 z", expression, "a | b", "z = a",
            "0", name)

    def double_kernel(expression: str, name: str):
        return cp.ReductionKernel(
            "float64 x, float64 y", "uint32 z", expression, "a | b", "z = a",
            "0", name)

    fa = "__int_as_float((int)x)"
    fb = "__int_as_float((int)y)"
    return {
        "plain-operator arithmetic": uint_kernel(
            f"(unsigned int)__float_as_int({fa} * {fb})", "ftz_probe_r4_mul"),
        "__f*_rn intrinsic": uint_kernel(
            f"(unsigned int)__float_as_int(__fmul_rn({fa}, {fb}))",
            "ftz_probe_r4_mul_rn"),
        "fminf/fmaxf": uint_kernel(
            f"(unsigned int)__float_as_int(fminf({fa}, {fb}))",
            "ftz_probe_r4_min"),
        "float compare": uint_kernel(
            f"(unsigned int)__float_as_int({fa} != {fb} ? 1.0f : 0.0f)",
            "ftz_probe_r4_compare"),
        "__double2float_rn": double_kernel(
            "(unsigned int)__float_as_int(__double2float_rn(x + y))",
            "ftz_probe_r4_d2f"),
        "(float) cast": double_kernel(
            "(unsigned int)__float_as_int((float)(x + y))",
            "ftz_probe_r4_cast"),
    }


# ---- execution ------------------------------------------------------------

def _launch_module(cp, route, kernel_name: str, op: str) -> list[int]:
    count = len(INPUT_ROWS)
    out = cp.zeros(count, dtype=cp.uint32)
    function = route["module"].get_function(kernel_name)
    threads = 32
    blocks = (count + threads - 1) // threads
    if op == "convert":
        ad = cp.asarray(np.array([row["d"] for row in INPUT_ROWS],
                                 dtype=np.float64))
        function((blocks,), (threads,), (ad, out, np.int32(count)))
    else:
        au = cp.asarray(np.array([_f32_bits(row["a"]) for row in INPUT_ROWS],
                                 dtype=np.uint32))
        bu = cp.asarray(np.array([_f32_bits(row["b"]) for row in INPUT_ROWS],
                                 dtype=np.uint32))
        function((blocks,), (threads,), (au, bu, out, np.int32(count)))
    cp.cuda.runtime.deviceSynchronize()
    return [int(value) for value in out.get()]


def _launch_reduction(cp, kernel, op: str) -> list[int]:
    """Drive the reduction one input at a time so it acts elementwise."""
    results = []
    for row in INPUT_ROWS:
        if op == "convert":
            x = cp.asarray(np.array([row["d"]], dtype=np.float64))
            y = cp.zeros(1, dtype=cp.float64)
        else:
            x = cp.asarray(np.array([_f32_bits(row["a"])], dtype=np.uint32))
            y = cp.asarray(np.array([_f32_bits(row["b"])], dtype=np.uint32))
        results.append(int(kernel(x, y)))
    cp.cuda.runtime.deviceSynchronize()
    return results


def run_probe(inventory: dict) -> tuple[list[dict], dict]:
    """Return (bit rows, compile-path capture) for one full pass."""
    import cupy as cp

    with NvrtcCapture() as capture:
        routes = build_routes(inventory, capture)
        rows: list[dict] = []
        control_bits: dict[tuple[str, str], list[int]] = {}
        for route in routes.values():
            for mechanism, op in MECHANISMS:
                kernel = route["kernels"].get(mechanism)
                if kernel is None:
                    continue
                capture.label = route["id"]
                if route["kind"] == "reduction":
                    device = _launch_reduction(cp, kernel, op)
                else:
                    device = _launch_module(cp, route, kernel, op)
                capture.label = None
                if route["id"] == "R1-ftztrue":
                    control_bits[(mechanism, op)] = device
                for row, bits in zip(INPUT_ROWS, device):
                    rows.append({
                        "route": route["id"],
                        "mechanism": mechanism,
                        "op": op,
                        "input_id": row["id"],
                        "kernel": kernel if isinstance(kernel, str)
                                  else kernel.name,
                        "device_bits": bits,
                        "ieee_reference_bits": _reference(
                            op, row, flushing=False),
                        "flushed_reference_bits": _reference(
                            op, row, flushing=True),
                    })

    for row in rows:
        control = control_bits.get((row["mechanism"], row["op"]))
        index = [item["id"] for item in INPUT_ROWS].index(row["input_id"])
        row["control_bits"] = control[index] if control is not None else None
    return rows, {
        "routes": {rid: {k: v for k, v in route.items()
                         if k in ("id", "label", "constructor", "site",
                                  "site_options", "arm_options")}
                   for rid, route in routes.items()},
        "nvrtc": {rid: [{k: v for k, v in record.items() if k != "object"}
                        for record in records]
                  for rid, records in capture.records.items()},
        "objects": {rid: [(record["object"], record["object_format"])
                          for record in records]
                    for rid, records in capture.records.items()},
    }


def mirror_ptx(meta: dict) -> dict:
    """Recompile each captured route to PTX so the text is readable.

    On this toolkit the shipped path targets ``-arch=sm_*`` and NVRTC hands
    back a cubin, so there is no PTX on the shipped path to commit.  These
    artifacts carry the SAME effective flags the production compile received
    -- CuPy's appended ``-ftz`` included -- with only the target changed, and
    they are labelled ``mirror`` for exactly that reason.
    """
    from cupy.cuda import compiler as cc
    source = _probe_source()
    arch = cc._get_arch()
    original = cc._get_arch_for_options_for_nvrtc
    out: dict[str, dict] = {}
    try:
        cc._get_arch_for_options_for_nvrtc = (
            lambda arch_arg=None: (f"-arch=compute_{arch_arg or arch}", "ptx"))
        for route_id, records in meta["nvrtc"].items():
            flags = [item for item in records[0]["effective_flags"]
                     if not str(item).startswith("-arch=")]
            if route_id == "R4":
                out[route_id] = {
                    "available": False,
                    "reason": "the CuPy-generated reduction source is built "
                              "inside CuPy and is not reproducible from this "
                              "tool without re-entering its kernel cache"}
                continue
            blob, _mapping = cc.compile_using_nvrtc(
                source, tuple(flags), None, "ftz_probe.cu")
            if isinstance(blob, bytes):
                blob = blob.decode("utf-8", "replace")
            out[route_id] = {"available": True, "text": blob,
                             "effective_flags": flags}
    finally:
        cc._get_arch_for_options_for_nvrtc = original
    return out


# ---- cells and serialization ---------------------------------------------

def cells(rows: list[dict], routes: dict) -> list[dict]:
    """Every (route, mechanism) pair, present or explicitly not-applicable."""
    out = []
    for route_id, route in routes.items():
        for mechanism, op in MECHANISMS:
            subset = [row for row in rows
                      if row["route"] == route_id
                      and row["mechanism"] == mechanism]
            record = {
                "route": route_id,
                "mechanism": mechanism,
                "verdict": classify(subset),
                "input_count": len(subset),
            }
            if not subset:
                record["not_applicable_reason"] = route.get(
                    "not_applicable", {}).get(
                        mechanism, "route does not express this mechanism")
            out.append(record)
    return out


def flag_sensitivity(rows: list[dict]) -> dict:
    """Per-arm bit-table digests, and whether any two arms differ.

    If every arm produced the same words, no compile-side statement in this
    receipt can mean anything: the pipeline would be insensitive to the
    options, and the control arm would be measuring nothing.  The answer is
    read off the digests, not asserted.
    """
    digests: dict[str, str] = {}
    for route in sorted({row["route"] for row in rows}):
        subset = [row for row in rows if row["route"] == route]
        payload = "".join(
            f"{row['mechanism']}|{row['input_id']}|{row['device_bits']:08x};"
            for row in subset)
        digests[route] = sha256_text(payload)
    distinct = sorted(set(digests.values()))
    return {
        "arm_bit_table_sha256": digests,
        "distinct_arm_bit_tables": len(distinct),
        "arms_differ": len(distinct) > 1,
        "basis": "sha256 over each arm's (mechanism, input, device word) "
                 "sequence; the mechanism set differs between arms that "
                 "cannot express a mechanism, which is itself a difference",
    }


def bitpatterns_csv(rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([
        "route", "mechanism", "input_id", "kernel", "device_bits",
        "ieee_reference_bits", "flushed_reference_bits", "control_bits"])
    for row in rows:
        writer.writerow([
            row["route"], row["mechanism"], row["input_id"], row["kernel"],
            f"0x{row['device_bits']:08x}",
            f"0x{row['ieee_reference_bits']:08x}",
            f"0x{row['flushed_reference_bits']:08x}",
            "" if row["control_bits"] is None
            else f"0x{row['control_bits']:08x}"])
    return buffer.getvalue()


#: A CUDA device UUID is 16 bytes.  ``cudaUUID_t`` is ``char bytes[16]``
#: with no terminator, so its length is the array bound and nothing else.
CUDA_UUID_BYTES = 16


def device_uuid_hex(raw):
    """Hex of the 16 bytes that are the UUID, and of no other bytes.

    The buffer CuPy hands back for ``getDeviceProperties()['uuid']`` is
    not bounded by that array.  The receipt this probe wrote carried 19
    bytes for it -- 38 hex characters -- because the conversion stops at
    the first zero byte rather than at the array end, and ``cudaDeviceProp``
    places ``luid`` immediately after ``uuid``, so the read ran three bytes
    into the neighbour before finding a zero there.

    Those three bytes are not device identity.  ``luid`` is re-issued when
    Windows re-enumerates the adapter, so a receipt regenerated on the same
    physical card could differ in this field -- churn that reads exactly
    like the probe having moved to a different GPU, which is the one thing
    the field exists to tell you.  Bounding the read to the defined length
    makes the field a property of the card again.

    A UUID whose own bytes contain a zero is truncated by that same
    conversion before this function ever sees it.  Such a value is short
    but still stable, since the card's UUID does not change; only the
    over-read was unstable, and only the over-read is cut here.
    """
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)[:CUDA_UUID_BYTES].hex()
    return raw


def device_identity() -> dict:
    import cupy as cp
    from cupy.cuda import runtime
    device = cp.cuda.Device()
    properties = runtime.getDeviceProperties(device.id)
    return {
        "name": properties["name"].decode()
        if isinstance(properties["name"], bytes) else str(properties["name"]),
        "uuid": device_uuid_hex(properties.get("uuid")),
        "pci_bus_id": runtime.deviceGetPCIBusId(device.id),
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "cuda_driver_version": runtime.driverGetVersion(),
        "cuda_runtime_version": runtime.runtimeGetVersion(),
        "nvrtc_version": ".".join(
            str(part) for part in _nvrtc_version()),
        "cupy_version": __import__("cupy").__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _nvrtc_version() -> tuple:
    from cupy.cuda import nvrtc
    return tuple(nvrtc.getVersion())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_receipt(rows_a: list[dict], rows_b: list[dict], meta: dict,
                  artifacts: dict) -> dict:
    table_a = bitpatterns_csv(rows_a)
    table_b = bitpatterns_csv(rows_b)
    return {
        "schema": SCHEMA_ID,
        "generator": "tools/ftz_receipt/probe.py",
        "device": device_identity(),
        "cupy_ftz_append_site": ftz_append_site(),
        "effective_option_capture": {
            "method": "wrap cupy.cuda.compiler._compile_using_nvrtc_no_warning"
                      " for the duration of the probe",
            "cache_mode": "CUPY_CACHE_IN_MEMORY=" + os.environ.get(
                "CUPY_CACHE_IN_MEMORY", ""),
            "per_route": meta["nvrtc"],
        },
        "routes": meta["routes"],
        "mechanisms": [name for name, _ in MECHANISMS],
        "inputs": [{"id": row["id"],
                    "a_bits": f"0x{_f32_bits(row['a']):08x}",
                    "b_bits": f"0x{_f32_bits(row['b']):08x}",
                    "d": repr(row["d"])} for row in INPUT_ROWS],
        "verdict_vocabulary": list(VERDICTS),
        "cells": cells(rows_a, meta["routes"]),
        "flag_sensitivity": flag_sensitivity(rows_a),
        "dual_run": {
            "runs": 2,
            "byte_identical": table_a == table_b,
            "bit_table_sha256": [sha256_text(table_a), sha256_text(table_b)],
        },
        "artifacts": artifacts,
    }


def write_receipt(out_dir: Path, inventory: dict) -> dict:
    rows_a, meta = run_probe(inventory)
    rows_b, _meta_b = run_probe(inventory)
    table_a = bitpatterns_csv(rows_a)
    table_b = bitpatterns_csv(rows_b)
    if table_a != table_b:
        raise SystemExit(
            "the two probe passes disagree; on a card without ECC a "
            "divergent dual run is the corruption detector, not a nuisance")

    mirrors = mirror_ptx(meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bitpatterns.csv").write_text(table_a, encoding="utf-8",
                                             newline="\n")
    artifacts = {"bitpatterns.csv": {
        "sha256": sha256_text(table_a),
        "provenance": "shipped-path",
        "description": "device bit table, both passes byte-identical"}}
    cubin_dir = out_dir / "cubin"
    cubin_dir.mkdir(parents=True, exist_ok=True)
    for route_id, blobs in sorted(meta["objects"].items()):
        stem = route_id.lower().replace("-", "_")
        for index, (blob, fmt) in enumerate(blobs):
            name = (f"{stem}.{fmt}" if len(blobs) == 1
                    else f"{stem}_{index}.{fmt}")
            (cubin_dir / name).write_bytes(blob)
            artifacts[f"cubin/{name}"] = {
                "sha256": hashlib.sha256(blob).hexdigest(),
                "format": fmt,
                "provenance": "shipped-path",
                "description": f"the compiled object NVRTC returned during "
                               f"route {route_id}'s own compile"}
    for route_id in meta["routes"]:
        if route_id not in meta["objects"]:
            artifacts[f"cubin/{route_id.lower().replace('-', '_')}"] = {
                "provenance": "shipped-path",
                "shares_module_with": "R1",
                "reason": "this route changes how the arithmetic is written, "
                          "not how it is compiled; its kernels live in the "
                          "module R1 compiled and are covered by that "
                          "route's object"}

    ptx_dir = out_dir / "ptx"
    ptx_dir.mkdir(parents=True, exist_ok=True)
    for route_id, record in sorted(mirrors.items()):
        stem = route_id.lower().replace("-", "_")
        if not record["available"]:
            artifacts[f"ptx/{stem}.ptx"] = {
                "provenance": "unavailable", "reason": record["reason"]}
            continue
        text = record["text"].replace("\x00", "")
        (ptx_dir / f"{stem}.ptx").write_text(text, encoding="utf-8",
                                             newline="\n")
        artifacts[f"ptx/{stem}.ptx"] = {
            "sha256": sha256_text(text),
            "provenance": "mirror",
            "mirror_of": route_id,
            "description": "same source and same effective flags as the "
                           "shipped compile, with CuPy's cubin target "
                           "replaced by a PTX target so the instruction "
                           "modifiers are readable",
            "ftz_modifiers": ptx_ftz_modifiers(text)}
    artifacts["sass"] = sass.collect(out_dir, artifacts)

    receipt = build_receipt(rows_a, rows_b, meta, artifacts)
    (out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else ri.repo_root()
    inventory = ri.load_inventory(root)
    out_dir = (Path(args.out) if args.out
               else root / "tools" / "ftz_receipt" / "receipt")
    receipt = write_receipt(out_dir, inventory)
    for cell in receipt["cells"]:
        print(f"{cell['route']:<12} {cell['mechanism']:<26} "
              f"{cell['verdict']}")
    print(f"dual run byte-identical: {receipt['dual_run']['byte_identical']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
