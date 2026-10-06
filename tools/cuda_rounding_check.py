"""Find architecture-dependent floating instruction choices in CUDA sources.

This is a compiler diagnostic, not a proof of output identity. It compares
multiply, add, subtract, division and fused multiply-add instructions at each
source location in each entry point. Equal total FMA counts are insufficient:
moving a contraction to another source line still fails the comparison.

The source inventory also names Python CUDA literals and compiler construction
sites. Generated units can be supplied as captures of the actual compiler
inputs. An unresolved template is reported explicitly rather than counted as a
compiled source. PTX differs legitimately when libdevice inlining or loop
unrolling changes, so reported differences require source review or word replay.

Run with CUDA hidden. NVRTC compilation opens no GPU::

    CUDA_VISIBLE_DEVICES= python -m tools.cuda_rounding_check --json out.json
    CUDA_VISIBLE_DEVICES= python -m tools.cuda_rounding_check --unit smag2d

The default command fails on compile errors and floating instruction deltas.
Use --report-only for a diagnostic sweep that records unresolved differences.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import shutil
from typing import Iterable

from tools.literal_division_census import Unit, compile_ptx, production_units
from tools.cuda_rounding_dataflow import ptx_dataflow_signatures, dataflow_differences
from tools.cuda_rounding_native import compile_sass, sass_floating_signatures, sass_dataflow_signatures

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHITECTURES = (120, 100, 89, 80, 90)
_CUDA_DECLARATION = re.compile(r"\b__(?:global|device)__\b")
_ENTRY = re.compile(r"\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([\w$]+)\s*\(")
_LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)(.*)$")
_INLINE = re.compile(r"inlined_at\s+(\d+)\s+(\d+)\s+(\d+)")
_FLOAT_OP = re.compile(
    r"^\s*(?:@\S+\s+)?((?:fma|mad|mul|add|sub|div|rcp|sqrt)"
    r"(?:\.[\w]+)*\.f(?:32|64))\s")
_PINNED = re.compile(r"\b__(?:f|d)(?:add|sub|mul|div|sqrt|ma[f]?)_rn\s*\(")
_CONSTRUCTORS = {"RawKernel", "RawModule", "compile_using_nvrtc",
                 "compile_program", "createProgram", "ElementwiseKernel",
                 "ReductionKernel"}


def source_statement_lines(source: str) -> dict[int, int]:
    """Group line-info locations belonging to the same CUDA statement.

    NVRTC can attach a ternary's multiply to its first or second source line
    depending on the target. That debug-location movement is not a rounding
    difference. Statement boundaries keep independent multiply-add expressions
    distinct while making a multiline statement one comparison location.
    """
    clean = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"',
                   lambda match: "".join("\n" if ch == "\n" else " " for ch in match[0]),
                   source, flags=re.S)
    mapping, start = {}, None
    for number, line in enumerate(clean.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for character in line:
            if character.isspace():
                continue
            if start is None:
                start = number
            mapping.setdefault(number, start)
            if character in ";{}":
                start = None
    return mapping


@dataclass(frozen=True, order=True)
class Location:
    function: str
    file_number: int
    line: int
    column: int
    inline_chain: tuple[tuple[int, int, int], ...]


def floating_signatures(ptx: str, source: str | None = None) -> dict[Location, Counter]:
    """Floating arithmetic by function and full source/inlining location.

    Register numbers and instruction scheduling are deliberately ignored.
    Opcode, precision, rounding and FTZ are retained. Instructions lacking
    source locations are kept at line zero, so missing line information cannot
    silently produce an empty passing result.
    """
    signatures: dict[Location, Counter] = {}
    function = "<global>"
    loc = (0, 0, 0, ())
    statement_lines = source_statement_lines(source) if source is not None else {}
    for text in ptx.splitlines():
        match = _ENTRY.search(text)
        if match:
            function = match.group(1)
            loc = (0, 0, 0, ())
        match = _LOC.match(text)
        if match:
            file_number, line, column = int(match[1]), int(match[2]), int(match[3])
            chain = tuple(tuple(int(x) for x in item)
                          for item in _INLINE.findall(match[4]))
            if source is not None:
                # Columns are debug attribution within one statement, not
                # distinct arithmetic sites. Preserve call-site statements.
                if file_number == 1:
                    line = statement_lines.get(line, line)
                    column = 0
                chain = tuple((file, statement_lines.get(line, line), 0)
                              if file == 1 else (file, line, column)
                              for file, line, column in chain)
            loc = (file_number, line, column, chain)
            continue
        match = _FLOAT_OP.match(text)
        if match:
            key = Location(function, *loc)
            signatures.setdefault(key, Counter())[match[1]] += 1
    return signatures


def signature_differences(left: dict[Location, Counter],
                          right: dict[Location, Counter]) -> list[dict]:
    """Report local deltas, including zero-sided and unlocated arithmetic."""
    differences = []
    for site in sorted(set(left) | set(right)):
        before, after = left.get(site, Counter()), right.get(site, Counter())
        if before != after:
            differences.append({
                "function": site.function, "file_number": site.file_number,
                "assembled_line": site.line, "column": site.column,
                "inline_chain": [list(item) for item in site.inline_chain],
                "left": dict(sorted(before.items())),
                "right": dict(sorted(after.items())),
                "contraction_changed": any(
                    before[key] != after[key]
                    for key in set(before) | set(after)
                    if key.startswith(("fma.", "mad.", "FFMA", "DFMA"))),
            })
    return differences


def _literal_value(node: ast.AST, bindings: dict) -> object:
    if isinstance(node, ast.Name):
        return bindings[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_literal_value(value, bindings) for value in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_value(node.left, bindings) + _literal_value(node.right, bindings)
    if isinstance(node, ast.JoinedStr):
        result = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                result.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                item = _literal_value(value.value, bindings)
                if value.conversion == 114:
                    item = repr(item)
                elif value.conversion == 115:
                    item = str(item)
                specification = (_literal_value(value.format_spec, bindings)
                                 if value.format_spec is not None else "")
                result.append(format(item, specification))
            else:
                raise ValueError("source is assembled at run time")
        return "".join(result)
    raise ValueError("source is assembled at run time")


def inventory(root: Path) -> dict:
    """Name every CUDA file, literal and compile site without importing CuPy."""
    package = Path(root) / "gpuwm"
    cuda_files, literals, constructors, compiler_files = [], [], [], []
    for path in sorted(package.rglob("*")):
        if path.suffix in (".cu", ".cuh"):
            source = path.read_text(encoding="utf-8")
            cuda_files.append({"file": path.relative_to(root).as_posix(),
                               "sha256": hashlib.sha256(source.encode()).hexdigest(),
                               "lines": len(source.splitlines()),
                               "pinned_intrinsics": len(_PINNED.findall(source))})
    for path in sorted(package.rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        label = path.relative_to(root).as_posix()
        if (_CUDA_DECLARATION.search(source)
                or any(re.search(r"\b" + re.escape(name) + r"\s*\(", source)
                       for name in _CONSTRUCTORS)):
            compiler_files.append({"file": label,
                                   "sha256": hashlib.sha256(source.encode()).hexdigest()})
        bindings = {}
        # Only module assignments are safe to resolve statically. Reusing a
        # local name from another function would silently invent a source.
        for statement in tree.body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = (statement.targets if isinstance(statement, ast.Assign)
                           else [statement.target])
                try:
                    value = _literal_value(statement.value, bindings)
                except (KeyError, ValueError, TypeError):
                    continue
                for target in targets:
                    if isinstance(target, ast.Name):
                        bindings[target.id] = value
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and _CUDA_DECLARATION.search(node.value)):
                literals.append({"file": label, "line": node.lineno,
                                 "sha256": hashlib.sha256(node.value.encode()).hexdigest(),
                                 "pinned_intrinsics": len(_PINNED.findall(node.value)),
                                 "source": node.value})
            if not isinstance(node, ast.Call):
                continue
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name) else "")
            if name not in _CONSTRUCTORS:
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            source_node = keywords.get("code", keywords.get("source"))
            if source_node is None and node.args:
                source_node = node.args[0]
            option_node = keywords.get("options")
            if option_node is None and name in ("compile_program", "compile_using_nvrtc"):
                option_node = node.args[1] if len(node.args) > 1 else None
            record = {"file": label, "line": node.lineno, "constructor": name,
                      "source_expression": ast.unparse(source_node) if source_node else "",
                      "status": "unresolved",
                      "call_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}
            scopes = [scope for scope in ast.walk(tree)
                      if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                      and scope.lineno <= node.lineno <= scope.end_lineno]
            scope = min(scopes, key=lambda item: item.end_lineno - item.lineno) if scopes else None
            record["factory"] = scope.name if scope else "<module>"
            record["factory_sha256"] = hashlib.sha256(
                ast.dump(scope if scope else node, include_attributes=False).encode()).hexdigest()
            local_bindings = dict(bindings)
            if scope is not None:
                # Resolve local literal assignments in source order. Unknown
                # assignments remove a binding, preventing stale guesses.
                for statement in scope.body:
                    if statement.lineno >= node.lineno:
                        break
                    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                        continue
                    targets = (statement.targets if isinstance(statement, ast.Assign)
                               else [statement.target])
                    try:
                        value = _literal_value(statement.value, local_bindings)
                    except (KeyError, ValueError, TypeError):
                        for target in targets:
                            if isinstance(target, ast.Name):
                                local_bindings.pop(target.id, None)
                        continue
                    for target in targets:
                        if isinstance(target, ast.Name):
                            local_bindings[target.id] = value
            if name == "ElementwiseKernel" and len(node.args) >= 4:
                try:
                    record["elementwise"] = {
                        "in_params": _literal_value(node.args[0], local_bindings),
                        "out_params": _literal_value(node.args[1], local_bindings),
                        "operation": _literal_value(node.args[2], local_bindings),
                        "name": _literal_value(node.args[3], local_bindings),
                        "preamble": _literal_value(keywords["preamble"], local_bindings) if "preamble" in keywords else "",
                        "options": list(_literal_value(option_node, local_bindings) if option_node else ()),
                    }
                except (KeyError, ValueError, TypeError):
                    record.pop("elementwise", None)
            try:
                code = _literal_value(source_node, local_bindings)
                options = _literal_value(option_node, local_bindings) if option_node else ()
                if not isinstance(code, str) or not _CUDA_DECLARATION.search(code):
                    raise ValueError("not a complete CUDA literal")
                if not isinstance(options, tuple) or not all(isinstance(x, str) for x in options):
                    raise ValueError("options are assembled at run time")
                record.update(status="resolved_literal", source=code, options=list(options),
                              sha256=hashlib.sha256(code.encode()).hexdigest())
            except (KeyError, ValueError, TypeError):
                pass
            constructors.append(record)
    return {"cuda_files": cuda_files, "python_cuda_literals": literals,
            "python_compiler_files": compiler_files,
            "compile_sites": constructors,
            "counts": {"cuda_files": len(cuda_files), "python_cuda_literals": len(literals),
                       "python_compiler_files": len(compiler_files),
                       "compile_sites": len(constructors),
                       "resolved_literal_sites": sum(x["status"] == "resolved_literal"
                                                      for x in constructors)}}


def source_units(root: Path, *, captures: Path | None = None) -> tuple[list[Unit], dict]:
    """Existing exact-source factories, resolved literals and runtime captures.

    The inventory preserves unresolved constructors. A capture binds every
    source byte and effective option; it is required for dynamic template
    coverage. Captures are JSON containing {units:[{key,source,options}]}.
    """
    stock = production_units(root)
    # These files are components, not the sources their runtime compiles.
    # Their real factories are included by composed_units. Compiling a
    # fragment without its defines/libm creates false coverage and errors.
    fragments = {"p3", "rrtmg_lw", "rrtmg_sw", "terrain_drag", "urban_bep_bem",
                 "rrtmg_lw_chain", "rrtmg_lw_chain_coalesced", "rrtmg_lw_zbatched",
                 "rrtmg_lw_taugb02_10_11_12", "rrtmg_lw_taugb03_05",
                 "rrtmg_lw_taugb06_09", "rrtmg_lw_taugb13_16"}
    direct = {"rrtmg_legacy_prep", "rrtmg_legacy_adapter", "rrtmg_mcica_wrf", "ruc_spp"}
    stock = [unit for unit in stock
             if unit.key not in {"kernels:" + name for name in fragments | direct}]
    source_inventory = inventory(root)
    seen = {(unit.source, unit.options) for unit in stock}
    for record in source_inventory["compile_sites"]:
        if record["status"] != "resolved_literal":
            continue
        options = tuple(record["options"])
        # The literal RawModule and RawKernel route adds FTZ at runtime.
        if record["constructor"] in ("RawModule", "RawKernel"):
            options += ("-ftz=true", "--device-as-default-execution-space")
        identity = (record["source"], options)
        if identity in seen:
            continue
        key = f"literal:{record['file']}:{record['line']}"
        stock.append(Unit(key, record["source"], options,
                          [(1, record["file"], record["line"])]))
        seen.add(identity)
    if captures is not None:
        data = json.loads(Path(captures).read_text(encoding="utf-8"))
        source_inventory["captures"] = []
        for record in data["units"]:
            if not record["source"].strip():
                source_inventory["captures"].append({"key": record["key"], "status": "empty_compiler_probe"})
                continue
            options = tuple(record["options"])
            identity = (record["source"], options)
            if identity in seen:
                source_inventory["captures"].append({"key": record["key"], "status": "duplicate_source_options"})
                continue
            stock.append(Unit("captured:" + record["key"], record["source"], options))
            source_inventory["captures"].append({"key": record["key"], "status": "compiled_source",
                                                  "source_sha256": hashlib.sha256(record["source"].encode()).hexdigest()})
            seen.add(identity)
    return stock, source_inventory


def composed_units(root: Path) -> list[Unit]:
    """Runtime compositions that cannot be compiled as standalone CUDA files.

    These factories run on the CPU. Their imported package must be the tree
    being checked, so a nearby installed wheel cannot supply another source.
    """
    root = Path(root).resolve()

    def module(name):
        value = importlib.import_module("gpuwm.core." + name)
        if not Path(value.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"source factory {name} was imported outside the checked tree")
        return value

    from gpuwm.core.kernels import module_source
    kdir = root / "gpuwm" / "core" / "kernels"
    lw = module("rrtmg_lw")
    sw = module("rrtmg_sw")
    _, sw_defines = sw._pack_cuda_tables(sw.load_sw_tables())
    terrain = module("terrain_drag")
    bem = module("urban_bem")
    mosaic = module("noah_mosaic")
    nt = module("ntiedtke_fused")
    units = [
        Unit("composed:rrtmg_lw", lw._gpu_source(), ("-std=c++17", "--ftz=false")),
        Unit("composed:rrtmg_sw", sw_defines + (kdir / "rrtmg_sw.cu").read_text(encoding="utf-8"),
             ("-std=c++17", "--ftz=false")),
        Unit("composed:terrain_drag", terrain.module_source(), tuple(terrain.MODULE_OPTIONS)),
        Unit("composed:urban_bem", bem.module_source(), tuple(bem.MODULE_OPTIONS)),
        Unit("composed:noah_mosaic", module_source("noah_mosaic", kernel_dir=kdir),
             tuple(mosaic.MOSAIC_NVRTC_OPTIONS)),
        Unit("composed:noah_mosaic_ucm", mosaic.mosaic_ucm_source(),
             tuple(mosaic.MOSAIC_NVRTC_OPTIONS)),
        Unit("composed:ntiedtke_fused", nt.source_plan()[0], ("-std=c++17",)),
        Unit("composed:nest", module_source("nest", kernel_dir=kdir),
             ("-std=c++17", "-fmad=false")),
    ]
    for name in ("rrtmg_legacy_prep", "rrtmg_legacy_adapter", "rrtmg_mcica_wrf"):
        units.append(Unit("direct:" + name,
                          (kdir / f"{name}.cu").read_text(encoding="utf-8"),
                          ("-std=c++17", "--ftz=false")))
    ruc = module("ruc_spp")
    units.append(Unit("direct:ruc_spp", (kdir / "ruc_spp.cu").read_text(encoding="utf-8"),
                      tuple(ruc.MODULE_OPTIONS)))
    return units


def elementwise_operation_unit(key, kernel, options=()) -> Unit:
    """Compile an operation in a typed scalar entry without opening a device.

    This preserves the factory's floating expressions. It is an operation
    probe, not a capture of CuPy's indexing wrapper or reduction schedule.
    Exact generated compiler inputs come from capture_compiler_sources.
    """
    parameters, setup, stores = [], [], []
    for parameter in kernel.in_params + kernel.out_params:
        name, ctype = parameter.name, parameter.ctype
        if parameter.raw:
            parameters.append(("const " if parameter.is_const else "") + ctype + "* " + name)
        elif parameter.is_const:
            parameters.append(ctype + " " + name)
        else:
            parameters.append(ctype + "* output_" + name)
            setup.append(ctype + " " + name + ";")
            stores.append("output_" + name + "[i] = " + name + ";")
    source = ("typedef float T;\n" + kernel.preamble
              + '\nextern "C" __global__ void operation_probe('
              + ", ".join(parameters) + ") {\n"
              + "const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
              + "\n".join(setup) + "\n" + kernel.operation + "\n" + "\n".join(stores) + "\n}\n")
    return Unit("operation:" + key, source, tuple(options) + ("-std=c++17", "-ftz=true"))


def operation_units(root: Path) -> list[Unit]:
    """All arithmetic variants of current ElementwiseKernel factories."""
    import itertools
    def module(name):
        value = importlib.import_module("gpuwm.core." + name)
        if not Path(value.__file__).resolve().is_relative_to(Path(root).resolve()):
            raise RuntimeError(f"operation factory {name} is outside the checked tree")
        return value
    dycore, moist, morrison = (module(name) for name in ("dycore", "moist", "morrison"))
    units = [elementwise_operation_unit("morrison", morrison._prepare_fields)]
    import cupy as cp
    for record in inventory(Path(root))["compile_sites"]:
        if "elementwise" not in record:
            continue
        values = record["elementwise"]
        if "_ind" in values["operation"]:
            # Exact CuPy indexers are handled by run-source captures.
            continue
        kernel = cp.ElementwiseKernel(values["in_params"], values["out_params"], values["operation"],
                                      values["name"], preamble=values["preamble"],
                                      options=tuple(values["options"]))
        unit = elementwise_operation_unit(f"{record['file']}:{record['factory']}[{record['call_sha256'][:10]}]", kernel,
                                          tuple(values["options"]))
        if unit.source not in {item.source for item in units}:
            units.append(unit)
    for mapped in (False, True):
        units.append(elementwise_operation_unit(f"omega[map={int(mapped)}]",
                                                dycore._omega_column_kernel(mapped), ("-fmad=false",)))
        for reciprocal in (False, True):
            units.append(elementwise_operation_unit(
                f"momentum[map={int(mapped)},reciprocal={int(reciprocal)}]",
                dycore._couple_momentum_kernel(mapped, reciprocal), ("-fmad=false",)))
    for flags in itertools.product((False, True), repeat=4):
        units.append(elementwise_operation_unit("scalar_update[" + ",".join(str(int(x)) for x in flags) + "]",
                                                moist._update_scalar_kernel(*flags), ("-fmad=false",)))
    # Validation factories generate integer flags. Their predicate and size
    # specializations are still inventoried and target-compiled.
    mynn, ruc = module("mynn_pbl_gpu"), module("ruc_gpu")
    for count in (1, 16):
        for predicate in mynn._VALIDATION_PREDICATES:
            kernel = mynn._validation_batch_kernel.__wrapped__(predicate, count)
            units.append(Unit(f"factory:mynn_validation[{predicate},{count}]", kernel.code,
                              tuple(kernel.options) + ("-ftz=true",)))
        kernel = ruc._validation_scan_kernel.__wrapped__(count)
        units.append(Unit(f"factory:ruc_validation[{count}]", kernel.code,
                          tuple(kernel.options) + ("-ftz=true",)))
    return units


@contextmanager
def capture_compiler_sources(destination: Path, compiler=None):
    """Capture the actual CUDA inputs of a real run, including cache hits.

    Arithmetic options and sources are passed through unchanged. Cache
    requests carry the RawKernel/RawModule route's implicit FTZ. The final
    NVRTC program hook also records direct compiles and generated CuPy kernels.
    A capture describes the exercised configuration, not every engine option.
    The hooks are removed on exit, including when the run raises an exception.
    """
    if compiler is None:
        from cupy.cuda import compiler
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_cache = compiler._compile_with_cache_cuda
    original_program = compiler._NVRTCProgram.compile
    units, lock = {}, threading.Lock()

    def record(source, options, route):
        options = tuple(str(option) for option in options)
        if not any(option.lstrip("-").startswith("ftz=") for option in options):
            options += (("-ftz=true",) if route == "cache_request" else ("-ftz=false",))
        if route == "cache_request" and "--device-as-default-execution-space" not in options:
            options += ("--device-as-default-execution-space",)
        # One actual source/options pair is one unit. Keep architecture in
        # the capture for evidence; the target comparator replaces it later.
        identity = hashlib.sha256((source + "\0" + repr(options)).encode()).hexdigest()
        with lock:
            units[identity] = {"key": identity[:16], "source": source,
                               "options": list(options), "route": route,
                               "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
            destination.write_text(json.dumps({"units": list(units.values())}, indent=1) + "\n",
                                   encoding="utf-8")

    @functools.wraps(original_cache)
    def cached(source, options, *args, **kwargs):
        record(source, options, "cache_request")
        return original_cache(source, options, *args, **kwargs)

    @functools.wraps(original_program)
    def program(self, options=(), log_stream=None):
        record(self.src, options, "nvrtc_compile")
        return original_program(self, options, log_stream)

    compiler._compile_with_cache_cuda = cached
    compiler._NVRTCProgram.compile = program
    try:
        yield
    finally:
        compiler._compile_with_cache_cuda = original_cache
        compiler._NVRTCProgram.compile = original_program


def _options(unit: Unit, arch: int) -> tuple[str, ...]:
    options = tuple(option for option in unit.options
                    if not option.startswith(("-arch=", "--gpu-architecture=")))
    if not any(option.lstrip("-").startswith("ftz=") for option in options):
        options += ("-ftz=true",)
    if "--device-as-default-execution-space" not in options:
        options += ("--device-as-default-execution-space",)
    return options + ("-lineinfo", f"-arch=compute_{arch}")


def compare_unit(unit: Unit, arches: Iterable[int] = DEFAULT_ARCHITECTURES,
                 *, keep_ptx: Path | None = None, nvrtc_library: Path | None = None,
                 nvdisasm: Path | None = None, native_only: bool = False,
                 native_temp_ledger: Path | None = None) -> dict:
    """Compile and compare the same source/options on every requested target."""
    arches = tuple(arches)
    if len(arches) < 2:
        raise ValueError("rounding comparison requires at least two targets")
    if native_only and (nvrtc_library is None or nvdisasm is None):
        raise ValueError("native-only comparison requires the NVRTC library and nvdisasm")
    signatures, dataflow, native, native_dataflow = {}, {}, {}, {}
    compiled, errors, native_errors = {}, {}, {}
    for arch in arches:
        try:
            ptx = "" if native_only else compile_ptx(unit.source, _options(unit, arch))
            signatures[arch] = {} if native_only else floating_signatures(ptx, unit.source)
            dataflow[arch] = {} if native_only else ptx_dataflow_signatures(ptx, source_statement_lines(unit.source))
            compiled[str(arch)] = {"ptx_sha256": None if native_only else hashlib.sha256(ptx.encode()).hexdigest(),
                                  "ptx_status": "not_requested" if native_only else "compiled",
                                  "floating_sites": len(signatures[arch]),
                                  "options": list(_options(unit, arch))}
            if keep_ptx is not None:
                path = Path(keep_ptx)
                path.mkdir(parents=True, exist_ok=True)
                key = re.sub(r"[^a-zA-Z0-9_.-]", "_", unit.key)
                if not native_only:
                    (path / f"{key}.compute_{arch}.ptx").write_text(ptx, encoding="utf-8")
            if nvrtc_library is not None and nvdisasm is not None:
                try:
                    native_options = tuple(option.replace(f"-arch=compute_{arch}", f"-arch=sm_{arch}")
                                           for option in _options(unit, arch))
                    sass, binary = compile_sass(unit.source, native_options, nvrtc_library=nvrtc_library,
                                               nvdisasm=nvdisasm, return_metadata=True,
                                               deletion_ledger=native_temp_ledger)
                    native[arch] = {Location(*key): value for key, value in
                                    sass_floating_signatures(sass, source_statement_lines(unit.source)).items()}
                    native_dataflow[arch] = sass_dataflow_signatures(sass, source_statement_lines(unit.source))
                    compiled[str(arch)]["sass_sha256"] = hashlib.sha256(sass.encode()).hexdigest()
                    compiled[str(arch)]["native_options"] = list(native_options)
                    compiled[str(arch)].update(binary)
                    if keep_ptx is not None:
                        (path / f"{key}.sm_{arch}.sass").write_text(sass, encoding="utf-8")
                except Exception as exc:
                    native_errors[str(arch)] = str(exc)[:2000]
        except Exception as exc:
            errors[str(arch)] = str(exc)[:2000]
    reference = 120 if 120 in signatures else next(iter(signatures), None)
    differences, expression_differences, native_differences, native_expression_differences = {}, {}, {}, {}
    for arch, signature in signatures.items():
        if arch == reference:
            continue
        rows = signature_differences(signatures[reference], signature)
        for row in rows:
            label, line = unit.locate(row["assembled_line"])
            row["file"], row["line"] = label, line
            source_lines = unit.source.splitlines()
            row["source"] = (source_lines[row["assembled_line"] - 1]
                             if 0 < row["assembled_line"] <= len(source_lines) else "")
            row["explicit_rounding_on_line"] = bool(_PINNED.search(row["source"]))
        differences[str(arch)] = rows
        expression_differences[str(arch)] = dataflow_differences(dataflow[reference], dataflow[arch])
        if reference in native and arch in native:
            native_differences[str(arch)] = signature_differences(native[reference], native[arch])
            native_expression_differences[str(arch)] = dataflow_differences(
                native_dataflow[reference], native_dataflow[arch])
    return {"unit": unit.key, "source_sha256": hashlib.sha256(unit.source.encode()).hexdigest(),
            "reference_arch": reference, "compiled": compiled, "errors": errors,
            "differences": differences, "dataflow_differences": expression_differences,
            "native_differences": native_differences, "native_errors": native_errors,
            "native_dataflow_differences": native_expression_differences,
            "dataflow_limits": "Register-definition DAGs do not solve branch joins or memory aliases; changed DAGs require review/replay.",
            "opcode_status": "different" if any(differences.values()) else "same_signatures",
            "status": ("compile_failed" if errors or native_errors else "different"
                       if (any(differences.values()) or any(expression_differences.values())
                           or any(native_differences.values()) or any(native_expression_differences.values()))
                       else "same_signatures")}


def audit_inventory(data: dict) -> dict:
    """Source-bound regression inventory, excluding unstable line numbers.

    The compile factory's AST and its module bytes bind generated arithmetic
    too. A changed or added source cannot silently inherit an old exception.
    """
    result = {}
    for category in ("cuda_files", "python_cuda_literals", "python_compiler_files", "compile_sites"):
        rows = []
        for row in data[category]:
            if category == "compile_sites":
                rows.append({key: row[key] for key in (
                    "file", "constructor", "factory", "call_sha256", "factory_sha256", "status")})
            else:
                rows.append({"file": row["file"], "sha256": row["sha256"]})
        result[category] = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
    return result


def audit_record(row: dict) -> dict:
    """Compact signatures retain diagnostics without calling them identity.

    Native IEEE division implementations and source attribution can differ
    despite fixed arithmetic. Unresolved DAG/CFG differences stay explicit.
    They are frozen regression evidence, not reviewed numerical exceptions.
    """
    record = {"source_sha256": row["source_sha256"], "errors": row["errors"],
              "native_errors": row["native_errors"], "assessment": "static_diagnostic_only"}
    for key in ("differences", "dataflow_differences", "native_differences", "native_dataflow_differences"):
        value = row.get(key, {})
        record[key] = {
            "sha256": hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "sites": {arch: len(sites) for arch, sites in value.items()},
        }
    record["options"] = {arch: details["options"] for arch, details in row["compiled"].items()}
    record["coverage"] = "compiled" if not (row["errors"] or row["native_errors"]) else "compile_failed"
    return record


def baseline_manifest(data: dict) -> dict:
    return {"schema": 1,
            "meaning": "Frozen source and target-compiler diagnostics. This is not cross-card numerical certification.",
            "architectures": data["architectures"],
            "inventory": audit_inventory(data["inventory"]),
            "units": {row["unit"]: audit_record(row) for row in data["units"]}}


def inventory_changes(current: dict, baseline: dict) -> list[str]:
    """Changed sources fail before a compiler result can inherit a baseline."""
    differences = []
    for category in sorted(set(current) | set(baseline)):
        expected = {json.dumps(row, sort_keys=True) for row in baseline.get(category, [])}
        actual = {json.dumps(row, sort_keys=True) for row in current.get(category, [])}
        differences.extend(f"{category}: {row}" for row in sorted(actual ^ expected))
    return differences


def public_inventory(data: dict) -> dict:
    """Keep the report bounded by omitting duplicate source text."""
    return {key: ([{k: v for k, v in row.items() if k != "source"} for row in value]
                  if isinstance(value, list) else value)
            for key, value in data.items()}


def _compare_job(payload):
    unit, arches, options = payload
    return compare_unit(unit, arches, **options)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--compiler-info", action="store_true", help="report NVRTC targets without opening a device")
    parser.add_argument("--unit", action="append", default=[], help="unit key substring, repeatable")
    parser.add_argument("--arch", type=int, action="append", help="NVRTC target, repeatable")
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--composed", action="store_true", help="include CPU runtime source factories")
    parser.add_argument("--operations", action="store_true", help="include typed operation probes for ElementwiseKernel variants")
    parser.add_argument("--keep-ptx", type=Path)
    parser.add_argument("--nvrtc-library", type=Path, help="NVRTC shared library for native cubin compilation")
    parser.add_argument("--nvdisasm", type=Path, help="native disassembler; requires --nvrtc-library")
    parser.add_argument("--native-only", action="store_true", help="compile native cubins without repeating PTX diagnostics")
    parser.add_argument("--workers", type=int, default=1, help="bounded CPU compiler processes (default: one)")
    parser.add_argument("--native-temp-ledger", type=Path, help="append owned temporary cubin deletions with sizes and hashes")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--write-baseline", type=Path, help="save frozen diagnostics for review; never a numerical certification")
    parser.add_argument("--reuse-audit", type=Path,
                        help="reuse same-source/options/target receipts from this compiler installation")
    args = parser.parse_args(argv)
    if args.compiler_info:
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
            raise RuntimeError("hide CUDA devices before querying the CPU compiler")
        try:
            from cupy.cuda import nvrtc
            print(json.dumps({"version": list(nvrtc.getVersion()),
                              "supported_architectures": list(nvrtc.getSupportedArchs())}))
        except Exception as exc:
            print(json.dumps({"unavailable": str(exc)}))
        return 0
    if args.json is None:
        parser.error("--json is required for an audit")
    if bool(args.nvrtc_library) != bool(args.nvdisasm):
        parser.error("native compilation requires both --nvrtc-library and --nvdisasm")
    if args.workers < 1:
        parser.error("--workers must be positive and capped to the available CPU quota")
    if args.inventory_only:
        data = {"inventory": public_inventory(inventory(args.root)), "units": []}
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
            raise RuntimeError("hide CUDA devices before this CPU compiler sweep")
        units, source_inventory = source_units(args.root, captures=args.captures)
        # Fragment replacement is required, not an optional coverage mode.
        units.extend(composed_units(args.root))
        if args.operations:
            units.extend(operation_units(args.root))
        if args.unit:
            units = [unit for unit in units if any(pattern in unit.key for pattern in args.unit)]
            if not units:
                raise ValueError("requested unit filter matched no source")
        data = {"inventory": public_inventory(source_inventory), "units": [],
                "architectures": args.arch or list(DEFAULT_ARCHITECTURES),
                "interpretation": "Compiler signatures are diagnostics; equal signatures do not prove word identity."}
        options = {"keep_ptx": args.keep_ptx, "nvrtc_library": args.nvrtc_library,
                   "nvdisasm": args.nvdisasm, "native_only": args.native_only,
                   "native_temp_ledger": args.native_temp_ledger}
        reused = {}
        if args.reuse_audit:
            previous = json.loads(args.reuse_audit.read_text(encoding="utf-8"))
            if previous["architectures"] != data["architectures"]:
                raise ValueError("reused compiler receipts must have the same target sequence")
            for row in previous["units"]:
                if row["errors"] or row["native_errors"]:
                    continue
                reused[row["unit"]] = row

        def record(row):
            data["units"].append(row)
            comparison = row["native_differences"] if args.native_only else row["differences"]
            counts = {arch: len(sites) for arch, sites in comparison.items()}
            print(f"{row['unit']}: {row['status']} {counts}", flush=True)
            # A detached interrupted sweep retains every completed receipt.
            args.json.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")

        pending_units = []
        for unit in units:
            row = reused.get(unit.key)
            if (row is not None
                    and row["source_sha256"] == hashlib.sha256(unit.source.encode()).hexdigest()
                    and all(row["compiled"].get(str(arch), {}).get("options") == list(_options(unit, arch))
                            for arch in data["architectures"])
                    and (not args.nvdisasm or all("sass_sha256" in row["compiled"].get(str(arch), {})
                                                 for arch in data["architectures"]))):
                row = dict(row, receipt_reused=True)
                record(row)
            else:
                pending_units.append(unit)
        if args.workers == 1:
            for unit in pending_units:
                record(compare_unit(unit, data["architectures"], **options))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_compare_job, (unit, data["architectures"], options)) for unit in pending_units]
                for future in as_completed(futures):
                    record(future.result())
        data["units"].sort(key=lambda row: row["unit"])
    args.json.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    if args.write_baseline:
        args.write_baseline.write_text(json.dumps(baseline_manifest(data), indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    failed = any(row["status"] != "same_signatures" for row in data["units"])
    return 1 if failed and not args.report_only else 0


if __name__ == "__main__":
    sys.exit(main())
