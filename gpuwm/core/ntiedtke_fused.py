"""Build a column-local stage group from the standalone source bodies."""
from functools import lru_cache
import re


@lru_cache(maxsize=1)
def source_plan():
    from gpuwm.core.kernels import module_source
    from gpuwm.core.ntiedtke import (
        NT_CALL_ORDER, NT_STAGE_ALIASES, NT_STAGE_SIGNATURE,
        _GEOMETRY_TAIL, nt_resolve,
    )
    source = module_source("ntiedtke")
    stages = NT_CALL_ORDER[NT_CALL_ORDER.index("ntiedtke_cuascn"):
                           NT_CALL_ORDER.index("ntiedtke_cududvn")]
    masked = re.sub(r'/\*.*?\*/|//[^\n]*',
                    lambda match: " " * len(match[0]), source, flags=re.S)
    definitions = {}
    for match in re.finditer(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\(', source):
        name = match[1]
        start = match.end() - 1
        end = masked.index(")", start)
        body = masked.index("{", end)
        depth, stop = 1, body + 1
        while depth:
            depth += (masked[stop] == "{") - (masked[stop] == "}")
            stop += 1
        parameters = [p.strip() for p in masked[start + 1:end].split(",")]
        names = [p.split()[-1].lstrip("*") for p in parameters]
        if name in stages:
            if tuple(names) != NT_STAGE_SIGNATURE[name]:
                raise ValueError(f"{name} no longer matches its launch signature")
            definitions[name] = (source[start:stop], parameters, names)

    bindings, types, symbols = [], [], []
    helpers, calls = [], []
    for stage in stages:
        fragment, parameters, names = definitions[stage]
        helpers.append("__device__ __forceinline__ void fused_" + stage + fragment)
        arguments = []
        for parameter, name in zip(parameters, names):
            if name in _GEOMETRY_TAIL:
                arguments.append(name)
                continue
            pointer = "*" in parameter
            kind = "array" if pointer else "scalar"
            target = nt_resolve(NT_STAGE_ALIASES.get((stage, name), name)) if pointer else name
            binding = (kind, target)
            dtype = ("int" if re.search(r"\bint\b", parameter) else "float") + (" *" if pointer else "")
            if binding not in bindings:
                bindings.append(binding)
                types.append(dtype)
                symbols.append("v" + str(len(bindings) - 1))
            index = bindings.index(binding)
            if types[index] != dtype:
                raise ValueError(f"inconsistent fused parameter type for {target}")
            arguments.append(symbols[index])
        calls.append("    fused_" + stage + "(" + ", ".join(arguments) + ");")
    parameters = [dtype + " " + symbol for dtype, symbol in zip(types, symbols)]
    parameters += ["const int *llo3_mask", "int expect_tpb", "int expect_nblocks",
                   "int *geom_report", "int *order_report", "int *ticket"]
    ncol = symbols[bindings.index(("scalar", "ncol"))]
    wrapper = ('extern "C" __global__ void ntiedtke_fused(' + ", ".join(parameters) + ') {\n'
               '    const int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
               '    if (i >= ' + ncol + ') return;\n' + "\n".join(calls) + '\n}\n')
    # The closure uses this indexing macro from outside its original body.
    macro = re.search(r"^#define NTK[^\n]*", source, re.M)[0]
    generated = source + "\n" + macro + "\n\n" + "\n\n".join(helpers) + "\n\n" + wrapper
    return generated, tuple(bindings), tuple(stages)


@lru_cache(maxsize=1)
def load_fused_module():
    import cupy as cp
    from gpuwm.core.kernels import _compile_observed
    from gpuwm.certify.kernel_manifest import record_module

    source, _, _ = source_plan()
    options = ("-std=c++17",)
    module = cp.RawModule(code=source, options=options, name_expressions=None)
    key = "gpuwm.core.ntiedtke:fused"
    _compile_observed(module, key)
    record_module(key, source=source, options=options, module=module)
    return module


def run_fused(pipeline):
    _, bindings, stages = source_plan()
    cached = pipeline.w._stage_args.get(("ntiedtke_fused", 1))
    if cached is None:
        arguments, scalar_slots = [], []
        for index, (kind, name) in enumerate(bindings):
            if kind == "scalar":
                scalar_slots.append((index, name))
                arguments.append(pipeline.scalars[name])
            else:
                arguments.append(pipeline.w.bind(name, 1))
        cached = (arguments, tuple(scalar_slots))
        pipeline.w._stage_args[("ntiedtke_fused", 1)] = cached
    arguments, scalar_slots = cached
    for index, name in scalar_slots:
        arguments[index] = pipeline.scalars[name]
    if "ntiedtke_fused" not in pipeline.stages._functions:
        pipeline.stages._functions["ntiedtke_fused"] = load_fused_module().get_function("ntiedtke_fused")
    pipeline.stages.launch("ntiedtke_fused", tuple(arguments))
    return stages
