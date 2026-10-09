"""CPU checks for address-only batch source and launch parameter construction."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.ensemble.batch_kernel import (
    BatchKernelUnsupported, KernelSpec, PointerSpec, batch_grid,
    generate_batch_source, get_batch_kernel, pack_pointer_strides,
    validate_pointer_arguments,
    prepare_batch_kernel_launch, prepare_batch_source_launch,
)


SOURCE = '''typedef float real;
extern "C" __global__ void advance(const real* __restrict__ shared,
                                  real *state, int nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y + gridDim.y * blockIdx.z;
    if (i < nx) state[j * nx + i] = __fadd_rn(state[j * nx + i], shared[i]);
}
'''
SPEC = KernelSpec("sample", "advance", (
    PointerSpec("shared", "shared"), PointerSpec("state", "member"),
))


def test_n1_source_is_identical_and_has_no_descriptor():
    assert generate_batch_source(SOURCE, SPEC, 1).encode() == SOURCE.encode()


@pytest.mark.parametrize("bounds", ["256", "256, 4"])
def test_launch_bounds_retains_original_declaration_and_arithmetic(bounds):
    declaration = f"__global__ __launch_bounds__({bounds}) void"
    source = SOURCE.replace("__global__ void", declaration)
    generated = generate_batch_source(source, SPEC, 4)
    assert declaration in generated
    assert generated.endswith(source[source.index("{") + 1:])
    assert generate_batch_source(source, SPEC, 1).encode() == source.encode()


def test_conditional_launch_bounds_preserves_original_compiler_choice():
    source = SOURCE.replace("__global__ void", "__global__\n#if EXACT\n#else\n__launch_bounds__(256, 4)\n#endif\nvoid")
    for options in (("-std=c++17",), ("-std=c++17", "-DEXACT=1")):
        spec = KernelSpec("sample", "advance", SPEC.pointers, options=options)
        generated = generate_batch_source(source, spec, 4)
        assert "#if EXACT\n#else\n__launch_bounds__(256, 4)\n#endif\nvoid" in generated
        assert generated.endswith(source[source.index("{") + 1:])


@pytest.mark.parametrize("bounds", ["0", "256, -1", "MAX_THREADS", "256,4,1"])
def test_unqualified_launch_bound_expressions_remain_refused(bounds):
    source = SOURCE.replace("__global__ void", f"__global__ __launch_bounds__({bounds}) void")
    with pytest.raises(BatchKernelUnsupported, match="one unconditional"):
        generate_batch_source(source, SPEC, 4)


def test_arithmetic_body_is_preserved_verbatim_and_only_member_is_offset():
    batched = generate_batch_source(SOURCE, SPEC, 4)
    original_body = SOURCE[SOURCE.index("{") + 1:]
    assert batched.endswith(original_body)
    assert "state = reinterpret_cast<real*>" in batched
    assert "shared = reinterpret_cast" not in batched
    assert "bytes[1]" in batched
    assert "__ensemble_physical_grid.x / 4u" in batched
    assert "const dim3 blockIdx" in batched and "const dim3 gridDim" in batched
    assert 'extern "C" __global__ void advance(' in batched


def test_irregular_three_dimensional_grid_has_no_member_coordinate_mix():
    gx, gy, gz = 7, 3, 5
    members = 4
    assert batch_grid((gx, gy, gz), members) == (28, 3, 5)
    recovered = {(physical_x // gx, physical_x % gx, y, z)
                 for physical_x in range(gx * members)
                 for y in range(gy) for z in range(gz)}
    expected = {(member, x, y, z) for member in range(members)
                for x in range(gx) for y in range(gy) for z in range(gz)}
    assert recovered == expected
    assert batch_grid((7,), 4) == (28, 1, 1)


@pytest.mark.parametrize("pointers", [
    (PointerSpec("state", "member"),),
    SPEC.pointers + (PointerSpec("absent", "member"),),
])
def test_pointer_inventory_must_match_exactly(pointers):
    spec = KernelSpec("sample", "advance", pointers)
    with pytest.raises(BatchKernelUnsupported, match="pointer ownership differs"):
        generate_batch_source(SOURCE, spec, 4)


def test_closed_undefined_macro_uses_cpp_absent_zero_semantics():
    source = SOURCE.replace("real *state", "#if OPTIONAL\nreal *extra,\n#endif\nreal *state")
    generated = generate_batch_source(source, SPEC, 4)
    assert "real *extra" not in generated
    assert "#if OPTIONAL" not in generated
    assert generate_batch_source(source, SPEC, 1) == source


def test_float_or_unknown_compiler_signature_condition_is_refused():
    for condition, reason in [("1.0f", "integer macro expressions"),
                              ("__CUDA_ARCH__ >= 700", "compiler builtin")]:
        source = SOURCE.replace("real *state", f"#if {condition}\nreal *extra,\n#endif\nreal *state")
        with pytest.raises(BatchKernelUnsupported, match=reason):
            generate_batch_source(source, SPEC, 4)
        assert generate_batch_source(source, SPEC, 1) == source


def test_signature_macro_selection_keeps_body_and_other_directives_exact():
    source = '''#define ENABLED 1
typedef float real;
extern "C" __global__ void advance(const real* shared,
#if ENABLED
real* state)
#else
double* different)
#endif
{
#if ENABLED
state[threadIdx.x] = shared[threadIdx.x];
#else
different[threadIdx.x] = 0;
#endif
}
'''
    generated = generate_batch_source(source, SPEC, 4)
    assert generated.startswith("struct __ensemble_strides_advance")
    assert "#define ENABLED 1" in generated
    assert "double* different)" not in generated
    assert generated.endswith(source[source.index("{\n#if ENABLED") + 1:])


def test_compile_option_and_undef_determine_signature_pointer_inventory():
    source = SOURCE.replace("real *state", "#ifdef EXTRA\nconst real *extra,\n#endif\nreal *state")
    spec = KernelSpec("sample", "advance", SPEC.pointers + (PointerSpec("extra", "shared"),),
                      options=("-std=c++17", "-DEXTRA=1"))
    assert "const real *extra" in generate_batch_source(source, spec, 4)
    with pytest.raises(BatchKernelUnsupported, match="missing=.*extra"):
        generate_batch_source(source, KernelSpec("sample", "advance", SPEC.pointers,
                                                ("-std=c++17", "-DEXTRA=1")), 4)
    source = "#undef EXTRA\n" + source
    absent = KernelSpec("sample", "advance", SPEC.pointers,
                        options=("-std=c++17", "-DEXTRA=1"))
    assert "const real *extra" not in generate_batch_source(source, absent, 4)


@pytest.mark.parametrize("expression,expected", [
    ("!1 < 2", 1), ("1 < 2 < 2", 1), ("2 & 1 == 0", 0),
    ("0 && (1 / 0)", 0), ("1 || (1 / 0)", 1), ("-5 / 2", -2),
    ("-5 % 2", -1), ("010 + 0xfL", 23), ("A * 3", 7),
    ("defined(A) && !defined(ABSENT)", 1),
])
def test_signature_integer_conditions_use_c_precedence_and_rounding(expression, expected):
    from gpuwm.ensemble.batch_kernel import _integer_condition
    assert _integer_condition(expression, {"A": "1 + 2"}) == expected


def test_macro_headers_and_malformed_conditions_do_not_escape_audit():
    for prefix, reason in [("#include <unknown.h>\n", "assemble all headers"),
                           ("#define FLOAT 1.0\n#if FLOAT\n#define ENABLED 1\n#endif\n",
                            "integer macro expressions"),
                           ("#if 1\n", "unterminated conditional")]:
        with pytest.raises(BatchKernelUnsupported, match=reason):
            generate_batch_source(prefix + SOURCE, SPEC, 4)


def test_wrf_hook_option_policy_is_used_directly(monkeypatch):
    from gpuwm import wrf_exact
    from gpuwm.ensemble.batch_kernel import _effective_options
    monkeypatch.setattr(wrf_exact, "ENABLED", True)
    requested = ("-std=c++17", "--ftz=true", "-DGPUWM_WRF_EXACT=0")
    assert _effective_options(requested) == wrf_exact.effective_options(requested)


@pytest.mark.parametrize("option", ["--pre-include=outside.h", "-includeoutside.h", "-include outside.h"])
def test_option_injected_headers_cannot_change_audited_signature(option):
    spec = KernelSpec("sample", "advance", SPEC.pointers, ("-std=c++17", option))
    with pytest.raises(BatchKernelUnsupported, match="pre-included headers"):
        generate_batch_source(SOURCE, spec, 4)
    assert generate_batch_source(SOURCE, spec, 1) == SOURCE


@pytest.mark.parametrize("token", ["const", "real", "float", "__restrict__", "size_t", "dim3"])
def test_type_and_readonly_macros_cannot_change_audited_storage(token):
    spec = KernelSpec("sample", "advance", SPEC.pointers, ("-std=c++17", f"-D{token}=altered"))
    with pytest.raises(BatchKernelUnsupported, match="audited CUDA type/qualifier"):
        generate_batch_source(SOURCE, spec, 4)
    with pytest.raises(BatchKernelUnsupported, match="audited CUDA type/qualifier"):
        generate_batch_source(f"#define {token} altered\n" + SOURCE, SPEC, 4)


@pytest.mark.parametrize("source", [SOURCE.replace("typedef float real;", "typedef double real;"),
                                    SOURCE.replace("typedef float real;", "using real = float;"),
                                    SOURCE.replace("typedef float real;", "")])
def test_real_alias_is_an_exact_float32_source_contract(source):
    with pytest.raises(BatchKernelUnsupported, match="exactly typedef float real"):
        generate_batch_source(source, SPEC, 4)
    assert generate_batch_source(source, SPEC, 1) == source


def test_continued_macro_grid_read_uses_virtual_components_in_helper():
    for token in ("blockIdx.x", "gridDim.x"):
        source = "#define PHYSICAL " + "\\\n" + f"    {token}\n__device__ int helper() {{ return PHYSICAL; }}\n" + SOURCE
        generated = generate_batch_source(source, SPEC, 4)
        accessor = "__ensemble_virtual_block()" if token.startswith("block") else "__ensemble_virtual_grid()"
        assert f"    {accessor}.x\n__device__ int helper()" in generated
        assert generate_batch_source(source, SPEC, 1) == source


def test_real_default_diagnostics_signature_is_resolved_with_original_body():
    from gpuwm.ensemble.batch_kernel import _active_source, _close
    from gpuwm.core.kernels import module_source
    source = module_source("diagnostics")
    members = {"thp", "php", "mup", "qv", "p", "al", "alt"}
    names = ("thp", "php", "mup", "thb", "phb", "dphbr", "alb", "rdnw",
             "c1h", "c2h", "c3h", "c4h", "c3f", "c4f", "dc3f", "dc4f", "mub",
             "qv", "p", "al", "alt")
    spec = KernelSpec("diagnostics", "calc_p_alpha", tuple(
        PointerSpec(name, "member" if name in members else "shared") for name in names))
    generated = generate_batch_source(source, spec, 4)
    body = source.index("{\n    int col", source.index("void calc_p_alpha"))
    assert generated.endswith(source[body + 1:])
    assert source[:source.index("void calc_p_alpha")] in generated
    # Optional helper definitions retain their raw bytes, even when their
    # selector is inactive. Only the compiled entry declaration sets its ABI.
    active = _active_source(generated, spec.options)
    opening = active.index("(", active.index("void calc_p_alpha"))
    closing = _close(active, opening, "(", ")")
    assert "p_perturbation" not in active[opening:closing]
    assert "__ensemble_pointer_strides" in generated
    assert generate_batch_source(source, spec, 1) == source


def test_real_diagnostics_exact_signature_requires_its_extra_pointer_roles(monkeypatch):
    from gpuwm import wrf_exact
    from gpuwm.core.kernels import module_source
    # Bind the actual optional header while supplying its explicit compile flag.
    monkeypatch.setattr(wrf_exact, "DIAGNOSTICS_ENABLED", True)
    source = module_source("diagnostics")
    members = {"thp", "php", "mup", "qv", "p", "al", "alt", "p_perturbation"}
    names = ("thp", "php", "mup", "thb", "phb", "dphbr", "alb", "pb", "rdnw",
             "c1h", "c2h", "c3h", "c4h", "c3f", "c4f", "dc3f", "dc4f", "mub",
             "qv", "p", "al", "alt", "p_perturbation")
    pointers = tuple(PointerSpec(name, "member" if name in members else "shared") for name in names)
    spec = KernelSpec("diagnostics", "calc_p_alpha", pointers,
                      ("-std=c++17", "-DGPUWM_WRF_EXACT_D_DIAGNOSTICS=1"))
    generated = generate_batch_source(source, spec, 4)
    body = source.index("{\n    int col", source.index("void calc_p_alpha"))
    assert generated.endswith(source[body + 1:])
    assert "p_perturbation = reinterpret_cast<real*>" in generated
    assert "pb = reinterpret_cast" not in generated
    incomplete = KernelSpec("diagnostics", "calc_p_alpha", pointers[:-1], spec.options)
    with pytest.raises(BatchKernelUnsupported, match="missing=.*p_perturbation"):
        generate_batch_source(source, incomplete, 4)


def test_acoustic_integer_tier_prefix_and_n1_bytes_are_preserved():
    from gpuwm.core.kernels import module_source_int_defines
    source = module_source_int_defines("acoustic", (("WPHI_MAX_LEV", 257),))
    spec = KernelSpec("acoustic", "advance_mu_th", ())
    assert generate_batch_source(source, spec, 1) == source
    # The closed view still retains the explicit tier in the emitted prelude.
    from gpuwm.ensemble.batch_kernel import _active_source
    active = _active_source(source, spec.options)
    assert len(active) == len(source)
    assert "__global__" in active and "ww_ref," not in active


@pytest.mark.parametrize("addition,reason", [
    ("cooperative_groups::this_grid().sync();", "synchronization"),
    ("child<<<1, 32>>>();", "device launches"),
    ('asm("mov.u32 %0, %ctaid.x;" : "=r"(i));', "PTX"),
    ('asm("mov.u32 %0, %nctaid.x;" : "=r"(i));', "PTX"),
    ("__threadfence();", "synchronization"),
])
def test_physical_grid_bypasses_require_a_dedicated_adapter(addition, reason):
    source = SOURCE.replace("int i =", addition + "\n    int i =")
    with pytest.raises(BatchKernelUnsupported, match=reason):
        generate_batch_source(source, SPEC, 4)


def test_helper_grid_reads_use_global_accessors_and_keep_float_text():
    source = "__device__ int physical_x() { return blockIdx.x; }\n" + SOURCE
    generated = generate_batch_source(source, SPEC, 4)
    assert "return __ensemble_virtual_block().x;" in generated
    assert "const dim3 blockIdx" not in generated
    assert "const uint3 __ensemble_physical_block = blockIdx" in generated
    body = generated[generated.index("int i = __ensemble_virtual_block()"):]
    restored = body.replace("__ensemble_virtual_block()", "blockIdx").replace("__ensemble_virtual_grid()", "gridDim")
    assert restored == SOURCE[SOURCE.index("int i ="):]


def test_inactive_helper_and_comment_string_coordinate_text_is_not_rewritten():
    prefix = '''#if 0
__device__ int inactive() { return blockIdx.x; }
#endif
__device__ int active() { return gridDim.x; }
// blockIdx.x and gridDim.x are explanatory text.
__device__ const char* text() { return "blockIdx.x gridDim.x"; }
'''
    generated = generate_batch_source(prefix + SOURCE, SPEC, 4)
    assert "return blockIdx.x; }\n#endif" in generated
    assert '// blockIdx.x and gridDim.x are explanatory text.' in generated
    assert 'return "blockIdx.x gridDim.x";' in generated
    assert "return __ensemble_virtual_grid().x;" in generated


@pytest.mark.parametrize("addition,reason", [
    ("__device__ int cache() { static int value = 0; return value++; }\n", "static storage"),
    ("__device__ int cache() { static const int* value; return 0; }\n", "static storage"),
    ("__device__ int helper() { uint3 value = blockIdx; return value.x; }\n", "whole CUDA coordinate"),
    ("__device__ int helper() { return *(&blockIdx.x); }\n", "qualified/addressed"),
    ("__device__ int __ensemble_virtual_block() { return 0; }\n", "helper name collides"),
    ("#define JOIN(x,y) x ## y\n", "token pasting"),
])
def test_unvirtualizable_coordinate_or_static_storage_is_refused(addition, reason):
    with pytest.raises(BatchKernelUnsupported, match=reason):
        generate_batch_source(addition + SOURCE, SPEC, 4)
    assert generate_batch_source(addition + SOURCE, SPEC, 1) == addition + SOURCE


@pytest.mark.parametrize("component", ["blockIdx.x", "gridDim.z"])
@pytest.mark.parametrize("mutation", [
    "{component} = 1", "{component} += 1", "{component} -= 1",
    "{component} *= 1", "{component} /= 1", "{component} %= 1",
    "{component} &= 1", "{component} |= 1", "{component} ^= 1",
    "{component} <<= 1", "{component} >>= 1",
    "++{component}", "--{component}", "{component}++", "{component}--",
    "++({component})", "--(({component}))", "({component})++",
    "(({component})) += 1", "{component} /* readonly */ += 1",
])
def test_coordinate_component_mutations_are_refused_without_changing_n1(component, mutation):
    source = "__device__ int helper() { " + mutation.format(component=component) + "; return 0; }\n" + SOURCE
    with pytest.raises(BatchKernelUnsupported, match="coordinate writes"):
        generate_batch_source(source, SPEC, 4)
    assert generate_batch_source(source, SPEC, 1) == source


def test_coordinate_component_comparisons_and_integer_arithmetic_remain_reads():
    source = "__device__ int helper() { return blockIdx.x == 1 || gridDim.z >= 2 || (blockIdx.y >> 1); }\n" + SOURCE
    generated = generate_batch_source(source, SPEC, 4)
    assert "return __ensemble_virtual_block().x == 1 || __ensemble_virtual_grid().z >= 2" in generated


def test_continued_coordinate_macro_component_write_is_refused():
    source = "#define WRITE_COORD " + "\\\n" + "    blockIdx.y /* readonly */ <<= 1\n" + SOURCE
    with pytest.raises(BatchKernelUnsupported, match="coordinate writes"):
        generate_batch_source(source, SPEC, 4)


def test_immutable_static_table_and_block_shared_helper_storage_are_allowed():
    prefix = '''__device__ int helper() {
static const int values[2] = {1, 2};
__shared__ int lanes[32];
lanes[threadIdx.x] = values[threadIdx.x % 2];
__syncthreads();
return lanes[threadIdx.x] + blockIdx.x;
}
'''
    generated = generate_batch_source(prefix + SOURCE, SPEC, 4)
    assert "static const int values[2] = {1, 2}" in generated
    assert "__shared__ int lanes[32]" in generated and "__syncthreads();" in generated


def test_actual_bandwidth_glue_unit_virtualizes_helpers_without_auditing_other_entry_tables():
    from gpuwm.core.kernels import module_source
    source = module_source("bandwidth_glue")
    spec = KernelSpec("bandwidth_glue", "glue_total_theta", (
        PointerSpec("thb", "shared"), PointerSpec("thp", "member"), PointerSpec("dst", "member")))
    generated = generate_batch_source(source, spec, 4, audit_options=spec.options + ("-arch=sm_120",))
    assert "GlueWordTable" in generated
    assert "__ensemble_virtual_block().x * blockDim.x" in generated
    assert "__fadd_rn(b.x, a.x)" in generated
    assert generate_batch_source(source, spec, 1) == source


def test_runtime_architecture_context_uses_compiler_target_and_rejects_conflicts(monkeypatch):
    import sys
    from types import ModuleType
    from gpuwm.ensemble.batch_kernel import _runtime_audit_options
    parent, cuda = ModuleType("cupy"), ModuleType("cupy.cuda")
    cuda.compiler = SimpleNamespace(_get_arch_for_options_for_nvrtc=lambda: ("-arch=sm_120", "cubin"))
    parent.cuda = cuda
    monkeypatch.setitem(sys.modules, "cupy", parent)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    spec = KernelSpec("sample", "advance", SPEC.pointers, ("-std=c++17", "-arch=compute_120"))
    assert _runtime_audit_options(spec)[-1] == "-arch=sm_120"
    other = KernelSpec("sample", "advance", SPEC.pointers, ("-std=c++17", "-arch=compute_90"))
    with pytest.raises(BatchKernelUnsupported, match="another ABI"):
        _runtime_audit_options(other)


def test_actual_strict_acoustic_helper_coordinates_are_virtual_and_math_text_is_kept(monkeypatch):
    from gpuwm import wrf_exact
    from gpuwm.core.kernels import module_source
    monkeypatch.setattr(wrf_exact, "ENABLED", True)
    source = module_source("acoustic")
    names = ("u_pp v_pp u v mup mu_pp mu_pp_old rmu_t mudf thp thb th_pp th_pp_old "
             "rth_t ww_pp ww_ref p_pp p_pp_old dnw rdnw fnm fnp c1h c2h mub2d").split()
    shared = {"thb", "dnw", "rdnw", "fnm", "fnp", "c1h", "c2h", "mub2d"}
    spec = KernelSpec("acoustic", "advance_mu_th", tuple(
        PointerSpec(name, "shared" if name in shared else "member") for name in names))
    generated = generate_batch_source(source, spec, 4,
                                      audit_options=wrf_exact.effective_options(spec.options) + ("-arch=sm_120",))
    assert "int c = __ensemble_virtual_block().x*blockDim.x+threadIdx.x;" in generated
    original_start = source.index("void wrf_advance_mu_theta(")
    original_end = source.index("// solve_em applies", original_start)
    generated_start = generated.index("void wrf_advance_mu_theta(")
    generated_end = generated.index("// solve_em applies", generated_start)
    restored = generated[generated_start:generated_end].replace("__ensemble_virtual_block()", "blockIdx")
    assert restored == source[original_start:original_end]
    assert generate_batch_source(source, spec, 1) == source


@pytest.mark.gpu
@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
def test_device_helper_virtual_coordinates_preserve_shared_barriers_and_scalar_words(members, monkeypatch):
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("no CUDA GPU / cupy")
    import cupy as cp
    from gpuwm.core import kernels
    source = '''#define COLUMN_X \\
    blockIdx.x
__device__ __forceinline__ unsigned int logical_block() {
return COLUMN_X + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}
__device__ void shared_rotate(const unsigned int* input, unsigned int* output) {
__shared__ unsigned int values[32];
unsigned int block = logical_block();
values[threadIdx.x] = input[block * blockDim.x + threadIdx.x];
__syncthreads();
output[block * blockDim.x + threadIdx.x] = values[(threadIdx.x + 1) % blockDim.x]
    ^ (gridDim.x + 16u * gridDim.y + 256u * gridDim.z);
}
extern "C" __global__ void helper_probe(const unsigned int* input, unsigned int* output) {
shared_rotate(input, output);
}
'''
    original = kernels.module_source
    monkeypatch.setattr(kernels, "module_source", lambda name: source if name == "ensemble_helper_probe" else original(name))
    grid, block = (7, 3, 5), (32,)
    words = int(np.prod(grid)) * block[0]
    rng = np.random.default_rng(728)
    host = rng.integers(0, 2**32, (members, words), dtype=np.uint32)
    input_array = cp.asarray(host)
    output = cp.empty_like(input_array)
    reference = cp.empty_like(input_array)
    scalar = kernels.get_kernel("ensemble_helper_probe", "helper_probe")
    for member in range(members):
        scalar(grid, block, (input_array[member], reference[member]))
    spec = KernelSpec("ensemble_helper_probe", "helper_probe", (
        PointerSpec("input", "member", "uint32"), PointerSpec("output", "member", "uint32")))
    prepared = prepare_batch_kernel_launch(spec, members, grid, block, (input_array, output),
                                           pointer_strides={"input": words * 4, "output": words * 4})
    prepared()
    cp.cuda.get_current_stream().synchronize()
    assert cp.asnumpy(output).tobytes() == cp.asnumpy(reference).tobytes()


def test_block_local_barriers_remain_unchanged():
    source = SOURCE.replace("int i =", "__syncthreads();\n    int i =")
    assert generate_batch_source(source, SPEC, 4).endswith(source[source.index("{") + 1:])


@pytest.mark.parametrize("declaration,reason", [
    ("real **state", "indirect pointer"),
    ("PointerTable state", "by-value aggregate"),
    ("PointerTable *state", "aggregate pointer"),
])
def test_hidden_pointer_addresses_are_refused(declaration, reason):
    source = SOURCE.replace("real *state", declaration)
    with pytest.raises(BatchKernelUnsupported, match=reason):
        generate_batch_source(source, SPEC, 4)


def test_shared_writable_pointer_would_mix_members():
    source = SOURCE.replace("const real*", "real*")
    with pytest.raises(BatchKernelUnsupported, match="shared pointer shared must be const"):
        generate_batch_source(source, SPEC, 4)
    with pytest.raises(BatchKernelUnsupported, match="pointer-level qualifiers"):
        generate_batch_source(SOURCE.replace("const real* __restrict__ shared",
                                             "real* const shared"), SPEC, 4)


def test_writable_device_global_is_not_member_local():
    with pytest.raises(BatchKernelUnsupported, match="writable device globals"):
        generate_batch_source("__device__ int counter;\n" + SOURCE, SPEC, 4)
    assert generate_batch_source("__device__ const int table[2] = {1, 2};\n" + SOURCE,
                                 SPEC, 4).endswith(SOURCE[SOURCE.index("{") + 1:])
    with pytest.raises(BatchKernelUnsupported, match="writable device globals"):
        generate_batch_source("__device__ int counter[2] = {0, 0};\n" + SOURCE, SPEC, 4)


def test_cooperative_macro_outside_entry_cannot_bypass_audit():
    with pytest.raises(BatchKernelUnsupported, match="cooperative grid"):
        generate_batch_source("#define BARRIER cooperative_groups::this_grid().sync()\n" + SOURCE,
                              SPEC, 4)


def test_stride_descriptor_has_parameter_order_and_shared_zero():
    packed = pack_pointer_strides(SPEC, {"state": 120, "shared": 0})
    assert np.frombuffer(packed.tobytes(), dtype=np.uint64).tolist() == [0, 120]
    with pytest.raises(ValueError, match="shared pointer"):
        pack_pointer_strides(SPEC, {"state": 120, "shared": 4})
    with pytest.raises(ValueError, match="positive"):
        pack_pointer_strides(SPEC, {"state": 0, "shared": 0})
    with pytest.raises(ValueError, match="exactly"):
        pack_pointer_strides(SPEC, {"state": 120})


@pytest.mark.parametrize("grid,members", [((0,), 4), ((1, 2, 3, 4), 4),
                                         ((2**30,), 4), ((1, 65536), 1)])
def test_bad_or_overflowing_grids_are_refused(grid, members):
    with pytest.raises(ValueError):
        batch_grid(grid, members)


def test_launch_uses_one_grid_and_one_by_value_descriptor(monkeypatch):
    calls = []
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._compiled",
                        lambda *_: lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._parameter_names",
                        lambda *_: ("shared", "state", "nx"))
    shared = SimpleNamespace(shape=(32,), nbytes=128, dtype=np.dtype("float32"),
                             flags=SimpleNamespace(c_contiguous=True),
                             __cuda_array_interface__={})
    state = SimpleNamespace(shape=(4, 32), nbytes=512, dtype=np.dtype("float32"),
                            flags=SimpleNamespace(c_contiguous=True),
                            __cuda_array_interface__={})
    args = (shared, state, 32)
    get_batch_kernel(SPEC, 4)((7, 3, 5), (32,), args,
                              pointer_strides={"shared": 0, "state": 128})
    assert calls[0][0][:2] == ((28, 3, 5), (32,))
    assert calls[0][0][2][:-1] == args
    assert np.frombuffer(calls[0][0][2][-1].tobytes(), dtype=np.uint64).tolist() == [0, 128]
    calls.clear()
    get_batch_kernel(SPEC, 1)((7,), (32,), args)
    assert calls == [(((7,), (32,), args), {})]


def test_constructing_launch_adapter_never_imports_cupy(monkeypatch):
    import builtins
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        assert name != "cupy", "adapter construction must remain CPU-only"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    assert callable(get_batch_kernel(SPEC, 4))


def test_member_backing_and_argument_order_are_validated():
    shared = SimpleNamespace(shape=(32,), nbytes=128, dtype=np.dtype("float32"),
                             flags=SimpleNamespace(c_contiguous=True),
                             __cuda_array_interface__={})
    backing = SimpleNamespace(shape=(4, 32), nbytes=512, dtype=np.dtype("float32"),
                              flags=SimpleNamespace(c_contiguous=True),
                              __cuda_array_interface__={})
    strides = {"shared": 0, "state": 128}
    names = ("shared", "state", "nx")
    validate_pointer_arguments(SPEC, 4, (shared, backing, 32), strides, names)
    with pytest.raises(ValueError, match="complete"):
        validate_pointer_arguments(SPEC, 4, (backing, shared, 32), strides, names)
    with pytest.raises(ValueError, match="exact slab"):
        validate_pointer_arguments(SPEC, 4, (shared, backing, 32),
                                   {"shared": 0, "state": 124}, names)
    with pytest.raises(ValueError, match="parameter order"):
        validate_pointer_arguments(SPEC, 4, (shared, backing), strides, names)
    with pytest.raises(TypeError, match="CUDA array"):
        validate_pointer_arguments(SPEC, 4, (shared, np.zeros((4, 32)), 32), strides, names)
    backing.dtype = np.dtype("float16")
    with pytest.raises(TypeError, match="requires float32"):
        validate_pointer_arguments(SPEC, 4, (shared, backing, 32), strides, names)


def test_padded_member_prefix_has_inner_contiguity_and_allocation_bounds():
    spec = KernelSpec("sample", "advance", (PointerSpec("state", "member"),))
    # Four logical 2x3 slabs begin after a guard word in physical 8-word rows.
    memory = SimpleNamespace(ptr=1024, size=128)
    array = SimpleNamespace(shape=(4, 2, 3), strides=(32, 12, 4), nbytes=96,
                            dtype=np.dtype("float32"), flags=SimpleNamespace(c_contiguous=False),
                            data=SimpleNamespace(ptr=1028, mem=memory), __cuda_array_interface__={})
    validate_pointer_arguments(spec, 4, (array,), {"state": 32}, ("state",))
    # Last-member coverage includes the logical suffix but not trailing guards.
    memory.size = 124
    validate_pointer_arguments(spec, 4, (array,), {"state": 32}, ("state",))
    memory.size = 123
    with pytest.raises(ValueError, match="allocation bounds"):
        validate_pointer_arguments(spec, 4, (array,), {"state": 32}, ("state",))


@pytest.mark.parametrize("change,reason", [
    ({"strides": (32, 16, 4)}, "contiguous inner"),
    ({"strides": (32, 12, -4)}, "contiguous inner"),
    ({"strides": (32, 0, 4)}, "contiguous inner"),
    ({"strides": (20, 12, 4)}, "leading stride"),
    ({"strides": (-32, 12, 4)}, "leading stride"),
    ({"strides": (31, 12, 4)}, "leading stride"),
    ({"data": SimpleNamespace(ptr=1028)}, "owned allocation"),
    ({"data": SimpleNamespace(ptr=1023, mem=SimpleNamespace(ptr=1024, size=128))}, "allocation bounds"),
    ({"data": SimpleNamespace(ptr=1026, mem=SimpleNamespace(ptr=1024, size=128))}, "allocation bounds"),
    ({"data": SimpleNamespace(ptr=1028, mem=SimpleNamespace(ptr=1024, size=100))}, "allocation bounds"),
])
def test_padded_views_cannot_hide_overlap_broadcast_or_unbounded_address(change, reason):
    spec = KernelSpec("sample", "advance", (PointerSpec("state", "member"),))
    array = SimpleNamespace(shape=(4, 2, 3), strides=(32, 12, 4), nbytes=96,
                            dtype=np.dtype("float32"), flags=SimpleNamespace(c_contiguous=False),
                            data=SimpleNamespace(ptr=1028, mem=SimpleNamespace(ptr=1024, size=128)),
                            __cuda_array_interface__={})
    array.__dict__.update(change)
    # Bind the actual stride where supplied; the overlap/alignment checks must
    # still refuse it. Other cases keep the original 32-byte descriptor.
    stride = array.strides[0]
    with pytest.raises(ValueError, match=reason):
        validate_pointer_arguments(spec, 4, (array,), {"state": stride}, ("state",))


def test_padded_pointer_span_uses_unbounded_host_integer_math():
    spec = KernelSpec("sample", "advance", (PointerSpec("state", "member"),))
    stride = 2**63
    array = SimpleNamespace(shape=(4, 2), strides=(stride, 4), nbytes=32,
                            dtype=np.dtype("float32"), flags=SimpleNamespace(c_contiguous=False),
                            data=SimpleNamespace(ptr=1024, mem=SimpleNamespace(ptr=1024, size=128)),
                            __cuda_array_interface__={})
    with pytest.raises(ValueError, match="allocation bounds"):
        validate_pointer_arguments(spec, 4, (array,), {"state": stride}, ("state",))


def test_prepared_binding_reuses_raw_handle_arguments_and_descriptor(monkeypatch):
    calls = []
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 0)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._runtime_audit_options", lambda spec: spec.options + ("-arch=sm_120",))
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._compiled",
                        lambda *_: lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._parameter_names",
                        lambda *_: ("shared", "state", "nx"))
    shared = SimpleNamespace(shape=(32,), strides=(4,), nbytes=128, dtype=np.dtype("float32"),
                             flags=SimpleNamespace(c_contiguous=True), device=SimpleNamespace(id=0),
                             data=SimpleNamespace(ptr=1024, mem=SimpleNamespace(ptr=1024, size=128)),
                             __cuda_array_interface__={})
    state = SimpleNamespace(shape=(4, 32), strides=(128, 4), nbytes=512, dtype=np.dtype("float32"),
                            flags=SimpleNamespace(c_contiguous=True), device=SimpleNamespace(id=0),
                            data=SimpleNamespace(ptr=2048, mem=SimpleNamespace(ptr=2048, size=512)),
                            __cuda_array_interface__={})
    args = (shared, state, 32)
    prepared = prepare_batch_kernel_launch(SPEC, 4, (7, 3, 5), (32,), args,
                                           pointer_strides={"shared": 0, "state": 128})
    # After binding neither descriptor packing, source inspection nor cache
    # lookup is needed, and the identical argument object is reused.
    def unexpected(*_, **__):
        raise AssertionError("binding metadata was rebuilt during submission")
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel.pack_pointer_strides", unexpected)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel.validate_pointer_arguments", unexpected)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._compiled", unexpected)
    prepared()
    prepared()
    assert calls[0][0][:2] == ((28, 3, 5), (32,))
    assert calls[0][0][2] is calls[1][0][2]
    assert calls[0][0][2][:-1] == args
    with pytest.raises(TypeError):
        prepared.binding_receipt["device"] = 1
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 1)
    with pytest.raises(ValueError, match="belongs to CUDA device 0"):
        prepared()
    assert len(calls) == 2


def test_prepared_n1_keeps_original_grid_args_and_skips_batch_validation(monkeypatch):
    calls = []
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 0)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._compiled",
                        lambda *_: lambda *args, **kwargs: calls.append((args, kwargs)))
    def unexpected(*_, **__):
        raise AssertionError("N=1 must not construct batch metadata")
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel.batch_grid", unexpected)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel.validate_pointer_arguments", unexpected)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel.pack_pointer_strides", unexpected)
    args = ("unchanged-scalar-arguments", 32)
    prepared = prepare_batch_kernel_launch(SPEC, 1, (7,), (32,), args)
    prepared()
    assert calls == [(((7,), (32,), args), {"shared_mem": 0})]


def test_prepared_binding_keeps_arrays_and_allocation_owners_live(monkeypatch):
    import gc
    import weakref
    class Array:
        __cuda_array_interface__ = {}
        shape, strides, nbytes = (4, 2), (8, 4), 32
        dtype = np.dtype("float32")
        flags = SimpleNamespace(c_contiguous=True)
        device = SimpleNamespace(id=0)
        def __init__(self):
            self.data = SimpleNamespace(ptr=1024, mem=SimpleNamespace(ptr=1024, size=32))
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 0)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._runtime_audit_options", lambda spec: spec.options + ("-arch=sm_120",))
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._compiled", lambda *_: lambda *_args, **_kwargs: None)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._parameter_names", lambda *_: ("state",))
    array = Array()
    reference = weakref.ref(array)
    spec = KernelSpec("sample", "advance", (PointerSpec("state", "member"),))
    prepared = prepare_batch_kernel_launch(spec, 4, (1,), (32,), (array,),
                                           pointer_strides={"state": 8})
    del array
    gc.collect()
    assert reference() is not None
    prepared()
    del prepared
    gc.collect()
    assert reference() is None


def test_prepared_binding_refuses_an_argument_from_another_device(monkeypatch):
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 0)
    argument = SimpleNamespace(device=SimpleNamespace(id=1), __cuda_array_interface__={})
    with pytest.raises(ValueError, match="not on owning CUDA device 0"):
        prepare_batch_kernel_launch(SPEC, 1, (1,), (32,), (argument,))


def _supplied_cpu_arrays(members):
    shared = SimpleNamespace(shape=(32,), strides=(4,), nbytes=128, dtype=np.dtype("float32"),
                             flags=SimpleNamespace(c_contiguous=True), device=SimpleNamespace(id=0),
                             data=SimpleNamespace(ptr=1024, mem=SimpleNamespace(ptr=1024, size=128)),
                             __cuda_array_interface__={})
    state = SimpleNamespace(shape=(members, 32), strides=(128, 4), nbytes=members * 128,
                            dtype=np.dtype("float32"), flags=SimpleNamespace(c_contiguous=True),
                            device=SimpleNamespace(id=0),
                            data=SimpleNamespace(ptr=2048, mem=SimpleNamespace(ptr=2048, size=members * 128)),
                            __cuda_array_interface__={})
    return shared, state


def test_supplied_prepared_binding_reuses_args_and_records_source_context(monkeypatch):
    from hashlib import sha256
    calls = []
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 0)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._runtime_audit_options", lambda spec: spec.options + ("-arch=sm_120",))
    def compile_source(source, spec, members, options):
        generated = generate_batch_source(source, spec, members, audit_options=options)
        return (lambda *args, **kwargs: calls.append((args, kwargs)),
                sha256(source.encode()).hexdigest(), sha256(generated.encode()).hexdigest())
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._compiled_source", compile_source)
    shared, state = _supplied_cpu_arrays(4)
    args = (shared, state, 32)
    prepared = prepare_batch_source_launch(SOURCE, SPEC, 4, (7, 3, 5), (32,), args,
                                           pointer_strides={"shared": 0, "state": 128},
                                           shared_mem=128, stream="owned-stream")
    def unexpected(*_, **__):
        raise AssertionError("supplied binding rebuilt metadata at submission")
    for name in ("pack_pointer_strides", "validate_pointer_arguments", "_entry_parts", "_compiled_source"):
        monkeypatch.setattr("gpuwm.ensemble.batch_kernel." + name, unexpected)
    prepared()
    prepared()
    assert calls[0][0][:2] == ((28, 3, 5), (32,))
    assert calls[0][0][2] is calls[1][0][2]
    assert calls[0][0][2][:-1] == args
    assert calls[0][1] == {"shared_mem": 128, "stream": "owned-stream"}
    receipt = prepared.binding_receipt
    assert receipt["source_kind"] == "supplied"
    assert receipt["source_sha256"] == sha256(SOURCE.encode()).hexdigest()
    assert receipt["compiled_source_sha256"] != receipt["source_sha256"]
    assert receipt["options"] == SPEC.options
    assert receipt["audit_options"][-1] == "-arch=sm_120"
    assert receipt["arrays"][1]["allocation_bytes"] == 512
    with pytest.raises(TypeError):
        receipt["source_sha256"] = "changed"
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 1)
    with pytest.raises(ValueError, match="belongs to CUDA device 0"):
        prepared()
    assert len(calls) == 2


def test_supplied_n1_audits_signature_but_retains_exact_args_source_and_grid(monkeypatch):
    from hashlib import sha256
    calls = []
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 0)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._runtime_audit_options", lambda spec: spec.options + ("-arch=sm_120",))
    def compile_source(source, spec, members, options):
        assert source == SOURCE and members == 1
        digest = sha256(source.encode()).hexdigest()
        return lambda *args, **kwargs: calls.append((args, kwargs)), digest, digest
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._compiled_source", compile_source)
    def unexpected(*_, **__):
        raise AssertionError("supplied N=1 must not construct batch metadata")
    for name in ("batch_grid", "pack_pointer_strides", "validate_pointer_arguments"):
        monkeypatch.setattr("gpuwm.ensemble.batch_kernel." + name, unexpected)
    shared, state = _supplied_cpu_arrays(1)
    args = (shared, state, 32)
    prepared = prepare_batch_source_launch(SOURCE, SPEC, 1, (7,), (32,), args)
    prepared()
    assert calls == [(((7,), (32,), args), {"shared_mem": 0})]
    assert prepared.binding_receipt["source_sha256"] == prepared.binding_receipt["compiled_source_sha256"]


@pytest.mark.parametrize("source", [None, "", "   ", b"CUDA source"])
def test_supplied_source_requires_nonempty_text_before_touching_a_device(monkeypatch, source):
    def unexpected():
        raise AssertionError("invalid source queried a CUDA device")
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", unexpected)
    with pytest.raises(TypeError, match="nonempty string"):
        prepare_batch_source_launch(source, SPEC, 4, (1,), (32,), ())


def test_supplied_binding_accepts_bounded_padded_member_slabs_without_copying(monkeypatch):
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 0)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._runtime_audit_options", lambda spec: spec.options + ("-arch=sm_120",))
    calls = []
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._compiled_source",
                        lambda *_: (lambda *args, **kwargs: calls.append(args), "original", "compiled"))
    shared, state = _supplied_cpu_arrays(4)
    state.flags.c_contiguous = False
    state.strides = (160, 4)
    state.data.ptr = 2052
    state.data.mem.size = 640
    prepared = prepare_batch_source_launch(SOURCE, SPEC, 4, (1,), (32,), (shared, state, 32),
                                           pointer_strides={"shared": 0, "state": 160})
    prepared()
    assert calls[0][2][1] is state
    row = prepared.binding_receipt["arrays"][1]
    assert row["pointer"] == 2052 and row["allocation_pointer"] == 2048
    assert row["strides"] == (160, 4) and row["allocation_bytes"] == 640


@pytest.mark.parametrize("members,problem", [
    (members, problem) for members in (1, 4)
    for problem in ("missing_pointer", "wrong_dtype", "storage_dtype", "wrong_arg_count", "other_device", "incomplete_members")
    if (members, problem) != (1, "incomplete_members")])
def test_supplied_source_binding_refuses_bad_contract_before_compilation(monkeypatch, members, problem):
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 0)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._runtime_audit_options", lambda spec: spec.options + ("-arch=sm_120",))
    def unexpected(*_, **__):
        raise AssertionError("invalid supplied binding reached compilation")
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._compiled_source", unexpected)
    shared, state = _supplied_cpu_arrays(members)
    source, spec, args = SOURCE, SPEC, (shared, state, 32)
    if problem == "missing_pointer":
        spec = KernelSpec("sample", "advance", SPEC.pointers[:-1])
    elif problem == "wrong_dtype":
        source = SOURCE.replace("real *state", "double *state")
    elif problem == "wrong_arg_count":
        args = args[:-1]
    elif problem == "storage_dtype":
        state.dtype = np.dtype("float16")
    elif problem == "other_device":
        state.device.id = 1
    else:
        state.shape = (1, 32)
    with pytest.raises((ValueError, TypeError)):
        prepare_batch_source_launch(source, spec, members, (1,), (32,), args,
                                    pointer_strides={"shared": 0, "state": 128})


def test_supplied_source_binding_keeps_array_and_memory_owner_live(monkeypatch):
    import gc
    import weakref
    class Owner:
        ptr, size = 2048, 512
    class Array:
        __cuda_array_interface__ = {}
        shape, strides, nbytes = (4, 32), (128, 4), 512
        dtype = np.dtype("float32")
        flags = SimpleNamespace(c_contiguous=True)
        device = SimpleNamespace(id=0)
        def __init__(self, owner):
            self.data = SimpleNamespace(ptr=2048, mem=owner)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._current_device", lambda: 0)
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._runtime_audit_options", lambda spec: spec.options + ("-arch=sm_120",))
    monkeypatch.setattr("gpuwm.ensemble.batch_kernel._compiled_source", lambda *_: (lambda *_a, **_k: None, "original", "compiled"))
    shared, _ = _supplied_cpu_arrays(4)
    owner = Owner()
    array = Array(owner)
    array_ref, owner_ref = weakref.ref(array), weakref.ref(owner)
    prepared = prepare_batch_source_launch(SOURCE, SPEC, 4, (1,), (32,), (shared, array, 32),
                                           pointer_strides={"shared": 0, "state": 128})
    del array, owner
    gc.collect()
    assert array_ref() is not None and owner_ref() is not None
    prepared()
    del prepared
    gc.collect()
    assert array_ref() is None and owner_ref() is None


def test_supplied_compiler_cache_manifest_and_original_options_are_exact(monkeypatch):
    import sys
    from types import ModuleType
    from hashlib import sha256
    from gpuwm.core import kernels
    from gpuwm.certify.kernel_manifest import kernel_manifest
    from gpuwm.ensemble.batch_kernel import _compiled_source
    modules, calls = [], []
    cp = ModuleType("cupy")
    cp.cuda = SimpleNamespace(Device=lambda: SimpleNamespace(id=0))
    class RawModule:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            modules.append(self)
        def get_function(self, entry):
            return lambda *args, **kwargs: calls.append((entry, args, kwargs))
    cp.RawModule = RawModule
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(kernels, "_compile_observed", lambda *_: None)
    _compiled_source.cache_clear()
    spec = KernelSpec("supplied_cache_probe", "advance", SPEC.pointers, ("-std=c++17", "-fmad=false"))
    audit = spec.options + ("-arch=sm_120",)
    original = _compiled_source(SOURCE, spec, 1, audit)
    assert _compiled_source(SOURCE, spec, 1, audit) is original
    batched = _compiled_source(SOURCE, spec, 4, audit)
    changed_source = SOURCE.replace("__fadd_rn", "__fsub_rn")
    _compiled_source(changed_source, spec, 4, audit)
    assert len(modules) == 3
    assert modules[0].kwargs == {"code": SOURCE, "options": spec.options, "name_expressions": None}
    assert modules[1].kwargs["options"] == spec.options
    assert modules[1].kwargs["code"] == generate_batch_source(SOURCE, spec, 4, audit_options=audit)
    assert original[1] == original[2] == sha256(SOURCE.encode()).hexdigest()
    assert batched[1] == original[1] and batched[2] != original[2]
    records = {key: row for key, row in kernel_manifest().items() if "supplied_cache_probe" in key}
    assert len(records) == 3
    assert {row["source_sha256"] for row in records.values()} == {
        sha256(module.kwargs["code"].encode()).hexdigest() for module in modules}
    assert all(row["options"] == list(spec.options) for row in records.values())
    _compiled_source.cache_clear()


def test_pointer_spec_dtype_must_equal_cuda_pointee_type():
    source = SOURCE.replace("real *state", "double *state")
    with pytest.raises(BatchKernelUnsupported, match="declares double"):
        generate_batch_source(source, SPEC, 4)
    spec = KernelSpec("sample", "advance", (
        PointerSpec("shared", "shared"), PointerSpec("state", "member", "float64"),
    ))
    assert "reinterpret_cast<double*>" in generate_batch_source(source, spec, 4)


def test_integer_pointer_dtype_is_explicit():
    source = SOURCE.replace("real *state", "unsigned int *state")
    spec = KernelSpec("sample", "advance", (
        PointerSpec("shared", "shared"), PointerSpec("state", "member", "uint32"),
    ))
    assert "reinterpret_cast<unsigned int*>" in generate_batch_source(source, spec, 4)


@pytest.mark.gpu
@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("yface", [0, 1])
def test_real_face_mass_kernel_batch_words_equal_separate_scalar_launches(members, yface):
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("no CUDA GPU / cupy")
    import cupy as cp
    from gpuwm.core import kernels

    ny, nx = 7, 11
    rng = np.random.default_rng(813)
    host = rng.uniform(60000.0, 100000.0, (members, ny, nx)).astype(np.float32)
    source = cp.asarray(host)
    shape = (ny + yface, nx + 1 - yface)
    result = cp.empty((members,) + shape, dtype=cp.float32)
    reference = cp.empty_like(result)
    spec = KernelSpec("face_mass", "average_mass_faces", (
        PointerSpec("mu", "member"), PointerSpec("out", "member"),
    ))
    grid, block = ((int(np.prod(shape)) + 31) // 32,), (32,)
    scalars = (np.int32(ny), np.int32(nx), np.int32(yface))
    scalar_kernel = kernels.get_kernel("face_mass", "average_mass_faces")
    for member in range(members):
        scalar_kernel(grid, block, (source[member], reference[member]) + scalars)
    if members == 1:
        get_batch_kernel(spec, members)(grid, block, (source[0], result[0]) + scalars)
    else:
        get_batch_kernel(spec, members)(
            grid, block, (source, result) + scalars,
            pointer_strides={"mu": host[0].nbytes,
                             "out": int(np.prod(shape)) * 4})
    cp.cuda.get_current_stream().synchronize()
    assert cp.asnumpy(result).tobytes() == cp.asnumpy(reference).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("hypso", [1, 2])
@pytest.mark.parametrize("terrain_opt", [0, 1])
def test_real_physical_diagnostics_batch_words_equal_scalar(members, hypso, terrain_opt):
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("no CUDA GPU / cupy")
    import cupy as cp
    from gpuwm import wrf_exact
    from gpuwm.config import RunConfig
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    if wrf_exact.ENABLED:
        pytest.skip("this physical fixture grades the default diagnostic; exact variants need their canonical WRF inputs")
    nz, ny, nx = 16, 5, 9
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0,
                    ztop=6400.0, dt=0.5, run_seconds=1.0, moist=True,
                    terrain_opt=terrain_opt, hypsometric_opt=hypso)
    terrain = (50.0 + 20.0 * np.sin(np.arange(ny)[:, None])
               + np.zeros((ny, nx))) if terrain_opt else None
    coord = make_vertical_coord(nz)
    base = make_base_state(coord, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=terrain)
    rng = np.random.default_rng(417)
    inputs = {
        "thp": cp.asarray(rng.normal(0, 0.3, (members, nz, ny, nx)).astype(np.float32)),
        "php": cp.asarray(rng.normal(0, 0.1, (members, nz + 1, ny, nx)).astype(np.float32)),
        "mup": cp.asarray(rng.normal(0, 20, (members, ny, nx)).astype(np.float32)),
        "qv": cp.asarray(rng.uniform(0.004, 0.012, (members, nz, ny, nx)).astype(np.float32)),
    }
    outputs = {name: cp.full((members, nz, ny, nx), cp.nan, dtype=cp.float32)
               for name in ("p", "al", "alt")}
    reference = {name: cp.empty_like(array) for name, array in outputs.items()}
    for member in range(members):
        for name, array in inputs.items():
            getattr(state, name)[...] = array[member]
        update_diagnostics(state, hypso)
        for name in outputs:
            reference[name][member] = getattr(state, name)
    aliases = {"dphbr": "dphb_resid", "mub": "mub2d"}
    shared_names = ("thb", "phb", "dphbr", "alb", "rdnw", "c1h", "c2h", "c3h",
                    "c4h", "c3f", "c4f", "dc3f", "dc4f", "mub")
    shared = {name: getattr(state, aliases.get(name, name)) for name in shared_names}
    pointer_names = ("thp", "php", "mup", "thb", "phb", "dphbr", "alb", "rdnw",
                     "c1h", "c2h", "c3h", "c4h", "c3f", "c4f", "dc3f", "dc4f",
                     "mub", "qv", "p", "al", "alt")
    spec = KernelSpec("diagnostics", "calc_p_alpha", tuple(
        PointerSpec(name, "shared" if name in shared else "member") for name in pointer_names))
    arguments = {**inputs, **shared, **outputs}
    first = tuple(arguments[name] for name in pointer_names[:-3])
    scalars = (np.float32(state.p_top), *(np.int32(value) for value in
               (hypso, 1, terrain_opt, nz, ny, nx, 0, 0, ny, nx)))
    args = first + scalars + tuple(outputs[name] for name in ("p", "al", "alt"))
    strides = {name: 0 if name in shared else arguments[name].nbytes // members
               for name in pointer_names}
    get_batch_kernel(spec, members)(((ny * nx + 127) // 128,), (128,), args,
                                     pointer_strides=strides)
    cp.cuda.get_current_stream().synchronize()
    for name, array in outputs.items():
        actual, expected = cp.asnumpy(array), cp.asnumpy(reference[name])
        assert np.isfinite(actual).all(), name
        assert actual.tobytes() == expected.tobytes(), (name, members, hypso, terrain_opt)


@pytest.mark.gpu
@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("yface", [0, 1])
def test_real_padded_member_views_preserve_face_mass_words_and_guards(members, yface):
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("no CUDA GPU / cupy")
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    ny, nx, before, after = 7, 11, 3, 5
    source_words = ny * nx
    shape = (ny + yface, nx + 1 - yface)
    target_words = int(np.prod(shape))
    guard = np.uint32(0xA5A5A5A5)
    source_backing = cp.full((members, before + source_words + after), guard, dtype=cp.uint32)
    target_backing = cp.full((members, before + target_words + after), guard, dtype=cp.uint32)
    source = source_backing.view(cp.float32)[:, before:before + source_words].reshape((members, ny, nx))
    target = target_backing.view(cp.float32)[:, before:before + target_words].reshape((members,) + shape)
    assert source.data.mem.ptr == source_backing.data.mem.ptr
    assert target.data.mem.ptr == target_backing.data.mem.ptr
    if members > 1:
        assert source.strides[0] == source_backing.strides[0]
        assert target.strides[0] == target_backing.strides[0]
    rng = np.random.default_rng(905)
    source[...] = cp.asarray(rng.uniform(60000.0, 100000.0, source.shape).astype(np.float32))
    original_source = cp.asnumpy(source_backing).tobytes()
    reference = cp.empty((members,) + shape, dtype=cp.float32)
    args_tail = (np.int32(ny), np.int32(nx), np.int32(yface))
    grid, block = ((target_words + 31) // 32,), (32,)
    scalar = get_kernel("face_mass", "average_mass_faces")
    for member in range(members):
        scalar(grid, block, (source[member], reference[member]) + args_tail)
    spec = KernelSpec("face_mass", "average_mass_faces", (
        PointerSpec("mu", "member"), PointerSpec("out", "member"),
    ))
    get_batch_kernel(spec, members)(grid, block, (source, target) + args_tail,
                                     pointer_strides={"mu": source.strides[0], "out": target.strides[0]})
    cp.cuda.get_current_stream().synchronize()
    assert cp.asnumpy(target).tobytes() == cp.asnumpy(reference).tobytes()
    expected = np.full(target_backing.shape, guard, dtype=np.uint32)
    expected[:, before:before + target_words] = cp.asnumpy(reference).view(np.uint32).reshape(members, target_words)
    assert cp.asnumpy(target_backing).tobytes() == expected.tobytes()
    assert cp.asnumpy(source_backing).tobytes() == original_source
    target_backing.fill(guard)
    prepared = prepare_batch_kernel_launch(
        spec, members, grid, block, (source, target) + args_tail,
        pointer_strides={"mu": source.strides[0], "out": target.strides[0]})
    prepared()
    cp.cuda.get_current_stream().synchronize()
    assert cp.asnumpy(target_backing).tobytes() == expected.tobytes()
    assert cp.asnumpy(source_backing).tobytes() == original_source


@pytest.mark.gpu
def test_real_prepared_launch_rebinding_uses_only_the_explicit_new_arrays():
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("no CUDA GPU / cupy")
    import cupy as cp
    members, ny, nx = 4, 3, 7
    first_source = cp.ones((members, ny, nx), dtype=cp.float32)
    first_output = cp.empty((members, ny, nx + 1), dtype=cp.float32)
    second_source = cp.full_like(first_source, 3)
    second_output = cp.empty_like(first_output)
    spec = KernelSpec("face_mass", "average_mass_faces", (
        PointerSpec("mu", "member"), PointerSpec("out", "member")))
    scalars = (np.int32(ny), np.int32(nx), np.int32(0))
    strides = {"mu": first_source.strides[0], "out": first_output.strides[0]}
    first = prepare_batch_kernel_launch(spec, members, (1,), (128,),
                                        (first_source, first_output) + scalars, pointer_strides=strides)
    second = prepare_batch_kernel_launch(spec, members, (1,), (128,),
                                         (second_source, second_output) + scalars, pointer_strides=strides)
    first_source[...] = 2
    first()
    second()
    cp.cuda.get_current_stream().synchronize()
    assert np.all(cp.asnumpy(first_output) == 2)
    assert np.all(cp.asnumpy(second_output) == 3)
    assert first.binding_receipt["arrays"][0]["pointer"] == first_source.data.ptr
    assert second.binding_receipt["arrays"][0]["pointer"] == second_source.data.ptr
    assert first.binding_receipt["arrays"][0]["pointer"] != second.binding_receipt["arrays"][0]["pointer"]


@pytest.mark.gpu
def test_real_prepared_launch_refuses_actual_wrong_device_context():
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("no CUDA GPU / cupy")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("actual prepared-launch device mismatch requires two visible CUDA devices")
    with cp.cuda.Device(0):
        source = cp.ones((1, 2, 3), dtype=cp.float32)
        output = cp.empty((1, 2, 4), dtype=cp.float32)
        spec = KernelSpec("face_mass", "average_mass_faces", (
            PointerSpec("mu", "member"), PointerSpec("out", "member")))
        prepared = prepare_batch_kernel_launch(spec, 1, (1,), (32,),
                                               (source, output, np.int32(2), np.int32(3), np.int32(0)))
    with cp.cuda.Device(1):
        with pytest.raises(ValueError, match="belongs to CUDA device 0"):
            prepared()


@pytest.mark.gpu
@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
def test_real_supplied_face_mass_binding_preserves_padded_guards_and_scalar_words(members):
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("no CUDA GPU / cupy")
    import cupy as cp
    from hashlib import sha256
    from gpuwm.core import kernels
    from gpuwm.ensemble.batch_kernel import _effective_options

    ny, nx, guard_words = 3, 11, 3
    shape = (ny, nx + 1)
    source_words, target_words = ny * nx, int(np.prod(shape))
    guard = np.uint32(0xA3A3A3A3)
    source_backing = cp.full((members, source_words + 2 * guard_words), guard, dtype=cp.uint32)
    target_backing = cp.full((members, target_words + 2 * guard_words), guard, dtype=cp.uint32)
    source = source_backing.view(cp.float32)[:, guard_words:guard_words + source_words].reshape(members, ny, nx)
    target = target_backing.view(cp.float32)[:, guard_words:guard_words + target_words].reshape((members,) + shape)
    host = np.random.default_rng(906).uniform(60000, 100000, source.shape).astype(np.float32)
    source[...] = cp.asarray(host)
    original_source = cp.asnumpy(source_backing).tobytes()
    reference = cp.empty((members,) + shape, dtype=cp.float32)
    spec = KernelSpec("face_mass", "average_mass_faces", (
        PointerSpec("mu", "member"), PointerSpec("out", "member")))
    unit = kernels.module_source("face_mass")
    grid, block = ((target_words + 31) // 32,), (32,)
    scalars = (np.int32(ny), np.int32(nx), np.int32(0))
    scalar = kernels.get_kernel(spec.module, spec.entry)
    prepared = prepare_batch_source_launch(unit, spec, members, grid, block, (source, target) + scalars,
                                           pointer_strides={"mu": source.strides[0], "out": target.strides[0]})
    for member in range(members):
        scalar(grid, block, (source[member], reference[member]) + scalars)
    prepared()
    cp.cuda.get_current_stream().synchronize()
    assert cp.asnumpy(target).tobytes() == cp.asnumpy(reference).tobytes()
    assert cp.asnumpy(source_backing).tobytes() == original_source
    expected = np.full(target_backing.shape, guard, dtype=np.uint32)
    expected[:, guard_words:guard_words + target_words] = cp.asnumpy(reference).view(np.uint32).reshape(members, target_words)
    assert cp.asnumpy(target_backing).tobytes() == expected.tobytes()
    receipt = prepared.binding_receipt
    assert receipt["source_sha256"] == sha256(unit.encode()).hexdigest()
    assert receipt["options"] == _effective_options(spec.options)
    if members == 1:
        assert receipt["compiled_source_sha256"] == receipt["source_sha256"]
    else:
        assert receipt["compiled_source_sha256"] != receipt["source_sha256"]
    assert receipt["arrays"][0]["pointer"] == source.data.ptr
    assert receipt["arrays"][0]["allocation_pointer"] == source_backing.data.mem.ptr
    # Reuse the fixed binding after an in-place input update. The original
    # scalar handle sees precisely the same new bytes and member-local views.
    source.fill(np.float32(70113))
    for member in range(members):
        scalar(grid, block, (source[member], reference[member]) + scalars)
    prepared()
    cp.cuda.get_current_stream().synchronize()
    assert cp.asnumpy(target).tobytes() == cp.asnumpy(reference).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
def test_real_supplied_signature_expansion_matches_original_macro_kernel_words(members):
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("no CUDA GPU / cupy")
    import cupy as cp
    from hashlib import sha256
    from gpuwm.ensemble.batch_kernel import _effective_options

    signature = "const unsigned int* src, unsigned int* dst, int count"
    original = '''#define GRID_ARGS const unsigned int* src, unsigned int* dst, int count
extern "C" __global__ void supplied_macro_probe(GRID_ARGS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = src[i] ^ (gridDim.x + 0x1300u);
}
'''
    supplied = original.replace("void supplied_macro_probe(GRID_ARGS)", "void supplied_macro_probe(" + signature + ")")
    assert supplied[supplied.index("{"):] == original[original.index("{"):]
    spec = KernelSpec("supplied_macro", "supplied_macro_probe", (
        PointerSpec("src", "member", "uint32"), PointerSpec("dst", "member", "uint32")))
    count = 77
    host = np.random.default_rng(907).integers(0, 2**32, (members, count), dtype=np.uint32)
    source, target = cp.asarray(host), cp.empty((members, count), dtype=cp.uint32)
    reference = cp.empty_like(target)
    grid, block = (3,), (32,)
    scalar_module = cp.RawModule(code=original, options=_effective_options(spec.options), name_expressions=None)
    scalar = scalar_module.get_function(spec.entry)
    for member in range(members):
        scalar(grid, block, (source[member], reference[member], np.int32(count)))
    prepared = prepare_batch_source_launch(supplied, spec, members, grid, block,
                                           (source, target, np.int32(count)),
                                           pointer_strides={"src": source.strides[0], "dst": target.strides[0]})
    prepared()
    cp.cuda.get_current_stream().synchronize()
    assert cp.asnumpy(target).tobytes() == cp.asnumpy(reference).tobytes()
    assert prepared.binding_receipt["source_sha256"] == sha256(supplied.encode()).hexdigest()
    if members == 1:
        assert prepared.binding_receipt["compiled_source_sha256"] == sha256(supplied.encode()).hexdigest()
