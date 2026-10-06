"""Architecture rounding diagnostics retain local differences and coverage."""
from pathlib import Path
import os
import json
import shutil
import subprocess
import sys

import pytest

from tools.cuda_rounding_check import (floating_signatures, inventory,
                                       signature_differences,
                                       capture_compiler_sources, compare_unit,
                                       audit_inventory, inventory_changes,
                                       audit_record, source_units, composed_units,
                                       operation_units)
from tools.literal_division_census import Unit
from tools.cuda_rounding_dataflow import ptx_dataflow_signatures, dataflow_differences
from tools.cuda_rounding_native import sass_floating_signatures, sass_dataflow_signatures

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools/cuda_rounding_baseline.json"


def test_replay_rejects_different_compiler_options_and_missing_variants():
    from tools.cuda_cross_target_replay import compiler_options, validate_artifact_variants

    variants = {name: {"options": compiler_options(100)}
                for name in ("production", "direct_w")}
    validate_artifact_variants(variants, 100, variants)
    variants["production"]["options"] += ("--fmad=false",)
    with pytest.raises(AssertionError, match="options differ from production"):
        validate_artifact_variants(variants, 100, variants)
    with pytest.raises(AssertionError, match="variants are incomplete"):
        validate_artifact_variants({}, 120, ("production", "direct_w"))


def _ptx(first, second):
    return f"""
.visible .entry example() {{
 .loc 1 10 1
 {first}
 .loc 1 20 1
 {second}
}}
"""


def test_equal_total_fmas_cannot_hide_a_contraction_at_another_site():
    fma = "fma.rn.f32 %f1, %f2, %f3, %f4;"
    mul_add = "mul.rn.f32 %f1, %f2, %f3;\nadd.rn.f32 %f1, %f1, %f4;"
    left = floating_signatures(_ptx(fma, mul_add))
    right = floating_signatures(_ptx(mul_add, fma))
    assert sum(count["fma.rn.f32"] for count in left.values()) == 1
    assert sum(count["fma.rn.f32"] for count in right.values()) == 1
    differences = signature_differences(left, right)
    assert [row["assembled_line"] for row in differences] == [10, 20]
    assert all(row["contraction_changed"] for row in differences)


def test_equal_counts_at_the_same_site_cannot_hide_the_opposite_rounded_product():
    loads = """
.visible .entry example() {
 .loc 1 10 1
 ld.param.f32 %f1, [a];
 ld.param.f32 %f2, [b];
 ld.param.f32 %f3, [c];
 ld.param.f32 %f4, [d];
"""
    left = loads + "mul.rn.f32 %f5, %f3, %f4;\nfma.rn.f32 %f6, %f1, %f2, %f5;\n}"
    right = loads + "mul.rn.f32 %f5, %f1, %f2;\nfma.rn.f32 %f6, %f3, %f4, %f5;\n}"
    assert signature_differences(floating_signatures(left), floating_signatures(right)) == []
    assert dataflow_differences(ptx_dataflow_signatures(left), ptx_dataflow_signatures(right))
    renamed = left.replace("%f", "%r")
    assert dataflow_differences(ptx_dataflow_signatures(left), ptx_dataflow_signatures(renamed)) == []


def test_native_equal_ffma_counts_cannot_hide_the_opposite_rounded_product():
    prefix = '''
.section .text.example,"ax",@progbits
//## File "unit.cu", line 10
/*0000*/ LDC R1, c[0x0][0x160];
/*0010*/ LDC R2, c[0x0][0x164];
/*0020*/ LDC R3, c[0x0][0x168];
/*0030*/ LDC R4, c[0x0][0x16c];
'''
    left = prefix + "/*0040*/ FMUL R5, R3, R4;\n/*0050*/ FFMA R6, R1, R2, R5;\n"
    right = prefix + "/*0040*/ FMUL R5, R1, R2;\n/*0050*/ FFMA R6, R3, R4, R5;\n"
    assert sass_floating_signatures(left) == sass_floating_signatures(right)
    assert dataflow_differences(sass_dataflow_signatures(left), sass_dataflow_signatures(right))


def test_native_inline_info_retains_the_helper_and_the_call_site():
    source = '''
.section .text.example,"ax",@progbits
//## File "unit.cu", line 10 inlined at "unit.cu", line 20
//## File "unit.cu", line 20
/*0000*/ FFMA R1, R2, R3, R4;
'''
    site, = sass_floating_signatures(source)
    assert site[2] == 10
    assert site[4] == ((1, 20, 0),)


def test_native_negative_zero_ffma_is_a_multiply_and_positive_zero_is_distinct():
    prefix = '''
.section .text.example,"ax",@progbits
//## File "unit.cu", line 10
/*0000*/ LDC R1, c[0x0][0x160];
/*0010*/ LDC R2, c[0x0][0x164];
'''
    multiply = prefix + "/*0020*/ FMUL.FTZ R3, R1, R2;\n"
    negative_zero = prefix + "/*0020*/ FFMA.FTZ R3, R1, R2, -RZ;\n"
    positive_zero = prefix + "/*0020*/ FFMA.FTZ R3, R1, R2, RZ;\n"
    assert sass_floating_signatures(multiply) == sass_floating_signatures(negative_zero)
    assert sass_floating_signatures(multiply) != sass_floating_signatures(positive_zero)
    assert not dataflow_differences(sass_dataflow_signatures(multiply), sass_dataflow_signatures(negative_zero))


def test_register_renaming_and_scheduling_do_not_create_differences():
    left = floating_signatures(_ptx("fma.rn.f32 %f1, %f2, %f3, %f4;", ""))
    right = floating_signatures(_ptx("fma.rn.f32 %f10, %f20, %f30, %f40;", ""))
    assert signature_differences(left, right) == []


@pytest.mark.parametrize("changed", ["fma.rn.f64", "fma.rz.f32", "fma.rn.ftz.f32"])
def test_precision_rounding_and_flush_mode_remain_part_of_the_signature(changed):
    left = floating_signatures(_ptx("fma.rn.f32 %f1, %f2, %f3, %f4;", ""))
    right = floating_signatures(_ptx(f"{changed} %f1, %f2, %f3, %f4;", ""))
    assert signature_differences(left, right)


def test_no_line_information_cannot_silently_pass():
    left = floating_signatures(".visible .entry a() {\nmul.rn.f32 %f1, %f2, %f3;\n}")
    right = floating_signatures(".visible .entry a() {\nfma.rn.f32 %f1, %f2, %f3, %f4;\n}")
    assert signature_differences(left, right)[0]["assembled_line"] == 0


def test_debug_line_movement_inside_one_statement_is_not_arithmetic_drift():
    source = "float scaled =\n    value * 16.0f;\nfloat out = scaled + other;\n"
    left = ".visible .entry a() {\n.loc 1 1 3\nmul.rn.f32 %f1, %f2, 16;\n}"
    right = ".visible .entry a() {\n.loc 1 2 12\nmul.rn.f32 %f3, %f4, 16;\n}"
    assert signature_differences(floating_signatures(left, source),
                                 floating_signatures(right, source)) == []


def test_inventory_names_every_cuda_file_and_dynamic_compiler_site():
    found = inventory(ROOT)
    assert {row["file"] for row in found["cuda_files"]} == {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "gpuwm").rglob("*")
        if path.suffix in (".cu", ".cuh")}
    assert any(row["file"] == "gpuwm/core/kernels/mynn_surface.cu"
               for row in found["cuda_files"])
    assert any(row["file"] == "gpuwm/ensemble/batch_kernel.py"
               and row["status"] == "unresolved" for row in found["compile_sites"])
    assert any(row["file"] == "gpuwm/ensemble/batch_perturbation.py"
               and row["status"] == "resolved_literal" for row in found["compile_sites"])


def test_new_cuda_literals_and_unresolved_templates_are_inventoried(tmp_path):
    package = tmp_path / "gpuwm"
    package.mkdir()
    (package / "new.py").write_text('''
SOURCE = 'extern "C" __global__ void added(float *out) { out[0] = 2.0f; }'
module = cp.RawModule(code=SOURCE, options=("-std=c++17",))
dynamic = cp.RawKernel(make_source(), "added")
''', encoding="utf-8")
    found = inventory(tmp_path)
    assert found["counts"]["python_cuda_literals"] == 1
    assert found["counts"]["compile_sites"] == 2
    assert found["counts"]["resolved_literal_sites"] == 1
    assert [row["status"] for row in found["compile_sites"]] == ["resolved_literal", "unresolved"]


def test_local_literal_factory_is_resolved_and_changed_generated_math_invalidates_audit(tmp_path):
    package = tmp_path / "gpuwm"
    package.mkdir()
    target = package / "new.py"
    target.write_text('''
def build():
    source = 'extern "C" __global__ void f(float *a) { a[0] = a[0]*2.0f + 1.0f; }'
    return cp.RawModule(code=source, options=("-std=c++17",))
''', encoding="utf-8")
    before = inventory(tmp_path)
    assert before["compile_sites"][0]["status"] == "resolved_literal"
    target.write_text(target.read_text().replace("*2.0f", "*3.0f"), encoding="utf-8")
    changed = inventory_changes(audit_inventory(inventory(tmp_path)), audit_inventory(before))
    assert any(row.startswith("compile_sites:") for row in changed)
    assert any(row.startswith("python_compiler_files:") for row in changed)


def test_every_cuda_source_and_compiler_factory_is_bound_to_the_rounding_audit():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    changes = inventory_changes(audit_inventory(inventory(ROOT)), baseline["inventory"])
    assert not changes, (
        "CUDA source or compiler factory changed without a target-rounding audit. "
        "A new implicit multiply-add choice can change weather words on sm_100 versus sm_120. "
        "Compile both targets, review actual arithmetic, and record unresolved diagnostics explicitly.\n"
        + "\n".join(changes[:20]))


def test_actual_source_capture_keeps_options_and_restores_hooks_on_error(tmp_path):
    import json

    class Program:
        src = 'extern "C" __global__ void captured(float *x) { x[0] += 1.0f; }'

        def compile(self, options, log_stream=None):
            assert options == ("-arch=compute_120", "--ftz=false")
            return "compiled"

    class Compiler:
        _NVRTCProgram = Program

        @staticmethod
        def _compile_with_cache_cuda(source, options):
            assert options == ("-std=c++17",)
            return "cache_hit"

    original_cache = Compiler._compile_with_cache_cuda
    original_compile = Program.compile
    target = tmp_path / "nested" / "capture.json"
    with pytest.raises(ValueError, match="stop"):
        with capture_compiler_sources(target, Compiler):
            assert Compiler._compile_with_cache_cuda(Program.src, ("-std=c++17",)) == "cache_hit"
            assert Program().compile(("-arch=compute_120", "--ftz=false")) == "compiled"
            raise ValueError("stop")
    assert Compiler._compile_with_cache_cuda is original_cache
    assert Program.compile is original_compile
    units = json.loads(target.read_text())["units"]
    assert len(units) == 2
    assert units[0]["options"] == ["-std=c++17", "-ftz=true", "--device-as-default-execution-space"]
    assert units[1]["options"] == ["-arch=compute_120", "--ftz=false"]


@pytest.fixture(scope="module")
def compiled_rounding_audit(tmp_path_factory):
    # Compilation is CPU-only and is enabled whenever a capable toolkit is
    # installed. It never requires an opt-in variable or a GPU owner hold.
    temporary = tmp_path_factory.mktemp("rounding-native")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1",
                       TMPDIR=str(temporary), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    probe = subprocess.run([sys.executable, "-m", "tools.cuda_rounding_check", "--compiler-info"],
                           cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
    assert probe.returncode == 0, probe.stderr[-4000:]
    information = json.loads(probe.stdout)
    if "unavailable" in information:
        pytest.skip(f"NVRTC compiler is unavailable: {information['unavailable']}")
    if not {100, 120}.issubset(information["supported_architectures"]):
        pytest.skip("installed NVRTC cannot target both sm_100 and sm_120")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    library = os.environ.get("GPUWM_NVRTC_LIBRARY")
    native_tool = os.environ.get("GPUWM_NVDISASM") or shutil.which("nvdisasm")
    if not library and os.environ.get("CUDA_PATH"):
        candidate = Path(os.environ["CUDA_PATH"]) / "lib" / "libnvrtc.so.13"
        if candidate.exists():
            library = str(candidate)
    if native_tool and not library:
        pytest.fail("nvdisasm is installed, but the matching NVRTC library was not found")
    result_file = temporary / "audit.json"
    command = [sys.executable, "-m", "tools.cuda_rounding_check", "--root", str(ROOT),
               "--arch", "120", "--arch", "100", "--operations", "--json", str(result_file),
               "--report-only"]
    if native_tool:
        command += ["--nvrtc-library", library, "--nvdisasm", native_tool]
    process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True,
                             timeout=3600)
    assert process.returncode == 0, process.stderr[-4000:]
    compiled = json.loads(result_file.read_text(encoding="utf-8"))
    results = [(row["unit"], audit_record(row)) for row in compiled["units"]]
    return baseline, results, bool(native_tool)


def test_every_compiled_source_keeps_its_sm100_sm120_rounding_diagnostics(compiled_rounding_audit):
    baseline, results, native = compiled_rounding_audit
    changes = []
    for key, record in results:
        expected = baseline["units"].get(key)
        if record["coverage"] != "compiled":
            changes.append(f"{key}: compile failed {record['errors']} {record['native_errors']}")
        elif expected is None:
            changes.append(f"{key}: new compiled source has no target-rounding audit")
        else:
            checked = ("source_sha256", "options", "differences", "dataflow_differences")
            if native:
                checked += ("native_differences", "native_dataflow_differences")
            for field in checked:
                if record[field] != expected[field]:
                    changes.append(f"{key}: {field} changed")
    expected_keys = {key for key in baseline["units"] if not key.startswith("captured:")}
    actual_keys = {key for key, _ in results}
    changes.extend(f"{key}: audited source is no longer compiled" for key in sorted(expected_keys - actual_keys))
    assert not changes, (
        "Target rounding diagnostics changed. Equal FFMA totals do not establish identity; "
        "review the operand DAG and replay changed weather kernels before refreshing the audit.\n"
        + "\n".join(changes[:30]))
