"""Replay advection compiler targets and an optional original source control.

Output words are compared, including the W surface, interior and lid rows.
Retargeted PTX is diagnostic: all cubins share one native linker target.
Paired physical-card forecasts remain necessary for hardware/ptxas identity.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cuda_cross_target_replay import TARGETS, NvJitLink, digest, retarget_ptx

SCENARIOS = tuple(product((False, True), repeat=6))
CERTIFICATION_TARGETS = (100, 120)
DEVICE_DEFAULT_OPTION = "--device-as-default-execution-space"


def effective_options(target, *, native=False):
    from gpuwm.core.kernels import module_options
    return module_options("advection") + (
        "-ftz=true", f"-arch={'sm' if native else 'compute'}_{target}", DEVICE_DEFAULT_OPTION)


def assembled_source(raw_source=None):
    from gpuwm.core.kernels import module_source
    source = module_source("advection")
    if raw_source is None:
        return source
    import gpuwm.core.kernels as kernels
    current = (Path(kernels.__file__).parent / "advection.cu").read_text(encoding="utf-8")
    if not source.endswith(current):
        raise AssertionError("advection source assembly has an unexpected suffix")
    return source[:-len(current)] + Path(raw_source).read_text(encoding="utf-8")


def compile_advection(outdir, host_sm, *, candidate_source=None, baseline_source=None):
    from gpuwm.core.kernels import module_options
    from gpuwm.nvrtc_cache_key import compile_program

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    candidate = assembled_source(candidate_source)
    source_set = {f"candidate_{target}": (candidate, target) for target in TARGETS}
    runtime_sources = {"native_candidate_120": candidate}
    if baseline_source is not None:
        baseline = assembled_source(baseline_source)
        source_set.update({f"original_{target}": (baseline, target)
                           for target in CERTIFICATION_TARGETS})
        runtime_sources["native_original_120"] = baseline
    linker = NvJitLink()
    receipt = {"schema": "gpuwm-advection-cross-target-compile-v1", "host_sm": int(host_sm),
               "candidate_source_sha256": digest(candidate.encode()),
               "nvjitlink_version": linker.version, "variants": {}, "runtime_ftz": True,
               "certification_targets": CERTIFICATION_TARGETS, "native_runtime": {},
               "limitations": "diagnostic PTX target rewrite; one native linker target; physical target pairs remain necessary"}
    if int(host_sm) == 120:
        for name, source in runtime_sources.items():
            filename = name + "-source.cu"
            (outdir / filename).write_text(source, encoding="utf-8")
            receipt["native_runtime"][name] = {
                "source": filename, "source_sha256": digest(source.encode()),
                "options": module_options("advection"),
                "effective_options": effective_options(120, native=True),
                "method": "production RawModule loader; native physical sm_120"}
    for name, (source, target) in source_set.items():
        # RawModule appends FTZ at runtime; the diagnostic compile must too.
        options = effective_options(target)
        row = receipt["variants"][name] = {"options": options, "source_sha256": digest(source.encode())}
        try:
            ptx = compile_program(source, options)
            ptx = ptx.encode() if isinstance(ptx, str) else ptx
        except Exception as error:
            message = str(error)
            unsupported = "gpu-architecture" in message and ("invalid value" in message or "not supported" in message)
            row.update(status="unsupported" if unsupported else "compile_error", error=message)
            continue
        replay = retarget_ptx(ptx, host_sm)
        (outdir / (name + ".ptx")).write_bytes(ptx)
        (outdir / (name + "-retargeted.ptx")).write_bytes(replay)
        row.update(ptx_sha256=digest(ptx), replay_ptx_sha256=digest(replay),
                   header_rewrites=[".target"], retargeted=ptx != replay)
        try:
            cubin = linker.link(replay, host_sm, name + ".ptx")
        except Exception as error:
            row.update(status="link_error", error=str(error))
            continue
        (outdir / (name + ".cubin")).write_bytes(cubin)
        row.update(status="linked", cubin=name + ".cubin", cubin_sha256=digest(cubin))
    (outdir / "compile-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def validate_artifacts(outdir, *, candidate_source=None):
    """Reject stale source, missing targets, altered options or damaged cubins."""
    from gpuwm.core.kernels import module_options

    outdir = Path(outdir)
    receipt = json.loads((outdir / "compile-receipt.json").read_text())
    if receipt.get("runtime_ftz") is not True:
        raise AssertionError("advection artifacts omit the runtime compiler's FTZ flag")
    if tuple(receipt.get("certification_targets", ())) != CERTIFICATION_TARGETS:
        raise AssertionError("advection artifacts omit required Blackwell compiler controls")
    if receipt["host_sm"] == 120 and "native_candidate_120" not in receipt.get("native_runtime", {}):
        raise AssertionError("advection artifacts omit the native production-loader control")
    if digest(assembled_source(candidate_source).encode()) != receipt["candidate_source_sha256"]:
        raise AssertionError("advection replay artifacts were compiled from a different candidate source")
    variants = receipt["variants"]
    if not {"candidate_100", "candidate_120"} <= set(variants):
        raise AssertionError("required advection compiler targets are absent")
    if "original_120" in variants and "original_100" not in variants:
        raise AssertionError("original advection source omits its compute_100 control")
    if receipt["host_sm"] == 120 and "original_120" in variants and "native_original_120" not in receipt["native_runtime"]:
        raise AssertionError("original advection source omits its native production-loader control")
    for name, row in receipt["variants"].items():
        if row["status"] == "unsupported" and name not in (
                "candidate_100", "candidate_120", "original_100", "original_120"):
            continue
        if row["status"] != "linked":
            raise AssertionError(f"advection variant {name} failed: {row}")
        target = int(name.rsplit("_", 1)[1])
        if tuple(row["options"]) != effective_options(target):
            raise AssertionError("advection compile options differ from the effective runtime options")
        expected_source = (receipt["candidate_source_sha256"] if name.startswith("candidate_")
                           else variants["original_120"]["source_sha256"])
        if row["source_sha256"] != expected_source:
            raise AssertionError("advection variants were compiled from different source controls")
        path = outdir / row["cubin"]
        if digest(path.read_bytes()) != row["cubin_sha256"]:
            raise AssertionError("advection cubin checksum changed")
    for name, row in receipt.get("native_runtime", {}).items():
        source = (outdir / row["source"]).read_text(encoding="utf-8")
        if digest(source.encode()) != row["source_sha256"]:
            raise AssertionError("native advection control source checksum changed")
        if tuple(row["options"]) != module_options("advection"):
            raise AssertionError("native advection control differs from production options")
        if tuple(row.get("effective_options", ())) != effective_options(120, native=True):
            raise AssertionError("native advection control omits effective runtime options")
        expected_source = (receipt["candidate_source_sha256"] if name == "native_candidate_120"
                           else variants["original_120"]["source_sha256"])
        if row["source_sha256"] != expected_source:
            raise AssertionError("native advection source does not match its compiled source control")
    return receipt


def load_advection(outdir, *, candidate_source=None):
    receipt = validate_artifacts(outdir, candidate_source=candidate_source)
    import cupy as cp
    from gpuwm.core.kernels import module_options
    if int(cp.cuda.Device().compute_capability) != receipt["host_sm"]:
        raise AssertionError("advection cubins belong to a different physical GPU target")
    outdir = Path(outdir)
    modules = {name: cp.RawModule(path=str(outdir / row["cubin"]))
               for name, row in receipt["variants"].items() if row["status"] == "linked"}
    for name, row in receipt.get("native_runtime", {}).items():
        source = (outdir / row["source"]).read_text(encoding="utf-8")
        modules[name] = cp.RawModule(code=source, options=module_options("advection"),
                                     name_expressions=None)
    return modules, receipt


def fixture_calls(scenario):
    import cupy as cp
    import numpy as np
    from gpuwm.core.grid import make_vertical_coord

    boundary_x, boundary_y, has_msf, specified, stretched, vertical = scenario
    nz, ny, nx = 12, 11, 17
    random = np.random.default_rng(497)
    coord = make_vertical_coord(nz, stretch=1.5 if stretched else None)

    def normal(shape, mean, sigma):
        return cp.asarray(random.normal(mean, sigma, shape), dtype=cp.float32)

    ru = normal((nz, ny, nx + 1), 0.0, 10.0)
    rv = normal((nz, ny + 1, nx), 0.0, 10.0)
    rw = normal((nz + 1, ny, nx), 0.0, 1.0)
    if not vertical:
        rw.fill(cp.float32(0))
    else:
        rw[0].fill(cp.float32(0))
    calls = []
    for kernel, shape, mean, sigma in (
        ("flux_div_u", (nz, ny, nx + 1), 0.0, 10.0),
        ("flux_div_v", (nz, ny + 1, nx), 0.0, 10.0),
        ("flux_div_w", (nz + 1, ny, nx), 0.0, 5.0),
        ("flux_div_scalar", (nz, ny, nx), 300.0, 5.0),
    ):
        field = normal(shape, mean, sigma)
        initial = normal(shape, 0.0, 0.2)
        map_factor = cp.asarray(random.uniform(0.85, 1.15, shape[-2:]), dtype=cp.float32)
        spacing = coord.rdn if kernel == "flux_div_w" else coord.rdnw
        args = (field, ru, rv, rw, initial, cp.asarray(spacing, dtype=cp.float32),
                cp.asarray(coord.fnm, dtype=cp.float32), cp.asarray(coord.fnp, dtype=cp.float32),
                map_factor, np.float32(1.0 / 900.0), np.float32(1.0 / 1100.0),
                np.int32(nz), np.int32(ny), np.int32(nx),
                np.int32(boundary_x), np.int32(boundary_y), np.int32(has_msf), np.int32(specified))
        calls.append((kernel, (1, shape[-2], shape[0]), args))
    return calls


def replay_scenario(modules, scenario):
    import cupy as cp
    import numpy as np

    rows = []
    for kernel, grid, arguments in fixture_calls(scenario):
        input_parts = []
        for index, arg in enumerate(arguments):
            if isinstance(arg, cp.ndarray):
                input_parts += [str((index, arg.dtype.str, arg.shape)).encode(), cp.asnumpy(arg).tobytes()]
            else:
                input_parts.append(str((index, type(arg).__name__, arg)).encode())
        input_sha = digest(b"".join(input_parts))
        outputs = {}
        for name, module in sorted(modules.items()):
            replay = tuple(arg.copy() if isinstance(arg, cp.ndarray) else arg for arg in arguments)
            module.get_function(kernel)(grid, (128, 1, 1), replay)
            cp.cuda.Device().synchronize()
            value = cp.asnumpy(replay[4])
            if not np.all(np.isfinite(value)):
                raise AssertionError(f"nonfinite {kernel} output in {name}")
            outputs[name] = value.view(np.uint32).copy()
        reference = outputs["candidate_120"]
        original = outputs.get("native_original_120", outputs.get("original_120", reference))
        original_compute120 = outputs.get("original_120", reference)
        boundary = np.zeros(reference.shape, dtype=bool)
        boundary[:, 0, :] = boundary[:, -1, :] = True
        boundary[:, :, 0] = boundary[:, :, -1] = True
        for name, value in sorted(outputs.items()):
            different = value != reference
            original_different = value != original
            coordinates = np.argwhere(original_different)
            row = {"kernel": kernel, "variant": name, "input_sha256": input_sha,
                   "output_sha256": digest(value.tobytes()),
                   "different_vs_candidate120": int(np.count_nonzero(different)),
                   "different_vs_original120": int(np.count_nonzero(original_different)),
                   "different_vs_original_compute120": int(np.count_nonzero(value != original_compute120)),
                   "original120_reference": "native_original_120" if "native_original_120" in outputs
                                            else "original_120" if "original_120" in outputs else "candidate_120",
                   "original_difference_boundary_words": int(np.count_nonzero(original_different & boundary)),
                   "original_difference_horizontal_interior_words": int(np.count_nonzero(original_different & ~boundary)),
                   "original_difference_per_vertical_row": np.count_nonzero(original_different, axis=(1, 2)).tolist(),
                   "first_original_differences": [{"index": point.tolist(), "actual_bits": int(value[tuple(point)]),
                                                   "original_bits": int(original[tuple(point)])}
                                                  for point in coordinates[:8]]}
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("compile", "replay"))
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--host-sm", type=int)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    options = parser.parse_args(argv)
    if options.action == "compile":
        if options.host_sm is None:
            parser.error("--host-sm is required for CPU-only compilation")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["GPUWM_NO_LOCAL_GPU"] = "1"
        receipt = compile_advection(options.outdir, options.host_sm,
                                    candidate_source=options.candidate_source, baseline_source=options.baseline_source)
        print(json.dumps({name: row["status"] for name, row in receipt["variants"].items()}, indent=2))
        return int(any(row["status"] not in ("linked", "unsupported") for row in receipt["variants"].values()))
    modules, receipt = load_advection(options.outdir, candidate_source=options.candidate_source)
    cases = [{"scenario": scenario, "kernels": replay_scenario(modules, scenario)} for scenario in SCENARIOS]
    cross_target = sum(row["different_vs_candidate120"] for case in cases for row in case["kernels"]
                       if row["variant"] == "candidate_100")
    diagnostic_cross_target = sum(row["different_vs_candidate120"] for case in cases for row in case["kernels"]
                                  if row["variant"].startswith("candidate_")
                                  and row["variant"] not in ("candidate_100", "candidate_120"))
    original_cross_target = sum(row["different_vs_original_compute120"] for case in cases for row in case["kernels"]
                                if row["variant"] == "original_100")
    native_changes = sum(row["different_vs_candidate120"] for case in cases for row in case["kernels"]
                         if row["variant"] == "native_candidate_120")
    original_native_changes = sum(row["different_vs_original120"] for case in cases for row in case["kernels"]
                                  if row["variant"] == "original_120")
    baseline_variant = ("native_candidate_120" if "native_original_120" in modules
                        else "candidate_120" if "original_120" in modules else None)
    blackwell_changes = sum(row["different_vs_original120"] for case in cases for row in case["kernels"]
                           if row["variant"] == baseline_variant)
    report = {"schema": "gpuwm-advection-cross-target-replay-v1", "compile": receipt,
              "scenario_fields": ["boundary_x", "boundary_y", "has_msf", "specified", "stretched", "vertical_flux"],
              "cross_target_different_words": cross_target, "blackwell_changed_words": blackwell_changes,
              "original_blackwell_cross_target_different_words": original_cross_target,
              "diagnostic_other_target_different_words": diagnostic_cross_target,
              "native_candidate_replay_different_words": native_changes,
              "native_original_replay_different_words": original_native_changes,
              "blackwell_source_control": baseline_variant,
              "native_pipeline_limit": "CuPy compiles native sm cubins; compute-target PTX replay can round differently on the same GPU",
              "cases": cases}
    (options.outdir / "replay-receipt.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"scenarios": len(cases), "variants": sorted(modules),
                      "cross_target_different_words": cross_target, "blackwell_changed_words": blackwell_changes,
                      "original_blackwell_cross_target_different_words": original_cross_target,
                      "diagnostic_other_target_different_words": diagnostic_cross_target,
                      "native_candidate_replay_different_words": native_changes,
                      "native_original_replay_different_words": original_native_changes}, indent=2))
    return int(bool(cross_target or blackwell_changes))


if __name__ == "__main__":
    raise SystemExit(main())
