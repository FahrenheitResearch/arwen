"""Exact-word replay of classic Thompson arithmetic across compiler targets.

State inputs are randomized once and cloned for every variant. Canonical
tables are verified immutable inputs. PTX is diagnostically retargeted to one
native GPU; paired physical targets remain necessary for hardware identity.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cuda_cross_target_replay import TARGETS, NvJitLink, digest, retarget_ptx

CERTIFICATION_TARGETS = (100, 120)
DEVICE_DEFAULT_OPTION = "--device-as-default-execution-space"


def effective_options(target, *, native=False, strict=False):
    from gpuwm.core.kernels import module_options
    from gpuwm.wrf_exact import STRICT_OPTIONS
    flags = STRICT_OPTIONS if strict else ("-ftz=true",)
    return module_options("thompson") + flags + (
        f"-arch={'sm' if native else 'compute'}_{target}", DEVICE_DEFAULT_OPTION)


def source(raw=None):
    import gpuwm.core.kernels as kernels
    assembled = kernels.module_source("thompson")
    if raw is None:
        return assembled
    current = (Path(kernels.__file__).parent / "thompson.cu").read_text(encoding="utf-8")
    if not assembled.endswith(current):
        raise AssertionError("Thompson source assembly has an unexpected suffix")
    return assembled[:-len(current)] + Path(raw).read_text(encoding="utf-8")


def compile_thompson(outdir, host_sm, *, candidate_source=None, baseline_source=None):
    from gpuwm.core.kernels import module_options
    from gpuwm.nvrtc_cache_key import compile_program

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    candidate = source(candidate_source)
    variants = {f"candidate_{target}": (candidate, target) for target in TARGETS}
    runtime_sources = {"native_candidate_120": candidate}
    if baseline_source is not None:
        baseline = source(baseline_source)
        variants.update({f"original_{target}": (baseline, target)
                         for target in CERTIFICATION_TARGETS})
        runtime_sources["native_original_120"] = baseline
        variants["strict_candidate_120"] = (candidate, 120)
        variants["strict_original_120"] = (baseline, 120)
    linker = NvJitLink()
    receipt = {"schema": "gpuwm-thompson-cross-target-compile-v1", "host_sm": int(host_sm),
               "source_sha256": digest(candidate.encode()), "runtime_ftz": True,
               "nvjitlink_version": linker.version, "variants": {},
               "certification_targets": CERTIFICATION_TARGETS, "native_runtime": {}}
    if int(host_sm) == 120:
        for name, text in runtime_sources.items():
            filename = name + "-source.cu"
            (outdir / filename).write_text(text, encoding="utf-8")
            receipt["native_runtime"][name] = {
                "source": filename, "source_sha256": digest(text.encode()),
                "options": module_options("thompson"),
                "effective_options": effective_options(120, native=True),
                "method": "production RawModule loader; native physical sm_120"}
    for name, (text, target) in variants.items():
        options = effective_options(target, strict=name.startswith("strict_"))
        row = receipt["variants"][name] = {"options": options, "source_sha256": digest(text.encode())}
        try:
            ptx = compile_program(text, options)
            ptx = ptx.encode() if isinstance(ptx, str) else ptx
            replay = retarget_ptx(ptx, host_sm)
            (outdir / (name + ".ptx")).write_bytes(ptx)
            (outdir / (name + "-retargeted.ptx")).write_bytes(replay)
            cubin = linker.link(replay, host_sm, name + ".ptx")
            (outdir / (name + ".cubin")).write_bytes(cubin)
            row.update(status="linked", cubin=name + ".cubin", cubin_sha256=digest(cubin),
                       ptx_sha256=digest(ptx), replay_ptx_sha256=digest(replay), header_rewrites=[".target"])
        except Exception as error:
            message = str(error)
            unsupported = "gpu-architecture" in message and ("invalid value" in message or "not supported" in message)
            row.update(status="unsupported" if unsupported else "error", error=message)
        (outdir / "compile-receipt.json").write_text(json.dumps(receipt, indent=2))
        print(name, row["status"], flush=True)
    return receipt


def validate_artifacts(outdir, *, candidate_source=None):
    """Reject stale source, missing targets, altered options or damaged cubins."""
    from gpuwm.core.kernels import module_options

    outdir = Path(outdir)
    receipt = json.loads((outdir / "compile-receipt.json").read_text())
    if receipt.get("runtime_ftz") is not True or receipt["source_sha256"] != digest(source(candidate_source).encode()):
        raise AssertionError("Thompson replay source or runtime FTZ does not match")
    if tuple(receipt.get("certification_targets", ())) != CERTIFICATION_TARGETS:
        raise AssertionError("Thompson artifacts omit required Blackwell compiler controls")
    if receipt["host_sm"] == 120 and "native_candidate_120" not in receipt.get("native_runtime", {}):
        raise AssertionError("Thompson artifacts omit the native production-loader control")
    variants = receipt["variants"]
    if not {"candidate_100", "candidate_120"} <= set(variants):
        raise AssertionError("required compiler target artifacts are absent")
    if "original_120" in variants and "original_100" not in variants:
        raise AssertionError("original Thompson source omits its compute_100 control")
    if receipt["host_sm"] == 120 and "original_120" in variants and "native_original_120" not in receipt["native_runtime"]:
        raise AssertionError("original Thompson source omits its native production-loader control")
    for name, row in receipt["variants"].items():
        if row["status"] == "unsupported" and name not in (
                "candidate_100", "candidate_120", "original_100", "original_120"):
            continue
        if row["status"] != "linked":
            raise AssertionError(f"Thompson compile failed: {name}: {row}")
        target = int(name.rsplit("_", 1)[1])
        if tuple(row["options"]) != effective_options(target, strict=name.startswith("strict_")):
            raise AssertionError("Thompson compile options differ from the effective runtime options")
        expected_source = (receipt["source_sha256"] if "candidate_" in name
                           else variants["original_120"]["source_sha256"])
        if row["source_sha256"] != expected_source:
            raise AssertionError("Thompson variants were compiled from different source controls")
        path = outdir / row["cubin"]
        if digest(path.read_bytes()) != row["cubin_sha256"]:
            raise AssertionError("Thompson cubin checksum changed")
    for name, row in receipt.get("native_runtime", {}).items():
        text = (outdir / row["source"]).read_text(encoding="utf-8")
        if digest(text.encode()) != row["source_sha256"]:
            raise AssertionError("native Thompson control source checksum changed")
        if tuple(row["options"]) != module_options("thompson"):
            raise AssertionError("native Thompson control differs from production options")
        if tuple(row.get("effective_options", ())) != effective_options(120, native=True):
            raise AssertionError("native Thompson control omits effective runtime options")
        expected_source = (receipt["source_sha256"] if name == "native_candidate_120"
                           else variants["original_120"]["source_sha256"])
        if row["source_sha256"] != expected_source:
            raise AssertionError("native Thompson source does not match its compiled source control")
    return receipt


def load_thompson(outdir, *, candidate_source=None):
    receipt = validate_artifacts(outdir, candidate_source=candidate_source)
    import cupy as cp
    from gpuwm.core.kernels import module_options
    if receipt["host_sm"] != int(cp.cuda.Device().compute_capability):
        raise AssertionError("Thompson cubins belong to a different native target")
    outdir = Path(outdir)
    modules = {name: cp.RawModule(path=str(outdir / row["cubin"]))
               for name, row in receipt["variants"].items() if row["status"] == "linked"}
    for name, row in receipt.get("native_runtime", {}).items():
        text = (outdir / row["source"]).read_text(encoding="utf-8")
        modules[name] = cp.RawModule(code=text, options=module_options("thompson"),
                                     name_expressions=None)
    return modules, receipt


def fixture_calls(n=517):
    import cupy as cp
    import numpy as np
    from gpuwm.core import thompson as launchers
    from gpuwm.core.thompson_runtime import load_classic_device_tables
    from gpuwm.physics_compat import thompson_table_root

    owner = load_classic_device_tables(thompson_table_root())
    cold = owner.cold_source_tables
    random = np.random.default_rng(817)
    fields = {name: random.uniform(1e-6, 2e-3, n).astype(np.float32)
              for name in ("qi", "qs", "qg", "qr", "qc")}
    fields.update(ni=random.uniform(1.0, 2e5, n).astype(np.float32),
                  nr=random.uniform(1.0, 2e5, n).astype(np.float32),
                  temperature=random.uniform(230.0, 305.0, n).astype(np.float32),
                  pressure=random.uniform(15000.0, 100000.0, n).astype(np.float32),
                  qv=random.uniform(1e-5, 0.018, n).astype(np.float32))
    for name in ("qi", "qs", "qg", "qr", "qc"):
        fields[name][::3] = np.float32(0)
        fields[name][1::7] = np.float32(1e-12)
        fields[name][2::19] = np.float32(-0.0)
    for name in ("effc", "effi", "effs", "reference_density", "reference_temperature",
                 "graupel_number_shadow", "velocity_boost", "snow_velocity_boost",
                 "graupel_melt_marker", "snow_melt_marker", "source_density", "condensation_marker"):
        fields[name] = np.zeros(n, dtype=np.float32)
    fields["graupel_number_shadow"][:] = random.uniform(1.0, 2e5, n).astype(np.float32)
    fields["source_density"][:] = np.float32(0.622) * fields["pressure"] / (
        np.float32(287.04) * fields["temperature"] * (fields["qv"] + np.float32(0.622)))
    device = {name: cp.asarray(value) for name, value in fields.items()}
    arguments = dict(device, dt=6.0,
                     ice_deposition_partition=cold.ice_deposition_partition,
                     ice_to_snow_mass=cold.ice_to_snow_mass, ice_to_snow_number=cold.ice_to_snow_number,
                     rain_snow_tables=cold.rain_snow_tables, rain_graupel_tables=cold.rain_graupel_tables,
                     rain_freezing_tables=cold.rain_freezing_tables,
                     rain_cloud_efficiency=cold.rain_cloud_efficiency,
                     snow_cloud_efficiency=owner.arrays["t_Efsw"],
                     cloud_freezing_tables=cold.cloud_freezing_tables,
                     cloud_to_ice_mass=cold.cloud_freezing_tables[0], cloud_to_ice_number=cold.cloud_freezing_tables[1])
    arguments["graupel_number"] = device["graupel_number_shadow"]
    arguments.update(zip(("rain_to_ice_mass", "rain_to_ice_number",
                          "rain_to_graupel_mass", "rain_to_graupel_number"),
                         cold.rain_freezing_tables))
    mutable = {id(value) for value in device.values()}
    calls = []
    original = launchers.get_kernel

    def capture(unit, kernel):
        assert unit == "thompson"

        def record(grid, block, args):
            outputs = tuple(index for index, arg in enumerate(args)
                            if isinstance(arg, cp.ndarray) and id(arg) in mutable)
            calls.append((kernel, grid, block, tuple(args), outputs))
        return record

    def invoke(name, optional=None):
        function = getattr(launchers, name)
        signature = inspect.signature(function)
        optional = optional or {}
        payload = {key: optional[key] if key in optional else arguments[key]
                   for key, parameter in signature.parameters.items()
                   if parameter.default is inspect.Parameter.empty}
        payload.update(optional)
        function(**payload)

    launchers.get_kernel = capture
    try:
        for name in ("launch_effective_radius", "launch_snow_sublimation", "launch_snow_melting",
                     "launch_snow_cloud_riming", "launch_snow_rime_conversion", "launch_snow_ice_collection",
                     "launch_rain_ice_collection", "launch_warm_process_network", "launch_warm_frozen_source_network",
                     "launch_cold_cloud_source_network", "launch_cold_rain_source_network",
                     "launch_graupel_cloud_riming", "launch_graupel_melting", "launch_ice_deposition",
                     "launch_rain_freezing", "launch_graupel_sublimation", "launch_ice_nucleation",
                     "launch_ice_autoconversion", "launch_cloud_freezing", "launch_warm_rain_collection",
                     "launch_warm_saturation_adjust", "launch_warm_autoconversion",
                     "launch_rain_self_collection", "launch_final_phase_cleanup",
                     "launch_cold_rain_snow_graupel_network"):
            invoke(name)
        invoke("launch_rain_graupel_collection", {"tables": cold.rain_graupel_tables})
        invoke("launch_rain_snow_collection", {"tables": cold.rain_snow_tables})
        invoke("launch_frozen_vapor_network")
        invoke("launch_frozen_vapor_network", {key: arguments[key] for key in (
            "rain_snow_tables", "rain_graupel_tables", "rain_freezing_tables", "qc", "rain_cloud_efficiency",
            "cloud_freezing_tables", "graupel_number_shadow", "snow_velocity_boost")})
        for optional in ({}, {"reference_density": device["reference_density"]},
                         {"reference_density": device["reference_density"], "reference_temperature": device["reference_temperature"]},
                         {"reference_density": device["reference_density"], "graupel_melt_marker": device["graupel_melt_marker"]},
                         {"reference_density": device["reference_density"], "source_density": device["source_density"]},
                         {"reference_density": device["reference_density"], "source_density": device["source_density"],
                          "density_carries_rain_presence": True}):
            invoke("launch_rain_evaporation", optional)
        # The column branch is a separate compiler context. Include dry and
        # cloudy columns, plus vapor values near both saturation boundaries.
        shape = (9, 5, 19)
        column_fields = {
            name: cp.zeros(shape, dtype=cp.float32)
            for name in ("qc", "qi", "qr", "qs", "qg")}
        temp = random.uniform(230.0, 305.0, shape).astype(np.float32)
        pressure = random.uniform(20000.0, 95000.0, shape).astype(np.float32)
        x = np.maximum(-80.0, temp.astype(np.float64) - 273.16)
        liquid = np.polynomial.polynomial.polyval(x, (
            0.611583699e3, 0.444606896e2, 0.143177157e1, 0.264224321e-1,
            0.299291081e-3, 0.203154182e-5, 0.702620698e-8,
            0.379534310e-11, -0.321582393e-13))
        ice = np.polynomial.polynomial.polyval(x, (
            0.609868993e3, 0.499320233e2, 0.184672631e1, 0.402737184e-1,
            0.565392987e-3, 0.521693933e-5, 0.307839583e-7,
            0.105785160e-9, 0.161444444e-12))
        vapor_pressure = np.minimum(np.where(temp <= 273.15, ice, liquid), pressure * 0.15)
        approximate_saturation = 0.622 * vapor_pressure / (pressure - vapor_pressure)
        qv = (approximate_saturation * random.uniform(1.0 - 2e-5, 1.0 + 2e-5, shape)).astype(np.float32)
        column_fields.update(temperature=cp.asarray(temp), pressure=cp.asarray(pressure),
                             qv=cp.asarray(qv), micro_columns=cp.zeros(shape[1:], dtype=cp.float32))
        column_fields["qc"][4, :, ::4] = cp.float32(2e-12)
        mutable.update(id(value) for value in column_fields.values())
        launchers.launch_microphysics_columns(**column_fields)

        volume = {name: cp.asarray(random.uniform(1e-6, 2e-3, shape).astype(np.float32))
                  for name in ("qc", "qi", "qs", "qg", "qr")}
        for value in volume.values():
            value[:, :, ::3] = cp.float32(0)
            value[::2, :, 1::7] = cp.float32(1e-12)
        density = (np.float32(0.622) * pressure /
                   (np.float32(287.04) * temp * (qv + np.float32(0.622)))).astype(np.float32)
        volume.update(temperature=cp.asarray(temp), pressure=cp.asarray(pressure), qv=cp.asarray(qv),
                      ni=cp.asarray(random.uniform(1.0, 2e5, shape).astype(np.float32)),
                      nr=cp.asarray(random.uniform(1.0, 2e5, shape).astype(np.float32)),
                      graupel_number_shadow=cp.zeros(shape, dtype=cp.float32),
                      micro_columns=cp.zeros(shape[1:], dtype=cp.float32),
                      reference_density=cp.asarray(density),
                      dz=cp.asarray(random.uniform(50.0, 300.0, shape).astype(np.float32)),
                      vertical_velocity=cp.asarray(random.uniform(-1.0, 1.0, shape).astype(np.float32)))
        surface = {name: cp.asarray(random.uniform(0.0, 2.0, shape[1:]).astype(np.float32))
                   for name in ("rainnc", "rainncv", "snownc", "snowncv", "graupelncv", "sr")}
        masks = {name: cp.asarray(random.integers(0, 2, shape[1:]).astype(np.float32))
                 for name in ("rain_active_columns", "cloud_active_columns")}
        presence_density = volume["reference_density"].copy()
        presence_density[volume["qr"] <= cp.float32(1e-12)] = cp.float32(0)
        mutable.update(id(value) for value in (*volume.values(), *surface.values(), *masks.values(), presence_density))
        launchers.launch_adapter_entry(**{key: volume[key] for key in (
            "qc", "qi", "qr", "qs", "qg", "temperature", "pressure", "qv",
            "graupel_number_shadow", "micro_columns")})

        for accumulate in (False, True):
            launchers.launch_rain_sedimentation(
                **{key: volume[key] for key in ("qr", "nr", "temperature", "pressure", "qv", "dz")},
                **{key: surface[key] for key in ("rainnc", "rainncv")}, dt=6.0,
                reference_density=presence_density, density_carries_rain_presence=True,
                accumulate_surface=accumulate)
        launchers.launch_ice_sedimentation(
            **{key: volume[key] for key in ("qi", "ni", "temperature", "pressure", "qv", "dz")},
            **{key: surface[key] for key in ("rainnc", "rainncv", "snownc", "snowncv")},
            dt=6.0, reference_density=volume["reference_density"])
        launchers.launch_cloud_sedimentation(
            **{key: volume[key] for key in ("qc", "temperature", "pressure", "qv", "vertical_velocity", "dz")},
            **masks, dt=6.0, reference_density=volume["reference_density"])

        from types import SimpleNamespace
        theta = temp / random.uniform(0.7, 1.0, shape).astype(np.float32)
        finish = {"temperature": volume["temperature"], "pii": cp.asarray(temp / theta),
                  "th": cp.asarray(theta), "thp": cp.asarray(random.normal(0.0, 3.0, shape).astype(np.float32)),
                  "h_diabatic": cp.asarray(theta + random.normal(0.0, 0.3, shape).astype(np.float32)),
                  **{key: surface[key] for key in ("rainncv", "snowncv", "graupelncv", "sr")}}
        mutable.update(id(value) for value in finish.values())
        for no_heating in (0, 1):
            launchers.launch_adapter_finish(**finish, cfg=SimpleNamespace(mp_tend_lim=0.02,
                                             no_mp_heating=no_heating), dt=6.0)
    finally:
        launchers.get_kernel = original
    return calls, mutable, owner.identity_sha256


def replay(modules, n=517):
    import cupy as cp
    import numpy as np

    calls, mutable, tables_sha = fixture_calls(n)
    rows = []
    for kernel, grid, block, arguments, outputs in calls:
        input_parts = [tables_sha.encode()]
        for index, arg in enumerate(arguments):
            if isinstance(arg, cp.ndarray) and id(arg) in mutable:
                input_parts += [str((index, arg.dtype.str, arg.shape)).encode(), cp.asnumpy(arg).tobytes()]
            elif not isinstance(arg, cp.ndarray):
                input_parts.append(str((index, type(arg).__name__, arg)).encode())
        values = {}
        for name, module in sorted(modules.items()):
            copies = {}
            args = []
            for arg in arguments:
                if isinstance(arg, cp.ndarray) and id(arg) in mutable:
                    if id(arg) not in copies:
                        copies[id(arg)] = arg.copy()
                    args.append(copies[id(arg)])
                else:
                    args.append(arg)
            module.get_function(kernel)(grid, block, tuple(args))
            cp.cuda.Device().synchronize()
            result = [cp.asnumpy(args[index]) for index in outputs]
            if any(not np.all(np.isfinite(value)) for value in result):
                raise AssertionError(f"nonfinite output from {kernel} in {name}")
            values[name] = [value.view(np.uint32).copy() for value in result]
        for name, result in sorted(values.items()):
            ref_name = "strict_original_120" if name.startswith("strict_") else "candidate_120"
            reference = values.get(ref_name, values["candidate_120"])
            original = values.get("native_original_120", values.get("original_120", values["candidate_120"]))
            original_compute120 = values.get("original_120", values["candidate_120"])
            counts = [int(np.count_nonzero(a != b)) for a, b in zip(result, reference)]
            baseline_counts = [int(np.count_nonzero(a != b)) for a, b in zip(result, original)]
            compute120_counts = [int(np.count_nonzero(a != b)) for a, b in zip(result, original_compute120)]
            examples = []
            for argument, actual, expected in zip(outputs, result, original):
                for point in np.argwhere(actual != expected)[:2]:
                    examples.append({"argument": argument, "index": point.tolist(),
                                     "actual_bits": int(actual[tuple(point)]),
                                     "original_bits": int(expected[tuple(point)])})
            rows.append({"kernel": kernel, "variant": name, "input_sha256": digest(b"".join(input_parts)),
                         "table_identity_sha256": tables_sha, "output_arg_indices": outputs,
                         "different_vs_reference": counts, "different_vs_original120": baseline_counts,
                         "different_vs_original_compute120": compute120_counts,
                         "original120_reference": "native_original_120" if "native_original_120" in values
                                                  else "original_120" if "original_120" in values else "candidate_120",
                         "output_sha256": [digest(value.tobytes()) for value in result],
                         "first_original_differences": examples[:8]})
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
        receipt = compile_thompson(options.outdir, options.host_sm,
                                   candidate_source=options.candidate_source, baseline_source=options.baseline_source)
        return int(any(row["status"] not in ("linked", "unsupported") for row in receipt["variants"].values()))
    modules, receipt = load_thompson(options.outdir, candidate_source=options.candidate_source)
    rows = replay(modules)
    cross = sum(sum(row["different_vs_reference"]) for row in rows if row["variant"] == "candidate_100")
    diagnostic_cross = sum(sum(row["different_vs_reference"]) for row in rows
                           if row["variant"].startswith("candidate_")
                           and row["variant"] not in ("candidate_100", "candidate_120"))
    original_cross = sum(sum(row["different_vs_original_compute120"]) for row in rows
                         if row["variant"] == "original_100")
    native_changes = sum(sum(row["different_vs_reference"]) for row in rows
                         if row["variant"] == "native_candidate_120")
    original_native_changes = sum(sum(row["different_vs_original120"]) for row in rows
                                  if row["variant"] == "original_120")
    baseline_variant = ("native_candidate_120" if "native_original_120" in modules
                        else "candidate_120" if "original_120" in modules else None)
    baseline = sum(sum(row["different_vs_original120"]) for row in rows
                   if row["variant"] == baseline_variant)
    strict = sum(sum(row["different_vs_reference"]) for row in rows if row["variant"] == "strict_candidate_120")
    report = {"schema": "gpuwm-thompson-cross-target-replay-v1", "compile": receipt, "rows": rows,
              "cross_target_different_words": cross, "blackwell_changed_words": baseline,
              "original_blackwell_cross_target_different_words": original_cross,
              "diagnostic_other_target_different_words": diagnostic_cross,
              "native_candidate_replay_different_words": native_changes,
              "native_original_replay_different_words": original_native_changes,
              "blackwell_source_control": baseline_variant,
              "native_pipeline_limit": "CuPy compiles native sm cubins; compute-target PTX replay can round differently on the same GPU",
              "wrf_exact_changed_words": strict}
    (options.outdir / "replay-receipt.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"kernel_calls": len(rows) // len(modules), "variants": sorted(modules),
                      "cross_target_different_words": cross, "blackwell_changed_words": baseline,
                      "original_blackwell_cross_target_different_words": original_cross,
                      "diagnostic_other_target_different_words": diagnostic_cross,
                      "native_candidate_replay_different_words": native_changes,
                      "native_original_replay_different_words": original_native_changes,
                      "wrf_exact_changed_words": strict}, indent=2))
    return int(bool(cross or baseline or strict))


if __name__ == "__main__":
    raise SystemExit(main())
