"""Small public-fixture proof for serial and packed DA member legs.

GPU commands are explicit, require the node operator's OWNER check, and use
one complete UUID. CPU comparison and plan creation never import CuPy.
The fixture is idealized WSM6/Noah over terrain, not a weather skill case.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUEST_SCHEMA = "gpuwm-da.member-leg-proof-request.v1"
PROOF_SCHEMA = "gpuwm-da.member-leg-proof.v1"
MEMBERS = 4
LEG_SECONDS = 60.0
OWNER_ENV = "GPUWM_DA_MEMBER_LEG_OWNER_VERIFIED"
CLOCK_FIELDS = ("elapsed_ticks", "tick_den", "elapsed_seconds", "dtbc_fp32_bits",
                "domain_start_ticks", "domain_lifecycle")


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _gpu_guard():
    if os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"):
        raise RuntimeError("GPU proof refused by GPUWM_NO_LOCAL_GPU")
    if os.environ.get(OWNER_ENV) != "1":
        raise RuntimeError(f"GPU proof requires the node operator's OWNER check and {OWNER_ENV}=1")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid or len(uuid) != 40:
        raise RuntimeError("GPU proof requires exactly one complete CUDA_VISIBLE_DEVICES GPU UUID")
    return uuid


def _fixture_imports():
    _gpu_guard()
    directory = ROOT / "tests"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    from test_da_cycle_join_gpu import _experiment_to, _wire_parent
    from test_da_nested_forecast_gpu import _geometry, _synthetic_land, _terrain
    return _experiment_to, _wire_parent, _geometry, _synthetic_land, _terrain


def fixture_context(request, *, output):
    """Bind the established public GPU fixture through the typed wire seam."""
    from gpuwm.da import nested_forecast, perturb
    from gpuwm.da.hotstart import HotStartConfig
    from gpuwm.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK, WSM6_PROFILE_ID
    from gpuwm.static.lambert import grids_from_projection_config
    from tools.da_cycle_prepared import build_parser
    from tools.da_ensemble_state import EnsembleIdentity
    from tools.da_member_leg import MemberLegContext

    experiment_to, wire_parent, geometry, synthetic_land, terrain = _fixture_imports()
    experiment = dataclasses.replace(experiment_to(3 * LEG_SECONDS),
                                    acknowledgements=(CONSTANT_DOWNWARD_LONGWAVE_ACK,), constant_glw_wm2=300.0)
    cfg = experiment.root.run
    static, surface, identity_land = synthetic_land(cfg, terrain(cfg))
    grid = grids_from_projection_config(experiment)[0]
    inputs = SimpleNamespace(experiment=experiment, grid=grid, static=static, landuse_identity=identity_land)
    source_hash = hashlib.sha256((ROOT / "tests/test_da_nested_forecast_gpu.py").read_bytes() +
                                 (ROOT / "tests/test_da_cycle_join_gpu.py").read_bytes()).hexdigest()
    identity = EnsembleIdentity(members=MEMBERS, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dt_s=cfg.dt,
                                mp_physics=cfg.mp_physics, physics_profile=WSM6_PROFILE_ID,
                                prepared_content_sha256=source_hash)
    args = build_parser().parse_args([
        "--prepared-root", str(output), "--proof-sha256", source_hash,
        "--source-manifest-sha256", source_hash, "--prepared-content-sha256", source_hash,
        "--physics-profile", WSM6_PROFILE_ID, "--run-seconds", "180", "--history-interval-seconds", "60",
        "--members", str(MEMBERS), "--out", str(output), "--no-hotstart", "--positivity-policy", "clip",
    ])
    args.seed = 20261004
    args.sfc_td_sigma_k = 1.0 if request.get("surface_dewpoint", False) else None
    args.obs = [Path("synthetic-observed-leg0"), Path("synthetic-observed-leg1")]
    perturbation = perturb.PerturbationConfig.from_mapping({
        "dx_km": 3.0, "dy_km": 3.0, "rim_width": 5,
        # The species draw changes the hydrostatic column, and the default
        # mass balance integrates it with the run's hypsometric option.
        "hypsometric_opt": int(cfg.hypsometric_opt),
        "fields": [{"name": "u", "amplitude": 0.1, "length_scale_km": 9.0},
                   {"name": "v", "amplitude": 0.1, "length_scale_km": 9.0}],
        "species": [{"mass_field": "qr", "amplitude": 0.1, "length_scale_km": 9.0,
                     "vertical_scale_levels": 3.0, "threshold_kg_kg": 1e-14, "clip_sigmas": 2.0}],
    })
    restart = None
    nest_birth = None
    if request.get("previous_result"):
        previous = read_json(request["previous_result"])
        restart = Path(previous["restart"])
        nest_birth = previous.get("nest_birth")
    pending = None
    if request.get("pending"):
        with np.load(request["pending"], allow_pickle=False) as data:
            pending = {key: np.array(data[key], copy=True) for key in data.files}
    if request.get("invalid_pending"):
        pending = dict(pending or {})
        pending["missing_carrier"] = np.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=np.float64)
    child = nested_forecast.nest_domain_config(experiment, geometry()) if request["nested"] else None
    context = MemberLegContext(args=args, inputs=inputs, identity=identity, cfg_perturb=perturbation,
        hot_cfg=HotStartConfig(), leg=int(request["leg"]), absolute_leg_number=int(request["absolute_leg_number"]),
        t_start=float(request["t_start"]), t_end=float(request["t_end"]), resumed=bool(request.get("resumed", False)),
        restart=restart, pending=pending, nest_child_dc=child, nested=bool(request["nested"]), nest_birth=nest_birth,
        analysis_due=int(request["absolute_leg_number"]) < 2, surface_enabled=True,
        stage_root=Path(output), out=Path(output))

    def wire_override(stop_seconds, *, child_dc=None):
        leg_experiment = dataclasses.replace(experiment_to(stop_seconds),
            acknowledgements=experiment.acknowledgements, constant_glw_wm2=experiment.constant_glw_wm2)
        wired = wire_parent(leg_experiment, child_dc=child_dc)
        # A declared rain-bearing initial column gives the real nonlinear
        # radar operator ensemble spread. These fields are restored over on
        # continued legs, before pending increments are applied.
        wired["root"].state.qr[:, 8:25, 8:25] = 2e-4
        return SimpleNamespace(node=wired["root"], restored=SimpleNamespace(surface=wired["surface"],
            initial_result=SimpleNamespace(state=wired["root"].state)), driver=wired["driver"],
            clocks=wired["clocks"], schedule=wired["schedule"], child_dc=child_dc)

    return context, wire_override


def array_inventory(path, *, skip=()):
    output = {}
    with np.load(path, allow_pickle=False) as arrays:
        for name in sorted(arrays.files):
            if name in skip:
                continue
            value = arrays[name]
            output[name] = {"dtype": value.dtype.str, "shape": list(value.shape), "bytes": value.nbytes,
                            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    return output


def compare_array_files(expected, actual, *, skip=()):
    """Compare actual dtype, shape and bytes, retaining only small evidence."""
    mismatches, checked_bytes, checked_arrays = [], 0, 0
    with np.load(expected, allow_pickle=False) as left, np.load(actual, allow_pickle=False) as right:
        names = sorted((set(left.files) | set(right.files)) - set(skip))
        for name in names:
            if name not in left or name not in right:
                mismatches.append({"array": name, "reason": "missing array"})
                continue
            a, b = left[name], right[name]
            if a.dtype != b.dtype:
                mismatches.append({"array": name, "reason": "dtype", "expected": a.dtype.str, "actual": b.dtype.str})
            elif a.shape != b.shape:
                mismatches.append({"array": name, "reason": "shape", "expected": list(a.shape), "actual": list(b.shape)})
            elif a.tobytes(order="C") != b.tobytes(order="C"):
                mismatches.append({"array": name, "reason": "raw bytes"})
            checked_arrays += 1
            checked_bytes += a.nbytes
    return {"equal": not mismatches, "checked_arrays": checked_arrays, "checked_bytes": checked_bytes, "mismatches": mismatches}


def checkpoint_inventory(root):
    from gpuwm.io.restart import read_restart_header, tree_restart_members
    output = {}
    for gid, path in tree_restart_members(root).items():
        header = read_restart_header(path)
        output[str(gid)] = {"path": str(path), "arrays": array_inventory(path, skip=("__gpuwm_restart_header__",)),
                            "clock": {field: header[field] for field in CLOCK_FIELDS}}
    return output


def compare_results(expected_path, actual_path):
    left, right = read_json(expected_path), read_json(actual_path)
    evidence = {"member": left["member"], "absolute_leg_number": left["absolute_leg_number"], "domains": {}, "equal": True}
    if left["member"] != right["member"] or left["absolute_leg_number"] != right["absolute_leg_number"]:
        raise ValueError("comparison results name different member/leg identities")
    if set(left["checkpoint"]) != set(right["checkpoint"]):
        evidence.update(equal=False, domain_error="different tree membership")
    for gid in sorted(set(left["checkpoint"]) & set(right["checkpoint"])):
        a, b = left["checkpoint"][gid], right["checkpoint"][gid]
        part = compare_array_files(a["path"], b["path"], skip=("__gpuwm_restart_header__",))
        part["clock_equal"] = a["clock"] == b["clock"]
        evidence["domains"][gid] = part
        evidence["equal"] &= part["equal"] and part["clock_equal"]
    evidence["operator_arrays"] = compare_array_files(left["arrays_path"], right["arrays_path"])
    evidence["equal"] &= evidence["operator_arrays"]["equal"]
    return evidence


def run_fixture_worker(request_path, result_path):
    _gpu_guard()
    from tools.da_member_leg import run_member_leg
    request_path, result_path = Path(request_path), Path(result_path)
    request = read_json(request_path)
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unknown public-fixture request schema")
    context, wire = fixture_context(request, output=request["output"])
    started = time.monotonic()
    # Observe the public calls without replacing their numerical bodies.
    # The independent call count catches applying a pending analysis twice
    # even if serial and packed both acquired that same implementation bug.
    from gpuwm.da import nested_forecast
    from gpuwm.ensemble import increments as increment_owner
    from gpuwm.io.restart import tree_restart_members
    real_apply, real_child = increment_owner.apply_increments, nested_forecast.build_nested_child
    order, applications, analysed_qr = [], [], []

    def observed_apply(state, pending, *args, **kwargs):
        order.append("apply_pending")
        output = real_apply(state, pending, *args, **kwargs)
        applications.append(sorted(pending))
        analysed_qr.append(hashlib.sha256(state.qr.get().tobytes()).hexdigest())
        return output

    def observed_child(parent, *args, **kwargs):
        order.append("build_child")
        parent_qr = hashlib.sha256(parent.state.qr.get().tobytes()).hexdigest()
        if analysed_qr and context.restart is not None and len(tree_restart_members(context.restart)) == 1:
            if parent_qr != analysed_qr[-1]:
                raise AssertionError("newborn child did not receive the analysed parent")
        return real_child(parent, *args, **kwargs)

    increment_owner.apply_increments, nested_forecast.build_nested_child = observed_apply, observed_child
    try:
        result = run_member_leg(context, int(request["member"]), wire_override=wire)
    finally:
        increment_owner.apply_increments, nested_forecast.build_nested_child = real_apply, real_child
    expected_calls = 1 if context.pending else 0
    if len(applications) != expected_calls:
        raise AssertionError(f"pending analysis application count {len(applications)}, expected {expected_calls}")
    if context.nested and context.restart is not None and context.pending:
        expected_order = ["build_child", "apply_pending"] if len(tree_restart_members(context.restart)) > 1 else ["apply_pending", "build_child"]
        if order != expected_order:
            raise AssertionError(f"nested restore/analysis/birth call order differs: {order}, expected {expected_order}")
    # The packed controller analyses each member against its leg-end
    # restart instead of a separately shipped host mirror.  That is only
    # the same analysis if the two hold the same bytes, so the proof
    # checks it on every member leg it runs.
    from gpuwm.da.radar_assimilation import CheckpointStateView
    view = CheckpointStateView(result.restart)
    try:
        if set(view) != set(result.snapshot):
            raise AssertionError(
                "restart state inventory differs from the leg-end mirror: "
                f"{sorted(set(view) ^ set(result.snapshot))}")
        for key, value in result.snapshot.items():
            stored = view[key]
            if (stored.dtype != value.dtype or stored.shape != value.shape
                    or stored.tobytes() != np.ascontiguousarray(value).tobytes()):
                raise AssertionError(
                    f"restart state/{key} differs from the leg-end mirror")
    finally:
        view.close()
    arrays = {f"state/{key}": value for key, value in result.snapshot.items()}
    surface_capture = {"dewpoint_selected": bool(request.get("surface_dewpoint", False))}
    if surface_capture["dewpoint_selected"]:
        surface = result.H_surface or {}
        if not {"q2", "psfc"}.issubset(surface):
            raise AssertionError("selected dewpoint capture lacks actual Q2/PSFC diagnostics")
        q2, psfc = surface["q2"], surface["psfc"]
        if not np.all(np.isfinite(q2)) or np.any(q2 < 0) or np.any(q2 > 0.2):
            raise AssertionError("native Q2 diagnostics are not finite mixing ratios")
        if not np.all(np.isfinite(psfc)) or np.any(psfc < 50000) or np.any(psfc > 120000):
            raise AssertionError("native PSFC diagnostics are not legitimate Pa surface pressures")
        surface_capture.update(q2_min=float(q2.min()), q2_max=float(q2.max()), psfc_min_pa=float(psfc.min()), psfc_max_pa=float(psfc.max()))
    if result.H_Z is not None:
        arrays["Hx/Z"] = result.H_Z
    for key, value in (result.H_surface or {}).items():
        arrays[f"Hx/surface/{key}"] = value
    for key, value in result.setup_arrays.items():
        arrays[f"setup/{key}"] = value
    arrays_path = result_path.with_suffix(".arrays.npz")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arrays_path, **arrays)
    checkpoint = checkpoint_inventory(result.restart)
    file_paths = {arrays_path.resolve(), *(Path(part["path"]).resolve() for part in checkpoint.values())}
    file_paths.update(path.resolve() for path in result.restart.parent.iterdir() if path.is_file())
    metadata = {"schema": "gpuwm-da.member-leg-result.v1", "complete": True, "member": int(request["member"]),
        "request_hash": digest_file(request_path), "absolute_leg_number": int(request["absolute_leg_number"]),
        "t_start": float(request["t_start"]), "t_end": float(request["t_end"]), "elapsed_seconds": float(request["t_end"]),
        "restart": str(result.restart.resolve()), "arrays_path": str(arrays_path.resolve()),
        "checkpoint": checkpoint, "arrays": array_inventory(arrays_path),
        "output_root": str(result_path.parent.resolve()),
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path)} for path in sorted(file_paths)],
        "nest_birth": result.nest_birth, "record": result.record,
        "consume_restart": str(result.consume_restart) if result.consume_restart is not None else None,
        "pending_consumed": bool(result.pending_consumed), "wall_seconds": time.monotonic()-started,
        "pending_apply_calls": len(applications), "nested_call_order": order,
        "surface_capture": surface_capture,
        "fixture": "public WSM6/Noah terrain, periodic parent, declared 300 W/m2 longwave, seeded rain",
        "is_stub": True, "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"]}
    write_json(result_path, metadata)
    return metadata


def analyze_leg(results, output):
    """Use a real GPU LETKF transform on four independent model H_Z fields."""
    _gpu_guard()
    import cupy as cp
    from gpuwm.da.letkf import GridGeometry, GriddedObs, LetkfConfig, LetkfDiagnostics, Localization, analyze
    started = time.monotonic()
    results = [read_json(path) for path in results]
    if sorted(row["member"] for row in results) != list(range(MEMBERS)):
        raise ValueError("analysis barrier needs each of the four members exactly once")
    results.sort(key=lambda row: row["member"])
    priors, simulated = [], []
    for row in results:
        with np.load(row["arrays_path"], allow_pickle=False) as data:
            priors.append(np.array(data["state/qr"], copy=True))
            simulated.append(np.array(data["Hx/Z"], copy=True))
    loaded = time.monotonic()
    prior = cp.asarray(np.stack(priors))
    hx = cp.asarray(np.stack(simulated))
    nz, ny, nx = priors[0].shape
    mask = cp.zeros((nz, ny, nx), dtype=cp.bool_)
    mask[5, ny//2, nx//2] = True
    observed = cp.mean(hx, axis=0) + 0.25
    config = LetkfConfig(localization=Localization(horizontal_m=5000.0, vertical_m=3000.0),
        analysis_fields=("qr",), rtps_alpha=0.5, prior_inflation=1.0, chunk_points=256,
        solve_dtype="float64", eigensolver="jacobi", matmul="fixed-order")
    geometry = GridGeometry(dx_m=3000.0, dy_m=3000.0, heights_m=np.linspace(100.0, 19900.0, nz))
    diagnostics = LetkfDiagnostics()
    increments = analyze({"qr": prior}, [GriddedObs("synthetic_radar_Z", observed, 0.5, hx, mask)], geometry, config, diagnostics)
    cp.cuda.Stream.null.synchronize()
    solved = time.monotonic()
    pending = cp.asnumpy(increments["qr"])
    maximum = float(np.max(np.abs(pending)))
    spread = float(cp.std(hx, axis=0, ddof=1)[5, ny//2, nx//2])
    if maximum == 0 or spread == 0 or diagnostics.active_points == 0:
        raise AssertionError("GPU LETKF fixture has no observation spread or nonzero analysis increment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pending_paths = []
    for member in range(MEMBERS):
        path = output / f"pending_m{member:03d}.npz"
        np.savez(path, qr=pending[member])
        pending_paths.append(str(path.resolve()))
    receipt = {"schema": "gpuwm-da.member-leg-proof-analysis.v1", "members": MEMBERS,
        "solve_device": "GPU", "eigensolver": diagnostics.eigensolver, "matmul": diagnostics.matmul,
        "active_points": diagnostics.active_points, "max_abs_increment": maximum,
        "radar_Z_spread_dbz_at_gate": spread, "observation_index": [5, ny//2, nx//2],
        "innovation_dbz": 0.25, "pending": pending_paths,
        "pending_arrays": [array_inventory(path) for path in pending_paths],
        "is_stub": True, "localization_geometry": "declared idealized 3 km spacing and monotone model-level heights",
        "phase_seconds": {"input_io": loaded-started, "gpu_analysis_including_transfer": solved-loaded,
                          "output_io": time.monotonic()-solved, "analysis_plus_io": time.monotonic()-started}}
    write_json(output / "analysis.receipt.json", receipt)
    del prior, hx, observed, mask, increments
    cp.get_default_memory_pool().free_all_blocks()
    return receipt


def make_request(output, member, absolute_leg, *, previous_result=None, pending=None, resumed=False, invalid_pending=False, surface_dewpoint=False):
    return {"schema": REQUEST_SCHEMA, "member": int(member), "leg": 0 if resumed else int(absolute_leg),
        "absolute_leg_number": int(absolute_leg), "t_start": absolute_leg*LEG_SECONDS,
        "t_end": (absolute_leg+1)*LEG_SECONDS, "nested": absolute_leg >= 1, "resumed": bool(resumed),
        "previous_result": str(previous_result) if previous_result is not None else None,
        "pending": str(pending) if pending is not None else None, "invalid_pending": bool(invalid_pending),
        "surface_dewpoint": bool(surface_dewpoint),
        "output": str(Path(output).resolve())}


def make_jobs(output, absolute_leg, *, previous_results=None, pending=None, resumed=False, invalid_member=None, surface_dewpoint=False):
    output = Path(output)
    jobs = []
    for member in range(MEMBERS):
        request = make_request(output / "stage", member, absolute_leg,
            previous_result=None if previous_results is None else previous_results[member],
            pending=None if pending is None else pending[member], resumed=resumed, invalid_pending=member == invalid_member,
            surface_dewpoint=surface_dewpoint)
        path = output / f"request_m{member:03d}.json"
        result = output / f"result_m{member:03d}.json"
        write_json(path, request)
        jobs.append({"member": member, "argv": [sys.executable, str(Path(__file__).resolve()), "worker",
            "--request", str(path.resolve()), "--result", str(result.resolve())],
            "result_path": str(result.resolve()), "request_hash": digest_file(path),
            "t_start": request["t_start"], "t_end": request["t_end"]})
    return jobs


def _wave(jobs, uuid, width, output, timeout):
    from gpuwm.da.member_wave import run_wave
    return run_wave(jobs, devices=[uuid], members_per_card=width, output=Path(output), timeout_seconds=timeout,
                    host_reserve_bytes=4*1024**3, decoder_budget_bytes=1024**3)


def run_suite(output, uuid, *, timeout_seconds=900, surface_dewpoint=False, default_capture_reference=None):
    _gpu_guard()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if uuid != os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("proof UUID and controller CUDA_VISIBLE_DEVICES differ")
    arms, analyses, waves, comparisons, timings = {}, {}, {}, [], {}
    started = time.monotonic()
    for arm, width in (("serial", 1), ("packed", 4)):
        previous, pending = None, None
        arms[arm], analyses[arm], waves[arm], timings[arm] = [], [], [], []
        for leg in range(3):
            directory = output / arm / f"leg{leg:02d}"
            jobs = make_jobs(directory, leg, previous_results=previous, pending=pending, surface_dewpoint=surface_dewpoint)
            cycle_started = time.monotonic()
            waves[arm].append(_wave(jobs, uuid, width, directory / "wave", timeout_seconds))
            forecast_done = time.monotonic()
            previous = [job["result_path"] for job in jobs]
            arms[arm].append(previous)
            if leg < 2:
                analysis = analyze_leg(previous, directory / "analysis")
                analyses[arm].append(analysis)
                pending = analysis["pending"]
            timings[arm].append({"leg": leg, "forecast_leg_plus_io": forecast_done-cycle_started,
                "analysis_plus_io": 0.0 if leg == 2 else analyses[arm][-1]["phase_seconds"]["analysis_plus_io"],
                "cycle_wall_seconds": time.monotonic()-cycle_started})
    for leg in range(3):
        for member in range(MEMBERS):
            evidence = compare_results(arms["serial"][leg][member], arms["packed"][leg][member])
            if not evidence["equal"]:
                raise AssertionError(f"serial/packed member {member} leg {leg} differs: {evidence}")
            comparisons.append(evidence)
        if leg < 2:
            for member in range(MEMBERS):
                difference = compare_array_files(analyses["serial"][leg]["pending"][member], analyses["packed"][leg]["pending"][member])
                if not difference["equal"]:
                    raise AssertionError("serial and packed GPU LETKF increments differ")
    # A leg 0 resumed context must restore, never perturb or apply twice.
    resume_jobs = make_jobs(output / "resume", 1, previous_results=arms["serial"][0], pending=analyses["serial"][0]["pending"], resumed=True,
                            surface_dewpoint=surface_dewpoint)
    resume_wave = _wave(resume_jobs, uuid, 4, output / "resume/wave", timeout_seconds)
    resume_comparisons = [compare_results(arms["serial"][1][m], resume_jobs[m]["result_path"]) for m in range(MEMBERS)]
    if not all(row["equal"] for row in resume_comparisons):
        raise AssertionError("resumed pending analysis was not applied exactly once")
    # Deliberately inadmissible pending data fails after restart restoration.
    protected = []
    for path in arms["packed"][1]:
        row = read_json(path)
        protected.extend(part["path"] for part in row["checkpoint"].values())
    protected.extend(analyses["packed"][1]["pending"])
    before = {path: digest_file(path) for path in protected}
    failure_jobs = make_jobs(output / "failure", 2, previous_results=arms["packed"][1], pending=analyses["packed"][1]["pending"], invalid_member=2,
                             surface_dewpoint=surface_dewpoint)
    failure = None
    try:
        _wave(failure_jobs, uuid, 4, output / "failure/wave", timeout_seconds)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    if failure is None:
        raise AssertionError("member failure did not stop the analysis barrier")
    if before != {path: digest_file(path) for path in protected}:
        raise AssertionError("failed packed wave mutated or deleted restart/pending inputs")
    default_capture = []
    if default_capture_reference is not None:
        original = read_json(default_capture_reference)
        for arm in ("serial", "packed"):
            for leg in range(3):
                for member in range(MEMBERS):
                    old = original["waves"][arm][leg][member]
                    new = read_json(arms[arm][leg][member])
                    for field in ("t2", "u10", "v10"):
                        key = f"Hx/surface/{field}"
                        if old["arrays"][key] != new["arrays"][key]:
                            raise AssertionError("opt-in dewpoint capture changed an existing surface operator array")
                    default_capture.append({"arm": arm, "leg": leg, "member": member, "t2_u10_v10_exact_inventory_equal": True})
    receipt = {"schema": PROOF_SCHEMA, "status": "pass", "members": MEMBERS, "device_uuid": uuid,
        "is_stub": True, "qualification": "member-leg software integrity only",
        "surface_dewpoint": bool(surface_dewpoint), "default_surface_comparisons": default_capture,
        "packing": {"serial_members_per_card": 1, "packed_members_per_card": 4, "mps_required": True},
        "legs_seconds": [LEG_SECONDS]*3, "observed_legs": 2, "nested_birth_seconds": LEG_SECONDS,
        "comparisons": comparisons, "analyses": analyses, "waves": waves, "resume_wave": resume_wave,
        "cycle_timings": timings,
        "resume_comparisons": resume_comparisons, "failure_barrier": failure,
        "failed_wave_inputs_unchanged": True, "wall_seconds": time.monotonic()-started,
        "source_files": {str(path.relative_to(ROOT)): digest_file(path) for path in (Path(__file__).resolve(), ROOT / "tools/da_member_leg.py",
            ROOT / "tools/da_cycle_prepared.py", ROOT / "gpuwm/da/member_wave.py", ROOT / "gpuwm/da/member_transport.py",
            ROOT / "tests/test_da_cycle_join_gpu.py", ROOT / "tests/test_da_nested_forecast_gpu.py")},
        "limitations": ["idealized WSM6/Noah integrity fixture, not forecast skill",
                        "one 4090 card with four members, not an eight-card Blackwell throughput measurement",
                        "typed fixture wire seam exercises the shared member worker and wave barrier; production prepared-data JSON transport and driver coordinator are not exercised"]}
    write_json(output / "proof.receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--result", type=Path, required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--results", type=Path, nargs=MEMBERS, required=True)
    analyze.add_argument("--out", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--expected", type=Path, required=True)
    compare.add_argument("--actual", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--uuid", required=True)
    run.add_argument("--timeout-seconds", type=float, default=900)
    run.add_argument("--surface-dewpoint", action="store_true")
    run.add_argument("--default-capture-reference", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            value = run_fixture_worker(args.request, args.result)
        elif args.command == "analyze":
            value = analyze_leg(args.results, args.out)
        elif args.command == "compare":
            value = compare_results(args.expected, args.actual)
            write_json(args.out, value)
        elif args.command == "plan":
            value = {"schema": PROOF_SCHEMA, "status": "planned", "jobs": make_jobs(args.out, 0), "requires_gpu": False}
            write_json(args.out / "plan.json", value)
        else:
            value = run_suite(args.out, args.uuid, timeout_seconds=args.timeout_seconds, surface_dewpoint=args.surface_dewpoint,
                              default_capture_reference=args.default_capture_reference)
        print(json.dumps({"schema": value["schema"] if "schema" in value else PROOF_SCHEMA,
                          "status": value.get("status", "complete")}, sort_keys=True))
        return 0
    except Exception as error:
        print(f"DA member-leg proof refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
