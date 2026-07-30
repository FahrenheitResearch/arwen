from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ecape_parcel.calc import calc_ecape_parcel, custom_cape_cin_lfc_el, units
from metpy.calc import specific_humidity_from_dewpoint


SCRIPT_DIR = Path(__file__).resolve().parent
CRATE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CRATE_ROOT.parent.parent
RUST_MANIFEST = CRATE_ROOT / "Cargo.toml"
DEFAULT_FIXTURE = CRATE_ROOT / "tests" / "fixtures" / "parity_column.json"
DEFAULT_OUTPUT = CRATE_ROOT / "target" / "parity" / "parity_column"
G = 9.81
PHI = 287.04 / 461.5


@dataclass(frozen=True)
class Config:
    name: str
    parcel_type: str
    entraining: bool
    pseudoadiabatic: bool
    storm_motion_type: str


def load_profile(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "input_column" in raw and raw["input_column"] is not None:
        col = raw["input_column"]
        request = raw.get("request", {})
        nearest = raw.get("nearest_grid_point", {})
        profile_id = raw.get(
            "target_id",
            (
                f"{request.get('date_yyyymmdd', path.stem)}_"
                f"{request.get('cycle_utc', 'xx')}z_"
                f"f{request.get('forecast_hour', 'xxx')}_"
                f"{nearest.get('lat', 'lat')}_{nearest.get('lon', 'lon')}"
            ),
        )
        return {
            "profile_id": profile_id,
            "source_path": str(path),
            "height_m": [float(v) for v in col["height_m"]],
            "pressure_pa": [float(v) for v in col["pressure_pa"]],
            "temperature_k": [float(v) for v in col["temperature_k"]],
            "dewpoint_k": [float(v) for v in col["dewpoint_k"]],
            "u_wind_ms": [float(v) for v in col["u_wind_ms"]],
            "v_wind_ms": [float(v) for v in col["v_wind_ms"]],
            "storm_motion_u_ms": raw.get("storm_motion_u_ms"),
            "storm_motion_v_ms": raw.get("storm_motion_v_ms"),
        }

    pressure_pa = raw.get("pressure_pa")
    if pressure_pa is None and raw.get("pressure_hpa") is not None:
        pressure_pa = [float(v) * 100.0 for v in raw["pressure_hpa"]]
    if pressure_pa is None:
        raise ValueError(f"{path} does not include pressure_pa or pressure_hpa")
    return {
        "profile_id": raw.get("profile_id", path.stem),
        "source_path": str(path),
        "height_m": [float(v) for v in raw["height_m"]],
        "pressure_pa": [float(v) for v in pressure_pa],
        "temperature_k": [float(v) for v in raw["temperature_k"]],
        "dewpoint_k": [float(v) for v in raw["dewpoint_k"]],
        "u_wind_ms": [float(v) for v in raw.get("u_wind_ms", raw.get("u_ms"))],
        "v_wind_ms": [float(v) for v in raw.get("v_wind_ms", raw.get("v_ms"))],
        "storm_motion_u_ms": raw.get("storm_motion_u_ms"),
        "storm_motion_v_ms": raw.get("storm_motion_v_ms"),
    }


def default_configs(storm_motion_type: str) -> list[Config]:
    configs: list[Config] = []
    for parcel_type in ["surface_based", "mixed_layer", "most_unstable"]:
        for entraining in [True, False]:
            for pseudoadiabatic in [True, False]:
                configs.append(
                    Config(
                        name=(
                            f"{parcel_type}_"
                            f"{'entraining' if entraining else 'nonentraining'}_"
                            f"{'pseudo' if pseudoadiabatic else 'irreversible'}_"
                            f"{storm_motion_type}"
                        ),
                        parcel_type=parcel_type,
                        entraining=entraining,
                        pseudoadiabatic=pseudoadiabatic,
                        storm_motion_type=storm_motion_type,
                    )
                )
    return configs


def qv_from_dewpoint(pressure_pa: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    q = specific_humidity_from_dewpoint(
        pressure_pa * units.pascal,
        dewpoint_k * units.kelvin,
    )
    return q.to("dimensionless").magnitude.astype(float)


def density_temperature_np(temp_k: np.ndarray, qv: np.ndarray, qt: np.ndarray) -> np.ndarray:
    return temp_k * (1.0 - qt + qv / PHI)


def interp_profile(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(target, x[order], y[order], left=np.nan, right=np.nan)


def buoyancy_profile(
    parcel_height_m: np.ndarray,
    parcel_temp_k: np.ndarray,
    parcel_qv: np.ndarray,
    parcel_qt: np.ndarray,
    env_height_m: np.ndarray,
    env_temp_k: np.ndarray,
    env_qv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parcel_trho = density_temperature_np(parcel_temp_k, parcel_qv, parcel_qt)
    env_trho = density_temperature_np(env_temp_k, env_qv, env_qv)
    env_trho_on_path = interp_profile(env_height_m, env_trho, parcel_height_m)
    buoyancy = G * (parcel_trho - env_trho_on_path) / env_trho_on_path
    return parcel_trho, env_trho_on_path, buoyancy


def integrate_positive_buoyancy(height_m: np.ndarray, buoyancy_ms2: np.ndarray) -> float:
    if len(height_m) < 2:
        return math.nan
    positive = np.where(np.isfinite(buoyancy_ms2), np.maximum(buoyancy_ms2, 0.0), 0.0)
    return float(np.nansum(0.5 * (positive[1:] + positive[:-1]) * np.diff(height_m)))


def to_quantity(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "pressure": np.asarray(profile["pressure_pa"], dtype=float) * units.pascal,
        "height": np.asarray(profile["height_m"], dtype=float) * units.meter,
        "temperature": np.asarray(profile["temperature_k"], dtype=float) * units.kelvin,
        "dewpoint": np.asarray(profile["dewpoint_k"], dtype=float) * units.kelvin,
        "u_wind": np.asarray(profile["u_wind_ms"], dtype=float) * units("m/s"),
        "v_wind": np.asarray(profile["v_wind_ms"], dtype=float) * units("m/s"),
    }


def qmag(value: Any, unit: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value.to(unit).magnitude)
    except Exception:
        return float(value)


def is_empty_ecape_parcel_result(result: Any) -> bool:
    if not isinstance(result, (list, tuple)):
        return False
    if len(result) == 0:
        return True
    for part in result:
        if part is None:
            return True
        if isinstance(part, list) and (len(part) == 0 or part[0] is None):
            return True
    return False


def empty_python_run(config: Config, elapsed_ms: float, reps: int) -> dict[str, Any]:
    return {
        "implementation": "ecape_parcel_python",
        "config": config.__dict__,
        "timing": {
            "reps": max(1, reps),
            "elapsed_ms": elapsed_ms,
            "per_call_ms": elapsed_ms / max(1, reps),
        },
        "scalars": {
            "epath_jkg": 0.0,
            "cin_jkg": None,
            "lfc_m": None,
            "el_m": None,
            "computed_epath_jkg": 0.0,
        },
        "parcel": {
            "pressure_pa": [],
            "height_m": [],
            "temperature_k": [],
            "qv_kgkg": [],
            "qt_kgkg": [],
            "density_temperature_k": [],
            "env_density_temperature_k": [],
            "buoyancy_ms2": [],
        },
        "empty_path": True,
    }


def call_python(profile: dict[str, Any], config: Config, reps: int) -> dict[str, Any]:
    q = to_quantity(profile)
    kwargs: dict[str, Any] = {
        "align_to_input_pressure_values": False,
        "entrainment_switch": config.entraining,
        "pseudoadiabatic_switch": config.pseudoadiabatic,
        "cape_type": config.parcel_type,
        "storm_motion_type": config.storm_motion_type,
    }
    if config.storm_motion_type == "user_defined":
        kwargs["storm_motion_u"] = profile.get("storm_motion_u_ms") * units("m/s")
        kwargs["storm_motion_v"] = profile.get("storm_motion_v_ms") * units("m/s")

    start = time.perf_counter()
    result = None
    for _ in range(max(1, reps)):
        result = calc_ecape_parcel(
            q["pressure"],
            q["height"],
            q["temperature"],
            q["dewpoint"],
            q["u_wind"],
            q["v_wind"],
            **kwargs,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert result is not None
    if is_empty_ecape_parcel_result(result):
        return empty_python_run(config, elapsed_ms, reps)

    parcel_pressure, parcel_height, parcel_temperature, parcel_qv, parcel_qt = result

    pressure_pa = np.asarray(parcel_pressure.to("pascal").magnitude, dtype=float)
    height_m = np.asarray(parcel_height.to("meter").magnitude, dtype=float)
    temp_k = np.asarray(parcel_temperature.to("kelvin").magnitude, dtype=float)
    qv = np.asarray(parcel_qv.to("dimensionless").magnitude, dtype=float)
    qt = np.asarray(parcel_qt.to("dimensionless").magnitude, dtype=float)
    env_qv = qv_from_dewpoint(
        np.asarray(profile["pressure_pa"], dtype=float),
        np.asarray(profile["dewpoint_k"], dtype=float),
    )
    parcel_trho, env_trho_on_path, buoyancy = buoyancy_profile(
        height_m,
        temp_k,
        qv,
        qt,
        np.asarray(profile["height_m"], dtype=float),
        np.asarray(profile["temperature_k"], dtype=float),
        env_qv,
    )
    integrated_positive, integrated_negative, lfc, el = custom_cape_cin_lfc_el(
        parcel_height,
        parcel_temperature,
        parcel_qv,
        parcel_qt,
        q["height"],
        q["temperature"],
        specific_humidity_from_dewpoint(q["pressure"], q["dewpoint"]),
    )

    return {
        "implementation": "ecape_parcel_python",
        "config": config.__dict__,
        "timing": {
            "reps": max(1, reps),
            "elapsed_ms": elapsed_ms,
            "per_call_ms": elapsed_ms / max(1, reps),
        },
        "scalars": {
            "epath_jkg": qmag(integrated_positive, "J/kg"),
            "cin_jkg": qmag(integrated_negative, "J/kg"),
            "lfc_m": qmag(lfc, "meter"),
            "el_m": qmag(el, "meter"),
            "computed_epath_jkg": integrate_positive_buoyancy(height_m, buoyancy),
        },
        "parcel": {
            "pressure_pa": pressure_pa.tolist(),
            "height_m": height_m.tolist(),
            "temperature_k": temp_k.tolist(),
            "qv_kgkg": qv.tolist(),
            "qt_kgkg": qt.tolist(),
            "density_temperature_k": parcel_trho.tolist(),
            "env_density_temperature_k": env_trho_on_path.tolist(),
            "buoyancy_ms2": buoyancy.tolist(),
        },
        "empty_path": False,
    }


def rust_payload(profile: dict[str, Any], config: Config, reps: int) -> dict[str, Any]:
    options: dict[str, Any] = {
        "cape_type": config.parcel_type,
        "storm_motion_type": config.storm_motion_type,
        "pseudoadiabatic": config.pseudoadiabatic,
    }
    if not config.entraining:
        options["entrainment_rate"] = 0.0
    if config.storm_motion_type == "user_defined":
        options["storm_motion_u_ms"] = profile.get("storm_motion_u_ms")
        options["storm_motion_v_ms"] = profile.get("storm_motion_v_ms")
    return {
        "pressure_hpa": [p / 100.0 for p in profile["pressure_pa"]],
        "height_m": profile["height_m"],
        "temperature_k": profile["temperature_k"],
        "dewpoint_k": profile["dewpoint_k"],
        "u_wind_ms": profile["u_wind_ms"],
        "v_wind_ms": profile["v_wind_ms"],
        "options": options,
        "reps": reps,
    }


def call_rust(
    profile: dict[str, Any],
    config: Config,
    reps: int,
    rust_bin: Path | None,
) -> dict[str, Any]:
    command = [str(rust_bin)] if rust_bin else [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--manifest-path",
        str(RUST_MANIFEST),
        "--bin",
        "run_case_raw",
    ]
    proc = subprocess.run(
        command,
        input=json.dumps(rust_payload(profile, config, reps)),
        text=True,
        capture_output=True,
        check=True,
    )
    raw = json.loads(proc.stdout)
    pressure_pa = np.asarray(raw["parcel_pressure_pa"], dtype=float)
    height_m = np.asarray(raw["parcel_height_m"], dtype=float)
    temp_k = np.asarray(raw["parcel_temperature_k"], dtype=float)
    qv = np.asarray(raw["parcel_qv_kgkg"], dtype=float)
    qt = np.asarray(raw["parcel_qt_kgkg"], dtype=float)
    env_qv = qv_from_dewpoint(
        np.asarray(profile["pressure_pa"], dtype=float),
        np.asarray(profile["dewpoint_k"], dtype=float),
    )
    parcel_trho, env_trho_on_path, buoyancy = buoyancy_profile(
        height_m,
        temp_k,
        qv,
        qt,
        np.asarray(profile["height_m"], dtype=float),
        np.asarray(profile["temperature_k"], dtype=float),
        env_qv,
    )
    return {
        "implementation": "ecape_rs_rust",
        "config": config.__dict__,
        "timing": {
            "reps": int(raw["reps"]),
            "elapsed_ms": float(raw["elapsed_ms"]),
            "per_call_ms": float(raw["per_call_ms"]),
        },
        "scalars": {
            "post_analytic_ecape_jkg": float(raw["ecape_jkg"]),
            "post_analytic_ncape_jkg": float(raw["ncape_jkg"]),
            "epath_jkg": float(raw["cape_jkg"]),
            "cin_jkg": float(raw["cin_jkg"]),
            "lfc_m": raw["lfc_m"],
            "el_m": raw["el_m"],
            "computed_epath_jkg": integrate_positive_buoyancy(height_m, buoyancy),
        },
        "parcel": {
            "pressure_pa": pressure_pa.tolist(),
            "height_m": height_m.tolist(),
            "temperature_k": temp_k.tolist(),
            "qv_kgkg": qv.tolist(),
            "qt_kgkg": qt.tolist(),
            "density_temperature_k": parcel_trho.tolist(),
            "env_density_temperature_k": env_trho_on_path.tolist(),
            "buoyancy_ms2": buoyancy.tolist(),
        },
    }


def common_height_grid(a: np.ndarray, b: np.ndarray, step_m: float = 20.0) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.asarray([], dtype=float)
    low = math.ceil(max(np.nanmin(a), np.nanmin(b)) / step_m) * step_m
    high = math.floor(min(np.nanmax(a), np.nanmax(b)) / step_m) * step_m
    if high < low:
        return np.asarray([], dtype=float)
    return np.arange(low, high + 0.5 * step_m, step_m)


def interp_on_height(run: dict[str, Any], field: str, grid: np.ndarray) -> np.ndarray:
    return interp_profile(
        np.asarray(run["parcel"]["height_m"], dtype=float),
        np.asarray(run["parcel"][field], dtype=float),
        grid,
    )


def compare_runs(
    profile_id: str,
    config: Config,
    py_run: dict[str, Any],
    rs_run: dict[str, Any],
) -> dict[str, Any]:
    py_height = np.asarray(py_run["parcel"]["height_m"], dtype=float)
    rs_height = np.asarray(rs_run["parcel"]["height_m"], dtype=float)
    py_empty = bool(py_run.get("empty_path", False) or len(py_height) == 0)
    rs_empty = bool(rs_run.get("empty_path", False) or len(rs_height) == 0)
    grid = common_height_grid(
        py_height,
        rs_height,
    )
    comparisons: dict[str, tuple[float, float]] = {}
    for field in ["temperature_k", "qv_kgkg", "qt_kgkg", "density_temperature_k", "buoyancy_ms2"]:
        if py_empty and rs_empty:
            comparisons[field] = (0.0, 0.0)
            continue
        if len(grid) == 0:
            comparisons[field] = (math.nan, math.nan)
            continue
        diff = interp_on_height(rs_run, field, grid) - interp_on_height(py_run, field, grid)
        comparisons[field] = (float(np.nanmax(np.abs(diff))), float(np.nanmean(np.abs(diff))))

    py_epath = py_run["scalars"].get("epath_jkg")
    rs_epath = rs_run["scalars"].get("epath_jkg")
    py_nonzero = py_epath is not None and abs(float(py_epath)) > 1e-6
    rs_nonzero = rs_epath is not None and abs(float(rs_epath)) > 1e-6
    py_lfc = py_run["scalars"].get("lfc_m")
    rs_lfc = rs_run["scalars"].get("lfc_m")
    py_el = py_run["scalars"].get("el_m")
    rs_el = rs_run["scalars"].get("el_m")
    empty_path_mismatch = py_empty != rs_empty
    lfc_zero_nonzero_mismatch = (py_lfc is None) != (rs_lfc is None)
    el_zero_nonzero_mismatch = (py_el is None) != (rs_el is None)
    return {
        "profile_id": profile_id,
        "config": config.name,
        "parcel_type": config.parcel_type,
        "entraining": config.entraining,
        "pseudoadiabatic": config.pseudoadiabatic,
        "storm_motion_type": config.storm_motion_type,
        "common_levels": int(len(grid)),
        "python_per_call_ms": py_run["timing"]["per_call_ms"],
        "rust_per_call_ms": rs_run["timing"]["per_call_ms"],
        "solver_speedup_python_over_rust": (
            py_run["timing"]["per_call_ms"] / rs_run["timing"]["per_call_ms"]
            if rs_run["timing"]["per_call_ms"] > 0
            else math.nan
        ),
        "python_epath_jkg": py_epath,
        "rust_epath_jkg": rs_epath,
        "epath_diff_jkg": None if py_epath is None or rs_epath is None else rs_epath - py_epath,
        "python_cin_jkg": py_run["scalars"].get("cin_jkg"),
        "rust_cin_jkg": rs_run["scalars"].get("cin_jkg"),
        "python_lfc_m": py_lfc,
        "rust_lfc_m": rs_lfc,
        "python_el_m": py_el,
        "rust_el_m": rs_el,
        "python_empty_path": py_empty,
        "rust_empty_path": rs_empty,
        "empty_path_mismatch": empty_path_mismatch,
        "lfc_zero_nonzero_mismatch": lfc_zero_nonzero_mismatch,
        "el_zero_nonzero_mismatch": el_zero_nonzero_mismatch,
        "zero_nonzero_mismatch": bool(
            py_nonzero != rs_nonzero
            or empty_path_mismatch
            or lfc_zero_nonzero_mismatch
            or el_zero_nonzero_mismatch
        ),
        "max_abs_parcel_temperature_k": comparisons["temperature_k"][0],
        "mean_abs_parcel_temperature_k": comparisons["temperature_k"][1],
        "max_abs_density_temperature_k": comparisons["density_temperature_k"][0],
        "mean_abs_density_temperature_k": comparisons["density_temperature_k"][1],
        "max_abs_buoyancy_ms2": comparisons["buoyancy_ms2"][0],
        "mean_abs_buoyancy_ms2": comparisons["buoyancy_ms2"][1],
        "max_abs_qv_kgkg": comparisons["qv_kgkg"][0],
        "mean_abs_qv_kgkg": comparisons["qv_kgkg"][1],
        "max_abs_qt_kgkg": comparisons["qt_kgkg"][0],
        "mean_abs_qt_kgkg": comparisons["qt_kgkg"][1],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("# ecape-parcel Parity Summary\n\nNo rows.\n", encoding="utf-8")
        return
    group_rows = []
    for parcel in ["surface_based", "mixed_layer", "most_unstable"]:
        sub = [row for row in rows if row["parcel_type"] == parcel]
        if not sub:
            continue
        group_rows.append(
            {
                "parcel_type": parcel,
                "max_abs_epath_diff_jkg": max(abs(float(row["epath_diff_jkg"])) for row in sub),
                "max_abs_density_temperature_k": max(
                    float(row["max_abs_density_temperature_k"]) for row in sub
                ),
                "python_ms_min": min(float(row["python_per_call_ms"]) for row in sub),
                "python_ms_max": max(float(row["python_per_call_ms"]) for row in sub),
                "rust_ms_min": min(float(row["rust_per_call_ms"]) for row in sub),
                "rust_ms_max": max(float(row["rust_per_call_ms"]) for row in sub),
                "mismatch_count": sum(bool(row["zero_nonzero_mismatch"]) for row in sub),
            }
        )
    fields = [
        "config",
        "epath_diff_jkg",
        "max_abs_density_temperature_k",
        "python_per_call_ms",
        "rust_per_call_ms",
        "zero_nonzero_mismatch",
    ]
    lines = [
        "# ecape-parcel Parity Summary",
        "",
        "This run calls Python `ecape_parcel.calc_ecape_parcel` directly and compares it with Rust `ecape-rs` on a common 20 m height grid.",
        "",
        "## Grouped Maxima",
        "",
        markdown_table(group_rows, list(group_rows[0].keys())),
        "",
        "## Per-Configuration Results",
        "",
        markdown_table(rows, fields),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rust-bin", type=Path)
    parser.add_argument("--python-reps", type=int, default=3)
    parser.add_argument("--rust-reps", type=int, default=100)
    parser.add_argument("--storm-motion-type", default="user_defined")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    if args.storm_motion_type == "user_defined" and (
        profile.get("storm_motion_u_ms") is None or profile.get("storm_motion_v_ms") is None
    ):
        raise ValueError("user_defined storm motion requires storm_motion_u_ms and storm_motion_v_ms")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for config in default_configs(args.storm_motion_type):
        try:
            py_run = call_python(profile, config, args.python_reps)
            rs_run = call_rust(profile, config, args.rust_reps, args.rust_bin)
            rows.append(compare_runs(profile["profile_id"], config, py_run, rs_run))
            runs.append({"profile_id": profile["profile_id"], "python": py_run, "rust": rs_run})
        except Exception as exc:
            failures.append(
                {
                    "profile_id": profile["profile_id"],
                    "config": config.__dict__,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
            print(f"FAILED {config.name}: {exc}", file=sys.stderr)

    (args.output_dir / "normalized_runs.json").write_text(
        json.dumps({"profile": profile, "runs": runs, "failures": failures}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    write_csv(rows, args.output_dir / "parity_summary.csv")
    write_markdown(rows, args.output_dir / "parity_summary.md")
    print(args.output_dir / "parity_summary.md")
    if failures:
        print(args.output_dir / "failures.json", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
