#!/usr/bin/env python3
"""Generate and audit the effective N5S four-domain WRF namelist."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Sequence
import tomllib


FORCING_INTERVAL_SECONDS = 6 * 60 * 60


def _value_text(text: str, key: str) -> str | None:
    pattern = re.compile(
        rf"(?im)^\s*{re.escape(key)}\s*=\s*(.*?)"
        rf"(?=^\s*[a-zA-Z][a-zA-Z0-9_]*\s*=|^\s*/)", re.S)
    match = pattern.search(text)
    return None if match is None else match.group(1).strip().rstrip(",").strip()


def _tokens(text: str | None) -> list[str]:
    if text is None:
        return []
    return [token.strip().strip("'\"") for token in text.replace("\n", " ").split(",")
            if token.strip()]


def _numbers(text: str | None, kind=float):
    return [kind(token.replace("d", "e").replace("D", "e"))
            for token in _tokens(text)]


def _fmt(values) -> str:
    def one(value):
        if isinstance(value, bool):
            return ".true." if value else ".false."
        if isinstance(value, str):
            return f"'{value}'"
        if isinstance(value, float):
            return f"{value:.9g}"
        return str(value)
    return ", ".join(one(value) for value in values) + ","


def _line(key: str, values) -> str:
    if not isinstance(values, (tuple, list)):
        values = [values]
    return f" {key:<36} = {_fmt(values)}"


def build_effective_namelist(bundle_namelist: str | Path,
                             config_path: str | Path, *, run_minutes: int,
                             history_minutes: int,
                             stage: str = "forecast") -> tuple[str, dict]:
    if not 30 <= run_minutes <= 75:
        raise ValueError("run_minutes must be in 30..75")
    if history_minutes <= 0 or run_minutes % history_minutes:
        raise ValueError("history_minutes must divide run_minutes")
    if stage not in ("forecast", "real"):
        raise ValueError("stage must be 'forecast' or 'real'")
    bundle_path = Path(bundle_namelist)
    config_path = Path(config_path)
    bundle = bundle_path.read_text(encoding="utf-8")
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    expected_bundle = {
        "max_dom": [4], "time_step": [60],
        "interval_seconds": [FORCING_INTERVAL_SECONDS],
        "e_we": [251, 501, 502, 601],
        "e_sn": [201, 401, 502, 601],
        "parent_grid_ratio": [1, 4, 3, 3],
        "parent_time_step_ratio": [1, 4, 3, 3],
        "diff_6th_factor": [0.12, 0.10, 0.08, 0.06],
        "spec_bdy_width": [5], "mp_physics": [55, 55, 55, 55],
        "bl_pbl_physics": [11, 11, 11, 11],
    }
    for key, expected in expected_bundle.items():
        actual = _numbers(_value_text(bundle, key),
                          int if all(isinstance(v, int) for v in expected) else float)
        if actual[:len(expected)] != expected:
            raise ValueError(
                f"binding bundle namelist drift for {key}: {actual} != {expected}")

    domains = config["domain"]
    experiment = config["experiment"]
    shared = config["shared"]
    case_data = config["case_data"]
    if shared["mp_physics"] != 10 or shared["bl_pbl_physics"] != 1:
        raise ValueError(
            "gpuwm config no longer carries the registered Morrison/YSU selection")

    derived_dx = [float(domains[0]["dx"])]
    for domain in domains[1:]:
        derived_dx.append(
            derived_dx[domain["parent_id"] - 1] / domain["parent_grid_ratio"])
    four_from_shared = lambda key: [shared[key]] * 4
    # All knobs represented on both authorities are checked here before any
    # intentional N5S override is applied.  Values present only in WRF are
    # copied verbatim into the generated namelist below.
    config_checks = [
        ("run_hours", [int(experiment["run_seconds"] // 3600)]),
        ("run_minutes", [int(experiment["run_seconds"] % 3600 // 60)]),
        ("interval_seconds", [int(case_data["forcing_interval_s"])]),
        ("history_interval",
         [dom["history_interval_s"] / 60.0 for dom in domains]),
        ("restart_interval", [experiment["restart_interval_s"] / 60.0]),
        ("time_step", [domains[0]["time_step"]]),
        ("max_dom", [len(domains)]),
        ("e_we", [dom["nx"] + 1 for dom in domains]),
        ("e_sn", [dom["ny"] + 1 for dom in domains]),
        ("e_vert", [shared["nz"] + 1] * 4),
        ("eta_levels", shared["eta_levels"]),
        ("p_top_requested", [shared["p_top"]]),
        ("dx", derived_dx), ("dy", derived_dx),
        ("grid_id", [dom["grid_id"] for dom in domains]),
        ("parent_id", [dom["parent_id"] for dom in domains]),
        ("i_parent_start", [dom["i_parent_start"] for dom in domains]),
        ("j_parent_start", [dom["j_parent_start"] for dom in domains]),
        ("parent_grid_ratio", [dom["parent_grid_ratio"] for dom in domains]),
        ("parent_time_step_ratio",
         [dom["parent_time_step_ratio"] for dom in domains]),
        ("feedback", [experiment["feedback"]]),
        ("smooth_option", [experiment["smooth_option"]]),
        ("sfcp_to_sfcp", [case_data["sfcp_to_sfcp"]]),
        ("ra_lw_physics", four_from_shared("ra_physics")),
        ("ra_sw_physics", four_from_shared("ra_physics")),
        ("radt", [dom["radt"] for dom in domains]),
        ("sf_sfclay_physics", four_from_shared("sf_sfclay_physics")),
        ("sf_surface_physics", four_from_shared("sf_surface_physics")),
        ("bldt", four_from_shared("bldt")),
        ("cu_physics", [dom["cu_physics"] for dom in domains]),
        ("cudt", [dom.get("cudt_minutes", 0) for dom in domains]),
        ("hybrid_opt", [shared["hybrid_opt"]]),
        ("etac", [shared["etac"]]),
        ("w_damping", [shared["w_damping"]]),
        ("epssm", [shared["epssm"]]),
        ("km_opt", four_from_shared("km_opt")),
        ("diff_6th_opt", four_from_shared("diff_6th_opt")),
        ("diff_6th_factor", [dom["diff_6th_factor"] for dom in domains]),
        ("diff_6th_slopeopt", four_from_shared("diff_6th_slopeopt")),
        ("base_temp", [shared["base_temp"]]),
        ("damp_opt", [shared["damp_opt"]]),
        ("zdamp", four_from_shared("zdamp")),
        ("dampcoef", four_from_shared("dampcoef")),
        ("khdif", four_from_shared("khdif")),
        ("kvdif", four_from_shared("kvdif")),
        ("moist_adv_opt", four_from_shared("moist_adv_opt")),
        ("spec_bdy_width", [experiment["spec_bdy_width"]]),
        ("spec_zone", [shared["spec_zone"]]),
        ("relax_zone", [shared["relax_zone"]]),
        ("specified", [dom["specified"] for dom in domains]),
        ("nested", [dom["nested"] for dom in domains]),
    ]
    agreement = []
    for key, config_values in config_checks:
        raw_value = _value_text(bundle, key)
        if raw_value is None:
            raise ValueError(f"bundle/config common knob is absent: {key}")
        if all(isinstance(v, bool) for v in config_values):
            bundle_values = [token.lower() == ".true." for token in _tokens(raw_value)]
        else:
            bundle_values = _numbers(raw_value, float)
        selected = bundle_values[:len(config_values)]
        equal = len(selected) == len(config_values) and all(
            left == right if isinstance(right, bool) else
            math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-8)
            for left, right in zip(selected, config_values))
        if not equal:
            raise ValueError(
                f"bundle/config conflict in required common knob {key}: "
                f"bundle={selected} config={config_values}")
        agreement.append({"key": key, "bundle": selected,
                          "gpuwm_config": config_values, "agrees": True})

    start = datetime(1974, 4, 3, 12, 0, 0)
    # real.exe must walk BOTH 6-hourly analysis times (12Z + 18Z) to build
    # wrfbdy_d01 ("Regional domains require more than one time-period to
    # process"); the forecast namelist bounds wrf.exe to run_minutes.  The
    # two stages therefore get different end times from the same builder.
    if stage == "real":
        end = start + timedelta(seconds=FORCING_INTERVAL_SECONDS)
        stage_run_minutes = FORCING_INTERVAL_SECONDS // 60
    else:
        end = start + timedelta(minutes=run_minutes)
        stage_run_minutes = run_minutes
    four = lambda value: [value] * 4
    eta_lines = [
        " eta_levels = 1.00000, 0.99780, 0.99519, 0.99212, 0.98849,",
        "              0.98422, 0.97918, 0.97325, 0.96627, 0.95808,",
        "              0.94846, 0.93719, 0.92402, 0.90866, 0.89079,",
        "              0.87006, 0.84612, 0.81857, 0.78706, 0.75124,",
        "              0.71080, 0.66556, 0.61547, 0.56067, 0.50519,",
        "              0.45474, 0.40886, 0.36713, 0.32918, 0.29466,",
        "              0.26328, 0.23473, 0.20877, 0.18516, 0.16369,",
        "              0.14417, 0.12641, 0.11026, 0.09557, 0.08222,",
        "              0.07007, 0.05902, 0.04898, 0.03984, 0.03153,",
        "              0.02398, 0.01710, 0.01085, 0.00517, 0.00000,",
    ]
    sections = []
    sections.extend([
        "&time_control",
        _line("run_days", 0), _line("run_hours", stage_run_minutes // 60),
        _line("run_minutes", stage_run_minutes % 60), _line("run_seconds", 0),
        _line("start_year", four(1974)), _line("start_month", four(4)),
        _line("start_day", four(3)), _line("start_hour", four(12)),
        _line("start_minute", four(0)), _line("start_second", four(0)),
        _line("end_year", four(end.year)), _line("end_month", four(end.month)),
        _line("end_day", four(end.day)), _line("end_hour", four(end.hour)),
        _line("end_minute", four(end.minute)), _line("end_second", four(0)),
        _line("interval_seconds", FORCING_INTERVAL_SECONDS),
        _line("input_from_file", four(True)),
        _line("history_interval", four(history_minutes)),
        _line("history_interval_s", four(0)), _line("history_begin", four(0)),
        _line("frames_per_outfile", four(1)), _line("restart", False),
        _line("restart_interval", 0), _line("nwp_diagnostics", 0),
        _line("nocolons", True), _line("io_form_history", 2),
        _line("io_form_restart", 2), _line("io_form_input", 2),
        _line("io_form_boundary", 2), "/", "",
        "&domains", _line("time_step", 60), _line("max_dom", 4),
        _line("e_we", [251, 501, 502, 601]),
        _line("e_sn", [201, 401, 502, 601]), _line("e_vert", four(50)),
        *eta_lines, _line("p_top_requested", 10000),
        _line("num_metgrid_levels", 38), _line("num_metgrid_soil_levels", 4),
        _line("dx", [12000.0, 3000.0, 1000.0, 333.333333]),
        _line("dy", [12000.0, 3000.0, 1000.0, 333.333333]),
        _line("grid_id", [1, 2, 3, 4]), _line("parent_id", [0, 1, 2, 3]),
        _line("i_parent_start", [1, 63, 167, 151]),
        _line("j_parent_start", [1, 51, 117, 151]),
        _line("parent_grid_ratio", [1, 4, 3, 3]),
        _line("parent_time_step_ratio", [1, 4, 3, 3]),
        _line("feedback", 0), _line("smooth_option", 0),
        _line("sfcp_to_sfcp", True), "/", "",
        "&physics", _line("mp_physics", four(10)),
        _line("ra_lw_physics", four(4)), _line("ra_sw_physics", four(4)),
        _line("radt", [12, 3, 1, 1]), _line("sf_sfclay_physics", four(91)),
        _line("sf_surface_physics", four(2)), _line("bl_pbl_physics", four(1)),
        _line("bldt", four(0)), _line("cu_physics", [1, 0, 0, 0]),
        _line("cudt", [5, 0, 0, 0]), _line("isfflx", 1),
        _line("ifsnow", 1), _line("icloud", 1),
        _line("surface_input_source", 1), _line("do_radar_ref", 1),
        _line("num_soil_layers", 4), _line("sf_urban_physics", four(0)),
        _line("sst_update", 0), "/", "", "&fdda", "/", "",
        "&dynamics", _line("use_theta_m", 0), _line("hybrid_opt", 2),
        _line("etac", 0.2), _line("w_damping", 1), _line("epssm", 0.5),
        _line("diff_opt", four(2)), _line("km_opt", four(4)),
        _line("mix_full_fields", four(True)), _line("diff_6th_opt", four(2)),
        _line("diff_6th_factor", [0.12, 0.10, 0.08, 0.06]),
        _line("diff_6th_slopeopt", four(1)), _line("base_temp", 290.0),
        _line("damp_opt", 3), _line("zdamp", four(5000.0)),
        _line("dampcoef", four(0.2)), _line("khdif", four(0)),
        _line("kvdif", four(0)), _line("non_hydrostatic", four(True)),
        _line("moist_adv_opt", four(1)), _line("scalar_adv_opt", four(1)),
        "/", "", "&bdy_control", _line("spec_bdy_width", 5),
        _line("spec_zone", 1), _line("relax_zone", 4),
        _line("specified", [True, False, False, False]),
        _line("nested", [False, True, True, True]), "/", "",
        "&grib2", "/", "", "&namelist_quilt",
        _line("nio_tasks_per_group", 0), _line("nio_groups", 1), "/", "",
    ])
    effective = "\n".join(sections)
    departures = [
        {"key": "time_control.run_duration", "bundle": "12 hours",
         "effective": f"{stage_run_minutes} minutes",
         "reason": ("real stage walks start + interval_seconds for BC "
                    "generation" if stage == "real"
                    else "N5S controller parameter")},
        {"key": "time_control.end_*", "bundle": "1974-04-04 00:00:00",
         "effective": end.isoformat(), "reason": "derived from run duration"},
        {"key": "time_control.history_interval", "bundle": [60, 15, 15, 15],
         "effective": four(history_minutes),
         "reason": "identical pinned cadence for WRF and gpuwm"},
        {"key": "time_control.restart_interval", "bundle": 180,
         "effective": 0, "reason": "shadow members do not checkpoint"},
        {"key": "time_control.nwp_diagnostics", "bundle": 1,
         "effective": 0, "reason": "binding oracle manifest"},
        {"key": "physics.mp_physics", "bundle": four(55),
         "effective": four(10), "reason": "Morrison matched physics"},
        {"key": "physics.bl_pbl_physics", "bundle": four(11),
         "effective": four(1), "reason": "YSU matched physics"},
        {"key": "physics.do_radar_ref", "bundle": 0, "effective": 1,
         "reason": "registered REFL_10CM metrics"},
        {"key": "dynamics.use_theta_m", "bundle": "omitted (WRF default 1)",
         "effective": 0, "reason": "binding oracle manifest dry-theta branch"},
    ]
    conflicts = [
        {"key": "mp_physics", "bundle": 55, "gpuwm_config": 10,
         "selected": 10, "authority": "gpuwm config + oracle manifest"},
        {"key": "bl_pbl_physics", "bundle": 11, "gpuwm_config": 1,
         "selected": 1, "authority": "gpuwm config + oracle manifest"},
    ]
    provenance = {
        "schema": 1,
        "bundle_namelist": str(bundle_path.resolve()),
        "gpuwm_config": str(config_path.resolve()),
        "selection_policy": "common knobs must agree; conflicts prefer gpuwm then N5S/oracle pins",
        "agreement": agreement,
        "conflicts": conflicts,
        "departures_from_bundle": departures,
        "effective": {
            # The stage's true duration.  The caller's forecast request is
            # retained ONLY under its explicit name (verify4 D5: a real-stage
            # object must never claim the forecast minutes as its duration).
            "run_minutes": stage_run_minutes,
            "requested_forecast_minutes": run_minutes,
            "history_minutes": history_minutes,
            "stage": stage,
            "effective_run_minutes": stage_run_minutes,
            "interval_seconds": FORCING_INTERVAL_SECONDS, "max_dom": 4,
            "physics": {"mp_physics": 10, "bl_pbl_physics": 1,
                        "sf_sfclay_physics": 91, "use_theta_m": 0,
                        "nwp_diagnostics": 0, "do_radar_ref": 1},
        },
        "effective_namelist_sha256": hashlib.sha256(
            effective.encode("utf-8")).hexdigest(),
    }
    return effective, provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-namelist", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-minutes", type=int, default=30)
    parser.add_argument("--history-minutes", type=int, default=5)
    parser.add_argument("--stage", choices=("forecast", "real"),
                        default="forecast")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args(argv)
    text, provenance = build_effective_namelist(
        args.bundle_namelist, args.config, run_minutes=args.run_minutes,
        history_minutes=args.history_minutes, stage=args.stage)
    args.output.write_text(text, encoding="utf-8")
    args.provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
