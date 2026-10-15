"""Exercise the real import CLI with the public WRF em_real namelist.

Select Noah-MP and its supported MYNN coupling explicitly. Generate a WPS
companion carrying the public input's dates and grid topology. No generation
override is added to the default arm.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from gpuwm.namelist_import import parse_namelist
from gpuwm.experiment import load_experiment
from gpuwm.physics_source_defaults import with_physics_selector_comment
from gpuwm.physics_registry import physics_registry
from gpuwm.checkpoint_identity import NOAHMP_SNOW_AGE_ALGORITHM_IDENTITIES

source, work = map(Path, sys.argv[1:])
work.mkdir(parents=True, exist_ok=True)
data = parse_namelist(source)
count = int(data["domains"]["max_dom"][0])
data["physics"].update(sf_surface_physics=[4] * count,
                        sf_sfclay_physics=[5] * count, bl_pbl_physics=[5] * count,
                        num_soil_layers=[4])


def spelling(value):
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def emit(sections):
    return "\n".join("&" + section + "\n" + "\n".join(
        " " + key + " = " + ", ".join(map(spelling, values)) + ","
        for key, values in fields.items()) + "\n/" for section, fields in sections.items()) + "\n"


tc, domains = data["time_control"], data["domains"]
for axis in ("dx", "dy"):
    spacing = [domains[axis][0]]
    for i in range(1, count):
        spacing.append(spacing[domains["parent_id"][i] - 1] /
                       domains["parent_grid_ratio"][i])
    domains[axis] = spacing
dates = {}
for prefix in ("start", "end"):
    dates[prefix + "_date"] = ["%04d-%02d-%02d_%02d:00:00" % tuple(
        tc[prefix + "_" + unit][i] for unit in ("year", "month", "day", "hour"))
        for i in range(count)]
geo = {name: domains[name] for name in (
    "parent_id", "parent_grid_ratio", "i_parent_start", "j_parent_start", "e_we", "e_sn", "dx", "dy")}
geo["parent_id"] = [1, *geo["parent_id"][1:]]
geo["dx"], geo["dy"] = [domains["dx"][0]], [domains["dy"][0]]
geo.update(geog_data_res=["default"] * count, map_proj=["lambert"],
           ref_lat=[40.0], ref_lon=[-100.0], truelat1=[30.0], truelat2=[60.0],
           stand_lon=[-100.0], geog_data_path=["./geog"])
wps = work / "namelist.wps"
wps.write_text(emit({"share": {"wrf_core": ["ARW"], "max_dom": [count], **dates,
                                "interval_seconds": tc["interval_seconds"], "io_form_geogrid": [2]},
                     "geogrid": geo, "ungrib": {"out_format": ["WPS"], "prefix": ["GFS"]},
                     "metgrid": {"fg_name": ["GFS"]}}))
receipt = {"public_input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "changes": {"sf_surface_physics": 4, "sf_sfclay_physics": 5,
                       "bl_pbl_physics": 5, "num_soil_layers": 4,
                       "dx_dy": "child spacings derived from the public parent chain"},
           "wps_companion": "public input dates and topology, generic Lambert projection",
           "arms": []}
for generation in (None, "wrf_461"):
    label = generation or "default"
    inp, output = work / (label + ".input"), work / (label + ".toml")
    text = emit(data)
    if generation:
        text = with_physics_selector_comment(text, {"noahmp_snow_age": generation})
    inp.write_text(text)
    command = [sys.executable, "-m", "gpuwm.cli", "import-namelist", str(wps), str(inp),
               "--output", str(output)]
    proc = subprocess.run(command, capture_output=True, text=True)
    (work / (label + ".log")).write_text(proc.stdout + proc.stderr)
    arm = {"command": command, "exit_code": proc.returncode}
    if proc.returncode == 0:
        experiment = load_experiment(output)
        arm["generations"] = [dc.run.noahmp_snow_age for dc in experiment.domains]
        arm["restart_algorithms"] = [NOAHMP_SNOW_AGE_ALGORITHM_IDENTITIES[v] for v in arm["generations"]]
        assert arm["generations"] == [generation or "wrf_470"] * count
    receipt["arms"].append(arm)
receipt["registry_row"] = physics_registry()["parameters"]["noahmp_snow_age"]
(work / "receipt.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
raise SystemExit(any(arm["exit_code"] for arm in receipt["arms"]))
