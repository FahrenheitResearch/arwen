"""Run the real import CLI and record selected settings and registry rows."""
from pathlib import Path
import dataclasses
import hashlib
import json
import subprocess
import sys
import tomllib
from test_urban_registry import _generic_hierarchy_pair
from gpuwm.experiment import load_experiment
from gpuwm.namelist_import import parse_namelist
from gpuwm.physics_registry import physics_registry,registry_sha256
from gpuwm.io.restart import _configuration_fingerprint

root = Path(sys.argv[1]).resolve()
dest = root/"cli"
dest.mkdir(exist_ok=True)
wps,inp = _generic_hierarchy_pair(dest,1)
inp.write_text(inp.read_text().replace("&physics", "&physics\n sf_urban_physics=1,\n"))
out = dest/"experiment.toml"
command = [sys.executable,"-m","gpuwm.cli","import-namelist",str(wps),str(inp),"--name","urban-water-balance","--output",str(out)]
result = subprocess.run(command,capture_output=True,text=True)
(dest/"import.log").write_text(result.stdout+result.stderr)
assert result.returncode == 0,result.stdout+result.stderr
experiment = load_experiment(out)
resolved = experiment.domains[0].run
assert resolved.urban_water_balance == "wrf_480"
registry = physics_registry()
(root/"cli.json").write_text(json.dumps({"command":command,"exit":result.returncode,
    "input_sha256":hashlib.sha256(inp.read_bytes()).hexdigest(),
    "wps_sha256":hashlib.sha256(wps.read_bytes()).hexdigest(),
    "config_sha256":_configuration_fingerprint(resolved),
    "selected":{"sf_urban_physics":resolved.sf_urban_physics,"urban_water_balance":resolved.urban_water_balance},
    "registry_sha256":registry_sha256(),"registry_parameter":registry["parameters"]["urban_water_balance"],
    "urban_row":registry["components"]["urban"]["options"]["slucm"],
    "namelist_kind":"standard synthetic WRF namelist, not an upstream public case"},indent=2)+"\n")

# Try the actual public WRF sample independently. Only the requested urban
# selector is added to it. WPS geometry and clock columns follow that sample.
public = root/"public-cli"
public.mkdir(exist_ok=True)
public_receipt = root/"public-cli.json"
# A failed preparation must not leave a previous sample's receipt to be
# mistaken for this attempt by the calling script.
public_receipt.unlink(missing_ok=True)
public_relative = sys.argv[2] if len(sys.argv) > 2 else "test/em_real/namelist.input"
data = subprocess.check_output(["git","-C",str(root/"sources/WRF"),"show",
                                f"v4.7.1:{public_relative}"])
(public/"upstream.input").write_bytes(data)
p = public/"namelist.input"
import re
text = re.sub(r"(?mi)^\s*sf_urban_physics\s*=.*$", " sf_urban_physics=1,",data.decode())
if "sf_urban_physics=1" not in text:
    text = re.sub(r"(?mi)^(\s*&physics\b)",r"\1\n sf_urban_physics=1,",text,count=1)
assert "sf_urban_physics=1" in text
p.write_text(text)
parsed = parse_namelist(p)
d,t = parsed["domains"],parsed["time_control"]
n = int(d["max_dom"][0])
def column(values):
    return ", ".join(".true." if v is True else ".false." if v is False else
                     repr(v) if isinstance(v,str) else str(v) for v in values[:n])
def date(prefix,i):
    return f"{int(t[prefix+'_year'][i]):04d}-{int(t[prefix+'_month'][i]):02d}-{int(t[prefix+'_day'][i]):02d}_{int(t[prefix+'_hour'][i]):02d}:00:00"
wps_text = "&share\n wrf_core='ARW',\n"
wps_text += f" max_dom={n},\n start_date={column([date('start',i) for i in range(n)])},\n end_date={column([date('end',i) for i in range(n)])},\n interval_seconds={t.get('interval_seconds',[21600])[0]},\n/\n&geogrid\n"
for key in ("parent_id","parent_grid_ratio","i_parent_start","j_parent_start","e_we","e_sn"):
    values = list(d[key])
    if key == "parent_id": values[0] = 1
    wps_text += f" {key}={column(values)},\n"
wps_text += f" dx={d['dx'][0]}, dy={d['dy'][0]},\n map_proj='lambert', ref_lat=39.7, ref_lon=-83.9,\n truelat1=30., truelat2=60., stand_lon=-83.9, geog_data_path='/geog',\n/\n"
w = public/"namelist.wps"
w.write_text(wps_text)
public_command = [sys.executable,"-m","gpuwm.cli","import-namelist",str(w),str(p),"--output",str(public/"experiment.toml")]
run = subprocess.run(public_command,capture_output=True,text=True)
(public/"import.log").write_text(run.stdout+run.stderr)
public_receipt.write_text(json.dumps({"command":public_command,"exit":run.returncode,
    "upstream":f"https://github.com/wrf-model/WRF/blob/v4.7.1/{public_relative}",
    "upstream_sha256":hashlib.sha256(data).hexdigest(),"selector_added":"sf_urban_physics=1",
    "modified_input_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
    "wps_sha256":hashlib.sha256(w.read_bytes()).hexdigest(),
    "stdout":run.stdout,"stderr":run.stderr},indent=2)+"\n")
if run.returncode == 0:
    public_run = load_experiment(public/"experiment.toml").domains[0].run
    assert public_run.urban_water_balance == "wrf_480"
    payload = json.loads(public_receipt.read_text())
    payload["selected"] = {"sf_urban_physics":public_run.sf_urban_physics,
                           "urban_water_balance":public_run.urban_water_balance}
    payload["configuration_sha256"] = _configuration_fingerprint(public_run)
    payload["registry_sha256"] = registry_sha256()
    payload["registry_parameter"] = registry["parameters"]["urban_water_balance"]
    payload["urban_row"] = registry["components"]["urban"]["options"]["slucm"]
    public_receipt.write_text(json.dumps(payload,indent=2)+"\n")
