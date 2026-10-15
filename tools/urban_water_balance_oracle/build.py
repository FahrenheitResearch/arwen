"""Compile the unmodified release scheme with the existing WRF argument driver.

The driver adds stress inputs and the required 4.8 state arguments. Tree and
vegetated-ground physics are disabled explicitly: this checks water balance.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
tag = sys.argv[2]
defined = "--defined-reference" in sys.argv[3:]
here = Path(__file__).resolve().parent
src = root / "sources" / tag
build = root / ("oracle-" + tag + ("-defined" if defined else ""))
build.mkdir(parents=True, exist_ok=True)
base = here.parent / "urban_wrf471_oracle"
driver = (base / "ucm_column_oracle.F90").read_text()
driver = driver.replace("nscen = 9", "nscen = 15")
driver = driver.replace("    case (9)            ! first level", "    case (10:15)\n      tsurf0 = 303.\n      tlay0 = (/301.,298.,295./)\n    case (9)            ! first level")
driver = driver.replace("    cmcr = 0.", """    if (sc >= 10) then
      drelr = 0.0001
      drelb = 0.0001
      drelg = 0.0001
      flxhumr = 0.00003
      flxhumb = 0.00003
      flxhumg = 0.00003
      if (sc == 14) then
        drelr = 0.0000001
        drelb = 0.0000001
        drelg = 0.0000001
      end if
      if (sc == 15) then
        flxhumr = -0.00003
        flxhumb = -0.00003
        flxhumg = -0.00003
      end if
    end if
    cmcr = 0.""")
# Use the warm forcing from scenario 1, then replace only rain in each arm.
start = driver.index("    case (1)", driver.index("  subroutine scenario_forcing"))
end = driver.index("    case (2)", start)
warm = driver[start:end]
arms = ""
for sc, rain in ((10,"1.e-8"),(11,"0."),(12,"transfer(1,0.)"),(13,"100."),(14,"1.e-8"),(15,"1.e-8")):
    arms += warm.replace("case (1)", f"case ({sc})").replace("rain = 0.", f"rain = {rain}")
driver = driver[:end] + arms + driver[end:]
if tag == "v4.8.0":
    driver = driver.replace("    real :: trlend, tblend, tglend, akanda, alh", "    real :: trlend, tblend, tglend, akanda, alh\n    real :: albvg_, albt_, fvg_, rtree_, rtree_m_, rw_m_, htree_, dtree_, lai_tree_, lai_veg_, z0vg_, z0hvg_, capvg_, epsvg_, epst_")
    driver = driver.replace("                      numdir, street_direction, street_width,", "                      albvg_, albt_, fvg_, rtree_, rtree_m_, rw_m_, htree_, dtree_, &\n                      lai_tree_, lai_veg_, z0vg_, z0hvg_, capvg_, epsvg_, epst_, &\n                      numdir, street_direction, street_width,")
    driver = driver.replace("  real :: lf_urb_s", "  real :: tvg, tt, xxxvg, tvgl(nsoil), smg(nsoil), cmcg, flxhumvg, flxhumt\n  real :: lf_urb_s")
    driver = driver.replace("                   lf_urb_s, z0_urb, vegfrac)", "                   tvg, tt, xxxvg, tvgl, smg, cmcg, flxhumvg, flxhumt, &\n                   lf_urb_s, z0_urb, vegfrac)")
    driver = driver.replace("    vegfrac = 0.", "    vegfrac = 0.\n    tvg=tsurf0; tt=tsurf0; xxxvg=0.; tvgl=trl; smg=0.2\n    cmcg=0.; flxhumvg=0.; flxhumt=0.")
    driver = driver.replace("  call urban_param_init(dzr, dzb, dzg, nsoil, 1, use_lcz, .false.)", "  call urban_param_init(dzr, dzb, dzg, nsoil, 1, use_lcz, .false.)\n  TREEOPTION=0\n  FVG_TBL=0.")
(build / "driver.F90").write_text(driver)
for f in ("URBPARM.TBL", "URBPARM_LCZ.TBL"):
    shutil.copyfile(src / f, build / f)
flags = ["-O0", "-g", "-cpp", "-ffree-form", "-ffree-line-length-none", "-fallow-argument-mismatch", "-ffp-contract=off",
         "-Dwrfmodel", "-DEM_CORE=1", "-DNMM_CORE=0", "-DRWORDSIZE=4", "-DIWORDSIZE=4", "-DDWORDSIZE=8", "-DLWORDSIZE=4"]
commands = []
def run(args):
    commands.append(args)
    subprocess.run(args, cwd=build, check=True)
objects = []
if defined:
    assert tag == "v4.8.0"
    candidate = build / "module_sf_urban.F"
    run([sys.executable,str(here.parent/"urban_wrf480_oracle/defined_state.py"),
         str(src/"module_sf_urban.F"),str(candidate)])
    flags += ["-fcheck=all", "-ffpe-trap=invalid,zero,overflow"]
    run(["gcc","-O0","-fno-fast-math","-c",
         str(here.parent/"urban_bem_regime_oracle/libm_shim.c"),"-o","reference_rounding.o"])
for path in (base / "stub_ucm.F90", src / "module_model_constants.F", src / "module_wrf_error.F", src / "module_sf_urban.F", build / "driver.F90"):
    if defined and path.name == "module_sf_urban.F":
        path = candidate
    obj = path.stem + ".o"
    run(["gfortran", *flags, "-finit-real=snan" if defined else "-finit-real=zero", "-c", str(path), "-o", obj])
    objects.append(obj)
run(["gfortran", "-o", "run", *objects,*(["reference_rounding.o"] if defined else [])])
variants = ["nlcd default -9 -9 -9 -9 -9 -9 -9 -9 -9", "lcz default -9 -9 -9 -9 -9 -9 -9 -9 -9",
 "nlcd ch1 1 -9 -9 -9 -9 -9 -9 -9 -9", "nlcd ts2 -9 2 -9 -9 -9 -9 -9 -9 -9",
 "nlcd ch1ts2 1 2 -9 -9 -9 -9 -9 -9 -9", "nlcd ahalh -9 -9 1 1 -9 -9 -9 -9 -9",
 "nlcd imp2 -9 -9 -9 -9 2 -9 -9 -9 -9", "nlcd bound2 -9 -9 -9 -9 -9 -9 -9 -9 2",
 "lcz imp2ahalh -9 -9 1 1 2 -9 -9 -9 -9", "nlcd gr -9 -9 -9 -9 -9 -9 1 0.5 -9",
 "nlcd griri -9 -9 1 -9 2 1 1 0.7 -9", "lcz gr -9 -9 -9 -9 -9 -9 1 0.4 -9"]
fixture = build / "fixture"
fixture.mkdir(exist_ok=True)
import gzip
for line in variants:
    table, variant, *switches = line.split()
    prefix = fixture / f"ucm-{table}-{variant}"
    run([str(build / "run"), table, variant, str(prefix), *switches])
    csv = prefix.with_suffix(".csv")
    csv.with_suffix(".csv.gz").write_bytes(gzip.compress(csv.read_bytes(), mtime=0))
receipt = {"tag":tag,"compiler":subprocess.check_output(["gfortran","--version"]).decode(),
 "defined_reference":defined,"reference_rounding_wrapper":defined,
 "commands":commands,"scalar_kinds":{"real":4,"integer":4,"double":8},
 "sources":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()},
 "driver_sha256":hashlib.sha256(driver.encode()).hexdigest(),
 "defined_corrections":"zero-initialized local green-roof ETR; tloc only exercised with AHOPTION=1 when irrigation is on"}
(build / "build-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
