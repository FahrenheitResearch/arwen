"""ncar_door_dir.py SRC DST CLOCK : one NCAR v4.4 benchmark directory made ready for WOOF's WRF-input door, exactly as the
2026-10-06 benchmark check's arm.sh did (CLOCK-CHECK-NCAR-2026-10-06/scripts/arm.sh, its namelist translation verbatim),
with CLOCK = measured | pinned | local_face named by a gpuwm selector comment (measured writes none).  wrfinput_d01 is
copied (I_PARENT_START/J_PARENT_START 0 -> 1 on the copy), wrfbdy_d01 linked or copied.  CPU only."""
import os, re, shutil, sys
import netCDF4
src, dst, clock = sys.argv[1:4]
os.makedirs(dst, exist_ok=True)
shutil.copy(os.path.join(src, "wrfinput_d01"), os.path.join(dst, "wrfinput_d01"))
b = os.path.join(dst, "wrfbdy_d01")
if not os.path.exists(b):
    try:
        os.symlink(os.path.abspath(os.path.join(src, "wrfbdy_d01")), b)
    except OSError:
        shutil.copy(os.path.join(src, "wrfbdy_d01"), b)
d = netCDF4.Dataset(os.path.join(dst, "wrfinput_d01"), "a"); changed = {}
for a in ("I_PARENT_START", "J_PARENT_START"):
    if a in d.ncattrs() and int(d.getncattr(a)) == 0: changed[a] = "0 -> 1"; d.setncattr(a, 1)
d.close(); print("wrfinput", changed)
t = open(os.path.join(src, "namelist.input")).read()
tropical = "tropical" in t
suite = ({"mp_physics": 6, "ra_lw_physics": 4, "ra_sw_physics": 4, "sf_sfclay_physics": 91, "sf_surface_physics": 2, "bl_pbl_physics": 1, "cu_physics": 16} if tropical
         else {"mp_physics": 8, "ra_lw_physics": 4, "ra_sw_physics": 4, "sf_sfclay_physics": 2, "sf_surface_physics": 2, "bl_pbl_physics": 2, "cu_physics": 16})
for k, v in suite.items():
    t, n = re.subn(r"(?m)^(\s*%s\s*=\s*)-1\b((?:\s*,\s*-1\b)*)" % k, lambda m: m.group(1) + ", ".join([str(v)] * (1 + m.group(2).count("-1"))), t)
    if n: print("expanded", k, "->", v)
t = re.sub(r"(?m)^\s*physics_suite\s*=.*\n", "", t)
t = re.sub(r"(?m)^( *[ij]_parent_start *= *)0,", r"\g<1>1,", t)
t = re.sub(r"(?m)^ *(perturb_input|ensdim|maxens|maxens2|maxens3|maxiens|use_baseparam_fr_nml|restart_interval) *=.*\n", "", t)
t = re.sub(r"(?m)^(\s*num_soil_layers\s*=\s*)1\b", r"\g<1>4", t)
m = re.search(r"(?m)^\s*time_step\s*=\s*(\d+)", t); dt = int(m.group(1)) if m else 0
mr = re.search(r"(?m)^(\s*radt\s*=\s*)([0-9.]+)", t)
if mr and dt and (float(mr.group(2)) * 60) % dt:
    new_radt = max(1, int(float(mr.group(2)) * 60 // dt) * dt // 60)
    while (new_radt * 60) % dt: new_radt -= 1
    t = t[:mr.start()] + mr.group(1) + str(new_radt) + t[mr.end():]; print("radt", mr.group(2), "->", new_radt, "min (dt", dt, "s)")
if clock in ("pinned", "local_face"):
    t = '! gpuwm-physics-selectors-v1: {"terrain_clock": "%s"}\n' % clock + t
    print("terrain_clock selector:", clock)
elif clock != "measured":
    sys.exit("CLOCK must be measured, pinned or local_face")
open(os.path.join(dst, "namelist.input"), "w").write(t)
