"""Per-arm copy of a real.exe pair that both WRF and WOOF can read for that arm.

    python prep_inputs.py ROOT CASE_DIR WRF_BUILD [FRAME0_ROOT]

FRAME0_ROOT, when given, holds an earlier WRF run of each arm from the same
pair (FRAME0_ROOT/<arm>/wrfout_d01_*); its t = 0 frame is used instead of a
new probe run.

A real.exe pair is made for one physics suite: its header names that
suite's MP_PHYSICS and it carries that suite's moisture and number species.
WRF runs another suite from it anyway: species the arm's scheme does not
have are never read, and species it has that the file lacks start at the
values WRF's own initialisation gives them.  WOOF refuses both (a header
naming another scheme, an inactive species, a missing one;
gpuwm/ingest/wrfinput_identity.py and gpuwm/ingest/wrfinput.py), so each arm
gets its own copy written to say exactly what WRF made of the file:

1. a short WRF run of the arm on the original pair (run_wrf_arm.sh, 0 steps
   of output: the t = 0 history frame) shows which species WRF's registry
   holds for the scheme and the values it starts them at;
2. the copy keeps every variable of the original byte for byte, drops the
   moisture/number species the arm's scheme does not hold, adds the ones it
   holds and the file lacked with WRF's own t = 0 words, and sets
   MP_PHYSICS to the arm's scheme.

Both sides then read the copy.  The comparison reruns WRF on it, so WRF's
side is the copy too.
"""

from __future__ import annotations

import glob
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

#: Moisture and number species a wrfinput can carry (WOOF's door refuses an
#: inactive one, gpuwm.ingest.wrfinput.ALL_MOISTURE_WRFINPUT's 3-D species).
SPECIES = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL",
           "QNDROP", "QNCLOUD", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
           "QNHAIL", "QNCCN", "QNWFA", "QNIFA", "QVGRAUPEL", "QVHAIL", "QIR",
           "QIB")


def _mp(namelist: Path) -> int:
    return int(re.search(r"mp_physics\s*=\s*(-?\d+)",
                         namelist.read_text()).group(1))


def _probe(root: Path, case: Path, build: Path, arm: str) -> Path:
    """WRF's t = 0 history frame for the arm, from the ORIGINAL pair."""
    probe = root / "probe" / arm
    if probe.exists():
        shutil.rmtree(probe)
    probe.mkdir(parents=True)
    for f in (build / "run").iterdir():
        if f.name not in ("namelist.input", "wrf.exe", "real.exe",
                          "ndown.exe", "tc.exe", "wrfinput_d01",
                          "wrfbdy_d01"):
            (probe / f.name).symlink_to(f)
    shutil.copyfile(build / "main" / "wrf.exe", probe / "wrf.exe")
    (probe / "wrf.exe").chmod(0o755)
    (probe / "wrfinput_d01").symlink_to(case / "wrfinput_d01")
    (probe / "wrfbdy_d01").symlink_to(case / "wrfbdy_d01")
    text = (root / "cases" / arm / "namelist.input").read_text()
    # One step: the t = 0 frame is written before the first step.
    text = re.sub(r"(run_minutes\s*=\s*)\d+", r"\g<1>0", text)
    dt = int(re.search(r"time_step\s*=\s*(\d+)", text).group(1))
    text = re.sub(r"(run_seconds\s*=\s*)\d+", rf"\g<1>{dt}", text)
    text = re.sub(r"(end_minute\s*=\s*)\d+", r"\g<1>0", text)
    text = re.sub(r"(end_second\s*=\s*)\d+", rf"\g<1>{dt}", text)
    (probe / "namelist.input").write_text(text)
    subprocess.run("ulimit -s unlimited; ./wrf.exe > wrf.stdout 2>&1",
                   shell=True, cwd=probe, check=True)
    return Path(sorted(glob.glob(str(probe / "wrfout_d01_*")))[0])


def write_copy(src: Path, dst: Path, frame0: Path, mp: int) -> dict:
    with Dataset(frame0) as ref:
        ref.set_auto_maskandscale(False)
        held = {n for n in SPECIES if n in ref.variables}
        with Dataset(src) as inp:
            inp.set_auto_maskandscale(False)
            present = {n for n in SPECIES if n in inp.variables}
            drop = sorted(present - held)
            add = sorted(held - present)
            with Dataset(dst, "w", format=inp.data_model) as out:
                out.set_fill_off()
                for name, dim in inp.dimensions.items():
                    out.createDimension(
                        name, None if dim.isunlimited() else len(dim))
                attrs = {k: inp.getncattr(k) for k in inp.ncattrs()}
                attrs["MP_PHYSICS"] = np.int32(mp)
                out.setncatts(attrs)
                for name, var in inp.variables.items():
                    if name in drop:
                        continue
                    v = out.createVariable(name, var.dtype, var.dimensions)
                    v.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                    v[:] = var[:]
                for name in add:
                    var = ref.variables[name]
                    v = out.createVariable(name, var.dtype, var.dimensions)
                    v.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                    v[:] = var[0:1]
    return {"dropped": drop, "added_from_wrf_t0": add}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root, case, build = Path(argv[0]), Path(argv[1]), Path(argv[2])
    frame0_root = Path(argv[3]) if len(argv) > 3 else None
    for arm_dir in sorted(p for p in (root / "cases").iterdir()
                          if p.is_dir()):
        arm = arm_dir.name
        earlier = (sorted(glob.glob(str(frame0_root / arm / "wrfout_d01_*")))
                   if frame0_root else [])
        frame0 = Path(earlier[0]) if earlier else _probe(
            root, case, build, arm)
        out = root / "inputs" / arm
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "wrfinput_d01"
        if dst.exists():
            dst.unlink()
        change = write_copy(case / "wrfinput_d01", dst, frame0,
                            _mp(arm_dir / "namelist.input"))
        print(arm, change)


if __name__ == "__main__":
    main()
