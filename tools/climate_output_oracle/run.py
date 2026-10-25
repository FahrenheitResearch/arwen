"""Build/replay the unchanged pinned climate routine with desktop WSL gfortran.

python tools/climate_output_oracle/run.py SOURCE_DIR OUTPUT_DIR
OUTPUT_DIR is owned build scratch; fixtures are copied deliberately afterwards.
Only framework stubs are supplied. No arithmetic/source expressions are edited.
"""
from pathlib import Path
import hashlib
import json
import re
import shlex
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gpuwm.core.climate_diagnostics import HISTORY

source, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
original = (source / "module_diag_cl.F").read_bytes()
(out / "module_diag_cl.F").write_bytes(original)
text = original.decode()
signature = text[text.index("   SUBROUTINE clwrf_output_calc"):text.index("!----------------------------------------------------------------------")]
formals = re.sub(r"!.*", "", signature).replace("&", "").split("(", 1)[1].split(")", 1)[0]
formals = formals.replace("\n", "").replace(" ", "").lower().split(",")
driver = (source / "module_diagnostics_driver.F").read_text()
a = driver.index("         CALL clwrf_output_calc(")
b = driver.index("         ENDIF", a)
mapping = {x.lower(): y.lower() for x, y in re.findall(r"(\w+)\s*=grid%(\w+)", driver[a:b], re.I)}
mapping.update(skintemp="tsk", swrad="swdown", swnorm="swddni", clwrfh="diag_int")
inputs = ("t2", "q2", "tsk", "swdown", "swddni", "u10", "v10", "raincv", "rainncv")
slots = tuple(HISTORY)
names = inputs + slots
decl = "\n".join("real :: " + name + "(3,2)" for name in names)
arguments = []
for arg in formals:
    if arg in mapping:
        value = mapping[arg]
    elif arg in ("is_restart", "wind_int", "dt", "xtime", "curr_secs2"):
        value = arg
    elif arg in ("i_start", "i_end", "j_start", "j_end"):
        value = {"i_start": "[1]", "i_end": "[3]", "j_start": "[1]", "j_end": "[2]"}[arg]
    else:
        value = {"ide": "4", "jde": "3", "ime": "3", "jme": "2", "ipe": "3", "jpe": "2"}.get(arg, "1")
    # The driver binds some loop inputs to grid fields; the harness owns one tile.
    value = {"i_start": "[1]", "j_start": "[1]", "num_tiles": "1"}.get(value, value)
    arguments.append(arg + "=" + value)

stubs = """module module_dm
contains
real function wrf_dm_sum_real(x)
real :: x
wrf_dm_sum_real=x
end function
subroutine wrf_dm_maxval
end subroutine
end module
module module_configure
end module
logical function wrf_dm_on_monitor()
wrf_dm_on_monitor=.false.
end function
subroutine wrf_debug(level,message)
integer :: level
character(*) :: message
end subroutine
"""
(out / "stubs.f90").write_text(stubs, newline="\n")
program = """program oracle
use module_diag_cl
implicit none
integer :: s,n,flag,diag_int,wind_int,nsteps,nsteps_wind
real :: dt,xtime,curr_secs2
logical :: is_restart
""" + decl + "\n"
program += "\n".join(name + "=-123.25" for name in slots) + "\n"
program += "read(*,*) n\ndo s=1,n\nread(*,*) dt,xtime,curr_secs2,flag,diag_int,wind_int\nis_restart=flag/=0\n"
program += "\n".join("read(*,*) " + name for name in inputs) + "\n"
program += "call clwrf_output_calc(&\n" + ", &\n".join(arguments) + ")\n"
program += "\n".join("print '(6(Z8.8,1X))',transfer(" + name + ",[0,0,0,0,0,0])" for name in slots)
program += "\nprint '(2(Z8.8,1X))',transfer(real(nsteps),0),transfer(real(nsteps_wind),0)\nenddo\nend program\n"
(out / "main.f90").write_text(program, newline="\n")

times = [(20., float(s * 20), 0, 15, 5) for s in range(100)]
# Native restart's explicit no-reset flag, changing dt, and REAL MOD tolerance.
times += [(30., 0., 1, 15, 5), (30., 870., 1, 15, 5),
          (7., 900., 0, 15, 5), (7., 1197., 0, 15, 5),
          (30., 1200., 0, 10, 5), (30., 1770., 0, 10, 5)]
data = []
for s, (dt, secs, restart, diag, wind) in enumerate(times):
    cell = np.arange(6, dtype=np.float32).reshape(2, 3)
    phase = (s % 7) - 3
    row = np.array([280. + cell + phase, .01 + cell * .001 + phase * .0001,
                    290. + cell + phase, 100. + cell * 2 + phase * 3,
                    150. + cell * 3 + phase * 4, 3. + cell + phase,
                    -4. + cell - phase, .1 + cell * .02 + (s % 4) * .01,
                    .2 + cell * .03 + (s % 3) * .02], dtype=np.float32)
    data.append(row)
data = np.array(data)
clock = np.array([[dt, np.float32(secs / 60), secs, restart, diag, wind]
                  for dt, secs, restart, diag, wind in times], dtype=np.float32)
lines = [str(len(times))]
for clock_row, values in zip(clock, data):
    lines.append(" ".join(format(float(x), ".9g") for x in clock_row))
    lines.extend(" ".join(format(float(x), ".9g") for x in field.ravel()) for field in values)
payload = "\n".join(lines) + "\n"
(out / "inputs.txt").write_text(payload)

posix = "/mnt/" + out.resolve().drive[0].lower() + out.resolve().as_posix()[2:]
cmd = "cd " + shlex.quote(posix) + " && gfortran -cpp -DNMM_CORE=0 -ffree-form -ffree-line-length-none -O0 -ffp-contract=off -fno-tree-vectorize stubs.f90 module_diag_cl.F main.f90 -o oracle && ./oracle < inputs.txt"
result = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-lc", cmd],
                        check=True, capture_output=True, text=True)
(out / "words.txt").write_text(result.stdout)
words = np.array([int(x, 16) for x in result.stdout.split()], dtype=np.uint32)
rows = words.reshape(len(times), len(slots) * 6 + 2)
np.savez_compressed(out / "vectors.npz", inputs=data, clock=clock,
                    expected=rows[:, :-2].reshape(len(times), len(slots), 2, 3),
                    counters=rows[:, -2:], slots=np.array(slots))
receipt = {"source_sha256": hashlib.sha256(original).hexdigest(),
           "authority": "NOAA-EMC/HRRR 40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827",
           "flags": "-O0 -ffp-contract=off -fno-tree-vectorize", "calls": len(times),
           "fields": len(slots), "compared_float32_words": int(words.size),
           "compiler": subprocess.check_output(["wsl", "-d", "Ubuntu-24.04", "--", "gfortran", "--version"], text=True).splitlines()[0]}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
