"""Byte-extract the native moisture scheduling block and label its correction."""
from pathlib import Path
import argparse
import hashlib
import json

SOURCE_SHA256 = "7662f29bb003697d08ccee585e272a3ed7362edc73bcef5ea8dc95d100d9e578"


def extract(source, destination):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Moisture scheduling needs the pinned WRF v4.7.1 driver")
    start = raw.index(b"    ! time - assume dt does not change")
    end = raw.index(b"!$OMP CRITICAL(FIRE_DRIVER_CRIT)", raw.index(b"    elseif(itimestep.eq.1.and.fmoist_interp)then", start))
    block = raw[start:end]
    old = b"time_start = itimestep * dt"
    new = b"time_start = max(itimestep-1,0) * dt"
    if block.count(old) != 1:
        raise ValueError("Native driver time expression changed")
    corrected = block.replace(old, new)
    header = b"""module moisture_driver_control
use module_fr_fire_util
implicit none
type clock_record
  integer :: itimestep=0
  real :: dt=0.,fmoist_lasttime=0.,fmoist_nexttime=0.
end type
type settings_record
  logical :: fmoist_run=.false.,fmoist_interp=.false.,fmoist_only=.false.
  integer :: fmoist_freq=0
  real :: fmoist_dt=600.
end type
contains
"""
    result = header
    for name, body in (("decide_original", block), ("decide_clock_corrected", corrected)):
        result += f"subroutine {name}(grid,config_flags,fire_ifun_start,fire_ifun_end,dt_moisture,run_advance_moisture,run_fuel_moisture,fire_run)\n".encode()
        result += b"""type(clock_record),intent(inout)::grid
type(settings_record),intent(in)::config_flags
integer,intent(in)::fire_ifun_start,fire_ifun_end
real,intent(inout)::dt_moisture
logical,intent(out)::run_advance_moisture,run_fuel_moisture,fire_run
logical::fmoist_run,fmoist_interp,moisture_initializing
integer::itimestep
real::dt,time_start,moisture_time
character(len=256)::msg
itimestep=grid%itimestep
""" + body + f"end subroutine {name}\n".encode()
    result += b"end module moisture_driver_control\n"
    Path(destination).write_bytes(result)
    receipt = dict(source_sha256=SOURCE_SHA256, original_block_sha256=hashlib.sha256(block).hexdigest(),
                   corrected_block_sha256=hashlib.sha256(corrected).hexdigest(),
                   wrapper_sha256=hashlib.sha256(result).hexdigest(),
                   clock_correction=dict(before=old.decode(), after=new.decode()),
                   initialization_correction="first actual advance initializes; original caller initializes only at itimestep==1",
                   support="original compiled WRF advance_moisture and fuel_moisture")
    Path(destination).with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("destination")
    a = p.parse_args()
    extract(a.source, a.destination)
