"""Compile the pinned fork's unchanged monthly_interp_to_date routine.

Usage: python tools/ruc_wrf461_oracle/build_monthly_interp.py SOURCE BUILD_DIR
SOURCE is the v4.1.21 module_initialize_real.F file. Only the calendar and
interior-point dependencies are stubbed; the interpolator is copied verbatim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

SOURCE_SHA256 = "e45234bd25dbc7a5d747384cb78781da1feda92f97bdcf6f691f3807c3b006d2"
DATES = ("2026-01-01", "2026-01-15", "2026-02-14", "2026-03-15",
         "2026-04-30", "2026-05-16", "2026-06-15", "2026-07-04",
         "2026-08-31", "2026-09-01", "2026-10-02", "2026-11-30",
         "2026-12-31", "2024-02-29", "2024-12-16")


def main():
    source, build = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("source differs from the pinned v4.1.21 file")
    text = raw.decode()
    start = text.index("   SUBROUTINE monthly_interp_to_date (")
    end = text.index("   END SUBROUTINE monthly_interp_to_date", start)
    end += len("   END SUBROUTINE monthly_interp_to_date")
    routine = text[start:end]
    build.mkdir(parents=True, exist_ok=True)
    monthly = np.random.default_rng(19).uniform(0, 60, (12, 7, 5)).astype(np.float32)
    words = monthly.reshape(12, 35).T.view(np.int32)
    np.savetxt(build / "input.txt", words, fmt="%d")
    declarations = ", &\n".join("'" + date + "_00:00:00.0000'" for date in DATES)
    wrapper = """module oracle
      implicit none
      integer :: em_width=0, hold_ups=0
      contains
      logical function skip_middle_points_t(ids,ide,jds,jde,i,j,e,h)
        integer :: ids,ide,jds,jde,i,j,e,h
        skip_middle_points_t=.false.
      end function
      subroutine get_julgmt(date,year,jday,gmt)
        character(len=*) :: date
        integer :: year,jday,m,d
        integer :: starts(12)=[0,31,59,90,120,151,181,212,243,273,304,334]
        real :: gmt
        read(date(1:4),*) year
        read(date(6:7),*) m
        read(date(9:10),*) d
        jday=starts(m)+d
        if(m>2.and.mod(year,4)==0.and.(mod(year,100)/=0.or.mod(year,400)==0)) jday=jday+1
        gmt=0.
      end subroutine
""" + routine + "\nend module\n" + """program driver
      use oracle
      implicit none
      real :: monthly(35,12,1), result(35,1)
      integer :: words(12), i,m,d
      character(len=24) :: dates(15)=[character(len=24) :: &
""" + declarations + """ ]
      open(10,file='input.txt')
      do i=1,35
        read(10,*) words
        do m=1,12
          monthly(i,m,1)=transfer(words(m),0.)
        enddo
      enddo
      close(10)
      do d=1,15
        call monthly_interp_to_date(monthly,dates(d),result, &
          1,36,1,2,1,1,1,35,1,1,1,1,1,35,1,1,1,1)
        do i=1,35
          write(*,'(A,1X,I0,1X,I0)') dates(d)(1:10),i,transfer(result(i,1),0)
        enddo
      enddo
end program
"""
    f90 = build / "monthly.f90"
    f90.write_text(wrapper)
    subprocess.run(["gfortran", "-O0", "-ffree-line-length-none", str(f90),
                    "-o", str(build / "monthly")], cwd=build, check=True)
    run = subprocess.run([str(build / "monthly")], cwd=build,
                         capture_output=True, text=True, check=True)
    result = np.array([int(line.split()[2]) for line in run.stdout.splitlines()],
                      np.int32).view(np.float32).reshape(15, 7, 5)
    fixture = {"source_sha256": SOURCE_SHA256, "dates": DATES,
               "shape": [7, 5], "monthly_bits": monthly.view(np.uint32).tolist(),
               "output_bits": result.view(np.uint32).tolist(),
               "compiler": subprocess.check_output(["gfortran", "--version"],
                                                    text=True).splitlines()[0],
               "routine_sha256": hashlib.sha256(routine.encode()).hexdigest()}
    (build / "monthly_interp.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print("525 REAL outputs from the unchanged pinned interpolator")


if __name__ == "__main__":
    main()
