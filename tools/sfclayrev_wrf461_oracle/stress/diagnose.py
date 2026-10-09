"""Instrument an owned copy of pinned WRF to locate the first nonfinite operation."""
import os
from pathlib import Path
import subprocess

root = Path(os.environ['SFCLAYREV_CHECK_ROOT'])
source = Path(os.environ['WRF_SOURCE_ROOT']) / 'phys/physics_mmm/sf_sfclayrev.F90'
text = source.read_text()
needle = '          z0t=znt(i)/exp(czil*karman*sqrt(restar))'
assert text.count(needle) == 1
text = text.replace(needle, needle + '''
          if (iz0tlnd.eq.2.and.(i.eq.92.or.i.eq.221.or.i.eq.309.or.i.eq.444)) then
            write(*,*) 'SCALAR_ROUGHNESS',i,ust(i),znt(i),visc,restar, &
              czil*karman*sqrt(restar),exp(czil*karman*sqrt(restar)), &
              z0t,(za(i)+z0t)/z0t,(2.+z0t)/z0t
          endif
''')
build = root / 'oracle'
path = root / 'diagnostic-scheme.F90'
path.write_text(text)
subprocess.run(['gfortran', '-c', '-O0', '-cpp', '-ffree-form', '-ffree-line-length-none',
                '-I', str(build), str(path), '-o', 'diagnostic-scheme.o'], cwd=build, check=True)
subprocess.run(['gfortran', '-O0', '-ffree-form', '-ffree-line-length-none', '-I', '.',
                str(root / 'extra-driver.F90'), 'ccpp_kind_types.o', 'diagnostic-scheme.o',
                'module_sf_sfclayrev_O0.o', '-o', 'run_diagnostic'], cwd=build, check=True)
subprocess.run([str(build / 'run_diagnostic'), str(root / 'extra/sfclayrev-inputs.hex'),
                str(root / 'diagnostic-outputs.hex')], check=True)
