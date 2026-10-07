"""Build observer copies after a pinned original build, inside lane scratch.

Each original numerical statement is preserved. Added observer writes expose
local plume tops and iteration counts. Drivers compare original and observed
emissions word for word before publishing diagnostics. The source pins are
checked before any observational transformation or compilation.
"""
from pathlib import Path
import hashlib
import argparse
import subprocess

p=argparse.ArgumentParser(); p.add_argument('arm',choices=['wrf','gsl']); p.add_argument('source_root'); p.add_argument('build'); p.add_argument('harness'); p.add_argument('--experiment',action='store_true')
a=p.parse_args(); root=Path(a.source_root).resolve(); build=Path(a.build).resolve(); h=Path(a.harness).resolve()
pins={}
for f in (h/'SOURCES.sha256',h/'SOURCES-smoke.sha256'):
    for l in f.read_text().splitlines():
        if l and not l.startswith('#'):
            digest,path=l.split(None,1); pins[path.lstrip('*')]=digest

def read(path):
    b=(root/path).read_bytes()
    if hashlib.sha256(b).hexdigest()!=pins[path]: raise RuntimeError('source hash mismatch: modified physics would invalidate the oracle')
    return b.decode()

scalar='chem/module_chem_plumerise_scalar.F' if a.arm=='wrf' else 'gsl-ccpp-physics/physics/smoke_dust/module_smoke_plumerise.F90'
driver='chem/module_plumerise1.F' if a.arm=='wrf' else 'gsl-ccpp-physics/physics/smoke_dust/module_plumerise.F90'
src=read(scalar); lines=src.splitlines(); result=[]; in_make=False
module='module_chem_plumerise_scalar' if a.arm=='wrf' else 'module_smoke_plumerise'
for line in lines:
    low=line.strip().lower()
    if low.startswith('module '+module):
        result.extend([line.replace(module,module+'_observed'),'use plume_trace'])
        if a.experiment: result.append('use plume_poison, only: poison_state')
        continue
    if low.startswith('end module'):
        result.append('end module '+module+'_observed'); continue
    if low.startswith('subroutine makeplume'): in_make=True
    # Declaration-safe initialization, immediately before the first executable line.
    if in_make and ('tstpf = 2.0' in low): result.append('trace_counter=0')
    result.append(line)
    if a.experiment and low=='j=1':
        result.insert(len(result)-1,'call poison_state' if a.arm=='wrf' else 'call poison_state(coms)')
    if in_make and ('time = time+dt' in low or 'coms%time = coms%time+coms%dt' in low): result.append('trace_counter=trace_counter+1')
    if 'call get_fire_properties(imm,iveg_ag,' in low: result.append('trace_group=iveg_ag')
    if 'call makeplume' in low and not low.startswith('!'):
        result.extend(['trace_tops(imm,trace_group)=real(ztopmax(imm),4)','trace_steps(imm,trace_group)=trace_counter'])
    if 'call set_flam_vert' in low and not low.startswith('!'):
        result.extend(['trace_tops(:,trace_group)=real(ztopmax,4)','trace_k1(trace_group)=k1','trace_k2(trace_group)=k2'])
    if low.startswith('end subroutine makeplume'): in_make=False

observed=build/(module+'_observed.F90'); observed.write_text('\n'.join(result)+'\n')
flags=['gfortran','-c','-O0','-cpp','-DRWORDSIZE=4','-ffree-form','-ffree-line-length-none','-fallow-argument-mismatch','-I',str(build)]
def compile(path):
    obj=build/(path.stem+'.o'); subprocess.run([*flags,'-o',str(obj),str(path)],cwd=build,check=True)
    undef=subprocess.check_output(['nm','-u',str(obj)],text=True)
    if '_ZGV' in undef: raise RuntimeError('observer object imports vector libm: scalar oracle identity would be lost')
    return obj
trace=compile(h/'plume_trace.F90')
poison=compile(h/('plume_poison_'+a.arm+'.F90')) if a.experiment else None
scalar_obj=compile(observed)
original_driver=read(driver)
if a.arm=='wrf':
    body=original_driver[original_driver.lower().index('subroutine plumerise_driver'):]
    body=body[:body.lower().index('end subroutine plumerise_driver')+len('end subroutine plumerise_driver')]
    dsrc='module plume_column_driver_observed\n'+(h/'plume_column_prelude.inc').read_text()+'\ncontains\n'+body+'\nend module\n'
    driver_module='plume_column_driver'
else: dsrc=original_driver; driver_module='module_plumerise'
dsrc=dsrc.replace('USE module_chem_plumerise_scalar','USE module_chem_plumerise_scalar_observed') if a.arm=='wrf' else dsrc.replace('USE module_smoke_plumerise','USE module_smoke_plumerise_observed')
if a.arm=='gsl':
    dsrc=dsrc.replace('module module_plumerise','module module_plumerise_observed')
driver_file=build/(driver_module+'_observed.F90'); driver_file.write_text(dsrc); driver_obj=compile(driver_file)
name='run_plumerise_wrfchem' if a.arm=='wrf' else 'gsl_run_plumerise_frp'
command=[*flags,'-DPLUME_TRACE','-o',str(build/(name+'_trace.o')),str(h/(name+'.F90'))]
subprocess.run(command,cwd=build,check=True)
objects=['stub_wrf','module_model_constants','module_zero_plumegen_coms','module_chem_plumerise_scalar','plume_column_driver','oracle_io'] if a.arm=='wrf' else ['machine','rrfs_smoke_config','module_zero_plumegen_coms','module_smoke_plumerise','module_plumerise','oracle_io']
exe=build/(name+'_trace')
subprocess.run(['gfortran','-o',str(exe),*[str(build/(o+'.o')) for o in objects],str(trace),*([str(poison)] if poison else []),str(scalar_obj),str(driver_obj),str(build/(name+'_trace.o'))],check=True)
out=build/'fixtures_observed'/('plumerise_wrfchem' if a.arm=='wrf' else 'plumerise_frp_gsl')
if not a.experiment: subprocess.run([str(exe),str(out)],cwd=build,check=True)
(build/'observer-sha256sums.txt').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in [observed,driver_file,h/'plume_trace.F90',h/(name+'.F90')]))
