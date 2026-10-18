"""Regenerate strict GF controls using the official scalar WRF build flags."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess
import numpy as np
import gf_regime_columns as gr

PINS = {
 'module_cu_gf_wrfdrv':'0cead99c913b6a74f3a492aea1e1c7ef6d34f95402574336022d1a6c716a4382',
 'module_cu_gf_deep':'798ad7271963757b88be36ce01e24222f7af045dc0692908d6e7c270d6b4ea11',
 'module_cu_gf_sh':'0dd1ed7c620b39e13f7ef77dc4ff1a269fa1f8d746492d360e203df0215b529a',
 'module_gfs_physcons':'2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52',
 'module_gfs_machine':'300ee4f75663e89d34c8a9b8733cc07a5dd48c3d973da019d31cb76524babe25',
}
FLAGS = ('-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize')
HERE=Path(__file__).resolve().parent

def record(source, configure, fixture, output, receipts):
 cfg=configure.read_bytes()
 match=re.search(r'^FCOPTIM\s*=\s*(.*)$',cfg.decode(),re.M)
 if not match or tuple(match.group(1).split())!=FLAGS:
  raise ValueError('official scalar WRF FCOPTIM differs from the qualified control flags')
 output.mkdir(parents=True,exist_ok=True);receipts.mkdir(parents=True,exist_ok=True)
 commands=[]
 def run(args):
  args=list(map(str,args));commands.append(args);subprocess.run(args,cwd=output,check=True)
 flags=[*FLAGS,'-cpp','-ffree-form','-ffree-line-length-none','-I',output,'-J',output]
 for name,digest in PINS.items():
  if hashlib.sha256((source/'phys'/f'{name}.F').read_bytes()).hexdigest()!=digest:
   raise ValueError('WRF source pin differs: '+name)
 for name in ('module_gfs_machine','module_gfs_physcons','module_cu_gf_deep','module_cu_gf_sh','module_cu_gf_wrfdrv'):
  run(['gfortran','-c',*flags,'-o',output/f'{name}.o',source/'phys'/f'{name}.F'])
 run(['gfortran','-c',*flags,'-o',output/'run_gf_columns.o',HERE/'run_gf_columns.F90'])
 original=(source/'phys/module_cu_gf_sh.F').read_text()
 safe,count=re.subn(r'(?im)^(\s*do\s+k\s*=\s*1\s*,\s*)kpbl\(i\)(\s*)$',r'\1min(kpbl(i),ktf-1)\2',original)
 if count!=1:raise ValueError('pressure-loop repair no longer has one source anchor')
 needle='k22(i)=maxloc(HEO_CUP(i,2:kbmax(i)),1)'
 if safe.count(needle)!=1:raise ValueError('MAXLOC repair no longer has one source anchor')
 base=fixture/'gf-regime-wrf-crgamma-k0.bin';old=gr.read_outputs(base)
 for arm,text in (('safe',safe),('corrected',safe.replace(needle,needle+'+1'))):
  code=output/f'{arm}.F';code.write_text(text)
  run(['gfortran','-c',*flags,'-fcheck=bounds','-o',output/f'{arm}.o',code])
  exe=output/f'run_{arm}'
  run(['gfortran','-o',exe,*[output/f'{n}.o' for n in ('module_gfs_machine','module_gfs_physcons','module_cu_gf_deep')],output/f'{arm}.o',output/'module_cu_gf_wrfdrv.o',output/'run_gf_columns.o'])
  binding=subprocess.check_output(['nm',str(exe)],text=True)
  if not re.search(r'\bU tgammaf(?:@|$)',binding,re.M):raise ValueError('scalar control no longer uses the stock dynamic binding')
  path=output/f'gf-regime-wrf-scalar-{arm}-k0.bin'
  run([exe,fixture/'gf-regime-inputs.bin',path,'0']);new=gr.read_outputs(path)
  words={}
  for key in ('lev','sca','isc'):
   a,b=old[key].view(np.uint32).ravel(),new[key].view(np.uint32).ravel()
   words[key]=[[int(i),int(b[i])] for i in np.flatnonzero(a!=b)]
  receipt=dict(base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ncol=new['ncol'],nz=new['nz'],words=words,
   provenance=dict(wrf_source_sha256=PINS,fortran_flags=list(FLAGS),official_config_sha256=hashlib.sha256(cfg).hexdigest(),modified_shallow_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),driver_sha256=hashlib.sha256((HERE/'run_gf_columns.F90').read_bytes()).hexdigest(),stock_dynamic_binding=True,repairs=['pressure loop capped at ktf-1']+(['MAXLOC section offset +1'] if arm=='corrected' else [])))
  (receipts/f'gf-regime-wrf-scalar-{arm}-delta.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('source','configure','fixture','output','receipts'):p.add_argument(name,type=Path)
 a=p.parse_args();record(a.source.resolve(),a.configure.resolve(),a.fixture.resolve(),a.output.resolve(),a.receipts.resolve())