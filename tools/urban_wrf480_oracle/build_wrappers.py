"""Compile the pristine WRF 4.8 Noah wrapper and exact release providers."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,urllib.request

PIN='06d4240ae989cc3e50af412bb472df3d9048783c'
NOAHMP_PIN='5da0b241e48ecfd9a2a1bd667ed554765856d589'
NOAHMP_FILES=['utility/Machine.F90','drivers/wrf/NoahmpIOVarType.F90',
              'drivers/wrf/NoahmpUrbanDriverMainMod.F90']
FILES=['share/module_model_constants.F','frame/module_wrf_error.F','phys/module_sf_urban.F',
       'phys/module_bep_bem_helper.F','phys/module_sf_bep.F','phys/module_sf_bem.F',
       'phys/module_sf_bep_bem.F','phys/module_sf_noahlsm.F','phys/module_sf_noahlsm_glacial_only.F',
       'phys/module_ra_gfdleta.F','phys/module_sf_noahdrv.F']


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=True)
    root=Path(__file__).resolve().parents[2];flags=['-c','-O0','-g','-cpp','-Dwrfmodel','-DEM_CORE=1','-DNMM_CORE=0','-DRWORDSIZE=4','-DIWORDSIZE=4','-DDWORDSIZE=8','-DLWORDSIZE=4','-ffree-form','-ffree-line-length-none','-fallow-argument-mismatch']
    sources=[];commands=[]
    for rel in FILES:
        raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/wrf-model/WRF/{PIN}/{rel}').read()
        file=out/Path(rel).name;file.write_bytes(raw)
        sources.append({'path':rel,'sha256':hashlib.sha256(raw).hexdigest()})
    text=(out/'module_ra_gfdleta.F').read_text();found=re.search(r'^ *SUBROUTINE CAL_MON_DAY\b.*?^ *END SUBROUTINE CAL_MON_DAY',text,re.M|re.S)
    if found is None:raise ValueError('calendar helper anchor moved')
    (out/'calendar.F90').write_text('module module_ra_gfdleta\ncontains\n'+found[0]+'\nend module module_ra_gfdleta\n')
    files=[root/'tools/urban_wrf471_oracle/stub_wrf.F90',*[out/Path(r).name for r in FILES[:-2]],out/'calendar.F90',out/'module_sf_noahdrv.F']
    for i,file in enumerate(files):
        command=['gfortran',*flags,str(file)];commands.append(command)
        q=subprocess.run(command,cwd=out,text=True,capture_output=True)
        (out/f'compile-{i:02d}.log').write_text(q.stdout+q.stderr)
        if q.returncode:
            receipt={'source_commit':PIN,'sources':sources,'commands':commands,'exit':q.returncode,'failed_source':str(file)}
            (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));print((q.stdout+q.stderr)[:5000]);raise SystemExit(q.returncode)
    receipt={'source_commit':PIN,'sources':sources,'commands':commands,'exit':0}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('PRISTINE_NOAH_WRAPPER_COMPILE_PASS')
    mp_sources=[];mp_commands=[]
    for rel in NOAHMP_FILES:
        raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/NCAR/noahmp/{NOAHMP_PIN}/{rel}').read()
        file=out/Path(rel).name;file.write_bytes(raw)
        mp_sources.append({'path':rel,'sha256':hashlib.sha256(raw).hexdigest()})
        command=['gfortran',*flags,str(file)];mp_commands.append(command)
        q=subprocess.run(command,cwd=out,text=True,capture_output=True)
        (out/(file.stem+'-compile.log')).write_text(q.stdout+q.stderr)
        if q.returncode:
            raise RuntimeError(f'Pinned Noah-MP wrapper compilation failed: {file}: {q.stderr}')
    (out/'noahmp-receipt.json').write_text(json.dumps({
        'source_commit':NOAHMP_PIN,'sources':mp_sources,'commands':mp_commands,
        'module_dependencies':'receipt.json','exit':0},indent=2))
    print('PINNED_NOAHMP_WRAPPER_COMPILE_PASS')


if __name__=='__main__':main()
