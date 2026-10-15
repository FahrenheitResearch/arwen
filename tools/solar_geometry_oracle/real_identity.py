"""Real CLI: old base, explicit legacy, and bare corrected default."""
from pathlib import Path
import hashlib
import json
import os
import subprocess as sp
import sys

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT.parent


def main():
    from tools.solar_geometry_oracle.hash_history import hashes
    work = LANE/'real-run'
    attempt = 1
    while work.exists():
        attempt += 1
        work = LANE/f'real-run-{attempt:02d}'
    work.mkdir()
    cache = Path('/work/sprint/cache/combo-rec-full')
    source_inputs = cache/'inputs/iowa-convective/A071'
    source_nml = cache/'strict/iowa-convective/A071/namelist.input'
    original_nml = source_nml.read_text()
    versions = [('base',LANE/'baseline-src',None),
                ('legacy',ROOT,'wrf_461'),('default',ROOT,None)]
    reports = {}
    for name,tree,generation in versions:
        inp = work/(name+'-inputs')
        inp.mkdir(exist_ok=True)
        for file in ('wrfinput_d01','wrfbdy_d01'):
            path = inp/file
            if not path.exists():
                path.symlink_to(source_inputs/file)
        text = original_nml
        if generation:
            from gpuwm.physics_source_defaults import with_physics_selector_comment
            text = with_physics_selector_comment(text,{'solar_generation':generation})
        (inp/'namelist.input').write_text(text)
        cmd = [sys.executable,'-m','gpuwm.cli','run','--wrfinput',str(inp),
               '--rrtmg-variant','rrtmg_legacy','--run-seconds','30',
               '--outdir',str(work/name),'--no-supervise']
        env = dict(os.environ, PYTHONPATH=str(tree)+':'+str(tree/'tests'))
        with (work/(name+'.log')).open('wb') as log:
            result = sp.run(cmd,cwd=tree,env=env,stdout=log,stderr=sp.STDOUT)
        reports[name] = {'argv':cmd,'source_tree':str(tree),'exit':result.returncode,
                         'strict':os.environ.get('GPUWM_WRF_EXACT','0'),
                         'generation':generation or ('wrf_461' if name=='base' else 'wrf_480')}
        if result.returncode == 0:
            reports[name]['fields'] = hashes(work/name)
            reports[name]['commit'] = sp.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()
            reports[name]['inputs_sha256'] = {file:hashlib.sha256((source_inputs/file).read_bytes()).hexdigest()
                                               for file in ('wrfinput_d01','wrfbdy_d01')}
            if name != 'base':
                from gpuwm.experiment import load_experiment
                from gpuwm.io.restart import configuration_echo, _configuration_fingerprint
                exp = load_experiment(work/name/'input/experiment.toml')
                selections = [domain.run.solar_generation for domain in exp.domains]
                assert selections == [reports[name]['generation']]*len(exp.domains), selections
                reports[name]['actual_selected_generations'] = selections
                reports[name]['restart_configuration'] = configuration_echo(exp.domains[0].run)
                reports[name]['restart_configuration_sha256'] = _configuration_fingerprint(exp.domains[0].run)
                registry = json.loads((ROOT/'gpuwm/physics_registry_v2.json').read_text())
                reports[name]['registry_row'] = registry['parameters']['solar_generation']
        (work/'real-run.json').write_text(json.dumps(reports,indent=2)+'\n')
        deleted = []
        for path in (work/name).rglob('wrfout*'):
            if path.is_file() and path.suffix not in ('.json','.log'):
                deleted.append({'path':str(path),'bytes':path.stat().st_size})
                path.unlink()
        (work/(name+'-deleted.json')).write_text(json.dumps(deleted,indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{name} real CLI failed: {work/(name+".log")}')
    assert reports['base']['fields'] == reports['legacy']['fields'], 'legacy full-field replay changed'
    comparisons = sum(len(frame) for frame in reports['base']['fields'].values())
    print('legacy full-field comparisons',comparisons,'all equal; corrected default CLI exit 0')

if __name__ == '__main__':
    main()
