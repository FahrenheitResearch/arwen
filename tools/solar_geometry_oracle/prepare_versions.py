"""Fetch release sources and build separately pinned solar referees on a box."""
from pathlib import Path
import configparser
import hashlib
import json
import subprocess as sp
import sys

PINS = {'v4.6.1':'d66e442fccc04111067e29274c9f9eaccc3cef28',
        'v4.7.0':'eeacb821626feb34c62c3c1e169594d886d13adb',
        'v4.7.1':'f52c197ed39d12e087d02c50f412d90d418f6186',
        'v4.8.0':'06d4240ae989cc3e50af412bb472df3d9048783c'}

def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for tag, pin in PINS.items():
        repo = root / tag
        if not repo.exists():
            sp.run(['git','clone','--depth=1','--filter=blob:none','--branch',tag,
                    '--no-checkout','https://github.com/wrf-model/WRF',str(repo)],check=True)
        rev = sp.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        assert rev == pin, (tag, rev)
        paths = ['phys/module_radiation_driver.F','share/module_model_constants.F',
                 'Registry/Registry.EM_COMMON','arch/Externals.cfg','.gitmodules']
        for path in paths:
            result = sp.run(['git','-C',str(repo),'show',f'HEAD:{path}'],capture_output=True)
            if result.returncode == 0:
                target = repo / path
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(result.stdout)
        receipts[tag] = {'commit':rev,'gitlinks':sp.check_output(
            ['git','-C',str(repo),'ls-tree','-r','HEAD'],text=True).splitlines()}
        receipts[tag]['gitlinks'] = [line for line in receipts[tag]['gitlinks'] if line.startswith('160000')]
        cfg = configparser.ConfigParser()
        cfg.read(repo/'.gitmodules')
        links = {line.split('\t')[1]:line.split()[2] for line in receipts[tag]['gitlinks']}
        pins = []
        for section in cfg.sections():
            path = cfg[section]['path']
            if path.startswith('phys/') and path in links:
                pins.append((path,cfg[section]['url'],links[path]))
        if tag in ('v4.7.0','v4.7.1'):
            external = configparser.ConfigParser()
            external.read(repo/'arch/Externals.cfg')
            row = external['MMM-physics']
            assert row['tag'] == '20240626-MPASv8.2'
            pins.append(('phys/physics_mmm',row['repo_url'],row['tag']))
        receipts[tag]['submodules'] = []
        for path,url,pin in pins:
            sub = root/'submodules'/tag/path
            if not sub.exists():
                sub.mkdir(parents=True)
                sp.run(['git','init',str(sub)],check=True)
                sp.run(['git','-C',str(sub),'remote','add','origin',url],check=True)
                sp.run(['git','-C',str(sub),'fetch','--depth=1','--filter=blob:none','origin',pin],check=True)
            rev = sp.check_output(['git','-C',str(sub),'rev-parse','FETCH_HEAD'],text=True).strip()
            tree = sp.check_output(['git','-C',str(sub),'ls-tree','-r','--name-only','FETCH_HEAD'],text=True).splitlines()
            assert tree, (tag,path,'empty submodule')
            if len(pin) == 40:
                assert rev == pin, (path,rev,pin)
            receipts[tag]['submodules'].append({'path':path,'url':url,'requested':pin,'commit':rev,'tree_paths':len(tree),'sample':tree[:10]})
        receipts[tag]['driver_sha256'] = hashlib.sha256((repo/'phys/module_radiation_driver.F').read_bytes()).hexdigest()
        if tag in ('v4.6.1','v4.8.0'):
            sp.run([sys.executable,str(Path(__file__).with_name('build.py')),
                    str(repo),str(root / ('oracle-'+tag))],check=True)
    (root/'sources.json').write_text(json.dumps(receipts,indent=2)+'\n')

if __name__ == '__main__':
    main()
