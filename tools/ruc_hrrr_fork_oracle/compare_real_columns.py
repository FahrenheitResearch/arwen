"""Compare compiled fork, host port, and captured production CUDA columns."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np


def load_csv(path, ncol, nsl, profiles):
    rows = list(csv.DictReader(Path(path).open()))
    if len(rows) != ncol*nsl:
        raise ValueError(f'{path}: expected {ncol*nsl} rows, got {len(rows)}')
    output = {}
    for key in rows[0]:
        if key in ('column','k','ivgtyp','isltyp'):
            continue
        words = np.array([float(row[key]) for row in rows],dtype=np.float32).reshape(ncol,nsl)
        output[key] = words.T if key in profiles else words[:,0]
    return output


def ulps(a,b):
    def ordered(value):
        word=np.asarray(value,dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(word<0,np.int64(np.iinfo(np.int32).min)-word,word)
    return np.abs(ordered(a)-ordered(b))


def grade(left,right):
    metrics={}
    for name in left:
        if name not in right:
            continue
        a,b=left[name],right[name]
        delta=(b.astype(np.float64)-a.astype(np.float64))
        metrics[name]={'max_abs':float(np.max(np.abs(delta))),
                       'mean':float(np.mean(delta)),
                       'max_ulp':int(np.max(ulps(a,b))),
                       'changed':int(np.count_nonzero(ulps(a,b)))}
    return metrics


def compare(snapshot,directory):
    from gpuwm.core.ruc import (ruc_land_surface_step,RUC_DRIVER_COLUMN_STATE,
                                RUC_DRIVER_PROFILE_STATE)
    snapshot, directory=Path(snapshot),Path(directory)
    capture=np.load(snapshot,allow_pickle=False)
    manifest=json.loads(snapshot.with_suffix('.json').read_text())
    kwargs=dict(manifest['driver_keywords'])
    kwargs.update({name:np.asarray(capture['keyword__'+name])
                   for name in ('ivgtyp','isltyp','landusef','soilctop')
                   if 'keyword__'+name in capture})
    kwargs['zs']=np.asarray(capture['zs'])
    values={name[7:]:np.asarray(capture[name]) for name in capture.files
            if name.startswith('input__')}
    result=ruc_land_surface_step(values,**kwargs)
    host={name:np.asarray(getattr(result,name)) for name in
          RUC_DRIVER_COLUMN_STATE+RUC_DRIVER_PROFILE_STATE}
    fused={name[8:]:np.asarray(capture[name]) for name in capture.files
           if name.startswith('output__')}
    ncol=int(manifest['columns']);nsl=len(capture['zs'])
    intel=load_csv(directory/'oracle-real-intel.csv',ncol,nsl,RUC_DRIVER_PROFILE_STATE)
    gnu=load_csv(directory/'oracle-real-gnu.csv',ncol,nsl,RUC_DRIVER_PROFILE_STATE)
    groups={}
    classes=np.asarray(capture['keyword__ivgtyp']).reshape(-1)
    for code in np.unique(classes):
        mask=classes==code
        entry={'columns':int(mask.sum()),'grid_population':manifest.get('land_class_census',{}).get(str(int(code)))}
        for arm,array in [('fork_intel',intel['lh']),('fork_gnu',gnu['lh']),
                          ('host',host['lh']),('fused',fused.get('lh'))]:
            if array is not None:
                entry[arm+'_sample_mean_lh_W_m2']=float(np.mean(array[mask],dtype=np.float64))
        if 'lh' in fused:
            delta=fused['lh'][mask].astype(np.float64)-intel['lh'][mask].astype(np.float64)
            entry['fused_minus_fork_mean_lh_W_m2']=float(np.mean(delta))
            entry['fused_minus_fork_max_abs_lh_W_m2']=float(np.max(np.abs(delta)))
        groups[str(int(code))]=entry
    receipt={'schema':1,'snapshot':snapshot.name,'step':manifest.get('step'),
             'columns':ncol,'nsl':nsl,'driver_keywords':manifest['driver_keywords'],
             'sample_note':'Wetness quantiles per land class, not population-weighted means.',
             'by_land_class':groups,
             'host_minus_fork':grade(intel,host),
             'fused_minus_fork':grade(intel,fused),
             'fused_minus_host':grade(host,fused),
             'gnu_minus_intel':grade(intel,gnu),
             'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (snapshot,snapshot.with_suffix('.json'),directory/'oracle-real-intel.csv',
                                 directory/'oracle-real-gnu.csv')}}
    target=directory/'comparison.json'
    target.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(directory/'host-output.npz',**host)
    print(json.dumps({name:receipt[name].get('lh') for name in
                     ('host_minus_fork','fused_minus_fork','fused_minus_host','gnu_minus_intel')},indent=2))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('snapshot');parser.add_argument('directory')
    args=parser.parse_args();compare(args.snapshot,args.directory)
