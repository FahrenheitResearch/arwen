import json,sys,hashlib
from pathlib import Path
import netCDF4,h5py,numpy as np
# QA only: independent readers compare artifacts, never production data processing.
# Usage: python compare_history.py REFERENCE_NC CANDIDATE_NC REPORT_JSON
if len(sys.argv) != 4:
    raise SystemExit("usage: compare_history.py REFERENCE_NC CANDIDATE_NC REPORT_JSON")
root=Path(sys.argv[3]).parent
paths=[Path(sys.argv[1]),Path(sys.argv[2])]
report={'readers':{'netCDF4':netCDF4.__version__,'netcdf_c':netCDF4.getlibversion(),'h5py':h5py.__version__},'paths':list(map(str,paths))}
with netCDF4.Dataset(paths[0]) as a,netCDF4.Dataset(paths[1]) as b:
    assert list(a.dimensions)==list(b.dimensions)
    assert [(k,len(v),v.isunlimited()) for k,v in a.dimensions.items()]==[(k,len(v),v.isunlimited()) for k,v in b.dimensions.items()]
    assert a.ncattrs()==b.ncattrs(),(a.ncattrs(),b.ncattrs())
    assert list(a.variables)==list(b.variables)
    for k in a.ncattrs():
        assert np.asarray(a.getncattr(k)).dtype==np.asarray(b.getncattr(k)).dtype,k
        assert np.array_equal(a.getncattr(k),b.getncattr(k)),k
    report['variables']={}
    for k,x in a.variables.items():
        y=b[k]
        assert x.dimensions==y.dimensions and x.dtype==y.dtype and x.shape==y.shape,k
        assert x.ncattrs()==y.ncattrs(),(k,x.ncattrs(),y.ncattrs())
        for attr in x.ncattrs():
            assert np.asarray(x.getncattr(attr)).dtype==np.asarray(y.getncattr(attr)).dtype,(k,attr)
            assert np.array_equal(x.getncattr(attr),y.getncattr(attr)),(k,attr)
        x.set_auto_maskandscale(False);y.set_auto_maskandscale(False)
        if x.dtype==str:
            assert list(x[:])==list(y[:]),k
        else:
            assert x[:].tobytes()==y[:].tobytes(),k
        fill=[str(z.get_fill_value()) if z.dtype!=str else None for z in (x,y)]
        assert fill[0]==fill[1],(k,fill)
        report['variables'][k]={'dimensions':x.dimensions,'dtype':str(x.dtype),'shape':x.shape,'fill_values':fill,'identical':True}
with h5py.File(paths[0]) as a,h5py.File(paths[1]) as b:
    for k in a.attrs:
        if k=='_NCProperties':continue
        assert a.attrs[k].dtype==b.attrs[k].dtype,(k,a.attrs[k],b.attrs[k])
        assert a.attrs[k].tobytes()==b.attrs[k].tobytes(),k
    for k in a:
        x,y=a[k],b[k]
        pa,pb=x.id.get_create_plist(),y.id.get_create_plist()
        assert pa.get_fill_time()==pb.get_fill_time(),k
        assert pa.get_alloc_time()==pb.get_alloc_time(),k
        assert pa.fill_value_defined()==pb.fill_value_defined(),k
        assert np.array_equal(x.fillvalue,y.fillvalue),k
    report['code_version_raw_hex']=b.attrs['code_version'].tobytes().hex()
report['status']='identical fields, dimensions, ordered typed attributes and NC_STRING'
report['sha256']={str(p.name if p.parent.name!='reference' else 'reference/'+p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
Path(sys.argv[3]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
