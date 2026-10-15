"""Small receipts from real CLI checkpoints and Rust-decoded history fields."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np

def checkpoint(path, *, diagnostics=False):
    with zipfile.ZipFile(path) as archive:
        return {name:hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()
                if name.startswith(("state/","acoustic/","scratch/","driver/","fields/","cumulus/","held/","pbl/"))
                and (diagnostics or not name.startswith("scratch/ltng_"))}

def capture(root,out):
    from gpuwm import netcdf_bridge
    result={"root":str(root),"checkpoints":{},"all_checkpoint_fields":{},"history":[]}
    for path in sorted(root.rglob("*.npz")):
        hashes=checkpoint(path)
        if hashes:
            result["checkpoints"][str(path.relative_to(root))]=hashes
            result["all_checkpoint_fields"][str(path.relative_to(root))]=checkpoint(path, diagnostics=True)
    for path in sorted(root.rglob("wrfout_d01*")):
        if not path.is_file() or path.suffix in (".json",".log"):
            continue
        with netcdf_bridge.open_dataset(path) as ds:
            item={"path":str(path.relative_to(root)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"fields":{}}
            for name in ("IC_FLASHRATE","CG_FLASHRATE","IC_FLASHCOUNT","CG_FLASHCOUNT","LPI","REFL_10CM"):
                if name in ds.variables:
                    values=np.asarray(ds.variables[name][:],dtype=np.float32)
                    item["fields"][name]={"shape":list(values.shape),"min":float(np.min(values)),
                                         "max":float(np.max(values)),"nonzero":int(np.count_nonzero(values)),
                                         "sha256":hashlib.sha256(values.tobytes()).hexdigest()}
        result["history"].append(item)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"history_files":len(result["history"]),"checkpoints":len(result["checkpoints"])}))

def compare(left,right,out,*,diagnostics=False):
    a=json.loads(left.read_text());b=json.loads(right.read_text())
    assert a["checkpoints"] and b["checkpoints"],"real CLI checkpoint hashes are required"
    key="all_checkpoint_fields" if diagnostics else "checkpoints"
    ha=list(a[key].values())[-1];hb=list(b[key].values())[-1]
    missing=sorted(set(ha)^set(hb))
    changed=sorted(k for k in ha.keys()&hb.keys() if ha[k]!=hb[k])
    result={"left":str(left),"right":str(right),"full_fields":len(ha),"missing":missing,"changed":changed,"pass":not missing and not changed}
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
    if not result["pass"]:raise SystemExit(1)

def ice_inventory_variant(source,destination,receipt):
    """Declared MP6 initial inventory with zero ice species.

    Retains all public atmospheric, cloud, rain and forcing fields.
    This is a new physics arm with explicit zero ice initial conditions.
    No stock-WRF MP6 initialization parity is claimed.
    """
    from gpuwm import netcdf_bridge
    from gpuwm.io.classic_tape import ClassicTape
    with netcdf_bridge.open_dataset(source) as ds:
        before={name:np.asarray(ds.variables[name][:],
                 dtype="S1" if ds.variables[name].dtype==np.dtype("O") else ds.variables[name].dtype)
                for name in ds.variables}
        attrs={name:ds.getncattr(name) for name in ds.ncattrs()}
        attrs["MP_PHYSICS"]=np.float32(6.)
        with ClassicTape(destination,container="cdf2") as tape:
            for name,dim in ds.dimensions.items():
                tape.createDimension(name,None if dim.isunlimited() else len(dim))
            tape.setncatts(attrs)
            variables={}
            for name,var in ds.variables.items():
                variables[name]=tape.createVariable(name,"S1" if var.dtype==np.dtype("O") else var.dtype,var.dimensions)
                variables[name].setncatts({n:var.getncattr(n) for n in var.ncattrs()})
            for name,description in (("QICE","Ice mixing ratio"),("QSNOW","Snow mixing ratio"),("QGRAUP","Graupel mixing ratio")):
                variables[name]=tape.createVariable(name,"f4",ds.variables["QCLOUD"].dimensions)
                variables[name].setncatts({"units":"kg kg-1","description":description,"stagger":"","FieldType":np.int32(104),"MemoryOrder":"XYZ"})
            for name,values in before.items():
                if ds.variables[name].dimensions and ds.dimensions[ds.variables[name].dimensions[0]].isunlimited():
                    for record in range(values.shape[0]):
                        variables[name][record] = values[record]
                else:
                    variables[name][0] = values
            for name in ("QICE","QSNOW","QGRAUP"):
                variables[name][0]=np.zeros_like(before["QCLOUD"][0],dtype=np.float32)
    with netcdf_bridge.open_dataset(destination) as ds:
        unchanged={name:hashlib.sha256(before[name].tobytes()).hexdigest()
                   for name in before if np.asarray(ds.variables[name][:],dtype=before[name].dtype).tobytes()==before[name].tobytes()}
        if len(unchanged)!=len(before):raise AssertionError("the cold ice arm changed an existing input field")
    receipt.write_text(json.dumps({"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
         "output_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),"existing_fields_unchanged":len(unchanged),
         "existing_field_hashes":unchanged,"new_zero_fields":["QICE","QSNOW","QGRAUP"],"changed_attribute":{"MP_PHYSICS":[1,6]}},indent=2)+"\n")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--compare",nargs=2,type=Path)
    p.add_argument("--include-diagnostics",action="store_true")
    p.add_argument("--ice-inventory",nargs=2,type=Path)
    a=p.parse_args()
    if a.compare:compare(*a.compare,a.out,diagnostics=a.include_diagnostics)
    elif a.ice_inventory:ice_inventory_variant(*a.ice_inventory,a.out)
    else:capture(a.root,a.out)
if __name__=="__main__":main()
