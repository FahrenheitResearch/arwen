"""Small real WRF state crops and deterministic edge transformations.

NetCDF decoding is done by the engine's Rust reader. NumPy only packs the
decoded fixture arrays and modifies named edge scenarios.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

VARIABLES = "U V W T P PB PH PHB QVAPOR QCLOUD QICE AL ALB MU MUB MAPFAC_M MAPFAC_U MAPFAC_V FNM FNP ZNW P_TOP DN DNW C1H C2H C1F C2F CF1 CF2 CF3 XLAT".split()

def raw_state(source, reader, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(reader), "dump", "--raw", str(source), str(out), *VARIABLES], check=True)
    metadata = json.loads((out/"metadata.json").read_text())
    data = {v["name"]: np.fromfile(out/v["filename"], dtype="<f8").reshape(v["shape"])[0].astype(np.float32)
            for v in metadata["variables"]}
    inventory = json.loads(subprocess.check_output([str(reader), "inventory", str(source)]))
    attrs = inventory["global_attributes"]
    if isinstance(attrs, list):
        attrs = {a["name"]: a["value"] for a in attrs}
    return data, {"source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  "reader_sha256": hashlib.sha256(Path(reader).read_bytes()).hexdigest(),
                  "dx": float(attrs["DX"]), "dy": float(attrs["DY"])}

def crop_state(raw, *, nx=16, ny=12, x=70, y=65):
    """Keep all vertical levels and actual C-grid staggering."""
    pairs = {"u":"U", "v":"V", "w":"W", "php":"PH", "phb":"PHB",
             "qv":"QVAPOR", "qc":"QCLOUD", "qi":"QICE", "msft":"MAPFAC_M",
             "msfu":"MAPFAC_U", "msfv":"MAPFAC_V", "lat":"XLAT"}
    a = {}
    for name, var in pairs.items():
        sy, sx = ny + (name == "v" or name == "msfv"), nx + (name == "u" or name == "msfu")
        a[name] = raw[var][...,y:y+sy,x:x+sx].copy()
    a["alt"] = np.add(raw["AL"],raw["ALB"],dtype=np.float32)[:,y:y+ny,x:x+nx].copy()
    a["p"] = np.add(raw["P"],raw["PB"],dtype=np.float32)[:,y:y+ny,x:x+nx].copy()
    a["thp"] = raw["T"][:,y:y+ny,x:x+nx].copy()
    a["thb"] = np.full_like(a["thp"],300.)
    a["mut"] = np.add(raw["MU"],raw["MUB"],dtype=np.float32)[y:y+ny,x:x+nx].copy()
    a["mub2d"] = raw["MUB"][y:y+ny,x:x+nx].copy()
    for name in ("fnm","fnp","znw","dn","dnw","c1h","c2h","c1f","c2f"):
        a[name] = raw[name.upper()].copy()
    return a

def state_cases(raw, metadata, *, nx=16, ny=12):
    base = crop_state(raw,nx=nx,ny=ny)
    definitions = [("real_open",1,1),("real_periodic",0,0),("steep_open",1,1),
                   ("map_extremes",0,0),("zero_flow",0,0),("near_zero_flow",0,0),("southern_open",1,1)]
    for name,bx,by in definitions:
        a = {k:v.copy() for k,v in base.items()}
        if name == "steep_open":
            iy,ix = np.meshgrid(np.arange(ny),np.arange(nx),indexing="ij")
            terrain = (1200*np.sin(2*np.pi*ix/nx)*np.cos(2*np.pi*iy/ny)).astype(np.float32)
            taper = np.linspace(1.,0.,a["phb"].shape[0],dtype=np.float32)
            a["phb"] += np.float32(9.81)*taper[:,None,None]*terrain[None]
        if name == "map_extremes":
            for key in ("msft","msfu","msfv"):
                iy,ix = np.meshgrid(np.arange(a[key].shape[0]),np.arange(a[key].shape[1]),indexing="ij")
                a[key] = (.25+3.75*(.5+.5*np.sin(2*np.pi*ix/nx)*np.cos(2*np.pi*iy/ny))).astype(np.float32)
        if name in ("zero_flow","near_zero_flow"):
            scale = np.float32(0. if name=="zero_flow" else 1.e-20)
            for key in ("u","v","w"):
                a[key] *= scale
            a["qv"].fill(0.); a["qc"].fill(0.); a["qi"].fill(0.)
        if name == "southern_open":
            a["lat"] *= -1
            a["v"] *= -1
        if not bx:
            a["u"][:,:,-1] = a["u"][:,:,0]
            a["msfu"][:,-1] = a["msfu"][:,0]
        if not by:
            a["v"][:,-1,:] = a["v"][:,0,:]
            a["msfv"][-1,:] = a["msfv"][0,:]
        yield name,a,{**metadata,"bx":bx,"by":by,"nx":nx,"ny":ny,"nz":a["u"].shape[0],
                      "p_top":float(raw["P_TOP"]),
                      "cf1":float(raw["CF1"]),"cf2":float(raw["CF2"]),"cf3":float(raw["CF3"])}

def pad3(array,nx,ny,nz,bx,by):
    """WRF (i,k,j) storage with two outside halos and redundant faces."""
    f = np.asarray(array,dtype=np.float32)
    if f.ndim==1:
        f = np.broadcast_to(f[:,None,None],(f.size,ny,nx))
    kk = np.minimum(np.arange(nz+1),f.shape[0]-1)
    ji = np.arange(-3,ny+3)
    ii = np.arange(-3,nx+3)
    ji = np.clip(ji,0,f.shape[1]-1) if by else ji%ny
    ii = np.clip(ii,0,f.shape[2]-1) if bx else ii%nx
    return np.asfortranarray(f[np.ix_(kk,ji,ii)].transpose(2,0,1))

def pad2(array,nx,ny,bx,by):
    return np.asfortranarray(pad3(np.asarray(array)[None],nx,ny,0,bx,by)[:,0,:])

def core(array,nx,ny,nz,stagger=""):
    return np.ascontiguousarray(array[3:3+nx+(stagger=="x"),:nz+(stagger=="z"),3:3+ny+(stagger=="y")].transpose(1,2,0))
