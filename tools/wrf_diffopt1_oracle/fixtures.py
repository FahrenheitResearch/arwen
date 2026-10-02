"""Create synthetic inputs and compiled WRF output words for diff_opt=1."""
from pathlib import Path
import argparse
import ctypes as ct
import hashlib
import json
import numpy as np


def ghost(value, nx, ny, nz, bx, by, stag=0):
    """Adapt engine C-grid storage to WRF's three-cell ghost storage."""
    value = np.asarray(value, dtype=np.float32)
    ix = np.arange(-3, nx + 3)
    iy = np.arange(-3, ny + 3)
    ix = np.clip(ix, 0, nx if stag == 1 else nx - 1) if bx else ix % nx
    iy = np.clip(iy, 0, ny if stag == 2 else ny - 1) if by else iy % ny
    if value.ndim == 2:
        return np.asfortranarray(value[iy[:, None], ix[None]].T)
    iz = np.minimum(np.arange(nz + 1), value.shape[0] - 1)
    padded = value[iz[:, None, None], iy[None, :, None], ix[None, None]]
    return np.asfortranarray(padded.transpose(2, 0, 1))


def unghost(value, shape):
    z, y, x = shape
    return np.ascontiguousarray(value[3:3+x, :z, 3:3+y].transpose(1, 2, 0))


def ptr(value):
    return ct.c_void_p(value.ctypes.data)


def build(library, output):
    output.mkdir(parents=True, exist_ok=True)
    native = ct.CDLL(str(library.resolve()))
    nx, ny, nz = 12, 9, 6
    arrays, cases = {}, []
    for case in range(8):
        winds={}
        bx, by = case % 2, (case // 2) % 2
        rng = np.random.default_rng(5100 + case)
        dx, dy = np.float32(800 + 23 * case), np.float32(1400 + 17 * case)
        c1 = np.linspace(1, .4, nz+1, dtype=np.float32)
        c2 = np.linspace(0, 16000, nz+1, dtype=np.float32)
        mu = (50000 + 13000 * rng.random((ny, nx))).astype(np.float32)
        km = (30 + 90 * rng.random((nz, ny, nx))).astype(np.float32)
        mt = (np.ones((ny, nx)) if case < 4 else .7 + .6 * rng.random((ny, nx))).astype(np.float32)
        mfu = np.concatenate([mt, mt[:, :1]], axis=1)
        mfv = np.concatenate([mt, mt[:1]], axis=0)
        for stag in range(4):
            for perturb in ([0, 1] if stag == 0 else [0]):
                shape = (nz + (stag == 3), ny + (stag == 2), nx + (stag == 1))
                field = rng.normal(size=shape).astype(np.float32)
                if stag == 1 and not bx: field[:, :, -1] = field[:, :, 0]
                if stag == 2 and not by: field[:, -1, :] = field[:, 0, :]
                winds[stag]=field
                base = rng.normal(size=(nz, ny, nx)).astype(np.float32)
                seed = (rng.normal(size=shape) * .01).astype(np.float32)
                if stag == 1 and not bx: seed[:, :, -1] = seed[:, :, 0]
                if stag == 2 and not by: seed[:, -1, :] = seed[:, 0, :]
                gf = ghost(field,nx,ny,nz,bx,by,stag)
                gk = ghost(km,nx,ny,nz,bx,by)
                gb = ghost(base,nx,ny,nz,bx,by)
                gm = ghost(mu,nx,ny,nz,bx,by)
                maps = [ghost(v,nx,ny,nz,bx,by,s) for v,s in ((mt,0),(mfu,1),(mfv,2))]
                gt = ghost(seed,nx,ny,nz,bx,by,stag)
                native.oracle_horizontal(*[ct.c_int(v) for v in (nx,ny,nz,bx,by,stag,perturb)],
                    *[ptr(v) for v in (gf,gk,gm,c1,c2,gb,*maps)],
                    ct.c_float(np.float32(1./dx)),ct.c_float(np.float32(1./dy)),ptr(gt))
                name = f"h{case}_{stag}_{perturb}"
                vals = dict(field=field,km=km,mu=mu,c1=c1,c2=c2,base=base,
                            mt=mt,mfu=mfu,mfv=mfv,seed=seed,expected=unghost(gt,shape))
                for key, value in vals.items(): arrays[f"{name}_{key}"] = value
                cases.append(dict(name=name,family="horizontal",bx=bx,by=by,stag=stag,
                                  perturb=perturb,dx=float(dx),dy=float(dy)))
        d11 = (rng.normal(size=km.shape)*.004).astype(np.float32)
        d22 = (rng.normal(size=km.shape)*.004).astype(np.float32)
        d12 = (rng.normal(size=km.shape)*.003).astype(np.float32)
        metrics = [ghost(np.full(km.shape,v,np.float32),nx,ny,nz,bx,by) for v in (.002,.3,-.2)]
        gkm = ghost(np.zeros_like(km),nx,ny,nz,bx,by); gkh=gkm.copy(order="F")
        native_inputs=[*[ghost(d,nx,ny,nz,bx,by) for d in (d11,d22,d12)],
                       ghost(mt,nx,ny,nz,bx,by),*metrics]
        native.oracle_km4(*[ct.c_int(v) for v in (nx,ny,nz,bx,by)],
            *[ptr(v) for v in native_inputs],
            ct.c_float(dx),ct.c_float(dy),ct.c_float(.25),ptr(gkm),ptr(gkh))
        name = f"k4_{case}"
        for key,value in dict(d11=d11,d22=d22,d12=d12,mt=mt,
                              expected_km=unghost(gkm,km.shape),expected_kh=unghost(gkh,km.shape)).items():
            arrays[f"{name}_{key}"]=value
        cases.append(dict(name=name,family="km4",bx=bx,by=by,dx=float(dx),dy=float(dy)))
        phb=(np.arange(nz+1,dtype=np.float32)*np.float32(4905.)).astype(np.float32)
        phb3=np.broadcast_to(phb[:,None,None],(nz+1,ny,nx)).copy()
        gp=ghost(phb3,nx,ny,nz,bx,by)
        grdz=ghost(np.zeros_like(km),nx,ny,nz,bx,by);grdzw=grdz.copy(order="F")
        native.oracle_metrics(*[ct.c_int(v) for v in (nx,ny,nz,bx,by)],ptr(gp),ptr(grdz),ptr(grdzw))
        # Flat-halo extension supplies all ghost metrics read by tke_km.
        rdz=unghost(grdz,km.shape);rdzw=unghost(grdzw,km.shape)
        dn=np.full(nz+1,-1./nz,"f4");dnw=dn.copy()
        weights=np.full(nz+1,.5,"f4")
        native_deform_inputs=[ghost(winds[s],nx,ny,nz,bx,by,s) for s in (1,2,3)]
        native_deform_inputs += [ghost(m,nx,ny,nz,bx,by,s) for m,s in ((mfu,1),(mfv,2),(mt,0))]
        native_deform_inputs += [ghost(m,nx,ny,nz,bx,by) for m in (rdz,rdzw,np.zeros_like(km),np.zeros_like(km))]
        tensors=[ghost(np.zeros_like(km),nx,ny,nz,bx,by) for _ in range(7)]
        native.oracle_deform(*[ct.c_int(v) for v in (nx,ny,nz,bx,by)],*[ptr(v) for v in native_deform_inputs],
            *[ptr(v) for v in (dn,dnw,weights,weights)],ct.c_float(np.float32(1./dx)),ct.c_float(np.float32(1./dy)),
            ct.c_float(1.875),ct.c_float(-1.25),ct.c_float(.375),*[ptr(v) for v in tensors])
        name=f"deform_{case}"
        payload=dict(u=winds[1],v=winds[2],w=winds[3],mt=mt,mfu=mfu,mfv=mfv,phb=phb,dn=dn,dnw=dnw,fnm=weights,fnp=weights)
        payload.update({"expected_"+field:unghost(value,km.shape) for field,value in
                        zip(("div","d11","d22","d33","d12","d13","d23"),tensors)})
        for key,value in payload.items():arrays[f"{name}_{key}"]=value
        cases.append(dict(name=name,family="deform",bx=bx,by=by,dx=float(dx),dy=float(dy)))
        gn2=ghost(np.zeros_like(km),nx,ny,nz,bx,by)
        gn2theta=ghost(np.full(km.shape,300.,"f4"),nx,ny,nz,bx,by)
        native.oracle_n2(*[ct.c_int(v) for v in (nx,ny,nz,bx,by)],ptr(gn2theta),ptr(grdz),ptr(grdzw),ptr(dn),ptr(dnw),ptr(gn2))
        expected_bn2=unghost(gn2,km.shape)
        for isotope in (0,1):
            for isfflx in (0,1):
                tke=(rng.random(km.shape)*.8).astype(np.float32)
                tke[:,:,0:2]=0.
                tke[:,:,2:4]=np.float32(1e-9)
                inputs=[ghost(v,nx,ny,nz,bx,by) for v in
                        (tke,np.full((nz+1,ny,nx),100000.,"f4"),
                         np.full((nz+1,ny,nx),300.,"f4"),np.full(km.shape,300.,"f4"),
                         expected_bn2,rdz,rdzw)]
                inputs.append(ghost(mt,nx,ny,nz,bx,by))
                outputs=[ghost(np.zeros_like(km),nx,ny,nz,bx,by) for _ in range(4)]
                native.oracle_km2(*[ct.c_int(v) for v in (nx,ny,nz,bx,by,isotope,isfflx)],
                    *[ptr(v) for v in inputs],ct.c_float(dx),ct.c_float(dy),ct.c_float(1.),
                    ct.c_float(.15),ct.c_float(.5),*[ptr(v) for v in outputs])
                name=f"k2_{case}_{isotope}_{isfflx}"
                payload=dict(tke=tke,mt=mt,mfu=mfu,mfv=mfv,phb=phb)
                payload["expected_bn2"]=expected_bn2
                payload.update({"expected_"+field:unghost(v,km.shape)
                                for field,v in zip(("km","kh","kmv","khv"),outputs)})
                payload["expected_tke"]=unghost(inputs[0],km.shape)
                for key,value in payload.items():arrays[f"{name}_{key}"]=value
                cases.append(dict(name=name,family="km2",bx=bx,by=by,isotropic=isotope,
                                  isfflx=isfflx,dx=float(dx),dy=float(dy)))
    np.savez_compressed(output/"wrf471.npz",**arrays)
    receipt = json.loads(library.with_name("build-receipt.json").read_text())
    # Portable source evidence without build-host paths.
    for key in ("commands",): receipt.pop(key,None)
    metadata = dict(schema="wrf471-diff-opt1-v1",cases=cases,build=receipt,
                    archive_sha256=hashlib.sha256((output/"wrf471.npz").read_bytes()).hexdigest())
    (output/"wrf471.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("library",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();build(a.library,a.output)
