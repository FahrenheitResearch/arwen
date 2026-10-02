"""Tools-only staging control for WRF scalar-flux rounding attribution."""
from deformation_arithmetic import _edit,reference_tensor_order

def reference_stress_order(source):
    for name,first,second in (
        ("wrf_smag_hd_u","msf*zx_u*(wrf_tau11_uavg(q,km,d11,k+1,j,i)-wrf_tau11_uavg(q,km,d11,k,j,i))/tmpdz",
         "msf*zy_u*(wrf_tau12_uavg(q,km,d12,k+1,j,i)-wrf_tau12_uavg(q,km,d12,k,j,i))/tmpdz"),
        ("wrf_smag_hd_v","msf*zx_v*(wrf_tau12_vavg(q,km,d12,k+1,j,i)-wrf_tau12_vavg(q,km,d12,k,j,i))/tmpdz",
         "msf*zy_v*(wrf_tau22_vavg(q,km,d22,k+1,j,i)-wrf_tau22_vavg(q,km,d22,k,j,i))/tmpdz")):
        old="G * tmpdz / dnw[k] * (divh - divz)"
        new=f"G*tmpdz/dnw[k]*((divh-({first}))-({second}))"
        # Global entry points have void returns; adapt the same brace walker.
        begin=source.index("void "+name+"(")
        end=source.index("\n}\n",begin)+2
        body=source[begin:end]
        if body.count(old)!=1:raise ValueError(name+" stress staging changed")
        body=body.replace(old,new)
        if name.endswith("_v"):
            import re
            pattern=r"real zx_v = 0.125f \* \([^;]*;"
            ordered="real zx_v=0.125f*(wrf_zx(q,k,jc,i)+wrf_zx(q,k,jc,i+1)+wrf_zx(q,k,jm,i)+wrf_zx(q,k,jm,i+1)+wrf_zx(q,k+1,jc,i)+wrf_zx(q,k+1,jc,i+1)+wrf_zx(q,k+1,jm,i)+wrf_zx(q,k+1,jm,i+1));"
            body,count=re.subn(pattern,ordered,body)
            if count!=1:raise ValueError("V slope average staging changed")
        source=source[:begin]+body+source[end:]
    for name,tau,face in (("wrf_tau13_mavg","wrf_tau13","j,i+1"),("wrf_tau23_mavg","wrf_tau23","j+1,i")):
        body=f"return 0.25f*({tau}(q,km,k+1,{face})+{tau}(q,km,k+1,j,i)+{tau}(q,km,k,{face})+{tau}(q,km,k,j,i));"
        source=_edit(source,name,lambda original,body=body:body)
    return source

def reference_flux_order(source):
    source=reference_tensor_order(source)
    for name,faces in (("wrf_scalar_w_xface",("j,i","j,i - 1")),
                       ("wrf_scalar_w_yface",("j,i","j - 1,i"))):
        at=lambda k,f:f"wrf_scalar(q,s,{k},{f})"
        terms=[f"q.cf{k+1} * {at(k,face)}" for face in faces for k in range(3)]
        bottom="\nif (kw <= 0) return 0.5f * ("+" + ".join(terms)+");\n"
        top=[]
        for face in faces:
            last,prev=at("q.nz-1",face),at("q.nz-2",face)
            top.append(f"{last}+({last}-{prev})*0.5f*q.dnw[q.nz-1]/q.dn[q.nz-1]")
        top="if (kw >= q.nz) return 0.5f * ("+" + ".join(top)+");\n"
        source=_edit(source,name,lambda body,bottom=bottom,top=top:bottom+top+body)
    for name,coords in (("wrf_h1",("j,i-1","j,i")),("wrf_h2",("j-1,i","j,i"))):
        old="return -map * rhoavg * kavg * grad;"
        new=f"real xkxavg=kavg*0.5f*(wrf_rho(q,k,{coords[0]})+wrf_rho(q,k,{coords[1]})); return -map*xkxavg*grad;"
        def replace(body):
            if body.count(old)!=1:raise ValueError(name+" flux return changed")
            return body.replace(old,new)
        source=_edit(source,name,replace)
    old="tend[IDX3(k, j, i)] += G / (dnw[k] * rz) * (divh - divz);"
    new="""real first=map*zx_m*(wrf_h1_mavg(q,fx,k+1,j,i)-wrf_h1_mavg(q,fx,k,j,i))*rz;
    real second=map*zy_m*(wrf_h2_mavg(q,fy,k+1,j,i)-wrf_h2_mavg(q,fy,k,j,i))*rz;
    tend[IDX3(k,j,i)] += G/(dnw[k]*rz)*((divh-first)-second);"""
    if source.count(old)!=1:raise ValueError("Scalar divergence staging changed")
    return reference_stress_order(source.replace(old,new))
