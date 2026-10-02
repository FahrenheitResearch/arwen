"""Tools-only float32 staging intervention for deformation attribution.

This preserves the production stencil and physics, but restores WRF's six-term
surface interpolation and separately rounded D12 contributions. It is never an
acceptance path or an engine option.
"""
from __future__ import annotations
import re


def _edit(source,name,transform):
    match=re.search(r"real "+name+r"\([^{}]*\)\s*\{",source)
    if match is None:raise ValueError(name)
    start=match.end();end=start;depth=1
    while depth:
        depth+=(source[end]=="{")-(source[end]=="}")
        end+=1
    return source[:start]+transform(source[start:end-1])+source[end-1:]


def reference_tensor_order(source):
    """Expose the exact lines whose float32 rounding causes the residual."""
    definitions={
        "wrf_u_w_xcenter":("wrf_uhat",("j,i","j,i + 1")),
        "wrf_v_w_ycenter":("wrf_vhat",("j,i","j + 1,i")),
        "wrf_u_w_corner":("wrf_uhat",("j - 1,i","j,i")),
        "wrf_v_w_corner":("wrf_vhat",("j,i - 1","j,i")),
    }
    for name,(fn,faces) in definitions.items():
        terms=[f"q.cf{k+1} * {fn}(q,{k},{face})" for face in faces for k in range(3)]
        bottom="\nif (kw <= 0) return 0.5f * ("+" + ".join(terms)+");\n"
        source=_edit(source,name,lambda body,bottom=bottom:bottom+body)
    def d12(body):
        old="* 0.25f * tmpzx * rr;"
        new="* 0.25f * tmpzx * (wrf_rdzw(q,k,j,i) + wrf_rdzw(q,k,j-1,i) + wrf_rdzw(q,k,j-1,i-1) + wrf_rdzw(q,k,j,i-1));"
        if body.count(old)!=1:raise ValueError("D12 slope staging changed")
        body=body.replace(old,new)
        old="return mm * (q.rdy * (wrf_uhat(q, k, j, i)\n                         - wrf_uhat(q, k, j - 1, i)) - uslope\n               + q.rdx * (wrf_vhat(q, k, j, i)\n                         - wrf_vhat(q, k, j, i - 1)) - vslope);"
        new="return mm * (q.rdy * (wrf_uhat(q,k,j,i)-wrf_uhat(q,k,j-1,i))-uslope) + mm * (q.rdx * (wrf_vhat(q,k,j,i)-wrf_vhat(q,k,j,i-1))-vslope);"
        if body.count(old)!=1:raise ValueError("D12 contribution staging changed")
        return body.replace(old,new)
    source=_edit(source,"wrf_defor12",d12)
    # phy_prep averages already rounded w-level heights. The production
    # reconstruction averages geopotential before dividing by gravity.
    pattern=r"__fdiv_rn\(0\.5f \* \(wrf_phi\(q, ([^()]*)\)\s*\+\s*wrf_phi\(q, ([^()]*)\)\), G\)"
    source,count=re.subn(pattern,lambda match:
                        "0.5f * (__fdiv_rn(wrf_phi(q, "+match[1]+"), G) + __fdiv_rn(wrf_phi(q, "+match[2]+"), G))",source)
    if count<4:raise ValueError("Surface height staging changed")
    return source


def reference_smag2d_coefficient_order(source):
    """Restore the WRF km_opt=4 strain and map-factor product stages."""
    substitutions={
        "sqrtf(0.25f * (d11 - d22) * (d11 - d22) + d12 * d12)":
        "sqrtf(0.25f * ((d11 - d22) * (d11 - d22)) + d12 * d12)",
        "real mlen = sqrtf(dxm * dym);":
        "real mlen = sqrtf((dxm * dy) / map);",
    }
    for old,new in substitutions.items():
        if source.count(old)!=1:raise ValueError("Smagorinsky coefficient staging changed: "+old)
        source=source.replace(old,new)
    return source
