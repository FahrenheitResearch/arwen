"""WRF default vertical mixing and deposition, dry_dep_driver.F:675-805.

The solver spans all nz mass levels (WRF kte=kde-1), but WRF writes only
kts:kte-1 back. The last mass level stays unchanged, not a fabricated
padding cell. z_at_w has nz+1 interfaces. No undefined WRF padding is read.
See tools/chem_wrf471_oracle/VERTMX-NOTES.md for bounds and conservation.
"""
from __future__ import annotations

import numpy as np
from gpuwm.core.chem_context import ChemAllocation

KEY = 'mixing.vertmx'
LEDGER = 'deposited'
REQUIRES = ('alt','z_at_w','z','dz8w','exch_h')
KERNEL_MODULES = ('chem_vertmx',)
ALLOCATES = (ChemAllocation('vertmx_deposited','rows_2d',restart='serialize',
             description='Accumulated WRF ddmassn; gas mol m-2, aerosol ug m-2'),)


def rows(table):
    return table.rows_for(KEY)


def refusal(cfg):
    """Why this configuration cannot run mixing.vertmx, or None.

    vertmx takes its eddy diffusivity from the PBL scheme's ``exch_h``
    (dry_dep_driver.F:686); with no PBL scheme nothing writes it, and every
    column would mix with the 1e-6 m2 s-1 floor alone -- chem would never
    leave the surface layer while the configuration claimed it was mixed.
    """
    if int(getattr(cfg, "bl_pbl_physics", 0)) == 0:
        return ("mixing.vertmx needs a PBL scheme's exch_h "
                "(dry_dep_driver.F:686), and bl_pbl_physics = 0 publishes "
                "none, so chem would stay in the lowest layer")
    return None


def init(ctx):
    """Driver allocates zeroed diagnostics; preserve restored accumulation."""
    pass






def launch_vertmx(chem, alt, z_at_w, z, dz8w, exch_h, vd, dt, deposited, *,
                  phase, floor=1e-16, anth_co_kts=None, fire_co_k1=None,
                  anth_pm25_pair=None, anth_pm25=None, sf_urban_physics=0,
                  mixed=None, ekmfull=None, ddmassn=None):
    """One column thread per row launch, independent of row storage order.

    Optional outputs let the node parity gate measure solver and floor words.
    The process path uses serialized accumulation and omits scratch outputs.
    """
    from gpuwm.core.kernels import get_kernel_int_defines
    if isinstance(chem, np.ndarray):
        raise TypeError(
            "vertical chemistry mixing requires CUDA arrays: NumPy fields "
            "would supply host pointers to the column solver")
    nz,ny,nx=chem.shape
    if nz<11:
        raise ValueError('vertmx needs at least 11 mass levels: WRF mixing floor writes kts+10')
    if any(x.shape!=chem.shape for x in (alt,z,dz8w,exch_h)) or z_at_w.shape!=(nz+1,ny,nx):
        raise ValueError('vertmx mass/w shapes disagree: diffusion would read the wrong interface')
    if phase not in ('gas','aerosol'):
        raise ValueError('vertmx phase must give gas molar or aerosol mass bookkeeping')
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('vertmx dt must be positive and finite: the solver divides by dt')
    optional=(anth_co_kts,fire_co_k1,
              None if anth_pm25_pair is None else anth_pm25_pair[0],
              None if anth_pm25_pair is None else anth_pm25_pair[1],anth_pm25)
    for x in (vd,deposited)+tuple(v for v in optional if v is not None):
        if x.shape!=(ny,nx):
            raise ValueError('vertmx source/velocity/accumulation planes must match columns')
    if (mixed is None)!=(ekmfull is None) or (mixed is None)!=(ddmassn is None):
        raise ValueError('vertmx debug outputs must be supplied together to prevent invalid writes')
    args=(chem,alt,z_at_w,z,dz8w,exch_h,vd,np.float32(dt),np.float32(floor),
          np.int32(phase=='gas'),deposited)+tuple(chem if x is None else x for x in optional)
    args+=tuple(np.int32(x is not None) for x in (anth_co_kts,fire_co_k1,anth_pm25_pair,anth_pm25))
    args+=(np.int32(sf_urban_physics),np.int32(nz),np.int32(ny*nx),
           chem if mixed is None else mixed,chem if ekmfull is None else ekmfull,
           chem if ddmassn is None else ddmassn,np.int32(mixed is not None))
    get_kernel_int_defines('chem_vertmx','chem_vertmx',(('CHEM_NZ',nz),))(
        ((ny*nx+127)//128,),(128,),args)


def step(ctx, dt, ktau):
    """chem_driver.F:1046-1047 and dry_dep_driver.F:672,741,765 gates.

    Integration supplies selected source-row planes in physics_fields under
    anth_co_kts, fire_co_k1, anth_pm25_pair, anth_pm25. No source name is
    interpreted here. WRF's CAM microphysics gate is a per-row ownership
    decision at integration; rows(table) must exclude its already-mixed rows.
    """
    if ktau<=2 or getattr(ctx.cfg,'vertmix_onoff',1)<=0:
        return
    if getattr(ctx.cfg,'mynn_chem_vertmx',False) or ctx.physics_fields.get('num_vert_mix',0)!=0:
        return
    f=ctx.physics_fields
    inputs={name:f.get(name) for name in ('anth_co_kts','fire_co_k1','anth_pm25_pair','anth_pm25')}
    inputs['sf_urban_physics']=getattr(ctx.cfg,'sf_urban_physics',0)
    met={name:ctx.met(name) for name in REQUIRES}
    for index,row in enumerate(rows(ctx.table)):
        chem=ctx.field(row);vel=ctx.ddvel[ctx.row_index(row)]
        accum=ctx.diag['vertmx_deposited'][index]
        launch_vertmx(chem,met['alt'],met['z_at_w'],met['z'],met['dz8w'],met['exch_h'],vel,dt,accum,
                      phase=row.phase,floor=row.floor,**inputs)
