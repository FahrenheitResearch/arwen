"""CUDA forcing hand-off and wave-age roughness; Python only binds launches."""
from __future__ import annotations
import math
from .interface import (AtmosphereForcing, WaveOutputs, validate_fields)

_CUDA = r"""
extern "C" __global__ void forcing(const float* u,const float* v,const float* p,
 const float* us,const float* t,const float* q,const float* ca,const float* sa,
 float* east,float* north,float* pressure,float* ustar,float* tx,float* ty,float* rho,
 int rotate,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float x=rotate?ca[i]*u[i]-sa[i]*v[i]:u[i];
 float y=rotate?sa[i]*u[i]+ca[i]*v[i]:v[i];
 float speed=hypotf(x,y);float a=fmaxf(us[i],0.f);
 float d=p[i]/(287.05f*t[i]*(1.f+0.61f*fmaxf(q[i],0.f)));
 east[i]=x;north[i]=y;pressure[i]=p[i];ustar[i]=a;rho[i]=d;
 float stress=d*a*a/fmaxf(speed,1.e-12f);
 tx[i]=stress*x;ty[i]=stress*y;
}
// ERA5 replay uses the physical neutral log layer when no model u* exists.
// Interpolation, humidity and drag diagnosis are CUDA, never host transforms.
extern "C" __global__ void replay(const float* u,const float* v,const float* p,
 const float* t,const float* td,const float* z0,float* ou,float* ov,float* op,
 float* ot,float* oq,float* us,float* speed,float a,int record,int cells) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cells)return;
 int k=record*cells+i;int j=k+cells;
 float x=u[k]+a*(u[j]-u[k]);float y=v[k]+a*(v[j]-v[k]);
 float pr=p[k]+a*(p[j]-p[k]);float temp=t[k]+a*(t[j]-t[k]);
 float dew=td[k]+a*(td[j]-td[k]);float e=611.2f*expf(17.67f*(dew-273.15f)/(dew-29.65f));
 ou[i]=x;ov[i]=y;op[i]=pr;ot[i]=temp;oq[i]=0.622f*e/(pr-0.378f*e);
 float w=hypotf(x,y);speed[i]=w;
 // Bounded ocean roughness includes the smooth-flow viscous contribution.
 float z=fmaxf(z0[i],1.e-4f);us[i]=0.4f*w/logf((10.f+z)/z);
}
extern "C" __global__ void stress_mag(const float* x,const float* y,float* out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=hypotf(x[i],y[i]);
}
extern "C" __global__ void roughness(const float* hs,const float* tp,const float* us,
 const float* depth,const float* wet,const float* ice,float* alpha,float* z0,int internal,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(wet[i]<0.5f||ice[i]>0.01f){if(internal)z0[i]=0.f;return;}
 float a=fmaxf(us[i],0.001f);float period=tp[i];float h=hs[i];
 if(!(h>0.f&&period>0.f)){alpha[i]=0.0185f;z0[i]=__fdiv_rn(0.0185f*a*a,9.80665f)+0.11f*1.5e-5f/a;return;}
 // Dispersion: omega^2=g*k*tanh(k*d), GPU Newton iteration at fixed depth.
 float om=6.28318530718f/period;float d=fmaxf(depth[i],0.05f);
 float k=fmaxf(__fdiv_rn(om*om,9.80665f),om/sqrtf(9.80665f*d));
 for(int j=0;j<8;j++){float th=tanhf(k*d);float f=9.80665f*k*th-om*om;
  k=fmaxf(1.e-8f,k-f/(9.80665f*(th+k*d*(1.f-th*th))));}
 float cp=om/k;
 // Drennan et al. 2003, doi:10.1029/2000JC000715: z0/Hs=3.35*(u*/cp)^3.4.
 float wave_z=3.35f*h*powf(a/cp,3.4f);
 float ch=fminf(0.1f,fmaxf(0.005f,wave_z*9.80665f/(a*a)));
 alpha[i]=ch;z0[i]=fminf(0.02f,__fdiv_rn(ch*a*a,9.80665f)+0.11f*1.5e-5f/a);
}
"""

class DeviceCoupler:
    def __init__(self, grid, bathymetry, *, waves=None, surge=None, interval=60.):
        import cupy as cp
        self.grid,self.bathymetry=grid,bathymetry
        grid.validate();validate_fields(bathymetry,grid)
        if not math.isfinite(interval) or interval<=0: raise ValueError("coupling interval must be finite and positive")
        self.interval=float(interval);self.waves=waves;self.surge=surge
        self.next_time=None;self.last_time=None;self.steps=0
        self.buffers={k:cp.empty(grid.shape,dtype=cp.float32) for k in ('u10','v10','psfc','ustar','tau_x','tau_y','rho_air')}
        self.alpha=cp.full(grid.shape,0.0185,dtype=cp.float32)
        self.zero=cp.zeros(grid.shape,dtype=cp.float32)
        self.z0=cp.where(bathymetry.wet>0.5,cp.float32(1.e-4),cp.float32(0.)).astype(cp.float32)
        from gpuwm.certify.kernel_manifest import record_module
        module=cp.RawModule(code=_CUDA,options=())
        record_module('gpuwm.ocean.coupling:device',source=_CUDA,options=(),module=module)
        self._forcing=module.get_function('forcing')
        self._roughness=module.get_function('roughness')
        self.forcing=None

    def handoff(self, *, u10,v10,psfc,ustar,temperature,qv,time_seconds,grid_relative=False):
        import cupy as cp
        import numpy as np
        from .interface import validate_plane
        for name,a in [('u10',u10),('v10',v10),('psfc',psfc),('ustar',ustar),('temperature',temperature),('qv',qv)]:
            validate_plane(a,self.grid.shape,name)
            if a.device.id!=self.buffers['u10'].device.id: raise ValueError("forcing belongs to another CUDA device")
        n=self.grid.nx*self.grid.ny;b=self.buffers
        self._forcing(((n+255)//256,),(256,),
            (u10,v10,psfc,ustar,temperature,qv,self.grid.cosalpha,self.grid.sinalpha,
             *(b[k] for k in ('u10','v10','psfc','ustar','tau_x','tau_y','rho_air')),
             np.int32(grid_relative),np.int32(n)))
        f=AtmosphereForcing(**b,time_seconds=float(time_seconds));self.forcing=f
        if self.waves is not None:self.waves.set_forcing(f)
        if self.surge is not None:self.surge.set_forcing(f)
        return f

    def advance(self, dt):
        if self.forcing is None:raise RuntimeError("ocean advance requires current atmospheric forcing")
        if not math.isfinite(dt) or dt<=0:raise ValueError("ocean dt must be finite and positive")
        if self.waves is not None:
            if self.surge is not None:self.waves.set_water_level(self.surge.outputs.eta)
            self.waves.step(dt)
        if self.surge is not None:
            if self.waves is not None:self.surge.set_waves(self.waves.outputs)
            self.surge.step(dt)
        self.steps+=1

    def surface_roughness(self, znt,ustar,*,ice=None):
        import numpy as np
        if self.waves is None:return
        o=self.waves.outputs;n=self.grid.nx*self.grid.ny
        if znt is not self.z0:
            self.surface_roughness(self.z0,ustar,ice=ice)
        self._roughness(((n+255)//256,),(256,),
            (o.hs,o.tp,ustar,self.bathymetry.depth,self.bathymetry.wet,
             self.zero if ice is None else ice,self.alpha,znt,np.int32(znt is self.z0),np.int32(n)))

    def atmosphere_step(self, fields, atmosphere, time_seconds,dt):
        # Surface diagnostics are refreshed upstream. Every atmosphere step
        # updates held forcing, and substeps partition that step exactly.
        self.handoff(u10=fields['u10'],v10=fields['v10'],psfc=fields['psfc'],
                     ustar=fields['ust'],temperature=atmosphere['temperature'][0],
                     qv=atmosphere['qv'][0],time_seconds=time_seconds,grid_relative=True)
        remaining=float(dt)
        while remaining>1.e-9:
            sub=min(self.interval,remaining);self.advance(sub);remaining-=sub
        self.last_time=float(time_seconds)+float(dt)

    def history_fields(self):
        """Output-cadence device views, consumed by the existing Rust writer."""
        from .door import rows
        values={'elevation':self.bathymetry.elevation,'wet':self.bathymetry.wet}
        if self.waves is not None:
            o=self.waves.outputs;values.update(hs=o.hs,tp=o.tp,direction=o.direction)
        if self.surge is not None:
            o=self.surge.outputs;values.update(eta=o.eta,inundation=o.inundation)
        return {rows()['products'][key]['history_name']:array for key,array in values.items()}
