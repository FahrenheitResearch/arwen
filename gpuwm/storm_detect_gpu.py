"""CUDA diagnostics for basin-driven mesoscale cyclone tracking.

Native-reader float buffers cross this driver. Atmospheric decode, unit
conversion, interpolation and weather rendering remain in the Rust tools.
"""
from __future__ import annotations
import math
from pathlib import Path


class Detector:
    def __init__(self, grid, row):
        import cupy as cp
        self.cp, self.row = cp, row
        self.lat_host = grid['lat']
        self.lon_host = grid['lon']
        self.lat = cp.asarray(grid['lat'], dtype=cp.float64)
        self.lon = cp.asarray(grid['lon'], dtype=cp.float64)
        self.phi = cp.deg2rad(self.lat)[:, None]
        self.lam = cp.deg2rad(self.lon)[None, :]
        self.shape = tuple(grid['shape'])
        self.periodic = abs(abs(grid['lon'][-1]-grid['lon'][0])+abs(grid['lon'][1]-grid['lon'][0])-360) < 1e-5
        self.dy = float(grid['lat'][1]-grid['lat'][0])*math.pi/180
        self.dx = float(grid['lon'][1]-grid['lon'][0])*math.pi/180
        self.weights = cp.broadcast_to(cp.cos(self.phi), self.shape)

    def distance(self, lat, lon):
        cp = self.cp
        a = cp.deg2rad(cp.asarray(lat))[..., None, None]
        b = cp.deg2rad(cp.asarray(lon))[..., None, None]
        h = cp.sin((self.phi-a)/2)**2+cp.cos(a)*cp.cos(self.phi)*cp.sin((self.lam-b)/2)**2
        return 12742.*cp.arcsin(cp.sqrt(cp.clip(h, 0, 1)))

    def vorticity(self, u, v):
        cp = self.cp
        dv = ((cp.roll(v, -1, 1)-cp.roll(v, 1, 1))/(2*self.dx) if self.periodic else cp.gradient(v, self.dx, axis=1))
        du = cp.gradient(u*cp.cos(self.phi), self.dy, axis=0)
        return (dv-du)/(6371000.*cp.maximum(cp.cos(self.phi), 1.e-8))

    def read(self, directory):
        cp = self.cp
        import numpy as np
        data = {key: cp.asarray(np.fromfile(Path(directory)/(key+'.f32'), dtype='<f4').reshape(self.shape)) for key in self.row['fields']}
        # SST may legitimately be missing over land. Rebuild the validity mask
        # from finite atmospheric fields and ocean-only SST on the GPU.
        data['mask'] = cp.ones(self.shape, dtype=bool)
        for key in self.row['fields']:
            if key != 'sst': data['mask'] &= cp.isfinite(data[key])
        data['mask'] &= cp.isfinite(data['sst']) | (data['land'] >= .5)
        data['vort'] = self.vorticity(data['u850'], data['v850'])
        data['wind'] = cp.hypot(data['u10m'], data['v10m'])
        return data

    def candidates(self, data):
        cp, r = self.cp, self.row
        from cupyx.scipy.ndimage import minimum_filter
        mask = data['mask'] & cp.isfinite(data['vort'])
        valid = minimum_filter(mask, size=3, mode=('constant','wrap' if self.periodic else 'constant'), cval=0)
        south,north,west,east = r['bounds']
        lon = (self.lon+180)%360-180
        inside = (self.lat[:,None]>=south)&(self.lat[:,None]<=north)&(lon[None,:]>=west)&(lon[None,:]<=east)
        pressure = cp.where(mask, data['msl'] if r['candidate_field']=='mslp' else -data['vort'], cp.inf)
        minima = pressure == minimum_filter(pressure, size=5, mode=('nearest','wrap' if self.periodic else 'nearest'))
        if r['candidate_field']=='vorticity850':
            inside &= data['vort'] >= r['vorticity_min_s']
        indices = cp.flatnonzero(minima & valid & inside & (data['msl'] <= r['candidate_max_mslp_pa']))
        indices = indices[cp.argsort(pressure.ravel()[indices])]
        selected = []
        while indices.size:
            k = int(indices[0].item()); y,x = divmod(k, self.shape[1])
            selected.append((y,x))
            if len(selected)>r['max_candidates']:
                raise ValueError('candidate limit exceeded; increase max_candidates in the basin row')
            # Separation is a candidate-metadata operation, not a full-field scan.
            a=cp.deg2rad(self.lat[indices//self.shape[1]])
            b=cp.deg2rad(self.lon[indices%self.shape[1]])
            phi=math.radians(self.lat_host[y]);lam=math.radians(self.lon_host[x])
            h=cp.sin((a-phi)/2)**2+cp.cos(a)*math.cos(phi)*cp.sin((b-lam)/2)**2
            d=12742.*cp.arcsin(cp.sqrt(cp.clip(h,0,1)))
            indices = indices[d > r['candidate_min_separation_km']]
        return selected

    def diagnostic(self, data, y, x, motion=None):
        return self.diagnostics(data, [(y,x)], [motion])[0]

    def diagnostics(self, data, indices, motions):
        """Reduce all candidate neighborhoods together, with one metadata copy.

        The spherical-cap longitude bound retains every grid point used by the
        physical diagnostics, including polar neighborhoods and the seam. It
        avoids launching full-domain reductions once for every candidate.
        """
        if not indices:return []
        cp,r=self.cp,self.row
        radius=max(r['wind_radius_km'],r['closed_low_radius_km'],r['diagnostic_radius_km'],r['environment_radius_km'])
        latitudes=[self.lat_host[y] for y,x in indices]
        longitudes=[self.lon_host[x] for y,x in indices]
        angular=radius/6371.
        polar=max(abs(a) for a in latitudes)+math.degrees(angular)>=90
        lon_angle=math.pi if polar else math.asin(min(1.,math.sin(angular)/math.cos(math.radians(max(abs(a) for a in latitudes)))))
        y_count=min(self.shape[0],2*(math.ceil(angular/abs(self.dy))+1)+1)
        x_count=min(self.shape[1],2*(math.ceil(lon_angle/abs(self.dx))+1)+1)
        y=cp.asarray([i[0] for i in indices])[:,None,None]+(cp.arange(y_count)-y_count//2)[None,:,None]
        x=cp.asarray([i[1] for i in indices])[:,None,None]+(cp.arange(x_count)-x_count//2)[None,None,:]
        inside=(y>=0)&(y<self.shape[0])
        if not self.periodic:inside=inside&(x>=0)&(x<self.shape[1])
        ys=cp.clip(y,0,self.shape[0]-1)
        xs=x%self.shape[1] if self.periodic else cp.clip(x,0,self.shape[1]-1)
        phi=cp.deg2rad(self.lat[ys]);lam=cp.deg2rad(self.lon[xs])
        a=cp.deg2rad(cp.asarray(latitudes))[:,None,None]
        b=cp.deg2rad(cp.asarray(longitudes))[:,None,None]
        h=cp.sin((phi-a)/2)**2+cp.cos(a)*cp.cos(phi)*cp.sin((lam-b)/2)**2
        distance=12742.*cp.arcsin(cp.sqrt(cp.clip(h,0,1)))
        env=inside&(distance<=r['environment_radius_km'])
        disk=inside&(distance<=r['diagnostic_radius_km'])
        winddisk=inside&(distance<=r['wind_radius_km'])
        ring=inside&(distance>=.85*r['closed_low_radius_km'])&(distance<=r['closed_low_radius_km'])
        weights=cp.broadcast_to(cp.cos(phi),distance.shape)
        values={k:v[ys,xs] for k,v in data.items()}
        def mean(v,mask):
            w=cp.where(mask,weights,0.)
            return cp.sum(cp.where(mask,v,0.)*w,axis=(1,2))/cp.maximum(w.sum(axis=(1,2)),1.e-12)
        boundary=[min(abs(a-self.lat_host[0]),abs(a-self.lat_host[-1]))*111.195 for a in latitudes]
        if not self.periodic:
            boundary=[min(edge,min(abs((lon-self.lon_host[i]+180)%360-180) for i in (0,-1))*111.195*math.cos(math.radians(lat))) for edge,lat,lon in zip(boundary,latitudes,longitudes)]
        coverage=(cp.asarray(boundary)>=radius)&cp.all(cp.where(inside&(distance<=radius),values['mask'],True),axis=(1,2))
        ocean=values['land']<.5
        center_msl=data['msl'][cp.asarray([y for y,x in indices]),cp.asarray([x for y,x in indices])]
        columns=dict(mslp_pa=center_msl,max_wind10_ms=cp.max(cp.where(winddisk,values['wind'],-cp.inf),axis=(1,2)),
                     vorticity850_s=cp.max(cp.where(env,values['vort']*cp.where(a>=0,1,-1),-cp.inf),axis=(1,2)),
                     ocean_fraction=mean(ocean.astype(cp.float32),env),sst_t500_k=mean(values['sst']-values['t500'],env&ocean),
                     closed_low_delta_pa=cp.min(cp.where(ring,values['msl'],cp.inf),axis=(1,2))-center_msl,coverage_complete=coverage)
        if r['kind']=='medicane':
            def thermal(levels):
                amplitudes=cp.stack([cp.max(cp.where(disk,values[f'z{p}'],-cp.inf),axis=(1,2))-cp.min(cp.where(disk,values[f'z{p}'],cp.inf),axis=(1,2)) for p in levels])
                logs=cp.log(cp.asarray(list(levels),dtype=cp.float64));centered=logs-logs.mean()
                return cp.sum(centered[:,None]*amplitudes,axis=0)/cp.sum(centered**2)
            # A grid cell centred on the dividing great circle contributes half
            # its area to each side. Resolve roundoff within 1e-14 radians
            # symmetrically, so batching cannot arbitrarily flip these cells.
            center_lon=cp.asarray([math.radians(lon) for lon in longitudes])[:,None,None]
            center_sin=cp.asarray([math.sin(math.radians(lat)) for lat in latitudes])[:,None,None]
            center_cos=cp.asarray([math.cos(math.radians(lat)) for lat in latitudes])[:,None,None]
            dx=cp.cos(phi)*cp.sin(lam-center_lon)
            dy=center_cos*cp.sin(phi)-center_sin*cp.cos(phi)*cp.cos(lam-center_lon)
            angles=[[math.atan2(m[1],m[0]) if m and math.hypot(*m)>1.e-9 else i*math.pi/18 for i in range(18)] for m in motions]
            sine=cp.asarray([[math.sin(v) for v in row] for row in angles])[:,:,None,None]
            cosine=cp.asarray([[math.cos(v) for v in row] for row in angles])[:,:,None,None]
            signed=dx[:,None]*sine-dy[:,None]*cosine
            right=cp.where(cp.abs(signed)<=1.e-14,.5,(signed>0).astype(cp.float64))
            thick=(values['z600']-values['z900'])[:,None]
            def side_mean(mask):
                w=cp.where(disk[:,None],weights[:,None]*mask,0.)
                return cp.sum(cp.where(disk[:,None],thick,0.)*w,axis=(2,3))/cp.maximum(w.sum(axis=(2,3)),1.e-12)
            columns.update(hart_b_abs_m=cp.max(cp.abs(side_mean(right)-side_mean(1-right)),axis=1),hart_lower_m=thermal(range(600,901,50)),hart_upper_m=thermal(range(300,601,50)))
        # These are bounded per-candidate diagnostics, never weather planes.
        host=cp.asnumpy(cp.stack(list(columns.values()),axis=1))
        results=[]
        for lat,lon,motion,values in zip(latitudes,longitudes,motions,host):
            result=dict(lat=float(lat),lon=(float(lon)+180)%360-180,**{k:float(v) for k,v in zip(columns,values)})
            result['coverage_complete']=bool(result['coverage_complete'])
            failures=[]
            for key,threshold in [('max_wind10_ms','wind_min_ms'),('vorticity850_s','vorticity_min_s'),('ocean_fraction','ocean_fraction_min'),('closed_low_delta_pa','closed_low_delta_pa')]:
                if r[threshold] is not None and result[key]<r[threshold]:failures.append(key)
            if not result['coverage_complete']:failures.append('coverage')
            if r['kind']=='medicane':
                result['hart_motion']='track_displacement' if motion and math.hypot(*motion)>1.e-9 else 'stationary_maximum_orientation'
                for key,threshold in [('hart_lower_m','hart_lower_min_m'),('hart_upper_m','hart_upper_min_m')]:
                    if result[key]<=r[threshold]:failures.append(key)
                if result['hart_b_abs_m']>r['hart_b_abs_max_m']:failures.append('hart_b_abs_m')
            elif result['sst_t500_k']<=r['sst_t500_min_k']:failures.append('sst_t500_k')
            for key,value in list(result.items()):
                if isinstance(value,float) and not math.isfinite(value):
                    failures.append('nonfinite_'+key);result[key]=None
            result['failed_criteria']=failures;result['classified']=not failures
            results.append(result)
        return results


def associate(detector, tracks, fixes, valid, max_speed_ms, max_gap_hours):
    """One-to-one distance ordering on GPU; bounded track metadata only."""
    from gpuwm.tc_tracks import utc
    cp=detector.cp
    active=[t for t in tracks if 0<(utc(valid)-utc(t['fixes'][-1]['valid'])).total_seconds()<=max_gap_hours*3600]
    links={}
    if active and fixes:
        a=cp.asarray([[t['fixes'][-1]['lat'],t['fixes'][-1]['lon']] for t in active])
        b=cp.asarray([[f['lat'],f['lon']] for f in fixes])
        phi=cp.deg2rad(a[:,0,None]);theta=cp.deg2rad(b[None,:,0])
        h=cp.sin((phi-theta)/2)**2+cp.cos(phi)*cp.cos(theta)*cp.sin(cp.deg2rad(a[:,1,None]-b[None,:,1])/2)**2
        cost=12742*cp.arcsin(cp.sqrt(cp.clip(h,0,1)))
        seconds=cp.asarray([(utc(valid)-utc(t['fixes'][-1]['valid'])).total_seconds() for t in active])
        # Constant-motion prediction prevents a nearby competing vortex from
        # stealing the identity of a moving low. The last-position speed gate
        # remains binding, including after a missing frame.
        previous=cp.asarray([[t['fixes'][-2]['lat'],t['fixes'][-2]['lon']] if len(t['fixes'])>1 else [t['fixes'][-1]['lat'],t['fixes'][-1]['lon']] for t in active])
        intervals=cp.asarray([(utc(t['fixes'][-1]['valid'])-utc(t['fixes'][-2]['valid'])).total_seconds() if len(t['fixes'])>1 else 1 for t in active])
        ratio=seconds/cp.maximum(intervals,1)
        predicted_lat=cp.clip(a[:,0]+(a[:,0]-previous[:,0])*ratio,-89.9,89.9)
        predicted_lon=a[:,1]+((a[:,1]-previous[:,1]+180)%360-180)*ratio
        phi_pred=cp.deg2rad(predicted_lat[:,None])
        h_pred=cp.sin((phi_pred-theta)/2)**2+cp.cos(phi_pred)*cp.cos(theta)*cp.sin(cp.deg2rad(predicted_lon[:,None]-b[None,:,1])/2)**2
        predicted_distance=12742*cp.arcsin(cp.sqrt(cp.clip(h_pred,0,1)))
        cost=cp.where(cost<=seconds[:,None]*max_speed_ms/1000,.75*predicted_distance+.25*cost,cp.inf)
        for _ in range(min(len(active),len(fixes))):
            k=int(cp.argmin(cost).item());i,j=divmod(k,len(fixes))
            if not bool(cp.isfinite(cost[i,j]).item()):break
            links[j]=active[i];cost[i,:]=cp.inf;cost[:,j]=cp.inf
    return links
