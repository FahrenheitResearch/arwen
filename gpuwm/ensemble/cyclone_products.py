"""CUDA diagnostic accumulation and orchestration of native cyclone products.

The common grid is fixed on Earth. No member is shifted to an ensemble mean
vortex. Missing coverage masks probabilities rather than changing weights.
"""
from pathlib import Path
import json
import subprocess


class CommonWind:
    def __init__(self, request):
        import cupy as cp
        self.cp, self.request = cp, request
        g = request['common_grid']
        self.lat, self.lon = cp.meshgrid(cp.linspace(g['south'], g['north'], g['ny'], dtype=cp.float32),
            cp.linspace(g['west'], g['east'], g['nx'], dtype=cp.float32), indexing='ij')
        self.maximum = cp.zeros(self.lat.shape, cp.float32)
        self.coverage = cp.ones(self.lat.shape, cp.bool_)
        self.onset = cp.full(self.lat.shape, cp.nan, cp.float32)
        self.duration = cp.zeros(self.lat.shape, cp.float32)
        self.key = None
        self.last_seconds = None
        self.samples = 0

    def bind(self, node, model):
        cp = self.cp
        key = (node.cfg.grid_id, id(node.grid))
        if key == self.key:
            return
        from gpuwm.io.history_layout import metadata_history_fields
        case = model._prepared_by_grid_id[node.cfg.grid_id]
        meta = metadata_history_fields(node.grid, case.static_fields)
        lat, lon = (cp.asarray(meta[k], dtype=cp.float32) for k in ('XLAT', 'XLONG'))
        ny, nx = lat.shape
        # Newton inversion of a smooth projected curvilinear grid. Retain
        # residual checks: a clamped edge is never valid out-of-domain data.
        x, y = cp.full(self.lat.shape, nx/2), cp.full(self.lat.shape, ny/2)
        for _ in range(8):
            i, j = cp.clip(x, 0, nx-2).astype(cp.int32), cp.clip(y, 0, ny-2).astype(cp.int32)
            a, b = x-i, y-j
            def value(q):
                return (1-a)*(1-b)*q[j,i]+a*(1-b)*q[j,i+1]+(1-a)*b*q[j+1,i]+a*b*q[j+1,i+1]
            dl, dn = self.lat-value(lat), (self.lon-value(lon)+180)%360-180
            lx = (1-b)*(lat[j,i+1]-lat[j,i])+b*(lat[j+1,i+1]-lat[j+1,i])
            ly = (1-a)*(lat[j+1,i]-lat[j,i])+a*(lat[j+1,i+1]-lat[j,i+1])
            nx_ = (1-b)*(lon[j,i+1]-lon[j,i])+b*(lon[j+1,i+1]-lon[j+1,i])
            ny_ = (1-a)*(lon[j+1,i]-lon[j,i])+a*(lon[j+1,i+1]-lon[j,i+1])
            determinant = lx*ny_-ly*nx_
            x += (dl*ny_-dn*ly)/determinant
            y += (dn*lx-dl*nx_)/determinant
        self.inside = (x >= 0)&(x <= nx-1)&(y >= 0)&(y <= ny-1)&cp.isfinite(x)&cp.isfinite(y)
        self.i = cp.nan_to_num(cp.clip(x, 0, nx-2)).astype(cp.int32)
        self.j = cp.nan_to_num(cp.clip(y, 0, ny-2)).astype(cp.int32)
        self.a, self.b = cp.clip(x-self.i, 0, 1), cp.clip(y-self.j, 0, 1)
        residual_lat = self.lat-self.sample(lat)
        residual_lon = (self.lon-self.sample(lon)+180)%360-180
        self.inside &= (cp.abs(residual_lat) < 1e-4)&(cp.abs(residual_lon) < 1e-4)
        self.rotation = tuple(cp.asarray(meta[k]) for k in ('SINALPHA', 'COSALPHA'))
        self.land = self.sample(cp.asarray(meta['LANDMASK']))
        smooth = cp.nan_to_num(self.land)
        for _ in range(3):
            p = cp.pad(smooth, 1, mode='edge')
            smooth = sum(p[j:j+smooth.shape[0], i:i+smooth.shape[1]] for j in range(3) for i in range(3))/9
        gy, gx = cp.gradient(smooth)
        spacing = self.request['common_grid']
        gx /= ((spacing['east']-spacing['west'])/(spacing['nx']-1))*cp.cos(cp.deg2rad(self.lat))
        gy /= (spacing['north']-spacing['south'])/(spacing['ny']-1)
        norm = cp.sqrt(gx*gx+gy*gy)
        # Keep the coast stencil independent of the requested grid spacing.
        edge_strength = cp.sqrt(cp.gradient(smooth)[0]**2+cp.gradient(smooth)[1]**2)
        self.coast = (edge_strength > .01)&self.inside
        self.normal = (gx/cp.maximum(norm, 1e-10), gy/cp.maximum(norm, 1e-10))
        self.bound_grid = node.grid  # retain the identity until the next bind
        self.key = key

    def sample(self, plane):
        i, j, a, b = self.i, self.j, self.a, self.b
        return (1-a)*(1-b)*plane[j,i]+a*(1-b)*plane[j,i+1]+(1-a)*b*plane[j+1,i]+a*b*plane[j+1,i+1]

    def observe(self, *, node, model):
        if node.cfg.grid_id != self.request.get('grid_id', 2):
            return
        cp = self.cp
        seconds = float(node.state.elapsed_seconds)
        if self.last_seconds == seconds:
            return
        self.bind(node, model)
        fields = node.state.physics.fields
        u, v = fields['u10'], fields['v10']
        sine, cosine = self.rotation
        east, north = self.sample(u*cosine-v*sine), self.sample(v*cosine+u*sine)
        wind = self.sample(cp.sqrt(u*u+v*v))
        finite = self.inside&cp.isfinite(wind)&cp.isfinite(east)&cp.isfinite(north)
        self.maximum = cp.maximum(self.maximum, cp.where(finite, wind, 0))
        self.coverage &= finite
        onshore = self.coast&finite&(east*self.normal[0]+north*self.normal[1] >= self.request.get('onshore_threshold_m_s', 10.))
        self.onset = cp.where(onshore&cp.isnan(self.onset), seconds, self.onset)
        if self.last_seconds is not None:
            self.duration += onshore*(seconds-self.last_seconds)
        self.last_seconds = seconds
        self.samples += 1

    def finish(self, directory):
        cp = self.cp
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {'latitude': self.lat, 'longitude': self.lon,
            'time_max_wind10': cp.where(self.coverage, self.maximum, cp.nan),
            'onshore_onset': cp.where(self.coverage&self.coast, self.onset, cp.nan),
            'onshore_duration': cp.where(self.coverage&self.coast, self.duration/3600, cp.nan)}
        for name, array in arrays.items():
            cp.asnumpy(array.astype(cp.float32)).tofile(directory/f'{name}.f32')
        return {'directory': str(directory), 'shape': list(self.lat.shape), 'samples': self.samples,
            'sampling': 'every committed fine-domain step, including restored initial state',
            'coverage': 'finite at every step; no denominator renormalization',
            'duration': 'right endpoint integration; hours; onshore component threshold in request'}


def render(root, receipt, request):
    from gpuwm import rustwx
    root = Path(root)
    manifest = root/'cyclone-products.json'
    payload = {**request, 'members': receipt['completed'], 'out': str(root/'maps'),
        'issue_time': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    manifest.write_text(json.dumps(payload, indent=2)+'\n')
    if request.get('defer_render', False):
        return {'status': 'deferred', 'request': str(manifest),
                'reason': 'member diagnostics retained for a combined ensemble render'}
    renderer = rustwx.find_renderer()
    if renderer is None:
        raise RuntimeError('cyclone products require rw_wrfbatch from this source tree')
    result = rustwx.run_cyclone_product_renderer(renderer, manifest)
    from gpuwm.ensemble.cyclone_charts import intensity_chart
    result['intensity_chart'] = str(intensity_chart(root/'maps',
        experiment_label=request['experiment_label']))
    return result
