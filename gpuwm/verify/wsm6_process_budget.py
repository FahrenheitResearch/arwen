"""Read-only, post-limiter WSM6 process ledger for isolated research probes.

No production import or default switch. The stock source is never edited.
One column thread owns every ledger entry; no atomics or synchronization.
Water is rho*dz weighted kg/m2, temperature is rho*dz weighted K kg/m2.
Rate entries integrate the FINAL limited rate, not its unlimited proposal.
SPDX-License-Identifier: Apache-2.0
"""
from contextlib import contextmanager
import hashlib

STAGES = ('input_clip', 'sedimentation', 'snow_graupel_melt',
          'instant_freeze_melt', 'limited_conversion', 'cloud_condensation',
          'small_value_padding', 'theta_roundtrip')
FIELDS = ('temperature', 'qv', 'qc', 'qi', 'qr', 'qs', 'qg')
RATES = ('prevp', 'psdep', 'pgdep', 'praut', 'psaut', 'pgaut', 'pracw',
         'praci', 'piacr', 'psaci', 'psacw', 'pracs', 'psacr', 'pgacw',
         'paacw', 'pgaci', 'pgacr', 'pgacs', 'pigen', 'pidep',
         'pseml', 'pgeml', 'psevp', 'pgevp')
FALL_SPEEDS = ('rain', 'snow_intrinsic', 'graupel_intrinsic',
               'snow_shared', 'graupel_shared')
LABELS = (tuple(s + ':' + f for s in STAGES for f in FIELDS) + RATES
          + tuple('fall:' + s + ':' + x for s in FALL_SPEEDS
                  for x in ('mass_time_speed', 'mass_time')))
NBAND = 4


def instrument_source(source):
    """Fail closed if any audited insertion point has moved or is ambiguous."""
    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise ValueError('WSM6 budget anchor changed: ' + old[:90])
        source = source.replace(old, new)

    helper = r'''
__device__ __forceinline__ void budget_add(double *b, const unsigned char *band,
    int term, int k, int stride, int col, double weight, double change) {
    int bin = band[k*stride+col];
    b[(term*4+bin)*stride+col] += weight*change;
}
__device__ __forceinline__ void budget_stage(double *b, const unsigned char *band,
    int stage, int k, int stride, int col, double weight, float *prev,
    float t, float q, float qc, float qi, float qr, float qs, float qg) {
    float now[7] = {t,q,qc,qi,qr,qs,qg};
    for (int v=0;v<7;++v) {
        budget_add(b,band,stage*7+v,k,stride,col,weight,(double)now[v]-prev[v]);
        prev[v]=now[v];
    }
}
'''
    replace('__device__ void wsm6_column_impl(', helper + '\n__device__ void wsm6_column_impl(')
    # Follow the production launcher's tiled global workspace arguments.
    replace('float delt, int nz, int stride, int col,\n    float *wsb, int ws)',
            'float delt, int nz, int stride, int col,\n    float *wsb, int ws, '
            'double *budget, const unsigned char *band)')
    replace('float delt, int nz, int ny, int nx,\n    float *ws, int col0)',
            'float delt, int nz, int ny, int nx,\n    float *ws, int col0, '
            'double *budget, const unsigned char *band)')
    replace('wsb, (int)blockDim.x);',
            'wsb, (int)blockDim.x, budget, band);')
    replace('        const float qq = wmax(q[k], W_QMIN);', '''
        float bp[7] = {t[k],qv[id],qc[id],qi[id],qr[id],qs[id],qg[id]};
        budget_stage(budget,band,0,k,stride,col,(double)den[id]*dz[id],bp,
                     t[k],q[k],qcl[k],qil[k],qrl[k],qsl[k],qgl[k]);
        const float qq = wmax(q[k], W_QMIN);''')
    for species, arr, rq in [('qr', 'qrl', 'rq1'), ('qs', 'qsl', 'rq2'), ('qg', 'qgl', 'rq3')]:
        old = f'            {arr}[k] = wmax({rq}[k] / denk, 0.0f);'
        idx = FIELDS.index(species)
        replace(old, f'''            budget_add(budget,band,{7+idx},k,stride,col,
                (double)denk*dzc[k*stride],(double)wmax({rq}[k]/denk,0.0f)-{arr}[k]);
''' + old)
    replace('''        for (int k = 0; k < nz; ++k)
            qil[k] = wmax(rq1[k] / denc[k * stride], 0.0f);''', '''
        for (int k = 0; k < nz; ++k) {
            budget_add(budget,band,10,k,stride,col,
                (double)denc[k*stride]*dzc[k*stride],
                (double)wmax(rq1[k]/denc[k*stride],0.0f)-qil[k]);
            qil[k] = wmax(rq1[k] / denc[k * stride], 0.0f);
        }''')
    replace('            const float cpmk = cpm[k], xlk = xl[k];', '''
            float bp[7] = {tk,qk,qck,qik,qrk,qsk,qgk};
            const double bw = (double)denk*dzc[k*stride];
            const float cpmk = cpm[k], xlk = xl[k];''')
    def checkpoint(stage):
        return f'''            budget_stage(budget,band,{stage},k,stride,col,bw,bp,
                         tk,qk,qck,qik,qrk,qsk,qgk);
'''
    for stage, anchor in [
        (2, '            // :777-829 instantaneous melting/freezing'),
        (3, '            // :835-856 slopes for the process rates'),
        (4, '            // :1392-1437 condensation/evaporation'),
        (5, '            // :1443-1448 padding for small values.'),
        (6, '            t[k] = tk; q[k] = qk;')]:
        replace(anchor, checkpoint(stage) + anchor)
    # Both temperature branches: after ALL resource limiters, before application.
    for anchor in ['                    const float work2u = -(prevp + psdep + pgdep + pigen + pidep);',
                   '                    const float work2u = -(prevp + psevp + pgevp);']:
        rates = '\n'.join(f'                    budget_add(budget,band,{56+i},k,stride,col,bw,(double){r}*dtcld);'
                          for i, r in enumerate(RATES))
        replace(anchor, rates + '\n' + anchor)
    replace('        const float tt = th * pii[id];', '''
        const float tt = th * pii[id];
        budget_add(budget,band,49,k,stride,col,(double)den[id]*dz[id],(double)tt-t[k]);''')
    # Departure speeds only: the semi-Lagrangian remap subsequently averages
    # departure/arrival speeds and limits face divergence. Actual net fallout
    # is separately measured by the sedimentation state-change ledger.
    speed_terms = [('qrl[k]', 'v1[k]'), ('qsl[k]', 'ss_.vt'),
                   ('qgl[k]', 'sg_.vt'), ('qsl[k]', 'v2[k]'),
                   ('qgl[k]', 'v2[k]')]
    fall = ''
    for i, (mass, speed) in enumerate(speed_terms):
        fall += f'''
            budget_add(budget,band,{80+2*i},k,stride,col,
                (double)denk*dzc[k*stride],(double){mass}*dtcld*{speed});
            budget_add(budget,band,{81+2*i},k,stride,col,
                (double)denk*dzc[k*stride],(double){mass}*dtcld);
'''
    replace('            v1[k] = workr;', '            v1[k] = workr;' + fall)
    return source


class ProcessLedger:
    """Explicit research observer; allocate once, clear between output intervals.

    bands: contiguous uint8 (nz,ny,nx), each in [0,3], supplied on device.
    It must describe fixed layer membership for telescoping interval budgets.
    The caller owns validation/output boundaries; this class adds no device sync.
    """
    def __init__(self, bands):
        import cupy as cp
        from gpuwm.core.kernels import module_source, module_options
        from gpuwm.core.wsm6_constants import wsm6_level_tier
        from gpuwm import wrf_exact
        if wrf_exact.ENABLED:
            raise ValueError('Research process observation is unavailable in WRF-exact mode')
        if bands.dtype != cp.uint8 or bands.ndim != 3 or not bands.flags.c_contiguous:
            raise ValueError('bands must be contiguous uint8 (nz,ny,nx)')
        self.bands = bands
        self.shape = bands.shape
        self.values = cp.zeros((len(LABELS), NBAND) + bands.shape[1:], cp.float64)
        source = module_source('wsm6')
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        source = '#define WSM6_KMAX %d\n' % wsm6_level_tier(bands.shape[0]) + instrument_source(source)
        self.instrumented_sha256 = hashlib.sha256(source.encode()).hexdigest()
        options = module_options('wsm6')
        self.module = cp.RawModule(code=source, options=options)
        # The manifest records every image this process compiles
        # (tests/test_kernel_manifest.py audits each RawModule site); the
        # instrumented WSM6 column is recorded under its own key, beside
        # the production wsm6 image it observes.
        from gpuwm.certify.kernel_manifest import record_module
        record_module('wsm6_process_budget', source=source, options=options,
                      module=self.module)
        self.kernel = self.module.get_function('wsm6_column')

    @contextmanager
    def observe(self):
        """Process-local WSM6 launcher hook; never use from concurrent host threads."""
        from gpuwm.core import wsm6
        old = wsm6.wsm6_kernel
        def select(func, nz, conservative_surface=False):
            if func != 'wsm6_column' or nz != self.shape[0] or conservative_surface:
                raise ValueError('Observer requires its original WSM6 shape and surface convention')
            def launch(grid, block, args):
                if tuple(args[0].shape) != self.shape:
                    raise ValueError('Observer horizontal shape changed')
                self.kernel(grid, block, args + (self.values, self.bands))
            return launch
        wsm6.wsm6_kernel = select
        try:
            yield self
        finally:
            wsm6.wsm6_kernel = old
