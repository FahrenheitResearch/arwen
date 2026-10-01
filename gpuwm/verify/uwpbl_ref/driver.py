"""WRF float32 column boundary for camuwpbl, bottom level at index zero.

inputs is a single column from uwpbl_oracle.step_arrays. state may override
kvm3d[_in], kvh3d[_in], tauresx2d[_in], tauresy2d[_in]. It is not mutated.
Returned REAL fields are numpy.float32 arrays/scalars; kpbl2d is int32, as
declared at module_bl_camuwpbl_driver.F:208. Arithmetic inside CAM
uses Python floats; default REAL subexpressions are rounded individually.
ftz=True emulates the product loader at the CUDA float32 boundary sites.
The default ftz=False retains the unflushed WRF oracle semantics.
"""
import math
import numpy as np

from gpuwm.core.uwpbl_constants import CPAIR, GRAVIT, LATVAP, LATICE, QMIN_Q
from .fortran import narrow, fmax, zeros
from .eddy import compute_eddy_diff


class _F32Boundary:
    """CUDA driver boundaries only; never flush binary64 scheme arithmetic."""
    def __init__(self, ftz):
        self.ftz = ftz

    def flush(self, value):
        value = float(value)
        if self.ftz and 0.0 < abs(value) < float.fromhex('0x1p-126'):
            return math.copysign(0.0, value)
        return value

    def widen(self, value):
        return self.flush(value)

    def narrow(self, value):
        return self.flush(narrow(value))

    def sub(self, a, b):
        return self.narrow(self.flush(a) - self.flush(b))

    def mul(self, a, b):
        return self.narrow(self.flush(a) * self.flush(b))

    def add(self, a, b):
        return self.narrow(self.flush(a) + self.flush(b))

    def sqrt(self, a):
        return self.narrow(math.sqrt(self.flush(a)))


def _prepare(inputs, state, ftz=False):
    """Lines 372-515, separately replayable against eddy_in."""
    boundary = _F32Boundary(ftz)
    widen, narrow = boundary.widen, boundary.narrow
    nk = len(inputs['u'])
    a = {n: zeros(nk) for n in ('u v pmid rpdel pdel z t s qrl wsedl exner '
                              'cldn pdeldry rpdeldry pmiddry').split()}
    a.update({n: zeros(nk+1) for n in ('pi zi kvq cgh cgs kvh kvm pintdry').split()})
    cloud = [None] + [zeros(nk) for _ in range(5)]
    cloudtnd = [None] + [zeros(nk) for _ in range(5)]
    first = int(inputs['itimestep']) == 1
    ht = widen(inputs['ht'])
    phis = ht * GRAVIT
    for k in range(nk):
        kflip = nk - k
        for m in range(1, 6):
            cloud[m][kflip] = -999888777.0
            cloudtnd[m][kflip] = -999888777.0
        for dest, src in (('u','u'), ('v','v'), ('pmid','p'), ('t','t'),
                          ('wsedl','wsedl3d'), ('exner','exner'), ('cldn','cldfra')):
            a[dest][kflip] = widen(inputs[src][k])
        dp = widen(boundary.sub(inputs['p8w'][k], inputs['p8w'][k+1]))
        a['pdel'][kflip] = dp
        a['rpdel'][kflip] = 1.0 / dp
        a['z'][kflip] = widen(boundary.sub(inputs['z'][k], inputs['ht']))
        a['s'][kflip] = CPAIR * a['t'][kflip] + GRAVIT * a['z'][kflip] + phis
        a['qrl'][kflip] = widen(boundary.mul(inputs['rthratenlw'][k], inputs['exner'][k])) * CPAIR * dp
        multFrc = 1.0 / (1.0 + widen(inputs['qv'][k]))
        cloud[1][kflip] = fmax(widen(inputs['qv'][k]) * multFrc, 1.e-30)
        for m, src in enumerate(('qc', 'qi', 'qnc', 'qni'), 2):
            cloud[m][kflip] = widen(inputs[src][k]) * multFrc
        a['pdeldry'][kflip] = a['pdel'][kflip] * (1.0 - cloud[1][kflip])
        a['rpdeldry'][kflip] = 1.0 / a['pdeldry'][kflip]
    def saved(name):
        return state[name] if name in state else state.get(name+'_in', inputs[name+'_in'])
    for k in range(nk+1):
        kflip = nk - k + 1
        a['pi'][kflip] = widen(inputs['p8w'][k])
        a['zi'][kflip] = widen(boundary.sub(inputs['z_at_w'][k], inputs['ht']))
        a['kvh'][kflip] = narrow(0.0) if first else widen(saved('kvh3d')[k])
        a['kvm'][kflip] = narrow(0.0) if first else widen(saved('kvm3d')[k])
    a['pintdry'][1] = a['pi'][1]
    for k in range(1, nk+1):
        a['pintdry'][k+1] = a['pintdry'][k] + a['pdeldry'][k]
        a['pmiddry'][k] = (a['pintdry'][k+1] + a['pintdry'][k]) * 0.5
    # sqrt(REAL) is evaluated before widening to uMean (real(r8)).
    u0, v0 = widen(inputs['u'][0]), widen(inputs['v'][0])
    uMean = widen(boundary.sqrt(boundary.add(
        boundary.mul(inputs['u'][0], inputs['u'][0]),
        boundary.mul(inputs['v'][0], inputs['v'][0]))))
    tauFac = widen(boundary.mul(boundary.mul(inputs['rho'][0], inputs['ust']),
                                 inputs['ust'])) / uMean
    a['taux'], a['tauy'] = [None, -(tauFac*u0)], [None, -(tauFac*v0)]
    a['shflx'], a['cflx'] = [None, widen(inputs['hfx'])], [None, widen(inputs['qfx']), 0.0, 0.0, 0.0, 0.0]
    a['tauresx'] = [None, narrow(0.0) if first else widen(saved('tauresx2d'))]
    a['tauresy'] = [None, narrow(0.0) if first else widen(saved('tauresy2d'))]
    a['cloud'], a['cloudtnd'] = cloud, cloudtnd
    return a


def camuwpbl_column(inputs: dict, state: dict, ftz=False) -> dict:
    from .vdiff import compute_vdiff
    from .wvsat import aqsat
    boundary = _F32Boundary(ftz)
    widen, narrow = boundary.widen, boundary.narrow
    nk = len(inputs['u'])
    ztodt = widen(inputs['dt'])
    rztodt = 1.0 / ztodt
    a = _prepare(inputs, state, ftz=ftz)
    for n in ('ustar pblh ipbl kpblh wstarPBL tpert qpert wpert').split():
        a[n] = zeros(1)
    for n in ('tke bprod sprod sfi turbtype smaw').split():
        a[n] = zeros(nk+1)
    cloud = a['cloud']
    compute_eddy_diff(nk, a['t'], cloud[1], ztodt, cloud[2], cloud[3], a['s'],
        a['rpdel'], a['cldn'], a['qrl'], a['wsedl'], a['z'], a['zi'], a['pmid'],
        a['pi'], a['u'], a['v'], a['taux'], a['tauy'], a['shflx'],
        [None, a['cflx'][1]], True, 5, a['ustar'], a['pblh'], a['kvm'].copy(),
        a['kvh'].copy(), a['kvm'], a['kvh'], a['kvq'], a['cgh'], a['cgs'],
        a['tpert'], a['qpert'], a['wpert'], a['tke'], a['bprod'], a['sprod'],
        a['sfi'], int(inputs['itimestep']) == 1, a['tauresx'], a['tauresy'],
        zeros(1), a['ipbl'], a['kpblh'], a['wstarPBL'], a['turbtype'], a['smaw'])
    cloudtnd = a['cloudtnd']
    for m in range(1, 6):
        cloudtnd[m][1:] = cloud[m][1:]
    stnd, utnd, vtnd = (a[n].copy() for n in ('s', 'u', 'v'))
    sl_pre, qt_pre = zeros(nk), zeros(nk)
    for k in range(1, nk+1):
        sl_pre[k] = stnd[k] - LATVAP*cloudtnd[2][k] - (LATVAP+LATICE)*cloudtnd[3][k]
        qt_pre[k] = cloudtnd[1][k] + cloudtnd[2][k] + cloudtnd[3][k]
    tem2, ftem = zeros(nk), zeros(nk)
    aqsat(a['t'], a['pmid'], tem2, ftem, 1, nk)
    ftem_pre = [None] + [cloud[1][k]/ftem[k]*100.0 for k in range(1, nk+1)]
    tautmsx, tautmsy, topflx, dtk = zeros(1), zeros(1), zeros(1), zeros(nk)
    errstring = compute_vdiff(nk, 5, a['pmid'], a['pi'], a['rpdel'], a['t'],
        ztodt, a['taux'], a['tauy'], a['shflx'], a['cflx'], 1, nk, a['kvh'],
        a['kvm'], a['kvq'], a['cgs'], a['cgh'], a['zi'], zeros(1),
        [None, QMIN_Q, 0.0, 0.0, 0.0, 0.0],
        {'u': True, 'v': True, 's': True, 'q': [None]+[True]*5},
        utnd, vtnd, cloudtnd, stnd, tautmsx, tautmsy, dtk, topflx,
        a['tauresx'], a['tauresy'], 1)
    if errstring:
        raise RuntimeError(errstring)
    slten, qtten, taft = zeros(nk), zeros(nk), zeros(nk)
    for k in range(1, nk+1):
        sl = stnd[k] - LATVAP*cloudtnd[2][k] - (LATVAP+LATICE)*cloudtnd[3][k]
        qt = cloudtnd[1][k] + cloudtnd[2][k] + cloudtnd[3][k]
        slten[k], qtten[k] = (sl-sl_pre[k])*rztodt, (qt-qt_pre[k])*rztodt
        stnd[k] = (stnd[k]-a['s'][k])*rztodt
        utnd[k], vtnd[k] = (utnd[k]-a['u'][k])*rztodt, (vtnd[k]-a['v'][k])*rztodt
        for m in range(1, 6):
            cloudtnd[m][k] = (cloudtnd[m][k]-cloud[m][k])*rztodt
        saft = a['s'][k] + stnd[k]*ztodt
        taft[k] = (saft-GRAVIT*a['z'][k])/CPAIR
        # Remaining diagnostic reconstruction has no effect on returned WRF fields.
        qaft = [None] + [cloud[m][k]+cloudtnd[m][k]*ztodt for m in range(1, 4)]
        uaft, vaft = a['u'][k]+utnd[k]*ztodt, a['v'][k]+vtnd[k]*ztodt
    aqsat(taft, a['pmid'], tem2, ftem, 1, nk)
    for k in range(1, nk+1):
        ftem_aft = (cloud[1][k]+cloudtnd[1][k]*ztodt)/ftem[k]*100.0
        tten, rhten = (taft[k]-a['t'][k])*rztodt, (ftem_aft-ftem_pre[k])*rztodt
    out = {n: np.empty(nk, np.float32) for n in ('rublten rvblten rthblten rqvblten rqcblten rqiblten rqniblten').split()}
    for k in range(nk):
        kflip = nk-k
        out['rublten'][k], out['rvblten'][k] = narrow(utnd[kflip]), narrow(vtnd[kflip])
        out['rthblten'][k] = narrow(stnd[kflip]/CPAIR/a['exner'][kflip])
        multFrc = 1.0 + widen(inputs['qv'][k])
        out['rqvblten'][k] = narrow(cloudtnd[1][kflip]*multFrc*multFrc)
        for n, m in (('rqcblten',2), ('rqiblten',3), ('rqniblten',5)):
            out[n][k] = narrow(cloudtnd[m][kflip]*multFrc)
    for n, src in (('kvm3d','kvm'), ('kvh3d','kvh'), ('tke_pbl','tke'), ('turbtype3d','turbtype'), ('smaw3d','smaw')):
        out[n] = np.asarray([narrow(a[src][nk-k+1]) for k in range(nk+1)], dtype=np.float32)
    out['kpbl2d'] = np.int32(nk-int(a['kpblh'][1])+1)
    for n, src in (('tauresx2d','tauresx'), ('tauresy2d','tauresy'),
                   ('pblh2d','pblh'), ('tpert2d','tpert'), ('qpert2d','qpert'), ('wpert2d','wpert')):
        out[n] = np.float32(narrow(a[src][1]))
    return out


def camuwpbl_step(step_arrays_dict: dict, ftz=False) -> dict:
    ncol = len(step_arrays_dict['u'])
    columns = []
    for col in range(ncol):
        inputs = {n: v if n in ('dt', 'itimestep') else v[col] for n, v in step_arrays_dict.items()}
        columns.append(camuwpbl_column(inputs, {}, ftz=ftz))
    return {n: np.stack([c[n] for c in columns]) for n in columns[0]}
