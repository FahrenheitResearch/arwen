"""Three real PBL column steps using the calm urban surface outputs."""
import numpy as np
from _urban_pbl_seed import YSU, YSU_DT, MYNN


def follow_calm_surface(ust, hfx, qfx):
    import cupy as cp
    from gpuwm.core.ysu import launch_ysu
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda

    def inputs(seed):
        return {k: np.asarray(words, np.int32 if k == 'kpbl' else np.float32).reshape(shape)
                for k, (shape, words) in seed.items()}

    values = inputs(YSU)
    values['u'].fill(0)
    values['v'].fill(0)
    for name in ('wspd', 'u10', 'v10'):
        values[name].fill(0)
    for k, x in [('ust', ust), ('hfx', hfx), ('qfx', qfx)]:
        values[k].fill(x)
    for step in range(3):
        dev = {k: cp.asarray(v) for k, v in values.items()}
        raw = launch_ysu(
            *(dev[k] for k in ('u', 'v', 'theta', 'qv', 'qc', 'qi', 'p',
                              'p_interface', 'exner', 'dz', 'rthraten')),
            **{k: dev[k] for k in ('psfc', 'znt', 'ust', 'hfx', 'qfx', 'wspd',
                                   'br', 'psim', 'psih', 'xland', 'u10', 'v10')},
            dt=YSU_DT, ysu_topdown_pblmix=0)
        out = {k: cp.asnumpy(v) for k, v in raw.items()}
        assert all(np.isfinite(a).all() for a in out.values()), ('ysu', step)
        for k, tend in [('u', 'du'), ('v', 'dv'), ('theta', 'dtheta'),
                        ('qv', 'dqv'), ('qc', 'dqc'), ('qi', 'dqi')]:
            values[k] += np.float32(YSU_DT) * out[tend]

    values = inputs(MYNN)
    values['u'].fill(0)
    values['v'].fill(0)
    # Resolved winds remain zero. MYNN's surface speed diagnostic must be
    # positive: its tendency guard prevents rho*ust**2/wspd dividing by zero.
    values['wspd'].fill(0.1)
    for k, x in [('ust', ust), ('hfx', hfx), ('qfx', qfx)]:
        values[k].fill(x)
    for step in range(3):
        raw = mynn_bl_driver_cuda(
            {k: cp.asarray(np.ascontiguousarray(v)) for k, v in values.items()},
            bl_mynn_mixlength=1, flag_qs=True, delt=np.float32(20),
            initflag=int(step == 0))
        out = {k: cp.asnumpy(v) for k, v in raw.items()}
        assert all(np.isfinite(a).all() for a in out.values()), ('mynn', step)
        for field, tend, floor in [('u', 'rublten', False), ('v', 'rvblten', False),
                                  ('th', 'rthblten', False), ('sqv', 'rqvblten', True),
                                  ('sqc', 'rqcblten', True), ('sqi', 'rqiblten', True),
                                  ('sqs', 'rqsblten', True)]:
            updated = values[field] + np.float32(20) * out[tend]
            values[field] = np.maximum(updated, np.float32(0)) if floor else updated
        for field in ('qke', 'tsq', 'qsq', 'cov', 'el', 'sh', 'sm',
                      'qc_bl', 'qi_bl', 'cldfra_bl', 'pblh', 'kpbl', 'rmol'):
            values[field] = out[field].copy()
        values['tk'] = values['th'] * values['exner']
        values['rho'] = values['p'] / (np.float32(287) * values['tk'])
