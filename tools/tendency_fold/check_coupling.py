"""Check actual raw CU words through the rounded mass-coupling expression."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'wrf_exact_localize'))
from cmpphys import compare, supported_scalar, wrf_array


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('wrf', 'woof'), required=True)
    p.add_argument('--dump', type=Path, required=True)
    p.add_argument('--input', type=Path)
    p.add_argument('--shape', type=int, nargs=3, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    nz, ny, nx = a.shape
    if a.arm == 'wrf':
        if a.input is None:
            p.error('--input is required for the WRF arm')
        from gpuwm import netcdf_bridge
        with netcdf_bridge.open_dataset(a.input) as ds:
            ds.set_auto_maskandscale(False)
            c1 = np.asarray(ds.variables['C1H'][:], np.float32).reshape(-1)[:nz]
            c2 = np.asarray(ds.variables['C2H'][:], np.float32).reshape(-1)[:nz]
        mut = wrf_array(a.dump, 'b_prep', 'mut', (ny, nx))
        raw = wrf_array(a.dump, 'fold_raw_cumulus', 'RTHCUTEN', (nz, ny, nx))
        coupled = wrf_array(a.dump, 'p40_mass_coupled', 'rthcuten', (nz, ny, nx))
        coupled = supported_scalar(coupled, specified=True)
    else:
        with np.load(a.dump / 'mass_inputs.npz') as z:
            c1, c2, mut = (z[k].copy() for k in ('c1h', 'c2h', 'mut'))
        with np.load(a.dump / 'raw_cumulus.npz') as z:
            raw = z['rtheta'].copy()
        with np.load(a.dump / 'components.npz') as z:
            coupled = z['cumulus_rtheta'].copy()
    chm = c1[:, None, None] * mut[None]
    chm += c2[:, None, None]
    reconstructed = supported_scalar(chm * raw, specified=True)
    result = dict(arm=a.arm, expression='float32(float32(c1*mut)+c2)*raw',
                  comparison=compare(coupled, reconstructed),
                  max_abs_diff=float(np.max(np.abs(coupled.astype(np.float64) - reconstructed))))
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return int(bool(result['comparison']['different_words']))


if __name__ == '__main__':
    raise SystemExit(main())
