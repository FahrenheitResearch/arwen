"""Per-word hashes of owned fold dumps; simulation arrays stay on their box."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'wrf_exact_localize'))
from cmpphys import wrf_array, supported_scalar, compare


def words(a, dictionary, identities, include_words):
    a = np.ascontiguousarray(a, dtype='<f4')
    raw = a.tobytes()
    hashes = []
    for i in range(0, len(raw) if include_words else 0, 4):
        word = raw[i:i+4]
        if word not in identities:
            identities[word] = len(dictionary)
            dictionary.append(hashlib.sha256(word).hexdigest())
        hashes.append(identities[word])
    indices = [(0, 1, 4), (0, 4, 4)] if a.ndim == 3 else []
    return dict(shape=list(a.shape), words=a.size,
                sha256=hashlib.sha256(raw).hexdigest(), word_ids=hashes,
                nonzero=int(np.count_nonzero(a)),
                samples=[dict(index=list(i), value=float(a[i]),
                              word=hex(int(a.view(np.uint32)[i])))
                         for i in indices])


def collect(arm, directory, shape):
    nz, ny, nx = shape
    fields, checks = {}, {}
    if arm == 'wrf':
        for stage in ('radiation', 'pbl', 'cumulus'):
            for name, sh in [('rt_tendf', shape), ('ru_tendf', (nz, ny, nx+1)),
                             ('rv_tendf', (nz, ny+1, nx))]:
                fields[f'after_{stage}/{name}'] = wrf_array(
                    directory, 'fold_after_' + stage, name, tuple(sh))
        fields['raw_cumulus/rtheta'] = wrf_array(
            directory, 'fold_raw_cumulus', 'RTHCUTEN', tuple(shape))
        for label, name in [('cumulus', 'rthcuten'), ('pbl', 'rthblten'),
                            ('radiation', 'rthraten')]:
            fields[f'component/{label}_rtheta'] = supported_scalar(
                wrf_array(directory, 'p40_mass_coupled', name, tuple(shape)),
                specified=True)
    else:
        with np.load(directory / 'components.npz', allow_pickle=False) as z:
            components = {k: z[k].copy() for k in z}
        # Same rounded expression as stock update_phy_ten. Independently
        # check the last prefix against the actual composer line capture.
        for name, source in [('rt_tendf', 'rtheta'), ('ru_tendf', 'ru'),
                             ('rv_tendf', 'rv')]:
            value = components['radiation_' + source].copy()
            fields[f'after_radiation/{name}'] = value.copy()
            value += components['pbl_' + source]
            fields[f'after_pbl/{name}'] = value.copy()
            value += components['cumulus_' + source]
            fields[f'after_cumulus/{name}'] = value.copy()
            with np.load(directory / 'woof_after_cumulus.npz') as z:
                checks[name] = compare(value, z[name])
        with np.load(directory / 'raw_cumulus.npz') as z:
            fields['raw_cumulus/rtheta'] = z['rtheta'].copy()
        for label in ('pbl', 'radiation', 'cumulus'):
            fields[f'component/{label}_rtheta'] = components[label + '_rtheta']
    dictionary, identities = [], {}
    per_word = {'raw_cumulus/rtheta', 'component/cumulus_rtheta',
                'after_cumulus/rt_tendf'}
    return dict(arm=arm, source=str(directory), reconstructed_final_checks=checks,
                fields={k: words(v, dictionary, identities, k in per_word)
                        for k, v in fields.items()},
                word_hash_dictionary=dictionary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['wrf', 'woof'], required=True)
    p.add_argument('--dump', type=Path, required=True)
    p.add_argument('--shape', nargs=3, type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = collect(a.arm, a.dump, a.shape)
    with gzip.open(a.out, 'wt') as f:
        json.dump(result, f, separators=(',', ':'))
    summary = {k: {n: v for n, v in row.items() if n != 'word_ids'}
               for k, row in result['fields'].items()}
    a.out.with_suffix('.summary.json').write_text(json.dumps(
        dict(arm=a.arm, fields=summary,
             reconstructed_final_checks=result['reconstructed_final_checks']), indent=2))
    print(json.dumps(dict(arm=a.arm, fields=len(summary), bytes=a.out.stat().st_size)))


if __name__ == '__main__':
    main()
