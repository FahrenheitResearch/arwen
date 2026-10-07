"""Throwaway table for the output diagnostics; no packaged species are changed."""
import json
import shutil
from pathlib import Path
import numpy as np
from gpuwm import chem_table

# Registry/registry.chem:4022, assigned in package order by gen_scalar_indices.c:153-169.
WRF_ORDER = 'so2 sulf dms msa p25 bc1 bc2 oc1 oc2 dust_1 dust_2 dust_3 dust_4 dust_5 seas_1 seas_2 seas_3 seas_4 p10'.split()


def staged_table(tmp_path):
    root = tmp_path / 'chem'
    shutil.copytree(chem_table.CHEM_DATA_ROOT, root, ignore=shutil.ignore_patterns('oracle'))
    # Only the test's own rows: a packaged row of the same name (a lane's
    # so2 or smoke) would be a second definition, which the loader refuses,
    # and a set left with no member row is refused too.
    for p in (root / 'species').glob('*.json'):
        if p.name != 'tracer_test.json':
            p.unlink()
    kept = {s for r in json.loads((root / 'species/tracer_test.json').read_text())['rows']
            for s in r['sets']} | {'tracer_test'}
    for p in (root / 'sets').glob('*.json'):
        if json.loads(p.read_text())['name'] not in kept:
            p.unlink()
    (root / 'diagnostics').mkdir(exist_ok=True)
    # Keep the staged catalog closed as packaged diagnostics acquire rows.
    # These fixture species have no physical processes or source inputs.
    known = {r['name'] for r in json.loads((root / 'species/tracer_test.json').read_text())['rows']}
    diagnostic_species = {name for p in (root / 'diagnostics').glob('*.json')
                          for term in json.loads(p.read_text())['terms']
                          for name in term['species']}
    names = WRF_ORDER + sorted(diagnostic_species - set(WRF_ORDER) - known)
    rows = [dict(name=n, output_name=None, long_name=n, units='ug kg-1',
                 phase='aerosol', family='tracer', sets=['tracer_test'],
                 default_inflow=0., processes=[], provenance='test only')
            for n in names]
    (root / 'species/test_pm.json').write_text(json.dumps(dict(schema='gpuwm.chem.species.v1', owner='test', rows=rows)))
    return chem_table.load_sets(['tracer_test'], root=root)


def fields_for(table, shape=(4,2,3)):
    rng = np.random.default_rng(230)
    fields = {r.state_attr: rng.uniform(.01, 12, shape).astype(np.float32) for r in table.rows}
    alt = rng.uniform(.7, 2, shape).astype(np.float32)
    dz = rng.uniform(20, 600, shape).astype(np.float32)
    return fields, alt, dz
