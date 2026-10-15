"""The pre-fix implementation must fail the corrected source oracle."""
from pathlib import Path
import json
import sys
import numpy as np
from gpuwm.core.rrtmg_legacy import radconst, calc_coszen

def main():
    fixture = np.load(Path(sys.argv[1])/'solar_refs.npz')
    rows = []
    n = fixture['lat'].size
    for k, (j, xt, g) in enumerate(fixture['scalars']):
        d, s = radconst(j)
        # The original failing check ran before the selector existed. This
        # final tool names that same preserved legacy arithmetic explicitly.
        coszen = calc_coszen(j,xt,g,fixture['lat'],fixture['lon'],d,generation='wrf_461')
        ref = fixture['ref'][k,2:2+n].reshape(coszen.shape)
        rows.append({'case':k,'coszen_words':n,'differing_words':int(np.count_nonzero(coszen.view('u4')!=ref.view('u4')))})
    print(json.dumps(rows,indent=2))
    assert sum(row['differing_words'] for row in rows) == 0, 'default sun differs from WRF v4.8.0 calc_coszen'

if __name__ == '__main__':
    main()
