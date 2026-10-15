"""Full numeric-field hashes from Rust-decoded real-run histories."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from gpuwm.netcdf_bridge import Dataset


def hashes(root):
    rows = {}
    for path in sorted(root.rglob('wrfout*')):
        if not path.is_file() or path.suffix in ('.json','.log'):
            continue
        with Dataset(path) as ds:
            fields = {}
            for name, var in ds.variables.items():
                if np.dtype(var.dtype).kind not in 'fiu':
                    continue
                raw = np.ascontiguousarray(var[:])
                fields[name] = {'shape':list(raw.shape),'dtype':str(raw.dtype),'words':raw.size,'sha256':hashlib.sha256(raw.tobytes()).hexdigest()}
            rows[path.name] = fields
    assert rows, f'no numeric histories in {root}'
    return rows


if __name__ == '__main__':
    a,b,out=map(Path,sys.argv[1:])
    left,right=hashes(a),hashes(b)
    differences = [f'{frame}/{field}' for frame in sorted(set(left)|set(right)) for field in sorted(set(left.get(frame,{}))|set(right.get(frame,{}))) if left.get(frame,{}).get(field)!=right.get(frame,{}).get(field)]
    report={'left':left,'right':right,'different_fields':differences,'field_comparisons':sum(len(fields) for fields in left.values())}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(report['field_comparisons'],'full-field comparisons;',len(differences),'differ')
    assert not differences, differences
