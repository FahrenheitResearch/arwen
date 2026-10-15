"""Reuse qualified regime builders, adding paired cloud and BEP probes."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np

from tools.ysu_wrf461_oracle.make_wide_columns import (
    build_columns, stratocumulus, write_records, _edge_base)


def main(path):
    # Exclude invalid zero-coupling upstream 0/0 probes from this finite gate.
    records = [c.finish() for c in build_columns() if c.regime != 9]
    for nz in (4, 128):
        for dry in (False, True):
            for stable in (False, True):
                for topdown in (0, 1):
                    # The unmodified get_pblh requires a 500 m search range.
                    # A separate low-top file demonstrates and checks its
                    # declared deterministic correction, not undefined words.
                    c = _edge_base(nz=nz, dz1=200. if nz == 4 else 25., grow=0.)
                    c.topdown = topdown
                    c.br = .1 if stable else -.1
                    c.hfx = -5. if stable else 50.
                    c.qv[:] = 0. if dry else .005
                    c.theta[:] = 300. + np.arange(nz) * (1. if stable else .1)
                    records.append(c.finish())
    rng = np.random.default_rng(20261008)
    for _ in range(20):
        base = stratocumulus(rng)
        for cloud in (0., 1.e-8, 1.e-6, 1.e-5, 1.e-4, 5.e-4, .001, .003, .01):
            for topdown in (0, 1):
                c = deepcopy(base)
                c.topdown = topdown
                c.qc[c.qc > 0] = cloud
                c.rthraten = np.minimum(c.rthraten, 0)
                rec = c.finish()
                records.append(rec)
                for fraction in (1., .6):
                    urban = deepcopy(rec)
                    urban["topo"] = 2
                    urban["scalars"]["_pad"] = np.float32(fraction)
                    records.append(urban)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_records(records, path)
    write_records([r for r in records if r["nz"] != 4], path.with_name("columns-defined.bin"))
    shallow = _edge_base(nz=4, dz1=25., grow=0.)
    write_records([shallow.finish()], path.with_name("low-top-undefined.bin"))
    print(f"wrote {len(records)} columns, seed=20261008")


if __name__ == "__main__":
    main(sys.argv[1])
