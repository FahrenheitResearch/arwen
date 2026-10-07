"""Grade CPU execution of mono CUDA source against WRF and NumPy.

Run from the export root with PYTHONPATH=. and GPUWM_NO_LOCAL_GPU=1.
No device is imported or opened. Tendencies use WRF's outputs; intermediate
extrema and scales use the separately transcribed float32 NumPy reference.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from gpuwm.core.chem_advect_mono import np_advect_mono
from gpuwm.verify.chem_oracle import ORACLE_ROOT, load, ulp_table


def main():
    reference = load(ORACLE_ROOT / "core" / "advect_mono")
    actual = load(sys.argv[1])
    if set(actual)!=set(reference): raise ValueError("case inventory differs")
    results = {}
    for case,c in reference.items():
        vol = lambda n: np.ascontiguousarray(c[n].transpose(1,2,0))
        plane = lambda n: np.ascontiguousarray(c[n].T)
        coord = SimpleNamespace(**{n:c[src] for n,src in (("c1h","c1"),("c2h","c2"),
            ("rdnw","rd"),("fnm","fnm"),("fnp","fnp"))})
        p = np_advect_mono(vol("q"),vol("q0"),vol("ru"),vol("rv"),vol("ww"),
            plane("mut"),plane("mu0"),coord,c["dx"],c["dy"],c["dt"],
            tend=vol("initial_tendency"),msft=plane("mx"),msfty=plane("my"),
            mub=plane("mub"),rw_implicit=vol("wi"),v_order=int(c["vorder"]),
            open_x=bool(c["boundary"]),open_y=bool(c["boundary"]),
            boundary="open" if c["boundary"]==2 else "specified")
        tables = {}
        for name in ("tendency","h_tendency","z_tendency","qmin","qmax","scale_in","scale_out"):
            expected = c[name] if name in c else p[name].transpose(2,0,1)
            tables[name] = ulp_table(actual[case][name],expected)
            tables[name]["n_word_diff"] = int(np.count_nonzero(actual[case][name].view(np.uint32)!=expected.copy().view(np.uint32)))
        results[case] = tables
    Path(sys.argv[2]).write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    for name in next(iter(results.values())):
        print(name,"max_ulp",max(t[name]["max_ulp"] for t in results.values()),
              "n_nonzero",sum(t[name]["n_nonzero"] for t in results.values()),
              "n",sum(t[name]["n"] for t in results.values()),
              "n_word_diff",sum(t[name]["n_word_diff"] for t in results.values()))
    assert all(t["max_ulp"]==0 and t["n_word_diff"]==0 for case in results.values() for t in case.values())


if __name__ == "__main__": main()
