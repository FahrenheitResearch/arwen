"""Prepare portable GPU-test tables using only the Rust NetCDF decoder."""
from pathlib import Path
import sys
import numpy as np
from gpuwm.core.chem_rrtmgp_aerosol import load_tables
root = str(Path(sys.argv[1]).resolve())
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for kind in ('sw', 'lw'):
    t = load_tables(kind, root)
    np.savez(out / f'packed-{kind}.npz', packed=t.packed, limits=t.limits,
             rh=t.rh, bands=t.bands)
