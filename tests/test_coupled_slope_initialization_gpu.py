"""The forecast slope initializer on the captured full terrain plane."""
import hashlib
import os
from pathlib import Path
import numpy as np
import pytest
cp=pytest.importorskip("cupy")

from gpuwm.core.topo_radiation import slope_geometry


def test_scalar_slope_initializer_matches_the_complete_stock_plane():
    directory=os.environ.get("GPUWM_COUPLED_SLOPE_CAPTURE")
    native=os.environ.get("GPUWM_COUPLED_SLOPE_INPUT")
    expected=os.environ.get("GPUWM_COUPLED_SLOPE_SHA256")
    if not (directory or native) or not expected:
        pytest.skip("real native input or captured terrain and stock hash are required")
    if native:
        from gpuwm.verify_exact.replay import read_frame
        fields=read_frame(Path(native))
        args=[cp.asarray(fields[n]) for n in
              ("HGT","MAPFAC_M","MAPFAC_M","SINALPHA","COSALPHA")]
    else:
        step=int(os.environ.get("GPUWM_COUPLED_SLOPE_STEP","5"))
        paths=list(Path(directory).glob(f"*_s{step:03d}_*_p20_noah_in.npz"))
        assert len(paths)==1
        with np.load(paths[0],allow_pickle=False) as z:
            args=[cp.asarray(z[n]) for n in ("ht","msft","msft","sina","cosa")]
    _,old=slope_geometry(*args,dx=3000.,dy=3000.)
    assert hashlib.sha256(cp.asnumpy(old).astype('<f4').tobytes()).hexdigest()!=expected
    _,actual=slope_geometry(*args,dx=3000.,dy=3000.,initialization="scalar_build")
    assert hashlib.sha256(cp.asnumpy(actual).astype('<f4').tobytes()).hexdigest()==expected
