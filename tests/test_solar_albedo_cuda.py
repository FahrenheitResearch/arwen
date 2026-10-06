"""CUDA solar albedo words against the source-pinned fork oracle."""
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_cuda_three_calls_match_compiled_fortran():
    import cupy as cp
    from gpuwm.core.solar_albedo import update_solar_albedo_cuda
    path = Path(__file__).parent / "data" / "solar_albedo_fork.npz"
    with np.load(path, allow_pickle=False) as fixture:
        fields = {key[3:]: cp.asarray(fixture[key].reshape(32, 64))
                  for key in fixture.files if key.startswith("in_")}
        for call, coszen in enumerate(fixture["coszen"]):
            if call:
                fields["albedo"][...] = cp.float32(0.123)
            update_solar_albedo_cuda(fields, cp.asarray(coszen),
                                     initialize=call == 0)
            for name in ("albsol", "albbcksol"):
                actual = cp.asnumpy(fields[name]).reshape(-1)
                expected = fixture[f"out_{name}"][call]
                np.testing.assert_array_equal(actual.view(np.uint32),
                                              expected.view(np.uint32))


def test_cuda_active_class_outside_modis_table_is_refused():
    import cupy as cp
    from gpuwm.core.solar_albedo import update_solar_albedo_cuda
    path = Path(__file__).parent / "data" / "solar_albedo_fork.npz"
    with np.load(path, allow_pickle=False) as fixture:
        fields = {key[3:]: cp.asarray(fixture[key])
                  for key in fixture.files if key.startswith("in_")}
        fields["ivgtyp"][0] = 24
        with pytest.raises(ValueError, match="MODIS IVGTYP classes 1 through 21"):
            update_solar_albedo_cuda(fields, cp.asarray(fixture["coszen"][0]))
