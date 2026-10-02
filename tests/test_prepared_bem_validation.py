"""Prepared counting preserves the runtime's named urban table refusals."""

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.urban_state import (prepared_urban_columns, option_spec,
                                    resolve_dimensions, urban_categories_for,
                                    urban_var_init_host)
from gpuwm.core import urban_bem
from gpuwm.core.urban_tables import load_urban_params
from test_urban_workspace_price import MODIS


@pytest.mark.parametrize("lcz,category,remedy", [
    (0, 55, "USE_WUDAPT_LCZ=1"),
    (1, 51, "USE_WUDAPT_LCZ=0"),
])
def test_prepared_count_and_cold_start_name_the_same_missing_table(
        lcz, category, remedy):
    cfg = SimpleNamespace(sf_urban_physics=3, sf_surface_physics=4,
                          use_wudapt_lcz=lcz)
    lu = np.full((2, 3), category, np.int32)
    if lcz == 0:
        with pytest.raises(ValueError, match=remedy) as counted:
            prepared_urban_columns(cfg, lu, landuse_dataset=MODIS)
    else:
        # These types fit the 11-row table. Counting must not introduce
        # the runtime's legend refusal at an earlier admission door.
        assert prepared_urban_columns(cfg, lu, landuse_dataset=MODIS) == lu.size
    one = np.ones(lu.shape, np.float32)
    soil = np.ones((4,) + lu.shape, np.float32)
    with pytest.raises(ValueError, match=remedy) as started:
        urban_var_init_host(
            option=3, use_wudapt_lcz=lcz, params=load_urban_params(3, lcz),
            categories=urban_categories_for(cfg, MODIS), ivgtyp=lu,
            tsk=one * 290, tslb=soil * 285, tmn=one * 280,
            smois=soil * .3, nz=59,
            dims=resolve_dimensions(3, urban_bem),
            spec=option_spec(3, urban_bem))
    if lcz == 0:
        assert str(counted.value) == str(started.value)
