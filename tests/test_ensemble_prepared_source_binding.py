"""The portable wrapper and raw source manifest are separate authorities."""
import hashlib
import json

import numpy as np
import pytest

from gpuwm import prepared_single_domain_forecast as runner
from gpuwm.ensemble.physical_store import physical_static_identity, BINDING_SCHEMA, SCHEMA
from physical_field_fixtures import analytic_field_contract


def test_physical_binding_uses_the_wrapped_raw_manifest_and_still_validates_proof():
    source = {"source_sha256": {name:"1"*64 for name in runner._HRRR_DECODE_SOURCES},
              "source_cycle":"2024-01-01T00:00:00", "model_start_time":"2024-01-01T00:00:00",
              "source_forecast_hours":[0,1], "model_forcing_hours":[0,1]}
    grid={"mass_shape":[2,3]}
    static=physical_static_identity({"HGT_M":np.zeros((2,3))})
    contract=analytic_field_contract(grid)
    document={"schema":SCHEMA,"source":dict(source,input_manifest_sha256="a"*64,static_identity=static),
              "grid":grid,"field_contract":contract,"frames":[{
                  "file":"frame.nc","sha256":"f"*64,"bytes":1,"valid_time":"2024-01-01T00:00:00",
                  "arrays":{"levels_hpa":{"shape":[5],"dtype":"<f8"},
                            "field__PRES":{"shape":[5,2,3],"dtype":"<f4"}},"metadata":{}}]}
    digest=hashlib.sha256((json.dumps(document,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()
    binding={"schema":BINDING_SCHEMA,"manifest_sha256":digest,"manifest":document}
    identity=dict(source,ensemble_physical_input=binding)
    files={"source_manifest":{"sha256":"a"*64}}
    assert runner._validate_source_identity("hrrr",identity,"b"*64,files,source,
                                            layout=runner.HRRR_DIRECT_LAYOUT)==identity
    with pytest.raises(ValueError,match="base source differs"):
        runner._validate_source_identity("hrrr",identity,"a"*64,
            {"source_manifest":{"sha256":"c"*64}},source,layout=runner.HRRR_DIRECT_LAYOUT)
    with pytest.raises(ValueError,match="source_cycle differs"):
        runner._validate_source_identity("hrrr",identity,"b"*64,files,
            dict(source,source_cycle="2024-01-02T00:00:00"),layout=runner.HRRR_DIRECT_LAYOUT)
