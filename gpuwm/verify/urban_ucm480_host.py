"""Call the production 4.8 numerical listing through its native argument ABI."""
import ctypes
import json
from pathlib import Path
import numpy as np
from gpuwm.core.urban_ucm import (
    UCM480_LOCAL_STATE, UCM_COLUMN_OUTPUTS, _column_arrays480, water_balance_params,
)


class Host480Twin:
    def __init__(self,library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.call = self.library.urban480_host
        self.call.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int]
        self.call.restype = None
        self.schema = json.loads((Path(__file__).resolve().parents[1]/"data/urban/oracle/ucm480/schema.json").read_text())["arguments"]

    def run_columns(self,params,inputs,state,*,generation=None):
        params = water_balance_params(params,generation)
        n = len(inputs["ta"])
        values = dict(_column_arrays480(params, inputs, state, self.schema, copy=True))
        pointers = np.asarray([a.ctypes.data for a in values.values()],np.uint64)
        codes = np.zeros(n,np.int32)
        self.call(pointers.ctypes.data,params.table.ctypes.data,params.globals_.ctypes.data,
                  params.switches.ctypes.data,codes.ctypes.data,n)
        return ({key:values[key] for key in UCM_COLUMN_OUTPUTS},
                {local:values[key] for key,local in UCM480_LOCAL_STATE.items() if key in values},codes)
