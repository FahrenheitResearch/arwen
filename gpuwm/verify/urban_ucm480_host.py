"""Call the production 4.8 numerical listing through its native argument ABI."""
import ctypes
import json
from pathlib import Path
import numpy as np
from gpuwm.core.urban_ucm import UCM480_LOCAL_STATE,UCM_COLUMN_OUTPUTS,water_balance_params


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
        values = {}
        for field in self.schema:
            name,layers = field["name"],field["layers"]
            kind = np.int32 if field["type"] in ("int","bool") else np.float32
            shape = (4,n) if layers else (n,)
            local = UCM480_LOCAL_STATE.get(name)
            if local is not None:
                value = state.get(local,np.zeros(shape,kind))
            elif name in ("dzr","dzb","dzg"):
                offset = {"dzr":0,"dzb":4,"dzg":8}[name]
                value = np.repeat(params.globals_[offset:offset+4,None],n,axis=1)
            elif name.startswith("num_"):
                value = np.full(n,4,kind)
            elif name == "znt":
                value = inputs["znt_in"]
            else:
                value = inputs.get(name,np.zeros(shape,kind))
            values[name] = np.array(value,dtype=kind,order="C",copy=True)
            if values[name].shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {values[name].shape}")
        pointers = np.asarray([a.ctypes.data for a in values.values()],np.uint64)
        codes = np.zeros(n,np.int32)
        self.call(pointers.ctypes.data,params.table.ctypes.data,params.globals_.ctypes.data,
                  params.switches.ctypes.data,codes.ctypes.data,n)
        return ({key:values[key] for key in UCM_COLUMN_OUTPUTS},
                {local:values[key] for key,local in UCM480_LOCAL_STATE.items() if key in values},codes)
