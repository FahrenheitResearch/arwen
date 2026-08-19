"""Drive the lane-2 seam through the REAL built cdylib via the real
ctypes bridge (verify-against-the-artifact probe; also usable after
integration)."""
import ctypes
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parents[4]))

from gpuwm.static import rust_bridge  # noqa: E402

reason = rust_bridge.unavailable_reason()
print("bridge unavailable_reason:", reason)
lib = rust_bridge.load()
print("abi:", lib.gpuwm_static_abi_version())
print("marker present:", hasattr(lib, "gpuwm_static_build_fields"))

fields = ("terrain", "landuse", "soil_top", "soil_bottom", "greenfrac",
          "lai", "albedo", "snow_albedo", "soil_temperature")
paths = {name: "Z:/absent" for name in fields}
try:
    rust_bridge.build_fields(999, paths)
    print("bogus grid: unexpectedly succeeded")
except RuntimeError as err:
    print("bogus grid refusal:", err)
print("len(bogus):", lib.gpuwm_static_fieldset_len(ctypes.c_uint64(7)),
      "err:", rust_bridge.last_error(lib))
