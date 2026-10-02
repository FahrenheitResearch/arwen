"""Record the new unit's loaded kernel attributes on the current card."""
import json

from gpuwm.certify.compile_platform import compile_platform_fingerprint
from gpuwm.core.kernels import get_kernel

print(json.dumps(dict(platform=compile_platform_fingerprint(), kernels={
    name: get_kernel("thompson_cold_start", name).attributes
    for name in ("cold_start_numbers", "cold_start_alt", "cold_start_surface", "cold_start_temperature")}), indent=2, default=str))
