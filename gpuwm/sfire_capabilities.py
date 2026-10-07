"""Read-only installed SFIRE contract probe for worker admission.

This declares executable interfaces, not observational forecast skill or an
installed GPU. Scientific qualification receipts are separate artifacts.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

CONTRACT = "gpuwm-sfire-worker-v1"


def capability_document():
    modules = ("gpuwm.core.sfire_coupler", "gpuwm.core.sfire_spotting",
               "gpuwm.core.chem_sfire", "gpuwm.ingest.wrfinput_sfire", "gpuwm.sfire_debug")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    package = Path(__file__).resolve().parent
    for kernel in ("sfire_core.cu", "sfire_phys.cu", "sfire_atm.cu", "sfire_coupling.cu",
                   "sfire_wind.cu", "sfire_moisture.cu", "sfire_spotting.cu", "chem_sfire.cu"):
        if not package.joinpath("core", "kernels", kernel).is_file():
            missing.append(kernel)
    symbols = ("gpuwm_static_sfire_experiment_grids", "gpuwm_static_sfire_load",
               "gpuwm_static_sfire_observed_perimeter", "gpuwm_static_sfire_debug_array")
    try:
        from gpuwm.static.sfire import _library
        library = _library()
        missing.extend(name for name in symbols if not hasattr(library, name))
    except (RuntimeError, ValueError, OSError) as exc:
        missing.append(f"native static bridge: {exc}")
    return dict(schema=CONTRACT, available=not missing, missing=missing,
                fire_static_request="gpuwm-sfire-static-v1", ifire=2,
                job_type="coupled-sfire", smoke_species="sfire_smoke",
                smoke_units="ug/kg dry air", smoke_is_pm25=False,
                observed_initialization_requires_burn_age=True,
                qualification="See source-bound scientific receipts; this probe checks installed interfaces")


def main():
    print(json.dumps(capability_document(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
