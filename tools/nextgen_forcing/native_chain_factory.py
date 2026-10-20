"""Fresh native cat-27 forcing/PET binding for the speed owner's timing door.

This is referee-fixture orchestration, not a product dataset or runtime. Set
NEXTGEN_FORCING_MANIFEST, NEXTGEN_PET_CAPTURE and GPUWM_NEXTGEN_FORCING_BRIDGE.
The speed lane provides gpuwm.nextgen_execution. At most 720 hourly advances,
including warmup, fit the captured series. No histories or per-step copies.
"""
from __future__ import annotations

import ctypes
from functools import partial
import json
import os
from pathlib import Path

PET_CAPTURE_SHA256 = "c1a17f49351fc7e43c303126cc5688a6f698194c07a94107b5767940456ef640"
FORCING_SHA256 = "4bc92c69725062a80a3bdb49c0c111776761cde6791629713232119b28a87ba8"


def _fixture_storage(manifest, capture, library):
    """Validate this fixed native case and read sealed bytes in Rust at setup."""
    from gpuwm.hydro.forcing_inputs import _library

    metadata = json.loads(manifest.read_text())
    expected = {
        "schema": "arwen.nextgen-forcing.v1", "shape": [720, 8, 1],
        "fields": ["T2D", "Q2D", "U2D", "V2D", "PSFC", "LWDOWN", "SWDOWN", "RAINRATE"],
        "units": ["K", "kg/kg", "m/s", "m/s", "Pa", "W/m2", "W/m2", "mm/s"],
        "time_origin": "2015-12-01T00:00:00Z", "interval_seconds": 3600.,
        "catchment_ids": ["cat-27"], "profile_row": "ngen-catchment-csv",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("native cat-27 packet clock, fields or ordering differ")
    if (metadata.get("files", {}).get("forcing.f64") != "<f8" or
            metadata.get("data_sha256", {}).get("forcing.f64") != FORCING_SHA256):
        raise ValueError("native cat-27 forcing seal differs")
    storage = (ctypes.c_ubyte * (720 * 39 * 8))()
    read = _library(library).arwen_forcing_read_packet
    read.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                     ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]
    read.restype = ctypes.c_int
    if read(bytes(capture), PET_CAPTURE_SHA256.encode(), storage, len(storage)):
        raise RuntimeError("Rust native PET capture bytes/hash differ from sealed fixture")
    return storage


def factory(*, strict=False):
    import cupy as cp
    import numpy as np
    from gpuwm import numerics
    from gpuwm.hydro.forcing_inputs import load_series
    from gpuwm.hydro.pet import PET_PIN, PreparedPET
    from gpuwm.nextgen_execution import ResidentSequence, Stage

    expected = "strict" if strict else "fast"
    if numerics.mode() != expected:
        raise ValueError("select requested numerics before constructing the factory")
    manifest = Path(os.environ["NEXTGEN_FORCING_MANIFEST"])
    capture = Path(os.environ["NEXTGEN_PET_CAPTURE"])
    library = os.environ["GPUWM_NEXTGEN_FORCING_BRIDGE"]
    storage = _fixture_storage(manifest, capture, library)
    # Rust-filled typed storage is uploaded unchanged; GPU selects initial state.
    records = cp.asarray(np.ndarray((720, 39), dtype="<f8", buffer=storage))
    forcing = load_series(manifest, library=library, native_precision=True,
                          device_id=cp.cuda.Device().id, start_seconds=0.)
    pet = PreparedPET(forcing.fields, records[:1, :12].copy(), method=5,
                      identity={"case": "native cat-27 hourly chain",
                                "capture_sha256": PET_CAPTURE_SHA256,
                                "forcing": forcing.restart_identity})
    return ResidentSequence(
        (Stage("forcing", partial(forcing.advance, 3600.), "1 block x128 threads,1 catchment,float64"),
         Stage("pet", partial(pet.advance, 3600.), "1 block x128 threads,1 catchment,float64")),
        step_seconds=3600., keepalive=(forcing, pet),
        metadata={"numerics": expected, "case": "pinned ngen cat-27, hourly native forcing/PET only",
                  "pet_source_pin": PET_PIN, "forcing_identity": forcing.restart_identity,
                  "native_pet_capture": str(capture), "native_pet_sha256": PET_CAPTURE_SHA256,
                  "maximum_intervals_including_warmup": 720, "launches_per_interval": 2,
                  "histories": False, "physical_runoff_routing_acceptance": False})
