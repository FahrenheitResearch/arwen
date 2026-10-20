"""Prepare area-weighted ArWen land fluxes; launch performs no host numerics."""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType


class PreparedCatchmentExchange:
    """Resident CSR exchange, one CUDA thread per catchment, stable pointers.

    Parent RUNSRF/RUNSUB and SOLDRAIN are mm per LSM step; canopy/soil ET
    carriers are mm/s. Surface head is a mm state, retained without a rate
    conversion. Outputs are catchment-average m/s. Hydrofabric must supply
    complete, normalized overlap rows in deterministic cell order. Status is
    resident; consumers/checkpoints inspect ``status_arrays`` at a diagnostic
    boundary. Its masks retain errors across subsequent healthy steps and
    checkpoint restore. They are not cleared by launch.
    """

    def __init__(self, *, offsets, indices, weights, runsrf, runsub, soldrain,
                 sfcheadrt, smois, sh2o, ecan, etran, edir, dt_seconds,
                 weights_sha256, fabric_sha256, numerics="fast"):
        import math
        import cupy as cp
        import numpy as np
        from gpuwm.core.ieee_module import compile_module

        if numerics not in {"fast", "strict"}:
            raise ValueError("numerics must be fast or strict")
        if not math.isfinite(dt_seconds) or dt_seconds <= 0:
            raise ValueError("parent LSM interval must be finite and positive")
        inputs = dict(offsets=offsets, indices=indices, weights=weights,
                      runsrf=runsrf, runsub=runsub, soldrain=soldrain,
                      sfcheadrt=sfcheadrt, smois=smois, sh2o=sh2o,
                      ecan=ecan, etran=etran, edir=edir)
        device = runsrf.device.id
        for name, array in inputs.items():
            dtype = "int64" if name == "offsets" else "int32" if name == "indices" else "float32"
            if array.device.id != device or str(array.dtype) != dtype or not array.flags.c_contiguous:
                raise ValueError(f"{name}: require contiguous {dtype} on parent card")
        n = offsets.size - 1
        cells = runsrf.size
        if n < 1 or indices.size != weights.size:
            raise ValueError("invalid CSR dimensions")
        if any(a.shape != runsrf.shape for a in (runsub, soldrain, sfcheadrt, ecan, etran, edir)):
            raise ValueError("parent land flux sizes differ")
        if smois.shape != sh2o.shape or smois.ndim != runsrf.ndim + 1 or smois.shape[1:] != runsrf.shape:
            raise ValueError("soil carriers must alias the parent's layers-first shape")
        # Final offset is checked once during setup, never while stepping.
        if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != indices.size:
            raise ValueError("CSR endpoints must contain exactly the supplied overlaps")
        outputs = {name: cp.empty(n, cp.float32) for name in
                   ("surface_runoff_m_s", "subsurface_runoff_m_s", "recharge_m_s",
                    "evapotranspiration_m_s")}
        outputs["status"] = cp.zeros(n, cp.int32)
        sticky_status = cp.zeros(n, cp.int32)
        runoff_depth = cp.empty(n, cp.float32)
        options = ["--std=c++17", "--ftz=false"]
        if numerics == "strict":
            options += ["--fmad=false", "--prec-div=true", "--prec-sqrt=true"]
        module = compile_module(Path(__file__).with_name("coupling.cu").read_text(),
                                options, "nextgen.noahowp.catchment_exchange")
        self._kernel = module.get_function("catchment_exchange_routed")
        self._grid, self._block = ((n + 127) // 128,), (128,)
        self._args = (np.int32(n), np.int32(cells), np.int64(indices.size), np.float32(dt_seconds),
                      offsets, indices, weights, runsrf, runsub, soldrain, ecan, etran, edir,
                      *outputs.values(), sticky_status, runoff_depth)
        self.arrays = MappingProxyType(inputs | {"infxsrt": runsrf} | outputs |
                                       {"sticky_status": sticky_status, "runoff_depth_m": runoff_depth})
        self.period_seconds = float(dt_seconds)
        self.status_arrays = MappingProxyType({"exchange": sticky_status})
        self.restart_identity = {
            "algorithm": "arwen.noahmp.catchment-area-mean/v3",
            "numerics": numerics, "weights_sha256": weights_sha256,
            "fabric_sha256": fabric_sha256, "catchments": n,
            "parent_cells": cells, "lsm_interval_seconds": float(dt_seconds),
            "input_units": {"runsrf": "mm step-1", "runsub": "mm step-1",
                            "infxsrt": "mm step-1", "soldrain": "mm step-1",
                            "sfcheadrt": "mm", "smois": "m3 m-3", "sh2o": "m3 m-3",
                            "ecan": "mm s-1", "edir": "mm s-1", "etran": "mm s-1"},
            "output_units": "m s-1",
            "runoff_depth_units": "m per parent LSM period",
            "runoff_partition": "RUNSRF + RUNSUB; SOLDRAIN excluded (already in parent soil budget)",
            "order": "hydrofabric CSR row order; serial float32 reduction per catchment",
            "diagnostics": {"schema": "arwen.resident-sticky-status/v1",
                            "current": "status", "sticky": "sticky_status",
                            "bits": {"1": "invalid CSR, index, weight or row normalization"}},
        }

    def launch(self, dt_seconds, *, stream=None):
        self._kernel(self._grid, self._block, self._args, stream=stream)

    advance = launch


class PreparedRasterExchange(PreparedCatchmentExchange):
    """Same area mean on Rust-prepared terrain destination rows.

    Rows must cover the destination grid in its declared raster order. Grid
    identities and CSR construction belong to preparation; equal array shapes
    alone do not establish that two physical grids coincide.
    """

    def __init__(self, *, destination_shape, parent_grid_sha256,
                 terrain_grid_sha256, **kwargs):
        shape = tuple(destination_shape)
        if len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError("terrain destination_shape must be positive integer (ny,nx)")
        if shape[0] * shape[1] != kwargs["offsets"].size - 1:
            raise ValueError("terrain CSR rows must match every destination cell")
        if not parent_grid_sha256 or not terrain_grid_sha256:
            raise ValueError("terrain exchange requires prepared parent and destination grid identities")
        super().__init__(**kwargs)
        self.arrays = MappingProxyType(dict(self.arrays) | {
            "surface_source": self.arrays["surface_runoff_m_s"].reshape(shape),
            "recharge": self.arrays["recharge_m_s"].reshape(shape),
        })
        self.restart_identity.update({
            "algorithm": "arwen.noahmp.terrain-area-mean/v1",
            "routing_cells": self.restart_identity.pop("catchments"),
            "destination_shape": shape, "parent_grid_sha256": parent_grid_sha256,
            "terrain_grid_sha256": terrain_grid_sha256,
        })
