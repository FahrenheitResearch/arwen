"""Extend a single-domain WRF forcing horizon while preserving an accepted restart.

The original checkpoint is restored normally against its original preparation. Only
an identical complete forcing prefix may acquire a later tail. A newly bound
checkpoint is then written by the engine; no checkpoint validation is bypassed.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np


def require_equal(a, b, path="prefix"):
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise ValueError(f"{path}: array words differ")
    elif dataclasses.is_dataclass(a):
        if type(a) is not type(b):
            raise ValueError(f"{path}: type differs")
        for f in dataclasses.fields(a):
            require_equal(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, Mapping):
        if set(a) != set(b):
            raise ValueError(f"{path}: keys differ")
        for k in a:
            require_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f"{path}: length differs")
        for i, (x, y) in enumerate(zip(a, b)):
            require_equal(x, y, f"{path}[{i}]")
    elif a != b:
        raise ValueError(f"{path}: {a!r} != {b!r}")


def merge_boundary(old_run, later_run, destination):
    """Create a new run directory with the exact old input and an audited tail."""
    import netCDF4 as nc
    old_run, later_run, destination = map(Path, (old_run, later_run, destination))
    if destination.exists():
        raise FileExistsError(destination)
    # Later forcing must refer to the same immutable base and horizontal/vertical grid.
    base = ("MUB", "PB", "PHB", "XLAT", "XLONG", "HGT", "ZNW", "ZNU", "DN", "DNW", "FNM", "FNP", "RDN", "RDNW")
    with nc.Dataset(old_run / "wrfinput_d01") as x, nc.Dataset(later_run / "wrfinput_d01") as y:
        for ds in (x, y):
            ds.set_auto_maskandscale(False)
        for name in base:
            require_equal(np.asarray(x[name][:]), np.asarray(y[name][:]), name)
    destination.mkdir(parents=True)
    try:
        with nc.Dataset(old_run / "wrfbdy_d01") as x, nc.Dataset(later_run / "wrfbdy_d01") as y:
            for ds in (x, y):
                ds.set_auto_maskandscale(False)
            tx = [b"".join(z).decode() for z in x["Times"][:]]
            ty = [b"".join(z).decode() for z in y["Times"][:]]
            if tx[-1] != ty[0] or len(ty) < 2 or any(b <= a for a, b in zip(ty, ty[1:])):
                raise ValueError("forcing tail must start at the last existing record and advance")
            if set(x.variables) != set(y.variables):
                raise ValueError("boundary variable set differs")
            for name, v in x.variables.items():
                w = y[name]
                if v.dimensions != w.dimensions or v.dtype != w.dtype:
                    raise ValueError(f"{name}: boundary schema differs")
                require_equal(np.asarray(v[-1] if v.dimensions[0] == "Time" else v[:]),
                              np.asarray(w[0] if w.dimensions[0] == "Time" else w[:]), name)
            with nc.Dataset(destination / "wrfbdy_d01", "w", format=x.data_model) as z:
                z.setncatts({k: x.getncattr(k) for k in x.ncattrs()})
                for name, dim in x.dimensions.items():
                    z.createDimension(name, None if name == "Time" else len(dim))
                for name, v in x.variables.items():
                    attrs = {k: v.getncattr(k) for k in v.ncattrs()}
                    fill = attrs.pop("_FillValue", None)
                    w = z.createVariable(name, v.dtype, v.dimensions, **({"fill_value": fill} if fill is not None else {}))
                    w.setncatts(attrs)
                    if v.dimensions[0] == "Time":
                        w[:len(tx)] = v[:]
                        w[len(tx):] = y[name][1:]
                    else:
                        w[:] = v[:]
        for source in old_run.iterdir():
            if source.is_file() and source.name != "wrfbdy_d01":
                (destination / source.name).symlink_to(source.resolve())
        return dict(old_records=len(tx), new_records=len(tx)+len(ty)-1,
                    overlap=tx[-1], last_record=ty[-1], immutable_base_fields=list(base))
    except BaseException:
        import shutil
        shutil.rmtree(destination)
        raise


def extend_checkpoint(old_run, new_run, checkpoint, destination):
    """Restore under the original identity, validate the full parsed prefix, then rebind."""
    from gpuwm.wrfinput_door import resolve_wrfinput_run
    from gpuwm.wrfinput_forecast import prepare_wrf_run, WrfInitialization
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree, tree_restart_identity_components, _with_terrain_acoustics
    from gpuwm.ingest.lateral_bc import attach_lateral_boundaries
    from gpuwm.io.restart import read_restart_header, write_tree_restart
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    old_resolved = resolve_wrfinput_run(Path(old_run))
    new_resolved = resolve_wrfinput_run(Path(new_run))
    old = prepare_wrf_run(old_resolved, destination / "old-input", run_seconds=old_resolved.coverage.coverage_seconds)
    new = prepare_wrf_run(new_resolved, destination / "new-input", run_seconds=new_resolved.coverage.coverage_seconds)
    new = _with_terrain_acoustics(new)
    if len(old.domains) != 1 or len(new.domains) != 1:
        raise ValueError("forcing renewal currently requires a single domain")
    if len(new.boundaries.intervals) <= len(old.boundaries.intervals):
        raise ValueError("no new forcing intervals")
    for f in dataclasses.fields(old.boundaries):
        if f.name != "intervals":
            require_equal(getattr(old.boundaries, f.name), getattr(new.boundaries, f.name), f"boundaries.{f.name}")
    for i, interval in enumerate(old.boundaries.intervals):
        require_equal(interval, new.boundaries.intervals[i], f"converted_interval[{i}]")
    captured = {}
    def bootstrap(**kw):
        captured.update(kw)
        return dict(status="PASS", ensemble_bootstrap_only=True)
    (destination / "restored").mkdir()
    run_prepared_tree(old, output_directory=destination / "restored", io_mode="none", restart=Path(checkpoint),
                      initialization=WrfInitialization(old), ensemble_bootstrap=bootstrap)
    model = captured["model"]
    components = model._experiment_fingerprint_components
    future = tree_restart_identity_components(new, components["runtime_source_identity"], WrfInitialization(new))
    for name in components:
        if name != "preparation_receipt_sha256":
            require_equal(components[name], future[name], f"identity.{name}")
    state = model.root.state
    elapsed = state.elapsed_seconds
    attach_lateral_boundaries(state, new.boundaries)
    state.elapsed_seconds = elapsed
    model._experiment_fingerprint_components = future
    model.experiment_fingerprint = hashlib.sha256(json.dumps(future, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    header = read_restart_header(checkpoint)
    valid = old.experiment.start_time + dt.timedelta(seconds=float(header["elapsed_seconds"]))
    result = write_tree_restart(destination / "checkpoint", model, valid,
                                run_trackers_by_grid_id={int(header["grid_id"]): header.get("run_trackers")})
    # Rebinding forcing must leave every carried prognostic, scratch array, physics state and clock untouched.
    new_header = read_restart_header(result)
    allowed = {"created", "checkpoint_set_id", "experiment_fingerprint", "experiment_fingerprint_components",
               "setup_fingerprint", "config"}
    require_equal(set(header), set(new_header), "header keys")
    for name in header:
        if name not in allowed:
            require_equal(header[name], new_header[name], f"header.{name}")
    require_equal({k: v for k, v in header["config"].items() if k != "run_seconds"},
                  {k: v for k, v in new_header["config"].items() if k != "run_seconds"}, "config")
    equal_arrays = 0
    with np.load(checkpoint, allow_pickle=False) as a, np.load(result, allow_pickle=False) as b:
        require_equal(set(a.files), set(b.files), "checkpoint array names")
        for name in a.files:
            if name == "__gpuwm_restart_header__":
                continue
            require_equal(a[name], b[name], f"checkpoint.{name}")
            equal_arrays += 1
    return dict(checkpoint=str(result), old_intervals=len(old.boundaries.intervals), new_intervals=len(new.boundaries.intervals),
                valid=valid.isoformat(), equal_state_arrays=equal_arrays,
                original_identity=components, extended_identity=future)
