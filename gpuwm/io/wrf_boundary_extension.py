"""Extend a single-domain WRF forcing horizon while preserving an accepted restart.

The original checkpoint is restored normally against its original preparation. Only
an identical complete forcing prefix may acquire a later tail. A newly bound
checkpoint is then written by the engine; no checkpoint validation is bypassed.
"""
from __future__ import annotations

import dataclasses
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
