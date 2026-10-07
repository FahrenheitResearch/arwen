"""Bounded completed-atmosphere column copies for one firebrand owner."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


ATMOSPHERE_FIELDS = ("u", "v", "w", "php", "phb", "p", "pb", "thp", "thb",
                     "al", "alb", "msft", "mup", "mub2d")
MOISTURE_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
VERTICAL_FIELDS = ("c1h", "c2h", "dnw", "fnm", "fnp")


def spotting_global_keys(keys):
    """The exact native per-domain inventory excluded from tile transport."""
    from gpuwm.core.sfire_spotting import (PARTICLE_COORDS, PARTICLE_PROPERTIES,
        PARTICLE_IDENTIFIERS, COARSE_REAL_FIELDS, COARSE_INTEGER_FIELDS)
    allowed={"fire/spotting."+name for name in (*PARTICLE_COORDS, *PARTICLE_PROPERTIES,
        *PARTICLE_IDENTIFIERS, *COARSE_REAL_FIELDS, *COARSE_INTEGER_FIELDS, "fs_fire_rosdt", "control")}
    found={key for key in keys if key.startswith("fire/spotting.")}
    if found-allowed:
        raise ValueError("unrecognized global firebrand carrier would bypass ordinary tile transport")
    return tuple(sorted(found))


def mapped_host_array(value):
    """Expose pinned native bytes to CUDA without a whole-domain device copy."""
    import cupy as cp
    from tilestream.gather import is_device_array, is_pinned, array_pointer
    if is_device_array(value):
        return value
    if not is_pinned(value):
        raise ValueError("mapped firebrand fields require page-locked host bytes for device access")
    attributes = cp.cuda.runtime.pointerGetAttributes(array_pointer(value))
    pointer = int(attributes.get("devicePointer", 0) if isinstance(attributes, dict)
                  else getattr(attributes, "devicePointer", 0))
    if not pointer:
        raise ValueError("pinned firebrand fields have no CUDA device address; mapped host access is required")
    memory = cp.cuda.UnownedMemory(pointer, value.nbytes, value, device_id=cp.cuda.Device().id)
    return cp.ndarray(value.shape, value.dtype, cp.cuda.MemoryPointer(memory, 0), strides=value.strides)


class StreamedSpottingOwner:
    """One particle owner attached to authoritative completed domain bytes."""
    def __init__(self, state, cfg, store, scalars, geography=None, template=None):
        import cupy as cp
        from gpuwm.core.sfire_spotting import SpottingState, SpottingOptions
        from tilestream.hoststore import alloc_pinned_array
        self.cfg, self.store, self.scalars = cfg, store, scalars
        self.keys = spotting_global_keys(store)
        metadata=scalars["fire_header"]["fire"]["spotting"]
        arrays={key.split(".",1)[1]:mapped_host_array(store[key]) for key in self.keys}
        def allocate(shape,dtype):
            return mapped_host_array(alloc_pinned_array(shape,dtype))
        self.spotting=SpottingState((cfg.ny,cfg.nx),cfg.sr_x,cfg.sr_y,
            SpottingOptions.from_config(cfg),field_allocator=allocate,
            field_storage={key:value for key,value in arrays.items()
                           if not key.startswith("fs_p_") and key!="control"})
        self.spotting.restore(arrays,metadata)
        self.sources={}
        geography=geography or {}
        base=state if state is not None else template
        for name in (*ATMOSPHERE_FIELDS,*MOISTURE_FIELDS,*VERTICAL_FIELDS):
            value=store.get("state/"+name,geography.get("setup/"+name,geography.get(name)))
            if value is None and base is not None:
                value=getattr(base,name,None)
            if value is not None:
                self.sources[name]=value
        self.p_top=float(base.p_top)
        self.columns=BoundedAtmosphereColumns(self.sources,cfg,p_top=self.p_top)

    def synchronize(self):
        """Restore global particles and clocks after a store checkpoint load."""
        self.spotting.restore({key.split(".",1)[1]:mapped_host_array(self.store[key])
                              for key in self.keys},self.scalars["fire_header"]["fire"]["spotting"])

    def advance(self,*,dt,history_alarm=False):
        import cupy as cp
        fine={name:mapped_host_array(self.store["fire/grid."+name])[1:-1,1:-1]
              for name in ("burnt_area_dt","fgip","fmc_g","nfuel_cat","fire_area")}
        self.spotting.advance_native({"dt":dt},fine,step=self.spotting.step_count+1,
            history_alarm=history_alarm,advector=self.columns)
        for name,value in self.spotting.arrays().items():
            if name.startswith("fs_p_") or name=="control":
                dest=self.store["fire/spotting."+name]
                if isinstance(dest,cp.ndarray):
                    cp.copyto(dest,value)
                else:
                    value.get(out=dest)
        cp.cuda.get_current_stream().synchronize()
        self.scalars["fire_header"]["fire"]["spotting"]=self.spotting.metadata()

    def bind(self,states):
        """Every atmospheric buffer borrows the same passive particle owner."""
        for state in states:
            fire=getattr(getattr(state,"physics",None),"fire",None)
            if fire is not None:
                fire.spotting=self.spotting
                if getattr(state,"_tile_buffer",False):
                    fire._tile_spotting_owner=True


class BoundedAtmosphereColumns:
    """Copy occupied horizontal blocks and transport each active brand once.

    ``sources`` contains completed full-domain native atmosphere arrays,
    either device arrays or pinned host arrays. Base profiles may be one
    dimensional. No full-domain meteorology allocation is made on the card.
    Two native transport passes each move at most two coarse cells. A six
    cell halo covers that motion and the staggered interpolation operands.
    The owner performs generation, removal, deposits and history once after
    these independent column groups have been transported.
    """
    def __init__(self, sources, cfg, *, p_top, block_shape=(128, 128)):
        from tilestream.gather import _check_host
        self.cfg = cfg
        self.sources = dict(sources)
        self.p_top = float(p_top)
        self.block_shape = tuple(map(int, block_shape))
        if len(self.block_shape) != 2 or min(self.block_shape) < 1:
            raise ValueError("firebrand column blocks must have positive horizontal dimensions")
        for name in (*ATMOSPHERE_FIELDS, *VERTICAL_FIELDS):
            if name not in self.sources:
                raise ValueError(f"completed firebrand columns require atmosphere field {name}")
        nz,ny,nx=int(cfg.nz),int(cfg.ny),int(cfg.nx)
        mass=(nz,ny,nx);full=(nz+1,ny,nx)
        shapes={"u":((nz,ny,nx+1),),"v":((nz,ny+1,nx),),
            "w":(full,),"php":(full,),"phb":(full,(nz+1,)),
            **{key:(mass,) for key in ("p","thp","al",*MOISTURE_FIELDS)},
            **{key:(mass,(nz,)) for key in ("pb","thb","alb")},
            **{key:((ny,nx),) for key in ("msft","mup","mub2d")}}
        for name, value in self.sources.items():
            if name not in (*ATMOSPHERE_FIELDS, *MOISTURE_FIELDS, *VERTICAL_FIELDS):
                continue
            if name in shapes and value.shape not in shapes[name]:
                raise ValueError(f"completed firebrand atmosphere field {name} does not cover its native full-domain extent")
            if value.dtype != np.float32 or not value.flags.c_contiguous:
                raise ValueError(f"firebrand atmosphere field {name} must retain contiguous native float32 bytes")
            if value.ndim > 1:
                _check_host(value, name, "completed atmosphere", False)
        self.max_window_shape = (0, 0)
        self.max_window_bytes = 0
        self.windows = []

    def _window(self, bounds):
        import cupy as cp
        from tilestream.gather import (_FieldCopy, TilePlan, _memcpy_kind,
                                      is_device_array)
        from gpuwm.core.sfire_spotting import prepare_atmosphere
        x0, x1, y0, y1 = bounds
        ny, nx = y1-y0, x1-x0
        device = {}
        source = {}
        copies = []
        for name in (*ATMOSPHERE_FIELDS, *MOISTURE_FIELDS):
            value = self.sources.get(name)
            if value is None:
                continue
            if value.ndim == 1:
                device[name] = cp.asarray(value)
                continue
            dy = int(name == "v")
            dx = int(name == "u")
            shape = value.shape[:-2] + (ny+dy, nx+dx)
            device[name] = cp.empty(shape, cp.float32)
            source[name] = value
            depth = int(value.shape[0]) if value.ndim == 3 else 1
            copies.append(_FieldCopy(name, "column", 4,
                _memcpy_kind(is_device_array(value), True), value.shape, shape,
                y0, x0, 0, 0, ny+dy, nx+dx, depth))
        copied = {key: device[key] for key in source}
        TilePlan("gather", None, copies, None, tuple(source)).execute(source, copied)
        device.update({name: cp.asarray(self.sources[name]) for name in VERTICAL_FIELDS})
        device["p_top"] = self.p_top
        # The temporary compact window and its prepared columns are bounded
        # by block dimensions, vertical levels and the fixed dependency halo.
        self.windows.append(bounds)
        self.max_window_shape = (max(self.max_window_shape[0], ny),
                                 max(self.max_window_shape[1], nx))
        self.max_window_bytes = max(self.max_window_bytes,
            sum(value.nbytes for value in device.values() if hasattr(value, "nbytes")))
        local = SimpleNamespace(nx=nx, ny=ny, dx=self.cfg.dx, dy=self.cfg.dy)
        return prepare_atmosphere(SimpleNamespace(**device), local)

    def __call__(self, coordinates, life, prop, fields, dt, *, origin, tile, **options):
        import cupy as cp
        from gpuwm.core.sfire_spotting import particle_blocks, advect_particles
        ny, nx = int(self.cfg.ny), int(self.cfg.nx)
        by, bx = self.block_shape
        groups = particle_blocks(coordinates, coarse_shape=(ny, nx), block_shape=self.block_shape)
        occupied = cp.asnumpy(cp.unique(groups)).tolist()
        out = cp.empty_like(coordinates)
        updated = cp.empty_like(prop)
        status = cp.empty(coordinates.shape[1], cp.int32)
        blocks_x = (nx+bx-1)//bx
        self.windows = []
        for group in occupied:
            indices = cp.flatnonzero(groups == group)
            gx, gy = int(group)%blocks_x, int(group)//blocks_x
            x0, x1 = max(0, gx*bx-6), min(nx, (gx+1)*bx+6)
            y0, y1 = max(0, gy*by-6), min(ny, (gy+1)*by+6)
            meteorology = self._window((x0, x1, y0, y1))
            coords, properties, flags = advect_particles(coordinates[:, indices], life[indices],
                prop[:, indices], meteorology, dt, origin=(x0-3, y0-3), tile=tile, **options)
            out[:, indices] = coords
            updated[:, indices] = properties
            status[indices] = flags
            # Drain this bounded group before its source store may be reused.
            cp.cuda.get_current_stream().synchronize()
        return out, updated, status
