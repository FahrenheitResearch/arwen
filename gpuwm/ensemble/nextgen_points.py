"""Original GPU point-product reduction for NextGen ensemble members.

No upstream hydrologic algorithm is implemented here. Samples are borrowed
only at output deadlines; all comparisons, counts and fractions run in CUDA.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
import math
import json
import threading

SCHEMA = "arwen.nextgen.reach-ensemble/v2"


@dataclass(frozen=True)
class ReachSample:
    reach_id: object
    discharge_m3_s: object
    channel_depth_m: object = None
    stream: object = None


def product_options(value=None):
    value = dict(value or {})
    if set(value) - {"thresholds_m3_s", "device_budget_bytes"}:
        raise ValueError("NextGen point products take thresholds_m3_s and device_budget_bytes")
    thresholds = tuple(value.get("thresholds_m3_s", (1., 10., 100.)))
    if (not thresholds or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                              for x in thresholds) or len(set(thresholds)) != len(thresholds)):
        raise ValueError("discharge thresholds need distinct finite nonnegative numbers in m3/s")
    budget = value.get("device_budget_bytes", 256 << 20)
    if type(budget) is not int or budget <= 0:
        raise ValueError("NextGen point-product device budget must be a positive integer")
    return {"thresholds_m3_s": list(map(float, thresholds)), "device_budget_bytes": budget}


_SOURCE = r'''
extern "C" __global__ void thresholds32(const double* raw, float* threshold,
 unsigned* error, int k) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=k) return;
 float t=__double2float_rn(raw[i]);
 threshold[i]=t;
 if(!isfinite(t)) atomicExch(error, 1u);
 for(int j=0;j<i;j++) if(t==__double2float_rn(raw[j])) atomicExch(error, 1u);
}
template<class T> __device__ void add_point(const long long* ids, const T* flow,
 const long long* reference, const float* threshold, unsigned* count,
 unsigned* finite, unsigned* error, unsigned long long n, int k) {
 unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
 if(i >= n) return;
 if(ids[i] != reference[i]) atomicExch(error, 1u);
 T q=flow[i];
 if(isfinite(q)) {
   finite[i] += 1;
   for(int t=0;t<k;t++) count[t*n+i] += ((double)q >= threshold[t]);
 }
}
extern "C" __global__ void add32(const long long* ids, const float* q,
 const long long* ref, const float* t, unsigned* c, unsigned* f,
 unsigned* bad, unsigned long long n, int k) {add_point(ids,q,ref,t,c,f,bad,n,k);}
extern "C" __global__ void add64(const long long* ids, const double* q,
 const long long* ref, const float* t, unsigned* c, unsigned* f,
 unsigned* bad, unsigned long long n, int k) {add_point(ids,q,ref,t,c,f,bad,n,k);}
extern "C" __global__ void fractions(const unsigned* count, const unsigned* finite,
 float* result, unsigned long long n, int k, unsigned members) {
 unsigned long long i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
 if(i<n*k) result[i] = finite[i%n] == members
   ? __double2float_rn(__ddiv_rn((double)count[i], (double)members))
   : __int_as_float(0x7fc00000);
}
'''


class ReachAccumulator:
    """A bounded private pool per card, serialized by output CUDA events.

    Each routing state stays owned by its member. Only IDs and integer product
    counts are retained. Member producer streams wait on preceding count work;
    this avoids racing counter updates without downloads or host waits.
    """
    def __init__(self, member_ids, *, options=None, array_module=None, depth_sink=None):
        self.members = tuple(member_ids)
        if (not self.members or len(set(self.members)) != len(self.members)
                or len(self.members) > 2**32-1
                or any(type(m) is not int or m < 0 for m in self.members)):
            raise ValueError("NextGen ensemble needs distinct nonnegative member IDs")
        self.options = product_options(options)
        self.cp, self.depth_sink = array_module, depth_sink
        self.frames, self.devices = {}, {}
        self.lock = threading.RLock()
        self.closed = False
        self.products = None
        self.published_frames = []
        self.restored = set()

    def _device(self, device):
        cp = self.cp
        if device not in self.devices:
            pool = cp.cuda.MemoryPool()
            pool.set_limit(size=self.options["device_budget_bytes"])
            from gpuwm.certify.kernel_manifest import record_module
            module = cp.RawModule(code=_SOURCE, options=())
            record_module("nextgen_points", source=_SOURCE, options=(), module=module)
            with cp.cuda.using_allocator(pool.malloc):
                raw = cp.asarray(self.options["thresholds_m3_s"], dtype=cp.float64)
                threshold = cp.empty(raw.shape, dtype=cp.float32)
                error = cp.zeros(1, dtype=cp.uint32)
                import numpy as np
                module.get_function("thresholds32")(((raw.size+255)//256,), (256,),
                    (raw, threshold, error, np.int32(raw.size)))
                if bool(error.item()):  # first output/setup boundary only
                    raise ValueError("discharge thresholds must remain finite and distinct after float32 rounding")
            self.devices[device] = {"pool": pool,
                "threshold": threshold,
                "add32": module.get_function("add32"), "add64": module.get_function("add64"),
                "fractions": module.get_function("fractions")}
        return self.devices[device]

    def _shard(self, ids):
        cp = self.cp
        with cp.cuda.using_allocator(self._device(int(ids.device.id))["pool"].malloc):
            return {"ids": ids.copy(), "threshold": self.devices[int(ids.device.id)]["threshold"],
                "count": cp.zeros((len(self.options["thresholds_m3_s"]), ids.size), dtype=cp.uint32),
                "finite": cp.zeros(ids.size, dtype=cp.uint32), "error": cp.zeros(1, dtype=cp.uint32),
                "done": None}

    def accept(self, *, member_id, valid_time, sample):
        if self.cp is None:
            import cupy
            self.cp = cupy
        cp = self.cp
        if isinstance(sample, Mapping):
            # Routing may expose velocity and reservoir elevation on its full
            # output map. Consume only named product fields; depth_m is not
            # channel_depth_m and must never be silently reinterpreted.
            sample = ReachSample(**{key: sample.get(key) for key in
                ("reach_id", "discharge_m3_s", "channel_depth_m", "stream")})
        ids, q = sample.reach_id, sample.discharge_m3_s
        if (not isinstance(ids, cp.ndarray) or not isinstance(q, cp.ndarray)
                or ids.dtype != cp.dtype("int64") or q.dtype not in (cp.dtype("float32"), cp.dtype("float64"))
                or ids.ndim != 1 or q.shape != ids.shape or not ids.size
                or not ids.flags.c_contiguous or not q.flags.c_contiguous
                or ids.device.id != q.device.id):
            raise ValueError("reach output needs same-card contiguous int64 IDs and native float32/64 discharge")
        depth = sample.channel_depth_m
        if self.depth_sink is not None and (not isinstance(depth, cp.ndarray) or depth.shape != ids.shape
                or depth.device.id != ids.device.id or depth.dtype not in (cp.dtype("float32"), cp.dtype("float64"))
                or not depth.flags.c_contiguous):
            raise ValueError("HAND consumer needs producer channel_depth_m above bed, on this reach axis")
        key = str(valid_time)
        with self.lock:
            if self.closed or member_id not in self.members:
                raise ValueError("NextGen output is closed or member is outside its requested roster")
            if (key, member_id) in self.restored:
                self.restored.remove((key, member_id))
                return  # restored checkpoint deadline may be emitted once again
            frame = self.frames.setdefault(key, {"members": set(), "n": ids.size, "cards": {}})
            if member_id in frame["members"] or frame["n"] != ids.size:
                raise ValueError("duplicate member/time or changed NextGen reach axis")
            device = int(ids.device.id)
            with cp.cuda.Device(device):
                stream = sample.stream or cp.cuda.get_current_stream()
                with stream:
                    shard = frame["cards"].get(device)
                    if shard is None:
                        shard = frame["cards"][device] = self._shard(ids)
                    elif shard["done"] is not None:
                        stream.wait_event(shard["done"])
                    import numpy as np
                    kind = "add32" if q.dtype == cp.dtype("float32") else "add64"
                    self.devices[device][kind](((ids.size+255)//256,), (256,),
                        (ids, q, shard["ids"], shard["threshold"], shard["count"], shard["finite"],
                         shard["error"], np.uint64(ids.size), np.int32(shard["count"].shape[0])))
                    if self.depth_sink is not None:
                        self.depth_sink(member_id=member_id, valid_time=valid_time,
                            feature_id=ids, channel_depth_m=depth, stream=stream)
                    shard["done"] = cp.cuda.Event(disable_timing=True)
                    shard["done"].record(stream)
            frame["members"].add(member_id)

    def require_complete(self):
        if not self.frames:
            raise RuntimeError("NextGen ensemble produced no reach output deadlines")
        for key, frame in self.frames.items():
            missing = set(self.members)-frame["members"]
            if missing:
                raise RuntimeError(f"NextGen reach output {key} is missing members {sorted(missing)}")

    def finish(self, root):
        with self.lock:
            if self.products is not None:
                return self.products
            self.require_complete()
            if self.closed:
                raise RuntimeError("NextGen product publication failed; restore a sealed checkpoint")
            self.closed = True
            self.products = []
            cp = self.cp
            try:
                for index, (key, frame) in enumerate(sorted(self.frames.items())):
                    device = min(frame["cards"])
                    with cp.cuda.Device(device), cp.cuda.using_allocator(self.devices[device]["pool"].malloc):
                        first = frame["cards"][device]
                        count, finite = cp.zeros_like(first["count"]), cp.zeros_like(first["finite"])
                        for source in frame["cards"].values():
                            if source["done"] is not None:
                                cp.cuda.get_current_stream().wait_event(source["done"])
                            if bool(source["error"].item()):
                                raise ValueError("NextGen member changed feature IDs/order")
                            ids = cp.asarray(source["ids"])
                            if not bool(cp.array_equal(ids, first["ids"])):
                                raise ValueError("NextGen cards disagree on feature IDs/order")
                            count += cp.asarray(source["count"])
                            finite += cp.asarray(source["finite"])
                        probability = cp.empty(count.shape, dtype=cp.float32)
                        import numpy as np
                        self.devices[device]["fractions"](((probability.size+255)//256,), (256,),
                            (count, finite, probability, np.uint64(frame["n"]), np.int32(count.shape[0]), np.uint32(len(self.members))))
                        path = Path(root)/f"reach-exceedance-{index:05d}.nc"
                        _write_product(path, key, self.members, first["ids"], first["threshold"], count, finite, probability, cp)
                        self.products.append(str(path))
                        self.published_frames.append(_frame_contract(path, key, self.members,
                            frame["n"], len(self.options["thresholds_m3_s"])))
                from gpuwm.ensemble.restart_roster import atomic_json
                manifest = Path(root) / "reach-ensemble.json"
                atomic_json(manifest, {"schema": SCHEMA, "frames": self.published_frames})
                self.products.append(str(manifest))
            except BaseException:
                self.products = None
                self.published_frames = []
                raise
            return self.products

    def receipt(self):
        return dict(schema=SCHEMA, members=list(self.members), options=self.options,
            comparison=">=", probability_units="1", discharge_units="m3 s-1",
            nonfinite="mask reach if any requested member is nonfinite", frames=len(self.frames),
            parallelism="CUDA one thread/reach; event-ordered member launches", cuda_graphs=False,
            routing_step_host_work=False, native_precision="float32 or float64, preserved through comparison",
            threshold_precision="float32 round-to-nearest-even; promoted exactly for native discharge comparison",
            published_buffers=self.published_buffers())

    def published_buffers(self):
        """Metadata-only file ABI; values remain in the Rust-written NetCDF."""
        return {"schema": SCHEMA, "manifest": None if not self.products else self.products[-1],
                "frames": self.published_frames}

    def save(self, root):
        """Immutable Rust NetCDF snapshot, including incomplete output cohorts."""
        from gpuwm.io.classic_product import ClassicProduct
        from gpuwm.ensemble.restart_roster import digest_file, relative_file
        root = Path(root)
        directory = root / ".ensemble-resume"
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        pending = directory / "nextgen-points.pending.nc"
        with self.lock, ClassicProduct(pending) as writer:
            for number, (key, frame) in enumerate(sorted(self.frames.items())):
                row = {"valid_time": key, "members": sorted(frame["members"]), "shards": []}
                for card, shard in sorted(frame["cards"].items()):
                    prefix = f"f{number}_d{card}"
                    with self.cp.cuda.Device(card):
                        if shard["done"] is not None:
                            shard["done"].synchronize()  # checkpoint boundary only
                        if bool(shard["error"].item()):
                            raise ValueError("NextGen member changed feature IDs/order")
                        # Rust reader's decoded transport is double. Preserve
                        # every int64 ID bit using exact uint32 words instead.
                        for label, value in (("ids", shard["ids"].view(self.cp.uint32)),
                                ("count", shard["count"]), ("finite", shard["finite"])):
                            dims = tuple(f"{prefix}_{label}_{axis}" for axis in range(value.ndim))
                            for dim, size in zip(dims, value.shape):
                                writer.createDimension(dim, size)
                            writer.createVariable(f"{prefix}_{label}", value.dtype, dims)[:] = self.cp.asnumpy(value)
                    row["shards"].append(prefix)
                rows.append(row)
        path = directory / (digest_file(pending) + ".nc")
        pending.replace(path)
        return {"schema": SCHEMA, "members": list(self.members), "options": self.options,
                "frames": rows, "arrays": relative_file(root, path)}

    def restore(self, root, document):
        from gpuwm.netcdf_bridge import Dataset
        from gpuwm.ensemble.restart_roster import resolve_file
        if (document.get("schema") != SCHEMA or document.get("members") != list(self.members)
                or document.get("options") != self.options or self.frames):
            raise ValueError("NextGen checkpoint differs from requested members/products or output already began")
        if self.cp is None:
            import cupy
            self.cp = cupy
        cp = self.cp
        # Uploads coexist with retained shards. Keep both inside the same
        # per-card hard limit reserved by ordinary ensemble admission; the
        # default allocator must not become an unpriced restore workspace.
        with (self.lock, Dataset(resolve_file(Path(root), document["arrays"])) as source,
              cp.cuda.using_allocator(self._device(int(cp.cuda.runtime.getDevice()))["pool"].malloc)):
            source.set_auto_maskandscale(False)
            for row in document["frames"]:
                members = set(row["members"])
                if len(members) != len(row["members"]) or not members <= set(self.members):
                    raise ValueError("invalid NextGen checkpoint cohort")
                target = None
                for prefix in row["shards"]:
                    ids = cp.asarray(source.variables[f"{prefix}_ids"][:], dtype=cp.uint32).view(cp.int64)
                    if target is None:
                        target = self._shard(ids)
                    elif not bool(cp.array_equal(ids, target["ids"])):
                        raise ValueError("NextGen checkpoint shard reach axes differ")
                    for label in ("count", "finite"):
                        value = cp.asarray(source.variables[f"{prefix}_{label}"][:], dtype=cp.uint32)
                        if value.shape != target[label].shape:
                            raise ValueError("invalid NextGen checkpoint array shape")
                        target[label] += value
                if target is None:
                    raise ValueError("NextGen checkpoint has an empty frame")
                target["done"] = cp.cuda.Event(disable_timing=True)
                target["done"].record()
                key = row["valid_time"]
                self.frames[key] = {"members": members, "n": target["ids"].size,
                                    "cards": {int(target["ids"].device.id): target}}
                self.restored.update((key, member) for member in members)

    def merge_member(self, root, document, member_id):
        """Consume a completed mass worker's integer shard through Rust.

        The parent uses its current GPU. No floating point probabilities are
        averaged across workers, and no denominator is inferred from files.
        """
        from gpuwm.netcdf_bridge import Dataset
        from gpuwm.ensemble.restart_roster import resolve_file
        if (document.get("schema") != SCHEMA or document.get("members") != [member_id]
                or document.get("options") != self.options or member_id not in self.members
                or not document.get("frames")):
            raise ValueError("mass NextGen shard differs from the requested member/products")
        if self.cp is None:
            import cupy
            self.cp = cupy
        cp = self.cp
        with self.lock, Dataset(resolve_file(Path(root), document["arrays"])) as source:
            if self.closed:
                raise ValueError("NextGen output already closed")
            source.set_auto_maskandscale(False)
            for row in document["frames"]:
                if row["members"] != [member_id] or len(row["shards"]) != 1:
                    raise ValueError("mass NextGen worker has an incomplete or split member cohort")
                key, prefix = row["valid_time"], row["shards"][0]
                frame = self.frames.get(key)
                if frame is not None and member_id in frame["members"]:
                    raise ValueError("duplicate mass NextGen member/time")
                device = int(cp.cuda.runtime.getDevice())
                with cp.cuda.using_allocator(self._device(device)["pool"].malloc):
                    ids = cp.asarray(source.variables[f"{prefix}_ids"][:], dtype=cp.uint32).view(cp.int64)
                    count = cp.asarray(source.variables[f"{prefix}_count"][:], dtype=cp.uint32)
                    finite = cp.asarray(source.variables[f"{prefix}_finite"][:], dtype=cp.uint32)
                    if (ids.ndim != 1 or not ids.size or finite.shape != ids.shape
                            or count.shape != (len(self.options["thresholds_m3_s"]), ids.size)
                            or bool(cp.any(finite > 1)) or bool(cp.any(count > finite[None, :]))):
                        raise ValueError("invalid mass NextGen integer counts")
                    if frame is None:
                        frame = self.frames[key] = {"members": set(), "n": ids.size, "cards": {}}
                    shard = frame["cards"].get(device)
                    if shard is None:
                        shard = frame["cards"][device] = self._shard(ids)
                    if shard["done"] is not None:
                        cp.cuda.get_current_stream().wait_event(shard["done"])
                    if not bool(cp.array_equal(ids, shard["ids"])):
                        raise ValueError("mass NextGen reach axes differ")
                    shard["count"] += count
                    shard["finite"] += finite
                    shard["done"] = cp.cuda.Event(disable_timing=True)
                    shard["done"].record()
                    frame["members"].add(member_id)


def _write_product(path, valid, members, ids, thresholds, count, finite, probability, cp):
    from gpuwm.io.classic_product import ClassicProduct
    path.parent.mkdir(parents=True, exist_ok=True)
    with ClassicProduct(path) as file:
        file.setncattr("schema", SCHEMA)
        file.setncattr("valid_time", valid)
        file.setncattr("requested_members", len(members))
        file.setncattr("requested_member_ids", json.dumps(list(members)))
        file.setncattr("comparison", ">=")
        file.setncattr("comparison_operator", "ge")
        file.setncattr("missing_policy", "complete_roster")
        file.setncattr("denominator", "full requested roster")
        file.createDimension("feature", ids.size)
        file.createDimension("word", 2)
        file.createDimension("threshold", thresholds.size)
        for name, value, dims, units in (("feature_id", ids, ("feature",), "1"),
                ("feature_id_words", ids.view(cp.uint32).reshape(-1, 2), ("feature", "word"), "1"),
                ("discharge_threshold", thresholds, ("threshold",), "m3 s-1"),
                ("exceedance_count", count, ("threshold", "feature"), "members"),
                ("finite_count", finite, ("feature",), "members"),
                ("probability", probability, ("threshold", "feature"), "1")):
            variable = file.createVariable(name, value.dtype, dims)
            variable.setncattr("units", units)
            variable[:] = cp.asnumpy(value)  # output-boundary Rust writer input only


def _frame_contract(path, valid, members, reaches, thresholds):
    def buffer(variable, dtype, shape, units):
        return {"path": path.name, "variable": variable, "dtype": dtype,
                "shape": shape, "units": units}
    return {"valid_time": valid, "requested_member_ids": list(members),
        "comparison": "ge", "missing_policy": "complete_roster", "denominator": len(members),
        "reach_id": buffer("feature_id", "<i8", [reaches], "1"),
        # The existing Rust reader's decoded-double transport cannot represent
        # every int64 ID. Low/high words retain exact ID bits on that transport.
        "reach_id_words": {**buffer("feature_id_words", "<u4", [reaches, 2], "1"),
                           "encoding": "low32,high32; two's-complement int64"},
        "threshold_m3_s": buffer("discharge_threshold", "<f4", [thresholds], "m3 s-1"),
        "exceedance_count": buffer("exceedance_count", "<u4", [thresholds, reaches], "members"),
        "finite_count": buffer("finite_count", "<u4", [reaches], "members"),
        "probability": buffer("probability", "<f4", [thresholds, reaches], "1")}
