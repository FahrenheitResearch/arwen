"""Run one NVIDIA Earth-2 model through Earth2Studio and write its NetCDF.

Executed by the interpreter that has ``earth2studio`` installed (named by
``GPUWM_EARTH2_PYTHON``), never imported by gpuwm itself: it imports no
gpuwm module, so the model runtime and its CUDA stack stay in their own
environment.  Every model-specific fact arrives on the command line from
the Earth-2 model table (``gpuwm/authorities/earth2-models.v1.json``); this
file names no model.

    python earth2_runner.py REQUEST.json

REQUEST: {"model_class", "ic_class", "init", "steps", "output",
          "area": [south, west, north, east] (west/east in 0..360) or null}
Prints one JSON line: load and run seconds, peak GPU memory, output path.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import sys
import time


def select_batch_size(request, coords, *, free_bytes, total_bytes):
    """Estimate capacity after loading weights, using the row workspace budget."""
    import math
    policy = request.get("member_policy") or {}
    count = len(request["members"])
    limit = request.get("batch_size", policy.get("batch_size", 1))
    cap = count if limit in (None, "auto") else min(count, int(limit))
    if cap < 1:
        raise ValueError("Earth-2 batch size must be positive")
    native = policy.get("native_rng", {})
    serial_rng = bool(native and not native.get("state_initializer") and
                      (native.get("seed_attribute") or native.get("seed_method") or
                       native.get("generator") == "torch"))
    if serial_rng:
        cap = 1
    memory = policy.get("memory")
    estimate = {"free_bytes_after_model_load": int(free_bytes),
                "total_bytes": int(total_bytes), "requested_cap": cap,
                "native_rng_serial": serial_rng, "capacity_is_estimate": True}
    if memory:
        state = math.prod(max(1, len(values)) for axis, values in coords.items()
                          if axis not in ("batch", "ensemble")) * int(memory["element_bytes"])
        per_member = state * int(memory["resident_state_copies"])
        fixed = int(memory["workspace_bytes"]) + int(memory["reserve_bytes"]) + state
        capacity = (int(free_bytes) - fixed) // per_member
        if capacity < 1:
            # The row contract is an unmeasured allowance; it sizes BATCHES.
            # One member at a time is the sequential forward every single
            # forecast of this row already runs (FCN3: 72 h Helene on an RTX
            # 5090, 2026-10-19, peak 29.06 GiB), so the batch degrades to it
            # and the runner's measured peak goes in the receipt. A real
            # out-of-memory still fails loudly as one. Breakage it prevents:
            # an 18-member FCN3 batch was refused on the 32 GB card that runs
            # the same model's single forecasts (contract 29.95e9 B for one
            # member, 28.76e9 B free after weights).
            print(f"earth2 runner: row memory contract needs {fixed + per_member} B for one member "
                  f"and {free_bytes} B are free after weights; running members one at a time "
                  "(the row's single-forecast path) and recording the measured peak",
                  file=sys.stderr, flush=True)
            estimate.update(below_contract=True)
            capacity = 1
        cap = min(cap, capacity)
        estimate.update(state_bytes=state, per_member_bytes=per_member,
                        fixed_bytes=fixed, policy=memory)
    elif limit in (None, "auto") and not serial_rng:
        raise ValueError("automatic Earth-2 batching requires a row memory contract")
    estimate.update(batch_size=cap, sub_batches=(count + cap - 1) // cap)
    return cap, estimate


def member_perturbation(model, policy, seeds):
    """Configure native noise or an Earth2Studio perturbation from TABLE DATA.

    Native models expose different RNG protocols. The table declares either a
    method plus a per-member state initializer, or the global torch generator.
    State initialization is rebound once per sub-batch, retaining the model's
    own iterator and forward path, including its stochastic time evolution.
    """
    import torch
    import earth2studio.perturbation as perturbations
    perturb = policy["perturbation"]
    operation = getattr(perturbations, perturb["class"])(**perturb.get("parameters", {}))
    active_seeds = []
    native = policy.get("native_rng")
    if native and native.get("state_initializer"):
        original = getattr(model, native["state_initializer"])

        def initialize(num_ensemble, num_time):
            if num_ensemble != len(active_seeds):
                raise ValueError("native RNG member count differs from workflow batch")
            states = []
            for seed in active_seeds:
                getattr(model, native["seed_method"])(seed=int(seed), reset=True)
                original(1, num_time)
                states.append(getattr(model, native["state_attribute"])[0])
            setattr(model, native["state_attribute"], states)

        setattr(model, native["state_initializer"], initialize)

    def apply(x, coords):
        active_seeds[:] = [int(seeds[int(index)]) for index in coords["ensemble"]]
        if native and (native.get("seed_attribute") or
                       (native.get("seed_method") and not native.get("state_initializer"))):
            if len(active_seeds) != 1:
                raise ValueError("model-level native RNG requires batch_size=1 for per-member replay")
            if native.get("seed_attribute"):
                setattr(model, native["seed_attribute"], active_seeds[0])
            if native.get("seed_method") and not native.get("state_initializer"):
                getattr(model, native["seed_method"])(seed=active_seeds[0])
        if native and native.get("generator") == "torch":
            if len(active_seeds) != 1:
                raise ValueError("global torch native RNG requires batch_size=1 for per-member replay")
            torch.manual_seed(active_seeds[0])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(active_seeds[0])
        if policy["kind"] == "perturbation":
            axis = list(coords).index("ensemble")
            for index, seed in enumerate(active_seeds):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                selection = [slice(None)] * x.ndim
                selection[axis] = slice(index, index + 1)
                subset = coords.copy()
                subset["ensemble"] = coords["ensemble"][index:index + 1]
                x[tuple(selection)], _ = operation(x[tuple(selection)], subset)
            return x, coords
        return operation(x, coords)
    return apply


def _band_contracted_disco():
    """Fit torch-harmonics' DISCO convolution on cards its fast kernel skips.

    torch-harmonics 0.9 ships its k-packed DISCO kernel for SM 9.0 and
    10.0 only; elsewhere (SM 12.0: RTX 5090, RTX PRO 6000) the fallback
    materialises a permuted copy of the expanded input for one einsum,
    about 20 GiB at 0.25 degrees, which does not fit a 32 GiB card beside
    the model.  Contracting one latitude band at a time computes the same
    sums without that copy.  A no-op when torch-harmonics is absent.
    """
    try:
        import torch
        from torch_harmonics.disco.optimized import disco_optimized as disco
    except Exception:
        return False

    def forward(ctx, inp, weight, roff_idx, ker_idx, row_idx, col_idx, vals,
                split_roff_idx, split_nnz_off, split_ker_idx, split_row_idx,
                split_col_idx, split_vals, kernel_size, nlat_out, nlon_out,
                groups, groupsize, split_row_offsets=(), split_nnz_offsets=()):
        itype = inp.dtype
        expanded = disco.disco_kernels.forward.default(
            inp.contiguous(), roff_idx, ker_idx, row_idx, col_idx,
            vals.to(disco._compute_dtype(itype)), kernel_size, nlat_out, nlon_out,
        ).to(itype)
        batch, channels, kernel, rows, columns = expanded.shape
        expanded = expanded.reshape(batch, groups, groupsize, kernel, rows, columns)
        weight = weight.to(itype)
        out = torch.empty((batch, groups, weight.shape[1], rows, columns),
                          dtype=itype, device=expanded.device)
        band = max(1, rows // 96)
        for row in range(0, rows, band):
            out[..., row:row + band, :] = torch.einsum(
                "bgckxy,gock->bgoxy", expanded[..., row:row + band, :], weight)
        return out.reshape(batch, groups * weight.shape[1], rows, columns)

    disco._DiscoSaveXConvFn.forward = staticmethod(forward)
    return True


def _cpu_count():
    import os
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def _sha256(path):
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def prepare_loader_inputs(request):
    """Materialize row-declared initial-state assets for unparameterized loaders.

    The caller holds the cache lock through model construction, so two cycles
    cannot replace a loader's fixed cache filename underneath each other.
    """
    from pathlib import Path
    import numpy as np
    import torch
    import earth2studio.data as data
    from earth2studio.data import fetch_data
    from earth2studio.models.auto import Package
    receipts = []
    for preparation in request.get("loader_preparations", ()):
        begin = time.perf_counter()
        root = Path(Package.default_cache(preparation["cache_subdir"])).resolve()
        path = (root / preparation["filename"]).resolve()
        if path.parent != root or preparation["format"] != "torch":
            raise ValueError("loader preparation needs a direct cache filename and torch format")
        root.mkdir(parents=True, exist_ok=True)
        source = getattr(data, preparation.get("data_class", request["ic_class"]))(
            **preparation.get("data_kwargs", request.get("ic_kwargs", {})))
        clock = request["init"] if preparation["time"] == "$init" else preparation["time"]
        tensor, _ = fetch_data(source=source, time=np.array([np.datetime64(clock)]),
                               variable=preparation["variables"], device=torch.device("cpu"))
        temporary = path.with_suffix(path.suffix + ".new")
        torch.save(tensor.squeeze(), temporary)
        temporary.replace(path)
        receipts.append(dict(_sha256(path), source=type(source).__name__, cycle=clock,
                             variables=preparation["variables"], seconds=round(time.perf_counter() - begin, 3),
                             parallelism=f"{_cpu_count()} threads / network"))
    return receipts


class PackageAudit:
    """Pin HF revisions and hash the assets the loader actually opens.

    Covers auxiliary packages as well as the checkpoint. Credentials on disk
    are never consulted. Package.open performs the transfer and the hashing
    workers run beside subsequent load/inference work.
    """
    def __init__(self, cache, workers, expected=None):
        import os
        from concurrent.futures import ThreadPoolExecutor
        from earth2studio.models.auto import Package
        from huggingface_hub import HfApi
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.assets = {}
        self.pending = {}
        self.expected = expected or {}
        self.original_init = Package.__init__
        self.original_open = Package.open
        audit = self
        token = os.environ.get("HF_TOKEN") or False

        def init(package, root, *args, **kwargs):
            import hashlib
            from pathlib import Path
            # All callers use public weights. The environment token is the
            # only allowed HF credential; a saved login is never loaded.
            if root.startswith("hf://"):
                repository, _, subpath = root[5:].partition("/")
                project, sep, tail = subpath.partition("/")
                repository += "/" + project
                if "@" in repository:
                    repository, revision = repository.split("@", 1)
                else:
                    revision = "main"
                revision = HfApi(token=token).model_info(repository, revision=revision).sha
                root = "hf://" + repository + "@" + revision + ("/" + tail if sep else "")
                kwargs["fs_options"] = dict(kwargs.get("fs_options") or {}, token=token)
            if root.startswith("ngc://"):
                kwargs["fs_options"] = dict(kwargs.get("fs_options") or {}, authenticated_api=False)
            if root.startswith(("gs://", "gcs://")):
                kwargs["fs_options"] = dict(kwargs.get("fs_options") or {}, token="anon", version_aware=True)
            if "://" in root:
                # Some loaders request same_names=True. Their fixed basename
                # must never alias a different resolved package revision.
                options = dict(kwargs.get("cache_options") or {})
                identity = hashlib.sha256(root.encode()).hexdigest()
                options["cache_storage"] = str(Path(cache) / "packages" / identity)
                kwargs["cache_options"] = options
            audit.original_init(package, root, *args, **kwargs)

        def opened(package, name):
            revision = None
            if str(package.root).startswith(("gs://", "gcs://")):
                identity = (str(package.root).split("://", 1)[1].rstrip("/") + "/" + name)
                if (audit.expected.get("revision_kind") == "object_generation"
                        and identity == audit.expected.get("repository")):
                    revision = str(audit.expected["revision"])
                    name += "#" + revision
            stream = audit.original_open(package, name)
            path = str(stream.name)
            key = (str(package.root), str(name))
            if key not in audit.assets:
                audit.assets[key] = {"root": str(package.root), "asset": str(name),
                                     "revision": revision or (str(package.root).split("@", 1)[1].split("/", 1)[0]
                                     if "@" in str(package.root) else None)}
                audit.pending[key] = audit.executor.submit(_sha256, path)
            return stream

        Package.__init__ = init
        Package.open = opened

    def finish(self):
        from earth2studio.models.auto import Package
        try:
            assets = [dict(self.assets[key], **future.result())
                      for key, future in self.pending.items()]
            for expected in self.expected.get("files", ()):
                name = expected.get("path", expected.get("name"))
                matching = [asset for asset in assets if asset["asset"] == name]
                for asset in matching:
                    if expected.get("sha256") and asset["sha256"] != expected["sha256"]:
                        raise ValueError(f"Weight sha256 differs from the model row: {name}")
            for asset in assets:
                if not asset["revision"]:
                    asset["revision"] = asset["sha256"]
                    asset["revision_kind"] = "content_sha256"
            return assets
        finally:
            self.executor.shutdown(wait=True)
            Package.__init__ = self.original_init
            Package.open = self.original_open


def annotate_output(store, attributes):
    """Publish the producer's units declared by the selected profile mapping.

    Earth2Studio supplies numeric coordinates and vocabulary but leaves units
    absent. Explicit CF metadata lets independent native readers validate the
    output, including coordinate readers that do not apply field overrides.
    """
    for name, values in attributes.items():
        if name in store.root.variables:
            store.root[name].setncatts(values)



def _frame_worker_init():
    # Run before task unpickling imports numpy: all cores belong to the
    # process pool, rather than to a BLAS pool inside every writer.
    import os
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"


def _write_frame(path, arrays, coords, variables, member, seed, attributes=None, geolocation=None):
    # Each process owns an independent HDF5 file. NetCDF is not thread safe.
    import os
    begin, cpu_begin = time.time(), time.process_time()
    os.environ["OMP_NUM_THREADS"] = "1"
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from earth2studio.io import NetCDF4Backend
    coords = OrderedDict(coords)
    for key, values in coords.items():
        if np.issubdtype(values.dtype, np.timedelta64):
            units = (attributes or {}).get(key, {}).get("units", "hours")
            code = {"hours": "h", "seconds": "s", "minutes": "m",
                    "days": "D", "years": "Y"}[units]
            converted = values.astype(f"timedelta64[{code}]")
            if not np.all(converted == values):
                raise ValueError(f"NetCDF forecast coordinate {key} needs whole {units}; fractional leads cannot be rounded")
            # Match the native profile contract; E2S has no ns units_map entry.
            coords[key] = converted
    io = NetCDF4Backend(path, backend_kwargs={"mode": "w"})
    try:
        io.add_array(coords, variables)
        annotate_output(io, attributes or {})
        io.write([torch.from_numpy(value) for value in arrays], coords, variables)
        for name, field in (geolocation or {}).items():
            variable = io.root.createVariable(name, "f8", tuple(field["axes"]))
            variable[:] = field["values"]
            variable.units = field["units"]
        io.root.setncattr("earth2_member", member)
        io.root.setncattr("earth2_generation_seed", seed)
    finally:
        io.close()
    return dict(_sha256(path), member=member, seed=seed, worker_pid=os.getpid(),
                started_unix=begin, seconds=round(time.time() - begin, 3),
                cpu_seconds=round(time.process_time() - cpu_begin, 3),
                lead_hours=int(coords["lead_time"][-1] / __import__("numpy").timedelta64(1, "h")))


class ParallelFrames:
    """Earth2Studio IO backend, one member/lead per independent worker file."""
    def __init__(self, output, members, seed, workers, seeds=None, attributes=None,
                 geolocation=None, member_outputs=None):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        from pathlib import Path
        self.attributes = attributes or {}
        self.member_outputs = member_outputs
        self.output = Path(output)
        self.members = members
        self.seed = seed
        self.seeds = seeds or [seed] * len(members)
        self.geolocation = geolocation
        self.executor = ProcessPoolExecutor(max_workers=workers,
                                            mp_context=multiprocessing.get_context("spawn"),
                                            initializer=_frame_worker_init)
        self.pending = []
        self.workers = workers

    def add_array(self, coords, variables, **kwargs):
        self.coords = coords

    def write(self, arrays, coords, variables):
        import numpy as np
        from collections import OrderedDict
        names = [variables] if isinstance(variables, str) else list(variables)
        values = [arrays] if not isinstance(arrays, list) else arrays
        # The tensors have already been cropped by Earth2Studio on the GPU.
        values = [value.detach().float().cpu().numpy() if hasattr(value, "detach") else value
                  for value in values]
        axis = list(coords).index("ensemble")
        for index, ensemble_index in enumerate(coords["ensemble"]):
            member = self.members[int(ensemble_index)]
            frame_coords = OrderedDict((key, value) for key, value in coords.items()
                                       if key != "ensemble")
            lead = int(frame_coords["lead_time"][-1] / np.timedelta64(1, "h"))
            path = self.output.with_name(self.output.stem + f"-m{member:04d}-f{lead:04d}.nc")
            if self.member_outputs:
                stem = self.output.__class__(self.member_outputs[int(ensemble_index)])
                path = stem.with_name(stem.stem + f"-f{lead:04d}.nc")
            frame = [np.take(value, index, axis=axis) for value in values]
            self.pending.append(self.executor.submit(_write_frame, str(path), frame,
                                                      frame_coords, names, member, self.seeds[int(ensemble_index)],
                                                      self.attributes, self.geolocation))
        # Bound queued host arrays while the GPU advances. Workers already
        # writing independent files can keep pace without retaining a run.
        if len(self.pending) > self.workers * 2:
            self.pending[-self.workers * 2 - 1].result()

    def close(self):
        try:
            return [future.result() for future in self.pending]
        finally:
            self.executor.shutdown(wait=True)


def main(argv) -> int:
    import os
    # Growable segments keep the decoder allocation beside live tensors.
    # Preserve an allocator setting supplied by the caller.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from pathlib import Path
    from importlib.util import module_from_spec, spec_from_file_location
    # Load the shared opt-out contract without importing the engine package
    # into the isolated Earth2Studio environment. This also guards direct use.
    spec = spec_from_file_location("_gpuwm_local_gpu", Path(__file__).with_name("local_gpu.py"))
    local_gpu = module_from_spec(spec)
    spec.loader.exec_module(local_gpu)
    if local_gpu.no_local_gpu():
        raise ValueError(f"{local_gpu.NO_LOCAL_GPU_ENV} is set; Earth2 generation cannot open the local GPU")
    from importlib.metadata import version
    request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    if version("earth2studio") != request["pinned_version"]:
        raise ValueError("Earth2Studio version differs from the pinned runtime table")
    # Configure allocation before importing Torch or initializing CUDA.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    workers = _cpu_count()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(workers)
    os.environ.update({key: str(value) for key, value in request.get("runtime_env", {}).items()})
    os.environ["EARTH2STUDIO_MODEL_CACHE"] = request["weight_cache"]
    os.environ["EARTH2STUDIO_CACHE"] = request["data_cache"]
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    import numpy as np
    import torch
    import earth2studio.data as data
    import earth2studio.models.px as px
    import earth2studio.run as run
    from earth2studio.perturbation import Zero

    if not torch.cuda.is_available():
        raise RuntimeError("Earth2 generation requires a GPU; no CUDA device was admitted")
    torch.set_num_threads(workers)
    seed = int(request["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    torch.cuda.manual_seed_all(seed)
    patched = False
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) not in ((9, 0), (10, 0)):
        patched = _band_contracted_disco()
    audit = PackageAudit(request["weight_cache"], workers, request.get("expected_weights"))
    started = time.perf_counter()
    stages = []
    def stage(name, begin, parallelism):
        stages.append({"stage": name, "started_unix": time.time() - (time.perf_counter() - begin),
                       "seconds": round(time.perf_counter() - begin, 3), "parallelism": parallelism})
    io = None
    try:
        model_class = getattr(px, request["model_class"])
        local_root = request.get("package_root") or request.get("weights_package")
        if local_root:
            from earth2studio.models.auto import Package
            package = Package(str(local_root), cache=False)
        else:
            package = model_class.load_default_package()
            if request.get("model_package"):
                from earth2studio.models.auto import Package
                package = Package(request["model_package"], cache_options=package.cache_options)
        kwargs = {key: seed if value == "$seed" else value
                  for key, value in request.get("loader_kwargs", {}).items()}
        import fcntl
        with open(Path(request["weight_cache"]) / "loader-inputs.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            loader_inputs = prepare_loader_inputs(request)
            model = model_class.load_model(package, **kwargs)
        for call in request.get("model_calls", []):
            kwargs = {key: seed if value == "$seed" else value
                      for key, value in call.get("kwargs", {}).items()}
            getattr(model, call["method"])(**kwargs)
        model = model.to(torch.device("cuda"))
        loaded = time.perf_counter()
        stage("model_load", started, f"{workers} threads / network")
        source = getattr(data, request["ic_class"])(**request.get("ic_kwargs", {}))
        output_coords = OrderedDict()
        area = request.get("area")
        # Use the model's declared output grid, including 720-row or regional
        # grids. No model-specific grid guesses belong in this runner.
        native = model.output_coords(model.input_coords())
        grid = request.get("output_grid")
        geolocation = {}
        if grid:
            # Regional grids retain the model's declared native window.
            # Cropping by a geographic rectangle would change its projection
            # window and invalidate the packaged source coordinates.
            axes = [grid["y"], grid["x"]]
            shape = tuple(len(native[axis]) for axis in axes)
            for kind, unit in (("latitude", "degrees_north"), ("longitude", "degrees_east")):
                values = getattr(model, grid[kind + "_attribute"])
                if hasattr(values, "detach"):
                    values = values.detach().cpu().numpy()
                values = np.asarray(values)
                if values.shape != shape:
                    raise ValueError(f"{kind} geolocation shape differs from declared output axes")
                geolocation[grid.get(kind + "_variable", "lat" if kind == "latitude" else "lon")] = {
                    "axes": axes, "values": values, "units": unit}
        elif area:
            south, west, north, east = (float(value) for value in area)
            lat, lon = native["lat"], native["lon"]
            output_coords["lat"] = lat[(lat >= south) & (lat <= north)]
            output_coords["lon"] = lon[(lon >= west) & (lon <= east)]
        timings = {"inference_seconds": 0., "initial_data_seconds": 0., "initial_data_calls": 0}
        original_fetch = run.fetch_data
        def timed_fetch(*args, **kwargs):
            begin = time.perf_counter()
            try:
                return original_fetch(*args, **kwargs)
            finally:
                timings["initial_data_seconds"] += time.perf_counter() - begin
                timings["initial_data_calls"] += 1
        run.fetch_data = timed_fetch
        original_iterator = model.create_iterator
        def timed_iterator(*args, **kwargs):
            iterator = original_iterator(*args, **kwargs)
            while True:
                torch.cuda.synchronize()
                begin = time.perf_counter()
                try:
                    value = next(iterator)
                except StopIteration:
                    return
                torch.cuda.synchronize()
                timings["inference_seconds"] += time.perf_counter() - begin
                yield value
        model.create_iterator = timed_iterator
        policy = request.get("member_policy")
        perturbation = (member_perturbation(model, policy, request["seeds"])
                        if policy else Zero())
        # Release temporary loader blocks so the query includes reusable space.
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        batch_size, batch_memory = select_batch_size(request, model.input_coords(),
            free_bytes=free_bytes, total_bytes=total_bytes)
        io = ParallelFrames(request["output"], request["members"], seed, workers,
                            request.get("seeds"), request.get("output_attributes"),
                            geolocation, request.get("member_outputs"))
        run.ensemble([request["init"]], int(request["steps"]), len(request["members"]),
                     model, source, io, perturbation, batch_size=batch_size,
                     output_coords=output_coords, device=torch.device("cuda"))
        ran = time.perf_counter()
        stage("initial_conditions_and_rollout", loaded, "GPU / " + str(workers) + " threads")
        closing, io = io, None
        files = closing.close()
        stage("staging_tail", ran, f"{workers} processes")
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        if io is not None:
            try:
                io.close()
            except Exception as error:
                cleanup_errors.append(error)
        try:
            assets = audit.finish()
            Path(request["output"]).with_suffix(".weights.json").write_text(
                json.dumps(assets, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is None:
                raise cleanup_errors[0]
            for error in cleanup_errors:
                primary_error.add_note(f"Earth2 cleanup also failed: {error!r}")
    finished = time.perf_counter()
    print(json.dumps({
        "schema": "gpuwm-earth2-run-v2", "stages": stages,
        "load_seconds": round(loaded - started, 3),
        "run_seconds": round(finished - loaded, 3),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gpu_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "band_contracted_disco": patched,
        "earth2studio": version("earth2studio"), "torch": torch.__version__,
        "members": request["members"], "seed": seed, "seeds": request.get("seeds"),
        "seed_scope": ("per-member replay seeds via row native RNG or perturbation protocol" if policy
                       else "one GPU RNG stream for the ordered member batch"),
        "workflow_calls": 1,
        "batch_size": batch_size, "batch_memory": batch_memory, "files": files,
        **{key: round(value, 3) for key, value in timings.items()},
        "loader_inputs": loader_inputs,
        "weight_cache": request["weight_cache"], "weights": assets,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
