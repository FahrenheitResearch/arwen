"""Borrow an independently ordered channel-only depth axis at output cadence."""
from collections.abc import Mapping
from pathlib import Path
import threading
import time


def submit_channel_depth(driver, *, member_id, valid_time, array_module, sink):
    """Optional MC-only output; reservoir elevations never become bed depths.

    The sink consumes borrowed storage on the producer stream before reuse.
    The default archive drains that stream and writes through Rust at this
    output boundary. Full network discharge keeps its own axis; no Q/depth
    conversion or routing-step host capture is performed.
    """
    if sink is None:
        return
    provider = getattr(driver, "channel_output", None)
    sample = provider() if provider is not None else None
    if sample is None:
        return  # Optional capability; the archive retains missing roster slots.
    if not isinstance(sample, Mapping):
        raise ValueError("channel_output must be a cached mapping with explicit channel-only IDs")
    cp = array_module
    if cp is None:
        import cupy as cp
    ids, depth = sample.get("reach_id"), sample.get("channel_depth_m")
    if (not isinstance(ids, cp.ndarray) or not isinstance(depth, cp.ndarray)
            or ids.dtype != cp.dtype("int64") or ids.ndim != 1 or not ids.size
            or depth.dtype not in (cp.dtype("float32"), cp.dtype("float64"))
            or depth.shape != ids.shape or ids.device.id != depth.device.id
            or int(ids.device.id) != int(driver.device_id)
            or not ids.flags.c_contiguous or not depth.flags.c_contiguous):
        raise ValueError("channel_output needs same-card contiguous int64 IDs and native above-bed depth")
    with cp.cuda.Device(ids.device.id):
        stream = sample.get("stream") or cp.cuda.get_current_stream()
        with stream:
            return sink(member_id=member_id, valid_time=valid_time, feature_id=ids,
                        channel_depth_m=depth, stream=stream)


class ChannelDepthArchive:
    """Immutable native-precision depth files at output boundaries only.

    Rust writes the NetCDF; Python transports bytes and records metadata.
    The producer stream is drained here before the borrowed storage is reused.
    No depth history or host work is added to a routing step.
    """
    schema = "arwen.nextgen.channel-depth-ensemble/v1"

    def __init__(self, member_ids, root, *, array_module=None):
        self.members, self._root, self.cp = tuple(member_ids), root, array_module
        self.samples, self.restored = {}, set()
        self.lock = threading.RLock()
        self.closed = False

    @property
    def root(self):
        return Path(self._root() if callable(self._root) else self._root)

    def accept(self, *, member_id, valid_time, feature_id, channel_depth_m, stream):
        from gpuwm.io.classic_product import ClassicProduct
        from gpuwm.ensemble.restart_roster import digest_file, relative_file
        key = (str(valid_time), member_id)
        with self.lock:
            if self.closed or member_id not in self.members:
                raise ValueError("channel depth output is closed or outside its requested roster")
            if key in self.restored:
                self.restored.remove(key)
                return
            if key in self.samples:
                raise ValueError("duplicate member/time in channel depth output")
            cp = self.cp
            if cp is None:
                import cupy as cp
                self.cp = cp
            started = time.perf_counter()
            directory = self.root / "nextgen" / "channel-depth"
            directory.mkdir(parents=True, exist_ok=True)
            pending = directory / f"member-{member_id}.pending.nc"
            with cp.cuda.Device(feature_id.device.id), stream, ClassicProduct(pending) as writer:
                writer.setncatts(dict(schema=self.schema, valid_time=key[0], member_id=member_id,
                    depth_reference="local_channel_bed", units="m", channel_axis="independent_of_discharge"))
                writer.createDimension("channel", feature_id.size)
                writer.createDimension("word", 2)
                for name, value, dims, units in (
                    ("reach_id", feature_id, ("channel",), "1"),
                    ("reach_id_words", feature_id.view(cp.uint32).reshape(-1, 2), ("channel", "word"), "1"),
                    ("channel_depth_m", channel_depth_m, ("channel",), "m")):
                    variable = writer.createVariable(name, value.dtype, dims)
                    variable.units = units
                    variable[:] = cp.asnumpy(value, stream=stream)  # byte transport, no Python arithmetic
            sha = digest_file(pending)
            path = directory / f"{sha}.nc"
            pending.replace(path)
            self.samples[key] = dict(member_id=member_id, valid_time=key[0],
                source=dict(path=relative_file(self.root, path), sha256=sha, variable="channel_depth_m",
                    dtype=channel_depth_m.dtype.str, shape=list(channel_depth_m.shape),
                    reach_ids_variable="reach_id", exact_reach_id_words_variable="reach_id_words",
                    reach_ids_dtype="<i8", depth_reference="local_channel_bed", units="m"),
                transport_rust_write_ms=(time.perf_counter()-started)*1000)

    def snapshot(self):
        with self.lock:
            return dict(schema=self.schema, requested_member_ids=list(self.members),
                samples=[self.samples[key] for key in sorted(self.samples)])

    def import_samples(self, root, document, *, expected_members, replay=False):
        from gpuwm.ensemble.restart_roster import digest_file, resolve_file, relative_file
        if (document.get("schema") != self.schema
                or document.get("requested_member_ids") != list(expected_members)):
            raise ValueError("channel depth checkpoint roster/schema differs")
        with self.lock:
            for row in document["samples"]:
                key = (row["valid_time"], row["member_id"])
                if key in self.samples or key[1] not in expected_members or key[1] not in self.members:
                    raise ValueError("duplicate or foreign member in channel depth checkpoint")
                source = row["source"]
                path = resolve_file(root, source["path"])
                if not path.is_file() or digest_file(path) != source["sha256"]:
                    raise ValueError("channel depth file hash mismatch")
                self.samples[key] = dict(row, source=dict(source, path=relative_file(self.root, path)))
                if replay:
                    self.restored.add(key)

    def finish(self, frames):
        """Publish a full roster, including explicit unavailable sources.

        A null source masks every HAND cell for that deadline. It must never
        be removed from a renderer's denominator; nonfinite per-reach values
        remain unmodified in the files for the Rust complete-roster mask.
        """
        from gpuwm.ensemble.restart_roster import atomic_json
        with self.lock:
            if not self.samples:
                return []
            rows = []
            for key in sorted(frames):
                members = [dict(member_id=member,
                    source=self.samples.get((key, member), {}).get("source")) for member in self.members]
                rows.append(dict(valid_time=key, members=members,
                    missing_member_ids=[row["member_id"] for row in members if row["source"] is None],
                    missing_member_action="mask_all_cells"))
            path = self.root / "nextgen" / "channel-depth-ensemble.json"
            atomic_json(path, dict(schema=self.schema, root="..",
                requested_member_ids=list(self.members), comparison="gt", units="m",
                depth_reference="local_channel_bed", missing_policy="complete_roster",
                frame_axis="per_member_independent_reach_ids; explicit_join_required",
                frames=rows, samples=self.snapshot()["samples"]))
            self.closed = True
            return [str(self.root / row["source"]["path"]) for row in self.samples.values()] + [str(path)]
