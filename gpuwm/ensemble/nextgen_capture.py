"""Optional NextGen orchestration at ordinary member output deadlines."""
from contextlib import contextmanager
from collections.abc import Mapping
import json
from pathlib import Path
import threading

from gpuwm.ensemble.nextgen_points import ReachAccumulator, ReachSample
from gpuwm.ensemble.optional_routing import (
    _file_settings, member_settings, enabled_for, member_settings_scope,
)
from gpuwm.ensemble.nextgen_inventory import ResidentInventory


def submit_reach_output(owner, *, member_id, valid_time, sample, driver=None):
    """Use the member capture API once at the existing root output deadline."""
    check_health = getattr(driver, "check_health", None)
    if check_health is not None:
        stream = (sample.get("stream") if isinstance(sample, Mapping)
                  else getattr(sample, "stream", None))
        check_health(stream=stream)
    from gpuwm.ensemble.runtime_context import current_capture
    capture = current_capture()
    if capture is not None and getattr(capture.callback, "__self__", capture.callback) is owner:
        if capture.member_id != member_id:
            raise ValueError("hydro deadline belongs to another ensemble member")
        if isinstance(sample, Mapping):
            sample = ReachSample(**{key: sample.get(key) for key in
                ("reach_id", "discharge_m3_s", "channel_depth_m", "stream")})
        return capture.submit_hydro(valid_time=valid_time, reach_ids=sample.reach_id,
            discharge=sample.discharge_m3_s, stream=sample.stream,
            channel_depth_m=sample.channel_depth_m)
    # Standalone qualification tools may use the collector without a forecast
    # capture context. They take the same single output-boundary path.
    return owner.submit_hydro(member_id=member_id, valid_time=valid_time, sample=sample)


class NextGenCollector:
    """Keep atmosphere and reach captures on one checkpoint transaction."""
    def __init__(self, original, member_ids, *, options=None, array_module=None, depth_sink=None,
                 channel_depth_sink=None):
        self.original = original
        self.nextgen_points = ReachAccumulator(member_ids, options=options,
            array_module=array_module, depth_sink=depth_sink,
            member_weights=getattr(original, "member_weights", None))
        self.resident_inventory = ResidentInventory()
        self.channel_depth_sink = channel_depth_sink
        from gpuwm.ensemble.nextgen_depth import ChannelDepthArchive
        self.channel_depth_archive = ChannelDepthArchive(self.nextgen_points.members, lambda: self.root,
            array_module=array_module)
        self._capture_lock = threading.RLock()

    def _capture_channel_depth(self, **row):
        self.channel_depth_archive.accept(**row)
        if self.channel_depth_sink is not None:
            return self.channel_depth_sink(**row)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def bind_member(self, *, member_id, model=None, state=None, cfg=None, clock=None, **unused):
        """Check initialized parent storage before its first forecast step."""
        grid_id = (clock.spec.grid_id if clock is not None else getattr(cfg, "grid_id", 1))
        if int(grid_id) != 1:
            return
        if model is not None:
            state = model.root.state
        driver = getattr(getattr(state, "physics", None), "hydro", None)
        if driver is None:
            raise RuntimeError("NextGen ensemble member has no resident parent driver")
        cp = self.nextgen_points.cp
        if cp is None:
            import cupy as cp
        return self.resident_inventory.inspect(driver, member_id=member_id,
            device_id=int(cp.cuda.runtime.getDevice()))

    @contextmanager
    def checkpoint_hold(self):
        with self._capture_lock:
            hold = getattr(self.original, "checkpoint_hold", None)
            if hold is None:
                yield
            else:
                with hold():
                    yield

    def submit(self, *, state, valid_time, grid_id, member_id, **kwargs):
        with self._capture_lock:
            # Hydro is fixed to the parent. Child output never recaptures it.
            if int(grid_id) == 1:
                driver = getattr(getattr(state, "physics", None), "hydro", None)
                if driver is None:
                    raise RuntimeError("NextGen ensemble member has no resident parent driver")
                self.bind_member(member_id=member_id, state=state)
                sample = driver.reach_output()
                submit_reach_output(self, member_id=member_id, valid_time=valid_time,
                                    sample=sample, driver=driver)
                from gpuwm.ensemble.nextgen_depth import submit_channel_depth
                submit_channel_depth(driver, member_id=member_id, valid_time=valid_time,
                    array_module=self.nextgen_points.cp, sink=self._capture_channel_depth)
            return self.original.submit(state=state, valid_time=valid_time,
                grid_id=grid_id, member_id=member_id, **kwargs)

    def submit_hydro(self, *, member_id, valid_time, sample=None, reach_ids=None,
                     discharge=None, stream=None, channel_depth_m=None):
        if sample is None:
            sample = ReachSample(reach_ids, discharge, stream=stream,
                                 channel_depth_m=channel_depth_m)
        with self._capture_lock:
            self.nextgen_points.accept(member_id=member_id, valid_time=valid_time, sample=sample)

    def require_complete(self):
        self.original.require_complete()
        self.nextgen_points.require_complete()

    def finish_run(self):
        with self._capture_lock:
            self.nextgen_points.require_complete()
            atmosphere = self.original.finish_run()
            hydro = (self.nextgen_points.finish(Path(self.root) / "nextgen")
                + self.channel_depth_archive.finish(self.nextgen_points.frames))
            if isinstance(atmosphere, Mapping):
                return {**atmosphere, "nextgen": hydro}
            return list(atmosphere) + hydro

    def save_resume(self, *args, **kwargs):
        from gpuwm.ensemble.restart_roster import atomic_json, relative_file
        with self.checkpoint_hold():
            document = self.original.save_resume(*args, **kwargs)
            document["nextgen_points"] = self.nextgen_points.save(self.root)
            document["nextgen_channel_depth"] = self.channel_depth_archive.snapshot()
            document["files"] = sorted(set(document["files"]) | {
                document["nextgen_points"]["arrays"],
                *(row["source"]["path"] for row in document["nextgen_channel_depth"]["samples"]),
                *(relative_file(Path(self.root), Path(path)) for path in self.nextgen_points.products or ())})
            atomic_json(Path(self.root) / ".ensemble-resume" / "collector.json", document)
            return document

    def restore_resume(self):
        with self.checkpoint_hold():
            document = json.loads((Path(self.root) / ".ensemble-resume" / "collector.json").read_text())
            if "nextgen_points" not in document:
                raise ValueError("NextGen continuation needs its sealed point-product state")
            self.nextgen_points.restore(self.root, document["nextgen_points"])
            if "nextgen_channel_depth" in document:
                self.channel_depth_archive.import_samples(self.root, document["nextgen_channel_depth"],
                    expected_members=self.nextgen_points.members, replay=True)
            return self.original.restore_resume()
