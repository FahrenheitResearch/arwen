"""NextGen hooks for a mass worker and its full-roster GPU product pass."""
from pathlib import Path

from gpuwm.ensemble.nextgen_points import ReachAccumulator, ReachSample, product_options
from gpuwm.ensemble.nextgen_inventory import ResidentInventory
from gpuwm.ensemble.optional_routing import read_product_options


class MemberReachCapture:
    """Wrap the launcher's synchronous thin atmosphere callback.

    The worker owns one original forecast/driver at a time. Call seal() only
    after that forecast succeeds, then attach its document to the member
    receipt under nextgen_points. A failed member cannot publish a shard.
    """
    def __init__(self, callback, *, member_id, directory, options=None, array_module=None,
                 depth_sink=None, channel_depth_sink=None):
        self.callback, self.member_id, self.directory = callback, member_id, Path(directory)
        self.points = ReachAccumulator((member_id,), options=options,
            array_module=array_module, depth_sink=depth_sink)
        self.resident_inventory = ResidentInventory()
        self.channel_depth_sink = channel_depth_sink
        from gpuwm.ensemble.nextgen_depth import ChannelDepthArchive
        self.channel_depth_archive = ChannelDepthArchive((member_id,), self.directory,
            array_module=array_module)

    def _capture_channel_depth(self, **row):
        self.channel_depth_archive.accept(**row)
        if self.channel_depth_sink is not None:
            return self.channel_depth_sink(**row)

    def __getattr__(self, name):
        return getattr(self.callback, name)

    def __call__(self, *, state, valid_time, grid_id, member_id, **kwargs):
        if member_id != self.member_id:
            raise ValueError("mass NextGen capture belongs to another member")
        if int(grid_id) == 1:
            driver = getattr(getattr(state, "physics", None), "hydro", None)
            if driver is None:
                raise RuntimeError("mass member has no resident NextGen driver")
            cp = self.points.cp
            if cp is None:
                import cupy as cp
            self.resident_inventory.inspect(driver, member_id=member_id,
                device_id=int(cp.cuda.runtime.getDevice()))
            from gpuwm.ensemble.nextgen_capture import submit_reach_output
            submit_reach_output(self, member_id=member_id, valid_time=valid_time,
                                sample=driver.reach_output(), driver=driver)
            from gpuwm.ensemble.nextgen_depth import submit_channel_depth
            submit_channel_depth(driver, member_id=member_id, valid_time=valid_time,
                array_module=self.points.cp, sink=self._capture_channel_depth)
        return self.callback(state=state, valid_time=valid_time, grid_id=grid_id,
                             member_id=member_id, **kwargs)

    def submit_hydro(self, *, member_id, valid_time, sample=None, reach_ids=None,
                     discharge=None, stream=None, channel_depth_m=None):
        if member_id != self.member_id:
            raise ValueError("mass NextGen capture belongs to another member")
        if sample is None:
            sample = ReachSample(reach_ids, discharge, stream=stream,
                                 channel_depth_m=channel_depth_m)
        self.points.accept(member_id=member_id, valid_time=valid_time, sample=sample)

    def seal(self):
        self.points.require_complete()
        document = self.points.save(self.directory)
        document["resident_inventory"] = self.resident_inventory.receipt()
        document["channel_depth"] = self.channel_depth_archive.snapshot()
        self.points.closed = True
        return document


def reduce_members(root, members, *, expected_member_ids, options=None, array_module=None):
    """Publish only the declared roster, after every worker has completed."""
    root = Path(root)
    points = ReachAccumulator(expected_member_ids, options=options, array_module=array_module)
    from gpuwm.ensemble.nextgen_depth import ChannelDepthArchive
    depths = ChannelDepthArchive(expected_member_ids, root, array_module=array_module)
    rows = {row["member_id"]: row for row in members}
    if len(rows) != len(members) or set(rows) != set(points.members):
        raise ValueError("mass NextGen reduction needs the full requested roster exactly once")
    for member in points.members:
        row = rows[member]
        if row.get("status") != "completed" or "nextgen_points" not in row:
            raise ValueError("mass NextGen reduction needs every completed member's sealed shard")
        # Explicit relative directory comes from the launcher's member receipt.
        from gpuwm.ensemble.restart_roster import resolve_file
        directory = resolve_file(root, row["nextgen_directory"])
        points.merge_member(directory, row["nextgen_points"], member)
        if "channel_depth" in row["nextgen_points"]:
            depths.import_samples(directory, row["nextgen_points"]["channel_depth"], expected_members=(member,))
    products = points.finish(root / "nextgen") + depths.finish(points.frames)
    return {"products": products, "reduction": points.receipt(),
            "channel_depth": depths.snapshot()}
