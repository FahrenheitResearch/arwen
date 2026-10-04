"""Native precipitation packs with explicit accumulation intervals.

One reader serves radar, gauge-derived grids and satellite sources. Their
decoders own units, masks and conversions; this adapter checks interval and
geometry metadata and exposes the existing observation-scoring protocol.
"""
from __future__ import annotations

from pathlib import Path

from gpuwm.obs.obspack import read_grid_pack, read_pack
from gpuwm.obs.sources import _GriddedSource, _parse_time


class PrecipitationSource(_GriddedSource):
    """A fixed-duration or fixed-start precipitation observation timeline.

    Matching is exact to the second. A neighboring accumulation is a different
    quantity even if its ending time is nearby; it cannot fill a missing hour.
    """

    def __init__(self, pack_paths, geo_pack, *, accumulation_seconds=None,
                 accumulation_start=None, minimum_observed_fraction=None):
        paths = tuple(Path(path) for path in pack_paths)
        if accumulation_seconds is None and accumulation_start is None:
            raise ValueError("precipitation matching needs a fixed duration or fixed start")
        if accumulation_seconds is not None and (
                isinstance(accumulation_seconds, bool)
                or not isinstance(accumulation_seconds, int)
                or accumulation_seconds <= 0):
            raise ValueError("precipitation duration must be a positive whole second count")
        start = None if accumulation_start is None else _parse_time(accumulation_start)
        geometry = read_pack(geo_pack)
        if geometry.meta.get("schema") != "gpuwm-obs.obs-geo.v1":
            raise ValueError("precipitation coordinates need the native geometry pack")
        geometry_digest = geometry.meta["content_sha256"]
        self.intervals = {}
        for path in paths:
            frame = read_grid_pack(path)
            meta = frame.meta
            begin = _parse_time(meta["accumulation_start"])
            end = _parse_time(meta["valid_time"])
            seconds = int((end - begin).total_seconds())
            if (seconds <= 0 or seconds != meta["accumulation_seconds"]
                    or (accumulation_seconds is not None and seconds != accumulation_seconds)
                    or (start is not None and begin != start)):
                raise ValueError(f"{path}: precipitation interval differs from the requested window")
            if meta["units"] != "mm" or meta.get("geometry_sha256") != geometry_digest:
                raise ValueError(f"{path}: precipitation units or native geometry identity differ")
            self.intervals[meta["valid_time"]] = meta["accumulation_start"]
        super().__init__(paths, geo_pack, quantity="precipitation_accumulation",
                         units="mm", match_seconds=1,
                         minimum_observed_fraction=minimum_observed_fraction)

    def field(self, valid_time):
        if str(valid_time) not in self.intervals:
            raise ValueError("no precipitation accumulation ends at the exact requested valid time")
        return super().field(valid_time)
