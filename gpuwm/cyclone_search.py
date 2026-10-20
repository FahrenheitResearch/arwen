"""Configured cyclone search bounds, independent of forecast model or tracker."""
import hashlib
import json
import math
from pathlib import Path

TABLE = Path(__file__).with_name("authorities") / "cyclone-search.v1.json"


def parse_area(value):
    """Validate south,west,north,east, retaining wrapped longitude bounds."""
    values = value.split(",") if isinstance(value, str) else value
    if len(values) != 4:
        raise ValueError("search area needs south,west,north,east")
    area = [float(v) for v in values]
    south, west, north, east = area
    if (not all(math.isfinite(v) for v in area) or
            not -90 <= south < north <= 90 or
            not 0 < abs(east - west) <= 360):
        raise ValueError("search area needs finite latitude bounds and a nonempty longitude span of at most 360 degrees")
    return area


def window(area):
    from gpuwm.source_coverage import RegularLatLonWindow
    south, west, north, east = parse_area(area)
    span = (east - west) % 360 or 360
    return RegularLatLonWindow(south, west, north, west + span)


def resolve_region(area=None, *, automatic=False):
    """Explicit job bounds win; automatic placement otherwise uses table data."""
    if area is not None:
        return dict(id="job-area", area=parse_area(area), basis="explicit job area")
    if not automatic:
        return dict(id="source-domain", area=None, basis="source coverage")
    raw = TABLE.read_bytes()
    table = json.loads(raw)
    name = table["default_auto_region"]
    row = table["regions"][name]
    return dict(id=name, area=parse_area(row["area"]), description=row["description"],
                basis="configured auto-search region", authority=dict(
                    file=TABLE.name, sha256=hashlib.sha256(raw).hexdigest()))
