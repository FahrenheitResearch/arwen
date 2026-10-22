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
    return dict(id=name, area=parse_area(row["area"]),
                **({"polygon_lon_lat": row["polygon_lon_lat"]} if "polygon_lon_lat" in row else {}),
                description=row["description"],
                basis="configured auto-search region", authority=dict(
                    file=TABLE.name, sha256=hashlib.sha256(raw).hexdigest()))


def contains(region, lat, lon):
    """Small track metadata geometry; boundary points are included."""
    from gpuwm.source_coverage import points_outside
    if region["area"] is None:
        return True
    if bool(points_outside(window(region["area"]), lat, lon)):
        return False
    polygon = region.get("polygon_lon_lat")
    if polygon is None:
        return True
    west = region["area"][1]
    x = (lon - west) % 360 + west
    vertices = [((px-west) % 360 + west, py) for px, py in polygon]
    inside = False
    for (ax, ay), (bx, by) in zip(vertices, vertices[1:] + vertices[:1]):
        cross = (x-ax)*(by-ay) - (lat-ay)*(bx-ax)
        if abs(cross) < 1e-9 and min(ax,bx) <= x <= max(ax,bx) and min(ay,by) <= lat <= max(ay,by):
            return True
        if (ay > lat) != (by > lat) and x < ax + (lat-ay)*(bx-ax)/(by-ay):
            inside = not inside
    return inside


def distance_to_region_km(region, lat, lon):
    """Approximate distance to the boundary, for ordering refusal diagnostics."""
    if contains(region, lat, lon):
        return 0.
    south, west, north, east = region["area"]
    vertices = region.get("polygon_lon_lat", [[west,south],[east,south],[east,north],[west,north]])
    # Local equirectangular projection suffices for ordering nearby alternatives;
    # this distance is not used to decide membership or cyclone strength.
    points = [(((x-lon+180)%360-180)*math.cos(math.radians(lat)), y-lat) for x,y in vertices]
    distances = []
    for (ax,ay),(bx,by) in zip(points, points[1:] + points[:1]):
        dx,dy = bx-ax,by-ay
        t = max(0.,min(1.,-(ax*dx+ay*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0.
        distances.append(math.hypot(ax+t*dx,ay+t*dy)*111.195)
    return min(distances)
