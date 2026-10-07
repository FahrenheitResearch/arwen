"""Byte-transfer geometry for the fire registry's refined horizontal fields."""
from __future__ import annotations


def fire_field_layout(name, shape, ny, nx):
    """Return (ry, rx, py, px) only for native fire-grid carrier names.

    A refined fire plane includes one continuation cell on each side. This
    is unrelated to an atmospheric face stagger, so it must not be inferred
    through the ordinary mass/u/v shape rules.
    """
    fine = (name.startswith("fire/grid.") and not name.startswith("fire/grid.moisture."))
    fine = fine or name in ("fire/fz0", "fire/fxlat", "fire/fxlong")
    if not fine:
        return None
    if len(shape) != 2:
        raise ValueError(f"native fine fire field {name} must be a horizontal plane")
    y, x = map(int, shape)
    ry, sy = divmod(y - 2, int(ny))
    rx, sx = divmod(x - 2, int(nx))
    if min(ry, rx) < 1 or sy or sx:
        raise ValueError(f"native fine fire field {name} has unbound refinement {shape} on {(ny, nx)}")
    return ry, rx, 1, 1


def fire_transfer_segments(spec, layout, direction):
    """Physical fire cells and their continuation halo, never periodic aliases.

    The atmospheric compute window may wrap, while WRF's fire domain has
    physical boundaries. Only the intersection with that physical domain
    and its native one-cell halo is used by a tile's fire integration.
    """
    ry, rx, py, px = layout
    fy, fx = spec.ny * ry, spec.nx * rx
    if direction == "gather":
        ay, ax = spec.cj0 * ry, spec.ci0 * rx
        y0, y1 = max(0, ay), min(fy + 2 * py, ay + spec.cny * ry + 2 * py)
        x0, x1 = max(0, ax), min(fx + 2 * px, ax + spec.cnx * rx + 2 * px)
        if y0 >= y1 or x0 >= x1:
            raise ValueError("atmospheric tile has no physical fire cells")
        return [( (slice(y0, y1), slice(x0, x1)),
                  (slice(y0 - ay, y1 - ay), slice(x0 - ax, x1 - ax)) )]
    if direction != "scatter":
        raise ValueError("fire transfer direction must be gather or scatter")
    # Only the tile owning a physical boundary writes its continuation cell.
    y0 = py + spec.j0 * ry - (py if spec.j0 == 0 else 0)
    y1 = py + spec.j1 * ry + (py if spec.j1 == spec.ny else 0)
    x0 = px + spec.i0 * rx - (px if spec.i0 == 0 else 0)
    x1 = px + spec.i1 * rx + (px if spec.i1 == spec.nx else 0)
    return [( (slice(y0 - spec.cj0 * ry, y1 - spec.cj0 * ry),
               slice(x0 - spec.ci0 * rx, x1 - spec.ci0 * rx)),
              (slice(y0, y1), slice(x0, x1)) )]


def fire_wind_segments(spec, name, direction):
    """Native fire wind terminal faces are physical, even with periodic air."""
    ey, ex = (0, 1) if name == "fire/uah" else (1, 0)
    if direction == "gather":
        y0, y1 = max(0, spec.cj0), min(spec.ny + ey, spec.cj0 + spec.cny + ey)
        x0, x1 = max(0, spec.ci0), min(spec.nx + ex, spec.ci0 + spec.cnx + ex)
        return [((slice(y0, y1), slice(x0, x1)),
                 (slice(y0-spec.cj0, y1-spec.cj0), slice(x0-spec.ci0, x1-spec.ci0)))]
    y0, y1 = spec.j0, spec.j1 + (ey if spec.j1 == spec.ny else 0)
    x0, x1 = spec.i0, spec.i1 + (ex if spec.i1 == spec.nx else 0)
    return [((slice(y0-spec.cj0, y1-spec.cj0), slice(x0-spec.ci0, x1-spec.ci0)),
             (slice(y0, y1), slice(x0, x1)))]
