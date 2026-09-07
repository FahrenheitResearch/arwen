"""Small horizontal-axis requirements shared by authors and their consumers.

These are operation bounds, not recommended forecast extents. Consumers retain
selection, type, and boundary-condition validation at their existing call sites.
"""

# Fifth-order horizontal advection uses a seven-point stencil. Open-boundary
# degrade bands also require seven cells so the two edge bands cannot overlap.
FIFTH_ORDER_STENCIL_AXIS = 7

# The native target constructor retains three cells between specified edges.
# This declaration is shared with its source adapter; it is not imposed on
# target producers or children which do not use that constructor.
NATIVE_TARGET_INTERIOR_AXIS = 3


def boundary_axis(width: int, *, interior_points: int = 0) -> int:
    """Cells occupied by both boundary strips plus the requested interior.

    Tables may abut (zero interior); a unique frame or health reduction needs
    one interior cell. A target producer may require a larger interior.
    """
    return 2 * width + interior_points
