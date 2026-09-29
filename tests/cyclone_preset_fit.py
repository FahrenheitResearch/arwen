"""Whether a source's declared window can hold the cyclone preset's root.

Shared by the cyclone door tests that walk every source.  A regional source
smaller than the 200x160 root at 12 km (ICON-D2 covers about 1,200 by
1,500 km) cannot carry it anywhere: the door refuses it by name at every
centre and lists the sources that can, so no configuration is emitted for
the other checks to read.  Both walks decide that from the row's own
window, never from a source name.
"""
from __future__ import annotations

import math

from gpuwm import cyclone_setup as tc
from gpuwm.source_adapters import get_source_adapter
from gpuwm.source_coverage import window_centre

KM_PER_DEGREE = 111.2


def center(source):
    """The middle of the source's declared window, or the global test point."""
    window = get_source_adapter(source).coverage_window
    return window_centre(window) if window is not None else (18., -65.)


def holds_the_preset_root(source):
    """Whether the source's declared window is as large as the preset root."""
    window = get_source_adapter(source).coverage_window
    if window is None:
        return True
    south, west, north, east = window.envelope()
    across = ((east - west) * KM_PER_DEGREE
              * math.cos(math.radians(0.5 * (south + north))))
    along = (north - south) * KM_PER_DEGREE
    return (across >= tc.ROOT_DIMS[0] * tc.ROOT_DX_M / 1000.
            and along >= tc.ROOT_DIMS[1] * tc.ROOT_DX_M / 1000.)
