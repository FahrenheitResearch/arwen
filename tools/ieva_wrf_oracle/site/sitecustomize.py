"""Install the IEVA capture in every Python process of a run (capture.py)."""

import os
import sys

if os.environ.get("IEVA_CAPTURE_DIR"):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    import capture

    capture._install(os.environ["IEVA_CAPTURE_DIR"])
