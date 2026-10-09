"""Run the localize kit's woof_run.py with the dump hook's own attribute (_loc_dt) declared as
restart infrastructure (no arithmetic), plus the base4 extra hooks when WOOF_HOOK3=1."""
import os
import runpy
import sys

import gpuwm.io.restart as R

R.STATE_INFRA_ATTRS = frozenset(R.STATE_INFRA_ATTRS | {"_loc_dt"})
script = sys.argv[1]
sys.argv = sys.argv[1:]
if os.environ.get("WOOF_LOCDUMP") and os.environ.get("WOOF_HOOK3") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    sys.path.insert(0, "/work/pverify/base4")
    import woof_hook  # noqa: F401
    import woof_hook2  # noqa: F401
    import woof_hook3  # noqa: F401
runpy.run_path(script, run_name="__main__")
