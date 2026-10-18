"""Run the real replay CLI with the delivered write-only step-1 trace hooks."""
import os
from pathlib import Path
import sys


def main():
    if not os.environ.get("WOOF_LOCDUMP"):
        raise ValueError("WOOF_LOCDUMP must name the owned trace directory")
    import gpuwm.io.restart as restart
    restart.STATE_INFRA_ATTRS = frozenset(restart.STATE_INFRA_ATTRS | {"_loc_dt"})
    sys.path.insert(0, str(Path(__file__).parent / "wrf_exact_localize"))
    import woof_hook
    import woof_hook2
    import woof_hook3
    if os.environ.get("WOOF_TRACE_GF_INPUTS") == "1":
        import woof_gf_inputs
    if os.environ.get("WOOF_TRACE_WSM6_INPUTS") == "1":
        import woof_wsm6_inputs
    from gpuwm.cli import main as cli
    return cli(["verify-exact", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
