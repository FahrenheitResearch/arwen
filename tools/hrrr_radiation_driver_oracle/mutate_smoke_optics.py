"""Show the source gate rejects dropping smoke from the combined optics.

Usage: python -m tools.hrrr_radiation_driver_oracle.mutate_smoke_optics
The mutation exists only in this process and never edits source files.
"""
import inspect

import pytest

from gpuwm.core import rrtmg_aerosol_optics as ao


def main():
    gate = ["-q", "tests/test_rrtmg_smoke_input.py",
            "-k", "smoke_twin_matches_the_compiled_source_oracle"]
    baseline = pytest.main(gate)
    if baseline != 0:
        raise SystemExit("the unmodified source oracle must pass before mutation is meaningful")
    text = inspect.getsource(ao.aer3_sw_optics)
    original = "taod = (taod + np.minimum(F(3.0), smoke)).astype(np.float32)"
    if text.count(original) != 1:
        raise ValueError("smoke addition statement changed; review this mutation before running it")
    mutated = text.replace(original, "taod = taod.astype(np.float32)")
    namespace = dict(vars(ao))
    exec(compile(mutated, "<smoke-addition-removed>", "exec"), namespace)
    ao.aer3_sw_optics = namespace["aer3_sw_optics"]
    code = pytest.main(gate)
    print(f"smoke-addition-removed mutation gate return code: {code}; expected 1")
    if code != 1:
        raise SystemExit("the source oracle did not reject the missing smoke addition")


if __name__ == "__main__":
    main()
