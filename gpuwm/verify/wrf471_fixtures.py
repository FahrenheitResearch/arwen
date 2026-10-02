"""Where the compiled WRF v4.7.1 dycore oracle fixtures live.

The advection, small-step and big-step oracle fixtures are test data kept in
a source checkout under ``tests/data`` (``wrf471_advect``,
``wrf471_smallstep``, ``wrf471_bigstep``), beside the diffusion oracle's
``tests/data/wrf471_diffusion``.  They are not package data: under
``gpuwm/data`` they made the 2.8.2 pure wheel 184 MB, over the
100,000,000-byte per-file distribution limit, and no runtime code reads them.

An installed gpuwm therefore carries these harness modules without their
fixtures, and :func:`require_fixture_dir` is the refusal for that case.
"""
from __future__ import annotations

from pathlib import Path


def require_fixture_dir(directory, oracle: str) -> Path:
    """Return ``directory``, or refuse by name when it does not exist.

    Breakage prevented: an installed gpuwm has no ``tests/data``, so an
    oracle replay from it would die on a bare ``FileNotFoundError`` for an
    archive path beside site-packages, which reads as a damaged install
    rather than as a harness that has no compiled WRF reference words to
    compare against outside a source checkout.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"The compiled WRF v4.7.1 {oracle} oracle fixtures are missing: "
            f"{directory} does not exist, so there are no compiled WRF "
            "reference words to compare against.  These fixtures are test "
            "data in a source checkout's tests/data and are not shipped in "
            "the gpuwm wheel; this oracle runs from a source checkout.")
    return directory
