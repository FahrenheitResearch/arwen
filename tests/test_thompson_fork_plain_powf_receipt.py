"""Run the source/receipt guard on CPU despite the device module's GPU marker."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def test_fork_plain_powf_site_receipt_runs_without_a_device():
    # b0556bd76, lane/286-fork-thompson: both new fork powers are retained
    # only with measured complete-site outputs. Invoke the original guard,
    # whose source/hash/receipt checks import no CUDA runtime.
    path = Path(__file__).with_name("test_thompson_aerosol_device_helpers.py")
    spec = importlib.util.spec_from_file_location("_thompson_powf_host_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_every_surviving_plain_powf_was_measured_and_is_inert()
