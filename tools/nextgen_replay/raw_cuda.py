"""Compatibility import for replay scripts; shared implementation is in gpuwm.

SPDX-License-Identifier: Apache-2.0
Stage gpuwm.nextgen.raw_cuda alongside this module when running from snapshots.
"""
from gpuwm.nextgen.raw_cuda import RawCuda

__all__ = ["RawCuda"]
