"""``GPUWM_NO_LOCAL_GPU``: the one definition of the local-device opt-out.

Setting the variable to anything but empty or ``0`` means **no gpuwm
process may open or read the local CUDA device**: no context stood up,
no ``memGetInfo``, no device-properties read, no kernel compiled -- not
in-process and not through a spawned probe.  It is the rented-GPU
workflow's switch: runs execute on another machine's card while this
one's stays untouched, and the CI legs use it to keep the CPU suites
off whatever card the box happens to hold.

Two boundaries, stated so no caller re-invents them:

* It does NOT claim cupy is uninstalled.  A box with the wheel present
  and the variable set is the ordinary rented-GPU shape; presence
  probes (``find_spec``) stay truthful and only device contact stops.
* It is not a sizing statement.  A gate that would have measured the
  card must gate against whatever budget was DECLARED (a ``--card``
  tier, ``--vram-gib``, a config's budget keys) and say the card was
  not read -- never silently measure anyway, and never refuse on a
  number it was told not to take.  The 2.5.0 upgrader walk measured the
  breakage: the variable was set for every step and the ``gpuwm go``
  memory gate still reported the local card's free VRAM, because its
  probe path never consulted the variable.

One definition on purpose.  Three hand-rolled ``os.environ`` reads in
:mod:`gpuwm.doctor` agreed with each other by luck while the memory
gate's probe read nothing at all; a scope that lives in one function
cannot fork like that, and the census test in
``tests/test_no_local_gpu_contract.py`` refuses a fourth raw read.
"""

from __future__ import annotations

import os

#: The variable, spelled once.
NO_LOCAL_GPU_ENV = "GPUWM_NO_LOCAL_GPU"


def no_local_gpu() -> bool:
    """Whether this process is forbidden to touch the local CUDA device."""

    return os.environ.get(NO_LOCAL_GPU_ENV, "") not in ("", "0")


__all__ = ["NO_LOCAL_GPU_ENV", "no_local_gpu"]
