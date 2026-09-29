"""CuPy's false "CUDA path could not be detected" warning on Windows.

On Windows, ``import cupy`` (CuPy 14.0 to 14.2) runs
``cupy._environment._setup_win32_dll_directory``, which asks for one CUDA
root folder and, when it gets none, writes::

    cupy/_environment.py:286: UserWarning: CUDA path could not be detected.
    Set CUDA_PATH environment variable if CuPy fails to load.

With the CUDA 12 toolkit that CuPy's own ``[ctk]`` extra installs from pip,
it never gets one.  cuda-pathfinder finds NVRTC in
``site-packages/nvidia/cuda_nvrtc/bin``, and CuPy names a root only for the
CUDA 13 wheels' single ``nvidia/cu13`` folder, so the CUDA 12 wheels'
folder-per-library layout reads as "no CUDA path".  Setting ``CUDA_PATH``
does not change that: CuPy 14.2 does not read the variable itself, and
cuda-pathfinder searches site-packages before it, so the wheel's NVRTC is
found first either way.  Nothing is missing, either: CuPy loads NVRTC and
its math libraries through cuda-pathfinder from those same wheels, which is
how every Windows install of the engine runs.  CuPy's
maintainers removed the step as obsolete (cupy/cupy#10044, commit 54522ab);
no released CuPy carries that removal yet.

The breakage this prevents: every engine command on the Windows managed
runtime, the desktop's own install, wrote the warning to stderr before doing
anything else, and the desktop shows a failed command's stderr as its error
text, so every failure a desktop user saw opened on a CUDA warning that was
not true.

:func:`quiet_false_cuda_path_warning` ignores that one warning exactly when
it is false: on Windows, when a site-packages folder cuda-pathfinder searches
holds the CUDA 12 wheel's NVRTC.  Anywhere else it is left alone, because an
install with no toolkit wheel may really be unable to find CUDA, and the
warning is the first sign of it.  ``gpuwm/__init__.py`` calls it, so it is in
place before any gpuwm module imports CuPy.

Retire this module when the runtime's CuPy floor is a release without
``_setup_win32_dll_directory`` (cupy/cupy#10044).
"""

from __future__ import annotations

import os
import site
import sys
import warnings
from collections.abc import Iterable

#: The warning's text, as CuPy 14.2 writes it, anchored at its start the way
#: :func:`warnings.filterwarnings` matches a message.
WARNING_MESSAGE = r"CUDA path could not be detected\."

#: The module that writes it, matched whole.
WARNING_MODULE = r"cupy\._environment\Z"

#: The CUDA 12 NVRTC wheel's DLL, relative to a site-packages folder: the
#: file cuda-pathfinder loads NVRTC from on x64 Windows when the
#: ``nvidia-cuda-nvrtc-cu12`` wheel is installed.
WHEEL_NVRTC = ("nvidia", "cuda_nvrtc", "bin", "nvrtc64_120_0.dll")


def pathfinder_site_packages() -> list[str]:
    """The folders cuda-pathfinder searches for a wheel's library, in order.

    ``site.getsitepackages()``, then the user site when it is enabled, the
    same list ``cuda.pathfinder._utils.find_sub_dirs`` builds.
    """
    try:
        folders = list(site.getsitepackages())
    except AttributeError:  # a ``site`` module without it (old virtualenv)
        folders = []
    if site.ENABLE_USER_SITE:
        user = site.getusersitepackages()
        if user:
            folders.append(user)
    return folders


def wheel_nvrtc(folders: Iterable[str]) -> str | None:
    """The CUDA 12 wheel's NVRTC DLL in the first of FOLDERS that holds it."""
    for folder in folders:
        candidate = os.path.join(folder, *WHEEL_NVRTC)
        if os.path.isfile(candidate):
            return candidate
    return None


def quiet_false_cuda_path_warning(
    *, platform: str | None = None, folders: Iterable[str] | None = None,
) -> str | None:
    """Ignore CuPy's CUDA path warning where it is false.

    Returns the NVRTC DLL that makes it false, after adding the filter, or
    ``None`` and adds nothing.  PLATFORM and FOLDERS default to this process
    (``sys.platform`` and :func:`pathfinder_site_packages`).
    """
    if (platform or sys.platform) != "win32":
        return None
    found = wheel_nvrtc(pathfinder_site_packages() if folders is None
                        else folders)
    if found is None:
        return None
    warnings.filterwarnings("ignore", message=WARNING_MESSAGE,
                            category=UserWarning, module=WARNING_MODULE)
    return found


__all__ = ["WARNING_MESSAGE", "WARNING_MODULE", "WHEEL_NVRTC",
           "pathfinder_site_packages", "quiet_false_cuda_path_warning",
           "wheel_nvrtc"]
