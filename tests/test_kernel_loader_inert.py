"""Prove the mp=28 kernel-loader hook is inert for every other module.

``gpuwm/core/kernels/__init__.py`` grew one additive ``_EXTRA_HEADERS``
allow-list so the six aerosol-aware Thompson translation units can share
``thompson_aerosol_common.cuh``; there is no ``#include`` path under
``cupy.RawModule``.

That hook touches the ONE file ``mp_physics=8`` also uses, so its inertness is
what keeps WP-00's freeze gate meaningful.  The guarantee claimed in the port
spec is *by construction*, not by measurement: a module absent from the
allow-list contributes the empty string and therefore assembles a
byte-identical source string, hence identical PTX, hence identical FP32
results.  This module enumerates every ``.cu`` in the tree and asserts exactly
that -- no spot checks.

These tests are CPU-only on purpose.  They read and compare source strings and
never compile anything, so they run in environments with no CUDA device.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuwm.core import kernels as kernel_loader


_KDIR = Path(kernel_loader.__file__).parent

#: The only modules that may appear in the allow-list.  Kept literal so a
#: future package cannot quietly add itself to the dict.
_ALLOWED_AEROSOL_MODULES = frozenset({
    "thompson_aerosol_probe",
    "thompson_aerosol_state",
    "thompson_aerosol_sat",
    "thompson_aerosol_cold",
    "thompson_aerosol_warm",
    "thompson_aerosol_sed",
})

#: The whole allow-list, module -> its exact header tuple.  Closed and
#: literal, which is the property these tests hold: NOTHING receives a
#: device header without being named here, in a second place.
#: rrtmgp_rte joined when the LW solver started deriving its own Planck
#: sources; rrtmgp_gas is deliberately absent (it keeps its own copies so
#: its assembled source and PTX stay byte-identical).
#: gf joined when the glibc 2.39 float32 transcendentals were lifted out of
#: gf.cu into glibc_flt32.cuh for New Tiedtke to share.  It is the one entry
#: whose module KNOWINGLY gives up the byte-identical-source guarantee the
#: unlisted modules keep -- gf's correctness is held by its 396 parity tests
#: at max_ulp 0, which is a stronger gate than source identity and the only
#: reason the trade was takeable.
_EXPECTED_HEADERS = {
    **{name: ("thompson_aerosol_common.cuh",)
       for name in _ALLOWED_AEROSOL_MODULES},
    "rrtmgp_rte": ("rrtmgp_planck_common.cuh",),
    "gf": ("glibc_flt32.cuh",),
    "ntiedtke": ("glibc_flt32.cuh",),
}


def _module_names() -> list[str]:
    names = sorted(path.stem for path in _KDIR.glob("*.cu"))
    assert names, "no CUDA translation units found"
    assert "thompson" in names
    return names


def _pre_hook_source(name: str) -> str:
    """The exact string the loader assembled BEFORE _EXTRA_HEADERS existed.

    Written out literally rather than derived from the loader, so a change to
    the loader cannot make this test agree with itself.

    The encoding is named for the same reason the loader names it: this test
    asserts two independent reads of one file are byte-identical, so if the
    two sides can disagree about how to decode it the assertion is about the
    locale rather than about the hook.  On Windows both sides used to decode
    the em dashes in acoustic.cu, advection.cu and coriolis_map.cu as cp1252
    and agree with each other while both being wrong.
    """
    return (kernel_loader._preamble()
            + (_KDIR / f"{name}.cu").read_text(encoding="utf-8"))


def _pre_hook_source_int_defines(name: str, prefix: str) -> str:
    return (kernel_loader._preamble() + prefix + "\n"
            + (_KDIR / f"{name}.cu").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", _module_names())
def test_non_aerosol_module_source_is_byte_identical(name):
    """Every module outside the allow-list assembles the pre-hook string."""
    if name in kernel_loader.EXTRA_HEADERS:
        pytest.skip(f"{name} is an allow-listed aerosol module")
    assembled = kernel_loader.module_source(name)
    expected = _pre_hook_source(name)
    assert assembled == expected, (
        f"{name}.cu no longer assembles its pre-hook source string; the "
        "mp=8 numerics guarantee rests on this being byte-identical")


@pytest.mark.parametrize("name", _module_names())
def test_non_aerosol_int_define_source_is_byte_identical(name):
    """The specialized-integer loader is inert for the same modules."""
    if name in kernel_loader.EXTRA_HEADERS:
        pytest.skip(f"{name} is an allow-listed aerosol module")
    defines = (("KMAX", 64), ("NLEVELS", 3))
    prefix = "\n".join(f"#define {key} {value}" for key, value in defines)
    assembled = kernel_loader.module_source_int_defines(name, defines)
    assert assembled == _pre_hook_source_int_defines(name, prefix)


def test_thompson_cu_source_is_byte_identical():
    """The single most load-bearing case, asserted without parametrization."""
    assert "thompson" not in kernel_loader.EXTRA_HEADERS
    assert (kernel_loader.module_source("thompson")
            == _pre_hook_source("thompson"))


def test_allow_list_is_a_closed_literal_mapping():
    extra = kernel_loader.EXTRA_HEADERS
    assert dict(extra) == _EXPECTED_HEADERS, (
        "the extra-header allow-list is a closed literal mapping; a module "
        "may receive a device header only by being named here too")
    for name, headers in extra.items():
        assert isinstance(headers, tuple)


def test_allow_listed_headers_exist_and_are_not_translation_units():
    for headers in kernel_loader.EXTRA_HEADERS.values():
        for header in headers:
            path = _KDIR / header
            assert path.is_file(), f"missing device header {header}"
            assert path.suffix == ".cuh", (
                "an allow-listed header must be a .cuh, never a .cu")


def test_allow_list_has_no_implicit_filesystem_behaviour():
    """A module not named in the dict gets nothing, even if a .cuh exists."""
    assert kernel_loader._extra_header_text("thompson") == ""
    assert kernel_loader._extra_header_text("no_such_module_at_all") == ""
    # A same-named .cuh sitting next to a .cu must NOT be picked up.
    assert (_KDIR / "thompson_aerosol_common.cuh").is_file()
    assert kernel_loader._extra_header_text("thompson_aerosol_common") == ""


def test_allow_listed_module_actually_receives_the_header():
    """The hook must not be inert for the modules it is FOR."""
    name = "thompson_aerosol_probe"
    assert name in kernel_loader.EXTRA_HEADERS
    header = (_KDIR / "thompson_aerosol_common.cuh").read_text(
        encoding="utf-8")
    assembled = kernel_loader.module_source(name)
    assert assembled != _pre_hook_source(name)
    assert assembled == (kernel_loader._preamble() + header
                         + (_KDIR / f"{name}.cu").read_text(encoding="utf-8"))
    # The header must precede the module body, or the helpers are undeclared.
    assert assembled.index("thompson_aa_droplet_bin") < assembled.index(
        "thompson_aa_probe_droplet_bin")


def test_launch_module_names_agree_with_the_allow_list():
    from gpuwm.core import thompson_aerosol_launch as launch

    assert set(launch.AEROSOL_KERNEL_MODULES) == (
        set(kernel_loader.EXTRA_HEADERS) & _ALLOWED_AEROSOL_MODULES)
    assert launch.CLASSIC_MODULE not in kernel_loader.EXTRA_HEADERS
    assert launch.AEROSOL_COMMON_HEADER == "thompson_aerosol_common.cuh"


def test_preamble_is_unchanged_by_the_hook():
    """_preamble() still ends with common.cuh and carries no aerosol text."""
    preamble = kernel_loader._preamble()
    assert preamble.endswith(
        (_KDIR / "common.cuh").read_text(encoding="utf-8") + "\n")
    assert "thompson_aa_" not in preamble
