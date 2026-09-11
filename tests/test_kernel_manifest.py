"""Every compiled site reports the translation unit it actually compiled.

The manifest is only worth carrying if it cannot drift from the compiler call
beside it.  These tests read the tree's own syntax: at each site the arguments
handed to :func:`record_module` must be the *same expressions* handed to the
compiler, so recording a different source, or collapsing the option tuples
into one global flags string, is a syntactic difference a test can see
without a device.

TWO compiler routes are audited, not one.  ``cp.RawModule`` is the common
one; ``cupy.cuda.compiler.compile_using_nvrtc`` is the bypass a site must
take when it needs an option CuPy would otherwise duplicate -- CuPy appends
``-ftz=true`` after the caller's options, NVRTC 13.0 rejects the repeat
outright, and ``gpuwm/core/rrtmg_sw.py`` moved onto the bypass for exactly
that reason.  Auditing only RawModule would have let that site keep
recording an option tuple no compiler was handed.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from conftest import requires_gpu

from gpuwm.certify.kernel_manifest import (KernelManifestConflict,
                                           kernel_manifest, record_module,
                                           reset_kernel_manifest,
                                           source_sha256)
from gpuwm.core.constants import CUDA_DEFINES

REPO = Path(__file__).resolve().parents[1]
KERNEL_DIR = REPO / "gpuwm" / "core" / "kernels"

#: The manifest sites: every module that constructs a ``cp.RawModule`` for
#: a forecast translation unit or records one.  The Noah-MP factories no
#: longer construct anything themselves -- they delegate to the one site in
#: ``noahmp_kernel_sources.py`` -- and stay listed so :func:`audit_source`
#: keeps reading them: a factory that grew a RawModule of its own again
#: would be caught here.  :func:`test_every_rawmodule_constructor_in_the_tree_is_named`
#: is the census behind this list; nothing under ``gpuwm/`` may construct a
#: RawModule without appearing either here or in its explained allow-list.
SITE_FILES = (
    "gpuwm/core/kernels/__init__.py",
    "gpuwm/core/nest_interp.py",
    "gpuwm/core/noahmp_driver_gpu.py",
    "gpuwm/core/noahmp_energy_gpu.py",
    "gpuwm/core/noahmp_glacier_gpu.py",
    "gpuwm/core/noahmp_kernel_sources.py",
    "gpuwm/core/noahmp_slab_libm.py",
    "gpuwm/core/noahmp_thermal_gpu.py",
    "gpuwm/core/noahmp_vegeflux_gpu.py",
    "gpuwm/core/rrtmg_sw.py",
)

#: Two cached loaders, nest interpolation, and the one Noah-MP compile
#: site (``noahmp_kernel_sources.compile_runtime_unit``); the six Noah-MP
#: factories that used to be sites each delegate to it.  ``rrtmg_sw.py`` is
#: NOT among them -- it compiles through :data:`EXPECTED_NVRTC_SITE_COUNT`'s
#: route instead, see the module docstring -- but it stays in
#: :data:`SITE_FILES` because it still records.
EXPECTED_SITE_COUNT = 4

#: ``cp.RawModule`` constructors under ``gpuwm/`` that are NOT manifest
#: sites, each with the reason.  Closed and literal: a new constructor
#: anywhere else fails the census below until it is either made a site or
#: explained here.
RAWMODULE_CONSTRUCTORS_OUTSIDE_THE_MANIFEST = {
    "gpuwm/core/p3_device.py": (
        "the P3 composed unit's own compile site; it records through "
        "record_module in the same function and its frame is re-audited on a "
        "device by tests/test_p3_cuda_gpu.py"),
    "gpuwm/doctor.py": (
        "a one-kernel self-contained probe inside a subprocess source string, "
        "compiled to prove the toolchain works; not a forecast translation unit"),
}

#: ``compile_using_nvrtc`` sites among :data:`SITE_FILES`: rrtmg_sw only.
#: (``rrtmg_lw.py`` and ``rrtmg_mcica.py`` take the same route but record
#: nothing, so they are not manifest sites and are not listed above.)
EXPECTED_NVRTC_SITE_COUNT = 1


def _is_rawmodule(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "RawModule")


def _is_nvrtc(node: ast.AST) -> bool:
    """A direct ``compile_using_nvrtc`` call, however it was imported."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = (func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None)
    return name == "compile_using_nvrtc"


#: Per route, how to reach the SOURCE and the OPTIONS argument:
#: ``(keyword, positional index)``.  ``compile_using_nvrtc(source, options,
#: arch, filename)`` takes both positionally at every site in this tree.
_COMPILE_ARGS = {
    "rawmodule": {"source": ("code", None), "options": ("options", None)},
    "nvrtc": {"source": ("source", 0), "options": ("options", 1)},
}


def _compiled_argument(call: ast.Call, route: str, which: str):
    keyword, index = _COMPILE_ARGS[route][which]
    found = _keyword(call, keyword)
    if found is not None:
        return found
    if index is not None and len(call.args) > index:
        return call.args[index]
    return None


def _is_record(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "record_module")


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def audit_source(path: str, text: str) -> list[str]:
    """Problems with the record/compile pairing in one module's source.

    Returns an empty list when every compile -- ``cp.RawModule`` or a direct
    ``compile_using_nvrtc`` -- is accompanied, in the same function, by a
    ``record_module`` call whose ``source`` and ``options`` expressions are
    syntactically the arguments the compiler was given.
    """
    tree = ast.parse(text)
    problems: list[str] = []
    for function in _functions(tree):
        compiles = [(node, "rawmodule" if _is_rawmodule(node) else "nvrtc")
                    for node in ast.walk(function)
                    if _is_rawmodule(node) or _is_nvrtc(node)]
        if not compiles:
            continue
        records = [node for node in ast.walk(function) if _is_record(node)]
        where = f"{path}:{function.name}"
        if len(records) != len(compiles):
            problems.append(
                f"{where} compiles {len(compiles)} module(s) but records "
                f"{len(records)}")
            continue
        for (compiled, route), recorded in zip(compiles, records):
            for which in ("source", "options"):
                left = _compiled_argument(compiled, route, which)
                right = _keyword(recorded, which)
                if left is None or right is None:
                    problems.append(f"{where} is missing {which}")
                    continue
                if ast.dump(left) != ast.dump(right):
                    problems.append(
                        f"{where} records {which}={ast.unparse(right)} "
                        f"but compiles {which}={ast.unparse(left)}")
    return problems


def _site_text(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def test_every_rawmodule_site_is_accounted_for():
    total = 0
    for path in SITE_FILES:
        tree = ast.parse(_site_text(path))
        total += sum(1 for node in ast.walk(tree) if _is_rawmodule(node))
    assert total == EXPECTED_SITE_COUNT, (
        f"expected {EXPECTED_SITE_COUNT} RawModule sites, found {total}; the "
        "manifest covers a fixed inventory and a new site must join it")


def test_every_rawmodule_constructor_in_the_tree_is_named():
    """The census behind SITE_FILES: no RawModule constructor may hide.

    The breakage this prevents: a module that compiles a translation unit
    outside the manifest records nothing about it, and a frame table, a
    parity receipt or a memory price can then describe a source the run
    never compiled.  Found when the Noah-MP factories were collapsed onto
    one site and the count dropped 8 -> 4: the old contract said "every
    module that constructs a RawModule" and nothing checked it.  In
    particular ``noahmp_frame_provenance.py`` constructs none -- its
    measurement compiles through the one Noah-MP site -- and this is
    where that stays asserted.
    """
    constructors = set()
    for path in sorted((REPO / "gpuwm").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "RawModule" not in text and "RawKernel" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name in ("RawModule", "RawKernel"):
                constructors.add(path.relative_to(REPO).as_posix())
    # gpuwm/core/preflight.py's probe compiles nothing; doctor's probe is a
    # string, so its constructor is found by the text scan below instead.
    doctor = (REPO / "gpuwm/doctor.py").read_text(encoding="utf-8")
    if "RawModule(" in doctor:
        constructors.add("gpuwm/doctor.py")
    named = set(SITE_FILES) | set(RAWMODULE_CONSTRUCTORS_OUTSIDE_THE_MANIFEST)
    assert constructors <= named, sorted(constructors - named)
    assert "gpuwm/core/noahmp_frame_provenance.py" not in constructors
    assert "gpuwm/core/noahmp_kernel_sources.py" in constructors
    for path in RAWMODULE_CONSTRUCTORS_OUTSIDE_THE_MANIFEST:
        assert path in constructors, f"{path} no longer constructs one; drop it"
    for path, reason in RAWMODULE_CONSTRUCTORS_OUTSIDE_THE_MANIFEST.items():
        assert len(reason.split()) >= 12, path


def test_every_direct_nvrtc_site_is_accounted_for():
    """The bypass route is inventoried too, or a site could leave silently.

    A site that moves off ``cp.RawModule`` and onto ``compile_using_nvrtc``
    -- which is what CUDA 13 forced on ``rrtmg_sw`` -- would otherwise drop
    out of the RawModule count and out of :func:`audit_source` at the same
    time, taking its manifest record with it and leaving both tests green.
    """
    total = 0
    for path in SITE_FILES:
        tree = ast.parse(_site_text(path))
        total += sum(1 for node in ast.walk(tree) if _is_nvrtc(node))
    assert total == EXPECTED_NVRTC_SITE_COUNT, (
        f"expected {EXPECTED_NVRTC_SITE_COUNT} compile_using_nvrtc site(s) "
        f"among the manifest files, found {total}")


@pytest.mark.parametrize("path", SITE_FILES)
def test_recorded_arguments_are_the_compiled_arguments(path):
    assert audit_source(path, _site_text(path)) == []


def test_option_tuples_are_per_site_and_not_one_global_flags_string():
    """The nest and shortwave modules carry options no other site carries."""
    recorded: dict[str, list[str]] = {}
    for path in SITE_FILES:
        tree = ast.parse(_site_text(path))
        for node in ast.walk(tree):
            if _is_record(node):
                options = _keyword(node, "options")
                recorded.setdefault(path, []).append(ast.unparse(options))
    nest = recorded["gpuwm/core/nest_interp.py"]
    shortwave = recorded["gpuwm/core/rrtmg_sw.py"]
    assert nest == ["('-std=c++17', '-fmad=false')"]
    assert shortwave == ["('-std=c++17', '--ftz=false')"]
    assert nest != shortwave
    flattened = [option for options in recorded.values() for option in options]
    assert len(set(flattened)) > 1, (
        "every site recording the same options expression is the collapse "
        "this manifest exists to prevent")


# --- failure controls ------------------------------------------------------

def test_control_a_site_that_stops_recording_is_caught():
    """Failure control 1: delete a site's record_module call."""
    path = "gpuwm/core/nest_interp.py"
    text = _site_text(path)
    mutated = text.replace(
        '    record_module("gpuwm.core.nest_interp:nest", source=src,\n'
        '                  options=("-std=c++17", "-fmad=false"), module=mod)\n',
        "")
    assert mutated != text, "the control did not modify the site"
    problems = audit_source(path, mutated)
    assert problems, "a site that records nothing must be reported"
    assert "records 0" in problems[0]


def test_control_options_collapsed_to_a_global_string_is_caught():
    """Failure control 2: record the common flags instead of the site's."""
    path = "gpuwm/core/nest_interp.py"
    text = _site_text(path)
    mutated = text.replace(
        '                  options=("-std=c++17", "-fmad=false"), module=mod)',
        '                  options=("-std=c++17",), module=mod)')
    assert mutated != text, "the control did not modify the site"
    problems = audit_source(path, mutated)
    assert problems, "a site recording the wrong option tuple must be reported"
    assert "records options=" in problems[0]


def test_control_an_nvrtc_site_recording_other_options_is_caught():
    """Failure control 3: the bypass route is audited as strictly.

    ``rrtmg_sw`` compiles with ``--ftz=false`` because CuPy's RawModule route
    would inject ``-ftz=true`` and flush the subnormal transmittance products
    (MEASURED: the LW preflight's ``1e-30 * 1e-10`` comes back 0.0 through
    RawModule and 1e-40 through this route).  Recording an option tuple the
    compiler never saw would put that claim in the manifest without it being
    true, so mutate the record and require a complaint.
    """
    path = "gpuwm/core/rrtmg_sw.py"
    text = _site_text(path)
    mutated = text.replace(
        '        record_module("gpuwm.core.rrtmg_sw:rrtmg_sw", source=code,\n'
        '                      options=("-std=c++17", "--ftz=false"),',
        '        record_module("gpuwm.core.rrtmg_sw:rrtmg_sw", source=code,\n'
        '                      options=("-std=c++17",),')
    assert mutated != text, "the control did not modify the site"
    problems = audit_source(path, mutated)
    assert problems, "an nvrtc site recording the wrong options must be caught"
    assert "records options=" in problems[0]


# --- the source hash, recomputed independently -----------------------------

def _independent_preamble() -> str:
    """Rebuild the preamble from the published construction, not from gpuwm.

    ``gpuwm/core/kernels/__init__.py`` builds one ``#define`` per entry of
    ``CUDA_DEFINES`` in ``float`` repr with an ``f`` suffix, then appends
    ``common.cuh``.  This is that construction written out again; if the
    production one changes, the two stop agreeing.
    """
    lines = [f"#define {key} {float(value)!r}f"
             for key, value in CUDA_DEFINES.items()]
    lines.append((KERNEL_DIR / "common.cuh").read_text())
    return "\n".join(lines) + "\n"


def test_independent_preamble_reconstruction_matches_production():
    from gpuwm.core.kernels import _preamble

    assert _independent_preamble() == _preamble()


@pytest.mark.parametrize("name", ["acoustic", "advection", "diff6"])
def test_recorded_source_hash_is_the_hash_of_the_translation_unit(name):
    reset_kernel_manifest()
    source = _independent_preamble() + (
        KERNEL_DIR / f"{name}.cu").read_text()
    record_module(f"gpuwm.core.kernels:{name}", source=source,
                  options=("-std=c++17",))
    expected = hashlib.sha256(source.encode("utf-8")).hexdigest()
    entry = kernel_manifest()[f"gpuwm.core.kernels:{name}"]
    assert entry["source_sha256"] == expected
    assert entry["options"] == ["-std=c++17"]
    assert entry["compiled_image"]["status"] == "unavailable"
    assert entry["compiled_image"]["reason"]
    reset_kernel_manifest()


def test_source_sha256_is_a_plain_sha256_over_utf8_bytes():
    assert source_sha256("abc") == hashlib.sha256(b"abc").hexdigest()


def test_one_key_records_two_images_without_losing_either():
    """A negative control's variant compile is recorded, never overwritten."""
    reset_kernel_manifest()
    record_module("gpuwm.core.kernels:probe", source="a",
                  options=("-std=c++17",))
    record_module("gpuwm.core.kernels:probe", source="a",
                  options=("-std=c++17",))
    assert list(kernel_manifest()) == ["gpuwm.core.kernels:probe"]

    # Same unit, different options: a second compiled image, a second entry.
    record_module("gpuwm.core.kernels:probe", source="a",
                  options=("-std=c++17", "-fmad=false"))
    # Same options, different source: a third.
    record_module("gpuwm.core.kernels:probe", source="b",
                  options=("-std=c++17",))
    manifest = kernel_manifest()
    assert len(manifest) == 3, manifest
    sources = {entry["source_sha256"] for entry in manifest.values()}
    assert sources == {source_sha256("a"), source_sha256("b")}
    options = {tuple(entry["options"]) for entry in manifest.values()}
    assert options == {("-std=c++17",), ("-std=c++17", "-fmad=false")}
    # Deterministic: recording the same three again adds nothing.
    record_module("gpuwm.core.kernels:probe", source="a",
                  options=("-std=c++17", "-fmad=false"))
    record_module("gpuwm.core.kernels:probe", source="b",
                  options=("-std=c++17",))
    assert kernel_manifest() == manifest
    reset_kernel_manifest()


def test_the_conflict_error_still_exists_for_an_indistinguishable_collision():
    assert issubclass(KernelManifestConflict, RuntimeError)


@requires_gpu
def test_gpu_a_compiled_module_records_the_source_the_test_rebuilds():
    """The GPU half: compile through the production loader and check the record.

    D-19's CPU-checkable half is the source and option record; this exercises
    it end to end on a device, and reports whichever compiled-image status the
    installed CuPy actually permits rather than requiring one.
    """
    from gpuwm.core.kernels import load_module

    reset_kernel_manifest()
    # The loader is lru_cached, so an earlier test in this process may already
    # hold the module; the record only fires on a real compile.
    load_module.cache_clear()
    load_module("diff6")
    entry = kernel_manifest()["gpuwm.core.kernels:diff6"]
    expected = hashlib.sha256(
        (_independent_preamble()
         + (KERNEL_DIR / "diff6.cu").read_text()).encode("utf-8")).hexdigest()
    assert entry["source_sha256"] == expected
    assert entry["options"] == ["-std=c++17"]
    assert entry["compiled_image"]["status"] in {"resolved", "unavailable"}
    reset_kernel_manifest()
