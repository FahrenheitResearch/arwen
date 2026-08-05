"""The ``WPHI_MAX_LEV`` tier ladder, and the proof it changed nothing below 128.

``gpuwm/core/kernels/acoustic.cu`` sizes one per-thread column,
``real rhs[WPHI_MAX_LEV]``, in the two implicit w''-phi'' solves.  That single
array was the whole ``nz <= 128`` ceiling.  The lift compiles the module at a
coarse tier chosen by ``nz`` -- the mechanism ``kf.cu``'s ``KF_KMAX`` already
uses -- and the acceptance that matters is NEGATIVE: every configuration that
ran before must compile the same kernel it compiled before.

Three independent things are asserted here, all on the CPU:

1. **The mechanism is inert below the ceiling.**  ``wphi_module_defines`` is
   EMPTY at ``nz <= 128``, so the launcher takes the unspecialized loader and
   the string handed to NVRTC is byte-identical to ``module_source`` -- the
   exact string the pre-ladder launcher produced.  Digested and compared.
2. **The source edit generates the same code.**  The ``#ifndef`` guard is
   reconstructed back to the pre-change single ``#define`` line, and the two
   forms are compared twice: token streams out of a real C preprocessor, and
   PTX out of a host ``nvcc -ptx`` compile.  Both must be equal.  Each has a
   negative control that injects the 193 tier and must DIFFER, so neither
   comparison is passing merely because it cannot fail.  Neither touches a
   device; the cubin leg of AC-P2.2 still belongs to the card.
3. **The ladder's edges are where they are claimed to be**, the level bound
   the CuPy-free front-door contract advertises is the launcher's own, and
   ``gpuwm.core.preflight`` prices a deeper tier instead of under-pricing the
   local-memory rail at the frame the shipped tier measured.

The host mirror is exercised at ``nz`` 160 and 192 against an independently
assembled dense solve, so the arithmetic above the old ceiling is checked by
something other than the code under test before the card ever sees it.

Every test in this file is CPU-only and imports no CuPy.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core import acoustic as ac
from gpuwm.core import constants as c
from gpuwm.core import preflight as pf
from gpuwm.core.kernels import module_source
from gpuwm.experiment import experiment_from_run_config
from gpuwm.physics_vertical_contract import ACOUSTIC_VERTICAL_LEVEL_BOUNDS
from gpuwm.verify.npref import np_advance_w_phi

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "gpuwm" / "core" / "kernels" / "acoustic.cu"

#: The literal the source keeps when nothing overrides it.  Written out here
#: rather than imported, so an edit to the ladder cannot make this file agree
#: with itself.
SHIPPED_TIER = 129
SHIPPED_LITERAL = "#define WPHI_MAX_LEV 129"


# ---------------------------------------------------------------------------
# 1. The ladder
# ---------------------------------------------------------------------------

def test_the_ladder_starts_at_the_source_literal_and_ascends():
    assert ac.WPHI_LEVEL_TIERS[0] == SHIPPED_TIER
    assert ac.UNSPECIALIZED_WPHI_LEVEL_TIER == SHIPPED_TIER
    assert list(ac.WPHI_LEVEL_TIERS) == sorted(set(ac.WPHI_LEVEL_TIERS))
    assert ac.MAX_ACOUSTIC_LEVELS == ac.WPHI_LEVEL_TIERS[-1] - 1


@pytest.mark.parametrize("nz", [1, 2, 4, 30, 40, 49, 64, 96, 127, 128])
def test_every_configuration_that_ran_before_takes_the_shipped_tier(nz):
    """The lift may not move a single pre-existing run to another tier."""
    assert ac.wphi_level_tier(nz) == SHIPPED_TIER
    assert ac.wphi_module_defines(nz) == ()


@pytest.mark.parametrize("nz,tier", [
    (128, 129),          # the last nz that fits the shipped tier
    (129, 193),          # the first nz that does not: the ceiling, lifted
    (160, 193),          # AC-P2.3's acceptance depth
    (192, 193),          # the last nz the middle tier holds
    (193, 257),          # and the step to the top tier
    (256, 257),          # the deepest column the ladder admits
])
def test_the_tier_steps_at_the_boundaries(nz, tier):
    assert ac.wphi_level_tier(nz) == tier
    expected = () if tier == SHIPPED_TIER else (("WPHI_MAX_LEV", tier),)
    assert ac.wphi_module_defines(nz) == expected


def test_the_solve_needs_one_entry_per_full_level_not_per_half_level():
    """``rhs`` is indexed 0..nz, so nz+1 entries, not nz."""
    for tier in ac.WPHI_LEVEL_TIERS:
        assert ac.wphi_level_tier(tier - 1) == tier
        if tier != ac.WPHI_LEVEL_TIERS[-1]:
            assert ac.wphi_level_tier(tier) > tier


def test_beyond_the_top_tier_the_refusal_is_loud():
    with pytest.raises(ValueError, match="exceeds the in-thread solve limit"):
        ac.wphi_level_tier(ac.MAX_ACOUSTIC_LEVELS + 1)
    with pytest.raises(ValueError, match="exceeds the in-thread solve limit"):
        ac.wphi_level_tier(10_000)


@pytest.mark.parametrize("nz", [0, -1])
def test_a_nonpositive_column_is_refused(nz):
    with pytest.raises(ValueError, match="nz must be positive"):
        ac.wphi_level_tier(nz)


def test_the_contract_bound_is_the_launcher_bound():
    """The CuPy-free front door and the launcher cannot drift apart."""
    assert ACOUSTIC_VERTICAL_LEVEL_BOUNDS == (1, ac.MAX_ACOUSTIC_LEVELS)
    low, high = ACOUSTIC_VERTICAL_LEVEL_BOUNDS
    assert ac.wphi_level_tier(low) == SHIPPED_TIER
    assert ac.wphi_level_tier(high) == ac.WPHI_LEVEL_TIERS[-1]
    with pytest.raises(ValueError):
        ac.wphi_level_tier(high + 1)
    with pytest.raises(ValueError):
        ac.wphi_level_tier(low - 1)


# ---------------------------------------------------------------------------
# 2. The generated source below the ceiling is the pre-ladder source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nz", [1, 4, 40, 49, 64, 96, 127, 128])
def test_below_the_ceiling_the_generated_source_is_byte_identical(nz):
    """The digest comparison the lift's acceptance rests on.

    ``module_source("acoustic")`` is exactly what the launcher handed NVRTC
    before the ladder existed, and it is exactly what it hands NVRTC now for
    any ``nz <= 128``.  Equality of the strings -- not of a pinned constant
    that would rot -- is the before/after proof.
    """
    generated = ac.wphi_kernel_source(nz)
    unspecialized = module_source("acoustic")
    assert generated == unspecialized
    assert (hashlib.sha256(generated.encode("utf-8")).hexdigest()
            == hashlib.sha256(unspecialized.encode("utf-8")).hexdigest())


def test_the_below_ceiling_source_carries_no_injected_define():
    generated = ac.wphi_kernel_source(128)
    assert "#define WPHI_MAX_LEV 193" not in generated
    assert "#define WPHI_MAX_LEV 257" not in generated
    assert generated.count(SHIPPED_LITERAL) == 1


@pytest.mark.parametrize("nz,tier", [(129, 193), (160, 193), (200, 257)])
def test_a_deeper_tier_adds_exactly_one_line_and_nothing_else(nz, tier):
    """Mutation control: the digest comparison above CAN fail.

    Above the ceiling the generated source must differ from the
    unspecialized module, and the entire difference must be the one injected
    define -- removing that line recovers the unspecialized string byte for
    byte, so a deeper tier cannot smuggle in any other edit.
    """
    generated = ac.wphi_kernel_source(nz)
    unspecialized = module_source("acoustic")
    assert generated != unspecialized

    injected = f"#define WPHI_MAX_LEV {tier}\n"
    assert generated.count(injected) == 1
    assert generated.replace(injected, "", 1) == unspecialized


def test_the_guard_admits_the_shipped_literal_and_only_that():
    """The ``#ifndef`` triple in the source, read structurally."""
    lines = KERNEL.read_text(encoding="utf-8").splitlines()
    defines = [i for i, line in enumerate(lines)
               if line.strip().startswith("#define WPHI_MAX_LEV")]
    assert len(defines) == 1, "exactly one WPHI_MAX_LEV definition"
    i = defines[0]
    assert lines[i].strip() == SHIPPED_LITERAL
    assert lines[i - 1].strip() == "#ifndef WPHI_MAX_LEV"
    assert lines[i + 1].strip() == "#endif"


# ---------------------------------------------------------------------------
# 3. The guard is a preprocessor no-op, measured with a real preprocessor
# ---------------------------------------------------------------------------

_GUARDED = ("#ifndef WPHI_MAX_LEV\n"
            f"{SHIPPED_LITERAL}\n"
            "#endif\n")


def _host_preprocessor() -> list[str] | None:
    """A C preprocessor command prefix, or ``None`` when none is installed."""
    for root in (Path("C:/Program Files/Microsoft Visual Studio"),
                 Path("C:/Program Files (x86)/Microsoft Visual Studio")):
        if not root.is_dir():
            continue
        found = sorted(root.glob(
            "*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"))
        if found:
            return [str(found[-1]), "-nologo", "-EP", "-TP"]
    return None


def _token_stream(source: str, tmp_path: Path, name: str,
                  extra: tuple[str, ...] = ()) -> list[str]:
    """Preprocessed non-blank lines.

    ``acoustic.cu`` has no ``#include``, so the preprocessor needs no header
    search path and its output is a pure macro expansion of this one file.
    ``-EP`` keeps the vertical whitespace a removed directive or comment left
    behind, which is invisible to the compiler, so blank lines are dropped
    before comparing.
    """
    command = _host_preprocessor()
    assert command is not None
    path = tmp_path / name
    path.write_text(source, encoding="utf-8", newline="")
    result = subprocess.run(command + list(extra) + [str(path)],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.skipif(_host_preprocessor() is None,
                    reason="no host C preprocessor installed")
def test_the_guard_is_a_preprocessor_no_op_at_the_shipped_tier(tmp_path):
    """With nothing defining it, the guard reduces to the pre-change line."""
    guarded = KERNEL.read_text(encoding="utf-8")
    assert guarded.count(_GUARDED) == 1
    pre_change = guarded.replace(_GUARDED, SHIPPED_LITERAL + "\n", 1)

    assert (_token_stream(guarded, tmp_path, "guarded.cpp")
            == _token_stream(pre_change, tmp_path, "pre_change.cpp"))


@pytest.mark.skipif(_host_preprocessor() is None,
                    reason="no host C preprocessor installed")
def test_the_preprocessor_comparison_can_fail(tmp_path):
    """Negative control: injecting a tier must move the token stream."""
    guarded = KERNEL.read_text(encoding="utf-8")
    pre_change = guarded.replace(_GUARDED, SHIPPED_LITERAL + "\n", 1)

    assert (_token_stream(guarded, tmp_path, "tiered.cpp",
                          ("-DWPHI_MAX_LEV=193",))
            != _token_stream(pre_change, tmp_path, "pre_change.cpp"))


# ---------------------------------------------------------------------------
# 3b. ... and the generated CODE is identical, compiled on the host
# ---------------------------------------------------------------------------
#
# Stronger than the preprocessor comparison and still device-free: nvcc's
# ``-ptx`` is a host-side compile.  It says nothing about NVRTC's cubin, which
# is what AC-P2.2 leg 1 measures on the card, but a PTX difference here would
# mean the guard changed the generated code and the device leg could not
# possibly pass.  Cheap enough to keep: ~2 s per compile.


def _nvcc() -> list[str] | None:
    """An ``nvcc -ptx`` command prefix, or ``None`` if the toolkit is absent."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    command = [nvcc, "-ptx", "-std=c++17", "-arch=sm_120"]
    host = _host_preprocessor()
    if host is not None:
        # nvcc drives a host compiler for the front end; point it at the one
        # this box has rather than requiring it on PATH.
        command += ["-ccbin", str(Path(host[0]).parent)]
    return command


def _ptx(source: str, tmp_path: Path, name: str,
         extra: tuple[str, ...] = ()) -> str:
    """Host-compiled PTX for one assembled translation unit.

    nvcc stamps the input file name into a leading comment, so comment lines
    are stripped before the text is compared.
    """
    command = _nvcc()
    assert command is not None
    src = tmp_path / f"{name}.cu"
    out = tmp_path / f"{name}.ptx"
    src.write_text(source, encoding="utf-8", newline="")
    result = subprocess.run(command + list(extra) + ["-o", str(out), str(src)],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    return re.sub(r"^//.*$", "", out.read_text(encoding="utf-8"), flags=re.M)


@pytest.mark.skipif(_nvcc() is None, reason="no CUDA toolkit installed")
def test_the_guard_generates_identical_code_at_the_shipped_tier(tmp_path):
    """The whole point, compiled: same PTX, guarded or not."""
    assembled = module_source("acoustic")
    assert assembled.count(_GUARDED) == 1
    pre_change = assembled.replace(_GUARDED, SHIPPED_LITERAL + "\n", 1)

    assert (_ptx(assembled, tmp_path, "guarded")
            == _ptx(pre_change, tmp_path, "pre_change"))


@pytest.mark.skipif(_nvcc() is None, reason="no CUDA toolkit installed")
def test_the_ptx_comparison_can_fail(tmp_path):
    """Negative control: a deeper tier enlarges the local frame, so the
    generated code MUST move.  Without this the comparison above could be
    passing on two identically-broken compiles."""
    assembled = module_source("acoustic")
    pre_change = assembled.replace(_GUARDED, SHIPPED_LITERAL + "\n", 1)

    assert (_ptx(assembled, tmp_path, "tiered", ("-DWPHI_MAX_LEV=193",))
            != _ptx(pre_change, tmp_path, "pre_change"))


# ---------------------------------------------------------------------------
# 4. The local-memory rail prices the tier it compiles
# ---------------------------------------------------------------------------

def test_the_shipped_tier_prices_the_driver_measured_frame():
    spec = pf.ACOUSTIC_TIER_FRAME
    assert spec.module == "acoustic" and spec.define == "WPHI_MAX_LEV"
    assert spec.shipped_tier == SHIPPED_TIER
    assert (spec.frame_bytes(SHIPPED_TIER)
            == pf.KERNEL_MAX_LOCAL_SIZE_BYTES["acoustic"])


@pytest.mark.parametrize("tier,frame", [(129, 544), (193, 800), (257, 1056)])
def test_a_deeper_tier_costs_four_bytes_per_added_full_level(tier, frame):
    assert pf.ACOUSTIC_TIER_FRAME.frame_bytes(tier) == frame


def test_a_tier_below_the_shipped_one_is_refused():
    with pytest.raises(ValueError, match="below the shipped tier"):
        pf.ACOUSTIC_TIER_FRAME.frame_bytes(SHIPPED_TIER - 1)


def _experiment(nz: int):
    cfg = RunConfig(nx=8, ny=6, nz=nz, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=1.0, run_seconds=10.0)
    return experiment_from_run_config(cfg, datetime(2026, 8, 4, 0))


@pytest.mark.parametrize("nz,frame", [(4, 544), (128, 544),
                                      (129, 800), (160, 800), (200, 1056)])
def test_the_experiment_price_follows_the_domain_depth(nz, frame):
    assert pf.kernel_local_frame_bytes(_experiment(nz))["acoustic"] == frame


def test_a_below_ceiling_experiment_prices_exactly_what_it_priced_before():
    """No pre-existing configuration may see its rail estimate move."""
    frames = pf.kernel_local_frame_bytes(_experiment(49))
    assert frames["acoustic"] == pf.KERNEL_MAX_LOCAL_SIZE_BYTES["acoustic"]


# ---------------------------------------------------------------------------
# 5. The host mirror above the old ceiling
# ---------------------------------------------------------------------------

def _deep_cfg(nz: int, *, top_lid: bool = False) -> RunConfig:
    return RunConfig(nx=1, ny=1, nz=nz, dx=1000.0, dy=1000.0,
                     ztop=20000.0, dt=1.0, run_seconds=0.0,
                     top_lid=top_lid)


def _deep_column(nz: int):
    """One non-hybrid column of ``nz`` layers with a stratified reference."""
    shape, fshape = (nz, 1, 1), (nz + 1, 1, 1)
    eta = np.linspace(1.0, 0.0, nz + 1)
    p_top, p_sfc = 5_000.0, 95_000.0
    p_full = p_top + eta * (p_sfc - p_top)
    p_half = 0.5 * (p_full[:-1] + p_full[1:])
    dnw = np.diff(eta)
    dn = np.zeros(nz)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    meta = {
        "c1h": np.ones(nz), "c2h": np.zeros(nz),
        "c1f": np.ones(nz + 1), "c2f": np.zeros(nz + 1),
        "mub2d": np.full((1, 1), p_sfc - p_top), "mup": np.zeros((1, 1)),
        "thb": np.linspace(300.0, 420.0, nz),
        "thp": np.zeros(shape),
        "p": p_half[:, None, None].copy(),
        "alt": np.linspace(0.85, 6.0, nz)[:, None, None].copy(),
        "phb": np.linspace(0.0, 20000.0 * c.G, nz + 1)[:, None, None].copy(),
        "php": np.zeros(fshape),
        "rdnw": 1.0 / dnw,
        "rdn": np.concatenate(([0.0], 1.0 / dn[1:])),
        "fnm": np.concatenate(([0.0], np.full(nz - 1, 0.5))),
        "fnp": np.concatenate(([0.0], np.full(nz - 1, 0.5))),
        "msft": np.ones((1, 1)), "ht": np.zeros((1, 1)),
        "cf1": 1.5, "cf2": -0.6, "cf3": 0.1,
        "rw_t": np.zeros(fshape), "rph_t": np.zeros(fshape),
        "w": np.zeros(fshape),
    }
    pp = {"mu_pp": np.zeros((1, 1)), "th_pp": np.zeros(shape),
          "ph_pp": np.zeros(fshape), "w_pp": np.zeros(fshape)}
    new = {"mu_pp": np.zeros((1, 1)), "th_pp": np.zeros(shape),
           "ww_pp": np.zeros(fshape), "u_pp": np.zeros((nz, 1, 2)),
           "v_pp": np.zeros((nz, 2, 1))}
    return pp, new, meta


@pytest.mark.parametrize("nz", [128, 160, 192, 256])
def test_the_quiescent_deep_column_stays_exactly_zero(nz):
    """A column with no forcing has no response, at any admitted depth."""
    cfg = _deep_cfg(nz)
    pp, new, meta = _deep_column(nz)
    w, ph = np_advance_w_phi(pp, new, meta, cfg, 1.0)
    assert np.count_nonzero(w) == 0
    assert np.count_nonzero(ph) == 0


@pytest.mark.parametrize("nz", [128, 160, 192])
def test_the_deep_solve_matches_an_independently_assembled_dense_solve(nz):
    """The Thomas sweep above the old ceiling, checked by dense algebra.

    Same construction as the nz=4 open-top case in
    ``tests/test_acoustic_npref.py``, generalized to depth: the tridiagonal
    operator is rebuilt row by row from the WRF coefficient expressions and
    inverted with ``numpy.linalg.solve``.  Nothing here reads the mirror's
    own factorization, so a depth-dependent indexing error in the sweep --
    exactly the failure a level lift could introduce -- has somewhere to
    show up.
    """
    dtau = 0.5
    cfg = _deep_cfg(nz)
    pp, new, meta = _deep_column(nz)
    meta["rw_t"][-1, 0, 0] = 2.0
    meta["rw_t"][nz // 2, 0, 0] = -1.25

    w, ph = np_advance_w_phi(pp, new, meta, cfg, dtau)

    mut = float(meta["mub2d"][0, 0])
    c2a = c.GAMMA * meta["p"][:, 0, 0] / meta["alt"][:, 0, 0]
    chm = meta["c1h"] * mut + meta["c2h"]
    cfm = meta["c1f"] * mut + meta["c2f"]
    cof = (0.5 * dtau * c.G * (1.0 + cfg.epssm)) ** 2
    matrix = np.zeros((nz, nz))
    for k in range(1, nz):
        row = k - 1
        matrix[row, row] = 1.0 + cof * meta["rdn"][k] * (
            meta["rdnw"][k] * c2a[k] / (chm[k] * cfm[k])
            + meta["rdnw"][k - 1] * c2a[k - 1] / (chm[k - 1] * cfm[k]))
        if k > 1:
            matrix[row, row - 1] = (
                -cof * meta["rdn"][k] * meta["rdnw"][k - 1]
                * c2a[k - 1] / (chm[k - 1] * cfm[k - 1]))
        matrix[row, row + 1] = (
            -cof * meta["rdn"][k] * meta["rdnw"][k]
            * c2a[k] / (chm[k] * cfm[k + 1]))
    matrix[-1, -2] = (-2.0 * cof * meta["rdnw"][-1] ** 2 * c2a[-1]
                      / (chm[-1] * cfm[-2]))
    matrix[-1, -1] = (1.0 + 2.0 * cof * meta["rdnw"][-1] ** 2 * c2a[-1]
                      / (chm[-1] * cfm[-1]))
    forcing = dtau * meta["rw_t"][1:, 0, 0]
    expected_w = np.linalg.solve(matrix, forcing)

    np.testing.assert_allclose(w[1:, 0, 0], expected_w, rtol=2e-10, atol=1e-14)
    np.testing.assert_allclose(
        ph[1:, 0, 0],
        0.5 * dtau * c.G * (1.0 + cfg.epssm) * expected_w / cfm[1:],
        rtol=2e-10, atol=1e-14)
    assert np.count_nonzero(w[1:-1, 0, 0]) == nz - 1, (
        "the whole deep column must couple, not just the forced rows")


def test_the_deep_mirror_is_not_silently_truncated_at_the_old_ceiling():
    """A perturbation above level 128 must reach the ground.

    The failure mode a naive lift produces is a solve that runs to 128 and
    leaves the rest untouched.  Forcing ONLY the top row of a 192-level
    column and requiring a nonzero response at every interior level below it
    is the shape of that falsification.
    """
    nz = 192
    cfg = _deep_cfg(nz)
    pp, new, meta = _deep_column(nz)
    meta["rw_t"][nz, 0, 0] = 3.0

    w, _ = np_advance_w_phi(pp, new, meta, cfg, 0.5)

    assert np.all(np.isfinite(w))
    assert np.count_nonzero(w[1:, 0, 0]) == nz
    assert abs(w[1, 0, 0]) > 0.0
