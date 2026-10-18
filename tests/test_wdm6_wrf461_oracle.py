"""WDM6 (mp_physics=16) against unmodified WRF v4.6.1, word for word.

The oracle is tools/wdm6_wrf461_oracle: WRF's own ``wdm6`` entry of the
byte-frozen ``phys/module_mp_wdm6.F`` (sha256 78b7c148...), called exactly as
``module_microphysics_driver.F`` calls it, at gfortran 13.3 -O0 with glibc
2.39 on the oracle host.  The fixture is 84 columns (convective land and
ocean, stable night, winter snow and ice, a melting layer, high terrain, a
cold anvil below -40 C, rain evaporating into dry air, numberless rain and
cloud, random mixtures and edge cases) in seven configurations: four time
steps (one, three or eight minor loops; 1 to 10 consecutive calls), both
``hail_opt`` arms, the cold-start CCN fill, and a 72-level copy that WOOF
compiles as its second kernel tier.

Two Fortran arms are recorded (tests/data/oracles/wdm6):

* ``stock`` -- the unmodified file, the comparison of record;
* ``defined`` -- the same file with the three documented defined behaviours
  of docs/wdm6_oracle_known_deltas.md substituted (sections 2, 5, 6).

WOOF's device kernel must equal the DEFINED arm in every output word of
every step under the strict build, and wherever the two Fortran arms agree
it therefore equals stock WRF too.  Under the default build the unit already
compiles without multiply-add contraction; the only remaining difference is
the loader's flush-to-zero route, and it may reach only columns into which a
subnormal moisture value enters a call.
"""

from __future__ import annotations

import gzip
import importlib.util
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "oracles" / "wdm6"
TOOLS = ROOT / "tools" / "wdm6_wrf461_oracle"
KERNEL = ROOT / "gpuwm" / "core" / "kernels" / "wdm6.cu"
CONFIGS = ("dt20_hail0", "dt20_hail1", "dt20_hail0_coldstart", "dt300_hail0",
           "dt150_hail1_x3", "dt60_hail0_x10", "deep72_dt30_hail0_x2")
FLT_MIN = np.float32(1.17549435e-38)


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(
        f"wdm6_oracle_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    sys.modules[spec.name] = module
    # The tool's modules import their siblings by bare name ("from oracle_io
    # import read_input").  Four oracle tools each have an oracle_io.py, and
    # in one GPU-shard process the Thompson column oracle's had already been
    # cached under that name, so this import resolved to it and failed (2.9.0
    # round 2, test host D).  Bare sibling names are cleared while this tool loads
    # and the caller's entries are put back afterwards.
    siblings = {path.stem for path in TOOLS.glob("*.py")}
    saved = {key: sys.modules.pop(key) for key in list(sys.modules)
             if key in siblings}
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(TOOLS))
        for key in siblings:
            sys.modules.pop(key, None)
        sys.modules.update(saved)
    return module


IO = _tool("oracle_io")


def _gunzip(src: Path, dst: Path) -> Path:
    dst.write_bytes(gzip.decompress(src.read_bytes()))
    return dst


def _case(tmp_path: Path, config: str):
    path = _gunzip(DATA / f"{config}.in.gz", tmp_path / f"{config}.in")
    return path, IO.read_input(path)


def _reference(tmp_path: Path, config: str, arm: str, case):
    path = _gunzip(DATA / f"{config}.{arm}.out.gz",
                   tmp_path / f"{config}.{arm}.out")
    return IO.read_output(path, case.ncol, case.nz, case.nsteps)


def _subnormal_columns(levels) -> np.ndarray:
    first = next(iter(levels.values()))
    flags = np.zeros(first.shape[1], bool)
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr"):
        a = np.abs(np.asarray(levels[name], np.float32))
        flags |= ((a > 0) & (a < FLT_MIN)).any(axis=0)
    return flags


def _columns_fed_a_subnormal(case, reference_steps) -> np.ndarray:
    """Columns where a subnormal moisture value ENTERS some call: from the
    input, or from the scheme's own output of an earlier step (WRF's words,
    which the strict build reproduces)."""
    fed = _subnormal_columns(case.levels)
    for levels, _ in reference_steps[:-1]:
        fed |= _subnormal_columns(levels)
    return fed


def _differing(woof_steps, ref_steps, *, skip=(), scale_reference_radii=True):
    """Per-field count of differing words, and the columns that differ."""
    counts, columns = {}, None
    for (wl, ws), (rl, rs) in zip(woof_steps, ref_steps):
        for name in IO.OUTPUT_LEVEL_FIELDS + IO.OUTPUT_SURFACE_FIELDS:
            if name in skip:
                continue
            level = name in IO.OUTPUT_LEVEL_FIELDS
            w = np.asarray((wl if level else ws)[name], np.float32)
            r = np.asarray((rl if level else rs)[name], np.float32)
            if scale_reference_radii and name in ("re_cloud", "re_ice", "re_snow"):
                # WRF's metres, scaled by RRTMG's own REAL multiply.
                r = (r * np.float32(1.0e6)).astype(np.float32)
            diff = w.view(np.uint32) != r.view(np.uint32)
            colhit = diff.any(axis=0) if level else diff
            columns = colhit if columns is None else (columns | colhit)
            counts[name] = counts.get(name, 0) + int(diff.sum())
    return counts, columns


# ---------------------------------------------------------------------------
# Provenance and the coefficient words (CPU)
# ---------------------------------------------------------------------------

def test_the_oracle_data_names_the_pinned_wrf_source():
    sums = (DATA / "oracle-sha256sums.txt").read_text(encoding="utf-8")
    assert ("78b7c148f7c29b55670f137636d632aba040bf338d8214e1a5dd191dacf67d70"
            "  <wrf-4.6.1-checkout>/phys/module_mp_wdm6.F") in sums
    assert (DATA / "wrf-commit.txt").read_text().strip() == (
        "d66e442fccc04111067e29274c9f9eaccc3cef28")
    assert "13.3.0" in (DATA / "compiler.txt").read_text()
    # The linked reference stays on scalar libm.
    report = (DATA / "libmvec-report.txt").read_text()
    o0 = report.split("# -O2")[0]
    assert "_ZGV" not in o0


def test_the_split_evidence_arms_edit_the_defined_arm_s_own_text():
    """split_arm.py (the only5t and fineN evidence arms) rewrites blocks of
    the defined arm's section 5 by exact text.  If defined_arm.py changes
    those blocks the evidence arms would refuse at build time on the box;
    this catches the drift on any CPU."""
    defined = _tool("defined_arm")
    split = _tool("split_arm")
    assert defined._RAIN_DEFINED.count(split._WOOF_COUNT) == 1
    assert defined._RAIN_DEFINED.count(split._STEPS) == 1
    # WRF's own count, verbatim from module_mp_wdm6.F:795-803.
    assert "numdt(i) = max(nint(max(work1(i,k,1),workn(i,k))*dtcld+.5),1)" \
        in split._WRF_COUNT


def _dump(hail_opt: int) -> dict[str, int]:
    words = {}
    for line in (DATA / f"wdm6init-hail{hail_opt}.txt").read_text().splitlines():
        name, bits, _ = line.split()
        words[name] = int(bits, 16)
    return words


def _f32_bits(text: str) -> int:
    return int(np.float32(float(text.rstrip("f"))).view(np.uint32))


def test_the_kernel_bakes_wdm6init_s_own_words():
    """Every coefficient literal in wdm6.cu is the word WRF's wdm6init
    leaves in its saved variable, not a float64 product rounded once."""
    from gpuwm.core.wdm6_constants import WDM6_WRF_RIMED_WORDS, WDM6_WRF_WORDS

    source = KERNEL.read_text(encoding="utf-8")
    defines = dict(re.findall(
        r"#define WDM6_([A-Z0-9]+)\s+([0-9.eE+-]+f)\b", source))
    checked = 0
    for name, bits in WDM6_WRF_WORDS.items():
        key = name.upper()
        assert key in defines, f"WDM6_{key} is not baked in wdm6.cu"
        assert _f32_bits(defines[key]) == bits, name
        checked += 1
    assert checked == len(WDM6_WRF_WORDS)
    for hail_opt in (0, 1):
        dump = _dump(hail_opt)
        assert {k: dump[k] for k in WDM6_WRF_WORDS} == WDM6_WRF_WORDS
        assert {k: dump[k] for k in WDM6_WRF_RIMED_WORDS[hail_opt]} == (
            WDM6_WRF_RIMED_WORDS[hail_opt])
    arms = re.split(r"\} else \{", source.split(
        "wdm6_rimed_constants(int hail_opt)")[1].split("return r;")[0])
    for hail_opt, arm in ((1, arms[0]), (0, arms[1])):
        baked = dict(re.findall(r"r\.(\w+) = ([0-9.eE+-]+f);", arm))
        for name, bits in WDM6_WRF_RIMED_WORDS[hail_opt].items():
            assert _f32_bits(baked[name]) == bits, (hail_opt, name)
    # The two folded constants of wdm62D the kernel spells out.
    assert "(actr*1.E-6)**3" in source and "WDM6_ACTR3   3.3750003e-18f" in source
    assert "2.88449907f" in source        # (24.)**(.3333333), 0x40389BA2


def test_the_float64_derivation_stays_within_the_wrf_words():
    """coefficients() is the readable formula; WRF's REAL words are the
    values.  They must stay close enough that a typo shows."""
    from gpuwm.core.wdm6_constants import (WDM6_WRF_RIMED_WORDS,
                                           WDM6_WRF_WORDS, coefficients,
                                           wrf_word)

    for hail_opt in (0, 1):
        c = coefficients(hail_opt)
        for name, bits in {**WDM6_WRF_WORDS,
                           **WDM6_WRF_RIMED_WORDS[hail_opt]}.items():
            if name == "pi":
                continue
            value = getattr(c, name, None)
            if value is None:
                value = getattr(c.rimed, name)
            assert wrf_word(bits) == pytest.approx(value, rel=1.0e-5), name


def test_the_reflectivity_words_are_wrf_s():
    from gpuwm.core.refl import wdm6_radar_words

    order = ("pi", "xam_r", "xcrg3", "xcrg4", "xorg2", "xobmr", "xcre2",
             "xcre4", "xam_s", "xcsg3", "xcsg4", "xcse1", "xcse4", "xocms",
             "xobms", None, None, "xam_g", "xcgg3", "xcgg4", "xcge1", "xcge4",
             "xocmg", "xobmg", None, None, "n0g")
    for hail_opt in (0, 1):
        dump = _dump(hail_opt)
        fw, dw = wdm6_radar_words(hail_opt)
        bits = fw.view(np.uint32)
        for index, name in enumerate(order):
            if name is not None:
                assert int(bits[index]) == dump[name], (hail_opt, name)
        # xbm = 3, xmu = 0 for snow and graupel (wdm6init :2201-2206).
        assert fw[15] == 3.0 and fw[16] == 0.0 and fw[24] == 3.0
        assert fw[25] == 0.0
        text = (DATA / f"wdm6init-hail{hail_opt}.txt").read_text()
        dbl = {line.split()[0]: int(line.split()[1], 16)
               for line in text.splitlines() if len(line.split()[1]) == 16}
        assert [int(v) for v in dw.view(np.uint64)] == [
            dbl[n] for n in ("K_w", "PI5", "lamda4", "m_w_0_re", "m_w_0_im",
                             "m_i_0_re", "m_i_0_im")]


def test_the_reflectivity_bin_tables_are_wrf_s():
    """radar_init's bins and Simpson weights, computed on this host, are
    WRF's words (the device kernel reads them from gpuwm.core.refl)."""
    if platform.libc_ver()[0] != "glibc":
        pytest.skip("the bin tables are graded on a glibc host")
    from gpuwm.core.refl import radar_init

    rc = radar_init()
    text = (DATA / "wdm6init-hail0.txt").read_text()
    dbl = {line.split()[0]: int(line.split()[1], 16)
           for line in text.splitlines()}
    for name, table in (("xxDs", rc.xxds), ("xdts", rc.xdts),
                        ("xxDg", rc.xxdg), ("xdtg", rc.xdtg),
                        ("simpson", rc.simpson)):
        got = np.asarray(table, np.float64).view(np.uint64)
        want = [dbl[f"{name}{n + 1:02d}"] for n in range(50)]
        assert [int(v) for v in got] == want, name


# ---------------------------------------------------------------------------
# The host mirror of the kernel source (CPU, glibc host)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config", ("dt20_hail0", "dt300_hail0",
                                    "dt150_hail1_x3"))
def test_the_kernel_source_on_the_host_is_the_defined_arm(tmp_path, config):
    """wdm6.cu compiled for the host (its WDM6_CPU_MIRROR arm, the C
    library's float32 exp/log/pow) lands on every state and radius word of
    WRF's defined arm.  Reflectivity has no host mirror."""
    compiler = shutil.which("g++")
    if compiler is None or platform.libc_ver()[0] != "glibc":
        pytest.skip("the host mirror is graded with g++ on a glibc host")
    exe = tmp_path / "wdm6_mirror"
    subprocess.run([compiler, "-std=c++17", "-O0", "-ffp-contract=off",
                    "-I", str(ROOT), str(TOOLS / "mirror_main.cpp"), "-o",
                    str(exe)], check=True, capture_output=True, text=True)
    inp, case = _case(tmp_path, config)
    out = tmp_path / f"{config}.mirror.out"
    subprocess.run([str(exe), str(inp), str(out)], check=True)
    mirror = IO.read_output(out, case.ncol, case.nz, case.nsteps)
    counts, _ = _differing(mirror, _reference(tmp_path, config, "defined",
                                              case), skip=("refl_10cm",))
    assert {k: v for k, v in counts.items() if v} == {}


# ---------------------------------------------------------------------------
# The device kernel (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("config", CONFIGS)
def test_the_device_kernel_is_wrf_word_for_word(tmp_path, config):
    from gpuwm.wrf_exact import ENABLED

    run_woof = _tool("run_woof")
    inp, case = _case(tmp_path, config)
    out = tmp_path / f"{config}.woof.out"
    run_woof.run(inp, out)
    woof = IO.read_output(out, case.ncol, case.nz, case.nsteps)
    defined = _reference(tmp_path, config, "defined", case)
    stock = _reference(tmp_path, config, "stock", case)

    counts, columns = _differing(woof, defined)
    if ENABLED:
        # Strict build: every word of every field of every step.
        assert {k: v for k, v in counts.items() if v} == {}
    else:
        # Default build: flush-to-zero may only reach the STATE of columns
        # into which a subnormal value enters a call; every other column's
        # state is still every word.  Outside them, a surface accumulator
        # may lose a vanishing fallout whose own product underflowed
        # (measured: under 5e-34 mm), and nothing else.
        fed = _columns_fed_a_subnormal(case, defined)
        _, state_columns = _differing(
            woof, defined, skip=IO.OUTPUT_SURFACE_FIELDS)
        assert not (state_columns & ~fed).any(), (
            counts, np.flatnonzero(state_columns & ~fed))
        for (wl, ws), (rl, rs) in zip(woof, defined):
            for name in IO.OUTPUT_SURFACE_FIELDS:
                w = np.asarray(ws[name], np.float64)[~fed]
                r = np.asarray(rs[name], np.float64)[~fed]
                assert np.abs(w - r).max(initial=0.0) < 1.0e-30, name

    # Where WRF's two arms agree, WOOF is stock WRF.
    _, arms_differ = _differing(defined, stock, scale_reference_radii=False)
    _, woof_vs_stock = _differing(woof, stock)
    if ENABLED:
        assert not (woof_vs_stock & ~arms_differ).any()
