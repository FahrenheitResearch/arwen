"""The WRF-Chem column-oracle builders must run end to end from this tree.

Rebuilding every published chem oracle on a clean Linux box (W1, gfortran
13.3, glibc 2.39, 2026-10-07; tools/chem_wrf461_verify/rebuild_oracles.sh)
found four ways the committed builders could not reproduce their own
fixtures.  Each test below names the breakage it prevents:

1. gocart/build.sh ran build_ageing.sh as a standalone builder, and
   build_ageing.sh hands off (through build_sulfur.sh) back to gocart/build.sh,
   which reaches build_ageing.sh again: unbounded recursion.
2. gocart/build.sh calls each standalone builder with four arguments, and
   build_inventory.sh / build_niwfa.sh refused anything but three (exit 2).
3. build_sulfur.sh compared its CRLF extract of the CRLF-pinned
   chemics_init.F with the LF-committed chemics_init_solar.inc byte for byte,
   a comparison that cannot pass anywhere.
4. Under ``set -o pipefail`` a bare ``ldd --version | head -1`` dies of
   SIGPIPE (exit 141) once head has its line, failing wesely/build.sh after it
   had already written its fixtures.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tools" / "chem_wrf471_oracle"
GOCART = ORACLE / "gocart"


def _wrapper_pattern() -> re.Pattern[str]:
    text = (GOCART / "build.sh").read_text(encoding="utf-8")
    found = re.search(r"grep -qE '([^']+)' \"\$\{here\}/build_\$\{key\}\.sh\"", text)
    assert found, "gocart/build.sh no longer classifies build_<key>.sh with grep -qE"
    return re.compile(found.group(1))


def _is_wrapper(script: str) -> bool:
    return bool(_wrapper_pattern().search((GOCART / script).read_text(encoding="utf-8")))


def test_ageing_and_sulfur_entry_points_are_not_run_as_their_own_builds():
    # Breakage 1: build_ageing.sh -> build_sulfur.sh -> build.sh -> build_ageing.sh ...
    assert _is_wrapper("build_ageing.sh")
    assert _is_wrapper("build_sulfur.sh")


def test_standalone_builders_never_hand_back_to_a_build_script():
    standalone = [p.name for p in sorted(GOCART.glob("build_*.sh")) if not _is_wrapper(p.name)]
    assert {"build_optics.sh", "build_inventory.sh", "build_niwfa.sh"} <= set(standalone)
    for name in standalone:
        text = (GOCART / name).read_text(encoding="utf-8")
        assert not re.search(r'exec bash "\$\{here\}/build', text), name


_BASH = shutil.which("bash")


@pytest.mark.skipif(_BASH is None, reason="needs bash")
@pytest.mark.parametrize("script", ["build_inventory.sh", "build_niwfa.sh"])
def test_standalone_builders_accept_the_dispatchers_four_arguments(script, tmp_path):
    # Breakage 2: exit 2 ("usage") on gocart/build.sh's own call.
    missing = tmp_path / "no-such-tree"
    run = subprocess.run([_BASH, str(GOCART / script), str(missing), str(tmp_path / "b"),
                          str(tmp_path / "f"), str(tmp_path / "t")],
                         capture_output=True, text=True)
    assert run.returncode != 2, run.stderr
    assert "usage" not in run.stderr


@pytest.mark.skipif(shutil.which("diff") is None, reason="needs GNU diff")
def test_sulfur_include_check_ignores_only_line_endings(tmp_path):
    # Breakage 3: the committed include is LF; the pinned source extract is CRLF.
    script = (GOCART / "build_sulfur.sh").read_text(encoding="utf-8")
    assert ('diff --strip-trailing-cr -q "${build}/chemics_init_solar.inc" '
            '"${here}/chemics_init_solar.inc"') in script
    assert not re.search(r"^\s*cmp .*chemics_init_solar\.inc", script, re.M)
    committed = GOCART / "chemics_init_solar.inc"
    lf = committed.read_bytes()
    assert b"\r" not in lf
    crlf = tmp_path / "crlf.inc"
    crlf.write_bytes(lf.replace(b"\n", b"\r\n"))
    same = subprocess.run(["diff", "--strip-trailing-cr", "-q", str(crlf), str(committed)])
    assert same.returncode == 0
    edited = tmp_path / "edited.inc"
    edited.write_bytes(lf.replace(b"\n", b"\r\n").replace(b"!", b"!x", 1))
    differs = subprocess.run(["diff", "--strip-trailing-cr", "-q", str(edited), str(committed)],
                             capture_output=True)
    assert differs.returncode == 1


def test_no_bare_ldd_version_probe_is_piped_into_head_under_pipefail():
    # Breakage 4: SIGPIPE exit 141 from `ldd --version | head -1` as a command.
    # ldd is a shell script printing line by line, so head's early exit kills
    # it; inside $(...) the status is discarded and nothing fails.
    offenders = []
    for script in sorted(ORACLE.rglob("*.sh")):
        text = script.read_text(encoding="utf-8")
        if "pipefail" not in text:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if re.search(r"\bldd --version\s*\|\s*head\b", line) and "$(" not in line:
                offenders.append(f"{script.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, offenders
