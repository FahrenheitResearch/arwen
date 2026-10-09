"""Compile WRF 4.6.1's own damping routines for the upper-damping oracle.

    python tools/upper_damping_wrf461_oracle/build.py WRF_SOURCE OUT [--as-built WRF_BUILD]

Two references are built from the pinned, unmodified source:

* ``OUT/O0``    gfortran -O0 -ffp-contract=off (the column-driver route).
* ``OUT/stock`` WRF's own GNU flags from configure.defaults
                (-O2 -ftree-vectorize -funroll-loops ...), plus -fPIC.

Each holds ``libsmallstep_oracle.so`` (all eight small-step routines,
advance_w among them, through the generated C-ABI wrappers of
tools/smallstep_wrf471_oracle/build.py) with its ``schema.json``, and
``libw_damp_oracle.so``: the ``w_damp`` subroutine text cut verbatim out
of dyn_em/module_big_step_utilities_em.F into a one-routine module, with
data-only stand-ins for the framework it names (grid_config_rec_type
fields, wrf_err_message, wrf_debug, the time/grid name strings).  No WRF
arithmetic is replaced.

``--as-built`` points at a compiled WRF tree (W1:
/work/pverify/wrf-build/serial/WRF-4.6.1).  The floating-point instruction
histogram and the libm imports of advance_w and w_damp in that tree's
objects are then compared with the stock-flag build here, as evidence that
the stock reference computes what wrf.exe computes.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.smallstep_wrf471_oracle.build import PIN as SMALLSTEP_PIN, generate  # noqa: E402

WRF_TAG = "v4.6.1"
WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
#: LF checkouts on W1 (src/WRF-4.6.1).  module_small_step_em.F and
#: module_model_constants.F are byte-identical to the v4.7.1 pins the
#: small-step generator carries, which build() re-checks.
BIG_STEP = "dyn_em/module_big_step_utilities_em.F"
BIG_STEP_SHA = "8f0649b458fceabd5c31c87ec9f266840fb164e2629322904a3f6ee04c72331a"
O0 = ["-O0", "-ffp-contract=off"]
STOCK = ["-O2", "-ftree-vectorize", "-funroll-loops", "-w", "-ffree-form",
         "-ffree-line-length-none", "-fconvert=big-endian", "-frecord-marker=4",
         "-fallow-argument-mismatch", "-fallow-invalid-boz"]
COMMON = ["-fPIC", "-cpp", "-ffree-form", "-ffree-line-length-none", "-Dwrfmodel",
          "-DEM_CORE=1", "-DRWORDSIZE=4", "-DNONSTANDARD_SYSTEM_SUBR", "-DWRF_USE_CLM"]
W_DAMP_CONFIG = {"w_crit_cfl": "real", "fft_filter_lat": "real",
                 "zadvect_implicit": "integer", "w_damping": "integer",
                 "polar": "logical"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def w_damp_sources(text: str):
    match = re.search(r"(?ims)^SUBROUTINE w_damp\(.*?^END SUBROUTINE W_DAMP\s*$", text)
    if match is None:
        raise ValueError("w_damp was not found in module_big_step_utilities_em.F")
    routine = match.group()
    fields = sorted(set(f.lower() for f in re.findall(r"config_flags%(\w+)", routine, re.I)))
    if set(fields) != set(W_DAMP_CONFIG):
        raise ValueError(f"w_damp reads config fields {fields}, expected {sorted(W_DAMP_CONFIG)}")
    support = ["module module_configure", "implicit none", "type grid_config_rec_type"]
    for name, kind in W_DAMP_CONFIG.items():
        support.append(f"{kind} :: {name}" + (" = .false." if kind == "logical" else " = 0"))
    support += ["end type", "end module",
                "module module_wrf_error", "character(len=256) :: wrf_err_message", "end module",
                "module module_llxy", "end module",
                "subroutine wrf_debug(level, str)", "integer :: level",
                "character(len=*) :: str", "end subroutine",
                "subroutine get_current_time_string(s)", "character(len=*) :: s",
                "s = 'oracle'", "end subroutine",
                "subroutine get_current_grid_name(s)", "character(len=*) :: s",
                "s = 'd01'", "end subroutine"]
    module = ("module module_w_damp_oracle\n use module_model_constants\n"
              " use module_configure, only: grid_config_rec_type\n use module_wrf_error\n"
              " implicit none\ncontains\n" + routine + "\nend module\n")
    wrapper = """subroutine oracle_w_damp(rw_tend, max_vert_cfl, max_horiz_cfl, u, v, ww, w, mut, &
    c1f, c2f, rdnw, rdx, rdy, msfux, msfuy, msfvx, msfvy, dt, w_crit_cfl, ieva, &
    nx, ny, nz) bind(C)
  use iso_c_binding
  use module_configure, only: grid_config_rec_type
  use module_w_damp_oracle, only: w_damp
  implicit none
  integer(c_int) :: nx, ny, nz, ieva
  real(c_float) :: max_vert_cfl, max_horiz_cfl, rdx, rdy, dt, w_crit_cfl
  real(c_float) :: rw_tend(nx+1, nz+1, ny+1), u(nx+1, nz+1, ny+1), v(nx+1, nz+1, ny+1)
  real(c_float) :: ww(nx+1, nz+1, ny+1), w(nx+1, nz+1, ny+1)
  real(c_float) :: mut(nx+1, ny+1), msfux(nx+1, ny+1), msfuy(nx+1, ny+1)
  real(c_float) :: msfvx(nx+1, ny+1), msfvy(nx+1, ny+1)
  real(c_float) :: c1f(nz+1), c2f(nz+1), rdnw(nz+1)
  type(grid_config_rec_type) :: cf
  cf%w_damping = 1
  cf%polar = .false.
  cf%fft_filter_lat = 45.
  cf%w_crit_cfl = w_crit_cfl
  cf%zadvect_implicit = ieva
  call w_damp(rw_tend, max_vert_cfl, max_horiz_cfl, u, v, ww, w, mut, c1f, c2f, rdnw, &
              rdx, rdy, msfux, msfuy, msfvx, msfvy, dt, cf, &
              1, nx+1, 1, ny+1, 1, nz+1, 1, nx+1, 1, ny+1, 1, nz+1, &
              1, nx+1, 1, ny+1, 1, nz+1)
end subroutine
"""
    return "\n".join(support) + "\n", module, wrapper


def compile_flavour(source_root: Path, out: Path, flags):
    out.mkdir(parents=True, exist_ok=True)
    text = (source_root / "dyn_em/module_small_step_em.F").read_text()
    stub, wrappers, schema = generate(text)
    (out / "configure_stub.F90").write_text(stub)
    (out / "wrappers.F90").write_text(wrappers)
    (out / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    commands = []

    def run(cmd, cwd):
        subprocess.run(cmd, cwd=cwd, check=True)
        commands.append(cmd)

    small = out / "smallstep"
    small.mkdir(exist_ok=True)
    for path in (out / "configure_stub.F90", source_root / "share/module_model_constants.F",
                 source_root / "dyn_em/module_small_step_em.F", out / "wrappers.F90"):
        run(["gfortran", "-c", *flags, *COMMON, str(path.resolve())], small)
    run(["gfortran", "-shared", "-o", str(out / "libsmallstep_oracle.so"), "configure_stub.o",
         "module_model_constants.o", "module_small_step_em.o", "wrappers.o"], small)

    support, module, wrapper = w_damp_sources((source_root / BIG_STEP).read_text())
    damp = out / "w_damp"
    damp.mkdir(exist_ok=True)
    for name, body in (("w_damp_support.F90", support), ("w_damp_module.F90", module),
                       ("w_damp_wrapper.F90", wrapper)):
        (damp / name).write_text(body)
    run(["gfortran", "-c", *flags, *COMMON, str((source_root / "share/module_model_constants.F").resolve())], damp)
    for name in ("w_damp_support.F90", "w_damp_module.F90", "w_damp_wrapper.F90"):
        run(["gfortran", "-c", *flags, *COMMON, name], damp)
    run(["gfortran", "-shared", "-o", str(out / "libw_damp_oracle.so"), "module_model_constants.o",
         "w_damp_support.o", "w_damp_module.o", "w_damp_wrapper.o"], damp)
    return commands


def fp_histogram(obj: Path, symbol: str):
    """Floating-point instruction counts and external symbols one function references."""
    text = subprocess.run(["objdump", "-dr", "--no-show-raw-insn", f"--disassemble={symbol}",
                           str(obj)], check=True, capture_output=True, text=True).stdout
    counts = collections.Counter()
    calls = collections.Counter()
    for line in text.splitlines():
        reloc = re.search(r"R_X86_64_(?:PLT32|PC32|GOTPCRELX?|REX_GOTPCRELX)\s+([A-Za-z_]\w*)", line)
        if reloc and not reloc.group(1).startswith((".", "__module")):
            calls[reloc.group(1)] += 1
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        op = parts[1].split()[0] if parts[1].split() else ""
        if re.match(r"v?(add|sub|mul|div|sqrt|min|max|fmadd|fmsub|fnmadd|cvt\w*|ucomi|comi|and|xor|andn)(ss|ps|sd|pd)?$", op) \
                and op.endswith(("ss", "ps", "sd", "pd")):
            counts[op] += 1
    return {"fp_instructions": dict(sorted(counts.items())), "calls": dict(sorted(calls.items()))}


def as_built(build_tree: Path, out: Path):
    report = {}
    for label, obj, symbol in (
            ("advance_w", build_tree / "dyn_em/module_small_step_em.o",
             "__module_small_step_em_MOD_advance_w"),
            ("w_damp", build_tree / "dyn_em/module_big_step_utilities_em.o",
             "__module_big_step_utilities_em_MOD_w_damp")):
        mine = (out / "stock/smallstep/module_small_step_em.o" if label == "advance_w"
                else out / "stock/w_damp/w_damp_module.o")
        my_symbol = symbol if label == "advance_w" else "__module_w_damp_oracle_MOD_w_damp"
        built, ours = fp_histogram(obj, symbol), fp_histogram(mine, my_symbol)
        report[label] = {"wrf_exe_object": str(obj), "wrf_exe": built, "stock_reference": ours,
                         "fp_instructions_equal": built["fp_instructions"] == ours["fp_instructions"],
                         "external_calls_equal": built["calls"] == ours["calls"]}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--as-built", type=Path)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    pins = {rel: sha(source / rel) for rel in (*SMALLSTEP_PIN, BIG_STEP)}
    for rel, digest in SMALLSTEP_PIN.items():
        if pins[rel] != digest:
            raise SystemExit(f"{rel} differs from the pinned source: {pins[rel]}")
    if pins[BIG_STEP] != BIG_STEP_SHA:
        raise SystemExit(f"{BIG_STEP} differs from the 4.6.1 pin: {pins[BIG_STEP]}")
    receipt = {"wrf_tag": WRF_TAG, "wrf_commit": WRF_COMMIT, "source_sha256": pins,
               "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
               "flavours": {}}
    for name, flags in (("O0", O0), ("stock", STOCK)):
        receipt["flavours"][name] = {"flags": flags + COMMON,
                                     "commands": compile_flavour(source, out / name, flags)}
    for name in ("O0", "stock"):
        for lib in ("libsmallstep_oracle.so", "libw_damp_oracle.so"):
            path = out / name / lib
            imports = subprocess.run(["nm", "-D", "--undefined-only", str(path)], check=True,
                                     capture_output=True, text=True).stdout.split()
            receipt["flavours"][name][lib] = {
                "sha256": sha(path),
                "libm_imports": sorted(s for s in imports if re.match(r"(_ZGV\w+|\w+f)(@.*)?$", s)
                                       and not s.startswith("_gfortran"))}
    if args.as_built:
        receipt["as_built"] = as_built(args.as_built.resolve(), out)
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("wrf_tag", "source_sha256")}, indent=2))
    if "as_built" in receipt:
        print(json.dumps({k: {x: v[x] for x in ("fp_instructions_equal", "external_calls_equal")}
                          for k, v in receipt["as_built"].items()}, indent=2))


if __name__ == "__main__":
    main()
