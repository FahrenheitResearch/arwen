"""Write the SPY copies of the two UW PBL sources that call private routines.

    python make_spy.py WRF_SOURCE_ROOT OUTDIR

Reads WRF_SOURCE_ROOT/phys/module_cam_bl_eddy_diff.F and
WRF_SOURCE_ROOT/phys/module_bl_camuwpbl_driver.F (build.sh has already
checked both against SOURCES.sha256) and writes OUTDIR/<same names> with
`call uwspy_*` lines inserted before and after five call statements:

  compute_eddy_diff:  trbintd, caleddy, the in-loop compute_vdiff
  caleddy:            exacol, zisocl
  camuwpbl:           compute_eddy_diff, the outer compute_vdiff

plus one `use uwspy` per module and the column/step/iteration context calls.
Nothing else in either file changes; the inserted calls only read.  build.sh
proves the copies are numerically inert by requiring the spy build's step
outputs to equal the pristine build's bit for bit.  The insertion points are
found by exact text, and the script fails if any is missing or ambiguous, so
it cannot silently instrument a different source.
"""
from __future__ import annotations

import sys
from pathlib import Path

# (name, size) -- size is a Fortran expression in the CALLER's scope, or
# 's' for a scalar real(r8).
TRBINTD = [
    ("z", "pver"), ("ufd", "pver"), ("vfd", "pver"), ("tfd", "pver"),
    ("pmid", "pver"), ("tautotx", "1"), ("tautoty", "1"), ("ustar", "1"),
    ("rrho", "1"), ("s2", "pver"), ("n2", "pver"), ("ri", "pver"),
    ("zi", "pver+1"), ("pi", "pver+1"), ("cldn", "pver"), ("qtfd", "pver"),
    ("qvfd", "pver"), ("qlfd", "pver"), ("qi", "pver"), ("sfi", "pver+1"),
    ("sfuh", "pver"), ("sflh", "pver"), ("slfd", "pver"), ("slv", "pver"),
    ("slslope", "pver"), ("qtslope", "pver"), ("chs", "pver+1"),
    ("chu", "pver+1"), ("cms", "pver+1"), ("cmu", "pver+1"),
    ("minpblh", "1"),
]
CALEDDY = [
    ("slfd", "pver"), ("qtfd", "pver"), ("qlfd", "pver"), ("slv", "pver"),
    ("ufd", "pver"), ("vfd", "pver"), ("pi", "pver+1"), ("z", "pver"),
    ("zi", "pver+1"), ("qflx", "1"), ("shflx", "1"), ("slslope", "pver"),
    ("qtslope", "pver"), ("chu", "pver+1"), ("chs", "pver+1"),
    ("cmu", "pver+1"), ("cms", "pver+1"), ("sfuh", "pver"), ("sflh", "pver"),
    ("n2", "pver"), ("s2", "pver"), ("ri", "pver"), ("rrho", "1"),
    ("pblh", "1"), ("ustar", "1"), ("kvh", "pver+1"), ("kvm", "pver+1"),
    ("kvh_out", "pver+1"), ("kvm_out", "pver+1"), ("tpert", "1"),
    ("qpert", "1"), ("qrl", "pver"), ("tke", "pver+1"), ("bprod", "pver+1"),
    ("sprod", "pver+1"), ("minpblh", "1"), ("wpert", "1"), ("tkes", "1"),
    ("turbtype", "pver+1"), ("sm_aw", "pver+1"), ("kbase_o", "ncvmax"),
    ("ktop_o", "ncvmax"), ("ncvfin_o", "1"), ("kbase_mg", "ncvmax"),
    ("ktop_mg", "ncvmax"), ("ncvfin_mg", "1"), ("kbase_f", "ncvmax"),
    ("ktop_f", "ncvmax"), ("ncvfin_f", "1"), ("wet", "ncvmax"),
    ("web", "ncvmax"), ("jtbu", "ncvmax"), ("jbbu", "ncvmax"),
    ("evhc", "ncvmax"), ("jt2slv", "ncvmax"), ("n2ht", "ncvmax"),
    ("n2hb", "ncvmax"), ("lwp", "ncvmax"), ("opt_depth", "ncvmax"),
    ("radinvfrac", "ncvmax"), ("radf", "ncvmax"), ("wstar", "ncvmax"),
    ("wstar3fact", "ncvmax"), ("ebrk", "ncvmax"), ("wbrk", "ncvmax"),
    ("lbrk", "ncvmax"), ("ricl", "ncvmax"), ("ghcl", "ncvmax"),
    ("shcl", "ncvmax"), ("smcl", "ncvmax"), ("ghi", "pver+1"),
    ("shi", "pver+1"), ("smi", "pver+1"), ("rii", "pver+1"),
    ("lengi", "pver+1"), ("wcap", "pver+1"), ("pblhp", "1"),
    ("cldn", "pver"), ("ipbl", "1"), ("kpblh", "1"), ("wsedl", "pver"),
]
VDIFF_INNER = [
    ("pmid", "pver"), ("pi", "pver+1"), ("rpdel", "pver"), ("t", "pver"),
    ("ztodt", "s"), ("taux", "1"), ("tauy", "1"), ("shflx", "1"),
    ("qflx", "1"), ("kvh_out", "pver+1"), ("kvm_out", "pver+1"),
    ("cgs", "pver+1"), ("cgh", "pver+1"), ("zi", "pver+1"),
    ("ksrftms", "1"), ("ufd", "pver"), ("vfd", "pver"), ("qtfd", "pver"),
    ("slfd", "pver"), ("tauresx", "1"), ("tauresy", "1"),
    ("jnk2d", "pver"), ("jnk1d", "1"),
]
# inside caleddy: (name, size, type) with type r8 (default), i4, l (array
# of logical), l1 (scalar logical)
EXACOL = [
    ("ri", "pver"), ("bflxs", "1"), ("minpblh", "1"), ("zi", "pver+1"),
    ("ktop", "ncvmax", "i4"), ("kbase", "ncvmax", "i4"),
    ("ncvfin", "1", "i4"),
]
ZISOCL = [
    ("z", "pver"), ("zi", "pver+1"), ("n2", "pver"), ("s2", "pver"),
    ("bprod", "pver+1"), ("sprod", "pver+1"), ("bflxs", "1"), ("tkes", "1"),
    ("ncvfin", "1", "i4"), ("kbase", "ncvmax", "i4"),
    ("ktop", "ncvmax", "i4"), ("belongcv", "pver+1", "l"),
    ("ricl", "ncvmax"), ("ghcl", "ncvmax"), ("shcl", "ncvmax"),
    ("smcl", "ncvmax"), ("lbrk", "ncvmax"), ("wbrk", "ncvmax"),
    ("ebrk", "ncvmax"), ("extend", "1", "l1"), ("extend_up", "1", "l1"),
    ("extend_dn", "1", "l1"),
]
EDDY = [
    ("t8", "kte"), ("cloud(:,:,1)", "kte"), ("ztodt", "s"),
    ("cloud(:,:,2)", "kte"), ("cloud(:,:,3)", "kte"), ("s8", "kte"),
    ("rpdel8", "kte"), ("cldn8", "kte"), ("qrl8", "kte"), ("wsedl8", "kte"),
    ("zm8", "kte"), ("zi8", "kte+1"), ("pmid8", "kte"), ("pint8", "kte+1"),
    ("u8", "kte"), ("v8", "kte"), ("taux", "1"), ("tauy", "1"),
    ("shflx", "1"), ("cflx(:,1)", "1"), ("ustar8", "1"), ("pblh", "1"),
    ("kvm_in", "kte+1"), ("kvh_in", "kte+1"), ("kvm", "kte+1"),
    ("kvh", "kte+1"), ("kvq", "kte+1"), ("cgh", "kte+1"), ("cgs", "kte+1"),
    ("tpert", "1"), ("qpert", "1"), ("wpert", "1"), ("tke8", "kte+1"),
    ("bprod", "kte+1"), ("sprod", "kte+1"), ("sfi", "kte+1"),
    ("tauresx", "1"), ("tauresy", "1"), ("ksrftms", "1"), ("ipbl", "1"),
    ("kpblh", "1"), ("wstarPBL", "1"), ("turbtype", "kte+1"),
    ("smaw", "kte+1"),
]
VDIFF_OUTER = [
    ("pmid8", "kte"), ("pint8", "kte+1"), ("rpdel8", "kte"), ("t8", "kte"),
    ("ztodt", "s"), ("taux", "1"), ("tauy", "1"), ("shflx", "1"),
    ("cflx(1,:)", "5"), ("kvh", "kte+1"), ("kvm", "kte+1"),
    ("kvq", "kte+1"), ("cgs", "kte+1"), ("cgh", "kte+1"), ("zi8", "kte+1"),
    ("ksrftms", "1"), ("wind_tends(:,:,1)", "kte"),
    ("wind_tends(:,:,2)", "kte"), ("cloudtnd(:,:,1)", "kte"),
    ("cloudtnd(:,:,2)", "kte"), ("cloudtnd(:,:,3)", "kte"),
    ("cloudtnd(:,:,4)", "kte"), ("cloudtnd(:,:,5)", "kte"),
    ("stnd", "kte"), ("tautmsx", "1"), ("tautmsy", "1"), ("dtk", "kte"),
    ("topflx", "1"), ("tauresx", "1"), ("tauresy", "1"),
]


def safe(name: str) -> str:
    return (name.replace("(:,:,", "_").replace("(:,", "_")
            .replace("(1,:)", "_all").replace(")", ""))


def spy_lines(stage: str, args, indent: str) -> list[str]:
    out = []
    for entry in args:
        name, size = entry[0], entry[1]
        kind = entry[2] if len(entry) > 2 else "r8"
        label = f"{stage}/{safe(name)}"
        if size == "s":
            out.append(f"{indent}call uwspy_s('{label}', {name})\n")
        elif kind == "i4":
            out.append(f"{indent}call uwspy_i4('{label}', {name}, {size})\n")
        elif kind == "l":
            out.append(f"{indent}call uwspy_l('{label}', {name}, {size})\n")
        elif kind == "l1":
            out.append(f"{indent}call uwspy_l1('{label}', {name})\n")
        else:
            out.append(f"{indent}call uwspy_r8('{label}', {name}, {size})\n")
    return out


def code_part(line: str) -> str:
    # strip a trailing comment that is not inside a string (no strings on
    # these call lines)
    i = line.find("!")
    return (line if i < 0 else line[:i]).rstrip()


def find_call(lines, start, needle, stop=None):
    hits = [i for i in range(start, stop if stop is not None else len(lines))
            if needle in lines[i] and not lines[i].lstrip().startswith("!")]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one '{needle}' in range, got {hits}")
    first = hits[0]
    last = first
    while code_part(lines[last]).endswith("&"):
        last += 1
    return first, last


def indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def instrument(lines, start, stop, needle, stage, args):
    first, last = find_call(lines, start, needle, stop)
    ind = indent_of(lines[first])
    before = spy_lines(stage + "_in", args, ind)
    after = spy_lines(stage + "_out", args, ind)
    lines[last + 1:last + 1] = after
    lines[first:first] = before
    return len(before) + len(after)


def locate(lines, needle, start=0):
    hits = [i for i in range(start, len(lines)) if needle in lines[i]]
    if not hits:
        raise SystemExit(f"'{needle}' not found")
    return hits[0]


def spy_eddy(text: str) -> str:
    lines = text.splitlines(keepends=True)
    use = locate(lines, "use diffusion_solver, only : vdiff_selector")
    lines.insert(use + 1, "  use uwspy\n")
    s = locate(lines, "subroutine compute_eddy_diff(")
    e = locate(lines, "end subroutine compute_eddy_diff", s)
    loop = locate(lines, "do iturb = 1, nturb", s)
    lines.insert(loop + 1, "       call uwspy_iter(iturb)\n")
    e += 1
    e += instrument(lines, s, e, "call trbintd(", "trbintd", TRBINTD)
    e += instrument(lines, s, e, "call caleddy(", "caleddy", CALEDDY)
    e += instrument(lines, s, e, "call compute_vdiff(", "vdiff_in_loop",
                    VDIFF_INNER)
    s = locate(lines, "subroutine caleddy(")
    e = locate(lines, "end subroutine caleddy", s)
    e += instrument(lines, s, e, "call exacol(", "exacol", EXACOL)
    e += instrument(lines, s, e, "call zisocl(", "zisocl", ZISOCL)
    return "".join(lines)


def spy_driver(text: str) -> str:
    lines = text.splitlines(keepends=True)
    use = locate(lines, "use shr_kind_mod,       only : r8 => shr_kind_r8")
    lines.insert(use + 1, "  use uwspy\n")
    s = locate(lines, "subroutine camuwpbl(")
    e = locate(lines, "end subroutine camuwpbl", s)
    lc = locate(lines, "lchnk   = (j - jts) * itile_len + (i - itsp1)", s)
    lines.insert(lc + 1, "          call uwspy_col(i)\n")
    lines.insert(lc + 2, "          call uwspy_step(itimestep)\n")
    lines.insert(lc + 3, "          call uwspy_iter(0)\n")
    e += 3
    e += instrument(lines, s, e, "call compute_eddy_diff(", "eddy", EDDY)
    first, _ = find_call(lines, s, "if( any(fieldlist_wet) ) then", e)
    ind = indent_of(lines[first])
    lines.insert(first, f"{ind}call uwspy_iter(-1)\n")
    e += 1
    # the outer compute_vdiff is the first compute_vdiff after
    # 'if( any(fieldlist_wet) ) then'; the dry-list call below it is never
    # reached (fieldlist_dry selects nothing, camuwpblinit lines 1048-1053)
    stop = locate(lines, "if( errstring .ne. '' ) then", first)
    instrument(lines, first, stop, "call compute_vdiff(", "vdiff_outer",
               VDIFF_OUTER)
    return "".join(lines)


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    root, out = Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    eddy = (root / "phys/module_cam_bl_eddy_diff.F").read_text(encoding="latin-1")
    drv = (root / "phys/module_bl_camuwpbl_driver.F").read_text(encoding="latin-1")
    (out / "module_cam_bl_eddy_diff.F").write_text(spy_eddy(eddy),
                                                   encoding="latin-1")
    (out / "module_bl_camuwpbl_driver.F").write_text(spy_driver(drv),
                                                     encoding="latin-1")


if __name__ == "__main__":
    main(sys.argv)
