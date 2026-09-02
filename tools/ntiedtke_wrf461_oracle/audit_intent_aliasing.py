"""Find dummies that are READ before they are WRITTEN in cu_ntiedtke.F90.

WHY THIS EXISTS.  `cutypen` declares `cutu`/`cuqu`/`culu`/`culab`
``intent(out)`` and then reads all four in its very first loop
(:1334-1337).  `cumastrn` passes cuinin's live `ptu`/`pqu`/`plu`/`ilab`
into those slots, so the incoming values are load-bearing -- and a harness
that hands the routine FRESH arrays instead produces a capture that is
wrong on exactly the columns where the routine's own writeback leaves the
incoming values in place.

That bug shipped into the cutypen fixture and was caught only because the
NumPy mirror disagreed with it.  Nothing inside the harness could have
found it: the harness agreed with itself, and every self-consistency check
it ran passed.

So the pattern has to be found by reading, once, for every routine, before
the next harness is written.  Reading an ``intent(out)`` dummy before
assigning it is undefined behaviour in Fortran and the compiler is under no
obligation to warn; the code does it anyway, and it works in situ only
because of what the caller happens to pass.

USAGE
    python audit_intent_aliasing.py <path to cu_ntiedtke.F90>

Reports, per routine, every ``intent(out)`` / ``intent(inout)`` dummy whose
FIRST appearance in the body is a read rather than a write, and classifies
that first use.  A dummy first appearing as an argument to a CALL is
reported separately -- the callee may write it, so those need a human.

WHAT A FIRST-USE SCAN STRUCTURALLY CANNOT FIND
----------------------------------------------
A dummy written ONLY inside a conditional is load-bearing in exactly the
same way -- a column that misses the branch keeps the caller's value -- but
its first textual use is a WRITE, so the scan above calls it clean.

That is not a bug to be fixed by a better regex; it is what "first use"
means.  ``cubasmcn`` is the case that proved it: all thirteen of its
outputs are conditionally written, the scan passed it, and it was found by
reading the routine by hand.

The third report below covers that class by looking at ALL writes to each
``intent(out)`` dummy and asking whether any of them is unconditional.  It
is a heuristic on Fortran block structure, not a parse, so treat a routine
it clears as unverified rather than safe.  Anyone extending this script
should know both halves are needed and neither subsumes the other.
"""
from __future__ import annotations

import re
import sys
from collections import OrderedDict

SUB = re.compile(r"^\s*subroutine\s+([A-Za-z_]\w*)", re.I)
END = re.compile(r"^\s*end\s+subroutine", re.I)
DECL = re.compile(r"^\s*(?:real|integer|logical)[^:]*?"
                  r"intent\s*\(\s*(out|inout)\s*\)[^:]*::\s*(.*)$", re.I)
CALL = re.compile(r"^\s*call\s+([A-Za-z_]\w*)", re.I)

#: Any type declaration, WITH OR WITHOUT an intent attribute.  A dummy
#: declared plainly -- `real(kind_phys),dimension(klon,klev):: pdpmel,plglac`
#: in cuflxn (:2838) -- has no intent, so DECL above never matches it and it
#: is invisible to ALL THREE reports.  cuflxn's plglac is exactly that, and
#: it is the dummy whose incoming value the mirror got wrong.
ANYINTENT = re.compile(r"^\s*(?:real|integer|logical)[^:]*?"
                      r"intent\s*\([^)]*\)[^:]*::\s*(.*)$", re.I)

ANYDECL = re.compile(r"^\s*(?:real|integer|logical)[^:]*::\s*(.*)$", re.I)


def routines(lines):
    out, cur, start = OrderedDict(), None, 0
    for n, line in enumerate(lines):
        m = SUB.match(line)
        if m and not line.lstrip().lower().startswith("end"):
            cur, start = m.group(1).lower(), n
            continue
        if END.match(line) and cur:
            out[cur] = (start, n)
            cur = None
    return out


def arg_names(lines, a, b):
    """The dummy names in a subroutine's own argument list."""
    text = ""
    for n in range(a, b):
        t = strip_comment(lines[n])
        text += t.rstrip().rstrip("&")
        if ")" in t and "(" in text:
            break
    if "(" not in text:
        return []
    inner = text[text.index("(") + 1:]
    inner = inner[:inner.rindex(")")] if ")" in inner else inner
    # Continuation markers sit inside the list, so strip them per TOKEN --
    # stripping only the line's trailing & leaves names like "ztmst      &".
    out = []
    for x in inner.split(","):
        x = x.replace("&", " ").strip().lower()
        if x:
            out.append(x)
    return out


def declared_names(lines, a, lo, with_intent):
    """Names declared in [a, lo), optionally only intent-bearing ones."""
    out = set()
    for n in range(a, lo):
        m = (ANYINTENT if with_intent else ANYDECL).match(lines[n])
        if not m:
            continue
        payload = m.group(1)
        for d in payload.split(","):
            d = d.split("(")[0].strip().lower()
            if d:
                out.add(d)
    return out


def body_start(lines, a, b):
    """First executable line: after the last declaration or `implicit`."""
    last = a
    for n in range(a, b):
        s = lines[n].strip().lower()
        if (s.startswith(("real", "integer", "logical", "character",
                          "implicit", "use ", "import"))
                or "intent(" in s or "::" in s):
            last = n
    return last + 1


def strip_comment(line):
    return line.split("!", 1)[0] if not line.lstrip().startswith("!") else ""


def classify_first_use(lines, lo, hi, name):
    """('write'|'read'|'call'|None, lineno, text) for the first use."""
    # name, optionally indexed, on the left of a top-level '='
    w = re.compile(rf"^\s*&?\s*{re.escape(name)}\s*(\([^=]*\))?\s*=[^=]", re.I)
    use = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                     re.I)
    for n in range(lo, hi):
        text = strip_comment(lines[n])
        if not text.strip():
            continue
        if not use.search(text):
            continue
        if w.match(text):
            # A THIRD LOAD-BEARING CLASS, and the audit missed it until a
            # cuflxn parity test failed on plglac.  `x = f(x)` reads the
            # incoming value on the RHS, but "first use" sees a write, so
            # class 1 skips it; and class 2 filters to intent(out), so an
            # intent(inout) self-assignment falls through both.
            #
            #   cuflxn:2887   plglac(jl,jk) = pmfu(jl,jk)*plglac(jl,jk)
            #
            # The RHS can continue across lines, so the continuation is
            # walked rather than assumed to be one line.
            rhs = text.split("=", 1)[1]
            j = n
            while strip_comment(lines[j]).rstrip().endswith("&")                     and j + 1 < hi:
                j += 1
                rhs += strip_comment(lines[j])
            if use.search(rhs):
                return "selfwrite", n + 1, text.strip()
            return "write", n + 1, text.strip()
        if CALL.match(text):
            return "call", n + 1, text.strip()
        # a continued call argument list: walk back for `call`
        j = n
        while j > lo and strip_comment(lines[j - 1]).rstrip().endswith("&"):
            j -= 1
            if CALL.match(strip_comment(lines[j])):
                return "call", n + 1, text.strip()
        return "read", n + 1, text.strip()
    return None, None, None


# ===========================================================================
# The vacuity guard -- asked for by review, and this file earned it first
# ===========================================================================
# Every report below answers "no violations" and "no coverage" WITH THE SAME
# OUTPUT: a line reading "(none)".  This script has already produced the
# second while being read as the first -- the DECL regex never matched a
# dummy declared without an intent attribute, so an entire class was
# reported as empty rather than as unscanned.
#
# So the patterns are checked against the corpus before any report is
# printed.  The numbers are MEASURED on the pinned v4.6.1 file (15 routines,
# 62 DECL lines, 158 ANYINTENT, 282 ANYDECL) and the floors sit well below
# them: the guard is meant to catch a pattern that has stopped matching, not
# to pin the file.  Named routines matter more than the counts, because a
# regex can keep matching a shrinking corpus.
#
# It exits non-zero rather than printing into the report, so the receipt
# stays byte-comparable against the shipped nt-aliasing-audit.txt.
_MUST_SEE = ("cumastrn", "cuascn", "cuinin", "cutypen", "cubasmcn",
             "cuentrn", "cudlfsn", "cuddrafn", "cuflxn", "cudtdqn",
             "cududvn", "cuadjtqn")


def guard(lines, found):
    """Refuse to report on a corpus the patterns are not seeing."""
    problems = []
    missing = [r for r in _MUST_SEE if r not in found]
    if missing:
        problems.append(f"SUB matched no such routine: {missing}")
    if len(found) < 12:
        problems.append(f"SUB found only {len(found)} routines")

    counts = {"DECL": 0, "ANYINTENT": 0, "ANYDECL": 0}
    for name, (a, b) in found.items():
        lo = body_start(lines, a, b)
        for n in range(a, lo):
            if DECL.match(lines[n]):
                counts["DECL"] += 1
            if ANYINTENT.match(lines[n]):
                counts["ANYINTENT"] += 1
            if ANYDECL.match(lines[n]):
                counts["ANYDECL"] += 1
    for key, floor in (("DECL", 40), ("ANYINTENT", 100), ("ANYDECL", 200)):
        if counts[key] < floor:
            problems.append(
                f"{key} matched {counts[key]} declaration lines, under the "
                f"floor of {floor} -- the pattern has stopped seeing its "
                f"corpus, so every report below would read '(none)'")
    if problems:
        import sys
        print("FATAL: the aliasing audit's patterns are not matching.",
              file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        sys.exit(2)


def main(path):
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    guard(lines, routines(lines))
    flagged, callish, selfish, clean = [], [], [], 0
    for name, (a, b) in routines(lines).items():
        lo = body_start(lines, a, b)
        dummies = []
        for n in range(a, lo):
            m = DECL.match(lines[n])
            if m:
                for d in m.group(2).split(","):
                    d = d.split("(")[0].strip()
                    if d:
                        dummies.append((d, m.group(1).lower()))
        for d, intent in dummies:
            kind, ln, text = classify_first_use(lines, lo, b, d)
            if kind == "read":
                flagged.append((name, d, intent, ln, text))
            elif kind == "call":
                callish.append((name, d, intent, ln, text))
            elif kind == "selfwrite":
                selfish.append((name, d, intent, ln, text))
            else:
                clean += 1

    print("=" * 74)
    print("READ BEFORE WRITTEN -- the caller's incoming value is LOAD-BEARING")
    print("=" * 74)
    if not flagged:
        print("  (none)")
    for name, d, intent, ln, text in flagged:
        print(f"  {name:<10} {d:<10} intent({intent:<5}) :{ln:<5} {text[:52]}")

    print()
    print("=" * 74)
    print("FIRST USE IS A CALL ARGUMENT -- needs a human; the callee may write")
    print("=" * 74)
    if not callish:
        print("  (none)")
    for name, d, intent, ln, text in callish:
        print(f"  {name:<10} {d:<10} intent({intent:<5}) :{ln:<5} {text[:52]}")

    print()
    print("=" * 74)
    print("SELF-REFERENTIAL FIRST WRITE (x = f(x)) -- also load-bearing")
    print("=" * 74)
    if not selfish:
        print("  (none)")
    for name, d, intent, ln, text in selfish:
        print(f"  {name:<10} {d:<10} intent({intent:<5}) :{ln:<5} {text[:52]}")

    # A FOURTH GAP, and the one that actually broke cuflxn's plglac: a
    # dummy declared WITHOUT an intent attribute is matched by no DECL, so
    # it never enters any of the three reports above.  It is still a dummy
    # and its incoming value is still whatever the caller had.
    print()
    print("=" * 74)
    print("DUMMY DECLARED WITHOUT intent -- invisible to every check above")
    print("=" * 74)
    silent = []
    for name, (a, b) in routines(lines).items():
        lo = body_start(lines, a, b)
        args = set(arg_names(lines, a, lo))
        if not args:
            continue
        intented = declared_names(lines, a, lo, True)
        plain = declared_names(lines, a, lo, False)
        for d in sorted(args & plain - intented):
            silent.append((name, d))
    if not silent:
        print("  (none)")
    for name, d in silent:
        print(f"  {name:<10} {d:<10} no intent attribute; treat as inout")

    # A SECOND CLASS the first-use test misses.  cubasmcn declares `ktype`
    # intent(out) and writes it ONLY at :3480, inside
    #   if(.not.ldcum(jl) .and. klab(jl,kk+1).eq.0)
    # so a column that does not take that branch keeps the CALLER's value.
    # Its first textual use is a write, so the scan above calls it safe --
    # and it is not.  Any intent(out) dummy whose writes are all conditional
    # is load-bearing in exactly the same way.
    print()
    print("=" * 74)
    print("intent(out) WITH ONLY CONDITIONAL WRITES -- also load-bearing")
    print("=" * 74)
    cond = []
    for name, (a, b) in routines(lines).items():
        lo = body_start(lines, a, b)
        outs = []
        for n in range(a, lo):
            m = DECL.match(lines[n])
            if m and m.group(1).lower() == "out":
                for d in m.group(2).split(","):
                    d = d.split("(")[0].strip()
                    if d:
                        outs.append(d)
        for d in outs:
            w = re.compile(rf"^\s*&?\s*{re.escape(d)}\s*(\([^=]*\))?\s*=[^=]",
                           re.I)
            writes, uncond = [], False
            depth = 0
            for n in range(lo, b):
                t = strip_comment(lines[n])
                ls = t.strip().lower()
                if re.match(r"^(if\s*\(.*\)\s*then|else|elseif|else if)", ls):
                    depth += 1 if not ls.startswith(("else",)) else 0
                if ls.startswith(("end if", "endif")):
                    depth = max(0, depth - 1)
                if w.match(t):
                    inline_if = bool(re.match(r"^\s*if\s*\(", ls))
                    writes.append(n + 1)
                    if depth == 0 and not inline_if:
                        uncond = True
            if writes and not uncond:
                cond.append((name, d, writes[:4]))
    if not cond:
        print("  (none)")
    for name, d, w in cond:
        print(f"  {name:<10} {d:<10} writes only at lines {w}")

    print()
    print(f"written before read (safe to pass fresh arrays): {clean}")
    print(f"READ FIRST (harness MUST pass the caller's array): {len(flagged)}")
    print(f"call-argument first (inspect by hand):            {len(callish)}")


if __name__ == "__main__":
    main(sys.argv[1])
