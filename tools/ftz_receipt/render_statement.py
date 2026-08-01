"""Render the public FTZ statement FROM the receipt, and nothing else.

Two delimited blocks are generated -- one in ``PROVENANCE.md``, one in
``docs/public/HARDWARE.md`` -- and both are written by substitution into
``statement_template.md``.  The template holds the framing; the receipt
holds every fact.  Three properties are load-bearing and each has a test:

*Nothing is authored.*  The template contains no member of the receipt's
verdict vocabulary, so the outcome words in the published text can only
have come from ``receipt.json``'s ``cells``.

*Nothing is quantified over loaders.*  The public sentence this replaces
read "No loader or kernel supplies an FTZ option" -- a universal that the
measurement falsifies and that no amount of re-measuring could have
repaired, because it was never scoped to a route.  ``render_routes`` and
``render_tuples`` emit an ENUMERATION instead: every route, its site, the
options it actually supplies, and the outcome measured for it.
``universal_quantifier_problems`` refuses to let a universal back in.

*Nothing rests on a mirror.*  ``receipt/ptx/*.ptx`` are recompiles to a
readable target, labelled ``mirror`` for that reason.  A claim whose only
support is a mirror is a claim about a compile nobody shipped, so
``mirror_only_problems`` fails any rendered line whose entire citation set
is mirrors.  The shipped-path tiers -- the device bit table and the
disassembly of the objects the routes themselves compiled -- carry the
statement.

Usage::

    python -m tools.ftz_receipt.render_statement            # write
    python -m tools.ftz_receipt.render_statement --check    # verify
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import textwrap
from pathlib import Path

TEMPLATE_NAME = "statement_template.md"

#: One generated region per (file, block).  The delimiters are HTML
#: comments so the rendered Markdown reads normally.
BEGIN = "<!-- BEGIN GENERATED ftz-statement: {name} " \
        "(tools/ftz_receipt/render_statement.py) -->"
END = "<!-- END GENERATED ftz-statement: {name} -->"

TARGETS = (
    ("provenance-compile-policy", "PROVENANCE.md"),
    ("hardware-fp32-subnormals", "docs/public/HARDWARE.md"),
)

#: Universal quantification over the compile surface, in the shapes the
#: retired sentence and its neighbours used.  The nouns are the ones a
#: reader would take as covering the whole codebase.
_QUANTIFIERS = r"(?:no|any|all|every|each and every|none of the|neither)"
_SURFACE = r"(?:loaders?|kernels?|routes?|call sites?|compile sites?)"
UNIVERSAL_PATTERNS = (
    re.compile(rf"\b{_QUANTIFIERS}\s+(?:\w+\s+){{0,2}}?{_SURFACE}\b",
               re.IGNORECASE),
    re.compile(rf"\b{_SURFACE}\s+(?:all|never|always)\b", re.IGNORECASE),
    re.compile(r"\bnever\s+(?:\w+\s+){0,2}?"
               rf"{_SURFACE}\b", re.IGNORECASE),
)

#: A receipt-relative artifact path as it appears in the rendered text.
CITATION = re.compile(r"tools/ftz_receipt/receipt/([A-Za-z0-9_./-]+)")

#: Published width.  Wrapping is deterministic, so a wrapped block still
#: regenerates byte-identically.
WIDTH = 76


def _wrap(text: str, *, indent: str = "  ", hanging: str | None = None) -> str:
    return textwrap.fill(
        text, width=WIDTH, initial_indent=indent,
        subsequent_indent=indent if hanging is None else hanging,
        break_long_words=False, break_on_hyphens=False)


# ---- inputs ---------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load(root: Path) -> tuple[dict, dict, str]:
    receipt_dir = root / "tools" / "ftz_receipt" / "receipt"
    receipt = json.loads((receipt_dir / "receipt.json").read_text(
        encoding="utf-8"))
    inventory = json.loads((receipt_dir / "route_inventory.json").read_text(
        encoding="utf-8"))
    template = (root / "tools" / "ftz_receipt" / TEMPLATE_NAME).read_text(
        encoding="utf-8")
    return receipt, inventory, template


# ---- derivations ----------------------------------------------------------

def verdict_summary(receipt: dict, route_id: str) -> str:
    """"<verdict> on k of n mechanisms", per verdict, most cells first.

    The verdict words come from ``cells``; the counting is arithmetic.
    Nothing here knows what any particular verdict is called.
    """
    total = len(receipt["mechanisms"])
    counts: dict[str, int] = {}
    for cell in receipt["cells"]:
        if cell["route"] == route_id:
            counts[cell["verdict"]] = counts.get(cell["verdict"], 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(
        f"`{verdict}` on {count} of {total} mechanisms"
        for verdict, count in ordered)


def verdict_set(receipt: dict, route_id: str) -> set[str]:
    return {cell["verdict"] for cell in receipt["cells"]
            if cell["route"] == route_id}


def shares_module_with(receipt: dict, route_id: str) -> str | None:
    """The route whose compile this one rides, if it does not have its own."""
    stem = route_id.lower().replace("-", "_")
    record = receipt["artifacts"].get(f"cubin/{stem}", {})
    return record.get("shares_module_with")


def effective_flags(receipt: dict, route_id: str) -> tuple[list[str], str]:
    """(flags NVRTC received, the route whose compile they came from)."""
    per_route = receipt["effective_option_capture"]["per_route"]
    records = per_route.get(route_id)
    if records:
        return list(records[0]["effective_flags"]), route_id
    other = shares_module_with(receipt, route_id)
    if other and per_route.get(other):
        return list(per_route[other][0]["effective_flags"]), other
    return [], route_id


def _route_stems(route_id: str) -> re.Pattern:
    stem = re.escape(route_id.lower().replace("-", "_"))
    return re.compile(rf"{stem}(?:_\d+)?$")


def route_sass_artifacts(receipt: dict, route_id: str) -> list[str]:
    """The disassembly artifacts derived from this route's own objects.

    ``R1`` must not collect ``R1-ftztrue``'s object: the control arm is a
    different compile and citing it under R1 would attribute the control's
    evidence to the route it exists to control.
    """
    objects = receipt["artifacts"].get("sass", {}).get("objects", {})
    pattern = _route_stems(route_id)
    return [record["artifact"] for name, record in objects.items()
            if record.get("available") and pattern.fullmatch(name)]


def route_evidence(receipt: dict, route_id: str) -> str:
    paths = route_sass_artifacts(receipt, route_id)
    borrowed = ""
    if not paths:
        other = shares_module_with(receipt, route_id)
        if other:
            paths = route_sass_artifacts(receipt, other)
            borrowed = f", the object `{other}` compiled"
    if not paths:
        return "device bit table only"
    if len(paths) == 1:
        return (f"disassembly `tools/ftz_receipt/receipt/{paths[0]}`"
                f"{borrowed}")
    return (f"disassembly of {len(paths)} objects, "
            f"`tools/ftz_receipt/receipt/{paths[0]}` and siblings{borrowed}")


def render_routes(receipt: dict, *, indent: str = "  ") -> str:
    lines = []
    for route_id, route in receipt["routes"].items():
        flags, source = effective_flags(receipt, route_id)
        if flags:
            shown = " ".join(f"`{flag}`" for flag in flags)
            if source != route_id:
                shown += f" (this route rides `{source}`'s compile)"
        else:
            shown = "no caller options at all"
        lines.append(_wrap(
            f"- `{route_id}` {route['label']} "
            f"(`{route['site']}`, {route['constructor']}), "
            f"effective NVRTC options {shown}: "
            f"{verdict_summary(receipt, route_id)} "
            f"[{route_evidence(receipt, route_id)}]",
            indent=indent, hanging=indent + "  "))
    return "\n".join(lines)


def package_tuples(inventory: dict) -> list[tuple[tuple, list[dict]]]:
    """Distinct literal option tuples over the shipped package surface."""
    groups: dict[tuple, list[dict]] = {}
    for record in inventory["sites"]:
        if not record["file"].startswith("gpuwm/"):
            continue
        options = record.get("options")
        key = tuple(options) if options else ()
        groups.setdefault(key, []).append(record)
    return sorted(groups.items(),
                  key=lambda item: (-len(item[1]), str(item[0])))


def render_tuples(inventory: dict, *, indent: str = "  ") -> str:
    lines = []
    for options, records in package_tuples(inventory):
        if options:
            shown = " ".join(
                f"`{item}`" if item is not None
                else "a target flag built from the device architecture "
                     "at run time"
                for item in options)
        else:
            shown = "no caller options"
        first = records[0]
        more = (f", and {len(records) - 1} other site(s)"
                if len(records) > 1 else "")
        lines.append(_wrap(
            f"- {shown} -- `{first['file']}:{first['line']}` "
            f"({first['constructor']}){more}",
            indent=indent, hanging=indent + "  "))
    return "\n".join(lines)


def tuple_lead_in(inventory: dict) -> str:
    groups = package_tuples(inventory)
    sites = sum(len(records) for _key, records in groups)
    return (f"The inventory records {len(groups)} distinct caller-supplied "
            f"option tuples across the {sites} compile sites in the shipped "
            f"package, each listed here with a site that supplies it:")


def shared_object_note(receipt: dict) -> str:
    """Fires only when two routes share one object and measured differently.

    That configuration is the whole of the "unconditional in hardware"
    question: same device, same object, same flags, two outcomes.  When the
    receipt does not show it, this returns the empty string and the
    published text simply does not contain the sentence.
    """
    for route_id, record in receipt["artifacts"].items():
        if not route_id.startswith("cubin/"):
            continue
        other = record.get("shares_module_with")
        if not other:
            continue
        rider = route_id.split("/", 1)[1].upper().replace("_", "-")
        if rider not in receipt["routes"] or other not in receipt["routes"]:
            continue
        mine, theirs = verdict_set(receipt, rider), verdict_set(receipt, other)
        if mine == theirs:
            continue
        return (f"`{rider}` and `{other}` are kernels inside ONE compiled "
                f"object -- same device, same flags, one compile -- and they "
                f"did not measure alike, so on this device the outcome "
                f"follows the instruction the compiler emitted rather than "
                f"the hardware alone.")
    return ""


def control_note(receipt: dict) -> str:
    sensitivity = receipt["flag_sensitivity"]
    if not sensitivity.get("arms_differ"):
        return ("The arms did not separate, so nothing above distinguishes "
                "a compile-side effect from a device-side one.")
    return (f"The control arm is load-bearing: the "
            f"{sensitivity['distinct_arm_bit_tables']} distinct bit tables "
            f"among the {len(sensitivity['arm_bit_table_sha256'])} arms are "
            f"what shows the pipeline responds to the flag at all.")


def evidence_line(receipt: dict) -> str:
    parts = [
        "the device bit table "
        "`tools/ftz_receipt/receipt/bitpatterns.csv` "
        f"(both passes byte-identical: "
        f"{str(receipt['dual_run']['byte_identical']).lower()})",
        "the objects the routes themselves compiled, "
        "`tools/ftz_receipt/receipt/cubin/`",
    ]
    sass = receipt["artifacts"].get("sass", {})
    if sass.get("produced"):
        parts.append(
            f"and their disassembly, "
            f"`tools/ftz_receipt/receipt/sass/` "
            f"({sass['disassembler']})")
    else:
        parts.append(
            f"and no disassembly tier "
            f"({sass.get('reason', 'not collected')})")
    parts.append("all recorded in `tools/ftz_receipt/receipt/receipt.json`")
    return ", ".join(parts)


# ---- rendering ------------------------------------------------------------

def driver_version(receipt: dict) -> str:
    """CUDA's packed driver integer, shown as itself and as its parts."""
    packed = receipt["device"]["cuda_driver_version"]
    return f"{packed // 1000}.{(packed % 1000) // 10} ({packed})"


def substitutions(receipt: dict, inventory: dict) -> dict:
    device = receipt["device"]
    append = receipt["cupy_ftz_append_site"]
    inline = {
        "tuple_lead_in": tuple_lead_in(inventory),
        "shared_object_note": shared_object_note(receipt),
        "control_note": control_note(receipt),
        "evidence": evidence_line(receipt),
    }
    return {
        "device_name": device["name"],
        "compute_capability": device["compute_capability"],
        "driver_version": driver_version(receipt),
        "nvrtc_version": device["nvrtc_version"],
        "cupy_version": device["cupy_version"],
        "mechanism_count": str(len(receipt["mechanisms"])),
        "route_count": str(len(receipt["routes"])),
        "route_bullets": render_routes(receipt),
        "tuple_bullets": render_tuples(inventory),
        "append_flag": append.get("appended", "no unconditional append"),
        "append_module": append.get("module", "unknown"),
        "append_line": str(append.get("line", "unknown")),
        "append_text": append.get("text", ""),
        **{key: _wrap(value, indent="", hanging="  ") if value else ""
           for key, value in inline.items()},
    }


def template_blocks(template: str) -> dict[str, str]:
    out = {}
    pattern = re.compile(
        r"\[\[block:(?P<name>[a-z0-9-]+)\]\]\n(?P<body>.*?)"
        r"\[\[end:(?P=name)\]\]",
        re.DOTALL)
    for match in pattern.finditer(template):
        out[match.group("name")] = match.group("body")
    return out


def substitute(body: str, values: dict) -> str:
    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"template asks for {key!r}, receipt has no such "
                           f"substitution")
        return values[key]
    text = re.sub(r"\{\{([a-z_]+)\}\}", replace, body)
    # An empty derivation leaves a blank continuation line behind; drop it
    # so the block never publishes an orphaned indent.
    return "\n".join(line for line in text.splitlines()
                     if line.strip() != "") + "\n"


def render_block(name: str, receipt: dict, inventory: dict,
                 template: str) -> str:
    blocks = template_blocks(template)
    if name not in blocks:
        raise KeyError(f"template has no block {name!r}")
    body = substitute(blocks[name], substitutions(receipt, inventory))
    return (BEGIN.format(name=name) + "\n" + body
            + END.format(name=name) + "\n")


def replace_region(text: str, name: str, block: str) -> str:
    begin, end = BEGIN.format(name=name), END.format(name=name)
    start = text.find(begin)
    if start == -1:
        raise SystemExit(
            f"{name}: no generated region in the target file; the region "
            f"delimiters are placed by hand once and regenerated therafter")
    stop = text.find(end, start)
    if stop == -1:
        raise SystemExit(f"{name}: generated region is not closed")
    return text[:start] + block + text[stop + len(end) + 1:]


# ---- the assertions the criteria name -------------------------------------

def universal_quantifier_problems(text: str) -> list[str]:
    """Every universal-over-the-compile-surface phrase in a rendered block."""
    problems = []
    for pattern in UNIVERSAL_PATTERNS:
        for match in pattern.finditer(text):
            problems.append(
                f"universal quantifier over the compile surface: "
                f"{match.group(0)!r}")
    return problems


def _lines_with_citations(text: str) -> list[tuple[str, list[str]]]:
    out = []
    for line in text.splitlines():
        found = CITATION.findall(line)
        if found:
            out.append((line, found))
    return out


def mirror_only_problems(text: str, receipt: dict) -> list[str]:
    """Lines whose entire citation set is ``mirror`` artifacts."""
    problems = []
    for line, citations in _lines_with_citations(text):
        provenances = []
        for citation in citations:
            record = receipt["artifacts"].get(citation.rstrip("/"))
            if record is None:
                continue
            provenances.append(record.get("provenance"))
        if provenances and set(provenances) == {"mirror"}:
            problems.append(
                f"line cites only mirror artifacts: {line.strip()[:120]!r}")
    return problems


def release_exclusions(root: Path) -> list[str]:
    path = root / "RELEASE-EXCLUDE.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _excluded(relpath: str, globs: list[str]) -> bool:
    for pattern in globs:
        if fnmatch.fnmatch(relpath, pattern):
            return True
        if pattern.endswith("/**") and relpath.startswith(pattern[:-2]):
            return True
    return False


def citation_problems(text: str, root: Path) -> list[str]:
    """Cited paths must exist AND survive the release exclusion list."""
    globs = release_exclusions(root)
    problems = []
    for line, citations in _lines_with_citations(text):
        for citation in citations:
            relpath = f"tools/ftz_receipt/receipt/{citation.rstrip('/')}"
            if not (root / relpath).exists():
                problems.append(f"citation does not resolve: {relpath}")
            elif _excluded(relpath, globs):
                problems.append(
                    f"citation is release-excluded: {relpath} "
                    f"(cited on {line.strip()[:60]!r})")
    return problems


def block_problems(text: str, receipt: dict, root: Path) -> list[str]:
    return (universal_quantifier_problems(text)
            + mirror_only_problems(text, receipt)
            + citation_problems(text, root))


# ---- driver ---------------------------------------------------------------

def render_all(root: Path) -> dict[str, str]:
    receipt, inventory, template = load(root)
    return {name: render_block(name, receipt, inventory, template)
            for name, _path in TARGETS}


def write_all(root: Path) -> list[str]:
    blocks = render_all(root)
    changed = []
    for name, relpath in TARGETS:
        path = root / relpath
        before = path.read_text(encoding="utf-8")
        after = replace_region(before, name, blocks[name])
        if after != before:
            path.write_text(after, encoding="utf-8", newline="\n")
            changed.append(relpath)
    return changed


def check_all(root: Path) -> list[str]:
    receipt, _inventory, _template = load(root)
    blocks = render_all(root)
    problems = []
    for name, relpath in TARGETS:
        text = (root / relpath).read_text(encoding="utf-8")
        begin, end = BEGIN.format(name=name), END.format(name=name)
        start, stop = text.find(begin), text.find(end)
        if start == -1 or stop == -1:
            problems.append(f"{relpath}: generated region {name} is missing")
            continue
        present = text[start:stop + len(end)] + "\n"
        if present != blocks[name]:
            problems.append(
                f"{relpath}: generated region {name} does not match the "
                f"receipt; re-run tools/ftz_receipt/render_statement.py")
        problems.extend(f"{relpath}: {problem}"
                        for problem in block_problems(present, receipt, root))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else repo_root()
    if args.check:
        problems = check_all(root)
        for problem in problems:
            print(problem)
        if problems:
            return 1
        print("rendered blocks reproduce from the receipt")
        return 0
    changed = write_all(root)
    for relpath in changed:
        print(f"regenerated {relpath}")
    if not changed:
        print("rendered blocks already match the receipt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
