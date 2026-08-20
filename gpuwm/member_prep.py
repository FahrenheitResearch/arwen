"""Ensemble member preparation: fetch by filename, verify by bytes.

The generic half of the ensemble member capability.  The declarative
half -- which members exist, how their files are named, what their bytes
must say -- is the packaged ``rw-wps.members.v1`` document read by
:mod:`gpuwm.member_grammar`.  This module does the work those documents
authorize and nothing else:

1. **Verification.**  Every message of a file claimed as member X is
   checked against X's declared ensemble identity: the product
   definition template, ``typeOfEnsembleForecast``, and
   ``perturbationNumber`` (the triple), plus the declared encoded
   ensemble size and generating-process pins where the grammar states
   them.  A mismatch refuses, naming what the filename claims AND what
   the bytes carry.  This fails closed because every failure mode here
   is silent: an ensemble mean decodes as a plausible member, a
   different member's trajectory decodes as the requested one, and no
   later stage can tell.

2. **Preparation.**  A verified member is staged into a member-addressed
   prepared tree -- ``<set>/<cycle>/<member>/<product>/<file>`` -- whose
   receipt pins the grammar document, the member identity, and the
   SHA-256 of every staged byte.  The tree exists because upstream leaf
   names cannot be trusted to carry identity: AIGEFS publishes
   byte-identical leaf filenames for all 31 members, so any flat store
   destroys the member axis.

The GRIB2 octets are read by the Rust ``grib2_inventory`` bridge; no
GRIB decoding happens in Python.  The inventory's ensemble-identity
columns (``ensemble_type``, ``ensemble_size``, ``derived_forecast``)
are REQUIRED: a stale bridge that predates them cannot tell a mean from
a member, so it refuses with a rebuild remedy rather than degrading.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

from gpuwm.member_grammar import (MemberGrammar, MemberIdentity,
                                  MemberIdentityRefusal, load_member_grammar)
from gpuwm.source_authorities import (packaged_member_grammar,
                                      packaged_member_grammar_ids,
                                      packaged_member_grammar_sha256)

RECEIPT_SCHEMA = "gpuwm-ensemble-member-preparation-receipt-v1"
RECEIPT_NAME = "member-receipt.json"

#: GRIB2 product definition templates that carry an ensemble statistic
#: rather than an individual trajectory, by WMO family.  Generic
#: vocabulary, not any model's: these are the templates under which a
#: mean, spread, probability or percentile product decodes cleanly as
#: ordinary fields and is indistinguishable from a member by anything
#: except this octet.
STATISTIC_TEMPLATE_FAMILIES: Mapping[int, str] = {
    2: "derived forecast on all ensemble members",
    12: "derived forecast on all ensemble members over a time interval",
    5: "probability forecast",
    9: "probability forecast over a time interval",
    6: "percentile forecast",
    10: "percentile forecast over a time interval",
}

#: WMO code table 4.7: what a derived-forecast product actually is.
DERIVED_FORECAST_NAMES: Mapping[int, str] = {
    0: "unweighted mean of all members",
    1: "weighted mean of all members",
    2: "standard deviation with respect to the cluster mean",
    3: "standard deviation with respect to the cluster mean, normalized",
    4: "spread of all members",
    5: "large anomaly index of all members",
    6: "unweighted mean of the cluster members",
    7: "interquartile range of all members",
    8: "minimum of all ensemble members",
    9: "maximum of all ensemble members",
}

#: Templates that carry no ensemble identity at all.
_DETERMINISTIC_TEMPLATES: Mapping[int, str] = {
    0: "deterministic forecast at a point in time",
    8: "deterministic forecast over a time interval",
    15: "deterministic forecast over a spatial area",
}

_ENSEMBLE_COLUMNS = ("ensemble_type", "ensemble_size", "derived_forecast")
_REQUIRED_COLUMNS = frozenset({
    "index", "pdt", "member", "generating_process",
    "forecast_generating_process_id", *_ENSEMBLE_COLUMNS,
})


@dataclass(frozen=True)
class MemberFileEvidence:
    """What the bytes of one verified member file actually said."""

    messages: int
    product_definition_templates: tuple[int, ...]
    type_of_ensemble_forecast: int
    perturbation_number: int
    ensemble_size: tuple[int, ...]
    type_of_generating_process: tuple[int, ...]
    forecast_generating_process_id: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "messages": self.messages,
            "product_definition_templates": list(
                self.product_definition_templates),
            "type_of_ensemble_forecast": self.type_of_ensemble_forecast,
            "perturbation_number": self.perturbation_number,
            "ensemble_size": list(self.ensemble_size),
            "type_of_generating_process": list(
                self.type_of_generating_process),
            "forecast_generating_process_id": list(
                self.forecast_generating_process_id),
        }


def _optional_int(row: Mapping[str, str], column: str) -> int | None:
    value = row.get(column, "-").strip()
    return None if value == "-" else int(value)


def member_inventory_rows(
    source: Path, executable: Path | None = None,
) -> list[dict[str, str]]:
    """Run the Rust inventory and require the ensemble-identity columns.

    The refusal for a stale binary names the concrete breakage -- the
    missing octets are the only thing separating an ensemble mean from a
    member -- because "rebuild your bridge" without the reason reads as
    a broken install rather than a contract the build enforces.
    """

    from gpuwm import bridges
    # The shared resolver/builder for the GRIB2 tools; the dump half of
    # the pair is unused here but resolving both keeps this route on the
    # exact ladder every mapped-source consumer already walks.
    from gpuwm.mapped_source import _build_grib2_tools, _grib2_inventory

    if executable is None:
        executable, _dump = _build_grib2_tools()
    rows = _grib2_inventory(Path(source), Path(executable))
    missing = sorted(_REQUIRED_COLUMNS - set(rows[0]))
    if missing:
        raise MemberIdentityRefusal(
            f"the grib2_inventory at {executable} does not report the "
            f"ensemble-identity columns {missing}.  Without them an "
            "ensemble mean is indistinguishable from a member and a "
            "control from a perturbed forecast, so member verification "
            "refuses rather than guesses.  The binary predates this "
            "release's inventory contract; rebuild it:\n"
            + bridges.bridge_remedy("grib2_inventory"))
    return rows


def _statistic_name(
    grammar: MemberGrammar, pdt: int, derived: int | None,
) -> str:
    """The best available name for a statistic product's bytes."""

    parts = [STATISTIC_TEMPLATE_FAMILIES[pdt]]
    if derived is not None:
        parts.append(DERIVED_FORECAST_NAMES.get(
            derived, f"derived-forecast code {derived}"))
    declared = [
        statistic for statistic in grammar.statistics()
        if statistic.product_definition_templates is not None
        and pdt in statistic.product_definition_templates
        and (statistic.derived_forecast is None
             or statistic.derived_forecast == derived)
    ]
    if len(declared) == 1:
        parts.append(
            f"the declared {grammar.name} statistic namespace names this "
            f"{declared[0].statistic_id!r} ({declared[0].statistic})")
    return "; ".join(parts)


def verify_member_rows(
    grammar: MemberGrammar,
    member: MemberIdentity,
    rows: Sequence[Mapping[str, str]],
    *,
    source_label: str,
) -> MemberFileEvidence:
    """Every message must carry the claimed member's declared identity.

    Pure over already-parsed inventory rows so the contract is testable
    without the bridge; :func:`verify_member_file` binds it to bytes.
    """

    declared = member.verification
    claim = (f"{source_label} is claimed as {grammar.name} member "
             f"{member.member_id} (class {member.class_name!r}, "
             f"perturbationNumber {member.ordinal})")
    pdts: set[int] = set()
    sizes: set[int] = set()
    generating: set[int] = set()
    processes: set[int] = set()
    for row in rows:
        index = row["index"]
        pdt = int(row["pdt"])
        if pdt not in declared.product_definition_templates:
            derived = _optional_int(row, "derived_forecast")
            if pdt in STATISTIC_TEMPLATE_FAMILIES:
                raise MemberIdentityRefusal(
                    f"{claim}, but field {index} is an ensemble STATISTIC: "
                    f"product definition template {pdt} -- "
                    f"{_statistic_name(grammar, pdt, derived)}.  A "
                    "statistic decodes cleanly as ordinary fields (the "
                    "spread file reads as a plausible temperature) and is "
                    "not a dynamically balanced trajectory; initializing "
                    "from it would silently smooth every field, so it is "
                    "refused as a member")
            if pdt in _DETERMINISTIC_TEMPLATES:
                raise MemberIdentityRefusal(
                    f"{claim}, but field {index} carries product "
                    f"definition template {pdt} "
                    f"({_DETERMINISTIC_TEMPLATES[pdt]}), which has no "
                    "ensemble identity octets at all: these bytes are a "
                    "deterministic product, not any ensemble's member")
            raise MemberIdentityRefusal(
                f"{claim}, but field {index} carries product definition "
                f"template {pdt}, outside the declared member templates "
                f"{list(declared.product_definition_templates)}; bytes "
                "whose template the grammar does not declare cannot be "
                "verified as this member, so they are refused")
        observed_type = _optional_int(row, "ensemble_type")
        if observed_type is None:
            raise MemberIdentityRefusal(
                f"{claim}, but field {index} (template {pdt}) carries no "
                "typeOfEnsembleForecast octet; an unverifiable message "
                "is refused, not assumed")
        if observed_type != declared.type_of_ensemble_forecast:
            raise MemberIdentityRefusal(
                f"{claim} with declared typeOfEnsembleForecast "
                f"{declared.type_of_ensemble_forecast}, but field {index} "
                f"carries typeOfEnsembleForecast {observed_type}.  Real "
                "ensembles use this octet incompatibly -- one measured "
                "source flags its control as a low-resolution control, "
                "another stamps its control exactly like a perturbed "
                "member -- so a value the grammar did not declare means "
                "these bytes are not the declared ensemble's member")
        observed_member = _optional_int(row, "member")
        if observed_member is None:
            raise MemberIdentityRefusal(
                f"{claim}, but field {index} (template {pdt}) carries no "
                "perturbationNumber; an unverifiable message is refused, "
                "not assumed")
        if observed_member != member.ordinal:
            actual = grammar.member_for_ordinal(observed_member)
            actually = (
                f" -- the bytes are {grammar.name} member "
                f"{actual.member_id}'s trajectory" if actual is not None
                else "")
            raise MemberIdentityRefusal(
                f"{claim}, but field {index} carries perturbationNumber "
                f"{observed_member}{actually}.  The filename addresses "
                "one member and the bytes belong to another; accepting "
                "the filename would silently initialize from a different "
                "trajectory, so the file is refused")
        observed_size = _optional_int(row, "ensemble_size")
        if declared.ensemble_size is not None:
            if observed_size != declared.ensemble_size:
                raise MemberIdentityRefusal(
                    f"{claim} with declared encoded ensemble size "
                    f"{declared.ensemble_size}, but field {index} encodes "
                    f"numberOfForecastsInEnsemble = {observed_size}.  The "
                    "encoded size is identity evidence, not a count "
                    "(sources include or exclude their control "
                    "incompatibly): a different value means these bytes "
                    "come from a different ensemble system whose members "
                    "this triple cannot otherwise distinguish")
        if observed_size is not None:
            sizes.add(observed_size)
        for column, declared_value, key_name in (
                ("generating_process",
                 declared.type_of_generating_process,
                 "typeOfGeneratingProcess"),
                ("forecast_generating_process_id",
                 declared.forecast_generating_process_id,
                 "generatingProcessIdentifier")):
            observed = int(row[column])
            if declared_value is not None and observed != declared_value:
                raise MemberIdentityRefusal(
                    f"{claim} with declared {key_name} {declared_value}, "
                    f"but field {index} carries {observed}: a different "
                    "generating process means a different producing "
                    "system, not this ensemble's member")
            if column == "generating_process":
                generating.add(observed)
            else:
                processes.add(observed)
        pdts.add(pdt)
    return MemberFileEvidence(
        messages=len(rows),
        product_definition_templates=tuple(sorted(pdts)),
        type_of_ensemble_forecast=declared.type_of_ensemble_forecast,
        perturbation_number=member.ordinal,
        ensemble_size=tuple(sorted(sizes)),
        type_of_generating_process=tuple(sorted(generating)),
        forecast_generating_process_id=tuple(sorted(processes)),
    )


def verify_member_file(
    grammar: MemberGrammar,
    member_id: str,
    source: Path,
    *,
    inventory_executable: Path | None = None,
) -> MemberFileEvidence:
    """Verify one real file's bytes against a member claim."""

    member = grammar.member(member_id)
    rows = member_inventory_rows(source, inventory_executable)
    return verify_member_rows(
        grammar, member, rows, source_label=str(source))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_grammar(
    grammar_id: str | None, grammar_path: Path | None,
) -> tuple[MemberGrammar, dict[str, object]]:
    """A packaged (pinned) grammar by id, or an explicit document."""

    if (grammar_id is None) == (grammar_path is None):
        raise MemberIdentityRefusal(
            "exactly one of a packaged member-set id or an explicit "
            "members-document path must be given; packaged sets: "
            f"{list(packaged_member_grammar_ids())}")
    if grammar_id is not None:
        path = packaged_member_grammar(grammar_id)
        identity: dict[str, object] = {
            "registry_id": grammar_id,
            "file": path.name,
            "sha256": packaged_member_grammar_sha256(grammar_id),
        }
    else:
        path = Path(grammar_path)
        identity = {
            "registry_id": None,
            "file": str(path),
            "sha256": _sha256_file(path),
        }
    return load_member_grammar(path), identity


def prepare_member(
    *,
    grammar_id: str | None = None,
    grammar_path: Path | None = None,
    member_id: str,
    cycle: datetime,
    steps: Sequence[int],
    inputs_root: Path,
    output_root: Path,
    products: Sequence[str] | None = None,
    inventory_executable: Path | None = None,
) -> Path:
    """Verify and stage one declared member into the prepared tree.

    ``inputs_root`` must hold the fetched files at their DECLARED
    upstream-relative paths -- the layout every fetch of these feeds
    naturally produces, and the only layout that preserves member
    identity for sources whose leaf filenames are member-ambiguous.
    The staged tree is assembled next to its final location and renamed
    into place, so a crash leaves no half-tree that looks prepared.
    """

    grammar, grammar_identity = _resolve_grammar(grammar_id, grammar_path)
    member = grammar.member(member_id)
    if not steps:
        raise MemberIdentityRefusal("at least one forecast step is required")
    chosen_products = tuple(products) if products else grammar.products()

    inputs_root = Path(inputs_root)
    output_root = Path(output_root)
    cycle_label = cycle.strftime("%Y%m%dT%H") + "Z"
    member_dir = output_root / grammar.name / cycle_label / member.member_id
    if member_dir.exists():
        raise MemberIdentityRefusal(
            f"{member_dir} already exists; this preparer never overwrites "
            "a prepared member -- remove it deliberately or choose "
            "another output root")

    plan: list[tuple[str, int, str, Path]] = []
    missing: list[str] = []
    for product in chosen_products:
        for step in sorted(set(int(step) for step in steps)):
            relative = grammar.relative_path(
                member.member_id, product, cycle, step)
            source = inputs_root / Path(relative)
            if source.is_file():
                plan.append((product, step, relative, source))
            else:
                missing.append(relative)
    if missing:
        raise MemberIdentityRefusal(
            f"{grammar.name} member {member.member_id} is missing "
            f"{len(missing)} declared input file(s) under {inputs_root}:\n"
            + "\n".join(f"  {relative}" for relative in missing)
            + "\n  # fetch each from a declared front door by appending "
            "the relative path to its base URL; verification happens "
            "here, on the bytes, after the fetch")

    partial = member_dir.parent / (member_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    files: list[dict[str, object]] = []
    try:
        for product, step, relative, source in plan:
            evidence = verify_member_file(
                grammar, member.member_id, source,
                inventory_executable=inventory_executable)
            staged_relative = Path(product) / Path(relative).name
            staged = partial / staged_relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            source_sha256 = _sha256_file(source)
            shutil.copy2(source, staged)
            staged_sha256 = _sha256_file(staged)
            if staged_sha256 != source_sha256:
                raise MemberIdentityRefusal(
                    f"staging {source} -> {staged} changed the bytes "
                    f"(sha256 {source_sha256} -> {staged_sha256}); the "
                    "copy cannot be trusted and the tree is not written")
            files.append({
                "product": product,
                "step_hours": step,
                "relative_source": relative,
                "staged": staged_relative.as_posix(),
                "bytes": staged.stat().st_size,
                "sha256": staged_sha256,
                "observed": evidence.to_dict(),
            })
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "member_set": grammar_identity | {"name": grammar.name},
            "member": {
                "id": member.member_id,
                "class": member.class_name,
                "ordinal": member.ordinal,
                "token": member.token,
                "declared_verification": {
                    "product_definition_templates": list(
                        member.verification.product_definition_templates),
                    "type_of_ensemble_forecast": (
                        member.verification.type_of_ensemble_forecast),
                    "perturbation_number": "ordinal",
                    "ensemble_size": member.verification.ensemble_size,
                    "type_of_generating_process": (
                        member.verification.type_of_generating_process),
                    "forecast_generating_process_id": (
                        member.verification.forecast_generating_process_id),
                },
            },
            "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
            "layout": grammar.layout,
            "counting_rule": (
                "the declared member set is the sizing authority; the "
                "encoded numberOfForecastsInEnsemble is verified as "
                "identity evidence only, because the measured ensembles "
                "count their own members incompatibly"),
            "files": files,
            "prepared_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "code_identity": _code_identity(),
        }
        receipt_path = partial / RECEIPT_NAME
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=False) + "\n",
            encoding="utf-8", newline="\n")
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    member_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, member_dir)
    return member_dir


def _code_identity() -> dict[str, object]:
    """The executing tree's identity, by the one shared ladder."""

    try:
        from gpuwm import runtime_manifest

        root = Path(__file__).resolve().parent.parent
        return dict(runtime_manifest.provenance(root))
    except Exception as error:                          # noqa: BLE001
        return {"identity_source": "unresolved", "error": str(error)}


# ---------------------------------------------------------------------------
# The front door
# ---------------------------------------------------------------------------

def _parse_cycle(text: str) -> datetime:
    for pattern in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y%m%dT%H", "%Y%m%d%H"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"cannot parse cycle {text!r}; use YYYY-MM-DDTHH")


def _parse_steps(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in text.split(",") if item != "")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"cannot parse steps {text!r}; use comma-separated hours "
            "such as 0,3,6") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuwm-member-prep",
        description=(
            "Verify and stage one declared ensemble member into a "
            "member-addressed prepared tree.  Members are fetched by "
            "filename but verified at decode on the GRIB ensemble "
            "identity, and every mismatch refuses naming both sides."),
    )
    parser.add_argument(
        "--list-member-sets", action="store_true",
        help="print the packaged member sets and exit")
    parser.add_argument(
        "--describe", metavar="SET",
        help="print one packaged member set's declared members, "
             "statistics and products, then exit")
    parser.add_argument(
        "--member-set", metavar="SET",
        help="packaged member-set id (see --list-member-sets)")
    parser.add_argument(
        "--members-document", type=Path, metavar="JSON",
        help="explicit rw-wps.members.v1 document instead of a packaged "
             "set (its SHA-256 is recorded in the receipt)")
    parser.add_argument("--member", metavar="ID", help="declared member id")
    parser.add_argument(
        "--cycle", type=_parse_cycle, metavar="YYYY-MM-DDTHH",
        help="the model cycle whose member is staged, in UTC; it selects "
             "the declared upstream-relative paths under --inputs and is "
             "recorded in the staging receipt")
    parser.add_argument(
        "--steps", type=_parse_steps, metavar="H,H,...",
        help="forecast hours to prepare, e.g. 0,3,6")
    parser.add_argument(
        "--products", metavar="P,P",
        help="comma-separated declared products (default: all declared)")
    parser.add_argument(
        "--inputs", type=Path, metavar="ROOT",
        help="root holding fetched files at their declared "
             "upstream-relative paths")
    parser.add_argument(
        "--output", type=Path, metavar="ROOT",
        help="root the member-addressed prepared tree is written under")
    parser.add_argument(
        "--verify-only", metavar="FILE", type=Path,
        help="verify one file's bytes against --member and print the "
             "evidence; nothing is staged")
    parser.add_argument("--grib2-inventory", type=Path,
                        help="override the resolved inventory executable")
    return parser


def _describe(grammar: MemberGrammar, identity: Mapping[str, object]) -> str:
    lines = [
        f"member set {grammar.name}",
        f"  document {identity['file']} sha256 {identity['sha256']}",
        f"  layout {grammar.layout}, "
        f"{grammar.declared_member_count} declared members",
        "  members:",
    ]
    for member in grammar.members():
        declared = member.verification
        lines.append(
            f"    {member.member_id}  class={member.class_name} "
            f"perturbationNumber={member.ordinal} "
            f"typeOfEnsembleForecast={declared.type_of_ensemble_forecast} "
            f"pdt={list(declared.product_definition_templates)}"
            + (f" encodedEnsembleSize={declared.ensemble_size}"
               if declared.ensemble_size is not None else ""))
    statistics = grammar.statistics()
    if statistics:
        lines.append("  statistics sharing the namespace (never members):")
        for statistic in statistics:
            lines.append(
                f"    {statistic.statistic_id}  {statistic.statistic}")
    lines.append("  products:")
    for product in grammar.products():
        lines.append(f"    {product}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.list_member_sets:
            for grammar_id in packaged_member_grammar_ids():
                print(f"{grammar_id}  "
                      f"sha256 {packaged_member_grammar_sha256(grammar_id)}")
            return 0
        if arguments.describe:
            grammar, identity = _resolve_grammar(arguments.describe, None)
            print(_describe(grammar, identity))
            return 0
        if arguments.member is None:
            parser.error("--member is required (see --describe SET)")
        if arguments.verify_only is not None:
            grammar, _identity = _resolve_grammar(
                arguments.member_set, arguments.members_document)
            evidence = verify_member_file(
                grammar, arguments.member, arguments.verify_only,
                inventory_executable=arguments.grib2_inventory)
            print(json.dumps({
                "verified": str(arguments.verify_only),
                "member_set": grammar.name,
                "member": arguments.member,
                "observed": evidence.to_dict(),
            }, indent=2))
            return 0
        for name in ("cycle", "steps", "inputs", "output"):
            if getattr(arguments, name) is None:
                parser.error(f"--{name} is required to prepare a member")
        products = (tuple(arguments.products.split(","))
                    if arguments.products else None)
        member_dir = prepare_member(
            grammar_id=arguments.member_set,
            grammar_path=arguments.members_document,
            member_id=arguments.member,
            cycle=arguments.cycle,
            steps=arguments.steps,
            products=products,
            inputs_root=arguments.inputs,
            output_root=arguments.output,
            inventory_executable=arguments.grib2_inventory,
        )
    except (MemberIdentityRefusal, FileNotFoundError, KeyError,
            RuntimeError) as error:
        print(f"error: {error}")
        return 2
    print(member_dir)
    return 0


__all__ = [
    "DERIVED_FORECAST_NAMES", "MemberFileEvidence", "RECEIPT_NAME",
    "RECEIPT_SCHEMA", "STATISTIC_TEMPLATE_FAMILIES", "build_parser", "main",
    "member_inventory_rows", "prepare_member", "verify_member_file",
    "verify_member_rows",
]
