"""The ensemble member-addressing grammar: ``rw-wps.members.v1``.

An ensemble source enters gpuwm the same way every source does -- as
table data.  This module is the one consumer of the ``rw-wps.members.v1``
document: a packaged JSON authority that declares, per ensemble, the
member set (classes, ordinals, id/token templates), the per-member
filename pattern for each published product, the byte-level verification
contract for each member class, and the statistic namespace (mean/spread
names) that shares the members' directory.

The contract the grammar exists to enforce, measured across four real
ensembles that disagree four incompatible ways about their own counts:

* members are FETCHED by filename, because that is how every
  file-per-member feed is published;
* members are VERIFIED at decode on the GRIB ensemble identity --
  (productDefinitionTemplateNumber, typeOfEnsembleForecast,
  perturbationNumber) -- because filenames are a distribution
  convention, not a format one, and because ensemble means and spreads
  share the members' namespace and decode cleanly as ordinary fields;
* verification FAILS CLOSED: a mismatch between what the filename claims
  and what the bytes carry is a refusal naming both, never a warning.

Nothing in this module knows any model's name.  What GEFS or AIGEFS
looks like -- an unflagged control, an ensemble-size octet that
excludes (or includes) the control, a member that is a path component
rather than a filename token -- is rows in the packaged documents, and
the next ensemble (ERA5-EDA, 20CRv3, AIFS-ENS members) is another
document, not another code path.

Verification itself lives in :mod:`gpuwm.member_prep`, which reads the
real bytes through the Rust GRIB2 inventory; this module owns the
declarative half only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

MEMBERS_SCHEMA = "rw-wps.members.v1"

#: The one layout this build implements.  Declaring any other refuses by
#: name: a concatenated-members layout (ECMWF AIFS-ENS packs all fifty
#: members into one file per step) would need metadata-driven member
#: extraction, and accepting the declaration without that engine would
#: silently treat a fifty-member file as one member.  The verification
#: triple contract is layout-independent, so adding that layout later is
#: an engine extension, not a grammar change.
SUPPORTED_LAYOUTS = ("file_per_member",)

#: Template placeholders a path/id template may use.  ``{ordinal:0Nd}``
#: zero-pads the member ordinal to N digits; the rest are filled at
#: resolution time from the cycle and forecast step.
_ORDINAL_FIELD = re.compile(r"\{ordinal:0([1-9])d\}")
_PATH_FIELDS = ("{yyyymmdd}", "{hh}", "{fff}", "{token}")


class MemberGrammarError(ValueError):
    """A members document that cannot be trusted, with the reason."""


class MemberIdentityRefusal(ValueError):
    """A member claim the grammar refuses, naming the concrete breakage."""


def _require_keys(
    value: Mapping[str, object], *, allowed: frozenset[str],
    required: frozenset[str], label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise MemberGrammarError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise MemberGrammarError(
            f"{label} declares unknown keys {unknown}; this build "
            f"understands {sorted(allowed)} -- an unknown key is a "
            "contract this build would silently not enforce, so it "
            "refuses instead")
    missing = sorted(required - set(value))
    if missing:
        raise MemberGrammarError(f"{label} is missing required keys {missing}")


def _expand_ordinal(template: str, ordinal: int, label: str) -> str:
    """Fill ``{ordinal:0Nd}`` and nothing else; refuse leftovers."""

    def _fill(match: re.Match[str]) -> str:
        width = int(match.group(1))
        text = f"{ordinal:0{width}d}"
        if len(text) > width:
            raise MemberGrammarError(
                f"{label}: ordinal {ordinal} does not fit in "
                f"{width} digit(s)")
        return text

    expanded = _ORDINAL_FIELD.sub(_fill, template)
    leftover = re.search(r"\{[^}]*\}", expanded)
    if leftover:
        raise MemberGrammarError(
            f"{label} uses unsupported placeholder {leftover.group(0)!r}; "
            "a member id/token template may use {ordinal:0Nd} only -- the "
            "path fields " + ", ".join(_PATH_FIELDS) + " belong to "
            "relative_path templates")
    return expanded


@dataclass(frozen=True)
class MemberVerification:
    """The declared byte contract for one member class.

    ``ensemble_size`` is the value the source ENCODES in
    ``numberOfForecastsInEnsemble``, declared so the bytes can be
    checked against it as identity evidence -- it is deliberately NOT
    the member count.  The four measured ensembles disagree about their
    own counts in four incompatible ways (GEFS encodes 30 for a 31-
    member set, AIGEFS 31, RRFS contradicts itself between products and
    domains), so the declared member set is the only sizing authority
    and this octet is never used to size anything.
    """

    product_definition_templates: tuple[int, ...]
    type_of_ensemble_forecast: int
    ensemble_size: int | None
    type_of_generating_process: int | None
    forecast_generating_process_id: int | None


@dataclass(frozen=True)
class MemberIdentity:
    """One declared ensemble member: identity plus its byte contract."""

    member_id: str
    class_name: str
    ordinal: int
    token: str
    verification: MemberVerification


@dataclass(frozen=True)
class StatisticIdentity:
    """One declared derived product sharing the members' namespace."""

    statistic_id: str
    statistic: str
    token: str | None
    relative_paths: tuple[str, ...]
    product_definition_templates: tuple[int, ...] | None
    derived_forecast: int | None


def _verification(value: Mapping[str, object], label: str) -> MemberVerification:
    _require_keys(
        value,
        allowed=frozenset({
            "product_definition_templates", "type_of_ensemble_forecast",
            "perturbation_number", "ensemble_size",
            "type_of_generating_process", "forecast_generating_process_id",
        }),
        required=frozenset({
            "product_definition_templates", "type_of_ensemble_forecast",
            "perturbation_number",
        }),
        label=label,
    )
    templates = value["product_definition_templates"]
    if (not isinstance(templates, Sequence) or isinstance(templates, str)
            or not templates
            or not all(isinstance(item, int) for item in templates)):
        raise MemberGrammarError(
            f"{label}.product_definition_templates must be a non-empty "
            "list of integers")
    if value["perturbation_number"] != "ordinal":
        # The only rule the four measured ensembles share is that the
        # perturbation number IS the member ordinal.  A grammar wanting
        # any other relationship is declaring a contract this build
        # would not verify, which must refuse rather than half-apply.
        raise MemberGrammarError(
            f"{label}.perturbation_number must be the literal 'ordinal': "
            "every supported ensemble encodes the member ordinal as "
            "perturbationNumber, and a different relationship needs an "
            "engine extension, not a silent pass-through")
    for key in ("type_of_ensemble_forecast", "ensemble_size",
                "type_of_generating_process",
                "forecast_generating_process_id"):
        item = value.get(key)
        if item is not None and not isinstance(item, int):
            raise MemberGrammarError(f"{label}.{key} must be an integer")
    return MemberVerification(
        product_definition_templates=tuple(int(t) for t in templates),
        type_of_ensemble_forecast=int(value["type_of_ensemble_forecast"]),
        ensemble_size=value.get("ensemble_size"),
        type_of_generating_process=value.get("type_of_generating_process"),
        forecast_generating_process_id=value.get(
            "forecast_generating_process_id"),
    )


def _class_ordinals(value: object, label: str) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        if not value or not all(
                isinstance(item, int) and item >= 0 for item in value):
            raise MemberGrammarError(
                f"{label} must be non-empty non-negative integers")
        ordinals = tuple(int(item) for item in value)
    elif isinstance(value, Mapping):
        _require_keys(
            value, allowed=frozenset({"first", "last"}),
            required=frozenset({"first", "last"}), label=label)
        first, last = value["first"], value["last"]
        if (not isinstance(first, int) or not isinstance(last, int)
                or first < 0 or last < first):
            raise MemberGrammarError(
                f"{label} range must satisfy 0 <= first <= last")
        ordinals = tuple(range(first, last + 1))
    else:
        raise MemberGrammarError(
            f"{label} must be a list of ordinals or a first/last range")
    if len(set(ordinals)) != len(ordinals):
        raise MemberGrammarError(f"{label} contains duplicate ordinals")
    return ordinals


class MemberGrammar:
    """A validated ``rw-wps.members.v1`` document, ready to resolve."""

    def __init__(self, document: Mapping[str, object], *, source: str):
        _require_keys(
            document,
            allowed=frozenset({
                "schema", "name", "format", "layout",
                "declared_member_count", "cycle_hours", "classes",
                "statistics", "products", "front_doors", "notes",
            }),
            required=frozenset({
                "schema", "name", "format", "layout",
                "declared_member_count", "classes", "products",
            }),
            label=f"members document {source}",
        )
        if document["schema"] != MEMBERS_SCHEMA:
            raise MemberGrammarError(
                f"{source} declares schema {document['schema']!r}; this "
                f"build implements {MEMBERS_SCHEMA!r} only")
        if document["layout"] not in SUPPORTED_LAYOUTS:
            raise MemberGrammarError(
                f"{source} declares layout {document['layout']!r}, which "
                "this build does not implement; supported: "
                f"{list(SUPPORTED_LAYOUTS)}.  A concatenated-members "
                "layout needs metadata-driven member extraction -- "
                "accepting the declaration without it would silently "
                "treat a whole-ensemble file as one member")
        if document["format"] != "grib2":
            raise MemberGrammarError(
                f"{source} declares format {document['format']!r}; member "
                "verification reads the GRIB2 ensemble octets, so only "
                "grib2 members documents are supported -- a NetCDF "
                "ensemble needs the dimension-based member coordinate of "
                "the mapping grammar instead")
        self.source = source
        self.name = str(document["name"])
        self.layout = str(document["layout"])
        self.declared_member_count = document["declared_member_count"]
        if (not isinstance(self.declared_member_count, int)
                or self.declared_member_count < 1):
            raise MemberGrammarError(
                f"{source}.declared_member_count must be a positive integer")

        members: dict[str, MemberIdentity] = {}
        ordinals_seen: dict[int, str] = {}
        classes = document["classes"]
        if not isinstance(classes, Mapping) or not classes:
            raise MemberGrammarError(f"{source}.classes must be a non-empty object")
        for class_name, declaration in classes.items():
            label = f"{source}.classes.{class_name}"
            _require_keys(
                declaration,
                allowed=frozenset({
                    "ordinals", "member_id", "token", "verification", "note",
                }),
                required=frozenset({
                    "ordinals", "member_id", "token", "verification",
                }),
                label=label,
            )
            verification = _verification(
                declaration["verification"], f"{label}.verification")
            for ordinal in _class_ordinals(declaration["ordinals"],
                                           f"{label}.ordinals"):
                member_id = _expand_ordinal(
                    str(declaration["member_id"]), ordinal,
                    f"{label}.member_id")
                token = _expand_ordinal(
                    str(declaration["token"]), ordinal, f"{label}.token")
                if member_id in members:
                    raise MemberGrammarError(
                        f"{source} declares member id {member_id!r} twice")
                if ordinal in ordinals_seen:
                    raise MemberGrammarError(
                        f"{source} declares ordinal {ordinal} in both "
                        f"{ordinals_seen[ordinal]!r} and {class_name!r}: "
                        "perturbationNumber could not identify one member")
                ordinals_seen[ordinal] = class_name
                members[member_id] = MemberIdentity(
                    member_id=member_id, class_name=str(class_name),
                    ordinal=ordinal, token=token, verification=verification)
        if len(members) != self.declared_member_count:
            raise MemberGrammarError(
                f"{source} declares declared_member_count="
                f"{self.declared_member_count} but its classes enumerate "
                f"{len(members)} members; a count the classes do not back "
                "is exactly the kind of self-disagreement this grammar "
                "exists to keep out of the engine")
        self._members = MappingProxyType(members)

        statistics: dict[str, StatisticIdentity] = {}
        for stat_id, declaration in dict(document.get("statistics") or {}).items():
            label = f"{source}.statistics.{stat_id}"
            _require_keys(
                declaration,
                allowed=frozenset({
                    "statistic", "token", "relative_paths",
                    "product_definition_templates", "derived_forecast",
                    "note",
                }),
                required=frozenset({"statistic"}),
                label=label,
            )
            token = declaration.get("token")
            paths = tuple(declaration.get("relative_paths") or ())
            if token is None and not paths:
                raise MemberGrammarError(
                    f"{label} names neither a token nor relative_paths; an "
                    "unaddressable statistic cannot be guarded by name")
            templates = declaration.get("product_definition_templates")
            statistics[str(stat_id)] = StatisticIdentity(
                statistic_id=str(stat_id),
                statistic=str(declaration["statistic"]),
                token=None if token is None else str(token),
                relative_paths=tuple(str(p) for p in paths),
                product_definition_templates=(
                    None if templates is None
                    else tuple(int(t) for t in templates)),
                derived_forecast=declaration.get("derived_forecast"),
            )
        self._statistics = MappingProxyType(statistics)

        products: dict[str, Mapping[str, object]] = {}
        declared_products = document["products"]
        if not isinstance(declared_products, Mapping) or not declared_products:
            raise MemberGrammarError(
                f"{source}.products must be a non-empty object")
        for product_id, declaration in declared_products.items():
            label = f"{source}.products.{product_id}"
            _require_keys(
                declaration,
                allowed=frozenset({
                    "relative_path", "statistics_published", "note",
                }),
                required=frozenset({"relative_path"}),
                label=label,
            )
            products[str(product_id)] = MappingProxyType(dict(declaration))
        self._products = MappingProxyType(products)

    # -- the declared sets -------------------------------------------------

    def members(self) -> tuple[MemberIdentity, ...]:
        return tuple(self._members.values())

    def statistics(self) -> tuple[StatisticIdentity, ...]:
        return tuple(self._statistics.values())

    def products(self) -> tuple[str, ...]:
        return tuple(self._products)

    def member(self, member_id: str) -> MemberIdentity:
        """Resolve a declared member id, or refuse usefully.

        A statistic id asked for as a member is the highest-value
        refusal this grammar owns: the file exists, follows the member
        naming pattern, and decodes cleanly, so nothing downstream
        would ever notice that an ensemble mean is not a trajectory.
        """

        found = self._members.get(member_id)
        if found is not None:
            return found
        statistic = self._statistics.get(member_id)
        if statistic is not None:
            raise MemberIdentityRefusal(
                f"{member_id!r} names the {self.name} "
                f"{statistic.statistic} ({statistic.statistic_id}), not a "
                "member: a derived statistic is not a dynamically "
                "balanced trajectory, and initializing from one would "
                "silently smooth every field it touched.  Declared "
                f"members: {self._member_span()}")
        raise MemberIdentityRefusal(
            f"{self.name} declares no member {member_id!r}; declared "
            f"members: {self._member_span()}")

    def member_for_ordinal(self, ordinal: int) -> MemberIdentity | None:
        for member in self._members.values():
            if member.ordinal == ordinal:
                return member
        return None

    def _member_span(self) -> str:
        ids = list(self._members)
        if len(ids) > 6:
            return f"{', '.join(ids[:3])} .. {ids[-1]} ({len(ids)} members)"
        return ", ".join(ids)

    # -- path resolution ---------------------------------------------------

    def relative_path(
        self, member_id: str, product: str, cycle: datetime, step_hours: int,
    ) -> str:
        """The declared upstream-relative path of one member file."""

        member = self.member(member_id)
        declaration = self._products.get(product)
        if declaration is None:
            raise MemberIdentityRefusal(
                f"{self.name} declares no product {product!r}; declared "
                f"products: {list(self._products)}")
        if step_hours < 0:
            raise MemberIdentityRefusal(
                f"forecast step must be non-negative, got {step_hours}")
        template = str(declaration["relative_path"])
        return (template
                .replace("{token}", member.token)
                .replace("{yyyymmdd}", cycle.strftime("%Y%m%d"))
                .replace("{hh}", f"{cycle.hour:02d}")
                .replace("{fff}", f"{step_hours:03d}"))

    def classify_relative_path(
        self, relative_path: str,
    ) -> MemberIdentity | StatisticIdentity | None:
        """Which declared name a path addresses, if any.

        Statistics are matched FIRST: on feeds where they share the
        members' filename pattern, the statistic token is the only
        difference, and a member-first match with a wildcard token
        would swallow them.
        """

        normalized = relative_path.replace("\\", "/")
        for statistic in self._statistics.values():
            for template in statistic.relative_paths:
                if _template_regex(template).fullmatch(normalized):
                    return statistic
            if statistic.token is not None:
                for declaration in self._products.values():
                    template = str(declaration["relative_path"]).replace(
                        "{token}", statistic.token)
                    if _template_regex(template).fullmatch(normalized):
                        return statistic
        for member in self._members.values():
            for declaration in self._products.values():
                template = str(declaration["relative_path"]).replace(
                    "{token}", member.token)
                if _template_regex(template).fullmatch(normalized):
                    return member
        return None


_REGEX_CACHE: dict[str, re.Pattern[str]] = {}


def _template_regex(template: str) -> re.Pattern[str]:
    cached = _REGEX_CACHE.get(template)
    if cached is not None:
        return cached
    pattern = re.escape(template)
    for field, expression in (
            (r"\{yyyymmdd\}", r"\d{8}"),
            (r"\{hh\}", r"\d{2}"),
            (r"\{fff\}", r"\d{3}")):
        pattern = pattern.replace(field, expression)
    compiled = re.compile(pattern)
    _REGEX_CACHE[template] = compiled
    return compiled


def load_member_grammar(path: Path) -> MemberGrammar:
    """Read and validate one packaged members document."""

    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MemberGrammarError(
            f"cannot read members document {path}: {error}") from None
    return MemberGrammar(document, source=path.name)


__all__ = [
    "MEMBERS_SCHEMA", "SUPPORTED_LAYOUTS",
    "MemberGrammar", "MemberGrammarError", "MemberIdentity",
    "MemberIdentityRefusal", "MemberVerification", "StatisticIdentity",
    "load_member_grammar",
]
