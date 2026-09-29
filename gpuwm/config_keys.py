"""Declared rows for configuration keys that no typed field carries.

Most keys a config document accepts land in a typed field: a
:class:`gpuwm.config.RunConfig` field, or a field of the dataclass its
table builds, and a front end reads the type and default from there.  A
few keys do not.  Their value is resolved into something else
(``physics_mode`` becomes a resolution object), inherited when absent (a
nest's ``start_time``), or validated in the loader and never stored under
its own name (the ``[fetch]`` hints).  Their types used to exist only in
the parsing code, so a front end could pass them but not learn what they
take.

For each such key, the table that owns it declares one :class:`KeyRow`
beside its key set: the TOML type, the value an absent key means, and one
line saying what it does.  The owning loader checks the value against the
row and takes its default from it, so what a front end is told and what
the loader accepts are one statement.  The loader never refuses a spelling
it read before the row existed unless that spelling breaks something: a
quoted number on a ``number`` row reads as the number, and the ``[shared]``
nest guard keys are compared with their one implemented value rather than
typed.  :func:`gpuwm.config.declared_key_rows` exports every row, next to
the RunConfig fields.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping

#: The TOML value types a row may declare.  ``number`` is an integer or
#: a float, and a string holding one finite decimal number reads as that
#: number (see :func:`_quoted_number`); ``datetime`` is a TOML local
#: date-time; ``table`` is an inline or standard table.
TOML_TYPES = ("string", "integer", "number", "boolean", "datetime",
              "array", "table")

_ARTICLES = {"string": "a string", "integer": "an integer",
             "number": "a number", "boolean": "true or false",
             "datetime": "a TOML date-time", "array": "an array",
             "table": "a table"}


def _is_type(kind: str, value: Any) -> bool:
    """Is ``value`` (as tomllib produces it) of TOML type ``kind``?"""

    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return (isinstance(value, (int, float))
                and not isinstance(value, bool))
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "datetime":
        return isinstance(value, datetime)
    if kind == "array":
        return isinstance(value, (list, tuple))
    if kind == "table":
        return isinstance(value, Mapping)
    raise ValueError(f"not a TOML type: {kind!r}")


def _quoted_number(value: Any) -> float | None:
    """A string holding one finite decimal number, as that number.

    The one ``number`` row's loader read its value through ``float()``
    before the row existed, so ``radius_km = "250"`` cropped 250 km.
    Refusing that spelling once the row arrived named no breakage, and
    a refusal that names none is a defect; the row reads it as the
    number it spells.  Anything that is not one finite number (``"250
    km"``, ``"nan"``) is still refused in the row's own words.
    """

    if not isinstance(value, str):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class KeyRow:
    """One accepted configuration key: its type, default and meaning.

    ``type`` is one TOML type name from :data:`TOML_TYPES`, or a tuple of
    them when the key takes either.  ``default`` is what an absent key
    means; ``None`` means the key is simply unset (its ``doc`` says what
    that does).  ``items`` is the element type of an array.  ``required``
    marks a key its table cannot omit.
    """

    name: str
    type: str | tuple[str, ...]
    default: Any
    doc: str
    items: str | None = None
    required: bool = False

    def __post_init__(self) -> None:
        for kind in self.types:
            if kind not in TOML_TYPES:
                raise ValueError(
                    f"key row {self.name!r}: {kind!r} is not a TOML type; "
                    f"use one of {list(TOML_TYPES)}")
        if self.items is not None and (
                "array" not in self.types or self.items not in TOML_TYPES):
            raise ValueError(
                f"key row {self.name!r}: items names the element type of "
                "an array, and must itself be a TOML type")
        doc = self.doc.strip()
        if not doc or "\n" in doc:
            raise ValueError(
                f"key row {self.name!r} needs a one-line doc")
        if self.default is not None:
            self.check(self.default, where="its own key row")

    @property
    def types(self) -> tuple[str, ...]:
        return (self.type,) if isinstance(self.type, str) else tuple(self.type)

    def spelled(self) -> str:
        return " or ".join(_ARTICLES[kind] for kind in self.types)

    def check(self, value: Any, *, where: str) -> Any:
        """``value`` when it has the declared type; refused otherwise.

        A ``number`` row given a quoted number returns the number
        (:func:`_quoted_number`).

        ``where`` is the location the loader already names in its own
        refusals, such as ``"[fetch] of case.toml"``.
        """

        if not any(_is_type(kind, value) for kind in self.types):
            number = (_quoted_number(value) if "number" in self.types
                      else None)
            if number is not None:
                return number
            raise ValueError(
                f"{self.name} in {where} must be {self.spelled()} "
                f"({self.doc}), got {value!r}.")
        if self.items is not None and isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if not _is_type(self.items, item):
                    raise ValueError(
                        f"{self.name}[{index}] in {where} must be "
                        f"{_ARTICLES[self.items]}, got {item!r}.")
        return value

    def get(self, table: Mapping[str, Any], *, where: str) -> Any:
        """The key's value in ``table``, checked, or the declared default."""

        if self.name not in table:
            return self.default
        return self.check(table[self.name], where=where)

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "type": self.type if isinstance(self.type, str)
            else list(self.type),
            "default": (self.default.isoformat()
                        if isinstance(self.default, datetime)
                        else self.default),
            "doc": self.doc,
        }
        if self.items is not None:
            row["items"] = self.items
        if self.required:
            row["required"] = True
        return row


def key_rows(*rows: KeyRow) -> Mapping[str, KeyRow]:
    """One table's rows by key name, read-only; a name given twice refuses."""

    table: dict[str, KeyRow] = {}
    for row in rows:
        if row.name in table:
            raise ValueError(f"key row {row.name!r} is declared twice")
        table[row.name] = row
    return MappingProxyType(table)


__all__ = ["KeyRow", "TOML_TYPES", "key_rows"]
