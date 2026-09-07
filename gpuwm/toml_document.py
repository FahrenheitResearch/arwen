"""Lossless TOML emission shared by preparation and forecast branching."""
from __future__ import annotations

import datetime as _datetime
import json
import re
import tomllib
from typing import Any, Mapping


_BARE_KEY = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _key(name: str) -> str:
    if name and set(name) <= _BARE_KEY:
        return name
    return json.dumps(name)


def _scalar(value: Any, where: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(
                f"{where} is {value!r}, which a branched configuration "
                "cannot carry: write a finite number")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, _datetime.datetime):
        return value.isoformat()
    if isinstance(value, _datetime.date):
        return value.isoformat()
    if isinstance(value, _datetime.time):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(
            _scalar(item, f"{where}[{index}]")
            for index, item in enumerate(value)) + "]"
    raise ValueError(
        f"{where} holds {type(value).__name__}, which is not a TOML "
        "value; a branched configuration is written from the parsed "
        "document, so every value must be one this emitter knows")


def _emit_table(raw: Mapping[str, Any], prefix: str,
                lines: list[str]) -> None:
    scalars = [(k, v) for k, v in raw.items()
               if not isinstance(v, dict)
               and not _is_table_array(v)]
    tables = [(k, v) for k, v in raw.items() if isinstance(v, dict)]
    arrays = [(k, v) for k, v in raw.items() if _is_table_array(v)]
    for key, value in scalars:
        where = f"{prefix}.{key}" if prefix else key
        lines.append(f"{_key(key)} = {_scalar(value, where)}")
    for key, value in tables:
        name = f"{prefix}.{_key(key)}" if prefix else _key(key)
        lines.append("")
        lines.append(f"[{name}]")
        _emit_table(value, name, lines)
    for key, value in arrays:
        name = f"{prefix}.{_key(key)}" if prefix else _key(key)
        for element in value:
            lines.append("")
            lines.append(f"[[{name}]]")
            _emit_table(element, name, lines)


def _is_table_array(value: Any) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(item, dict) for item in value))


def emit_experiment_toml(raw: Mapping[str, Any]) -> str:
    """``raw`` as TOML text that parses back to ``raw``, exactly.

    Round-tripped here rather than trusted: this emitter writes the
    configuration a branch will INTEGRATE, and a value it silently
    mangled would be a physics change nobody asked for.  The check is
    the cheapest possible proof that it did not.
    """

    lines: list[str] = []
    _emit_table(raw, "", lines)
    text = "\n".join(line for line in lines).lstrip("\n") + "\n"
    reparsed = tomllib.loads(text)
    if reparsed != dict(raw):
        raise RuntimeError(
            "the branched configuration did not survive its own "
            "round-trip; refusing to write a config whose bytes and "
            "meaning disagree")
    return text


def iter_toml_statements(text: str):
    """Yield complete TOML statements with decoded keys and original lines.

    Validate the complete document before rewriting it. Parsing one statement
    at a time lets TOML itself recognize quoted/escaped table and key names,
    while keeping arrays and multiline strings together during rewriting.
    """
    tomllib.loads(text)
    key_token = r'''(?:[A-Za-z0-9_-]+|'[^'\r\n]*'|"(?:[^"\\\r\n]|\\.)*")'''
    assignment = re.compile(
        rf"^\s*({key_token}(?:\s*\.\s*{key_token})*)\s*=")

    def key_path(parsed):
        path = []
        while isinstance(parsed, dict) and len(parsed) == 1:
            key, parsed = next(iter(parsed.items()))
            path.append(key)
        return tuple(path)

    pending = []
    # str.splitlines also splits valid string/key characters such as U+2028.
    # TOML physical lines are LF or CRLF only.
    for physical_line in text.removesuffix("\n").split("\n"):
        line = physical_line.removesuffix("\r")
        if not pending and (not line.strip() or line.lstrip().startswith("#")):
            yield "trivia", (), [line]
            continue
        pending.append(line)
        try:
            parsed = tomllib.loads("\n".join(pending) + "\n")
        except tomllib.TOMLDecodeError:
            # A valid document can contain an incomplete multiline value.
            continue
        first = pending[0].lstrip()
        if first.startswith("["):
            kind = "array" if first.startswith("[[") else "table"
            path = key_path(parsed)
        else:
            match = assignment.match(pending[0])
            if match is None:
                raise ValueError("Cannot identify a validated TOML assignment")
            kind = "assignment"
            path = key_path(tomllib.loads(f"{match.group(1)} = 0"))
        yield kind, path, pending
        pending = []
    if pending:
        raise ValueError("Cannot identify a complete validated TOML statement")
