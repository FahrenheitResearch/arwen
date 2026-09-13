"""Preserve preparation authorities when reusing or replacing a request."""
from __future__ import annotations

import json
from pathlib import Path


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def preparation_directory(directory, *, receipt, source_files, documents,
                          required_files=()):
    """Reuse matching authorities, or reserve a separate preparation tree."""
    from gpuwm.stage_reuse import prepared_inputs_reusable

    directory = Path(directory)
    try:
        reusable = prepared_inputs_reusable(
            directory, receipt=receipt, source_files=source_files)
        if not directory.exists() and not directory.is_symlink():
            return directory, False
        matching = all((directory / name).read_bytes() == content
                       for name, content in documents.items())
        complete = all((directory / name).is_file() for name in required_files)
        if reusable and matching and complete:
            return directory, True
    except (OSError, ValueError):
        pass
    sequence = 1
    while True:
        generation = directory.with_name(f"{directory.name}-attempt-{sequence:03d}")
        try:
            generation.mkdir()
            return generation / directory.name, False
        except FileExistsError:
            sequence += 1


def write_document(path, content: bytes, *, reused: bool) -> None:
    """A retained authority is compared, never reopened for writing."""
    path = Path(path)
    if reused:
        try:
            matching = path.read_bytes() == content
        except OSError:
            matching = False
        if not matching:
            raise ValueError(
                f"Prepared authority {path} does not match this preparation. "
                "Keep the retained input tree and prepare a separate generation.")
        return
    with path.open("xb") as handle:
        handle.write(content)
