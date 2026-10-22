"""Licence policy for model rows, shared by acquisition and every source door.

The concrete breakage prevented here is running restricted or unlicensed
weights through an otherwise permissive source family without a named opt-in.
No model identities or licence exceptions live in this module.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import shlex

TABLE = Path(__file__).with_name("authorities") / "earth2-models.v1.json"
_accepted = ContextVar("earth2_accepted_licences", default=())


def model_row(source, model=None):
    """Return a model row without applying runtime/profile availability checks."""
    name = str(model or source or "").strip().lower().replace("_", "-")
    if name.startswith("earth2-"):
        name = name[len("earth2-"):]
    rows = json.loads(TABLE.read_text(encoding="utf-8"))["models"]
    for key, row in rows.items():
        if name == key or name in (str(a).lower() for a in row.get("aliases", ())):
            return key, row
    return None


def consumer_row(name):
    """Return a consumers row (a diagnostic or conditioning model) by its exact name."""
    name = str(name or "").strip().lower()
    rows = json.loads(TABLE.read_text(encoding="utf-8")).get("consumers", {})
    return (name, rows[name]) if name in rows else None


def require_licence(key, row, accept_licence=None):
    """Restricted rows require an exact, explicit licence name, before any I/O."""
    category = row.get("licence_class", "unresolved")
    if category == "permissive":
        return
    licence = (row.get("weights") or {}).get("licence", "unresolved weight licence")
    if isinstance(accept_licence, str):
        accept_licence = (accept_licence,)
    accepted = (*_accepted.get(), *(accept_licence or ()))
    if licence not in accepted:
        raise ValueError(
            f"earth2-{key}: {category} weights, licence {licence}. "
            "Default paths require permissive weights; explicitly opt in with "
            f"--accept-licence {shlex.quote(licence)}")


def accepted_licences():
    """Explicit runtime acceptance to pass across child-process boundaries."""
    return tuple(_accepted.get())


def add_licence_argument(parser):
    if "--accept-licence" not in parser._option_string_actions:
        parser.add_argument(
            "--accept-licence", action="append", default=[], metavar="LICENCE",
            help="explicitly accept a model row's exact restricted/unresolved weight licence; repeat for multiple licences")


@contextmanager
def licence_scope(accepted):
    token = _accepted.set(tuple(accepted or ()))
    try:
        yield
    finally:
        _accepted.reset(token)


def _values(tokens, option):
    for i, token in enumerate(tokens):
        if token == option and i + 1 < len(tokens):
            yield tokens[i + 1]
        elif token.startswith(option + "="):
            yield token.split("=", 1)[1]


def _check_source(source, model=None, accepted=()):
    # A bare non-family source can have the same spelling as a model key.
    # Only the family namespace (or an explicit family --model) selects it.
    source = str(source or "").strip().lower().replace("_", "-")
    if source.startswith("earth2-") or (
            source in ("earth2", "earth2studio") and model):
        found = model_row(source, model)
        if found:
            require_licence(*found, accept_licence=accepted)


def _check_receipt(path, accepted):
    """Preparation retains the same policy when the source is the bare family."""
    path = Path(path)
    receipt = path if path.name == "earth2-run.json" else path.parent / "earth2-run.json"
    if receipt.is_file():
        document = json.loads(receipt.read_text(encoding="utf-8"))
        found = model_row(document.get("model"))
        if found:
            # Read current authority, never trust a receipt's licence classification.
            require_licence(*found, accept_licence=accepted)
    # Also protect manually produced files with the family's conventional name.
    if path.name.startswith("earth2-"):
        stem = path.stem.rsplit("-", 1)[0]
        _check_source(stem, accepted=accepted)


def check_argv(tokens):
    """Check raw names before argparse canonicalizes aliases or rejects go choices."""
    tokens = list(tokens)
    if not tokens or tokens[0] not in ("fetch", "prep", "domain", "go", "ensemble", "source-intermediate",
                                       "earth2-diagnose", "hf-fetch"):
        return
    if "--help" in tokens or "-h" in tokens:
        return
    accepted = tuple(_values(tokens, "--accept-licence"))
    if tokens[0] == "hf-fetch":
        # The positional target is a bare model row (an hf:// URL has no row).
        rest = iter(tokens[1:])
        for token in rest:
            if token in ("--cache", "--connections", "--accept-licence"):
                next(rest, None)
            elif not token.startswith("-"):
                found = None if token.startswith("hf://") else model_row(token)
                if found:
                    require_licence(*found, accept_licence=accepted)
                break
        return
    if tokens[0] == "earth2-diagnose":
        # The diagnostics door runs consumers rows (their weights are what the
        # licence names), never a models row; nothing below applies to it.
        for names in _values(tokens, "--for"):
            for name in names.split(","):
                found = consumer_row(name)
                if found:
                    require_licence(*found, accept_licence=accepted)
        return
    model = next(_values(tokens, "--model"), None)
    for source in _values(tokens, "--source"):
        _check_source(source, model, accepted)
    # go CONFIG and prep/domain --experiment-config must not hide a restricted row.
    configs = list(_values(tokens, "--experiment-config"))
    if tokens[0] in ("go", "ensemble") and len(tokens) > 1 and not tokens[1].startswith("-"):
        configs.append(tokens[1])
    for config in configs:
        path = Path(config)
        if path.is_file():
            import tomllib
            try:
                document = tomllib.loads(path.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError:
                # The owning command reports malformed configuration in its
                # own vocabulary. An undecodable file cannot launch a model.
                continue
            fetch = document.get("fetch", {})
            _check_source(fetch.get("source"), fetch.get("model"), accepted)
            root = fetch.get("source_root")
            if root:
                _check_receipt(Path(root) / "earth2-run.json", accepted)
    for option in ("--input", "--source-manifest", "--author-input-manifest"):
        for path in _values(tokens, option):
            _check_receipt(path, accepted)
    for option in ("--source-root", "--data-dir"):
        for path in _values(tokens, option):
            _check_receipt(Path(path) / "earth2-run.json", accepted)
    for path in _values(tokens, "--input-list"):
        if Path(path).is_file():
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    _check_receipt(line.strip(), accepted)


def check_namespace(args):
    accepted = getattr(args, "accept_licence", ()) or ()
    _check_source(getattr(args, "source", None), getattr(args, "model", None), accepted)
    for path in getattr(args, "mapped_inputs", ()) or ():
        _check_receipt(path, accepted)
    config = getattr(args, "config", None) or getattr(args, "experiment_config", None)
    if config:
        tokens = ["go", str(config)]
        for licence in accepted:
            tokens.extend(("--accept-licence", licence))
        check_argv(tokens)
