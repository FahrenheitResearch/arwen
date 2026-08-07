"""Publish the experiment TOML a programmatic preparation actually used.

Some preparation routes build their :class:`ExperimentConfig` in code
rather than reading a config file: the native HRRR route derives every
domain and physics control from the target-domain document, the WRF
namelist and the selected physics profile, and hands the resulting
tables straight to :func:`gpuwm.experiment.build_experiment`.  That is
the right shape for a preparer -- there is no file for a user to get
wrong -- but it leaves the STAGE AFTER IT with nothing to bind.

A prepared-cache identity contains ``domain_config``, compared by
strict equality on roughly a hundred and ten run fields.  A downstream
runner that has to supply an experiment config for the same case can
either be handed one that provably round-trips to what the preparation
used, or it can be hand-matched field by field and quietly refused
months later on a default nobody remembers changing.  This module is
the first option: the preparation renders the exact tables it built,
writes them, reads them back through the ordinary front door, and
refuses to publish if the reloaded domain identity is not identical.

The renderer is deliberately narrow.  It accepts the value types an
experiment document actually holds -- bool, int, float, str, datetime
and homogeneous lists of those -- and raises on anything else rather
than quoting it, because a number emitted as a string reads as valid
TOML under the right key and is not what was computed.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

#: Header stamped into every published document, so a reader who finds
#: one in a prepared bundle knows it was generated and what it is for.
_HEADER = (
    "# gpuwm generated experiment authority -- do not hand-edit.\n"
    "# Rendered from the tables the preparation itself built, and\n"
    "# verified to reload to the same per-domain identity.\n"
)

#: Emission order.  ``[[domain]]`` last, matching every shipped config,
#: because ``[shared]`` defaults have to be in scope when they are read.
_TABLE_ORDER = ("experiment", "projection", "shared")

#: Long numeric arrays (eta ladders) wrap at this many values per line.
_ARRAY_WRAP = 5


class ExperimentDocumentError(ValueError):
    """A value that cannot be rendered, or a document that did not reload."""


def _literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # repr round-trips a float exactly; tomllib parses it back to the
        # same double, which is what identity equality needs.
        if value != value or value in (float("inf"), float("-inf")):
            raise ExperimentDocumentError(
                "experiment document contains a non-finite float")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.isoformat()
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    raise ExperimentDocumentError(
        f"cannot render {value!r} ({type(value).__name__}) into an "
        "experiment document: it is neither a scalar, a string, a "
        "datetime, nor a list of those")


def _render_value(value: object, indent: str = "") -> str:
    if isinstance(value, (list, tuple)):
        items = [_literal(item) for item in value]
        if len(items) <= _ARRAY_WRAP:
            return "[" + ", ".join(items) + "]"
        lines = ["["]
        for start in range(0, len(items), _ARRAY_WRAP):
            lines.append(
                indent + "    " + ", ".join(items[start:start + _ARRAY_WRAP])
                + ",")
        lines.append(indent + "]")
        return "\n".join(lines)
    return _literal(value)


def _render_table(name: str, entries: Mapping[str, object],
                  *, array_of_tables: bool = False) -> str:
    header = f"[[{name}]]" if array_of_tables else f"[{name}]"
    lines = [header]
    for key, value in entries.items():
        if not isinstance(key, str) or not key:
            raise ExperimentDocumentError(f"unsupported table key {key!r}")
        lines.append(f"{key} = {_render_value(value)}")
    return "\n".join(lines) + "\n"


def render_experiment_document(raw: Mapping[str, object]) -> str:
    """Render ``raw`` -- the tables ``build_experiment`` accepts -- as TOML."""

    tables = dict(raw)
    domains = tables.pop("domain", None)
    if not isinstance(domains, Sequence) or isinstance(domains, (str, bytes)) \
            or not domains:
        raise ExperimentDocumentError(
            "an experiment document needs at least one [[domain]] table")
    unknown = set(tables) - set(_TABLE_ORDER)
    if unknown:
        raise ExperimentDocumentError(
            f"unsupported experiment table(s) {sorted(unknown)}")
    if "experiment" not in tables:
        raise ExperimentDocumentError(
            "an experiment document needs an [experiment] table")
    chunks = [_HEADER]
    for name in _TABLE_ORDER:
        entries = tables.get(name)
        if entries is None:
            continue
        if not isinstance(entries, Mapping):
            raise ExperimentDocumentError(f"[{name}] must be a table")
        chunks.append(_render_table(name, entries))
    for domain in domains:
        if not isinstance(domain, Mapping):
            raise ExperimentDocumentError("[[domain]] entries must be tables")
        chunks.append(_render_table("domain", domain, array_of_tables=True))
    return "\n".join(chunks)


def publish_experiment_document(
        path: str | Path, raw: Mapping[str, object], experiment,
) -> Path:
    """Write ``raw`` as TOML and prove it reloads to ``experiment``.

    The proof is the point.  ``prepared_domain_config_identity`` is what
    a prepared cache stores and what every restore compares, so
    comparing exactly that -- per domain -- is comparing the thing that
    will be checked, not a proxy for it.  A mismatch refuses and leaves
    no file behind: publishing a document that does not reload would
    hand the next stage an authority that fails at its front door with a
    message about the user's config.

    The VERTICAL grid is compared too, and separately, because it is not
    in the domain identity: ``eta_levels``, ``p_top`` and their hybrid
    controls live on the experiment, so a document that reloaded with a
    different ladder would pass a domain-identity comparison and then
    run a different model.  The start time is compared for the same
    reason -- it dates every forcing frame and is not a domain field.
    """

    from gpuwm.experiment import load_experiment
    from gpuwm.ingest.prepared_cache import prepared_domain_config_identity

    target = Path(path)
    text = render_experiment_document(raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Create-only: a published authority is bound by digest downstream,
    # so silently replacing one would invalidate a receipt already written.
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    try:
        reloaded = load_experiment(target)
        expected = [prepared_domain_config_identity(domain)
                    for domain in experiment.domains]
        observed = [prepared_domain_config_identity(domain)
                    for domain in reloaded.domains]
        if observed != expected:
            raise ExperimentDocumentError(
                "published experiment document does not reload to the "
                "prepared domain identity it was rendered from")
        if reloaded.start_time != experiment.start_time:
            raise ExperimentDocumentError(
                "published experiment document reloads to a different "
                f"start time ({reloaded.start_time.isoformat()} vs "
                f"{experiment.start_time.isoformat()})")
        if reloaded.vertical != experiment.vertical:
            raise ExperimentDocumentError(
                "published experiment document does not reload to the "
                "vertical grid it was rendered from")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "ExperimentDocumentError",
    "publish_experiment_document",
    "render_experiment_document",
]
