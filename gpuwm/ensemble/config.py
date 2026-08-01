"""The ``[ensemble]`` config surface (EXPERIMENTAL).

An ensemble is declared as a small overlay TOML *on top of* an existing
experiment config, never inside it: ``gpuwm.experiment.build_experiment``
rejects unknown top-level tables, and the experiment TOMLs are
hash-bound, so adding a table to one would change its sha256 and break
every receipt that cites it.  The overlay names the base config and
carries only the ensemble's own knobs::

    [ensemble]
    base_config = "configs/some_experiment.toml"
    n_members = 30
    base_seed = 20260730
    perturbation = "gpuwm.da.perturb"

    [ensemble.perturbation_options]
    # free-form; hashed into the manifest, passed to the perturbation hook

Nothing here is wired into a default route.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

#: Versioned schema of the overlay file itself.
ENSEMBLE_CONFIG_SCHEMA = "gpuwm-ensemble-config.v1"

_KNOWN_KEYS = frozenset({
    "base_config", "n_members", "base_seed", "perturbation",
    "perturbation_options", "ens_root",
})
_REQUIRED_KEYS = ("base_config", "n_members", "base_seed", "perturbation")

#: An ensemble of more than this many members is refused as a typo
#: guard.  Sequential execution means N members cost N forecasts; a
#: mistyped n_members would otherwise burn a night of GPU time.
MAX_MEMBERS = 512


def sha256_of_file(path: Path) -> str:
    """Content sha256 of ``path`` (streamed; used for base-config binding)."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    """sha256 of a canonical JSON encoding (sorted keys, no whitespace)."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EnsembleConfig:
    """A validated ``[ensemble]`` overlay."""

    source: Path
    source_sha256: str
    base_config: Path
    base_config_sha256: str
    n_members: int
    base_seed: int
    perturbation: str
    perturbation_options: Mapping[str, object] = field(default_factory=dict)
    ens_root: Path | None = None

    @property
    def perturbation_options_sha256(self) -> str:
        return canonical_sha256(dict(self.perturbation_options))

    def describe(self) -> dict[str, object]:
        """The provenance block every manifest embeds."""
        return {
            "schema": ENSEMBLE_CONFIG_SCHEMA,
            "source": str(self.source),
            "source_sha256": self.source_sha256,
            "base_config": str(self.base_config),
            "base_config_sha256": self.base_config_sha256,
            "n_members": self.n_members,
            "base_seed": self.base_seed,
            "perturbation": self.perturbation,
            "perturbation_options": dict(self.perturbation_options),
            "perturbation_options_sha256": self.perturbation_options_sha256,
        }


def load_ensemble_config(path: str | Path) -> EnsembleConfig:
    """Parse and validate an ensemble overlay TOML.  Fails closed."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"ensemble config {source} does not exist")
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    unknown_tables = [name for name in raw if name != "ensemble"]
    if unknown_tables:
        raise ValueError(
            f"unknown table(s) {unknown_tables} in ensemble config "
            f"{source}; the overlay carries exactly one [ensemble] table "
            "and points at the base experiment config with base_config.")
    entries = raw.get("ensemble")
    if not isinstance(entries, Mapping):
        raise ValueError(
            f"ensemble config {source} must carry an [ensemble] table")
    unknown = sorted(set(entries) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown} in [ensemble] of {source}; known "
            f"keys: {sorted(_KNOWN_KEYS)}")
    missing = [key for key in _REQUIRED_KEYS if key not in entries]
    if missing:
        raise ValueError(
            f"[ensemble] of {source} is missing required key(s) {missing}")

    base_raw = entries["base_config"]
    if not isinstance(base_raw, str) or not base_raw.strip():
        raise ValueError(
            f"base_config in [ensemble] of {source} must be a non-empty "
            f"path string, got {base_raw!r}")
    base_config = Path(base_raw)
    if not base_config.is_absolute():
        base_config = (source.parent / base_config)
    base_config = base_config.resolve()
    if not base_config.is_file():
        raise ValueError(
            f"base_config {base_config} declared by {source} does not "
            "exist; the ensemble runs an existing experiment config, it "
            "does not create one")

    n_members = entries["n_members"]
    if not isinstance(n_members, int) or isinstance(n_members, bool) \
            or n_members < 1:
        raise ValueError(
            f"n_members in [ensemble] of {source} must be an integer >= 1, "
            f"got {n_members!r}")
    if n_members > MAX_MEMBERS:
        raise ValueError(
            f"n_members = {n_members} in [ensemble] of {source} exceeds the "
            f"{MAX_MEMBERS}-member guard; members run sequentially, so this "
            "is almost certainly a typo. Raise MAX_MEMBERS deliberately if "
            "it is not.")

    base_seed = entries["base_seed"]
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) \
            or base_seed < 0:
        raise ValueError(
            f"base_seed in [ensemble] of {source} must be a non-negative "
            f"integer, got {base_seed!r}")

    perturbation = entries["perturbation"]
    if not isinstance(perturbation, str) or not perturbation.strip():
        raise ValueError(
            f"perturbation in [ensemble] of {source} must be a non-empty "
            "provenance reference string (for example "
            "\"gpuwm.da.perturb\", \"experimental-stub\", or \"none\"), "
            f"got {perturbation!r}")

    options = entries.get("perturbation_options", {})
    if not isinstance(options, Mapping):
        raise ValueError(
            f"[ensemble.perturbation_options] of {source} must be a table, "
            f"got {options!r}")

    ens_root_raw = entries.get("ens_root")
    ens_root = None
    if ens_root_raw is not None:
        if not isinstance(ens_root_raw, str) or not ens_root_raw.strip():
            raise ValueError(
                f"ens_root in [ensemble] of {source} must be a non-empty "
                f"path string, got {ens_root_raw!r}")
        ens_root = Path(ens_root_raw)
        if not ens_root.is_absolute():
            ens_root = source.parent / ens_root

    return EnsembleConfig(
        source=source,
        source_sha256=sha256_of_file(source),
        base_config=base_config,
        base_config_sha256=sha256_of_file(base_config),
        n_members=n_members,
        base_seed=base_seed,
        perturbation=perturbation.strip(),
        perturbation_options=dict(options),
        ens_root=ens_root,
    )
