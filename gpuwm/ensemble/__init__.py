"""Sequential single-GPU ensemble orchestration (EXPERIMENTAL).

One member is resident in VRAM at a time.  Members are run by the
existing single-domain experiment runner -- this package adds member
layout, deterministic seeding, a manifest, resume, a cycling skeleton
with a data-assimilation seam, and a bench harness.  It contains no
model loop of its own and no physics.

Everything published here is stamped ``experimental`` in provenance and
is not wired into any default route: the entry point is
``tools/ensemble_forecast.py``.

THE DOOR IS LAZY.  Each exported name is resolved from its submodule the
first time it is read, so importing one submodule does not import the
orchestration stack.  The breakage this prevents: every preparation asks
``gpuwm.ensemble.posted_preparation`` whether a member input is bound, and
the namelist importer reads ``gpuwm.ensemble.stochastic``; with eager
exports each of those imports also pulled ``cycle`` and ``engine``, which
reach the forecast executor, so the standalone RW-WPS preprocessing wheel
(which stages no forecast executor) could not be staged and none of its
preparations could run.
"""

from __future__ import annotations

import importlib
import importlib.machinery

#: exported name -> the submodule that defines it.
_EXPORTS = {
    "ENSEMBLE_CONFIG_SCHEMA": "config",
    "EnsembleConfig": "config",
    "load_ensemble_config": "config",
    # The supported way to observe a leg's analyses, exported at the package
    # door so a consumer never has to reach for the directory instead.  See
    # gpuwm.ensemble.cycle for why a raw listing is out of contract.
    "read_analysis_roster": "cycle",
    "recover_analysis_publication": "cycle",
    "INCREMENT_CONTRACT": "increments",
    "apply_increments": "increments",
    "apply_increments_to_checkpoint": "increments",
    "CYCLE_MANIFEST_SCHEMA": "manifest",
    "ENSEMBLE_MANIFEST_SCHEMA": "manifest",
    "MEMBER_STATUSES": "manifest",
    "member_directory_name": "manifest",
    "read_manifest": "manifest",
    "write_manifest_atomically": "manifest",
    "SEED_DERIVATION": "seeds",
    "member_seed": "seeds",
    "WRFOUT_INVENTORY_CONTRACT": "wrfout_inventory",
    "WRFOUT_INVENTORY_KEY": "wrfout_inventory",
    "member_inventory": "wrfout_inventory",
}

__all__ = [
    "CYCLE_MANIFEST_SCHEMA",
    "ENSEMBLE_CONFIG_SCHEMA",
    "ENSEMBLE_MANIFEST_SCHEMA",
    "INCREMENT_CONTRACT",
    "MEMBER_STATUSES",
    "SEED_DERIVATION",
    "WRFOUT_INVENTORY_CONTRACT",
    "WRFOUT_INVENTORY_KEY",
    "EnsembleConfig",
    "member_inventory",
    "apply_increments",
    "apply_increments_to_checkpoint",
    "load_ensemble_config",
    "member_directory_name",
    "member_seed",
    "read_analysis_roster",
    "read_manifest",
    "recover_analysis_publication",
    "write_manifest_atomically",
]


def request_installed() -> bool:
    """Whether this installation carries the ``[ensemble]`` table's validator.

    False in the standalone RW-WPS preprocessing wheel, which stages the
    package's preparation side only: ``gpuwm.ensemble.request`` reads the
    batched forecast's product table.  The config loaders ask this before
    they validate the table, so a preparation there (which consumes nothing
    from it) does not die on a bare ModuleNotFoundError; the table is
    validated by the ensemble door, where the forecast is installed.

    The validator counts only when it is in this package's own directories
    (``__path__``), the rule ``gpuwm.stage_cli.missing_forecast_runners``
    states for the forecast runners: what another gpuwm tree on the import
    path carries is not something this installation can run.
    """

    return importlib.machinery.PathFinder.find_spec(
        f"{__name__}.request", list(__path__)) is not None


def __getattr__(name: str):
    submodule = _EXPORTS.get(name)
    if submodule is None:
        # Also the answer `from gpuwm.ensemble import <submodule>` needs:
        # the import system imports the submodule after this refusal.
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
