"""Sequential member orchestration with resume (EXPERIMENTAL).

Members run one at a time -- one member resident in VRAM -- and the
manifest is republished atomically after every state change, so the
answer to "where did it get to" is always a complete file on disk.

Three rules the engine enforces and the tests pin:

* a member's seed is ``member_seed(base_seed, index)`` and nothing else;
* a member already recorded ``DONE`` is *refused*, not silently redone --
  rerunning it would overwrite the output another cycle may already
  have hashed;
* resume starts at the first member that is not ``DONE`` and leaves the
  finished ones alone.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from gpuwm.ensemble.config import EnsembleConfig
from gpuwm.ensemble.manifest import (
    ENSEMBLE_MANIFEST_NAME, ENSEMBLE_MANIFEST_SCHEMA, first_incomplete_member,
    member_directory_name, member_record, new_ensemble_manifest,
    read_manifest, rollup_status, write_manifest_atomically,
)
from gpuwm.ensemble.seeds import member_seed
from gpuwm.ensemble.wrfout_inventory import WRFOUT_INVENTORY_KEY


class MemberAlreadyCompleteError(RuntimeError):
    """Raised when a completed member would be rerun."""


@dataclass(frozen=True)
class EnsembleResult:
    ens_root: Path
    manifest_path: Path
    ran: tuple[int, ...]
    skipped: tuple[int, ...]
    status: str


def default_member_runner(**kwargs):
    """The real member runner.  Imported lazily: it pulls in the GPU stack."""
    from gpuwm.ensemble.member import run_member

    return run_member(**kwargs)


def member_seeds(cfg: EnsembleConfig) -> list[int]:
    return [member_seed(cfg.base_seed, index)
            for index in range(cfg.n_members)]


def prepare_ensemble(cfg: EnsembleConfig, ens_root: str | Path, *,
                     run_seconds: float | None = None) -> Path:
    """Create ``ens_root``, its member directories, and the manifest.

    Idempotent: an existing manifest for the same base config and member
    count is kept (that is what makes resume possible).  A manifest that
    disagrees with the config is a refusal -- silently reconciling them
    would let a config edit corrupt a half-finished ensemble.

    ``run_seconds`` is bound the first time a run states one and checked
    on every later one.  Members that ran to different horizons are not
    one ensemble, and without this binding a resume that changed the
    forecast length still rolled up ``COMPLETE``.
    """
    root = Path(ens_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ENSEMBLE_MANIFEST_NAME
    seeds = member_seeds(cfg)
    if path.is_file():
        existing = read_manifest(path, schema=ENSEMBLE_MANIFEST_SCHEMA)
        _check_compatible(existing, cfg, seeds, path,
                          run_seconds=run_seconds)
        if not existing.get("run_seconds_bound"):
            existing["run_seconds"] = (None if run_seconds is None
                                       else float(run_seconds))
            existing["run_seconds_bound"] = True
            write_manifest_atomically(path, existing)
    else:
        manifest = new_ensemble_manifest(cfg, seeds=seeds, ens_root=root)
        manifest["run_seconds"] = (None if run_seconds is None
                                   else float(run_seconds))
        manifest["run_seconds_bound"] = True
        write_manifest_atomically(path, manifest)
    for index in range(cfg.n_members):
        (root / member_directory_name(index)).mkdir(parents=True,
                                                    exist_ok=True)
    return path


def _check_compatible(existing, cfg: EnsembleConfig, seeds, path, *,
                      run_seconds: float | None = None) -> None:
    """Everything a resume must agree with.  Fails closed, naming each gap.

    The list is the whole set of facts that decide what a member IS: the
    roster, the two configuration hashes (base experiment AND the
    ensemble overlay), the seed derivation inputs, the perturbation and
    the exact options it was given, and the forecast horizon.  Binding a
    subset is what let a resumed ensemble mix perturbation options and
    forecast lengths and still report ``COMPLETE``.
    """
    mismatches = []
    if existing.get("n_members") != cfg.n_members:
        mismatches.append(
            f"n_members {existing.get('n_members')} != {cfg.n_members}")
    if existing.get("base_config_sha256") != cfg.base_config_sha256:
        mismatches.append(
            f"base config sha256 {existing.get('base_config_sha256')} != "
            f"{cfg.base_config_sha256}")
    if existing.get("base_seed") != cfg.base_seed:
        mismatches.append(
            f"base_seed {existing.get('base_seed')} != {cfg.base_seed}")
    if existing.get("perturbation") != cfg.perturbation:
        mismatches.append(
            f"perturbation {existing.get('perturbation')!r} != "
            f"{cfg.perturbation!r}")
    recorded_options = (
        existing.get("perturbation_options_sha256")
        or (existing.get("ensemble_config") or {})
        .get("perturbation_options_sha256"))
    if recorded_options != cfg.perturbation_options_sha256:
        mismatches.append(
            f"perturbation_options_sha256 {recorded_options} != "
            f"{cfg.perturbation_options_sha256}; the completed members "
            "carry the recorded options and any new one would carry these")
    recorded_source = (
        existing.get("ensemble_source_sha256")
        or (existing.get("ensemble_config") or {}).get("source_sha256"))
    if recorded_source != cfg.source_sha256:
        mismatches.append(
            f"ensemble overlay sha256 {recorded_source} != "
            f"{cfg.source_sha256}")
    recorded = [record.get("seed") for record in existing.get("members", ())]
    if recorded != list(seeds):
        mismatches.append("derived member seeds differ from the recorded ones")
    if existing.get("run_seconds_bound"):
        bound = existing.get("run_seconds")
        wanted = None if run_seconds is None else float(run_seconds)
        if bound != wanted:
            mismatches.append(
                f"run_seconds {bound!r} != {wanted!r}; members that ran to "
                "different horizons are not one ensemble")
    if mismatches:
        raise ValueError(
            f"{path} was written for a different ensemble: "
            + "; ".join(mismatches)
            + ". Point --ens-root at a new directory, or restore the "
              "config that produced this manifest.")


def run_ensemble(cfg: EnsembleConfig, ens_root: str | Path, *,
                 members: Sequence[int] | None = None,
                 run_seconds: float | None = None,
                 resume: bool = True,
                 runner: Callable = default_member_runner,
                 restarts: Mapping[int, object] | None = None,
                 on_event: Callable[[dict], None] | None = None
                 ) -> EnsembleResult:
    """Run the members of ``cfg`` sequentially into ``ens_root``.

    ``restarts`` maps member index to the checkpoint that member starts
    from -- how a cycling run's leg N+1 begins at leg N's analysis rather
    than re-preparing from the base config.  A member with no entry starts
    from the base config as before.  It is a per-member mapping and not a
    single path because the whole point of the analysis is that the
    members differ.
    """
    root = Path(ens_root)
    manifest_path = prepare_ensemble(cfg, root, run_seconds=run_seconds)
    manifest = read_manifest(manifest_path, schema=ENSEMBLE_MANIFEST_SCHEMA)

    if members is None:
        if resume:
            first = first_incomplete_member(manifest)
            start = cfg.n_members if first is None else first
        else:
            start = 0
        requested = list(range(start, cfg.n_members))
    else:
        requested = [int(index) for index in members]
        for index in requested:
            member_record(manifest, index)

    ran: list[int] = []
    skipped: list[int] = []
    for index in requested:
        record = member_record(manifest, index)
        if record.get("status") == "DONE":
            # Skipping a DONE member is *resume*, and only resume: an
            # explicit --member for finished work, or a non-resume run
            # over it, is a refusal.
            if members is not None or not resume:
                raise MemberAlreadyCompleteError(
                    f"member {index} is already DONE in {manifest_path} "
                    f"(final_state_sha256 "
                    f"{record.get('final_state_sha256')}). Rerunning it "
                    "would overwrite output another cycle may already "
                    "have hashed. Delete the member directory and its "
                    "manifest entry deliberately if that is what you "
                    "want.")
            skipped.append(index)
            _emit(on_event, {"event": "member-skipped", "index": index})
            continue

        member_dir = root / record["member_dir"]
        member_dir.mkdir(parents=True, exist_ok=True)
        record["status"] = "RUNNING"
        record["error"] = None
        manifest["status"] = rollup_status(manifest)
        write_manifest_atomically(manifest_path, manifest)
        _emit(on_event, {"event": "member-started", "index": index,
                         "seed": record["seed"]})

        restart = None if restarts is None else restarts.get(index)
        if restart is not None:
            restart = Path(restart)
            if not restart.is_file():
                raise ValueError(
                    f"member {index} was told to restart from {restart}, "
                    "which does not exist. A leg that silently re-prepared "
                    "from the base config instead would be a cycling run "
                    "that quietly stopped cycling.")
        started = time.perf_counter()
        try:
            outcome = runner(
                base_config=cfg.base_config, member_dir=member_dir,
                index=index, seed=int(record["seed"]),
                perturbation=cfg.perturbation,
                perturbation_options=dict(cfg.perturbation_options),
                run_seconds=run_seconds, restart=restart)
        except BaseException as error:
            record["status"] = "FAILED"
            record["wall_seconds"] = time.perf_counter() - started
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            manifest["status"] = rollup_status(manifest)
            write_manifest_atomically(manifest_path, manifest)
            _emit(on_event, {"event": "member-failed", "index": index,
                             "error": str(error)})
            raise

        record["status"] = "DONE"
        # Recorded, not inferred: "this member started from an analysis"
        # is exactly the fact a cycling receipt must not have to be
        # reconstructed from directory layout.
        record["restart_from"] = None if restart is None else str(restart)
        record["final_state_sha256"] = outcome.final_state_sha256
        record["initial_state_sha256"] = outcome.initial_state_sha256
        # Recorded so a hash over a truncated inventory is legible AS a
        # hash over a truncated inventory (see gpuwm.ensemble.state_sha).
        record["final_state_inventory"] = (
            dict(outcome.final_state_inventory)
            if outcome.final_state_inventory else None)
        record["wall_seconds"] = float(outcome.wall_seconds)
        record["sim_seconds"] = float(outcome.sim_seconds)
        record["wrfout_count"] = int(outcome.wrfout_count)
        # The count alone is not an identity: one stale wrfout replacing
        # one real one satisfies it exactly, which is how
        # ``enprod --accept-status`` came to admit bytes it had never
        # verified.  The inventory binds path, domain, valid times, frame
        # indices, size and sha256 per file; a consumer that admits this
        # member past the default statuses checks THAT.
        record[WRFOUT_INVENTORY_KEY] = (
            [dict(entry) for entry in outcome.wrfout_inventory]
            if outcome.wrfout_inventory is not None else None)
        record["perturbation"] = {
            "provenance": cfg.perturbation,
            "options_sha256": cfg.perturbation_options_sha256,
            "applied": dict(outcome.perturbation),
        }
        manifest["status"] = rollup_status(manifest)
        write_manifest_atomically(manifest_path, manifest)
        ran.append(index)
        _emit(on_event, {"event": "member-finished", "index": index,
                         "final_state_sha256": outcome.final_state_sha256,
                         "wall_seconds": float(outcome.wall_seconds)})

    manifest["status"] = rollup_status(manifest)
    write_manifest_atomically(manifest_path, manifest)
    return EnsembleResult(ens_root=root, manifest_path=manifest_path,
                          ran=tuple(ran), skipped=tuple(skipped),
                          status=manifest["status"])


def _emit(on_event, payload) -> None:
    if on_event is not None:
        on_event(payload)
