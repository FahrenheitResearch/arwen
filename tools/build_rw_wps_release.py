#!/usr/bin/env python3
"""Build the complete Linux RW-WPS runtime from one clean source checkout.

The lower-level distribution builder deliberately accepts already-built
artifacts so it can verify each one independently.  This command is the
end-user entry point: it builds the Python wheel and every vendored Rust
decoder/CPU backend offline, then delegates to that verifier and packager.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.build_native_wrf_distribution import build_distribution  # noqa: E402
from gpuwm.native_wrf_distribution import (  # noqa: E402
    BRIDGE_WORKSPACES,
    BUNDLED_BRIDGES,
    CPU_BACKEND_LIBRARY,
    CUDA_KERNEL_SOURCES,
    HRRR_HELPERS,
    PYTHON_DISTRIBUTION,
)


_TOP_LEVEL_EXCLUDES = {
    "cli.py",
    # ``python -m gpuwm`` is the gpuwm console script by another name: its
    # only import is gpuwm.cli, excluded on the line above, so it leaves
    # with it.  The standalone project has no such door.
    "__main__.py",
    # The spectral-numerics seam: the hook the MODEL LOOP calls once per
    # completed slow large step, plus receipt binding into the run
    # capsule.  Pure runtime -- a preprocessing wheel completes no model
    # step, so nothing here is reachable from any staged door -- and it
    # imports gpuwm.spectral_ops, which this wheel does not stage.
    "spectral_seam.py",
    # gpuwm-product front doors (CUDA forecast side): the domain wizard
    # imports the memory preflight + gpuwm.cli, and the downscale front
    # door drives the offline CUDA child.  Neither belongs in the
    # standalone RW-WPS preprocessing wheel.
    "domain_wizard.py",
    # The wizard's prompt session, for the same reason and one more: it
    # reaches gpuwm.domain_wizard and gpuwm.core.preflight, both absent
    # here, so staging it fails this builder's own unresolved-import
    # scan.  A preprocessing wheel has no domain to size.
    "domain_interactive.py",
    # `gpuwm go` drives tools/prepared_single_domain_forecast.py through
    # the GPU forecast.  Its imports happen to resolve against what RW-WPS
    # stages, so the scan would not have caught it -- but the runner it
    # exists to call is not in this wheel, so shipping it would offer a
    # command that cannot run.
    "go_cli.py",
    # Its only importer, go_cli.py, is excluded, as is its dependency
    # gpuwm.supervisor. A preprocessing wheel does not run forecasts.
    "forecast_supervisor.py",
    # `gpuwm warm-kernels` compiles the forecast's GPU kernels by stepping
    # a synthetic domain through gpuwm.core.dycore, moist and physics,
    # none of which this wheel stages; its only importer is gpuwm.cli,
    # excluded above.  Staged, it was the one module reaching for those
    # three, and this builder's unresolved-import scan refused the whole
    # staging.  A preprocessing wheel runs no forecast to warm.
    "warm_kernels.py",
    # The live renderer that draws each frame as a forecast writes it.  Its
    # importers are go_cli.py, first_products.py and runplan.py, all
    # excluded above, and it reaches for gpuwm.first_products, gpuwm.render,
    # gpuwm.render_receipts and gpuwm.go_cli, none of which this wheel
    # stages, so the unresolved-import scan refused the whole staging.  A
    # preprocessing wheel runs no forecast whose frames it could draw.
    "live_products.py",
    # The physics catalog behind `gpuwm physics-catalog` and New forecast's
    # Physics step.  Its importers are gpuwm/cli.py and gpuwm/runplan.py,
    # both excluded, and it reaches for gpuwm.domain_wizard,
    # gpuwm.companion_domains and gpuwm.core.pace, none of which this wheel
    # stages, so the unresolved-import scan refused the whole staging.  A
    # preprocessing wheel picks no physics for a forecast.
    "physics_catalog.py",
    # `gpuwm go`'s own telemetry writer: the append-only events.jsonl a
    # go chain emits per stage.  Its ONLY importer anywhere in gpuwm/ or
    # tools/ is gpuwm/go_cli.py:1674 -- excluded directly above -- and
    # the three modules it reaches inside its functions are
    # gpuwm.progress_log, gpuwm.runplan and gpuwm.first_products, all
    # three already excluded below.  Staging it therefore put a module
    # in the wheel reaching for three deliberately absent ones, and
    # this builder's own unresolved-import scan refused the staging
    # outright: at that point `tools/build_rw_wps_release.py` could not
    # stage, build or publish the standalone wheel at all.  A
    # preprocessing wheel runs no `gpuwm go` chain, so it has no stage
    # walls to record.
    "chain_events.py",
    # The early-render worker `gpuwm go` and run-plan arm to draw the
    # analysis frame while the forecast runs.  It is reached only from a
    # running forecast, its only two referrers are go_cli.py and
    # runplan.py -- both excluded directly above and below -- and it
    # imports gpuwm.go_cli itself, so staging it put a module in the
    # wheel reaching for one deliberately absent from it and this
    # builder's own unresolved-import scan refused the staging.  A
    # preprocessing wheel renders no forecast frames.
    "first_products.py",
    # The two prepared-cache GPU forecast runners.  Their substance
    # moved out of tools/ and into the package so a `pip install gpuwm`
    # can finish a forecast; that makes them top-level modules here, and
    # they are exactly what this preprocessing wheel does not do.  They
    # reach gpuwm.core.model and the whole CUDA side, which RW-WPS does
    # not stage.
    "prepared_single_domain_forecast.py",
    "prepared_domain_tree_forecast.py",
    # Per-model-time-step progress for those two runners: the WRF-shaped
    # `Timing for main:` line, progress.jsonl and the frame-ready
    # markers.  A preprocessing wheel takes no model time steps and
    # renders no history frames, so it has nothing to report; the only
    # importers are the two runners immediately above, both excluded.
    # It also reaches gpuwm.supervisor for the atomic marker write, and
    # supervisor is excluded below -- which is how this surfaced: the
    # staging scan refused on an unresolved internal import rather than
    # shipping a module that would raise at first use.
    "progress_log.py",
    # The machine front door.  It is an envelope over the run routes, so it
    # reaches every one of them -- gpuwm.cli, gpuwm.go_cli, gpuwm.supervisor,
    # gpuwm.domain_wizard, gpuwm.certify.capsule and gpuwm.core.preflight --
    # and all six are deliberately absent here.  A preprocessing wheel has no
    # run to plan, and staging it would offer a command that cannot execute
    # one, which is exactly the reason go_cli.py is excluded above.
    "runplan.py",
    # The run's disk projection and the download and preparation pricing it
    # reads.  Their only importer in the package is gpuwm/runplan.py,
    # excluded directly above (tools/wiki_seed reads them too, and is not
    # staged), and download_budget.request_from_arguments reads a fetch argv
    # through gpuwm.cli's parser, which this wheel does not carry.  Staged,
    # download_budget was a module reaching for a deliberately absent one:
    # this builder's own unresolved-import scan refused the whole staging and
    # the 2.8.0 cut battery went red on it.  Its measured table
    # (gpuwm/data/download-bytes.v1.json) is not staged either, so the module
    # could not have priced anything here.  A preprocessing wheel plans no
    # run, so it has no run disk to price.
    "disk_budget.py",
    "download_budget.py",
    # Direct forecast disk admission reads the same excluded pricing,
    # checkpoint-retention policy and restart reader. No preprocessing
    # entry point integrates a forecast or writes its history output.
    "output_disk.py",
    # `gpuwm speedrun` and the capsule module behind it.  A speedrun
    # times the SHIPPED chain end to end by driving `gpuwm go` as a
    # subprocess -- prepare, forecast, render -- and seals a capsule of
    # the walls; `gpuwm go` is excluded directly above, so the door
    # here would offer a command whose only action is to invoke a
    # command this wheel does not carry.  A preprocessing wheel runs no
    # forecast, so it has no course to run and no clock to seal.
    #
    # It is also how this surfaced: gpuwm/speedrun.py reaches
    # gpuwm.verify.cases._repo_config to find a course's config, and
    # gpuwm.verify is the developer verification package RW-WPS omits
    # entirely -- so the staging scan refused rather than shipping a
    # module that would raise at first use.
    "speedrun.py",
    "speedrun_cli.py",
    # `gpuwm sources` -- the human listing of the source registry.  It is
    # a CLI door and nothing else: its only importer anywhere is
    # gpuwm/cli.py, excluded at the top of this list, and it reaches
    # gpuwm.runplan (excluded above) for the inventory it prints.  A
    # wheel with no CLI has no door to register it on, so staging it put
    # a module in the wheel reaching for a deliberately absent one and
    # this builder's own unresolved-import scan refused the staging.
    "sources_cli.py",
    # These product orchestrators both execute forecasts: multi-run dispatches
    # isolated supervisor workers, while stream extends forecasts through
    # restart checkpoints.  Neither exposes an RW-WPS preprocessing surface,
    # and both import executor modules deliberately absent from this wheel.
    "multi_run.py",
    "stream.py",
    "downscale.py",
    "offline_child.py",
    # Own-geography preparation belongs to the offline forecast child;
    # its parent-history readers are excluded with that route.
    "offline_child_geography.py",
    "offline_child_run.py",
    "offline_child_smoke.py",
    # `gpuwm resume` locates a forecast checkpoint and hands it to the
    # supervisor's `run --restart` dispatch.  It reached this staging
    # only because it is a top-level module: nothing RW-WPS ships
    # imports it (the one importer is the excluded gpuwm.cli), no
    # standalone entry point exposes it, and its own two lookups are the
    # restart validator in gpuwm.supervisor and the header reader in
    # gpuwm.io.restart -- one forbidden here, the other not staged at
    # all.  A preprocessing wheel has no checkpoints to resume from.
    "resume.py",
    # `gpuwm branch` is resume's other half: a NEW run seeded from an
    # existing run's checkpoint (gpuwm.branch).  It reached this staging
    # the same way resume.py did -- it is a top-level module and nothing
    # RW-WPS ships imports it (the one importer is the excluded
    # gpuwm.cli) -- and its own lookups are the forecast executor in
    # gpuwm.core.model and the checkpoint locator in gpuwm.resume, one
    # forbidden here and the other excluded directly above, so staging
    # it fails this builder's own unresolved-import scan on four
    # imports.  A preprocessing wheel has no checkpoints to branch from.
    "branch.py",
    # The saved history a resumed forecast's first pictures read from
    # beside its checkpoint.  Its importers are go_cli.py, live_products.py
    # and first_products.py, all excluded above, and it reaches
    # gpuwm.io.restart (refused below), gpuwm.resume and
    # gpuwm.live_products (both excluded above).  Staged, it was a module
    # reaching for three deliberately absent ones, and this builder's own
    # unresolved-import scan refused the whole staging.  A preprocessing
    # wheel resumes no forecast and draws no pictures.
    "restart_render.py",
    "runtime.py",
    "state_digest.py",
    "supervisor.py",
    # Forecast input runners and terminal job/catalog entry points have no
    # standalone preprocessing command. Their readers remain staged below.
    "metem_forecast.py", "wrfinput_forecast.py",
    "launchpad_api.py", "tui_worker.py",
    # These forecast UI doors size research runs, dispatch remote workers,
    # select runtime products or validate starter run plans. They reach the
    # excluded wizard, supervisor, restart and runplan modules; no RW-WPS
    # entry point or staged preparation module consumes them.
    "remote_cli.py", "remote_worker.py", "research_workspaces.py",
    "starter_template.py", "tui_products.py",
    "case_catalog.py", "case_catalog_import.py",
    # The companion's query/edit/fit doors are registered only by gpuwm.cli;
    # the remote modules transfer and supervise ArWen jobs. Configuration
    # recovery is consumed by those excluded run/catalog doors. None belongs
    # to source_cli's standalone preprocessing argument surface.
    "companion_query.py", "companion_domains.py", "companion_forcing.py",
    # The saved-setups door (`gpuwm companion-setup save|start`) is the same
    # surface one step on: a library of FORECAST configurations a person
    # copies and re-times.  Its only importer is gpuwm/cli.py:374, and its
    # own work is done through gpuwm.companion_domains (the native editor
    # excluded directly above), gpuwm.starter_template and gpuwm.fetch_routes
    # -- so staging it put a module in the wheel reaching for a deliberately
    # absent one and this builder's own unresolved-import scan refused the
    # staging outright.  A preprocessing wheel prepares inputs for a forecast
    # it does not run; it has no forecast setups to save or start from.
    "companion_setups.py",
    "configuration_recovery.py", "remote_artifacts.py",
    "remote_input_transfer.py", "remote_plan.py", "remote_processed.py",
    # Companion forecast configuration and remote rendering/preparation
    # orchestrators belong to the same excluded ArWen UI/worker surface.
    # Standalone source_cli has no entry point or importer for these doors.
    "companion_physics.py", "cyclone_setup.py", "remote_native_plots.py",
    "remote_preparation_v2.py", "remote_processed_cache_v2.py",
    "remote_processed_v2.py",
    # The product renderer publishes supervisor/first-products receipts.
    # Its callers are the excluded ArWen CLI/go/run-plan/remote doors and
    # unstaged DA/cells/verification packages. RW-WPS has no render command;
    # keep the renderer and its receipt owner together in full ArWen, rather
    # than staging a renderer whose result publication cannot be imported.
    "render.py", "render_receipts.py",
    # Analysis-cycle orchestration and cyclone setup are full ArWen entry
    # points. Their shared source facts remain in source_drivability; the
    # standalone source_cli never invokes an analysis or a tracking setup.
    "background_contract.py", "regional_preparation.py",
    "local_da.py", "local_da_fetch.py", "local_da_observations.py",
    "local_da_runtime.py", "cyclone_seed.py", "cyclone_sources.py",
    # The rapid-cycling controller and the nowcast scorer belong to the same
    # analysis-cycle surface as the doors above, and both are reached only
    # from them (`gpuwm/local_da.py` and `gpuwm/local_da_runtime.py`, already
    # excluded here).  Each is unstageable on its own terms as well: the
    # controller reaches gpuwm.ensemble, gpuwm.mcp and gpuwm.supervisor, none
    # of which this wheel stages, and the scorer reaches gpuwm.verify.obs
    # nine times -- the verification package this wheel omits by design.
    # Staging either put a module in the wheel that ImportErrors the moment
    # it is reached, and this builder's own staging scans refused outright.
    "local_da_controller.py", "local_da_score.py",
    # The steep-terrain clock: the acoustic substep rule and the long step a
    # domain's ground and crest-level wind allow.  Both set the time step a
    # forecast integrates with, when the forecast starts, and besides each
    # other their only importers are gpuwm/runtime.py and the two prepared
    # forecast runners, all excluded above.  Both reach gpuwm.core.adaptive_clock, which this
    # wheel does not stage (it reaches the physics cadence in
    # gpuwm.core.physics), so staging them made this builder's own
    # unresolved-import scan refuse the whole staging and the RW-WPS
    # package could not be built.  A preprocessing wheel takes no step.
    "acoustic_adaptation.py", "terrain_clock.py",
}
_CORE_MODULES = {
    "__init__.py",
    "constants.py",
    "diagnostics.py",
    # The C-library transcriptions the CPU paths hash through (A126):
    # diagnostics.py imports noahmp_libm.powf_array, and
    # gpuwm/static/lambert.py and projection.py import host_libm.  Both
    # need only math, struct and numpy.
    "noahmp_libm.py",
    "host_libm.py",
    # The portable transcendentals the host preparation takes (A130):
    # real.py, nest_init.py, grid.py, nest_interp.py and the other
    # preparation paths import it.  numpy, ctypes and the stdlib at
    # module scope; the CPU preprocessing library it calls is staged.
    "portable_math.py",
    # The per-card cache keys the staged noah.py and milbrandt2_constants.py
    # take for their device tables (lane 281-multigpu-2card): functools and
    # threading at module scope, CuPy only inside the call.
    "device_cache.py",
    # The [devices] options table config.py and experiment.py parse beside
    # [tiles] (lane 281-multigpu-2card): dataclasses and typing at module
    # scope; the rank machinery it names is imported only by the forecast.
    "devices.py",
    "track_boundary.py",  # NumPy-only boundary diagnostic used by storm_tracking.
    # Prepared/wrfinput initialization imports the lazy tile door, and the
    # CPU fit estimator reads the import-free mosaic array inventory.
    "noah_mosaic.py", "noah_mosaic_door.py",
    "grid.py",
    "landuse.py",
    # landuse.py reads the LCZ categories through urban_tables under the
    # urban land-use legend; urban_tables imports only noahmp_libm and noah,
    # both staged here.
    "urban_tables.py",
    "microphysics_transition.py",
    # The Milbrandt-Yau constant table, reached by microphysics_transition
    # above when a mixed nest edge enters mp=9 and the kernel needs the
    # scheme's own ck vector.  Pure numpy, no kernels and no scheme: the
    # forecast module gpuwm/core/milbrandt2.py stays out of this wheel and
    # reads its device cache from here.
    "milbrandt2_constants.py",
    # The three mp=28 scalars the same mixed-edge module seeds a
    # Thompson-aerosol child from (NT_C and the two aerosol floors).
    # Import-free constants: the table contract that re-exports them
    # (thompson_aerosol_contract.py) stays out, as recorded below.
    "thompson_aerosol_constants.py",
    # The card probe `--preprocess-backend auto` reads before it keeps a
    # certified card: free memory, utilization and the CUDA error of a
    # card too full to open, from a short-lived subprocess.  Without it
    # this package had no load reading and prepared on a card another
    # program held busy or nearly full, where gpuwm moves that
    # preparation to the CPU.  A leaf: stdlib at module scope plus a
    # function-local gpuwm.local_gpu (staged); the forecast memory
    # preflight that re-exports it stays out.
    "device_probe.py",
    # The inventories a CUDA preparation is priced from before its first
    # device allocation: the exact DomainState allocation list, one
    # boundary interval's side tables and the card's CUDA context.
    # gpuwm/ingest/preparation_price.py (staged) reads all three for every
    # GFS, ERA5 and mapped preparation; while they lived in the forecast
    # memory preflight, which stays out, this builder refused the staging
    # on three unresolved imports and a staged wheel raised
    # ModuleNotFoundError at the price.  Module scope is stdlib plus
    # gpuwm.config (staged), gpuwm.boundary_fields (staged) is
    # function-local, and preflight re-exports every name.  No CuPy, no
    # forecast executor.
    "device_inventory.py",
    # The resident admission the prepared-cache restore, the wrfinput
    # restore and init_at_rest (all staged) take before their DomainState
    # constructor: the exact state inventory and the boundary tables
    # against the card's free memory, refused by name.  Module scope is
    # stdlib plus gpuwm.ingest.memory_refusal (staged); CuPy and
    # device_inventory are function-local, and the one forecast import
    # (the loader slab's estimate) is recorded in _OPTIONAL_STAGED_IMPORTS.
    "resident_admission.py",
    # The sea-level pressure reduction and its nine-point smoother,
    # reached by storm_tracking.py below when a follow block tracks
    # `field = 'pressure'`.  Same shape as sase_limits.py further down:
    # the arithmetic lives in gpuwm/core rather than in its historical
    # home under gpuwm/verify because that tree is developer
    # verification and this distribution omits it -- staging
    # storm_tracking.py while it reached across that boundary is what
    # the verification-import scan below refused.  Leaf module, numpy at
    # module scope and nothing internal, so it carries no CuPy and no
    # forecast executor.
    "mslp.py",
    "nest_interp.py",
    # Shared config/initialization contracts, without memory or GPU executors.
    "nest_fields.py", "ozone_contract.py", "inflow_perturbation.py",
    # The config reader validates attribute-following fields and source
    # domains through this module; storm_tracking also imports its grammar.
    # Its imports are NumPy and the already-staged streaming options module.
    # Its device reduction is function-local and is never called by config
    # validation, so importing it requires no CuPy or forecast executor.
    "attribute_tracking.py",
    # Config VALIDATION for the two storm-following blocks, which is
    # preprocessing work: `gpuwm/experiment.py` is staged, and it calls
    # `build_follow_config` for `[relocation.follow]` and
    # `build_spawn_config` for a `[[domain]]` `spawn` table while LOADING a
    # config, long before any state exists.  Excluding them would make this
    # wheel refuse to read a perfectly valid storm-following TOML.  Both are
    # stdlib + numpy (nest_spawn imports only storm_tracking beyond that),
    # so they carry no CuPy and no forecast executor.
    "storm_tracking.py",
    "nest_spawn.py",
    # The SAME config-validation reason, for the `[relocation.track]`
    # table: `gpuwm/experiment.py` calls `build_track_config` while
    # LOADING a config, and reads `POSITION_ONLY_FIELDS` and
    # `SURFACE_LEVEL` from here to refuse a track block whose columns the
    # configured tracker could never fill.  Those three imports sit at
    # function scope inside the loader, so leaving the module behind made
    # this wheel refuse to read a valid storm-following TOML -- and the
    # staging scan below refused the wheel outright rather than shipping
    # a half-readable one.  Module scope is stdlib plus numpy, and its one
    # internal lookup is `gpuwm.core.storm_tracking`, staged directly
    # above for the same reason.  No CuPy, no forecast executor: the CSV
    # this module writes is driven from `gpuwm/runtime.py`, excluded.
    "storm_track_writer.py",
    # The SAME reason as the two directly above, for the three lifecycle
    # tables that sit beside `spawn` on a `[[domain]]`: `gpuwm/experiment.py`
    # calls `build_retire_config` for `retire`, `build_rearm_config` for
    # `rearm` and `build_domain_follow_config` for a per-domain `follow`
    # while LOADING a config, so a wheel without this module refuses to read
    # a valid storm-following TOML -- and, because those three imports sit
    # at function scope inside the loader, the staging scan below refused
    # the wheel outright rather than shipping a half-readable one.
    # Module scope is math/dataclasses plus numpy, and its only two internal
    # lookups are gpuwm.core.uh_diag and gpuwm.core.storm_tracking, both
    # already staged for the same config-validation reason.  No CuPy, no
    # forecast executor: the tree surgery a retirement decision feeds is
    # gpuwm/runtime.py's, and that module is excluded above.
    "nest_lifecycle.py",
    # Reached by ALL THREE above -- storm_tracking, nest_spawn and
    # nest_lifecycle -- for the tracking-window slot
    # NAMES: the windows are one source of truth rather than string
    # literals copied into each consumer, so validating a spawn, retire or
    # follow block imports the module that owns them.  It costs nothing this
    # wheel refuses -- module scope is numpy plus core/constants.py
    # (already staged), and the only CUDA reference is a function-local
    # get_kernel inside the device fold, which no config-validation path
    # ever calls.
    "uh_diag.py",
    # The same config-validation reason again: `gpuwm/experiment.py` and
    # nest_lifecycle call `validate_reach_speed` for a follow block's
    # `reach_speed_m_s` while LOADING a config, and the statics corridor
    # (gpuwm/static/corridor.py, staged) sizes a moving nest from its
    # reach.  Stdlib only: no CuPy, no forecast executor.
    "nest_reach.py",
    # The [tiles] option surface, and NOT the streamed transport.  Module
    # scope here is dataclasses plus typing: every tilestream import in the
    # file is function-local, so staging it carries no forecast executor
    # and no CuPy.  It is not optional either -- gpuwm/experiment.py holds
    # `tiles: StreamingOptions = OFF` as a field default, so the class
    # cannot be defined without it, and a wheel that stages experiment.py
    # and not this one fails on `import gpuwm.era5_direct`.  That is how
    # the tilestream port broke this wheel: the scan below read
    # `from gpuwm.core import streaming` as an import of `gpuwm.core`
    # alone, which IS staged, so nothing refused the staging.  It now
    # checks gpuwm.core.streaming too (A129).
    "streaming.py",
    "nssl2_contract.py",
    "noah.py",
    # state.py allocates the SASE prognostic and reads its realizability
    # floor from here.  A dependency-free constants module, which is why
    # those two values live in gpuwm/core rather than in the closure's
    # authority module under gpuwm/verify -- that tree is developer
    # verification and this distribution omits it.
    # The benchmark's mp=50 table-authority arm validates the P3 lookup
    # table at profile binding -- the same stage and the same reason as
    # the Thompson arm beside it.  Leaf module: stdlib + numpy at module
    # scope, no CuPy, no forecast executor.
    "p3_tables.py",
    "sase_limits.py",
    "state.py",
    "thompson_contract.py",
    # The mp=28 real-data cold start closes cloud droplet, rain and ice
    # number over the analyzed mass through the scheme's own entry block
    # (gpuwm/ingest/real.py, staged above), and that block is the host
    # mirror gpuwm/core/thompson_entry.py: numpy at module scope, its
    # powers from noahmp_libm.py and host_libm.py (staged above), its
    # gamma moments from thompson_aerosol_contract.py, whose only other
    # internal reaches are thompson_contract.py (staged) and, function-
    # locally, gpuwm.physics_compat (staged); the contract's exp/log are
    # correctly_rounded_libm.py, 75 lines of the decimal module.  Three
    # leaf modules, no CuPy and no forecast executor, so a preparation-only
    # install starts an mp=28 case with the same numbers the full one does.
    "thompson_entry.py",
    "thompson_aerosol_contract.py",
    "correctly_rounded_libm.py",
    # state.py allocates WDM6's three number prognostics by NAME and reads
    # that tuple from `wdm6_constants.WDM6_NUMBER_SPECIES` rather than
    # copying the spelling, so an mp_physics=16 config cannot be loaded
    # without it.  Same shape as sase_limits.py above: a leaf constants
    # module, stdlib-only at module scope (dataclasses/functools/math), no
    # CuPy and no forecast executor.  It reaches one sibling --
    # `wsm6_constants.py`, staged just below -- because WDM6's cold half IS
    # WSM6's and the coefficients are imported rather than duplicated.
    # `gpuwm/core/microphysics.py` and `gpuwm/core/wdm6.py` import it too,
    # but both are already excluded from this wheel, so state.py is the
    # only module that forces it here.
    "wdm6_constants.py",
    # Reached only by wdm6_constants.py, for the shared rimed-ice
    # constants and the Weierstrass Gamma.  Also stdlib-only at module
    # scope; it is staged as that module's dependency, not on its own
    # account -- nothing else in this wheel names it.
    "wsm6_constants.py",
}
#: `nest_spawn_init.py` and `relocation_init.py` are forecast-time child
#: initializers, not preprocessing: one builds a nest that is born mid-run,
#: the other rebuilds one that just moved, and both need a LIVE parent state
#: to do it.  They reach `gpuwm.core.nest_relocation`, `gpuwm.core.health`
#: and `gpuwm.ensemble.state_sha`, none of which this wheel stages.  Their
#: only importers are `gpuwm/core/spawn_runner.py` (not in _CORE_MODULES)
#: and `gpuwm/runtime.py` (already excluded above), so leaving them behind
#: strands nothing.
_INGEST_EXCLUDES = {"preflight.py", "nest_spawn_init.py",
                    "relocation_init.py", "relocation_continuation.py",
                    "case_store.py"}
#: `gpuwm/obs/sources.py` is the seam between the ingest lane and the scoring
#: lane: it builds the scorer's dataclasses and reaches
#: `gpuwm.verify.obs.contracts` to do it.  RW-WPS ships no verification
#: package, and that boundary is absolute rather than exemptible, so the one
#: module that crosses it stays behind.  Nothing else in `gpuwm/obs` imports
#: it -- `__init__` pulls radar_grid, superob, sweeps and target_grid -- so
#: the radar front door `gpuwm doctor` checks for is unaffected.
# GOES window acquisition publishes cycle/ensemble manifests. The lower-level
# observation decoders remain available to standalone preprocessing.
_OBS_EXCLUDES = {"sources.py", "goes_window.py"}
#: The only two files of ``gpuwm/io`` this wheel stages -- named
#: individually rather than by excluding the rest of the package,
#: because the package is the forecast executor's output side and the
#: allowed set is the small half.
#:
#: ``gpuwm/doctor.py`` imports ``gpuwm.io.nc_writer_bridge`` to report
#: on the NetCDF writer, and that check RETURNS A BLOCKING ``missing``
#: Check on ImportError.  So the two ways of leaving gpuwm/io out
#: entirely both break something concrete: not staging it at all fails
#: this builder's unresolved-import scan, and allowlisting the import
#: instead ships a wheel whose `rw-wps doctor` always reports the
#: NetCDF writer missing -- on a wheel that BUNDLES the netcdf_writer
#: cdylib as one of its artifacts, and whose own suite states the
#: doctor belongs here precisely because a preprocessing install is
#: the one that needs to be told which bridge is missing.
#:
#: Staging the whole subpackage is what does not work: measured, it
#: cascades to 26 fresh unresolved imports, because ``io/restart.py``
#: reaches gpuwm.core.model/physics/microphysics/kf and
#: gpuwm.supervisor, and ``io/wrfout.py`` reaches gpuwm.runtime and the
#: physics runtimes -- the forecast executor this wheel exists not to
#: carry.  ``nc_writer_bridge.py`` is the seam that does not: module
#: scope is ctypes/os/pathlib/typing plus numpy, and its two internal
#: lookups (gpuwm.bridges, gpuwm.rustwx) are function-local and both
#: already staged.  ``__init__.py`` is empty and comes along only so
#: ``gpuwm.io`` is a package in the wheel.
#:
#: ``classic_tape.py`` is the third, and it is here for the same reason
#: and by the same measurement.  ``gpuwm/wrf_direct.py`` -- the
#: wrfinput/wrfbdy writer, which IS preprocessing and which this wheel
#: therefore stages -- imports ``ClassicTape`` at module scope, so
#: leaving the module behind fails this builder's own unresolved-import
#: scan and the wheel cannot be staged at all.  It does not cascade:
#: module scope is pathlib plus numpy, and its one internal lookup
#: (``gpuwm.io.nc_writer_bridge``) is function-local and already staged.
#: It carries no forecast executor -- it is the shared classic-NetCDF
#: tape both NetCDF products write through, nothing more.
#: The gpuwm.io modules this wheel stages.  An ALLOWLIST, and the
#: comment on ``_FORBIDDEN_STAGED_FILES`` below says why that is a
#: hazard: gpuwm/io is partially staged, so "add one more name" is a
#: mistake somebody can make.  The bar for a name here is that the
#: module is DATA or GRAMMAR -- no CuPy, no netCDF4, no runtime, no
#: forecast executor -- and that something the wheel actually does
#: needs it.
#:
#: ``wrf_output_schema`` and ``history_selection`` joined on 2026-08-20
#: and both clear that bar: the schema module is a transcribed WRF
#: Registry table importing nothing but the standard library, and
#: history_selection is the ``[output]`` block's grammar, importing only
#: it and gpuwm.explain.  What needs them is the CONFIG READER --
#: ``gpuwm.config.load_history_selection`` resolves ``[output]`` at the
#: one load every front door shares, and ``gpuwm.render`` reads the same
#: module's product-input table at module scope.  Without them staged, a
#: preprocessing install could not read a config carrying an ``[output]``
#: block at all.  Neither reaches ``gpuwm.io.wrfout``: the Registry
#: metadata table the vocabulary is built from moved INTO
#: wrf_output_schema the same day, precisely so that reading a config
#: does not import the forecast output writer (and gpuwm.supervisor and
#: netCDF4 behind it).  tests/test_history_selection.py pins that
#: boundary against the artifact.
#:
#: ``classic_product`` is the whole-file classic NetCDF writer the staged
#: gpuwm.obs product writers open by default
#: (``gpuwm.obs.grid_product.open_obs_grid_product``), so
#: ``gpuwm.obs.write_radar_grid``, which the staged ``gpuwm.obs`` exports,
#: needs it.  Missing, that call raised ImportError in this package.
#: Module scope is pathlib, typing and numpy, and its one internal lookup
#: (``gpuwm.io.nc_writer_bridge``) is function-local and staged.
_IO_MODULES = {"__init__.py", "classic_product.py", "classic_tape.py",
               "history_selection.py", "nc_writer_bridge.py",
               "wrf_output_schema.py"}
_ROOT_DATA = {
    "native_wrf_support_v1.json",
    "physics_registry_v2.json",
    "wrf_direct_v461_contract.json",
}
_TOOL_FILES = {"__init__.py", *HRRR_HELPERS}
_FORBIDDEN_STAGED_FILES = {
    f"gpuwm/{name}" for name in _TOP_LEVEL_EXCLUDES
} | {
    "gpuwm/core/model.py",
    "gpuwm/core/dycore.py",
    "gpuwm/core/physics.py",
    # The forecast executor's OUTPUT side, named here for the same
    # belt-and-braces reason as the three above.  `_IO_MODULES` makes
    # gpuwm/io a partially-staged package, so "add one more name to the
    # allowlist" is now a mistake somebody can make, and these two are
    # what it must not reach: measured, staging them cascades to 26
    # fresh unresolved imports (restart.py reaches gpuwm.core.model /
    # physics / microphysics / kf and gpuwm.supervisor, wrfout.py
    # reaches gpuwm.runtime and the physics runtimes).
    #
    # The unresolved-import scan below would catch that TODAY, so this
    # is redundant today -- deliberately.  The scan only objects while
    # those imports stay at module scope; the tilestream port already
    # broke this wheel exactly once by making the scan blind (see
    # streaming.py in _CORE_MODULES), and a later edit that made
    # wrfout.py's executor imports function-local would ship the
    # forecast output side of a preprocessing wheel with nothing
    # refusing.  This line refuses on the FILE, which no import
    # rearrangement can talk out of.
    "gpuwm/io/restart.py",
    "gpuwm/io/wrfout.py",
}

_OPTIONAL_STAGED_IMPORTS = {
    ("gpuwm/config.py", "gpuwm.offline_child_geography"):
        "the [static] parser runs only when load_config admits child_static, "
        "which only the excluded offline child route requests; ordinary "
        "RunConfig loading refuses [static] before reaching this import",
    ("gpuwm/core/streaming.py", "gpuwm.core.preflight"):
        "forecast tree admission and execution estimates; standalone preparation "
        "only reads StreamingOptions and does not call these planners",
    ("gpuwm/stage_cli.py", "gpuwm.prepared_single_domain_forecast"):
        "the forecast runner, imported inside _schema_index (behind "
        "resolve_bundle and resolve_head_bundle), streaming_flags, sim_main "
        "and register_cli.  This package's one route into stage_cli is "
        "gpuwm.prep_output's handoff after a finished preparation, which asks "
        "missing_forecast_runners() first and, with both runners absent "
        "here, prints the preparation-only line and resolves no bundle",
    ("gpuwm/stage_cli.py", "gpuwm.prepared_domain_tree_forecast"):
        "the tree forecast runner, imported inside _schema_index and "
        "sim_main.  Same route and same reason as "
        "gpuwm.prepared_single_domain_forecast directly above: the "
        "preparation handoff asks missing_forecast_runners() before it "
        "resolves a bundle, so a finished preparation here imports neither",
    ("gpuwm/core/streaming.py", "gpuwm.core.pace"):
        "the step-cost sentence of an auto [tiles] refusal "
        "(_refused_tiling_clause), reached only from decide() after "
        "tilestream.autoplan refused a tiling.  decide imports tilestream "
        "first, which this package does not carry, and its one staged "
        "caller prices a downscaled forecast child "
        "(downscale_pricing.price_child).  The clause also catches a failed "
        "import and states the refusal without the cost",
    ("gpuwm/core/streaming.py", "gpuwm.core.adaptive_clock"):
        "adaptive forecast tile planning/step execution; StreamingOptions "
        "and config validation reach none of these function-local imports",
    ("gpuwm/core/streaming.py", "gpuwm.io.restart"):
        "live forecast tile builder inventories restart tracker slots; "
        "standalone preparation constructs no tile stepper",
    ("gpuwm/core/streaming.py", "gpuwm.core.streamed_relocation"):
        "forecast-only replacement/adoption of a child store after a move",
    ("gpuwm/core/streaming.py", "gpuwm.core.physics_step_control"):
        "per-step physics cadence in an executing tile; no config path calls it",
    ("gpuwm/core/streaming.py", "gpuwm.core.streamed_state"):
        "running-state publication into a forecast tile store",
    ("gpuwm/core/streaming.py", "gpuwm.core.nest_relocation"):
        "the relocation host-snapshot term of the tree admission, priced "
        "only when a forecast tree carries a moving nest; standalone "
        "preparation decides no tree and moves no nest",
    ("gpuwm/ingest/boundary_stream.py", "gpuwm.runplan"):
        "the interrupt test in _is_stop, reached only from run_chained, "
        "whose callers (gpuwm/go_cli.py and gpuwm/runplan.py) run a "
        "forecast and are not staged; standalone preparation never "
        "chains, so it never calls run_chained",
    ("gpuwm/hrrr_route_inputs.py", "gpuwm.companion_domains"):
        "the doors' WPS namelist renderer (candidate_wps_text) inside "
        "run_route_inputs, reached only when a native hrrr configuration "
        "has no namelist.wps beside it.  run_route_inputs writes the set "
        "a run of `gpuwm go` or `gpuwm run-plan` hands the HRRR chain, "
        "and its only callers are gpuwm/runplan.py's plan resolution and "
        "HRRR chain, excluded above; this wheel plans and chains no run",
    ("gpuwm/ingest/boundary_stream.py", "gpuwm.core.preflight"):
        "the forecast's memory admission (chained_admission) and the host "
        "RAM reader (host_admission), both reached only by a "
        "PreparedTreeWriter that chains, i.e. publishes its head for a "
        "forecast to start on while boundaries are still built.  Its "
        "constructor declines chaining when forecast_installed() is false, "
        "as it is here without gpuwm.core.preflight and gpuwm.core.model, "
        "so an era5, gfs or mapped preparation publishes at its seal and "
        "neither import runs",
    ("gpuwm/ingest/boundary_stream.py", "gpuwm.core.urban_state"):
        "the BEM workspace count reader in prepared_head_urban_columns; "
        "forecast_installed() is checked before this import, so a "
        "preparation-only installation reads no forecast workspace count "
        "and publishes at the seal",
    ("gpuwm/ingest/boundary_stream.py", "gpuwm.prepared_domain_tree_forecast"):
        "the prepared tree's land-cover count reader, reused for chained "
        "head admission after forecast_installed() confirms this package "
        "can forecast; standalone preparation has no forecast executor "
        "and returns before importing this reader",
    ("gpuwm/downscale_pricing.py", "gpuwm.core.preflight"):
        "the downscaled child's memory admission (estimate_experiment) "
        "inside price_child, reached only by the downscale door's plan "
        "review and the offline child runner, both forecast routes; the "
        "preparation wheel prices no child",
    ("gpuwm/core/streaming.py", "gpuwm.core.cam_ozone"):
        "live tile physics initialization; pure ozone config dependencies "
        "are staged separately in ozone_contract",
    ("gpuwm/stage_reuse.py", "gpuwm.prepared_single_domain_forecast"):
        "claim_run_output delegates the output-directory refusal to the "
        "forecast runner's own claim_output_directory rather than "
        "restating it; the import is function-local and only a run door "
        "claims an output directory.  This wheel prepares inputs and "
        "claims none",
    ("gpuwm/metem_door.py", "gpuwm.core.preflight"):
        "metgrid_memory_admission is called only by metem_forecast's GPU "
        "run door; metadata and analyzed-field validation use staged contracts",
    ("gpuwm/ingest/hrrr_physics.py", "gpuwm.core.radiation_composition"):
        "GPU physics initialization after the standalone --prepare-only return, "
        "on the same boundary as this module's existing core.physics import",
    ("gpuwm/ingest/wrfinput.py", "gpuwm.core.physics"):
        "initialize_wrfinput_physics imports CuPy first and initializes an "
        "executing forecast; file reading and host state restoration do not call it",
    ("gpuwm/experiment.py", "gpuwm.spectral_ops.config"):
        "the [spectral_numerics] TABLE PARSER, imported function-locally "
        "and only when a config actually carries the table.  The "
        "subsystem is model runtime and this wheel stages none of it; "
        "experiment.py guards the import and refuses such a config by "
        "name (install the full gpuwm distribution to run spectral "
        "numerics), so a preparation-only install parses every config "
        "it can act on and refuses the one table it cannot.",
    ("gpuwm/physics_menu.py", "gpuwm.domain_wizard"):
        "the WIZARD'S PROSE, and only that: physics_summary() inside "
        "profile_facts() and _radiation_words() inside "
        "nocturnal_remedy().  Their only callers anywhere are "
        "gpuwm/runplan.py (`gpuwm run-plan --physics-profiles`) and "
        "gpuwm/domain_wizard.py itself, and both modules are excluded "
        "above -- a preprocessing wheel plans no run and builds no "
        "domain.  The route a preparation-only install DOES reach this "
        "module on is physics_compat.nocturnal_radiation_refusal() -> "
        "universally_admissible_profile(), which runs at the one "
        "experiment load every front door shares; that path reads "
        "gpuwm.physics_compat and gpuwm.source_adapters, both staged, "
        "and touches neither name above.  It reached the wizard until "
        "2026-08-20, when the suite list, the declared default and "
        "profile_route_blocker moved to gpuwm.physics_compat for "
        "exactly this reason: a table the whole product reads must not "
        "sit behind a door this wheel does not have",
    ("gpuwm/core/kernels/__init__.py", "gpuwm.certify.kernel_manifest"):
        "certification kernel-manifest recording, reached only after the "
        "CuPy import inside the loader; RW-WPS stages no forecast executor "
        "and compiles no CUDA module",
    ("gpuwm/core/kernels/__init__.py", "gpuwm.core.noahmp_kernel_sources"):
        "the one Noah-MP compile site, reached only after the CuPy import "
        "inside the loader and only for a module whose name starts with "
        "noahmp_; RW-WPS stages no forecast executor and compiles no CUDA "
        "module, so the loader never takes that branch here",
    ("gpuwm/core/nest_interp.py", "gpuwm.certify.kernel_manifest"):
        "certification kernel-manifest recording, reached only after the "
        "CuPy import inside the loader; RW-WPS stages no forecast executor "
        "and compiles no CUDA module",
    ("gpuwm/core/noah_mosaic.py", "gpuwm.certify.kernel_manifest"):
        "certification kernel-manifest recording inside the Noah mosaic "
        "tile loop's own compile (_mosaic_module, _mosaic_ucm_module), "
        "reached only after the CuPy import when a forecast launches the "
        "loop; RW-WPS stages no forecast executor and compiles no CUDA "
        "module.  The module is staged for the doors' tile initialization "
        "and the fit estimator's import-free array inventory",
    ("gpuwm/core/noah_mosaic.py", "gpuwm.core.urban_ucm"):
        "the urban canopy's packed tables and state pointers, imported "
        "function-locally in launch_noah_mosaic's urban arm, which only a "
        "forecast's surface step calls; a preparation-only install never "
        "launches the tile loop",
    ("gpuwm/physics_compat.py", "gpuwm.core.ruc_contract"):
        "RUC'S SOIL GEOMETRY COUNTS, and only those.  The import is "
        "function-local inside pending_wrf_physics_components' "
        "sf_surface_physics == 3 arm (physics_compat.py, the deferred "
        "block), so importing this module in a preprocessing wheel does "
        "not touch it.  It is reached only when a caller asks whether a "
        "RUC forecast column is admissible -- a forecast-side question a "
        "preparation-only install has no arm to answer, and one that "
        "gpuwm.config defers for exactly the same reason.  Staging "
        "gpuwm/core/ruc_contract.py instead would pull "
        "gpuwm.core.noahmp_mynn_contract and gpuwm.ingest.ruc_soil in "
        "behind it for two integer tuples.",
    ("gpuwm/core/state.py", "gpuwm.core.preflight"):
        "CUDA forecast scratch-preflight path; RW-WPS constructs host state",
    ("gpuwm/core/resident_admission.py", "gpuwm.core.preflight"):
        "the loader slab's device estimate (estimate_domain), reached only "
        "by store_from_prepared_cache building a forecast's pinned host "
        "store; standalone preparation builds no store, and the constructor "
        "floor it does reach reads only the staged device_inventory",
    ("gpuwm/core/uh_diag.py", "gpuwm.core.streaming"):
        "the streamed view of the tracking accumulators, reached only from "
        "_zero_domain_slot -- i.e. only when a run RESETS a running max "
        "after publishing a history frame. uh_diag is staged for the slot "
        "NAMES that [relocation.follow] and [[domain]].spawn validation "
        "read while LOADING a config, and no config-validation path calls "
        "the reset. Kept as an OPTIONAL entry after gpuwm.core.streaming "
        "joined _CORE_MODULES, because what it records is that uh_diag "
        "does not need the module at import time and the record is worth "
        "more than the redundancy: the module is staged for the option "
        "surface experiment.py cannot define itself without, and if that "
        "reason ever goes away this line says uh_diag does not supply a "
        "second one",
    ("gpuwm/core/streaming.py", "gpuwm.core.dycore"):
        "the tile sweep's stepper and its stability record, imported inside "
        "the four functions that run a tile: attach, step, and the health "
        "fold over the store. gpuwm.core.streaming is staged for the "
        "[tiles] option surface alone -- the dataclass experiment.py holds "
        "as a field default -- and every route that reaches these imports "
        "is the forecast executor this wheel omits",
    ("gpuwm/core/streaming.py", "gpuwm.core.refl"):
        "the reflectivity stash the streamed history writer consumes, "
        "imported inside the publish path. Same reason as gpuwm.core.dycore "
        "directly above: no config-loading path reaches it",
    ("gpuwm/core/streaming.py", "gpuwm.core.nest_stream"):
        "the NEST streamed transport, imported inside three functions that "
        "each require a forecast: prepared_domain_builder.build wires the "
        "tile hook only for a domain whose decision STREAMS, _tree_reservations "
        "prices a live tree's coupling corridors against a VRAM budget, and "
        "steppers_for_tree hands its result to "
        "gpuwm.core.model.execute_experiment -- a module _FORBIDDEN_STAGED_FILES "
        "refuses outright. This is the boundary the _CORE_MODULES entry for "
        "streaming.py already draws in its own words: that file is staged for "
        "the [tiles] OPTION SURFACE and NOT the streamed transport. Staging "
        "nest_stream would cross it, and would pay for the crossing twice -- "
        "the module's own nested imports reach gpuwm.core.preflight, which "
        "this wheel deliberately omits, so admitting it would need a second "
        "exception to cover the first. Nothing staged here needs a NAME from "
        "it: experiment.py cannot define StreamingOptions without streaming.py, "
        "but no class definition and no config-validation path reaches "
        "nest_stream, and its only other referrer, gpuwm/core/nest.py, is not "
        "staged at all",
    ("gpuwm/core/streaming.py", "gpuwm.core.prepared_tile_memory"):
        "prepared forecast VRAM pricing, imported function-locally inside "
        "radiation_footprint after tilestream.autoplan. RW-WPS stages "
        "streaming.py for StreamingOptions and config validation; no "
        "standalone preprocessing path calls the forecast footprint planner",
    ("gpuwm/core/streaming.py", "gpuwm.state_digest"):
        "the canonical end-of-run digest taken from the STORE, imported "
        "inside StreamedDomain.canonical_digest. gpuwm/state_digest.py is on "
        "_TOP_LEVEL_EXCLUDES above by deliberate choice, and what reaches it "
        "here is evidence a run records after it has integrated: the method "
        "refuses outright on a domain carrying no scalars. A preprocessing "
        "wheel steps nothing and so digests nothing",
    ("gpuwm/hrrr_prepared_bundle.py",
     "gpuwm.prepared_single_domain_forecast"):
        "physics-selection helpers on the opt-in portable-bundle branch. "
        "publish_hrrr_prepared_bundle runs only when prepare_hrrr_wrf is "
        "given --wps-namelist, and the three documents it writes exist "
        "for a config-driven FORECAST stage to bind -- which is exactly "
        "what this preprocessing wheel does not ship. The certified "
        "native preparation above that branch never reaches it",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.core.clock"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.core.dycore"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.core.health"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.core.model"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.core.refl"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.io.wrfout"):
        "forecast-only output branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.state_digest"):
        "forecast-only evidence branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.runtime"):
        "forecast-only physics setup after the public --prepare-only "
        "return: declared_constant_glw supplies initialize_hrrr_physics "
        "with the declared downward longwave, and nothing on the "
        "--prepare-only path calls it. The tool itself cannot be dropped "
        "instead -- it is a member of native_wrf_distribution.HRRR_HELPERS, "
        "so the sealed-runtime receipt refuses an installation that is "
        "missing it",
    ("gpuwm/ingest/hrrr_physics.py", "gpuwm.core.physics"):
        "forecast-only physics setup after the public --prepare-only return",
    ("gpuwm/source_availability.py", "gpuwm.domain_wizard"):
        "availability() prices the download a run of a given length makes with "
        "the wizard's fetch_window; it answers the web page's calendar and the "
        "case catalog, neither staged here. The module stays because "
        "gpuwm/source_adapters.py reads ArchiveWindow from it at import",
    ("gpuwm/static/corridor.py", "gpuwm.core.nest_relocation"):
        "the older-corridor check inside load_child_statics_corridor, which "
        "compares a corridor sealed under an earlier build contract with "
        "the tree's own child statics at the prepared placement. Its only "
        "caller is the prepared-tree forecast runner "
        "(gpuwm/prepared_domain_tree_forecast.py, excluded above) when it "
        "loads a corridor for a moving nest; preparation seals corridors "
        "under the current contract and loads none. The module stays "
        "because gpuwm/source_hierarchy.py and "
        "gpuwm/hrrr_hierarchy_direct.py build and seal corridors through "
        "it at module scope",
    ("gpuwm/static/corridor.py", "gpuwm.ingest.relocation_init"):
        "the overlap rule the same older-corridor check reads, beside "
        "gpuwm.core.nest_relocation directly above and on the same "
        "forecast-only route; relocation_init.py is a forecast-time child "
        "initializer and stays out with _INGEST_EXCLUDES",
    ("gpuwm/ingest/init_perturbation.py", "gpuwm.core.rrtmgp"):
        "radiation_temperature_ceiling reads the RTE+RRTMGP gas-table span "
        "when a forecast runner builds the initial-state perturbation "
        "(gpuwm/runtime.py, the prepared-tree runner and a nest's "
        "initialization in gpuwm/core/model.py, none staged here); "
        "preparation never builds one. The module ships with the rest of "
        "gpuwm/ingest, and gpuwm/ingest/nest_init.py imports it on that same "
        "forecast-only branch",
    ("gpuwm/fetch.py", "tools.download_gfs_native_subset"):
        "the GFS subset transport (the NOMADS crop and whole-object "
        "downloads, their level ladder and record counts), imported inside "
        "the locked bodies of fetch_gfs and fetch_gfs_fullfile and five level "
        "and record helpers.  Their callers are the `gpuwm fetch` door "
        "(fetch_main and the route fetch under it, registered only by "
        "gpuwm/cli.py, excluded above), gpuwm/core/preflight.py (not staged) "
        "and gpuwm/download_budget.py (excluded above).  This package "
        "has no fetch door: rw-wps --source gfs prepares a series already "
        "fetched and authors its front-door manifest from the fetch receipt "
        "on disk, which reads no transport",
}


def _names_verify(module: str) -> bool:
    return module == "gpuwm.verify" or module.startswith("gpuwm.verify.")


def _staged_verification_imports(destination: Path) -> list[dict[str, object]]:
    """Return developer-only ``gpuwm.verify`` imports in staged runtime code.

    RW-WPS intentionally excludes the verification package.  Scanning every
    staged Python module closes the package boundary at build time instead of
    relying on a hand-maintained list of public entry points.  Direct imports
    and literal dynamic imports are both rejected.
    """
    violations: list[dict[str, object]] = []
    for source in sorted(destination.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            candidates: list[tuple[str, str]] = []
            if isinstance(node, ast.Import):
                candidates.extend((alias.name, "import") for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                candidates.append((node.module, "from"))
                # `from gpuwm import verify` imports gpuwm.verify.
                if not _names_verify(node.module):
                    candidates.extend(
                        (f"{node.module}.{alias.name}", "from")
                        for alias in node.names if alias.name != "*")
            elif isinstance(node, ast.Call) and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    function = node.func
                    dynamic = (
                        isinstance(function, ast.Name)
                        and function.id == "__import__"
                    ) or (
                        isinstance(function, ast.Attribute)
                        and function.attr == "import_module"
                    )
                    if dynamic:
                        candidates.append((argument.value, "dynamic"))
            for module, kind in candidates:
                if _names_verify(module):
                    violations.append({
                        "path": source.relative_to(destination).as_posix(),
                        "line": int(getattr(node, "lineno", 0)),
                        "module": module,
                        "kind": kind,
                    })
    return violations


def _is_source_module(source_root: Path, module: str) -> bool:
    """Is ``module`` a module or a regular package in the source tree?"""

    path = source_root.joinpath(*module.split("."))
    return (path.with_suffix(".py").is_file()
            or (path / "__init__.py").is_file())


def _staged_internal_imports(
    destination: Path, *, source_root: Path, optional: bool = False,
) -> list[dict[str, object]]:
    """Return imports whose internal module is absent from wheel staging.

    ``from package import name`` imports ``package.name`` as a module
    whenever the source tree has one by that name, so each such alias is
    checked as well as ``package``: ``from gpuwm.core import host_libm``
    needs ``gpuwm/core/host_libm.py`` staged, not only
    ``gpuwm/core/__init__.py``.  Read as an import of ``gpuwm.core`` alone
    it passed this scan while host_libm was not staged, and the staged
    map projections could not be imported.  Only the source
    tree (``source_root``) can say whether a name is a submodule or an
    attribute of its package, which is why the scan is given one.
    """
    modules: set[str] = set()
    sources: list[tuple[Path, str, bool]] = []
    for source in sorted(destination.rglob("*.py")):
        relative = source.relative_to(destination)
        parts = list(relative.with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module = ".".join(parts)
        if module:
            modules.add(module)
            sources.append((source, module, is_package))

    violations: list[dict[str, object]] = []
    for source, current_module, is_package in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        type_checking_nodes = set()
        for conditional in ast.walk(tree):
            if not isinstance(conditional, ast.If):
                continue
            marker = conditional.test
            is_type_checking = (
                isinstance(marker, ast.Name) and marker.id == "TYPE_CHECKING"
            ) or (
                isinstance(marker, ast.Attribute)
                and marker.attr == "TYPE_CHECKING"
            )
            if is_type_checking:
                for statement in conditional.body:
                    type_checking_nodes.update(ast.walk(statement))
        current_package = (
            current_module if is_package else current_module.rpartition(".")[0]
        )
        for node in ast.walk(tree):
            if node in type_checking_nodes:
                continue
            candidates: list[tuple[str, str]] = []
            if isinstance(node, ast.Import):
                candidates.extend((alias.name, "import") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = current_package.split(".") if current_package else []
                    ascend = node.level - 1
                    if ascend > len(package_parts):
                        resolved = ""
                    else:
                        base = package_parts[:len(package_parts) - ascend]
                        if node.module:
                            base.extend(node.module.split("."))
                        resolved = ".".join(base)
                else:
                    resolved = node.module or ""
                if resolved:
                    candidates.append((resolved, "from"))
                    # A missing package is reported as itself; its
                    # submodules cannot be staged without it.
                    if resolved in modules:
                        candidates.extend(
                            (f"{resolved}.{alias.name}", "from")
                            for alias in node.names
                            if alias.name != "*" and _is_source_module(
                                source_root, f"{resolved}.{alias.name}"))
            elif isinstance(node, ast.Call) and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    function = node.func
                    dynamic = (
                        isinstance(function, ast.Name)
                        and function.id == "__import__"
                    ) or (
                        isinstance(function, ast.Attribute)
                        and function.attr == "import_module"
                    )
                    if dynamic:
                        candidates.append((argument.value, "dynamic"))
            for module, kind in candidates:
                root = module.partition(".")[0]
                if root not in {"gpuwm", "tools"}:
                    continue
                if module not in modules:
                    relative_path = source.relative_to(destination).as_posix()
                    reason = _OPTIONAL_STAGED_IMPORTS.get(
                        (relative_path, module))
                    if (reason is not None) != optional:
                        continue
                    record = {
                        "path": relative_path,
                        "line": int(getattr(node, "lineno", 0)),
                        "module": module,
                        "kind": kind,
                    }
                    if reason is not None:
                        record["optional_reason"] = reason
                    violations.append(record)
    return violations

#: The token :func:`_standalone_pyproject` replaces with this release.
_STANDALONE_VERSION_PLACEHOLDER = "0.0.0+placeholder"

#: The standalone RW-WPS wheel's own pyproject.  Its version is a
#: placeholder that :func:`_standalone_pyproject` fills in from the
#: running package, because it is not free to be anything else: the
#: sealed-runtime receipt refuses unless the installed ``rw-wps``
#: distribution version equals ``gpuwm.__version__``
#: (``native_wrf_distribution._installed_record_receipt``).  It read
#: ``0.1.1`` and agreed with the stale package constant by coincidence;
#: the moment that constant started telling the truth, a hardcoded
#: version here would have failed the very seal it feeds.
_STANDALONE_PYPROJECT = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rw-wps"
version = "0.0.0+placeholder"
description = "Native parallel preprocessing and stock-WRF initialization"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "netCDF4>=1.6"]
keywords = ["WRF", "WPS", "GRIB", "NetCDF", "weather"]
classifiers = ["License :: OSI Approved :: Apache Software License"]

[tool.setuptools]
license-files = ["LICENSE", "NOTICE", "licenses/LICENSE-rust-libm.txt", "licenses/LICENSE-FDLIBM-SunPro.txt"]

[project.optional-dependencies]
# [ctk] and the 14.0 floor for the same measured reason as the parent
# project's own GPU extras (see pyproject.toml): a CuPy wheel carries the
# compiler and no CUDA headers, so on a driver-only box every device call
# dies with "Failed to find CUDA headers", and [ctk] -- a CuPy 14 extra --
# is what supplies a matching toolkit as wheels.
gpu = ["cupy-cuda12x[ctk]>=14.0"]
geog = ["rasterio>=1.3", "pyproj>=3.6"]

[project.scripts]
rw-wps = "gpuwm.source_cli:main"
gpuwm-wrf-init = "gpuwm.source_cli:main"
gpuwm-wrf-runtime-check = "gpuwm.native_wrf_distribution:main"
gpuwm-mapped-inspect = "gpuwm.mapped_source:main"

[tool.setuptools.packages.find]
include = ["gpuwm*", "tools"]

[tool.setuptools.package-data]
gpuwm = [
  "native_wrf_support_v1.json",
  "physics_registry_v2.json",
  "wrf_direct_v461_contract.json",
  "authorities/*.json",
  "core/kernels/*.cu",
  "core/kernels/*.cuh",
  "data/noah_tables/*.TBL",
  "data/noah_tables/*.md",
]
tools = ["*.sh"]
"""


def _standalone_pyproject() -> str:
    """The standalone wheel's pyproject, stamped with THIS release.

    A plain substitution rather than ``str.format``: the template is
    TOML and carries braces of its own (``license = { file = ... }``).
    """

    from gpuwm import __version__

    stamped = _STANDALONE_PYPROJECT.replace(
        _STANDALONE_VERSION_PLACEHOLDER, __version__)
    if _STANDALONE_VERSION_PLACEHOLDER in stamped:
        raise RuntimeError(
            "the standalone pyproject template lost its version placeholder")
    return stamped


class NotAGitCheckout(RuntimeError):
    """The tree cannot answer whether a file is tracked.

    Distinct from "this file is untracked", which is a refusal.  A
    snapshot extracted from a tarball or a `git archive` has no index at
    all, so the question has no answer there -- and the two are not the
    same finding.  Conflating them cost a Linux gate run a 40-line
    `CalledProcessError` traceback ending in `exit status 128`, which
    reads as a staging bug rather than as "this directory is not a
    checkout".
    """


def _tree_is_a_git_checkout() -> bool:
    """Does ``REPO`` have a git index to ask?  Asked once, cached."""

    global _GIT_CHECKOUT
    if _GIT_CHECKOUT is None:
        probe = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--git-dir"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _GIT_CHECKOUT = probe.returncode == 0
    return _GIT_CHECKOUT


#: Memo for :func:`_tree_is_a_git_checkout` (None = not yet asked).
_GIT_CHECKOUT: bool | None = None


def _require_tracked(source: Path) -> None:
    relative = source.resolve().relative_to(REPO).as_posix()
    if not _tree_is_a_git_checkout():
        raise NotAGitCheckout(
            f"{REPO} is not a git checkout, so whether {relative} is "
            "tracked cannot be answered.  RW-WPS wheel staging copies "
            "tracked files only; run it from a clone (CI's "
            "actions/checkout leaves one), not from an extracted "
            "archive")
    subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", relative],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _copy_source(source: Path, destination: Path) -> None:
    _require_tracked(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _stage_rw_wps_python_project(destination: Path) -> dict[str, object]:
    """Create the source-matched, forecast-executor-free wheel project."""

    destination.mkdir()
    package = destination / "gpuwm"
    for source in sorted((REPO / "gpuwm").glob("*.py")):
        if source.name not in _TOP_LEVEL_EXCLUDES:
            _copy_source(source, package / source.name)
    for name in sorted(_ROOT_DATA):
        _copy_source(REPO / "gpuwm" / name, package / name)
    for subpackage, excludes in (
        ("ingest", _INGEST_EXCLUDES),
        # Radar observation ingest.  `gpuwm doctor` checks for the NEXRAD
        # front door by name, unconditionally, so that a green report can
        # never mean "radar is fine" on a machine that cannot read a radar
        # volume -- and that check imports `gpuwm.obs.nexrad`.  Staging the
        # subpackage is what keeps the check unconditional here too: making
        # the import lazy instead would reintroduce, in a new place, exactly
        # the silent-green hole the check exists to close.  It costs nothing
        # a preprocessing wheel should refuse -- the whole subpackage is
        # stdlib plus numpy, imports only `gpuwm.bridges` and
        # `gpuwm.static.projection` (both already staged), and contains no
        # CuPy and no forecast executor.  Turning radar bytes into gridded
        # observation files is preprocessing, which is what this wheel is.
        ("obs", _OBS_EXCLUDES),
        ("static", set()),
    ):
        for source in sorted((REPO / "gpuwm" / subpackage).glob("*.py")):
            if source.name not in excludes:
                _copy_source(source, package / subpackage / source.name)
    for name in sorted(_IO_MODULES):
        _copy_source(
            REPO / "gpuwm" / "io" / name,
            package / "io" / name,
        )
    for name in sorted(_CORE_MODULES):
        _copy_source(
            REPO / "gpuwm" / "core" / name,
            package / "core" / name,
        )
    _copy_source(
        REPO / "gpuwm" / "core" / "kernels" / "__init__.py",
        package / "core" / "kernels" / "__init__.py",
    )
    for name in CUDA_KERNEL_SOURCES:
        _copy_source(
            REPO / "gpuwm" / "core" / "kernels" / name,
            package / "core" / "kernels" / name,
        )
    for source in sorted((REPO / "gpuwm" / "authorities").glob("*.json")):
        _copy_source(source, package / "authorities" / source.name)
    for source in sorted((REPO / "gpuwm" / "data" / "noah_tables").iterdir()):
        if source.is_file():
            _copy_source(
                source,
                package / "data" / "noah_tables" / source.name,
            )
    for name in sorted(_TOOL_FILES):
        _copy_source(REPO / "tools" / name, destination / "tools" / name)
    _copy_source(REPO / "README.md", destination / "README.md")
    _copy_source(REPO / "LICENSE", destination / "LICENSE")
    _copy_source(REPO / "NOTICE", destination / "NOTICE")
    for name in ("LICENSE-rust-libm.txt", "LICENSE-FDLIBM-SunPro.txt"):
        _copy_source(REPO / "licenses" / name, destination / "licenses" / name)
    (destination / "pyproject.toml").write_text(
        _standalone_pyproject(),
        encoding="utf-8",
        newline="\n",
    )

    files = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*") if path.is_file()
    )
    forbidden = sorted(_FORBIDDEN_STAGED_FILES & set(files))
    if forbidden:
        raise RuntimeError(
            f"RW-WPS wheel staging contains forecast executors: {forbidden}"
        )
    verification_imports = _staged_verification_imports(destination)
    if verification_imports:
        raise RuntimeError(
            "RW-WPS wheel staging imports omitted developer verification "
            f"modules: {verification_imports}"
        )
    internal_imports = _staged_internal_imports(
        destination, source_root=REPO)
    if internal_imports:
        raise RuntimeError(
            "RW-WPS wheel staging has unresolved internal imports: "
            f"{internal_imports}"
        )
    optional_internal_imports = _staged_internal_imports(
        destination, source_root=REPO, optional=True)
    return {
        "distribution": PYTHON_DISTRIBUTION,
        "file_count": len(files),
        "files": files,
        "forecast_executor_files": forbidden,
        "verification_imports": verification_imports,
        "unresolved_internal_imports": internal_imports,
        "optional_internal_imports": optional_internal_imports,
    }


def _run(
    argv: list[str], *, cwd: Path = REPO,
    env: dict[str, str] | None = None,
) -> None:
    # ``main`` reserves stdout for its one machine-readable JSON receipt.
    # pip and Cargo both write progress to stdout, so inheriting their streams
    # produced a file that looked like a receipt but could not be parsed as
    # JSON.  Preserve all build diagnostics on stderr while keeping stdout a
    # strict, single-document interface suitable for atomic capture.
    subprocess.run(
        argv, cwd=cwd, env=env, check=True,
        stdout=sys.stderr, stderr=sys.stderr,
    )


def _cargo_release_environment(
    *, source_root: Path, target_dir: Path, source_date_epoch: str,
) -> dict[str, str]:
    """Return a path-stable environment for the sealed Rust build.

    Rust dependencies can encode absolute source filenames in panic-location
    tables even when application code never prints them.  A release assembled
    from a different clean checkout would then have different bridge hashes.
    Use Cargo's encoded flag channel so checkout paths containing whitespace
    are handled as one rustc argument, and discard caller-provided rust flags
    that would otherwise make the sealed artifact host-dependent.
    """
    environment = os.environ.copy()
    environment.pop("RUSTFLAGS", None)
    rust_flags = [
        f"--remap-path-prefix={source_root.resolve()}=/usr/src/rw-wps",
    ]
    if platform.system() == "Windows":
        # MSVC's linker otherwise writes the wall-clock link time into the PE
        # header and every debug-directory entry. /Brepro replaces those
        # timestamps and the CodeView identity with content-derived values.
        rust_flags.extend((
            "-C", "link-arg=/Brepro",
            "-C", "target-feature=+crt-static",
        ))
    environment["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(rust_flags)
    environment["CARGO_TARGET_DIR"] = str(target_dir)
    environment["SOURCE_DATE_EPOCH"] = source_date_epoch
    return environment


def build_release(args: argparse.Namespace) -> dict[str, object]:
    if platform.system() != "Linux" or platform.machine() not in {
        "x86_64", "AMD64",
    }:
        raise RuntimeError("RW-WPS release assembly requires Linux x86-64")
    if sys.version_info < (3, 11):
        raise RuntimeError("RW-WPS release assembly requires Python 3.11+")

    output = args.output_dir.resolve()
    archive = args.archive.resolve()
    if output.exists() or archive.exists():
        raise FileExistsError("output directory and archive must not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("refusing to build RW-WPS from a dirty source tree")

    source_date_epoch = subprocess.check_output(
        ["git", "-C", str(REPO), "show", "-s", "--format=%ct", "HEAD"],
        text=True,
    ).strip()
    with tempfile.TemporaryDirectory(prefix="rw-wps-release-build-") as raw:
        temporary = Path(raw)
        python_project = temporary / "python-project"
        python_inventory = _stage_rw_wps_python_project(python_project)
        wheel_dir = temporary / "wheel"
        wheel_dir.mkdir()
        wheel_environment = os.environ.copy()
        wheel_environment["SOURCE_DATE_EPOCH"] = source_date_epoch
        wheel_environment["PYTHONHASHSEED"] = "0"
        _run([
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(python_project),
        ], env=wheel_environment)
        wheels = sorted(wheel_dir.glob("rw_wps-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                f"wheel build produced {len(wheels)} RW-WPS wheels: {wheels}")

        cargo_target = temporary / "cargo-target"
        environment = _cargo_release_environment(
            source_root=REPO,
            target_dir=cargo_target,
            source_date_epoch=source_date_epoch,
        )
        # One build per workspace the bundle's declaration names, into
        # one target directory, which is the shape the release bridge
        # builder already uses.  Reading the workspaces off the table
        # instead of hard-coding the decoder one is what makes a bridge
        # that lives elsewhere -- rw_fetch does -- actually get built:
        # this command used to compile only tools/grib1_bridge and then
        # hand the packager a namespace with no rw_fetch in it at all.
        for workspace in BRIDGE_WORKSPACES:
            manifest = REPO / workspace / "Cargo.toml"
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"the standalone bundle declares bridges from "
                    f"{workspace} and this checkout has no {manifest}")
            _run([
                "cargo",
                "build",
                "--manifest-path",
                str(manifest),
                "--release",
                "--locked",
                "--offline",
            ], cwd=manifest.parent, env=environment)
        native = cargo_target / "release"
        distribution_args = argparse.Namespace(
            wheel=wheels[0],
            cpu_backend=native / CPU_BACKEND_LIBRARY,
            output_dir=output,
            archive=archive,
            **{bridge.dest: native / bridge.name
               for bridge in BUNDLED_BRIDGES},
        )
        result = build_distribution(distribution_args)

    result["build_interface"] = "tools/build_rw_wps_release.py"
    result["rust_build"] = "cargo-release-locked-offline"
    result["python_build"] = "pip-wheel-no-build-isolation-no-deps"
    result["python_package_inventory"] = python_inventory
    result["source_date_epoch"] = source_date_epoch
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        build_release(args), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
