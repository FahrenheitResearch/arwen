"""Sequential single-GPU ensemble orchestration (EXPERIMENTAL).

One member is resident in VRAM at a time.  Members are run by the
existing single-domain experiment runner -- this package adds member
layout, deterministic seeding, a manifest, resume, a cycling skeleton
with a data-assimilation seam, and a bench harness.  It contains no
model loop of its own and no physics.

Everything published here is stamped ``experimental`` in provenance and
is not wired into any default route: the entry point is
``tools/ensemble_forecast.py``.
"""

from gpuwm.ensemble.config import (
    ENSEMBLE_CONFIG_SCHEMA, EnsembleConfig, load_ensemble_config,
)
# The supported way to observe a leg's analyses, exported at the package
# door so a consumer never has to reach for the directory instead.  See
# gpuwm.ensemble.cycle for why a raw listing is out of contract.
from gpuwm.ensemble.cycle import (
    read_analysis_roster, recover_analysis_publication,
)
from gpuwm.ensemble.increments import (
    INCREMENT_CONTRACT, apply_increments, apply_increments_to_checkpoint,
)
from gpuwm.ensemble.manifest import (
    CYCLE_MANIFEST_SCHEMA, ENSEMBLE_MANIFEST_SCHEMA, MEMBER_STATUSES,
    member_directory_name, read_manifest, write_manifest_atomically,
)
from gpuwm.ensemble.seeds import SEED_DERIVATION, member_seed
from gpuwm.ensemble.wrfout_inventory import (
    WRFOUT_INVENTORY_CONTRACT, WRFOUT_INVENTORY_KEY, member_inventory,
)

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
