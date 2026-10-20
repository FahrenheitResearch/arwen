"""Initialize perturbation members from immutable, validated t0 checkpoints.

Parents are shared only within one launch, source trajectory and physics
configuration. The existing restart writer/reader own every array and check;
this module only orders checkpoint restore before the perturbation callback.
"""
from pathlib import Path
import hashlib
import json
import threading


class CheckpointFill:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.receipts = {}
        self.reconstructions = {}
        self.parents = {}

    def callback(self, provenance, *, member_id, previous=None):
        def initialize(*, model=None, state=None, cfg=None, **kwargs):
            nodes = ([(node.cfg.grid_id, node.state, node.cfg.run)
                      for node in model.walk_parent_first()
                      if node.state is not None and getattr(node, "_started", True)]
                     if model is not None else [(getattr(cfg, "grid_id", 1), state, cfg)])
            from gpuwm.io.restart import configuration_echo, write_restart, restore_restart
            from gpuwm.bridge_assets import sha256_file
            for grid_id, selected, run in nodes:
                if getattr(selected, "_ensemble_checkpoint_fill", None) == member_id:
                    continue
                # A normal resume restores its own advanced checkpoint after
                # this initialization hook; never save that advanced state as t0.
                if float(selected.elapsed_seconds) != 0:
                    # A moved or newly born nest is rebuilt mid-run from this
                    # member's own, already branched parent, so it carries the
                    # member's perturbation; a shared t0 parent restored here
                    # would overwrite it with another member's state.  Breakage
                    # this prevents: a WoFS hurricane ensemble (two-domain
                    # moving nest) stopped every fill member at its first nest
                    # move (WOOF 1.1.0 dry cut, d02 move at 900 s).
                    with self.lock:
                        branched = bool(self.receipts.get(member_id))
                        if branched:
                            self.reconstructions.setdefault(member_id, []).append({
                                "member_id": member_id, "grid_id": grid_id,
                                "model_elapsed_seconds": float(selected.elapsed_seconds),
                                "applied": False,
                                "reason": "rebuilt mid-run from the member's branched parent"})
                    if branched:
                        continue
                    raise ValueError("perturbation-fill parents must be checkpointed at initialization")
                identity = {key: provenance[key] for key in
                    ("source", "cycle", "source_member", "lag_hours") if key in provenance}
                identity.update(grid_id=grid_id, configuration=configuration_echo(run))
                key = hashlib.sha256(json.dumps(identity, sort_keys=True,
                    separators=(",", ":"), default=str).encode()).hexdigest()
                parent = self.root / (key + ".npz")
                with self.lock:
                    parent_lock = self.parents.setdefault(key, threading.Lock())
                # Only publication of the same immutable parent is shared.
                # Different physics/source parents and each child's restore
                # use independent member streams, rather than one run lock.
                with parent_lock:
                    self.root.mkdir(parents=True, exist_ok=True)
                    reused = parent.exists()
                    if not reused:
                        write_restart(parent, selected, run)
                restored = restore_restart(parent, selected, run)
                row = {"member_id": member_id, "grid_id": grid_id,
                           "parent_checkpoint": str(parent),
                           "parent_sha256": sha256_file(parent), "parent_reused": reused,
                           "model_elapsed_seconds": restored.elapsed_seconds,
                           "applied": "after initialized parent checkpoint restore, before the first branch step"}
                with self.lock:
                    self.receipts.setdefault(member_id, {})[grid_id] = row
                selected._ensemble_checkpoint_fill = member_id
            if previous is not None:
                previous(model=model, state=state, cfg=cfg, **kwargs)
        return initialize
