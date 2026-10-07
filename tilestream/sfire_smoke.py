"""Whole-domain accounting for the native bulk smoke source on tiles."""
from __future__ import annotations

from contextlib import nullcontext

from gpuwm.core.chem_context import LEDGER_BUCKETS
from gpuwm.core.chem_state import LEDGER_TOTALS, ledger_attr

OLD_MU_KEY = "transport/sfire_smoke/mup0"
GLOBAL_KEYS = tuple("state/" + ledger_attr(name)
    for name in (*LEDGER_TOTALS, *LEDGER_BUCKETS, "started"))


def time_row_key(row):
    return "transport/sfire_smoke/" + row.time_attr


def accepts(cfg):
    """Only the source-only endogenous bulk profile is partitioned here."""
    from gpuwm.chem_table import chem_names, load
    if chem_names(cfg.chem_sets) != ("sfire_smoke",) or cfg.chem_sources:
        return False
    table = load(cfg)
    return (len(table.rows) == 1 and table.processes == ("emission.sfire",)
            and not table.enabled_sources)


def transport_fields(state):
    """Rebuilt time-t operands, carried for accounting after the sweep."""
    if state.chem is None:
        return {}
    return {OLD_MU_KEY:state.mup0, **{
        time_row_key(row):getattr(state,row.time_attr) for row in state.chem.transported}}


def configure_tile(state):
    if state.chem is not None:
        state.chem.externally_managed_ledger = True


def refresh_history(streamed):
    """Rebuild output-only chemistry in bounded buffers after a disk restore.

    This copies authoritative operands and scatters only diagnostics. It
    integrates no atmospheric or fire step and changes no domain clock.
    """
    import cupy as cp
    from tilestream import gather, driver
    from tilestream.global_inventory import global_keys
    from tilestream.sfire import configure_tile as configure_fire_tile
    from gpuwm.core.chem_history import streaming_fields
    run = streamed.tiled_run
    run.drain()
    keys = global_keys(streamed.store)
    output_keys = tuple(key for key in streamed.store if key.startswith('diag/chem/'))
    ranked = bool(getattr(run, 'ranked', False))
    for rank, spec in enumerate(run.specs):
        tile = run.tiles[rank if ranked else 0]
        with cp.cuda.Device(run.devices[rank]) if ranked else nullcontext():
            configure_fire_tile(tile, spec)
            if streamed._geography is not None:
                gather.gather_tile(streamed._geography, tile, spec, nz=run.cfg.nz,
                                   inventory_fn=driver.geography_inventory)
            gather.gather_tile(streamed.store, tile, spec, nz=run.cfg.nz,
                               inventory_fn=streamed.inventory_fn, global_keys=keys)
            streaming_fields(tile)
            gather.scatter_tile(tile, streamed.store, spec, nz=run.cfg.nz,
                                inventory_fn=streamed.inventory_fn, names=output_keys)
            cp.cuda.get_current_stream().synchronize()
    cp.cuda.get_current_stream().synchronize()


class GlobalSmokeLedger:
    """The resident ordered reduction over complete mapped host fields.

    The caller drains outstanding tile transfers before both boundaries.
    Time-row and old-mu carriers are outputs of the tile's real RK save,
    so source accounting does not reconstruct or infer those operands.
    """
    def __init__(self, cfg, geography, template):
        if not accepts(cfg):
            raise ValueError("tiled SFIRE accounting supports only the source-only bulk smoke profile")
        self.cfg, self.geography, self.template = cfg, geography, template
        self.rows = tuple(template.chem.transported)
        self.current_keys = tuple("state/"+row.state_attr for row in self.rows)
        self.time_keys = tuple(time_row_key(row) for row in self.rows)
        self._before = None

    def _measure(self, store, keys, mu):
        from tilestream.chem_mass import global_masses
        return global_masses(store, keys, mu, self.cfg, self.geography,
                             self.template, rows=self.rows)

    @staticmethod
    def _vectors(store):
        from tilestream.sfire_spotting import mapped_host_array
        return {name:mapped_host_array(store["state/"+ledger_attr(name)])
            for name in (*LEDGER_TOTALS, *LEDGER_BUCKETS, "started")}

    def begin(self, store):
        import cupy as cp
        if self._before is not None:
            raise RuntimeError("the prior tiled smoke sweep has not closed its ledger")
        vectors = self._vectors(store)
        mass = self._measure(store,self.current_keys,"state/mup")
        if not bool(vectors["started"][0].item()):
            vectors["initial"][...] = mass
            vectors["current"][...] = mass
            vectors["started"][...] = 1
        vectors["transport"][...] += mass-vectors["current"]
        self._before = mass
        # The next sweep mutates the mapped fields read by this reduction.
        cp.cuda.get_current_stream().synchronize()

    def end(self, store):
        import cupy as cp
        if self._before is None:
            raise RuntimeError("tiled smoke accounting must begin before the sweep")
        vectors = self._vectors(store)
        after_source = self._measure(store,self.time_keys,OLD_MU_KEY)
        vectors["emitted"][...] += after_source-self._before
        current = self._measure(store,self.current_keys,"state/mup")
        vectors["transport"][...] += current-after_source
        vectors["current"][...] = current
        # Checkpoint/history readers access these mapped vectors on the host.
        cp.cuda.get_current_stream().synchronize()
        self._before = None
