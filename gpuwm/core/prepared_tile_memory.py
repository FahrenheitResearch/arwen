"""Itemize independent prepared RTE+RRTMGP tile buffers without CUDA.

The prepared factory executes RRTMGPRadiation without chunk_workspace.
Its finalized optics and Planck arrays therefore cannot be priced from the
fused shared-workspace inventory.  Every buffer below pays its own complete
LW+SW allocation set: CuPy keeps separate free arenas for separate compute
streams.  This deliberately forgoes both phase aliasing and cross-buffer
pool reuse.  The column-bounded loader and its retained last slab are also
inside the envelope.  No physics implementation or user setting is changed.

THE MODEL PRICES THE ROUTE, NOT ONE CONFIGURATION.  Every term below is
read off the experiment it is handed -- ``nz``, ``p_top``, ``column_chunk``
from the vertical and chunking, the resident and transient bytes from
:func:`gpuwm.core.preflight.estimate_domain` for whatever physics is
selected -- so it applies to any prepared single-root host-store run on the
RTE+RRTMGP solver.  It was first shipped behind a twelve-value config
fingerprint (Morrison, KF, YSU, nz 49, chunk 3125, ...), and every other
configuration on the same route silently kept the fused-inventory price,
measured 23 % low against this itemization at 980,000 cells with two
buffers (9.304 GiB itemized against 7.580 GiB fused, ENG-013).  The
fingerprint survives only as :func:`measured_anchor`, the configuration the
itemization was MEASURED against, for reports and tests -- never as a gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math


def supported(exp, cfg, options) -> bool:
    """The route this module itemizes: a prepared single-root RTE+RRTMGP host store.

    The ROUTE condition, and nothing narrower (ENG-013).  Two limits remain
    and each names what it cannot price:

    * legacy RRTMG -- :func:`standalone_rte_storage_bytes` itemizes the
      RTE+RRTMGP solver's own storage (its gas tables, g-points and Planck
      arrays); legacy RRTMG's prepared factory was not itemized, so it keeps
      the fused-inventory price.  Retire by itemizing legacy RRTMG.
    * CAM ozone -- adds interpolation arrays the itemization does not carry.
      Retire by adding them to :func:`standalone_rte_storage_bytes`.
    """
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.physics_compat import RRTMG_VARIANT_RTE_RRTMGP, rrtmg_variant
    return bool(
        exp is not None and len(exp.domains) == 1
        and exp.root.run == cfg and cfg.specified and not cfg.nested
        # ON THE ROUTE: [tiles] configured for this domain (mode on/auto)
        # with a host store.  A resident run has no prepared tile buffers
        # to price and keeps the resident estimate untouched.
        and options.enabled and options.store == "host"
        # The implementation selector retains its default even when no
        # spectrum requests RRTMG. Only an active RRTMG spectrum allocates
        # the unfused RTE storage itemized below; other schemes keep their
        # own price. A single active spectrum is conservatively charged
        # both until the itemization is split by spectrum.
        and 4 in radiation_scheme_ids(cfg)
        and rrtmg_variant(cfg) == RRTMG_VARIANT_RTE_RRTMGP
        and options.follower_context is None
        and not getattr(options.radiation_context, "cam_ozone", False))


def measured_anchor(exp, cfg, options) -> bool:
    """Is this the configuration the itemization was MEASURED against?

    The Sep08 prepared-store probes on the RTX 3080 (allocator peak,
    loader template, unfused LW+SW arrays -- see
    tests/test_prepared_tile_memory.py) were taken on exactly this
    configuration.  A report may say so; a test may pin it.  It is NOT a
    gate on :func:`supported`: gating the model on it left every other
    configuration on the fused-inventory price measured 23 % low (ENG-013).
    """
    from gpuwm.config import radiation_scheme_ids
    return bool(
        supported(exp, cfg, options)
        and cfg.nz == 49 and 0.0 < exp.vertical.p_top <= 10000.0
        and exp.column_chunk == 3125
        and cfg.mp_physics == 10 and cfg.cu_physics == 1
        and cfg.bl_pbl_physics == 1 and cfg.sf_sfclay_physics == 91
        and cfg.sf_surface_physics == 2 and cfg.num_soil_layers == 4
        and radiation_scheme_ids(cfg) == (4, 4)
        and not cfg.use_adaptive_time_step)


def standalone_rte_storage_bytes(nz: int, columns: int, column_chunk: int,
                                 p_top: float) -> dict[str, int]:
    """Named storage of the actual workspace=None branch, without aliasing.

    Columns/upper profiles/metadata are already in estimate_domain. This
    prices the solver allocations beside them, keeping both spectra and an
    old Planck result simultaneously even where the actual code frees them.
    The old sources variable survives into SW and the next chunk's RHS.
    Partial-flux storage takes the largest non-folded tile (16 g-points),
    independent of the attached device; folded launches allocate none.
    This bound stays monotone across chunk tails and SM coverage thresholds.
    """
    from gpuwm.core.rrtmgp import (
        load_gas_tables, rrtmgp_above_model_layer_counts)

    c = min(int(columns), int(column_chunk))
    if c < 1:
        return {}
    ul, us = rrtmgp_above_model_layer_counts(p_top)
    nl, ns = nz + ul, nz + us
    lw, sw = load_gas_tables("lw"), load_gas_tables("sw")
    planck = c * lw.ngpt * (nl + nl + 1 + 1) * 4
    return {
        "lw/gas_tau": c * nl * lw.ngpt * 4,
        "lw/vmr": c * nl * (lw.ngas + 1) * 4,
        "lw/cloud_tau_ssa_g": 3 * c * nl * lw.nband * 4,
        "lw/col_dry": c * nl * 4,
        "lw/mcica_mask": c * nl * lw.ngpt,
        "lw/finalized_tau": c * nl * lw.ngpt * 4,
        "lw/planck_current": planck,
        # The previous chunk's sources are still bound while the new result
        # is constructed. Always charge both, including a one-chunk tile.
        "lw/planck_previous": planck,
        "lw/emiss_incident": 2 * c * lw.ngpt * 4,
        "lw/flux_up_dn": 2 * c * (nl + 1) * 4,
        "lw/partial_flux": 2 * 16 * c * (nl + 1) * 4,
        "lw/ozone_interp_log": c * nl * (8 + 4),
        "sw/gas_tau_ssa": 2 * c * ns * sw.ngpt * 4,
        "sw/vmr": c * ns * (sw.ngas + 1) * 4,
        "sw/cloud_tau_ssa_g": 3 * c * ns * sw.nband * 4,
        "sw/col_dry": c * ns * 4,
        "sw/mcica_mask": c * ns * sw.ngpt,
        "sw/finalized_tau_ssa_g": 3 * c * ns * sw.ngpt * 4,
        "sw/albedo_incident": 2 * c * sw.ngpt * 4,
        "sw/mu0_layers": c * ns * 4,
        "sw/flux_up_dn_dir": 3 * c * (ns + 1) * 4,
        "sw/partial_flux": 3 * 16 * c * (ns + 1) * 4,
        "sw/ozone_interp_log": c * ns * (8 + 4),
    }


@dataclass(frozen=True)
class PreparedTileMemory:
    """Monotone current-code envelope, suitable for planner inversion."""
    experiment: object = field(repr=False, compare=False)
    profile: object = field(repr=False, compare=False)
    forcing_intervals: int = 0
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def nz(self):
        return self.experiment.root.run.nz

    def _domain(self, nx, ny):
        from gpuwm.core import preflight as pf
        exp = self.experiment
        dc = replace(exp.root, run=replace(exp.root.run, nx=nx, ny=ny))
        return pf.estimate_domain(
            dc, spec_bdy_width=exp.spec_bdy_width,
            n_lbc_intervals=self.forcing_intervals,
            p_top=exp.vertical.p_top, column_chunk=exp.column_chunk)

    def fixed_terms(self):
        if "fixed" not in self._cache:
            from gpuwm.core import preflight as pf
            from gpuwm.ingest.prepared_store import default_slab_rows
            exp = self.experiment
            nx, ny = exp.root.run.nx, exp.root.run.ny
            rows = default_slab_rows(nx, ny)
            last = ny % rows or rows
            template, slab = self._domain(nx, last), self._domain(nx, rows)
            self._cache["fixed"] = {
                "loader_rows": rows,
                "template_resident_bytes": template.resident_bytes,
                "loader_pool_peak_bytes": slab.resident_bytes + slab.transient_bytes,
                "k_tables_bytes": pf.k_distribution_bytes(),
                "cuda_context_bytes": self.profile.cuda_context_bytes,
                "local_memory_bytes": pf.kernel_local_memory_bytes(exp, profile=self.profile),
                "unmodelled_bytes": pf.ENVELOPE_UNMODELLED_BYTES,
            }
        return self._cache["fixed"]

    def buffer_terms(self, window_cells):
        from gpuwm.core import preflight as pf
        columns, rem = divmod(int(window_cells), self.nz)
        if rem or columns < 1:
            raise ValueError("a prepared tile must contain whole vertical columns")
        if columns not in self._cache:
            exp = self.experiment
            # At fixed area N, N x 1 maximizes every horizontal face/edge
            # product in this selected inventory: nx+ny <= N+1. This is a
            # sizing shape only, never an emitted or executed grid. It makes
            # the existing cell-count planner safe for rectangular windows.
            itemized = self._domain(columns, 1)
            run = replace(exp.root.run, nx=columns, ny=1)
            work_exp = replace(exp, domains=(replace(exp.root, run=run),))
            radiation = standalone_rte_storage_bytes(
                self.nz, columns, exp.column_chunk, exp.vertical.p_top)
            self._cache[columns] = {
                "resident_bytes": itemized.resident_bytes,
                "step_transient_bytes": itemized.transient_bytes,
                "radiation_named_storage_bytes": sum(radiation.values()),
                "column_workspace_bytes": pf.column_workspace_bytes(work_exp, profile=self.profile),
            }
        return self._cache[columns]

    def buffer_bytes(self, window_cells):
        return sum(self.buffer_terms(window_cells).values())

    def vram_bytes(self, window_cells, nbuffers):
        from gpuwm.core import preflight as pf
        fixed = self.fixed_terms()
        pool = (int(nbuffers) * self.buffer_bytes(window_cells)
                + fixed["template_resident_bytes"] + fixed["k_tables_bytes"])
        pool = max(pool, fixed["loader_pool_peak_bytes"] + fixed["k_tables_bytes"])
        return (math.ceil(pf.ALLOCATOR_HEADROOM * pool)
                + fixed["cuda_context_bytes"] + fixed["local_memory_bytes"]
                + fixed["unmodelled_bytes"])

    @property
    def process_overhead_bytes(self):
        from gpuwm.core import preflight as pf
        f = self.fixed_terms()
        return (math.ceil(pf.ALLOCATOR_HEADROOM * (
                    f["template_resident_bytes"] + f["k_tables_bytes"]))
                + f["cuda_context_bytes"] + f["local_memory_bytes"]
                + f["unmodelled_bytes"])

    def max_window_cells(self, nbuffers, budget):
        # Binary search calls the exact same monotone function as the final
        # candidate/envelope. No stale linear inversion can bypass its peak.
        lo, hi = 0, 1
        while self.vram_bytes(hi * self.nz, nbuffers) <= budget:
            lo, hi = hi, hi * 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.vram_bytes(mid * self.nz, nbuffers) <= budget:
                lo = mid
            else:
                hi = mid
        return lo * self.nz


def for_options(cfg, options, *, profile=None, estimate=None):
    """Return an inventoried-contract model, otherwise retain old pricing."""
    context = getattr(options, "resident_context", None)
    exp = None if context is None else context.experiment
    if not supported(exp, cfg, options):
        return None
    from gpuwm.core import preflight as pf
    # The supplied experiment estimate carries the resolved forcing schedule,
    # including wider prepared input intervals. No assumed GFS cadence is
    # substituted into HRRR/RAP or a differently retained boundary archive.
    if estimate is None:
        return None
    return PreparedTileMemory(exp, profile or pf.MEASURED_LOCAL_MEMORY_PROFILE,
                              estimate.retained_forcing_intervals)
