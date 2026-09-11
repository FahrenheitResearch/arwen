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

THE BUFFER IS PRICED AT THE WINDOW'S OWN SHAPE, NOT AT ``N x 1``.  Until
2.7.3 every buffer was itemized as an ``N x 1`` domain, ``N`` its column
count, on the argument that a one-row rectangle maximizes every horizontal
face and edge product and so bounds any rectangle of the same area.  It
does -- and for the terms that scale with the PERIMETER rather than the
area it bounds them by the perimeter of a rectangle no tiling can produce.
REPRODUCED on a user's 572x524x49 icon-eu forecast (Thompson, RTE+RRTMGP,
MYNN, Noah, six retained forcing intervals) with a pinned 250x250 tiling:
the 286x286 compute window is 81,796 columns, so the sizing rectangle was
81,796 x 1 with a perimeter 143x the window's, and its eager
``lbc_forcing_tables`` came to 9.03 GiB PER BUFFER against 0.06 GiB for the
window itself.  Two buffers of that, under the allocator headroom, put
20.8 GiB of forcing tables that no buffer allocates into the streamed
envelope, which read 38.96 GiB against a 17.43 GiB resident run -- a
streamed forecast priced at 2.2x the resident one, refused before fetch.

So a caller that knows the window (the pinned road, and the planner once
it has chosen a tile) passes ``shape=(window_nx, window_ny)`` and the
buffer is itemized at exactly that rectangle.  A caller that has only a
cell count (the planner's binary search over window sizes) gets
:meth:`PreparedTileMemory.sizing_shape`: the most elongated rectangle of
that area a tiling can LEGALLY produce -- no side narrower than the
smallest compute window, ``2 * halo + 1``, and none longer than the domain
plus its halo -- which still bounds every legal rectangle of that area on
every term, and does so by a perimeter a tiling can actually have.

MEASURED against the fix, 2026-09-10, node-1 (RTX 5070 Ti, 15.51 GiB,
cupy 14.2.0 / CUDA 13), the same 572x524x49 icon-eu forecast on real
2026-09-10T12 data, per-process device memory read by nvidia-smi
(``compute-apps used_memory``: the process's whole device allocation,
CUDA context included) at 0.2 s:

* 250x250 tiles, nbuffers = 1: this model 8.89 GiB; measured peak 6.18 GiB
  (6,326 MiB, at the radiation call), steady 5.97 GiB, over 40 steps and
  four radiation calls.
* 250x250 tiles, nbuffers = 2 (the user's tiling): this model 15.54 GiB;
  measured peak 11.11 GiB (11,374 MiB, recurring at each radiation call),
  steady 7.83 GiB between calls, over 122 steps, no allocation failure.

So the corrected envelope brackets the run on both roads with 28-31 %
to spare, and the per-buffer radiation storage it carries is a real
transient (about 1.6 GiB per buffer measured against the 2.01 GiB priced
per 3,125-column chunk here).  The run the user asked for does reach
11.1 GiB on a card of this class, above the 10.69 GiB they had free; one
buffer of the same tile measured 6.2 GiB and is the remedy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

GIB = 1024 ** 3


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

    def _columns(self, window_cells):
        columns, rem = divmod(int(window_cells), self.nz)
        if rem or columns < 1:
            raise ValueError("a prepared tile must contain whole vertical columns")
        return columns

    @property
    def halo(self) -> int:
        """The domain's own per-step dependency radius, in mass cells."""
        if "halo" not in self._cache:
            from tilestream.harness import halo_radius
            self._cache["halo"] = int(halo_radius(self.experiment.root.run))
        return self._cache["halo"]

    def sizing_shape(self, columns) -> tuple[int, int]:
        """The rectangle a buffer of ``columns`` is priced at when its shape
        is not known: the most elongated one a tiling can LEGALLY produce.

        Every compute window is a tile plus a halo on both sides of both
        axes, so no side is narrower than ``2 * halo + 1``; and a window is
        clamped inside a non-periodic domain or wraps a periodic one, so no
        side is longer than the domain's longer axis plus two halos.  Among
        rectangles of area ``N`` inside those limits the one with the
        longest perimeter -- the one that bounds every perimeter-scaled
        term for all of them -- puts one side at the lower limit, or, when
        that would make the other side too long, one side at the upper.
        Its area is at least ``N``, never less, so the per-cell terms are
        bounded too.  Monotone in ``N`` on every side, which is what the
        planner's binary inversion needs.

        A one-row ``N x 1`` rectangle was the sizing shape before this: see
        the module docstring for what its 143x perimeter did to a real
        user's envelope.
        """
        columns = int(columns)
        run = self.experiment.root.run
        lo = 2 * self.halo + 1
        hi = max(int(run.nx), int(run.ny)) + 2 * self.halo
        a = max(1, min(lo, math.isqrt(columns)))
        b = -(-columns // a)
        if b > hi:
            b = hi
            a = -(-columns // hi)
        return a, b

    def _shape_for(self, window_cells, shape):
        columns = self._columns(window_cells)
        if shape is None:
            return self.sizing_shape(columns)
        nx, ny = int(shape[0]), int(shape[1])
        if nx < 1 or ny < 1 or nx * ny != columns:
            raise ValueError(
                f"window shape {nx}x{ny} does not hold {columns} columns")
        return nx, ny

    def buffer_terms(self, window_cells, shape=None):
        """Itemized bytes of ONE tile buffer holding ``window_cells``.

        ``shape`` is the compute window ``(window_nx, window_ny)`` when the
        caller knows it, and the buffer is itemized at exactly that
        rectangle; without it the buffer is itemized at
        :meth:`sizing_shape`, which bounds every legal rectangle of that
        area.  Either way this is a sizing shape only, never an emitted or
        executed grid.
        """
        from gpuwm.core import preflight as pf
        key = self._shape_for(window_cells, shape)
        if key not in self._cache:
            exp = self.experiment
            nx, ny = key
            itemized = self._domain(nx, ny)
            run = replace(exp.root.run, nx=nx, ny=ny)
            work_exp = replace(exp, domains=(replace(exp.root, run=run),))
            radiation = standalone_rte_storage_bytes(
                self.nz, nx * ny, exp.column_chunk, exp.vertical.p_top)
            self._cache[key] = {
                "resident_bytes": itemized.resident_bytes,
                "step_transient_bytes": itemized.transient_bytes,
                "radiation_named_storage_bytes": sum(radiation.values()),
                "column_workspace_bytes": pf.column_workspace_bytes(work_exp, profile=self.profile),
            }
        return self._cache[key]

    def buffer_bytes(self, window_cells, shape=None):
        return sum(self.buffer_terms(window_cells, shape).values())

    def vram_bytes(self, window_cells, nbuffers, shape=None):
        from gpuwm.core import preflight as pf
        fixed = self.fixed_terms()
        pool = (int(nbuffers) * self.buffer_bytes(window_cells, shape)
                + fixed["template_resident_bytes"] + fixed["k_tables_bytes"])
        pool = max(pool, fixed["loader_pool_peak_bytes"] + fixed["k_tables_bytes"])
        return (math.ceil(pf.ALLOCATOR_HEADROOM * pool)
                + fixed["cuda_context_bytes"] + fixed["local_memory_bytes"]
                + fixed["unmodelled_bytes"])

    def terms(self, window_cells, nbuffers, shape=None):
        """Every term of :meth:`vram_bytes`, named, in the order they add up.

        The arithmetic a refusal prints: a reader holding a screenshot of
        it can check the total with a calculator and see which term the
        tile can move (the buffers) and which it cannot (the floors).
        Byte-valued entries end in ``_bytes``; the rest are labels.
        """
        from gpuwm.core import preflight as pf
        nx, ny = self._shape_for(window_cells, shape)
        columns = nx * ny
        exp = self.experiment
        run = exp.root.run
        fixed = self.fixed_terms()
        buffer = self.buffer_terms(window_cells, shape)
        per_buffer = sum(buffer.values())
        nbuffers = int(nbuffers)
        buffers = nbuffers * per_buffer
        pool = buffers + fixed["template_resident_bytes"] + fixed["k_tables_bytes"]
        loader_peak = fixed["loader_pool_peak_bytes"] + fixed["k_tables_bytes"]
        pool_priced = max(pool, loader_peak)
        pool_with_headroom = math.ceil(pf.ALLOCATOR_HEADROOM * pool_priced)
        itemized = self._domain(nx, ny)
        return {
            "domain": f"{int(run.nx)}x{int(run.ny)}x{int(run.nz)}",
            "window": (f"{nx}x{ny}x{int(run.nz)} = {columns:,} columns"
                       + ("" if shape is not None else
                          " (sizing rectangle; the window's shape was not given)")),
            "buffer/state_bytes": itemized.category_bytes("state"),
            "buffer/physics_bytes": itemized.category_bytes("physics"),
            "buffer/scratch_bytes": itemized.category_bytes("scratch"),
            "buffer/lbc_bytes": (itemized.category_bytes("lbc")
                                 + itemized.category_bytes("nest")),
            "buffer/diagnostic_bytes": itemized.category_bytes("diagnostic"),
            "buffer/step_transient_bytes": buffer["step_transient_bytes"],
            "buffer/radiation_named_storage_bytes": buffer["radiation_named_storage_bytes"],
            "buffer/column_workspace_bytes": buffer["column_workspace_bytes"],
            "buffer/total_bytes": per_buffer,
            "buffer/per_column_bytes": per_buffer // columns,
            "buffers": f"{nbuffers} x {per_buffer / GIB:.3f} GiB",
            "buffers_bytes": buffers,
            "fixed/template_resident_bytes": fixed["template_resident_bytes"],
            "fixed/k_tables_bytes": fixed["k_tables_bytes"],
            "fixed/loader_pool_peak_bytes": fixed["loader_pool_peak_bytes"],
            "pool_bytes": pool_priced,
            "pool_basis": ("the buffers plus the retained template and the k-tables"
                           if pool >= loader_peak else
                           f"the {fixed['loader_rows']}-row loader's initialization "
                           "peak, which is larger than the buffers"),
            "pool_headroom": f"x {pf.ALLOCATOR_HEADROOM:.2f}",
            "pool_with_headroom_bytes": pool_with_headroom,
            "fixed/cuda_context_bytes": fixed["cuda_context_bytes"],
            "fixed/local_memory_bytes": fixed["local_memory_bytes"],
            "fixed/unmodelled_bytes": fixed["unmodelled_bytes"],
            "radiation_chunk_columns": min(columns, int(exp.column_chunk)),
            "radiation_transient_bytes": 0,
            "vram_bytes": int(self.vram_bytes(window_cells, nbuffers, shape)),
        }

    @property
    def process_overhead_bytes(self):
        from gpuwm.core import preflight as pf
        f = self.fixed_terms()
        return (math.ceil(pf.ALLOCATOR_HEADROOM * (
                    f["template_resident_bytes"] + f["k_tables_bytes"]))
                + f["cuda_context_bytes"] + f["local_memory_bytes"]
                + f["unmodelled_bytes"])

    def max_window_cells(self, nbuffers, budget):
        # Binary search over the shape-free BOUND (sizing_shape), which is
        # monotone in the column count; the planner then prices the tile it
        # chose at that tile's own window, and the exact rectangle never
        # costs more than the bound that admitted it.
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
