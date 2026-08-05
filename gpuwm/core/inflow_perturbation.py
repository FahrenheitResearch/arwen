"""CPM-style inflow turbulence seeding for one-way LES nest children.

A one-way LES child receives smoothed parent fields through its lateral
boundary and has to grow its own turbulence while the inflow advects
across the domain.  The measured cost on the shipped 250 m child is a
spin-up fetch larger than the domain itself
(docs/superpowers/receipts/les/INFLOW-FETCH-D90-2026-08-03.md).  This
module seeds that transition: cell-blocked potential-temperature
perturbations applied to the child's rolling boundary VALUE tables on
the inflow-side relax-zone rows, refreshed on the FORCE cadence, so the
existing Davies relaxation nudges the relax zone toward a perturbed
target and advection carries the blocks into the interior where
buoyancy amplifies them.

Every constant is pinned in
docs/superpowers/receipts/les/INFLOW-GENERATOR-ACCEPTANCE-V2.md,
registered before any perturbed integration existed.  The mechanism in
one paragraph:

* Only the theta VALUE tables are touched, only on width rows
  ``d in [spec_zone, relax_zone)`` — the rows whose ``fcx[d]``
  relaxation target the state_specified_relaxation kernel reads.  The
  spec-zone row, all tendency tables, and every other field are
  byte-untouched, so mass, momentum and the outer specified rim remain
  exactly the parent's.
* The increment is applied in the tables' own WRF-coupled units:
  ``delta_table = (c1h[k]*(mub2d + mu_table) + c2h[k]) * delta_theta``,
  the same hybrid mass coupling ``coupled_current`` (lbc_state.cu)
  uses, so ``delta_theta`` Kelvin shift the relaxation equilibrium by
  ``delta_theta`` Kelvin.
* Amplitude follows the pinned Eckert-number convention
  ``theta_max = scale * U^2 / (Ec * cp)`` with Ec = 0.2 and U the
  face-mean boundary-normal wind over the perturbed depth.
* The perturbed depth is the parent-diagnosed PBLH over the child's
  footprint (the parent physics driver's own ``pblh`` field), times a
  pinned fraction of 1.0, expressed in child base-state half-level
  heights.
* Draws are uniform in [-1, 1), one per (mass level, 8-cell along-face
  block), held for 100 s of model time, from a counter-based NumPy
  Philox generator keyed on (seed, grid_id, face, refresh index) — the
  same draw on every card, every run, every restart.

OFF is absolute: when ``inflow_perturbation`` is false (the default)
:func:`build_inflow_perturbation` returns ``None`` and the coupler
executes not one instruction of this module — the OFF trajectory is
byte-identical to a build without this file (acceptance gate G1).
With ``inflow_perturbation_amplitude_scale = 0.0`` the hook runs its
read-only classification and skips every table write (gate G2).

This is an ArWen-over-WRF extension, entered in PROVENANCE.md (D10):
stock WRF v4.6.1 ships ``perturb_bdy`` (stochastic-pattern boundary
TENDENCY perturbation in the stoch package) and no cell-perturbation
path; the OFF default keeps every WRF-parity configuration untouched.
"""

from __future__ import annotations

import math

import numpy as np

from gpuwm.core import constants as c

#: Along-face cell-block size, child cells (acceptance v2 pin).
BLOCK_CELLS = 8
#: Perturbation Eckert number (acceptance v2 pin).
ECKERT = 0.2
#: Fraction of the parent-footprint PBLH the perturbation reaches.
VERTICAL_FRACTION = 1.0
#: Model seconds one unit draw is held before being redrawn.
REFRESH_SECONDS = 100.0
#: Stable face -> RNG key codes.  Never renumber: the code is part of
#: every registered draw's key.
FACE_CODES = {"west": 0, "east": 1, "south": 2, "north": 3}

_FACE_MODES = ("inflow", "outflow")


def refresh_index(force_ticks: int, parent_step_ticks: int,
                  parent_dt_seconds: float) -> int:
    """The draw-holding counter for one force, in integer clock ticks.

    ``REFRESH_SECONDS`` is snapped to a whole number of parent forces
    (at least one), and the index is cut from the absolute tick count,
    so a restarted run recomputes the same index at the same model time.
    """
    if parent_step_ticks <= 0:
        raise ValueError("parent_step_ticks must be positive")
    if not (parent_dt_seconds > 0.0 and math.isfinite(parent_dt_seconds)):
        raise ValueError("parent dt must be a positive finite number")
    forces_per_refresh = max(
        1, int(round(REFRESH_SECONDS / parent_dt_seconds)))
    return int(force_ticks) // (forces_per_refresh * int(parent_step_ticks))


def draw_unit_pattern(seed: int, grid_id: int, face: str,
                      refresh: int, nz: int, n_blocks: int) -> np.ndarray:
    """One held unit draw: (nz, n_blocks) float64 uniform in [-1, 1).

    Counter-based and fully keyed: Philox seeded by SeedSequence
    entropy (seed, grid_id, face code, refresh index), consumed in a
    fixed (level, block) order, so each (domain, face, cell-block,
    refresh) owns a fixed counter position.  Identical on every
    platform NumPy guarantees stream stability for.
    """
    if seed < 0 or refresh < 0:
        raise ValueError("seed and refresh index must be non-negative")
    sequence = np.random.SeedSequence(
        entropy=(int(seed), int(grid_id), FACE_CODES[face], int(refresh)))
    generator = np.random.Generator(np.random.Philox(sequence))
    return generator.uniform(-1.0, 1.0, size=(int(nz), int(n_blocks)))


def expand_blocks(pattern: np.ndarray, face_length: int) -> np.ndarray:
    """Spread per-block draws onto per-cell columns along the face."""
    if pattern.ndim != 2:
        raise ValueError("pattern must be (nz, n_blocks)")
    needed = -(-int(face_length) // BLOCK_CELLS)
    if pattern.shape[1] != needed:
        raise ValueError(
            f"pattern has {pattern.shape[1]} blocks, face of "
            f"{face_length} cells needs {needed}")
    return np.repeat(pattern, BLOCK_CELLS, axis=1)[:, :int(face_length)]


def perturbed_level_count(z_half_agl: np.ndarray, z_i: float) -> int:
    """Levels (contiguous from the surface) inside the pinned extent."""
    if not math.isfinite(z_i) or z_i <= 0.0:
        return 0
    limit = VERTICAL_FRACTION * float(z_i)
    below = np.asarray(z_half_agl, dtype=np.float64) < limit
    count = 0
    for flag in below:
        if not flag:
            break
        count += 1
    return count


def select_faces(inward_means: dict[str, float], mode: str) -> list[str]:
    """flow_dep_bdy face selection from signed inward-normal means.

    ``inward_means[face]`` is positive where the boundary-normal wind
    enters the domain.  ``"inflow"`` takes strictly-entering faces,
    ``"outflow"`` (the AC-P3.4 mutation control) the strictly-leaving
    ones; an exactly-zero mean face belongs to neither.
    """
    if mode not in _FACE_MODES:
        raise ValueError(f"faces mode must be one of {_FACE_MODES}")
    wanted = (lambda value: value > 0.0) if mode == "inflow" \
        else (lambda value: value < 0.0)
    return [face for face in ("west", "east", "south", "north")
            if wanted(inward_means[face])]


def face_amplitude(inward_mean: float, amplitude_scale: float) -> float:
    """The pinned Eckert-number amplitude for one face, in Kelvin."""
    speed = abs(float(inward_mean))
    return float(amplitude_scale) * speed * speed / (ECKERT * c.CP)


def add_coupled_theta(fields, face: str, delta_theta, state, *,
                      spec_zone: int, relax_zone: int) -> None:
    """Add one face's Kelvin increment to the coupled theta value table.

    ``delta_theta`` is (nz, face_length) float32 Kelvin (already
    amplitude-scaled and zeroed above the perturbed depth).  The write
    touches exactly the relax-target rows ``[spec_zone, relax_zone)``
    of the theta VALUE table; tendencies and all other tables are
    untouched.  All arithmetic is float32 elementwise on device.
    """
    import cupy as cp

    theta_value = fields["theta"][face][0]
    mu_value = fields["mu"][face][0][0]      # (ny, W) x-sides / (W, nx)
    c1h = state.c1h
    c2h = state.c2h
    rows = slice(int(spec_zone), int(relax_zone))
    delta = cp.asarray(np.ascontiguousarray(delta_theta, dtype=np.float32))
    if face in ("west", "east"):
        width = int(theta_value.shape[-1])
        columns = state.mub2d[:, :width] if face == "west" \
            else state.mub2d[:, ::-1][:, :width]
        mass = columns + mu_value                       # (ny, W)
        ch = c1h[:, None, None] * mass[None] + c2h[:, None, None]
        theta_value[:, :, rows] += ch[:, :, rows] * delta[:, :, None]
    else:
        width = int(theta_value.shape[1])
        columns = state.mub2d[:width, :] if face == "south" \
            else state.mub2d[::-1, :][:width, :]
        mass = columns + mu_value                       # (W, nx)
        ch = c1h[:, None, None] * mass[None] + c2h[:, None, None]
        theta_value[:, rows, :] += ch[:, rows, :] * delta[:, None, :]


class InflowPerturbation:
    """Per-coupler generator state: pinned geometry plus the config keys."""

    def __init__(self, child_node):
        run = child_node.cfg.run
        self.grid_id = int(child_node.cfg.grid_id)
        self.seed = int(run.inflow_perturbation_seed)
        self.amplitude_scale = float(run.inflow_perturbation_amplitude_scale)
        self.faces_mode = str(run.inflow_perturbation_faces)
        self.nz = int(run.nz)
        self._face_length = {"west": int(run.ny), "east": int(run.ny),
                             "south": int(run.nx), "north": int(run.nx)}
        self._n_blocks = {
            face: -(-length // BLOCK_CELLS)
            for face, length in self._face_length.items()}
        ratio = int(child_node.cfg.parent_grid_ratio)
        self._footprint = (
            int(child_node.cfg.j_parent_start) - 1,
            int(child_node.cfg.i_parent_start) - 1,
            max(1, int(run.ny) // ratio),
            max(1, int(run.nx) // ratio))
        self._z_half_agl: np.ndarray | None = None

    # -- deterministic per-force ingredients --------------------------------

    def _base_half_heights(self, state) -> np.ndarray:
        """Domain-mean base-state half-level height AGL, cached (static)."""
        if self._z_half_agl is None:
            import cupy as cp

            phb = state.phb
            if getattr(phb, "ndim", 1) == 3:
                full = cp.asnumpy(phb.mean(axis=(1, 2), dtype=cp.float64))
            else:
                full = cp.asnumpy(cp.asarray(phb, dtype=cp.float64))
            full = full / c.G
            self._z_half_agl = (0.5 * (full[:-1] + full[1:]) - full[0])
        return self._z_half_agl

    def _parent_footprint_pblh(self, parent_node) -> float:
        import cupy as cp

        driver = parent_node.state.physics
        fields = getattr(driver, "fields", None) if driver is not None \
            else None
        if not fields or "pblh" not in fields:
            raise RuntimeError(
                "inflow_perturbation reads the parent's PBLH diagnostic "
                "and the parent physics driver does not carry one; the "
                "coupler-construction guard should have refused this "
                "configuration")
        j0, i0, nj, ni = self._footprint
        window = fields["pblh"][j0:j0 + nj, i0:i0 + ni]
        return float(window.mean(dtype=cp.float64))

    def _inward_means(self, state, n_levels: int) -> dict[str, float]:
        import cupy as cp

        levels = slice(0, int(n_levels))
        u, v = state.u, state.v
        return {
            "west": float(u[levels, :, 0].mean(dtype=cp.float64)),
            "east": -float(u[levels, :, -1].mean(dtype=cp.float64)),
            "south": float(v[levels, 0, :].mean(dtype=cp.float64)),
            "north": -float(v[levels, -1, :].mean(dtype=cp.float64)),
        }

    # -- the force-time hook ------------------------------------------------

    def apply_at_force(self, node, fields) -> None:
        """Perturb the freshly written theta tables for one FORCE."""
        parent = node.parent
        run = node.cfg.run
        z_i = self._parent_footprint_pblh(parent)
        n_levels = perturbed_level_count(
            self._base_half_heights(node.state), z_i)
        if n_levels == 0:
            return
        inward = self._inward_means(node.state, n_levels)
        refresh = refresh_index(
            int(parent.clock.ticks),
            int(parent.clock.spec.step_ticks),
            float(parent.clock.spec.dt_fp32))
        for face in select_faces(inward, self.faces_mode):
            theta_max = face_amplitude(inward[face], self.amplitude_scale)
            if theta_max == 0.0:
                continue
            pattern = draw_unit_pattern(
                self.seed, self.grid_id, face, refresh,
                self.nz, self._n_blocks[face])
            delta = expand_blocks(pattern, self._face_length[face])
            delta = (theta_max * delta).astype(np.float32)
            delta[n_levels:, :] = np.float32(0.0)
            add_coupled_theta(
                fields, face, delta, node.state,
                spec_zone=int(run.spec_zone),
                relax_zone=int(run.relax_zone))


def build_inflow_perturbation(child_node):
    """The coupler-construction entry: ``None`` unless configured ON.

    The ``None`` return is the whole OFF contract — the coupler holds no
    object and executes no instruction of this module (acceptance G1).
    A parent without a PBL scheme is refused here, at construction, not
    silently at the first force: the vertical extent is defined by the
    parent's PBLH diagnostic and a PBL-off parent has none.
    """
    run = child_node.cfg.run
    if not run.inflow_perturbation:
        return None
    parent = child_node.parent
    if parent is None or parent.cfg.run.bl_pbl_physics == 0:
        raise ValueError(
            "inflow_perturbation defines its vertical extent from the "
            "parent-diagnosed PBLH and therefore requires a parent "
            "running a PBL scheme; parent "
            f"bl_pbl_physics={0 if parent is None else parent.cfg.run.bl_pbl_physics}"
        )
    return InflowPerturbation(child_node)


__all__ = [
    "BLOCK_CELLS", "ECKERT", "FACE_CODES", "REFRESH_SECONDS",
    "VERTICAL_FRACTION", "InflowPerturbation", "add_coupled_theta",
    "build_inflow_perturbation", "draw_unit_pattern", "expand_blocks",
    "face_amplitude", "perturbed_level_count", "refresh_index",
]
