"""Vertical eta coordinate and discretely-balanced hydrostatic base state.

Pure NumPy (float64) setup-time module.  Everything here is computed once at
model initialization and cast to FP32 only when loaded into ``DomainState``.

Conventions (WRF-ARW):
- ``znw[0] = 1.0`` (surface) ... ``znw[nz] = 0.0`` (model top);
  ``dnw[k] = znw[k+1] - znw[k] < 0``.
- The base geopotential ``phb`` satisfies the *discrete* hydrostatic relation
  ``phb[k+1] = phb[k] - dnw[k]*(c1h[k]*mub + c2h[k])*alb[k]`` exactly; the
  model's at-rest balance depends on this recurrence, not on the continuous
  integral.  For ``hybrid_opt=0`` (c1h = 1, c2h = 0) this reduces bitwise to
  the Phase 1 form ``phb[k+1] = phb[k] - dnw[k]*mub*alb[k]``.

Phase 2 additions (WRF v4 hybrid coordinate, terrain):
- ``compute_hybrid_coeffs`` builds the WRF v4 c1/c2/c3/c4 arrays
  (``c3 = B(eta)``, ``c4 = (eta - B)(p0 - pt)``, ``c1 = dB/deta``,
  ``c2 = (1 - c1)(p0 - pt)``), transcribed from WRF v4.6.1
  ``dyn_em/module_initialize_ideal.F`` (also ``nest_init_utils.F``
  ``compute_vcoord_1d_coeffs``).
- ``make_base_state`` accepts a 2-D ``terrain_z`` and then returns per-column
  base profiles with 2-D dry mass ``mub[j,i] = p_s(z_sfc[j,i]) - p_top``;
  flat terrain keeps the Phase 1 scalar/1-D broadcast forms bitwise.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from gpuwm.core import constants as c

#: Sounding: maps height z (m) -> potential temperature theta (K).
Sounding = Callable[[np.ndarray], np.ndarray]


@dataclass
class VerticalCoord:
    """WRF vertical coordinate arrays (names/meanings match WRF).

    ``dn``, ``rdn``, ``fnp``, ``fnm`` are defined for k = 1..nz-1 and zero at
    the k = 0 endpoint, as in WRF; ``fnp``/``fnm`` are the interpolation
    weights for averaging half-level quantities to full levels.

    Phase 2: the WRF v4 hybrid-coordinate coefficient arrays ``c1f..c4f``
    (full levels, nz+1) and ``c1h..c4h`` (half levels, nz) ride along, with
    ``hybrid_opt``/``etac`` recording how they were built.  The
    pressure-weighted pairs c2/c4 carry a factor ``(p0 - p_top)``:
    ``make_vertical_coord`` fills them as zeros (p_top unknown) and
    ``make_base_state`` finalizes them in place once p_top is computed,
    recording that p_top in ``p_top`` (None until then); reuse with an
    inconsistent p_top raises instead of silently re-finalizing.
    For ``hybrid_opt=0`` every stage is the exact Phase 1 identity
    (c1 = 1, c2 = 0, c3 = eta, c4 = 0).
    """

    znw: np.ndarray   # (nz+1,) full-level eta
    znu: np.ndarray   # (nz,)   half-level eta
    dnw: np.ndarray   # (nz,)   dnw[k] = znw[k+1] - znw[k]  (< 0)
    rdnw: np.ndarray  # (nz,)   1/dnw
    dn: np.ndarray    # (nz,)   dn[k] = 0.5*(dnw[k] + dnw[k-1])
    rdn: np.ndarray   # (nz,)   1/dn
    fnp: np.ndarray   # (nz,)   fnp[k] = 0.5*dnw[k]   / dn[k]
    fnm: np.ndarray   # (nz,)   fnm[k] = 0.5*dnw[k-1] / dn[k]
    # --- Phase 2 hybrid-coordinate additions ---
    hybrid_opt: int = 0
    etac: float = 0.2
    # p_top (Pa) the pressure-weighted c2/c4 were finalized for; None until
    # make_base_state first finalizes the coord (make_vertical_coord's
    # placeholder install with pt = P0 records nothing).
    p_top: float | None = None
    c1f: np.ndarray | None = None  # (nz+1,) dB/deta at full levels
    c2f: np.ndarray | None = None  # (nz+1,) (1 - c1f)*(p0 - pt)
    c3f: np.ndarray | None = None  # (nz+1,) B(eta) at full levels
    c4f: np.ndarray | None = None  # (nz+1,) (eta - B)*(p0 - pt)
    c1h: np.ndarray | None = None  # (nz,)   half-level counterparts
    c2h: np.ndarray | None = None  # (nz,)
    c3h: np.ndarray | None = None  # (nz,)
    c4h: np.ndarray | None = None  # (nz,)


def hybrid_b_poly(etac: float) -> np.ndarray:
    """Coefficients (b1, b2, b3, b4) of the WRF v4 hybrid cubic B(eta).

    ``B(eta) = b1 + b2*eta + b3*eta^2 + b4*eta^3`` is uniquely determined by
    B(1) = 1, B'(1) = 1, B(etac) = 0, B'(etac) = 0 (ARW v4 Tech Note sec. 2);
    solved as a 4x4 linear system in float64.  The closed form in WRF v4.6.1
    (module_initialize_ideal.F B1..B5) satisfies the same four constraints;
    the test suite cross-checks the two constructions.
    """
    e = float(etac)
    A = np.array([[1.0, 1.0, 1.0, 1.0],          # B(1)  = 1
                  [0.0, 1.0, 2.0, 3.0],          # B'(1) = 1
                  [1.0, e, e * e, e ** 3],       # B(etac)  = 0
                  [0.0, 1.0, 2.0 * e, 3.0 * e * e]], dtype=np.float64)
    rhs = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    return np.linalg.solve(A, rhs)


def compute_hybrid_coeffs(znw: np.ndarray, hybrid_opt: int, etac: float,
                          p0: float, pt: float) -> dict[str, np.ndarray]:
    """WRF v4 hybrid coefficients c1/c2/c3/c4 on full and half levels.

    Transcribed from WRF v4.6.1 ``compute_vcoord_1d_coeffs``
    (dyn_em/nest_init_utils.F; identical code in module_initialize_ideal.F):
    ``c3 = B(eta)``, ``c4 = (eta - B)(p0 - pt)``, ``c1 = dB/deta`` by the
    same finite differences WRF uses (c1h from full-level c3/eta differences,
    interior c1f from half-level differences, endpoint c1f pinned to 1 at the
    surface and to 0 at the top for the true hybrid options), and
    ``c2 = (1 - c1)(p0 - pt)``.  ``hybrid_opt`` 0/1 give the exact identity
    B = eta (c1 = 1, c2 = 0, c4 = 0); 2 gives the Klemp cubic with B = 0
    above (below in eta) ``etac``.  All float64.
    """
    znw = np.asarray(znw, dtype=np.float64)
    if hybrid_opt in (0, 1):
        c3f = znw.copy()
    elif hybrid_opt == 2:
        b1, b2, b3, b4 = hybrid_b_poly(etac)
        c3f = b1 + b2 * znw + b3 * znw ** 2 + b4 * znw ** 3
        c3f[znw < etac] = 0.0
        c3f[0] = 1.0
        c3f[-1] = 0.0
    else:
        raise ValueError(
            f"hybrid_opt must be 0/1 (B=eta) or 2 (Klemp cubic), "
            f"got {hybrid_opt}")

    dp = p0 - pt
    znu = 0.5 * (znw[:-1] + znw[1:])
    c4f = (znw - c3f) * dp
    c3h = 0.5 * (c3f[:-1] + c3f[1:])
    c4h = (znu - c3h) * dp

    c1f = np.empty_like(znw)
    c1f[1:-1] = (c3h[1:] - c3h[:-1]) / (znu[1:] - znu[:-1])
    c1f[0] = 1.0
    c1f[-1] = 1.0 if hybrid_opt in (0, 1) else 0.0
    c2f = (1.0 - c1f) * dp
    c1h = (c3f[1:] - c3f[:-1]) / (znw[1:] - znw[:-1])
    c2h = (1.0 - c1h) * dp
    return {"c1f": c1f, "c2f": c2f, "c3f": c3f, "c4f": c4f,
            "c1h": c1h, "c2h": c2h, "c3h": c3h, "c4h": c4h}


#: Analytic base-state lapse used by WRF's ``module_initialize_real.F``
#: surface-pressure formula (``base_lapse``, WRF Registry default 50.0).
_BASE_LAPSE_K = 50.0


def _max_hybrid_slope(znw: np.ndarray, hybrid_opt: int, etac: float,
                      p_top: float) -> float:
    """Largest discrete ``dB/deta`` on either level set of one coordinate.

    The reference dry pressure is checked on full levels by WRF
    (``dyn_em/nest_init_utils.F:1158-1182``) and on half levels by this
    package's own base state; the tighter of the two governs.
    """

    hy = compute_hybrid_coeffs(np.asarray(znw, dtype=np.float64),
                               hybrid_opt, etac, c.P0, p_top)
    znu = 0.5 * (np.asarray(znw)[:-1] + np.asarray(znw)[1:])
    full = np.diff(hy["c3f"]) / np.diff(np.asarray(znw, dtype=np.float64))
    half = np.diff(hy["c3h"]) / np.diff(znu)
    return float(max(full.max(), half.max()))


def hybrid_surface_pressure_floor(znw: np.ndarray, hybrid_opt: int,
                                  etac: float, p_top: float) -> float:
    """Lowest surface pressure (Pa) this hybrid coordinate can order.

    Writing ``pd[k] = c3[k]*(ps - p_top) + c4[k] + p_top`` and
    ``s = dB/deta``, ``pd`` decreases with height only where
    ``s*(P0 - ps) < P0 - p_top``.  The steepest segment of the Klemp
    cubic therefore sets a hard floor under the surface pressure -- and
    so a ceiling over the terrain -- for a given ``etac``/``p_top`` pair.
    ``hybrid_opt`` 0/1 have ``B = eta`` exactly, hence no floor.
    """

    slope = _max_hybrid_slope(znw, hybrid_opt, etac, float(p_top))
    if slope <= 1.0:
        return 0.0
    return c.P0 - (c.P0 - float(p_top)) / slope


def analytic_base_terrain_height(surface_pressure: float,
                                 base_temp: float = 290.0) -> float:
    """Invert WRF's analytic base-state ``p_s(z)`` for terrain height.

    ``module_initialize_real.F:3787-3803`` sets
    ``p_s = p00*exp(-t00/a + sqrt((t00/a)^2 - 2*g*z/(a*Rd)))``; this is
    that relation solved for ``z``, used only to express a pressure floor
    in the metres a user chose the domain by.
    """

    ratio = float(base_temp) / _BASE_LAPSE_K
    inner = np.log(float(surface_pressure) / c.P0) + ratio
    return float((ratio ** 2 - inner ** 2) * _BASE_LAPSE_K * c.RD
                 / (2.0 * c.G))


def analytic_base_pressure(height: float,
                           base_temp: float = 290.0) -> float | None:
    """WRF's analytic base-state ``p_s(z)``: the forward relation.

    ``module_initialize_real.F:3787-3803`` again, unsolved this time --
    ``p_s = p00*exp(-t00/a + sqrt((t00/a)^2 - 2*g*z/(a*Rd)))`` -- so that
    a caller holding a model top in METRES can express it as the pressure
    :func:`base_layer_depths` needs, in the SAME base state that function
    inverts.  The pair round-trips: a coordinate resolved this way spans
    exactly ``[0, height]``, which is what makes it usable as a stand-in
    for a ladder a config declined to write out.

    ``None`` when the square root's argument goes negative.  The profile
    carries a 50 K lapse in ``ln p`` and so runs out of atmosphere at
    ``(t00/a)^2 * a*Rd/(2g)`` -- about 24.6 km for the 290 K default.  A
    model top above that has no representable base pressure here, and a
    caller must be told that rather than handed a fabricated one.
    """

    ratio = float(base_temp) / _BASE_LAPSE_K
    inner = ratio ** 2 - 2.0 * c.G * float(height) / (_BASE_LAPSE_K * c.RD)
    if inner < 0.0:
        return None
    return float(c.P0 * np.exp(-ratio + np.sqrt(inner)))


def base_layer_depths(znw: np.ndarray, hybrid_opt: int, etac: float,
                      p_top: float, base_temp: float = 290.0) -> np.ndarray:
    """Base-state layer depths in metres for one vertical coordinate.

    ``dz[k] = z_full[k+1] - z_full[k]`` at the reference column
    ``ps = P0``, where half- and full-level dry pressure is
    ``c3*(P0 - pt) + c4 + pt`` (:func:`compute_hybrid_coeffs`) and
    :func:`analytic_base_terrain_height` turns pressure into height.  No
    sounding, no state, no GPU: a loader can ask how deep this
    coordinate's layers are before anything has been allocated.

    These are the depths a vertical mixing length is built from, so they
    are also what a HORIZONTAL operator handed a vertical exchange
    coefficient has to be judged against
    (:func:`gpuwm.config.anisotropic_w_mixing_ratio`).
    """

    znw = np.asarray(znw, dtype=np.float64)
    hy = compute_hybrid_coeffs(znw, hybrid_opt, etac, c.P0, float(p_top))
    pd_full = hy["c3f"] * (c.P0 - float(p_top)) + hy["c4f"] + float(p_top)
    z_full = np.array([analytic_base_terrain_height(float(p), base_temp)
                       for p in pd_full])
    return np.diff(z_full)


def largest_supported_etac(znw: np.ndarray, p_top: float,
                           surface_pressure: float) -> float | None:
    """Largest ``etac`` whose reference column stays ordered at ``ps``.

    ``dB/deta`` grows with ``etac``, so the constraint is monotone and a
    bisection is exact to the printed precision.  ``None`` means no
    positive ``etac`` suffices -- the cubic's own minimum slope of 4/3
    already inverts the column, and only a lower ``p_top`` can help.
    """

    def ordered(candidate: float) -> bool:
        return float(surface_pressure) > hybrid_surface_pressure_floor(
            znw, 2, candidate, p_top)

    low = 1.0e-3
    if not ordered(low):
        return None
    high = 0.5
    if ordered(high):
        return high
    for _ in range(40):
        middle = 0.5 * (low + high)
        if ordered(middle):
            low = middle
        else:
            high = middle
    # A remedy is only useful if the number printed is itself usable, so
    # step down to the coarsest value that still orders the column rather
    # than reporting a supremum that rounds back onto the refusal.
    quantized = np.floor(low * 1000.0) / 1000.0
    while quantized > 0.0 and not ordered(quantized):
        quantized -= 0.001
    return float(quantized) if quantized > 0.0 else None


def hybrid_column_ordering_refusal(coord: VerticalCoord, p_top: float,
                                   surface_pressure, *, terrain=None,
                                   quantity: str = "reference dry pressure",
                                   row_offset: int = 0) -> str | None:
    """WRF's coordinate-validity refusal, with the constraint and remedy.

    Pristine WRF v4.6.1 makes exactly this check and calls it fatal --
    ``dyn_em/nest_init_utils.F:1158-1182``, whose own message says the
    cause "tends to be caused by very high topography" and whose only
    remedy is "reduce etac".  ``doc/README.hybrid_vert_coord:65-74``
    names the same failure over the Himalaya.  This returns that refusal
    with the numbers WRF's message leaves the user to find: which column,
    how high it is, the surface pressure the coordinate can actually
    order, and the largest ``etac`` (or ``p_top``) that would order this
    one.  Returns ``None`` when the column is representable.
    """

    pressure = np.asarray(surface_pressure, dtype=np.float64)
    floor = hybrid_surface_pressure_floor(
        coord.znw, coord.hybrid_opt, coord.etac, p_top)
    if floor <= 0.0 or not np.any(pressure <= floor):
        return None
    index = np.unravel_index(int(np.argmin(pressure)), pressure.shape)
    lowest = float(pressure[index])
    reported = tuple(int(value) for value in index)
    if row_offset and len(reported) >= 1:
        reported = (reported[0] + int(row_offset),) + reported[1:]
    if terrain is None:
        height = analytic_base_terrain_height(lowest)
        height_source = "implied"
    else:
        height = float(np.asarray(terrain, dtype=np.float64)[index])
        height_source = "terrain"
    etac_remedy = largest_supported_etac(coord.znw, p_top, lowest)
    slope = _max_hybrid_slope(coord.znw, coord.hybrid_opt, coord.etac, p_top)
    top_remedy = c.P0 - slope * (c.P0 - lowest)
    remedies = []
    if etac_remedy is not None and etac_remedy < coord.etac:
        remedies.append(f"reduce etac from {coord.etac:g} to at most "
                        f"{etac_remedy:.3f}")
    if 0.0 < top_remedy < float(p_top):
        remedies.append(f"lower p_top from {float(p_top):g} to at most "
                        f"{top_remedy:.0f} Pa")
    if not remedies:
        remedies.append("raise the model top and coarsen the terrain: no "
                        "positive etac orders a column this high")
    return (
        f"{quantity} is not monotonic: the hybrid coordinate "
        f"(hybrid_opt={coord.hybrid_opt}, etac={coord.etac:g}, "
        f"p_top={float(p_top):g} Pa) can only order columns whose surface "
        f"pressure stays above {floor:.0f} Pa "
        f"(about {analytic_base_terrain_height(floor):.0f} m of terrain), "
        f"and mass point {reported} is at "
        f"{lowest:.0f} Pa ({height_source} height {height:.0f} m) -- "
        f"{', or '.join(remedies)} "
        "(WRF v4.6.1 dyn_em/nest_init_utils.F:1158-1182 refuses the same "
        "column and names the same remedy)")


def _install_hybrid_coeffs(coord: VerticalCoord, pt: float) -> None:
    """(Re)build the hybrid coefficient arrays on ``coord`` for model-top
    pressure ``pt`` (in place: the pressure-weighted c2/c4 need p_top)."""
    hy = compute_hybrid_coeffs(coord.znw, coord.hybrid_opt, coord.etac,
                               c.P0, pt)
    for name, arr in hy.items():
        setattr(coord, name, arr)


def finalize_vertical_coord(coord: VerticalCoord, p_top: float) -> None:
    """Finalize pressure-bearing hybrid coefficients for an explicit top.

    Real-data initialization obtains ``p_top`` from the namelist rather than
    from an idealized sounding.  This is the public counterpart of the
    setup-time finalization performed by :func:`make_base_state`; it retains
    the same single-finalization guard.
    """
    p_top = float(p_top)
    if not (0.0 < p_top < c.P0):
        raise ValueError(f"p_top must lie in (0, {c.P0}), got {p_top}")
    if coord.p_top is None:
        _install_hybrid_coeffs(coord, p_top)
        coord.p_top = p_top
    elif coord.p_top != p_top:
        if coord.hybrid_opt not in (0, 1):
            raise ValueError(
                f"VerticalCoord already finalized for p_top={coord.p_top} Pa, "
                f"cannot reuse it for p_top={p_top} Pa")
        coord.p_top = p_top


def make_vertical_coord(nz: int, stretch: float | None = None,
                        hybrid_opt: int = 0, etac: float = 0.2,
                        eta_levels: np.ndarray | None = None,
                        ) -> VerticalCoord:
    """Build an eta coordinate with nz half levels.

    ``stretch=None`` gives the Phase 1 uniform spacing (bitwise unchanged).
    A positive ``stretch`` applies tanh clustering toward the surface,
    ``znw[k] = tanh(stretch*(1 - k/nz)) / tanh(stretch)``: layers thinnest
    at eta = 1 and thickest at the model top.  Any nonuniform spacing makes
    the fnm/fnp interpolation weights and rdn/rdnw spacings non-degenerate
    (on the uniform grid fnm == fnp == 0.5), which the acoustic matching
    tests exploit.

    ``hybrid_opt``/``etac`` select the WRF v4 hybrid coordinate (see
    ``compute_hybrid_coeffs``); the pressure-weighted c2/c4 coefficients are
    finalized by ``make_base_state`` once p_top is known.
    """
    if eta_levels is not None:
        if stretch is not None:
            raise ValueError("stretch and eta_levels are mutually exclusive")
        znw = np.asarray(eta_levels, dtype=np.float64).copy()
        if znw.shape != (nz + 1,):
            raise ValueError(f"eta_levels must have shape ({nz + 1},), "
                             f"got {znw.shape}")
        if not np.isfinite(znw).all():
            raise ValueError("eta_levels must be finite")
        if znw[0] != 1.0 or znw[-1] != 0.0 or not np.all(np.diff(znw) < 0.0):
            raise ValueError("eta_levels must decrease strictly from 1.0 to 0.0")
    elif stretch is None:
        znw = np.linspace(1.0, 0.0, nz + 1)
    else:
        if stretch <= 0.0:
            raise ValueError(f"stretch must be positive, got {stretch}")
        zeta = np.linspace(0.0, 1.0, nz + 1)     # 0 at surface, 1 at top
        znw = np.tanh(stretch * (1.0 - zeta)) / np.tanh(stretch)
        znw[0] = 1.0
        znw[-1] = 0.0
    znu = 0.5 * (znw[:-1] + znw[1:])
    dnw = znw[1:] - znw[:-1]
    rdnw = 1.0 / dnw

    dn = np.zeros(nz)
    rdn = np.zeros(nz)
    fnp = np.zeros(nz)
    fnm = np.zeros(nz)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    rdn[1:] = 1.0 / dn[1:]
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]

    coord = VerticalCoord(znw=znw, znu=znu, dnw=dnw, rdnw=rdnw,
                          dn=dn, rdn=rdn, fnp=fnp, fnm=fnm,
                          hybrid_opt=hybrid_opt, etac=etac)
    # c1/c3 are final here; c2/c4 (factor p0 - p_top) start at zero and are
    # finalized by make_base_state.  With pt = P0 the dp factor is exactly 0.
    _install_hybrid_coeffs(coord, pt=c.P0)
    return coord


@dataclass
class BaseState:
    """Hydrostatic base state.

    Flat terrain (Phase 1): ``mub`` is a scalar float and the profiles are
    1-D columns that broadcast against ``(nz, ny, nx)``.  With terrain,
    ``mub`` is ``(ny, nx)``, the profiles are full 3-D ``(nz[,+1], ny, nx)``
    arrays, and ``terrain_z`` records the surface height.
    """

    mub: float | np.ndarray    # base-state column dry-air mass, p_s - p_top (Pa)
    p_top: float               # pressure at model top (Pa)
    pb: np.ndarray             # (nz[,ny,nx])   base dry pressure at half levels
    alb: np.ndarray            # (nz[,ny,nx])   base inverse density
    thb: np.ndarray            # (nz[,ny,nx])   base potential temperature
    phb: np.ndarray            # (nz+1[,ny,nx]) base geopotential at full levels
    terrain_z: np.ndarray | None = None  # (ny, nx) surface height (None = flat)

    @property
    def t_init(self) -> np.ndarray:
        """Alias of ``thb`` (WRF name for the base theta profile)."""
        return self.thb


def make_base_state(coord: VerticalCoord, sounding: Sounding,
                    p_surf: float, ztop: float,
                    terrain_z: np.ndarray | None = None) -> BaseState:
    """Build a discretely hydrostatically balanced base state.

    Integrates the Exner function on a 1 m fine grid (with ``ztop`` appended
    so ``p_top = p(ztop)`` exactly, integer or not) to find p(z) for the
    given sounding, then evaluates the base profiles on the eta levels and
    integrates the discrete hydrostatic recurrence for ``phb`` with the
    hybrid column-mass increments ``c1h*mub + c2h``.

    With ``terrain_z (ny, nx)``: per-column surface pressure comes from the
    same hydrostatic integration evaluated at ``terrain_z[j,i]`` (the Exner
    profile is exactly piecewise linear between fine-grid nodes, so linear
    interpolation of pi is exact), dry mass ``mub[j,i] = p_s - p_top``, and
    half-level dry pressures ``pb = c3h*mub + c4h + p_top`` (WRF v4.6.1
    module_initialize_ideal.F lines 962-985); ``phb[0] = g*terrain_z``.

    Side effect: the FIRST call finalizes the pressure-weighted hybrid
    coefficients (c2f/c4f/c2h/c4h) on ``coord`` in place (they carry a
    factor ``(p0 - p_top)`` and p_top is first known here) and records
    ``coord.p_top``.  Reusing the coord with a different p_top raises
    ValueError for ``hybrid_opt=2`` instead of silently re-finalizing,
    which would corrupt every state built from the earlier base state;
    for ``hybrid_opt`` 0/1 the coefficients are identically zero for any
    p_top, so reuse is inert (and every output is bitwise Phase 1).
    """
    # 1. Fine-grid Exner integration: pi[n+1] = pi[n] - G*dz/(CP*theta(z_mid)).
    zf = np.append(np.arange(0.0, ztop, 1.0), ztop)
    dz = np.diff(zf)
    z_mid = 0.5 * (zf[:-1] + zf[1:])
    pi_surf = (p_surf / c.P0) ** c.RCP
    pi = np.concatenate(([pi_surf],
                         pi_surf - np.cumsum(c.G * dz / (c.CP * sounding(z_mid)))))
    p_of_z = c.P0 * pi ** (1.0 / c.RCP)

    # 2. Model top pressure; finalize the hybrid coefficients on the coord
    #    ONCE, recording p_top (final-review T2-1: reuse with a different
    #    p_top must not silently re-finalize c2/c4 under the first caller).
    p_top = float(p_of_z[-1])
    if coord.p_top is None:
        _install_hybrid_coeffs(coord, pt=p_top)
        coord.p_top = p_top
    elif coord.p_top != p_top:
        if coord.hybrid_opt not in (0, 1):
            raise ValueError(
                f"VerticalCoord already finalized for p_top={coord.p_top} "
                f"Pa but this base state has p_top={p_top} Pa "
                f"(hybrid_opt={coord.hybrid_opt}): the pressure-weighted "
                "c2/c4 coefficients carry a factor (p0 - p_top), so "
                "re-finalizing would corrupt every state built from the "
                "earlier base state.  Build a fresh coord with "
                "make_vertical_coord for the new model top."
            )
        # hybrid_opt 0/1: c2/c4 are identically zero for ANY p_top, so the
        # coefficients are p_top-independent; just re-record.
        coord.p_top = p_top
    c1h, c2h = coord.c1h, coord.c2h
    c3h, c4h = coord.c3h, coord.c4h

    if terrain_z is None:
        # 3. Flat: scalar column mass, 1-D profiles (Phase 1 broadcast form).
        mub = float(p_surf - p_top)
        pb = c3h * mub + c4h + p_top
        phb = np.zeros(coord.znw.size)
    else:
        # 3. Terrain: per-column surface pressure and 2-D dry mass.
        #    np.array (not asarray): BaseState.terrain_z must own a copy so
        #    later caller-side mutation cannot alias into the base state
        #    (final-review T2-5).
        terrain_z = np.array(terrain_z, dtype=np.float64)
        if terrain_z.ndim != 2:
            raise ValueError(f"terrain_z must be 2-D (ny, nx), got shape "
                             f"{terrain_z.shape}")
        if terrain_z.min() < 0.0 or terrain_z.max() >= ztop:
            raise ValueError(
                f"terrain_z must lie in [0, ztop): got range "
                f"[{terrain_z.min()}, {terrain_z.max()}] with ztop={ztop}")
        pi_s = np.interp(terrain_z, zf, pi)          # exact: pi is pw-linear
        p_s = c.P0 * pi_s ** (1.0 / c.RCP)
        mub = p_s - p_top                            # (ny, nx)
        pb = (c3h[:, None, None] * mub[None]
              + c4h[:, None, None] + p_top)          # (nz, ny, nx)
        phb = np.zeros((coord.znw.size,) + terrain_z.shape)
        phb[0] = c.G * terrain_z

    # 4. WRF's coordinate-validity check (compute_vcoord_1d_coeffs): the
    #    reference dry pressure must decrease with k in every column.  High
    #    terrain + large etac folds the hybrid coordinate (the cubic's
    #    dB/deta exceeds 1 near the surface, making c2h < 0, and columns
    #    with small mub then get c1h*mub + c2h < 0).
    if np.ndim(mub) == 0:
        pd_f = coord.c3f * mub + coord.c4f + p_top
    else:
        pd_f = (coord.c3f[:, None, None] * mub[None]
                + coord.c4f[:, None, None] + p_top)
    if not np.all(np.diff(pd_f, axis=0) < 0.0):
        raise ValueError(
            "hybrid reference dry pressure is not monotonically decreasing "
            f"(hybrid_opt={coord.hybrid_opt}, etac={coord.etac}): terrain "
            "too high for this etac - reduce etac (WRF "
            "compute_vcoord_1d_coeffs validity check)")

    # 5. Invert z(p) by interpolation; base theta and inverse density.
    #    np.interp needs increasing xp, and p decreases with z -> reverse.
    z_of_pb = np.interp(pb, p_of_z[::-1], zf[::-1])
    thb = sounding(z_of_pb)
    alb = c.RD * thb * (pb / c.P0) ** c.RCP / pb

    # 6. Discrete hydrostatic base geopotential (load-bearing recurrence).
    for k in range(coord.dnw.size):
        phb[k + 1] = phb[k] - coord.dnw[k] * (c1h[k] * mub + c2h[k]) * alb[k]

    return BaseState(mub=mub, p_top=p_top, pb=pb, alb=alb, thb=thb, phb=phb,
                     terrain_z=terrain_z)


def rebalance_hydrostatic(th_total: np.ndarray, mub: float,
                          coord: VerticalCoord, p_surf: float) -> np.ndarray:
    """Integrate the discrete hydrostatic recurrence with perturbed theta.

    Given total potential temperature ``th_total`` of shape (nz, ny, nx),
    returns the balanced total geopotential (nz+1, ny, nx).  Used by case
    initializers so ICs start balanced.  Phase-1 simplification: scalar
    ``mub``, exact for mu' = 0 initial conditions.
    """
    nz, ny, nx = th_total.shape
    p_top = p_surf - mub
    p = (coord.znu * mub + p_top)[:, None, None]           # (nz,1,1)
    alpha = c.RD * th_total * (p / c.P0) ** c.RCP / p       # (nz,ny,nx)

    ph = np.zeros((nz + 1, ny, nx))
    for k in range(nz):
        ph[k + 1] = ph[k] - coord.dnw[k] * mub * alpha[k]
    return ph
