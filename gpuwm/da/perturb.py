"""Initial-condition perturbations for storm-scale ensembles (experimental).

A 30-member ensemble whose members differ only by their random seed is not an
ensemble; it is thirty runs of the same forecast.  Spread has to be *put* into
the initial condition, and it has to be put in at scales the model can carry:
white noise on the grid is annihilated by the first diffusion call, and a
domain-wide offset is a bias, not a perturbation.  This module draws smooth
Gaussian random fields with a prescribed horizontal and vertical correlation
length, tapers them to zero against the lateral boundary so a member's
interior can diverge while its boundary forcing stays the unperturbed one,
and adds them to the prognostic state under hard physical bounds.

Construction
------------
For each field an independent unit-variance white-noise draw ``w`` is filtered
in spectral space by a separable Gaussian kernel

    ``H(k) = exp(-(kx^2 + ky^2) Lh^2 / 4) * exp(-kz^2 Lv^2 / 4)``

so the realized field has the Gaussian correlation ``C(r) = exp(-r^2/2L^2)``
(the autocorrelation of a Gaussian of width ``L/sqrt(2)``).  Its *radially
binned* power spectrum, which carries the ``2 pi k`` annulus Jacobian, is
therefore ``E(k) ~ k exp(-k^2 Lh^2 / 2)`` and peaks at ``k = 1/Lh`` exactly.
That peak is the module's contract with :func:`radial_power_spectrum` and is
what ``tests/test_da_perturb.py`` measures; a spectrum that peaks somewhere
else means the prescribed length scale is not the length scale you got.

Amplitudes are normalized *analytically*, not by dividing out the sample
standard deviation.  With unit-variance white noise and NumPy's DFT
convention the field variance is exactly ``mean_k |H(k)|^2``, and because
``H`` is separable that mean is a product of three cheap 1-D means.  Dividing
by the sample standard deviation would force every member to the same
realized variance, which is wrong: sample variance is supposed to fluctuate
between members.  The realized RMS of each draw is recorded in the provenance
instead.

What this does NOT do (v1, stated plainly)
------------------------------------------
* **No mass balance.**  ``mu'`` is untouched and the perturbed state is not
  re-balanced hydrostatically.  The perturbations are added to ``theta'`` and
  the moisture field directly; the column mass they imply is not imposed.
* **No wind balance.**  The ``u``/``v`` perturbations are not divergence-free
  and are not in gradient-wind or geostrophic balance with the temperature
  perturbation.  Expect the first minutes of the forecast to radiate gravity
  waves while the model adjusts.  For storm-scale ensembles with a short
  spin-up this is the accepted cost of a simple, auditable perturbation; a
  balanced route (a streamfunction/velocity-potential control vector, or a
  digital filter over the first steps) is future work.
* **No perturbation of the boundary forcing.**  Members share one boundary
  file.  The rim taper is what keeps that legal; it does not make the members
  independent out to the boundary, and it means ensemble spread decays toward
  the rim by construction.  :func:`recycled_difference_perturbations` and
  :func:`perturbed_lateral_boundaries` are documented stubs, not code.
* **No perturbation of surface, soil, or physics parameters**, and no
  perturbation of ``w``, ``mu'``, or any hydrometeor species.
* **No vertical taper.**  Perturbations reach the model top and the surface
  at full amplitude; only the lateral rim is tapered.
* **No flow dependence.**  The correlation length is prescribed and uniform,
  not derived from the analysis error covariance of the day.
* **The draw is periodic, because the FFT is, and the vertical wrap is NOT
  controlled.**  Opposite edges of the raw field are correlated.
  Horizontally the rim taper hides it completely -- both edges are
  multiplied by zero.  Vertically nothing hides it, and the quarter-column
  cap on ``vertical_scale_levels`` does not fix it: on a periodic column
  level ``0`` and level ``nz-1`` are ONE grid interval apart, not ``nz/2``,
  so their correlation is about ``exp(-1 / (2 Lv^2))`` -- which at the
  admitted cap (``nz=24``, ``Lv=6``) is ``0.983``, essentially locked
  together, and was previously documented as ``exp(-2) ~ 0.14``.  That
  claim confused the maximum circular separation with the seam and was
  simply wrong.  What the cap does control is the mid-column correlation,
  and even there periodic image covariance keeps it above the Gaussian
  value (measured ``0.297`` against a claimed ``0.135``).  Every draw with
  a nonzero vertical scale now reports its exact seam and half-column
  correlations in ``provenance["fields"][i]["vertical_wrap"]``, computed
  from the filter that was actually used; read them before interpreting
  any top-versus-bottom spread.

Nothing here is on a certified forecast path.  Every provenance dict this
module returns carries ``status="experimental"``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gpuwm.core import constants as c
from gpuwm.da import STATUS

__all__ = [
    "FieldPerturbation",
    "PerturbationConfig",
    "SUPPORTED_FIELDS",
    "PROVENANCE_SCHEMA",
    "apply_perturbations",
    "gaussian_random_field",
    "boundary_taper",
    "radial_power_spectrum",
    "spectral_peak_wavenumber",
    "fit_gaussian_length_scale",
    "default_array_module",
    "recycled_difference_perturbations",
    "perturbed_lateral_boundaries",
]

#: Provenance schema identifier.  Bump on any change to the emitted keys.
PROVENANCE_SCHEMA = "gpuwm.da.perturb/provenance/v1"

#: Domain-separation string mixed into every random stream key, so a seed
#: reused by an unrelated module cannot reproduce these draws.
_STREAM_DOMAIN = "gpuwm.da.perturb/noise/v1"

#: A prescribed length scale below this many grid spacings is not resolved.
_MIN_SCALE_CELLS = 2.0

#: A horizontal length scale above this fraction of the shorter domain span
#: makes the "prescribed length scale" meaningless -- the field is then a
#: domain-wide offset with a few wiggles, and its spectrum has no peak inside
#: the resolved band.
#:
#: The fraction is ``1/(2 pi)``, not the quarter this used to admit.  The
#: documented contract is that the radial spectrum peaks at ``k = 1/L``; on a
#: periodic span ``S`` the lowest nonzero angular wavenumber is ``2 pi / S``,
#: so that peak is resolved only when ``1/L >= 2 pi / S``, i.e.
#: ``L <= S / (2 pi) ~ 0.159 S``.  At the old quarter-span limit probes on
#: 32-, 64- and 128-point domains all measured ``peak * L = 2.356`` instead of
#: 1: the peak had fallen below the fundamental and the estimator was reading
#: the fundamental back.  This is a tightening, and configurations sitting
#: between 0.159 and 0.25 of the span are now refused rather than quietly
#: given a domain-wide offset.
_MAX_HORIZONTAL_SPAN_FRACTION = 1.0 / (2.0 * math.pi)
#: The vertical ceiling, which is a DIFFERENT and weaker claim than the
#: horizontal one.  It bounds how much of the column one coherent blob may
#: span; it does NOT control the FFT seam.  On a periodic column the top and
#: bottom levels are one interval apart, so their correlation is about
#: ``exp(-1/(2 Lv^2))`` -- 0.98 at this ceiling, not the ``exp(-2)`` this
#: constant used to claim.  See the module docstring, and
#: :func:`vertical_wrap_correlations`, which reports the real numbers per
#: draw.
_MAX_VERTICAL_SPAN_FRACTION = 0.25


# --------------------------------------------------------------------------
# Field table
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Target:
    """Where one perturbable field lives on the state, and in what units."""

    attribute: str
    stagger: str
    units: str
    #: True when the amplitude is a temperature and has to be divided by the
    #: Exner function before it can be added to the potential-temperature
    #: perturbation the state actually stores.
    exner_from_temperature: bool = False


#: The fields this module knows how to perturb.
#:
#: ``"t"`` and ``"theta"`` both land on ``state.thp``; they differ only in
#: what the configured amplitude means.  ``"t"`` is a temperature amplitude in
#: K and is converted with the Exner function (and therefore needs a
#: diagnosed pressure on the state); ``"theta"`` is a potential-temperature
#: amplitude in K and is added straight to ``thp``.  Requesting both is a
#: configuration error rather than a silent double perturbation.
SUPPORTED_FIELDS: Mapping[str, _Target] = {
    "t": _Target("thp", "mass", "K", exner_from_temperature=True),
    "theta": _Target("thp", "mass", "K"),
    "qv": _Target("qv", "mass", "kg/kg"),
    "u": _Target("u", "u_face", "m s-1"),
    "v": _Target("v", "v_face", "m s-1"),
}

#: Fields are applied in this order regardless of the order they appear in
#: the configuration, so two configurations that list the same perturbations
#: differently produce byte-identical states.  Temperature moves before
#: moisture because the supersaturation cap is evaluated against the
#: *perturbed* temperature.
_APPLICATION_ORDER = ("t", "theta", "qv", "u", "v")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldPerturbation:
    """One field's perturbation amplitude and correlation lengths.

    ``amplitude`` is the ensemble 1-sigma of the perturbation in the field's
    own units (see :data:`SUPPORTED_FIELDS`), *before* the rim taper.  The
    realized standard deviation of any single draw fluctuates around it, and
    inside the rim it is deliberately smaller.

    ``vertical_scale_levels`` is measured in model levels, not metres,
    because that is the only vertical coordinate this module can see without
    reaching into the base state; ``0`` decorrelates the levels entirely.
    """

    name: str
    amplitude: float
    length_scale_km: float
    vertical_scale_levels: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_FIELDS:
            raise ValueError(
                f"unknown perturbation field {self.name!r}; supported: "
                + ", ".join(sorted(SUPPORTED_FIELDS)))
        for label, value in (("amplitude", self.amplitude),
                             ("length_scale_km", self.length_scale_km),
                             ("vertical_scale_levels",
                              self.vertical_scale_levels)):
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{self.name}: {label} must be finite, got {value!r}")
        if self.amplitude < 0.0:
            raise ValueError(
                f"{self.name}: amplitude must be non-negative, got "
                f"{self.amplitude!r} (a negative 1-sigma is not a sign flip, "
                "it is a typo)")
        if self.length_scale_km <= 0.0:
            raise ValueError(
                f"{self.name}: length_scale_km must be positive, got "
                f"{self.length_scale_km!r}")
        if self.vertical_scale_levels < 0.0:
            raise ValueError(
                f"{self.name}: vertical_scale_levels must be non-negative "
                f"(0 decorrelates the levels), got "
                f"{self.vertical_scale_levels!r}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FieldPerturbation":
        """Build one spec from a configuration table."""
        known = {"name", "amplitude", "length_scale_km",
                 "vertical_scale_levels"}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                "unknown perturbation field keys: " + ", ".join(unknown))
        missing = sorted({"name", "amplitude", "length_scale_km"}
                         - set(mapping))
        if missing:
            raise ValueError(
                "perturbation field is missing required keys: "
                + ", ".join(missing))
        return cls(
            name=str(mapping["name"]),
            amplitude=float(mapping["amplitude"]),
            length_scale_km=float(mapping["length_scale_km"]),
            vertical_scale_levels=float(
                mapping.get("vertical_scale_levels", 0.0)),
        )


@dataclass(frozen=True)
class PerturbationConfig:
    """Everything :func:`apply_perturbations` needs besides the state+seed.

    ``rim_width`` is in grid cells and must be at least 1: there is no way to
    ask this module to perturb the outermost row, because a member whose
    boundary row disagrees with the shared boundary file is a member with a
    discontinuity, not a member with spread.

    ``rh_cap`` is the relative-humidity ceiling the perturbed moisture field
    is clipped to (``1.0`` = no supersaturation).  ``None`` disables the cap
    entirely, which also removes the module's only need for a diagnosed
    pressure -- but only if no ``"t"`` perturbation is requested, since the
    Exner conversion needs pressure too.
    """

    dx_km: float
    dy_km: float
    fields: tuple[FieldPerturbation, ...]
    rim_width: int = 5
    rim_taper: str = "cosine"
    qv_floor: float = 0.0
    rh_cap: float | None = 1.0
    compute_dtype: str = "float64"

    def __post_init__(self) -> None:
        for label, value in (("dx_km", self.dx_km), ("dy_km", self.dy_km)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"{label} must be a positive finite length, got "
                    f"{value!r}")
        specs = tuple(self.fields)
        if not specs:
            raise ValueError(
                "PerturbationConfig.fields is empty: a perturbation "
                "configuration that perturbs nothing is a silent no-op")
        for spec in specs:
            if not isinstance(spec, FieldPerturbation):
                raise TypeError(
                    "PerturbationConfig.fields must hold FieldPerturbation "
                    f"instances, got {type(spec).__name__}")
        names = [spec.name for spec in specs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                "duplicate perturbation fields: " + ", ".join(duplicates))
        if "t" in names and "theta" in names:
            raise ValueError(
                "'t' and 'theta' both perturb state.thp; configure exactly "
                "one of them ('t' is a temperature amplitude converted with "
                "the Exner function, 'theta' is a potential-temperature "
                "amplitude applied directly)")
        object.__setattr__(self, "fields", specs)

        if int(self.rim_width) != self.rim_width or int(self.rim_width) < 1:
            raise ValueError(
                f"rim_width must be an integer >= 1 grid cell, got "
                f"{self.rim_width!r}; there is no 'no taper' setting, "
                "because an untapered member contradicts its own boundary "
                "forcing")
        object.__setattr__(self, "rim_width", int(self.rim_width))
        if self.rim_taper not in ("cosine", "linear"):
            raise ValueError(
                f"rim_taper must be 'cosine' or 'linear', got "
                f"{self.rim_taper!r}")
        if not math.isfinite(float(self.qv_floor)) or self.qv_floor < 0.0:
            raise ValueError(
                f"qv_floor must be finite and non-negative, got "
                f"{self.qv_floor!r}")
        if self.rh_cap is not None:
            if not math.isfinite(float(self.rh_cap)) or self.rh_cap <= 0.0:
                raise ValueError(
                    f"rh_cap must be positive and finite (or None to "
                    f"disable), got {self.rh_cap!r}")
        if self.compute_dtype not in ("float32", "float64"):
            raise ValueError(
                f"compute_dtype must be 'float32' or 'float64', got "
                f"{self.compute_dtype!r}")

    @property
    def field_names(self) -> tuple[str, ...]:
        """Configured field names in the canonical application order."""
        configured = {spec.name for spec in self.fields}
        return tuple(n for n in _APPLICATION_ORDER if n in configured)

    def spec(self, name: str) -> FieldPerturbation:
        """Return the configured spec for one field name."""
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PerturbationConfig":
        """Build a configuration from a TOML/JSON table.

        ``fields`` may be a list of tables (each with ``name``) or a table
        keyed by field name.
        """
        known = {"dx_km", "dy_km", "fields", "rim_width", "rim_taper",
                 "qv_floor", "rh_cap", "compute_dtype"}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                "unknown perturbation config keys: " + ", ".join(unknown))
        missing = sorted({"dx_km", "dy_km", "fields"} - set(mapping))
        if missing:
            raise ValueError(
                "perturbation config is missing required keys: "
                + ", ".join(missing))
        raw = mapping["fields"]
        if isinstance(raw, Mapping):
            specs = tuple(
                FieldPerturbation.from_mapping({"name": name, **dict(entry)})
                for name, entry in raw.items())
        else:
            specs = tuple(FieldPerturbation.from_mapping(entry)
                          for entry in raw)
        kwargs: dict[str, Any] = {
            "dx_km": float(mapping["dx_km"]),
            "dy_km": float(mapping["dy_km"]),
            "fields": specs,
        }
        if "rim_width" in mapping:
            kwargs["rim_width"] = int(mapping["rim_width"])
        if "rim_taper" in mapping:
            kwargs["rim_taper"] = str(mapping["rim_taper"])
        if "qv_floor" in mapping:
            kwargs["qv_floor"] = float(mapping["qv_floor"])
        if "rh_cap" in mapping:
            cap = mapping["rh_cap"]
            kwargs["rh_cap"] = None if cap is None else float(cap)
        if "compute_dtype" in mapping:
            kwargs["compute_dtype"] = str(mapping["compute_dtype"])
        return cls(**kwargs)


# --------------------------------------------------------------------------
# Array-module plumbing
# --------------------------------------------------------------------------

def default_array_module():
    """CuPy when it is importable and a device is visible, else NumPy."""
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return np
    return cupy


def _array_module(array):
    """The array module that owns ``array``, without importing CuPy first."""
    if isinstance(array, np.ndarray):
        return np
    root = type(array).__module__.split(".")[0]
    if root == "cupy":
        import cupy
        return cupy
    raise TypeError(
        f"unsupported array type {type(array).__module__}."
        f"{type(array).__name__}: expected a NumPy or CuPy array")


def _to_host(array) -> np.ndarray:
    """Copy any supported array to host without assuming which backend."""
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array.get())


#: Tri-state cache for the one-off cuFFT probe: None = not yet asked.
_DEVICE_FFT_AVAILABLE: bool | None = None


def _device_fft_available(xp) -> bool:
    """Whether ``xp`` can actually run an FFT, probed once and remembered.

    A CuPy install can have a perfectly good allocator, kernels, and
    reductions while ``cupy.cuda.cufft`` fails to load -- most commonly when
    the wheel's CUDA minor version and the installed toolkit's cuFFT
    soname disagree.  Asking the library whether it *has* cuFFT is not the
    same as asking whether cuFFT *loads*, so this runs a two-cubed transform
    and believes the answer.
    """
    global _DEVICE_FFT_AVAILABLE
    if xp is np:
        return True
    if _DEVICE_FFT_AVAILABLE is None:
        try:
            xp.fft.rfftn(xp.zeros((2, 2, 2), dtype=np.float64),
                         axes=(0, 1, 2))
        except Exception:
            _DEVICE_FFT_AVAILABLE = False
        else:
            _DEVICE_FFT_AVAILABLE = True
    return _DEVICE_FFT_AVAILABLE


# --------------------------------------------------------------------------
# Deterministic noise
# --------------------------------------------------------------------------

def _stream_key(seed: int, name: str, shape: Sequence[int]) -> int:
    """A 128-bit Philox key derived from ``(seed, field name, shape)``.

    Deriving the key by hash rather than by ``seed + offset`` means adjacent
    seeds give unrelated streams, two fields of one member never share a
    stream, and the same field on a differently shaped grid is a different
    draw (so a resolution change cannot silently reproduce the old noise on a
    subset of points).
    """
    dims = ",".join(str(int(extent)) for extent in shape)
    payload = f"{_STREAM_DOMAIN}|seed={int(seed)}|field={name}|shape=({dims})"
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:16], "big")


def _white_noise(shape: Sequence[int], *, seed: int, name: str,
                 dtype) -> tuple[np.ndarray, int, str]:
    """Unit-variance host white noise plus its key and SHA-256.

    The draw happens on the host with NumPy's Philox counter-based generator
    even when the filtering will run on the GPU.  That is deliberate: it
    makes ``noise_sha256`` identical on a CUDA box and a CPU box, so the
    provenance stamp identifies the *perturbation* rather than the machine.
    The filtered field itself is not bit-identical across backends -- cuFFT
    and pocketfft round differently -- and this module does not claim it is.
    """
    key = _stream_key(seed, name, shape)
    generator = np.random.Generator(np.random.Philox(key=key))
    noise = generator.standard_normal(tuple(int(s) for s in shape),
                                      dtype=dtype)
    digest = hashlib.sha256(noise.tobytes()).hexdigest()
    return noise, key, digest


# --------------------------------------------------------------------------
# Gaussian random fields
# --------------------------------------------------------------------------

def _axis_filter(n: int, spacing: float, length_scale: float,
                 *, half: bool) -> np.ndarray:
    """Per-axis Gaussian amplitude filter on the (half-)spectrum."""
    freq = (np.fft.rfftfreq(n, d=spacing) if half
            else np.fft.fftfreq(n, d=spacing))
    k = 2.0 * np.pi * freq
    return np.exp(-0.25 * (k * length_scale) ** 2)


def _analytic_variance(n: int, spacing: float,
                       length_scale: float) -> float:
    """``mean_k |H(k)|^2`` on one full axis.

    With unit-variance white noise and NumPy's unnormalized forward DFT, the
    filtered field's variance is the mean of ``|H|^2`` over the *full* k-grid;
    because ``H`` is separable the 3-D mean is the product of these.
    """
    return float(np.mean(
        _axis_filter(n, spacing, length_scale, half=False) ** 2))


def vertical_wrap_correlations(nz: int,
                               vertical_scale_levels: float) -> dict[str, Any]:
    """The column's real periodic correlations, from the filter in use.

    The field is white noise multiplied by the separable amplitude filter
    ``H``, so along the vertical axis its circular autocorrelation at lag
    ``d`` is exactly

    ``rho(d) = sum_k |H(k)|^2 cos(2 pi k d / nz) / sum_k |H(k)|^2``

    -- no sampling, no fitting, no approximation.  Evaluated at ``d=1``
    (adjacent interior levels), at ``d=nz-1`` (the FFT SEAM: on a circle
    the top and bottom levels are one interval apart, which is why the
    seam correlation is close to the adjacent one and nowhere near the
    ``exp(-(nz/2)^2/(2 Lv^2))`` the cap used to be justified by), and at
    ``d=nz//2`` (the maximum circular separation, which is the quantity
    the quarter-column cap actually bounds).

    ``gaussian_random_field`` puts this in its ``info`` and
    :func:`apply_perturbations` puts it in every field record, so the
    number a reader needs is in the manifest instead of in a docstring.
    """
    nz = int(nz)
    scale = float(vertical_scale_levels)
    if nz < 2:
        return {
            "vertical_scale_levels": scale,
            "note": "a single-level column has no vertical correlation",
        }
    power = _axis_filter(nz, 1.0, scale, half=False) ** 2
    total = float(power.sum())
    if not (total > 0.0):
        return {
            "vertical_scale_levels": scale,
            "note": "the vertical filter has no power on this column",
        }
    modes = np.fft.fftfreq(nz, d=1.0) * nz

    def rho(lag: int) -> float:
        return float((power * np.cos(2.0 * np.pi * modes * lag / nz)).sum()
                     / total)

    seam = rho(nz - 1)
    return {
        "vertical_scale_levels": scale,
        "levels": nz,
        "adjacent_interior": rho(1),
        #: Level 0 against level nz-1 -- the periodic seam.
        "top_to_bottom_seam": seam,
        #: The largest circular separation, which is what the cap bounds.
        "half_column": rho(nz // 2),
        "method": ("exact circular autocorrelation of the applied "
                   "amplitude filter: sum_k |H(k)|^2 cos(2 pi k d/nz) / "
                   "sum_k |H(k)|^2"),
        "caveat": (
            "the top and bottom levels are ONE grid interval apart on a "
            "periodic column, so top_to_bottom_seam tracks "
            "adjacent_interior and is not controlled by the "
            "quarter-column cap on vertical_scale_levels"),
    }


def _check_resolvable(shape: Sequence[int], dx_km: float, dy_km: float,
                      spec: FieldPerturbation) -> None:
    """Refuse length scales the grid cannot represent, in either direction."""
    nz, ny, nx = (int(shape[0]), int(shape[1]), int(shape[2]))
    cell = max(float(dx_km), float(dy_km))
    floor = _MIN_SCALE_CELLS * cell
    if spec.length_scale_km < floor:
        raise ValueError(
            f"{spec.name}: length_scale_km={spec.length_scale_km} is below "
            f"{_MIN_SCALE_CELLS:g} grid spacings ({floor:g} km on a "
            f"{dx_km:g} x {dy_km:g} km grid); the grid cannot carry it and "
            "the model's diffusion would remove what is left")
    span = min(nx * float(dx_km), ny * float(dy_km))
    ceiling = _MAX_HORIZONTAL_SPAN_FRACTION * span
    if spec.length_scale_km > ceiling:
        raise ValueError(
            f"{spec.name}: length_scale_km={spec.length_scale_km} exceeds "
            f"span/(2*pi) = {ceiling:g} km on the shorter domain span "
            f"({span:g} km); the documented spectral peak k=1/L is resolved "
            f"only at or below that limit, because the lowest nonzero "
            f"wavenumber a periodic span of {span:g} km carries is "
            f"2*pi/{span:g}. Above it the draw is a domain-wide offset and "
            "'prescribed length scale' stops meaning anything")
    if spec.vertical_scale_levels > 0.0:
        if spec.vertical_scale_levels < _MIN_SCALE_CELLS:
            raise ValueError(
                f"{spec.name}: vertical_scale_levels="
                f"{spec.vertical_scale_levels} is below "
                f"{_MIN_SCALE_CELLS:g} levels; use 0 to decorrelate the "
                "levels outright rather than a sub-level scale that only "
                "looks like a correlation")
        vertical_ceiling = _MAX_VERTICAL_SPAN_FRACTION * nz
        if spec.vertical_scale_levels > vertical_ceiling:
            raise ValueError(
                f"{spec.name}: vertical_scale_levels="
                f"{spec.vertical_scale_levels} exceeds "
                f"{_MAX_VERTICAL_SPAN_FRACTION:g} of the {nz}-level column "
                f"(limit {vertical_ceiling:g})")


def gaussian_random_field(shape: Sequence[int], *, seed: int, name: str,
                          dx_km: float, dy_km: float,
                          length_scale_km: float,
                          vertical_scale_levels: float = 0.0,
                          xp=None, dtype: str = "float64",
                          ) -> tuple[Any, dict[str, Any]]:
    """A unit-variance Gaussian random field of ``shape`` ``(nz, ny, nx)``.

    Returns ``(field, info)``.  ``info`` carries the stream key, the SHA-256
    of the white-noise draw, the analytic and realized RMS, and the backend
    the FFT ran on -- everything :func:`apply_perturbations` needs to build
    its provenance, and enough for a caller using this helper directly to
    reproduce the draw.

    The field's *expected* variance is exactly 1 by analytic normalization;
    its realized variance fluctuates like any finite sample, which is the
    behaviour an ensemble wants.
    """
    shape = tuple(int(extent) for extent in shape)
    if len(shape) != 3 or any(extent < 1 for extent in shape):
        raise ValueError(
            f"gaussian_random_field needs a positive (nz, ny, nx) shape, "
            f"got {shape}")
    if dtype not in ("float32", "float64"):
        raise ValueError(
            f"dtype must be 'float32' or 'float64', got {dtype!r}")
    if xp is None:
        xp = default_array_module()
    nz, ny, nx = shape
    np_dtype = np.float32 if dtype == "float32" else np.float64

    noise, key, digest = _white_noise(shape, seed=seed, name=name,
                                      dtype=np_dtype)

    # Filter wherever the FFT actually works; move the finished field to the
    # requested backend afterwards.  A half-device pipeline would be worse
    # than either whole one.
    fft_xp = xp if _device_fft_available(xp) else np
    working = noise if fft_xp is np else fft_xp.asarray(noise)

    spectrum = fft_xp.fft.rfftn(working, axes=(0, 1, 2))
    hz = _axis_filter(nz, 1.0, float(vertical_scale_levels), half=False)
    hy = _axis_filter(ny, float(dy_km), float(length_scale_km), half=False)
    hx = _axis_filter(nx, float(dx_km), float(length_scale_km), half=True)
    kernel = (hz[:, None, None] * hy[None, :, None]
              * hx[None, None, :]).astype(np_dtype, copy=False)
    if fft_xp is not np:
        kernel = fft_xp.asarray(kernel)
    spectrum = spectrum * kernel
    field = fft_xp.fft.irfftn(spectrum, s=shape, axes=(0, 1, 2))

    variance = (_analytic_variance(nz, 1.0, float(vertical_scale_levels))
                * _analytic_variance(ny, float(dy_km),
                                     float(length_scale_km))
                * _analytic_variance(nx, float(dx_km),
                                     float(length_scale_km)))
    if not (variance > 0.0) or not math.isfinite(variance):
        raise RuntimeError(
            f"{name}: the Gaussian filter has no power on this grid "
            f"(analytic variance {variance!r}); the requested scales are "
            "degenerate")
    field = field / math.sqrt(variance)
    field = field.astype(np_dtype, copy=False)

    realized = float(fft_xp.sqrt(fft_xp.mean(field.astype(np.float64) ** 2)))
    if fft_xp is not xp:
        field = xp.asarray(field)
    info = {
        "shape": shape,
        "stream_key_hex": f"{key:032x}",
        "noise_sha256": digest,
        "noise_dtype": dtype,
        #: RMS of the filtered field *before* normalization; the returned
        #: field was divided by exactly this, so its expected RMS is 1.
        "pre_normalization_analytic_rms": math.sqrt(variance),
        "realized_rms": realized,
        "backend": "numpy" if xp is np else "cupy",
        "fft_backend": "numpy" if fft_xp is np else "cupy",
        "length_scale_km": float(length_scale_km),
        "vertical_scale_levels": float(vertical_scale_levels),
        # The periodic column's real correlations, including the seam the
        # quarter-column cap does NOT control.  Reported on every draw so
        # a reader never has to take the docstring's word for it.
        "vertical_wrap": vertical_wrap_correlations(
            nz, float(vertical_scale_levels)),
    }
    return field, info


# --------------------------------------------------------------------------
# Boundary taper
# --------------------------------------------------------------------------

def boundary_taper(ny: int, nx: int, rim_width: int, *, kind: str = "cosine",
                   xp=None, dtype=np.float64):
    """A ``(ny, nx)`` rim taper: exactly 0 on the edge, exactly 1 inside.

    The taper is a function of the *minimum* index distance to any of the
    four lateral edges, so it is a frame rather than a separable product;
    corners are treated the same as edges instead of being doubly damped.

    Both endpoints are exact in IEEE arithmetic, not merely close: at
    ``d = 0`` the cosine form evaluates ``0.5 * (1 - cos 0) = 0`` and at
    ``d >= rim_width`` it evaluates ``0.5 * (1 - cos pi) = 1``, because
    ``cos(pi)`` is exactly ``-1``.  ``tests/test_da_perturb.py`` asserts
    equality, not tolerance.
    """
    ny, nx, rim_width = int(ny), int(nx), int(rim_width)
    if ny < 1 or nx < 1:
        raise ValueError(f"boundary_taper needs a positive shape, got "
                         f"({ny}, {nx})")
    if rim_width < 1:
        raise ValueError(
            f"rim_width must be >= 1 grid cell, got {rim_width}")
    if kind not in ("cosine", "linear"):
        raise ValueError(f"unknown taper kind {kind!r}")
    if xp is None:
        xp = np
    if 2 * rim_width >= min(ny, nx):
        raise ValueError(
            f"rim_width={rim_width} leaves no untapered interior on a "
            f"({ny}, {nx}) field; the two rims meet")

    dj = np.minimum(np.arange(ny), ny - 1 - np.arange(ny))
    di = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))
    distance = np.minimum(dj[:, None], di[None, :]).astype(np.float64)
    ratio = np.minimum(distance / float(rim_width), 1.0)
    if kind == "cosine":
        taper = 0.5 * (1.0 - np.cos(np.pi * ratio))
    else:
        taper = ratio
    taper = taper.astype(dtype, copy=False)
    return taper if xp is np else xp.asarray(taper)


# --------------------------------------------------------------------------
# Spectral diagnostic
# --------------------------------------------------------------------------

def radial_power_spectrum(field, dx_km: float, dy_km: float | None = None,
                          *, bins: int | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Annulus-summed 2-D power spectrum ``E(k)`` of a field.

    Accepts one ``(ny, nx)`` level or a ``(nz, ny, nx)`` stack, in which case
    the per-level spectra are averaged -- with vertically decorrelated levels
    that is an average over ``nz`` independent realizations and it is the
    difference between a peak you can put a tolerance on and one you cannot.

    Returns ``(k, energy)`` with ``k`` in radians per kilometre and

        ``energy[b] = k[b] * mean(|F(k)|^2 over annulus b)``

    The ``k`` factor is the annulus Jacobian -- it is what makes a
    Gaussian-correlated field's spectrum *peak*, at ``k = 1/L``, instead of
    decaying monotonically from ``k = 0``.  It is applied analytically to the
    per-mode *mean* rather than by summing the annulus, because the number of
    discrete modes actually landing in a low-wavenumber annulus is small and
    lumpy, and a storm-scale correlation length puts its peak exactly there.
    Summing instead of averaging moves the measured peak by tens of percent
    from one realization to the next; this form does not.

    The zero mode is dropped: a constant offset is not a length scale.
    Empty annuli come back as zero energy.
    """
    host = _to_host(field)
    if host.ndim == 2:
        host = host[None, :, :]
    if host.ndim != 3:
        raise ValueError(
            f"radial_power_spectrum takes an (ny, nx) level or an "
            f"(nz, ny, nx) stack, got shape {np.shape(field)}")
    nlev, ny, nx = host.shape
    if dy_km is None:
        dy_km = dx_km
    if dx_km <= 0.0 or dy_km <= 0.0:
        raise ValueError("grid spacings must be positive")

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=float(dx_km))
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=float(dy_km))
    kmag = np.hypot(ky[:, None], kx[None, :])

    # Bin only out to the smaller of the two Nyquist wavenumbers, so no
    # annulus is partly outside the sampled rectangle and thus artificially
    # starved of modes.
    kmax = float(min(np.abs(kx).max(), np.abs(ky).max()))
    if bins is None:
        bins = max(8, min(ny, nx) // 2)
    bins = int(bins)
    edges = np.linspace(0.0, kmax, bins + 1)
    flat_k = kmag.ravel()
    keep = (flat_k > 0.0) & (flat_k <= kmax)
    index = np.clip(np.digitize(flat_k[keep], edges) - 1, 0, bins - 1)

    counts = np.bincount(index, minlength=bins).astype(np.float64)
    transform = np.fft.fft2(host.astype(np.float64), axes=(1, 2))
    power = (np.abs(transform) ** 2).reshape(nlev, -1)[:, keep]
    total = np.zeros(bins, dtype=np.float64)
    for level in range(nlev):
        total += np.bincount(index, weights=power[level], minlength=bins)
    occupied = counts > 0.0
    per_mode = np.zeros(bins, dtype=np.float64)
    per_mode[occupied] = total[occupied] / (counts[occupied] * float(nlev))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, centres * per_mode


def spectral_peak_wavenumber(k: np.ndarray, energy: np.ndarray) -> float:
    """Sub-bin peak of a radial spectrum, by log-parabolic interpolation.

    The bin containing ``1/L`` can be several percent wide at the
    wavenumbers a storm-scale perturbation lives at, so taking the argmax
    bin centre reports the binning as much as the field.  Fitting a parabola
    to ``log E`` over the peak bin and its two neighbours -- exact for a
    Gaussian, which this spectrum locally is -- removes that.
    """
    k = np.asarray(k, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    if k.shape != energy.shape or k.ndim != 1 or k.size < 3:
        raise ValueError("k and energy must be matching 1-D arrays of >= 3")
    peak = int(np.argmax(energy))
    if peak == 0 or peak == k.size - 1:
        return float(k[peak])
    left, centre, right = energy[peak - 1:peak + 2]
    if not (left > 0.0 and centre > 0.0 and right > 0.0):
        return float(k[peak])
    ll, lc, lr = math.log(left), math.log(centre), math.log(right)
    denominator = ll - 2.0 * lc + lr
    if denominator == 0.0:
        return float(k[peak])
    offset = 0.5 * (ll - lr) / denominator
    if not (-1.0 <= offset <= 1.0):
        return float(k[peak])
    spacing = float(k[1] - k[0])
    return float(k[peak] + offset * spacing)


def fit_gaussian_length_scale(k: np.ndarray, energy: np.ndarray, *,
                              band: tuple[float, float] = (0.3, 3.0)
                              ) -> float:
    """Recover ``L`` from a radial spectrum, in the same units as ``1/k``.

    The inverse of what :func:`gaussian_random_field` promises.  Removing the
    annulus Jacobian leaves ``P(k) = E(k)/k``, and for the Gaussian
    correlation this module imposes ``log P`` is *exactly* linear in ``k^2``
    with slope ``-L^2/2``.  A straight-line fit over a band around the peak
    therefore pins ``L`` far more tightly than reading the peak off the axis
    does -- and it checks the whole spectral shape, not one point of it.

    ``band`` is the fitted range as a multiple of the measured peak
    wavenumber.
    """
    k = np.asarray(k, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    peak = spectral_peak_wavenumber(k, energy)
    low, high = float(band[0]) * peak, float(band[1]) * peak
    usable = (energy > 0.0) & (k > 0.0) & (k >= low) & (k <= high)
    if int(np.count_nonzero(usable)) < 4:
        raise ValueError(
            "not enough occupied spectral bins around the peak to fit a "
            "length scale; use a larger domain or fewer bins")
    slope = np.polyfit(k[usable] ** 2,
                       np.log(energy[usable] / k[usable]), 1)[0]
    if slope >= 0.0:
        raise ValueError(
            "the spectrum does not fall off with wavenumber (fitted slope "
            f"{slope!r}); this is not a Gaussian-correlated field")
    return float(math.sqrt(-2.0 * slope))


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

def _expected_shape(name: str, nz: int, ny: int, nx: int
                    ) -> tuple[int, int, int]:
    """The ARW-staggered shape one perturbable field must have."""
    stagger = SUPPORTED_FIELDS[name].stagger
    if stagger == "mass":
        return (nz, ny, nx)
    if stagger == "u_face":
        return (nz, ny, nx + 1)
    if stagger == "v_face":
        return (nz, ny + 1, nx)
    raise AssertionError(f"unhandled stagger {stagger!r}")


def _resolve_target(state, name: str):
    """Fetch the state array for one field, failing closed if it is absent."""
    attribute = SUPPORTED_FIELDS[name].attribute
    array = getattr(state, attribute, None)
    if array is None:
        raise ValueError(
            f"cannot perturb {name!r}: state.{attribute} is None "
            "(a dry state has no moisture arrays; perturbing qv on it would "
            "be a silent no-op)")
    if not hasattr(array, "shape"):
        raise TypeError(
            f"state.{attribute} is not an array (got "
            f"{type(array).__name__})")
    return array


def _total_theta(state, xp):
    """Full potential temperature, whether ``thb`` is a column or 3-D."""
    if hasattr(state, "total_theta"):
        return state.total_theta()
    thb = getattr(state, "thb", None)
    if thb is None:
        raise ValueError(
            "state carries neither total_theta() nor thb; the "
            "supersaturation cap and the Exner conversion both need the "
            "full potential temperature")
    base = thb if getattr(thb, "ndim", 0) == 3 else thb[:, None, None]
    return base + state.thp


def _require_pressure(state, xp, reason: str):
    """Return a strictly positive full pressure or refuse to continue."""
    pressure = getattr(state, "p", None)
    if pressure is None:
        raise ValueError(
            f"{reason} needs state.p (full pressure) and the state has none")
    minimum = float(xp.min(pressure))
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError(
            f"{reason} needs a diagnosed positive pressure; state.p has "
            f"minimum {minimum!r}. An un-diagnosed (all-zero) pressure "
            "field would silently produce infinite Exner factors, so this "
            "fails rather than guesses")
    return pressure


def _saturation_mixing_ratio(temperature, pressure, xp):
    """Tetens ``qvs`` over liquid water, in WRF's constants.

    Uses ``SVP1/SVP2/SVP3/SVPT0`` and ``EP2`` from
    :mod:`gpuwm.core.constants` -- the same numbers WRF's own saturation
    calls use -- so the cap this module enforces and the saturation the
    microphysics will see the next step are the same curve.
    """
    denominator = temperature - c.SVP3
    if float(xp.min(denominator)) <= 0.0:
        raise ValueError(
            "the perturbed temperature field reaches the Tetens pole "
            f"(T <= {c.SVP3} K); the temperature perturbation amplitude is "
            "not physical for this state")
    es = (c.SVP1 * 1000.0) * xp.exp(
        c.SVP2 * (temperature - c.SVPT0) / denominator)
    margin = pressure - es
    if float(xp.min(margin)) <= 0.0:
        raise ValueError(
            "saturation vapour pressure meets or exceeds the total "
            "pressure somewhere in the perturbed state; the moisture cap "
            "cannot be evaluated there")
    return c.EP2 * es / margin


def apply_perturbations(state, seed: int, cfg: PerturbationConfig
                        ) -> dict[str, Any]:
    """Perturb ``state`` in place and return the provenance for that member.

    This exact signature is the contract the ensemble engine codes against.
    ``state`` is mutated; the return value is a JSON-serializable dict that
    fully identifies the perturbation (seed, per-field amplitudes and scales,
    the SHA-256 of each white-noise draw, the realized statistics, the taper,
    and every bound that fired).

    The state is expected to look like ``gpuwm.core.state.DomainState``:
    ``thp``/``qv`` on mass points ``(nz, ny, nx)``, ``u`` on ``(nz, ny,
    nx+1)``, ``v`` on ``(nz, ny+1, nx)``, backed by either NumPy or CuPy.
    Shapes are checked against each other; a field whose staggering does not
    match the mass grid is an error, not a broadcast.

    Read the module docstring for what balance is **not** imposed.  Nothing
    here re-balances mass or wind, and the members share one boundary file.

    **Caller post-condition.**  ``state.p``/``al``/``alt`` are diagnostics of
    ``(thp, php, mup, qv)``; perturbing ``thp`` and ``qv`` leaves them stale.
    The caller must run ``gpuwm.core.diagnostics.update_diagnostics(state,
    ...)`` before the first step.  This module deliberately does not call it:
    the diagnostic is a CUDA-only path and importing it here would make a
    NumPy-backed perturbation impossible.  The returned provenance records
    the requirement under ``"post_conditions"``.  For the same reason the
    supersaturation cap is evaluated against the pressure *as it stands on
    entry* -- a few-kelvin theta perturbation moves the pressure by well
    under a percent, so the cap is accurate to that, and it is a cap rather
    than an equality anyway.

    **Rim invariant.**  Wherever the taper is exactly zero this call is the
    identity, byte for byte -- including the moisture bounds, which are
    confined to the taper-active region.  A clamp that repaired the rim
    would break exactly the boundary consistency the taper exists to keep.

    Perturbing these fields does not disturb ``setup_fingerprint``
    (``gpuwm.state_serialization_contract``): ``u``/``v``/``thp``/``qv`` are
    serialized arrays, not setup arrays, so a perturbed member still restarts
    against the same base state and boundary tables.
    """
    if not isinstance(cfg, PerturbationConfig):
        raise TypeError(
            "cfg must be a PerturbationConfig (build one with "
            "PerturbationConfig.from_mapping for table-driven callers), got "
            f"{type(cfg).__name__}")
    if int(seed) != seed:
        raise TypeError(f"seed must be an integer, got {seed!r}")
    seed = int(seed)

    mass = _resolve_target(state, "theta")  # state.thp, always present
    if getattr(mass, "ndim", 0) != 3:
        raise ValueError(
            f"state.thp must be 3-D (nz, ny, nx), got shape "
            f"{getattr(mass, 'shape', None)}")
    xp = _array_module(mass)
    nz, ny, nx = (int(extent) for extent in mass.shape)

    names = cfg.field_names
    targets: dict[str, Any] = {}
    for name in names:
        array = _resolve_target(state, name)
        expected = _expected_shape(name, nz, ny, nx)
        if tuple(int(e) for e in array.shape) != expected:
            raise ValueError(
                f"state.{SUPPORTED_FIELDS[name].attribute} has shape "
                f"{tuple(array.shape)} but the ARW staggering for {name!r} "
                f"on this ({nz}, {ny}, {nx}) mass grid requires {expected}")
        if _array_module(array) is not xp:
            raise TypeError(
                f"state.{SUPPORTED_FIELDS[name].attribute} is on a "
                "different array backend than state.thp; a half-migrated "
                "state is not a state")
        targets[name] = array

    needs_pressure_for = [n for n in names
                          if SUPPORTED_FIELDS[n].exner_from_temperature]
    exner = None
    if needs_pressure_for:
        pressure = _require_pressure(
            state, xp, f"the temperature perturbation {needs_pressure_for}")
        exner = (pressure / c.P0) ** c.RCP

    field_records: list[dict[str, Any]] = []
    qv_increment = None
    for name in names:
        spec = cfg.spec(name)
        target = targets[name]
        shape = tuple(int(e) for e in target.shape)
        _check_resolvable(shape, cfg.dx_km, cfg.dy_km, spec)
        draw, info = gaussian_random_field(
            shape, seed=seed, name=name, dx_km=cfg.dx_km, dy_km=cfg.dy_km,
            length_scale_km=spec.length_scale_km,
            vertical_scale_levels=spec.vertical_scale_levels,
            xp=xp, dtype=cfg.compute_dtype)
        taper = boundary_taper(shape[1], shape[2], cfg.rim_width,
                               kind=cfg.rim_taper, xp=xp,
                               dtype=draw.dtype)
        increment = draw * float(spec.amplitude) * taper[None, :, :]
        if SUPPORTED_FIELDS[name].exner_from_temperature:
            increment = increment / exner
        increment = increment.astype(target.dtype, copy=False)
        # Write ONLY where the taper is active.  ``target += increment``
        # over the whole array looks like the identity wherever the
        # increment is exactly zero, and is -- except for signed zero:
        # IEEE ``-0.0 + 0.0`` is ``+0.0``, so a rim holding -0.0 came back
        # numerically equal and byte-different, and the state sha sees
        # bytes.  Selecting with ``where`` keeps the original words.
        active = (taper > 0.0)[None, :, :]
        target[...] = xp.where(active, target + increment, target)
        if name == "qv":
            qv_increment = increment

        record = {
            "name": name,
            "attribute": SUPPORTED_FIELDS[name].attribute,
            "stagger": SUPPORTED_FIELDS[name].stagger,
            "units": SUPPORTED_FIELDS[name].units,
            "amplitude": float(spec.amplitude),
            "length_scale_km": float(spec.length_scale_km),
            "vertical_scale_levels": float(spec.vertical_scale_levels),
            "shape": list(shape),
            "stream_key_hex": info["stream_key_hex"],
            "noise_sha256": info["noise_sha256"],
            "noise_dtype": info["noise_dtype"],
            "fft_backend": info["fft_backend"],
            "unit_field_realized_rms": info["realized_rms"],
            "increment_rms": float(
                xp.sqrt(xp.mean(increment.astype(np.float64) ** 2))),
            "increment_min": float(xp.min(increment)),
            "increment_max": float(xp.max(increment)),
            "vertical_wrap": info["vertical_wrap"],
        }
        if SUPPORTED_FIELDS[name].exner_from_temperature:
            record["exner_converted"] = True
            record["applied_to"] = "potential temperature (theta')"
        field_records.append(record)

    bounds = _enforce_bounds(state, cfg, xp, perturbed=set(names),
                             qv_increment=qv_increment)

    combined = hashlib.sha256()
    for record in field_records:
        combined.update(record["noise_sha256"].encode("ascii"))
    return {
        "schema": PROVENANCE_SCHEMA,
        "module": "gpuwm.da.perturb",
        "status": STATUS,
        "seed": seed,
        "backend": "numpy" if xp is np else "cupy",
        #: "numpy" alongside a "cupy" backend means the device FFT was
        #: unavailable and the filtering ran on the host -- correct, slower,
        #: and never silent.
        "fft_backend": ("numpy" if not _device_fft_available(xp)
                        else ("numpy" if xp is np else "cupy")),
        "compute_dtype": cfg.compute_dtype,
        "mass_grid": [nz, ny, nx],
        "grid_spacing_km": {"dx": float(cfg.dx_km), "dy": float(cfg.dy_km)},
        "application_order": list(names),
        "fields": field_records,
        "noise_sha256": combined.hexdigest(),
        "taper": {
            "kind": cfg.rim_taper,
            "rim_width_cells": cfg.rim_width,
            "boundary_value": 0.0,
            "interior_value": 1.0,
            "axes": "lateral only (no vertical taper)",
        },
        "bounds": bounds,
        "post_conditions": [
            "wherever the rim taper is zero this call was the identity, "
            "byte for byte, bounds included",
            "state.p / state.al / state.alt are now stale: run "
            "gpuwm.core.diagnostics.update_diagnostics(state, ...) before "
            "the first step",
            "the supersaturation cap was evaluated against the pressure as "
            "it stood on entry, not against the re-diagnosed pressure",
        ],
        "balance_not_imposed": [
            "mass: mu' is untouched and the column is not re-balanced "
            "hydrostatically",
            "wind: the u/v increments are neither non-divergent nor in "
            "geostrophic/gradient balance with the theta increment",
            "boundary: members share one unperturbed boundary file; only "
            "the rim taper keeps them consistent with it",
            "vertical: the draw is FFT-periodic in the column and nothing "
            "tapers it, so the top and bottom levels are correlated at the "
            "figure each field record's vertical_wrap.top_to_bottom_seam "
            "states -- near 1 for any usable vertical scale. The "
            "quarter-column cap on vertical_scale_levels bounds the "
            "half-column correlation, not the seam",
        ],
    }


def _enforce_bounds(state, cfg: PerturbationConfig, xp,
                    perturbed: Iterable[str], qv_increment
                    ) -> dict[str, Any]:
    """Clamp the perturbed moisture field, and report what the clamp did.

    Runs whenever moisture was perturbed.  Leaving a mixing ratio negative is
    not a rounding detail: the microphysics will happily advect it, and the
    first positive-definite renormalization will manufacture mass to hide it.

    **The clamp is confined to the taper-active region.**  Where the taper is
    exactly zero this function is the identity, byte for byte, even if the
    incoming state violates the bound there.  Two reasons.  The rim has to
    match the shared boundary file, and a clamp that "helpfully" repaired it
    would break exactly the consistency the taper exists to preserve; and a
    perturbation module that silently repairs a state defect it did not
    cause is a module that hides the defect.  Pre-existing violations inside
    the perturbed region are counted and reported rather than hidden.
    """
    perturbed = set(perturbed)
    report: dict[str, Any] = {
        "qv_floor": float(cfg.qv_floor),
        "qv_floor_clipped_points": 0,
        "rh_cap": None if cfg.rh_cap is None else float(cfg.rh_cap),
        "rh_cap_clipped_points": 0,
        "evaluated": False,
        "scope": "taper-active points only; the untapered rim is returned "
                 "byte-identical even where it violates a bound",
    }
    if "qv" not in perturbed:
        report["skipped_reason"] = "no moisture perturbation was configured"
        return report

    qv = _resolve_target(state, "qv")
    report["evaluated"] = True
    ny, nx = int(qv.shape[1]), int(qv.shape[2])
    active = boundary_taper(ny, nx, cfg.rim_width, kind=cfg.rim_taper,
                            xp=xp, dtype=np.float64) > 0.0
    active = active[None, :, :]

    # The incoming state, reconstructed BEFORE either clamp runs.  Doing
    # it afterwards -- ``qv - qv_increment`` on the already-clipped field
    # -- reconstructed the ceiling minus the increment at every clipped
    # point, which is not the state that arrived, and undercounted
    # pre-existing supersaturation wherever the increment was positive.
    incoming = None
    if cfg.rh_cap is not None and qv_increment is not None:
        incoming = qv - qv_increment

    floor = qv.dtype.type(cfg.qv_floor)
    breaches = active & (qv < floor)
    below = int(xp.count_nonzero(breaches))
    if below:
        qv[...] = xp.where(breaches, floor, qv)
    report["qv_floor_clipped_points"] = below

    if cfg.rh_cap is not None:
        pressure = _require_pressure(state, xp, "the supersaturation cap")
        theta = _total_theta(state, xp)
        temperature = theta * (pressure / c.P0) ** c.RCP
        qvs = _saturation_mixing_ratio(temperature, pressure, xp)
        ceiling = (qvs * cfg.rh_cap).astype(qv.dtype, copy=False)
        breaches = active & (qv > ceiling)
        above = int(xp.count_nonzero(breaches))
        if above:
            qv[...] = xp.where(breaches, ceiling, qv)
        report["rh_cap_clipped_points"] = above
        if incoming is not None:
            # What the incoming state was already doing, so a repair this
            # module performs is visible in the manifest rather than silent.
            report["pre_existing_supersaturated_points"] = int(
                xp.count_nonzero(active & (incoming > ceiling)))
            report["pre_existing_basis"] = (
                "qv as it stood on entry (post-perturbation minus the "
                "increment, snapshotted before either clamp), against the "
                "same ceiling the cap applied")
        report["saturation_formula"] = (
            "Tetens over liquid water, gpuwm.core.constants "
            "SVP1/SVP2/SVP3/SVPT0 with EP2")
    else:
        report["skipped_reason"] = "rh_cap is None (cap disabled by config)"
    return report


# --------------------------------------------------------------------------
# Documented stubs -- routes that exist on paper only
# --------------------------------------------------------------------------

def recycled_difference_perturbations(*args, **kwargs):
    """Not built in v1.  Perturbations recycled from forecast differences.

    The idea: take the difference between two forecasts valid at the same
    time (or one forecast and its own state some hours earlier), rescale it
    to the desired analysis-error magnitude, and use *that* as the
    perturbation.  Because the difference is a difference of two model
    trajectories it is already in the model's own balance -- no gravity-wave
    shock on the first step -- and it carries flow-dependent structure that a
    prescribed isotropic Gaussian cannot.

    What it needs that this module does not have: two archived states on the
    same grid, a rescaling target (a spread climatology or an observed
    innovation variance), and a policy for the boundary rows, since a
    recycled difference is nonzero everywhere including the rim.  Until all
    three exist, calling this raises rather than silently falling back to the
    Gaussian route.
    """
    raise NotImplementedError(
        "recycled-difference perturbations are a documented v1 non-goal; "
        "use apply_perturbations with a PerturbationConfig, or implement "
        "this route with its own provenance schema")


def perturbed_lateral_boundaries(*args, **kwargs):
    """Not built in v1.  Per-member perturbation of the boundary forcing.

    The idea: give each member its own boundary tendencies so spread does not
    collapse toward the rim as the forecast runs and the unperturbed inflow
    floods the domain.  For a multi-hour storm-scale forecast on a small
    domain this matters more than the initial-condition perturbation does.

    What it needs that this module does not have: a per-member boundary file
    (or an in-memory equivalent) that the lateral-BC reader can be pointed
    at, a perturbation that is consistent in time across boundary intervals
    rather than redrawn each interval, and agreement with the driving model's
    own uncertainty.  Until then the rim taper in
    :func:`apply_perturbations` is the whole boundary story, and the
    provenance says so.
    """
    raise NotImplementedError(
        "perturbed lateral boundaries are a documented v1 non-goal; members "
        "share one boundary file and apply_perturbations tapers the rim to "
        "zero so that stays legal")
