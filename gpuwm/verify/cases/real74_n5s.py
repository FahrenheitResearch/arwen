"""N5S gpuwm runner restored directly from WRF ``real.exe`` products.

No config file and no ERA5/WPS ingest surface is used here.  The four-domain
experiment is constructed programmatically, atmospheric and physics state is
restored from ``wrfinput_d01..d04``, and d01's native Davies value/tendency
tables are restored from ``wrfbdy_d01``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Mapping, Sequence

import netCDF4
import numpy as np

from gpuwm.verify.n5s_common import restored_input_sha256, stable_hash, write_json
from gpuwm.verify.n5s_metrics import (
    load_registration, make_registration, require_matching_registrations,
)


ETA_LEVELS = (
    1.00000, 0.99780, 0.99519, 0.99212, 0.98849,
    0.98422, 0.97918, 0.97325, 0.96627, 0.95808,
    0.94846, 0.93719, 0.92402, 0.90866, 0.89079,
    0.87006, 0.84612, 0.81857, 0.78706, 0.75124,
    0.71080, 0.66556, 0.61547, 0.56067, 0.50519,
    0.45474, 0.40886, 0.36713, 0.32918, 0.29466,
    0.26328, 0.23473, 0.20877, 0.18516, 0.16369,
    0.14417, 0.12641, 0.11026, 0.09557, 0.08222,
    0.07007, 0.05902, 0.04898, 0.03984, 0.03153,
    0.02398, 0.01710, 0.01085, 0.00517, 0.00000,
)
#: This case's initial instant.  The metric registration takes it as a
#: required argument, so the one place that knows which campaign is being
#: scored is this case module.
N5S_START_TIME = datetime(1974, 4, 3, 12, 0, 0)
N5S_FORCING_INTERVAL_SECONDS = 6 * 60 * 60
N5S_SOIL_LAYERS = 4
REFLECTIVITY_MICROPHYSICS = frozenset((1, 6, 8, 10, 18))

# Every science-field entry below has one explicit gpuwm consumer.  The
# importer is intentionally closed-world: variables outside these inventories
# are errors, rather than being hidden in a catch-all auxiliary bucket.
REQUIRED_WRFINPUT = (
    "U", "V", "W", "T", "PH", "MU", "PHB", "MUB", "T_INIT",
    "P", "PB", "AL", "ALB", "QVAPOR", "QCLOUD", "QRAIN",
    "QICE", "QSNOW", "QGRAUP", "QNRAIN", "QNICE", "QNSNOW",
    "QNGRAUPEL", "HGT", "FNM", "FNP", "RDNW", "RDN", "DNW",
    "DN", "ZNU", "ZNW", "C1H", "C2H", "C1F", "C2F", "C3H",
    "C4H", "C3F", "C4F", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
    "F", "E", "SINALPHA", "COSALPHA", "LANDMASK", "LU_INDEX",
    "ISLTYP", "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SNOW",
    "SNOWH", "VEGFRA", "SNOALB", "SHDMIN", "SHDMAX", "PSFC", "T2",
    "Q2", "TH2", "U10", "V10", "XLAND", "IVGTYP",
    "CF1", "CF2", "CF3",
)

ALIASES = {
    "XICE": ("XICE", "SEAICE"),
    "ALBBCK": ("ALBBCK", "ALBEDO"),
    "LAI": ("LAI", "LAI12M"),
    "P_TOP": ("P_TOP",),
}

MOISTURE_MAP = {
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg",
    "QNRAIN": "nr", "QNICE": "ni", "QNSNOW": "ns",
    "QNGRAUPEL": "ng", "QNCLOUD": "nc",
}

# NSSL reuses several WRF Registry names carried by Morrison, but the state
# names are scheme-native (for example QNRAIN -> qnr, not nr).  Keep this as
# a distinct map so restoration cannot choose a target by whichever attribute
# happens to exist on DomainState.
NSSL_MOISTURE_MAP = {
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg", "QHAIL": "qh",
    "QNDROP": "qndrop", "QNRAIN": "qnr", "QNICE": "qni",
    "QNSNOW": "qns", "QNGRAUPEL": "qng", "QNHAIL": "qnh",
    "QNCCN": "qnn", "QVGRAUPEL": "qvolg", "QVHAIL": "qvolh",
}
ALL_MOISTURE_WRFINPUT = frozenset(MOISTURE_MAP) | frozenset(
    NSSL_MOISTURE_MAP)

# ``real.exe`` writes a physics-package-specific moisture inventory.  The
# first three mass species are active in every gpuwm moist configuration;
# WSM6 adds the three ice-category masses, while Morrison additionally owns
# four transported number moments.  Native option-18 NSSL defaults add hail,
# five two-moment number fields, predicted CCN, and graupel/hail volume.  Its
# Registry aliases overlap Morrison but map to distinct scheme-native state.
# QNCLOUD is a documented optional Morrison restart field (WRF's matched
# default diagnoses cloud number).
BASE_MOISTURE_WRFINPUT = ("QVAPOR", "QCLOUD", "QRAIN")
ICE_MASS_WRFINPUT = ("QICE", "QSNOW", "QGRAUP")
MORRISON_NUMBER_WRFINPUT = ("QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL")
MORRISON_OPTIONAL_MOISTURE_WRFINPUT = ("QNCLOUD",)
THOMPSON_NUMBER_WRFINPUT = ("QNRAIN", "QNICE")
NSSL_MOISTURE_WRFINPUT = (
    "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
    "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
)

PHYSICS_FIELD_ALIASES = {
    "landmask": ("LANDMASK",), "xland": ("XLAND",),
    "tsk": ("TSK",), "pblh": ("PBLH",), "ivgtyp": ("IVGTYP", "LU_INDEX"),
    "isltyp": ("ISLTYP",), "vegfra": ("VEGFRA",), "tmn": ("TMN",),
    "xice": ("XICE", "SEAICE"), "swdown": ("SWDOWN",), "glw": ("GLW",),
    "snow": ("SNOW",), "snowh": ("SNOWH",), "smois": ("SMOIS",),
    "tslb": ("TSLB",), "sh2o": ("SH2O",), "psfc": ("PSFC",),
    "t2": ("T2",), "q2": ("Q2",), "th2": ("TH2",),
    "u10": ("U10",), "v10": ("V10",), "snoalb": ("SNOALB",),
    "albbck": ("ALBBCK", "ALBEDO"), "lai": ("LAI",),
    "shdmin": ("SHDMIN",), "shdmax": ("SHDMAX",),
    "ust": ("UST",), "znt": ("ZNT",), "hfx": ("HFX",),
    "qfx": ("QFX",), "lh": ("LH",), "grdflx": ("GRDFLX",),
}

# Optional restart-state fields have explicit consumers in
# ``restore_domain_state``.  They are permitted when present but are not
# synthesized when absent.
OPTIONAL_WRFINPUT = (
    "H_DIABATIC", "RAINNC", "RAINC", "QNCLOUD",
)

# These are the only non-science variable records allowed through the reader.
# ``Times`` is character metadata and is not copied into ``RestoredDomain.raw``.
EXPLICIT_AUXILIARY_WRFINPUT = (
    "Times", "XTIME", "ITIMESTEP",
    "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
)

_MASS_3D_DIMS = ("bottom_top", "south_north", "west_east")
_MASS_2D_DIMS = ("south_north", "west_east")
_U_3D_DIMS = ("bottom_top", "south_north", "west_east_stag")
_V_3D_DIMS = ("bottom_top", "south_north_stag", "west_east")
_W_3D_DIMS = ("bottom_top_stag", "south_north", "west_east")
_U_2D_DIMS = ("south_north", "west_east_stag")
_V_2D_DIMS = ("south_north_stag", "west_east")
_SOIL_DIMS = ("soil_layers_stag", "south_north", "west_east")
_WRFINPUT_GEOMETRY_DIMENSIONS = frozenset({
    "bottom_top", "bottom_top_stag", "south_north", "south_north_stag",
    "west_east", "west_east_stag", "soil_layers_stag",
})

# Field-specific staggering is checked while bytes are read, before any GPU
# state exists.  The dimension names are part of the contract as well as the
# resulting shape, so a truncated variable cannot borrow an unrelated
# dimension of the same length.
WRFINPUT_DIMENSIONS: dict[str, tuple[str, ...]] = {
    **{name: _MASS_3D_DIMS for name in (
        "T", "T_INIT", "P", "PB", "AL", "ALB", "QVAPOR", "QCLOUD",
        "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNRAIN", "QNICE",
        "QNSNOW", "QNGRAUPEL", "QNCLOUD", "QHAIL", "QNDROP",
        "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL", "H_DIABATIC",
    )},
    "U": _U_3D_DIMS, "V": _V_3D_DIMS, "W": _W_3D_DIMS,
    "PH": _W_3D_DIMS, "PHB": _W_3D_DIMS,
    **{name: _MASS_2D_DIMS for name in (
        "MU", "MUB", "HGT", "MAPFAC_M", "F", "E", "SINALPHA",
        "COSALPHA", "LANDMASK", "LU_INDEX", "ISLTYP", "TSK", "TMN",
        "SNOW", "SNOWH", "VEGFRA", "SNOALB", "SHDMIN", "SHDMAX",
        "PSFC", "T2", "Q2", "TH2", "U10", "V10", "XLAND", "IVGTYP",
        "XICE", "SEAICE", "ALBBCK", "ALBEDO", "LAI", "LAI12M",
        "SWDOWN", "GLW",
        "PBLH", "UST", "ZNT", "HFX", "QFX", "LH", "GRDFLX", "RAINNC",
        "RAINC", "XLAT", "XLONG",
    )},
    "MAPFAC_U": _U_2D_DIMS, "MAPFAC_V": _V_2D_DIMS,
    "XLAT_U": _U_2D_DIMS, "XLONG_U": _U_2D_DIMS,
    "XLAT_V": _V_2D_DIMS, "XLONG_V": _V_2D_DIMS,
    **{name: _SOIL_DIMS for name in ("TSLB", "SMOIS", "SH2O")},
    **{name: ("bottom_top",) for name in (
        "FNM", "FNP", "RDNW", "RDN", "DNW", "DN", "ZNU",
        "C1H", "C2H", "C3H", "C4H",
    )},
    **{name: ("bottom_top_stag",) for name in (
        "ZNW", "C1F", "C2F", "C3F", "C4F",
    )},
    **{name: () for name in (
        "P_TOP", "CF1", "CF2", "CF3", "XTIME", "ITIMESTEP",
    )},
}


def _mapped_wrfinput_names() -> set[str]:
    names = set(REQUIRED_WRFINPUT) | set(OPTIONAL_WRFINPUT)
    names.update(alias for aliases in ALIASES.values() for alias in aliases)
    names.update(ALL_MOISTURE_WRFINPUT)
    names.update(
        alias for aliases in PHYSICS_FIELD_ALIASES.values() for alias in aliases)
    return names


MAPPED_WRFINPUT = frozenset(_mapped_wrfinput_names())
ALLOWED_WRFINPUT = MAPPED_WRFINPUT | frozenset(EXPLICIT_AUXILIARY_WRFINPUT)

#: Standard real.exe wrfinput variables the restored model does not consume
#: (F20 conformance defines exactly what is restored; everything else is
#: skipped).  Enumerated explicitly — first contact with the production
#: handoff inputs (registered identity e2c6fdf4...) surfaced these — so an
#: unanticipated variable still fails loudly instead of being silently
#: ignored.  THM/P_HYD are present-but-unconsumed under the registered
#: use_theta_m=0 restoration; SST/land-use climatology fields are superseded
#: by the mapped surface restoration set.
IGNORED_WRFINPUT = frozenset({
    "BATHYMETRY_FLAG", "CANWAT", "CFN", "CFN1", "CLAT", "CLDFRA", "CPLMASK",
    "DTS", "DTSEPS", "DZS", "EROD", "FCX", "FNDALBSI", "FNDICEDEPTH",
    "FNDSNOWH", "FNDSNOWSI", "FNDSOILW", "FRC_URB2D", "GCX", "GOT_VAR_SSO",
    "LAKEFLAG", "LAKEMASK", "LAKE_DEPTH", "LAKE_DEPTH_FLAG", "LANDUSEF",
    "LAT_LL_D", "LAT_LL_T", "LAT_LL_U", "LAT_LL_V", "LAT_LR_D", "LAT_LR_T",
    "LAT_LR_U", "LAT_LR_V", "LAT_UL_D", "LAT_UL_T", "LAT_UL_U", "LAT_UL_V",
    "LAT_UR_D", "LAT_UR_T", "LAT_UR_U", "LAT_UR_V", "LON_LL_D", "LON_LL_T",
    "LON_LL_U", "LON_LL_V", "LON_LR_D", "LON_LR_T", "LON_LR_U", "LON_LR_V",
    "LON_UL_D", "LON_UL_T", "LON_UL_U", "LON_UL_V", "LON_UR_D", "LON_UR_T",
    "LON_UR_U", "LON_UR_V", "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX",
    "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY", "MF_VX_INV", "O3_GFS_DU", "P00",
    "PC", "PCB", "P_HYD", "P_STRAT", "QV_BASE", "RDX", "RDY", "RESM",
    "SAVE_TOPO_FROM_REAL", "SHDAVG", "SMCREL", "SNOWC", "SOILCBOT",
    "SOILCTOP", "SR", "SST", "STEP_NUMBER", "T00", "THIS_IS_AN_IDEAL_RUN",
    "THM", "TISO", "TLP", "TLP_STRAT", "TOPOSLPX", "TOPOSLPY", "T_BASE",
    "UOCE", "U_BASE", "U_FRAME", "VAR", "VAR_SSO", "VOCE", "V_BASE",
    "V_FRAME", "WATER_DEPTH", "ZETATOP", "ZS", "Z_BASE",
})


@dataclass(frozen=True)
class RestoredDomain:
    path: Path
    raw: Mapping[str, np.ndarray]
    dimensions: Mapping[str, int]
    global_attributes: Mapping[str, object]
    mapped_variables: tuple[str, ...]
    auxiliary_variables: tuple[str, ...]

    def wrf_frame(self) -> dict[str, np.ndarray]:
        """CPU inverse of the restored atmospheric mapping."""
        raw = self.raw
        theta_prime = np.asarray(
            (raw["T"] + np.float32(300.0)) - raw["T_INIT"],
            dtype=np.float32)
        return {
            "U": raw["U"].copy(), "V": raw["V"].copy(),
            "W": raw["W"].copy(),
            "T": np.asarray(raw["T_INIT"] + theta_prime
                            - np.float32(300.0), dtype=np.float32),
            "PH": raw["PH"].copy(), "MU": raw["MU"].copy(),
            "PHB": raw["PHB"].copy(), "MUB": raw["MUB"].copy(),
            "QVAPOR": raw["QVAPOR"].copy(),
        }


def _read_numeric(variable) -> np.ndarray:
    value = np.ma.asarray(variable[...])
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"WRF input variable {variable.name} contains masked data")
    array = np.asarray(value)
    if variable.dimensions and variable.dimensions[0] == "Time":
        if array.shape[0] != 1:
            raise ValueError(
                f"WRF input {variable.name} must carry exactly one Time record")
        array = array[0]
    if array.dtype.kind not in "iufb":
        raise TypeError(f"WRF input {variable.name} is not numeric")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"WRF input {variable.name} contains non-finite values")
    # ``np.ascontiguousarray`` promotes a zero-dimensional value to ``(1,)``.
    # WRF writes Registry scalars as ``(Time,)`` records, so preserve the
    # scalar produced by removing that singleton Time dimension.
    return np.array(array, copy=True, order="C")


def _n5s_grid_id_for_wrfinput(path: Path) -> int:
    prefix = "wrfinput_d"
    suffix = path.name.removeprefix(prefix)
    if not path.name.startswith(prefix) or not suffix.isdigit():
        raise ValueError(
            f"cannot identify registered N5S domain from WRF input {path}")
    return int(suffix)


@lru_cache(maxsize=1)
def _registered_wrfinput_dimensions() -> Mapping[int, Mapping[str, int]]:
    """Return WRF dimension extents from the registered N5S experiment."""
    experiment = build_n5s_experiment()
    geometries = {}
    for domain in experiment.domains:
        run = domain.run
        geometries[domain.grid_id] = MappingProxyType({
            "bottom_top": run.nz,
            "bottom_top_stag": run.nz + 1,
            "south_north": run.ny,
            "south_north_stag": run.ny + 1,
            "west_east": run.nx,
            "west_east_stag": run.nx + 1,
            "soil_layers_stag": N5S_SOIL_LAYERS,
        })
    return MappingProxyType(geometries)


def _explicit_wrfinput_dimensions(
        dimensions: Mapping[str, int]) -> Mapping[str, int]:
    """Validate a caller-pinned, non-N5S WRF domain geometry.

    Requiring the complete seven-dimension contract keeps an arbitrary file
    from defining its own expected geometry and thereby making a truncated
    but self-consistent handoff look valid.
    """
    if not isinstance(dimensions, Mapping):
        raise TypeError("expected_dimensions must be a mapping")
    names = set(dimensions)
    missing = sorted(_WRFINPUT_GEOMETRY_DIMENSIONS - names)
    extra = sorted(names - _WRFINPUT_GEOMETRY_DIMENSIONS)
    if missing or extra:
        raise ValueError(
            "explicit WRF geometry dimension inventory mismatch: "
            f"missing={missing}, extra={extra}")
    normalized = {}
    for name in sorted(_WRFINPUT_GEOMETRY_DIMENSIONS):
        value = dimensions[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0):
            raise ValueError(
                f"explicit WRF geometry {name} must be a positive integer")
        normalized[name] = int(value)
    for mass, staggered in (
            ("bottom_top", "bottom_top_stag"),
            ("south_north", "south_north_stag"),
            ("west_east", "west_east_stag")):
        if normalized[staggered] != normalized[mass] + 1:
            raise ValueError(
                f"explicit WRF geometry {staggered} must equal {mass} + 1")
    return MappingProxyType(normalized)


def _active_moisture_inventory(cfg) -> tuple[frozenset[str], frozenset[str]]:
    """Return required/allowed wrfinput moisture names for ``cfg``.

    ``cfg=None`` is the registered N5S compatibility path and retains the
    historical Morrison contract verbatim.
    """
    if cfg is None:
        required = frozenset(
            BASE_MOISTURE_WRFINPUT + ICE_MASS_WRFINPUT
            + MORRISON_NUMBER_WRFINPUT)
        return required, required | frozenset(
            MORRISON_OPTIONAL_MOISTURE_WRFINPUT)
    if not hasattr(cfg, "moist") or not hasattr(cfg, "mp_physics"):
        raise TypeError("cfg must expose moist and mp_physics")
    if not isinstance(cfg.moist, (bool, np.bool_)):
        raise TypeError("cfg.moist must be boolean")
    if (isinstance(cfg.mp_physics, (bool, np.bool_))
            or not isinstance(cfg.mp_physics, (int, np.integer))):
        raise TypeError("cfg.mp_physics must be an integer")
    moist = bool(cfg.moist)
    mp_physics = int(cfg.mp_physics)
    if mp_physics not in (0, 1, 6, 8, 10, 18):
        raise ValueError(
            f"unsupported active wrfinput mp_physics={mp_physics}")
    if not moist:
        if mp_physics != 0:
            raise ValueError(
                f"mp_physics={mp_physics} requires cfg.moist=True")
        return frozenset(), frozenset()
    required = BASE_MOISTURE_WRFINPUT
    if mp_physics in (6, 8, 10, 18):
        required += ICE_MASS_WRFINPUT
    if mp_physics == 8:
        required += THOMPSON_NUMBER_WRFINPUT
    elif mp_physics == 10:
        required += MORRISON_NUMBER_WRFINPUT
    elif mp_physics == 18:
        required += NSSL_MOISTURE_WRFINPUT
    allowed = frozenset(required)
    if mp_physics == 10:
        allowed |= frozenset(MORRISON_OPTIONAL_MOISTURE_WRFINPUT)
    return frozenset(required), allowed


def _active_moisture_map(cfg) -> Mapping[str, str]:
    """Return the exact WRF-name -> DomainState-name map for ``cfg``."""
    _, allowed = _active_moisture_inventory(cfg)
    if cfg is None or int(cfg.mp_physics) != 18:
        candidates = MOISTURE_MAP
    else:
        candidates = NSSL_MOISTURE_MAP
    return MappingProxyType({
        wrf_name: state_name
        for wrf_name, state_name in candidates.items()
        if wrf_name in allowed
    })


def _validate_wrfinput_geometry(name: str, variable,
                                expected_extents: Mapping[str, int],
                                value: np.ndarray) -> None:
    expected_dimensions = WRFINPUT_DIMENSIONS.get(name)
    if expected_dimensions is None:
        raise ValueError(f"WRF input variable {name} has no mapped geometry")
    actual_dimensions = tuple(variable.dimensions)
    if actual_dimensions[:1] == ("Time",):
        actual_dimensions = actual_dimensions[1:]
    try:
        expected_shape = tuple(
            expected_extents[dim] for dim in expected_dimensions)
    except KeyError as exc:
        raise ValueError(
            f"pinned WRF geometry has no dimension {exc.args[0]} for "
            f"WRF input {name}") from exc
    if actual_dimensions != expected_dimensions or value.shape != expected_shape:
        raise ValueError(
            f"WRF input {name} shape mismatch: expected pinned {expected_shape} "
            f"on {expected_dimensions}, got {value.shape} on {actual_dimensions}")


def read_wrfinput(path: str | Path, *, require_complete: bool = True,
                  grid_id: int | None = None,
                  expected_dimensions: Mapping[str, int] | None = None,
                  cfg=None,
                  ) -> RestoredDomain:
    """Read one wrfinput file without importing CuPy.

    Omitting ``expected_dimensions`` retains the registered N5S geometry
    lookup.  Generic real.exe handoffs must instead provide all seven pinned
    WRF extents explicitly.  ``cfg`` selects the exact active hydrometeor
    inventory; omitting it retains the registered Morrison/N5S contract.
    """
    path = Path(path)
    if expected_dimensions is None:
        if grid_id is None:
            grid_id = _n5s_grid_id_for_wrfinput(path)
        try:
            expected_extents = _registered_wrfinput_dimensions()[grid_id]
        except KeyError as exc:
            raise ValueError(
                f"WRF input {path} is not a registered N5S domain") from exc
    else:
        if grid_id is not None:
            raise ValueError(
                "grid_id and expected_dimensions are mutually exclusive")
        expected_extents = _explicit_wrfinput_dimensions(expected_dimensions)
    required_moisture, allowed_moisture = _active_moisture_inventory(cfg)
    with netCDF4.Dataset(path) as dataset:
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        unknown = sorted(
            set(dataset.variables) - ALLOWED_WRFINPUT - IGNORED_WRFINPUT)
        if unknown:
            raise ValueError(f"{path} has unmapped WRF variable(s): {unknown}")
        raw = {}
        for name, variable in dataset.variables.items():
            if name == "Times" or name in IGNORED_WRFINPUT:
                continue
            value = _read_numeric(variable)
            _validate_wrfinput_geometry(
                name, variable, expected_extents, value)
            raw[name] = value
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
    present_moisture = set(raw) & ALL_MOISTURE_WRFINPUT
    extra_moisture = sorted(present_moisture - allowed_moisture)
    if extra_moisture:
        raise ValueError(
            f"{path} has inactive WRF moisture variable(s) for the active "
            f"physics: {extra_moisture}")
    if require_complete:
        non_moisture_required = (
            set(REQUIRED_WRFINPUT) - ALL_MOISTURE_WRFINPUT)
        missing = sorted(name for name in non_moisture_required
                         if name not in raw)
        missing.extend(sorted(required_moisture - present_moisture))
        missing.extend(sorted(
            name for name, alternatives in ALIASES.items()
            if not any(alias in raw for alias in alternatives)))
        if missing:
            raise ValueError(f"{path} is missing mapped WRF variable(s): {missing}")
    mapped = MAPPED_WRFINPUT & set(raw)
    auxiliary = (set(EXPLICIT_AUXILIARY_WRFINPUT) - {"Times"}) & set(raw)
    return RestoredDomain(
        path=path, raw=MappingProxyType(raw),
        dimensions=MappingProxyType(dimensions),
        global_attributes=MappingProxyType(attrs),
        mapped_variables=tuple(sorted(mapped)),
        auxiliary_variables=tuple(sorted(auxiliary)))


def _expect_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"WRF {name} shape {value.shape} != expected {expected}")


def _first(raw: Mapping[str, np.ndarray], names: Sequence[str], *,
           required: bool = True):
    for name in names:
        if name in raw:
            return raw[name]
    if required:
        raise ValueError(f"WRF input is missing every alias in {tuple(names)}")
    return None


def _restore_active_moisture(state, raw: Mapping[str, np.ndarray], cfg,
                             array_module) -> None:
    """Restore the scheme-native moisture fields and their RK ``*0`` copies."""
    state_names = []
    for wrf_name, state_name in _active_moisture_map(cfg).items():
        target = getattr(state, state_name, None)
        if target is None:
            raise ValueError(
                f"DomainState lacks active mp_physics={cfg.mp_physics} "
                f"field {state_name} for WRF {wrf_name}")
        state_names.append(state_name)
        if wrf_name in raw:
            _expect_shape(wrf_name, raw[wrf_name], target.shape)
            target[...] = array_module.asarray(
                raw[wrf_name], dtype=array_module.float32)
        elif wrf_name not in MORRISON_OPTIONAL_MOISTURE_WRFINPUT:
            raise ValueError(
                f"WRF input lacks active mp_physics={cfg.mp_physics} "
                f"field {wrf_name}")

    # DomainState owns RK-beginning copies only for prognostic fields.  Sync
    # every active one that exists, including all ten NSSL-only fields.
    for state_name in dict.fromkeys(state_names):
        source = getattr(state, state_name)
        initial = getattr(state, f"{state_name}0", None)
        if initial is not None:
            initial[...] = source


def restore_domain_state(restored: RestoredDomain, cfg, *, scratch_arena=None,
                         dycore_state_workspace=None, radiation=None,
                         radiation_start_time=None, radiation_latitude=None,
                         radiation_longitude=None):
    """Upload one CPU-restored domain into a production ``DomainState``."""
    import cupy as cp
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.constants import G
    from gpuwm.core.state import DomainState

    raw = restored.raw
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    expected = {
        "U": (nz, ny, nx + 1), "V": (nz, ny + 1, nx),
        "W": (nz + 1, ny, nx), "T": (nz, ny, nx),
        "PH": (nz + 1, ny, nx), "MU": (ny, nx),
        "PHB": (nz + 1, ny, nx), "MUB": (ny, nx),
        "T_INIT": (nz, ny, nx), "P": (nz, ny, nx),
        "PB": (nz, ny, nx), "AL": (nz, ny, nx),
        "ALB": (nz, ny, nx),
    }
    for name, shape in expected.items():
        _expect_shape(name, raw[name], shape)
    state_kwargs = {}
    if scratch_arena is not None:
        state_kwargs["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        state_kwargs["dycore_state_workspace"] = dycore_state_workspace
    state = DomainState(cfg, **state_kwargs)

    for wrf_name, state_name in (("U", "u"), ("V", "v"), ("W", "w"),
                                 ("PH", "php"), ("MU", "mup")):
        getattr(state, state_name)[...] = cp.asarray(raw[wrf_name], dtype=cp.float32)
    state.thb[...] = cp.asarray(raw["T_INIT"], dtype=cp.float32)
    state.thp[...] = cp.asarray(
        (raw["T"].astype(np.float32) + np.float32(300.0))
        - raw["T_INIT"].astype(np.float32), dtype=cp.float32)
    state.phb[...] = cp.asarray(raw["PHB"], dtype=cp.float32)
    state.mub = None
    state.mub2d[...] = cp.asarray(raw["MUB"], dtype=cp.float32)
    state.pb[...] = cp.asarray(raw["PB"], dtype=cp.float32)
    state.alb[...] = cp.asarray(raw["ALB"], dtype=cp.float32)
    state.p[...] = cp.asarray(raw["P"] + raw["PB"], dtype=cp.float32)
    state.al[...] = cp.asarray(raw["AL"], dtype=cp.float32)
    state.alt[...] = cp.asarray(raw["AL"] + raw["ALB"], dtype=cp.float32)
    state.ht[...] = cp.asarray(raw["HGT"], dtype=cp.float32)
    state.p_top = cp.float32(np.asarray(_first(raw, ALIASES["P_TOP"])).reshape(-1)[0])

    one_dimensional = (
        "fnm", "fnp", "rdnw", "rdn", "dnw", "dn", "znu", "znw",
        "c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f",
    )
    for name in one_dimensional:
        target = getattr(state, name)
        source = np.asarray(raw[name.upper()]).reshape(-1)
        _expect_shape(name.upper(), source, target.shape)
        target[...] = cp.asarray(source, dtype=cp.float32)
    for name in ("CF1", "CF2", "CF3"):
        source = np.asarray(raw[name]).reshape(-1)
        _expect_shape(name, source, (1,))
        setattr(state, name.lower(), cp.float32(source[0]))
    state._phb_host = np.asarray(raw["PHB"], dtype=np.float64).copy()
    z_half = 0.5 * (state._phb_host[:-1] + state._phb_host[1:]) / G
    state._dz_min = float(np.diff(z_half, axis=0).min())

    state.set_map_coriolis(
        raw["MAPFAC_M"], raw["MAPFAC_U"], raw["MAPFAC_V"], raw["F"],
        raw["E"], sina=raw["SINALPHA"], cosa=raw["COSALPHA"])
    _restore_active_moisture(state, raw, cfg, cp)
    if "H_DIABATIC" in raw:
        state.h_diabatic[...] = cp.asarray(raw["H_DIABATIC"], dtype=cp.float32)

    for current, initial in (
            ("u", "u0"), ("v", "v0"), ("w", "w0"),
            ("thp", "thp0"), ("php", "php0"), ("mup", "mup0")):
        source = getattr(state, current, None)
        target = getattr(state, initial, None)
        if source is not None and target is not None:
            target[...] = source

    xice = _first(raw, ALIASES["XICE"])
    albbck = _first(raw, ALIASES["ALBBCK"])
    lai = _first(raw, ALIASES["LAI"])
    swdown = _first(raw, ("SWDOWN",), required=False)
    glw = _first(raw, ("GLW",), required=False)
    pblh = _first(raw, ("PBLH",), required=False)
    # real.exe leaves soil columns 0.0 at water points (WRF never reads
    # them there; it uses SST/TSK).  gpuwm's health gate bounds TSLB
    # globally, so fill water columns with TSK — the same values WRF's
    # own surface init uses over water — before restoration.
    landmask = np.asarray(raw["LANDMASK"])
    tslb = np.array(raw["TSLB"], copy=True)
    water = landmask < 0.5
    if np.any(water):
        tslb[:, water] = np.broadcast_to(
            np.asarray(raw["TSK"])[water], tslb[:, water].shape)
    raw = {**raw, "TSLB": tslb}  # alias loop below re-reads raw (immutable)
    driver = initialize_physics(
        state, cfg, landmask=raw["LANDMASK"], tsk=raw["TSK"],
        soil_temperature=tslb, soil_moisture=raw["SMOIS"],
        liquid_moisture=raw["SH2O"], ivgtyp=raw["LU_INDEX"],
        isltyp=raw["ISLTYP"], vegfra=raw["VEGFRA"], tmn=raw["TMN"],
        xice=xice, snow=raw["SNOW"], snow_depth=raw["SNOWH"],
        swdown=(0.0 if swdown is None else swdown),
        # These diagnostics are not wrfinput fields for this Registry.  WRF
        # v4.6.1 phy_init initializes all three to zero before the first
        # radiation/PBL calls, so preserve that start-of-run convention.
        glw=(0.0 if glw is None else glw),
        pblh=(0.0 if pblh is None else pblh),
        radiation=radiation,
        radiation_start_time=radiation_start_time,
        radiation_latitude=radiation_latitude,
        radiation_longitude=radiation_longitude)
    for field in driver.fields:
        aliases = PHYSICS_FIELD_ALIASES.get(field, (field.upper(),))
        value = _first(raw, aliases, required=False)
        if value is not None:
            if value.shape != driver.fields[field].shape:
                raise ValueError(
                    f"WRF physics field {aliases[0]} shape {value.shape} != "
                    f"gpuwm {field} shape {driver.fields[field].shape}")
            driver.fields[field][...] = cp.asarray(
                value, dtype=driver.fields[field].dtype)
    driver.fields["albbck"][...] = cp.asarray(albbck, dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    if "RAINNC" in raw:
        driver.microphysics.rainnc[...] = cp.asarray(raw["RAINNC"], dtype=cp.float32)
    if driver.rainc is not None and "RAINC" in raw:
        driver.rainc[...] = cp.asarray(raw["RAINC"], dtype=cp.float32)
    return state


def _decode_times(variable) -> tuple[datetime, ...]:
    data = np.ma.asarray(variable[...])
    if np.ma.isMaskedArray(data) and np.any(np.ma.getmaskarray(data)):
        raise ValueError(f"WRF time variable {variable.name} contains masked data")
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 19:
        raise ValueError(
            f"WRF time variable {variable.name} must have shape (records, 19), "
            f"got {data.shape}")
    values = []
    for row in data:
        try:
            text = b"".join(np.asarray(row, dtype="S1").tolist()).decode(
                "ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"WRF time variable {variable.name} is not ASCII") from exc
        try:
            value = datetime.strptime(text, "%Y-%m-%d_%H:%M:%S")
        except ValueError as exc:
            raise ValueError(
                f"WRF time variable {variable.name} has invalid timestamp "
                f"{text!r}") from exc
        if value.strftime("%Y-%m-%d_%H:%M:%S") != text:
            raise ValueError(
                f"WRF time variable {variable.name} timestamp is not canonical: "
                f"{text!r}")
        values.append(value)
    return tuple(values)


_WRFBDY_FIELDS = {
    "u": ("U", "bottom_top", "south_north", "west_east_stag"),
    "v": ("V", "bottom_top", "south_north_stag", "west_east"),
    "theta": ("T", "bottom_top", "south_north", "west_east"),
    "phi": ("PH", "bottom_top_stag", "south_north", "west_east"),
    "mu": ("MU", None, "south_north", "west_east"),
    "qv": ("QVAPOR", "bottom_top", "south_north", "west_east"),
}


def _wrfbdy_side_table(variable, index: int, side_name: str) -> np.ndarray:
    """Transpose WRF ``(width,z,side)`` into gpuwm's side convention."""
    value = np.ma.asarray(variable[index])
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"wrfbdy_d01 {variable.name} contains masked data")
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        axes = ((1, 2, 0) if side_name in ("west", "east")
                else (1, 0, 2))
        value = np.transpose(value, axes)
    elif value.ndim == 2:
        value = (np.transpose(value, (1, 0))[None]
                 if side_name in ("west", "east") else value[None])
    else:
        raise ValueError(
            f"wrfbdy_d01 {variable.name} must be a 2-D or 3-D side table")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"wrfbdy_d01 {variable.name} contains non-finite data")
    return np.ascontiguousarray(value)


def _boundary_strip(field: np.ndarray, side_name: str,
                    width: int) -> np.ndarray:
    if side_name == "west":
        return field[..., :width]
    if side_name == "east":
        return field[..., -width:][..., ::-1]
    if side_name == "south":
        return field[..., :width, :]
    return field[..., -width:, :][..., ::-1, :]


def _mass_from_boundary_sides(initial_mu: np.ndarray,
                              sides: Mapping[str, np.ndarray]) -> np.ndarray:
    """Restore a boundary-frame MU field from gpuwm-oriented side tables."""
    mu = np.asarray(initial_mu, dtype=np.float32).copy()
    width = sides["west"].shape[-1]
    for distance in range(width):
        mu[:, distance] = sides["west"][0, :, distance]
        mu[:, -1 - distance] = sides["east"][0, :, distance]
        mu[distance, :] = sides["south"][0, distance, :]
        mu[-1 - distance, :] = sides["north"][0, distance, :]
    return mu


def _wrf_and_gpuwm_mass_weights(restored: RestoredDomain,
                                mu: np.ndarray
                                ) -> tuple[dict[str, np.ndarray],
                                           dict[str, np.ndarray]]:
    """Return source-WRF and target-gpuwm FP32 dry-mass weights.

    ``real_em.F`` calls ``couple`` before packing wrfbdy.  At mass points
    that routine retains separate perturbation/base products, whereas
    gpuwm's producer first forms total column mass.  U/V additionally use
    WRF's one-sided physical faces and staggered mass averages.  Keeping the
    two expression trees distinct makes read-time normalization deterministic
    instead of relying on algebraic equivalence across FP32 roundoff.
    """
    raw = restored.raw
    required = {
        "MUB", "C1H", "C2H", "C1F", "C2F",
        "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            f"wrfinput lacks wrfbdy coupling variable(s): {missing}")
    mub = np.asarray(raw["MUB"], dtype=np.float32)
    if mu.shape != mub.shape:
        raise ValueError(
            f"wrfbdy MU frame {mu.shape} != wrfinput MUB {mub.shape}")
    c1h = np.asarray(raw["C1H"], dtype=np.float32)[:, None, None]
    c2h = np.asarray(raw["C2H"], dtype=np.float32)[:, None, None]
    c1f = np.asarray(raw["C1F"], dtype=np.float32)[:, None, None]
    c2f = np.asarray(raw["C2F"], dtype=np.float32)[:, None, None]

    # WRF v4.6.1 module_big_step_utilities_em.F:423-506,576-606.
    wrf_muu = np.empty((mu.shape[0], mu.shape[1] + 1), dtype=np.float32)
    wrf_muv = np.empty((mu.shape[0] + 1, mu.shape[1]), dtype=np.float32)
    wrf_muu[:, 1:-1] = np.asarray(
        np.float32(0.5) * (mu[:, 1:] + mu[:, :-1]
                           + mub[:, 1:] + mub[:, :-1]), dtype=np.float32)
    wrf_muv[1:-1, :] = np.asarray(
        np.float32(0.5) * (mu[1:, :] + mu[:-1, :]
                           + mub[1:, :] + mub[:-1, :]), dtype=np.float32)
    wrf_muu[:, 0] = np.asarray(mu[:, 0] + mub[:, 0], dtype=np.float32)
    wrf_muu[:, -1] = np.asarray(mu[:, -1] + mub[:, -1], dtype=np.float32)
    wrf_muv[0, :] = np.asarray(mu[0, :] + mub[0, :], dtype=np.float32)
    wrf_muv[-1, :] = np.asarray(mu[-1, :] + mub[-1, :], dtype=np.float32)
    wrf_half = np.asarray(
        c1h * mu[None] + (c1h * mub[None] + c2h), dtype=np.float32)
    wrf_full = np.asarray(
        c1f * mu[None] + (c1f * mub[None] + c2f), dtype=np.float32)

    # gpuwm lateral_bc._coupled_device_fields: total MU is formed first.
    total = np.asarray(mub + mu, dtype=np.float32)
    gpu_muu = np.empty_like(wrf_muu)
    gpu_muv = np.empty_like(wrf_muv)
    gpu_muu[:, 1:-1] = np.asarray(
        np.float32(0.5) * (total[:, 1:] + total[:, :-1]),
        dtype=np.float32)
    gpu_muv[1:-1, :] = np.asarray(
        np.float32(0.5) * (total[1:, :] + total[:-1, :]),
        dtype=np.float32)
    gpu_muu[:, 0], gpu_muu[:, -1] = total[:, 0], total[:, -1]
    gpu_muv[0, :], gpu_muv[-1, :] = total[0, :], total[-1, :]
    gpu_half = np.asarray(c1h * total[None] + c2h, dtype=np.float32)
    gpu_full = np.asarray(c1f * total[None] + c2f, dtype=np.float32)

    wrf = {
        "u": np.asarray(c1h * wrf_muu[None] + c2h, dtype=np.float32),
        "v": np.asarray(c1h * wrf_muv[None] + c2h, dtype=np.float32),
        "theta": wrf_half, "phi": wrf_full, "qv": wrf_half,
    }
    gpuwm = {
        "u": np.asarray(c1h * gpu_muu[None] + c2h, dtype=np.float32),
        "v": np.asarray(c1h * gpu_muv[None] + c2h, dtype=np.float32),
        "theta": gpu_half, "phi": gpu_full, "qv": gpu_half,
    }
    if any(not np.all(np.isfinite(weight)) or np.any(weight <= 0.0)
           for weight in (*wrf.values(), *gpuwm.values())):
        raise ValueError("wrfinput produces invalid wrfbdy dry-mass weights")
    return wrf, gpuwm


def _normalize_wrfbdy_endpoint(
        restored: RestoredDomain, gpu_name: str, side_name: str,
        coupled: np.ndarray, width: int,
        wrf_weights: Mapping[str, np.ndarray],
        gpuwm_weights: Mapping[str, np.ndarray]) -> np.ndarray:
    """Decouple one WRF endpoint and recouple it in gpuwm producer order."""
    if gpu_name == "mu":
        return np.asarray(coupled, dtype=np.float32)
    wrf_weight = _boundary_strip(
        wrf_weights[gpu_name], side_name, width)
    gpuwm_weight = _boundary_strip(
        gpuwm_weights[gpu_name], side_name, width)
    if np.array_equal(wrf_weight, gpuwm_weight):
        # Avoid a lossy divide/multiply round trip when the expression trees
        # already agree bitwise (normally U/V).
        return np.asarray(coupled, dtype=np.float32)
    primitive = np.asarray(coupled / wrf_weight, dtype=np.float32)
    if gpu_name in ("u", "v"):
        map_name = "MAPFAC_U" if gpu_name == "u" else "MAPFAC_V"
        map_factor = _boundary_strip(
            np.asarray(restored.raw[map_name], dtype=np.float32)[None],
            side_name, width)[0]
        primitive = np.asarray(primitive * map_factor[None], dtype=np.float32)
        return np.asarray(
            np.asarray(primitive * gpuwm_weight, dtype=np.float32)
            / map_factor[None], dtype=np.float32)
    return np.asarray(primitive * gpuwm_weight, dtype=np.float32)


def read_wrfbdy(path: str | Path, *, run_seconds: float,
                restored: RestoredDomain,
                forcing_interval_seconds: float =
                N5S_FORCING_INTERVAL_SECONDS):
    """Build coupled gpuwm d01 boundary intervals from real.exe tables."""
    from gpuwm.ingest.lateral_bc import (
        BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
    )

    if not isinstance(restored, RestoredDomain):
        raise TypeError("read_wrfbdy requires the restored d01 wrfinput")
    if (isinstance(forcing_interval_seconds, (bool, np.bool_))
            or not isinstance(
                forcing_interval_seconds,
                (int, float, np.integer, np.floating))
            or not np.isfinite(float(forcing_interval_seconds))
            or float(forcing_interval_seconds) <= 0.0
            or not float(forcing_interval_seconds).is_integer()):
        raise ValueError(
            "forcing_interval_seconds must be a positive whole number")
    forcing_interval = int(forcing_interval_seconds)
    field_names = {
        gpu_name: layout[0] for gpu_name, layout in _WRFBDY_FIELDS.items()}
    with netCDF4.Dataset(path) as dataset:
        if "Times" not in dataset.variables:
            raise ValueError("wrfbdy_d01 has no Times variable")
        if "bdy_width" not in dataset.dimensions:
            raise ValueError("wrfbdy_d01 has no bdy_width dimension")
        width = len(dataset.dimensions["bdy_width"])
        if width != 5:
            raise ValueError(
                f"wrfbdy_d01 bdy_width {width} != registered N5S width 5")
        times = _decode_times(dataset.variables["Times"])
        if not times:
            raise ValueError("wrfbdy_d01 has no records")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("wrfbdy_d01 records must increase")
        next_name = ("md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_")
        if len(times) == 1:
            # real.exe's standard single-interval product: one record of
            # boundary values at thisbdytime plus _BT tendencies valid to
            # nextbdytime.  The interval loop below already extrapolates the
            # last record by the registered cadence; here we only verify the
            # file's own metadata pins that same interval.
            if next_name not in dataset.variables:
                raise ValueError(
                    "single-record wrfbdy_d01 needs nextbdytime metadata")
            next_times = _decode_times(dataset.variables[next_name])
            expected_next = times[0] + timedelta(
                seconds=forcing_interval)
            if len(next_times) != 1 or next_times[0] != expected_next:
                raise ValueError(
                    "single-record wrfbdy_d01 nextbdytime does not pin the "
                    f"requested {forcing_interval}-second forcing interval")
        elif next_name in dataset.variables:
            # WRF products encountered in the wild use either one metadata
            # record for the final endpoint or one next-time record per Times
            # record.  Accept only those two explicit encodings.
            next_times = _decode_times(dataset.variables[next_name])
            expected_all = tuple(
                value + timedelta(seconds=forcing_interval)
                for value in times)
            expected_final = (expected_all[-1],)
            if next_times not in (expected_all, expected_final):
                raise ValueError(
                    "multi-record wrfbdy_d01 nextbdytime does not match "
                    f"Times plus {forcing_interval} seconds")
        required_records = []
        for gpu_name, wrf_name in field_names.items():
            _, zdim, ydim, xdim = _WRFBDY_FIELDS[gpu_name]
            for side_name, suffix in (
                    ("west", "XS"), ("east", "XE"),
                    ("south", "YS"), ("north", "YE")):
                value_name = f"{wrf_name}_B{suffix}"
                tendency_name = f"{wrf_name}_BT{suffix}"
                for record_name in (value_name, tendency_name):
                    if record_name not in dataset.variables:
                        raise ValueError(f"wrfbdy_d01 is missing {record_name}")
                    records = dataset.variables[record_name].shape[0]
                    if records != len(times):
                        raise ValueError(
                            f"wrfbdy_d01 {record_name} record count mismatch: "
                            f"expected {len(times)}, got {records}")
                if (dataset.variables[value_name].shape
                        != dataset.variables[tendency_name].shape):
                    raise ValueError(
                        f"wrfbdy_d01 {value_name}/{tendency_name} shape "
                        "mismatch")
                side_dim = ydim if side_name in ("west", "east") else xdim
                expected_dimensions = (["Time", "bdy_width"]
                                       + ([] if zdim is None else [zdim])
                                       + [side_dim])
                actual_dimensions = list(
                    dataset.variables[value_name].dimensions)
                if actual_dimensions != expected_dimensions:
                    raise ValueError(
                        f"wrfbdy_d01 {value_name} dimensions "
                        f"{tuple(actual_dimensions)} != expected real.exe "
                        f"{tuple(expected_dimensions)}")
                required_records.append(
                    (gpu_name, wrf_name, side_name, value_name, tendency_name))
        origin = times[0]
        starts = [(time - origin).total_seconds() for time in times]
        forcing_intervals = [
            later - earlier for earlier, later in zip(starts, starts[1:])]
        if any(interval != forcing_interval
               for interval in forcing_intervals):
            if forcing_interval == N5S_FORCING_INTERVAL_SECONDS:
                raise ValueError(
                    "wrfbdy_d01 must use the bundle's six-hour forcing cadence")
            raise ValueError(
                "wrfbdy_d01 must use the requested "
                f"{forcing_interval}-second forcing cadence")
        first_interval_end = (starts[1] if len(starts) > 1
                              else starts[0] + forcing_interval)
        # The registered/default N5S contract deliberately stays inside its
        # first six-hour interval.  An explicitly non-N5S cadence may consume
        # every validated record, including the final record's BT endpoint.
        legacy_n5s_cadence = (
            forcing_interval == N5S_FORCING_INTERVAL_SECONDS)
        coverage_end = (first_interval_end if legacy_n5s_cadence
                        else starts[-1] + forcing_interval)
        if not 0.0 < float(run_seconds) <= coverage_end:
            if legacy_n5s_cadence:
                raise ValueError(
                    "N5S run must remain inside the first six-hour boundary "
                    "interval")
            raise ValueError(
                "run exceeds the validated requested-cadence boundary "
                "coverage")
        intervals = []
        for index, start in enumerate(starts):
            if index + 1 < len(starts):
                end = starts[index + 1]
            else:
                end = start + forcing_interval
            raw_tables = {gpu_name: {} for gpu_name in field_names}
            for (gpu_name, wrf_name, side_name, value_name,
                 tendency_name) in required_records:
                value = _wrfbdy_side_table(
                    dataset.variables[value_name], index, side_name)
                tendency = _wrfbdy_side_table(
                    dataset.variables[tendency_name], index, side_name)
                raw_tables[gpu_name][side_name] = (value, tendency)
            duration = float(end - start)
            mu_start_sides = {
                side_name: pair[0]
                for side_name, pair in raw_tables["mu"].items()}
            mu_end_sides = {
                side_name: np.asarray(
                    pair[0] + np.float32(duration) * pair[1],
                    dtype=np.float32)
                for side_name, pair in raw_tables["mu"].items()}
            mu_start = _mass_from_boundary_sides(
                restored.raw["MU"], mu_start_sides)
            mu_end = _mass_from_boundary_sides(
                restored.raw["MU"], mu_end_sides)
            wrf_start, gpuwm_start = _wrf_and_gpuwm_mass_weights(
                restored, mu_start)
            wrf_end, gpuwm_end = _wrf_and_gpuwm_mass_weights(
                restored, mu_end)
            side_tables = {gpu_name: {} for gpu_name in field_names}
            for gpu_name, sides in raw_tables.items():
                for side_name, (value, tendency) in sides.items():
                    future = np.asarray(
                        value + np.float32(duration) * tendency,
                        dtype=np.float32)
                    normalized = _normalize_wrfbdy_endpoint(
                        restored, gpu_name, side_name, value, width,
                        wrf_start, gpuwm_start)
                    normalized_future = _normalize_wrfbdy_endpoint(
                        restored, gpu_name, side_name, future, width,
                        wrf_end, gpuwm_end)
                    target_tendency = (
                        normalized_future.astype(np.float64)
                        - normalized.astype(np.float64)) / duration
                    side_tables[gpu_name][side_name] = SideBoundary(
                        normalized.astype(np.float64), target_tendency)
            fields = {
                gpu_name: FieldBoundary(**side_tables[gpu_name])
                for gpu_name in field_names
            }
            intervals.append(BoundaryInterval(start, end, fields))
    return LateralBoundaries(tuple(intervals), width, 1, 4)


def build_n5s_experiment(*, run_minutes: int = 30,
                         history_minutes: int = 5):
    """Construct the registered four-domain experiment without configs/."""
    from gpuwm.experiment import build_experiment

    run_seconds = int(run_minutes) * 60
    history_seconds = int(history_minutes) * 60
    raw = {
        "experiment": {
            "name": "N5S_matched_physics_wrf_shadow",
            "start_time": N5S_START_TIME,
            "run_seconds": run_seconds,
            "feedback": 0, "smooth_option": 0, "blend_width": 5,
            "spec_bdy_width": 5, "restart_interval_s": 0.0,
        },
        "projection": {
            "map_proj": "lambert", "ref_lat": 39.6848,
            "ref_lon": -83.9297, "truelat1": 30.0, "truelat2": 60.0,
            "stand_lon": -83.9297,
        },
        "shared": {
            "nz": 49, "ztop": 20000.0, "p_top": 10000.0,
            "eta_levels": list(ETA_LEVELS), "hybrid_opt": 2, "etac": 0.2,
            "base_temp": 290.0, "time_step_sound": 4, "epssm": 0.5,
            "emdiv": 0.01, "hypsometric_opt": 2, "h_sca_adv_order": 5,
            "smdiv": 0.1, "moist": True, "mp_physics": 10,
            "moist_adv_opt": 1, "km_opt": 4, "diff_6th_opt": 2,
            "diff_6th_slopeopt": 1, "w_damping": 1, "damp_opt": 3,
            "zdamp": 5000.0, "dampcoef": 0.2, "khdif": 0.0, "kvdif": 0.0,
            "spec_zone": 1, "relax_zone": 4, "terrain_opt": 1,
            "map_proj": 1, "sf_sfclay_physics": 91,
            "sf_surface_physics": 2, "bl_pbl_physics": 1,
            "ra_physics": 4, "bldt": 0.0,
        },
        "domain": [
            {"grid_id": 1, "parent_id": 0, "i_parent_start": 1,
             "j_parent_start": 1, "parent_grid_ratio": 1,
             "parent_time_step_ratio": 1, "nx": 250, "ny": 200,
             "time_step": 60, "dx": 12000.0,
             "history_interval_s": history_seconds, "specified": True,
             "nested": False, "radt": 12.0, "cu_physics": 1,
             "cudt_minutes": 5.0, "diff_6th_factor": 0.12,
             "spec_exp": 0.0},
            {"grid_id": 2, "parent_id": 1, "i_parent_start": 63,
             "j_parent_start": 51, "parent_grid_ratio": 4,
             "parent_time_step_ratio": 4, "nx": 500, "ny": 400,
             "history_interval_s": history_seconds, "specified": False,
             "nested": True, "radt": 3.0, "cu_physics": 0,
             "diff_6th_factor": 0.10, "spec_exp": 0.0},
            {"grid_id": 3, "parent_id": 2, "i_parent_start": 167,
             "j_parent_start": 117, "parent_grid_ratio": 3,
             "parent_time_step_ratio": 3, "nx": 501, "ny": 501,
             "history_interval_s": history_seconds, "specified": False,
             "nested": True, "radt": 1.0, "cu_physics": 0,
             "diff_6th_factor": 0.08, "spec_exp": 0.0},
            {"grid_id": 4, "parent_id": 3, "i_parent_start": 151,
             "j_parent_start": 151, "parent_grid_ratio": 3,
             "parent_time_step_ratio": 3, "nx": 600, "ny": 600,
             "history_interval_s": history_seconds, "specified": False,
             "nested": True, "radt": 1.0, "cu_physics": 0,
             "diff_6th_factor": 0.06, "spec_exp": 0.0},
        ],
    }
    return build_experiment(raw, "programmatic:N5S_matched_physics_wrf_shadow")


def _restored_model_radiation(cfg, start_time, latitude, longitude, workspace):
    """Build only the explicitly shared RRTMGP adapter for a restored domain."""
    from gpuwm.config import radiation_scheme_ids

    if radiation_scheme_ids(cfg) != (4, 4):
        return None
    if workspace is None:
        raise ValueError("RRTMGP restored model requires a shared workspace")
    from gpuwm.core.rrtmgp import RRTMGPRadiation

    radiation = RRTMGPRadiation(
        start_time, latitude, longitude,
        trace_gas_overrides={"co2": 330.0e-6})
    radiation.column_chunk = workspace.column_chunk
    radiation.chunk_workspace = workspace
    return radiation


def build_restored_model(wrf_inputs: str | Path, *, run_minutes: int = 30,
                         history_minutes: int = 5):
    """Build the production tree from restored files, bypassing real ingest."""
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.core.model import (
        DomainNode, ExperimentState, ModelRuntimeStatus,
        SharedRRTMGPChunkWorkspace,
    )
    from gpuwm.core.nest import NestCoupler
    from gpuwm.core.preflight import DEFAULT_COLUMN_CHUNK
    from gpuwm.core.state import (build_shared_dycore_state_workspace,
                                  build_shared_scratch_arena)
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         bind_lateral_boundary_clock)
    from gpuwm.static.lambert import grids_from_projection_config

    root = Path(wrf_inputs)
    exp = build_n5s_experiment(
        run_minutes=run_minutes, history_minutes=history_minutes)
    restored = {
        domain.grid_id: read_wrfinput(
            root / f"wrfinput_d{domain.grid_id:02d}", grid_id=domain.grid_id)
        for domain in exp.domains}
    clock = resolve_clock(
        exp, lbc_interval_s=float(N5S_FORCING_INTERVAL_SECONDS))
    schedule = build_schedule(exp, clock)
    clocks = clock.clocks()
    dycore_state_workspace = build_shared_dycore_state_workspace(exp.domains)
    arena = build_shared_scratch_arena(exp.domains)
    grids = grids_from_projection_config(exp)
    uses_rrtmgp = any(
        radiation_scheme_ids(dc.run) == (4, 4) for dc in exp.domains)
    workspace = (
        SharedRRTMGPChunkWorkspace(
            nz=exp.root.run.nz, column_chunk=DEFAULT_COLUMN_CHUNK,
            p_top=exp.vertical.p_top)
        if uses_rrtmgp else None)
    nodes = {}
    prepared = {}
    for dc, grid in zip(exp.domains, grids):
        lat, lon = grid.latlon_mass()
        radiation = _restored_model_radiation(
            dc.run, exp.start_time, lat, lon, workspace)
        state = restore_domain_state(
            restored[dc.grid_id], dc.run, scratch_arena=arena,
            dycore_state_workspace=dycore_state_workspace,
            radiation=radiation, radiation_start_time=exp.start_time,
            radiation_latitude=lat, radiation_longitude=lon)
        parent = None if dc.parent_id == 0 else nodes[dc.parent_id]
        node = DomainNode(
            cfg=dc, grid=grid, state=state, clock=clocks[dc.grid_id],
            parent=parent, children=[], coupler=None)
        if parent is not None:
            node.coupler = NestCoupler(node)
            parent.children.append(node)
            state._nest_restart_classification = "REBUILT"
        nodes[dc.grid_id] = node
        prepared[dc.grid_id] = SimpleNamespace(
            grid=grid,
            static_fields={
                "HGT_M": np.asarray(restored[dc.grid_id].raw["HGT"]),
                "LANDMASK": np.asarray(restored[dc.grid_id].raw["LANDMASK"]),
                "LU_INDEX": np.asarray(restored[dc.grid_id].raw["LU_INDEX"]),
            },
            geog_selection=None)
    boundaries = read_wrfbdy(
        root / "wrfbdy_d01", run_seconds=exp.run_seconds,
        restored=restored[1])
    attach_lateral_boundaries(nodes[1].state, boundaries)
    # Davies clock bind (2026-07-28, retires the F20 adjudication): this
    # builder constructs DomainNodes manually instead of calling
    # build_experiment, so it must bind the restored root itself --
    # immediately after attachment, before ExperimentState -- or the
    # matched-physics shadow would keep measuring the retired one-step
    # root-clock lag after production was corrected (dossier section 5.3).
    bind_lateral_boundary_clock(nodes[1].state, nodes[1].clock)
    fingerprint = stable_hash({
        "experiment": exp.name,
        "restored_input_sha256": restored_input_sha256(root),
    })
    model = ExperimentState(
        root=nodes[1], nodes_by_grid_id=MappingProxyType(nodes),
        schedule=schedule, memory_ledger=None,
        experiment_fingerprint=fingerprint)
    model._scratch_arena = arena
    model._dycore_state_workspace = dycore_state_workspace
    model._prepared_by_grid_id = MappingProxyType(prepared)
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._io_manager = None
    model._last_checkpoint = None
    model._n5s_restored_domains = MappingProxyType(restored)
    return exp, model


def run_restored_experiment(wrf_inputs: str | Path, outdir: str | Path, *,
                            run_minutes: int = 30, history_minutes: int = 5,
                            registration: Mapping[str, object] | None = None
                            ) -> dict[str, object]:
    import cupy as cp
    from gpuwm.core.model import execute_experiment
    from gpuwm.core.refl import consume_refl_10cm
    from gpuwm.io.wrfout import PerDomainWrfoutWriters

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    created = make_registration(
        start_time=N5S_START_TIME.isoformat(),
        run_minutes=run_minutes, history_minutes=history_minutes)
    reg = created if registration is None else require_matching_registrations(
        created, registration)
    write_json(outdir / "n5s-preregistration.json", reg)
    digest = restored_input_sha256(wrf_inputs)
    (outdir / "restored_input_sha256.txt").write_text(
        digest + "\n", encoding="utf-8")
    exp, model = build_restored_model(
        wrf_inputs, run_minutes=run_minutes, history_minutes=history_minutes)
    with PerDomainWrfoutWriters(
            model, outdir, start_time=exp.start_time,
            title="gpuwm N5S matched-physics WRF shadow") as writers:
        model._io_manager = writers

        def history_handler(tree, node, ticks):
            refl_field = None
            if (ticks != 0 and node.state.qv is not None
                    and node.state.physics.mp_physics
                    in REFLECTIVITY_MICROPHYSICS):
                refl_field = consume_refl_10cm(node.state)
            writers.submit(node, ticks, refl_field=refl_field)

        report = execute_experiment(model, history_handler=history_handler)
        writers.drain()
        paths = writers.paths
    cp.cuda.runtime.deviceSynchronize()
    payload = {
        "schema": 1, "restored_input_sha256": digest,
        "domain_durations_seconds": {
            f"d{node.cfg.grid_id:02d}": float(node.clock.elapsed_seconds)
            for node in model.walk_parent_first()},
        "wrfout_paths": [str(path) for path in paths],
        "steps": report.steps, "forces": report.forces,
        "input_mapping": {
            f"d{grid_id:02d}": {
                "mapped_variables": list(restored.mapped_variables),
                "explicit_auxiliary_variables": list(
                    restored.auxiliary_variables),
            }
            for grid_id, restored in model._n5s_restored_domains.items()
        },
    }
    write_json(outdir / "N5S-gpu-run.json", payload)
    (outdir / "exit.status").write_text("0\n", encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-inputs", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--run-minutes", type=int, default=30)
    parser.add_argument("--history-minutes", type=int, default=5)
    parser.add_argument(
        "--registration", type=Path,
        help="optional controller-created preregistration; it must match the CLI timing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registration = (None if args.registration is None
                    else load_registration(args.registration))
    run_restored_experiment(
        args.wrf_inputs, args.outdir, run_minutes=args.run_minutes,
        history_minutes=args.history_minutes, registration=registration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RestoredDomain", "build_n5s_experiment", "build_restored_model",
    "read_wrfbdy", "read_wrfinput", "restore_domain_state",
    "run_restored_experiment",
]
