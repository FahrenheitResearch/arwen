"""Declarative stock-WRF initialization inventories for RW-WPS.

This module deliberately answers a narrower question than gpuwm's forecast
physics dispatcher: which package fields must exist when RW-WPS writes a
``wrfinput_dNN`` for an unchanged WRF v4.6.1 executable?  The inventory is
derived from ``Registry/Registry.EM_COMMON`` package declarations and field
I/O flags.  It must not be inferred from the schemes implemented by gpuwm.

WRF Registry dimensions ``ikjftb`` become the NetCDF dimensions below.  WRF
``real.exe`` initializes package hydrometeors/moments to zero when the source
analysis does not provide them; water vapour is populated from the source.
Package ``state:`` auxiliaries whose Registry flags contain ``r`` but not
``i`` are runtime/restart state and are intentionally not invented as
``wrfinput`` variables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


SCHEMA = "rw-wps.stock-wrf-physics-inventory.v1"
WRFINPUT_3D_DIMS = (
    "Time",
    "bottom_top",
    "south_north",
    "west_east",
)


@dataclass(frozen=True)
class WrfInputField:
    """One package member written to stock-WRF initialization files."""

    registry_name: str
    netcdf_name: str
    collection: str
    dtype: str = "float32"
    dimensions: tuple[str, ...] = WRFINPUT_3D_DIMS
    units: str = "kg kg-1"
    initialization: str = "zero_if_source_absent"


@dataclass(frozen=True)
class RuntimeStateField:
    """Scheme state allocated by WRF but not required in wrfinput."""

    registry_name: str
    netcdf_name: str
    dtype: str = "float32"
    dimensions: tuple[str, ...] = WRFINPUT_3D_DIMS
    initialization: str = "wrf_runtime"


@dataclass(frozen=True)
class StockWrfPhysicsInventory:
    mp_physics: int
    scheme: str
    registry_package: str
    wrfinput_fields: tuple[WrfInputField, ...]
    runtime_state_not_wrfinput: tuple[RuntimeStateField, ...]
    registry_authority: str = "WRF-v4.6.1 Registry/Registry.EM_COMMON"

    def as_report(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "target": "stock_wrf_v4.6.1",
            "mp_physics": self.mp_physics,
            "scheme": self.scheme,
            "registry_package": self.registry_package,
            "registry_authority": self.registry_authority,
            "wrfinput_fields": [asdict(field) for field in self.wrfinput_fields],
            "runtime_state_not_wrfinput": [
                asdict(field) for field in self.runtime_state_not_wrfinput
            ],
        }


def _moist(name: str, netcdf_name: str) -> WrfInputField:
    return WrfInputField(
        registry_name=name,
        netcdf_name=netcdf_name,
        collection="moist",
        initialization=(
            "source_specific_humidity" if name == "qv" else "zero_if_source_absent"
        ),
    )


def _scalar(
        name: str, netcdf_name: str, *, units: str = "# kg-1",
) -> WrfInputField:
    return WrfInputField(
        registry_name=name,
        netcdf_name=netcdf_name,
        collection="scalar",
        units=units,
    )


_ICE_MASS = (
    _moist("qv", "QVAPOR"),
    _moist("qc", "QCLOUD"),
    _moist("qr", "QRAIN"),
    _moist("qi", "QICE"),
    _moist("qs", "QSNOW"),
    _moist("qg", "QGRAUP"),
)

_EFFECTIVE_RADII = (
    RuntimeStateField("re_cloud", "RE_CLOUD"),
    RuntimeStateField("re_ice", "RE_ICE"),
    RuntimeStateField("re_snow", "RE_SNOW"),
)

_INVENTORIES = {
    # Registry.EM_COMMON:3021
    6: StockWrfPhysicsInventory(
        mp_physics=6,
        scheme="WSM6",
        registry_package="wsm6scheme",
        wrfinput_fields=_ICE_MASS,
        runtime_state_not_wrfinput=_EFFECTIVE_RADII,
    ),
    # Registry.EM_COMMON:3024.  Thompson's package includes both qni and qnr.
    8: StockWrfPhysicsInventory(
        mp_physics=8,
        scheme="Thompson",
        registry_package="thompson",
        wrfinput_fields=_ICE_MASS + (
            _scalar("qni", "QNICE"),
            _scalar("qnr", "QNRAIN", units="# kg(-1)"),
        ),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII,
    ),
    # Registry.EM_COMMON:3026.  Do not collapse Morrison moments into mass.
    10: StockWrfPhysicsInventory(
        mp_physics=10,
        scheme="Morrison two-moment",
        registry_package="morr_two_moment",
        wrfinput_fields=_ICE_MASS + (
            _scalar("qni", "QNICE"),
            _scalar("qns", "QNSNOW", units="# kg(-1)"),
            _scalar("qnr", "QNRAIN", units="# kg(-1)"),
            _scalar("qng", "QNGRAUPEL", units="# kg(-1)"),
        ),
        runtime_state_not_wrfinput=(
            RuntimeStateField("rqrcuten", "RQRCUTEN"),
            RuntimeStateField("rqscuten", "RQSCUTEN"),
            RuntimeStateField("rqicuten", "RQICUTEN"),
        ),
    ),
    # Registry.EM_COMMON:3033,3049,3052,3054,3056 after WRF's option-18
    # selector defaults resolve to two moments, hail, predicted CCN, and
    # predicted graupel/hail volume (density).
    18: StockWrfPhysicsInventory(
        mp_physics=18,
        scheme="NSSL-2",
        registry_package=(
            "nssl_2mom+nssl2mconc+nssl_hail+nssl_ccn_opt+nssl_hailvol"
        ),
        wrfinput_fields=_ICE_MASS + (
            _moist("qh", "QHAIL"),
            _scalar("qndrop", "QNDROP"),
            _scalar("qnr", "QNRAIN", units="# kg(-1)"),
            _scalar("qni", "QNICE"),
            _scalar("qns", "QNSNOW", units="# kg(-1)"),
            _scalar("qng", "QNGRAUPEL", units="# kg(-1)"),
            _scalar("qnh", "QNHAIL", units="# kg(-1)"),
            _scalar("qnn", "QNCCN", units="# kg(-1)"),
            _scalar("qvolg", "QVGRAUPEL", units="m(3) kg(-1)"),
            _scalar("qvolh", "QVHAIL", units="m(3) kg(-1)"),
        ),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII,
    ),
}


def supported_stock_wrf_mp_physics() -> tuple[int, ...]:
    """Return the declaratively inventoried stock-WRF microphysics ids."""

    return tuple(sorted(_INVENTORIES))


def stock_wrf_physics_inventory(mp_physics: int) -> StockWrfPhysicsInventory:
    """Return an exact v4.6.1 package inventory or fail closed."""

    if isinstance(mp_physics, bool) or not isinstance(mp_physics, int):
        raise TypeError(
            f"mp_physics must be a WRF integer, got {mp_physics!r}"
        )
    try:
        return _INVENTORIES[mp_physics]
    except KeyError:
        supported = ", ".join(str(value) for value in sorted(_INVENTORIES))
        raise ValueError(
            f"stock-WRF initialization inventory for mp_physics={mp_physics} "
            f"is not declared; currently evidenced from WRF v4.6.1: {supported}. "
            "Add the exact Registry package and real.exe initialization policy "
            "before enabling this configuration."
        ) from None


__all__ = [
    "SCHEMA",
    "StockWrfPhysicsInventory",
    "RuntimeStateField",
    "WRFINPUT_3D_DIMS",
    "WrfInputField",
    "stock_wrf_physics_inventory",
    "supported_stock_wrf_mp_physics",
]
