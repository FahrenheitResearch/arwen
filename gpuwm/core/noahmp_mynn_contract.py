"""Pinned WRF v4.6.1 contract for Noah-MP 4 and coupled MYNN 5/5.

This module records source identity, default option identity, and Registry
state allocation before either scheme is admitted as executable.  It is
deliberately CPU-only and has no runtime-selector side effects.

The WRF authority is commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``.  WRF pins its Noah-MP
submodule at ``848f54ad3d28c4303151fe5ad83724e232694422``.  A source tree
that differs by even one byte is a different oracle and fails this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


WRF_REFERENCE_VERSION = "v4.6.1"
WRF_REFERENCE_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
NOAHMP_REFERENCE_COMMIT = "848f54ad3d28c4303151fe5ad83724e232694422"
CONTRACT_ID = "wrf-v4.6.1-noahmp4-mynn5-defaults-v1"

SF_SURFACE_PHYSICS = 4
SF_SFCLAY_PHYSICS = 5
BL_PBL_PHYSICS = 5


@dataclass(frozen=True)
class ReferenceAsset:
    """One byte-pinned file in the authoritative WRF checkout."""

    relative_path: str
    bytes: int
    sha256: str


MYNN_SOURCE_ASSETS = (
    ReferenceAsset(
        "phys/module_sf_mynn.F",
        90_941,
        "86395534a6c9bfc79dcad50094bce290eff05756777a95794b2673795f9761c3",
    ),
    ReferenceAsset(
        "phys/module_bl_mynn.F",
        303_225,
        "b36c8b935cd9c8265359e78dbe8db285d99e992c7cab29917b5fca9d57d49452",
    ),
    ReferenceAsset(
        "phys/module_bl_mynn_common.F",
        4_424,
        "ba3e6909447408210fce490e95bdf2decc1e205d3446ddc5c715277d2b23924c",
    ),
    ReferenceAsset(
        "phys/module_bl_mynn_wrapper.F",
        32_623,
        "790d177dd4765855ad1b65efa9844e3001ab1c1b96c8335cac54deb09380ac66",
    ),
)

NOAHMP_SOURCE_ASSETS = (
    ReferenceAsset(
        "phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F",
        213_680,
        "9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890",
    ),
    ReferenceAsset(
        "phys/noahmp/src/module_sf_noahmplsm.F",
        577_175,
        "bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282",
    ),
    ReferenceAsset(
        "phys/noahmp/src/module_sf_noahmp_glacier.F",
        137_313,
        "bf94f3522c3b9c2c9cfbb34fa7e485ff58519106db434520968793409a520579",
    ),
    ReferenceAsset(
        "phys/noahmp/src/module_sf_noahmp_groundwater.F",
        24_027,
        "ff4b2d7bcb4e948d43af2797e29cbec1e3bb2875b3e9bbf2421c39a6f840cf50",
    ),
)

NOAHMP_PARAMETER_ASSETS = (
    ReferenceAsset(
        "phys/noahmp/parameters/MPTABLE.TBL",
        56_140,
        "7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d",
    ),
    ReferenceAsset(
        "phys/noahmp/parameters/SOILPARM.TBL",
        6_557,
        "1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f",
    ),
    ReferenceAsset(
        "phys/noahmp/parameters/GENPARM.TBL",
        261,
        "9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2",
    ),
)

REGISTRY_ASSETS = (
    ReferenceAsset(
        "Registry/Registry.EM_COMMON",
        436_138,
        "28cf7a2d369e898217d7231f4ed19310af7f0bec920f3cb57831e3dc4c80beec",
    ),
    ReferenceAsset(
        "Registry/registry.noahmp",
        38_937,
        "1d4c0e0dfb94d67b9ec2cc1044860f038d0096d5d8bb8ea73d6d82eeaff3b019",
    ),
)

PINNED_REFERENCE_ASSETS = (
    MYNN_SOURCE_ASSETS
    + NOAHMP_SOURCE_ASSETS
    + NOAHMP_PARAMETER_ASSETS
    + REGISTRY_ASSETS
)


# Registry/Registry.EM_COMMON:2680-2703.  The first executable lane is the
# exact WRF default option set.  Alternative Noah-MP science switches are
# separate validation targets and cannot silently inherit this identity.
NOAHMP_NAMELIST_DEFAULTS: Mapping[str, int | float] = MappingProxyType({
    "dveg": 4,
    "opt_crs": 1,
    "opt_btr": 1,
    "opt_run": 3,
    "opt_sfc": 1,
    "opt_frz": 1,
    "opt_inf": 1,
    "opt_rad": 3,
    "opt_alb": 2,
    "opt_snf": 1,
    "opt_tbot": 2,
    "opt_stc": 1,
    "opt_gla": 1,
    "opt_rsf": 1,
    "opt_soil": 1,
    "opt_pedo": 1,
    "opt_crop": 0,
    "opt_irr": 0,
    "opt_irrm": 0,
    "opt_infdv": 0,
    "opt_tdrn": 0,
    "soiltstep": 0.0,
    "noahmp_acc_dt": 0.0,
    "noahmp_output": 1,
})


# Registry/Registry.EM_COMMON:2473-2485.  MYNN surface layer and PBL are one
# coupled admission unit; these defaults are not aliases for YSU/MM5.
MYNN_NAMELIST_DEFAULTS: Mapping[str, bool | int | float] = MappingProxyType({
    "bl_mynn_tkeadvect": False,
    "bl_mynn_cloudpdf": 2,
    "bl_mynn_mixlength": 1,
    "bl_mynn_edmf": 1,
    "bl_mynn_edmf_mom": 1,
    "bl_mynn_edmf_tke": 0,
    "bl_mynn_mixscalars": 0,
    "bl_mynn_output": 0,
    "bl_mynn_cloudmix": 1,
    "bl_mynn_mixqt": 0,
    "icloud_bl": 1,
    "bl_mynn_closure": 2.6,
    "mfshconv": 1,
})


MYNN_ADVECTED_FIELDS = ("qke_adv",)
MYNN_PACKAGE_STATE_FIELDS = (
    "qke",
    "tke_pbl",
    "sh3d",
    "sm3d",
    "tsq",
    "qsq",
    "cov",
    "el_pbl",
)
MYNN_EDMF_MASS_FLUX_FIELDS = (
    "ktop_plume",
    "ztop_plume",
    "maxmf",
    "maxwidth",
)
MYNN_OPTIONAL_OUTPUT_FIELDS = (
    "edmf_a",
    "edmf_w",
    "edmf_thl",
    "edmf_qt",
    "edmf_ent",
    "edmf_qc",
    "sub_thl3d",
    "sub_sqv3d",
    "det_thl3d",
    "det_sqv3d",
)
MYNN_PBL_CLOUD_FIELDS = ("cldfra_bl", "qc_bl", "qi_bl")

MYNN_SURFACE_ORACLE_ASSET = ReferenceAsset(
    "surface-layer.csv",
    8_600,
    "049caf3b3add0a68730fc7e8953041425797715b3136169a7e290bef72ff9b04",
)
MYNN_SURFACE_ORACLE_CASES = (
    "stable_land",
    "unstable_land",
    "neutral_land",
    "snow_land",
    "stable_water",
    "unstable_water",
)
MYNN_PBL_LEVEL2_ORACLE_ASSET = ReferenceAsset(
    "pbl-level2.csv",
    18_056,
    "a953224da302091dc7a3ec805ea44393f452c481368d35d3ce4b8e150b5b8ea9",
)
MYNN_PBL_LEVEL2_ORACLE_CASES = (
    "stable_dry",
    "convective_dry",
    "neutral_shear",
    "moist_cloud",
)
MYNN_PBLH_SCALE_ORACLE_ASSET = ReferenceAsset(
    "pblh-scale.csv",
    10_682,
    "f6f8ef33b36048257786fab3096f77bc0ea320b7f6a377b023518b78218407b1",
)
MYNN_PBLH_SCALE_ORACLE_CASES = (
    "convective_land",
    "stable_land",
    "marine",
    "cold_pool",
)
MYNN_MIXLENGTH_ORACLE_ASSET = ReferenceAsset(
    "mixlength.csv",
    28_310,
    "733b689bfaf5ecfe522e7949003229fb0cdf8bea16e8759132b2f296ab7a0e77",
)
MYNN_MIXLENGTH_ORACLE_CASES = (
    "stable",
    "convective",
    "high_shear",
    "edmf_active",
)
MYNN_TURBULENCE_ORACLE_ASSET = ReferenceAsset(
    "turbulence.csv",
    51_156,
    "68671e652ea0ad871ae02d5f2b84e0906888aa8dce9f2b2935d98639d7852144",
)
MYNN_TURBULENCE_ORACLE_CASES = (
    "stable",
    "convective",
    "cloudy",
    "edmf_active",
)
MYNN_PREDICT_ORACLE_ASSET = ReferenceAsset(
    "predict.csv",
    17_300,
    "685d4780e32010f0a3e015108b4e5867bb82fe2d99b8e489b0f14bbba881f738",
)
MYNN_PREDICT_ORACLE_CASES = (
    "stable",
    "convective",
    "cloudy",
    "edmf_active",
)


# Registry/Registry.EM_COMMON:3149.  This is the complete state allocation
# expression for package noahmpscheme, normalized to lowercase exactly as the
# package line spells it.  It contains 224 fields and is hash-bound below so a
# dropped tail or accidental reorder cannot masquerade as the pinned contract.
NOAHMP_PACKAGE_STATE_FIELDS = (
    "isnowxy", "tvxy", "tgxy", "canliqxy", "canicexy", "eahxy",
    "tahxy", "cmxy", "chxy", "fwetxy", "sneqvoxy", "alboldxy",
    "qsnowxy", "qrainxy", "wslakexy", "zwtxy", "waxy", "wtxy",
    "tsnoxy", "zsnsoxy", "snicexy", "snliqxy", "lfmassxy",
    "rtmassxy", "stmassxy", "woodxy", "stblcpxy", "fastcpxy",
    "xsaixy", "taussxy", "t2mvxy", "t2mbxy", "q2mvxy", "q2mbxy",
    "tradxy", "neexy", "gppxy", "nppxy", "fvegxy", "qinxy",
    "runsfxy", "runsbxy", "ecanxy", "edirxy", "etranxy", "fsaxy",
    "firaxy", "aparxy", "psnxy", "savxy", "sagxy", "rssunxy",
    "rsshaxy", "bgapxy", "wgapxy", "tgvxy", "tgbxy", "chvxy",
    "chbxy", "shgxy", "shcxy", "shbxy", "evgxy", "evbxy", "ghvxy",
    "ghbxy", "irgxy", "ircxy", "irbxy", "trxy", "evcxy",
    "chleafxy", "chucxy", "chv2xy", "chb2xy", "chstarxy", "smoiseq",
    "smcwtdxy", "rechxy", "deeprechxy", "fdepthxy", "areaxy",
    "rivercondxy", "riverbedxy", "eqzwt", "pexpxy", "qrfxy", "qrfsxy",
    "qspringxy", "qspringsxy", "qslatxy", "stepwtd", "rechclim",
    "gddxy", "grainxy", "croptype", "planting", "harvest",
    "season_gdd", "cropcat", "pgsxy", "soilcomp", "soilcl1",
    "soilcl2", "soilcl3", "soilcl4", "irfract", "sifract", "mifract",
    "fifract", "irnumsi", "irnummi", "irnumfi", "irwatsi", "irwatmi",
    "irwatfi", "irsivol", "irmivol", "irfivol", "ireloss", "irrsplh",
    "td_fraction", "qtdrain", "acrech", "acqspring", "qlatxy",
    "qintsxy", "qintrxy", "qdripsxy", "qdriprxy", "qthrosxy",
    "qthrorxy", "qsnsubxy", "qsnfroxy", "qsubcxy", "qfrocxy",
    "qevacxy", "qdewcxy", "qfrzcxy", "qmeltcxy", "qsnbotxy",
    "pondingxy", "pahxy", "pahgxy", "pahvxy", "pahbxy", "fpicexy",
    "rainlsm", "snowlsm", "acints", "acintr", "acdripr", "acthror",
    "acevac", "acdewc", "forctlsm", "forcqlsm", "forcplsm",
    "forczlsm", "forcwlsm", "acrainlsm", "acrunsb", "acrunsf",
    "acecan", "acetran", "acedir", "acqlat", "acqrf", "acetlsm",
    "acsnowlsm", "acsubc", "acfroc", "acfrzc", "acmeltc", "acsnbot",
    "acponding", "acsnsub", "acsnfro", "acrainsnow", "acdrips",
    "acthros", "acsagb", "acirb", "acshb", "acevb", "acghb",
    "acpahb", "acsagv", "acirg", "acshg", "acevg", "acghv",
    "acpahg", "acsav", "acirc", "acshc", "acevc", "actr", "acpahv",
    "acswdnlsm", "acswuplsm", "aclwdnlsm", "aclwuplsm", "acshflsm",
    "aclhflsm", "acghflsm", "acpahlsm", "accanhs", "canhsxy",
    "soilenergy", "snowenergy", "acc_ssoil", "acc_qinsur", "acc_qseva",
    "acc_etrani", "aceflxb", "eflxbxy", "acc_dwaterxy", "acc_prcpxy",
    "acc_ecanxy", "acc_etranxy", "acc_edirxy", "qmeltxy", "acsnmelt",
)
NOAHMP_PACKAGE_STATE_SHA256 = (
    "717e40add8c199ab4d03eca449f103bd343cf835f68269f41323ebca54cd9a6d"
)


def _sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def validate_reference_assets(
    wrf_root: str | Path,
    assets: tuple[ReferenceAsset, ...] = PINNED_REFERENCE_ASSETS,
) -> tuple[ReferenceAsset, ...]:
    """Fail closed on a missing, resized, or byte-substituted WRF oracle."""

    root = Path(wrf_root)
    for asset in assets:
        current = root / asset.relative_path
        if not current.is_file():
            raise FileNotFoundError(f"missing WRF reference asset {current}")
        actual_bytes = current.stat().st_size
        if actual_bytes != asset.bytes:
            raise ValueError(
                f"WRF reference asset {current} has {actual_bytes} bytes; "
                f"expected {asset.bytes}"
            )
        actual_sha256 = _sha256_file(current)
        if actual_sha256 != asset.sha256:
            raise ValueError(
                f"WRF reference asset {current} SHA-256 {actual_sha256}; "
                f"expected {asset.sha256}"
            )
    return assets


def _same_option_type_and_value(
    actual: bool | int | float,
    expected: bool | int | float,
) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and math.isfinite(float(actual))
        and float(actual) == expected
    )


def _resolve_default_options(
    suite: str,
    supplied: Mapping[str, bool | int | float],
    defaults: Mapping[str, bool | int | float],
) -> Mapping[str, bool | int | float]:
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        raise ValueError(f"unknown {suite} options: {', '.join(unknown)}")
    changed = [
        f"{key}={supplied[key]!r} (validated default {defaults[key]!r})"
        for key in defaults
        if key in supplied
        and not _same_option_type_and_value(supplied[key], defaults[key])
    ]
    if changed:
        raise ValueError(
            f"{suite} first executable lane is pinned to WRF v4.6.1 "
            f"defaults; unsupported overrides: " + "; ".join(changed)
        )
    resolved = dict(defaults)
    resolved.update(supplied)
    return MappingProxyType(resolved)


def resolve_default_noahmp_options(
    supplied: Mapping[str, int | float] | None = None,
) -> Mapping[str, int | float]:
    """Resolve omitted Noah-MP values and reject unvalidated option modes."""

    return _resolve_default_options(
        "Noah-MP", supplied or {}, NOAHMP_NAMELIST_DEFAULTS
    )


def resolve_default_mynn_options(
    supplied: Mapping[str, bool | int | float] | None = None,
) -> Mapping[str, bool | int | float]:
    """Resolve omitted MYNN values and reject unvalidated option modes."""

    return _resolve_default_options("MYNN", supplied or {}, MYNN_NAMELIST_DEFAULTS)


def reference_contract_receipt(
    wrf_root: str | Path,
) -> dict[str, object]:
    """Validate the pinned tree and return a serializable provenance receipt."""

    root = Path(wrf_root).resolve()
    assets = validate_reference_assets(root)
    return {
        "schema": "gpuwm.noahmp-mynn-reference-contract/v1",
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "wrf_version": WRF_REFERENCE_VERSION,
        "wrf_commit": WRF_REFERENCE_COMMIT,
        "noahmp_commit": NOAHMP_REFERENCE_COMMIT,
        "root": str(root),
        "assets": [
            {
                "relative_path": asset.relative_path,
                "bytes": asset.bytes,
                "sha256": asset.sha256,
            }
            for asset in assets
        ],
        "selectors": {
            "sf_surface_physics": SF_SURFACE_PHYSICS,
            "sf_sfclay_physics": SF_SFCLAY_PHYSICS,
            "bl_pbl_physics": BL_PBL_PHYSICS,
        },
        "noahmp_package_state_count": len(NOAHMP_PACKAGE_STATE_FIELDS),
        "noahmp_package_state_sha256": NOAHMP_PACKAGE_STATE_SHA256,
    }


_actual_noahmp_state_sha256 = hashlib.sha256(
    ",".join(NOAHMP_PACKAGE_STATE_FIELDS).encode("ascii")
).hexdigest()
if len(NOAHMP_PACKAGE_STATE_FIELDS) != 224:
    raise RuntimeError("Noah-MP Registry package state count drift")
if _actual_noahmp_state_sha256 != NOAHMP_PACKAGE_STATE_SHA256:
    raise RuntimeError("Noah-MP Registry package state inventory drift")


__all__ = [
    "BL_PBL_PHYSICS",
    "CONTRACT_ID",
    "MYNN_ADVECTED_FIELDS",
    "MYNN_EDMF_MASS_FLUX_FIELDS",
    "MYNN_NAMELIST_DEFAULTS",
    "MYNN_MIXLENGTH_ORACLE_ASSET",
    "MYNN_MIXLENGTH_ORACLE_CASES",
    "MYNN_OPTIONAL_OUTPUT_FIELDS",
    "MYNN_PACKAGE_STATE_FIELDS",
    "MYNN_PBL_CLOUD_FIELDS",
    "MYNN_PBL_LEVEL2_ORACLE_ASSET",
    "MYNN_PBL_LEVEL2_ORACLE_CASES",
    "MYNN_PBLH_SCALE_ORACLE_ASSET",
    "MYNN_PBLH_SCALE_ORACLE_CASES",
    "MYNN_PREDICT_ORACLE_ASSET",
    "MYNN_PREDICT_ORACLE_CASES",
    "MYNN_SOURCE_ASSETS",
    "MYNN_SURFACE_ORACLE_ASSET",
    "MYNN_SURFACE_ORACLE_CASES",
    "MYNN_TURBULENCE_ORACLE_ASSET",
    "MYNN_TURBULENCE_ORACLE_CASES",
    "NOAHMP_NAMELIST_DEFAULTS",
    "NOAHMP_PACKAGE_STATE_FIELDS",
    "NOAHMP_PACKAGE_STATE_SHA256",
    "NOAHMP_PARAMETER_ASSETS",
    "NOAHMP_REFERENCE_COMMIT",
    "NOAHMP_SOURCE_ASSETS",
    "PINNED_REFERENCE_ASSETS",
    "REGISTRY_ASSETS",
    "ReferenceAsset",
    "SF_SFCLAY_PHYSICS",
    "SF_SURFACE_PHYSICS",
    "WRF_REFERENCE_COMMIT",
    "WRF_REFERENCE_VERSION",
    "reference_contract_receipt",
    "resolve_default_mynn_options",
    "resolve_default_noahmp_options",
    "validate_reference_assets",
]
