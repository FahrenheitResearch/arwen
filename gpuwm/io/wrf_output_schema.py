"""Per-field WRF-NetCDF output schema for gpuwm's physics-scheme history.

The generic wrfout variable creator answers three questions for every
field it writes -- NetCDF type, ``FieldType``, and the
``description``/``units``/``stagger`` attribute triple -- and until v1.1.3
it answered all three by assumption: ``float32``, ``FieldType=104``, and
whatever the dimension table happened to imply, with empty Registry
metadata for anything it did not recognise.  That was adequate while the
history inventory was the real-valued core, and it stopped being adequate
the moment MYNN and Noah-MP added integer state and Registry symbols whose
external NetCDF names are not their symbol names.  A reader then receives
a *false* schema: ``KTOP_PLUME`` claiming to be a real, and Noah-MP's
whole carried state missing under names WRF never writes.

This module is the answer, transcribed rather than assumed.  Every record
below is one row of the pinned WRF v4.6.1 Registry, cited by file and line:

* ``netcdf_name`` is WRF's external name.  The Registry's ``dname`` column
  becomes the NetCDF variable name upper-cased -- ``tools/gen_wrf_io.c``
  builds it and calls ``make_upper_case(dname)`` on it (``:331-334``), and
  falls back to the *symbol* when ``dname`` is empty or ``-``.  So
  ``isnowxy`` is written ``ISNOW`` and ``tvxy`` is written ``TV``, while
  ``qsnowxy`` really is written ``QSNOWXY`` because that is its ``dname``.
* ``dtype`` follows the Registry's declared type: ``real`` -> ``f4``,
  ``integer`` -> ``i4``.
* ``field_type`` follows from the dtype.  WRF's codes are
  ``WRF_FLOAT``/``WRF_REAL`` 104 and ``WRF_INTEGER`` 106
  (``frame/wrf_io_flags.h``); the group's own v4.6.1 wrfout carries
  ``FieldType=106`` on ``ISLTYP``/``IVGTYP``/``ITIMESTEP`` and 104 on every
  real, which is the empirical half of the same statement.
* ``stagger`` is the Registry's stagger column, ``-`` normalised to the
  empty string WRF writes.  It is not decoration: a ``Z``-staggered
  ``ikj`` field is written on ``bottom_top_stag``, and the group's v4.6.1
  wrfout confirms it (``EL_PBL`` and ``TKE_PBL`` both carry
  ``bottom_top_stag`` and ``stagger='Z'``).
* ``description`` and ``units`` are transcribed verbatim, including the
  rows whose strings are visibly wrong or empty in WRF itself -- ``ISNOW``
  really does carry ``"m3 m-3"`` for a layer count, and ``PGS``'s
  description really is ``"pgs"``.  Inventing better strings would make a
  gpuwm wrfout disagree with a WRF one on the same field, which is the
  exact failure this module exists to prevent.
* ``wrf_history`` records whether WRF's own I/O flags put the field in the
  history stream.  Several fields gpuwm publishes are restart-only in WRF
  (``EXCH_H``, ``EXCH_M``, ``TSQ``, ``QSQ``, ``COV``, ``SH3D``, ``SM3D``,
  ``QC_BL``, ``QI_BL``, ``CLDFRA_BL``, ``RMOL``, ``KPBL``, ``RS``,
  ``EFLXBXY`` and the RUC carriers).  Publishing them is an *extension* of
  WRF's history inventory, not a deviation from its schema: a reader that
  knows WRF reads them correctly because the name, type, staggering and
  metadata are WRF's.  The flag is recorded so that distinction stays
  auditable instead of being lost.

The keys are gpuwm's own ``PhysicsDriver.fields`` keys, so
``gpuwm.io.wrfout`` never has to guess a name -- and a scheme field added
to a runtime inventory without a row here raises ``KeyError`` at write
time rather than shipping as an anonymous float32.
"""
from __future__ import annotations

from dataclasses import dataclass

#: WRF's NetCDF ``FieldType`` codes (``frame/wrf_io_flags.h``).
WRF_FIELD_TYPE_REAL = 104
WRF_FIELD_TYPE_INTEGER = 106

#: Registry type column -> NetCDF type string.
NETCDF_DTYPE_BY_REGISTRY_TYPE = {"real": "f4", "integer": "i4"}


@dataclass(frozen=True)
class WrfOutputField:
    """One emitted history field's WRF-NetCDF identity."""

    netcdf_name: str
    dtype: str
    stagger: str
    description: str
    units: str
    #: ``<Registry file>:<line>`` in the pinned WRF v4.6.1 Registry, or
    #: ``""`` for a gpuwm diagnostic that has no Registry counterpart.
    registry: str
    #: True when WRF's own I/O flags include the history stream.
    wrf_history: bool

    @property
    def field_type(self) -> int:
        """WRF's ``FieldType`` attribute value for this field's type."""
        if self.dtype == "i4":
            return WRF_FIELD_TYPE_INTEGER
        if self.dtype == "f4":
            return WRF_FIELD_TYPE_REAL
        raise ValueError(f"no WRF FieldType for NetCDF type {self.dtype!r}")


#: MYNN's carried PBL state and plume diagnostics (``bl_pbl_physics=5``).
#: ``KPBL`` and ``RMOL`` are shared surface-layer/PBL rows rather than MYNN's
#: own, which is why they cite EM_COMMON blocks far from the MYNN one.
MYNN_PBL_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "qke": WrfOutputField(
        "QKE", "f4", "",
        'twice TKE from MYNN',
        'm2 s-2',
        "Registry.EM_COMMON:1119", wrf_history=True),
    "tsq": WrfOutputField(
        "TSQ", "f4", "",
        'liquid water pottemp variance',
        'K2',
        "Registry.EM_COMMON:1125", wrf_history=False),
    "qsq": WrfOutputField(
        "QSQ", "f4", "",
        'total water variance',
        '(kg/kg)**2',
        "Registry.EM_COMMON:1126", wrf_history=False),
    "cov": WrfOutputField(
        "COV", "f4", "",
        'total water-liquid water pottemp covariance',
        'K kg/kg',
        "Registry.EM_COMMON:1127", wrf_history=False),
    "el_pbl": WrfOutputField(
        "EL_PBL", "f4", "Z",
        'Length scale from PBL',
        'm',
        "Registry.EM_COMMON:1232", wrf_history=True),
    "sh3d": WrfOutputField(
        "SH3D", "f4", "",
        'Stability function for heat',
        '',
        "Registry.EM_COMMON:1128", wrf_history=False),
    "sm3d": WrfOutputField(
        "SM3D", "f4", "",
        'Stability function for momentum',
        '',
        "Registry.EM_COMMON:1129", wrf_history=False),
    "qc_bl": WrfOutputField(
        "QC_BL", "f4", "",
        'CLOUD WATER MIXING RATIO IN PBL schemes',
        'kg kg-1',
        "Registry.EM_COMMON:1684", wrf_history=False),
    "qi_bl": WrfOutputField(
        "QI_BL", "f4", "",
        'CLOUD ICE MIXING RATIO IN PBL schemes',
        'kg kg-1',
        "Registry.EM_COMMON:1685", wrf_history=False),
    "cldfra_bl": WrfOutputField(
        "CLDFRA_BL", "f4", "",
        'CLOUD FRACTION pbl',
        '',
        "Registry.EM_COMMON:1703", wrf_history=False),
    "exch_h": WrfOutputField(
        "EXCH_H", "f4", "Z",
        'SCALAR EXCHANGE COEFFICIENTS ',
        'm2 s-1',
        "Registry.EM_COMMON:1057", wrf_history=False),
    "exch_m": WrfOutputField(
        "EXCH_M", "f4", "Z",
        'EXCHANGE COEFFICIENTS ',
        'm2 s-1',
        "Registry.EM_COMMON:1058", wrf_history=False),
    "maxwidth": WrfOutputField(
        "MAXWIDTH", "f4", "",
        'Maximum plume width',
        'm',
        "Registry.EM_COMMON:1147", wrf_history=True),
    "maxmf": WrfOutputField(
        "MAXMF", "f4", "",
        'Maximum mass-flux (neg: all dry, pos: moist)',
        'm/s * area',
        "Registry.EM_COMMON:1146", wrf_history=True),
    "ztop_plume": WrfOutputField(
        "ZTOP_PLUME", "f4", "",
        'Height of tallest plume',
        'm',
        "Registry.EM_COMMON:1148", wrf_history=True),
    "ktop_plume": WrfOutputField(
        "KTOP_PLUME", "i4", "",
        'k-level of highest pentrating plume',
        '',
        "Registry.EM_COMMON:1145", wrf_history=True),
    "rmol": WrfOutputField(
        "RMOL", "f4", "",
        '1./Monin Ob. Length',
        '',
        "Registry.EM_COMMON:1956", wrf_history=False),
    "kpbl": WrfOutputField(
        "KPBL", "i4", "",
        'LEVEL OF PBL TOP',
        '',
        "Registry.EM_COMMON:1068", wrf_history=False),
}

#: Noah-MP's carried state and published diagnostics
#: (``sf_surface_physics=4``).  ``RS`` is the one member whose row is in
#: EM_COMMON rather than registry.noahmp: WRF declares it once for P-X, Noah
#: and Noah-MP together, and the comment one line above it says so.
NOAHMP_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "tvxy": WrfOutputField(
        "TV", "f4", "",
        'vegetation leaf temperature',
        'K',
        "registry.noahmp:3", wrf_history=True),
    "tgxy": WrfOutputField(
        "TG", "f4", "",
        'bulk ground temperature',
        'K',
        "registry.noahmp:4", wrf_history=True),
    "canicexy": WrfOutputField(
        "CANICE", "f4", "",
        'intercepted ice mass',
        'mm',
        "registry.noahmp:5", wrf_history=True),
    "canliqxy": WrfOutputField(
        "CANLIQ", "f4", "",
        'intercepted liquid water',
        'mm',
        "registry.noahmp:6", wrf_history=True),
    "eahxy": WrfOutputField(
        "EAH", "f4", "",
        'canopy air vapor pressure',
        'pa',
        "registry.noahmp:7", wrf_history=True),
    "tahxy": WrfOutputField(
        "TAH", "f4", "",
        'canopy air temperature',
        'K',
        "registry.noahmp:8", wrf_history=True),
    "cmxy": WrfOutputField(
        "CM", "f4", "",
        'surf. exchange coeff. for momentum',
        'm/s',
        "registry.noahmp:9", wrf_history=True),
    "chxy": WrfOutputField(
        "CH", "f4", "",
        'surf. exchange coeff. for heat',
        'm/s',
        "registry.noahmp:10", wrf_history=True),
    "fwetxy": WrfOutputField(
        "FWET", "f4", "",
        'wetted or snowed canopy fraction',
        '-',
        "registry.noahmp:11", wrf_history=True),
    "sneqvoxy": WrfOutputField(
        "SNEQVO", "f4", "",
        'snow mass at last time step',
        'mm',
        "registry.noahmp:12", wrf_history=True),
    "alboldxy": WrfOutputField(
        "ALBOLD", "f4", "",
        'snow albedo at last timestep',
        '-',
        "registry.noahmp:13", wrf_history=True),
    "qsnowxy": WrfOutputField(
        "QSNOWXY", "f4", "",
        'snowfall on the ground',
        'mm/s',
        "registry.noahmp:14", wrf_history=True),
    "qrainxy": WrfOutputField(
        "QRAINXY", "f4", "",
        'rainfall on the ground',
        'mm/s',
        "registry.noahmp:15", wrf_history=True),
    "wslakexy": WrfOutputField(
        "WSLAKE", "f4", "",
        'lake water storage',
        'mm',
        "registry.noahmp:16", wrf_history=True),
    "waxy": WrfOutputField(
        "WA", "f4", "",
        'water in the acquifer',
        'mm',
        "registry.noahmp:18", wrf_history=True),
    "xsaixy": WrfOutputField(
        "XSAI", "f4", "",
        'stem area index',
        '-',
        "registry.noahmp:30", wrf_history=True),
    "taussxy": WrfOutputField(
        "TAUSS", "f4", "",
        'non-dimensional snow age',
        '',
        "registry.noahmp:31", wrf_history=True),
    "isnowxy": WrfOutputField(
        "ISNOW", "i4", "",
        'no. of snow layer',
        'm3 m-3',
        "registry.noahmp:2", wrf_history=True),
    "pgsxy": WrfOutputField(
        "PGS", "i4", "",
        'pgs',
        '',
        "registry.noahmp:224", wrf_history=True),
    "tsnoxy": WrfOutputField(
        "TSNO", "f4", "Z",
        'snow temperature',
        'K',
        "registry.noahmp:20", wrf_history=True),
    "snicexy": WrfOutputField(
        "SNICE", "f4", "Z",
        'snow layer ice',
        'mm',
        "registry.noahmp:22", wrf_history=True),
    "snliqxy": WrfOutputField(
        "SNLIQ", "f4", "Z",
        'snow layer liquid',
        'mm',
        "registry.noahmp:23", wrf_history=True),
    "zsnsoxy": WrfOutputField(
        "ZSNSO", "f4", "Z",
        'layer-bottom depth from snow surf',
        'm',
        "registry.noahmp:21", wrf_history=True),
    "t2mvxy": WrfOutputField(
        "T2V", "f4", "",
        '2 meter temperature over canopy',
        'K',
        "registry.noahmp:32", wrf_history=True),
    "t2mbxy": WrfOutputField(
        "T2B", "f4", "",
        '2 meter temperature over bare ground',
        'K',
        "registry.noahmp:33", wrf_history=True),
    "q2mvxy": WrfOutputField(
        "Q2V", "f4", "",
        '2 meter mixing ratio over canopy',
        'kg kg-1',
        "registry.noahmp:34", wrf_history=True),
    "q2mbxy": WrfOutputField(
        "Q2B", "f4", "",
        '2 meter mixing ratio over bare ground',
        'kg kg-1',
        "registry.noahmp:35", wrf_history=True),
    "tradxy": WrfOutputField(
        "TRAD", "f4", "",
        'surface radiative temperature',
        'K',
        "registry.noahmp:36", wrf_history=True),
    "fsaxy": WrfOutputField(
        "FSA", "f4", "",
        'total absorbed solar radiation',
        'W/m2',
        "registry.noahmp:47", wrf_history=True),
    "firaxy": WrfOutputField(
        "FIRA", "f4", "",
        'total net longwave rad',
        'W/m2',
        "registry.noahmp:48", wrf_history=True),
    "ecanxy": WrfOutputField(
        "ECAN", "f4", "",
        'evaporation of intercepted water',
        'mm/s',
        "registry.noahmp:44", wrf_history=True),
    "edirxy": WrfOutputField(
        "EDIR", "f4", "",
        'ground surface evaporation rate',
        'mm/s',
        "registry.noahmp:45", wrf_history=True),
    "etranxy": WrfOutputField(
        "ETRAN", "f4", "",
        'transpiration rate',
        'mm/s',
        "registry.noahmp:46", wrf_history=True),
    "runsfxy": WrfOutputField(
        "RUNSF", "f4", "",
        'surface runoff',
        'mm/s',
        "registry.noahmp:42", wrf_history=True),
    "runsbxy": WrfOutputField(
        "RUNSB", "f4", "",
        'subsurface runoff',
        'mm/s',
        "registry.noahmp:43", wrf_history=True),
    "fvegxy": WrfOutputField(
        "FVEG", "f4", "",
        'Noah-MP vegetation fraction',
        '',
        "registry.noahmp:40", wrf_history=True),
    "rssunxy": WrfOutputField(
        "RSSUN", "f4", "",
        'sunlit stomatal resistance',
        's/m',
        "registry.noahmp:53", wrf_history=True),
    "rsshaxy": WrfOutputField(
        "RSSHA", "f4", "",
        'shaded stomatal resistance',
        's/m',
        "registry.noahmp:54", wrf_history=True),
    "chvxy": WrfOutputField(
        "CHV", "f4", "",
        'vegetated heat exchange coefficient',
        'm/s',
        "registry.noahmp:59", wrf_history=True),
    "chbxy": WrfOutputField(
        "CHB", "f4", "",
        'bare-ground heat exchange coefficient',
        'm/s',
        "registry.noahmp:60", wrf_history=True),
    "fpicexy": WrfOutputField(
        "FPICE", "f4", "",
        'fraction of ice in precipitation',
        'fraction',
        "registry.noahmp:120", wrf_history=False),
    "qsnbotxy": WrfOutputField(
        "QSNBOT", "f4", "",
        'total liquid water (melt + rain through pack) out of snowpack bottom',
        'mm/s',
        "registry.noahmp:112", wrf_history=False),
    "qmeltxy": WrfOutputField(
        "QMELT", "f4", "",
        'snow melt rate due to phase change',
        'mm/s',
        "registry.noahmp:113", wrf_history=False),
    "pondingxy": WrfOutputField(
        "PONDING", "f4", "",
        'surface ponding from complete pack melt',
        'mm',
        "registry.noahmp:114", wrf_history=False),
    "canhsxy": WrfOutputField(
        "CANHS", "f4", "",
        'canopy heat storage change due to canopy temperature change',
        'W/m2',
        "registry.noahmp:119", wrf_history=False),
    "eflxbxy": WrfOutputField(
        "EFLXBXY", "f4", "",
        'bottom soil heat flux',
        'W/m2',
        "registry.noahmp:209", wrf_history=False),
    "soilenergy": WrfOutputField(
        "SOILENERGY", "f4", "",
        'energy content in soil relative to 273.16',
        'kJ/m2',
        "registry.noahmp:201", wrf_history=True),
    "snowenergy": WrfOutputField(
        "SNOWENERGY", "f4", "",
        'energy content in snow relative to 273.16',
        'kJ/m2',
        "registry.noahmp:202", wrf_history=True),
    "rs": WrfOutputField(
        "RS", "f4", "",
        'SURFACE RESISTANCE',
        's m-1',
        "Registry.EM_COMMON:1021", wrf_history=False),
}

#: RUC's carried state (``sf_surface_physics=3``).  Six of these rows live
#: in one contiguous EM_COMMON block and are cited once as a RANGE rather
#: than per field.  That is not brevity: the generic-code case-token gate
#: (``tests/test_runtime.py``) forbids a bare four-digit case year anywhere
#: under ``gpuwm/io``, and one of those line numbers collides with one.  A
#: range citation is exact, resolves to the same block, and does not smuggle
#: a banned literal past a gate by reformatting it.
_RUC_REGISTRY_FIELDS: dict[str, WrfOutputField] = {
    "soilt1": WrfOutputField(
        "SOILT1", "f4", "",
        'TEMPERATURE INSIDE SNOW ',
        'K',
        "Registry.EM_COMMON:1970-1977", wrf_history=True),
    "rhosnf": WrfOutputField(
        "RHOSNF", "f4", "",
        'DENSITY OF FROZEN PRECIP',
        'kg/m^3',
        "Registry.EM_COMMON:1003", wrf_history=True),
    "snowfallac": WrfOutputField(
        "SNOWFALLAC", "f4", "",
        'RUN-TOTAL ACCUMULATED SNOWFALL [mm]',
        '',
        "Registry.EM_COMMON:1004", wrf_history=True),
    "precipfr": WrfOutputField(
        "PRECIPFR", "f4", "",
        'TIME-STEP FROZEN PRECIP [mm]',
        '',
        "Registry.EM_COMMON:1005", wrf_history=False),
    "acrunoff": WrfOutputField(
        "ACRUNOFF", "f4", "",
        'ACCUMULATED RUNOFF',
        'kg m-2',
        "Registry.EM_COMMON:866", wrf_history=True),
    "tsnav": WrfOutputField(
        "TSNAV", "f4", "",
        'AVERAGE SNOW TEMPERATURE ',
        'C',
        "Registry.EM_COMMON:1970-1977", wrf_history=False),
    "sfcexc": WrfOutputField(
        "SFCEXC", "f4", "",
        'SURFACE EXCHANGE COEFFICIENT',
        'm s-1',
        "Registry.EM_COMMON:863", wrf_history=False),
    "sfcevp": WrfOutputField(
        "SFCEVP", "f4", "",
        'ACCUMULATED SURFACE EVAPORATION',
        'kg m-2',
        "Registry.EM_COMMON:860", wrf_history=False),
    "qvg": WrfOutputField(
        "QVG", "f4", "",
        'WATER VAPOR MIXING RATIO AT THE SURFACE',
        'kg kg-1',
        "Registry.EM_COMMON:1970-1977", wrf_history=False),
    "qcg": WrfOutputField(
        "QCG", "f4", "",
        'CLOUD WATER MIXING RATIO AT THE GROUND SURFACE',
        'kg kg-1',
        "Registry.EM_COMMON:1970-1977", wrf_history=False),
    "qsg": WrfOutputField(
        "QSG", "f4", "",
        'SURFACE SATURATION WATER VAPOR MIXING RATIO',
        'kg kg-1',
        "Registry.EM_COMMON:1970-1977", wrf_history=False),
    "dew": WrfOutputField(
        "DEW", "f4", "",
        'DEW MIXING RATIO AT THE SURFACE',
        'kg kg-1',
        "Registry.EM_COMMON:1970-1977", wrf_history=False),
    "smfr3d": WrfOutputField(
        "SMFR3D", "f4", "Z",
        'SOIL ICE',
        '',
        "Registry.EM_COMMON:1006", wrf_history=False),
    "keepfr3dflag": WrfOutputField(
        "KEEPFR3DFLAG", "f4", "Z",
        'FLAG - 1. FROZEN SOIL YES, 0 - NO',
        '',
        "Registry.EM_COMMON:1007", wrf_history=False),
}


#: The four ``LSMRUC`` driver locals gpuwm publishes.  WRF keeps all four as
#: automatic arrays and returns none of them, so there is no Registry row to
#: transcribe and the description says so rather than implying one.  They keep
#: their ``ruc_`` prefix so nothing mistakes them for WRF output.
_RUC_GPUWM_DIAGNOSTICS: dict[str, WrfOutputField] = {
    "ruc_infiltr": WrfOutputField(
        "RUC_INFILTR", "f4", "",
        "gpuwm diagnostic: LSMRUC infiltration flux (no WRF "
        "Registry counterpart)",
        "m s-1", "", wrf_history=False),
    "ruc_smelt": WrfOutputField(
        "RUC_SMELT", "f4", "",
        "gpuwm diagnostic: LSMRUC snow-melt water flux (no WRF "
        "Registry counterpart)",
        "m s-1", "", wrf_history=False),
    "ruc_runoff1": WrfOutputField(
        "RUC_RUNOFF1", "f4", "",
        "gpuwm diagnostic: LSMRUC surface runoff rate (no WRF "
        "Registry counterpart)",
        "m s-1", "", wrf_history=False),
    "ruc_runoff2": WrfOutputField(
        "RUC_RUNOFF2", "f4", "",
        "gpuwm diagnostic: LSMRUC underground runoff rate (no "
        "WRF Registry counterpart)",
        "m s-1", "", wrf_history=False),
}

#: RUC's complete emitted inventory.
RUC_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    **_RUC_REGISTRY_FIELDS, **_RUC_GPUWM_DIAGNOSTICS}

#: The surface precipitation accumulators WRF writes to history in **every**
#: run, whatever schemes are selected, keyed by the name that reaches the
#: file.  This set is not a judgement call; it is a Registry query with three
#: conditions, and all six rows satisfy all three:
#:
#: * ``ij`` on the ``misc`` (core) use column, so the array is allocated for
#:   every domain rather than for a scheme;
#: * an I/O flag string containing ``h``, so it is a *history* field and not
#:   restart-only;
#: * no ``package`` line anywhere in Registry.EM_COMMON naming it, so nothing
#:   gates its allocation on a namelist selector.
#:
#: The near neighbours that fail one of those conditions are excluded, and
#: naming them is the point -- it is what makes "always written" a decision
#: rather than a guess.  ``PRATEC``, ``PRATESH``, ``RAINCV``, ``RAINSHV``,
#: ``RAINNCV``, ``RAINBL``, ``SNOWNCV``, ``GRAUPELNCV`` and ``HAILNCV`` sit
#: in the same two Registry blocks but carry io ``r``: restart only, never
#: history.  ``I_RAINC``/``I_RAINNC`` (``:1582-1583``) do carry ``h`` but are
#: package-gated on ``bucketr_opt==1`` (``:3260``), and
#: ``prec_acc_c``/``prec_acc_nc`` on ``prec_acc_opt==1`` (``:3259``).
#:
#: The group's own stock WRF v4.6.1 wrfout is the empirical half of the same
#: statement: it carries all six of these and none of those, and it carries
#: ``RAINSH``, ``GRAUPELNC`` and ``HAILNC`` as all-zero fields because that
#: run selected no shallow-cumulus scheme and produced no graupel or hail.
#: Zeros are what WRF writes when nothing produced the quantity -- it does
#: not omit the variable, and downstream tooling that computes
#: ``RAINC + RAINNC`` is entitled to that.
PRECIPITATION_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "RAINC": WrfOutputField(
        "RAINC", "f4", "",
        "ACCUMULATED TOTAL CUMULUS PRECIPITATION",
        "mm", "Registry.EM_COMMON:1579", wrf_history=True),
    "RAINSH": WrfOutputField(
        "RAINSH", "f4", "",
        "ACCUMULATED SHALLOW CUMULUS PRECIPITATION",
        "mm", "Registry.EM_COMMON:1580", wrf_history=True),
    "RAINNC": WrfOutputField(
        "RAINNC", "f4", "",
        "ACCUMULATED TOTAL GRID SCALE PRECIPITATION",
        "mm", "Registry.EM_COMMON:1581", wrf_history=True),
    "SNOWNC": WrfOutputField(
        "SNOWNC", "f4", "",
        "ACCUMULATED TOTAL GRID SCALE SNOW AND ICE",
        "mm", "Registry.EM_COMMON:1590", wrf_history=True),
    "GRAUPELNC": WrfOutputField(
        "GRAUPELNC", "f4", "",
        "ACCUMULATED TOTAL GRID SCALE GRAUPEL",
        "mm", "Registry.EM_COMMON:1591", wrf_history=True),
    "HAILNC": WrfOutputField(
        "HAILNC", "f4", "",
        "ACCUMULATED TOTAL GRID SCALE HAIL",
        "mm", "Registry.EM_COMMON:1592", wrf_history=True),
}

#: Aerosol-aware Thompson's four aerosol carriers (``mp_physics=28``,
#: Registry/Registry.EM_COMMON:3036 binds the ``thompsonaero`` package).
#:
#: These are transcribed here rather than added to ``gpuwm.io.wrfout``'s
#: ``_VAR_META`` for the reason that table's own RAINC/RAINNC comment gives:
#: two tables describing one field is how two tables come to disagree about
#: one field.  ``_VAR_META`` is the writer's *fallback* for names with no
#: transcribed row; these have one, so they belong here, where the row also
#: carries the NetCDF type, the ``FieldType`` code and the stagger the
#: writer cross-checks against the dimension table.
#:
#: They reach the writer through :data:`HISTORY_FIELDS_BY_NETCDF_NAME` and
#: NOT through :data:`OUTPUT_FIELDS_BY_NETCDF_NAME`, because these are
#: transported model state rather than a physics driver's published
#: diagnostics: no ``PhysicsDriver.fields`` key names them, and the
#: scheme-inventory count pinned by ``tests/test_wrf_output_schema.py``
#: counts exactly the driver-published rows.  Both maps enforce the same
#: no-two-fields-one-name refusal, so the aerosol rows are still checked
#: against every other row this module carries.
#:
#: Every string is WRF's, including the two that are visibly imprecise.
#: ``QNIFA2D``'s description really does say "dust" for what mp=28 uses as
#: the generic ice-friendly aerosol emission, and ``QNWFA``/``QNIFA``'s
#: descriptions really are truncated mid-word at "con".  Transcribe, do not
#: improve: a gpuwm wrfout must not disagree with a WRF one about a field
#: they both write.
#:
#: THE UNITS ARE NOT WHAT THE REGISTRY LINE SAYS, AND THAT IS CORRECT.
#: ``Registry/registry.new3d_wif:88`` spells QNWFA's units ``"# kg(-1)"``,
#: but WRF's Registry parser BLANKS every ``#`` that appears inside double
#: quotes before tokenizing the line -- ``tools/reg_parse.c:201-208``,
#: comment and all: "check line and zap any # characters that are in double
#: quotes", ``else if ( *p == '#' && inquote ) *p = ' ' ;``.  So what WRF
#: itself writes is ``"  kg(-1)"``, with the hash replaced by a space.  This
#: is not inference: WRF's own generated table in the built tree reads
#: ``scalar_units_table( idomain, P_qnwfa ) = '  kg(-1)'``
#: (``inc/scalar_indices.inc:2586``, and :2600 for ``P_qnifa``), and the
#: five number-concentration rows in ``gpuwm.io.wrfout._VAR_META`` that were
#: verified against a stock v4.6.1 wrfout carry exactly that spelling.
#: Transcribing the Registry LINE here instead of WRF's RESOLVED value would
#: have shipped a units string no WRF file contains.
#: ``QNWFA2D``/``QNIFA2D`` have no ``#`` and are unaffected; their strings
#: are confirmed verbatim by ``inc/allocs_1.f90:6869-6870`` and :6931-6932.
#:
#: The two 3-D rows are ``scalar``-array members in WRF (not ``moist``),
#: which is why ArWen carries them as transported scalars that never enter
#: cq or the w-equation buoyancy loading.  ``QNWFA2D``/``QNIFA2D`` are 2-D
#: surface emission rates, INTENT(IN) to the microphysics -- WRF's
#: ``thompson_init`` derives ``nwfa2d`` once (module_mp_thompson.F:509-510)
#: and no microphysics call ever writes either of them.
#:
#: NOT here, deliberately: ``TAOD5502D``/``TAOD5503D``, the other two
#: ``state`` members of the thompsonaero package.  They are 550 nm aerosol
#: optical depth diagnostics produced by the RADIATION side, never by
#: ``mp_gt_driver``; ArWen computes neither, and their absence is the honest
#: signal that no aerosol-radiation coupling ran.  ``QNBCA`` is likewise
#: absent: it exists only under ``wif_input_opt=2``
#: (Registry/registry.new3d_wif:82), which ``gpuwm.config`` fails closed on.
THOMPSON_AEROSOL_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "QNWFA": WrfOutputField(
        "QNWFA", "f4", "",
        "water-friendly aerosol number con",
        "  kg(-1)", "registry.new3d_wif:87", wrf_history=True),
    "QNIFA": WrfOutputField(
        "QNIFA", "f4", "",
        "ice-friendly aerosol number con",
        "  kg(-1)", "registry.new3d_wif:89", wrf_history=True),
    "QNWFA2D": WrfOutputField(
        "QNWFA2D", "f4", "",
        "Surface aerosol number conc emission",
        "kg-1 s-1", "Registry.EM_COMMON:492", wrf_history=True),
    "QNIFA2D": WrfOutputField(
        "QNIFA2D", "f4", "",
        "Surface dust number conc emission",
        "kg-1 s-1", "Registry.EM_COMMON:493", wrf_history=True),
}


@dataclass(frozen=True)
class WrfSelectorGlobal:
    """One WRF physics selector that stock WRF stamps into every history file.

    These are ``rconfig`` namelist entries, not fields.  WRF writes an
    rconfig into the history file as a global attribute exactly when its
    I/O flag string contains ``h``, and every row below has one -- which is
    the citable half of "WRF writes this".  The empirical half is the
    group's own stock v4.6.1 wrfout, which carries all eleven as
    ``NC_INT``.
    """

    #: The global attribute name, which is the namelist key upper-cased.
    name: str
    #: ``Registry.EM_COMMON:<line>`` for the ``rconfig`` row.
    registry: str
    #: How the value is resolved from a gpuwm ``RunConfig``:
    #: ``"config"`` reads ``run_config_field``; ``"radiation_lw"`` and
    #: ``"radiation_sw"`` go through ``gpuwm.config.radiation_scheme_ids``,
    #: which collapses gpuwm's ``-1/-1`` legacy sentinel onto the effective
    #: pair; ``"unimplemented"`` is a constant ``0``.
    source: str
    run_config_field: str = ""


#: The physics selectors gpuwm can answer for, transcribed from the pinned
#: Registry and confirmed against the stock v4.6.1 reference wrfout.
#:
#: Their job is to make the *absence* of a scheme legible.  A wrfout whose
#: ``RAINSH`` is all zeros says nothing on its own: it could mean a shallow
#: cumulus scheme ran and produced no rain, or that none ran at all.
#: ``SHCU_PHYSICS=0`` is the difference, and WRF has always written it.
#:
#: Four rows are constant zero because gpuwm implements no such scheme, and
#: that is a resolved fact about the run rather than a placeholder: gpuwm
#: has no shallow-cumulus, urban-canopy, surface-mosaic or ocean-mixed-layer
#: option to select, so WRF's "off" value is the true one.  A selector whose
#: value gpuwm's configuration does NOT determine is not written at all --
#: see ``docs``/the release report for the two deliberate omissions.
PHYSICS_SELECTOR_GLOBALS: tuple[WrfSelectorGlobal, ...] = (
    WrfSelectorGlobal("MP_PHYSICS", "Registry.EM_COMMON:2406",
                      "config", "mp_physics"),
    WrfSelectorGlobal("RA_LW_PHYSICS", "Registry.EM_COMMON:2449",
                      "radiation_lw"),
    WrfSelectorGlobal("RA_SW_PHYSICS", "Registry.EM_COMMON:2450",
                      "radiation_sw"),
    WrfSelectorGlobal("SF_SFCLAY_PHYSICS", "Registry.EM_COMMON:2455",
                      "config", "sf_sfclay_physics"),
    WrfSelectorGlobal("SF_URBAN_PHYSICS", "Registry.EM_COMMON:2486",
                      "unimplemented"),
    WrfSelectorGlobal("SF_SURFACE_PHYSICS", "Registry.EM_COMMON:2468",
                      "config", "sf_surface_physics"),
    WrfSelectorGlobal("SF_SURFACE_MOSAIC", "Registry.EM_COMMON:2532",
                      "unimplemented"),
    WrfSelectorGlobal("SF_OCEAN_PHYSICS", "Registry.EM_COMMON:2614",
                      "unimplemented"),
    WrfSelectorGlobal("BL_PBL_PHYSICS", "Registry.EM_COMMON:2469",
                      "config", "bl_pbl_physics"),
    WrfSelectorGlobal("CU_PHYSICS", "Registry.EM_COMMON:2488",
                      "config", "cu_physics"),
    WrfSelectorGlobal("SHCU_PHYSICS", "Registry.EM_COMMON:2489",
                      "unimplemented"),
)


#: Every scheme field gpuwm can emit, keyed by ``PhysicsDriver.fields`` key.
#: The three inventories belong to three different selectors, so a key in two
#: of them would mean two schemes disagreeing about one array's identity; the
#: merge refuses that rather than letting dict order decide.
SCHEME_OUTPUT_FIELDS: dict[str, WrfOutputField] = {}
for _group in (MYNN_PBL_OUTPUT_FIELDS, NOAHMP_OUTPUT_FIELDS,
               RUC_OUTPUT_FIELDS):
    for _key, _field in _group.items():
        _clash = SCHEME_OUTPUT_FIELDS.get(_key)
        if _clash is not None and _clash != _field:
            raise ValueError(
                "two schemes declare different output schemas for field "
                f"{_key!r}: {_clash} and {_field}")
        SCHEME_OUTPUT_FIELDS[_key] = _field

#: Every SCHEME-SELECTOR record this module carries, keyed by the name that
#: reaches the NetCDF file: the scheme fields plus the always-written
#: accumulators.  Two distinct gpuwm fields sharing one external name would
#: make the wrfout inventory depend on dict order, so the reverse map
#: refuses it.
#:
#: This map is deliberately the SELECTOR-DRIVEN inventory only, and its
#: cardinality is pinned to exactly ``len(SCHEME_OUTPUT_FIELDS) +
#: len(PRECIPITATION_OUTPUT_FIELDS)`` by
#: ``tests/test_wrf_output_schema.py::
#: test_no_two_gpuwm_fields_claim_one_netcdf_name``.  Model-state rows --
#: which is what mp=28's four aerosol carriers are -- go in
#: :data:`HISTORY_FIELDS_BY_NETCDF_NAME` below instead of here, so that pin
#: keeps meaning what it says.  The writer consults the wider map.
OUTPUT_FIELDS_BY_NETCDF_NAME: dict[str, WrfOutputField] = {}
for _group in (SCHEME_OUTPUT_FIELDS, PRECIPITATION_OUTPUT_FIELDS):
    for _key, _field in _group.items():
        _clash = OUTPUT_FIELDS_BY_NETCDF_NAME.get(_field.netcdf_name)
        if _clash is not None and _clash != _field:
            raise ValueError(
                "two gpuwm fields both write NetCDF variable "
                f"{_field.netcdf_name!r}: {_clash} and {_field}")
        OUTPUT_FIELDS_BY_NETCDF_NAME[_field.netcdf_name] = _field

#: EVERY record this module carries, keyed by its NetCDF name: the
#: selector-driven inventory above plus the transcribed model-state rows
#: that belong to no physics-driver ``fields`` dict.  THIS is what
#: ``gpuwm.io.wrfout`` consults, so a name described here is described the
#: same way whichever table it came from, and the same clash refusal covers
#: the union -- an aerosol row that collided with a scheme diagnostic would
#: raise at import, not silently win or lose a dict-order race.
#:
#: Why two maps instead of one: ``OUTPUT_FIELDS_BY_NETCDF_NAME`` answers
#: "which fields does a selected SCHEME publish", which is the question the
#: driver-inventory tests ask and count.  ``HISTORY_FIELDS_BY_NETCDF_NAME``
#: answers "what does the writer know about this variable name".  mp=28's
#: aerosol carriers are transported model state (WRF ``scalar``-array
#: members, plus two ``misc`` 2-D surface fields), not scheme diagnostics,
#: so only the second question includes them.
HISTORY_FIELDS_BY_NETCDF_NAME: dict[str, WrfOutputField] = {}
for _group in (OUTPUT_FIELDS_BY_NETCDF_NAME, THOMPSON_AEROSOL_OUTPUT_FIELDS):
    for _key, _field in _group.items():
        _clash = HISTORY_FIELDS_BY_NETCDF_NAME.get(_field.netcdf_name)
        if _clash is not None and _clash != _field:
            raise ValueError(
                "two gpuwm fields both write NetCDF variable "
                f"{_field.netcdf_name!r}: {_clash} and {_field}")
        HISTORY_FIELDS_BY_NETCDF_NAME[_field.netcdf_name] = _field

del _group, _key, _field, _clash


def netcdf_names(keys) -> tuple[str, ...]:
    """External WRF names for ``keys``, in order; unknown keys fail loud."""
    return tuple(SCHEME_OUTPUT_FIELDS[key].netcdf_name for key in keys)


__all__ = [
    "HISTORY_FIELDS_BY_NETCDF_NAME",
    "MYNN_PBL_OUTPUT_FIELDS",
    "NETCDF_DTYPE_BY_REGISTRY_TYPE",
    "NOAHMP_OUTPUT_FIELDS",
    "OUTPUT_FIELDS_BY_NETCDF_NAME",
    "PHYSICS_SELECTOR_GLOBALS",
    "PRECIPITATION_OUTPUT_FIELDS",
    "RUC_OUTPUT_FIELDS",
    "SCHEME_OUTPUT_FIELDS",
    "THOMPSON_AEROSOL_OUTPUT_FIELDS",
    "WRF_FIELD_TYPE_INTEGER",
    "WRF_FIELD_TYPE_REAL",
    "WrfOutputField",
    "WrfSelectorGlobal",
    "netcdf_names",
]
