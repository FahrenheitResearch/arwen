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
``wrfinput`` variables.  The converse half of that rule first bites at
mp_physics=28: ``qnwfa2d``/``qnifa2d`` are ``state:`` members whose flags DO
carry ``i0``, so they are inventoried as ``wrfinput`` fields (with 2-D
dimensions) rather than as runtime state.  See :func:`_aerosol_emission`.

Units are the value WRF's Registry parser RESOLVES, not the text of the
Registry line; see :data:`WRF_RESOLVED_UNITS_NUMBER_PAREN`, which also
records the one place this module is knowingly inconsistent.
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
#: Registry dimension spec ``ij``.  Needed by mp_physics=28, the first
#: inventoried package with a 2-D member that carries an input-stream flag.
WRFINPUT_2D_DIMS = (
    "Time",
    "south_north",
    "west_east",
)

#: What WRF actually stores as a field's ``units``, which is NOT the text of
#: the Registry line.  ``tools/reg_parse.c:203-208`` walks every raw Registry
#: line before tokenizing and, for a ``#`` INSIDE double quotes, replaces it
#: with a blank (``:206``, ``else if ( *p == '#' && inquote ) *p = ' ' ;``);
#: a ``#`` outside quotes starts a comment (``:207``).  So the Registry text
#: ``"# kg(-1)"`` becomes ``'  kg(-1)'`` -- two leading blanks, one from the
#: blanked ``#`` and one that was already there -- and that string is what
#: WRF's generated tables carry and what its own writer emits as the NetCDF
#: ``units`` attribute.  Inside the generated ``mp_physics(idomain)==28``
#: block (``inc/scalar_indices.inc:2449-2618``) the six scalar members are
#: spelled ``'  kg-1'`` for ``P_qni`` (:2544) and ``'  kg(-1)'`` for
#: ``P_qnr`` (:2558), ``P_qnc`` (:2572), ``P_qnwfa`` (:2586), ``P_qnifa``
#: (:2600) and ``P_qnbca`` (:2614); the table is consumed as "! Units" at
#: ``inc/wrf_bdyout.inc:1449``.  gpuwm's own WRF-output authority already
#: agrees: ``gpuwm/io/wrfout.py::_VAR_META`` was verified against a stock
#: v4.6.1 wrfout and spells QNCLOUD/QNRAIN ``"  kg(-1)"`` and QNICE
#: ``"  kg-1"``.
#:
#: KNOWN DIVERGENCE, stated rather than hidden: the mp=6/8/10/18 rows below
#: predate this finding and carry the pre-parse Registry spelling
#: (``"# kg(-1)"``).  They are left byte-identical here on purpose -- they
#: are consumed by a shipped compatibility report and are not this change's
#: subject -- so the same NetCDF variable (QNRAIN) is spelled two ways in
#: this module depending on the package.  Only the mp=28 row is
#: authoritative on units.
WRF_RESOLVED_UNITS_NUMBER_PAREN = "  kg(-1)"
WRF_RESOLVED_UNITS_NUMBER_PLAIN = "  kg-1"


#: Registry members stock WRF allocates for a package that gpuwm has no
#: species for.  They are still WRITTEN -- real.exe writes them too, at
#: zero, and a stock file missing a registered member is not a stock file --
#: but there is no prepared state to look for, so the exporter must not go
#: looking and must not report the absence as a gap.
_NO_GPUWM_SPECIES = frozenset({"qnbca"})


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
    #: The gpuwm state attribute holding this member's prepared values,
    #: when it differs from WRF's Registry name, and ``None`` when gpuwm
    #: has no such species at all (the member is still written, at zero,
    #: because stock WRF registers it for the package).
    #:
    #: It exists because the exporter used to read
    #: ``state/{registry_name}`` and gpuwm drops WRF's leading ``q`` on
    #: exactly the aerosol-aware rows: qnc/qnwfa/qnifa/qnwfa2d/qnifa2d
    #: are nc/nwfa/nifa/nwfa2d/nifa2d in the prepared cache, so every one
    #: of them missed and exported zeros even on a run whose WIF
    #: climatology lane had filled them.  A transform ("strip a q") would
    #: be wrong for qni/qnr/qns/qng/qnh/qndrop/qnn/qvolg/qvolh/qir/qib,
    #: which DO exist under their Registry names, so the exception is
    #: declared per row rather than computed.
    state_name: str | None = None

    @property
    def state_key(self) -> str | None:
        """Prepared-cache key for this member, or ``None`` when gpuwm has none."""

        if self.state_name is None and self.registry_name in _NO_GPUWM_SPECIES:
            return None
        return self.state_name or self.registry_name


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
        state_name: str | None = None,
) -> WrfInputField:
    return WrfInputField(
        registry_name=name,
        netcdf_name=netcdf_name,
        collection="scalar",
        units=units,
        state_name=state_name,
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


def _aerosol_emission(name: str, netcdf_name: str, *,
                      state_name: str | None = None) -> WrfInputField:
    """One of mp=28's two 2-D surface aerosol emission members.

    These are ``state:`` members of the ``thompsonaero`` package rather than
    ``scalar:`` members, and the module's own rule ("Package ``state:``
    auxiliaries whose Registry flags contain ``r`` but not ``i`` are
    runtime/restart state") therefore has to be applied and not assumed.
    ``Registry/Registry.EM_COMMON:492-493`` gives them the I/O string
    ``i01{17}rhdu`` -- an ``i`` list that BEGINS with stream 0, the same
    ``i0`` prefix carried by unambiguous wrfinput members such as HGT
    (:1407 ``i012rh056dus``) and TSK (:1417).  So unlike ``re_cloud`` /
    ``re_ice`` / ``re_snow`` (:497-499, bare ``r``) and ``taod5503d`` /
    ``taod5502d`` (:1738-1739, ``r`` and ``rh``), these two ARE
    initialization-file variables and belong in ``wrfinput_fields``.

    ``real.exe`` writes them: ``dyn_em/module_initialize_real.F:4496-4653``
    is the 2-D counterpart of the 3-D aerosol block, and its
    ``aer_init_opt = 0`` arm sets both to exactly ``0.0`` (:4501-4510) with
    the comment "Surface emissions of QNWFA will be computed in
    microphysics" -- which is ``thompson_init`` deriving ``nwfa2d`` from the
    synthetic CCN profile at ``phys/module_mp_thompson.F:509-510``.
    ``nifa2d`` has no such derivation anywhere in that file and stays zero.
    """
    return WrfInputField(
        registry_name=name,
        netcdf_name=netcdf_name,
        collection="state",
        dimensions=WRFINPUT_2D_DIMS,
        units="kg-1 s-1",
        state_name=state_name,
    )


#: P3's moist list, and it is a SUBSET of :data:`_ICE_MASS` rather than an
#: extension of it.  ``Registry.EM_COMMON:3038`` declares
#: ``moist:qv,qc,qr,qi`` for ``p3_1category`` -- there is no ``qs`` and no
#: ``qg``, because P3 carries ONE ice category whose rime mass and rime
#: volume (``qir``/``qib``) span the graupel-to-snow continuum instead of
#: splitting it into species.  Every other inventoried package so far has
#: been WSM6's six plus additions, so this is the first row where the
#: package declares FEWER moist members than the frozen WSM6 contract
#: carries; see ``gpuwm/wrf_direct.py::_physics_contract_bundle``, which
#: prunes what the package does not declare.
_P3_MASS = (
    _moist("qv", "QVAPOR"),
    _moist("qc", "QCLOUD"),
    _moist("qr", "QRAIN"),
    _moist("qi", "QICE"),
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
    # Registry.EM_COMMON:3036.  Aerosol-aware Thompson.  Its moist list is
    # character for character mp=8's (:3024); everything new is a scalar or
    # state member.
    #
    # SIX scalar members, not five.  ``qnbca`` is declared by the
    # ``thompsonaero`` package itself and is therefore registered for EVERY
    # mp_physics=28 run, not only for wif_input_opt=2 -- WRF's generated
    # code proves it directly: the ``mp_physics(idomain)==28`` block at
    # inc/scalar_indices.inc:2449-2618 ends with
    # ``scalar_dname_table( idomain, P_qnbca ) = 'QNBCA'`` (:2612) and
    # ``F_qnbca = .TRUE.`` (:2617), with the wif_input_opt==2 block at
    # :18973-18988 registering it a SECOND, idempotent time.  It is listed
    # here because this module answers what an unchanged WRF v4.6.1
    # executable expects in wrfinput, not what gpuwm implements; gpuwm has
    # no nbca species anywhere (see gpuwm.config.MP28_AEROSOL_SOURCE_OPTIONS
    # for wif_input_opt), and that is a gpuwm scope statement, not a claim
    # about WRF's package.
    #
    # THAT GAP IS CLOSED (audit R-054).  The block that stood here recorded
    # a downstream defect and filed a request instead of fixing it: for
    # want of six rows in gpuwm/wrf_direct.py's ``_PACKAGE_FIELD_METADATA``
    # a stock export of an mp=28 domain wrote 8 of these 14 variables and
    # dropped 6 with no error and no receipt line, and the two 2-D members
    # would have been written with the 3-D QCLOUD prototype's rank if the
    # rows had simply been added.  wrf_direct.py now carries all six rows,
    # a 2-D prototype branch, and the per-row ``state_name`` above; the
    # agreement check at its import holds this inventory and that table
    # equal, so a future package member cannot be dropped silently either.
    28: StockWrfPhysicsInventory(
        mp_physics=28,
        scheme="Thompson aerosol-aware",
        registry_package="thompsonaero",
        wrfinput_fields=_ICE_MASS + (
            # Units are WRF's post-reg_parse values; see
            # WRF_RESOLVED_UNITS_NUMBER_PAREN above for why they carry a
            # blank where the Registry line shows '#'.
            _scalar("qni", "QNICE", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN),
            _scalar("qnc", "QNCLOUD", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nc"),
            _scalar("qnwfa", "QNWFA", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nwfa"),
            _scalar("qnifa", "QNIFA", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nifa"),
            # No gpuwm species: written at zero, which IS stock behaviour
            # (real.exe's aer_init_opt=0 arm writes exact 0.0 for it).
            _scalar("qnbca", "QNBCA", units=WRF_RESOLVED_UNITS_NUMBER_PAREN),
            _aerosol_emission("qnwfa2d", "QNWFA2D", state_name="nwfa2d"),
            _aerosol_emission("qnifa2d", "QNIFA2D", state_name="nifa2d"),
        ),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII + (
            # Registry.EM_COMMON:1738-1739.  taod5503d is bare ``r`` and
            # taod5502d is ``rh``: restart and history, never input.  They
            # are the 550 nm aerosol optical depth diagnostics WRF's
            # aerosol-aware Thompson publishes for radiation; gpuwm does
            # not compute them, which is a gpuwm gap and not a reason to
            # misreport WRF's package.
            RuntimeStateField("taod5503d", "TAOD5503D"),
            RuntimeStateField(
                "taod5502d", "TAOD5502D", dimensions=WRFINPUT_2D_DIMS),
        ),
    ),
    # Registry.EM_COMMON:3038.  P3 one-category, two-moment ice:
    #
    #   package p3_1category mp_physics==50 -
    #     moist:qv,qc,qr,qi;
    #     scalar:qni,qnr,qir,qib;
    #     state:re_cloud,re_ice,vmi3d,rhopo3d,di3d,refl_10cm,th_old,qv_old
    #
    # THE STATE HALF IS ENTIRELY RUNTIME, and that is measured from the I/O
    # flags rather than assumed, because this module's own rule ("Package
    # ``state:`` auxiliaries whose Registry flags contain ``r`` but not
    # ``i`` are runtime/restart state") is what mp=28's qnwfa2d/qnifa2d
    # already proved has to be applied case by case.  Not one of P3's eight
    # state members carries an ``i``: re_cloud (:497) and re_ice (:498) are
    # bare ``r``; vmi3d (:1600), di3d (:1601), rhopo3d (:1602) and
    # refl_10cm (:1596) are ``hdu``; th_old (:1598) and qv_old (:1599) are
    # ``rusd``.  So mp=50 adds NO state member to wrfinput, unlike mp=28.
    #
    # re_snow is deliberately absent from the runtime list.  P3's package
    # does not declare it, which is the same fact WRF acts on when
    # module_physics_init.F:1027-1033 sets has_reqs=0 for P3 -- and that is
    # in turn why the RTE+RRTMGP coupling for mp=50 (rrtmgp's ``50: "p3"``
    # row) remaps the single ice category onto the snow species at P3's own
    # ice radius instead of consuming a snow radius.  Listing RE_SNOW here
    # would invent the field that remap exists to do without.
    #
    # Units are WRF's post-reg_parse RESOLVED values, the mp=28 convention
    # rather than the pre-parse spelling the 6/8/10/18 rows carry: qni is
    # Registry ``"# kg-1"`` (:523-524) and qnr ``"# kg(-1)"`` (:533-534).
    # qir (:555-556) and qib (:557-558) carry NO ``#`` at all -- their
    # Registry text is ``"kg kg(-1)"`` and ``"m(3) kg(-1)"`` -- so for those
    # two the resolved value and the Registry text are the same string and
    # there is nothing to resolve.
    #
    # mp=51 (p3_1category_nc, :3039) is the same port with qnc added and is
    # NOT inventoried here: gpuwm.config accepts 50 only.  52 and 53 are
    # refused by name (gpuwm/config.py:1134-1172).
    50: StockWrfPhysicsInventory(
        mp_physics=50,
        scheme="P3 one-category two-moment ice",
        registry_package="p3_1category",
        wrfinput_fields=_P3_MASS + (
            _scalar("qni", "QNICE", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN),
            _scalar("qir", "QIR", units="kg kg(-1)"),
            _scalar("qib", "QIB", units="m(3) kg(-1)"),
        ),
        runtime_state_not_wrfinput=(
            RuntimeStateField("re_cloud", "RE_CLOUD"),
            RuntimeStateField("re_ice", "RE_ICE"),
            RuntimeStateField("vmi3d", "v_ice"),
            RuntimeStateField("di3d", "d_ice"),
            RuntimeStateField("rhopo3d", "rho_ice"),
            RuntimeStateField("refl_10cm", "refl_10cm"),
            RuntimeStateField("th_old", "TH_OLD"),
            RuntimeStateField("qv_old", "QV_OLD"),
        ),
    ),
}


def supported_stock_wrf_mp_physics() -> tuple[int, ...]:
    """Return the declaratively inventoried stock-WRF microphysics ids."""

    return tuple(sorted(_INVENTORIES))


def _require_agreement_with_the_registry() -> None:
    """The registry's ``consumers.stock_wrf_export`` rows ARE this table.

    Generated from ``_INVENTORIES`` by tools/build_registry.py and held to
    it here, so a package row added without a rebuilt registry -- or a
    registry claiming an inventory this module does not carry -- fails this
    import.  The four uninventoried schemes are an export-only scope
    decision the row itself states (``inventoried: false`` with its
    reason); they are not absences to cite, because the registry publishes
    a row for them too.
    """

    from gpuwm.physics_registry import require_consumer_rows_agreement

    require_consumer_rows_agreement(
        "gpuwm.wrf_physics_inventory._INVENTORIES",
        "microphysics", "stock_wrf_export",
        {mp: {"inventoried": True,
              "netcdf_names": [field.netcdf_name
                               for field in inventory.wrfinput_fields]}
         for mp, inventory in _INVENTORIES.items()},
        project=lambda row: (
            {"inventoried": True,
             "netcdf_names": [field["netcdf_name"]
                              for field in row["wrfinput_fields"]]}
            if row.get("inventoried") is True else None),
        cited_absences={
            0: "export-only scope: no evidenced Registry.EM_COMMON package contract packaged (audit R-014)",
            1: "export-only scope: no evidenced Registry.EM_COMMON package contract packaged (audit R-014)",
            9: "export-only scope: no evidenced Registry.EM_COMMON package contract packaged (audit R-014)",
            16: "export-only scope: no evidenced Registry.EM_COMMON package contract packaged (audit R-014)",
        })


_require_agreement_with_the_registry()


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
            f"stock-WRF wrfinput export for mp_physics={mp_physics} is not "
            "available on this route: it writes initialization files for an "
            "UNCHANGED WRF v4.6.1 executable, and no evidenced "
            "Registry.EM_COMMON package contract (member list plus "
            "real.exe initialized-state policy) is packaged for that "
            "selector, so the exported wrfinput would under-declare its own "
            "package's hydrometeors and moments. This says NOTHING about "
            "running the scheme in ArWen -- ArWen's own forecast route runs "
            "every scheme the physics registry publishes as implemented, "
            "and the runtime verdict is answered separately. Evidenced "
            f"package contracts: {supported}."
        ) from None


__all__ = [
    "SCHEMA",
    "StockWrfPhysicsInventory",
    "RuntimeStateField",
    "WRFINPUT_2D_DIMS",
    "WRFINPUT_3D_DIMS",
    "WRF_RESOLVED_UNITS_NUMBER_PAREN",
    "WRF_RESOLVED_UNITS_NUMBER_PLAIN",
    "WrfInputField",
    "stock_wrf_physics_inventory",
    "supported_stock_wrf_mp_physics",
]
