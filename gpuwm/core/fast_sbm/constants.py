"""Scalar constants, the scalar layout, the refusal type, behaviour presets.

Transcribed from WRF v4.8.0 ``phys/module_mp_fast_sbm.F`` (LF SHA-256
``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``,
identical to v4.6.1).  WRF is NCAR/UCAR public domain; the notice is in the
repository ``NOTICE`` and in ``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.  The
scheme is FSBM-2 by Jacob Shpund and co-authors (see the package docstring).

Lines transcribed: :3872-3873 (bin indices in ``chem_new``), :3880-3886 and
:3891 (the process switches, ``hail_opt``, ``ICEPROCS``), :3894-3910 (bin
counts, split bins, ``COL``, ``ALCR``, ``kr_icempl``), :3991-3995 (aerosol
composition), :4088-4099 (``FAST_SBM`` DATA statements, the freezing and
coalescence constants among them), :4204-4205 (``NCOND``, ``NCOLL``), :6830
(``ONECOND1`` DATA for ``AL1``, ``AL2``), the ``module_mp_SBM_Nucleation``
parameters (:3024-3033), the ``module_mp_SBM_BreakUp`` snow bins (:110-111)
and the four fusion-heat literals (:3108, :6627, :6741, :8590).

Every literal is written exactly as WRF writes it.  The host reference is
float64, so a WRF ``REAL`` literal such as ``2.53E12`` enters here as the
double nearest the decimal, which is what the promoted oracle build
(``-freal-4-real-8``) holds.  WRF's native build holds the float32 nearest
the same decimal; that difference belongs to the native-envelope gate, not
to these values.

The behaviour presets (``FsbmBehavior``) name the WRF findings D1, D2, D3,
D5, D7, D8, D9 and D10 of the fast-sbm design (section 3.7), and the
findings of the cold half, D11 to D15, D17 and S1 and S2, which milestone
2's plan and contract catalogue (``catalogue.FINDINGS`` holds every row).
``wrf_v480()`` reproduces WRF as written, for the oracle gates.
``corrected()`` is the reference default.  They are keyword arguments of
the process functions, never a silent change of what a caller gets.

The dense-ice class is a named option (``DenseIce``, ``dense_ice``), hail
or graupel, each with its own identity, never WRF's bare ``hail_opt``
integer passed as a silent change of a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Bin counts and split bins (:3894-3902, :3910)
# ---------------------------------------------------------------------------

#: Number of mass-doubling bins for each hydrometeor class (:3902, the
#: 33-bin line; the 43-bin line at :3901 is commented out).
NKR = 33
#: Number of aerosol dry-radius bins (:3902).
NKR_AEROSOL = 43
#: Number of ice-crystal habits (columns, plates, dendrites) (:3894).
ICEMAX = 3
#: Collision grid size (:3902).
JMAX = 33
#: Number of the largest drop bins the collisional breakup acts on (:3902).
JBREAK = 18
#: Collisional breakup trigger, a strict bound (:3902).  COAL_BOTT_NEW
#: switches collisional breakup on only when a drop bin above this one holds
#: mass: ``if(kr .gt. KRMIN_BREAKUP .and. g1(kr) > g_lim) icol_drop_brk = 1``
#: (:8335), so on the 33-bin grid only bins 32 and 33 trigger it.  The
#: comparison is ``>``, not ``>=``.  The breakup itself then acts on the top
#: ``JBREAK`` bins, 16 to 33 (``coll_breakup_KS``, :1546, :1574).
KRMIN_BREAKUP = 31
#: First rain bin, radius 50.8 um; QC sums bins 1-14 (:3900).
KRDROP = 15
#: Last snow bin reported as cloud ice QI (:3900).
KRICE = 18
#: Smallest drop bin counted in the Hallett-Mossop drop depletion (:3910).
#: COAL_BOTT_NEW sums the drop number of bins ``KR_ICEMPL`` to ``NKR`` just
#: before graupel or hail collects drops (:8490, :8506, :8529) and again
#: after (:8542); inside the splinter window that loss sets the splinter
#: number, and the splinters are added to snow bin 1 (:8553).  It is not
#: the bin the splinters go to.
KR_ICEMPL = 9
#: Freezing threshold bin (``FAST_SBM`` DATA, :4090-4092).
KRFREEZ = 21

# ---------------------------------------------------------------------------
# Grid geometry and process counts
# ---------------------------------------------------------------------------

#: Logarithmic radius spacing of the mass-doubling grid, ln 2 / 3 rounded
#: to the literal WRF writes (:3905).
COL = 0.23105
#: Condensation substeps per call (:4204).
NCOND = 3
#: Collision substeps per call (:4205).
NCOLL = 1

# ---------------------------------------------------------------------------
# Saturation vapour pressure fit and latent heats
# ---------------------------------------------------------------------------

#: Water saturation fit ``AA1_MY exp(-BB1_MY / T)`` (dyn/cm2, K) and the ice
#: fit ``AA2_MY exp(-BB2_MY / T)`` (``FAST_SBM`` DATA, :4088).
AA1_MY = 2.53e12
BB1_MY = 5.42e3
AA2_MY = 3.41e13
BB2_MY = 6.13e3
#: Latent heat of condensation divided by the specific heat of air, K per
#: unit mixing ratio, as WRF writes it (``ONECOND1`` DATA, :6830, used as
#: ``TPN = TPS + DAL1*DELMASSL1`` at :7059; also ``WATER_NUCLEATION`` :3220).
AL1 = 2500.0
#: Latent heat of sublimation divided by the specific heat of air, K per
#: unit mixing ratio (``ONECOND1`` DATA, :6830; also ``JERNUCL01_KS`` :3062).
AL2 = 2834.0

# ---------------------------------------------------------------------------
# Transitions and thresholds
# ---------------------------------------------------------------------------

#: Liquid water content above which rimed snow is transferred to hail or
#: graupel, g/m3 (:3907; the header comment at :56-60 describes it).
ALCR = 0.5
#: Graupel to hail transition threshold, g/m3; 100 disables the transition
#: in this version (:3908).
ALCR_G = 100.0
#: Upper temperature bound of the Hallett-Mossop splinter window, K
#: (``FAST_SBM`` DATA, :4099).  Its only use is inside COAL_BOTT_NEW:
#: splinters form when ``tt .ge. 265.15 .and. tt .le. tcrit`` (:8540), and
#: above 268.15 K their number falls linearly to zero at TCRIT (:8550).  It
#: is not a freezing threshold: immediate freezing (``FREEZ``) is keyed on
#: the literal 273.15 (:4794).
TCRIT = 270.15
#: Cold bound of the ice collision-kernel temperature factor, K
#: (``FAST_SBM`` DATA, :4097).  FAST_SBM passes it through COAL_BOTT_NEW
#: (:4776) to ``MODKRN_KS`` (:8320), where only ``CASE(1)`` reads it, as
#: ``tc_min = ttcoal - 273.15`` (:1455), the cold end of the linear factor
#: below -20 C.  The live call at :8320 selects ``CASE(11)``, which never
#: reads it, so in WRF v4.8.0 TTCOAL changes no result.  It is not the
#: collision guard: collisions run when ``TT >= 233.15`` (:4769), a literal.
TTCOAL = 233.15

# ---------------------------------------------------------------------------
# Aerosol composition, ammonium sulfate (:3991-3995)
# ---------------------------------------------------------------------------

#: Number of ions per solute molecule (:3993).
IONS = 3
#: Molecular weight of the solute, g/mol (:3991).
MWAERO = 115.0
#: Solute density, g/cm3 (:3995).
RO_SOLUTE = 1.79

# ---------------------------------------------------------------------------
# Compile-time switches (:3886, :3891)
# ---------------------------------------------------------------------------

#: 1: the dense-ice class is hail (WRF's default); 0: graupel (:3886).
HAIL_OPT_DEFAULT = 1
#: 1: ice processes active (WRF's default); 0: liquid only (:3891).
ICEPROCS_DEFAULT = 1

# ---------------------------------------------------------------------------
# The cold half: process switches (:3880-3894) and module_mp_SBM_BreakUp's
# snow bins (:110-111)
# ---------------------------------------------------------------------------

#: Collisional breakup of drops on (:3880); ``COAL_BOTT_NEW`` clears its
#: breakup flag when this is not 1 (:8336).
IBREAKUP = 1
#: Snow breakup switch (:3881), the first term of the guard at :4836.
SNOW_BREAKUP_ON = 1
#: Spontaneous rain breakup switch (:3882), the gate at :4826.
SPONT_RAIN_BREAKUP_ON = 1
#: Jiwen Fan melting, the only melting ``FAST_SBM`` does (:3884, gate at
#: :4804).
JIWEN_FAN_MELT = 1
#: Hallett-Mossop switch (:3894; ``COAL_BOTT_NEW`` declares its own
#: ``icempl = 1`` at :8302 and tests that one at :8539).
ICEMPL = 1
#: First snow bin of ``BreakUp_Snow`` (``module_mp_SBM_BreakUp`` :111).
#: The guard at :4836 sums ``FF3R(KR_SNOW_MIN:NKR)``, an empty section on
#: the 33-bin grid.
KR_SNOW_MIN = 34
#: Last snow bin of ``BreakUp_Snow`` (``module_mp_SBM_BreakUp`` :110).  The
#: call at :4841 runs only ``IF (KR_SNOW_MAX <= NKR)``, so it never runs
#: with 33 bins.
KR_SNOW_MAX = 35

# ---------------------------------------------------------------------------
# Freezing (``FREEZ``, Bigg 1953): the FAST_SBM DATA statement (:4090-4099)
# ---------------------------------------------------------------------------
# DATA names at :4090-4092, values in the same order at :4092-4099.  KRFREEZ
# (21, :4092), TTCOAL (:4097) and TCRIT (:4099) are above.

#: Bigg's freezing coefficient ``AF`` (:4097, ``0.3333E-04``), used as
#: ``PF_1 = AF*EXP(-BF*DEL_T)`` (:6608).
AFREEZMY = 0.3333e-4
#: Bigg's temperature coefficient for the smallest drop, ``BF`` at bin mass
#: zero (:4097, ``0.6600E00``): ``BF = BFREEZMY + CFREEZ*XL(KR)`` (:6607).
BFREEZMY = 0.66
#: The same coefficient at the largest drop bin (:4093, ``0.6600E00``):
#: ``CFREEZ = (BFREEZMAX-BFREEZMY)/XL(NKR)`` (:6601), zero here because the
#: two are equal, so every drop bin freezes with the same ``BF``.
BFREEZMAX = 0.66

# ---------------------------------------------------------------------------
# Nucleation: module_mp_SBM_Nucleation parameters (:3024-3033)
# ---------------------------------------------------------------------------

#: Cloud-base activation branch on (:3024).
USE_CLOUD_BASE_NUC = 1
#: Drop activation runs only above this temperature, degrees C (:3025,
#: tested at :3094).
T_NUCL_DROP_MIN = -80.0
#: Deposition and condensation-freezing nucleation runs only at or above
#: this temperature, degrees C (:3026, tested at :3117).
T_NUCL_ICE_MIN = -37.0
#: 0: Meyers et al. (1992) ice nucleation (:3030).  The DeMott branch the
#: comment at :3029 names has no code in this file.
ICE_NUCL_METHOD = 0
#: 1: the ice branch of ``JERNUCL01_KS`` takes vapour and releases heat for
#: the nucleated crystals (:3031, :3129-3179).
ISIGN_TQ_ICENUCL = 1
#: Clamp on the ice supersaturation ``ICE_NUCL`` works with, percent
#: (:3033, "DELSUPICE_MAX=59%" at :3032).
DELSUPICE_MAX = 59.0
#: Homogeneous freezing of every drop at or below this temperature,
#: degrees C (:3101), inside the water-supersaturated branch only (D14).
T_HOMOGENEOUS_FREEZING = -38.0

# ---------------------------------------------------------------------------
# Latent heat of fusion over the heat capacity of air (finding D13)
# ---------------------------------------------------------------------------

#: The four places the cold path converts frozen or melted mass into a
#: temperature change, and the literal WRF writes at each, K per unit mass
#: mixing ratio: ``FREEZ`` (:6627), ``J_W_MELT`` (:6741), riming in
#: ``COAL_BOTT_NEW`` (:8590) and homogeneous freezing in ``JERNUCL01_KS``
#: (:3108).  The scheme's own ``AL2 - AL1`` is 334.
FUSION_HEAT_WRF = {
    "freez": 333.0,
    "j_w_melt": 333.0,
    "riming": 320.0,
    "homogeneous": 334.0,
}
#: The WRF line of each literal in ``FUSION_HEAT_WRF``.
FUSION_HEAT_LINES = {"freez": 6627, "j_w_melt": 6741, "riming": 8590,
                     "homogeneous": 3108}

# ---------------------------------------------------------------------------
# The WRF scalar layout of ``chem_new`` (:3872-3873)
# ---------------------------------------------------------------------------
# Fortran one-based indices into the fourth dimension of ``chem_new``.
# Index 1 is WRF's unused first scalar slot, so the 142 bins occupy 2..143.

P_FF1I01 = 2
P_FF1I33 = 34
P_FF5I01 = 35
P_FF5I33 = 67
P_FF6I01 = 68
P_FF6I33 = 100
P_FF8I01 = 101
P_FF8I43 = 143
#: Number of bin scalars, ``P_FF8I43 - P_FF1I01 + 1``.
N_BIN_SCALARS = P_FF8I43 - P_FF1I01 + 1


class FsbmRefusal(RuntimeError):
    """A WRF fatal check, or an input the reference cannot honour.

    ``where`` names the WRF routine (and line) or the port function that
    refused; ``condition`` states what was found.  WRF's ``wrf_error_fatal``
    calls on the transcribed paths become this exception.
    """

    def __init__(self, where: str, condition: str) -> None:
        self.where = where
        self.condition = condition
        super().__init__(f"{where}: {condition}")


_RAINNC_VELOCITY = ("tile_last", "own_column")
_RAINNC_SOURCE = ("level1_after_fall", "bottom_outflow")

#: Cloud-base detection rules (finding D9): ``"wrf_v480"`` is the test at
#: :4387 as gfortran evaluates it, ``"first_level"`` the test the code's
#: comment describes.
CLOUD_BASE_RULES = ("wrf_v480", "first_level")

#: Fall-speed lookups inside the coalescence efficiencies (finding D10):
#: ``"wrf_v480"`` is ``vTBeard`` as WRF runs it, which hands every
#: bin-centre drop the next bin's fall speed; ``"own_bin"`` gives each drop
#: its own bin's speed.
FALL_SPEED_LOOKUPS = ("wrf_v480", "own_bin")

#: What the exit clip of the snow bins is keyed on (finding D11): ``"qs"``
#: is WRF, snow bins 19-33 only (:5291); ``"qi_plus_qs"`` all 33 snow bins.
SNOW_CLIP_KEYS = ("qs", "qi_plus_qs")

#: How ``J_W_MELT`` removes a melting bin over one step (finding D12):
#: ``"linear"`` is WRF, ``f - f rate dt`` (:6683-6734), which goes negative
#: once ``rate dt`` exceeds 1; ``"exponential"`` the exact solution of the
#: same constant-rate decay, ``f exp(-rate dt)``, the form ``FREEZ`` itself
#: uses (``YKK = EXP(-PF*DT)``, :6610).
MELT_STEPS = ("linear", "exponential")

#: The latent heat of fusion the cold path converts frozen, melted and
#: rimed mass with (finding D13): ``"wrf_v480"`` is WRF's four literals
#: (``FUSION_HEAT_WRF``); ``"al2_minus_al1"`` the scheme's own
#: ``AL2 - AL1`` at every site.
FUSION_HEATS = ("wrf_v480", "al2_minus_al1")

#: Where homogeneous freezing at -38 C runs (finding D14):
#: ``"water_supersaturated"`` is WRF, only inside ``JERNUCL01_KS``'s
#: ``SUP1 > 0`` branch (:3094-3111); ``"any_saturation"`` at every point
#: at or below -38 C.
HOMOGENEOUS_FREEZING_RULES = ("water_supersaturated", "any_saturation")

#: The vertical velocity in the ``DSUPICE_XYZ`` term (finding D15):
#: ``"wrf_v480"`` is WRF, ``w_stag`` set only where 238.15 <= T_old <
#: 274.15 K (:4405-4410), so a colder point reads the ``w_stag`` of the
#: most recent point, in the pre-pass's (j, i, k) order, with 238.15 <=
#: T_old < 274.15 (undefined if no such point came earlier in the call);
#: ``"own_point"`` each point's own staggered w.
DSUPICE_W_RULES = ("wrf_v480", "own_point")

#: The horizontal wind in the ``DSUPICE_XYZ`` term (finding D17):
#: ``"wrf_v480"`` is WRF, the four-point average of ``U`` and ``V`` over
#: (I..I+1, J..J+1) (:4497-4499), a wind at (I+1/2, J+1/2) because the
#: winds WRF passes are already at mass points, and the point's own wind
#: at I = IDE-1 or J = JDE-1 (:4500-4502); ``"own_point"`` each point's
#: own mass-point wind.
DSUPICE_WIND_RULES = ("wrf_v480", "own_point")

#: The small-ratio branches of ``JERSUPSAT_KS`` (findings S1 and S2):
#: ``"wrf_v480"`` is WRF, ``ALFA`` expanded in ``X/2`` (:2252-2256,
#: :2286-2290) and branch 7's ``GAMA`` clamped to 5e-14 (:2202);
#: ``"rederived"`` each form rederived from the closed-form solution, the
#: expansion variable ``X/PI**2`` and ``GAMA`` keeping its sign.
SUPSAT_SMALL_RATIO_RULES = ("wrf_v480", "rederived")

#: The depth ``FALFLUXHUCM_Z`` divides a level's flux divergence by
#: (finding D8): ``"level_spacing"`` is WRF, the distance between level
#: centres (:5364-5373); ``"layer_depth"`` the layer's own depth.
FALL_DEPTHS = ("level_spacing", "layer_depth")


class _NoValue:
    """The marker a cold field holds when a caller left it out."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<no value>"

    def __reduce__(self):
        return (_NoValue, ())


_NO_VALUE = _NoValue()

#: The fields milestone 2 adds for the cold half, with the finding each
#: selects.  None has a usable default (see ``FsbmBehavior``).
COLD_FIELDS = {
    "snow_clip_key": ("D11",),
    "melt_step": ("D12",),
    "fusion_heat": ("D13",),
    "homogeneous_freezing": ("D14",),
    "dsupice_w": ("D15",),
    "dsupice_wind": ("D17",),
    "supsat_small_ratio": ("S1", "S2"),
    "fall_depth": ("D8",),
}

_COLD_VALUES = {
    "snow_clip_key": SNOW_CLIP_KEYS,
    "melt_step": MELT_STEPS,
    "fusion_heat": FUSION_HEATS,
    "homogeneous_freezing": HOMOGENEOUS_FREEZING_RULES,
    "dsupice_w": DSUPICE_W_RULES,
    "dsupice_wind": DSUPICE_WIND_RULES,
    "supsat_small_ratio": SUPSAT_SMALL_RATIO_RULES,
    "fall_depth": FALL_DEPTHS,
}


@dataclass(frozen=True)
class FsbmBehavior:
    """Which WRF behaviour each finding of the design (3.7) follows.

    ``rainnc_velocity``
        ``"tile_last"`` (WRF, D1): surface precipitation uses the fall-speed
        pressure factor of the tile's last column, because ``VR1_Z`` has no
        i, j index (:4523, read at :5064).  ``"own_column"``: each column's
        own factor.
    ``rainnc_source``
        ``"level1_after_fall"`` (WRF, D2): RAINNC is fall speed times the
        lowest-level spectrum after sedimentation, times dt (:5058-5121).
        ``"bottom_outflow"``: the sedimentation outflow through the bottom
        face.
    ``nucleation_water_sink``
        ``False`` (WRF, D3): ``WATER_NUCLEATION`` discards the activated drop
        mass (:3313), so activation takes no vapour and releases no heat.
        ``True``: the sink and heating are applied as the ice branch does.
    ``dtcond_carry``
        ``True`` (WRF, D5): ``DTCOND`` is not reset per point and carries a
        last-place drift across points in loop order (:4620).  ``False``:
        reset per point.
    ``breakup_retry_cap``
        Upper bound on the collisional breakup retry doubling, which WRF
        leaves unbounded (D7, :8423).  64 in both presets; exceeding it is a
        ``FsbmRefusal``.
    ``cloud_base_rule``
        ``"wrf_v480"`` (WRF, D9): the cloud-base test at :4387 joins its
        terms with ``.and.`` around ``K_found .eqv. .FALSE.``; ``.EQV.``
        binds more loosely than ``.AND.``, so ``KZ_Cloud_Base`` ends as the
        highest level that is not a supersaturated updraft, or level 1.
        ``"first_level"``: the test the code's comment describes, the first
        level above the second that is supersaturated with w above 10 cm/s
        and lies below 3 km, and no level when none is.  It decides where
        the ``Cloud_Base_Super`` activation branch (Pinsky et al. 2012)
        runs.  The field has no default, so every behaviour, an A/B arm
        included, names its rule: an arm made with ``dataclasses.replace``
        from a preset keeps that preset's rule.
    ``fall_speed_lookup``
        ``"wrf_v480"`` (WRF, D10): ``ecoalmass`` forms a drop's diameter
        with pi written ``3.1415927`` (:8795) while ``DROPRADII``, the radii
        ``vTBeard`` searches, uses ``3.141593`` (:6100), so every bin-centre
        drop of bins 1-32 lands just above its own radius and gets the next
        bin's fall speed (:9041-9045) in every Beard-Ochs and Low-List
        efficiency of ``ECOALMASSM``.  ``"own_bin"``: each drop gets its own
        bin's speed.  ``derived.build_derived`` builds ``ECOALMASSM`` with
        the behaviour's lookup, and the warm call refuses derived tables
        built with the other one.  Like ``cloud_base_rule`` the field has no
        default.

    The cold half (milestone 2) adds eight fields, keyword-only.  Each is a
    required argument in effect: leaving one out raises ``TypeError``
    naming it, as leaving out ``cloud_base_rule`` does, so an A/B arm built
    field by field cannot take either behaviour silently.  (They carry a
    marker default only so that a behaviour built from milestone 1's seven
    fields reaches the checks of those seven first, and a malformed one of
    those is still refused by name.)

    ``snow_clip_key``
        ``"qs"`` (WRF, D11): the exit zeroes every snow bin where qs, the
        snow of bins 19-33, is below 1e-13 (:5291), so small ice (qi, bins
        1-18) is deleted at the end of every call unless large snow exists.
        ``"qi_plus_qs"``: the clip is keyed on all the snow, qi + qs.
    ``melt_step``
        ``"linear"`` (WRF, D12): ``J_W_MELT`` removes ``rate dt`` of a bin
        (:6683-6734) with no limit, so a bin goes negative once dt exceeds
        100 s in the middle bands (rate 0.5/50) or 176 s in the large ones
        (rate 0.683/120).  ``"exponential"``: the bin keeps
        ``exp(-rate dt)`` of itself, the exact solution of the same decay,
        which ``FREEZ`` already uses for freezing (:6610).
    ``fusion_heat``
        ``"wrf_v480"`` (WRF, D13): 333 in ``FREEZ`` and ``J_W_MELT``
        (:6627, :6741), 320 in riming (:8590), 334 in homogeneous freezing
        (:3108), so liquid-ice static energy does not close.
        ``"al2_minus_al1"``: 334, the scheme's own ``AL2 - AL1``, at every
        site (``fusion_heat``).
    ``homogeneous_freezing``
        ``"water_supersaturated"`` (WRF, D14): drops freeze at -38 C only
        inside the ``SUP1 > 0`` branch of ``JERNUCL01_KS`` (:3094-3111), so
        supercooled drops below -38 C survive in subsaturated air.
        ``"any_saturation"``: every point at or below -38 C.  The
        cold-column item measures both on KC-2 and records the decision
        with its evidence.
    ``dsupice_w``
        ``"wrf_v480"`` (WRF, D15): the ice-nucleation forcing
        ``DSUPICE_XYZ`` (:4504) uses ``w_stag``, a scalar set only where
        238.15 <= T_old < 274.15 K (:4405-4410); below 238.15 K it is the
        ``w_stag`` of the most recent point, in the pre-pass's (j, i, k)
        order, with 238.15 <= T_old < 274.15, and undefined if no such
        point came earlier in the call.  In a column whose cold levels lie
        above its warm-enough ones that is the highest warm-enough level
        below, not the level just below.  This preset defines the
        undefined read as the point's own ``w_stag`` (the catalogue records
        the choice).  ``"own_point"``: every point uses its own staggered
        w.
    ``dsupice_wind``
        ``"wrf_v480"`` (WRF, D17): the horizontal part of ``DSUPICE_XYZ``
        multiplies the centred differences ``DERIVS_X`` and ``DERIVS_Y``
        at (I, J) by ``UX`` and ``VX``, the average of ``U`` and ``V`` over
        (I..I+1, J..J+1) (:4497-4499), and by the point's own wind at I =
        IDE-1 or J = JDE-1 (:4500-4502).  WRF passes ``FAST_SBM`` the
        mass-point winds: ``solve_em`` calls the microphysics driver with
        ``u=grid%u_phy, v=grid%v_phy``, and ``u_phy = 0.5*(u(i)+u(i+1))``,
        ``v_phy = 0.5*(v(j)+v(j+1))`` (``module_big_step_utilities_em.F``
        :4857-4858, read in WRF v4.6.1 and v4.7.1).  The average is
        therefore a wind at (I+1/2, J+1/2), half a cell off diagonally from
        the differences it multiplies.  (Read as staggered winds it would
        still be off by half a cell, in j for U and in i for V.)
        ``"own_point"``: each point's own mass-point wind, ``U(I,K,J)`` and
        ``V(I,K,J)``.  The same average at :4411-4417 feeds only
        ``DTFREEZ_XYZ``, which nothing reads.
    ``supsat_small_ratio``
        ``"wrf_v480"`` (WRF, S1 and S2): ``JERSUPSAT_KS`` branches 8 and 9
        expand ``ALFA = sqrt(PI**2 + X)`` as ``PI + X/2`` (:2252-2256,
        :2286-2290) where ``PI + X/(2 PI)`` is right, and branch 7 clamps
        any ``GAMA`` below 1e-10 to 5e-14 (:2202), which fires whenever
        both phases are present.  ``"rederived"``: the forms rederived from
        the closed-form solution.  Only ``ONECOND2`` and ``ONECOND3`` reach
        these branches.
    ``fall_depth``
        ``"level_spacing"`` (WRF, D8): ``FALFLUXHUCM_Z`` divides a level's
        flux divergence by the distance between level centres, and at the
        top by the spacing below (:5364-5373), so column water is not
        conserved exactly on a stretched grid.  ``"layer_depth"``: by the
        layer's own depth.  Milestone 1 measured D8 and left it open; both
        presets keep WRF's divisor until the fall-exit item builds the
        layer-depth fall, and the warm call refuses ``"layer_depth"`` until
        then, because its sedimentation divides by the level spacing.
    """

    rainnc_velocity: str
    rainnc_source: str
    nucleation_water_sink: bool
    dtcond_carry: bool
    breakup_retry_cap: int
    cloud_base_rule: str
    fall_speed_lookup: str
    snow_clip_key: str = field(default=_NO_VALUE, kw_only=True)
    melt_step: str = field(default=_NO_VALUE, kw_only=True)
    fusion_heat: str = field(default=_NO_VALUE, kw_only=True)
    homogeneous_freezing: str = field(default=_NO_VALUE, kw_only=True)
    dsupice_w: str = field(default=_NO_VALUE, kw_only=True)
    dsupice_wind: str = field(default=_NO_VALUE, kw_only=True)
    supsat_small_ratio: str = field(default=_NO_VALUE, kw_only=True)
    fall_depth: str = field(default=_NO_VALUE, kw_only=True)

    def __post_init__(self) -> None:
        if self.rainnc_velocity not in _RAINNC_VELOCITY:
            raise FsbmRefusal(
                "FsbmBehavior",
                f"rainnc_velocity {self.rainnc_velocity!r} is not one of "
                f"{_RAINNC_VELOCITY}")
        if self.rainnc_source not in _RAINNC_SOURCE:
            raise FsbmRefusal(
                "FsbmBehavior",
                f"rainnc_source {self.rainnc_source!r} is not one of "
                f"{_RAINNC_SOURCE}")
        for name in ("nucleation_water_sink", "dtcond_carry"):
            if not isinstance(getattr(self, name), bool):
                raise FsbmRefusal(
                    "FsbmBehavior", f"{name} must be a bool, got "
                    f"{getattr(self, name)!r}")
        cap = self.breakup_retry_cap
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise FsbmRefusal(
                "FsbmBehavior",
                f"breakup_retry_cap must be a positive int, got {cap!r}")
        if self.cloud_base_rule not in CLOUD_BASE_RULES:
            raise FsbmRefusal(
                "FsbmBehavior",
                f"cloud_base_rule {self.cloud_base_rule!r} is not one of "
                f"{CLOUD_BASE_RULES}")
        if self.fall_speed_lookup not in FALL_SPEED_LOOKUPS:
            raise FsbmRefusal(
                "FsbmBehavior",
                f"fall_speed_lookup {self.fall_speed_lookup!r} is not one of "
                f"{FALL_SPEED_LOOKUPS}")
        # The cold fields are checked after milestone 1's, so a behaviour
        # built from those seven alone still reaches their refusals first.
        missing = [name for name in COLD_FIELDS
                   if getattr(self, name) is _NO_VALUE]
        if missing:
            raise TypeError(
                f"FsbmBehavior() missing {len(missing)} required keyword "
                f"argument{'s' if len(missing) > 1 else ''}: "
                + ", ".join(repr(name) for name in missing)
                + " (no cold-half field has a default; name each, or start "
                  "from a preset with dataclasses.replace)")
        for name, allowed in _COLD_VALUES.items():
            value = getattr(self, name)
            if not isinstance(value, str) or value not in allowed:
                raise FsbmRefusal(
                    "FsbmBehavior",
                    f"{name} {value!r} is not one of {allowed}")

    @classmethod
    def wrf_v480(cls) -> "FsbmBehavior":
        """WRF v4.8.0 as written: the preset the oracle gates run."""
        return cls(rainnc_velocity="tile_last",
                   rainnc_source="level1_after_fall",
                   nucleation_water_sink=False,
                   dtcond_carry=True,
                   breakup_retry_cap=64,
                   cloud_base_rule="wrf_v480",
                   fall_speed_lookup="wrf_v480",
                   snow_clip_key="qs",
                   melt_step="linear",
                   fusion_heat="wrf_v480",
                   homogeneous_freezing="water_supersaturated",
                   dsupice_w="wrf_v480",
                   dsupice_wind="wrf_v480",
                   supsat_small_ratio="wrf_v480",
                   fall_depth="level_spacing")

    @classmethod
    def corrected(cls) -> "FsbmBehavior":
        """The reference default: D1, D2, D3, D5, D9, D10, D11, D12, D13,
        D14, D15, D17, S1 and S2 corrected, D7 capped.  D8 keeps WRF's divisor
        until the fall-exit item builds the layer-depth fall (see
        ``fall_depth``)."""
        return cls(rainnc_velocity="own_column",
                   rainnc_source="bottom_outflow",
                   nucleation_water_sink=True,
                   dtcond_carry=False,
                   breakup_retry_cap=64,
                   cloud_base_rule="first_level",
                   fall_speed_lookup="own_bin",
                   snow_clip_key="qi_plus_qs",
                   melt_step="exponential",
                   fusion_heat="al2_minus_al1",
                   homogeneous_freezing="any_saturation",
                   dsupice_w="own_point",
                   dsupice_wind="own_point",
                   supsat_small_ratio="rederived",
                   fall_depth="level_spacing")


#: The four fusion-heat sites of ``FUSION_HEAT_WRF``.
FUSION_HEAT_SITES = tuple(FUSION_HEAT_WRF)


def fusion_heat(behavior: FsbmBehavior, site: str) -> float:
    """The latent heat of fusion over the heat capacity of air the cold path
    uses at ``site`` under ``behavior`` (finding D13), K per unit mass
    mixing ratio.

    ``site`` is one of ``FUSION_HEAT_SITES``: ``"freez"`` (:6627),
    ``"j_w_melt"`` (:6741), ``"riming"`` (:8590) and ``"homogeneous"``
    (:3108).  ``fusion_heat="wrf_v480"`` returns WRF's literal there;
    ``"al2_minus_al1"`` returns ``AL2 - AL1`` (334) everywhere.
    """
    if not isinstance(behavior, FsbmBehavior):
        raise FsbmRefusal("fusion_heat", f"behavior must be an FsbmBehavior, "
                          f"got {type(behavior).__name__}")
    if site not in FUSION_HEAT_WRF:
        raise FsbmRefusal("fusion_heat", f"site {site!r} is not one of "
                          f"{FUSION_HEAT_SITES}")
    if behavior.fusion_heat == "wrf_v480":
        return FUSION_HEAT_WRF[site]
    return AL2 - AL1


# ---------------------------------------------------------------------------
# The dense-ice class, a named option (:3886)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DenseIce:
    """One choice of the dense-ice class, which WRF fixes at compile time
    with ``hail_opt`` (:3886).

    The advected bins ``ff6`` hold this class.  Each option names the WRF
    arrays that go with it: the mass grid of the entry and exit
    conversions (:4239-4251, :5300-5313), the scheme-internal spectrum
    (``FF5R`` for hail, ``FF4R`` for graupel, :4560-4577), the fall speeds
    of its sedimentation and surface precipitation (:4941-4945, :5092-5119),
    the capacities and bulk densities of its slot in ``ONECOND2`` and
    ``ONECOND3``, which receive both slots and find the other one empty
    (:4735-4736, :4754-4755), and the two collection kernels it meets drops
    with (:8476-8537).

    ``identity`` names the option, so a graupel run is never mistaken for
    a hail run: a cold call's result records the option it ran with
    (``column.FsbmResult.dense_ice``, whose ``identity`` this is).
    Obtain an option with ``dense_ice(name)``.  The cold call and every
    process item of milestone 2 take a ``DenseIce``, never an integer.
    ``hail_opt`` is WRF's integer, for the oracle builds (the graupel
    builds change only :3886) and for milestone 1's state and
    sedimentation functions, which still take it: the cold path hands them
    ``option.hail_opt`` and nothing else.
    """

    name: str
    hail_opt: int
    mass_grid: str
    distribution: str
    fall_speed: str
    capacity: str
    bulk_density: str
    kernels: tuple[str, str]

    @property
    def identity(self) -> str:
        return f"fast-sbm/dense-ice={self.name}"


#: The two options.  ``hail`` is WRF's compiled default.
DENSE_ICE_OPTIONS = {
    "hail": DenseIce(name="hail", hail_opt=1, mass_grid="xh",
                     distribution="ff5r", fall_speed="vr5", capacity="rhec",
                     bulk_density="ro5bl", kernels=("cwlh", "cwhl")),
    "graupel": DenseIce(name="graupel", hail_opt=0, mass_grid="xg",
                        distribution="ff4r", fall_speed="vr4",
                        capacity="rgec", bulk_density="ro4bl",
                        kernels=("cwlg", "cwgl")),
}
#: WRF's compiled choice (``hail_opt = 1``, :3886).
DENSE_ICE_DEFAULT = "hail"


def dense_ice(name: str = DENSE_ICE_DEFAULT) -> DenseIce:
    """The dense-ice option called ``name``, ``"hail"`` or ``"graupel"``.

    Anything else is refused, WRF's ``hail_opt`` integer included: the
    option is chosen by name, so a changed default can never pass as the
    same scheme.
    """
    if not isinstance(name, str) or name not in DENSE_ICE_OPTIONS:
        raise FsbmRefusal("dense_ice", f"the dense-ice class is named "
                          f"{tuple(DENSE_ICE_OPTIONS)}, got {name!r}")
    return DENSE_ICE_OPTIONS[name]
