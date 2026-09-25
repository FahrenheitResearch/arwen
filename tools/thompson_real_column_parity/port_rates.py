"""Read every mp=28 and mp=8 process rate back out of the port's own
kernels, on the host, under WRF's names.

The production kernels keep each rate in a local variable and fold it into
the state before returning, so nothing outside can see it.  This module
rewrites a COPY of the kernel sources (three aerosol units for mp=28, the
shared ``thompson`` unit for mp=8) -- the text
``gpuwm.core.kernels.module_source`` returns -- by inserting one
``HOST_RATE(slot, idx, value)`` call per rate at the single point where the
rate is final (after every limiter, immediately before the state update),
guarded by ``#ifdef GPUWM_HOST_RATES``.  Nothing else in the text changes;
``real_column_parity.py`` proves it by running the instrumented and the
pristine host builds on the same columns and requiring byte-identical
state.  The committed ``.cu`` files are never touched.

The slot of a rate is its index in ``instrument_wrf_rates.RATES``, so the
port's buffer and WRF's cp1/cp2 streams line up by construction.  Each map
below is ``{WRF name: expression in the kernel}``; a WRF rate a kernel does
not compute stays zero there, which is WRF's value for it on that branch
(the cold kernel owns no melting, the warm kernel no ice processes).
"""

from __future__ import annotations

from instrument_wrf_rates import RATES

SLOT = {name: i for i, name in enumerate(RATES)}

#: Diagnostic readbacks after the process rates, in slots len(RATES) + i.
#: The rain-evaporation kernel's post-adjustment supersaturation, for the
#: conditioning analysis of prv_rev / pnr_rev (WRF's is ``ssatw`` at cp2).
DIAGNOSTICS = ("ssatw_rev",)
for _i, _name in enumerate(DIAGNOSTICS):
    SLOT[_name] = len(RATES) + _i
NSLOTS = len(RATES) + len(DIAGNOSTICS)

#: thompson_aa_cold_network: every cell with entry T < 273.15 K.
COLD = {
    "pri_ide": "ice_rate", "prs_ide": "ice_to_snow_rate",
    "pni_ide": "ice_number_rate",
    "prs_iau": "autoconversion_rate", "pni_iau": "autoconversion_number_rate",
    "pri_rci": "rain_ice_ice_rate", "prr_rci": "rain_ice_rain_rate",
    "pni_rci": "rain_ice_ice_number_rate",
    "pnr_rci": "rain_ice_rain_number_rate",
    "prg_rci": "rain_ice_graupel_rate",
    "pnr_rcr": "rain_self_number_rate",
    "pri_rfz": "freeze_ice_rate", "pni_rfz": "freeze_ice_number_rate",
    "prg_rfz": "freeze_graupel_rate", "pnr_rfz": "freeze_graupel_number_rate",
    "prr_rcs": "rain_snow_rain_rate", "prs_rcs": "rain_snow_category_rate",
    "prg_rcs": "rain_snow_graupel_rate", "pnr_rcs": "rain_snow_number_rate",
    "prr_rcg": "rain_graupel_rain_rate", "prg_rcg": "rain_graupel_graupel_rate",
    "pnr_rcg": "rain_graupel_number_rate",
    "prr_wau": "cloud_autoconversion_rate",
    "pnr_wau": "cloud_autoconversion_number_rate",
    "pnc_wau": "cloud_number_autoconversion_rate",
    "prr_rcw": "cloud_rain_accretion_rate", "pnc_rcw": "cloud_number_rain_rate",
    "pna_rca": "nwfa_rain_rate", "pnd_rcd": "nifa_rain_rate",
    "pri_wfz": "cloud_freezing_rate", "pni_wfz": "cloud_freezing_number_rate",
    "prs_scw": "snow_riming_rate", "prg_scw": "snow_graupel_conversion_rate",
    "prg_gcw": "graupel_riming_rate",
    "pnc_scw": "cloud_number_snow_rate", "pnc_gcw": "cloud_number_graupel_rate",
    "pni_ihm": "hm_number_rate", "pri_ihm": "hm_mass_rate",
    "prs_ihm": "snow_hm_rate", "prg_ihm": "graupel_hm_rate",
    "pna_sca": "nwfa_snow_rate", "pnd_scd": "nifa_snow_rate",
    "pna_gca": "nwfa_graupel_rate", "pnd_gcd": "nifa_graupel_rate",
    "pri_inu": "nucleation_rate", "pni_inu": "nucleation_number_rate",
    "pri_iha": "koop_rate", "pni_iha": "koop_number_rate",
    "prs_sde": "snow_rate", "prg_gde": "graupel_rate",
    "prs_sci": "snow_collection_rate",
    "pni_sci": "snow_collection_number_rate",
    "png_scw": "snow_graupel_conversion_number_rate",
    "png_rcs": "rain_snow_number_rate",
}

#: The cold kernel's graupel-vapour number rate lives inside the shadow
#: update block, so it is read back there.
COLD_SHADOW = {"png_gde": "graupel_vapor_number_rate"}

#: thompson_aa_warm_source_network: every cell with entry T >= 273.15 K.
WARM = {
    "prr_wau": "autoconversion_rate", "pnr_wau": "autoconversion_number_rate",
    "prr_rcw": "rain_cloud_rate", "pnr_rcr": "rain_self_number_rate",
    "pnc_wau": "cloud_autoconversion_number_sink",
    "pnc_rcw": "cloud_accretion_number_sink",
    "pna_rca": "ccn_rain_scavenge", "pnd_rcd": "in_rain_scavenge",
    "prs_scw": "snow_cloud_rate", "prg_gcw": "graupel_cloud_rate",
    "pnc_scw": "cloud_snow_number_sink", "pnc_gcw": "cloud_graupel_number_sink",
    "pna_sca": "ccn_snow_scavenge", "pnd_scd": "in_snow_scavenge",
    "pna_gca": "ccn_graupel_scavenge", "pnd_gcd": "in_graupel_scavenge",
    "prr_rcs": "rain_snow_rain_rate", "prs_rcs": "rain_snow_snow_rate",
    "prg_rcs": "rain_snow_graupel_rate", "pnr_rcs": "rain_snow_number_rate",
    "prr_rcg": "rain_graupel_rain_rate", "prg_rcg": "rain_graupel_graupel_rate",
    "pnr_rcg": "rain_graupel_rain_number_rate",
    "prr_sml": "snow_melt_rate", "pnr_sml": "snow_melt_number_rate",
    "prr_gml": "graupel_melt_rate", "pnr_gml": "graupel_melt_number_rate",
    "prs_sde": "snow_vapor_rate", "prg_gde": "graupel_vapor_rate",
    "png_rcs": "rain_snow_number_rate", "png_rcg": "rain_graupel_number_rate",
    "png_gde": "graupel_vapor_number_rate",
}

#: thompson_aa_saturation_adjust_impl and thompson_aa_rain_evaporation_impl.
CONDENSATION = {"prw_vcd": "prw_vcd", "pnc_wcd": "pnc_wcd"}
RAIN_EVAPORATION = {"prv_rev": "prv_rev", "pnr_rev": "pnr_rev"}

#: module -> [(anchor line, rate map)].  The block goes immediately BEFORE
#: the anchor, which must occur exactly once in the module's source.
ANCHORS = {
    "thompson_aerosol_cold": [
        ("    qi[idx] = fmaxf(0.0f, thompson_aa_add(qi[idx],\n", COLD),
        ("        graupel_number_shadow[idx] = thompson_aa_add("
         "initial_number_per_kg,\n", COLD_SHADOW)],
    "thompson_aerosol_warm": [
        ("    snow_melt_marker[idx] = snow_melt_rate > 0.0 ? 1.0f : 0.0f;\n",
         WARM)],
    "thompson_aerosol_sat": [
        ("    const float prw = (float)prw_vcd;\n", CONDENSATION),
        ("    // :3501.\n    if (ssatw >= -THOMPSON_AA_SAT_EPS) return;\n",
         {"ssatw_rev": "ssatw"}),
        ("    const float qr_tendency = (float)(-prv_rev);\n",
         RAIN_EVAPORATION)],
}

# ---------------------------------------------------------------------------
# mp_physics=8: the classic kernels of the shared thompson.cu.
# ---------------------------------------------------------------------------

#: The WRF rates that reach nothing but the droplet number and the two
#: aerosol numbers, which classic Thompson does not carry (mp_gt_driver
#: writes nc, nwfa and nifa back only when is_aerosol_aware, :1316-1340), and
#: Koop homogeneous freezing of deliquesced aerosol, which :2635 forms only
#: when is_aerosol_aware.  The mp=8 port computes none of them.  Their one
#: indirect route into the state, the droplet number the cloud freezing below
#: HGFR hands the ice (:3959, ``xnc = nc1d + ncten*DT``), is graded through
#: the final ice number instead.
NOT_CARRIED_MP8 = (
    "pnc_wcd", "pnc_wau", "pnc_rcw", "pnc_scw", "pnc_gcw",
    "pna_rca", "pna_sca", "pna_gca", "pnd_rcd", "pnd_scd", "pnd_gcd",
    "pri_iha", "pni_iha",
)

#: thompson_frozen_vapor_cloud_network: every cell with entry T < 273.15 K.
#: The classic cold kernel is the one the mp=28 cold network was built
#: from, so its locals carry the same names; the aerosol terms are absent.
COLD_MP8 = {name: expr for name, expr in COLD.items()
            if name not in NOT_CARRIED_MP8}

#: thompson_warm_frozen_source_network: every cell with entry T >= 273.15 K.
WARM_MP8 = {name: expr for name, expr in WARM.items()
            if name not in NOT_CARRIED_MP8}

#: thompson_cloud_saturation_adjust_impl keeps the adjustment in mixing
#: ratio (``clap``, kg/kg); WRF's rate is ``prw_vcd = clap*odt`` (:3412,
#: and ``-rc*orho*odt`` where the cloud would empty, :3473, which the kernel
#: forms as ``clap = -rc/rho``).  The readback is ``clap`` and
#: ``real_column_parity`` divides it by the step: ``PER_STEP_MP8``.
CONDENSATION_MP8 = {"prw_vcd": "clap"}
PER_STEP_MP8 = ("prw_vcd",)

#: thompson_rain_evaporation_impl.
RAIN_EVAPORATION_MP8 = {"prv_rev": "evaporation_rate",
                        "pnr_rev": "number_rate"}

ANCHORS_MP8 = {
    "thompson": [
        ("    qi[idx] = fmaxf(0.0f, qi[idx]\n"
         "        + (float)((nucleation_rate + hm_mass_rate\n", COLD_MP8),
        ("        graupel_number_shadow[idx] = initial_number_per_kg\n"
         "            + (float)(number_rate * (double)orho) * dt;\n",
         COLD_SHADOW),
        ("    snow_melt_marker[idx] = snow_melt_rate > 0.0 ? 1.0f : 0.0f;\n",
         WARM_MP8),
        ("    if (condensation_marker != nullptr) {\n"
         "        condensation_marker[idx] = clap > 0.0f ? 1.0f : 0.0f;\n",
         CONDENSATION_MP8),
        ("    if (ssatw >= -1.0e-15f) return;\n\n"
         "    const float orho = 1.0f / rho;\n"
         "    const float rr = qr[idx] * rain_density;\n",
         {"ssatw_rev": "ssatw"}),
        ("    const float qr_tendency = (float)(-evaporation_rate);\n",
         RAIN_EVAPORATION_MP8)],
}

#: The anchor set of each scheme.
ANCHORS_BY_MP = {28: ANCHORS, 8: ANCHORS_MP8}
#: Rates each scheme's port does not compute, reported but not graded.
NOT_CARRIED = {28: (), 8: NOT_CARRIED_MP8}
#: Readbacks that are per-step changes; divided by dt to give WRF's rate.
PER_STEP = {28: (), 8: PER_STEP_MP8}


def _block(rate_map: dict[str, str]) -> str:
    lines = ["#ifdef GPUWM_HOST_RATES\n"]
    for wrf_name, expr in rate_map.items():
        lines.append(f"    HOST_RATE({SLOT[wrf_name]}, (long long)idx, "
                     f"(double)({expr}));  /* {wrf_name} */\n")
    lines.append("#endif\n")
    return "".join(lines)


def instrument(module: str, text: str, mp: int = 28) -> str:
    """The module source with the rate readback inserted."""
    for anchor, rate_map in ANCHORS_BY_MP[mp][module]:
        count = text.count(anchor)
        if count != 1:
            raise ValueError(
                f"{module}: anchor {anchor.strip()!r} found {count} times")
        unknown = set(rate_map) - set(SLOT)
        if unknown:
            raise ValueError(f"{module}: not WRF rates or diagnostics: "
                             f"{sorted(unknown)}")
        text = text.replace(anchor, _block(rate_map) + anchor)
    return text


def instrumented_modules(mp: int = 28) -> tuple[str, ...]:
    return tuple(ANCHORS_BY_MP[mp])


__all__ = ["ANCHORS", "ANCHORS_BY_MP", "ANCHORS_MP8", "COLD", "COLD_MP8",
           "COLD_SHADOW", "CONDENSATION", "CONDENSATION_MP8",
           "DIAGNOSTICS", "NOT_CARRIED", "NOT_CARRIED_MP8", "NSLOTS",
           "PER_STEP", "RAIN_EVAPORATION", "RAIN_EVAPORATION_MP8", "SLOT",
           "WARM", "WARM_MP8", "instrument", "instrumented_modules"]
