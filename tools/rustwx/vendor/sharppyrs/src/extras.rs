//! Small numerics the display needs that `sharprs` does not (yet) provide.

use sharprs::Profile;
use sharprs::params::cape::{self, LiftedParcelLevel, ParcelResult, ParcelType};
use sharprs::params::indices;
use sharprs::thermo;
use sharprs::winds;

use crate::utils::qc;

/// Replace sharprs' eager Bolton/approximate profile arrays with the
/// formulations used by SHARPpy's `BasicProfile`.
///
/// A single normalization point is important: downstream indices borrow
/// these cached arrays, so correcting only the values exported through the
/// Python bridge leaves calculations such as wet-bulb-zero and lapse rates
/// on a different thermodynamic basis. The raw observations are unchanged.
pub fn normalize_sharppy_thermodynamics(profile: &mut Profile) {
    let n = profile.pres.len();
    profile.vtmp = Vec::with_capacity(n);
    profile.theta = Vec::with_capacity(n);
    profile.thetae = Vec::with_capacity(n);
    profile.wvmr = Vec::with_capacity(n);
    profile.relh = Vec::with_capacity(n);
    profile.wetbulb = Vec::with_capacity(n);

    for i in 0..n {
        let p = profile.pres[i];
        let t = profile.tmpc[i];
        let td = profile.dwpc[i];
        let have_pt = p.is_finite() && p > 0.0 && t.is_finite();
        let have_td = have_pt && td.is_finite();

        // SHARPpy's virtemp deliberately falls back to the dry temperature
        // when the dewpoint is masked.
        profile.vtmp.push(if have_pt {
            thermo::virtemp(p, t, have_td.then_some(td))
        } else {
            f64::NAN
        });
        profile.theta.push(if have_pt {
            thermo::theta(p, t, 1000.0) + 273.15
        } else {
            f64::NAN
        });
        profile.thetae.push(if have_td {
            thermo::thetae(p, t, td) + 273.15
        } else {
            f64::NAN
        });
        profile.wvmr.push(if have_td {
            thermo::mixratio(p, td)
        } else {
            f64::NAN
        });
        profile.relh.push(if have_td {
            thermo::relh(p, t, td)
        } else {
            f64::NAN
        });
        profile.wetbulb.push(if have_td {
            thermo::wetbulb(p, t, td)
        } else {
            f64::NAN
        });
    }
}

/// Build the parcel solver's lean profile while retaining SHARPpy's virtual-
/// temperature fallback at levels with temperature but no reported dewpoint.
/// Candidate parcel moisture remains missing; only the environment virtual
/// temperature used during ascent is repaired.
pub fn cape_profile(inner: &Profile) -> cape::Profile {
    let mut profile = cape::Profile::new(
        inner.pres.clone(),
        inner.hght.clone(),
        inner.tmpc.clone(),
        inner.dwpc.clone(),
        inner.sfc,
    );
    // The lean CAPE profile already uses SHARPpy's Wobus theta-e/wet-bulb
    // functions. Copy the centrally normalized arrays so it also inherits
    // SHARPpy's dry-temperature virtual-temperature fallback.
    profile.vtmp.clone_from(&inner.vtmp);
    profile.thetae = inner.thetae.iter().map(|value| value - 273.15).collect();
    profile.wetbulb.clone_from(&inner.wetbulb);
    profile
}

/// Whether every reported wind vector is the same finite vector.
///
/// Parcel Bunkers motion is mathematically undefined when the layer shear is
/// exactly zero.  Legacy NumPy subsequently reduces its all-NaN helicity
/// layers to zero; callers use this predicate to preserve that narrow public
/// behavior without turning genuinely missing wind profiles into zeros.
pub fn has_constant_wind(profile: &Profile) -> bool {
    let mut reference: Option<(f64, f64)> = None;
    let mut count = 0usize;
    for (&u, &v) in profile.u.iter().zip(&profile.v) {
        if !u.is_finite() || !v.is_finite() {
            continue;
        }
        count += 1;
        if let Some((u0, v0)) = reference {
            if (u - u0).abs() > 1.0e-9 || (v - v0).abs() > 1.0e-9 {
                return false;
            }
        } else {
            reference = Some((u, v));
        }
    }
    count >= 2
}

/// The literal one-hPa pressure sequence produced by SHARPpy's
/// ``np.arange(pbot, ptop - 1, -1)`` fast mean routines.
///
/// Fractional effective-layer bounds make the final sample extend slightly
/// above the requested layer top.  Clamping that last point to ``ptop`` looks
/// cleaner, but changes the effective-parcel moisture enough to move its
/// LFC/EL and CAP metadata, so preserve NumPy's historical sequence exactly.
fn sharppy_fast_pressure_grid(pbot: f64, ptop: f64) -> Option<Vec<f64>> {
    if !pbot.is_finite() || !ptop.is_finite() || pbot < ptop {
        return None;
    }
    let stop = ptop - 1.0;
    let mut values = Vec::with_capacity((pbot - ptop).ceil() as usize + 1);
    let mut pressure = pbot;
    while pressure > stop && values.len() < 2_000 {
        values.push(pressure);
        pressure -= 1.0;
    }
    (!values.is_empty()).then_some(values)
}

/// SHARPpy's default (``exact=False``) mean mixing ratio.
pub fn sharppy_mean_mixratio(prof: &Profile, pbot: f64, ptop: f64) -> Option<f64> {
    if !prof.interp_tmpc(pbot).is_finite() || !prof.interp_tmpc(ptop).is_finite() {
        return None;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for pressure in sharppy_fast_pressure_grid(pbot, ptop)? {
        let dewpoint = prof.interp_dwpc(pressure);
        let mixing_ratio = thermo::mixratio(pressure, dewpoint);
        if mixing_ratio.is_finite() {
            total += mixing_ratio;
            count += 1;
        }
    }
    (count > 0).then_some(total / count as f64)
}

/// SHARPpy's default pressure-weighted mean potential temperature.
pub fn sharppy_mean_theta(prof: &Profile, pbot: f64, ptop: f64) -> Option<f64> {
    if !prof.interp_tmpc(pbot).is_finite() || !prof.interp_tmpc(ptop).is_finite() {
        return None;
    }
    let mut weighted = 0.0;
    let mut weights = 0.0;
    for pressure in sharppy_fast_pressure_grid(pbot, ptop)? {
        let temperature = prof.interp_tmpc(pressure);
        let theta = thermo::theta(pressure, temperature, 1000.0);
        if theta.is_finite() {
            weighted += theta * pressure;
            weights += pressure;
        }
    }
    (weights > 0.0).then_some(weighted / weights)
}

/// SHARPpy's stripped `cape(..., trunc=True)` CAPE/CIN calculation.
///
/// `convective_temp` intentionally stops the observed-level ascent at 500
/// hPa, then closes the calculation with one interpolated layer from that
/// level to the sounding top.  A full `parcelx` result is not equivalent:
/// elevated and sparse soundings can retain CIN for several extra observed
/// layers and therefore require too much surface heating.  Keep this helper
/// narrow (surface parcel, full sounding, one-hPa sub-LCL integration) so the
/// historical SHARPpy search semantics are explicit and inexpensive.
pub fn cape_truncated_sharppy(prof: &Profile, pres: f64, tmpc: f64, dwpc: f64) -> (f64, f64) {
    cape_sharppy_impl(
        prof,
        pres,
        tmpc,
        dwpc,
        prof.sfc_pressure(),
        prof.pres[prof.top],
        true,
    )
}

/// SHARPpy's stripped `cape()` over an explicit pressure layer.
///
/// The local HGZ/0-3/0-6 km diagnostics use this exact bounded integral. A
/// full `parcelx` computes different bookkeeping at an interpolated upper
/// boundary, while sharprs' current lightweight helper can return missing on
/// sparse upper-moisture profiles. This port preserves the reference result
/// without constructing a Python profile.
pub fn cape_bounded_sharppy(
    prof: &Profile,
    pres: f64,
    tmpc: f64,
    dwpc: f64,
    pbot: f64,
    ptop: f64,
) -> (f64, f64) {
    cape_sharppy_impl(prof, pres, tmpc, dwpc, pbot, ptop, false)
}

fn cape_sharppy_impl(
    prof: &Profile,
    pres: f64,
    tmpc: f64,
    dwpc: f64,
    requested_pbot: f64,
    ptop: f64,
    truncate_at_500_hpa: bool,
) -> (f64, f64) {
    const G: f64 = 9.80665;
    let missing = (f64::NAN, f64::NAN);
    if prof.pres.len() < 2
        || ![pres, tmpc, dwpc, requested_pbot, ptop]
            .iter()
            .all(|v| v.is_finite())
    {
        return missing;
    }

    let mut pbot = requested_pbot.min(pres);
    let interp_vtmp = |p: f64| prof.interp_by_pressure(&prof.vtmp, p);
    if !interp_vtmp(pbot).is_finite() || !interp_vtmp(ptop).is_finite() {
        return missing;
    }

    let (lcl_pres, lcl_temp) = thermo::drylift(pres, tmpc, dwpc);
    if !lcl_pres.is_finite() || !lcl_temp.is_finite() {
        return missing;
    }
    let theta_parcel = thermo::theta(lcl_pres, lcl_temp, 1000.0);
    let parcel_mixratio = thermo::mixratio(pres, dwpc);
    let mut positive = 0.0;
    let mut negative = 0.0;

    // Literal `np.arange(pbot, lcl_pres - 1, -1)` sampling used by Python.
    let stop = lcl_pres - 1.0;
    let mut pressure = pbot;
    let mut previous: Option<(f64, f64)> = None;
    let mut samples = 0usize;
    while pressure > stop && samples < 2_000 {
        let height = prof.interp_hght(pressure);
        let env_theta = thermo::theta(pressure, prof.interp_tmpc(pressure), 1000.0);
        let env_dewpoint = prof.interp_dwpc(pressure);
        let env_vtmp = thermo::virtemp(
            pressure,
            env_theta,
            env_dewpoint.is_finite().then_some(env_dewpoint),
        );
        let parcel_vtmp = thermo::virtemp(
            pressure,
            theta_parcel,
            Some(thermo::temp_at_mixrat(parcel_mixratio, pressure)),
        );
        let deficit = (parcel_vtmp - env_vtmp) / thermo::ctok(env_vtmp);
        if let Some((previous_deficit, previous_height)) = previous {
            if [deficit, height, previous_deficit, previous_height]
                .iter()
                .all(|v| v.is_finite())
            {
                let energy = G * (previous_deficit + deficit) / 2.0 * (height - previous_height);
                if energy < 0.0 {
                    negative += energy;
                }
            }
        }
        previous = Some((deficit, height));
        pressure -= 1.0;
        samples += 1;
    }

    if pbot > lcl_pres {
        pbot = lcl_pres;
    }
    if pbot < ptop {
        return missing;
    }
    let Some(first) = prof.pres.iter().position(|level| pbot > *level) else {
        return missing;
    };
    let upper_index = prof
        .pres
        .iter()
        .rposition(|level| ptop < *level)
        .unwrap_or(prof.top);

    let mut pe1 = pbot;
    let mut h1 = prof.interp_hght(pe1);
    let mut te1 = interp_vtmp(pe1);
    let mut tp1 = lcl_temp;
    for index in first..prof.pres.len() {
        if !prof.tmpc[index].is_finite() {
            continue;
        }
        let pe2 = prof.pres[index];
        let h2 = prof.hght[index];
        let te2 = prof.vtmp[index];
        let tp2 = thermo::wetlift(pe1, tp1, pe2);
        let tdef1 = (thermo::virtemp(pe1, tp1, Some(tp1)) - te1) / thermo::ctok(te1);
        let tdef2 = (thermo::virtemp(pe2, tp2, Some(tp2)) - te2) / thermo::ctok(te2);
        let layer = G * (tdef1 + tdef2) / 2.0 * (h2 - h1);
        if !layer.is_finite() {
            continue;
        }
        if layer > 0.0 {
            positive += layer;
        } else if pe2 > 500.0 {
            negative += layer;
        }

        pe1 = pe2;
        h1 = h2;
        te1 = te2;
        tp1 = tp2;

        if (truncate_at_500_hpa && pe2 <= 500.0) || index >= upper_index {
            // SHARPpy first removes/re-adds the terminating observed layer,
            // then integrates directly from that observation to `ptop`.
            let mut bplus;
            let mut bminus;
            if layer > 0.0 {
                bplus = positive - layer;
                bminus = negative;
            } else {
                bplus = positive;
                bminus = if pe2 > 500.0 {
                    negative + layer
                } else {
                    negative
                };
            }

            let final_height = prof.interp_hght(ptop);
            let final_env_vtmp = interp_vtmp(ptop);
            let final_parcel_temp = thermo::wetlift(pe1, tp1, ptop);
            let start_deficit = (thermo::virtemp(pe1, tp1, Some(tp1)) - te1) / thermo::ctok(te1);
            let final_deficit = (thermo::virtemp(ptop, final_parcel_temp, Some(final_parcel_temp))
                - final_env_vtmp)
                / thermo::ctok(final_env_vtmp);
            let final_layer = G * (start_deficit + final_deficit) / 2.0 * (final_height - h1);
            if final_layer.is_finite() {
                if final_layer > 0.0 {
                    bplus += final_layer;
                } else if ptop > 500.0 {
                    bminus += final_layer;
                }
            }
            if bplus == 0.0 {
                bminus = 0.0;
            }
            return (bplus, bminus);
        }
    }
    missing
}

fn interp_log_pressure(p: f64, pres: &[f64], field: &[f64]) -> f64 {
    if !p.is_finite() || p <= 0.0 || pres.len() != field.len() {
        return f64::NAN;
    }
    let mut lower: Option<(f64, f64)> = None;
    let mut upper: Option<(f64, f64)> = None;
    for (&level_p, &value) in pres.iter().zip(field) {
        if !level_p.is_finite() || level_p <= 0.0 || !value.is_finite() {
            continue;
        }
        if (level_p - p).abs() < 1.0e-9 {
            return value;
        }
        if level_p > p && lower.is_none_or(|(old, _)| level_p < old) {
            lower = Some((level_p, value));
        }
        if level_p < p && upper.is_none_or(|(old, _)| level_p > old) {
            upper = Some((level_p, value));
        }
    }
    let (p0, v0) = lower.unwrap_or((f64::NAN, f64::NAN));
    let (p1, v1) = upper.unwrap_or((f64::NAN, f64::NAN));
    if ![p0, v0, p1, v1].iter().all(|value| value.is_finite()) {
        return f64::NAN;
    }
    let fraction = (p.ln() - p0.ln()) / (p1.ln() - p0.ln());
    v0 + (v1 - v0) * fraction
}

/// SHARPpy's pressure-weighted, interpolated mean theta-e.
///
/// This is deliberately separate from sharprs' DCAPE helper.  The reference
/// routine interpolates the profile's cached Wobus theta-e field on a 1-hPa
/// grid and weights each sample by pressure.  Recomputing theta-e at every
/// sample and taking an arithmetic mean can select a different downdraft
/// source layer on sparse or strongly perturbed soundings.
fn sharppy_mean_thetae(prof: &cape::Profile, mut pbot: f64, ptop: f64) -> Option<f64> {
    if !interp_log_pressure(pbot, &prof.pres, &prof.tmpc).is_finite() {
        pbot = prof.pres[prof.sfc];
    }
    if !interp_log_pressure(ptop, &prof.pres, &prof.tmpc).is_finite() {
        return None;
    }

    let mut weighted = 0.0;
    let mut weights = 0.0;
    for pressure in sharppy_fast_pressure_grid(pbot, ptop)? {
        let thetae = interp_log_pressure(pressure, &prof.pres, &prof.thetae);
        if thetae.is_finite() {
            weighted += thetae * pressure;
            weights += pressure;
        }
    }
    (weights > 0.0).then_some(weighted / weights)
}

/// Literal SHARPpy downdraft-CAPE source selection, integration, and trace.
///
/// sharprs exposes the same diagnostic, but its current implementation uses
/// an unweighted recomputed-theta-e mean and skips the reported level at the
/// top of the descent.  Both differences are normally tiny; together they
/// can move the chosen 100-hPa source layer by hundreds of hPa in sparse
/// upper-moisture profiles.  Keep this compatibility routine here until that
/// upstream implementation adopts the reference semantics.
pub fn sharppy_dcape(prof: &cape::Profile) -> cape::DcapeResult {
    const G: f64 = 9.80665;
    let sfc_pres = prof.pres[prof.sfc];
    let valid: Vec<usize> = (0..prof.pres.len())
        .filter(|&index| prof.pres[index].is_finite() && prof.thetae[index].is_finite())
        .collect();

    let mut minimum = 1000.0;
    let mut source_pressure = f64::NAN;
    for &index in &valid {
        let pressure = prof.pres[index];
        if pressure < sfc_pres - 400.0 {
            continue;
        }
        let Some(mean) = sharppy_mean_thetae(prof, pressure, pressure - 100.0) else {
            continue;
        };
        if mean < minimum {
            minimum = mean;
            source_pressure = pressure - 50.0;
        }
    }

    if !source_pressure.is_finite() {
        return cape::DcapeResult {
            dcape: 0.0,
            ttrace: Vec::new(),
            ptrace: Vec::new(),
        };
    }
    let Some(source_position) = valid
        .iter()
        .rposition(|&index| prof.pres[index] >= source_pressure)
    else {
        return cape::DcapeResult {
            dcape: 0.0,
            ttrace: Vec::new(),
            ptrace: Vec::new(),
        };
    };
    let source_temp = interp_log_pressure(source_pressure, &prof.pres, &prof.tmpc);
    let source_dewpoint = interp_log_pressure(source_pressure, &prof.pres, &prof.dwpc);
    let source_wetbulb = thermo::wetbulb(source_pressure, source_temp, source_dewpoint);
    let mut pe1 = source_pressure;
    let mut te1 = source_temp;
    let mut h1 = interp_log_pressure(source_pressure, &prof.pres, &prof.hght);
    let mut tp1 = source_wetbulb;
    let mut total = 0.0;
    let mut descending_ttrace = Vec::with_capacity(source_position + 1);
    let mut descending_ptrace = Vec::with_capacity(source_position + 1);

    // SHARPpy walks the filtered reported levels from the source down through
    // the surface, including both endpoints.
    for position in (0..=source_position).rev() {
        let index = valid[position];
        let pe2 = prof.pres[index];
        let te2 = prof.tmpc[index];
        let h2 = prof.hght[index];
        let tp2 = thermo::wetlift(pe1, tp1, pe2);
        if te1.is_finite() && te2.is_finite() {
            let tdef1 = (tp1 - te1) / (te1 + 273.15);
            let tdef2 = (tp2 - te2) / (te2 + 273.15);
            total += G * (tdef1 + tdef2) / 2.0 * (h2 - h1);
        }
        descending_ttrace.push(tp2);
        descending_ptrace.push(pe2);
        pe1 = pe2;
        te1 = te2;
        h1 = h2;
        tp1 = tp2;
    }

    // The historical object starts with the interpolated source point, then
    // appends the reported levels in descent order through the surface.
    let mut ttrace = Vec::with_capacity(descending_ttrace.len() + 1);
    let mut ptrace = Vec::with_capacity(descending_ptrace.len() + 1);
    ttrace.push(source_wetbulb);
    ptrace.push(source_pressure);
    ttrace.extend(descending_ttrace);
    ptrace.extend(descending_ptrace);

    cape::DcapeResult {
        dcape: total,
        ttrace,
        ptrace,
    }
}

/// SHARPpy parcel metadata that sharprs' otherwise-equivalent lift does not
/// reproduce literally.
///
/// In particular, SHARPpy's 1-hPa maximum-parcel-level loop never updates
/// its `h3` base height. That historical quirk materially changes MPL
/// pressure, and later LFC/EL crossings clear MPL pressure without clearing a
/// height already found. Release parity requires preserving both quirks. The
/// same pass records CAP/CAPPRES only on the LFC search branch, exactly as the
/// Python routine does. It intentionally does not replace CAPE/CIN or the
/// parcel trace.
fn sharppy_parcel_metadata(prof: &cape::Profile, lpl: &LiftedParcelLevel) -> (f64, f64, f64, f64) {
    const G: f64 = 9.80665;
    let mut cap = f64::NAN;
    let mut cappres = f64::NAN;
    let mut mplpres = f64::NAN;
    let mut mplhght = f64::NAN;

    if ![lpl.pres, lpl.tmpc, lpl.dwpc]
        .iter()
        .all(|value| value.is_finite())
    {
        return (cap, cappres, mplpres, mplhght);
    }

    let (lcl_pres, lcl_temp) = thermo::drylift(lpl.pres, lpl.tmpc, lpl.dwpc);
    if !lcl_pres.is_finite() || !lcl_temp.is_finite() {
        return (cap, cappres, mplpres, mplhght);
    }
    let mut pbot = prof.pres[prof.sfc].min(lpl.pres);
    if pbot > lcl_pres {
        pbot = lcl_pres;
    }
    let Some(lptr) = (prof.sfc..prof.pres.len()).find(|&i| pbot >= prof.pres[i]) else {
        return (cap, cappres, mplpres, mplhght);
    };

    let interp_hght = |p| interp_log_pressure(p, &prof.pres, &prof.hght);
    let interp_vtmp = |p| interp_log_pressure(p, &prof.pres, &prof.vtmp);
    let mut pe1 = pbot;
    let mut h1 = interp_hght(pe1);
    let mut te1 = interp_vtmp(pe1);
    let mut tp1 = thermo::wetlift(lcl_pres, lcl_temp, pe1);
    if ![h1, te1, tp1].iter().all(|value| value.is_finite()) {
        return (cap, cappres, mplpres, mplhght);
    }

    let mut lyre = 0.0;
    let mut tote = 0.0;
    let mut cap_strength = -9999.0_f64;
    let mut cap_strengthpres = -9999.0_f64;
    let mut elpres = f64::NAN;

    for i in lptr..prof.pres.len() {
        if !prof.tmpc[i].is_finite() {
            continue;
        }
        let pe2 = prof.pres[i];
        let h2 = prof.hght[i];
        let te2 = prof.vtmp[i];
        let tp2 = thermo::wetlift(pe1, tp1, pe2);
        if ![pe2, h2, te2, tp2].iter().all(|value| value.is_finite()) {
            continue;
        }
        let tdef1 = (thermo::virtemp(pe1, tp1, Some(tp1)) - te1) / (te1 + 273.15);
        let tdef2 = (thermo::virtemp(pe2, tp2, Some(tp2)) - te2) / (te2 + 273.15);
        let lyrlast = lyre;
        lyre = G * (tdef1 + tdef2) / 2.0 * (h2 - h1);
        if !lyre.is_finite() {
            continue;
        }

        let mli = thermo::virtemp(pe2, tp2, Some(tp2)) - te2;
        // Preserve the literal SHARPpy parcelx CAP diagnostic.  Although
        // ``-mli`` looks like the natural inversion strength, the reference
        // routine records ``te2 - mli`` and existing public CAP/CAPPRES values
        // depend on that historical expression.
        let mcap = te2 - mli;
        if mcap > cap_strength {
            cap_strength = mcap;
            cap_strengthpres = pe2;
        }
        tote += lyre;

        let pelast = pe1;
        pe1 = pe2;
        h1 = h2;
        te1 = te2;
        tp1 = tp2;

        // LFC possibility. The direct-crossing branch deliberately does not
        // save CAP/CAPPRES or reset `tote`; this is SHARPpy's behavior.
        if lyre >= 0.0 && lyrlast <= 0.0 {
            let parcel_at_previous = thermo::wetlift(pe1, tp1, pelast);
            if interp_vtmp(pelast)
                < thermo::virtemp(pelast, parcel_at_previous, Some(parcel_at_previous))
            {
                elpres = f64::NAN;
                mplpres = f64::NAN;
            } else {
                let mut crossing = pelast;
                let mut found = false;
                for _ in 0..400 {
                    if crossing <= 0.0 {
                        break;
                    }
                    let parcel = thermo::wetlift(pe1, tp1, crossing);
                    let env = interp_vtmp(crossing);
                    if !parcel.is_finite() || !env.is_finite() {
                        break;
                    }
                    if env <= thermo::virtemp(crossing, parcel, Some(parcel)) {
                        found = true;
                        break;
                    }
                    crossing -= 5.0;
                }
                if found && crossing > 0.0 {
                    tote = 0.0;
                    if cap_strength < 0.0 {
                        cap_strength = 0.0;
                    }
                    cap = cap_strength;
                    cappres = cap_strengthpres;
                    elpres = f64::NAN;
                    mplpres = f64::NAN;
                }
            }
        }

        // EL possibility.
        if lyre <= 0.0 && lyrlast >= 0.0 {
            let mut crossing = pelast;
            for _ in 0..400 {
                if crossing <= 0.0 {
                    break;
                }
                let parcel = thermo::wetlift(pe1, tp1, crossing);
                let env = interp_vtmp(crossing);
                if !parcel.is_finite() || !env.is_finite() {
                    crossing = f64::NAN;
                    break;
                }
                if env >= thermo::virtemp(crossing, parcel, Some(parcel)) {
                    break;
                }
                crossing -= 5.0;
            }
            elpres = crossing;
            mplpres = f64::NAN;
        }

        // MPL possibility, including the reference routine's intentionally
        // non-updated `h3` inside the 1-hPa loop.
        if tote < 0.0 && !mplpres.is_finite() && elpres.is_finite() {
            let mut pe3 = pelast;
            let h3 = interp_hght(pe3);
            let mut te3 = interp_vtmp(pe3);
            let mut tp3 = thermo::wetlift(pe1, tp1, pe3);
            let mut totx = tote - lyre;
            let mut pe2m = pelast;
            let mut valid = [h3, te3, tp3, totx].iter().all(|value| value.is_finite());
            for _ in 0..2000 {
                if !valid || totx <= 0.0 || pe2m <= 0.0 {
                    break;
                }
                pe2m -= 1.0;
                let te2m = interp_vtmp(pe2m);
                let tp2m = thermo::wetlift(pe3, tp3, pe2m);
                let h2m = interp_hght(pe2m);
                valid = [te2m, tp2m, h2m].iter().all(|value| value.is_finite());
                if !valid {
                    break;
                }
                let td3 = (thermo::virtemp(pe3, tp3, Some(tp3)) - te3) / (te3 + 273.15);
                let td2 = (thermo::virtemp(pe2m, tp2m, Some(tp2m)) - te2m) / (te2m + 273.15);
                // Do not update h3 here: that is the legacy SHARPpy quirk.
                let lyrf = G * (td3 + td2) / 2.0 * (h2m - h3);
                if !lyrf.is_finite() {
                    valid = false;
                    break;
                }
                totx += lyrf;
                tp3 = tp2m;
                te3 = te2m;
                pe3 = pe2m;
            }
            let height = interp_hght(pe2m);
            if valid && totx <= 0.0 && pe2m.is_finite() && height.is_finite() {
                mplpres = pe2m;
                mplhght = height - prof.hght[prof.sfc];
            }
        }
    }

    (cap, cappres, mplpres, mplhght)
}

/// Full SHARPpy parcel lift with literal legacy CAP/MPL metadata semantics.
pub fn parcelx_sharppy(
    prof: &cape::Profile,
    lpl: &LiftedParcelLevel,
    pbot: Option<f64>,
    ptop: Option<f64>,
) -> ParcelResult {
    let mut parcel = cape::parcelx(prof, lpl, pbot, ptop);
    if pbot.is_none() && ptop.is_none() {
        let (cap, cappres, mplpres, mplhght) = sharppy_parcel_metadata(prof, lpl);
        parcel.cap = cap;
        parcel.cappres = cappres;
        parcel.mplpres = mplpres;
        parcel.mplhght = mplhght;
    }
    parcel
}

/// SHARPpy-style forecast surface parcel: forecast max temperature (100-hPa
/// mixed layer warmed 2 K) with the mean boundary-layer mixing ratio.
/// sharprs's own `ParcelType::Forecast` uses the current surface temperature
/// instead; the original renderer's FCST row uses this definition.
pub fn forecast_parcel(inner: &Profile, cape_prof: &cape::Profile) -> ParcelResult {
    let pres = inner.pres[inner.sfc];
    let tmpc = indices::max_temp(inner, Some(100.0)).unwrap_or(f64::NAN);
    // SHARPpy's forecast parcel uses `mean_mixratio(..., exact=True)`, the
    // same exact observed-level average used by its 100-hPa mixed-layer
    // parcel.  Reuse sharprs' exact mixed-layer definition for the moisture
    // source instead of the 1-hPa fast mean in `indices::mean_mixratio`.
    let mixed = cape::define_parcel(cape_prof, ParcelType::MixedLayer { depth_hpa: 100.0 });
    let dwpc = mixed.dwpc;
    let lpl = LiftedParcelLevel {
        pres,
        tmpc,
        dwpc,
        parcel_type: ParcelType::UserDefined { pres, tmpc, dwpc },
    };
    parcelx_sharppy(cape_prof, &lpl, None, None)
}

/// Parcel-based Bunkers storm motion `(rstu, rstv, lstu, lstv)` in kts —
/// the Bunkers et al. 2014 method the original renderer uses (effective
/// inflow base to 65% of the MU parcel EL, pressure-weighted mean wind),
/// falling back to the non-parcel method (port of SHARPpy
/// `params.bunkers_storm_motion`).
pub fn bunkers_storm_motion(
    prof: &Profile,
    mupcl: &ParcelResult,
    pbot: f64,
) -> (f64, f64, f64, f64) {
    let d = 7.5 * 1.94384449; // 7.5 m/s deviation, in kts
    let mucape = mupcl.bplus;
    let muel = mupcl.elhght;
    let base = prof.to_agl(prof.interp_hght(pbot));
    if mucape > 100.0 && qc(muel) && qc(base) {
        let depth = muel - base;
        let htop = base + depth * (65.0 / 100.0);
        let ptop = prof.pres_at_height(prof.to_msl(htop));
        let (mnu, mnv) =
            winds::mean_wind(prof, pbot, ptop, -1.0, 0.0, 0.0).unwrap_or((f64::NAN, f64::NAN));
        let (sru, srv) = winds::wind_shear(prof, pbot, ptop).unwrap_or((f64::NAN, f64::NAN));
        let srmag = (sru * sru + srv * srv).sqrt();
        let uchg = d / srmag * srv;
        let vchg = d / srmag * sru;
        (mnu + uchg, mnv - vchg, mnu - uchg, mnv + vchg)
    } else {
        winds::non_parcel_bunkers_motion(prof).unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN))
    }
}

/// Storm-relative helicity (m2/s2) over `lower..upper` m AGL for storm motion
/// `(stu, stv)` kts — SHARPpy's exact path, tolerant of missing wind levels
/// and of the surface-boundary float round-trip that makes
/// `sharprs::winds::helicity` return an error on some soundings (the
/// height->pressure interpolation can land a hair above the surface
/// pressure; here the boundary is clamped back into the profile).
/// Returns `(total, positive, negative)`.
pub fn helicity(prof: &Profile, lower: f64, upper: f64, stu: f64, stv: f64) -> (f64, f64, f64) {
    let nan3 = (f64::NAN, f64::NAN, f64::NAN);
    if !qc(lower) || !qc(upper) || !qc(stu) || !qc(stv) {
        return nan3;
    }
    if lower == upper {
        return (0.0, 0.0, 0.0);
    }
    let sfc_pres = prof.pres[prof.sfc];
    let mut plower = prof.pres_at_height(prof.to_msl(lower));
    let pupper = prof.pres_at_height(prof.to_msl(upper));
    // Clamp the boundaries into the profile's pressure range (float
    // round-trips can put them an ulp outside it).
    if plower.is_finite() {
        plower = plower.min(sfc_pres);
    } else if lower.abs() < 1.0 {
        plower = sfc_pres;
    }
    if !qc(plower) || !qc(pupper) {
        return nan3;
    }
    let interp_uv = |p: f64| -> (f64, f64) {
        (
            prof.interp_by_pressure(&prof.u, p),
            prof.interp_by_pressure(&prof.v, p),
        )
    };
    let mut us: Vec<f64> = Vec::new();
    let mut vs: Vec<f64> = Vec::new();
    let (u1, v1) = interp_uv(plower);
    if qc(u1) && qc(v1) {
        us.push(u1);
        vs.push(v1);
    }
    for i in 0..prof.pres.len() {
        let p = prof.pres[i];
        if p.is_finite() && p < plower && p > pupper && qc(prof.u[i]) && qc(prof.v[i]) {
            us.push(prof.u[i]);
            vs.push(prof.v[i]);
        }
    }
    let (u2, v2) = interp_uv(pupper);
    if qc(u2) && qc(v2) {
        us.push(u2);
        vs.push(v2);
    }
    if us.len() < 2 {
        return nan3;
    }
    const KTS2MS: f64 = 0.514444;
    let mut phel = 0.0;
    let mut nhel = 0.0;
    for k in 0..us.len() - 1 {
        let sru0 = (us[k] - stu) * KTS2MS;
        let srv0 = (vs[k] - stv) * KTS2MS;
        let sru1 = (us[k + 1] - stu) * KTS2MS;
        let srv1 = (vs[k + 1] - stv) * KTS2MS;
        let layer = sru1 * srv0 - sru0 * srv1;
        if layer.is_nan() {
            continue;
        }
        if layer > 0.0 {
            phel += layer;
        } else if layer < 0.0 {
            nhel += layer;
        }
    }
    (phel + nhel, phel, nhel)
}

fn lapse_rate_agl_field(prof: &Profile, values: &[f64], lower_m: f64, upper_m: f64) -> f64 {
    let z1 = prof.to_msl(lower_m);
    let z2 = prof.to_msl(upper_m);
    let mut p1 = prof.pres_at_height(z1);
    let p2 = prof.pres_at_height(z2);
    if p1.is_finite() {
        p1 = p1.min(prof.pres[prof.sfc]);
    } else if lower_m.abs() < 1.0 {
        p1 = prof.pres[prof.sfc];
    }
    if !qc(p1) || !qc(p2) {
        return f64::NAN;
    }
    let tv1 = prof.interp_by_pressure(values, p1);
    let tv2 = prof.interp_by_pressure(values, p2);
    let dz = z2 - z1;
    if !qc(tv1) || !qc(tv2) || dz.abs() < 1.0 {
        return f64::NAN;
    }
    (tv2 - tv1) / dz * -1000.0
}

/// Virtual-temperature lapse rate (C/km) over an AGL layer, boundary-clamped
/// like [`helicity`]. This is the vendored SHARPpy `params.lapse_rate`
/// convention used by the 0–3 and 3–6 km diagnostics.
pub fn lapse_rate_agl(prof: &Profile, lower_m: f64, upper_m: f64) -> f64 {
    lapse_rate_agl_field(prof, &prof.vtmp, lower_m, upper_m)
}

/// Plain-temperature lapse rate used by sharpmod's displayed SFC–500 m and
/// SFC–1 km companion parameters.
pub fn temperature_lapse_rate_agl(prof: &Profile, lower_m: f64, upper_m: f64) -> f64 {
    lapse_rate_agl_field(prof, &prof.tmpc, lower_m, upper_m)
}
