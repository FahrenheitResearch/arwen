use rustwx_sounding::{NativeSounding, SoundingColumn, SoundingMetadata};
use sharprs::{
    params::{cape, composites, indices},
    winds,
};
use std::{env, error::Error, fs, path::PathBuf};

const KTS_TO_MS: f64 = 0.514_444_444_444_444_5;

fn main() -> Result<(), Box<dyn Error>> {
    let input = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(r"C:\weather-data\sounding_20110427-200600_lat32.98_lon-88.59.txt")
    });
    let text = fs::read_to_string(&input)?;
    let column = parse_sounding(&text)?;
    let sounding = NativeSounding::from_column(&column)?;

    let out_dir = PathBuf::from(r"C:\weather-data");
    fs::create_dir_all(&out_dir)?;
    let png_path = out_dir.join("sounding_20110427_200600_lat32.98_lon-88.59_rustwx.png");
    sounding.write_full_png(&png_path)?;
    let report_path = out_dir.join("sounding_20110427_200600_lat32.98_lon-88.59_rustwx_calcs.txt");

    let mut report = String::new();
    let p = &sounding.params;
    let e = &sounding.verified_ecape;

    line(&mut report, format_args!("input: {}", input.display()));
    line(&mut report, format_args!("png:   {}", png_path.display()));
    line(
        &mut report,
        format_args!("txt:   {}", report_path.display()),
    );
    blank(&mut report);

    line(
        &mut report,
        format_args!("Rustwx sounding table parcel rows:"),
    );
    line(
        &mut report,
        format_args!(
            "Parcel          ANALYTIC ECAPE  NCAPE   CAPE  3CAPE  6CAPE  CINH   LCL   LFC     EL"
        ),
    );
    append_parcel(
        &mut report,
        "Surface-Based",
        &e.surface_based,
        p.sfcpcl.lclhght,
    );
    append_parcel(&mut report, "Mixed-Layer", &e.mixed_layer, p.mlpcl.lclhght);
    append_parcel(
        &mut report,
        "Most-Unstable",
        &e.most_unstable,
        p.mupcl.lclhght,
    );
    blank(&mut report);

    line(&mut report, format_args!("Native sharprs parcel rows:"));
    line(
        &mut report,
        format_args!("Parcel              CAPE  3CAPE  6CAPE  CINH   LCL   LFC     EL"),
    );
    append_native_parcel(&mut report, "Surface-Based", &p.sfcpcl);
    append_native_parcel(&mut report, "Mixed-Layer", &p.mlpcl);
    append_native_parcel(&mut report, "Most-Unstable", &p.mupcl);
    blank(&mut report);

    line(&mut report, format_args!("Lapse rates:"));
    line(
        &mut report,
        format_args!("  Sfc-3km:      {}", fmt_lr(p.lr03)),
    );
    line(
        &mut report,
        format_args!("  3km-6km:      {}", fmt_lr(p.lr36)),
    );
    line(
        &mut report,
        format_args!("  Sfc-LCL:      {}", fmt_lr(sfc_lcl_lr(&sounding))),
    );
    line(
        &mut report,
        format_args!(
            "  950-850 mb:   {}",
            fmt_lr(indices::lapse_rate(&sounding.profile, 950.0, 850.0, true))
        ),
    );
    line(
        &mut report,
        format_args!("  850-500 mb:   {}", fmt_lr(p.lr85)),
    );
    line(
        &mut report,
        format_args!("  700-500 mb:   {}", fmt_lr(p.lr75)),
    );
    blank(&mut report);

    append_shear_table(&mut report, &sounding);
    blank(&mut report);
    append_storm_motions(&mut report, &sounding);
    blank(&mut report);
    append_misc_table(&mut report, &sounding);

    fs::write(&report_path, &report)?;
    print!("{report}");

    Ok(())
}

fn line(report: &mut String, args: std::fmt::Arguments<'_>) {
    use std::fmt::Write;
    writeln!(report, "{args}").expect("write to string");
}

fn blank(report: &mut String) {
    report.push('\n');
}

fn append_parcel(
    report: &mut String,
    label: &str,
    parcel: &rustwx_sounding::VerifiedEcapeParcelParams,
    lcl_m: f64,
) {
    line(
        report,
        format_args!(
            "{:<16} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5} {:>6}",
            label,
            fmt0(parcel.ecape),
            fmt0(parcel.ncape),
            fmt0(parcel.cape),
            fmt0(parcel.cape_3km),
            fmt0(parcel.cape_6km),
            fmt0(parcel.cinh),
            fmt0(lcl_m),
            fmt0(parcel.lfc_m),
            fmt0(parcel.el_m),
        ),
    );
}

fn append_native_parcel(report: &mut String, label: &str, parcel: &cape::ParcelResult) {
    line(
        report,
        format_args!(
            "{:<16} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5} {:>6}",
            label,
            fmt0(parcel.bplus),
            fmt0(parcel.b3km),
            fmt0(parcel.b6km),
            fmt0(parcel.bminus),
            fmt0(parcel.lclhght),
            fmt0(parcel.lfchght),
            fmt0(parcel.elhght),
        ),
    );
}

fn append_shear_table(report: &mut String, sounding: &NativeSounding) {
    let p = &sounding.params;
    let profile = &sounding.profile;
    let p_sfc = profile.sfc_pressure();
    let p500m = profile.pres_at_height(profile.to_msl(500.0));
    let p1km = profile.pres_at_height(profile.to_msl(1000.0));
    let p2km = profile.pres_at_height(profile.to_msl(2000.0));
    let p3km = profile.pres_at_height(profile.to_msl(3000.0));
    let p6km = profile.pres_at_height(profile.to_msl(6000.0));
    let eff_bot_h = pressure_to_agl(profile, p.eff_inflow.0);
    let eff_top_h = pressure_to_agl(profile, p.eff_inflow.1);

    line(report, format_args!("Shear / SRH / SR wind / EHI:"));
    line(
        report,
        format_args!(
            "Level       Layer m AGL      Shear kt   SRH m2/s2  SRWind kt   EHI   Mean kt"
        ),
    );
    append_shear_row(report, sounding, "Sfc-500m", p_sfc, p500m, 0.0, 500.0);
    append_shear_row(report, sounding, "Sfc-1km", p_sfc, p1km, 0.0, 1000.0);
    append_shear_row(
        report,
        sounding,
        "Eff Inflow",
        p.eff_inflow.0,
        p.eff_inflow.1,
        eff_bot_h,
        eff_top_h,
    );
    append_shear_row(report, sounding, "Sfc-3km", p_sfc, p3km, 0.0, 3000.0);
    append_shear_row(report, sounding, "1km-3km", p1km, p3km, 1000.0, 3000.0);
    append_shear_row(report, sounding, "3km-6km", p3km, p6km, 3000.0, 6000.0);
    append_shear_row(report, sounding, "Sfc-6km", p_sfc, p6km, 0.0, 6000.0);
    append_shear_row(report, sounding, "C 0-2km", p_sfc, p2km, 0.0, 2000.0);
}

fn append_shear_row(
    report: &mut String,
    sounding: &NativeSounding,
    label: &str,
    pbot: f64,
    ptop: f64,
    bottom_agl: f64,
    top_agl: f64,
) {
    let profile = &sounding.profile;
    let p = &sounding.params;
    let shear = winds::wind_shear(profile, pbot, ptop)
        .map(|(u, v)| sharprs::profile::comp2vec(u, v))
        .unwrap_or((f64::NAN, f64::NAN));
    let srh = winds::helicity(profile, bottom_agl, top_agl, p.rstu, p.rstv, -1.0, false)
        .map(|value| value.0)
        .unwrap_or(f64::NAN);
    let srw = winds::sr_wind(profile, pbot, ptop, p.rstu, p.rstv, -1.0)
        .map(|(u, v)| sharprs::profile::comp2vec(u, v))
        .unwrap_or((f64::NAN, f64::NAN));
    let mean = winds::mean_wind(profile, pbot, ptop, -1.0, 0.0, 0.0)
        .map(|(u, v)| sharprs::profile::comp2vec(u, v))
        .unwrap_or((f64::NAN, f64::NAN));
    let ehi = composites::ehi(p.sfcpcl.bplus, srh).unwrap_or(f64::NAN);

    line(
        report,
        format_args!(
            "{:<11} {:>5}-{:>5}    {:>9} {:>9}  {:>9} {:>5} {:>9}",
            label,
            fmt0(bottom_agl),
            fmt0(top_agl),
            fmt_dir_spd(shear.0, shear.1),
            fmt0(srh),
            fmt_dir_spd(srw.0, srw.1),
            fmt1(ehi),
            fmt_dir_spd(mean.0, mean.1),
        ),
    );
}

fn append_storm_motions(report: &mut String, sounding: &NativeSounding) {
    let p = &sounding.params;
    let (rm_dir, rm_spd) = sharprs::profile::comp2vec(p.rstu, p.rstv);
    let (lm_dir, lm_spd) = sharprs::profile::comp2vec(p.lstu, p.lstv);
    let (cu_dir, cu_spd) = sharprs::profile::comp2vec(p.corfidi_up_u, p.corfidi_up_v);
    let (cd_dir, cd_spd) = sharprs::profile::comp2vec(p.corfidi_dn_u, p.corfidi_dn_v);

    line(report, format_args!("Storm motions:"));
    line(
        report,
        format_args!("  Bunkers RM:    {} kt", fmt_dir_spd(rm_dir, rm_spd)),
    );
    line(
        report,
        format_args!("  Bunkers LM:    {} kt", fmt_dir_spd(lm_dir, lm_spd)),
    );
    line(
        report,
        format_args!("  Corfidi Down:  {} kt", fmt_dir_spd(cd_dir, cd_spd)),
    );
    line(
        report,
        format_args!("  Corfidi Up:    {} kt", fmt_dir_spd(cu_dir, cu_spd)),
    );
    line(
        report,
        format_args!(
            "  1km wind:      {} kt",
            fmt_dir_spd(p.wind_1km.0, p.wind_1km.1)
        ),
    );
    line(
        report,
        format_args!(
            "  6km wind:      {} kt",
            fmt_dir_spd(p.wind_6km.0, p.wind_6km.1)
        ),
    );
}

fn append_misc_table(report: &mut String, sounding: &NativeSounding) {
    let p = &sounding.params;
    let profile = &sounding.profile;
    let p1km = profile.pres_at_height(profile.to_msl(1000.0));
    let p3500m = profile.pres_at_height(profile.to_msl(3500.0));
    let p3km = profile.pres_at_height(profile.to_msl(3000.0));
    let p12km = profile.pres_at_height(profile.to_msl(12000.0));
    let (dgz_bot, dgz_top) = indices::dgz(profile);
    let dgz_rh = indices::mean_relh(profile, Some(dgz_bot), Some(dgz_top));
    let lcl_temp_c = cape::lcl(
        profile.sfc_pressure(),
        profile.tmpc[profile.sfc],
        profile.dwpc[profile.sfc],
    )
    .1;
    let mean_wind_1_35_ms = mean_wind_mag(profile, p1km, p3500m) * KTS_TO_MS;
    let wndg = composites::wndg(
        p.mlpcl.bplus,
        p.lr03.unwrap_or(f64::NAN),
        mean_wind_1_35_ms,
        p.mlpcl.bminus,
    );
    let shr06_mag = vector_mag(p.shr06.0, p.shr06.1);
    let mean06_mag = vector_mag(p.mean_wind_06.0, p.mean_wind_06.1);
    let dcp = composites::dcp(p.dcape.dcape, p.mupcl.bplus, shr06_mag, mean06_mag);
    let esp = composites::esp(p.mlpcl.b3km, p.lr03.unwrap_or(f64::NAN), p.mlpcl.bplus);
    let lr38 = indices::lapse_rate(profile, 3000.0, 8000.0, false).unwrap_or(f64::NAN);
    let mean_wind_3_12_ms = mean_wind_mag(profile, p3km, p12km) * KTS_TO_MS;
    let mmp = {
        let max_bulk_shear = max_bulk_shear_0_1_to_6_10_mps(profile);
        if max_bulk_shear.is_finite() && lr38.is_finite() && mean_wind_3_12_ms.is_finite() {
            Some(indices::coniglio(
                p.mupcl.bplus,
                max_bulk_shear,
                lr38,
                mean_wind_3_12_ms,
            ))
        } else {
            None
        }
    };

    line(report, format_args!("Other table values:"));
    line(
        report,
        format_args!("  PWAT:              {}", fmt_unit(p.precip_water, "in", 2)),
    );
    line(
        report,
        format_args!("  Mean Mix Rat:      {}", fmt_unit(p.mean_mixr, "g/kg", 2)),
    );
    line(
        report,
        format_args!(
            "  Sfc RH:            {}",
            fmt_pct(profile.relh.get(profile.sfc).copied())
        ),
    );
    line(
        report,
        format_args!("  Low RH:            {}", fmt_pct(p.mean_rh_low)),
    );
    line(
        report,
        format_args!("  Mid RH:            {}", fmt_pct(p.mean_rh_mid)),
    );
    line(
        report,
        format_args!("  Avg DGZ RH:        {}", fmt_pct(dgz_rh)),
    );
    line(
        report,
        format_args!("  Freezing Level:    {}", fmt_unit(p.frz_lvl, "m", 0)),
    );
    line(
        report,
        format_args!("  WB Zero:           {}", fmt_unit(p.wb_zero, "m", 0)),
    );
    line(
        report,
        format_args!("  MU MPL:            {:.0} m", p.mupcl.mplhght),
    );
    line(
        report,
        format_args!("  3km Theta Diff:    {}", fmt_unit(p.tei, "K", 0)),
    );
    line(
        report,
        format_args!("  LCL Temp:          {:.1} C", lcl_temp_c),
    );
    line(
        report,
        format_args!("  ConvT:             {}", fmt_unit(p.conv_t, "C", 1)),
    );
    line(
        report,
        format_args!("  MaxT:              {}", fmt_unit(p.max_temp, "C", 1)),
    );
    line(
        report,
        format_args!("  K Index:           {}", fmt_unit(p.k_index, "", 1)),
    );
    line(
        report,
        format_args!("  T Totals:          {}", fmt_unit(p.t_totals, "", 1)),
    );
    line(
        report,
        format_args!("  C Totals:          {}", fmt_unit(p.c_totals, "", 1)),
    );
    line(
        report,
        format_args!("  V Totals:          {}", fmt_unit(p.v_totals, "", 1)),
    );
    line(
        report,
        format_args!("  EHI 0-1km:         {}", fmt_unit(p.ehi01, "", 1)),
    );
    line(
        report,
        format_args!("  EHI 0-3km:         {}", fmt_unit(p.ehi03, "", 1)),
    );
    line(
        report,
        format_args!("  Supercell Comp:    {}", fmt_unit(p.scp, "", 1)),
    );
    line(
        report,
        format_args!("  STP cin:           {}", fmt_unit(p.stp_cin, "", 1)),
    );
    line(
        report,
        format_args!("  STP fixed:         {}", fmt_unit(p.stp_fixed, "", 1)),
    );
    line(
        report,
        format_args!("  SHIP:              {}", fmt_unit(p.ship, "", 1)),
    );
    line(
        report,
        format_args!("  DCP:               {}", fmt_unit(dcp, "", 1)),
    );
    line(
        report,
        format_args!("  WNDG:              {}", fmt_unit(wndg, "", 1)),
    );
    line(
        report,
        format_args!("  DCAPE:             {:.0} J/kg", p.dcape.dcape),
    );
    line(
        report,
        format_args!(
            "  DownT:             {}",
            fmt1(p.dcape.ttrace.last().copied().unwrap_or(f64::NAN))
        ),
    );
    line(
        report,
        format_args!("  MMP:               {}", fmt_unit(mmp, "", 2)),
    );
    line(
        report,
        format_args!("  ESP:               {}", fmt_unit(esp, "", 1)),
    );
}

fn parse_sounding(text: &str) -> Result<SoundingColumn, Box<dyn Error>> {
    let mut in_raw = false;
    let mut pressure_hpa = Vec::new();
    let mut height_m_msl = Vec::new();
    let mut temperature_c = Vec::new();
    let mut dewpoint_c = Vec::new();
    let mut u_ms = Vec::new();
    let mut v_ms = Vec::new();
    let mut omega_pa_s = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "%RAW%" {
            in_raw = true;
            continue;
        }
        if !in_raw || trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }

        let values: Vec<f64> = trimmed
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::parse::<f64>)
            .collect::<Result<_, _>>()?;
        if values.len() < 6 {
            continue;
        }

        let pressure = values[0];
        let height = values[1];
        let temp = values[2];
        let dewpoint = values[3].min(temp);
        let wind_dir_deg = values[4];
        let wind_speed_ms = values[5] * KTS_TO_MS;
        let dir_rad = wind_dir_deg.to_radians();

        pressure_hpa.push(pressure);
        height_m_msl.push(height);
        temperature_c.push(temp);
        dewpoint_c.push(dewpoint);
        u_ms.push(-wind_speed_ms * dir_rad.sin());
        v_ms.push(-wind_speed_ms * dir_rad.cos());
        if let Some(omega) = values.get(6) {
            omega_pa_s.push(*omega);
        }
    }

    Ok(SoundingColumn {
        pressure_hpa,
        height_m_msl,
        temperature_c,
        dewpoint_c,
        u_ms,
        v_ms,
        omega_pa_s,
        metadata: SoundingMetadata {
            station_id: "WRF-ARW".to_string(),
            valid_time: "2011-04-27T20:06:00Z".to_string(),
            latitude_deg: Some(32.98),
            longitude_deg: Some(-88.59),
            elevation_m: Some(0.0),
            sample_method: Some("raw text sounding".to_string()),
            box_radius_lat_deg: None,
            box_radius_lon_deg: None,
        },
    })
}

fn sfc_lcl_lr(sounding: &NativeSounding) -> Option<f64> {
    let lcl = sounding.params.sfcpcl.lclhght;
    if !lcl.is_finite() || lcl <= 1.0 {
        return None;
    }
    let sfc_temp = sounding.profile.tmpc[sounding.profile.sfc];
    let lcl_pres = sounding.params.sfcpcl.lclpres;
    let lcl_temp = sounding.profile.interp_tmpc(lcl_pres);
    let lr = (sfc_temp - lcl_temp) / lcl * 1000.0;
    lr.is_finite().then_some(lr)
}

fn pressure_to_agl(profile: &rustwx_sounding::SharprsProfile, pressure_hpa: f64) -> f64 {
    if !pressure_hpa.is_finite() {
        return f64::NAN;
    }
    let height = profile.interp_hght(pressure_hpa);
    if height.is_finite() {
        profile.to_agl(height)
    } else {
        f64::NAN
    }
}

fn vector_mag(u: f64, v: f64) -> f64 {
    (u * u + v * v).sqrt()
}

fn mean_wind_mag(profile: &rustwx_sounding::SharprsProfile, pbot: f64, ptop: f64) -> f64 {
    winds::mean_wind(profile, pbot, ptop, -1.0, 0.0, 0.0)
        .map(|(u, v)| vector_mag(u, v))
        .unwrap_or(f64::NAN)
}

fn max_bulk_shear_0_1_to_6_10_mps(profile: &rustwx_sounding::SharprsProfile) -> f64 {
    let low = wind_indices_in_layer(profile, 0.0, 1000.0);
    let high = wind_indices_in_layer(profile, 6000.0, 10_000.0);
    let mut max_shear = f64::NAN;
    for &i in &low {
        for &j in &high {
            let du = profile.u[j] - profile.u[i];
            let dv = profile.v[j] - profile.v[i];
            if du.is_finite() && dv.is_finite() {
                let shear = vector_mag(du, dv) * KTS_TO_MS;
                if !max_shear.is_finite() || shear > max_shear {
                    max_shear = shear;
                }
            }
        }
    }
    max_shear
}

fn wind_indices_in_layer(
    profile: &rustwx_sounding::SharprsProfile,
    bottom_agl: f64,
    top_agl: f64,
) -> Vec<usize> {
    profile
        .hght
        .iter()
        .enumerate()
        .filter_map(|(index, &height_msl)| {
            let agl = profile.to_agl(height_msl);
            if agl >= bottom_agl
                && agl <= top_agl
                && profile.u[index].is_finite()
                && profile.v[index].is_finite()
            {
                Some(index)
            } else {
                None
            }
        })
        .collect()
}

fn fmt0(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.0}")
    } else {
        "NA".to_string()
    }
}

fn fmt1(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.1}")
    } else {
        "NA".to_string()
    }
}

fn fmt_dir_spd(direction: f64, speed: f64) -> String {
    if direction.is_finite() && speed.is_finite() {
        format!("{direction:.0}/{speed:.0}")
    } else {
        "NA".to_string()
    }
}

fn fmt_lr(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.1} C/km"))
        .unwrap_or_else(|| "NA".to_string())
}

fn fmt_unit(value: Option<f64>, unit: &str, digits: usize) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| {
            if unit.is_empty() {
                format!("{v:.digits$}")
            } else {
                format!("{v:.digits$} {unit}")
            }
        })
        .unwrap_or_else(|| "NA".to_string())
}

fn fmt_pct(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.0}%"))
        .unwrap_or_else(|| "NA".to_string())
}
