//! CfRadial 1.x decoder (classic-netCDF radar moments).
//!
//! Format reference: M. Dixon and W.-C. Lee, "CfRadial Data File Format —
//! CF-compliant netCDF Format for Moments Data for RADAR and LIDAR",
//! NCAR/EOL, version 1.4 (2016) (versions 1.1–1.4 share the layout read
//! here). CfRadial 1 files are classic netCDF (`CDF\x01`/`CDF\x02`) with:
//! - dimensions `time` (rays, usually unlimited) and `range` (gates),
//! - per-ray `azimuth(time)`, `elevation(time)`, `time(time)` and the
//!   instrument variables `nyquist_velocity`, `unambiguous_range`, `prt`,
//!   `prt_ratio`, `n_samples`, `pulse_width`, `scan_rate`,
//!   `antenna_transition`, `r_calib_index` (all optional),
//! - per-sweep `fixed_angle(sweep)`, `sweep_start_ray_index(sweep)`,
//!   `sweep_end_ray_index(sweep)`, `sweep_mode(sweep, string_length)` and
//!   the other Table 301-8a string variables,
//! - scalar `latitude`/`longitude`/`altitude` (per ray for moving
//!   platforms), `time_coverage_start`, `volume_number`, `platform_type`,
//! - field variables dimensioned `(time, range)`, optionally packed with
//!   `scale_factor`/`add_offset` and flagged with `_FillValue`
//!   (CF packing: physical = raw * scale_factor + add_offset), or with
//!   `n_gates_vary = "true"` (section 2.3.1) dimensioned `(n_points)`, each
//!   ray's gates located by `ray_start_index(time)` and `ray_n_gates(time)`,
//! - the gate centres in `range(range)`, or per sweep in `range(sweep,
//!   range)` (section 4.4), or per ray in `range(time, range)` (as LROSE Radx
//!   writes a volume whose geometry varies), and per ray in
//!   `ray_start_range(time)` / `ray_gate_spacing(time)`.
//!
//! CfRadial 2 (one group per sweep) is [`crate::cfradial2`]'s.
//!
//! [`read_cfradial1_volume`] builds the FM301 model ([`Volume`]; design note
//! `docs/design/fm301-model.md` sections 7.3, 9 and 11) the way xradar's
//! `open_cfradial1_datatree` and Py-ART's `read_cfradial` read the same
//! file: one sweep per `sweep` index in file order, field variables under
//! their names verbatim in file order, packed `byte`/`short`/`int` fields kept
//! as `i8`/`i16`/`i32` with the file's `scale_factor`, `add_offset` and `_FillValue`
//! (attribute width preserved), float fields verbatim with their fill, the
//! `range` coordinate as the file's gate centres (each sweep its own: its
//! row of a two-dimensional `range`, or `ray_start_range` /
//! `ray_gate_spacing` where they state one geometry for the sweep that
//! differs from `range(range)`; rays of one sweep with different geometries
//! are [`CfRadialError::PerRayGeometry`]), `n_points` storage laid out as
//! rows of the sweep's longest `ray_n_gates`, each ray padded after its own
//! with the field's `_FillValue` (else the netCDF default fill, which then
//! becomes the field's fill), `time(time)` as seconds
//! since the `time.units` reference (else `time_coverage_start`), and every
//! variable without a typed slot kept verbatim in `extra_vars`: per-ray ones
//! in their sweeps, `(sweep, ...)` ones as a scalar of each sweep, anything
//! else at the root. A file `sweep_number` that is not the sweep's 0-based
//! index is kept in `Sweep::other`; a `missing_value` that is not the fill
//! code stays among the field's attributes. The attributes of every variable
//! a typed slot holds (`azimuth:comment`, `range:meters_between_gates`, ...)
//! are kept in `Volume::variable_attrs`; a global attribute whose typed slot
//! cannot hold its value (a flag that is not a boolean, a coverage time that
//! does not parse or differs from the variable) stays in `attrs.other`.
//! Integer attributes keep their stored type. A netCDF-4 file's
//! `_NCProperties` joins the global attributes and `provenance.compression`
//! says `cfradial1-netcdf4` (else `cfradial1-netcdf3`). The BowEcho export attributes (`vcp_*`, `polarization`,
//! `calibration`, `forward_operator`, ...) fill `ScanStrategy`, `Provenance`
//! and `SimulationProvenance`; sweeps whose `vcp_moment_coverage_code`
//! restricts their moments get only those fields. For RHI sweeps the fixed
//! angle is the AZIMUTH (CfRadial §5.8).

use std::borrow::Cow;

use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use recast_radar_core::bounded_read::{DecodeBudget, check_gate_count, check_sweep_count};
use recast_radar_core::model::{
    ArrayBuf, AttrValue, ExtraVariable, Field, FieldData, FieldName, FloatCoding, FloatWidth,
    FollowMode, GateMapping, InstrumentType, IntCoding, LinearTransform, PlatformTrack,
    PlatformType, PolarizationMode, PrimaryAxis, PrtMode, Quantity, RadarCalibration,
    RadarParameters, RangeCoord, Scalar, ScanDefinition, ScanLeg, SimulationProvenance,
    SourceFormat, Sweep, SweepMode, VariableAttrs, Volume,
};
use recast_radar_hdf5::netcdf4::NcFile as Nc4File;

use crate::netcdf::NcFile;
pub use crate::netcdf3::looks_like_netcdf3_bytes;
use crate::netcdf3::{IntKind, Nc3File, NcArray, NcValue, NcVar};
use crate::{CfRadialError, Result};

/// Decode a CfRadial 1.x byte buffer (classic netCDF, or netCDF-4) into the
/// FM301 model.
pub fn read_cfradial1_volume(bytes: &[u8]) -> Result<Volume> {
    decode(bytes, DecodeBudget::volume())
}

/// [`read_cfradial1_volume`] with an explicit output budget.
#[cfg(test)]
fn read_cfradial1_volume_within(bytes: &[u8], budget: DecodeBudget) -> Result<Volume> {
    decode(bytes, budget)
}

/// Global attributes with a typed slot in the model; every other global
/// attribute goes to `GlobalAttrs::other` verbatim.
pub(crate) const SLOTTED_GLOBAL_ATTRS: &[&str] = &[
    "Conventions",
    "version",
    "title",
    "institution",
    "references",
    "source",
    "history",
    "comment",
    "instrument_name",
    "site_name",
    "scan_name",
    "scan_id",
    "platform_is_mobile",
    "ray_times_increase",
    "simulated",
    "vcp_pattern",
    "vcp_source_document",
    "vcp_source_revision",
    "vcp_source_rda_build",
    "vcp_source_figure",
    "vcp_pulse_length",
    "vcp_adaptations",
    "polarization",
    "calibration",
    "forward_operator",
    "forward_operator_config",
    "source_model",
    "microphysics_scheme",
    "scattering_model",
    "time_coverage_start",
    "time_coverage_end",
];

/// Per-ray variables with a typed slot; the rest of the `(time)` variables
/// go to `Sweep::extra_vars`.
pub(crate) const SLOTTED_RAY_VARS: &[&str] = &[
    "time",
    "azimuth",
    "elevation",
    "nyquist_velocity",
    "unambiguous_range",
    "prt",
    "prt_ratio",
    "n_samples",
    "pulse_count",
    "pulse_width",
    "scan_rate",
    "antenna_transition",
    "r_calib_index",
    "calib_index",
    "independent_samples",
    "latitude",
    "longitude",
    "altitude",
    "altitude_agl",
    "heading",
    "roll",
    "pitch",
    "drift",
    "rotation",
    "tilt",
    "measured_transmit_power_h",
    "measured_transmit_power_v",
    "radar_measured_transmit_power_h",
    "radar_measured_transmit_power_v",
];

/// Root variables with a typed slot; other scalar root variables go to
/// `Volume::extra_vars`.
pub(crate) const SLOTTED_ROOT_VARS: &[&str] = &[
    "volume_number",
    "platform_type",
    "instrument_type",
    "primary_axis",
    "status_str",
    "time_coverage_start",
    "time_coverage_end",
    "latitude",
    "longitude",
    "altitude",
    "altitude_agl",
    "frequency",
    "radar_antenna_gain_h",
    "radar_antenna_gain_v",
    "radar_beam_width_h",
    "radar_beam_width_v",
    "radar_rx_bandwidth",
    "radar_receiver_bandwidth",
];

pub(crate) fn decode(bytes: &[u8], budget: DecodeBudget) -> Result<Volume> {
    if recast_radar_hdf5::looks_like_hdf5_bytes(bytes) {
        let file = Nc4File::open(bytes)?;
        return decode_netcdf4(&file, budget);
    }
    let file = NcFile::classic(Nc3File::open(bytes)?);
    decode_file(&file, budget)
}

/// Decode the root group of a netCDF-4 CfRadial 1.x file.
pub(crate) fn decode_netcdf4(file: &Nc4File<'_>, budget: DecodeBudget) -> Result<Volume> {
    let root = NcFile::netcdf4_group(file, "/")?;
    let mut volume = decode_file(&root, budget)?;
    note_netcdf4_container(&mut volume, file, "cfradial1-netcdf4");
    Ok(volume)
}

/// Record a netCDF-4 container: `compression` names it and the root
/// `_NCProperties` (writer library versions) joins the global attributes.
pub(crate) fn note_netcdf4_container(volume: &mut Volume, file: &Nc4File<'_>, container: &str) {
    volume.provenance.compression = Some(container.to_owned());
    if let Some(properties) = file.nc_properties() {
        volume
            .attrs
            .other
            .push(("_NCProperties".into(), AttrValue::Text(properties.into())));
    }
}

/// Decode a CfRadial 1.x file (classic or the root group of a netCDF-4
/// file).
pub(crate) fn decode_file(file: &NcFile<'_>, mut budget: DecodeBudget) -> Result<Volume> {
    let dim = |name: &str| file.dims.iter().position(|(dim_name, _)| dim_name == name);
    let (Some(time_dim), Some(range_dim)) = (dim("time"), dim("range")) else {
        return Err(invalid(
            "netCDF file lacks time/range dimensions — not CfRadial 1.x",
        ));
    };
    let n_rays = file.dims[time_dim].1;
    let n_gates = file.dims[range_dim].1;
    if n_rays == 0 || n_gates == 0 {
        return Err(invalid("CfRadial volume has no rays or gates"));
    }
    check_gate_count(n_gates, "CfRadial range dimension").map_err(CfRadialError::LimitExceeded)?;
    let ngates = u32::try_from(n_gates).map_err(|_| invalid("CfRadial range overflow"))?;

    let rays = RayData::read(file, time_dim, n_rays, &mut budget)?;

    // `n_gates_vary` storage (CfRadial 1.4 section 2.3.1): fields over
    // `n_points`, located by `ray_start_index` and `ray_n_gates`.
    let points_dim = dim("n_points");
    let ragged = match points_dim {
        Some(points_dim) => Some(Ragged::read(
            file,
            time_dim,
            n_rays,
            file.dims[points_dim].1,
            n_gates,
            &mut budget,
        )?),
        None => None,
    };

    // Gate geometry (CfRadial 1.4 section 4.4): range(range) gate centres
    // in metres for the whole volume, range(sweep, range) when the geometry
    // varies from sweep to sweep, or LROSE Radx's range(time, range).
    let sweep_dim = dim("sweep");
    let range_rows = match file.vars.get("range").map(|var| var.dim_ids.as_slice()) {
        Some([r]) if *r == range_dim => RangeRows::Volume,
        Some([s, r]) if Some(*s) == sweep_dim && *r == range_dim => RangeRows::Sweep,
        Some([t, r]) if *t == time_dim && *r == range_dim => RangeRows::Ray,
        Some(_) => {
            return Err(invalid(
                "CfRadial range variable is not range(range), range(sweep, range) or range(time, range)",
            ));
        }
        None => return Err(invalid("CfRadial file lacks the range variable")),
    };
    let range = read_f64s(file, "range", &mut budget)?;
    if range.len() < 2 || range.len() % n_gates != 0 {
        return Err(invalid("range coordinate needs at least two gates"));
    }
    let range_coord = range_coordinate(&range[..n_gates.min(range.len())], ngates);
    let ray_start_range =
        aligned_time_values(file, "ray_start_range", time_dim, n_rays, &mut budget)?;
    let ray_gate_spacing =
        aligned_time_values(file, "ray_gate_spacing", time_dim, n_rays, &mut budget)?;

    // Sweep index ranges; a missing sweep dimension means one sweep.
    let fixed_angles = optional_f64s(file, "fixed_angle", &mut budget)?.unwrap_or_default();
    check_sweep_count(fixed_angles.len(), "CfRadial fixed_angle")
        .map_err(CfRadialError::LimitExceeded)?;
    let sweep_starts =
        optional_f64s(file, "sweep_start_ray_index", &mut budget)?.unwrap_or_default();
    let sweep_ends = optional_f64s(file, "sweep_end_ray_index", &mut budget)?.unwrap_or_default();
    let sweep_count = fixed_angles.len().max(1);
    let sweep_modes = read_sweep_strings(file, "sweep_mode", sweep_count);
    let follow_modes = read_sweep_strings(file, "follow_mode", sweep_count);
    let prt_modes = read_sweep_strings(file, "prt_mode", sweep_count);
    let polarization_modes = read_sweep_strings(file, "polarization_mode", sweep_count);
    let target_scan_rates = optional_f64s(file, "target_scan_rate", &mut budget)?;
    let rays_are_indexed = read_sweep_strings(file, "rays_are_indexed", sweep_count);
    let ray_angle_res = optional_f64s(file, "ray_angle_res", &mut budget)?;

    // Time: seconds since the `time.units` reference, else since
    // `time_coverage_start`.
    let coverage_start = parse_time_var_or_attr(file, "time_coverage_start");
    let coverage_end = parse_time_var_or_attr(file, "time_coverage_end");
    let time_reference = time_units_reference(file)
        .or(coverage_start)
        .unwrap_or(DateTime::<Utc>::UNIX_EPOCH);

    let mut volume = describe_volume(
        file,
        Some(time_dim),
        time_reference,
        coverage_start,
        coverage_end,
    );
    volume.radar_parameters = radar_parameters(file);
    volume.radar_calibration = read_radar_calibration(file, time_reference);

    // Scan legs (BowEcho catalog-backed synthetic volumes) join the scan
    // definition `describe_volume` read.
    let mut definition = volume
        .scan
        .definition
        .take()
        .map(|definition| *definition)
        .unwrap_or_default();

    // Scan legs (BowEcho catalog-backed synthetic volumes).
    let source_row_indices = optional_f64s(file, "vcp_source_row_index", &mut budget)?;
    let vcp_azimuth_rates = optional_f64s(file, "vcp_azimuth_rate", &mut budget)?;
    let vcp_source_periods = optional_f64s(file, "vcp_source_period", &mut budget)?;
    let vcp_waveform_codes = optional_f64s(file, "vcp_waveform_code", &mut budget)?;
    let vcp_moment_coverage_codes = optional_f64s(file, "vcp_moment_coverage_code", &mut budget)?;
    let surveillance_prf_codes = optional_f64s(file, "vcp_surveillance_prf_code", &mut budget)?;
    let surveillance_pulse_counts =
        optional_f64s(file, "vcp_surveillance_pulse_count", &mut budget)?;
    let doppler_prf_codes = optional_f64s(file, "vcp_doppler_prf_code", &mut budget)?;
    let doppler_pulse_counts = optional_f64s(file, "vcp_doppler_pulse_count", &mut budget)?;
    let has_scan_leg_metadata = source_row_indices.is_some()
        || vcp_azimuth_rates.is_some()
        || vcp_source_periods.is_some()
        || vcp_waveform_codes.is_some()
        || vcp_moment_coverage_codes.is_some()
        || surveillance_prf_codes.is_some()
        || surveillance_pulse_counts.is_some()
        || doppler_prf_codes.is_some()
        || doppler_pulse_counts.is_some();

    // Field variables: anything shaped (time, range), or (n_points) in
    // `n_gates_vary` storage, in file order.
    let is_points = |var: &NcVar| {
        ragged.is_some()
            && points_dim.is_some()
            && var.dim_ids.as_slice() == [points_dim.unwrap_or(usize::MAX)]
    };
    let mut fields: Vec<&NcVar> = file
        .vars
        .values()
        .filter(|var| {
            var.name != "range"
                && (var.dim_ids.as_slice() == [time_dim, range_dim] || is_points(var))
        })
        .collect();
    fields.sort_by_key(|var| var.index);
    if fields.is_empty() {
        return Err(invalid("CfRadial volume has no (time, range) fields"));
    }
    // Per-ray variables without a typed slot, in file order.
    let mut extra_ray_vars: Vec<&NcVar> = file
        .vars
        .values()
        .filter(|var| {
            var.dim_ids.first() == Some(&time_dim)
                && var.dim_ids.len() <= 2
                && var.dim_ids.as_slice() != [time_dim, range_dim]
                && !SLOTTED_RAY_VARS.contains(&var.name.as_str())
        })
        .collect();
    extra_ray_vars.sort_by_key(|var| var.index);
    // Root variables without a typed slot, verbatim: everything that is not
    // a field, a per-ray or per-sweep variable (those go to the sweeps), a
    // calibration entry, the range coordinate or the frequency coordinate.
    let mut extra_root_vars: Vec<&NcVar> = file
        .vars
        .values()
        .filter(|var| {
            let first = var.dim_ids.first().copied();
            first != Some(time_dim)
                && (first != sweep_dim || first.is_none())
                && var.dim_ids.as_slice() != [range_dim]
                && !is_points(var)
                && !(var.name.starts_with("r_calib_") && file.dim_name(var, 0) == Some("r_calib"))
                && !(var.name == "frequency" && file.dim_name(var, 0) == Some("frequency"))
                && !SLOTTED_ROOT_VARS.contains(&var.name.as_str())
        })
        .collect();
    extra_root_vars.sort_by_key(|var| var.index);
    for var in extra_root_vars {
        if let Some(extra) = extra_variable(file, var, None) {
            volume.extra_vars.push(extra);
        }
    }
    // Per-sweep variables without a typed slot: one scalar per sweep.
    let mut extra_sweep_vars: Vec<&NcVar> = file
        .vars
        .values()
        .filter(|var| {
            sweep_dim.is_some()
                && var.dim_ids.first().copied() == sweep_dim
                && var.name != "range"
                && !SLOTTED_SWEEP_VARS.contains(&var.name.as_str())
        })
        .collect();
    extra_sweep_vars.sort_by_key(|var| var.index);
    let sweep_numbers = optional_f64s(file, "sweep_number", &mut budget)?;

    // Validate every sweep's ray range before building anything. Sweeps that
    // share rays would each copy the same field rows, so a header claiming
    // many overlapping sweeps multiplied the decoded size (fuzz regression
    // `fuzz-cfradial-overlapping-sweep-ray-ranges`).
    let ray_ranges: Vec<Option<(usize, usize)>> = (0..sweep_count)
        .map(|sweep| sweep_ray_range(&sweep_starts, &sweep_ends, sweep, n_rays))
        .collect();
    check_disjoint_sweeps(&ray_ranges)?;

    // Build sweep geometry first, then read each full (time, range) field
    // once and distribute its rows across every sweep.
    let mut sweeps: Vec<SweepBuild> = Vec::with_capacity(sweep_count);
    for (index, ray_range) in ray_ranges.into_iter().enumerate() {
        let Some((start_ray, end_ray)) = ray_range else {
            volume.provenance.decode.skipped_message_count += 1;
            continue;
        };
        let mode = sweep_modes
            .get(index)
            .cloned()
            .flatten()
            .map(|text| SweepMode::parse(&text))
            .unwrap_or(SweepMode::AzimuthSurveillance);
        let fixed = fixed_angles.get(index).copied().unwrap_or_else(|| {
            fallback_fixed_angle(
                matches!(mode, SweepMode::Rhi | SweepMode::ManualRhi),
                &rays.azimuth[start_ray..=end_ray],
                &rays.elevation[start_ray..=end_ray],
            )
        }) as f32;
        let sweep_rays = end_ray - start_ray + 1;
        budget
            .charge(sweep_rays, 16 * size_of::<f64>(), "CfRadial sweep rays")
            .map_err(CfRadialError::LimitExceeded)?;
        let mut sweep = Sweep::new(sweeps.len() as u32, mode, fixed);
        sweep.elevation_number = u16::try_from(index).ok();
        sweep.follow_mode = follow_modes
            .get(index)
            .cloned()
            .flatten()
            .map(|text| FollowMode::parse(&text));
        sweep.prt_mode = prt_modes
            .get(index)
            .cloned()
            .flatten()
            .map(|text| PrtMode::parse(&text));
        sweep.polarization_mode = polarization_modes
            .get(index)
            .cloned()
            .flatten()
            .map(|text| PolarizationMode::parse(&text));
        sweep.target_scan_rate_deg_per_s = numeric_f32_at(&target_scan_rates, index);
        sweep.rays_are_indexed = rays_are_indexed
            .get(index)
            .cloned()
            .flatten()
            .and_then(|text| parse_bool(&text));
        sweep.rays_angle_resolution_deg = numeric_f32_at(&ray_angle_res, index);
        let sweep_ngates = match &ragged {
            Some(ragged) => ragged.sweep_gates(start_ray..=end_ray),
            None => n_gates,
        };
        if sweep_ngates == 0 {
            return Err(invalid(format!("CfRadial sweep {index} has no gates")));
        }
        let row = |row: usize| range.get(row * n_gates..(row + 1) * n_gates).unwrap_or(&[]);
        sweep.range = match range_rows {
            RangeRows::Volume if sweep_ngates == n_gates => range_coord.clone(),
            RangeRows::Volume => truncated(&range_coord, sweep_ngates),
            RangeRows::Sweep => row_coordinate(row(index), sweep_ngates)?,
            RangeRows::Ray => {
                let first = row(start_ray);
                if (start_ray..=end_ray).any(|ray| !same_centres(row(ray), first, sweep_ngates)) {
                    return Err(CfRadialError::PerRayGeometry { sweep: index });
                }
                row_coordinate(first, sweep_ngates)?
            }
        };
        if range_rows == RangeRows::Volume
            && let Some((first_center_m, spacing_m)) = ray_geometry(
                ray_start_range.as_deref(),
                ray_gate_spacing.as_deref(),
                start_ray..=end_ray,
                &sweep.range,
            )
            .map_err(|()| CfRadialError::PerRayGeometry { sweep: index })?
        {
            sweep.range = RangeCoord::Uniform {
                first_center_m,
                spacing_m,
                ngates: sweep_ngates as u32,
            };
        }
        let ray_range = start_ray..=end_ray;
        rays.fill(&mut sweep, ray_range.clone(), 0.0);
        if let Some(number) = numeric_at(&sweep_numbers, index)
            && number != sweep.sweep_number as f64
        {
            // The file's own sweep number, when it is not the 0-based index
            // the model numbers sweeps by.
            sweep.other.push((
                "sweep_number".into(),
                AttrValue::Scalar(Scalar::F64(number)),
            ));
        }
        for var in &extra_sweep_vars {
            if let Some(extra) = sweep_scalar_variable(file, var, index) {
                sweep.extra_vars.push(extra);
            }
        }
        for var in &extra_ray_vars {
            if let Some(extra) = extra_variable(file, var, Some(ray_range.clone())) {
                sweep.extra_vars.push(extra);
            }
        }
        let leg = ScanLeg {
            source_row_index: numeric_u16_at(&source_row_indices, index),
            elevation_deg: has_scan_leg_metadata.then_some(fixed),
            azimuth_rate_deg_per_second: numeric_f32_at(&vcp_azimuth_rates, index),
            source_period_seconds: numeric_f32_at(&vcp_source_periods, index),
            waveform: numeric_u8_at(&vcp_waveform_codes, index)
                .and_then(waveform_from_code)
                .map(str::to_owned),
            moment_coverage: numeric_u8_at(&vcp_moment_coverage_codes, index)
                .and_then(moment_coverage_from_code)
                .map(str::to_owned),
            surveillance_prf_code: numeric_u8_at(&surveillance_prf_codes, index),
            surveillance_pulse_count: numeric_u16_at(&surveillance_pulse_counts, index),
            doppler_prf_code: numeric_u8_at(&doppler_prf_codes, index),
            doppler_pulse_count: numeric_u16_at(&doppler_pulse_counts, index),
        };
        sweeps.push(SweepBuild {
            start_ray,
            end_ray,
            ngates: sweep_ngates,
            sweep,
            leg,
        });
    }
    if sweeps.is_empty() {
        return Err(invalid("CfRadial volume decoded no sweeps"));
    }

    let grid_values = n_rays
        .checked_mul(n_gates)
        .ok_or_else(|| invalid("CfRadial field dimensions overflow addressable memory"))?;
    for var in fields {
        let (quantity, _) = Quantity::classify(&var.name, var.attr_str("standard_name"));
        let points = is_points(var);
        let expected_values = match &ragged {
            Some(ragged) if points => ragged.points,
            _ => grid_values,
        };
        let raw = file.read_var(&var.name)?;
        if raw.len() < expected_values {
            return Err(invalid(format!(
                "CfRadial field '{}' has {} values; expected at least {expected_values}",
                var.name,
                raw.len()
            )));
        }
        for build in &mut sweeps {
            if !scan_leg_allows_quantity(&build.leg, quantity) {
                continue;
            }
            let field = match &ragged {
                Some(ragged) if points => {
                    let (rows, padded) = ragged.gather(
                        &raw,
                        build.start_ray..=build.end_ray,
                        build.ngates,
                        var,
                        &mut budget,
                    )?;
                    let len = rows.len();
                    let mut field =
                        build_field(var, &rows, 0..len, build.ngates as u32, &mut budget)?;
                    if padded && let Some(field) = &mut field {
                        default_fill(&mut field.data);
                    }
                    field
                }
                _ => {
                    let rows = build.start_ray * n_gates..(build.end_ray + 1) * n_gates;
                    build_field(var, &raw, rows, ngates, &mut budget)?
                }
            };
            let Some(field) = field else {
                continue;
            };
            build
                .sweep
                .add_field(field)
                .map_err(|err| invalid(format!("CfRadial field '{}': {err}", var.name)))?;
        }
    }

    let legs: Vec<ScanLeg> = sweeps.iter().map(|build| build.leg.clone()).collect();
    if legs.iter().any(|leg| *leg != ScanLeg::default()) {
        definition.legs = legs;
    }
    if definition != ScanDefinition::default() {
        volume.scan.definition = Some(Box::new(definition));
    }
    volume.provenance.decode.message_count = sweep_count;
    volume.sweeps = sweeps.into_iter().map(|build| build.sweep).collect();
    volume.provenance.decode.decoded_ray_count = volume.sweeps.iter().map(Sweep::nrays).sum();
    // The source's own attributes of every variable a typed slot holds.
    let kept = kept_names(&volume);
    volume.variable_attrs = slotted_variable_attrs(file, "", &|var| {
        var.dim_ids.as_slice() == [time_dim, range_dim] || kept.contains(var.name.as_str())
    });
    charge_passthrough(&volume, &mut budget)?;
    volume.seal().map_err(|err| invalid(err.to_string()))?;
    if volume.time_coverage.is_none() {
        volume.time_coverage = volume.ray_time_extent();
    }
    Ok(volume)
}

/// The volume-level description shared by CfRadial 1 and 2: identity,
/// global attributes (slotted ones typed, the rest verbatim), time coverage,
/// `volume_number`, platform and instrument types, the site location (a
/// moving platform's first ray, from a `time`-dimensioned variable or, when
/// `time_dim` is `None`, one whose only dimension is called `time`), and the
/// provenance of a CfRadial 1 file (the CfRadial 2 decoder adjusts it).
pub(crate) fn describe_volume(
    file: &NcFile<'_>,
    time_dim: Option<usize>,
    time_reference: DateTime<Utc>,
    coverage_start: Option<DateTime<Utc>>,
    coverage_end: Option<DateTime<Utc>>,
) -> Volume {
    let instrument_name = file
        .gattr_str("instrument_name")
        .or_else(|| file.gattr_str("site_name"))
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("CFRAD")
        .to_owned();
    let mut volume = Volume::new(instrument_name, time_reference);
    volume.attrs.site_name = file.gattr_str("site_name").map(str::to_owned);
    volume.attrs.title = metadata_text(file, "title");
    volume.attrs.institution = metadata_text(file, "institution");
    volume.attrs.references = metadata_text(file, "references");
    volume.attrs.source = metadata_text(file, "source");
    volume.attrs.history = metadata_text(file, "history");
    volume.attrs.comment = metadata_text(file, "comment");
    volume.attrs.platform_is_mobile = gattr_bool(file, "platform_is_mobile").unwrap_or(false);
    volume.attrs.ray_times_increase = gattr_bool(file, "ray_times_increase");
    volume.attrs.simulated = gattr_bool(file, "simulated").unwrap_or(false);
    volume.time_coverage = match (coverage_start, coverage_end) {
        (Some(start), Some(end)) => Some(recast_radar_core::model::TimeCoverage { start, end }),
        _ => None,
    };
    // Every global attribute without a slot, and a slotted one whose value
    // the slot does not hold (a flag that is not a boolean, a coverage time
    // that does not parse or differs from the `time_coverage_*` variable).
    let coverage_attr_held = |name: &str, held: Option<DateTime<Utc>>| {
        file.gattr_str(name)
            .and_then(parse_iso_instant)
            .is_some_and(|instant| Some(instant) == held)
    };
    volume.attrs.other = file
        .gattrs
        .iter()
        .filter(|(name, _)| {
            let name = name.as_str();
            if !SLOTTED_GLOBAL_ATTRS.contains(&name) {
                return true;
            }
            match name {
                "platform_is_mobile" | "ray_times_increase" | "simulated" => {
                    gattr_bool(file, name).is_none()
                }
                "time_coverage_start" => !coverage_attr_held(name, coverage_start),
                "time_coverage_end" => !coverage_attr_held(name, coverage_end),
                _ => false,
            }
        })
        .map(|(name, value)| (name.as_str().into(), attr_value(value)))
        .collect();
    volume.volume_number = file
        .read_var("volume_number")
        .ok()
        .and_then(|array| array.get_f64(0))
        .filter(|value| value.is_finite() && value.fract() == 0.0)
        .and_then(|value| i32::try_from(value as i64).ok());
    volume.platform_type = char_var_text(file, "platform_type")
        .and_then(|text| PlatformType::parse(&text))
        .unwrap_or(PlatformType::Fixed);
    volume.instrument_type = char_var_text(file, "instrument_type")
        .and_then(|text| InstrumentType::parse(&text))
        .unwrap_or(InstrumentType::Radar);
    volume.primary_axis =
        char_var_text(file, "primary_axis").and_then(|text| PrimaryAxis::parse(&text));
    volume.status_str = char_var_text(file, "status_str").filter(|text| !text.is_empty());
    let scalar = |name: &str| -> Option<f64> {
        let var = file.vars.get(name)?;
        let per_ray = var.dim_ids.len() == 1
            && (time_dim == var.dim_ids.first().copied()
                || (time_dim.is_none() && file.dim_name(var, 0) == Some("time")));
        if !var.dim_ids.is_empty() && !per_ray {
            return None;
        }
        // A NaN location (a writer without one) is no location.
        file.read_var(name)
            .ok()
            .and_then(|array| array.get_f64(0))
            .filter(|value| value.is_finite())
    };
    // A moving platform's root location is its first ray's.
    volume.location.latitude_deg = scalar("latitude");
    volume.location.longitude_deg = scalar("longitude");
    volume.location.altitude_m = scalar("altitude");
    volume.location.altitude_agl_m = scalar("altitude_agl");
    volume.provenance.source_format = SourceFormat::CfRadial1;
    volume.provenance.source_version = Some(
        file.gattr_str("version")
            .map(str::to_owned)
            .unwrap_or_else(|| "CfRadial-1".to_owned()),
    );
    volume.provenance.source_conventions = file.gattr_str("Conventions").map(str::to_owned);
    volume.provenance.compression = Some("cfradial1-netcdf3".to_owned());
    describe_scan(file, &mut volume);
    volume.provenance.polarization_note = metadata_text(file, "polarization");
    volume.provenance.calibration_note = metadata_text(file, "calibration");
    let simulation = SimulationProvenance {
        forward_operator: metadata_text(file, "forward_operator"),
        forward_operator_config: metadata_text(file, "forward_operator_config"),
        source_model: metadata_text(file, "source_model"),
        microphysics_scheme: metadata_text(file, "microphysics_scheme"),
        scattering_model: metadata_text(file, "scattering_model"),
    };
    if simulation != SimulationProvenance::default() {
        volume.attrs.simulated = true;
        volume.simulation = Some(Box::new(simulation));
    }
    volume
}

/// `scan_name`, `scan_id` (an integer, else kept as text), `vcp_pattern` and
/// the BowEcho scan-table provenance attributes.
fn describe_scan(file: &NcFile<'_>, volume: &mut Volume) {
    volume.scan.name = metadata_text(file, "scan_name");
    let scan_id_text = metadata_text(file, "scan_id").or_else(|| {
        file.gattr_f64("scan_id")
            .filter(|value| value.is_finite())
            .map(|value| {
                if value.fract() == 0.0 {
                    format!("{value:.0}")
                } else {
                    value.to_string()
                }
            })
    });
    volume.scan.id = scan_id_text
        .as_deref()
        .and_then(|text| text.parse::<i64>().ok());
    volume.scan.vcp_pattern = file
        .gattr_f64("vcp_pattern")
        .filter(|value| value.is_finite() && value.fract() == 0.0)
        .and_then(|value| u16::try_from(value as i64).ok())
        .filter(|pattern| *pattern > 0);
    let definition = ScanDefinition {
        source_document: metadata_text(file, "vcp_source_document"),
        source_revision: metadata_text(file, "vcp_source_revision"),
        source_rda_build: metadata_text(file, "vcp_source_rda_build"),
        source_figure: metadata_text(file, "vcp_source_figure"),
        pulse_length: metadata_text(file, "vcp_pulse_length"),
        adaptations: metadata_text(file, "vcp_adaptations"),
        scan_id_text: scan_id_text
            .filter(|text| volume.scan.id.is_none_or(|id| id.to_string() != *text)),
        legs: Vec::new(),
    };
    if definition != ScanDefinition::default() {
        volume.scan.definition = Some(Box::new(definition));
    }
}

/// Radar parameters (Table 301-12) from the variables of `file`: the root
/// of a CfRadial 1 file, the `radar_parameters` group of a CfRadial 2 file.
pub(crate) fn radar_parameters(file: &NcFile<'_>) -> RadarParameters {
    RadarParameters {
        frequency_hz: cfradial_frequency_hz(file),
        beam_width_h_deg: cfradial_beam_width_deg(
            file,
            &["radar_beam_width_h", "radar_beam_width_h_deg"],
        ),
        beam_width_v_deg: cfradial_beam_width_deg(
            file,
            &["radar_beam_width_v", "radar_beam_width_v_deg"],
        ),
        antenna_gain_h_db: numeric_var_first(file, "radar_antenna_gain_h").map(|v| v as f32),
        antenna_gain_v_db: numeric_var_first(file, "radar_antenna_gain_v").map(|v| v as f32),
        receiver_bandwidth_hz: numeric_var_first(file, "radar_rx_bandwidth")
            .or_else(|| numeric_var_first(file, "radar_receiver_bandwidth"))
            .map(|v| v as f32),
        pulse_width_s: cfradial_pulse_width_us(file).map(|us| us * 1e-6),
        prt_s: cfradial_prt_s(file),
        unambiguous_range_m: cfradial_unambiguous_range_km(file).map(|km| km * 1000.0),
    }
}

/// One field of a sweep: the `rows` elements (whole rays) of a `(time,
/// range)` variable in their stored type with the variable's CF packing, or
/// `None` for text data. Integer storage keeps its width (`byte`, `short`,
/// `int`; netCDF-4 `ubyte` and `ushort`); netCDF-4 `uint`, `int64` and
/// `uint64` fields, which the model has no integer storage for, keep their
/// codes as `f64` (exact up to 2^53) with the packing as the float transform.
/// Charges the sweep's rows to `budget` before copying them.
pub(crate) fn build_field(
    var: &NcVar,
    raw: &NcArray,
    rows: std::ops::Range<usize>,
    ngates: u32,
    budget: &mut DecodeBudget,
) -> Result<Option<Field>> {
    let word_bytes = match raw {
        NcArray::I8(_) | NcArray::Char(_) | NcArray::U8(_) => 1,
        NcArray::I16(_) | NcArray::U16(_) => 2,
        NcArray::I32(_) | NcArray::F32(_) => 4,
        NcArray::F64(_) | NcArray::U32(_) | NcArray::I64(_) | NcArray::U64(_) => 8,
        NcArray::Str(_) => return Ok(None),
    };
    let gates = (ngates as usize).max(1);
    budget
        .charge(rows.len() / gates, gates * word_bytes, "CfRadial field")
        .map_err(CfRadialError::LimitExceeded)?;
    let coding = FieldCoding::of(var, raw);
    let wide = |values: Vec<f64>| FieldData::F64 {
        values,
        coding: FloatCoding {
            transform: coding.float_transform(),
            fill_value: coding.fill,
            undetect: coding.undetect,
        },
    };
    let data = match raw {
        // `_Unsigned = "true"`: classic `byte`/`short`/`int` holding
        // unsigned codes (the classic format has no unsigned types).
        NcArray::I8(values) if coding.unsigned => FieldData::U8 {
            values: values[rows].iter().map(|v| *v as u8).collect(),
            coding: coding.int(),
        },
        NcArray::I16(values) if coding.unsigned => FieldData::U16 {
            values: values[rows].iter().map(|v| *v as u16).collect(),
            coding: coding.int(),
        },
        NcArray::I32(values) if coding.unsigned => {
            wide(values[rows].iter().map(|v| f64::from(*v as u32)).collect())
        }
        NcArray::I8(values) => FieldData::I8 {
            values: values[rows].to_vec(),
            coding: coding.int(),
        },
        NcArray::Char(values) | NcArray::U8(values) => FieldData::U8 {
            values: values[rows].to_vec(),
            coding: coding.int(),
        },
        NcArray::I16(values) => FieldData::I16 {
            values: values[rows].to_vec(),
            coding: coding.int(),
        },
        NcArray::U16(values) => FieldData::U16 {
            values: values[rows].to_vec(),
            coding: coding.int(),
        },
        NcArray::I32(values) => FieldData::I32 {
            values: values[rows].to_vec(),
            coding: coding.int(),
        },
        NcArray::F32(values) => FieldData::F32 {
            values: values[rows].to_vec(),
            coding: FloatCoding {
                transform: coding.float_transform(),
                fill_value: coding.fill.map(|fill| fill as f32),
                undetect: coding.undetect.map(|undetect| undetect as f32),
            },
        },
        NcArray::F64(values) => FieldData::F64 {
            values: values[rows].to_vec(),
            coding: FloatCoding {
                transform: coding.float_transform(),
                fill_value: coding.fill,
                undetect: coding.undetect,
            },
        },
        NcArray::U32(values) => wide(values[rows].iter().map(|v| f64::from(*v)).collect()),
        NcArray::I64(values) => wide(values[rows].iter().map(|v| *v as f64).collect()),
        NcArray::U64(values) => wide(values[rows].iter().map(|v| *v as f64).collect()),
        NcArray::Str(_) => return Ok(None),
    };
    let (quantity, polarization) = Quantity::classify(&var.name, var.attr_str("standard_name"));
    let mut field = Field::new(
        FieldName::parse(&var.name),
        GateMapping::IDENTITY,
        ngates,
        data,
    );
    field.quantity = quantity;
    field.polarization = polarization;
    field.attrs.standard_name = var
        .attr_str("standard_name")
        .map(|s| Cow::Owned(s.to_owned()));
    field.attrs.long_name = var.attr_str("long_name").map(|s| Cow::Owned(s.to_owned()));
    field.attrs.units = var.attr_str("units").map(|s| Cow::Owned(s.to_owned()));
    field.attrs.sampling_ratio = var.attr_f64("sampling_ratio").map(|v| v as f32);
    let table = table_attrs(var);
    field.attrs.is_discrete = table.is_discrete;
    field.attrs.field_folds = table.field_folds;
    field.attrs.fold_limit_lower = table.fold_limit_lower;
    field.attrs.fold_limit_upper = table.fold_limit_upper;
    field.attrs.is_quality_field = table.is_quality_field;
    field.attrs.qualified_variables = table.qualified_variables;
    field.attrs.ancillary_variables = table.ancillary_variables;
    field.attrs.thresholding_xml = table.thresholding_xml;
    field.attrs.flag_values = coding.flag_values.clone();
    field.attrs.flag_masks = coding.flag_masks.clone();
    field.attrs.flag_meanings = coding.flag_meanings.clone();
    // `missing_value` stays verbatim unless it is the fill code the coding
    // took (a file with both keeps `_FillValue` as the fill).
    let missing_is_fill = !var.attrs.contains_key("_FillValue");
    field.attrs.other = var
        .attrs
        .iter()
        .filter(|(attr, _)| {
            let attr = attr.as_str();
            let slotted = matches!(
                attr,
                "standard_name"
                    | "long_name"
                    | "units"
                    | "sampling_ratio"
                    | "scale_factor"
                    | "add_offset"
                    | "_FillValue"
            ) || coding.slotted.contains(&attr)
                || table.slotted.contains(&attr);
            let fill = attr == "missing_value" && missing_is_fill;
            !slotted && !fill
        })
        .map(|(attr, value)| (attr.as_str().into(), attr_value(value)))
        .collect();
    Ok(Some(field))
}

/// FM301 Table 301-10 attributes of a field variable that parse into their
/// typed slots, and the names of those that did.
#[derive(Default)]
struct TableAttrs {
    is_discrete: Option<bool>,
    field_folds: Option<bool>,
    fold_limit_lower: Option<f32>,
    fold_limit_upper: Option<f32>,
    is_quality_field: Option<bool>,
    qualified_variables: Vec<FieldName>,
    ancillary_variables: Vec<FieldName>,
    thresholding_xml: Option<String>,
    slotted: Vec<&'static str>,
}

fn table_attrs(var: &NcVar) -> TableAttrs {
    let mut table = TableAttrs::default();
    let mut flag = |name: &'static str| -> Option<bool> {
        let value = match var.attrs.get(name)? {
            NcValue::Str(text) => parse_bool(text),
            _ => None,
        };
        if value.is_some() {
            table.slotted.push(name);
        }
        value
    };
    let is_discrete = flag("is_discrete");
    let field_folds = flag("field_folds");
    let is_quality_field = flag("is_quality_field");
    table.is_discrete = is_discrete;
    table.field_folds = field_folds;
    table.is_quality_field = is_quality_field;
    for name in ["fold_limit_lower", "fold_limit_upper"] {
        let value = match var.attrs.get(name) {
            Some(NcValue::Floats(v)) if v.len() == 1 => Some(v[0]),
            Some(NcValue::Doubles(v)) if v.len() == 1 && f64::from(v[0] as f32) == v[0] => {
                Some(v[0] as f32)
            }
            _ => None,
        };
        if value.is_some() {
            table.slotted.push(name);
            if name == "fold_limit_lower" {
                table.fold_limit_lower = value;
            } else {
                table.fold_limit_upper = value;
            }
        }
    }
    for name in ["qualified_variables", "ancillary_variables"] {
        if let Some(NcValue::Str(text)) = var.attrs.get(name) {
            let names: Vec<FieldName> = text.split_whitespace().map(FieldName::parse).collect();
            // Only a list the view writes back the same way.
            if names
                .iter()
                .map(FieldName::as_str)
                .collect::<Vec<_>>()
                .join(" ")
                == *text
            {
                table.slotted.push(name);
                if name == "qualified_variables" {
                    table.qualified_variables = names;
                } else {
                    table.ancillary_variables = names;
                }
            }
        }
    }
    if let Some(NcValue::Str(text)) = var.attrs.get("thresholding_xml") {
        table.slotted.push("thresholding_xml");
        table.thresholding_xml = Some(text.clone());
    }
    table
}

/// Per-sweep variables with a typed slot (or that define the sweep table);
/// any other `(sweep, ...)` variable becomes a scalar of each sweep.
const SLOTTED_SWEEP_VARS: &[&str] = &[
    "sweep_number",
    "fixed_angle",
    "sweep_start_ray_index",
    "sweep_end_ray_index",
    "sweep_mode",
    "follow_mode",
    "prt_mode",
    "polarization_mode",
    "target_scan_rate",
    "rays_are_indexed",
    "ray_angle_res",
    "vcp_source_row_index",
    "vcp_azimuth_rate",
    "vcp_source_period",
    "vcp_waveform_code",
    "vcp_moment_coverage_code",
    "vcp_surveillance_prf_code",
    "vcp_surveillance_pulse_count",
    "vcp_doppler_prf_code",
    "vcp_doppler_pulse_count",
];

/// Entry `sweep` of a `(sweep)` or `(sweep, string_length)` variable as a
/// scalar extra variable of that sweep (a `(sweep, n)` numeric variable
/// keeps its inner dimension).
fn sweep_scalar_variable(file: &NcFile<'_>, var: &NcVar, sweep: usize) -> Option<ExtraVariable> {
    let array = file.read_var(&var.name).ok()?;
    let dims = file.var_dims(var);
    let row = dims.iter().skip(1).product::<usize>().max(1);
    let range = sweep.checked_mul(row)?..sweep.checked_add(1)?.checked_mul(row)?;
    if range.end > array.len() {
        return None;
    }
    let (names, shape): (Vec<Box<str>>, Vec<u32>) = if array.is_text() || row == 1 {
        (Vec::new(), Vec::new())
    } else {
        (
            var.dim_ids[1..]
                .iter()
                .map(|id| file.dims[*id].0.as_str().into())
                .collect(),
            dims[1..]
                .iter()
                .map(|len| u32::try_from(*len).unwrap_or(u32::MAX))
                .collect(),
        )
    };
    let values = match &array {
        NcArray::Char(chars) => ArrayBuf::Text(vec![text_of(&chars[range]).into()]),
        NcArray::Str(strings) => {
            ArrayBuf::Text(strings[range].iter().map(|s| s.as_str().into()).collect())
        }
        other => slice_buf(other, range)?,
    };
    Some(ExtraVariable {
        name: var.name.as_str().into(),
        dims: names,
        shape,
        values,
        attrs: var
            .attrs
            .iter()
            .map(|(name, value)| (name.as_str().into(), attr_value(value)))
            .collect(),
    })
}

/// The per-ray variables of one ray dimension: coordinates, the Table
/// 301-8a instrument variables, the monitoring powers and a moving
/// platform's track, each only when exactly aligned to the ray dimension.
pub(crate) struct RayData {
    pub(crate) azimuth: Vec<f64>,
    pub(crate) elevation: Vec<f64>,
    /// `time`, in seconds since its own `time.units` reference.
    pub(crate) seconds: Option<Vec<f64>>,
    nyquist: Option<Vec<f64>>,
    unambiguous_range: Option<Vec<f64>>,
    unambiguous_range_scale: f64,
    prt: Option<Vec<f64>>,
    prt_scale: f64,
    prt_ratio: Option<Vec<f64>>,
    n_samples: Option<Vec<f64>>,
    pulse_count: Option<Vec<f64>>,
    pulse_width: Option<Vec<f64>>,
    pulse_width_scale: f64,
    scan_rate: Option<Vec<f64>>,
    antenna_transition: Option<Vec<f64>>,
    calib_index: Option<Vec<f64>>,
    independent_samples: Option<Vec<f64>>,
    transmit_power_h: Option<Vec<f64>>,
    transmit_power_v: Option<Vec<f64>>,
    pub(crate) platform: Option<PlatformTrack>,
}

impl RayData {
    /// Read the per-ray variables of `file` along dimension `ray_dim`
    /// (`n_rays` long). `azimuth` and `elevation` are required.
    pub(crate) fn read(
        file: &NcFile<'_>,
        ray_dim: usize,
        n_rays: usize,
        budget: &mut DecodeBudget,
    ) -> Result<Self> {
        let azimuth = read_f64s(file, "azimuth", budget)?;
        let elevation = read_f64s(file, "elevation", budget)?;
        if azimuth.len() < n_rays || elevation.len() < n_rays {
            return Err(invalid("azimuth/elevation shorter than the time dimension"));
        }
        let mut data = Self::instruments(file, ray_dim, n_rays, budget)?;
        data.azimuth = azimuth;
        data.elevation = elevation;
        data.seconds = optional_f64s(file, "time", budget)?;
        Ok(data)
    }

    /// Only the instrument, monitoring and platform variables (what a
    /// CfRadial 2 `monitoring` group holds).
    pub(crate) fn instruments(
        file: &NcFile<'_>,
        ray_dim: usize,
        n_rays: usize,
        budget: &mut DecodeBudget,
    ) -> Result<Self> {
        let mut aligned = |name: &str| aligned_time_values(file, name, ray_dim, n_rays, budget);
        Ok(Self {
            azimuth: Vec::new(),
            elevation: Vec::new(),
            seconds: None,
            nyquist: aligned("nyquist_velocity")?,
            unambiguous_range: aligned("unambiguous_range")?,
            unambiguous_range_scale: range_units_to_m_scale(units_of(file, "unambiguous_range")),
            prt: aligned("prt")?,
            prt_scale: time_units_scale(units_of(file, "prt")),
            prt_ratio: aligned("prt_ratio")?,
            n_samples: aligned("n_samples")?,
            pulse_count: aligned("pulse_count")?,
            pulse_width: aligned("pulse_width")?,
            pulse_width_scale: time_units_scale(units_of(file, "pulse_width")),
            scan_rate: aligned("scan_rate")?,
            antenna_transition: aligned("antenna_transition")?,
            // CfRadial 1's name, then FM301-2022's.
            calib_index: match aligned("r_calib_index")? {
                Some(values) => Some(values),
                None => aligned("calib_index")?,
            },
            independent_samples: aligned("independent_samples")?,
            transmit_power_h: match aligned("measured_transmit_power_h")? {
                Some(values) => Some(values),
                None => aligned("radar_measured_transmit_power_h")?,
            },
            transmit_power_v: match aligned("measured_transmit_power_v")? {
                Some(values) => Some(values),
                None => aligned("radar_measured_transmit_power_v")?,
            },
            platform: read_platform_track(file, ray_dim, n_rays, budget)?,
        })
    }

    /// Take every instrument variable `self` lacks from `other`.
    pub(crate) fn merge_missing(&mut self, other: Self) {
        fn take<T>(slot: &mut Option<T>, other: Option<T>) -> bool {
            if slot.is_none() && other.is_some() {
                *slot = other;
                true
            } else {
                false
            }
        }
        take(&mut self.nyquist, other.nyquist);
        if take(&mut self.unambiguous_range, other.unambiguous_range) {
            self.unambiguous_range_scale = other.unambiguous_range_scale;
        }
        if take(&mut self.prt, other.prt) {
            self.prt_scale = other.prt_scale;
        }
        take(&mut self.prt_ratio, other.prt_ratio);
        take(&mut self.n_samples, other.n_samples);
        take(&mut self.pulse_count, other.pulse_count);
        if take(&mut self.pulse_width, other.pulse_width) {
            self.pulse_width_scale = other.pulse_width_scale;
        }
        take(&mut self.scan_rate, other.scan_rate);
        take(&mut self.antenna_transition, other.antenna_transition);
        take(&mut self.calib_index, other.calib_index);
        take(&mut self.independent_samples, other.independent_samples);
        take(&mut self.transmit_power_h, other.transmit_power_h);
        take(&mut self.transmit_power_v, other.transmit_power_v);
        take(&mut self.platform, other.platform);
    }

    /// Push rays `rays` into `sweep` (times shifted by `time_offset_s` onto
    /// the volume reference) with their instrument variables, monitoring
    /// powers and platform track.
    pub(crate) fn fill(
        &self,
        sweep: &mut Sweep,
        rays: std::ops::RangeInclusive<usize>,
        time_offset_s: f64,
    ) {
        sweep.reserve_rays(rays.end() - rays.start() + 1);
        for ray in rays.clone() {
            let time_s = self
                .seconds
                .as_ref()
                .and_then(|seconds| seconds.get(ray))
                .copied()
                .unwrap_or(0.0);
            sweep.push_ray(
                time_s + time_offset_s,
                azimuth_f32(self.azimuth[ray]),
                self.elevation[ray] as f32,
            );
        }
        let slice_f32 = |values: &Option<Vec<f64>>, scale: f64| -> Option<Vec<f32>> {
            values.as_ref().map(|values| {
                values[rays.clone()]
                    .iter()
                    .map(|value| {
                        if value.is_finite() {
                            (value * scale) as f32
                        } else {
                            f32::NAN
                        }
                    })
                    .collect()
            })
        };
        sweep.ray_vars.nyquist_velocity_mps = slice_f32(&self.nyquist, 1.0);
        sweep.ray_vars.unambiguous_range_m =
            slice_f32(&self.unambiguous_range, self.unambiguous_range_scale);
        sweep.ray_vars.prt_s = slice_f32(&self.prt, self.prt_scale);
        sweep.ray_vars.prt_ratio = slice_f32(&self.prt_ratio, 1.0);
        sweep.ray_vars.pulse_width_s = slice_f32(&self.pulse_width, self.pulse_width_scale);
        sweep.ray_vars.scan_rate_deg_per_s = slice_f32(&self.scan_rate, 1.0);
        sweep.ray_vars.independent_samples = slice_f32(&self.independent_samples, 1.0);
        let slice_i32 = |values: &Option<Vec<f64>>| -> Option<Vec<i32>> {
            values.as_ref().map(|values| {
                values[rays.clone()]
                    .iter()
                    .map(|value| {
                        if value.is_finite()
                            && value.fract() == 0.0
                            && (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(value)
                        {
                            *value as i32
                        } else {
                            -9999
                        }
                    })
                    .collect()
            })
        };
        sweep.ray_vars.n_samples =
            slice_i32(&self.n_samples).or_else(|| slice_i32(&self.pulse_count));
        sweep.ray_vars.calib_index = slice_i32(&self.calib_index);
        sweep.ray_vars.antenna_transition = self.antenna_transition.as_ref().map(|values| {
            values[rays.clone()]
                .iter()
                .map(|value| u8::from(*value != 0.0 && value.is_finite()))
                .collect()
        });
        if let Some(platform) = &self.platform {
            sweep.platform_track = Some(Box::new(platform.slice(rays.clone())));
        }
        if self.transmit_power_h.is_some() || self.transmit_power_v.is_some() {
            sweep.monitoring = Some(Box::new(recast_radar_core::model::Monitoring {
                radar_measured_transmit_power_h_dbm: slice_f32(&self.transmit_power_h, 1.0),
                radar_measured_transmit_power_v_dbm: slice_f32(&self.transmit_power_v, 1.0),
                ..Default::default()
            }));
        }
    }
}

/// Text of a char array: up to the first NUL, lossy UTF-8, trimmed.
pub(crate) fn text_of(chars: &[u8]) -> String {
    let text = chars.split(|byte| *byte == 0).next().unwrap_or_default();
    String::from_utf8_lossy(text).trim().to_owned()
}

/// Elements `range` of a numeric array in their stored type.
fn slice_buf(array: &NcArray, range: std::ops::Range<usize>) -> Option<ArrayBuf> {
    Some(match array {
        NcArray::I8(v) => ArrayBuf::I8(v.get(range)?.to_vec()),
        NcArray::U8(v) => ArrayBuf::U8(v.get(range)?.to_vec()),
        NcArray::I16(v) => ArrayBuf::I16(v.get(range)?.to_vec()),
        NcArray::U16(v) => ArrayBuf::U16(v.get(range)?.to_vec()),
        NcArray::I32(v) => ArrayBuf::I32(v.get(range)?.to_vec()),
        NcArray::U32(v) => ArrayBuf::U32(v.get(range)?.to_vec()),
        NcArray::I64(v) => ArrayBuf::I64(v.get(range)?.to_vec()),
        NcArray::U64(v) => {
            let v = v.get(range)?;
            // The model has no u64 arrays: i64 when every value fits, else
            // f64.
            if v.iter().all(|x| i64::try_from(*x).is_ok()) {
                ArrayBuf::I64(v.iter().map(|x| *x as i64).collect())
            } else {
                ArrayBuf::F64(v.iter().map(|x| *x as f64).collect())
            }
        }
        NcArray::F32(v) => ArrayBuf::F32(v.get(range)?.to_vec()),
        NcArray::F64(v) => ArrayBuf::F64(v.get(range)?.to_vec()),
        NcArray::Char(v) => ArrayBuf::U8(v.get(range)?.to_vec()),
        NcArray::Str(v) => {
            ArrayBuf::Text(v.get(range)?.iter().map(|s| s.as_str().into()).collect())
        }
    })
}

struct SweepBuild {
    /// Gates of each of the sweep's rows.
    ngates: usize,
    start_ray: usize,
    end_ray: usize,
    sweep: Sweep,
    leg: ScanLeg,
}

/// A field's CF packing as the file states it: `scale_factor`,
/// `add_offset`, `_FillValue` (else `missing_value`), `_Unsigned`,
/// `_Undetect`, `valid_range`, and `flag_values`/`flag_masks`/
/// `flag_meanings` (a `range_folded` meaning is the coding's range-folded
/// code). Integer sentinels are read in the variable's type (unsigned with
/// `_Unsigned = "true"`).
struct FieldCoding {
    scale: f64,
    offset: f64,
    attr_width: FloatWidth,
    fill: Option<f64>,
    unsigned: bool,
    undetect: Option<f64>,
    valid_range: Option<[f64; 2]>,
    range_folded: Option<f64>,
    flag_values: Vec<i64>,
    flag_masks: Vec<i64>,
    flag_meanings: Vec<Box<str>>,
    /// Attributes the coding holds (besides scale, offset and fill).
    slotted: Vec<&'static str>,
}

/// Integer values of a numeric attribute, as a variable of `bits` wide
/// (unsigned when `unsigned`) holds them; `None` for anything else.
fn attr_codes(value: &NcValue, unsigned: bool, bits: u32) -> Option<Vec<f64>> {
    let values: Vec<f64> = match value {
        NcValue::Ints(values, _) => values.iter().map(|v| *v as f64).collect(),
        NcValue::Floats(values) => values.iter().map(|v| f64::from(*v)).collect(),
        NcValue::Doubles(values) => values.clone(),
        _ => return None,
    };
    Some(
        values
            .into_iter()
            .map(|v| {
                if unsigned && v < 0.0 && bits < 64 {
                    v + 2f64.powi(bits as i32)
                } else {
                    v
                }
            })
            .collect(),
    )
}

impl FieldCoding {
    fn of(var: &NcVar, raw: &NcArray) -> Self {
        let attr_width = match (var.attrs.get("scale_factor"), var.attrs.get("add_offset")) {
            (Some(NcValue::Floats(_)), _) | (None, Some(NcValue::Floats(_))) => FloatWidth::F32,
            _ => FloatWidth::F64,
        };
        let bits = match raw {
            NcArray::I8(_) => Some(8),
            NcArray::I16(_) => Some(16),
            NcArray::I32(_) => Some(32),
            _ => None,
        };
        let integer = matches!(
            raw,
            NcArray::I8(_)
                | NcArray::U8(_)
                | NcArray::Char(_)
                | NcArray::I16(_)
                | NcArray::U16(_)
                | NcArray::I32(_)
        );
        let unsigned = bits.is_some()
            && var
                .attr_str("_Unsigned")
                .is_some_and(|text| text.trim().eq_ignore_ascii_case("true"));
        let bits = bits.unwrap_or(64);
        let mut slotted = Vec::new();
        if unsigned {
            slotted.push("_Unsigned");
        }
        let code = |name: &str| -> Option<f64> {
            var.attrs
                .get(name)
                .and_then(|value| attr_codes(value, unsigned, bits))
                .and_then(|values| (values.len() == 1).then(|| values[0]))
        };
        let fill = code("_FillValue").or_else(|| code("missing_value"));
        let undetect = code("_Undetect");
        if undetect.is_some() {
            slotted.push("_Undetect");
        }
        let valid_range = if integer {
            var.attrs
                .get("valid_range")
                .and_then(|value| attr_codes(value, unsigned, bits))
                .and_then(|values| match values.as_slice() {
                    [lo, hi] if lo.fract() == 0.0 && hi.fract() == 0.0 => Some([*lo, *hi]),
                    _ => None,
                })
        } else {
            None
        };
        if valid_range.is_some() {
            slotted.push("valid_range");
        }
        // Flags of integer fields: whole numbers, one meaning per value.
        let mut range_folded = None;
        let mut flag_values = Vec::new();
        let mut flag_meanings: Vec<Box<str>> = Vec::new();
        let mut flag_masks = Vec::new();
        if integer {
            let values = var
                .attrs
                .get("flag_values")
                .and_then(|value| attr_codes(value, unsigned, bits))
                .filter(|values| values.iter().all(|v| v.fract() == 0.0));
            let meanings: Option<Vec<&str>> = var
                .attr_str("flag_meanings")
                .map(|text| text.split_whitespace().collect());
            if let (Some(values), Some(meanings)) = (values, meanings)
                && values.len() == meanings.len()
                && !values.is_empty()
            {
                for (value, meaning) in values.iter().zip(&meanings) {
                    if *meaning == "range_folded" && range_folded.is_none() {
                        range_folded = Some(*value);
                    } else {
                        flag_values.push(*value as i64);
                        flag_meanings.push((*meaning).into());
                    }
                }
                slotted.push("flag_values");
                slotted.push("flag_meanings");
            }
            if let Some(masks) = var
                .attrs
                .get("flag_masks")
                .and_then(|value| attr_codes(value, unsigned, bits))
                .filter(|values| values.iter().all(|v| v.fract() == 0.0))
            {
                flag_masks = masks.into_iter().map(|v| v as i64).collect();
                slotted.push("flag_masks");
            }
        }
        Self {
            scale: var.attr_f64("scale_factor").unwrap_or(1.0),
            offset: var.attr_f64("add_offset").unwrap_or(0.0),
            attr_width,
            fill,
            unsigned,
            undetect,
            valid_range,
            range_folded,
            flag_values,
            flag_masks,
            flag_meanings,
            slotted,
        }
    }

    fn transform(&self) -> LinearTransform {
        LinearTransform::CfScaleOffset {
            scale_factor: self.scale,
            add_offset: self.offset,
            attr_width: self.attr_width,
        }
    }

    /// The transform of a float field: `None` when the values are physical.
    fn float_transform(&self) -> Option<LinearTransform> {
        (self.scale != 1.0 || self.offset != 0.0).then(|| self.transform())
    }

    fn int<T: recast_radar_core::model::PackedInt>(&self) -> IntCoding<T> {
        let code = |value: Option<f64>| {
            value
                .filter(|value| value.is_finite() && value.fract() == 0.0)
                .and_then(|value| T::from_i64(value as i64))
        };
        IntCoding {
            transform: self.transform(),
            fill_value: code(self.fill),
            undetect: code(self.undetect),
            range_folded: code(self.range_folded),
            valid_range: self
                .valid_range
                .and_then(|[lo, hi]| Some([code(Some(lo))?, code(Some(hi))?])),
        }
    }
}

/// How the `range` variable states the gate centres (CfRadial 1.4 section
/// 4.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RangeRows {
    /// `range(range)`: one geometry for the volume.
    Volume,
    /// `range(sweep, range)`: one row per sweep.
    Sweep,
    /// `range(time, range)`: one row per ray (LROSE Radx writes this when
    /// the geometry varies); the rows of a sweep must agree.
    Ray,
}

/// `n_gates_vary` storage: where each ray's gates are in the `(n_points)`
/// field arrays (CfRadial 1.4 sections 2.3.1 and 4.5).
struct Ragged {
    /// `ray_n_gates`.
    counts: Vec<usize>,
    /// `ray_start_index`.
    starts: Vec<usize>,
    /// Length of the `n_points` dimension.
    points: usize,
}

impl Ragged {
    /// Read and check `ray_n_gates` and `ray_start_index`: whole numbers,
    /// every ray inside `n_points` and no longer than the `range`
    /// dimension (the gate centres there are).
    fn read(
        file: &NcFile<'_>,
        time_dim: usize,
        n_rays: usize,
        points: usize,
        n_gates: usize,
        budget: &mut DecodeBudget,
    ) -> Result<Self> {
        let mut read = |name: &str| -> Result<Vec<usize>> {
            let values =
                aligned_time_values(file, name, time_dim, n_rays, budget)?.ok_or_else(|| {
                    invalid(format!(
                        "CfRadial n_points storage without a {name}(time) variable"
                    ))
                })?;
            values
                .iter()
                .map(|value| {
                    (value.is_finite() && *value >= 0.0 && value.fract() == 0.0)
                        .then_some(*value as usize)
                        .ok_or_else(|| invalid(format!("CfRadial {name} value {value}")))
                })
                .collect()
        };
        let counts = read("ray_n_gates")?;
        let starts = read("ray_start_index")?;
        for (ray, (count, start)) in counts.iter().zip(&starts).enumerate() {
            if *count > n_gates || start.checked_add(*count).is_none_or(|end| end > points) {
                return Err(invalid(format!(
                    "CfRadial ray {ray}: {count} gates from n_points index {start} do not fit \
                     {points} points and {n_gates} range gates"
                )));
            }
        }
        Ok(Self {
            counts,
            starts,
            points,
        })
    }

    /// The gates of a sweep's rows: its longest ray.
    fn sweep_gates(&self, rays: std::ops::RangeInclusive<usize>) -> usize {
        self.counts
            .get(rays)
            .map_or(0, |counts| counts.iter().copied().max().unwrap_or(0))
    }

    /// The rays `rays` of the `(n_points)` array `raw` as rows of `ngates`,
    /// each padded after its `ray_n_gates` with the variable's
    /// `_FillValue` (else `missing_value`, else the netCDF default fill).
    /// Returns the rows and whether any gate was padded with the default
    /// fill (which the field's coding then has to name).
    fn gather(
        &self,
        raw: &NcArray,
        rays: std::ops::RangeInclusive<usize>,
        ngates: usize,
        var: &NcVar,
        budget: &mut DecodeBudget,
    ) -> Result<(NcArray, bool)> {
        let nrays = rays.clone().count();
        let len = nrays
            .checked_mul(ngates)
            .ok_or_else(|| invalid("CfRadial ragged sweep size overflow"))?;
        let word = match raw {
            NcArray::I8(_) | NcArray::Char(_) | NcArray::U8(_) => 1,
            NcArray::I16(_) | NcArray::U16(_) => 2,
            NcArray::I32(_) | NcArray::F32(_) | NcArray::U32(_) => 4,
            NcArray::F64(_) | NcArray::I64(_) | NcArray::U64(_) => 8,
            NcArray::Str(_) => return Ok((NcArray::Str(Vec::new()), false)),
        };
        budget
            .charge(len, word, "CfRadial ragged field rows")
            .map_err(CfRadialError::LimitExceeded)?;
        let stated = var
            .attr_f64("_FillValue")
            .or_else(|| var.attr_f64("missing_value"));
        let short = rays.clone().any(|ray| self.counts[ray] < ngates);
        let default_padded = short && stated.is_none();
        macro_rules! rows {
            ($variant:ident, $values:expr, $default:expr, $t:ty) => {{
                let pad: $t = match stated {
                    Some(value) => value as $t,
                    None => $default,
                };
                let mut out = Vec::with_capacity(len);
                for ray in rays {
                    let (start, count) = (self.starts[ray], self.counts[ray]);
                    let row = $values.get(start..start + count).ok_or_else(|| {
                        invalid(format!("CfRadial ray {ray} lies outside its field"))
                    })?;
                    out.extend_from_slice(row);
                    out.resize(out.len() + (ngates - count), pad);
                }
                NcArray::$variant(out)
            }};
        }
        let rows = match raw {
            NcArray::I8(values) => rows!(I8, values, -127, i8),
            NcArray::Char(values) => rows!(Char, values, 0, u8),
            NcArray::U8(values) => rows!(U8, values, u8::MAX, u8),
            NcArray::I16(values) => rows!(I16, values, -32767, i16),
            NcArray::U16(values) => rows!(U16, values, u16::MAX, u16),
            NcArray::I32(values) => rows!(I32, values, -2_147_483_647, i32),
            NcArray::U32(values) => rows!(U32, values, u32::MAX, u32),
            NcArray::I64(values) => rows!(I64, values, i64::MIN + 2, i64),
            NcArray::U64(values) => rows!(U64, values, u64::MAX - 1, u64),
            NcArray::F32(values) => rows!(F32, values, NC_FILL_FLOAT as f32, f32),
            NcArray::F64(values) => rows!(F64, values, NC_FILL_FLOAT, f64),
            NcArray::Str(_) => return Ok((NcArray::Str(Vec::new()), false)),
        };
        Ok((rows, default_padded))
    }
}

/// netCDF's default fill for `float` and `double` (`NC_FILL_FLOAT`,
/// `NC_FILL_DOUBLE`).
const NC_FILL_FLOAT: f64 = 9.969_209_968_386_869e36;

/// A field padded with the netCDF default fill ([`Ragged::gather`]) names
/// that fill as its `_FillValue` when the variable had none.
fn default_fill(data: &mut FieldData) {
    match data {
        FieldData::U8 { coding, .. } => {
            coding.fill_value.get_or_insert(u8::MAX);
        }
        FieldData::I8 { coding, .. } => {
            coding.fill_value.get_or_insert(-127);
        }
        FieldData::U16 { coding, .. } => {
            coding.fill_value.get_or_insert(u16::MAX);
        }
        FieldData::I16 { coding, .. } => {
            coding.fill_value.get_or_insert(-32767);
        }
        FieldData::I32 { coding, .. } => {
            coding.fill_value.get_or_insert(-2_147_483_647);
        }
        FieldData::F32 { coding, .. } => {
            coding.fill_value.get_or_insert(NC_FILL_FLOAT as f32);
        }
        FieldData::F64 { coding, .. } => {
            coding.fill_value.get_or_insert(NC_FILL_FLOAT);
        }
    }
}

/// `coord` cut to its first `ngates` gates.
fn truncated(coord: &RangeCoord, ngates: usize) -> RangeCoord {
    match coord {
        RangeCoord::Uniform {
            first_center_m,
            spacing_m,
            ..
        } => RangeCoord::Uniform {
            first_center_m: *first_center_m,
            spacing_m: *spacing_m,
            ngates: ngates as u32,
        },
        RangeCoord::Explicit { centers_m } => RangeCoord::Explicit {
            centers_m: centers_m.iter().take(ngates).copied().collect(),
        },
    }
}

/// The geometry of a `range(sweep, range)` or `range(time, range)` row
/// whose first `ngates` centres are the sweep's (later ones may be padding).
fn row_coordinate(row: &[f64], ngates: usize) -> Result<RangeCoord> {
    let take = ngates.max(2).min(row.len());
    if take < 2 || row[..take].iter().any(|centre| !centre.is_finite()) {
        return Err(invalid(
            "CfRadial range row without two finite gate centres",
        ));
    }
    Ok(truncated(
        &range_coordinate(&row[..take], take as u32),
        ngates,
    ))
}

/// Whether two `range(time, range)` rows give the same first `ngates`
/// centres (within float32 rounding).
fn same_centres(a: &[f64], b: &[f64], ngates: usize) -> bool {
    a.len() >= ngates
        && b.len() >= ngates
        && a[..ngates]
            .iter()
            .zip(&b[..ngates])
            .all(|(x, y)| (x - y).abs() <= 1e-3 * (1.0 + y.abs() * 1e-4))
}

/// A sweep's gate geometry from `ray_start_range` and `ray_gate_spacing`
/// (metres) when they give one first centre and spacing for its rays that
/// differs from `range` (a volume whose sweeps differ, stated per ray):
/// `Ok(None)` when they are absent, fill values or agree with `range`,
/// `Err` when the sweep's rays disagree.
fn ray_geometry(
    starts: Option<&[f64]>,
    spacings: Option<&[f64]>,
    rays: std::ops::RangeInclusive<usize>,
    range: &RangeCoord,
) -> std::result::Result<Option<(f64, f64)>, ()> {
    let (Some(starts), Some(spacings)) = (starts, spacings) else {
        return Ok(None);
    };
    let RangeCoord::Uniform {
        first_center_m,
        spacing_m,
        ..
    } = range
    else {
        return Ok(None);
    };
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-3 * spacing_m.max(1.0);
    let mut geometry: Option<(f64, f64)> = None;
    let mut differs = false;
    for ray in rays {
        let (Some(start), Some(spacing)) = (starts.get(ray), spacings.get(ray)) else {
            continue;
        };
        // Fill values (-9999, NaN) and nonsense say nothing.
        if !start.is_finite() || !spacing.is_finite() || *spacing <= 0.0 || *start <= -9999.0 {
            continue;
        }
        differs |= !close(*start, *first_center_m) || !close(*spacing, *spacing_m);
        match geometry {
            None => geometry = Some((*start, *spacing)),
            Some((a, b)) if close(a, *start) && close(b, *spacing) => {}
            Some(_) => return if differs { Err(()) } else { Ok(None) },
        }
    }
    Ok(geometry.filter(|_| differs))
}

/// The `range` coordinate: uniform when every centre sits on the line
/// through the first and last centres within 1% of a gate (float32 files
/// carry rounding of that order), else the explicit centres.
pub(crate) fn range_coordinate(range: &[f64], ngates: u32) -> RangeCoord {
    let first = range[0];
    let spacing = (range[range.len() - 1] - first) / (range.len() - 1) as f64;
    let uniform = spacing > 0.0
        && spacing.is_finite()
        && range.iter().enumerate().all(|(gate, center)| {
            (center - (first + gate as f64 * spacing)).abs() <= 1e-2 * spacing
        });
    if uniform {
        RangeCoord::Uniform {
            first_center_m: first,
            spacing_m: spacing,
            ngates,
        }
    } else {
        RangeCoord::Explicit {
            centers_m: range.iter().map(|value| *value as f32).collect(),
        }
    }
}

/// The rays `start..=end` of `sweep` from `sweep_start_ray_index` and
/// `sweep_end_ray_index`; a missing value means the first or last ray.
/// `None` (the sweep is skipped) when an index is not a non-negative integer
/// (fill values, garbage) or the sweep starts after it ends. An end past the
/// last ray is clamped to it, which keeps the rays a truncated `time`
/// dimension still holds.
fn sweep_ray_range(
    starts: &[f64],
    ends: &[f64],
    sweep: usize,
    n_rays: usize,
) -> Option<(usize, usize)> {
    let last_ray = n_rays.checked_sub(1)?;
    let index = |values: &[f64], missing: usize| match values.get(sweep) {
        None => Some(missing),
        // `as` saturates, and a start past the last ray fails `start <= end`.
        Some(value) => {
            (value.is_finite() && *value >= 0.0 && value.fract() == 0.0).then_some(*value as usize)
        }
    };
    let start = index(starts, 0)?;
    let end = index(ends, last_ray)?.min(last_ray);
    (start <= end).then_some((start, end))
}

/// CfRadial sweeps partition the rays: reject two sweeps whose ray ranges
/// overlap instead of duplicating the shared rows into both.
fn check_disjoint_sweeps(ray_ranges: &[Option<(usize, usize)>]) -> Result<()> {
    let mut by_start: Vec<(usize, usize, usize)> = ray_ranges
        .iter()
        .enumerate()
        .filter_map(|(sweep, range)| range.map(|(start, end)| (start, end, sweep)))
        .collect();
    by_start.sort_unstable();
    // Sorted by start, any overlap shows up between neighbours.
    for pair in by_start.windows(2) {
        if let [(_, previous_end, previous), (start, end, sweep)] = pair
            && start <= previous_end
        {
            return Err(invalid(format!(
                "CfRadial sweeps {} and {} overlap: sweep_start_ray_index/sweep_end_ray_index \
                 put rays {start}..={} in both",
                previous.min(sweep),
                previous.max(sweep),
                end.min(previous_end)
            )));
        }
    }
    Ok(())
}

pub(crate) fn numeric_at(values: &Option<Vec<f64>>, index: usize) -> Option<f64> {
    values
        .as_ref()?
        .get(index)
        .copied()
        .filter(|value| value.is_finite() && *value != -9999.0)
}

pub(crate) fn numeric_f32_at(values: &Option<Vec<f64>>, index: usize) -> Option<f32> {
    numeric_at(values, index)
        .filter(|value| value.abs() <= f32::MAX as f64)
        .map(|value| value as f32)
}

fn numeric_u8_at(values: &Option<Vec<f64>>, index: usize) -> Option<u8> {
    let value = numeric_at(values, index)?;
    (value.fract() == 0.0)
        .then(|| u8::try_from(value as i64).ok())
        .flatten()
}

fn numeric_u16_at(values: &Option<Vec<f64>>, index: usize) -> Option<u16> {
    let value = numeric_at(values, index)?;
    (value.fract() == 0.0)
        .then(|| u16::try_from(value as i64).ok())
        .flatten()
}

fn waveform_from_code(code: u8) -> Option<&'static str> {
    Some(match code {
        1 => "CS",
        2 => "CD/W",
        3 => "B",
        4 => "CD/WO",
        5 => "SZCS",
        6 => "SZCD",
        _ => return None,
    })
}

fn moment_coverage_from_code(code: u8) -> Option<&'static str> {
    Some(match code {
        1 => "surveillance",
        2 => "doppler",
        3 => "all",
        _ => return None,
    })
}

fn scan_leg_allows_quantity(leg: &ScanLeg, quantity: Quantity) -> bool {
    let doppler = matches!(
        quantity,
        Quantity::RadialVelocity | Quantity::DealiasedRadialVelocity | Quantity::SpectrumWidth
    );
    match leg.moment_coverage.as_deref() {
        Some("surveillance") => !doppler,
        Some("doppler") => doppler,
        _ => true,
    }
}

/// CfRadial's fixed angle is elevation for PPI sweeps and azimuth for RHI
/// sweeps. Azimuth needs a circular mean so a 359-degree/1-degree RHI points
/// north, rather than being mislabeled as 180 degrees when `fixed_angle` is
/// absent.
pub(crate) fn fallback_fixed_angle(rhi: bool, azimuth: &[f64], elevation: &[f64]) -> f64 {
    if rhi {
        circular_mean_degrees(azimuth)
            .or_else(|| azimuth.iter().copied().find(|value| value.is_finite()))
            .map(wrap_degrees)
            .unwrap_or(0.0)
    } else {
        arithmetic_mean(elevation).unwrap_or(0.0)
    }
}

fn circular_mean_degrees(values: &[f64]) -> Option<f64> {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    let mut count = 0usize;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        let radians = value.to_radians();
        sin_sum += radians.sin();
        cos_sum += radians.cos();
        count += 1;
    }
    if count == 0 || sin_sum.hypot(cos_sum) <= f64::EPSILON {
        return None;
    }
    Some(wrap_degrees(sin_sum.atan2(cos_sum).to_degrees()))
}

/// An angle in [0, 360) degrees. `rem_euclid` alone returns 360 for a
/// negative angle within rounding of 0 (-1e-40 + 360 rounds to 360), which
/// is the angle 0. A non-finite angle is returned as it is: it names no
/// direction to wrap (`rem_euclid` would turn an infinity into NaN).
pub(crate) fn wrap_degrees(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let wrapped = value.rem_euclid(360.0);
    if wrapped >= 360.0 { 0.0 } else { wrapped }
}

/// [`wrap_degrees`] in single precision.
pub(crate) fn wrap_degrees_f32(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let wrapped = value.rem_euclid(360.0);
    if wrapped >= 360.0 { 0.0 } else { wrapped }
}

/// A stored azimuth as the model's single-precision angle in [0, 360):
/// wrapped in double precision first, so a finite angle beyond the float
/// range still names its direction (cast first, it became infinite), then
/// again after the cast (359.99999999 rounds to 360 in single precision).
pub(crate) fn azimuth_f32(value: f64) -> f32 {
    wrap_degrees_f32(wrap_degrees(value) as f32)
}

fn arithmetic_mean(values: &[f64]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        sum += value;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

/// Read a numeric variable as f64, charging the widened array to `budget`
/// (an 8-bit variable grows eightfold).
pub(crate) fn read_f64s(
    file: &NcFile<'_>,
    name: &str,
    budget: &mut DecodeBudget,
) -> Result<Vec<f64>> {
    let raw = file.read_var(name)?;
    if raw.is_text() {
        return Err(invalid(format!("variable '{name}' is not numeric")));
    }
    let count = raw.len();
    budget
        .charge(count, size_of::<f64>(), "CfRadial numeric variable")
        .map_err(CfRadialError::LimitExceeded)?;
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        out.push(
            raw.get_f64(index)
                .ok_or_else(|| invalid(format!("variable '{name}' is not numeric")))?,
        );
    }
    Ok(out)
}

/// [`read_f64s`] for an optional variable: a missing or unusable variable is
/// `None`, but exceeding a resource limit is still an error.
pub(crate) fn optional_f64s(
    file: &NcFile<'_>,
    name: &str,
    budget: &mut DecodeBudget,
) -> Result<Option<Vec<f64>>> {
    match read_f64s(file, name, budget) {
        Ok(values) => Ok(Some(values)),
        Err(error @ CfRadialError::LimitExceeded(_)) => Err(error),
        Err(_) => Ok(None),
    }
}

/// Read a numeric variable only when it is exactly aligned to the CfRadial
/// `time` dimension. A scalar, sweep-level value, or malformed short array is
/// not silently broadcast across rays.
pub(crate) fn aligned_time_values(
    file: &NcFile<'_>,
    name: &str,
    time_dim: usize,
    n_rays: usize,
    budget: &mut DecodeBudget,
) -> Result<Option<Vec<f64>>> {
    let Some(var) = file.vars.get(name) else {
        return Ok(None);
    };
    if var.dim_ids.as_slice() != [time_dim] {
        return Ok(None);
    }
    Ok(optional_f64s(file, name, budget)?.filter(|values| values.len() == n_rays))
}

/// Per-ray platform position and attitude of a moving platform
/// (`latitude(time)`, ...). `None` for a fixed platform.
fn read_platform_track(
    file: &NcFile<'_>,
    time_dim: usize,
    n_rays: usize,
    budget: &mut DecodeBudget,
) -> Result<Option<PlatformTrack>> {
    let (Some(latitude), Some(longitude), Some(altitude)) = (
        aligned_time_values(file, "latitude", time_dim, n_rays, budget)?,
        aligned_time_values(file, "longitude", time_dim, n_rays, budget)?,
        aligned_time_values(file, "altitude", time_dim, n_rays, budget)?,
    ) else {
        return Ok(None);
    };
    let angles = |name: &str, budget: &mut DecodeBudget| -> Result<Option<Vec<f32>>> {
        Ok(aligned_time_values(file, name, time_dim, n_rays, budget)?
            .map(|values| values.into_iter().map(|v| v as f32).collect()))
    };
    Ok(Some(PlatformTrack {
        latitude_deg: latitude,
        longitude_deg: longitude,
        altitude_m: altitude,
        altitude_agl_m: aligned_time_values(file, "altitude_agl", time_dim, n_rays, budget)?,
        heading_deg: angles("heading", budget)?,
        roll_deg: angles("roll", budget)?,
        pitch_deg: angles("pitch", budget)?,
        drift_deg: angles("drift", budget)?,
        rotation_deg: angles("rotation", budget)?,
        tilt_deg: angles("tilt", budget)?,
    }))
}

pub(crate) trait SliceRays {
    fn slice(&self, rays: std::ops::RangeInclusive<usize>) -> Self;
}

impl SliceRays for PlatformTrack {
    fn slice(&self, rays: std::ops::RangeInclusive<usize>) -> Self {
        let take_f64 = |values: &Vec<f64>| values[rays.clone()].to_vec();
        let take_f32 =
            |values: &Option<Vec<f32>>| values.as_ref().map(|v| v[rays.clone()].to_vec());
        Self {
            latitude_deg: take_f64(&self.latitude_deg),
            longitude_deg: take_f64(&self.longitude_deg),
            altitude_m: take_f64(&self.altitude_m),
            altitude_agl_m: self
                .altitude_agl_m
                .as_ref()
                .map(|v| v[rays.clone()].to_vec()),
            heading_deg: take_f32(&self.heading_deg),
            roll_deg: take_f32(&self.roll_deg),
            pitch_deg: take_f32(&self.pitch_deg),
            drift_deg: take_f32(&self.drift_deg),
            rotation_deg: take_f32(&self.rotation_deg),
            tilt_deg: take_f32(&self.tilt_deg),
        }
    }
}

/// Charge the attributes kept verbatim to the decode budget by the memory
/// they hold (integers widen to 64 bits, and a netCDF-4 string array holds
/// one box per element).
pub(crate) fn charge_passthrough(volume: &Volume, budget: &mut DecodeBudget) -> Result<()> {
    fn list(attrs: &[(Box<str>, AttrValue)]) -> usize {
        attrs
            .iter()
            .map(|(name, value)| name.len().saturating_add(attr_bytes(value)))
            .fold(0usize, usize::saturating_add)
    }
    let mut bytes = list(&volume.attrs.other);
    for entry in &volume.variable_attrs {
        bytes = bytes.saturating_add(list(&entry.attrs));
    }
    let extras = volume
        .extra_vars
        .iter()
        .chain(volume.sweeps.iter().flat_map(|sweep| &sweep.extra_vars));
    for extra in extras {
        bytes = bytes.saturating_add(list(&extra.attrs));
    }
    for sweep in &volume.sweeps {
        bytes = bytes.saturating_add(list(&sweep.other));
        for field in &sweep.fields {
            bytes = bytes.saturating_add(list(&field.attrs.other));
        }
    }
    budget
        .charge(1, bytes, "CfRadial attributes")
        .map_err(CfRadialError::LimitExceeded)
}

/// Heap bytes an attribute value holds (a text element also costs its box).
fn attr_bytes(value: &AttrValue) -> usize {
    const BOX: usize = size_of::<Box<str>>();
    match value {
        AttrValue::Text(text) => text.len(),
        AttrValue::Bool(_) | AttrValue::Scalar(_) => 0,
        AttrValue::Array(ArrayBuf::Text(texts)) => texts
            .iter()
            .map(|text| text.len().saturating_add(BOX))
            .fold(0usize, usize::saturating_add),
        AttrValue::Array(array) => {
            let width = match array {
                ArrayBuf::I8(_) | ArrayBuf::U8(_) => 1,
                ArrayBuf::I16(_) | ArrayBuf::U16(_) => 2,
                ArrayBuf::I32(_) | ArrayBuf::U32(_) | ArrayBuf::F32(_) => 4,
                _ => 8,
            };
            array.len().saturating_mul(width)
        }
    }
}

/// Names of the extra variables of `volume` and of its sweeps.
pub(crate) fn kept_names(volume: &Volume) -> std::collections::BTreeSet<String> {
    volume
        .extra_vars
        .iter()
        .chain(volume.sweeps.iter().flat_map(|sweep| &sweep.extra_vars))
        .map(|extra| extra.name.to_string())
        .collect()
}

/// The attributes of the variables of `file` that `kept` does not claim (a
/// field or an extra variable keeps its own): their values are in typed
/// slots, their attributes are kept verbatim as `VariableAttrs` of `group`,
/// in file order.
pub(crate) fn slotted_variable_attrs(
    file: &NcFile<'_>,
    group: &str,
    kept: &dyn Fn(&NcVar) -> bool,
) -> Vec<VariableAttrs> {
    let mut vars: Vec<&NcVar> = file
        .vars
        .values()
        .filter(|var| !var.attrs.is_empty() && !kept(var))
        .collect();
    vars.sort_by_key(|var| var.index);
    vars.into_iter()
        .map(|var| VariableAttrs {
            group: group.into(),
            name: var.name.as_str().into(),
            attrs: var
                .attrs
                .iter()
                .map(|(name, value)| (name.as_str().into(), attr_value(value)))
                .collect(),
        })
        .collect()
}

/// A variable without a typed slot, verbatim: the whole array (root
/// variables) or the rows of `rays` (per-ray variables, first dimension
/// `time` in the model whatever the file calls it). A `char` array becomes
/// one text per row (its last dimension is the string length).
pub(crate) fn extra_variable(
    file: &NcFile<'_>,
    var: &NcVar,
    rays: Option<std::ops::RangeInclusive<usize>>,
) -> Option<ExtraVariable> {
    let dims = file.var_dims(var);
    let array = file.read_var(&var.name).ok()?;
    let mut dim_names: Vec<Box<str>> = (0..dims.len())
        .map(|axis| file.dim_name(var, axis).unwrap_or_default().into())
        .collect();
    let mut shape: Vec<u32> = dims
        .iter()
        .map(|len| u32::try_from(*len).unwrap_or(u32::MAX))
        .collect();
    let range = match &rays {
        Some(rays) => {
            let row = dims.iter().skip(1).product::<usize>().max(1);
            dim_names[0] = "time".into();
            shape[0] = u32::try_from(rays.end() - rays.start() + 1).ok()?;
            *rays.start() * row..(*rays.end() + 1) * row
        }
        None => 0..array.len(),
    };
    if range.end > array.len() {
        return None;
    }
    let values = match &array {
        NcArray::Char(chars) => {
            // One string per row of the last (string length) dimension.
            let width = dims.last().copied().unwrap_or(chars.len()).max(1);
            dim_names.pop();
            shape.pop();
            ArrayBuf::Text(
                chars[range]
                    .chunks(width)
                    .map(|chunk| {
                        let text = chunk.split(|byte| *byte == 0).next().unwrap_or_default();
                        String::from_utf8_lossy(text).into()
                    })
                    .collect(),
            )
        }
        other => slice_buf(other, range)?,
    };
    Some(ExtraVariable {
        name: var.name.as_str().into(),
        dims: dim_names,
        shape,
        values,
        attrs: var
            .attrs
            .iter()
            .map(|(name, value)| (name.as_str().into(), attr_value(value)))
            .collect(),
    })
}

pub(crate) fn attr_value(value: &NcValue) -> AttrValue {
    match value {
        NcValue::Str(text) => AttrValue::Text(text.as_str().into()),
        NcValue::Strings(texts) => AttrValue::Array(ArrayBuf::Text(
            texts.iter().map(|t| t.as_str().into()).collect(),
        )),
        NcValue::Floats(values) => match values.as_slice() {
            [single] => AttrValue::Scalar(Scalar::F32(*single)),
            _ => AttrValue::Array(ArrayBuf::F32(values.clone())),
        },
        NcValue::Doubles(values) => match values.as_slice() {
            [single] => AttrValue::Scalar(Scalar::F64(*single)),
            _ => AttrValue::Array(ArrayBuf::F64(values.clone())),
        },
        NcValue::Ints(values, kind) => int_attr_value(values, *kind),
    }
}

/// Integer attribute values in their stored type (`uint64` as `int64`: the
/// model has no unsigned 64-bit arrays, and every value fits).
fn int_attr_value(values: &[i64], kind: IntKind) -> AttrValue {
    macro_rules! typed {
        ($ty:ty, $scalar:ident, $array:ident) => {
            match values {
                [single] => AttrValue::Scalar(Scalar::$scalar(*single as $ty)),
                _ => AttrValue::Array(ArrayBuf::$array(
                    values.iter().map(|value| *value as $ty).collect(),
                )),
            }
        };
    }
    match kind {
        IntKind::I8 => typed!(i8, I8, I8),
        IntKind::U8 => typed!(u8, U8, U8),
        IntKind::I16 => typed!(i16, I16, I16),
        IntKind::U16 => typed!(u16, U16, U16),
        IntKind::I32 => typed!(i32, I32, I32),
        IntKind::U32 => typed!(u32, U32, U32),
        IntKind::U64 => match values {
            [single] => AttrValue::Scalar(Scalar::U64(*single as u64)),
            _ => AttrValue::Array(ArrayBuf::I64(values.to_vec())),
        },
        IntKind::I64 => typed!(i64, I64, I64),
    }
}

/// `name(sweep, string_length)` char matrix (or netCDF-4 `name(sweep)`
/// strings) → per-sweep strings.
fn read_sweep_strings(file: &NcFile<'_>, name: &str, sweep_count: usize) -> Vec<Option<String>> {
    let Some(var) = file.vars.get(name) else {
        return vec![None; sweep_count];
    };
    let dims = file.var_dims(var);
    match (file.read_var(name), dims.as_slice()) {
        (Ok(NcArray::Char(chars)), [rows, width]) => (0..sweep_count)
            .map(|sweep| {
                if sweep >= *rows || (sweep + 1) * width > chars.len() {
                    return None;
                }
                Some(text_of(&chars[sweep * width..(sweep + 1) * width]))
            })
            .collect(),
        (Ok(NcArray::Str(strings)), [_]) => (0..sweep_count)
            .map(|sweep| strings.get(sweep).map(|text| text.trim().to_owned()))
            .collect(),
        _ => vec![None; sweep_count],
    }
}

/// A scalar char variable (or netCDF-4 string) as text.
pub(crate) fn char_var_text(file: &NcFile<'_>, name: &str) -> Option<String> {
    match file.read_var(name).ok()? {
        NcArray::Char(chars) => Some(text_of(&chars)),
        NcArray::Str(strings) => strings.first().map(|text| text.trim().to_owned()),
        _ => None,
    }
}

pub(crate) fn parse_bool(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

fn gattr_bool(file: &NcFile<'_>, name: &str) -> Option<bool> {
    match file.gattrs.get(name)? {
        NcValue::Str(text) => parse_bool(text),
        other => other.as_f64().map(|value| value != 0.0),
    }
}

pub(crate) fn units_of<'f>(file: &'f NcFile<'_>, name: &str) -> Option<&'f str> {
    file.vars.get(name).and_then(|var| var.attr_str("units"))
}

/// The reference instant of `time.units` ("seconds since <ISO 8601>").
pub(crate) fn time_units_reference(file: &NcFile<'_>) -> Option<DateTime<Utc>> {
    let units = units_of(file, "time")?;
    let (_, rest) = units.trim().split_once("since")?;
    parse_iso_instant(rest)
}

pub(crate) fn parse_time_var_or_attr(file: &NcFile<'_>, name: &str) -> Option<DateTime<Utc>> {
    // Either a char (or netCDF-4 string) variable or a global attribute,
    // ISO 8601 "...Z".
    let text = match file.read_var(name) {
        Ok(NcArray::Char(chars)) => text_of(&chars),
        Ok(NcArray::Str(strings)) => strings.first()?.clone(),
        _ => file.gattr_str(name)?.to_owned(),
    };
    parse_iso_instant(&text)
}

/// An ISO 8601 instant: `YYYY-MM-DDTHH:MM:SS[.fff]` (or a space instead of
/// `T`), UTC when it ends in `Z` or has no offset, else converted from its
/// `+HH:MM` offset.
pub(crate) fn parse_iso_instant(text: &str) -> Option<DateTime<Utc>> {
    let trimmed = text.trim().trim_end_matches('Z').trim();
    for format in [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Some(Utc.from_utc_datetime(&naive));
        }
    }
    for format in [
        "%Y-%m-%dT%H:%M:%S%:z",
        "%Y-%m-%d %H:%M:%S%:z",
        "%Y-%m-%dT%H:%M:%S%.f%:z",
        "%Y-%m-%d %H:%M:%S%.f%:z",
    ] {
        if let Ok(instant) = DateTime::parse_from_str(trimmed, format) {
            return Some(instant.with_timezone(&Utc));
        }
    }
    None
}

/// Operating frequencies in Hz: the `frequency` coordinate variable (values
/// above 1 MHz verbatim; MHz or GHz spellings normalized), else the
/// historical global-attribute spellings.
pub(crate) fn cfradial_frequency_hz(file: &NcFile<'_>) -> Vec<f64> {
    if let Ok(values) = file.read_var("frequency") {
        let hz: Vec<f64> = (0..values.len())
            .filter_map(|index| values.get_f64(index))
            .filter(|value| value.is_finite() && *value > 0.0)
            .filter_map(|value| {
                if value > 1.0e6 {
                    Some(value)
                } else {
                    normalize_frequency_mhz(value).map(|mhz| f64::from(mhz) * 1e6)
                }
            })
            .collect();
        if !hz.is_empty() {
            return hz;
        }
    }
    for name in [
        "radar_frequency",
        "frequency",
        "frequency_ghz",
        "instrument_frequency",
    ] {
        if let Some(value) = file.gattr_f64(name)
            && let Some(mhz) = normalize_frequency_mhz(value)
        {
            return vec![f64::from(mhz) * 1e6];
        }
    }
    for name in ["radar_wavelength", "radar_wavelength_cm", "wavelength"] {
        if let Some(value) = file.gattr_f64(name)
            && let Some(mhz) = frequency_mhz_from_wavelength(value)
        {
            return vec![f64::from(mhz) * 1e6];
        }
    }
    Vec::new()
}

pub(crate) fn cfradial_beam_width_deg(file: &NcFile<'_>, names: &[&str]) -> Option<f32> {
    for name in names {
        if let Some(value) = numeric_var_first(file, name).or_else(|| file.gattr_f64(name)) {
            let units = file
                .vars
                .get(*name)
                .and_then(|var| var.attr_str("units"))
                .unwrap_or("degrees")
                .to_ascii_lowercase();
            let degrees = if units.contains("rad") {
                value.to_degrees()
            } else {
                value
            };
            if degrees.is_finite() && degrees > 0.0 && degrees <= 180.0 {
                return Some(degrees as f32);
            }
        }
    }
    None
}

fn cfradial_pulse_width_us(file: &NcFile<'_>) -> Option<f32> {
    if let Some(seconds) = time_var_seconds(file, "pulse_width") {
        return positive_f32(seconds * 1.0e6);
    }
    file.gattr_f64("pulse_width_us")
        .and_then(positive_f32)
        .or_else(|| {
            file.gattr_f64("pulse_width")
                .and_then(|seconds| positive_f32(seconds * 1.0e6))
        })
}

fn cfradial_prt_s(file: &NcFile<'_>) -> Option<f32> {
    // A time-aligned `prt(time)` belongs to each ray, not to the volume.
    // Only a true scalar variable participates in this volume-level
    // fallback.
    if let Some(value) = numeric_scalar_var_first(file, "prt") {
        let scale = time_units_scale(file.vars.get("prt").and_then(|var| var.attr_str("units")));
        return positive_f32(value * scale);
    }
    file.gattr_f64("prt_s")
        .or_else(|| file.gattr_f64("prt"))
        .and_then(positive_f32)
}

fn cfradial_unambiguous_range_km(file: &NcFile<'_>) -> Option<f32> {
    // As with PRT, do not collapse varying per-ray values to the first ray.
    if let Some(value) = numeric_scalar_var_first(file, "unambiguous_range") {
        let scale = range_units_to_km_scale(
            file.vars
                .get("unambiguous_range")
                .and_then(|var| var.attr_str("units")),
        );
        return positive_f32(value * scale);
    }
    file.gattr_f64("unambiguous_range_km")
        .and_then(positive_f32)
        .or_else(|| {
            file.gattr_f64("unambiguous_range")
                .and_then(|meters| positive_f32(meters / 1000.0))
        })
}

/// The `r_calib_*(r_calib)` variables as one [`RadarCalibration`] per
/// `r_calib` entry (design note 12.4: `r_calib_<name>` is Table 301-14a
/// `<name>`, `r_calib_base_dbz_1km_*` is `base_1km_*`; every other name goes
/// to `RadarCalibration::extra` verbatim). A value equal to the variable's
/// `_FillValue` (or not finite) is left unset. `r_calib_time` (ISO text or
/// seconds) becomes seconds since the volume time reference. `r_calib_index`
/// is a per-ray variable and is read with the ray variables. Empty when the
/// file has no `r_calib` dimension.
fn read_radar_calibration(
    file: &NcFile<'_>,
    time_reference: DateTime<Utc>,
) -> Vec<RadarCalibration> {
    let Some(&(_, count)) = file.dims.iter().find(|(name, _)| name == "r_calib") else {
        return Vec::new();
    };
    if count == 0 || count > 4096 {
        return Vec::new();
    }
    let mut entries = vec![RadarCalibration::default(); count];
    for (name, var) in &file.vars {
        let Some(suffix) = name.strip_prefix("r_calib_") else {
            continue;
        };
        if suffix == "index"
            || var.dim_ids.first().map(|dim| file.dims[*dim].0.as_str()) != Some("r_calib")
        {
            continue;
        }
        let Ok(array) = file.read_var(name) else {
            continue;
        };
        if suffix == "time" {
            match &array {
                NcArray::Str(strings) => {
                    for (entry, text) in entries.iter_mut().zip(strings) {
                        entry.time_s = parse_iso_instant(text).map(|instant| {
                            (instant - time_reference).num_milliseconds() as f64 / 1000.0
                        });
                    }
                }
                NcArray::Char(chars) => {
                    let width = chars.len() / count;
                    for (entry, text) in entries.iter_mut().zip(chars.chunks(width.max(1))) {
                        let text = text.split(|byte| *byte == 0).next().unwrap_or_default();
                        entry.time_s = parse_iso_instant(String::from_utf8_lossy(text).trim()).map(
                            |instant| (instant - time_reference).num_milliseconds() as f64 / 1000.0,
                        );
                    }
                }
                _ => {
                    let scale = time_units_scale(var.attr_str("units"));
                    for (index, entry) in entries.iter_mut().enumerate() {
                        entry.time_s = array
                            .get_f64(index)
                            .filter(|value| value.is_finite())
                            .map(|value| value * scale);
                    }
                }
            }
            continue;
        }
        let fill = var.attr_f64("_FillValue");
        let table_name = match suffix {
            "base_dbz_1km_hc" => "base_1km_hc",
            "base_dbz_1km_vc" => "base_1km_vc",
            "base_dbz_1km_hx" => "base_1km_hx",
            "base_dbz_1km_vx" => "base_1km_vx",
            other => other,
        };
        for (index, entry) in entries.iter_mut().enumerate() {
            let value = array
                .get_f64(index)
                .filter(|value| value.is_finite() && Some(*value) != fill);
            let Some(value) = value else {
                continue;
            };
            if !entry.set_float_entry(table_name, Some(value as f32)) {
                entry
                    .extra
                    .push((suffix.into(), AttrValue::Scalar(Scalar::F32(value as f32))));
            }
        }
    }
    entries
}

pub(crate) fn numeric_var_first(file: &NcFile<'_>, name: &str) -> Option<f64> {
    let values = file.read_var(name).ok()?;
    (0..values.len()).find_map(|index| {
        values
            .get_f64(index)
            .filter(|value| value.is_finite() && *value > 0.0)
    })
}

fn numeric_scalar_var_first(file: &NcFile<'_>, name: &str) -> Option<f64> {
    file.vars.get(name)?.dim_ids.is_empty().then_some(())?;
    numeric_var_first(file, name)
}

fn time_var_seconds(file: &NcFile<'_>, name: &str) -> Option<f64> {
    let value = numeric_var_first(file, name)?;
    let units = file
        .vars
        .get(name)
        .and_then(|var| var.attr_str("units"))
        .unwrap_or("seconds")
        .trim()
        .to_ascii_lowercase();
    let seconds = if units.contains("microsecond") || matches!(units.as_str(), "us" | "µs") {
        value * 1.0e-6
    } else if units.contains("millisecond") || units == "ms" {
        value * 1.0e-3
    } else {
        value
    };
    seconds.is_finite().then_some(seconds)
}

pub(crate) fn time_units_scale(units: Option<&str>) -> f64 {
    let units = units.unwrap_or("seconds").trim().to_ascii_lowercase();
    if units.contains("microsecond") || matches!(units.as_str(), "us" | "µs") {
        1.0e-6
    } else if units.contains("millisecond") || units == "ms" {
        1.0e-3
    } else {
        1.0
    }
}

fn range_units_to_km_scale(units: Option<&str>) -> f64 {
    let units = units.unwrap_or("meters").trim().to_ascii_lowercase();
    if units.contains("kilometer") || units.contains("kilometre") || units == "km" {
        1.0
    } else {
        1.0e-3
    }
}

fn range_units_to_m_scale(units: Option<&str>) -> f64 {
    range_units_to_km_scale(units) * 1000.0
}

pub(crate) fn positive_f32(value: f64) -> Option<f32> {
    (value.is_finite() && value > 0.0 && value <= f32::MAX as f64).then_some(value as f32)
}

pub(crate) fn metadata_text(file: &NcFile<'_>, name: &str) -> Option<String> {
    file.gattr_str(name)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn normalize_frequency_mhz(value: f64) -> Option<u32> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let mhz = if value > 1.0e6 {
        value / 1.0e6
    } else if value > 1000.0 {
        value
    } else {
        value * 1000.0
    };
    (1000.0..=12_000.0)
        .contains(&mhz)
        .then_some(mhz.round() as u32)
}

fn frequency_mhz_from_wavelength(value: f64) -> Option<u32> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let meters = if value > 1.0 { value / 100.0 } else { value };
    let mhz = 299.792_458 / meters;
    (1000.0..=12_000.0)
        .contains(&mhz)
        .then_some(mhz.round() as u32)
}

pub(crate) fn invalid(reason: impl Into<String>) -> CfRadialError {
    CfRadialError::InvalidMessage {
        offset: 0,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A negative azimuth within rounding of 0 wrapped to 360 (the `writers`
    /// fuzz target: an ODIM ray at -8.6e-41 degrees read back from CfRadial
    /// as 360); it is 0.
    #[test]
    fn angles_wrap_into_0_to_360() {
        assert_eq!(wrap_degrees_f32(-8.6e-41), 0.0);
        assert_eq!(wrap_degrees(-1e-300), 0.0);
        assert_eq!(wrap_degrees_f32(-0.5), 359.5);
        assert_eq!(wrap_degrees(360.0), 0.0);
        assert_eq!(wrap_degrees(725.0), 5.0);
    }

    /// An azimuth beyond the float range reads as its direction, not as an
    /// infinity that a later wrap turns into NaN; a stored infinity or NaN
    /// stays what it is.
    #[test]
    fn azimuths_wrap_before_narrowing_and_keep_non_finite_values() {
        assert_eq!(azimuth_f32(1.0e300), 1.0e300_f64.rem_euclid(360.0) as f32);
        assert!(azimuth_f32(1.0e300).is_finite());
        assert_eq!(azimuth_f32(-1.0e-300), 0.0);
        assert_eq!(azimuth_f32(359.999_999_99), 0.0);
        assert_eq!(azimuth_f32(f64::INFINITY), f32::INFINITY);
        assert_eq!(azimuth_f32(f64::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(azimuth_f32(f64::NAN).is_nan());
        assert_eq!(wrap_degrees_f32(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn real_volume_exceeding_the_output_budget_is_rejected() {
        let path = recast_radar_testdata::path("cfrad1-irene-sr2-20110827-120420-sur-sweeps01")
            .unwrap_or_else(|e| panic!("{e}"));
        let bytes = std::fs::read(path).expect("read committed CfRadial file");
        read_cfradial1_volume_within(&bytes, DecodeBudget::volume())
            .expect("real file fits the default budget");
        // The int8 fields alone (719 rays x 1,107 gates x two fields) take
        // 1.5 MiB on top of the widened coordinates; a 1 MiB budget must fail
        // cleanly.
        let Err(error) = read_cfradial1_volume_within(&bytes, DecodeBudget::new(1 << 20)) else {
            panic!("1 MiB budget must fail");
        };
        assert!(
            matches!(&error, CfRadialError::LimitExceeded(reason) if reason.contains("limit")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rhi_fixed_angle_fallback_uses_wrap_aware_azimuth_mean() {
        let fixed = fallback_fixed_angle(true, &[359.0, 0.0, 1.0], &[10.0, 20.0, 30.0]);
        assert!(!(0.01..=359.99).contains(&fixed), "fixed angle was {fixed}");
    }

    #[test]
    fn ppi_fixed_angle_fallback_still_uses_mean_elevation() {
        let fixed = fallback_fixed_angle(false, &[80.0, 90.0, 100.0], &[0.4, 0.5, 0.6]);
        assert!((fixed - 0.5).abs() < 1.0e-9);
    }

    #[test]
    fn range_coordinate_detects_uniform_spacing() {
        assert_eq!(
            range_coordinate(&[0.0, 75.0, 150.0, 225.0], 4),
            RangeCoord::Uniform {
                first_center_m: 0.0,
                spacing_m: 75.0,
                ngates: 4
            }
        );
        // float32 centres of a 124.913025 m spacing, as DOW8 stores them.
        let dow8: Vec<f64> = (0..950)
            .map(|gate| f64::from((62.456512_f64 + f64::from(gate) * 124.913025) as f32))
            .collect();
        let RangeCoord::Uniform { spacing_m, .. } = range_coordinate(&dow8, 950) else {
            panic!("float32 rounding must still read as uniform");
        };
        assert!((spacing_m - 124.913025).abs() < 1e-4);
        assert!(matches!(
            range_coordinate(&[0.0, 75.0, 200.0], 3),
            RangeCoord::Explicit { .. }
        ));
    }

    #[test]
    fn time_units_reference_parses_cf_epoch() {
        assert_eq!(
            parse_iso_instant(" 2011-08-27T12:04:20Z"),
            Some(Utc.with_ymd_and_hms(2011, 8, 27, 12, 4, 20).unwrap())
        );
    }
}
