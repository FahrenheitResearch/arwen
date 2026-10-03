//! CfRadial 2 (FM 301) decoder: the netCDF-4 group layout.
//!
//! Format references: WMO FM 301 (CF-Radial 2.x as adopted by WMO, 2022
//! draft) and NCAR/EOL "CfRadial Data File Format, version 2.0" (M. Dixon,
//! 2017). A CfRadial 2 file is netCDF-4 with:
//!
//! - global attributes as in CfRadial 1;
//! - root variables `volume_number`, `platform_type`, `instrument_type`,
//!   `primary_axis`, `time_coverage_start`/`_end`, `latitude`, `longitude`,
//!   `altitude` (per ray, dimension `time`, on a moving platform),
//!   `sweep_group_name(sweep)` and `sweep_fixed_angle(sweep)`;
//! - optional groups `radar_parameters` (Table 301-12, with the `frequency`
//!   coordinate), `radar_calibration` (Table 301-14, one entry per element
//!   of an `r_calib`/`calib` dimension, or scalars for one entry) and
//!   `georeferencing_correction`;
//! - one group per sweep (`sweep_0`, `sweep_0001`, ...) holding what a
//!   CfRadial 1 file holds for one sweep: `time`, `range`, `azimuth`,
//!   `elevation`, the Table 301-8a per-ray variables, the sweep scalars
//!   (`sweep_number`, `sweep_mode`, `sweep_fixed_angle` or `fixed_angle`,
//!   `follow_mode`, `prt_mode`, `polarization_mode`, `rays_are_indexed`,
//!   `ray_angle_res`, `target_scan_rate`, ...), the fields `(time, range)`,
//!   and a `monitoring` subgroup of per-ray variables.
//!
//! The files the CfRadial 2 writers produce differ, and all read here:
//! Radx (`Cf2RadxFile`) names sweep groups `sweep_0001` (1-based), writes
//! text as netCDF-4 strings, puts `scan_rate` and the measured transmit
//! powers in `monitoring` and the calibration along `r_calib`; xradar names
//! them `sweep_0`, and its current writer calls the ray dimension `azimuth`
//! (a PPI) and puts a moving platform's track at the root; this crate's
//! FM301-2022 writer ([`crate::write_cfradial2`]) uses Table 301-12a's
//! `radar_parameters` names without the `radar_` prefix, a `frequency`
//! coordinate in every sweep group, `calib_index` per ray and in
//! `radar_calibration`, and every Table 301-11 variable in `monitoring`.
//!
//! [`read_cfradial2_volume`] builds the FM301 model the way
//! [`crate::read_cfradial1_volume`] does for CfRadial 1 (same field,
//! packing, coordinate and passthrough rules), one sweep per group in
//! `sweep_group_name` order (else by the number in the group name), each
//! sweep with its own range coordinate and its own field packing. Ray times
//! are seconds since the volume reference: the first sweep's `time.units`
//! reference, else `time_coverage_start`; each sweep's own reference is
//! applied. What has no typed slot is kept verbatim: sweep scalars and
//! per-ray variables in `Sweep::extra_vars`, sweep group attributes in
//! `Sweep::other` (the `monitoring` group's as `monitoring.<name>`), a file
//! `sweep_number` that is not the sweep's index and a source group name
//! that is not `sweep_<index>` (`sweep_group_name`, Radx's `sweep_0001`) in
//! `Sweep::other`, root variables and unknown variables of the metadata
//! groups in `Volume::extra_vars` (`<group>.<name>` when the name is taken;
//! calibration variables that are neither numbers nor the entry time too),
//! metadata group attributes in `Volume::attrs.other` as `<group>.<name>`,
//! any other group (a root group's or a sweep's, with its subgroups) as
//! `<group>.<name>` variables and attributes of its level, and the
//! attributes of every variable a typed slot holds in
//! `Volume::variable_attrs` under its FM301 group (`sweep_<index>`,
//! `sweep_<index>/monitoring`, `radar_parameters`, ...).

use std::collections::BTreeSet;

use chrono::{DateTime, Utc};
use recast_radar_core::bounded_read::{DecodeBudget, check_gate_count, check_sweep_count};
use recast_radar_core::model::{
    AttrValue, ExtraVariable, FollowMode, GeoreferencingCorrection, PolarizationMode, PrtMode,
    RadarCalibration, RadarParameters, RangeCoord, Scalar, SourceFormat, Sweep, SweepMode, Volume,
};
use recast_radar_hdf5::netcdf4::NcFile as Nc4File;

use crate::cfradial::{
    RayData, SLOTTED_RAY_VARS, SLOTTED_ROOT_VARS, aligned_time_values, attr_value, build_field,
    cfradial_beam_width_deg, char_var_text, charge_passthrough, describe_volume, extra_variable,
    fallback_fixed_angle, invalid, note_netcdf4_container, numeric_var_first, parse_bool,
    parse_iso_instant, parse_time_var_or_attr, radar_parameters, range_coordinate, read_f64s,
    slotted_variable_attrs, time_units_reference, time_units_scale,
};
use crate::netcdf::NcFile;
use crate::netcdf3::{NcArray, NcVar};
use crate::{CfRadialError, Result};

/// Decode a CfRadial 2 (netCDF-4 group layout) byte buffer into the FM301
/// model.
pub fn read_cfradial2_volume(bytes: &[u8]) -> Result<Volume> {
    let file = Nc4File::open(bytes)?;
    decode(&file, DecodeBudget::volume())
}

/// True when a netCDF-4 file uses the CfRadial 2 group layout: a root
/// `sweep_group_name` variable, or a root child group named `sweep_<n>`.
pub fn is_cfradial2(file: &Nc4File<'_>) -> bool {
    file.root().variable("sweep_group_name").is_some()
        || file
            .root()
            .groups
            .iter()
            .any(|path| sweep_number_of(path.trim_start_matches('/')).is_some())
}

/// Sweep scalars and coordinates of a sweep group with a typed slot.
const SLOTTED_SWEEP_VARS: &[&str] = &[
    "time",
    "range",
    "azimuth",
    "elevation",
    "sweep_number",
    "sweep_mode",
    "sweep_fixed_angle",
    "fixed_angle",
    "follow_mode",
    "prt_mode",
    "polarization_mode",
    "rays_are_indexed",
    "ray_angle_res",
    "rays_angle_resolution",
    "target_scan_rate",
    "qc_procedures",
    "latitude",
    "longitude",
    "altitude",
    "altitude_agl",
];

/// Root groups with a typed home in the model.
const METADATA_GROUPS: &[&str] = &[
    "radar_parameters",
    "radar_calibration",
    "georeferencing_correction",
    "georeference_correction",
];

/// `sweep_0`, `sweep_0001`, ... → the number.
fn sweep_number_of(name: &str) -> Option<u32> {
    name.strip_prefix("sweep_")?.parse().ok()
}

/// The sweep group paths in volume order: `sweep_group_name` when every
/// name it lists is a root group, else every `sweep_<n>` root group by `n`.
fn sweep_groups(file: &Nc4File<'_>, root: &NcFile<'_>) -> Vec<String> {
    let listed: Option<Vec<String>> = match root.read_var("sweep_group_name") {
        Ok(NcArray::Str(names)) => Some(names),
        Ok(NcArray::Char(chars)) => root.vars.get("sweep_group_name").map(|var| {
            let width = root.var_dims(var).last().copied().unwrap_or(1).max(1);
            chars.chunks(width).map(crate::cfradial::text_of).collect()
        }),
        _ => None,
    };
    if let Some(names) = listed {
        let paths: Vec<String> = names
            .iter()
            .map(|name| format!("/{}", name.trim().trim_start_matches('/')))
            .collect();
        if !paths.is_empty() && paths.iter().all(|path| file.group(path).is_some()) {
            return paths;
        }
    }
    let mut numbered: Vec<(u32, String)> = file
        .root()
        .groups
        .iter()
        .filter_map(|path| sweep_number_of(path.trim_start_matches('/')).map(|n| (n, path.clone())))
        .collect();
    numbered.sort();
    numbered.into_iter().map(|(_, path)| path).collect()
}

pub(crate) fn decode(file: &Nc4File<'_>, mut budget: DecodeBudget) -> Result<Volume> {
    let root = NcFile::netcdf4_group(file, "/")?;
    let sweep_paths = sweep_groups(file, &root);
    if sweep_paths.is_empty() {
        return Err(invalid(
            "netCDF-4 file has no sweep groups — not CfRadial 2",
        ));
    }
    check_sweep_count(sweep_paths.len(), "CfRadial 2 sweep groups")
        .map_err(CfRadialError::LimitExceeded)?;
    let views = sweep_paths
        .iter()
        .map(|path| NcFile::netcdf4_group(file, path))
        .collect::<Result<Vec<_>>>()?;

    let coverage_start = parse_time_var_or_attr(&root, "time_coverage_start");
    let coverage_end = parse_time_var_or_attr(&root, "time_coverage_end");
    let time_reference = views
        .iter()
        .find_map(time_units_reference)
        .or(coverage_start)
        .unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
    let root_time_dim = root.dims.iter().position(|(name, _)| name == "time");
    let mut volume = describe_volume(
        &root,
        root_time_dim,
        time_reference,
        coverage_start,
        coverage_end,
    );
    volume.provenance.source_format = SourceFormat::CfRadial2;
    volume.provenance.source_version = Some(
        root.gattr_str("version")
            .map(str::to_owned)
            .unwrap_or_else(|| "CfRadial-2".to_owned()),
    );
    note_netcdf4_container(&mut volume, file, "cfradial2-netcdf4");

    // Radar parameters: the group (CfRadial 2.0 / xradar names, then
    // FM301-2022's), the `frequency` FM301 puts in every sweep group, then
    // anything a writer left at the root.
    let parameters_view = group_view(file, "/radar_parameters")?;
    let mut parameters = parameters_view
        .as_ref()
        .map(radar_parameters)
        .unwrap_or_default();
    if let Some(view) = &parameters_view {
        fm301_parameters(view, &mut parameters);
    }
    if parameters.frequency_hz.is_empty()
        && let Some(hz) = views.iter().find_map(sweep_frequency_hz)
    {
        parameters.frequency_hz = hz;
    }
    let at_root = radar_parameters(&root);
    if parameters.frequency_hz.is_empty() {
        parameters.frequency_hz = at_root.frequency_hz;
    }
    parameters.beam_width_h_deg = parameters.beam_width_h_deg.or(at_root.beam_width_h_deg);
    parameters.beam_width_v_deg = parameters.beam_width_v_deg.or(at_root.beam_width_v_deg);
    parameters.antenna_gain_h_db = parameters.antenna_gain_h_db.or(at_root.antenna_gain_h_db);
    parameters.antenna_gain_v_db = parameters.antenna_gain_v_db.or(at_root.antenna_gain_v_db);
    parameters.receiver_bandwidth_hz = parameters
        .receiver_bandwidth_hz
        .or(at_root.receiver_bandwidth_hz);
    volume.radar_parameters = parameters;
    if let Some(view) = &parameters_view {
        keep_variables(
            view,
            PARAMETER_VARS,
            "",
            "radar_parameters",
            &mut volume.extra_vars,
        );
        keep_group_attrs(view, "radar_parameters", &mut volume.attrs.other);
        volume
            .variable_attrs
            .extend(slotted_variable_attrs(view, "radar_parameters", &|var| {
                !PARAMETER_VARS.contains(&var.name.as_str())
            }));
    }

    if let Some(view) = group_view(file, "/radar_calibration")? {
        let (entries, used) = calibration(&view, time_reference);
        volume.radar_calibration = entries;
        let used: Vec<&str> = used.iter().map(String::as_str).collect();
        keep_variables(
            &view,
            &used,
            "radar_calibration.",
            "radar_calibration",
            &mut volume.extra_vars,
        );
        keep_group_attrs(&view, "radar_calibration", &mut volume.attrs.other);
        volume
            .variable_attrs
            .extend(slotted_variable_attrs(&view, "radar_calibration", &|var| {
                !used.contains(&var.name.as_str())
            }));
    }
    for name in ["georeferencing_correction", "georeference_correction"] {
        if let Some(view) = group_view(file, &format!("/{name}"))? {
            let (correction, used) = georeferencing(&view);
            if correction != GeoreferencingCorrection::default() {
                volume.georeferencing_correction = Some(Box::new(correction));
            }
            let used: Vec<&str> = used.to_vec();
            keep_variables(&view, &used, "", name, &mut volume.extra_vars);
            keep_group_attrs(&view, name, &mut volume.attrs.other);
            volume.variable_attrs.extend(slotted_variable_attrs(
                &view,
                "georeferencing_correction",
                &|var| !used.contains(&var.name.as_str()),
            ));
        }
    }

    // Root variables without a slot, verbatim (the platform track and the
    // sweep table are the sweeps').
    let mut extra: Vec<&NcVar> = root
        .vars
        .values()
        .filter(|var| {
            let per_ray = root_time_dim.is_some() && var.dim_ids.first().copied() == root_time_dim;
            !per_ray
                && !matches!(
                    var.name.as_str(),
                    "sweep_group_name" | "sweep_fixed_angle" | "frequency"
                )
                && !SLOTTED_ROOT_VARS.contains(&var.name.as_str())
        })
        .collect();
    extra.sort_by_key(|var| var.index);
    for var in extra {
        if let Some(extra) = extra_variable(&root, var, None) {
            volume.extra_vars.push(extra);
        }
    }
    // Any other root group, with its subgroups: variables as
    // `<group>.<name>`, group attributes as `<group>.<attribute>`.
    for path in &file.root().groups {
        let name = path.trim_start_matches('/');
        if sweep_paths.contains(path) || METADATA_GROUPS.contains(&name) {
            continue;
        }
        keep_group_tree(
            file,
            path,
            name,
            &mut volume.extra_vars,
            &mut volume.attrs.other,
            0,
        )?;
    }

    // A moving platform's track at the root (xradar): split over the
    // sweeps when it has one entry per ray of the volume.
    let root_rays = root_time_dim.map_or(0, |dim| root.dims[dim].1);
    let root_track = match root_time_dim {
        Some(dim) if root_rays > 0 => {
            let track = RayData::instruments(&root, dim, root_rays, &mut budget)?;
            track.platform
        }
        _ => None,
    };

    let root_fixed = root.read_var("sweep_fixed_angle").ok();
    let mut first_ray = 0usize;
    let total_rays: usize = views
        .iter()
        .map(|view| ray_dimension(view).map_or(0, |(_, n)| n))
        .sum();
    for (index, (view, path)) in views.iter().zip(&sweep_paths).enumerate() {
        let root_fixed_angle = root_fixed.as_ref().and_then(|array| array.get_f64(index));
        let mut sweep = decode_sweep(
            file,
            view,
            path,
            index,
            &SweepContext {
                time_reference,
                root_fixed_angle,
                frequency_hz: &volume.radar_parameters.frequency_hz,
            },
            &mut budget,
        )?;
        // The source's own attributes of the sweep's slotted variables, and
        // the source group name when it is not the view's `sweep_<index>`.
        let group = format!("sweep_{index}");
        let kept: BTreeSet<String> = sweep
            .extra_vars
            .iter()
            .map(|extra| extra.name.to_string())
            .collect();
        let field_shape: Option<Vec<usize>> = ray_dimension(view).and_then(|(ray_dim, _)| {
            view.vars
                .get("range")
                .and_then(|var| var.dim_ids.first().copied())
                .map(|range_dim| vec![ray_dim, range_dim])
        });
        volume
            .variable_attrs
            .extend(slotted_variable_attrs(view, &group, &|var| {
                field_shape.as_deref() == Some(var.dim_ids.as_slice())
                    || kept.contains(var.name.as_str())
            }));
        if let Some(monitoring) = group_view(file, &format!("{path}/monitoring"))? {
            volume.variable_attrs.extend(slotted_variable_attrs(
                &monitoring,
                &format!("{group}/monitoring"),
                &|var| kept.contains(var.name.as_str()),
            ));
            keep_group_attrs(&monitoring, "monitoring", &mut sweep.other);
        }
        // Any other subgroup of the sweep, verbatim.
        let subgroups = file
            .group(path)
            .map(|group| group.groups.clone())
            .unwrap_or_default();
        for child in subgroups {
            let name = child.rsplit('/').next().unwrap_or_default().to_owned();
            if name != "monitoring" {
                keep_group_tree(
                    file,
                    &child,
                    &name,
                    &mut sweep.extra_vars,
                    &mut sweep.other,
                    0,
                )?;
            }
        }
        let source_name = path.trim_start_matches('/');
        if source_name != group {
            sweep.other.push((
                "sweep_group_name".into(),
                AttrValue::Text(source_name.into()),
            ));
        }
        let nrays = sweep.nrays();
        if sweep.platform_track.is_none()
            && let Some(track) = &root_track
            && root_rays == total_rays
            && first_ray + nrays <= root_rays
            && nrays > 0
        {
            use crate::cfradial::SliceRays;
            sweep.platform_track = Some(Box::new(track.slice(first_ray..=first_ray + nrays - 1)));
        }
        first_ray += nrays;
        volume.sweeps.push(sweep);
    }

    let root_kept: BTreeSet<String> = volume
        .extra_vars
        .iter()
        .map(|extra| extra.name.to_string())
        .collect();
    let mut root_attrs =
        slotted_variable_attrs(&root, "", &|var| root_kept.contains(var.name.as_str()));
    root_attrs.append(&mut volume.variable_attrs);
    volume.variable_attrs = root_attrs;
    charge_passthrough(&volume, &mut budget)?;
    volume.provenance.decode.message_count = sweep_paths.len();
    volume.provenance.decode.decoded_ray_count = volume.sweeps.iter().map(Sweep::nrays).sum();
    volume.seal().map_err(|err| invalid(err.to_string()))?;
    if volume.time_coverage.is_none() {
        volume.time_coverage = volume.ray_time_extent();
    }
    Ok(volume)
}

/// FM301-2022 Table 301-12a radar parameters (no `radar_` prefix), for the
/// slots the CfRadial 2.0 names left empty.
fn fm301_parameters(view: &NcFile<'_>, parameters: &mut RadarParameters) {
    let number = |name: &str| numeric_var_first(view, name).map(|value| value as f32);
    parameters.antenna_gain_h_db = parameters
        .antenna_gain_h_db
        .or_else(|| number("antenna_gain_h"));
    parameters.antenna_gain_v_db = parameters
        .antenna_gain_v_db
        .or_else(|| number("antenna_gain_v"));
    parameters.beam_width_h_deg = parameters
        .beam_width_h_deg
        .or_else(|| cfradial_beam_width_deg(view, &["beam_width_h"]));
    parameters.beam_width_v_deg = parameters
        .beam_width_v_deg
        .or_else(|| cfradial_beam_width_deg(view, &["beam_width_v"]));
    parameters.receiver_bandwidth_hz = parameters
        .receiver_bandwidth_hz
        .or_else(|| number("receiver_bandwidth"));
}

/// A sweep group's `frequency(frequency)` coordinate in Hz (FM301 Table
/// 301-5), when it has one.
fn sweep_frequency_hz(view: &NcFile<'_>) -> Option<Vec<f64>> {
    let var = view.vars.get("frequency")?;
    if var.dim_ids.len() != 1 || view.dim_name(var, 0) != Some("frequency") {
        return None;
    }
    let values = view.read_var("frequency").ok()?;
    let hz: Vec<f64> = (0..values.len())
        .filter_map(|index| values.get_f64(index))
        .collect();
    (!hz.is_empty() && hz.iter().all(|value| value.is_finite() && *value > 0.0)).then_some(hz)
}

/// Table 301-11 monitoring variables other than the transmit powers, read
/// into `Sweep::monitoring` from a sweep's `monitoring` group.
const MONITORING_VARS: &[&str] = &[
    "radar_measured_sky_noise",
    "radar_measured_cold_noise",
    "radar_measured_hot_noise",
    "phase_difference_transmit_hv",
    "antenna_pointing_accuracy_elev",
    "antenna_pointing_accuracy_az",
    "calibration_offset_h",
    "calibration_offset_v",
    "zdr_offset",
];

/// Radar parameter variables with a typed slot.
const PARAMETER_VARS: &[&str] = &[
    "frequency",
    "antenna_gain_h",
    "antenna_gain_v",
    "beam_width_h",
    "beam_width_v",
    "receiver_bandwidth",
    "radar_antenna_gain_h",
    "radar_antenna_gain_v",
    "radar_beam_width_h",
    "radar_beam_width_v",
    "radar_beam_width_h_deg",
    "radar_beam_width_v_deg",
    "radar_rx_bandwidth",
    "radar_receiver_bandwidth",
    "pulse_width",
    "prt",
    "unambiguous_range",
];

/// The view of a group, when the file has it.
fn group_view<'f>(file: &'f Nc4File<'f>, path: &str) -> Result<Option<NcFile<'f>>> {
    if file.group(path).is_none() {
        return Ok(None);
    }
    NcFile::netcdf4_group(file, path).map(Some)
}

/// Every variable of `view` not named in `used`, verbatim into `extras`,
/// named `<prefix><name>` (`<group>.<name>` when that name is taken).
fn keep_variables(
    view: &NcFile<'_>,
    used: &[&str],
    prefix: &str,
    group: &str,
    extras: &mut Vec<ExtraVariable>,
) {
    let mut vars: Vec<&NcVar> = view
        .vars
        .values()
        .filter(|var| !used.contains(&var.name.as_str()))
        .collect();
    vars.sort_by_key(|var| var.index);
    for var in vars {
        let Some(mut extra) = extra_variable(view, var, None) else {
            continue;
        };
        let mut name = format!("{prefix}{}", var.name);
        if extras.iter().any(|have| *have.name == *name) {
            name = format!("{group}.{}", var.name);
        }
        extra.name = name.into();
        extras.push(extra);
    }
}

/// The attributes of a group as `<group>.<name>`.
fn keep_group_attrs(view: &NcFile<'_>, group: &str, out: &mut Vec<(Box<str>, AttrValue)>) {
    out.extend(
        view.gattrs
            .iter()
            .map(|(name, value)| (format!("{group}.{name}").into(), attr_value(value))),
    );
}

/// A group and its subgroups verbatim: variables and group attributes as
/// `<prefix>.<name>`, where `prefix` is the group's path below the level
/// that keeps it, dot-separated.
fn keep_group_tree(
    file: &Nc4File<'_>,
    path: &str,
    prefix: &str,
    extras: &mut Vec<ExtraVariable>,
    attrs: &mut Vec<(Box<str>, AttrValue)>,
    depth: usize,
) -> Result<()> {
    if depth > MAX_GROUP_DEPTH {
        return Ok(());
    }
    let view = NcFile::netcdf4_group(file, path)?;
    keep_variables(&view, &[], &format!("{prefix}."), prefix, extras);
    keep_group_attrs(&view, prefix, attrs);
    let children = file
        .group(path)
        .map(|group| group.groups.clone())
        .unwrap_or_default();
    for child in children {
        let name = child.rsplit('/').next().unwrap_or_default().to_owned();
        keep_group_tree(
            file,
            &child,
            &format!("{prefix}.{name}"),
            extras,
            attrs,
            depth + 1,
        )?;
    }
    Ok(())
}

/// Nested groups are kept this deep (the HDF5 reader bounds depth too).
const MAX_GROUP_DEPTH: usize = 16;

/// The ray dimension of a sweep group (the dimension of its `time`
/// variable, else of `azimuth`; xradar's writer calls it `azimuth`) and its
/// length.
fn ray_dimension(view: &NcFile<'_>) -> Option<(usize, usize)> {
    let var = view
        .vars
        .get("time")
        .filter(|var| var.dim_ids.len() == 1)
        .or_else(|| view.vars.get("azimuth"))?;
    let dim = *var.dim_ids.first()?;
    Some((dim, view.dims.get(dim)?.1))
}

/// The first finite element of a numeric variable (a sweep scalar).
fn scalar(view: &NcFile<'_>, name: &str) -> Option<f64> {
    let var = view.vars.get(name)?;
    let value = view.read_var(name).ok()?.get_f64(0)?;
    let fill = var.attr_f64("_FillValue");
    (value.is_finite() && Some(value) != fill).then_some(value)
}

/// A text sweep scalar: a string or char variable, else a group attribute.
fn text(view: &NcFile<'_>, name: &str) -> Option<String> {
    char_var_text(view, name)
        .or_else(|| view.gattr_str(name).map(|text| text.trim().to_owned()))
        .filter(|text| !text.is_empty())
}

/// What a sweep group is read against: the volume time reference, the
/// root's `sweep_fixed_angle` entry for the sweep, and the volume frequency.
#[derive(Clone, Copy)]
struct SweepContext<'a> {
    time_reference: DateTime<Utc>,
    root_fixed_angle: Option<f64>,
    frequency_hz: &'a [f64],
}

fn decode_sweep(
    file: &Nc4File<'_>,
    view: &NcFile<'_>,
    path: &str,
    index: usize,
    context: &SweepContext<'_>,
    budget: &mut DecodeBudget,
) -> Result<Sweep> {
    let SweepContext {
        time_reference,
        root_fixed_angle,
        frequency_hz,
    } = *context;
    let (ray_dim, n_rays) = ray_dimension(view)
        .ok_or_else(|| invalid(format!("CfRadial 2 sweep {path} has no time coordinate")))?;
    let range_dim = view
        .vars
        .get("range")
        .and_then(|var| var.dim_ids.first().copied())
        .or_else(|| view.dims.iter().position(|(name, _)| name == "range"))
        .ok_or_else(|| invalid(format!("CfRadial 2 sweep {path} has no range coordinate")))?;
    let n_gates = view.dims[range_dim].1;
    // A sweep of rays without gates (a Level II cut whose radials carry no
    // moments) is a sweep; one without rays is not.
    if n_rays == 0 {
        return Err(invalid(format!("CfRadial 2 sweep {path} has no rays")));
    }
    check_gate_count(n_gates, "CfRadial 2 range dimension")
        .map_err(CfRadialError::LimitExceeded)?;
    let ngates = u32::try_from(n_gates).map_err(|_| invalid("CfRadial range overflow"))?;
    budget
        .charge(n_rays, 16 * size_of::<f64>(), "CfRadial sweep rays")
        .map_err(CfRadialError::LimitExceeded)?;

    let mut rays = RayData::read(view, ray_dim, n_rays, budget)?;
    // `time` in the units of this sweep's reference.
    if let Some(seconds) = &mut rays.seconds {
        let scale = time_units_scale(
            view.vars
                .get("time")
                .and_then(|var| var.attr_str("units"))
                .and_then(|units| units.split_once("since"))
                .map(|(unit, _)| unit.trim()),
        );
        if scale != 1.0 {
            seconds.iter_mut().for_each(|value| *value *= scale);
        }
    }
    let monitoring_path = format!("{path}/monitoring");
    let monitoring = group_view(file, &monitoring_path)?;
    if let Some(monitoring) = &monitoring {
        let ray_name = &view.dims[ray_dim].0;
        if let Some(dim) = monitoring
            .dims
            .iter()
            .position(|(name, _)| name == ray_name)
        {
            rays.merge_missing(RayData::instruments(monitoring, dim, n_rays, budget)?);
        }
    }
    let offset_s = time_units_reference(view)
        .map(|reference| (reference - time_reference).num_milliseconds() as f64 / 1000.0)
        .unwrap_or(0.0);
    // The rest of the Table 301-11 monitoring variables (the transmit powers
    // come with the instrument variables).
    let mut monitoring_values: Vec<(&'static str, Vec<f32>)> = Vec::new();
    if let Some(monitoring) = &monitoring {
        let ray_name = &view.dims[ray_dim].0;
        if let Some(dim) = monitoring
            .dims
            .iter()
            .position(|(name, _)| name == ray_name)
        {
            for name in MONITORING_VARS {
                if let Some(values) = aligned_time_values(monitoring, name, dim, n_rays, budget)? {
                    monitoring_values.push((name, values.iter().map(|v| *v as f32).collect()));
                }
            }
        }
    }

    let range = read_f64s(view, "range", budget)?;
    if range.is_empty() && n_gates > 0 {
        return Err(invalid(format!(
            "CfRadial 2 sweep {path} has an empty range"
        )));
    }

    let mode = text(view, "sweep_mode")
        .map(|text| SweepMode::parse(&text))
        .unwrap_or(SweepMode::AzimuthSurveillance);
    let rhi = matches!(mode, SweepMode::Rhi | SweepMode::ManualRhi);
    let fixed = scalar(view, "sweep_fixed_angle")
        .or_else(|| scalar(view, "fixed_angle"))
        .or(root_fixed_angle.filter(|value| value.is_finite()))
        .unwrap_or_else(|| {
            fallback_fixed_angle(rhi, &rays.azimuth[..n_rays], &rays.elevation[..n_rays])
        }) as f32;
    let mut sweep = Sweep::new(index as u32, mode, fixed);
    sweep.elevation_number = u16::try_from(index).ok();
    sweep.follow_mode = text(view, "follow_mode").map(|text| FollowMode::parse(&text));
    sweep.prt_mode = text(view, "prt_mode").map(|text| PrtMode::parse(&text));
    sweep.polarization_mode =
        text(view, "polarization_mode").map(|text| PolarizationMode::parse(&text));
    sweep.rays_are_indexed = text(view, "rays_are_indexed").and_then(|text| parse_bool(&text));
    sweep.rays_angle_resolution_deg = scalar(view, "ray_angle_res")
        .or_else(|| scalar(view, "rays_angle_resolution"))
        .map(|value| value as f32);
    sweep.target_scan_rate_deg_per_s = scalar(view, "target_scan_rate").map(|value| value as f32);
    sweep.qc_procedures = text(view, "qc_procedures");
    if let Some(number) = scalar(view, "sweep_number")
        && number != index as f64
    {
        sweep.other.push((
            "sweep_number".into(),
            AttrValue::Scalar(Scalar::F64(number)),
        ));
    }
    sweep.range = if n_gates == 0 {
        // No centres: the geometry the range attributes state, if any.
        let attr = |name: &str| {
            view.vars
                .get("range")
                .and_then(|var| var.attr_f64(name))
                .filter(|value| value.is_finite())
                .unwrap_or(0.0)
        };
        RangeCoord::Uniform {
            first_center_m: attr("meters_to_center_of_first_gate"),
            spacing_m: attr("meters_between_gates"),
            ngates: 0,
        }
    } else {
        range_coordinate(&range[..n_gates.min(range.len())], ngates)
    };
    rays.fill(&mut sweep, 0..=n_rays - 1, offset_s);
    if !monitoring_values.is_empty() {
        let slots = sweep.monitoring.get_or_insert_with(Default::default);
        for (name, values) in monitoring_values {
            if let Some(slot) = slots.variable_mut(name) {
                *slot = Some(values);
            }
        }
    }
    // Sweep group attributes without a slot, verbatim.
    sweep.other.extend(
        view.gattrs
            .iter()
            .filter(|(name, _)| name.as_str() != "sweep_mode")
            .map(|(name, value)| (name.as_str().into(), attr_value(value))),
    );

    // Fields `(ray, range)`, per-ray variables and scalars, in file order.
    let mut vars: Vec<&NcVar> = view.vars.values().collect();
    vars.sort_by_key(|var| var.index);
    let field_shape = [ray_dim, range_dim];
    for var in &vars {
        if var.dim_ids.as_slice() != field_shape {
            continue;
        }
        let raw = view.read_var(&var.name)?;
        let expected = n_rays
            .checked_mul(n_gates)
            .ok_or_else(|| invalid("CfRadial field dimensions overflow addressable memory"))?;
        if raw.len() < expected {
            return Err(invalid(format!(
                "CfRadial 2 field '{}' in {path} has {} values; expected {expected}",
                var.name,
                raw.len()
            )));
        }
        if let Some(field) = build_field(var, &raw, 0..expected, ngates, budget)? {
            sweep
                .add_field(field)
                .map_err(|err| invalid(format!("CfRadial 2 field '{}': {err}", var.name)))?;
        }
    }
    for var in &vars {
        if var.dim_ids.as_slice() == field_shape || SLOTTED_SWEEP_VARS.contains(&var.name.as_str())
        {
            continue;
        }
        // The sweep's `frequency` coordinate is the volume's frequency slot
        // when it holds the same values.
        if var.name == "frequency"
            && sweep_frequency_hz(view).is_some_and(|hz| hz.as_slice() == frequency_hz)
        {
            continue;
        }
        let per_ray = var.dim_ids.first() == Some(&ray_dim) && var.dim_ids.len() <= 2;
        if per_ray && SLOTTED_RAY_VARS.contains(&var.name.as_str()) {
            continue;
        }
        let extra = if per_ray {
            extra_variable(view, var, Some(0..=n_rays - 1))
        } else {
            extra_variable(view, var, None)
        };
        if let Some(extra) = extra {
            sweep.extra_vars.push(extra);
        }
    }
    if let Some(monitoring) = &monitoring {
        let mut vars: Vec<&NcVar> = monitoring.vars.values().collect();
        vars.sort_by_key(|var| var.index);
        let ray_name = view.dims[ray_dim].0.as_str();
        for var in vars {
            let per_ray = monitoring.dim_name(var, 0) == Some(ray_name);
            if per_ray
                && var.dim_ids.len() == 1
                && (SLOTTED_RAY_VARS.contains(&var.name.as_str())
                    || MONITORING_VARS.contains(&var.name.as_str()))
            {
                continue;
            }
            let extra = if per_ray && var.dim_ids.len() <= 2 {
                extra_variable(monitoring, var, Some(0..=n_rays - 1))
            } else {
                extra_variable(monitoring, var, None)
            };
            if let Some(extra) = extra {
                sweep.extra_vars.push(extra);
            }
        }
    }
    Ok(sweep)
}

/// Calibration entries of a `radar_calibration` group: one per element of
/// its `r_calib`/`calib` dimension (Radx), or one from scalars (xradar).
/// Names are Table 301-14a's, with CfRadial 1's `r_calib_` prefix and the
/// `base_dbz_1km_*` spellings accepted; `time`/`calibration_time` is the
/// entry time; anything else goes to `RadarCalibration::extra`. A value equal
/// to the variable's `_FillValue` (or not finite) is left unset.
fn calibration(
    view: &NcFile<'_>,
    time_reference: DateTime<Utc>,
) -> (Vec<RadarCalibration>, Vec<String>) {
    let count_dim = view
        .dims
        .iter()
        .position(|(name, _)| matches!(name.as_str(), "r_calib" | "calib"));
    let count = count_dim.map_or(1, |dim| view.dims[dim].1);
    if count == 0 || count > 4096 {
        return (Vec::new(), Vec::new());
    }
    let mut used = Vec::new();
    let mut entries = vec![RadarCalibration::default(); count];
    let mut vars: Vec<&NcVar> = view.vars.values().collect();
    vars.sort_by_key(|var| var.index);
    for var in vars {
        let along = match (var.dim_ids.first().copied(), count_dim) {
            (None, _) => false,
            (Some(first), Some(dim)) if first == dim => true,
            // A text scalar is a char array along its string length.
            _ if var.is_text() && var.dim_ids.len() == 1 => false,
            _ => continue,
        };
        let Ok(array) = view.read_var(&var.name) else {
            continue;
        };
        let name = var.name.strip_prefix("r_calib_").unwrap_or(&var.name);
        if matches!(name, "time" | "calibration_time") {
            let texts: Vec<String> = match &array {
                NcArray::Str(strings) => strings.clone(),
                NcArray::Char(chars) => {
                    let width = view.var_dims(var).last().copied().unwrap_or(1).max(1);
                    chars.chunks(width).map(crate::cfradial::text_of).collect()
                }
                _ => Vec::new(),
            };
            if texts.is_empty() {
                continue;
            }
            for (entry, text) in entries.iter_mut().zip(texts) {
                entry.time_s = parse_iso_instant(&text)
                    .map(|instant| (instant - time_reference).num_milliseconds() as f64 / 1000.0);
            }
            used.push(var.name.clone());
            continue;
        }
        if array.is_text() {
            // Kept verbatim by the caller.
            continue;
        }
        used.push(var.name.clone());
        let table_name = match name {
            "base_dbz_1km_hc" => "base_1km_hc",
            "base_dbz_1km_vc" => "base_1km_vc",
            "base_dbz_1km_hx" => "base_1km_hx",
            "base_dbz_1km_vx" => "base_1km_vx",
            other => other,
        };
        let fill = var.attr_f64("_FillValue");
        for (index, entry) in entries.iter_mut().enumerate() {
            let value = if along {
                array.get_f64(index)
            } else {
                array.get_f64(0).filter(|_| index == 0)
            };
            let Some(value) = value.filter(|value| value.is_finite() && Some(*value) != fill)
            else {
                continue;
            };
            // FM301's calibration index (a byte in FM301-2022, an int in
            // xradar's files).
            if table_name == "calib_index"
                && value.fract() == 0.0
                && let Ok(index) = i32::try_from(value as i64)
            {
                entry.calib_index = Some(index);
                continue;
            }
            if !entry.set_float_entry(table_name, Some(value as f32)) {
                let value = match array {
                    NcArray::F32(_) => Scalar::F32(value as f32),
                    _ => Scalar::F64(value),
                };
                entry.extra.push((name.into(), AttrValue::Scalar(value)));
            }
        }
    }
    (entries, used)
}

/// The `georeferencing_correction` scalars, and the variable names used.
fn georeferencing(view: &NcFile<'_>) -> (GeoreferencingCorrection, Vec<&'static str>) {
    let mut correction = GeoreferencingCorrection::default();
    let mut used = Vec::new();
    let names = GeoreferencingCorrection::default()
        .entries()
        .map(|(name, _)| name);
    for name in names {
        if !view.vars.contains_key(name) {
            continue;
        }
        used.push(name);
        let value = scalar(view, name).map(|value| value as f32);
        let slot = match name {
            "azimuth_correction" => &mut correction.azimuth_correction,
            "elevation_correction" => &mut correction.elevation_correction,
            "range_correction" => &mut correction.range_correction,
            "longitude_correction" => &mut correction.longitude_correction,
            "latitude_correction" => &mut correction.latitude_correction,
            "pressure_altitude_correction" => &mut correction.pressure_altitude_correction,
            "radar_altitude_correction" => &mut correction.radar_altitude_correction,
            "eastward_ground_speed_correction" => &mut correction.eastward_ground_speed_correction,
            "northward_ground_speed_correction" => {
                &mut correction.northward_ground_speed_correction
            }
            "vertical_velocity_correction" => &mut correction.vertical_velocity_correction,
            "heading_correction" => &mut correction.heading_correction,
            "roll_correction" => &mut correction.roll_correction,
            "pitch_correction" => &mut correction.pitch_correction,
            "drift_correction" => &mut correction.drift_correction,
            "rotation_correction" => &mut correction.rotation_correction,
            "tilt_correction" => &mut correction.tilt_correction,
            _ => continue,
        };
        *slot = value;
    }
    (correction, used)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_group_numbers() {
        assert_eq!(sweep_number_of("sweep_0"), Some(0));
        assert_eq!(sweep_number_of("sweep_0012"), Some(12));
        assert_eq!(sweep_number_of("sweep_x"), None);
        assert_eq!(sweep_number_of("radar_parameters"), None);
    }
}
