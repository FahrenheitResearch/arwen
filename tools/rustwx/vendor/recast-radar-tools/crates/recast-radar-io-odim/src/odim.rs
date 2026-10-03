//! ODIM_H5 polar volume/scan decoder (the European/research HDF5 standard).
//!
//! Information model: D. B. Michelson, R. Lewandowski, M. Szewczykowski,
//! H. Beekhuis, and G. Haase, "EUMETNET OPERA weather radar information
//! model for implementation with the HDF5 file format" (ODIM_H5), EUMETNET
//! OPERA Working Document WD_2008_03 (v2.2, 2014; v2.4, 2021). Layout:
//! `/what` (object, date, time, source), `/where` (lat, lon, height),
//! `/datasetN` per sweep with `where` (elangle, nbins, nrays, rstart,
//! rscale, a1gate), `what` (start/end date and time), `how` (per-ray angle
//! and time arrays, `NI`) and `dataM` per quantity with `what` (quantity,
//! gain, offset, nodata, undetect) and the `data` plane (nrays x nbins).
//!
//! Decodes `PVOL` (polar volume) and `SCAN` (single sweep) objects into the
//! FM301 model ([`Volume`]; `docs/design/fm301-model.md` sections 5.2, 7.2
//! and 8.1), following what xradar's `open_odim_datatree` returns for the
//! same file:
//! - One sweep per `datasetN` in file order; every `dataM` plane is a field
//!   named by its `what/quantity` verbatim (`DBZH`, `TH`, `VRAD`, ...); a
//!   second plane of the same quantity is kept as `<quantity>_<dataM>`.
//! - Planes keep their stored encoding: `u8`, `i8`, `u16`, `i16` and `i32`
//!   with the CF packing `physical = gain * raw + offset` and `nodata` as
//!   `_FillValue`, `undetect` as `_Undetect` (kept distinct; Table 301-10);
//!   `float32` and `float64` planes verbatim with their sentinels as float
//!   codings. `u32`, `i64` and `u64` planes (no writer seen) widen to
//!   `float64` codes, the model having no wider integer storage.
//! - Quality groups are quality fields (Table 301-10 `is_quality_field`,
//!   `qualified_variables`): a plane's `dataM/qualityK` is
//!   `<quantity>_qualityK` and qualifies that plane, a dataset's
//!   `qualityK` is `qualityK` and qualifies every plane; each qualified
//!   field lists them in `ancillary_variables`. Their `what/quantity`
//!   (QIND, CLASS, ...) and `how/task` stay among their attributes.
//! - A `legend` dataset (ODIM_H5 v2.4 key/value strings, or FMI's
//!   code/class compound) and an enumerated plane (h5py's `bool` quality
//!   flags) become `flag_values`/`flag_meanings`.
//! - Ray azimuths are `(how/startazA + how/stopazA) / 2` when present
//!   (wrapped into [0, 360); a non-finite mean is kept as it is; without a
//!   `stopazA` of one angle per ray, each ray stops where the next starts),
//!   else the storage-order centres `(i + 0.5) * 360 / nrays`; ray elevations
//!   `(how/startelA + how/stopelA) / 2`, else `how/elangles`, else
//!   `where/elangle`; ray times `(how/startazT + how/stopazT) / 2`, else
//!   spread evenly between `what/starttime` and `endtime` starting at
//!   `where/a1gate` (all rays at `starttime` when the two are equal). The
//!   `how` arrays a coordinate is read from also stay verbatim in
//!   `Sweep::other`: a mean does not give back the start and stop angles.
//! - The `range` coordinate holds gate centres: `rstart` (the start of the
//!   first bin: metres in ODIM_H5 v2.4, km before) plus half a `rscale` (bin
//!   spacing in metres). Implausibly large km values are reinterpreted as
//!   metres — see `first_gate_m_from_rstart`.
//! - `nyquist_velocity(time)` broadcasts `how/NI` (dataset, else root).
//! - Every attribute a coordinate or a typed slot does not take is kept
//!   verbatim, whatever its datatype (string arrays, compound members as
//!   `name.member`, references as the target path, enums by member name,
//!   anything else as bytes): the root group's and the root `what`,
//!   `where` and `how` groups' in `Volume::attrs.other`, a dataset's in
//!   `Sweep::other`, a plane's or quality group's in `Field::attrs.other`.
//!   Attributes of a nested group (DWD `how/radar_system`, SMHI
//!   `how/process_chain`) are `<group>.<name>`, those of any other group
//!   too, and a name the level already has is written `<what|where|how>.<name>`.
//!   Only the arrays a ray coordinate is actually built from are held back,
//!   so a per-ray array this decoder has no slot for (`TXpower`,
//!   `startelT`/`stopelT`) reaches the model instead of being dropped for
//!   having one entry per ray. `tests/odim_every_value.rs` checks every
//!   attribute and plane of the corpus against h5py.
//! - Planes are stored verbatim (design note 7.2): no rewrite pass, so the
//!   raw arrays hash equal to xradar's. Some IRIS exporters (AEMET Spain,
//!   IRIS 10.3) copy the REFLECTIVITY `what` group onto the velocity plane —
//!   the VRADH `nodata`/`undetect` carry the dBZ sentinels (e.g. 95.5 / -32)
//!   while no-echo velocity gates are filled with the physical `offset`
//!   (0 m/s for gain=1/offset=0). Those fill gates match neither declared
//!   sentinel, so the decoded plane is a spurious 0 m/s wall (as it is in
//!   xradar and Py-ART). [`recover_copied_whatgroup_velocity_nodata`] is the
//!   opt-in post-pass that masks a velocity gate sitting on `offset` when
//!   the co-located reflectivity gate is no-echo, and only when the velocity
//!   sentinels equal the reflectivity sentinels (the copied-what-group
//!   signature) — genuine 0 m/s gates with echo, and conformant writers with
//!   distinct velocity sentinels, are untouched.
//!
//! Known limitations (explicit, not silent): non-polar objects (ELEV/RHI
//! cross-section products, CVOL, IMAGE) are rejected with a clear error; a
//! plane whose shape differs from the dataset's first plane (malformed per
//! ODIM_H5) is skipped and counted in `skipped_message_count`.

use std::collections::BTreeSet;

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};
use recast_radar_core::bounded_read::{DecodeBudget, check_gate_count, check_sweep_count};
use recast_radar_core::model::{
    ArrayBuf, AttrValue, Field, FieldData, FieldName, FloatCoding, FloatWidth, FollowMode,
    GateMapping, IntCoding, LinearTransform, PackedInt, Quantity, RadarCalibration,
    RadarParameters, RangeCoord, SourceFormat, Sweep, SweepMode, Volume, floor_to_second,
};

use crate::h5::{H5Attr, H5Data, H5Dataset, H5File};
use crate::tables::Level;
use crate::{OdimError, Result};
pub use recast_radar_hdf5::looks_like_hdf5_bytes;

/// Decode an ODIM_H5 PVOL/SCAN byte buffer into the FM301 model.
pub fn read_odim_h5_volume(bytes: &[u8]) -> Result<Volume> {
    decode(&H5File::open(bytes)?)
}

/// [`read_odim_h5_volume`] for an HDF5 file already opened (the format
/// router opens a file once to tell ODIM from netCDF-4).
pub fn read_odim_hdf5_volume(file: recast_radar_hdf5::H5File<'_>) -> Result<Volume> {
    decode(&H5File::from_hdf5(file)?)
}

fn decode(file: &H5File<'_>) -> Result<Volume> {
    let object = file
        .attr("/what", "object")
        .and_then(|attr| attr.as_str().map(str::to_owned))
        .ok_or_else(|| {
            invalid(
                "HDF5 file has no /what 'object' attribute, so it is not ODIM_H5 \
                 (netCDF-4 CfRadial decodes with recast_radar_io_cfradial)",
            )
        })?;
    if object != "PVOL" && object != "SCAN" {
        return Err(invalid(format!(
            "ODIM_H5 object '{object}' unsupported (PVOL and SCAN only)"
        )));
    }

    let source = file
        .attr("/what", "source")
        .and_then(|attr| attr.as_str().map(str::to_owned))
        .unwrap_or_default();
    let identity = site_identity_from_source(&source);
    let nominal_time = parse_datetime(file, "/what");
    let mut volume = Volume::new(
        identity.id,
        nominal_time.unwrap_or(DateTime::<Utc>::UNIX_EPOCH),
    );
    volume.attrs.site_name = identity.name;
    volume.attrs.source = (!source.is_empty()).then_some(source);
    volume.attrs.wmo.id = identity.wmo;
    volume.attrs.wmo.wsi = identity.wigos;
    // A NaN location (a writer without one) is no location.
    let finite = |name: &str| attr_f64(file, "/where", name).filter(|value| value.is_finite());
    volume.location.latitude_deg = finite("lat");
    volume.location.longitude_deg = finite("lon");
    volume.location.altitude_m = finite("height");
    volume.provenance.source_format = SourceFormat::OdimH5;
    volume.provenance.source_version = file
        .attr("/what", "version")
        .and_then(|attr| attr.as_str().map(str::to_owned))
        .or(Some("ODIM_H5".to_owned()));
    volume.provenance.source_conventions = file
        .attr("/", "Conventions")
        .and_then(|attr| attr.as_str().map(str::to_owned));
    volume.provenance.compression = Some("odim-h5".to_owned());
    if let Some(mhz) = odim_radar_frequency_mhz(file) {
        volume.radar_parameters.frequency_hz = vec![mhz * 1e6];
    }
    let root_how = How::read(file, "/how");

    let mut dataset_names: Vec<String> = file
        .child_names("/")
        .into_iter()
        .filter(|name| is_numbered(name, "dataset"))
        .collect();
    dataset_names.sort_by_key(|name| name[7..].parse::<u32>().unwrap_or(u32::MAX));
    if dataset_names.is_empty() {
        return Err(invalid("ODIM_H5 volume has no /datasetN groups"));
    }
    check_sweep_count(dataset_names.len(), "ODIM_H5 volume").map_err(OdimError::LimitExceeded)?;

    let first_how = How::read(file, &format!("/{}/how", dataset_names[0]));
    let mut root_used = describe_volume(&root_how, &first_how, &mut volume);

    let mut budget = DecodeBudget::volume();
    // Absolute ray times (seconds since the Unix epoch) until the reference
    // is known.
    let mut ray_epoch_s: Vec<Vec<f64>> = Vec::with_capacity(dataset_names.len());
    let mut skipped_planes = 0usize;
    for (index, name) in dataset_names.iter().enumerate() {
        let (mut sweep, times) = decode_sweep(
            file,
            name,
            index,
            &root_how,
            &volume.radar_parameters,
            &mut budget,
        )?;
        skipped_planes += sweep.skipped_planes;
        ray_epoch_s.push(times);
        root_used.extend(sweep.root_used);
        if let Some(calibration) = sweep.calibration {
            let index = calibration_index(&mut volume.radar_calibration, calibration);
            sweep.sweep.ray_vars.calib_index = Some(vec![index; sweep.sweep.nrays()]);
        }
        volume.sweeps.push(sweep.sweep);
    }
    // Every root attribute no typed slot holds, verbatim.
    volume.attrs.other = root_passthrough(file, nominal_time.is_some(), &root_how, &root_used);
    charge_attrs(&mut budget, &volume.attrs.other)?;

    // Time reference: the nominal volume time, else the earliest ray.
    if nominal_time.is_none() {
        let earliest = ray_epoch_s
            .iter()
            .flatten()
            .copied()
            .filter(|time| time.is_finite())
            .fold(f64::INFINITY, f64::min);
        if earliest.is_finite()
            && let Some(instant) =
                DateTime::<Utc>::from_timestamp_millis((earliest * 1000.0).floor() as i64)
        {
            volume.time_reference = floor_to_second(instant);
        }
    }
    let reference_s = volume.time_reference.timestamp() as f64;
    for (sweep, times) in volume.sweeps.iter_mut().zip(ray_epoch_s) {
        sweep.rays.time_s = times.into_iter().map(|time| time - reference_s).collect();
    }

    volume.provenance.decode.decoded_ray_count = volume.sweeps.iter().map(Sweep::nrays).sum();
    volume.provenance.decode.message_count = dataset_names.len();
    volume.provenance.decode.skipped_message_count = skipped_planes;
    volume.seal().map_err(|err| invalid(err.to_string()))?;
    volume.time_coverage = volume.ray_time_extent();
    Ok(volume)
}

struct DecodedSweep {
    sweep: Sweep,
    skipped_planes: usize,
    /// The dataset's radar constants, for `radar_calibration`.
    calibration: Option<RadarCalibration>,
    /// Root `how` attributes the sweep took as its defaults.
    root_used: BTreeSet<&'static str>,
}

fn decode_sweep(
    file: &H5File<'_>,
    dataset: &str,
    index: usize,
    root_how: &How,
    parameters: &RadarParameters,
    budget: &mut DecodeBudget,
) -> Result<(DecodedSweep, Vec<f64>)> {
    let where_path = format!("/{dataset}/where");
    let what_path = format!("/{dataset}/what");
    let how_path = format!("/{dataset}/how");
    let elangle = attr_f64(file, &where_path, "elangle")
        .ok_or_else(|| invalid(format!("{dataset} has no where/elangle")))?
        as f32;
    let rstart_km = attr_f64(file, &where_path, "rstart").unwrap_or(0.0);
    let rscale_m = attr_f64(file, &where_path, "rscale").unwrap_or(0.0);
    let how = How::read(file, &how_path);
    let mut settings = SweepHow::new(&how, root_how);
    let nyquist = settings
        .number("NI")
        .map(|value| value as f32)
        .filter(|value| *value > 0.0);
    settings.used("NI", nyquist.is_some());
    // `rpm` (revolutions per minute), else `antspeed` (deg/s, v2.4).
    let rpm = settings
        .number("rpm")
        .filter(|rpm| *rpm > 0.0)
        .map(|rpm| (rpm * 6.0) as f32);
    settings.used("rpm", rpm.is_some());
    let scan_rate = rpm.or_else(|| {
        let speed = settings
            .number("antspeed")
            .filter(|speed| *speed > 0.0)
            .map(|speed| speed as f32);
        settings.used("antspeed", speed.is_some());
        speed
    });
    // ODIM_H5 v2.2+ gives microseconds; a value outside 0.05 to 10 is
    // another unit (AEMET writes 1e-06) and stays verbatim.
    let pulse_width_s = settings
        .number("pulsewidth")
        .filter(|us| (0.05..=10.0).contains(us))
        .map(|us| (us * 1e-6) as f32);
    settings.used("pulsewidth", pulse_width_s.is_some());
    let calibration = settings.calibration(pulse_width_s);
    settings.site_constants(parameters);

    let mut data_names: Vec<String> = file
        .child_names(&format!("/{dataset}"))
        .into_iter()
        .filter(|name| is_numbered(name, "data"))
        .collect();
    data_names.sort_by_key(|name| name[4..].parse::<u32>().unwrap_or(u32::MAX));
    if data_names.is_empty() {
        return Err(invalid(format!("{dataset} has no dataM planes")));
    }

    // All planes in a sweep share ray geometry; read the first to size it.
    let first_plane = file.dataset(&format!("/{dataset}/{}/data", data_names[0]))?;
    let (nrays, nbins) = match first_plane.dims.as_slice() {
        [rays, bins] => (*rays, *bins),
        other => {
            return Err(invalid(format!(
                "{dataset} data has rank {} (need 2)",
                other.len()
            )));
        }
    };
    if nrays == 0 || nbins == 0 {
        return Err(invalid(format!("{dataset} data plane is empty")));
    }
    check_gate_count(nbins, dataset).map_err(OdimError::LimitExceeded)?;
    let ngates = u32::try_from(nbins).map_err(|_| invalid(format!("{dataset} nbins overflow")))?;
    let spacing_m = if rscale_m > 0.0 && rscale_m.is_finite() {
        rscale_m
    } else {
        1.0
    };
    let first_center_m = first_gate_m_from_rstart(rstart_km, rstart_unit(file)) + spacing_m / 2.0;

    let mut sweep = Sweep::new(index as u32, SweepMode::AzimuthSurveillance, elangle);
    sweep.follow_mode = Some(FollowMode::None);
    sweep.target_scan_rate_deg_per_s = scan_rate;
    let SweepHow {
        dataset_used,
        root_used,
        ..
    } = settings;
    sweep.range = RangeCoord::Uniform {
        first_center_m,
        spacing_m,
        ngates,
    };
    budget
        .charge(nrays, 4 * size_of::<f64>(), "ODIM_H5 sweep rays")
        .map_err(OdimError::LimitExceeded)?;

    // Ray coordinates (xradar's rules; module docs). The `how` arrays they
    // are read from stay in `sweep.other` too: a coordinate is a mean (or a
    // float32) of them, which does not give the arrays back.
    sweep.rays.azimuth_deg = match (
        attr_array(file, &how_path, "startazA"),
        attr_array(file, &how_path, "stopazA"),
    ) {
        (Some(start), stop) if start.len() == nrays => {
            // Without a `stopazA` of one angle per ray, each ray stops where
            // the next starts.
            let stop = stop.filter(|stop| stop.len() == nrays).unwrap_or_else(|| {
                let mut next: Vec<f64> = start[1..].to_vec();
                next.push(start[0] + 360.0);
                next
            });
            start
                .iter()
                .zip(&stop)
                .map(|(start, stop)| {
                    let stop = if *stop < *start { stop + 360.0 } else { *stop };
                    azimuth_f32(mean(*start, stop))
                })
                .collect()
        }
        _ => (0..nrays)
            .map(|ray| ((ray as f32 + 0.5) * 360.0 / nrays as f32).rem_euclid(360.0))
            .collect(),
    };
    sweep.rays.elevation_deg = match (
        attr_array(file, &how_path, "startelA"),
        attr_array(file, &how_path, "stopelA"),
    ) {
        (Some(start), Some(stop)) if start.len() == nrays && stop.len() == nrays => start
            .iter()
            .zip(&stop)
            .map(|(start, stop)| mean(*start, *stop) as f32)
            .collect(),
        _ => match attr_array(file, &how_path, "elangles") {
            Some(angles) if angles.len() == nrays => {
                angles.iter().map(|angle| *angle as f32).collect()
            }
            _ => vec![elangle; nrays],
        },
    };
    let times = match (
        attr_array(file, &how_path, "startazT"),
        attr_array(file, &how_path, "stopazT"),
    ) {
        (Some(start), Some(stop)) if start.len() == nrays && stop.len() == nrays => start
            .iter()
            .zip(&stop)
            .map(|(start, stop)| mean(*start, *stop))
            .collect(),
        _ => ray_times_from_what(file, &what_path, &where_path, nrays),
    };

    // Every dataset attribute no typed slot holds, verbatim: the per-ray
    // `how` arrays included, those the ray coordinates are read from too.
    sweep.other = dataset_passthrough(file, dataset, (nrays, nbins), &how, &dataset_used);
    charge_attrs(budget, &sweep.other)?;
    sweep.rays.time_s = vec![0.0; nrays];
    if let Some(nyquist) = nyquist {
        sweep.ray_vars.nyquist_velocity_mps = Some(vec![nyquist; nrays]);
    }
    if let Some(pulse_width_s) = pulse_width_s {
        sweep.ray_vars.pulse_width_s = Some(vec![pulse_width_s; nrays]);
    }

    let shape = PlaneShape {
        nrays,
        nbins,
        ngates,
    };
    let mut skipped_planes = 0usize;
    let mut first_plane = Some(first_plane);
    for (plane_index, plane_name) in data_names.iter().enumerate() {
        let plane_path = format!("/{dataset}/{plane_name}");
        let quantity = file
            .attr(&format!("{plane_path}/what"), "quantity")
            .and_then(|attr| attr.as_str().map(str::to_owned))
            .unwrap_or_else(|| plane_name.to_uppercase());
        let plane = match (plane_index, first_plane.take()) {
            (0, Some(plane)) => plane,
            _ => file.dataset(&format!("{plane_path}/data"))?,
        };
        let mut name = FieldName::parse(&quantity);
        if sweep.field(&name).is_some() {
            // A second plane of the same quantity (malformed): kept under
            // `<quantity>_<dataM>`; the first plane keeps the name.
            name = FieldName::parse(&format!("{quantity}_{plane_name}"));
        }
        let Some(mut field) = plane_field(file, &plane_path, plane, name, &shape, budget)? else {
            skipped_planes += 1;
            continue;
        };
        if field.name == FieldName::Th {
            // ODIM TH is logarithmic total power in dBZ (design note 8.2,
            // note 1), whatever FM301 Table 301-9 says about the spelling.
            field.quantity = Quantity::TotalPower;
            field.attrs.units = Some("dBZ".into());
        }
        // The plane's own quality groups (`dataM/qualityK`).
        let mut quality_fields = Vec::new();
        for quality in quality_names(file, &plane_path) {
            let quality_name = FieldName::parse(&format!("{}_{quality}", field.name.as_str()));
            match quality_field(
                file,
                &format!("{plane_path}/{quality}"),
                quality_name,
                vec![field.name.clone()],
                &shape,
                budget,
            )? {
                Some(quality) => quality_fields.push(quality),
                None => skipped_planes += 1,
            }
        }
        field
            .attrs
            .ancillary_variables
            .extend(quality_fields.iter().map(|quality| quality.name.clone()));
        for field in std::iter::once(field).chain(quality_fields) {
            if sweep.field(&field.name).is_some() {
                skipped_planes += 1;
                continue;
            }
            sweep
                .add_field(field)
                .map_err(|err| invalid(format!("{dataset}/{plane_name}: {err}")))?;
        }
    }
    // Dataset quality groups (`datasetN/qualityK`) qualify every plane.
    let data_fields: Vec<FieldName> = sweep
        .fields
        .iter()
        .filter(|field| field.attrs.is_quality_field != Some(true))
        .map(|field| field.name.clone())
        .collect();
    for quality in quality_names(file, &format!("/{dataset}")) {
        let name = FieldName::parse(&quality);
        let built = quality_field(
            file,
            &format!("/{dataset}/{quality}"),
            name,
            data_fields.clone(),
            &shape,
            budget,
        )?;
        let Some(field) = built.filter(|field| sweep.field(&field.name).is_none()) else {
            skipped_planes += 1;
            continue;
        };
        for data_field in &data_fields {
            if let Some(qualified) = sweep.field_mut(data_field) {
                qualified.attrs.ancillary_variables.push(field.name.clone());
            }
        }
        sweep
            .add_field(field)
            .map_err(|err| invalid(format!("{dataset}/{quality}: {err}")))?;
    }
    Ok((
        DecodedSweep {
            sweep,
            skipped_planes,
            calibration,
            root_used,
        },
        times,
    ))
}

/// Charge passthrough attributes to the volume's decode budget by the memory
/// they hold: an HDF5 attribute may be 16 MiB of bytes, and widening or
/// splitting it into strings multiplies that.
fn charge_attrs(budget: &mut DecodeBudget, attrs: &[(Box<str>, AttrValue)]) -> Result<()> {
    let bytes: usize = attrs
        .iter()
        .map(|(name, value)| name.len().saturating_add(attr_bytes(value)))
        .fold(0usize, usize::saturating_add);
    budget
        .charge(1, bytes, "ODIM_H5 attributes")
        .map_err(OdimError::LimitExceeded)
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

/// The shape every plane of a dataset shares.
struct PlaneShape {
    nrays: usize,
    nbins: usize,
    ngates: u32,
}

/// A plane group's `what` coding: `gain` (0 reads as 1) and `offset`, and
/// the `nodata` and `undetect` codes.
struct PlaneCoding {
    gain: f64,
    offset: f64,
    nodata: Option<f64>,
    undetect: Option<f64>,
}

impl PlaneCoding {
    fn read(file: &H5File<'_>, what: &str) -> Self {
        let gain = attr_f64(file, what, "gain").unwrap_or(1.0);
        Self {
            gain: if gain.abs() > 1.0e-9 { gain } else { 1.0 },
            offset: attr_f64(file, what, "offset").unwrap_or(0.0),
            nodata: attr_f64(file, what, "nodata"),
            undetect: attr_f64(file, what, "undetect"),
        }
    }

    /// The `what` attributes this coding holds exactly: `gain` unless it
    /// was 0, `offset`, and `nodata`/`undetect` when the plane's storage
    /// type represents them (a code the type cannot hold stays verbatim).
    fn slotted(&self, file: &H5File<'_>, what: &str, data: &H5Data) -> BTreeSet<&'static str> {
        let mut slotted = BTreeSet::new();
        if attr_f64(file, what, "offset").is_some() {
            slotted.insert("offset");
        }
        if attr_f64(file, what, "gain").is_some_and(|gain| gain == self.gain) {
            slotted.insert("gain");
        }
        for (name, code) in [("nodata", self.nodata), ("undetect", self.undetect)] {
            if code.is_some_and(|code| data.holds(code)) {
                slotted.insert(name);
            }
        }
        slotted
    }

    /// The field data of `data` with this coding: integer planes with the
    /// CF packing `physical = gain * raw + offset`, `nodata` as
    /// `_FillValue` and `undetect` as `_Undetect`; float planes verbatim
    /// (the packing only when it is not the identity).
    fn field_data(&self, data: H5Data) -> FieldData {
        let transform = LinearTransform::CfScaleOffset {
            scale_factor: self.gain,
            add_offset: self.offset,
            attr_width: FloatWidth::F64,
        };
        let (nodata, undetect) = (self.nodata, self.undetect);
        // Float planes with the identity packing hold physical values.
        let float_transform = (self.gain != 1.0 || self.offset != 0.0).then_some(transform);
        match data {
            H5Data::U8(values) => FieldData::U8 {
                values,
                coding: int_coding(transform, nodata, undetect),
            },
            H5Data::I8(values) => FieldData::I8 {
                values,
                coding: int_coding(transform, nodata, undetect),
            },
            H5Data::U16(values) => FieldData::U16 {
                values,
                coding: int_coding(transform, nodata, undetect),
            },
            H5Data::I16(values) => FieldData::I16 {
                values,
                coding: int_coding(transform, nodata, undetect),
            },
            H5Data::I32(values) => FieldData::I32 {
                values,
                coding: int_coding(transform, nodata, undetect),
            },
            H5Data::F32(values) => FieldData::F32 {
                values,
                coding: FloatCoding {
                    transform: float_transform,
                    fill_value: nodata.map(|value| value as f32),
                    undetect: undetect.map(|value| value as f32),
                },
            },
            H5Data::F64(values) => FieldData::F64 {
                values,
                coding: FloatCoding {
                    transform: float_transform,
                    fill_value: nodata,
                    undetect,
                },
            },
        }
    }
}

impl H5Data {
    /// True when `code` is exactly representable in this storage type.
    fn holds(&self, code: f64) -> bool {
        fn exact<T: OdimCode>(code: f64) -> bool {
            T::from_f64(code).as_f64() == code
        }
        match self {
            Self::U8(_) => exact::<u8>(code),
            Self::I8(_) => exact::<i8>(code),
            Self::U16(_) => exact::<u16>(code),
            Self::I16(_) => exact::<i16>(code),
            Self::I32(_) => exact::<i32>(code),
            Self::F32(_) => f64::from(code as f32) == code || code.is_nan(),
            Self::F64(_) => true,
        }
    }
}

/// One `dataM` (or quality) plane group as a field named `name`: the
/// `data` plane with its `what` coding, the group's other attributes
/// verbatim ([`group_passthrough`]), and a `legend` or an enumerated
/// datatype as `flag_values`/`flag_meanings`. `None` when the plane's shape
/// is not the dataset's.
fn plane_field(
    file: &H5File<'_>,
    path: &str,
    plane: H5Dataset,
    name: FieldName,
    shape: &PlaneShape,
    budget: &mut DecodeBudget,
) -> Result<Option<Field>> {
    if plane.dims.as_slice() != [shape.nrays, shape.nbins] {
        return Ok(None);
    }
    budget
        .charge(
            shape.nrays,
            shape.nbins.saturating_mul(plane.data.word_bytes()),
            "ODIM_H5 field",
        )
        .map_err(OdimError::LimitExceeded)?;
    let what = format!("{path}/what");
    let coding = PlaneCoding::read(file, &what);
    let mut slotted = coding.slotted(file, &what, &plane.data);
    slotted.insert("quantity");
    let other = group_passthrough(file, path, &slotted);
    charge_attrs(budget, &other)?;
    let legend = file
        .legend(&format!("{path}/legend"))
        .unwrap_or_else(|| plane.enum_members.clone());
    let mut field = Field::new(
        name,
        GateMapping::IDENTITY,
        shape.ngates,
        coding.field_data(plane.data),
    );
    field.attrs.other = other;
    for (code, meaning) in legend {
        field.attrs.flag_values.push(code);
        // CF flag meanings are blank-separated words.
        let word = meaning.split_whitespace().collect::<Vec<_>>().join("_");
        field.attrs.flag_meanings.push(word.into());
    }
    Ok(Some(field))
}

/// A quality group (`qualityK` of a plane or of a dataset) as a quality
/// field (FM301 Table 301-10 `is_quality_field`, `qualified_variables`)
/// named `name`; its `what/quantity` (QIND, CLASS, ...) and `how/task`
/// stay among its attributes. `None` when its plane has another shape or cannot
/// be read (the caller counts it as skipped).
fn quality_field(
    file: &H5File<'_>,
    path: &str,
    name: FieldName,
    qualified: Vec<FieldName>,
    shape: &PlaneShape,
    budget: &mut DecodeBudget,
) -> Result<Option<Field>> {
    // A quality group without a readable plane is skipped (and counted), not
    // fatal to the volume; a limit still is.
    let plane = match file.dataset(&format!("{path}/data")) {
        Ok(plane) => plane,
        Err(err @ OdimError::LimitExceeded(_)) => return Err(err),
        Err(_) => return Ok(None),
    };
    let Some(mut field) = plane_field(file, path, plane, name, shape, budget)? else {
        return Ok(None);
    };
    // `quantity` does not name a quality field: keep it.
    if let Some(quantity) = file
        .attr(&format!("{path}/what"), "quantity")
        .and_then(|attr| attr.as_str().map(str::to_owned))
    {
        field
            .attrs
            .other
            .insert(0, ("quantity".into(), AttrValue::Text(quantity.into())));
    }
    field.quantity = Quantity::Other;
    field.attrs.is_quality_field = Some(true);
    field.attrs.qualified_variables = qualified;
    Ok(Some(field))
}

/// The `qualityK` groups of `path`, by `K`.
fn quality_names(file: &H5File<'_>, path: &str) -> Vec<String> {
    let mut names: Vec<String> = file
        .child_names(path)
        .into_iter()
        .filter(|name| is_numbered(name, "quality"))
        .collect();
    names.sort_by_key(|name| name[7..].parse::<u32>().unwrap_or(u32::MAX));
    names
}

/// `<prefix><n>`, as in `dataset1`, `data2`, `quality3`.
fn is_numbered(name: &str, prefix: &str) -> bool {
    name.strip_prefix(prefix)
        .is_some_and(|rest| rest.parse::<u32>().is_ok())
}

/// Model attributes with ODIM's names: `<group>.<name>` for an attribute
/// whose bare name would not place it back in its group (a `what` or
/// `where` attribute the ODIM_H5 tables do not list there, a `how` one
/// named like a table attribute; [`crate::tables`]) or whose bare name is
/// already taken, else the bare name.
#[derive(Default)]
struct Passthrough {
    attrs: Vec<(Box<str>, AttrValue)>,
    /// The level whose tables decide which names stay bare (`None`: only a
    /// taken name gets its group).
    level: Option<Level>,
}

impl Passthrough {
    fn at(level: Level) -> Self {
        Self {
            attrs: Vec::new(),
            level: Some(level),
        }
    }

    fn push(&mut self, group: &str, name: &str, value: AttrValue) {
        let taken = self.attrs.iter().any(|(have, _)| &**have == name);
        let foreign = self
            .level
            .is_some_and(|level| !group.is_empty() && !level.bare_name_places(group, name));
        let name: Box<str> = if (taken || foreign) && !group.is_empty() {
            format!("{group}.{name}").into()
        } else {
            name.into()
        };
        self.attrs.push((name, value));
    }

    /// Every attribute of every group below `path` (depth first) as
    /// `<prefix><group>.<name>`, skipping the children of `path` that
    /// `skip` names.
    fn push_tree(
        &mut self,
        file: &H5File<'_>,
        path: &str,
        prefix: &str,
        skip: &dyn Fn(&str) -> bool,
        depth: usize,
    ) {
        if depth > MAX_GROUP_DEPTH {
            return;
        }
        let parent = if path.is_empty() { "" } else { path };
        for child in file.child_names(if path.is_empty() { "/" } else { path }) {
            if skip(&child) {
                continue;
            }
            let child_path = format!("{parent}/{child}");
            let child_prefix = format!("{prefix}{child}.");
            for (name, value) in file.attr_entries(&child_path) {
                self.push("", &format!("{child_prefix}{name}"), value);
            }
            self.push_tree(file, &child_path, &child_prefix, &|_| false, depth + 1);
        }
    }
}

/// Groups below a `how` group (DWD `how/radar_system`, SMHI
/// `how/process_chain`, ...) and other unknown groups are read this deep.
const MAX_GROUP_DEPTH: usize = 8;

/// The attributes of a plane or quality group no typed slot holds,
/// verbatim: the group's own, its `what` attributes but `slotted`, its
/// `how` attributes (those of `how` subgroups as `<sub>.<name>`), and those
/// of any other member but the quality groups (as `<member>.<name>`: the
/// `data` dataset's `data.CLASS` and `data.IMAGE_VERSION`, HDF5
/// image-convention markers; a `legend` dataset's).
fn group_passthrough(
    file: &H5File<'_>,
    path: &str,
    slotted: &BTreeSet<&str>,
) -> Vec<(Box<str>, AttrValue)> {
    let mut out = Passthrough::at(Level::Plane);
    for (name, value) in file.attr_entries(path) {
        out.push("", &name, value);
    }
    for (name, value) in file.attr_entries(&format!("{path}/what")) {
        if !slotted.contains(&*name) {
            out.push("what", &name, value);
        }
    }
    for (name, value) in How::read(file, &format!("{path}/how")).0 {
        out.push("how", &name, value);
    }
    out.push_tree(
        file,
        path,
        "",
        &|child| matches!(child, "what" | "how") || is_numbered(child, "quality"),
        0,
    );
    out.attrs
}

/// The dataset attributes no typed slot holds, verbatim: the group's own,
/// every `what` attribute (the product; the start and end times, from which
/// ray times are only derived), the `where` attributes but `elangle` (the
/// fixed angle), `nbins`/`nrays` when they are the plane shape and
/// `rscale`/`rstart` when the range coordinate holds them (`rstart` read as
/// metres stays, see [`first_gate_m_from_rstart`]), the `how` attributes
/// not in `used`, and those of any other group but the planes and quality
/// groups (as `<group>.<name>`).
fn dataset_passthrough(
    file: &H5File<'_>,
    dataset: &str,
    (nrays, nbins): (usize, usize),
    how: &How,
    used: &BTreeSet<&str>,
) -> Vec<(Box<str>, AttrValue)> {
    let path = format!("/{dataset}");
    let mut out = Passthrough::at(Level::Dataset);
    for (name, value) in file.attr_entries(&path) {
        out.push("", &name, value);
    }
    for (name, value) in file.attr_entries(&format!("{path}/what")) {
        out.push("what", &name, value);
    }
    let where_path = format!("{path}/where");
    let number = |name: &str| attr_f64(file, &where_path, name);
    for (name, value) in file.attr_entries(&where_path) {
        let slotted = match &*name {
            "elangle" => number("elangle").is_some(),
            "nbins" => number("nbins") == Some(nbins as f64),
            "nrays" => number("nrays") == Some(nrays as f64),
            "rscale" => number("rscale").is_some_and(|rscale| rscale > 0.0 && rscale.is_finite()),
            "rstart" => number("rstart").is_some_and(|rstart| {
                rstart_unit(file) != RstartUnit::KmOrMetres || rstart <= RSTART_SANE_MAX_KM
            }),
            _ => false,
        };
        if !slotted {
            out.push("where", &name, value);
        }
    }
    for (name, value) in how.unused(used) {
        out.push("how", &name, value);
    }
    out.push_tree(
        file,
        &path,
        "",
        &|child| {
            matches!(child, "what" | "where" | "how")
                || is_numbered(child, "data")
                || is_numbered(child, "quality")
        },
        0,
    );
    out.attrs
}

/// The root attributes no typed slot holds, verbatim: the root group's own
/// but `Conventions` (`source_conventions`); the `/what` attributes but
/// `version` (`source_version`), `source` (`attrs.source`) and
/// `date`/`time` when they parse (the time reference), so `object` (PVOL or
/// SCAN) stays; the `/where` attributes but a numeric `lat`, `lon` and
/// `height` (the location); the `/how` attributes not in `used` (those of
/// `how` subgroups as `<sub>.<name>`); and those of any other root group
/// but the datasets (as `<group>.<name>`).
fn root_passthrough(
    file: &H5File<'_>,
    time_parsed: bool,
    how: &How,
    used: &BTreeSet<&str>,
) -> Vec<(Box<str>, AttrValue)> {
    let mut out = Passthrough::at(Level::Root);
    for (name, value) in file.attr_entries("/") {
        if &*name != "Conventions" {
            out.push("", &name, value);
        }
    }
    for (name, value) in file.attr_entries("/what") {
        let slotted = match &*name {
            "version" | "source" => true,
            "date" | "time" => time_parsed,
            _ => false,
        };
        if !slotted {
            out.push("what", &name, value);
        }
    }
    for (name, value) in file.attr_entries("/where") {
        let slotted = matches!(&*name, "lat" | "lon" | "height")
            && attr_f64(file, "/where", &name).is_some_and(|value| value.is_finite());
        if !slotted {
            out.push("where", &name, value);
        }
    }
    for (name, value) in how.unused(used) {
        out.push("how", &name, value);
    }
    out.push_tree(
        file,
        "",
        "",
        &|child| matches!(child, "what" | "where" | "how") || is_numbered(child, "dataset"),
        0,
    );
    out.attrs
}

/// Integer types ODIM planes are stored in, with the saturating `as` cast
/// the sentinel attributes need (writers disagree about whether `nodata` is
/// a long or a double).
trait OdimCode: PackedInt {
    fn from_f64(value: f64) -> Self;
}

macro_rules! odim_code {
    ($($ty:ty),*) => {$(
        impl OdimCode for $ty {
            fn from_f64(value: f64) -> Self {
                value as $ty
            }
        }
    )*};
}

odim_code!(u8, i8, u16, i16, i32);

/// Integer plane coding: `nodata` is `_FillValue`, `undetect` is
/// `_Undetect`. A plane that declares only `undetect` uses it as the fill
/// code too (the view needs one for padding).
fn int_coding<T: OdimCode>(
    transform: LinearTransform,
    nodata: Option<f64>,
    undetect: Option<f64>,
) -> IntCoding<T> {
    let nodata = nodata.map(T::from_f64);
    let undetect = undetect.map(T::from_f64);
    IntCoding {
        transform,
        fill_value: nodata.or(undetect),
        undetect,
        range_folded: None,
        valid_range: None,
    }
}

/// The mean of a start and a stop value, `a / 2 + b / 2`: the same as
/// `(a + b) / 2` wherever that is finite (halving is exact), and finite for
/// any two finite values (a corrupt `startazT` of -1.8e308 made the sum
/// overflow to -inf).
fn mean(a: f64, b: f64) -> f64 {
    a / 2.0 + b / 2.0
}

/// A ray azimuth (degrees, computed in double precision) as the model's
/// single-precision angle in [0, 360): wrapped before the cast, so a finite
/// angle beyond the float range still names its direction (cast first, a
/// corrupt `startazA` of 1.6e185 became an infinite azimuth), and after it
/// (359.99999999 rounds to 360 in single precision). A non-finite angle
/// stays what it is.
fn azimuth_f32(degrees: f64) -> f32 {
    if !degrees.is_finite() {
        return degrees as f32;
    }
    let wrapped = degrees.rem_euclid(360.0) as f32;
    if wrapped >= 360.0 { 0.0 } else { wrapped }
}

/// Ray times from the dataset's `what/startdate,starttime,enddate,endtime`
/// (seconds since the Unix epoch): spread evenly over the rays starting at
/// `where/a1gate`; every ray at `starttime` when start and end are equal.
fn ray_times_from_what(
    file: &H5File<'_>,
    what_path: &str,
    where_path: &str,
    nrays: usize,
) -> Vec<f64> {
    let start = parse_datetime_pair(file, what_path, "startdate", "starttime");
    let end = parse_datetime_pair(file, what_path, "enddate", "endtime").or(start);
    let (Some(start), Some(end)) = (start, end) else {
        return vec![f64::NAN; nrays];
    };
    let start = start.timestamp() as f64;
    let end = end.timestamp() as f64;
    if start == end {
        return vec![start; nrays];
    }
    let delta = (end - start) / nrays as f64;
    let a1gate = file
        .attr(where_path, "a1gate")
        .as_ref()
        .and_then(H5Attr::as_i64)
        .unwrap_or(0)
        .rem_euclid(nrays as i64) as usize;
    (0..nrays)
        .map(|ray| {
            // Storage index `ray` was radiated `(ray - a1gate) mod nrays`-th.
            let order = (ray + nrays - a1gate) % nrays;
            start + delta / 2.0 + order as f64 * delta
        })
        .collect()
}

/// A first bin starting this many km downrange is physically implausible
/// for a PVOL/SCAN: conformant writers put `rstart` at 0 or within a few km
/// (0.0 across the bejab/bewid/norst fixtures, 22 sweeps), and even
/// long-pulse blind ranges are single-digit km. Values beyond this are
/// metre-valued writer output (see `first_gate_m_from_rstart`).
const RSTART_SANE_MAX_KM: f64 = 20.0;

/// Metres to the start of the first bin, from the `where/rstart` attribute,
/// to the millimetre (which drops the float noise of km times 1000, 0.05 km
/// being 50.00000000000001 m, and keeps a half-metre start: a first bin of
/// 601 m gates centred at 0 m starts at -300.5 m).
///
/// A file whose `Conventions` is `ODIM_H5/V2_4` states `rstart` in metres
/// (`metres`): the only v2.4 producer in the corpus, AEMET Spain (IRIS
/// 8.13/10.3 exports; all 11 sites surveyed on the OPERA ORD bucket,
/// 2026-07-07), writes 125/167/200, and xradar reads v2.4 `rstart` as metres.
/// Earlier versions state km. A km value over [`RSTART_SANE_MAX_KM`] is read
/// as metres too: read as km those would start every ray 125–200 km
/// downrange, past the 150 km extent of the very sweeps they describe (299
/// bins x 500 m), so any writer whose first bin "starts" that far out is
/// reporting metres. (Metre values below the threshold in a pre-v2.4 file
/// are indistinguishable from km; range starts sit at gate-size scale,
/// hundreds of metres, and 0 reads identically in either unit.) A file this
/// crate's writer made (root `how/software`, [`crate::write`]) states km
/// before v2.4 at any distance: a sweep whose first bin starts 20.48 km out
/// read back 20 km short (the `writers` fuzz target).
pub(crate) fn first_gate_m_from_rstart(rstart: f64, unit: RstartUnit) -> f64 {
    let metres = match unit {
        RstartUnit::Metres => rstart,
        RstartUnit::Km => rstart * 1000.0,
        RstartUnit::KmOrMetres if rstart > RSTART_SANE_MAX_KM => rstart,
        RstartUnit::KmOrMetres => rstart * 1000.0,
    };
    (metres * 1000.0).round() / 1000.0
}

/// The unit of a file's `where/rstart` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RstartUnit {
    /// ODIM_H5 v2.4: metres.
    Metres,
    /// Km, by the file's own writer (this crate's, before v2.4).
    Km,
    /// Km by the specification, metres beyond [`RSTART_SANE_MAX_KM`].
    KmOrMetres,
}

/// How the file states `rstart`: metres when it declares ODIM_H5 v2.4, km
/// when this crate wrote it (root `how/software`), else km or metres by
/// size.
pub(crate) fn rstart_unit(file: &H5File<'_>) -> RstartUnit {
    let v2_4 = file
        .attr("/", "Conventions")
        .and_then(|attr| attr.as_str().map(|text| text.trim() == "ODIM_H5/V2_4"))
        .unwrap_or(false);
    let ours = file
        .attr("/how", "software")
        .and_then(|attr| {
            attr.as_str()
                .map(|software| software == crate::write::SOFTWARE)
        })
        .unwrap_or(false);
    if v2_4 {
        RstartUnit::Metres
    } else if ours {
        RstartUnit::Km
    } else {
        RstartUnit::KmOrMetres
    }
}

/// A velocity gate within this distance of the plane's physical `offset` is
/// treated as sitting exactly on the collapsed no-data / zero code. The gate
/// spacing of any Doppler quantum (≈0.3 m/s for a 40 m/s Nyquist) is orders of
/// magnitude larger, so this only ever catches the exact `offset` fill.
const VELOCITY_OFFSET_EPS: f32 = 1.0e-6;

/// The `nodata` / `undetect` sentinels and physical offset a field's coding
/// declares, in physical units: the values the plane's `what` group wrote.
/// The offset is 0 for a transform without one ([`LinearTransform::add_offset`]).
fn plane_sentinels(field: &Field) -> (Option<f64>, Option<f64>, f64) {
    fn packed<T: Copy + Into<f64>>(code: Option<T>, transform: LinearTransform) -> Option<f64> {
        code.map(|code| f64::from(transform.apply(code.into())))
    }
    match &field.data {
        FieldData::U8 { coding, .. } => (
            packed(coding.fill_value, coding.transform),
            packed(coding.undetect, coding.transform),
            coding.transform.add_offset().unwrap_or(0.0),
        ),
        FieldData::U16 { coding, .. } => (
            packed(coding.fill_value, coding.transform),
            packed(coding.undetect, coding.transform),
            coding.transform.add_offset().unwrap_or(0.0),
        ),
        FieldData::I8 { coding, .. } => (
            packed(coding.fill_value, coding.transform),
            packed(coding.undetect, coding.transform),
            coding.transform.add_offset().unwrap_or(0.0),
        ),
        FieldData::I16 { coding, .. } => (
            packed(coding.fill_value, coding.transform),
            packed(coding.undetect, coding.transform),
            coding.transform.add_offset().unwrap_or(0.0),
        ),
        FieldData::I32 { coding, .. } => (
            packed(coding.fill_value, coding.transform),
            packed(coding.undetect, coding.transform),
            coding.transform.add_offset().unwrap_or(0.0),
        ),
        FieldData::F32 { coding, .. } => {
            let offset = coding
                .transform
                .and_then(LinearTransform::add_offset)
                .unwrap_or(0.0);
            (
                coding.fill_value.map(f64::from),
                coding.undetect.map(f64::from),
                offset,
            )
        }
        FieldData::F64 { coding, .. } => {
            let offset = coding
                .transform
                .and_then(LinearTransform::add_offset)
                .unwrap_or(0.0);
            (coding.fill_value, coding.undetect, offset)
        }
    }
}

/// Recover no-echo Doppler-velocity gates that a copied-`what`-group writer
/// (AEMET Spain / IRIS 10.3) leaves decoding to a spurious `offset` (0 m/s).
/// Returns the number of gates masked.
///
/// Such writers stamp the velocity plane's `nodata`/`undetect` with the
/// REFLECTIVITY sentinels while filling no-echo velocity gates with the
/// physical `offset`, so those gates match no declared sentinel and decode
/// as a wall of 0 m/s (in xradar and Py-ART too). There is no per-plane
/// attribute that distinguishes the fill from a genuine 0 m/s reading, so
/// the file's own reflectivity no-echo mask is the only ground truth: a
/// velocity gate on `offset` with no co-located reflectivity echo is the
/// writer's collapsed no-data code; the same value where reflectivity IS
/// present is a real 0 m/s reading and is preserved.
///
/// This is an opt-in post-pass: [`read_odim_h5_volume`] stores every plane
/// verbatim. The velocity and reflectivity planes are the highest-priority
/// ones of each kind (`canonical_field`). Guarded by the copied-what-group
/// signature (velocity sentinels equal the reflectivity sentinels) so
/// conformant writers — which give velocity its own distinct sentinels,
/// already masked by the coding — are never touched. Also a no-op for a
/// sweep whose two planes do not share ray/gate geometry.
pub fn recover_copied_whatgroup_velocity_nodata(volume: &mut Volume) -> usize {
    volume
        .sweeps
        .iter_mut()
        .map(recover_sweep_velocity_nodata)
        .sum()
}

fn recover_sweep_velocity_nodata(sweep: &mut Sweep) -> usize {
    let (Some(velocity_name), Some(reflectivity_name)) = (
        canonical_field(sweep, CanonicalMoment::Velocity),
        canonical_field(sweep, CanonicalMoment::Reflectivity),
    ) else {
        return 0;
    };
    let (Some(velocity_index), Some(reflectivity_index)) = (
        sweep.field_index(&velocity_name),
        sweep.field_index(&reflectivity_name),
    ) else {
        return 0;
    };
    let (vel_nodata, vel_undetect, offset) = plane_sentinels(&sweep.fields[velocity_index]);
    let (ref_nodata, ref_undetect, _) = plane_sentinels(&sweep.fields[reflectivity_index]);
    // Copied-what-group signature: the velocity plane carries the reflectivity
    // plane's no-data sentinels verbatim (and at least one is present).
    let sentinels_copied = vel_nodata == ref_nodata
        && vel_undetect == ref_undetect
        && (vel_nodata.is_some() || vel_undetect.is_some());
    if !sentinels_copied {
        return 0;
    }
    // Mask in place (reflectivity borrowed, velocity taken out for the pass)
    // so no gate-index list proportional to the sweep is built.
    let mut velocity = std::mem::replace(
        &mut sweep.fields[velocity_index],
        Field::new(
            velocity_name.clone(),
            GateMapping::IDENTITY,
            0,
            FieldData::U8 {
                values: Vec::new(),
                coding: IntCoding::new(LinearTransform::IcdScaleOffset {
                    scale: 1.0,
                    offset: 0.0,
                }),
            },
        ),
    );
    let masked = mask_offset_fill_without_echo(
        &mut velocity,
        &sweep.fields[reflectivity_index],
        offset as f32,
    );
    sweep.fields[velocity_index] = velocity;
    masked
}

/// Set velocity gates that sit on the plane's physical `offset` where the
/// reflectivity field has no echo to the velocity field's fill code. A no-op
/// unless both fields share ray/gate geometry. Returns the gates masked.
fn mask_offset_fill_without_echo(velocity: &mut Field, reflectivity: &Field, offset: f32) -> usize {
    let (rows, gates) = velocity.shape();
    if reflectivity.shape() != (rows, gates) {
        return 0; // differing geometry: do not risk mis-masking
    }
    let mut masked = 0;
    for row in 0..rows {
        for gate in 0..gates {
            // Reflectivity no-echo: no value (sentinel) or NaN.
            let ref_no_echo = reflectivity.value(row, gate).is_none_or(|z| !z.is_finite());
            if !ref_no_echo {
                continue;
            }
            if let Some(value) = velocity.value(row, gate)
                && value.is_finite()
                && (value - offset).abs() <= VELOCITY_OFFSET_EPS
            {
                mask_gate_no_data(velocity, row * gates + gate);
                masked += 1;
            }
        }
    }
    masked
}

/// Set one flat gate index to the field's fill code: `_FillValue` for
/// integer storage (unchanged when the plane declares none — nothing
/// transparent to write), `_FillValue` else NaN for float storage.
fn mask_gate_no_data(field: &mut Field, index: usize) {
    match &mut field.data {
        FieldData::U8 { values, coding } => {
            if let (Some(fill), Some(slot)) = (coding.fill_value, values.get_mut(index)) {
                *slot = fill;
            }
        }
        FieldData::U16 { values, coding } => {
            if let (Some(fill), Some(slot)) = (coding.fill_value, values.get_mut(index)) {
                *slot = fill;
            }
        }
        FieldData::I8 { values, coding } => {
            if let (Some(fill), Some(slot)) = (coding.fill_value, values.get_mut(index)) {
                *slot = fill;
            }
        }
        FieldData::I16 { values, coding } => {
            if let (Some(fill), Some(slot)) = (coding.fill_value, values.get_mut(index)) {
                *slot = fill;
            }
        }
        FieldData::I32 { values, coding } => {
            if let (Some(fill), Some(slot)) = (coding.fill_value, values.get_mut(index)) {
                *slot = fill;
            }
        }
        FieldData::F32 { values, coding } => {
            if let Some(slot) = values.get_mut(index) {
                *slot = coding.fill_code();
            }
        }
        FieldData::F64 { values, coding } => {
            if let Some(slot) = values.get_mut(index) {
                *slot = coding.fill_code();
            }
        }
    }
}

/// Canonical moment kinds of ODIM quantity codes (spec Table 16), used to
/// pick the reflectivity and velocity planes of the copied-what-group
/// recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum CanonicalMoment {
    Reflectivity,
    Velocity,
    SpectrumWidth,
    DifferentialReflectivity,
    CorrelationCoefficient,
    DifferentialPhase,
    SpecificDifferentialPhase,
}

/// Map ODIM quantity codes (spec Table 16) onto the canonical moment set.
pub(crate) fn canonical_quantity(quantity: &str) -> Option<CanonicalMoment> {
    match quantity {
        "DBZH" | "DBZV" | "TH" | "TV" | "DBZ" => Some(CanonicalMoment::Reflectivity),
        "VRADH" | "VRADV" | "VRAD" | "VRADDH" => Some(CanonicalMoment::Velocity),
        "WRADH" | "WRADV" | "WRAD" => Some(CanonicalMoment::SpectrumWidth),
        // The unfiltered dual-pol spellings come in both orders in the
        // wild: spec-style trailing U (ZDRU) and DWD's leading U (UZDR,
        // URHOHV — live opendata.dwd.de sweep files, 2026-06-12).
        "ZDR" | "ZDRU" | "UZDR" => Some(CanonicalMoment::DifferentialReflectivity),
        "RHOHV" | "RHOHVU" | "URHOHV" => Some(CanonicalMoment::CorrelationCoefficient),
        "PHIDP" | "PHIDPU" | "UPHIDP" => Some(CanonicalMoment::DifferentialPhase),
        "KDP" | "KDPU" => Some(CanonicalMoment::SpecificDifferentialPhase),
        _ => None,
    }
}

pub(crate) fn canonical_quantity_priority(quantity: &str) -> u8 {
    match quantity {
        // Prefer filtered reflectivity when a file carries both DBZH and
        // unfiltered TH/TV. ODIM writers do not guarantee plane order.
        "DBZH" | "DBZV" => 30,
        "DBZ" => 25,
        "TH" | "TV" => 10,
        // Prefer filtered dual-pol spellings over explicitly unfiltered ones.
        "ZDR" | "RHOHV" | "PHIDP" | "KDP" => 30,
        "ZDRU" | "UZDR" | "RHOHVU" | "URHOHV" | "PHIDPU" | "UPHIDP" | "KDPU" => 10,
        _ => 20,
    }
}

/// The field of `sweep` that best represents `moment`: the plane with the
/// highest [`canonical_quantity_priority`], earliest first among equals.
pub(crate) fn canonical_field(sweep: &Sweep, moment: CanonicalMoment) -> Option<FieldName> {
    let mut best: Option<(u8, &FieldName)> = None;
    for field in &sweep.fields {
        let name = field.name.as_str();
        if canonical_quantity(name) != Some(moment) {
            continue;
        }
        let priority = canonical_quantity_priority(name);
        if best.is_none_or(|(existing, _)| priority > existing) {
            best = Some((priority, &field.name));
        }
    }
    best.map(|(_, name)| name.clone())
}

/// Identifiers from the `/what` `source` attribute.
pub(crate) struct SiteIdentity {
    pub id: String,
    pub name: Option<String>,
    pub wmo: Option<String>,
    pub wigos: Option<String>,
}

/// Pick site id + display name out of the `/what` `source` attribute:
/// comma-separated "TYP:value" identifier pairs (spec Table 3), e.g.
/// "WMO:02606,RAD:SE50,PLC:Karlskrona,NOD:sekkr". Preference is
/// NOD > RAD > WMO regardless of pair order — operational files (RMI
/// Belgium, met.no) list WMO first but NOD is the canonical OPERA site
/// code (validated against bejab/norst sample volumes).
pub(crate) fn site_identity_from_source(source: &str) -> SiteIdentity {
    let (mut nod, mut rad, mut wmo, mut name, mut wigos) = (None, None, None, None, None);
    for pair in source.split(',') {
        let Some((key, value)) = pair.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        match key.trim() {
            "NOD" => nod = Some(value.to_uppercase()),
            "RAD" => rad = Some(value.to_owned()),
            "WMO" => wmo = Some(value.to_owned()),
            "PLC" => name = Some(value.to_owned()),
            "WIGOS" => wigos = Some(value.to_owned()),
            _ => {}
        }
    }
    let id = nod
        .or(rad)
        .or_else(|| wmo.clone())
        .unwrap_or_else(|| "ODIM".to_owned());
    SiteIdentity {
        id,
        name,
        wmo,
        wigos,
    }
}

fn parse_datetime(file: &H5File<'_>, group: &str) -> Option<DateTime<Utc>> {
    parse_datetime_pair(file, group, "date", "time")
}

fn parse_datetime_pair(
    file: &H5File<'_>,
    group: &str,
    date_attr: &str,
    time_attr: &str,
) -> Option<DateTime<Utc>> {
    let date = file.attr(group, date_attr)?.as_str()?.to_owned();
    let time = file.attr(group, time_attr)?.as_str()?.to_owned();
    let date = NaiveDate::parse_from_str(&date, "%Y%m%d").ok()?;
    let time = NaiveTime::parse_from_str(&time, "%H%M%S").ok()?;
    Some(Utc.from_utc_datetime(&NaiveDateTime::new(date, time)))
}

fn attr_f64(file: &H5File<'_>, path: &str, name: &str) -> Option<f64> {
    file.attr(path, name).as_ref().and_then(H5Attr::as_f64)
}

/// A numeric array attribute as f64 (a scalar counts as a one-element array).
fn attr_array(file: &H5File<'_>, path: &str, name: &str) -> Option<Vec<f64>> {
    match file.attr(path, name)? {
        H5Attr::F64Array(values) => Some(values),
        H5Attr::I64Array(values) => Some(values.into_iter().map(|v| v as f64).collect()),
        H5Attr::F64(value) => Some(vec![value]),
        H5Attr::I64(value) => Some(vec![value as f64]),
        H5Attr::Str(_) => None,
    }
}

/// The attributes of one `how` group, in header order, then those of its
/// subgroups as `<sub>.<name>` (DWD `how/radar_system`, MeteoSwiss
/// `how/MeteoSwiss`, SMHI `how/process_chain`); empty when the group does
/// not exist.
struct How(Vec<(Box<str>, AttrValue)>);

impl How {
    fn read(file: &H5File<'_>, path: &str) -> Self {
        let mut all = Passthrough {
            attrs: file.attr_entries(path),
            level: None,
        };
        if file.has_object(path) {
            all.push_tree(file, path, "", &|_| false, 0);
        }
        Self(all.attrs)
    }

    /// The finite numeric scalar `name`.
    fn number(&self, name: &str) -> Option<f64> {
        self.0
            .iter()
            .find(|(key, _)| &**key == name)
            .and_then(|(_, value)| match value {
                AttrValue::Scalar(scalar) => Some(scalar.as_f64()),
                _ => None,
            })
            .filter(|value| value.is_finite())
    }

    /// The attributes not named in `used`, verbatim.
    fn unused(&self, used: &BTreeSet<&str>) -> Vec<(Box<str>, AttrValue)> {
        self.0
            .iter()
            .filter(|(name, _)| !used.contains(&**name))
            .cloned()
            .collect()
    }
}

/// Site constants from ODIM_H5 `how` (v2.4 Table 8) into
/// `radar_parameters`: `beamwH`/`beamwV` (deg; the older `beamwidth` for
/// either), `antgainH`/`antgainV` (dB) and `RXbandwidth` (MHz), each from
/// the root `how`, else the first dataset's. Returns the root attributes it
/// used.
fn describe_volume(root: &How, first: &How, volume: &mut Volume) -> BTreeSet<&'static str> {
    let mut root_used = BTreeSet::new();
    let mut positive = |name: &'static str| {
        let root_value = root.number(name).filter(|value| *value > 0.0);
        if root_value.is_some() {
            root_used.insert(name);
        }
        root_value
            .or_else(|| first.number(name).filter(|value| *value > 0.0))
            .map(|value| value as f32)
    };
    let beam_width_h = positive("beamwH");
    let beam_width_v = positive("beamwV");
    let antenna_gain_h = positive("antgainH");
    let antenna_gain_v = positive("antgainV");
    let bandwidth_mhz = positive("RXbandwidth");
    let beamwidth = match (beam_width_h, beam_width_v) {
        (Some(_), Some(_)) => None,
        _ => positive("beamwidth"),
    };
    let parameters = &mut volume.radar_parameters;
    parameters.beam_width_h_deg = beam_width_h.or(beamwidth);
    parameters.beam_width_v_deg = beam_width_v.or(beamwidth);
    parameters.antenna_gain_h_db = antenna_gain_h;
    parameters.antenna_gain_v_db = antenna_gain_v;
    parameters.receiver_bandwidth_hz = bandwidth_mhz.map(|mhz| mhz * 1e6);
    root_used
}

/// A dataset's `how` settings: the dataset's value of an attribute, else
/// the root's (ODIM_H5: lower-level `how` attributes override higher ones),
/// and which attributes of each level a typed slot took.
struct SweepHow<'a> {
    dataset: &'a How,
    root: &'a How,
    dataset_used: BTreeSet<&'static str>,
    root_used: BTreeSet<&'static str>,
}

impl<'a> SweepHow<'a> {
    fn new(dataset: &'a How, root: &'a How) -> Self {
        Self {
            dataset,
            root,
            dataset_used: BTreeSet::new(),
            root_used: BTreeSet::new(),
        }
    }

    fn number(&self, name: &str) -> Option<f64> {
        self.dataset.number(name).or_else(|| self.root.number(name))
    }

    /// Record that the value [`Self::number`] returned for `name` went into
    /// a typed slot.
    fn used(&mut self, name: &'static str, used: bool) {
        if !used {
            return;
        }
        if self.dataset.number(name).is_some() {
            self.dataset_used.insert(name);
        } else if self.root.number(name).is_some() {
            self.root_used.insert(name);
        }
    }

    /// `radconstH`/`radconstV` (dB) with the antenna gains and the pulse
    /// width they apply to, as a `radar_calibration` entry (without an
    /// index); `None` without a radar constant.
    fn calibration(&mut self, pulse_width_s: Option<f32>) -> Option<RadarCalibration> {
        let mut value = |name: &'static str, positive: bool| {
            let value = self
                .number(name)
                .filter(|value| !positive || *value > 0.0)
                .map(|value| value as f32);
            self.used(name, value.is_some());
            value
        };
        let radar_constant_h = value("radconstH", false);
        let radar_constant_v = value("radconstV", false);
        if radar_constant_h.is_none() && radar_constant_v.is_none() {
            return None;
        }
        Some(RadarCalibration {
            radar_constant_h,
            radar_constant_v,
            antenna_gain_h_db: value("antgainH", true),
            antenna_gain_v_db: value("antgainV", true),
            pulse_width_s,
            ..RadarCalibration::default()
        })
    }

    /// Mark the dataset's site constants that equal the volume's
    /// `radar_parameters` as held there.
    fn site_constants(&mut self, parameters: &RadarParameters) {
        let same = |value: Option<f64>, slot: Option<f32>| {
            value.is_some_and(|value| Some(value as f32) == slot)
        };
        let dataset = self.dataset;
        let beamwidth = dataset.number("beamwidth");
        let checks: [(&'static str, bool); 6] = [
            (
                "beamwH",
                same(dataset.number("beamwH"), parameters.beam_width_h_deg),
            ),
            (
                "beamwV",
                same(dataset.number("beamwV"), parameters.beam_width_v_deg),
            ),
            (
                "beamwidth",
                same(beamwidth, parameters.beam_width_h_deg)
                    && same(beamwidth, parameters.beam_width_v_deg),
            ),
            (
                "antgainH",
                same(dataset.number("antgainH"), parameters.antenna_gain_h_db),
            ),
            (
                "antgainV",
                same(dataset.number("antgainV"), parameters.antenna_gain_v_db),
            ),
            (
                "RXbandwidth",
                same(
                    dataset.number("RXbandwidth").map(|mhz| mhz * 1e6),
                    parameters.receiver_bandwidth_hz,
                ),
            ),
        ];
        for (name, same) in checks {
            if same {
                self.dataset_used.insert(name);
            }
        }
    }
}

/// The `calib_index` of `entry` in `calibration`: an equal entry's, else a
/// new entry's.
fn calibration_index(calibration: &mut Vec<RadarCalibration>, entry: RadarCalibration) -> i32 {
    let found = calibration.iter().find(|existing| {
        RadarCalibration {
            calib_index: None,
            ..(*existing).clone()
        } == entry
    });
    if let Some(index) = found.and_then(|existing| existing.calib_index) {
        return index;
    }
    let index = i32::try_from(calibration.len()).unwrap_or(i32::MAX);
    calibration.push(RadarCalibration {
        calib_index: Some(index),
        ..entry
    });
    index
}

fn odim_radar_frequency_mhz(file: &H5File<'_>) -> Option<f64> {
    for name in ["frequency", "freq", "radar_frequency", "radar_frequency_hz"] {
        if let Some(value) = attr_f64(file, "/how", name)
            && let Some(mhz) = normalize_frequency_mhz(value)
        {
            return Some(mhz);
        }
    }
    for name in ["wavelength", "radar_wavelength", "wavelength_cm"] {
        if let Some(value) = attr_f64(file, "/how", name)
            && let Some(mhz) = frequency_mhz_from_wavelength(value)
        {
            return Some(mhz);
        }
    }
    None
}

fn normalize_frequency_mhz(value: f64) -> Option<f64> {
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
    (1000.0..=12_000.0).contains(&mhz).then_some(mhz)
}

fn frequency_mhz_from_wavelength(value: f64) -> Option<f64> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let meters = if value > 1.0 { value / 100.0 } else { value };
    let mhz = 299.792_458 / meters;
    (1000.0..=12_000.0).contains(&mhz).then_some(mhz)
}

pub(crate) fn invalid(reason: impl Into<String>) -> OdimError {
    OdimError::InvalidMessage {
        offset: 0,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `startazA` of 1.6e185 (the `writers` fuzz target, a corrupt FRALE
    /// scan) gave an infinite azimuth when the mean was cast to f32 before
    /// wrapping, which the CfRadial reader's wrap then read back as NaN.
    #[test]
    fn azimuths_wrap_before_narrowing() {
        let mean = (1.556e185 + 148.0 + 360.0) / 2.0;
        assert_eq!(azimuth_f32(mean), mean.rem_euclid(360.0) as f32);
        assert!(azimuth_f32(mean).is_finite());
        assert_eq!(azimuth_f32(359.999_999_99), 0.0);
        assert_eq!(azimuth_f32(-1.0e-300), 0.0);
        assert_eq!(azimuth_f32(725.0), 5.0);
        assert_eq!(azimuth_f32(f64::INFINITY), f32::INFINITY);
        assert!(azimuth_f32(f64::NAN).is_nan());
    }

    #[test]
    fn quantity_codes_map_to_moments() {
        assert_eq!(
            canonical_quantity("DBZH"),
            Some(CanonicalMoment::Reflectivity)
        );
        assert_eq!(canonical_quantity("VRADH"), Some(CanonicalMoment::Velocity));
        assert_eq!(
            canonical_quantity("WRADH"),
            Some(CanonicalMoment::SpectrumWidth)
        );
        assert_eq!(
            canonical_quantity("RHOHV"),
            Some(CanonicalMoment::CorrelationCoefficient)
        );
        assert_eq!(canonical_quantity("QIND"), None);
    }

    /// Met Eireann Shannon carries DBZH, TH and VRADH on every sweep: the
    /// filtered DBZH is the reflectivity whatever the dataset order (the real
    /// sweep with its fields reversed), VRADH the velocity, and no quantity
    /// is spectrum width.
    #[test]
    fn filtered_odim_quantities_win_duplicate_canonical_moments() {
        let bytes = recast_radar_testdata::bytes("odim-iesha-20260305-0115-pvol")
            .unwrap_or_else(|err| panic!("{err}"));
        let volume = read_odim_h5_volume(&bytes).unwrap();
        let mut sweep = volume.sweeps[0].clone();
        let names: Vec<&str> = sweep.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["DBZH", "TH", "VRADH"]);
        for reversed in [false, true] {
            if reversed {
                sweep.fields.reverse();
                assert_eq!(sweep.fields[1].name.as_str(), "TH");
            }
            assert_eq!(
                canonical_field(&sweep, CanonicalMoment::Reflectivity),
                Some(FieldName::Dbzh)
            );
            assert_eq!(
                canonical_field(&sweep, CanonicalMoment::Velocity),
                Some(FieldName::Vradh)
            );
            assert_eq!(
                canonical_field(&sweep, CanonicalMoment::SpectrumWidth),
                None
            );
        }
        assert!(canonical_quantity_priority("DBZH") > canonical_quantity_priority("TH"));
    }

    #[test]
    fn site_identity_prefers_nod_over_wmo_regardless_of_pair_order() {
        // Real operational source strings put WMO first; NOD must win.
        let identity = site_identity_from_source(
            "WMO:06410,RAD:BX42,PLC:Jabbeke,NOD:bejab,CTY:605,CMT:bejab_scan_v3_Z_dBZ",
        );
        assert_eq!(identity.id, "BEJAB");
        assert_eq!(identity.name.as_deref(), Some("Jabbeke"));
        assert_eq!(identity.wmo.as_deref(), Some("06410"));
        // No NOD: fall back RAD, then WMO; empty values are skipped.
        let identity = site_identity_from_source("RAD:AU40,PLC:CapFlat,CTY:500,STN:70341");
        assert_eq!(identity.id, "AU40");
        let identity = site_identity_from_source("WMO:01104,NOD:");
        assert_eq!(identity.id, "01104");
        let identity = site_identity_from_source("CMT:whatever");
        assert_eq!(identity.id, "ODIM");
    }

    #[test]
    fn rstart_beyond_sane_range_reinterprets_as_metres() {
        use RstartUnit::{Km, KmOrMetres, Metres};
        // Spec-conformant km values pass through unchanged.
        assert_eq!(first_gate_m_from_rstart(0.0, KmOrMetres), 0.0);
        assert_eq!(first_gate_m_from_rstart(0.05, KmOrMetres), 50.0);
        assert_eq!(first_gate_m_from_rstart(5.0, KmOrMetres), 5_000.0);
        // Half a metre survives (601 m gates centred from 0 m).
        assert_eq!(first_gate_m_from_rstart(-0.3005, KmOrMetres), -300.5);
        // ODIM_H5 v2.4 states metres.
        assert_eq!(first_gate_m_from_rstart(5.0, Metres), 5.0);
        assert_eq!(first_gate_m_from_rstart(2000.0, Metres), 2000.0);
        // Just under the physical-sanity bound: still km.
        assert_eq!(first_gate_m_from_rstart(19.9, KmOrMetres), 19_900.0);
        assert_eq!(first_gate_m_from_rstart(20.0, KmOrMetres), 20_000.0);
        // AEMET metre-valued rstart (125/167/200 observed across the
        // network, 2026-07-07): reinterpreted as metres, not 125+ km.
        assert_eq!(first_gate_m_from_rstart(125.0, KmOrMetres), 125.0);
        assert_eq!(first_gate_m_from_rstart(167.0, KmOrMetres), 167.0);
        assert_eq!(first_gate_m_from_rstart(200.0, KmOrMetres), 200.0);
        // This crate's writer states km at any distance.
        assert_eq!(first_gate_m_from_rstart(20.48, Km), 20_480.0);
        assert_eq!(first_gate_m_from_rstart(125.0, Km), 125_000.0);
    }

    #[test]
    fn integer_coding_keeps_nodata_and_undetect_apart() {
        let coding: IntCoding<u8> = int_coding(
            LinearTransform::CfScaleOffset {
                scale_factor: 0.5,
                add_offset: -32.0,
                attr_width: FloatWidth::F64,
            },
            Some(255.0),
            Some(0.0),
        );
        assert_eq!(coding.fill_value, Some(255));
        assert_eq!(coding.undetect, Some(0));
        // Only `undetect` declared: it doubles as the fill code.
        let coding: IntCoding<u16> = int_coding(
            LinearTransform::CfScaleOffset {
                scale_factor: 1.0,
                add_offset: 0.0,
                attr_width: FloatWidth::F64,
            },
            None,
            Some(3.0),
        );
        assert_eq!(coding.fill_value, Some(3));
        assert_eq!(coding.undetect, Some(3));
    }

    // ----- copied-what-group velocity recovery on real AEMET planes --------
    //
    // Input: corpus entry `odim-espdg-20260707-1927-pvol-dbzh-vradh` (AEMET
    // Perdiguera, IRIS 10.3 export). Both datasets stamp the DBZH `what`
    // sentinels (nodata 95.5, undetect -32.0, offset 0.0, gain 1.0) onto
    // VRADH, whose no-echo gates hold the offset (0 m/s). The decoder keeps
    // the datasets in file order: sweep 0 is dataset1 (1.5 deg), sweep 1 is
    // dataset2 (0.5 deg).
    //
    // Expected values: tools/golden_io_formats.py, section `odim`, keys
    // `espdg_recovery` (h5py raw planes: a fill gate is DBZH no-echo and VRADH
    // on offset; a genuine zero is DBZH echo and VRADH on offset) and
    // `espdg_distinct_sentinel_mutation` (the file offset of dataset2 VRADH
    // what/nodata, found by editing candidates and reading them back with
    // libhdf5; the v2 object-header checksum recomputed with lookup3; h5py
    // reads the edited file).

    const ESPDG: &str = "odim-espdg-20260707-1927-pvol-dbzh-vradh";
    /// dataset2 VRADH what/nodata f64 value (95.5) and its OHDR checksum.
    const VRADH_NODATA_OFFSET: usize = 101_591;
    const VRADH_OHDR_CHECKSUM_OFFSET: usize = 101_629;
    const ORIGINAL_CHECKSUM: u32 = 320_803_438;
    const DISTINCT_NODATA: f64 = -9999.0;
    const DISTINCT_CHECKSUM: u32 = 3_075_319_015;
    /// Sweep indexes of the two datasets.
    const DATASET1: usize = 0;
    const DATASET2: usize = 1;

    fn espdg_bytes() -> Vec<u8> {
        recast_radar_testdata::bytes(ESPDG).unwrap_or_else(|err| panic!("{err}"))
    }

    /// The same file with dataset2 (0.5 deg) VRADH what/nodata rewritten to a
    /// value no gate holds: a writer that gives velocity its own sentinels.
    fn espdg_with_distinct_velocity_nodata() -> Vec<u8> {
        let mut bytes = espdg_bytes();
        let nodata = &bytes[VRADH_NODATA_OFFSET..VRADH_NODATA_OFFSET + 8];
        assert_eq!(f64::from_le_bytes(nodata.try_into().unwrap()), 95.5);
        let checksum = &bytes[VRADH_OHDR_CHECKSUM_OFFSET..VRADH_OHDR_CHECKSUM_OFFSET + 4];
        assert_eq!(
            u32::from_le_bytes(checksum.try_into().unwrap()),
            ORIGINAL_CHECKSUM
        );
        bytes[VRADH_NODATA_OFFSET..VRADH_NODATA_OFFSET + 8]
            .copy_from_slice(&DISTINCT_NODATA.to_le_bytes());
        bytes[VRADH_OHDR_CHECKSUM_OFFSET..VRADH_OHDR_CHECKSUM_OFFSET + 4]
            .copy_from_slice(&DISTINCT_CHECKSUM.to_le_bytes());
        bytes
    }

    fn velocity(sweep: &Sweep) -> &Field {
        sweep.field(&FieldName::Vradh).expect("VRADH")
    }

    /// (valid gates, missing flat-index sum, missing flat-index square sum,
    /// 0 m/s gates, 0 m/s flat-index sum) of a plane, with flat index
    /// `ray * ngates + gate`.
    fn velocity_summary(field: &Field) -> (usize, u64, u64, usize, u64) {
        let ngates = field.ngates as usize;
        let mut summary = (0usize, 0u64, 0u64, 0usize, 0u64);
        for ray in 0..field.nrays as usize {
            for gate in 0..ngates {
                let index = (ray * ngates + gate) as u64;
                match field.value(ray, gate) {
                    Some(value) => {
                        summary.0 += 1;
                        if value == 0.0 {
                            summary.3 += 1;
                            summary.4 += index;
                        }
                    }
                    None => {
                        summary.1 += index;
                        summary.2 += index * index;
                    }
                }
            }
        }
        summary
    }

    /// golden espdg_recovery.dataset2: 89237 fill gates masked (index sum
    /// 4625669116, square sum 326315664938928), 12869 genuine 0 m/s gates
    /// with echo kept (index sum 821953830), 18403 valid gates left.
    const DATASET2_RECOVERED: (usize, u64, u64, usize, u64) = (
        18_403,
        4_625_669_116,
        326_315_664_938_928,
        12_869,
        821_953_830,
    );
    /// golden espdg_recovery.dataset1: 90846 fill gates, 8141 genuine zeros,
    /// 16794 valid gates.
    const DATASET1_RECOVERED: (usize, u64, u64, usize, u64) = (
        16_794,
        4_709_191_624,
        333_031_625_592_866,
        8_141,
        511_192_353,
    );

    #[test]
    fn copied_whatgroup_recovery_masks_only_no_echo_offset_gates() {
        // The decoder stores the planes verbatim: the 0.5 deg VRADH plane
        // decodes with the full 0 m/s wall, 107640 valid gates, 102106 of
        // them zero (89237 fill + 12869 genuine).
        let mut volume = read_odim_h5_volume(&espdg_bytes()).expect("decode espdg");
        assert!((volume.sweeps[DATASET1].fixed_angle_deg - 1.5).abs() < 0.01);
        let low = &volume.sweeps[DATASET2];
        assert!((low.fixed_angle_deg - 0.5).abs() < 0.01);
        let before = velocity_summary(velocity(low));
        assert_eq!((before.0, before.1, before.3), (107_640, 0, 102_106));
        // Velocity carries the reflectivity sentinels: the copied-what-group
        // signature.
        let dbzh = low.field(&FieldName::Dbzh).expect("DBZH");
        assert_eq!(plane_sentinels(velocity(low)), plane_sentinels(dbzh));
        assert_eq!(plane_sentinels(dbzh), (Some(95.5), Some(-32.0), 0.0));

        let masked = recover_copied_whatgroup_velocity_nodata(&mut volume);

        assert_eq!(masked, 89_237 + 90_846);
        let low = velocity(&volume.sweeps[DATASET2]);
        assert_eq!(velocity_summary(low), DATASET2_RECOVERED);
        // First fill gate (0,0) masked; first genuine zero (0,32) kept.
        assert_eq!(low.value(0, 0), None);
        assert_eq!(low.value(0, 32), Some(0.0));
        assert_eq!(
            velocity_summary(velocity(&volume.sweeps[DATASET1])),
            DATASET1_RECOVERED
        );
        // A second pass finds nothing left to mask.
        assert_eq!(recover_copied_whatgroup_velocity_nodata(&mut volume), 0);
    }

    #[test]
    fn distinct_velocity_sentinels_are_never_reflectivity_gated() {
        // dataset2 VRADH now declares nodata -9999.0 while DBZH keeps 95.5: a
        // conformant writer. golden espdg_distinct_sentinel_mutation: h5py
        // reads 107640 non-sentinel VRADH gates, 102106 of them 0 m/s.
        let edited = espdg_with_distinct_velocity_nodata();
        let mut volume = read_odim_h5_volume(&edited).expect("decode edited espdg");
        let low = &volume.sweeps[DATASET2];
        let summary = velocity_summary(velocity(low));
        assert_eq!((summary.0, summary.1, summary.3), (107_640, 0, 102_106));
        assert_eq!(plane_sentinels(velocity(low)).0, Some(DISTINCT_NODATA));
        // Co-located no-echo reflectivity at (0,0) does not mask velocity.
        assert_eq!(low.field(&FieldName::Dbzh).expect("DBZH").value(0, 0), None);
        assert_eq!(velocity(low).value(0, 0), Some(0.0));

        // The recovery leaves the edited sweep alone and still recovers the
        // unedited 1.5 deg sweep, which carries the copied sentinels.
        let masked = recover_copied_whatgroup_velocity_nodata(&mut volume);
        assert_eq!(masked, 90_846);
        assert_eq!(
            velocity_summary(velocity(&volume.sweeps[DATASET2])),
            summary
        );
        assert_eq!(
            velocity_summary(velocity(&volume.sweeps[DATASET1])),
            DATASET1_RECOVERED
        );
    }
}
