//! Reading an ODIM_H5 polar volume off disk.
//!
//! One entry point, [`read_volume`], handles both shapes the EUMETNET OPERA
//! feed actually serves:
//!
//! * `/what/object = "SCAN"` -- a single elevation cut, one `/dataset1`,
//!   carrying one moment (Germany) or several (Finland);
//! * `/what/object = "PVOL"` -- a whole volume, `/dataset1`..`/datasetN`,
//!   carrying one moment per sweep (Romania) or nine (the Netherlands).
//!
//! Nothing about the read is conditioned on which one it is: a `SCAN` is a
//! volume with one sweep. That is deliberate, because the split is a national
//! packaging choice rather than a difference in the data model, and a decoder
//! with two paths would have two sets of bugs.
//!
//! # What this decoder refuses
//!
//! Geometry that contradicts the payload is refused rather than repaired. If
//! `/datasetN/where` says 360 rays of 720 bins and the payload is 360x360,
//! there is no reading of the file that makes the gates land where the
//! metadata says they do, and every gate would be assimilated at the wrong
//! range. The same goes for a non-finite calibration, an elevation off the
//! sphere, and a missing `undetect` when `nodata` is present.

use std::path::Path;

use hdf5_reader::group::Group;
use hdf5_reader::messages::datatype::Datatype;
use hdf5_reader::{Dataset, Hdf5File};

use crate::attrs::{Attrs, indexed_suffix, optional_group};
use crate::censor::{self, Census};
use crate::error::{OdimError, Result};
use crate::quantity;
use crate::volume::{
    AzimuthSource, Calibration, Moment, Nyquist, PolarVolume, Site, Source, Sweep, SystemNotes,
    parse_odim_datetime,
};

/// The ODIM `/what/object` values this decoder claims to read.
const POLAR_OBJECTS: [&str; 2] = ["PVOL", "SCAN"];

/// What to read, and what to skip.
#[derive(Debug, Clone, Default)]
pub struct DecodeOptions {
    /// Read only these ODIM quantities. `None` reads all of them.
    pub quantities: Option<Vec<String>>,
    /// Read only these `/datasetN` indices. `None` reads all of them.
    pub sweep_indices: Option<Vec<usize>>,
    /// Read geometry and calibration but no payload, leaving every moment's
    /// `values`/`censor` empty and its census zero. This is what `inspect`
    /// uses so that surveying a 31 MB Dutch volume does not decode 20 million
    /// gates to answer a question about elevations.
    pub geometry_only: bool,
}

impl DecodeOptions {
    /// Read everything.
    pub fn all() -> Self {
        DecodeOptions::default()
    }

    /// Read geometry only.
    pub fn geometry_only() -> Self {
        DecodeOptions {
            geometry_only: true,
            ..DecodeOptions::default()
        }
    }

    /// Restrict to one quantity.
    pub fn quantity(mut self, quantity: impl Into<String>) -> Self {
        self.quantities
            .get_or_insert_with(Vec::new)
            .push(quantity.into());
        self
    }

    fn wants_quantity(&self, name: &str) -> bool {
        match &self.quantities {
            None => true,
            Some(list) => list.iter().any(|q| q == name),
        }
    }

    fn wants_sweep(&self, index: usize) -> bool {
        match &self.sweep_indices {
            None => true,
            Some(list) => list.contains(&index),
        }
    }
}

/// Read an ODIM_H5 polar volume, decoding every sweep and every moment.
pub fn read_volume(path: impl AsRef<Path>) -> Result<PolarVolume> {
    read_volume_with(path, &DecodeOptions::all())
}

/// Read an ODIM_H5 polar volume, decoding only what `options` asks for.
pub fn read_volume_with(path: impl AsRef<Path>, options: &DecodeOptions) -> Result<PolarVolume> {
    let path = path.as_ref().to_path_buf();
    if !path.exists() {
        return Err(OdimError::Io {
            path: path.clone(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        });
    }
    let file = Hdf5File::open(&path).map_err(|err| OdimError::Hdf5 {
        path: path.clone(),
        detail: err.to_string(),
    })?;

    let root = file.root_group().map_err(|err| OdimError::Hdf5 {
        path: path.clone(),
        detail: format!("no root group: {err}"),
    })?;
    let root_attrs = Attrs::read("/", &root)?;
    let conventions = root_attrs.string_opt("Conventions")?;

    // --- /what -----------------------------------------------------------
    let what = file.group("/what").map_err(|_| OdimError::MissingGroup {
        path: "/what".to_string(),
    })?;
    let what = Attrs::read("/what", &what)?;
    let object = what.string("object")?;
    if !POLAR_OBJECTS.contains(&object.as_str()) {
        return Err(OdimError::UnsupportedObject {
            path: path.clone(),
            object,
            detail: format!(
                "this decoder reads the polar objects {POLAR_OBJECTS:?}. A composite \
                 (COMP), a vertical profile (VP), an RHI (ELEV) or a sector (AZIM) has a \
                 different geometry -- rays and range bins are not what it is made of -- \
                 and reading one here would produce a polar volume that never existed"
            ),
        });
    }
    let source = Source::parse(&what.string_opt("source")?.unwrap_or_default());
    let nominal_time = match (what.string_opt("date")?, what.string_opt("time")?) {
        (Some(date), Some(time)) => parse_odim_datetime(&date, &time),
        _ => None,
    };

    // --- /where ----------------------------------------------------------
    let where_group = file.group("/where").map_err(|_| OdimError::MissingGroup {
        path: "/where".to_string(),
    })?;
    let where_attrs = Attrs::read("/where", &where_group)?;
    let site = read_site(&where_attrs)?;

    // --- /how (optional) -------------------------------------------------
    let root_how = match optional_group(&root, "how") {
        Some(group) => Attrs::read("/how", &group)?,
        None => Attrs::empty("/how"),
    };
    let system = SystemNotes {
        wavelength_cm: root_how.f64_opt("wavelength")?,
        beamwidth_deg: root_how
            .f64_opt("beamwidth")?
            .or(root_how.f64_opt("beamwH")?),
        system: root_how.string_opt("system")?,
        software: root_how.string_opt("software")?,
        sw_version: root_how.string_opt("sw_version")?,
    };

    // --- sweeps ----------------------------------------------------------
    let mut indices = sweep_indices(&root)?;
    if indices.is_empty() {
        return Err(OdimError::format(
            format!("{} declares /what/object {object}", path.display()),
            "but carries no /datasetN group, so there is no sweep to read".to_string(),
        ));
    }
    // PyART orders sweeps by `int(name[7:])`, and so does this: `dataset10`
    // sorts before `dataset2` as a string, which would silently renumber a
    // twelve-sweep volume.
    indices.sort_unstable();

    let mut sweeps = Vec::new();
    for index in indices {
        if !options.wants_sweep(index) {
            continue;
        }
        sweeps.push(read_sweep(
            &file,
            &path,
            index,
            system.wavelength_cm,
            options,
        )?);
    }

    Ok(PolarVolume {
        object,
        conventions,
        version: what.string_opt("version")?,
        source,
        nominal_time,
        site,
        system,
        sweeps,
    })
}

fn read_site(attrs: &Attrs) -> Result<Site> {
    let latitude_deg = attrs.f64("lat")?;
    let longitude_deg = attrs.f64("lon")?;
    let height_m = attrs.f64("height")?;
    if !(-90.0..=90.0).contains(&latitude_deg) || !longitude_deg.is_finite() {
        return Err(OdimError::format(
            "reading /where",
            format!(
                "the antenna is placed at lat {latitude_deg}, lon {longitude_deg}, which is \
                 not on the Earth; every gate in this volume would be georeferenced from it"
            ),
        ));
    }
    if !height_m.is_finite() {
        return Err(OdimError::format(
            "reading /where",
            format!("the antenna height is {height_m}, which is not a height"),
        ));
    }
    Ok(Site {
        latitude_deg,
        longitude_deg,
        height_m,
    })
}

/// The `N` of every `/datasetN` group at the root.
fn sweep_indices(root: &Group) -> Result<Vec<usize>> {
    let groups = root.groups().map_err(|err| OdimError::Format {
        context: "listing the root group".to_string(),
        detail: err.to_string(),
    })?;
    Ok(groups
        .iter()
        .filter_map(|g| indexed_suffix(basename(g.name()), "dataset"))
        .collect())
}

fn basename(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

fn read_sweep(
    file: &Hdf5File,
    path: &Path,
    index: usize,
    volume_wavelength_cm: Option<f64>,
    options: &DecodeOptions,
) -> Result<Sweep> {
    let sweep_path = format!("/dataset{index}");
    let sweep_group = file
        .group(&sweep_path)
        .map_err(|_| OdimError::MissingGroup {
            path: sweep_path.clone(),
        })?;

    // --- /datasetN/where -------------------------------------------------
    let where_path = format!("{sweep_path}/where");
    let where_group = file
        .group(&where_path)
        .map_err(|_| OdimError::MissingGroup {
            path: where_path.clone(),
        })?;
    let where_attrs = Attrs::read(&where_path, &where_group)?;

    let elevation_deg = where_attrs.f64("elangle")?;
    if !(-90.0..=90.0).contains(&elevation_deg) {
        return Err(OdimError::format(
            format!("reading {where_path}"),
            format!("elangle is {elevation_deg} degrees, which is not an elevation"),
        ));
    }
    let nrays = where_attrs.usize("nrays")?;
    let nbins = where_attrs.usize("nbins")?;
    if nrays == 0 || nbins == 0 {
        return Err(OdimError::format(
            format!("reading {where_path}"),
            format!("the sweep declares {nrays} rays of {nbins} bins, which is not a sweep"),
        ));
    }
    let range_scale_m = where_attrs.f64("rscale")?;
    if !range_scale_m.is_finite() || range_scale_m <= 0.0 {
        return Err(OdimError::format(
            format!("reading {where_path}"),
            format!(
                "rscale is {range_scale_m} m; without a positive bin spacing no gate has a range"
            ),
        ));
    }
    // ODIM states rstart in kilometres and rscale in metres, in the same
    // group. The unit change is real, not a transcription slip.
    let range_start_km = where_attrs.f64_opt("rstart")?.unwrap_or(0.0);
    if !range_start_km.is_finite() {
        return Err(OdimError::format(
            format!("reading {where_path}"),
            format!("rstart is {range_start_km}, which is not a range"),
        ));
    }
    let range_start_m = range_start_km * 1000.0;
    let a1gate = where_attrs.usize_opt("a1gate")?;

    // --- /datasetN/what --------------------------------------------------
    let what_attrs = match optional_group(&sweep_group, "what") {
        Some(group) => Attrs::read(format!("{sweep_path}/what"), &group)?,
        None => Attrs::empty(format!("{sweep_path}/what")),
    };
    let start_time = match (
        what_attrs.string_opt("startdate")?,
        what_attrs.string_opt("starttime")?,
    ) {
        (Some(date), Some(time)) => parse_odim_datetime(&date, &time),
        _ => None,
    };
    let end_time = match (
        what_attrs.string_opt("enddate")?,
        what_attrs.string_opt("endtime")?,
    ) {
        (Some(date), Some(time)) => parse_odim_datetime(&date, &time),
        _ => None,
    };

    // --- /datasetN/how (optional) ----------------------------------------
    let how_attrs = match optional_group(&sweep_group, "how") {
        Some(group) => Attrs::read(format!("{sweep_path}/how"), &group)?,
        None => Attrs::empty(format!("{sweep_path}/how")),
    };
    let nyquist = Nyquist::establish(
        how_attrs.f64_opt("NI")?,
        how_attrs.f64_opt("highprf")?,
        how_attrs.f64_opt("lowprf")?,
        how_attrs.f64_opt("wavelength")?.or(volume_wavelength_cm),
    );
    let (azimuth_deg, azimuth_source) = resolve_azimuths(nrays, &how_attrs)?;
    let ray_elevation_deg = resolve_ray_elevations(nrays, &how_attrs)?;

    // --- /datasetN/dataM -------------------------------------------------
    let mut data_indices: Vec<usize> = sweep_group
        .groups()
        .map_err(|err| OdimError::Format {
            context: format!("listing {sweep_path}"),
            detail: err.to_string(),
        })?
        .iter()
        .filter_map(|g| indexed_suffix(basename(g.name()), "data"))
        .collect();
    data_indices.sort_unstable();

    let mut moments = Vec::new();
    for data_index in data_indices {
        let moment_path = format!("{sweep_path}/data{data_index}");
        let what_path = format!("{moment_path}/what");
        let what_group = file
            .group(&what_path)
            .map_err(|_| OdimError::MissingGroup {
                path: what_path.clone(),
            })?;
        let moment_what = Attrs::read(&what_path, &what_group)?;
        let quantity_name = moment_what.string("quantity")?;
        if !options.wants_quantity(&quantity_name) {
            continue;
        }
        moments.push(read_moment(
            file,
            path,
            &moment_path,
            &quantity_name,
            &moment_what,
            nrays,
            nbins,
            options,
        )?);
    }

    Ok(Sweep {
        index,
        path: sweep_path,
        elevation_deg,
        nrays,
        nbins,
        range_scale_m,
        range_start_m,
        a1gate,
        start_time,
        end_time,
        nyquist,
        azimuth_deg,
        azimuth_source,
        ray_elevation_deg,
        moments,
    })
}

/// Per-ray azimuths, measured if the file recorded them and nominal otherwise.
fn resolve_azimuths(nrays: usize, how: &Attrs) -> Result<(Vec<f64>, AzimuthSource)> {
    let start = how.f64_vec_opt("startazA")?;
    let stop = how.f64_vec_opt("stopazA")?;
    if let (Some(start), Some(stop)) = (start, stop) {
        if start.len() != nrays || stop.len() != nrays {
            return Err(OdimError::format(
                format!("reading {}", how.path()),
                format!(
                    "startazA has {} entries and stopazA has {}, but the sweep declares {nrays} \
                     rays; a per-ray azimuth list that does not match the ray count cannot be \
                     assigned to rays without guessing which rays it describes",
                    start.len(),
                    stop.len()
                ),
            ));
        }
        let azimuths = start
            .iter()
            .zip(&stop)
            .map(|(a, b)| circular_mean_deg(*a, *b))
            .collect();
        return Ok((azimuths, AzimuthSource::MeasuredStartStopMean));
    }
    // ODIM section 5.1: the first ray points north and the sweep proceeds
    // clockwise through a full rotation.
    let astart_deg = how.f64_opt("astart")?.unwrap_or(0.0);
    let step = 360.0 / nrays as f64;
    let azimuths = (0..nrays)
        .map(|i| normalise_deg(astart_deg + (i as f64 + 0.5) * step))
        .collect();
    Ok((azimuths, AzimuthSource::NominalFromRayCount { astart_deg }))
}

/// Per-ray elevations, when the file recorded the antenna's actual elevation.
fn resolve_ray_elevations(nrays: usize, how: &Attrs) -> Result<Option<Vec<f64>>> {
    let (Some(start), Some(stop)) = (how.f64_vec_opt("startelA")?, how.f64_vec_opt("stopelA")?)
    else {
        return Ok(None);
    };
    if start.len() != nrays || stop.len() != nrays {
        return Ok(None);
    }
    // Elevations do not wrap, so this is an arithmetic mean, unlike azimuth.
    Ok(Some(
        start
            .iter()
            .zip(&stop)
            .map(|(a, b)| 0.5 * (a + b))
            .collect(),
    ))
}

/// The mean of two azimuths, taken the short way round the circle.
///
/// A ray that starts at 359.5 and stops at 0.5 is centred on 0, not on 180.
/// This is the same construction Py-ART uses
/// (`np.angle(exp(i*start) + exp(i*stop))`), stated as an `atan2` because the
/// magnitude is irrelevant to the angle.
fn circular_mean_deg(start_deg: f64, stop_deg: f64) -> f64 {
    let a = start_deg.to_radians();
    let b = stop_deg.to_radians();
    let mean = (a.sin() + b.sin()).atan2(a.cos() + b.cos());
    normalise_deg(mean.to_degrees())
}

/// Fold an angle into `[0, 360)`.
fn normalise_deg(degrees: f64) -> f64 {
    let folded = degrees.rem_euclid(360.0);
    // rem_euclid can return exactly 360.0 for inputs a hair below zero.
    if folded >= 360.0 { 0.0 } else { folded }
}

#[allow(clippy::too_many_arguments)]
fn read_moment(
    file: &Hdf5File,
    path: &Path,
    moment_path: &str,
    quantity_name: &str,
    what: &Attrs,
    nrays: usize,
    nbins: usize,
    options: &DecodeOptions,
) -> Result<Moment> {
    let gain = what.f64_opt("gain")?.unwrap_or(1.0);
    let offset = what.f64_opt("offset")?.unwrap_or(0.0);
    if !gain.is_finite() || !offset.is_finite() {
        return Err(OdimError::format(
            format!("reading {moment_path}/what"),
            format!(
                "the calibration is gain {gain}, offset {offset}; a non-finite affine map turns \
                 every gate in this moment into a non-number"
            ),
        ));
    }
    let nodata = what.f64_opt("nodata")?;
    let undetect = what.f64_opt("undetect")?;
    if nodata.is_some() && undetect.is_none() {
        return Err(OdimError::format(
            format!("reading {moment_path}/what"),
            "the moment declares `nodata` but no `undetect`, so a gate the radar looked at and \
             found empty cannot be told from a gate it never measured. Every correct negative a \
             skill score is built on lives in that distinction, and reading an unmarked gate as \
             either one would silently invent observations"
                .to_string(),
        ));
    }
    let sentinels_collide = match (nodata, undetect) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    };
    let calibration = Calibration {
        gain,
        offset,
        nodata,
        undetect,
        sentinels_collide,
    };

    let info = quantity::describe(quantity_name);
    let data_path = format!("{moment_path}/data");
    let dataset = file.dataset(&data_path).map_err(|err| OdimError::Hdf5 {
        path: path.to_path_buf(),
        detail: format!("{data_path} did not open: {err}"),
    })?;

    // The shape check is the integrity gate: it is what makes a truncated or
    // mis-declared sweep a refusal rather than a plausible field.
    let shape = dataset.shape();
    if shape.len() != 2 {
        return Err(OdimError::format(
            format!("reading {data_path}"),
            format!(
                "the payload has rank {} with shape {shape:?}; a polar sweep is a \
                 [ray, bin] rectangle",
                shape.len()
            ),
        ));
    }
    if shape[0] as usize != nrays || shape[1] as usize != nbins {
        return Err(OdimError::format(
            format!("reading {data_path}"),
            format!(
                "the payload is {}x{} but the sweep's /where declares {nrays}x{nbins}. There is \
                 no reading of this file that puts the gates where the metadata says they are, \
                 so every gate would be assimilated at the wrong azimuth or range",
                shape[0], shape[1]
            ),
        ));
    }

    let mut moment = Moment {
        quantity: quantity_name.to_string(),
        path: moment_path.to_string(),
        unit: info.unit.to_string(),
        kind: info.kind,
        calibration,
        nrays,
        nbins,
        values: Vec::new(),
        censor: Vec::new(),
        census: Census::default(),
    };
    if options.geometry_only {
        return Ok(moment);
    }

    let raw = read_raw_f64(&dataset, path, &data_path)?;
    if raw.len() != nrays * nbins {
        return Err(OdimError::format(
            format!("reading {data_path}"),
            format!(
                "the payload decoded to {} values but the shape says {}",
                raw.len(),
                nrays * nbins
            ),
        ));
    }

    let mut values = Vec::with_capacity(raw.len());
    let mut codes = Vec::with_capacity(raw.len());
    let mut census = Census::default();
    for stored in raw {
        let code = classify(stored, &calibration);
        let value = if code == censor::MEASURED {
            let decoded = calibration.apply(stored);
            if !decoded.is_finite() {
                return Err(OdimError::format(
                    format!("reading {data_path}"),
                    format!(
                        "a gate holding raw {stored} calibrated to {decoded} with gain {gain} \
                         and offset {offset}; a non-finite value is neither a measurement nor a \
                         mask, and this decoder will not emit one as either"
                    ),
                ));
            }
            decoded
        } else {
            f64::NAN
        };
        census.tally(code);
        values.push(value);
        codes.push(code);
    }

    moment.values = values;
    moment.censor = codes;
    moment.census = census;
    Ok(moment)
}

/// Classify one raw storage value against the moment's declared sentinels.
///
/// The comparison is against the **raw** value, not the calibrated one,
/// because ODIM states both sentinels as raw storage values. It is exact
/// equality: every ODIM payload type in the wild is an integer, and an
/// integer raw value and an integer-valued `f64` sentinel compare exactly.
/// Py-ART masks the same way (`np.ma.masked_equal` on the raw array), so the
/// two agree gate for gate.
fn classify(raw: f64, calibration: &Calibration) -> u8 {
    if !raw.is_finite() {
        // Not a declared sentinel, but certainly not a measurement, and it
        // must never be admitted as a correct negative.
        return censor::NODATA;
    }
    let is_nodata = calibration.nodata.is_some_and(|s| raw == s);
    let is_undetect = calibration.undetect.is_some_and(|s| raw == s);
    match (is_nodata, is_undetect) {
        (true, true) => censor::SENTINEL_AMBIGUOUS,
        (true, false) => censor::NODATA,
        (false, true) => censor::UNDETECT,
        (false, false) => censor::MEASURED,
    }
}

/// Read a payload of any ODIM-legal storage type as `f64`, row-major.
///
/// Widening every type to `f64` here is what keeps the classification and the
/// affine decode written once. It mirrors `rustwx-io`'s
/// `hdf5_dataset_values_f64`, which does the same for the OPERA composite.
fn read_raw_f64(dataset: &Dataset, path: &Path, data_path: &str) -> Result<Vec<f64>> {
    let hdf5 = |err: hdf5_reader::error::Error| OdimError::Hdf5 {
        path: path.to_path_buf(),
        detail: format!("{data_path} did not decode: {err}"),
    };
    let values = match dataset.dtype() {
        Datatype::FixedPoint { size, signed, .. } => match (size, signed) {
            (1, false) => dataset
                .read_array::<u8>()
                .map_err(hdf5)?
                .iter()
                .map(|v| f64::from(*v))
                .collect(),
            (1, true) => dataset
                .read_array::<i8>()
                .map_err(hdf5)?
                .iter()
                .map(|v| f64::from(*v))
                .collect(),
            (2, false) => dataset
                .read_array::<u16>()
                .map_err(hdf5)?
                .iter()
                .map(|v| f64::from(*v))
                .collect(),
            (2, true) => dataset
                .read_array::<i16>()
                .map_err(hdf5)?
                .iter()
                .map(|v| f64::from(*v))
                .collect(),
            (4, false) => dataset
                .read_array::<u32>()
                .map_err(hdf5)?
                .iter()
                .map(|v| f64::from(*v))
                .collect(),
            (4, true) => dataset
                .read_array::<i32>()
                .map_err(hdf5)?
                .iter()
                .map(|v| f64::from(*v))
                .collect(),
            _ => {
                return Err(OdimError::format(
                    format!("reading {data_path}"),
                    format!(
                        "payload is a {size}-byte integer (signed: {signed}), which ODIM does not use for polar data"
                    ),
                ));
            }
        },
        Datatype::FloatingPoint { size, .. } => match size {
            4 => dataset
                .read_array::<f32>()
                .map_err(hdf5)?
                .iter()
                .map(|v| f64::from(*v))
                .collect(),
            8 => dataset
                .read_array::<f64>()
                .map_err(hdf5)?
                .iter()
                .copied()
                .collect(),
            _ => {
                return Err(OdimError::format(
                    format!("reading {data_path}"),
                    format!("payload is a {size}-byte float"),
                ));
            }
        },
        other => {
            return Err(OdimError::format(
                format!("reading {data_path}"),
                format!("payload has datatype {other:?}, which is not a number"),
            ));
        }
    };
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ray_spanning_north_is_centred_on_north_not_on_south() {
        // The wrap case: 359.5 -> 0.5 must average to 0, not 180.
        let mean = circular_mean_deg(359.5, 0.5);
        assert!(mean < 1e-9 || (360.0 - mean) < 1e-9, "{mean}");
    }

    #[test]
    fn ordinary_rays_average_the_obvious_way() {
        assert!((circular_mean_deg(10.0, 11.0) - 10.5).abs() < 1e-9);
        assert!((circular_mean_deg(180.0, 181.0) - 180.5).abs() < 1e-9);
        assert!((circular_mean_deg(270.0, 271.0) - 270.5).abs() < 1e-9);
    }

    #[test]
    fn azimuths_are_reported_in_zero_to_threesixty() {
        // Py-ART's np.angle yields (-180, 180]; this crate reports [0, 360),
        // so a westward ray is 270 here and -90 there. Same ray.
        let mean = circular_mean_deg(269.5, 270.5);
        assert!((mean - 270.0).abs() < 1e-9, "{mean}");
        assert_eq!(normalise_deg(-90.0), 270.0);
        assert_eq!(normalise_deg(360.0), 0.0);
        assert_eq!(normalise_deg(-1e-18), 0.0);
    }

    #[test]
    fn the_nominal_geometry_centres_the_first_ray_half_a_step_off_north() {
        let attrs = Attrs::empty("/dataset1/how");
        let (azimuths, source) = resolve_azimuths(360, &attrs).expect("nominal");
        assert_eq!(
            source,
            AzimuthSource::NominalFromRayCount { astart_deg: 0.0 }
        );
        assert_eq!(azimuths.len(), 360);
        assert!((azimuths[0] - 0.5).abs() < 1e-12);
        assert!((azimuths[359] - 359.5).abs() < 1e-12);
    }

    #[test]
    fn the_two_sentinels_are_told_apart_by_the_values_the_file_declares() {
        let cal = Calibration {
            gain: 0.5,
            offset: -32.0,
            nodata: Some(255.0),
            undetect: Some(0.0),
            sentinels_collide: false,
        };
        assert_eq!(classify(255.0, &cal), censor::NODATA);
        assert_eq!(classify(0.0, &cal), censor::UNDETECT);
        assert_eq!(classify(128.0, &cal), censor::MEASURED);
        // Neither sentinel is a tolerance band: 254 is a measurement.
        assert_eq!(classify(254.0, &cal), censor::MEASURED);
        assert_eq!(classify(1.0, &cal), censor::MEASURED);
    }

    #[test]
    fn colliding_sentinels_are_classified_ambiguous_rather_than_picked() {
        // Finnish VRADH declares nodata = undetect = 0. Neither state can be
        // proven, so neither is asserted.
        let cal = Calibration {
            gain: 0.059_805_118_110_236_22,
            offset: -7.655_055_118_110_236,
            nodata: Some(0.0),
            undetect: Some(0.0),
            sentinels_collide: true,
        };
        assert_eq!(classify(0.0, &cal), censor::SENTINEL_AMBIGUOUS);
        assert_eq!(classify(128.0, &cal), censor::MEASURED);
        // And it is not silently readable as clear air.
        assert!(!censor::is_observed(classify(0.0, &cal)));
    }

    #[test]
    fn a_moment_with_no_sentinels_at_all_is_all_measurement() {
        let cal = Calibration {
            gain: 1.0,
            offset: 0.0,
            nodata: None,
            undetect: None,
            sentinels_collide: false,
        };
        assert_eq!(classify(0.0, &cal), censor::MEASURED);
        assert_eq!(classify(65535.0, &cal), censor::MEASURED);
    }

    #[test]
    fn a_non_finite_raw_gate_is_never_admitted_as_clear_air() {
        let cal = Calibration {
            gain: 1.0,
            offset: 0.0,
            nodata: Some(-999.0),
            undetect: Some(0.0),
            sentinels_collide: false,
        };
        assert_eq!(classify(f64::NAN, &cal), censor::NODATA);
        assert!(!censor::is_observed(classify(f64::NAN, &cal)));
    }

    #[test]
    fn decode_options_filter_by_quantity_and_sweep() {
        let options = DecodeOptions::all().quantity("VRADH");
        assert!(options.wants_quantity("VRADH"));
        assert!(!options.wants_quantity("DBZH"));
        // No sweep filter means every sweep.
        assert!(options.wants_sweep(7));

        let options = DecodeOptions {
            sweep_indices: Some(vec![1, 3]),
            ..DecodeOptions::all()
        };
        assert!(options.wants_sweep(1));
        assert!(!options.wants_sweep(2));
        assert!(options.wants_quantity("anything"));
    }

    #[test]
    fn group_names_reduce_to_their_last_segment() {
        assert_eq!(basename("/dataset1"), "dataset1");
        assert_eq!(basename("dataset1"), "dataset1");
        assert_eq!(basename("/dataset1/data2"), "data2");
    }
}
