//! Observation reference semantics are rows. Acquisition, decoding, time
//! validation and missing-cell handling do not name an observation source.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, NaiveDateTime, Timelike, Utc};
use grib_core::grib2::{Grib2File, Grib2Message, flip_rows, grid_latlon, unpack_message};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct ObservationSpec {
    pub name: String,
    pub label: String,
    pub source_label: String,
    pub bucket: String,
    pub region: String,
    pub file_prefix: String,
    pub parameter_table: String,
    pub products: Vec<ObservationProduct>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ObservationProduct {
    pub product: String,
    pub message: String,
    pub candidates: Vec<String>,
    pub parameters: Vec<[u8; 3]>,
    pub center: u16,
    pub subcenter: u16,
    pub level_type: u8,
    pub level_value: f64,
    pub units: String,
    pub message_template: u16,
    pub forecast_time: u32,
    pub valid_time_basis: ObservationTimeBasis,
    pub missing_values: Vec<f64>,
    pub time: TimePolicy,
    pub accumulation_seconds: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationTimeBasis {
    ReferenceTime,
    IntervalEnd,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TimePolicy {
    Nearest { tolerance_seconds: i64 },
    ExactHourEnd,
}

impl TimePolicy {
    fn tolerance(&self) -> i64 {
        match self {
            Self::Nearest { tolerance_seconds } => *tolerance_seconds,
            Self::ExactHourEnd => 0,
        }
    }

    fn accepts(&self, valid: DateTime<Utc>, target: DateTime<Utc>) -> bool {
        match self {
            Self::Nearest { tolerance_seconds } => {
                (valid.timestamp() - target.timestamp()).abs() <= *tolerance_seconds
            }
            Self::ExactHourEnd => valid == target && target.minute() == 0 && target.second() == 0,
        }
    }
}

pub fn specifications() -> Result<Vec<ObservationSpec>, String> {
    let rows: Vec<ObservationSpec> =
        serde_json::from_str(include_str!("../data/comparison-observations.json"))
            .map_err(|error| format!("comparison observation table: {error}"))?;
    for row in &rows {
        for product in &row.products {
            if product.candidates.is_empty()
                || product.candidates.len() != product.parameters.len()
                || !(0..=3600).contains(&product.time.tolerance())
                || matches!(product.time, TimePolicy::ExactHourEnd)
                    && product.accumulation_seconds != Some(3600)
            {
                return Err(format!(
                    "invalid observation row {} / {}",
                    row.name, product.product
                ));
            }
        }
    }
    Ok(rows)
}

pub struct ObservationField {
    pub values: Vec<f32>,
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    pub ny: usize,
    pub nx: usize,
    pub valid: DateTime<Utc>,
    pub file_name: String,
    pub origin: String,
    pub candidate: String,
    pub sha256: String,
    pub missing_cells: usize,
}

/// The filename narrows acquisition; the decoded message remains authoritative.
fn filename_time(name: &str, prefix: &str, product: &str) -> Option<DateTime<Utc>> {
    let name = name.strip_prefix(prefix)?;
    let stamp = name.strip_prefix(product)?.strip_prefix('_')?;
    let stamp = stamp
        .strip_suffix(".grib2.gz")
        .or_else(|| stamp.strip_suffix(".grib2"))?;
    NaiveDateTime::parse_from_str(stamp, "%Y%m%d-%H%M%S")
        .ok()
        .map(|time| time.and_utc())
}

/// Search only explicit reference and cache roots, without following directory
/// links. A bounded traversal also accepts the archive's region/product/day tree.
fn local_files(root: &Path, depth: usize, files: &mut Vec<PathBuf>) -> Result<(), String> {
    if !root.is_dir() || depth == 0 {
        return Ok(());
    }
    for entry in
        std::fs::read_dir(root).map_err(|error| format!("read {}: {error}", root.display()))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        if kind.is_file() {
            files.push(entry.path());
        } else if kind.is_dir() {
            local_files(&entry.path(), depth - 1, files)?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Identity {
    parameter: [u8; 3],
    center: u16,
    subcenter: u16,
    level_type: u8,
    level_value: f64,
    template: u16,
    lead: u32,
    valid: DateTime<Utc>,
}

impl Identity {
    fn of(message: &Grib2Message, product: &ObservationProduct) -> Option<Self> {
        let valid = match product.valid_time_basis {
            ObservationTimeBasis::ReferenceTime => message.reference_time,
            ObservationTimeBasis::IntervalEnd => message.product.end_of_interval?,
        }.and_utc();
        Some(Self {
            parameter: [
                message.discipline,
                message.product.parameter_category,
                message.product.parameter_number,
            ],
            center: message.identification.center_id,
            subcenter: message.identification.subcenter_id,
            level_type: message.product.level_type,
            level_value: message.product.level_value,
            template: message.product.template,
            lead: message.product.forecast_time,
            valid,
        })
    }

    fn accepts(
        &self,
        product: &ObservationProduct,
        parameter: [u8; 3],
        target: DateTime<Utc>,
    ) -> bool {
        self.parameter == parameter
            && self.center == product.center
            && self.subcenter == product.subcenter
            && self.level_type == product.level_type
            && (self.level_value - product.level_value).abs() < 1.0e-6
            // These rows define analyses and hour-ending estimates in PDT 0.
            // Forecast time is zero; reference_time is the actual observation
            // time, including its seconds, or the accumulation's hour end.
            && self.template == product.message_template
            && self.lead == product.forecast_time
            && product.time.accepts(self.valid, target)
    }
}

fn mask_value(value: f64, product: &ObservationProduct) -> f32 {
    if !value.is_finite() || product.missing_values.contains(&value) {
        f32::NAN
    } else {
        value as f32
    }
}

fn decode(
    raw: &[u8],
    origin: &str,
    name: &str,
    candidate: &str,
    parameter: [u8; 3],
    product: &ObservationProduct,
    target: DateTime<Utc>,
) -> Result<ObservationField, String> {
    let (bytes, _) =
        rw_nexrad::pack::gunzip_if_wrapped(raw).map_err(|error| format!("{origin}: {error}"))?;
    let grib = Grib2File::from_bytes(&bytes).map_err(|error| format!("{origin}: {error}"))?;
    let messages: Vec<_> = grib
        .messages
        .iter()
        .filter(|message| Identity::of(message, product).is_some_and(|identity| identity.accepts(product, parameter, target)))
        .collect();
    if messages.len() != 1 {
        return Err(format!(
            "{origin}: expected one {} message with parameter {parameter:?}, center {}, level {} / {}, PDT {}, lead {} and the required observation time; found {}",
            product.product,
            product.center,
            product.level_type,
            product.level_value, product.message_template, product.forecast_time,
            messages.len()
        ));
    }
    let message = messages[0];
    let (nx, ny) = (message.grid.nx as usize, message.grid.ny as usize);
    if nx == 0 || ny == 0 || message.grid.is_reduced || message.grid.scan_mode & 0x20 != 0 {
        return Err(format!(
            "{origin}: observation grid requires a rectangular row-major scan"
        ));
    }
    let mut values = unpack_message(message).map_err(|error| format!("{origin}: {error}"))?;
    let (mut lat, mut lon) =
        grid_latlon(&message.grid).map_err(|error| format!("{origin}: {error}"))?;
    let cells = nx.checked_mul(ny).ok_or("observation grid size overflow")?;
    if values.len() != cells || lat.len() != cells || lon.len() != cells {
        return Err(format!(
            "{origin}: observation values and coordinates disagree with {ny}x{nx}"
        ));
    }
    if message.grid.scan_mode & 0x10 != 0 {
        for row in (1..ny).step_by(2) {
            values[row * nx..(row + 1) * nx].reverse();
        }
    }
    if message.grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut values, nx, ny);
        flip_rows(&mut lat, nx, ny);
        flip_rows(&mut lon, nx, ny);
    }
    for value in &mut lon {
        *value = (*value + 180.0).rem_euclid(360.0) - 180.0;
    }
    // Keep coordinates and values together when a geographic row wraps.
    for row in 0..ny {
        let range = row * nx..(row + 1) * nx;
        if let Some(index) = lon[range.clone()]
            .windows(2)
            .position(|pair| pair[1] - pair[0] < -180.0)
        {
            lat[range.clone()].rotate_left(index + 1);
            lon[range.clone()].rotate_left(index + 1);
            values[range].rotate_left(index + 1);
        }
    }
    let values: Vec<_> = values
        .into_iter()
        .map(|value| mask_value(value, product))
        .collect();
    let missing_cells = values.iter().filter(|value| !value.is_finite()).count();
    Ok(ObservationField {
        values,
        lat: lat.into_iter().map(|value| value as f32).collect(),
        lon: lon.into_iter().map(|value| value as f32).collect(),
        nx,
        ny,
        valid: Identity::of(message, product).expect("selected identity").valid,
        file_name: name.to_string(),
        origin: origin.to_string(),
        candidate: candidate.to_string(),
        sha256: rw_nexrad::s3::hex_sha256(raw),
        missing_cells,
    })
}

pub fn load(
    spec: &ObservationSpec,
    product_name: &str,
    target: DateTime<Utc>,
    directory: Option<&Path>,
    cache: &Path,
    explicit_file: Option<&Path>,
    offline: bool,
) -> Result<ObservationField, String> {
    let product = spec
        .products
        .iter()
        .find(|product| product.product == product_name)
        .ok_or_else(|| {
            format!(
                "{} has no {product_name} product in the reference table",
                spec.label
            )
        })?;
    let mut local = Vec::new();
    if let Some(directory) = directory {
        local_files(directory, 6, &mut local)?;
    }
    local_files(cache, 6, &mut local)?;
    let mut refusals = Vec::new();
    for (candidate, parameter) in product.candidates.iter().zip(&product.parameters) {
        let mut files: Vec<_> = local
            .iter()
            .filter_map(|path| {
                let valid =
                    filename_time(path.file_name()?.to_str()?, &spec.file_prefix, candidate)?;
                product
                    .time
                    .accepts(valid, target)
                    .then(|| (valid, path.clone()))
            })
            .collect();
        files.sort_by_key(|(valid, path)| {
            (
                (valid.timestamp() - target.timestamp()).abs(),
                *valid > target,
                path.clone(),
            )
        });
        if let Some(file) = explicit_file {
            files = vec![(target, file.to_path_buf())];
        }
        for (_, path) in files {
            let raw = std::fs::read(&path)
                .map_err(|error| format!("read {}: {error}", path.display()))?;
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            match decode(
                &raw,
                &path.display().to_string(),
                &name,
                candidate,
                *parameter,
                product,
                target,
            ) {
                Ok(field) => return Ok(field),
                Err(reason) => refusals.push(reason),
            }
        }
        if offline || explicit_file.is_some() {
            continue;
        }
        let agent = rw_nexrad::s3::build_agent();
        let tolerance = Duration::seconds(product.time.tolerance());
        let start = (target - tolerance).date_naive();
        let end = (target + tolerance).date_naive();
        let mut day = start;
        let mut objects = Vec::new();
        while day <= end {
            let prefix = format!("{}/{candidate}/{}/", spec.region, day.format("%Y%m%d"));
            match rw_nexrad::s3::list_s3(
                &agent,
                rw_nexrad::s3::ListRequest::new(&spec.bucket, &prefix),
            ) {
                Ok(listing) => objects.extend(listing.objects.into_iter().filter_map(|object| {
                    let valid = filename_time(
                        object.key.rsplit('/').next()?,
                        &spec.file_prefix,
                        candidate,
                    )?;
                    product
                        .time
                        .accepts(valid, target)
                        .then_some((valid, object))
                })),
                Err(error) => refusals.push(format!("list {} / {prefix}: {error}", spec.bucket)),
            }
            let Some(next) = day.succ_opt() else {
                break;
            };
            day = next;
        }
        objects.sort_by_key(|(valid, object)| {
            (
                (valid.timestamp() - target.timestamp()).abs(),
                *valid > target,
                object.key.clone(),
            )
        });
        for (_, object) in objects {
            match rw_nexrad::s3::download_object(&agent, &spec.bucket, cache, &object, true) {
                Ok(download) => {
                    let name = object.key.rsplit('/').next().unwrap_or(&object.key);
                    let origin = rw_nexrad::s3::object_url(&spec.bucket, &object.key);
                    match decode(
                        &download.bytes,
                        &origin,
                        name,
                        candidate,
                        *parameter,
                        product,
                        target,
                    ) {
                        Ok(field) => return Ok(field),
                        Err(reason) => refusals.push(reason),
                    }
                }
                Err(error) => refusals.push(error.to_string()),
            }
        }
    }
    Err(format!(
        "{} {}: no accepted observation for {} (tolerance {} s, candidate preference {}{}): {}",
        spec.label,
        product_name,
        target.to_rfc3339(),
        product.time.tolerance(),
        product.candidates.join(", "),
        if offline { ", offline" } else { "" },
        if refusals.is_empty() {
            "no candidate file".to_string()
        } else {
            refusals.join("; ")
        }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> DateTime<Utc> {
        "2026-10-03T13:00:00Z".parse().unwrap()
    }

    #[test]
    fn table_pins_candidate_preference_and_official_parameters() {
        let rows = specifications().unwrap();
        let product = &rows[0].products[1];
        assert_eq!(product.parameters, vec![[209, 6, 37], [209, 6, 30]]);
        assert!(product.candidates[0].contains("Pass2"));
        assert!(product.candidates[1].contains("Pass1"));
        assert_eq!(product.accumulation_seconds, Some(3600));
    }

    #[test]
    fn observation_time_is_bounded_and_hour_end_is_exact() {
        let rows = specifications().unwrap();
        let composite = &rows[0].products[0];
        assert!(
            composite
                .time
                .accepts(target() - Duration::seconds(120), target())
        );
        assert!(
            composite
                .time
                .accepts(target() + Duration::seconds(120), target())
        );
        assert!(
            !composite
                .time
                .accepts(target() + Duration::seconds(121), target())
        );
        let qpe = &rows[0].products[1];
        assert!(qpe.time.accepts(target(), target()));
        assert!(!qpe.time.accepts(target() - Duration::seconds(1), target()));
        assert!(!qpe.time.accepts(
            target() + Duration::minutes(1),
            target() + Duration::minutes(1)
        ));
    }

    #[test]
    fn identity_refuses_other_parameters_levels_centers_and_forecasts() {
        let rows = specifications().unwrap();
        let product = &rows[0].products[1];
        let good = Identity {
            parameter: [209, 6, 37],
            center: 161,
            subcenter: 0,
            level_type: 102,
            level_value: 0.0,
            template: 0,
            lead: 0,
            valid: target(),
        };
        assert!(good.accepts(product, [209, 6, 37], target()));
        for bad in [
            Identity {
                parameter: [209, 6, 30],
                ..good.clone()
            },
            Identity {
                center: 7,
                ..good.clone()
            },
            Identity {
                subcenter: 1,
                ..good.clone()
            },
            Identity {
                level_type: 1,
                ..good.clone()
            },
            Identity {
                level_value: 500.0,
                ..good.clone()
            },
            Identity {
                template: 8,
                ..good.clone()
            },
            Identity {
                lead: 1,
                ..good.clone()
            },
            Identity {
                valid: target() - Duration::hours(1),
                ..good.clone()
            },
        ] {
            assert!(!bad.accepts(product, [209, 6, 37], target()));
        }
    }

    #[test]
    fn dry_cells_are_observations_and_flags_stay_missing() {
        let rows = specifications().unwrap();
        let qpe = &rows[0].products[1];
        assert_eq!(mask_value(0.0, qpe), 0.0);
        assert_eq!(mask_value(12.5, qpe), 12.5);
        for missing in [-1.0, -3.0, f64::NAN] {
            assert!(mask_value(missing, qpe).is_nan());
        }
        let refc = &rows[0].products[0];
        assert!(mask_value(-99.0, refc).is_nan());
        assert!(mask_value(-999.0, refc).is_nan());
        assert_eq!(mask_value(-10.0, refc), -10.0);
    }

    #[test]
    fn published_names_keep_seconds_and_match_the_whole_candidate() {
        let candidate = "MergedReflectivityQCComposite_00.50";
        assert_eq!(
            filename_time(
                "MRMS_MergedReflectivityQCComposite_00.50_20261003-130037.grib2.gz",
                "MRMS_",
                candidate
            ),
            Some(target() + Duration::seconds(37))
        );
        assert!(
            filename_time(
                "MRMS_MergedReflectivityQCComposite_00.25_20261003-130000.grib2.gz",
                "MRMS_",
                candidate
            )
            .is_none()
        );
        assert_eq!(
            filename_time(
                "OBS_MergedReflectivityQCComposite_00.50_20261003-130037.grib2",
                "OBS_",
                candidate
            ),
            Some(target() + Duration::seconds(37))
        );
    }
}
