//! Canonical frames and the frameset the seam writes.
//!
//! Port of `mapped_source._materialize_frames` and `_frame_header`, plus
//! the `gpuwm-mapped-frameset-v1` writer.  The frameset is the ABI: its
//! schema name is the compiled ABI marker, so anything that changes the
//! shape below changes the marker (design doc §3.4).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use chrono::NaiveDateTime;
use serde_json::{json, Map, Value};

use crate::array;
use crate::assemble::DecodedCollection;
use crate::derive::{evaluate_derivation, CanonicalField};
use crate::model::{Mapping, GRID_FAMILY_LAMBERT, PROJECTED_AXIS_UNIT_M};
use crate::refusal::{frame_invalid, mapping_invalid, Result};

/// `gpuwm.source_frame.SCHEMA`.
pub const SOURCE_FRAME_SCHEMA: &str = "gpuwm-canonical-source-frame-v1";

/// One materialized frame — `mapped_source.MappedSourceFrame`.
pub struct MappedSourceFrame {
    pub valid_time: NaiveDateTime,
    pub member: Option<String>,
    pub source_cycle: NaiveDateTime,
    pub latitude: Vec<f64>,
    pub longitude: Vec<f64>,
    pub vertical_kind: String,
    pub vertical_units: String,
    pub vertical_values: Vec<f64>,
    /// Fields in the mapping's declared order.
    pub fields: Vec<CanonicalField>,
    pub header: Value,
}

/// Python's `datetime.isoformat()` on a naive datetime.
pub fn naive_isoformat(value: NaiveDateTime) -> String {
    value.format("%Y-%m-%dT%H:%M:%S").to_string()
}

/// Python's `datetime.replace(tzinfo=utc).isoformat()`.
fn utc_isoformat(value: NaiveDateTime) -> String {
    format!("{}+00:00", naive_isoformat(value))
}

/// `mapped_source._materialize_frames`.
pub fn materialize_frames(
    mapping: &Mapping,
    collection: &DecodedCollection,
) -> Result<Vec<MappedSourceFrame>> {
    let declared_names = mapping.field_names()?;
    let mut unbound: Vec<String> = Vec::new();
    for field in mapping.fields()? {
        if field.provider() == Some("composition_bound") {
            unbound.push(field.name.clone());
        }
    }
    if !unbound.is_empty() {
        unbound.sort();
        let unbound = crate::refusal::python_list_repr(&unbound);
        return Err(frame_invalid(format!(
            "fields {unbound} are composition_bound: this mapping cannot \
             materialize alone, because those values live in another packaged \
             source's decode; a cross-source composition must bind each of \
             them to a contributing source"
        )));
    }
    let keys: Vec<&(NaiveDateTime, Option<String>)> = collection.source_cycles.keys().collect();
    if keys.is_empty() {
        return Err(frame_invalid("mapped source has no valid times"));
    }
    let members: BTreeSet<&Option<String>> = keys.iter().map(|(_time, member)| member).collect();
    if members.len() != 1 {
        return Err(frame_invalid(
            "mapped WRF initialization requires exactly one member",
        ));
    }
    let required_names: BTreeSet<String> = mapping.required_field_names()?.into_iter().collect();
    let mut frames = Vec::with_capacity(keys.len());
    for (valid_time, member) in &keys {
        let mut available: BTreeMap<String, CanonicalField> = BTreeMap::new();
        for ((time, member_value, field_name), direct) in &collection.direct {
            if time != valid_time || member_value != member {
                continue;
            }
            let field = mapping.field(field_name)?;
            available.insert(
                field_name.clone(),
                CanonicalField {
                    name: field_name.clone(),
                    units: field.units_target()?.to_owned(),
                    axes: direct.axes.clone(),
                    location: field.location()?.to_owned(),
                    staggering: field.staggering().to_owned(),
                    values: direct.values.clone(),
                    missing_count: direct.missing_count,
                    source_references: direct.references.clone(),
                }
                .validated()?,
            );
        }
        let mut pending: BTreeSet<String> = BTreeSet::new();
        for field in mapping.fields()? {
            if field.derivation().is_some() {
                pending.insert(field.name.clone());
            }
        }
        while !pending.is_empty() {
            let mut progress = false;
            for name in pending.clone() {
                let field = mapping.field(&name)?;
                let derivation_name = field.derivation().expect("pending entries are derived");
                let operation = mapping.derivation(derivation_name).ok_or_else(|| {
                    mapping_invalid(format!(
                        "field {name} names unknown derivation '{derivation_name}'"
                    ))
                })?;
                let Some((values, axes, references)) =
                    evaluate_derivation(operation, &available, collection, &field, &name)?
                else {
                    continue;
                };
                let missing_count = array::count_nan(&values);
                available.insert(
                    name.clone(),
                    CanonicalField {
                        name: name.clone(),
                        units: field.units_target()?.to_owned(),
                        axes,
                        location: field.location()?.to_owned(),
                        staggering: field.staggering().to_owned(),
                        values,
                        missing_count,
                        source_references: references,
                    }
                    .validated()?,
                );
                pending.remove(&name);
                progress = true;
            }
            if !progress {
                let stuck: Vec<String> = pending.iter().cloned().collect();
                return Err(frame_invalid(format!(
                    "derived fields have missing dependencies or a cycle: {}",
                    stuck.join(", ")
                )));
            }
        }
        // The frame states its fields in the mapping's own declared order:
        // assembly order is the PRODUCER's record layout, and a broadcast
        // invariant lands after the per-time records, so two frames with
        // identical field SETS could otherwise disagree about sequence.
        let ordered: Vec<CanonicalField> = declared_names
            .iter()
            .filter_map(|name| available.get(name).cloned())
            .collect();
        let present: BTreeSet<&str> = ordered.iter().map(|field| field.name.as_str()).collect();
        let missing: Vec<&String> = required_names
            .iter()
            .filter(|name| !present.contains(name.as_str()))
            .collect();
        if !missing.is_empty() {
            let missing = crate::refusal::python_list_repr(&missing);
            return Err(frame_invalid(format!(
                "mapped frame at {valid_time} lacks required fields {missing}"
            )));
        }
        for field in &ordered {
            let finite_required = required_names.contains(&field.name)
                && field.name != "soil_temperature"
                && field.name != "volumetric_soil_moisture";
            if finite_required && field.values.iter().any(|value| !value.is_finite()) {
                return Err(frame_invalid(format!(
                    "required mapped field {} is not finite at {valid_time}",
                    field.name
                )));
            }
        }
        if let Some(soil_count) = mapping.soil_layer_count()?.filter(|count| *count > 0) {
            for name in ["soil_temperature", "volumetric_soil_moisture"] {
                let field = ordered
                    .iter()
                    .find(|field| field.name == name)
                    .ok_or_else(|| {
                        frame_invalid(format!("mapped frame at {valid_time} lacks {name}"))
                    })?;
                let axis = field
                    .axes
                    .iter()
                    .position(|axis| axis == "soil")
                    .ok_or_else(|| frame_invalid(format!("{name} has no soil axis")))?;
                let observed = field.values.shape()[axis] as i64;
                if observed != soil_count {
                    return Err(frame_invalid(format!(
                        "{name} has {observed} layers, target declares {soil_count}"
                    )));
                }
            }
        }
        let source_cycle = collection.source_cycles[&(*valid_time, (*member).clone())];
        validate_frame_axes(collection, &ordered)?;
        let header = frame_header(
            mapping,
            *valid_time,
            source_cycle,
            collection,
            &ordered,
        )?;
        require_wrf_initial_state(&header)?;
        frames.push(MappedSourceFrame {
            valid_time: *valid_time,
            member: (*member).clone(),
            source_cycle,
            latitude: collection.latitude.clone(),
            longitude: collection.longitude.clone(),
            vertical_kind: mapping
                .vertical()?
                .get("kind")
                .and_then(crate::node::Node::as_str)
                .unwrap_or_default()
                .to_owned(),
            vertical_units: mapping
                .vertical()?
                .get("units")
                .and_then(crate::node::Node::as_str)
                .unwrap_or_default()
                .to_owned(),
            vertical_values: collection.vertical_values.clone(),
            fields: ordered,
            header,
        });
    }

    let times: Vec<NaiveDateTime> = frames.iter().map(|frame| frame.valid_time).collect();
    let mut sorted = times.clone();
    sorted.sort();
    sorted.dedup();
    if sorted != times {
        return Err(frame_invalid(
            "mapped forcing times are not unique and increasing",
        ));
    }
    let target = mapping.target()?;
    if target
        .field("require_lateral_boundaries")
        .and_then(crate::node::Node::as_bool)
        .unwrap_or(false)
    {
        if times.len() < 2 {
            // The owned class, not frame_invalid: the Python engine
            // raises ForcingSeriesRefusal for this exact condition, and
            // the refusal contract maps class -> exception 1:1.
            return Err(crate::refusal::forcing_series(
                "mapped lateral-boundary forcing requires at least two times",
            ));
        }
        let deltas: BTreeSet<i64> = times
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).num_seconds())
            .collect();
        if deltas.len() != 1 || *deltas.iter().next().expect("one delta") <= 0 {
            return Err(frame_invalid(
                "mapped forcing cadence must be positive and uniform",
            ));
        }
        let observed = *deltas.iter().next().expect("one delta");
        let declared = target
            .field("boundary_interval_seconds")
            .and_then(crate::node::Node::as_i64);
        if declared != Some(observed) {
            return Err(frame_invalid(format!(
                "mapped cadence {observed} seconds differs from target contract \
                 {declared:?}"
            )));
        }
    }
    let inventories: BTreeSet<Vec<String>> = frames
        .iter()
        .map(|frame| frame.fields.iter().map(|field| field.name.clone()).collect())
        .collect();
    if inventories.len() != 1 {
        return Err(frame_invalid(
            "mapped field inventory changes between valid times",
        ));
    }
    Ok(frames)
}

/// `MappedSourceFrame.__post_init__`'s coordinate and grid-sharing checks.
fn validate_frame_axes(collection: &DecodedCollection, fields: &[CanonicalField]) -> Result<()> {
    for (name, axis) in [
        ("latitude", &collection.latitude),
        ("longitude", &collection.longitude),
        ("vertical", &collection.vertical_values),
    ] {
        if axis.is_empty() || axis.iter().any(|value| !value.is_finite()) {
            return Err(frame_invalid(format!(
                "mapped {name} coordinate must be finite non-empty 1-D"
            )));
        }
    }
    if collection.latitude.len() < 2 || collection.longitude.len() < 2 {
        return Err(frame_invalid("mapped horizontal grid must be at least 2x2"));
    }
    for (name, axis) in [
        ("latitude", &collection.latitude),
        ("longitude", &collection.longitude),
    ] {
        let differences: Vec<f64> = axis.windows(2).map(|pair| pair[1] - pair[0]).collect();
        let increasing = differences.iter().all(|value| *value > 0.0);
        let decreasing = differences.iter().all(|value| *value < 0.0);
        if !increasing && !decreasing {
            return Err(frame_invalid(format!(
                "mapped {name} coordinate is not strictly monotonic"
            )));
        }
        let first = differences[0];
        // `np.allclose(difference, difference[0], rtol=1e-10, atol=1e-12)`.
        if differences
            .iter()
            .any(|value| (value - first).abs() > 1e-12 + 1e-10 * first.abs())
        {
            return Err(frame_invalid(format!(
                "mapped {name} coordinate is not a regular axis; curvilinear \
                 export is not enabled"
            )));
        }
    }
    for field in fields {
        let shape: BTreeMap<&str, usize> = field
            .axes
            .iter()
            .map(String::as_str)
            .zip(field.values.shape().iter().copied())
            .collect();
        if shape.get("y") != Some(&collection.latitude.len())
            || shape.get("x") != Some(&collection.longitude.len())
        {
            return Err(frame_invalid(format!(
                "{} does not share the source horizontal grid",
                field.name
            )));
        }
        if let Some(levels) = shape.get("vertical") {
            if *levels != collection.vertical_values.len() {
                return Err(frame_invalid(format!(
                    "{} does not share the vertical coordinate",
                    field.name
                )));
            }
        }
    }
    Ok(())
}

/// `mapped_source._frame_header`.
fn frame_header(
    mapping: &Mapping,
    valid_time: NaiveDateTime,
    source_cycle: NaiveDateTime,
    collection: &DecodedCollection,
    fields: &[CanonicalField],
) -> Result<Value> {
    let vertical = mapping.vertical()?;
    let vertical_kind = vertical
        .get("kind")
        .and_then(crate::node::Node::as_str)
        .unwrap_or_default();
    let coordinate = match vertical_kind {
        "hybrid_sigma_pressure" => "hybrid",
        "embedded_levels" => "model_level",
        other => other,
    };
    let mut vertical_coordinates = Map::new();
    vertical_coordinates.insert(
        "atmosphere".to_owned(),
        json!({
            "coordinate": coordinate,
            "level_count": collection.vertical_values.len(),
            "level_values": collection.vertical_values,
            "a_coefficients": Vec::<f64>::new(),
            "b_coefficients": Vec::<f64>::new(),
            "positive": vertical
                .field("positive")
                .and_then(crate::node::Node::as_str)
                .unwrap_or("down"),
            "units": vertical
                .get("units")
                .and_then(crate::node::Node::as_str)
                .unwrap_or_default(),
        }),
    );
    if let Some(soil) = fields.iter().find(|field| field.axes.iter().any(|axis| axis == "soil")) {
        let axis = soil
            .axes
            .iter()
            .position(|axis| axis == "soil")
            .expect("soil axis proven present");
        vertical_coordinates.insert(
            "soil".to_owned(),
            json!({
                "coordinate": "soil_depth",
                "level_count": soil.values.shape()[axis],
                "level_values": Vec::<f64>::new(),
                "a_coefficients": Vec::<f64>::new(),
                "b_coefficients": Vec::<f64>::new(),
                "positive": "down",
                "units": "index",
            }),
        );
    }
    let time = json!({
        "reference_time": utc_isoformat(source_cycle),
        "valid_time": utc_isoformat(valid_time),
        "lead_seconds": (valid_time - source_cycle).num_seconds(),
        "statistic": "instantaneous",
        "interval_start": Value::Null,
        "interval_end": Value::Null,
        "accumulation_reset": Value::Null,
    });
    // One descriptor per field, hashed CONCURRENTLY.  sha256 over a
    // field's bytes is the whole cost here and the fields are
    // independent; the indexed collect puts field i's descriptor at
    // index i whatever order the threads finished in, so the header --
    // and therefore its digest -- is the serial engine's.
    let field_descriptors: Vec<Value> = crate::threads::install(|| {
        use rayon::prelude::*;
        fields.par_iter().map(|field| {
            let flat = array::contiguous(&field.values);
            json!({
                "canonical_name": field.name,
                "units": field.units,
                "dimensions": field.axes,
                "grid_location": field.location,
                "vertical_coordinate": if field.axes.iter().any(|axis| axis == "vertical") {
                    Value::String("atmosphere".to_owned())
                } else if field.axes.iter().any(|axis| axis == "soil") {
                    Value::String("soil".to_owned())
                } else {
                    Value::Null
                },
                "time": time,
                "data_reference": format!(
                    "sha256:{}",
                    crate::digest::array_sha256(field.values.shape(), &flat)
                ),
                "dtype": "<f8",
                "shape": field.values.shape(),
                "missing_value_policy": if field.missing_count > 0 {
                    "explicit_missing"
                } else {
                    "reject_nonfinite"
                },
                "source_field": field.source_references.join(";"),
            })
        }).collect()
    });
    let declaration = mapping.grid_declaration()?;
    let grid = if declaration.family == GRID_FAMILY_LAMBERT {
        let parameters = declaration
            .parameters
            .as_ref()
            .expect("lambert declarations carry parameters");
        json!({
            "projection": GRID_FAMILY_LAMBERT,
            "nx": collection.longitude.len(),
            "ny": collection.latitude.len(),
            "earth_shape": format!("grib_shape_of_earth:{}", parameters.shape_of_earth),
            "scan_order": "+x,+y",
            // Grid-relative sources were rotated to the earth basis at decode
            // time; the frame states what its arrays ARE, not what the
            // producer published.
            "wind_basis": "earth_relative",
            "parameters": {
                "axis_unit_m": PROJECTED_AXIS_UNIT_M,
                "dx_m": parameters.dx_m,
                "dy_m": parameters.dy_m,
                "earth_radius_m": parameters.earth_radius_m,
                "lat1": parameters.lat1,
                "latin1": parameters.latin1,
                "latin2": parameters.latin2,
                "lon1": parameters.lon1,
                "lov": parameters.lov,
                "nx": parameters.nx,
                "ny": parameters.ny,
                "shape_of_earth": parameters.shape_of_earth,
                "source_wind_basis": declaration.wind_basis,
            },
        })
    } else {
        let latitude = &collection.latitude;
        let longitude = &collection.longitude;
        json!({
            "projection": "regular_latitude_longitude",
            "nx": longitude.len(),
            "ny": latitude.len(),
            "earth_shape": "source_metadata_bound",
            "scan_order": format!(
                "{},{}",
                if longitude[longitude.len() - 1] > longitude[0] { "+x" } else { "-x" },
                if latitude[latitude.len() - 1] > latitude[0] { "+y" } else { "-y" },
            ),
            "wind_basis": "earth_relative",
            "parameters": {
                "latitude_first": latitude[0],
                "latitude_last": latitude[latitude.len() - 1],
                "longitude_first": longitude[0],
                "longitude_last": longitude[longitude.len() - 1],
            },
        })
    };
    let policies = mapping
        .target()?
        .field("initialization_policies")
        .map(crate::node::Node::to_value)
        .unwrap_or_else(|| Value::Object(Map::new()));
    Ok(json!({
        "source_id": mapping.name()?,
        "source_cycle": utc_isoformat(source_cycle),
        "grid": grid,
        "vertical_coordinates": Value::Object(vertical_coordinates),
        "fields": field_descriptors,
        "initialization_policies": policies,
        "schema": SOURCE_FRAME_SCHEMA,
    }))
}

/// Values per pass when encoding the frame stream: 1 MiB of f64.
const STREAM_CHUNK: usize = 128 * 1024;

/// `gpuwm.source_frame.REQUIRED_3D_FIELDS`.
const REQUIRED_3D_FIELDS: [&str; 5] = [
    "air_temperature",
    "eastward_wind",
    "geopotential_height",
    "northward_wind",
    "specific_humidity",
];

/// `gpuwm.source_frame.REQUIRED_SURFACE_FIELDS`.
const REQUIRED_SURFACE_FIELDS: [&str; 10] = [
    "air_temperature_2m",
    "eastward_wind_10m",
    "land_fraction",
    "northward_wind_10m",
    "skin_temperature",
    "soil_temperature",
    "specific_humidity_2m",
    "surface_pressure",
    "terrain_height",
    "volumetric_soil_moisture",
];

/// `gpuwm.source_frame.POLICY_CONTROLLED_FIELDS`.
const POLICY_CONTROLLED_FIELDS: [&str; 9] = [
    "cloud_ice_mixing_ratio",
    "cloud_water_mixing_ratio",
    "graupel_or_hail_mixing_ratio",
    "rain_water_mixing_ratio",
    "sea_ice_fraction",
    "snow_depth",
    "snow_mixing_ratio",
    "snow_water_equivalent",
    "vertical_velocity",
];

/// `gpuwm.source_frame.validate_source_frame_header`'s WRF-initial-state
/// block, run on the header this engine just built.
///
/// The Python engine validates the header at construction, so a frame it
/// cannot initialize WRF from is refused THERE — before the next frame is
/// even reached.  Without this the engine ran on and refused a LATER
/// frame for a different reason, which is not the same answer: on a real
/// staged analysis whose f00 carries the 3-D state and whose f06 carries
/// only surface records, Python refused f00 naming the absent surface
/// fields and this engine refused f06 naming the absent 3-D fields.  Both
/// refuse, so nothing wrong was ever produced — but a caller reading the
/// message would be sent to the wrong end of the source.
fn require_wrf_initial_state(header: &Value) -> Result<()> {
    let names: BTreeSet<&str> = header
        .get("fields")
        .and_then(Value::as_array)
        .map(|fields| {
            fields
                .iter()
                .filter_map(|field| field.get("canonical_name").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    let mut missing: BTreeSet<&str> = REQUIRED_3D_FIELDS
        .iter()
        .chain(REQUIRED_SURFACE_FIELDS.iter())
        .copied()
        .filter(|name| !names.contains(name))
        .collect();
    // Pressure may be stated as a field or implied by a hybrid vertical
    // coordinate; one of the two must be there.
    let hybrid = header
        .get("vertical_coordinates")
        .and_then(Value::as_object)
        .map(|coordinates| {
            coordinates.values().any(|vertical| {
                vertical.get("coordinate").and_then(Value::as_str) == Some("hybrid")
            })
        })
        .unwrap_or(false);
    if !names.contains("air_pressure") && !hybrid {
        missing.insert("air_pressure_or_hybrid_coordinate");
    }
    if !missing.is_empty() {
        let missing: Vec<&str> = missing.into_iter().collect();
        return Err(frame_invalid(format!(
            "source frame is incomplete for WRF initialization: {}",
            missing.join(", ")
        )));
    }
    let policies: BTreeSet<&str> = header
        .get("initialization_policies")
        .and_then(Value::as_object)
        .map(|policies| policies.keys().map(String::as_str).collect())
        .unwrap_or_default();
    let missing_policies: Vec<&str> = POLICY_CONTROLLED_FIELDS
        .iter()
        .copied()
        .filter(|name| !names.contains(name) && !policies.contains(name))
        .collect();
    if !missing_policies.is_empty() {
        return Err(frame_invalid(format!(
            "missing explicit initialization policy for absent fields: {}",
            missing_policies.join(", ")
        )));
    }
    Ok(())
}

/// One axis as `mapped_engine_bridge._axis_document`.
///
/// The numbers ride the JSON, and the sha256 of their little-endian
/// `<f8` bytes rides beside them, so a reader can tell a grid that was
/// re-parsed exactly from one that a JSON round trip moved by an ulp.
/// A bare array could not: the values would still look like an axis.
fn axis_document(values: &[f64]) -> Value {
    let mut payload = Vec::with_capacity(values.len() * 8);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    json!({
        "values": values,
        "count": values.len(),
        "sha256": crate::digest::bytes_sha256(&payload),
    })
}

/// Write `frames.json` + `frames.f64` into `directory`.
pub fn write_frameset(
    directory: &std::path::Path,
    mapping: &Mapping,
    collection: &DecodedCollection,
    frames: &[MappedSourceFrame],
    input_sha256: &BTreeMap<String, String>,
) -> Result<Value> {
    std::fs::create_dir_all(directory).map_err(|error| {
        crate::refusal::missing_input(format!(
            "cannot create output directory {}: {error}",
            directory.display()
        ))
    })?;
    let stream_path = directory.join("frames.f64");
    let file = std::fs::File::create(&stream_path).map_err(|error| {
        crate::refusal::missing_input(format!(
            "cannot write {}: {error}",
            stream_path.display()
        ))
    })?;
    let mut stream = std::io::BufWriter::new(file);
    let mut offset: u64 = 0;
    // The whole-stream digest the reader re-computes before it trusts a
    // single array.  Accumulated as the bytes go out rather than by
    // re-reading the file: a real frameset is multi-gigabyte.
    let mut stream_digest = <sha2::Sha256 as sha2::Digest>::new();
    let mut frame_documents = Vec::with_capacity(frames.len());
    for frame in frames {
        // Every field's array digest for this frame, computed
        // CONCURRENTLY and consumed by position below.  The digests are
        // independent of the stream layout -- each covers one field's
        // own dtype, shape and bytes -- so they can be taken before a
        // byte is written, while the WRITE stays strictly sequential
        // because the offsets are cumulative and the whole-stream digest
        // is order-dependent.
        let field_digests: Vec<String> = crate::threads::install(|| {
            use rayon::prelude::*;
            frame
                .fields
                .par_iter()
                .map(|field| {
                    crate::digest::array_sha256(
                        field.values.shape(),
                        &array::contiguous(&field.values),
                    )
                })
                .collect()
        });
        let mut field_documents = Vec::with_capacity(frame.fields.len());
        for (field, field_sha256) in frame.fields.iter().zip(field_digests) {
            let flat = array::contiguous(&field.values);
            let length = (flat.len() * 8) as u64;
            // Encoded, digested into the whole-stream hash, and
            // written through a FIXED buffer, a chunk at a time.  One
            // field of a 0.25-degree analysis is ~700 MB as f64;
            // encoding a whole field into a second Vec first would
            // double the peak of every decode this seam exists to make
            // cheaper -- the same reason the reader streams its hash
            // instead of reading the stream whole.  The field's OWN
            // digest was taken above, off this thread.
            let mut chunk: Vec<u8> = Vec::with_capacity(STREAM_CHUNK * 8);
            for values in flat.chunks(STREAM_CHUNK) {
                chunk.clear();
                for value in values {
                    chunk.extend_from_slice(&value.to_le_bytes());
                }
                sha2::Digest::update(&mut stream_digest, &chunk);
                stream.write_all(&chunk).map_err(|error| {
                    crate::refusal::missing_input(format!(
                        "cannot write the frame stream: {error}"
                    ))
                })?;
            }
            field_documents.push(json!({
                "name": field.name,
                "units": field.units,
                "axes": field.axes,
                "location": field.location,
                "staggering": field.staggering,
                "shape": field.values.shape(),
                "dtype": "<f8",
                "offset": offset,
                "length": length,
                "sha256": field_sha256,
                "missing_count": field.missing_count,
                "source_references": field.source_references,
            }));
            offset += length;
        }
        // Every MappedSourceFrame scalar rides the frame, not the
        // document: the reader rebuilds one dataclass per entry and
        // re-runs its validators, and a frame that had to borrow its
        // mapping digest, its input digests or its grid fingerprint
        // from an enclosing object could be replayed under a different
        // decode's provenance without anything noticing.
        frame_documents.push(json!({
            "valid_time": naive_isoformat(frame.valid_time),
            "member": frame.member,
            "source_cycle": naive_isoformat(frame.source_cycle),
            "latitude": axis_document(&frame.latitude),
            "longitude": axis_document(&frame.longitude),
            "vertical_kind": frame.vertical_kind,
            "vertical_units": frame.vertical_units,
            "vertical_values": axis_document(&frame.vertical_values),
            "grid_fingerprint": collection.grid_fingerprint,
            "mapping_sha256": mapping.sha256,
            "input_sha256": input_sha256,
            "fields": field_documents,
            "header": frame.header,
        }));
    }
    stream.flush().map_err(|error| {
        crate::refusal::missing_input(format!("cannot flush the frame stream: {error}"))
    })?;
    let document = json!({
        "schema": crate::FRAMESET_SCHEMA,
        "engine": {"name": crate::ENGINE_NAME, "version": crate::ENGINE_VERSION},
        // One object, not a bare name beside a loose byte count: the
        // reader verifies path, size and whole-stream digest together
        // before it maps a byte, and two spellings of the same number
        // are how a stream and its manifest drift apart.
        "stream": {
            "path": "frames.f64",
            "dtype": "<f8",
            "bytes": offset,
            "sha256": crate::digest::hex_digest(stream_digest),
        },
        "mapping_sha256": mapping.sha256,
        "mapping_path": mapping.path,
        "input_sha256": input_sha256,
        "grid_fingerprint": collection.grid_fingerprint,
        "frames": frame_documents,
    });
    let manifest_path = directory.join("frames.json");
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&document).unwrap_or_default())
        .map_err(|error| {
            crate::refusal::missing_input(format!(
                "cannot write {}: {error}",
                manifest_path.display()
            ))
        })?;
    Ok(document)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_and_utc_isoformats_match_pythons_two_spellings() {
        let value =
            NaiveDateTime::parse_from_str("2026-08-17 06:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(naive_isoformat(value), "2026-08-17T06:00:00");
        assert_eq!(utc_isoformat(value), "2026-08-17T06:00:00+00:00");
    }
}
