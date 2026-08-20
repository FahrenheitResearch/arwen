//! The composition JOINS: the coordinate-subset solve, terrain
//! composition and the cross-source bound-field borrow.
//!
//! Port of `gpuwm.mapped_composition._exact_subset_indices`,
//! `_compose_terrain`, `_binding_subset_indices`, `_take_subset` and
//! `_compose_bound_fields`.  These four are the numeric heart of
//! `compose`: they decide WHICH CELLS of a donor grid are borrowed and
//! under which clock, so an error here is silently offset data rather
//! than a refusal.
//!
//! Three transcription hazards are called out where they occur, because
//! each one reads as a harmless simplification:
//!
//! * the coordinate comparison is EXACT float equality, never a
//!   tolerance — an epsilon would accept grids the Python engine refuses;
//! * `np.mod(x, 360.0)` is floor-modulo, which is `f64::rem_euclid`, not
//!   `%`;
//! * the terrain path INHERITS the supplement's missing count while the
//!   bound-field path RECOMPUTES it on the subset.  The two are
//!   deliberately different and unifying them moves a number in every
//!   receipt.
//!
//! **What the compose goldens do and do not grade here.**  Measured
//! across every staged composed source: each one's supplement or donor
//! grid EQUALS its primary grid, so the index window solved below spans
//! the whole donor axis on all of them.  The goldens therefore grade
//! this solve in its degenerate full-window form only.  The strict
//! subset -- an interior window, a reversed axis, a donor spelling its
//! longitudes in [-180, 180) against a primary in [0, 360), a
//! non-contiguous match -- is graded by the unit tests at the bottom of
//! this file and by nothing else.  Do not read a green battery as
//! coverage of the window arithmetic; a source with a genuinely larger
//! donor grid would be the first golden that does.

use std::collections::{BTreeMap, BTreeSet};

use chrono::NaiveDateTime;
use ndarray::{ArrayD, Axis};
use serde_json::{json, Map, Value};

use crate::assemble::{DecodedCollection, DirectValue, TimeKey};
use crate::compose::{Binding, EXTERNAL_FIELD};
use crate::frames::naive_isoformat;
use crate::refusal::{frame_invalid, python_float_repr, python_list_repr, Result};

/// `gpuwm-mapped-exact-subset-binding-v1`.
pub const SUBSET_RECEIPT_SCHEMA: &str = "gpuwm-mapped-exact-subset-binding-v1";
/// `mapped_composition.BINDING_RECEIPT_SCHEMA`.
pub const BINDING_RECEIPT_SCHEMA: &str = "gpuwm-cross-source-binding-v1";

/// The two DECLARED shapes a terrain supplement's clock can take
/// (`mapped_composition._TERRAIN_TIME_ALIGNMENTS`).
pub const TERRAIN_TIME_ALIGNMENTS: [&str; 2] = ["valid_time_exact", "cycle_invariant_broadcast"];

/// The DECLARED clocks a cross-source binding can run on
/// (`mapped_composition._BOUND_TIME_ALIGNMENTS`).
pub const BOUND_TIME_ALIGNMENTS: [&str; 3] = [
    "cycle_invariant_broadcast",
    "source_cycle_analysis_broadcast",
    "valid_time_exact",
];

/// `datetime.__str__`: `2026-08-17 06:00:00`, the spelling `str(cycle)`
/// puts in the mixed-cycle refusals.
fn naive_str(value: NaiveDateTime) -> String {
    value.format("%Y-%m-%d %H:%M:%S").to_string()
}

/// `repr(datetime.datetime(...))`, as a refusal that interpolates a list
/// of valid times spells them.
fn naive_repr(value: NaiveDateTime) -> String {
    use chrono::{Datelike, Timelike};

    let mut parts = vec![
        value.year().to_string(),
        value.month().to_string(),
        value.day().to_string(),
        value.hour().to_string(),
        value.minute().to_string(),
    ];
    // Python drops trailing zero second/microsecond components, and every
    // valid time in a mapped source is whole-second.
    if value.second() != 0 {
        parts.push(value.second().to_string());
    }
    format!("datetime.datetime({})", parts.join(", "))
}

/// `repr` of a `(valid_time, member)` key, as `_compose_terrain`'s
/// "lacks exact primary valid time(s)" refusal interpolates it.
fn time_key_repr(key: &TimeKey) -> String {
    let member = match &key.1 {
        Some(value) => crate::refusal::python_repr(value),
        None => "None".to_owned(),
    };
    format!("({}, {member})", naive_repr(key.0))
}

fn time_key_list_repr(keys: &[TimeKey]) -> String {
    let rendered: Vec<String> = keys.iter().map(time_key_repr).collect();
    format!("[{}]", rendered.join(", "))
}

/// `mapped_composition._exact_subset_indices`.
///
/// Every coordinate of `smaller` must have EXACTLY ONE exact match in
/// `larger`, and the matches must be contiguous and monotone.  The
/// comparison is `==` on float64 and nothing else: this is the function
/// that decides which cells of the donor are borrowed, and a tolerance
/// would accept a grid that is merely close, taking the wrong cells and
/// reporting a PASS receipt for them.
pub fn exact_subset_indices(
    larger: &[f64],
    smaller: &[f64],
    label: &str,
    cyclic_degrees: bool,
) -> Result<Vec<usize>> {
    // `np.mod(x, 360.0)`: floor-modulo, whose result takes the divisor's
    // sign.  Rust's `%` is truncating and would leave -0.25 where numpy
    // leaves 359.75, so a donor spelling longitudes in [-180, 180) would
    // match nothing against a primary spelling them in [0, 360).
    let fold = |values: &[f64]| -> Vec<f64> {
        if cyclic_degrees {
            values.iter().map(|value| value.rem_euclid(360.0)).collect()
        } else {
            values.to_vec()
        }
    };
    let larger_comparison = fold(larger);
    let smaller_comparison = fold(smaller);
    let mut indices: Vec<usize> = Vec::with_capacity(smaller_comparison.len());
    for value in &smaller_comparison {
        let matches: Vec<usize> = larger_comparison
            .iter()
            .enumerate()
            .filter(|(_, candidate)| *candidate == value)
            .map(|(index, _)| index)
            .collect();
        if matches.len() != 1 {
            return Err(frame_invalid(format!(
                "primary {label} coordinate np.float64({}) has {} exact matches \
                 in the terrain grid",
                python_float_repr(*value),
                matches.len()
            )));
        }
        indices.push(matches[0]);
    }
    if indices.len() > 1 {
        let ascending = indices.windows(2).all(|pair| pair[1] == pair[0] + 1);
        let descending = indices.windows(2).all(|pair| pair[0] == pair[1] + 1);
        if !(ascending || descending) {
            return Err(frame_invalid(format!(
                "primary {label} is not a contiguous terrain-grid subset"
            )));
        }
    }
    Ok(indices)
}

/// `int(np.sign(last - first))` over the solved index range.
fn index_direction(indices: &[usize]) -> i64 {
    let first = indices[0] as i64;
    let last = indices[indices.len() - 1] as i64;
    (last - first).signum()
}

/// `values[np.ix_(latitude_indices, longitude_indices)]` on a 2-D field.
fn take_grid(values: &ArrayD<f64>, latitude: &[usize], longitude: &[usize]) -> ArrayD<f64> {
    values
        .select(Axis(0), latitude)
        .select(Axis(1), longitude)
}

/// `mapped_composition._take_subset`: take along the axes NAMED `y` and
/// `x`, wherever the field's own axis order puts them.
fn take_subset(
    values: &ArrayD<f64>,
    axes: &[String],
    latitude: &[usize],
    longitude: &[usize],
) -> Result<ArrayD<f64>> {
    let position = |name: &str| -> Result<usize> {
        axes.iter().position(|axis| axis == name).ok_or_else(|| {
            frame_invalid(format!(
                "a borrowed field declares axes {} with no '{name}' axis to \
                 subset along",
                python_list_repr(axes)
            ))
        })
    };
    let result = values.select(Axis(position("y")?), latitude);
    Ok(result.select(Axis(position("x")?), longitude))
}

fn array_digest(values: &ArrayD<f64>) -> String {
    crate::digest::array_sha256(values.shape(), &crate::array::contiguous(values))
}

fn axis_digest(values: &[f64]) -> String {
    crate::digest::array_sha256(&[values.len()], values)
}

/// The `(valid_time, member)` keys of a collection, in Python's order.
fn primary_keys(primary: &DecodedCollection) -> Vec<TimeKey> {
    primary.source_cycles.keys().cloned().collect()
}

/// `mapped_composition._compose_terrain`.
pub fn compose_terrain(
    mut primary: DecodedCollection,
    terrain: &DecodedCollection,
    time_alignment: &str,
) -> Result<(DecodedCollection, Value)> {
    if !TERRAIN_TIME_ALIGNMENTS.contains(&time_alignment) {
        let allowed: Vec<&str> = {
            let mut names = TERRAIN_TIME_ALIGNMENTS.to_vec();
            names.sort_unstable();
            names
        };
        return Err(frame_invalid(format!(
            "terrain supplement time_alignment must be one of {}, got {}",
            python_list_repr(&allowed),
            crate::refusal::python_repr(time_alignment)
        )));
    }
    if primary
        .direct
        .keys()
        .any(|(_time, _member, name)| name == EXTERNAL_FIELD)
    {
        return Err(frame_invalid(
            "terrain has two providers: the primary source and the declared supplement",
        ));
    }
    let inventory: BTreeSet<&str> = terrain
        .direct
        .keys()
        .map(|(_time, _member, name)| name.as_str())
        .collect();
    if inventory.iter().copied().ne([EXTERNAL_FIELD]) {
        let names: Vec<&str> = inventory.into_iter().collect();
        return Err(frame_invalid(format!(
            "terrain supplement decoded unexpected fields {}",
            python_list_repr(&names)
        )));
    }
    let latitude_indices = exact_subset_indices(
        &terrain.latitude,
        &primary.latitude,
        "latitude",
        false,
    )?;
    let longitude_indices = exact_subset_indices(
        &terrain.longitude,
        &primary.longitude,
        "longitude",
        true,
    )?;
    // `sorted(terrain.direct.items(), key=(valid_time, str(member)))`.
    // The map is already keyed `(valid_time, member, field)` and the
    // inventory above proved there is exactly one field name, so the map
    // order IS that sort.
    let terrain_items: Vec<(&NaiveDateTime, &Option<String>, &DirectValue)> = terrain
        .direct
        .iter()
        .map(|((time, member, _field), value)| (time, member, value))
        .collect();
    if terrain_items.is_empty() {
        return Err(frame_invalid("terrain supplement decoded no terrain messages"));
    }
    let full_reference = &terrain_items[0].2.values;
    if terrain_items[1..]
        .iter()
        .any(|(_time, _member, value)| &value.values != full_reference)
    {
        return Err(frame_invalid(
            "terrain supplement changes across supplied valid times",
        ));
    }
    let keys = primary_keys(&primary);
    let terrain_by_time: BTreeMap<TimeKey, &DirectValue> = terrain_items
        .iter()
        .map(|(time, member, value)| ((**time, (*member).clone()), *value))
        .collect();
    let missing_times: Vec<TimeKey> = keys
        .iter()
        .filter(|key| !terrain_by_time.contains_key(key))
        .cloned()
        .collect();
    if !missing_times.is_empty() && time_alignment == "valid_time_exact" {
        return Err(frame_invalid(format!(
            "terrain supplement lacks exact primary valid time(s) {}",
            time_key_list_repr(&missing_times)
        )));
    }
    if time_alignment != "valid_time_exact" {
        // The producer publishes terrain at the analysis step only; the
        // supplement's one proven-invariant record answers every primary
        // valid time of the one source cycle.  One broadcast belongs to
        // one cycle: mixed primary cycles refuse rather than share an
        // invariant record no cycle proved for itself.
        let cycles: BTreeSet<String> = primary.source_cycles.values().copied().map(naive_str).collect();
        if cycles.len() != 1 {
            let names: Vec<String> = cycles.into_iter().collect();
            return Err(frame_invalid(format!(
                "cycle-invariant terrain broadcast cannot span mixed primary \
                 source cycles {}",
                python_list_repr(&names)
            )));
        }
    }
    let carrier = terrain_items[0].2;
    // TAKEN from the primary, not cloned out of it.  The caller hands
    // its collection over and replaces it with the composed one, so the
    // clone made a full second copy of every valid time's arrays live
    // at once: 4.2 GiB per forcing time on a 3 km CONUS source, which is
    // what a seven-time compose was carrying twice.
    let mut direct = std::mem::take(&mut primary.direct);
    let mut subset_reference: Option<ArrayD<f64>> = None;
    for key in &keys {
        let supplied = terrain_by_time.get(key).copied().unwrap_or(carrier);
        let values = take_grid(&supplied.values, &latitude_indices, &longitude_indices);
        match &subset_reference {
            None => subset_reference = Some(values.clone()),
            Some(reference) => {
                if reference != &values {
                    return Err(frame_invalid(
                        "terrain subset changes across primary valid times",
                    ));
                }
            }
        }
        direct.insert(
            (key.0, key.1.clone(), EXTERNAL_FIELD.to_owned()),
            DirectValue {
                name: EXTERNAL_FIELD.to_owned(),
                valid_time: key.0,
                member: key.1.clone(),
                source_cycle: supplied.source_cycle,
                axes: supplied.axes.clone(),
                values,
                // INHERITED, not recomputed.  The bound-field path a few
                // hundred lines down recomputes its own on the subset;
                // the two are deliberately different, and unifying them
                // would move `missing_count` in every terrain receipt
                // ever written.
                missing_count: supplied.missing_count,
                references: supplied.references.clone(),
            },
        );
    }
    // A primary with no valid times would leave the receipt with no
    // subset to hash.  `materialize_frames` refuses that collection a
    // step later; refusing here keeps the composition from writing a
    // receipt describing a binding it never performed.
    let Some(subset_reference) = subset_reference else {
        return Err(frame_invalid(
            "terrain composition has no primary valid time to bind to",
        ));
    };
    let mut receipt = Map::new();
    receipt.insert("schema".into(), json!(SUBSET_RECEIPT_SCHEMA));
    receipt.insert("status".into(), json!("PASS"));
    receipt.insert("field".into(), json!(EXTERNAL_FIELD));
    receipt.insert(
        "primary_shape".into(),
        json!([primary.latitude.len(), primary.longitude.len()]),
    );
    receipt.insert(
        "supplement_shape".into(),
        json!([terrain.latitude.len(), terrain.longitude.len()]),
    );
    receipt.insert(
        "latitude_index_range".into(),
        json!([
            latitude_indices[0],
            latitude_indices[latitude_indices.len() - 1]
        ]),
    );
    receipt.insert(
        "longitude_index_range".into(),
        json!([
            longitude_indices[0],
            longitude_indices[longitude_indices.len() - 1]
        ]),
    );
    receipt.insert("latitude_sha256".into(), json!(axis_digest(&primary.latitude)));
    receipt.insert(
        "longitude_sha256".into(),
        json!(axis_digest(&primary.longitude)),
    );
    receipt.insert(
        "terrain_full_sha256".into(),
        json!(array_digest(full_reference)),
    );
    receipt.insert(
        "terrain_subset_sha256".into(),
        json!(array_digest(&subset_reference)),
    );
    receipt.insert(
        "supplement_valid_times".into(),
        json!(terrain_items
            .iter()
            .map(|(time, _member, _value)| naive_isoformat(**time))
            .collect::<Vec<String>>()),
    );
    receipt.insert(
        "matched_primary_valid_times".into(),
        json!(keys
            .iter()
            .map(|key| naive_isoformat(key.0))
            .collect::<Vec<String>>()),
    );
    if time_alignment != "valid_time_exact" {
        // Recorded ONLY under the broadcast, so every receipt written
        // before these two keys existed is byte-identical to what the
        // same preparation writes today.
        receipt.insert("time_alignment".into(), json!(time_alignment));
        receipt.insert(
            "broadcast_primary_valid_times".into(),
            json!(missing_times
                .iter()
                .map(|key| naive_isoformat(key.0))
                .collect::<Vec<String>>()),
        );
    }
    receipt.insert("invariant_across_all_supplement_times".into(), json!(true));
    receipt.insert(
        "latitude_index_direction".into(),
        json!(index_direction(&latitude_indices)),
    );
    receipt.insert(
        "longitude_index_direction".into(),
        json!(index_direction(&longitude_indices)),
    );
    receipt.insert(
        "coordinate_match".into(),
        json!("exact_equivalent_contiguous_subset"),
    );
    receipt.insert("longitude_equivalence".into(), json!("modulo_360_exact"));
    Ok((
        DecodedCollection {
            latitude: primary.latitude,
            longitude: primary.longitude,
            vertical_values: primary.vertical_values,
            direct,
            source_cycles: primary.source_cycles,
            grid_fingerprint: primary.grid_fingerprint,
            // The composed frame keeps the PRIMARY's vertical identity,
            // its hybrid coefficient ladder included.  TAKEN, not cloned:
            // the current line hands the whole primary over rather than
            // duplicating it, and the ladder follows that same rule.
            hybrid_a: primary.hybrid_a,
            hybrid_b: primary.hybrid_b,
        },
        Value::Object(receipt),
    ))
}

/// `mapped_composition._binding_subset_indices`: the same solve, with the
/// refusal that names the missing regrid capability.
fn binding_subset_indices(
    primary: &DecodedCollection,
    donor: &DecodedCollection,
    binding_name: &str,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let wrap = |error: crate::refusal::Refusal| -> crate::refusal::Refusal {
        frame_invalid(format!(
            "contributing source binding {} is cross-grid ({}); borrowing a \
             field across grids requires a horizontal regrid capability this \
             composition does not declare, so it refuses rather than \
             interpolate",
            crate::refusal::python_repr(binding_name),
            error.message
        ))
    };
    let latitude = exact_subset_indices(&donor.latitude, &primary.latitude, "latitude", false)
        .map_err(wrap)?;
    let longitude =
        exact_subset_indices(&donor.longitude, &primary.longitude, "longitude", true).map_err(wrap)?;
    Ok((latitude, longitude))
}

/// `mapped_composition._compose_bound_fields`.
pub fn compose_bound_fields(
    mut primary: DecodedCollection,
    donor: &DecodedCollection,
    binding: &Binding,
) -> Result<(DecodedCollection, Value)> {
    let binding_name = binding.name.as_str();
    let quoted = crate::refusal::python_repr(binding_name);
    let alignment = binding.time_alignment.as_str();
    if !BOUND_TIME_ALIGNMENTS.contains(&alignment) {
        return Err(frame_invalid(format!(
            "binding time_alignment must be one of {}, got {}",
            python_list_repr(&BOUND_TIME_ALIGNMENTS),
            crate::refusal::python_repr(alignment)
        )));
    }
    let donor_members: BTreeSet<Option<String>> = donor
        .source_cycles
        .keys()
        .map(|(_time, member)| member.clone())
        .collect();
    if donor_members.len() != 1 {
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} decoded {} members; \
             cross-source borrowing has no member-alignment capability and \
             requires a single-member donor",
            donor_members.len()
        )));
    }
    let donor_member = donor_members.into_iter().next().expect("one donor member");
    let donor_inventory: BTreeSet<&str> = donor
        .direct
        .keys()
        .map(|(_time, _member, name)| name.as_str())
        .collect();
    let bound: BTreeSet<&str> = binding.fields.iter().map(String::as_str).collect();
    if donor_inventory != bound {
        let decoded: Vec<&str> = donor_inventory.into_iter().collect();
        let wanted: Vec<&str> = bound.into_iter().collect();
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} decoded fields {} instead of \
             the bound {}",
            python_list_repr(&decoded),
            python_list_repr(&wanted)
        )));
    }
    let provided_twice: BTreeSet<&str> = primary
        .direct
        .keys()
        .map(|(_time, _member, name)| name.as_str())
        .filter(|name| binding.fields.iter().any(|field| field == name))
        .collect();
    if let Some(name) = provided_twice.into_iter().next() {
        return Err(frame_invalid(format!(
            "field {} has two providers: the primary decode and contributing \
             source binding {quoted}",
            crate::refusal::python_repr(name)
        )));
    }
    if donor
        .direct
        .values()
        .any(|value| value.axes.iter().any(|axis| axis == "vertical"))
        && donor.vertical_values != primary.vertical_values
    {
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} borrows a vertical-bearing \
             field on a different vertical ladder; cross-ladder borrowing \
             requires a vertical interpolation capability this composition \
             does not declare"
        )));
    }
    let (latitude_indices, longitude_indices) = binding_subset_indices(&primary, donor, binding_name)?;
    let keys = primary_keys(&primary);
    let primary_cycles: Vec<NaiveDateTime> = primary
        .source_cycles
        .values()
        .copied()
        .collect::<BTreeSet<NaiveDateTime>>()
        .into_iter()
        .collect();
    let mut by_field: BTreeMap<&str, BTreeMap<NaiveDateTime, &DirectValue>> = BTreeMap::new();
    for ((time, _member, name), value) in &donor.direct {
        by_field.entry(name.as_str()).or_default().insert(*time, value);
    }

    // TAKEN from the primary for the same reason the terrain join takes
    // it: the caller replaces its collection with the composed one, so a
    // clone held every valid time's arrays twice.
    let mut direct = std::mem::take(&mut primary.direct);
    let mut matched_times: BTreeSet<NaiveDateTime> = BTreeSet::new();
    let mut broadcast_times: BTreeSet<NaiveDateTime> = BTreeSet::new();
    let mut subset_hashes: BTreeMap<&str, String> = BTreeMap::new();
    for name in &binding.fields {
        let supplied = &by_field[name.as_str()];
        let carrier: Option<&DirectValue> = match alignment {
            "valid_time_exact" => {
                let missing: Vec<NaiveDateTime> = keys
                    .iter()
                    .map(|key| key.0)
                    .filter(|time| !supplied.contains_key(time))
                    .collect();
                if !missing.is_empty() {
                    let rendered: Vec<String> = missing.iter().copied().map(naive_repr).collect();
                    return Err(frame_invalid(format!(
                        "contributing source binding {quoted} lacks {} at \
                         primary valid time(s) [{}]",
                        crate::refusal::python_repr(name),
                        rendered.join(", ")
                    )));
                }
                None
            }
            "source_cycle_analysis_broadcast" => {
                if primary_cycles.len() != 1 {
                    return Err(mixed_cycles(
                        "source-cycle analysis broadcast",
                        &primary_cycles,
                    ));
                }
                if supplied.len() != 1 {
                    return Err(frame_invalid(format!(
                        "contributing source binding {quoted} supplies {} at {} \
                         valid times; the analysis broadcast requires exactly \
                         one analysis record",
                        crate::refusal::python_repr(name),
                        supplied.len()
                    )));
                }
                let (analysis_time, value) = supplied.iter().next().expect("one record");
                if *analysis_time != primary_cycles[0] {
                    return Err(frame_invalid(format!(
                        "contributing source binding {quoted} supplies {} at {}, \
                         which is not the primary source cycle {}; a hybrid \
                         borrows its initialization state from the SAME cycle's \
                         analysis",
                        crate::refusal::python_repr(name),
                        naive_isoformat(*analysis_time),
                        naive_isoformat(primary_cycles[0])
                    )));
                }
                Some(*value)
            }
            _ => {
                if primary_cycles.len() != 1 {
                    return Err(mixed_cycles("cycle-invariant broadcast", &primary_cycles));
                }
                let ordered: Vec<&DirectValue> = supplied.values().copied().collect();
                let reference = ordered[0];
                if ordered[1..]
                    .iter()
                    .any(|value| value.values != reference.values)
                {
                    return Err(frame_invalid(format!(
                        "contributing source binding {quoted} field {} changes \
                         across supplied valid times; the cycle-invariant \
                         broadcast is for proven statics only",
                        crate::refusal::python_repr(name)
                    )));
                }
                Some(reference)
            }
        };
        let mut subset_reference: Option<ArrayD<f64>> = None;
        for key in &keys {
            let value = match supplied.get(&key.0) {
                Some(value) => {
                    matched_times.insert(key.0);
                    *value
                }
                None => {
                    broadcast_times.insert(key.0);
                    // Unreachable through the clock rules above -- the
                    // exact alignment refuses every missing time before
                    // this loop, and both broadcasts set a carrier -- and
                    // written as a refusal rather than a panic anyway: an
                    // engine that aborts leaves no refusal object on
                    // stderr, so the caller gets "exited without a class"
                    // instead of a sentence naming what was missing.
                    let Some(carrier) = carrier else {
                        return Err(frame_invalid(format!(
                            "contributing source binding {quoted} has no \
                             record for {} at {} and no carrier to broadcast; \
                             the {alignment} clock rule left the borrow \
                             unresolved",
                            crate::refusal::python_repr(name),
                            naive_isoformat(key.0)
                        )));
                    };
                    carrier
                }
            };
            let values = take_subset(
                &value.values,
                &value.axes,
                &latitude_indices,
                &longitude_indices,
            )?;
            if subset_reference.is_none() {
                subset_hashes.insert(name.as_str(), array_digest(&values));
                subset_reference = Some(values.clone());
            }
            let missing_count = crate::array::contiguous(&values)
                .iter()
                .filter(|item| item.is_nan())
                .count();
            direct.insert(
                (key.0, key.1.clone(), name.clone()),
                DirectValue {
                    name: name.clone(),
                    valid_time: key.0,
                    member: key.1.clone(),
                    source_cycle: value.source_cycle,
                    axes: value.axes.clone(),
                    values,
                    // RECOMPUTED on the subset, unlike the terrain path's
                    // inherited count: a donor grid's missing cells need
                    // not fall inside the borrowed window.
                    missing_count,
                    references: value.references.clone(),
                },
            );
        }
    }
    let donor_cycles: BTreeSet<NaiveDateTime> = donor
        .direct
        .values()
        .map(|value| value.source_cycle)
        .collect();
    let mut fields = binding.fields.clone();
    fields.sort();
    let receipt = json!({
        "schema": BINDING_RECEIPT_SCHEMA,
        "status": "PASS",
        "binding": binding_name,
        "source_id": binding.source_id,
        "fields": fields,
        "grid_alignment": binding.grid_alignment,
        "coordinate_match": "exact_equivalent_contiguous_subset",
        "longitude_equivalence": "modulo_360_exact",
        "primary_shape": [primary.latitude.len(), primary.longitude.len()],
        "donor_shape": [donor.latitude.len(), donor.longitude.len()],
        "latitude_index_range": [
            latitude_indices[0],
            latitude_indices[latitude_indices.len() - 1],
        ],
        "longitude_index_range": [
            longitude_indices[0],
            longitude_indices[longitude_indices.len() - 1],
        ],
        "time_alignment": alignment,
        "donor_member": donor_member,
        "donor_source_cycles": donor_cycles
            .iter()
            .copied()
            .map(naive_isoformat)
            .collect::<Vec<String>>(),
        "donor_valid_times": donor
            .direct
            .keys()
            .map(|(time, _member, _name)| naive_isoformat(*time))
            .collect::<BTreeSet<String>>(),
        "matched_primary_valid_times": matched_times
            .iter()
            .copied()
            .map(naive_isoformat)
            .collect::<Vec<String>>(),
        "broadcast_primary_valid_times": broadcast_times
            .iter()
            .copied()
            .map(naive_isoformat)
            .collect::<Vec<String>>(),
        "field_subset_sha256": subset_hashes,
    });
    Ok((
        DecodedCollection {
            latitude: primary.latitude,
            longitude: primary.longitude,
            vertical_values: primary.vertical_values,
            direct,
            source_cycles: primary.source_cycles,
            grid_fingerprint: primary.grid_fingerprint,
            // The composed frame keeps the PRIMARY's vertical identity,
            // its hybrid coefficient ladder included.  TAKEN, not cloned:
            // the current line hands the whole primary over rather than
            // duplicating it, and the ladder follows that same rule.
            hybrid_a: primary.hybrid_a,
            hybrid_b: primary.hybrid_b,
        },
        receipt,
    ))
}

fn mixed_cycles(label: &str, cycles: &[NaiveDateTime]) -> crate::refusal::Refusal {
    let rendered: Vec<String> = cycles.iter().copied().map(naive_repr).collect();
    frame_invalid(format!(
        "{label} cannot span mixed primary source cycles [{}]",
        rendered.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_exact_subset_solves_to_its_index_window() {
        let larger: Vec<f64> = (0..10).map(|index| index as f64 * 0.25).collect();
        let smaller = vec![0.5, 0.75, 1.0];
        assert_eq!(
            exact_subset_indices(&larger, &smaller, "latitude", false).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn a_near_miss_is_a_refusal_rather_than_a_tolerance() {
        // The named breakage: 0.7000000000000001 is the f64 next to 0.7,
        // and a tolerance would borrow the 0.7 cell for it and record a
        // PASS receipt saying the grids matched exactly.
        let larger = vec![0.5, 0.7, 0.9];
        let refusal =
            exact_subset_indices(&larger, &[0.700_000_000_000_000_1], "latitude", false).unwrap_err();
        assert!(refusal.message.contains("0 exact matches"), "{refusal}");
    }

    #[test]
    fn the_refusal_spells_the_coordinate_the_way_python_repr_does() {
        let refusal = exact_subset_indices(&[1.0], &[45.0], "latitude", false).unwrap_err();
        assert_eq!(
            refusal.message,
            "primary latitude coordinate np.float64(45.0) has 0 exact matches \
             in the terrain grid"
        );
    }

    #[test]
    fn longitudes_are_compared_modulo_360_by_floor_not_truncation() {
        // A donor spelling longitudes in [-180, 180) and a primary
        // spelling them in [0, 360) name the same meridian.  Rust's `%`
        // would leave -0.25 where numpy's floor-modulo leaves 359.75, so
        // this row would find no match at all.
        let donor = vec![-0.5, -0.25, 0.0];
        let primary = vec![359.75];
        assert_eq!(
            exact_subset_indices(&donor, &primary, "longitude", true).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn a_scattered_subset_refuses_as_non_contiguous() {
        let larger = vec![0.0, 1.0, 2.0, 3.0];
        let refusal = exact_subset_indices(&larger, &[0.0, 2.0], "latitude", false).unwrap_err();
        assert!(refusal.message.contains("not a contiguous"), "{refusal}");
    }

    #[test]
    fn a_reversed_subset_is_contiguous_and_records_its_direction() {
        let larger = vec![0.0, 1.0, 2.0, 3.0];
        let indices = exact_subset_indices(&larger, &[2.0, 1.0], "latitude", false).unwrap();
        assert_eq!(indices, vec![2, 1]);
        assert_eq!(index_direction(&indices), -1);
    }

    #[test]
    fn a_datetime_list_is_spelled_the_way_python_repr_spells_it() {
        let value =
            NaiveDateTime::parse_from_str("2026-08-17 06:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(naive_repr(value), "datetime.datetime(2026, 8, 17, 6, 0)");
        assert_eq!(
            time_key_list_repr(&[(value, None)]),
            "[(datetime.datetime(2026, 8, 17, 6, 0), None)]"
        );
    }
}
