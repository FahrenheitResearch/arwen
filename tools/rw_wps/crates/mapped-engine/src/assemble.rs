//! Record assembly: matched GRIB records to a canonical decoded collection.
//!
//! Port of `mapped_source._assemble_grib`, `_broadcast_invariant_fields`,
//! `_apply_landmask_water_missing` and `_rotate_grid_relative_winds`.  The
//! order of the four steps is the Python order and is not incidental:
//! invariants broadcast BEFORE the landmask keys off `land_fraction` (an
//! invariant on most sources), and the wind rotation runs LAST so it sees
//! every component the frame will carry.

use std::collections::{BTreeMap, BTreeSet};

use chrono::NaiveDateTime;
use ndarray::ArrayD;

use crate::array;
use crate::grib::{declared_vertical_admits, selector_matches, GribRecord};
use crate::model::{Mapping, ROTATED_WIND_PAIRS};
use crate::refusal::{frame_invalid, selector_unmatched, Result};

/// A (valid_time, member, field) address into the decoded collection.
pub type DirectKey = (NaiveDateTime, Option<String>, String);
/// A (valid_time, member) address.
pub type TimeKey = (NaiveDateTime, Option<String>);

/// `mapped_source._DirectValue`.
#[derive(Debug, Clone)]
pub struct DirectValue {
    pub name: String,
    pub valid_time: NaiveDateTime,
    pub member: Option<String>,
    pub source_cycle: NaiveDateTime,
    pub axes: Vec<String>,
    pub values: ArrayD<f64>,
    pub missing_count: usize,
    pub references: Vec<String>,
}

/// `mapped_source._DecodedCollection`.
#[derive(Debug, Clone)]
pub struct DecodedCollection {
    pub latitude: Vec<f64>,
    pub longitude: Vec<f64>,
    pub vertical_values: Vec<f64>,
    pub direct: BTreeMap<DirectKey, DirectValue>,
    pub source_cycles: BTreeMap<TimeKey, NaiveDateTime>,
    pub grid_fingerprint: String,
}

/// `mapped_source._assemble_grib`.
pub fn assemble_grib(mapping: &Mapping, records: &[GribRecord]) -> Result<DecodedCollection> {
    if records.is_empty() {
        return Err(selector_unmatched("no mapped GRIB records were decoded"));
    }
    let source_format = mapping.format()?.to_owned();
    let declared_levels = mapping.declared_levels()?;
    let field_names = mapping.field_names()?;

    // matched[(valid_time, member, field)] -> record positions, in decode order.
    let mut matched: BTreeMap<DirectKey, Vec<usize>> = BTreeMap::new();
    for name in &field_names {
        let field = mapping.field(name)?;
        if field.derivation().is_some() {
            continue;
        }
        for (position, record) in records.iter().enumerate() {
            let identity = record.identity();
            let hit = field
                .selectors()
                .iter()
                .any(|selector| selector_matches(selector, &identity, &source_format));
            if hit && declared_vertical_admits(&declared_levels, &field, record.level_value)? {
                matched
                    .entry((record.valid_time, record.member.clone(), name.clone()))
                    .or_default()
                    .push(position);
            }
        }
    }
    if matched.is_empty() {
        return Err(selector_unmatched(
            "no GRIB messages match the mapping selectors",
        ));
    }

    let selected: Vec<usize> = matched.values().flatten().copied().collect();
    let fingerprints: BTreeSet<&str> = selected
        .iter()
        .map(|position| records[*position].grid_fingerprint.as_str())
        .collect();
    if fingerprints.len() != 1 {
        return Err(frame_invalid(
            "selected GRIB fields do not share one source grid",
        ));
    }
    let grid_fingerprint = (*fingerprints.iter().next().expect("one fingerprint")).to_owned();
    let latitude = records[selected[0]].latitude.clone();
    let longitude = records[selected[0]].longitude.clone();
    if selected.iter().any(|position| {
        records[*position].latitude != latitude || records[*position].longitude != longitude
    }) {
        return Err(frame_invalid("selected GRIB coordinate axes differ"));
    }

    let members: BTreeSet<Option<String>> = selected
        .iter()
        .map(|position| records[*position].member.clone())
        .collect();
    if members.len() != 1 {
        return Err(frame_invalid(
            "mapped WRF initialization requires exactly one GRIB member",
        ));
    }

    let mut processes: BTreeMap<TimeKey, BTreeSet<(i64, i64)>> = BTreeMap::new();
    for position in &selected {
        let record = &records[*position];
        if let Some(identity) = record.process_identity {
            processes
                .entry((record.valid_time, record.member.clone()))
                .or_default()
                .insert(identity);
        }
    }
    let mixed: Vec<&TimeKey> = processes
        .iter()
        .filter(|(_key, identities)| identities.len() > 1)
        .map(|(key, _)| key)
        .collect();
    if !mixed.is_empty() {
        return Err(frame_invalid(format!(
            "selected GRIB2 fields mix generating-process identities within \
             a valid time: {mixed:?}"
        )));
    }

    let mut unsupported: Vec<(usize, Vec<i64>)> = Vec::new();
    for position in &selected {
        let record = &records[*position];
        let supported = if source_format == "grib1" {
            record.time_semantics.first() == Some(&0) && record.time_semantics.get(2) == Some(&0)
        } else {
            matches!(record.time_semantics.first(), Some(0) | Some(1))
        };
        if !supported {
            unsupported.push((record.index, record.time_semantics.clone()));
        }
    }
    if !unsupported.is_empty() {
        unsupported.truncate(8);
        return Err(frame_invalid(format!(
            "selected GRIB fields use interval/derived time semantics that \
             rw-wps.mapping.v1 cannot bind: {unsupported:?}"
        )));
    }

    let vertical_values = if !declared_levels.is_empty() {
        declared_levels.clone()
    } else {
        let mut level_sets: BTreeSet<Vec<u64>> = BTreeSet::new();
        let mut chosen: Option<Vec<f64>> = None;
        for ((_time, _member, name), group) in &matched {
            if !mapping
                .field(name)?
                .target_axes()?
                .iter()
                .any(|axis| axis == "vertical")
            {
                continue;
            }
            let mut levels: Vec<f64> = group
                .iter()
                .map(|position| records[*position].level_value)
                .collect();
            levels.sort_by(f64::total_cmp);
            levels.dedup();
            level_sets.insert(levels.iter().map(|value| value.to_bits()).collect());
            chosen = Some(levels);
        }
        if level_sets.len() != 1 {
            return Err(frame_invalid(
                "GRIB atmospheric fields do not share one complete vertical inventory",
            ));
        }
        chosen.expect("one level set")
    };

    // ONE TASK PER MAPPED FIELD.  Below this point every field is
    // independent work over disjoint slices of the same immutable
    // records: stack the group, resolve missing cells, convert units,
    // transpose to the target axes, count the NaNs.  The tasks land in
    // pre-assigned slots (`crate::threads`) and are drained in the
    // BTreeMap's own key order below, so the collection, the refusal
    // and the cycle cross-check are the serial engine's.  Nothing new
    // is allocated by doing it this way: each task's stacked array is
    // the array that ends up in the collection either way.
    let entries: Vec<(&DirectKey, &Vec<usize>)> = matched.iter().collect();
    let slots: Vec<Result<(DirectValue, NaiveDateTime)>> = crate::threads::install(|| {
        use rayon::prelude::*;
        entries
            .par_iter()
            .map(|(key, group)| {
                assemble_one_field(mapping, records, &source_format, &vertical_values, key, group)
            })
            .collect()
    });
    let mut direct: BTreeMap<DirectKey, DirectValue> = BTreeMap::new();
    let mut cycles: BTreeMap<TimeKey, NaiveDateTime> = BTreeMap::new();
    for ((key, _group), slot) in entries.iter().zip(slots) {
        let (value, cycle) = slot?;
        let (valid_time, member, _field_name) = *key;
        let cycle_key = (*valid_time, member.clone());
        if let Some(existing) = cycles.get(&cycle_key) {
            if *existing != cycle {
                return Err(frame_invalid(format!(
                    "mapped GRIB fields mix source cycles at {valid_time}"
                )));
            }
        }
        cycles.insert(cycle_key, cycle);
        direct.insert((*key).clone(), value);
    }

    let (direct, cycles) = broadcast_invariant_fields(mapping, direct, cycles)?;
    let direct = apply_landmask_water_missing(mapping, direct)?;
    let declaration = mapping.grid_declaration()?;
    let direct = if declaration.rotates_winds() {
        rotate_grid_relative_winds(
            direct,
            declaration
                .parameters
                .as_ref()
                .expect("a rotating declaration carries parameters"),
        )?
    } else {
        direct
    };

    Ok(DecodedCollection {
        latitude,
        longitude,
        vertical_values,
        direct,
        source_cycles: cycles,
        grid_fingerprint,
    })
}

/// One mapped field, assembled from the records its selectors matched.
///
/// The body of `assemble_grib`'s per-field loop, lifted out so it can
/// run as one task.  It returns the field and the source cycle its
/// records agree on; the caller cross-checks that cycle against the
/// other fields of the same valid time, in order, because that check is
/// about the collection rather than about one field.
fn assemble_one_field(
    mapping: &Mapping,
    records: &[GribRecord],
    source_format: &str,
    vertical_values: &[f64],
    key: &DirectKey,
    group: &[usize],
) -> Result<(DirectValue, NaiveDateTime)> {
    {
        let (valid_time, member, field_name) = key;
        let field = mapping.field(field_name)?;
        let source_axes = field.source_axes()?;
        if source_axes.iter().any(|axis| axis == "time" || axis == "member") {
            return Err(frame_invalid(format!(
                "GRIB embedded time/member metadata must not also appear in \
                 {field_name}.source_axes"
            )));
        }
        let selectors = field.selectors();
        let stacking_axis = if source_axes.iter().any(|axis| axis == "vertical") {
            Some("vertical")
        } else if source_axes.iter().any(|axis| axis == "soil") {
            Some("soil")
        } else {
            None
        };

        let mut values = match stacking_axis {
            None => {
                if group.len() != 1 {
                    return Err(frame_invalid(format!(
                        "duplicate GRIB messages for scalar field {field_name} at {valid_time}"
                    )));
                }
                records[group[0]].values.clone()
            }
            Some("vertical") => {
                let mut by_level: BTreeMap<u64, usize> = BTreeMap::new();
                for position in group {
                    let level = records[*position].level_value;
                    if by_level.insert(level.to_bits(), *position).is_some() {
                        return Err(frame_invalid(format!(
                            "duplicate {field_name} GRIB level {level} at {valid_time}"
                        )));
                    }
                }
                let missing: Vec<f64> = vertical_values
                    .iter()
                    .copied()
                    .filter(|level| !by_level.contains_key(&level.to_bits()))
                    .collect();
                let declared: BTreeSet<u64> =
                    vertical_values.iter().map(|value| value.to_bits()).collect();
                let extra: Vec<f64> = by_level
                    .keys()
                    .filter(|bits| !declared.contains(bits))
                    .map(|bits| f64::from_bits(*bits))
                    .collect();
                if !missing.is_empty() || !extra.is_empty() {
                    let missing = crate::refusal::python_float_list_repr(&missing);
                    let extra = crate::refusal::python_float_list_repr(&extra);
                    return Err(selector_unmatched(format!(
                        "{field_name} vertical coverage mismatch; missing={missing}, \
                         extra={extra}"
                    )));
                }
                let ordered: Vec<ArrayD<f64>> = vertical_values
                    .iter()
                    .map(|level| records[by_level[&level.to_bits()]].values.clone())
                    .collect();
                let axis = source_axes
                    .iter()
                    .position(|item| item == "vertical")
                    .expect("vertical axis proven present");
                array::stack(&ordered, axis, field_name)?
            }
            Some("soil") => {
                if mapping.soil_layer_count()?.is_none() {
                    return Err(frame_invalid(
                        "soil_layer_count is required to stack GRIB soil fields",
                    ));
                }
                if selectors.is_empty() {
                    return Err(frame_invalid(format!(
                        "{field_name} soil stacking requires ordered GRIB selectors"
                    )));
                }
                if group.len() != selectors.len() {
                    return Err(frame_invalid(format!(
                        "{field_name} has {} GRIB soil records; the mapping declares \
                         {} ordered soil selectors",
                        group.len(),
                        selectors.len()
                    )));
                }
                let mut by_selector: BTreeMap<usize, usize> = BTreeMap::new();
                for position in group {
                    let identity = records[*position].identity();
                    let hits: Vec<usize> = selectors
                        .iter()
                        .enumerate()
                        .filter(|(_index, selector)| {
                            selector_matches(selector, &identity, &source_format)
                        })
                        .map(|(index, _)| index)
                        .collect();
                    if hits.len() != 1 {
                        return Err(frame_invalid(format!(
                            "{field_name} GRIB soil record {} matches {} selectors; \
                             expected exactly one",
                            records[*position].index,
                            hits.len()
                        )));
                    }
                    if by_selector.insert(hits[0], *position).is_some() {
                        return Err(frame_invalid(format!(
                            "duplicate {field_name} GRIB soil selector {} at {valid_time}",
                            hits[0]
                        )));
                    }
                }
                let absent: Vec<usize> = (0..selectors.len())
                    .filter(|index| !by_selector.contains_key(index))
                    .collect();
                if !absent.is_empty() {
                    return Err(selector_unmatched(format!(
                        "{field_name} is missing GRIB soil selectors {absent:?}"
                    )));
                }
                let ordered: Vec<ArrayD<f64>> = (0..selectors.len())
                    .map(|index| records[by_selector[&index]].values.clone())
                    .collect();
                let axis = source_axes
                    .iter()
                    .position(|item| item == "soil")
                    .expect("soil axis proven present");
                array::stack(&ordered, axis, field_name)?
            }
            Some(other) => unreachable!("unexpected stacking axis {other}"),
        };

        // The missing cells are answered in ONE pass over the field.
        // The mask this replaces was a `Vec<bool>` as long as the field
        // -- 63 million bytes for one 3-km pressure-level variable --
        // materialized only to be walked once and dropped.  A cell that
        // is replaced is finite afterwards, so a single in-place pass
        // decides and rewrites exactly the cells the mask selected.
        let policy = field.missing_kind()?;
        if policy == "reject" {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(frame_invalid(format!(
                    "{field_name} contains missing/non-finite GRIB data"
                )));
            }
        } else {
            let replacement = if policy == "value" {
                field.missing_value()?
            } else {
                f64::NAN
            };
            for value in values.iter_mut() {
                if !value.is_finite() {
                    *value = replacement;
                }
            }
        }
        let values = array::unit_transform(
            values,
            field.unit_scale(),
            field.unit_offset(),
            field_name,
        )?;
        let target_axes = field.target_axes()?;
        let values = array::transpose_to_target(values, &source_axes, &target_axes, field_name)?;

        let mut ordered_group = group.to_vec();
        ordered_group.sort_by_key(|position| records[*position].index);
        let references: Vec<String> = ordered_group
            .iter()
            .map(|position| format!("{}:{}", records[*position].source, records[*position].index))
            .collect();
        let group_cycles: BTreeSet<NaiveDateTime> = group
            .iter()
            .map(|position| records[*position].reference_time)
            .collect();
        if group_cycles.len() != 1 {
            return Err(frame_invalid(format!(
                "{field_name} GRIB records mix source cycles at {valid_time}"
            )));
        }
        let cycle = *group_cycles.iter().next().expect("one cycle");
        let missing_count = array::count_nan(&values);
        Ok((
            DirectValue {
                name: field_name.clone(),
                valid_time: *valid_time,
                member: member.clone(),
                source_cycle: cycle,
                axes: target_axes,
                values,
                missing_count,
                references,
            },
            cycle,
        ))
    }
}

/// `mapped_source._broadcast_invariant_fields`.
pub fn broadcast_invariant_fields(
    mapping: &Mapping,
    direct: BTreeMap<DirectKey, DirectValue>,
    cycles: BTreeMap<TimeKey, NaiveDateTime>,
) -> Result<(BTreeMap<DirectKey, DirectValue>, BTreeMap<TimeKey, NaiveDateTime>)> {
    let mut invariant: BTreeSet<String> = BTreeSet::new();
    for field in mapping.fields()? {
        if field.time_binding() == Some("cycle_invariant") {
            invariant.insert(field.name.clone());
        }
    }
    if invariant.is_empty() {
        return Ok((direct, cycles));
    }
    let dependent_keys: BTreeSet<TimeKey> = direct
        .keys()
        .filter(|(_time, _member, name)| !invariant.contains(name))
        .map(|(time, member, _name)| (*time, member.clone()))
        .collect();
    if dependent_keys.is_empty() {
        return Err(frame_invalid(
            "every decoded mapped field is declared time-invariant; there is \
             no time-dependent state to define the forcing axis",
        ));
    }
    let dependent_cycles: BTreeSet<NaiveDateTime> = dependent_keys
        .iter()
        .filter_map(|key| cycles.get(key).copied())
        .collect();
    if dependent_cycles.len() > 1 {
        return Err(frame_invalid(format!(
            "cycle-invariant fields cannot broadcast across mixed source \
             cycles {dependent_cycles:?}; one broadcast belongs to one cycle"
        )));
    }
    // The time-dependent fields are MOVED into the result, not copied
    // into it.  Rebuilding this map by cloning every value duplicated
    // the whole decoded frameset -- 8 GB on a 3-km CONUS source -- to
    // produce a map whose only difference is which keys it holds.  The
    // invariant entries are lifted out instead, in the same BTreeMap key
    // order the filter walked, and the broadcast below reads them from
    // there.
    let lifted: Vec<DirectKey> = direct
        .keys()
        .filter(|(_time, _member, name)| invariant.contains(name))
        .cloned()
        .collect();
    let mut result = direct;
    let mut invariant_values: Vec<(DirectKey, DirectValue)> = Vec::with_capacity(lifted.len());
    for key in lifted {
        if let Some(value) = result.remove(&key) {
            invariant_values.push((key, value));
        }
    }
    let kept_cycles: BTreeMap<TimeKey, NaiveDateTime> = cycles
        .iter()
        .filter(|(key, _)| dependent_keys.contains(key))
        .map(|(key, value)| (key.clone(), *value))
        .collect();
    for name in &invariant {
        let instances: Vec<&DirectValue> = invariant_values
            .iter()
            .filter(|((_time, _member, field), _value)| field == name)
            .map(|(_key, value)| value)
            .collect();
        let Some(reference) = instances.first() else {
            continue;
        };
        for instance in instances.iter().skip(1) {
            if instance.values != reference.values {
                return Err(frame_invalid(format!(
                    "cycle-invariant field {name} changes across its supplied \
                     valid times; the declaration promises one array per source cycle"
                )));
            }
        }
        for (valid_time, member) in &dependent_keys {
            result.insert(
                (*valid_time, member.clone(), name.clone()),
                DirectValue {
                    name: name.clone(),
                    valid_time: *valid_time,
                    member: member.clone(),
                    source_cycle: kept_cycles[&(*valid_time, member.clone())],
                    axes: reference.axes.clone(),
                    values: reference.values.clone(),
                    missing_count: reference.missing_count,
                    references: reference.references.clone(),
                },
            );
        }
    }
    Ok((result, kept_cycles))
}

/// `mapped_source._apply_landmask_water_missing`.
pub fn apply_landmask_water_missing(
    mapping: &Mapping,
    direct: BTreeMap<DirectKey, DirectValue>,
) -> Result<BTreeMap<DirectKey, DirectValue>> {
    let mut masked_names: Vec<String> = Vec::new();
    for field in mapping.fields()? {
        if field.missing_kind()? == "landmask_water" {
            masked_names.push(field.name.clone());
        }
    }
    if masked_names.is_empty() {
        return Ok(direct);
    }
    masked_names.sort();
    // The replacements are computed first and applied to the map this
    // was HANDED, rather than to a full copy of it: only the masked
    // fields change, and cloning the whole decoded frameset to overwrite
    // a handful of its entries duplicated every array in it.
    let mut replacements: Vec<(DirectKey, DirectValue)> = Vec::new();
    let time_keys: BTreeSet<TimeKey> = direct
        .keys()
        .map(|(time, member, _name)| (*time, member.clone()))
        .collect();
    for (valid_time, member) in time_keys {
        let present: Vec<&String> = masked_names
            .iter()
            .filter(|name| {
                direct.contains_key(&(valid_time, member.clone(), (*name).clone()))
            })
            .collect();
        if present.is_empty() {
            continue;
        }
        let land = direct
            .get(&(valid_time, member.clone(), "land_fraction".to_owned()))
            .ok_or_else(|| {
                frame_invalid(format!(
                    "landmask_water fields {present:?} at {valid_time} have no \
                     decoded land_fraction to key water cells from"
                ))
            })?;
        if land.values.ndim() != 2 || land.values.iter().any(|value| !value.is_finite()) {
            return Err(frame_invalid(
                "land_fraction must be a finite 2-D field to serve as the \
                 landmask_water mask",
            ));
        }
        let water: Vec<bool> = array::contiguous(&land.values)
            .iter()
            .map(|value| *value < 0.5)
            .collect();
        for name in present {
            let key = (valid_time, member.clone(), name.clone());
            let value = &direct[&key];
            let mut masked = value.values.as_standard_layout().to_owned();
            let cells = water.len();
            let total = masked.len();
            for (position, cell) in masked.iter_mut().enumerate() {
                if water[position % cells] {
                    *cell = f64::NAN;
                }
            }
            debug_assert_eq!(total % cells, 0, "a masked field is whole horizontal planes");
            let missing_count = array::count_nan(&masked);
            replacements.push((
                key,
                DirectValue {
                    values: masked,
                    missing_count,
                    ..value.clone()
                },
            ));
        }
    }
    let mut result = direct;
    for (key, value) in replacements {
        result.insert(key, value);
    }
    Ok(result)
}

/// `mapped_source._rotate_grid_relative_winds`.
pub fn rotate_grid_relative_winds(
    direct: BTreeMap<DirectKey, DirectValue>,
    parameters: &crate::model::LambertParameters,
) -> Result<BTreeMap<DirectKey, DirectValue>> {
    let (sina, cosa) = crate::lambert::declared_grid_rotation(parameters);
    // Same reasoning as the landmask: the rotated components are
    // computed first and written into the map this was handed.  Only
    // the wind pairs change, and copying every other field to rewrite
    // two of them duplicated the whole decoded frameset.
    let mut rotated: Vec<(DirectKey, DirectValue)> = Vec::new();
    let time_keys: BTreeSet<TimeKey> = direct
        .keys()
        .map(|(time, member, _name)| (*time, member.clone()))
        .collect();
    for (valid_time, member) in time_keys {
        for (u_name, v_name) in ROTATED_WIND_PAIRS {
            let u_key = (valid_time, member.clone(), u_name.to_owned());
            let v_key = (valid_time, member.clone(), v_name.to_owned());
            let has_u = direct.contains_key(&u_key);
            let has_v = direct.contains_key(&v_key);
            if !has_u && !has_v {
                continue;
            }
            if !has_u || !has_v {
                let absent = if has_u { v_name } else { u_name };
                return Err(frame_invalid(format!(
                    "grid-relative wind rotation at {valid_time} needs both \
                     components of ({u_name}, {v_name}); {absent} is not mapped"
                )));
            }
            let u = &direct[&u_key];
            let v = &direct[&v_key];
            if u.axes != v.axes || u.values.shape() != v.values.shape() {
                return Err(frame_invalid(format!(
                    "({u_name}, {v_name}) at {valid_time} disagree in axes/shape; \
                     rotation needs one shared grid"
                )));
            }
            let rank = u.axes.len();
            if rank < 2 || u.axes[rank - 2] != "y" || u.axes[rank - 1] != "x" {
                return Err(frame_invalid(format!(
                    "({u_name}, {v_name}) must end in y/x axes to rotate"
                )));
            }
            let shape = u.values.shape();
            if shape[rank - 2] != parameters.ny as usize
                || shape[rank - 1] != parameters.nx as usize
            {
                return Err(frame_invalid(format!(
                    "({u_name}, {v_name}) at {valid_time} do not share the \
                     declared grid shape"
                )));
            }
            let cells = sina.len();
            let u_values = array::contiguous(&u.values);
            let v_values = array::contiguous(&v.values);
            let mut earth_u = u.values.as_standard_layout().to_owned();
            let mut earth_v = v.values.as_standard_layout().to_owned();
            for (position, cell) in earth_u.iter_mut().enumerate() {
                let plane = position % cells;
                *cell = u_values[position] * cosa[plane] - v_values[position] * sina[plane];
            }
            for (position, cell) in earth_v.iter_mut().enumerate() {
                let plane = position % cells;
                *cell = v_values[position] * cosa[plane] + u_values[position] * sina[plane];
            }
            rotated.push((
                u_key,
                DirectValue {
                    values: earth_u,
                    ..u.clone()
                },
            ));
            rotated.push((
                v_key,
                DirectValue {
                    values: earth_v,
                    ..v.clone()
                },
            ));
        }
    }
    let mut result = direct;
    for (key, value) in rotated {
        result.insert(key, value);
    }
    Ok(result)
}
