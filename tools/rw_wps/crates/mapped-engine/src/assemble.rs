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
    /// Resolved hybrid A (Pa) / B coefficients for a
    /// hybrid_sigma_pressure vertical: N+1 half-level interfaces or N
    /// full-level values, top of the atmosphere first.  Empty on every
    /// other vertical kind (`_DecodedCollection.hybrid_a/hybrid_b`).
    pub hybrid_a: Vec<f64>,
    pub hybrid_b: Vec<f64>,
}

/// `mapped_source._HYBRID_LITERAL_ABS_TOL` / `_REL_TOL`: pv rides IEEE
/// f32 and literals are authored from the provider's published table of
/// the same numbers, so print rounding separates them by well under
/// 1e-3 Pa in A and 1e-6 in B — while a wrong-model ladder moves
/// adjacent coefficients by tens of Pa.
const HYBRID_LITERAL_ABS_TOL: f64 = 1e-3;
const HYBRID_LITERAL_REL_TOL: f64 = 1e-6;

/// `mapped_source._resolve_hybrid_coefficients`.
fn resolve_hybrid_coefficients(
    mapping: &Mapping,
    nlevels: usize,
    record_pv: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let literals = mapping.hybrid_literals()?;
    if !record_pv.is_empty() {
        if record_pv.len() % 2 != 0 {
            return Err(frame_invalid(format!(
                "pv coordinate list length {} is not an even A+B split",
                record_pv.len()
            )));
        }
        let half = record_pv.len() / 2;
        if half != nlevels + 1 && half != nlevels {
            return Err(frame_invalid(format!(
                "hybrid coefficient count mismatch: the pv coordinate \
                 octets carry {half} A and {half} B coefficients; \
                 {nlevels} levels accept {} (half-level interfaces) or \
                 {nlevels} (full levels)",
                nlevels + 1
            )));
        }
        let a_values = &record_pv[..half];
        let b_values = &record_pv[half..];
        if a_values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(frame_invalid(
                "pv A coefficients must be finite and non-negative (Pa)",
            ));
        }
        if b_values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
        {
            return Err(frame_invalid(
                "pv B coefficients must be finite within [0, 1]",
            ));
        }
        if let Some((literal_a, literal_b)) = literals {
            for (label, declared, observed) in [
                ("hybrid_a", &literal_a, a_values),
                ("hybrid_b", &literal_b, b_values),
            ] {
                if declared.len() != observed.len() {
                    return Err(frame_invalid(format!(
                        "inline vertical.{label} declares {} coefficients \
                         but the source's pv octets carry {}; the literals \
                         disagree with the bytes",
                        declared.len(),
                        observed.len()
                    )));
                }
                for (index, (lit, pv)) in
                    declared.iter().zip(observed.iter()).enumerate()
                {
                    let tolerance =
                        HYBRID_LITERAL_ABS_TOL + HYBRID_LITERAL_REL_TOL * pv.abs();
                    if (lit - pv).abs() > tolerance {
                        return Err(frame_invalid(format!(
                            "inline vertical.{label} literals disagree with \
                             the source's pv octets at index {index}: \
                             literal {lit} vs pv {pv}"
                        )));
                    }
                }
            }
        }
        return Ok((a_values.to_vec(), b_values.to_vec()));
    }
    if let Some((literal_a, literal_b)) = literals {
        let count = literal_a.len();
        if count != nlevels + 1 && count != nlevels {
            return Err(frame_invalid(format!(
                "hybrid coefficient count mismatch: vertical.hybrid_a \
                 declares {count} coefficients; {nlevels} levels accept \
                 {} (half-level interfaces) or {nlevels} (full levels)",
                nlevels + 1
            )));
        }
        return Ok((literal_a, literal_b));
    }
    Err(frame_invalid(
        "hybrid_sigma_pressure source supplies no A/B coefficients: the \
         selected records carry no pv coordinate octets and the mapping \
         declares no inline vertical.hybrid_a/hybrid_b literals",
    ))
}

/// The literal-only resolution the NetCDF route uses (its bytes carry
/// no pv channel).
pub fn resolve_hybrid_literals_only(
    mapping: &Mapping,
    nlevels: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    resolve_hybrid_coefficients(mapping, nlevels, &[])
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

    let (hybrid_a, hybrid_b) = if mapping.vertical_kind()? == "hybrid_sigma_pressure" {
        // Every selected vertical-bearing record states the whole ladder
        // in its pv octets; one source has one ladder, so disagreement
        // is a mixed-source input, not a choice to make.
        let mut pv_lists: BTreeSet<Vec<u64>> = BTreeSet::new();
        let mut chosen_pv: Vec<f64> = Vec::new();
        for ((_time, _member, name), group) in &matched {
            if !mapping
                .field(name)?
                .target_axes()?
                .iter()
                .any(|axis| axis == "vertical")
            {
                continue;
            }
            for position in group {
                let pv = &records[*position].coordinate_values;
                pv_lists.insert(pv.iter().map(|value| value.to_bits()).collect());
                if !pv.is_empty() {
                    chosen_pv = pv.clone();
                }
            }
        }
        let nonempty = pv_lists.iter().filter(|list| !list.is_empty()).count();
        if nonempty > 1 || (nonempty == 1 && pv_lists.len() > 1) {
            return Err(frame_invalid(
                "selected GRIB records do not share one pv coordinate \
                 list; hybrid A/B coefficients must be identical across \
                 the source",
            ));
        }
        resolve_hybrid_coefficients(mapping, vertical_values.len(), &chosen_pv)?
    } else {
        (Vec::new(), Vec::new())
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
        hybrid_a,
        hybrid_b,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn hybrid_mapping(literals: Option<(&str, &str)>) -> Mapping {
        let vertical = match literals {
            Some((a, b)) => format!(
                r#""kind": "hybrid_sigma_pressure", "units": "1",
                   "surface_pressure_field": "surface_pressure",
                   "hybrid_a": {a}, "hybrid_b": {b}"#
            ),
            None => r#""kind": "hybrid_sigma_pressure", "units": "1",
                       "surface_pressure_field": "surface_pressure""#
                .to_owned(),
        };
        let text = format!(
            r#"{{"schema": "rw-wps.mapping.v1", "name": "t", "format": "grib2",
                "coordinates": {{"vertical": {{{vertical}}}}}, "fields": {{}}}}"#
        );
        let bytes = text.as_bytes().to_vec();
        Mapping {
            sha256: crate::digest::bytes_sha256(&bytes),
            doc: crate::node::Node::parse(&bytes).unwrap(),
            path: "<test>".to_owned(),
        }
    }

    const PV: [f64; 8] = [0.0, 6000.0, 4000.0, 0.0, 0.0, 0.24, 0.7, 1.0];

    #[test]
    fn pv_octets_resolve_into_an_a_then_b_split() {
        let mapping = hybrid_mapping(None);
        let (a, b) = resolve_hybrid_coefficients(&mapping, 3, &PV).unwrap();
        assert_eq!(a, vec![0.0, 6000.0, 4000.0, 0.0]);
        assert_eq!(b, vec![0.0, 0.24, 0.7, 1.0]);
    }

    #[test]
    fn literals_supply_the_ladder_when_the_bytes_carry_no_pv() {
        let mapping = hybrid_mapping(Some((
            "[0.0, 6000.0, 4000.0, 0.0]",
            "[0.0, 0.24, 0.7, 1.0]",
        )));
        let (a, b) = resolve_hybrid_coefficients(&mapping, 3, &[]).unwrap();
        assert_eq!(a, vec![0.0, 6000.0, 4000.0, 0.0]);
        assert_eq!(b, vec![0.0, 0.24, 0.7, 1.0]);
    }

    #[test]
    fn literals_disagreeing_with_pv_refuse_naming_the_index() {
        let mapping = hybrid_mapping(Some((
            "[0.0, 6100.0, 4000.0, 0.0]",
            "[0.0, 0.24, 0.7, 1.0]",
        )));
        let refusal = resolve_hybrid_coefficients(&mapping, 3, &PV).unwrap_err();
        assert!(refusal.message.contains("index 1"), "{}", refusal.message);
    }

    #[test]
    fn no_pv_and_no_literals_refuses_naming_both_channels() {
        let mapping = hybrid_mapping(None);
        let refusal = resolve_hybrid_coefficients(&mapping, 3, &[]).unwrap_err();
        assert!(refusal.message.contains("pv"), "{}", refusal.message);
        assert!(refusal.message.contains("hybrid_a"), "{}", refusal.message);
    }

    #[test]
    fn pv_half_count_is_held_to_the_declared_ladder() {
        let mapping = hybrid_mapping(None);
        let long_pv: Vec<f64> = (0..12).map(f64::from).collect();
        let refusal = resolve_hybrid_coefficients(&mapping, 3, &long_pv).unwrap_err();
        assert!(refusal.message.contains('6'), "{}", refusal.message);
        assert!(refusal.message.contains('4'), "{}", refusal.message);
    }
}
