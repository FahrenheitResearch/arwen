//! Cross-source composition.
//!
//! `compose` mirrors `gpuwm.mapped_composition.decode_composed_source`:
//! a base mapping plus a `gpuwm-mapped-composition-v2` document, role-bound
//! supplements and provenance, terrain composition, bound-field subsetting
//! and cross-source partition decode.
//!
//! **What is here and what stays in Python.**  The BYTE WORK is here:
//! partition decode, the coordinate-subset solve, terrain composition, the
//! bound-field borrow, the union mapping, frame materialization and the two
//! evidence products only that work can produce.  Everything above it stays
//! in `mapped_composition` and runs around this call —  manifest
//! verification, the authority hash window, the role inventory, the
//! contributing-mapping hash pins, the primary-versus-donor field
//! agreement, the soil-layer contract against the donor table, and the two
//! post-return checks (the frames' `mapping_sha256`, and "did the engine
//! compose exactly the declared bindings").
//!
//! The document grammar below is therefore a SECOND reader of a contract
//! Python has already validated on the same bytes: through the front door
//! `load_composition` runs first and refuses first.  It is ported anyway
//! because the engine must not proceed on a contract it has not read, and
//! it is transcribed condition for condition so it can never refuse a
//! composition Python accepts — a stricter engine here would refuse a
//! preparation that is in fact correct.  Where Python raises `TypeError`
//! for a non-object, the engine refuses `mapping_invalid` (`ValueError`);
//! that divergence is reachable only by calling the exe directly with a
//! document the Python door would have refused first.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::engine::Invocation;
use crate::model::Mapping;
use crate::node::Node;
use crate::refusal::{mapping_invalid, missing_input, python_list_repr, python_repr, Result};

pub const COMPOSITION_SCHEMA: &str = "gpuwm-mapped-composition-v2";
/// `mapped_composition.PENDING_COMPOSITION_SCHEMA`.
pub const PENDING_COMPOSITION_SCHEMA: &str = "gpuwm-cross-source-composition-pending-v1";
/// What `mapped_engine_bridge.read_composition_evidence` reads.
pub const COMPOSITION_EVIDENCE_SCHEMA: &str = "gpuwm-mapped-composition-evidence-v1";
/// The one field a terrain supplement may provide.
pub const EXTERNAL_FIELD: &str = "terrain_height";
/// `mapped_composition._SOIL_PAIR`.
pub const SOIL_PAIR: [&str; 2] = ["soil_temperature", "volumetric_soil_moisture"];
/// `mapped_composition._BOUND_GRID_ALIGNMENTS`.
pub const BOUND_GRID_ALIGNMENTS: [&str; 1] = ["exact_coordinate_subset"];

const COMPOSITION_KEYS: [&str; 6] = [
    "schema",
    "name",
    "mapping_binding",
    "soil_layers",
    "supplements",
    "field_sources",
];
const COMPOSITION_REQUIRED: [&str; 5] = [
    "schema",
    "name",
    "mapping_binding",
    "soil_layers",
    "supplements",
];
const TERRAIN_KEYS: [&str; 8] = [
    "data_role",
    "provenance_role",
    "format",
    "field",
    "selector_authority",
    "grid_alignment",
    "time_alignment",
    "require_invariant_across_time",
];
const BINDING_KEYS: [&str; 8] = [
    "source_id",
    "mapping_role",
    "mapping_sha256",
    "data_role",
    "provenance_role",
    "fields",
    "grid_alignment",
    "time_alignment",
];

/// `composition.supplements.terrain_height`.
#[derive(Debug, Clone)]
pub struct TerrainSupplement {
    pub data_role: String,
    pub provenance_role: String,
    pub time_alignment: String,
}

/// One `composition.field_sources[name]` binding.
#[derive(Debug, Clone)]
pub struct Binding {
    pub name: String,
    pub source_id: String,
    pub mapping_role: String,
    pub mapping_sha256: String,
    pub data_role: String,
    pub provenance_role: String,
    pub fields: Vec<String>,
    pub grid_alignment: String,
    pub time_alignment: String,
}

/// The composition document, validated against the mapping's own bytes.
#[derive(Debug)]
pub struct Composition {
    pub doc: Node,
    pub sha256: String,
    pub path: String,
    pub terrain: Option<TerrainSupplement>,
    /// Bindings in NAME order, which is the order
    /// `decode_composed_source` composes them in (`sorted(bindings)`).
    pub bindings: Vec<Binding>,
}

/// `_object`: a closed key set, checked the way the Python engine checks it.
fn object(value: &Node, label: &str, allowed: &[&str], required: &[&str]) -> Result<()> {
    if !value.is_object() {
        return Err(mapping_invalid(format!("{label} must be an object")));
    }
    let declared: BTreeSet<&str> = value.entries().iter().map(|(key, _)| key.as_str()).collect();
    let unknown: Vec<&str> = declared
        .iter()
        .copied()
        .filter(|key| !allowed.contains(key))
        .collect();
    if !unknown.is_empty() {
        return Err(mapping_invalid(format!(
            "{label} has unknown key(s): {}",
            python_list_repr(&unknown)
        )));
    }
    let mut missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|key| !declared.contains(key))
        .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(mapping_invalid(format!(
            "{label} is missing required key(s): {}",
            python_list_repr(&missing)
        )));
    }
    Ok(())
}

/// `_string`: a non-empty, non-whitespace string.
fn string<'a>(value: Option<&'a Node>, label: &str) -> Result<&'a str> {
    match value.and_then(Node::as_str) {
        Some(text) if !text.trim().is_empty() => Ok(text),
        _ => Err(mapping_invalid(format!(
            "{label} must be a non-empty string"
        ))),
    }
}

/// `_digest`: 64 lowercase hex characters.
fn digest(value: Option<&Node>, label: &str) -> Result<String> {
    let text = string(value, label)?.to_ascii_lowercase();
    if text.len() != 64 || !text.chars().all(|character| character.is_ascii_hexdigit()) {
        return Err(mapping_invalid(format!("{label} must be a SHA-256 digest")));
    }
    Ok(text)
}

impl Composition {
    /// `mapped_composition.load_composition`, against the mapping in hand.
    pub fn load(path: &str, mapping: &Mapping) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            missing_input(format!("cannot read composition {path}: {error}"))
        })?;
        let sha256 = crate::digest::bytes_sha256(&bytes);
        let doc = Node::parse(&bytes)
            .map_err(|error| mapping_invalid(format!("composition {path} is not JSON: {error}")))?;
        if doc.get("schema").and_then(Node::as_str) == Some(PENDING_COMPOSITION_SCHEMA) {
            return Err(mapping_invalid(pending_composition_refusal(&doc, path)?));
        }
        object(&doc, "composition", &COMPOSITION_KEYS, &COMPOSITION_REQUIRED)?;
        let schema = doc.get("schema").and_then(Node::as_str).unwrap_or("");
        if schema != COMPOSITION_SCHEMA {
            return Err(mapping_invalid(format!(
                "unsupported composition schema {}",
                python_repr(schema)
            )));
        }
        string(doc.get("name"), "composition.name")?;
        if doc.get("mapping_binding").and_then(Node::as_str) != Some("input_manifest_sha256") {
            return Err(mapping_invalid(
                "composition mapping_binding must be input_manifest_sha256",
            ));
        }
        let mut declared_bound: BTreeSet<String> = BTreeSet::new();
        for field in mapping.fields()? {
            if field.provider() == Some("composition_bound") {
                declared_bound.insert(field.name.clone());
            }
        }
        let bindings = validate_field_sources(
            doc.field("field_sources"),
            mapping,
            &declared_bound,
        )?;
        let mut covered: BTreeMap<&str, &str> = BTreeMap::new();
        for binding in &bindings {
            for name in &binding.fields {
                covered.insert(name.as_str(), binding.name.as_str());
            }
        }
        let unbound: Vec<&String> = declared_bound
            .iter()
            .filter(|name| !covered.contains_key(name.as_str()))
            .collect();
        if !unbound.is_empty() {
            let names: Vec<&str> = unbound.into_iter().map(String::as_str).collect();
            return Err(mapping_invalid(format!(
                "composition_bound field(s) {} have no contributing source; \
                 every declared gap must be bound in field_sources",
                python_list_repr(&names)
            )));
        }
        let soil_bindings: BTreeSet<&str> = SOIL_PAIR
            .iter()
            .filter_map(|name| covered.get(name).copied())
            .collect();
        if soil_bindings.len() > 1
            || (!soil_bindings.is_empty()
                && SOIL_PAIR.iter().any(|name| !covered.contains_key(name)))
        {
            return Err(mapping_invalid(
                "the canonical soil pair must ride one contributing source: a \
                 soil-depth contract validates against a single donor \
                 mapping's layer geometry",
            ));
        }
        // `validate_soil_layer_contract` is deliberately NOT ported: it
        // reads sealed documents rather than bytes, it validates against
        // the DONOR table once the donor mapping is pinned, and
        // `decode_composed_source` runs it on both routes.  A second
        // implementation here could only disagree with the first.
        let supplements = doc
            .get("supplements")
            .ok_or_else(|| mapping_invalid("composition.supplements must be an object"))?;
        object(
            supplements,
            "composition.supplements",
            &[EXTERNAL_FIELD],
            &[],
        )?;
        let terrain_declaration = supplements.field(EXTERNAL_FIELD);
        if terrain_declaration.is_some() && covered.contains_key(EXTERNAL_FIELD) {
            return Err(mapping_invalid(
                "terrain has two providers: the declared supplement and a \
                 contributing source binding",
            ));
        }
        if terrain_declaration.is_none() && !covered.contains_key(EXTERNAL_FIELD) {
            return Err(mapping_invalid(
                "terrain_height has no contributing source and no supplement; \
                 the composition must provide terrain through exactly one",
            ));
        }
        let terrain = match terrain_declaration {
            Some(value) => Some(validate_terrain_supplement(value, mapping)?),
            None => None,
        };
        Ok(Self {
            doc,
            sha256,
            path: path.to_owned(),
            terrain,
            bindings,
        })
    }
}

/// `mapped_composition._validate_terrain_supplement`.
fn validate_terrain_supplement(value: &Node, mapping: &Mapping) -> Result<TerrainSupplement> {
    let label = format!("composition.supplements.{EXTERNAL_FIELD}");
    object(value, &label, &TERRAIN_KEYS, &TERRAIN_KEYS)?;
    for key in [
        "data_role",
        "provenance_role",
        "format",
        "field",
        "selector_authority",
        "grid_alignment",
        "time_alignment",
    ] {
        string(value.get(key), &format!("terrain supplement.{key}"))?;
    }
    if value.get("field").and_then(Node::as_str) != Some(EXTERNAL_FIELD) {
        return Err(mapping_invalid(
            "terrain supplement must provide terrain_height",
        ));
    }
    let format = value.get("format").and_then(Node::as_str).unwrap_or("");
    if format != mapping.format()? || !matches!(format, "grib1" | "grib2" | "netcdf") {
        return Err(mapping_invalid(
            "terrain supplement format must equal the mapping format",
        ));
    }
    if value.get("selector_authority").and_then(Node::as_str) != Some("mapping_field_exact") {
        return Err(mapping_invalid(
            "terrain selector authority must be mapping_field_exact",
        ));
    }
    if value.get("grid_alignment").and_then(Node::as_str) != Some("exact_coordinate_subset") {
        return Err(mapping_invalid(
            "terrain grid alignment must be exact_coordinate_subset",
        ));
    }
    let time_alignment = value
        .get("time_alignment")
        .and_then(Node::as_str)
        .unwrap_or("");
    if !crate::join::TERRAIN_TIME_ALIGNMENTS.contains(&time_alignment) {
        let mut allowed = crate::join::TERRAIN_TIME_ALIGNMENTS.to_vec();
        allowed.sort_unstable();
        return Err(mapping_invalid(format!(
            "terrain time alignment must be one of {}",
            python_list_repr(&allowed)
        )));
    }
    if value.get("require_invariant_across_time").and_then(Node::as_bool) != Some(true) {
        return Err(mapping_invalid(
            "terrain supplement must be invariant across all supplied times",
        ));
    }
    let field = mapping.field(EXTERNAL_FIELD).map_err(|_| {
        mapping_invalid("mapping terrain_height must have a direct selector")
    })?;
    if field.derivation().is_some() || field.selectors().is_empty() {
        return Err(mapping_invalid(
            "mapping terrain_height must have a direct selector",
        ));
    }
    let materialized: Vec<String> = field
        .source_axes()?
        .into_iter()
        .filter(|axis| axis != "time" && axis != "member")
        .collect();
    if materialized != ["y", "x"]
        || field.target_axes()? != ["y", "x"]
        || field.location()? != "surface"
        || field.staggering() != "none"
        || field.units_target()? != "m"
        || field.missing_kind()? != "reject"
    {
        return Err(mapping_invalid(
            "mapping terrain_height must be finite unstaggered surface metres",
        ));
    }
    Ok(TerrainSupplement {
        data_role: value
            .get("data_role")
            .and_then(Node::as_str)
            .unwrap_or_default()
            .to_owned(),
        provenance_role: value
            .get("provenance_role")
            .and_then(Node::as_str)
            .unwrap_or_default()
            .to_owned(),
        time_alignment: time_alignment.to_owned(),
    })
}

/// `mapped_composition._validate_field_sources`.
fn validate_field_sources(
    value: Option<&Node>,
    mapping: &Mapping,
    declared_bound: &BTreeSet<String>,
) -> Result<Vec<Binding>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if !value.is_object() {
        return Err(mapping_invalid("composition.field_sources must be an object"));
    }
    let mut bindings: Vec<Binding> = Vec::new();
    let mut covered: BTreeMap<String, String> = BTreeMap::new();
    let mut roles: BTreeMap<String, String> = BTreeMap::new();
    for (binding_name, raw) in value.entries() {
        string(Some(&Node::String(binding_name.clone())), "field_sources binding name")?;
        let label = format!("composition.field_sources.{binding_name}");
        object(raw, &label, &BINDING_KEYS, &BINDING_KEYS)?;
        let source_id = string(raw.get("source_id"), &format!("{label}.source_id"))?.to_owned();
        let mapping_sha256 = digest(
            raw.get("mapping_sha256"),
            &format!("{label}.mapping_sha256"),
        )?;
        let mut role_values: Vec<String> = Vec::new();
        for key in ["mapping_role", "data_role", "provenance_role"] {
            let role = string(raw.get(key), &format!("{label}.{key}"))?.to_owned();
            if let Some(previous) = roles.get(&role) {
                return Err(mapping_invalid(format!(
                    "{label}.{key} reuses role {} already claimed by {previous}; \
                     every binding role must be unique",
                    python_repr(&role)
                )));
            }
            roles.insert(role.clone(), format!("{label}.{key}"));
            role_values.push(role);
        }
        let grid_alignment = raw
            .get("grid_alignment")
            .and_then(Node::as_str)
            .unwrap_or("");
        if !BOUND_GRID_ALIGNMENTS.contains(&grid_alignment) {
            let mut allowed = BOUND_GRID_ALIGNMENTS.to_vec();
            allowed.sort_unstable();
            return Err(mapping_invalid(format!(
                "{label}.grid_alignment must be one of {}: a cross-grid \
                 contribution has no landed regrid capability",
                python_list_repr(&allowed)
            )));
        }
        let time_alignment = raw
            .get("time_alignment")
            .and_then(Node::as_str)
            .unwrap_or("");
        if !crate::join::BOUND_TIME_ALIGNMENTS.contains(&time_alignment) {
            let mut allowed = crate::join::BOUND_TIME_ALIGNMENTS.to_vec();
            allowed.sort_unstable();
            return Err(mapping_invalid(format!(
                "{label}.time_alignment must be one of {}",
                python_list_repr(&allowed)
            )));
        }
        let raw_fields = raw.get("fields").filter(|node| node.is_array());
        let names: Vec<String> = match raw_fields {
            Some(node) if !node.items().is_empty() => node
                .items()
                .iter()
                .map(|item| {
                    string(Some(item), &format!("{label}.fields[]")).map(str::to_owned)
                })
                .collect::<Result<Vec<String>>>()?,
            _ => {
                return Err(mapping_invalid(format!(
                    "{label}.fields must be a non-empty list"
                )))
            }
        };
        if names.iter().collect::<BTreeSet<&String>>().len() != names.len() {
            return Err(mapping_invalid(format!("{label}.fields repeats a field")));
        }
        for name in &names {
            if mapping.field(name).is_err() {
                return Err(mapping_invalid(format!(
                    "{label} binds unknown field {}",
                    python_repr(name)
                )));
            }
            if !declared_bound.contains(name) {
                return Err(mapping_invalid(format!(
                    "field {} has two providers: the primary mapping and \
                     contributing source binding {}",
                    python_repr(name),
                    python_repr(binding_name)
                )));
            }
            if let Some(previous) = covered.get(name) {
                return Err(mapping_invalid(format!(
                    "field {} is bound by more than one contributing source \
                     ({} and {})",
                    python_repr(name),
                    python_repr(previous),
                    python_repr(binding_name)
                )));
            }
            covered.insert(name.clone(), binding_name.clone());
        }
        bindings.push(Binding {
            name: binding_name.clone(),
            source_id,
            mapping_role: role_values[0].clone(),
            mapping_sha256,
            data_role: role_values[1].clone(),
            provenance_role: role_values[2].clone(),
            fields: names,
            grid_alignment: grid_alignment.to_owned(),
            time_alignment: time_alignment.to_owned(),
        });
    }
    bindings.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(bindings)
}

/// `mapped_composition._pending_composition_refusal`.
fn pending_composition_refusal(doc: &Node, path: &str) -> Result<String> {
    let file_name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let label = format!("pending composition {file_name}");
    object(
        doc,
        &label,
        &["schema", "name", "pending"],
        &["schema", "name", "pending"],
    )?;
    let name = string(doc.get("name"), "pending composition.name")?;
    let pending = doc
        .get("pending")
        .ok_or_else(|| mapping_invalid("pending composition.pending must be an object"))?;
    object(
        pending,
        "pending composition.pending",
        &["missing_canonical_state", "reason", "supply_route"],
        &["missing_canonical_state", "reason", "supply_route"],
    )?;
    let missing = pending.get("missing_canonical_state");
    let entries = match missing {
        Some(node) if node.is_array() && !node.items().is_empty() => node.items(),
        _ => {
            return Err(mapping_invalid(format!(
                "{file_name} pending.missing_canonical_state must be a \
                 non-empty list of canonical field names"
            )))
        }
    };
    let mut names: Vec<&str> = Vec::new();
    for entry in entries {
        names.push(string(Some(entry), "pending.missing_canonical_state entry")?);
    }
    let reason = string(pending.get("reason"), "pending composition.reason")?;
    let route = string(pending.get("supply_route"), "pending composition.supply_route")?;
    Ok(format!(
        "composition {} is an explicit PENDING declaration, not a runnable \
         contract: the source publishes no {}. {reason}. Initializing from \
         this source alone would invent those fields, so it is refused; \
         {route}.",
        python_repr(name),
        names.join(", ")
    ))
}

/// `mapped_composition._partition_mapping`: a decoder-only view of one
/// side of the composition.
///
/// The sealed, fully validated mapping remains the sole semantic
/// authority; partitioning only prevents a decoder from demanding a field
/// the composition contract explicitly assigns to the other product.
/// Without it the primary decode demands `terrain_height` from bytes the
/// composition assigns to the supplement, and refuses.
pub fn partition_mapping(mapping: &Mapping, terrain_only: bool) -> Result<Mapping> {
    let fields = mapping
        .doc
        .get("fields")
        .filter(|node| node.is_object())
        .ok_or_else(|| mapping_invalid("mapping.fields must be an object"))?;
    let kept = fields.object_filtered(|name, field| {
        (name == EXTERNAL_FIELD) == terrain_only
            && field.field("provider").and_then(Node::as_str) != Some("composition_bound")
    });
    let names: Vec<&str> = kept.entries().iter().map(|(name, _)| name.as_str()).collect();
    if terrain_only && names != [EXTERNAL_FIELD] {
        return Err(mapping_invalid("mapping lacks the direct terrain field"));
    }
    if !terrain_only && names.is_empty() {
        return Err(mapping_invalid(
            "mapping has no primary fields after terrain partitioning",
        ));
    }
    Ok(Mapping {
        doc: mapping.doc.with_entry("fields", kept),
        sha256: mapping.sha256.clone(),
        path: mapping.path.clone(),
    })
}

/// `mapped_composition._partition_contributing`: a decoder-only view of
/// one contributing source's bound fields.
pub fn partition_contributing(
    donor: &Mapping,
    field_names: &[String],
    binding_name: &str,
) -> Result<Mapping> {
    let fields = donor
        .doc
        .get("fields")
        .filter(|node| node.is_object())
        .ok_or_else(|| mapping_invalid("mapping.fields must be an object"))?;
    let quoted = python_repr(binding_name);
    let mut kept: Vec<(String, Node)> = Vec::with_capacity(field_names.len());
    for name in field_names {
        let Some(field) = fields.get(name) else {
            return Err(mapping_invalid(format!(
                "contributing source binding {quoted} borrows {} but the \
                 contributing mapping does not map it",
                python_repr(name)
            )));
        };
        if field
            .field("selectors")
            .map(|node| node.items().is_empty())
            .unwrap_or(true)
        {
            return Err(mapping_invalid(format!(
                "contributing source binding {quoted} borrows {}, which the \
                 contributing mapping does not provide with direct selectors; \
                 a derived or externally bound donor field is not decodable in \
                 a partitioned donor decode",
                python_repr(name)
            )));
        }
        kept.push((name.clone(), field.clone()));
    }
    Ok(Mapping {
        doc: donor.doc.with_entry("fields", Node::Object(kept)),
        sha256: donor.sha256.clone(),
        path: donor.path.clone(),
    })
}

/// The UNION view a composed frameset materializes against: the primary
/// mapping's own fields, plus each borrowed field's spec taken from the
/// DONOR mapping — the decode authority for those octets.
///
/// Built structurally, never through JSON text: the crate's own
/// `Cargo.toml` records a one-ULP float parse that moved every cell of a
/// scaled field.  The union keeps the PRIMARY's identity (sha256 and
/// path), because a frame states the mapping the caller sealed and
/// `_compose_through_engine` re-checks exactly that digest on return.
pub fn union_mapping(
    primary: &Mapping,
    donors: &BTreeMap<String, Mapping>,
    bindings: &[Binding],
) -> Result<Mapping> {
    let mut fields = primary
        .doc
        .get("fields")
        .filter(|node| node.is_object())
        .cloned()
        .ok_or_else(|| mapping_invalid("mapping.fields must be an object"))?;
    for binding in bindings {
        let donor = &donors[&binding.name];
        let donor_fields = donor
            .doc
            .get("fields")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("contributing mapping.fields must be an object"))?;
        for name in &binding.fields {
            let spec = donor_fields.get(name).ok_or_else(|| {
                mapping_invalid(format!(
                    "contributing source binding {} borrows {} but the \
                     contributing mapping does not map it",
                    python_repr(&binding.name),
                    python_repr(name)
                ))
            })?;
            fields = fields.with_entry(name, spec.clone());
        }
    }
    Ok(Mapping {
        doc: primary.doc.with_entry("fields", fields),
        sha256: primary.sha256.clone(),
        path: primary.path.clone(),
    })
}

/// `mapped_composition._bind_manifest_member`: stamp the manifest's
/// EXPLICIT member onto every composed record.
///
/// For archives whose product octets carry no ensemble identity: the
/// caller's verified authority named the member, the sealed input
/// manifest declares it, and this replaces ABSENT product-defined
/// membership only.  Bytes that already encode a member contradict the
/// declaration and refuse rather than being overwritten -- overwriting
/// would let one archive's frames claim another member's identity.  The
/// refusal wording is the Python engine's, verbatim, so the two engines
/// refuse the same bytes with the same sentence.
pub fn bind_manifest_member(
    collection: crate::assemble::DecodedCollection,
    member: &str,
) -> Result<crate::assemble::DecodedCollection> {
    let encoded: BTreeSet<Option<String>> = collection
        .direct
        .values()
        .map(|value| value.member.clone())
        .collect();
    if encoded != BTreeSet::from([None]) {
        let mut names: Vec<String> = encoded.into_iter().flatten().collect();
        names.sort();
        return Err(crate::refusal::manifest_mismatch(format!(
            "the composition input manifest binds explicit member {}, but \
             the decoded records already encode product-defined member(s) \
             {}; an explicit member binding replaces ABSENT product \
             membership only, so stamping it here would overwrite a \
             producer-declared identity",
            python_repr(member),
            python_list_repr(&names)
        )));
    }
    let mut direct = BTreeMap::new();
    for ((valid_time, _old_member, field_name), value) in collection.direct {
        direct.insert(
            (valid_time, Some(member.to_owned()), field_name),
            crate::assemble::DirectValue {
                member: Some(member.to_owned()),
                ..value
            },
        );
    }
    let mut source_cycles = BTreeMap::new();
    for ((valid_time, _old_member), cycle) in collection.source_cycles {
        source_cycles.insert((valid_time, Some(member.to_owned())), cycle);
    }
    Ok(crate::assemble::DecodedCollection {
        latitude: collection.latitude,
        longitude: collection.longitude,
        vertical_values: collection.vertical_values,
        direct,
        source_cycles,
        grid_fingerprint: collection.grid_fingerprint,
    })
}

/// Role bindings from argv, in the shape the composition reads them.
fn role_inventory(bindings: &[(String, String)]) -> BTreeMap<String, Vec<String>> {
    let mut inventory: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (role, path) in bindings {
        inventory.entry(role.clone()).or_default().push(path.clone());
    }
    inventory
}

fn single_role(inventory: &BTreeMap<String, Vec<String>>, kind: &str) -> Result<BTreeMap<String, String>> {
    let mut single = BTreeMap::new();
    for (role, paths) in inventory {
        if paths.len() != 1 {
            return Err(crate::refusal::usage(format!(
                "--{kind} {role}= was supplied {} times; a {kind} role binds \
                 exactly one document",
                paths.len()
            )));
        }
        single.insert(role.clone(), paths[0].clone());
    }
    Ok(single)
}

/// `compose`: mapping + composition + role bindings → a frameset directory
/// plus the composition evidence document.
pub fn run_compose(invocation: &Invocation, progress: &mut dyn FnMut(Value)) -> Result<Value> {
    let mapping = Mapping::load(&invocation.mapping)?;
    let Some(composition_path) = invocation.composition.as_deref() else {
        return Err(crate::refusal::usage(
            "compose requires --composition COMPOSITION.json",
        ));
    };
    let composition = Composition::load(composition_path, &mapping)?;
    let primary = crate::engine::read_input_list(&invocation.input_list)?;
    let supplements = role_inventory(&invocation.supplements);
    let provenance = single_role(&role_inventory(&invocation.provenance), "provenance")?;
    let contributing = single_role(
        &role_inventory(&invocation.contributing_mappings),
        "contributing-mapping",
    )?;

    // The role inventory the contract declares, checked against argv
    // before a byte is read.  Named breakage: a supplement bound under a
    // role the composition does not declare would be hashed into the
    // frameset's inputs and never read, so the frameset would name bytes
    // that contributed nothing.
    let mut expected_data: BTreeSet<&str> = BTreeSet::new();
    let mut expected_provenance: BTreeSet<&str> = BTreeSet::new();
    if let Some(terrain) = &composition.terrain {
        expected_data.insert(terrain.data_role.as_str());
        expected_provenance.insert(terrain.provenance_role.as_str());
    }
    let mut expected_mappings: BTreeSet<&str> = BTreeSet::new();
    for binding in &composition.bindings {
        expected_data.insert(binding.data_role.as_str());
        expected_provenance.insert(binding.provenance_role.as_str());
        expected_mappings.insert(binding.mapping_role.as_str());
    }
    for (kind, declared, supplied) in [
        ("supplement", expected_data, supplements.keys().map(String::as_str).collect::<BTreeSet<&str>>()),
        ("provenance", expected_provenance, provenance.keys().map(String::as_str).collect()),
        ("contributing mapping", expected_mappings, contributing.keys().map(String::as_str).collect()),
    ] {
        if declared != supplied {
            let declared: Vec<&str> = declared.into_iter().collect();
            let supplied: Vec<&str> = supplied.into_iter().collect();
            return Err(mapping_invalid(format!(
                "{kind} role inventory differs from the composition: expected \
                 {}, got {}",
                python_list_repr(&declared),
                python_list_repr(&supplied)
            )));
        }
    }

    // Every byte the frameset will name, hashed once.  The frames carry
    // the primary inputs AND every supplement (a borrowed field's donor
    // object is an input to the composed frame), exactly as
    // `decode_composed_source` builds `input_hashes`.
    let mut data_files: Vec<String> = primary.clone();
    for paths in supplements.values() {
        for path in paths {
            if !data_files.contains(path) {
                data_files.push(path.clone());
            }
        }
    }
    let digests = crate::engine::input_digests(&data_files)?;
    // The provenance documents are sealed by the manifest and named in
    // the composition evidence, but they are NOT frameset inputs: no
    // number in a frame came from them, and hashing them into
    // `input_sha256` would claim they did.
    let provenance_files: Vec<String> = provenance.values().cloned().collect();
    let provenance_digests = crate::engine::input_digests(&provenance_files)?;
    let mut member_binding: Option<(String, String)> = None;
    if let (Some(manifest), Some(expected)) = (
        invocation.input_manifest.as_ref(),
        invocation.input_manifest_sha256.as_ref(),
    ) {
        let mut sealed = digests.clone();
        sealed.extend(
            provenance_digests
                .iter()
                .map(|(path, value)| (path.clone(), value.clone())),
        );
        member_binding = crate::engine::verify_composition_manifest(
            manifest,
            expected,
            &mapping,
            &composition.sha256,
            &primary,
            &supplements,
            &provenance,
            &sealed,
        )?;
    }

    let mut combined = crate::engine::decode_collection(
        &partition_mapping(&mapping, false)?,
        &primary,
        progress,
    )?;
    progress(json!({
        "event": "composed_primary",
        "valid_times": combined.source_cycles.len(),
    }));
    let mut alignment_receipt: Option<Value> = None;
    if let Some(terrain) = &composition.terrain {
        let files = &supplements[&terrain.data_role];
        let supplement = crate::engine::decode_collection(
            &partition_mapping(&mapping, true)?,
            files,
            progress,
        )?;
        let (composed, receipt) =
            crate::join::compose_terrain(&combined, &supplement, &terrain.time_alignment)?;
        combined = composed;
        alignment_receipt = Some(receipt);
        progress(json!({"event": "composed_terrain", "objects": files.len()}));
    }

    let mut donors: BTreeMap<String, Mapping> = BTreeMap::new();
    let mut contributing_records: Vec<Value> = Vec::new();
    let mut terrain_binding_receipt: Option<Value> = None;
    for binding in &composition.bindings {
        let donor_path = &contributing[&binding.mapping_role];
        let donor = Mapping::load(donor_path)?;
        if donor.sha256 != binding.mapping_sha256 {
            return Err(mapping_invalid(format!(
                "contributing source binding {} authority hash mismatch: the \
                 composition pins {}, the supplied mapping bytes hash to {}",
                python_repr(&binding.name),
                binding.mapping_sha256,
                donor.sha256
            )));
        }
        if donor.name()? != binding.source_id {
            return Err(mapping_invalid(format!(
                "contributing source binding {} names source_id {} but the \
                 pinned mapping is {}",
                python_repr(&binding.name),
                python_repr(&binding.source_id),
                python_repr(donor.name()?)
            )));
        }
        let donor_files = &supplements[&binding.data_role];
        let donor_collection = crate::engine::decode_collection(
            &partition_contributing(&donor, &binding.fields, &binding.name)?,
            donor_files,
            progress,
        )?;
        let (composed, receipt) =
            crate::join::compose_bound_fields(&combined, &donor_collection, binding)?;
        combined = composed;
        let provenance_path = &provenance[&binding.provenance_role];
        let mut fields = binding.fields.clone();
        fields.sort();
        contributing_records.push(json!({
            "binding": binding.name,
            "source_id": binding.source_id,
            "mapping": {"path": donor_path, "sha256": donor.sha256},
            "data": donor_files
                .iter()
                .map(|path| json!({"path": path, "sha256": digests[path]}))
                .collect::<Vec<Value>>(),
            "provenance": {
                "path": provenance_path,
                "sha256": provenance_digests[provenance_path],
            },
            "fields": fields,
            "alignment": receipt,
        }));
        if binding.fields.iter().any(|name| name == EXTERNAL_FIELD) {
            terrain_binding_receipt = Some(
                contributing_records
                    .last()
                    .expect("just pushed")
                    .get("alignment")
                    .cloned()
                    .expect("the record carries its alignment"),
            );
        }
        donors.insert(binding.name.clone(), donor);
        progress(json!({
            "event": "composed_binding",
            "binding": binding.name,
            "fields": binding.fields.len(),
        }));
    }

    // The manifest's explicit member binding lands LAST, after every
    // join: the composition operates on the identity the bytes carry
    // (donor alignment refuses member-bearing donors on its own terms),
    // and the stamp then reaches every canonical frame at once.  Twin of
    // `mapped_composition._bind_manifest_member`.
    if let Some((member, _identity)) = &member_binding {
        combined = bind_manifest_member(combined, member)?;
    }

    let union = union_mapping(&mapping, &donors, &composition.bindings)?;
    let frames = crate::frames::materialize_frames(&union, &combined)?;
    let output = std::path::PathBuf::from(
        invocation
            .output
            .as_ref()
            .expect("compose requires --output"),
    );
    let document =
        crate::frames::write_frameset(&output, &union, &combined, &frames, &digests)?;
    // The alignment receipt a composed bundle is judged on: the terrain
    // supplement's when there is one, otherwise the receipt of the
    // binding that supplies terrain.  `MappedSourceBundle` requires one,
    // and the grammar above proved terrain has exactly one provider.
    let mut alignment_receipt = alignment_receipt.or(terrain_binding_receipt).ok_or_else(|| {
        mapping_invalid(
            "composed source has neither a terrain supplement nor a terrain \
             field binding",
        )
    })?;
    if let Some((member, identity)) = &member_binding {
        // The sealed alignment receipt keeps the ensemble identity the
        // manifest bound, beside the alignment it describes.  Recorded
        // only under an explicit binding, so every receipt written
        // before the binding existed is byte-identical today.
        if let Some(receipt) = alignment_receipt.as_object_mut() {
            receipt.insert("member".to_owned(), json!(member));
            receipt.insert("member_identity".to_owned(), json!(identity));
        }
    }
    let evidence = json!({
        "schema": COMPOSITION_EVIDENCE_SCHEMA,
        "engine": {"name": crate::ENGINE_NAME, "version": crate::ENGINE_VERSION},
        "mapping": {"path": mapping.path, "sha256": mapping.sha256},
        "composition": {"path": composition.path, "sha256": composition.sha256},
        "alignment_receipt": alignment_receipt,
        "contributing_sources": contributing_records,
    });
    let evidence_path = output.join("composition.json");
    std::fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&evidence).unwrap_or_default(),
    )
    .map_err(|error| {
        missing_input(format!("cannot write {}: {error}", evidence_path.display()))
    })?;
    Ok(json!({
        "event": "receipt",
        "subcommand": "compose",
        "schema": crate::FRAMESET_SCHEMA,
        "frames": frames.len(),
        "output": output.display().to_string(),
        "grid_fingerprint": combined.grid_fingerprint,
        "bindings": composition.bindings.len(),
        "stream_bytes": document
            .get("stream")
            .and_then(|stream| stream.get("bytes"))
            .cloned()
            .unwrap_or(Value::Null),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, body: &str) -> String {
        let directory = std::env::temp_dir().join("gpuwm-mapped-engine-compose-test");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join(name);
        std::fs::write(&path, body).unwrap();
        path.to_str().unwrap().to_owned()
    }

    const TERRAIN_FIELD: &str = r#"{
        "selectors": [{"parameter": "HGT"}],
        "source_axes": ["time", "y", "x"],
        "target_axes": ["y", "x"],
        "location": "surface",
        "units": {"target": "m"},
        "missing": {"kind": "reject"}
    }"#;

    fn mapping_document(extra_fields: &str) -> String {
        format!(
            r#"{{"schema": "rw-wps.mapping.v1", "format": "grib2", "name": "m",
                "fields": {{"terrain_height": {TERRAIN_FIELD}{extra_fields}}}}}"#
        )
    }

    fn composition_document(supplements: &str, field_sources: &str) -> String {
        format!(
            r#"{{"schema": "gpuwm-mapped-composition-v2", "name": "c",
                "mapping_binding": "input_manifest_sha256",
                "soil_layers": {{}},
                "supplements": {supplements}{field_sources}}}"#
        )
    }

    const TERRAIN_SUPPLEMENT: &str = r#"{"terrain_height": {
        "data_role": "terrain_data", "provenance_role": "terrain_provenance",
        "format": "grib2", "field": "terrain_height",
        "selector_authority": "mapping_field_exact",
        "grid_alignment": "exact_coordinate_subset",
        "time_alignment": "valid_time_exact",
        "require_invariant_across_time": true}}"#;

    #[test]
    fn a_composition_with_the_wrong_schema_refuses_by_naming_it() {
        let mapping = Mapping::load(&write("m1.json", &mapping_document(""))).unwrap();
        let path = write(
            "c1.json",
            &composition_document(TERRAIN_SUPPLEMENT, "")
                .replace(COMPOSITION_SCHEMA, "something-else"),
        );
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::MAPPING_INVALID);
        assert_eq!(
            refusal.message,
            "unsupported composition schema 'something-else'"
        );
    }

    #[test]
    fn a_document_missing_a_required_key_names_every_one_it_lacks() {
        // The closed key set is checked BEFORE the schema string, exactly
        // as `_object` runs before the schema comparison: a document that
        // is neither a v2 composition nor anything else earns the
        // structural refusal, not a schema sentence.
        let mapping = Mapping::load(&write("m1b.json", &mapping_document(""))).unwrap();
        let path = write("c1b.json", r#"{"schema": "something-else"}"#);
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert_eq!(
            refusal.message,
            "composition is missing required key(s): ['mapping_binding', \
             'name', 'soil_layers', 'supplements']"
        );
    }

    #[test]
    fn a_terrain_supplement_composition_parses_its_roles() {
        let mapping = Mapping::load(&write("m2.json", &mapping_document(""))).unwrap();
        let path = write("c2.json", &composition_document(TERRAIN_SUPPLEMENT, ""));
        let composition = Composition::load(&path, &mapping).unwrap();
        let terrain = composition.terrain.expect("a declared supplement");
        assert_eq!(terrain.data_role, "terrain_data");
        assert_eq!(terrain.time_alignment, "valid_time_exact");
        assert!(composition.bindings.is_empty());
    }

    #[test]
    fn a_composition_bound_field_with_no_contributing_source_refuses() {
        // The named breakage: the mapping declares a gap, the composition
        // fills none of it, and materialization would then emit a frame
        // whose bound field silently came from the base source.
        let mapping = Mapping::load(&write(
            "m3.json",
            &mapping_document(
                r#", "skin_temperature": {"provider": "composition_bound",
                    "source_axes": ["y", "x"], "target_axes": ["y", "x"],
                    "location": "surface", "units": {"target": "K"},
                    "missing": {"kind": "reject"}}"#,
            ),
        ))
        .unwrap();
        let path = write("c3.json", &composition_document(TERRAIN_SUPPLEMENT, ""));
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert!(
            refusal
                .message
                .contains("composition_bound field(s) ['skin_temperature'] have no"),
            "{refusal}"
        );
    }

    #[test]
    fn terrain_with_two_providers_refuses() {
        let mapping = Mapping::load(&write(
            "m4.json",
            &mapping_document("").replace(
                r#""selectors": [{"parameter": "HGT"}]"#,
                r#""provider": "composition_bound", "selectors": []"#,
            ),
        ))
        .unwrap();
        let path = write(
            "c4.json",
            &composition_document(
                TERRAIN_SUPPLEMENT,
                r#", "field_sources": {"donor": {
                    "source_id": "d", "mapping_role": "donor_mapping",
                    "mapping_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "data_role": "donor_data", "provenance_role": "donor_provenance",
                    "fields": ["terrain_height"],
                    "grid_alignment": "exact_coordinate_subset",
                    "time_alignment": "valid_time_exact"}}"#,
            ),
        );
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert!(refusal.message.contains("terrain has two providers"), "{refusal}");
    }

    #[test]
    fn a_pending_declaration_refuses_by_naming_the_state_it_lacks() {
        let mapping = Mapping::load(&write("m5.json", &mapping_document(""))).unwrap();
        let path = write(
            "c5.json",
            r#"{"schema": "gpuwm-cross-source-composition-pending-v1",
                "name": "atmosphere-only",
                "pending": {"missing_canonical_state": ["soil_temperature"],
                            "reason": "the producer publishes no land surface",
                            "supply_route": "bind a physical analysis donor"}}"#,
        );
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert_eq!(
            refusal.message,
            "composition 'atmosphere-only' is an explicit PENDING declaration, \
             not a runnable contract: the source publishes no soil_temperature. \
             the producer publishes no land surface. Initializing from this \
             source alone would invent those fields, so it is refused; bind a \
             physical analysis donor."
        );
    }

    #[test]
    fn partitioning_splits_terrain_from_the_primary_fields() {
        let mapping = Mapping::load(&write(
            "m6.json",
            &mapping_document(
                r#", "surface_pressure": {"selectors": [{"parameter": "PRES"}],
                    "source_axes": ["y", "x"], "target_axes": ["y", "x"],
                    "location": "surface", "units": {"target": "Pa"},
                    "missing": {"kind": "reject"}}"#,
            ),
        ))
        .unwrap();
        assert_eq!(
            partition_mapping(&mapping, true).unwrap().field_names().unwrap(),
            vec!["terrain_height".to_owned()]
        );
        assert_eq!(
            partition_mapping(&mapping, false).unwrap().field_names().unwrap(),
            vec!["surface_pressure".to_owned()]
        );
    }

    #[test]
    fn a_partition_keeps_the_sealed_identity_of_the_mapping_it_came_from() {
        // The frameset states the mapping the caller sealed.  A partition
        // that re-hashed its own edited document would bind the frames to
        // a document nobody verified.
        let mapping = Mapping::load(&write("m7.json", &mapping_document(""))).unwrap();
        let partition = partition_mapping(&mapping, true).unwrap();
        assert_eq!(partition.sha256, mapping.sha256);
        assert_eq!(partition.path, mapping.path);
    }

    fn member_collection(member: Option<&str>) -> crate::assemble::DecodedCollection {
        use chrono::NaiveDateTime;
        use ndarray::ArrayD;

        let valid = NaiveDateTime::parse_from_str(
            "1932-03-21 00:00:00", "%Y-%m-%d %H:%M:%S",
        )
        .unwrap();
        let member = member.map(str::to_owned);
        let mut direct = std::collections::BTreeMap::new();
        direct.insert(
            (valid, member.clone(), "surface_pressure".to_owned()),
            crate::assemble::DirectValue {
                name: "surface_pressure".to_owned(),
                valid_time: valid,
                member: member.clone(),
                source_cycle: valid,
                axes: vec!["y".to_owned(), "x".to_owned()],
                values: ArrayD::from_elem(ndarray::IxDyn(&[2, 2]), 100000.0),
                missing_count: 0,
                references: vec!["fixture".to_owned()],
            },
        );
        let mut source_cycles = std::collections::BTreeMap::new();
        source_cycles.insert((valid, member), valid);
        crate::assemble::DecodedCollection {
            latitude: vec![10.0, 10.5],
            longitude: vec![20.0, 20.5],
            vertical_values: vec![100000.0],
            direct,
            source_cycles,
            grid_fingerprint: "fixture-grid".to_owned(),
        }
    }

    #[test]
    fn the_manifest_member_replaces_absent_product_membership() {
        let bound = bind_manifest_member(member_collection(None), "072").unwrap();
        let keys: Vec<_> = bound.direct.keys().cloned().collect();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].1.as_deref(), Some("072"));
        assert!(bound
            .source_cycles
            .keys()
            .all(|(_time, member)| member.as_deref() == Some("072")));
        assert!(bound
            .direct
            .values()
            .all(|value| value.member.as_deref() == Some("072")));
    }

    #[test]
    fn a_producer_declared_member_refuses_the_explicit_binding() {
        // The Python engine's sentence, verbatim: the two engines must
        // refuse the same bytes the same way.
        let refusal = bind_manifest_member(member_collection(Some("9")), "072").unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::MANIFEST_MISMATCH);
        assert_eq!(
            refusal.message,
            "the composition input manifest binds explicit member '072', but \
             the decoded records already encode product-defined member(s) \
             ['9']; an explicit member binding replaces ABSENT product \
             membership only, so stamping it here would overwrite a \
             producer-declared identity"
        );
    }
}
