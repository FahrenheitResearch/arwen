//! Subcommand orchestration: argv grammar, input staging, the three verbs.
//!
//! The invocation grammar is normative (design doc §3.1).  `--input-list`
//! is a file of one path per line rather than argv entries because the
//! 251-file icon-eu prep hit Windows' argv limit; that lesson is baked in
//! from day one, and argv here never carries an input path.
//!
//! Paths are used EXACTLY as the input list spells them.  The engine does
//! not canonicalize: provenance strings (`<path>:<index>`) must match what
//! the Python engine wrote, and Windows canonicalization would prepend the
//! `\\?\` verbatim prefix and change every reference in every receipt.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::{json, Value};

use crate::assemble::{assemble_grib, DecodedCollection};
use crate::grib::{grib2_identities, validate_grib2_envelopes, wanted_indices, GribRecord};
use crate::model::Mapping;
use crate::node::Node;
use crate::refusal::{manifest_mismatch, missing_input, usage, Refusal, Result};

/// The parsed argv of one engine run.
#[derive(Debug, Clone, Default)]
pub struct Invocation {
    pub subcommand: String,
    pub mapping: String,
    pub input_list: String,
    pub output: Option<String>,
    pub composition: Option<String>,
    pub supplements: Vec<(String, String)>,
    pub provenance: Vec<(String, String)>,
    pub contributing_mappings: Vec<(String, String)>,
    pub input_manifest: Option<String>,
    pub input_manifest_sha256: Option<String>,
}

pub const USAGE: &str = "usage: gpuwm_mapped_engine {decode|compose|inspect} \
--mapping MAPPING.json --input-list FILES.txt --output DIR \
[--composition COMPOSITION.json] [--supplement ROLE=PATH]... \
[--provenance ROLE=PATH]... [--contributing-mapping ROLE=PATH]... \
[--input-manifest MANIFEST.json --input-manifest-sha256 HEX]\n\
   or: gpuwm_mapped_engine inventory --input-list FILES.txt\n\
   or: gpuwm_mapped_engine capabilities";

/// `capabilities`: what THIS build implements, as one JSON object.
///
/// The port is arriving in stages, so "which paths does the engine
/// decode?" is a question about the binary in hand, not about the
/// release notes.  gpuwm routes a bare run on the answer, and its
/// `ENGINE_CAPABILITIES` table is checked against this output — so a
/// build that gains `compose` is picked up by the route the moment it
/// says so here, and a table that drifts from the binary fails a test
/// instead of misrouting a user.
pub fn run_capabilities() -> Value {
    json!({
        "schema": crate::CAPABILITIES_SCHEMA,
        "engine": {"name": crate::ENGINE_NAME, "version": crate::ENGINE_VERSION},
        "frameset_schema": crate::FRAMESET_SCHEMA,
        // Per subcommand, the mapped source formats it decodes in
        // process.  An empty list means the subcommand is declared by
        // the contract and refuses `not_implemented` in this build.
        //
        // GRIB1 joined GRIB2 when `crate::grib1` landed: the ERA5 1974
        // reference object's forty-two decoded arrays, its grid
        // fingerprint and its materialization refusal are byte-identical
        // to the Python engine's through this binary, so a bare run of a
        // GRIB1 mapping decodes here rather than routing back to Python.
        //
        // NetCDF joined on the corpus fix.  It was held back because the
        // engine read one hand-made file and misread the corpus: the
        // vendored HDF5 reader enumerated ONE variable out of a
        // `netCDF4.Dataset(path, "w")` file, so a latitude selector
        // matched nothing.  Both causes are now named and fixed -- the
        // rw_wps workspace was linking the STOCK crates.io hdf5-reader
        // instead of the hardened vendored copy, and NetCDF-4 coordinate
        // variables are HDF5 dimension scales that netcrust's variable
        // index omits.  The evidence for the declaration is not the one
        // golden: it is the whole Python NetCDF suite green under
        // GPUWM_MAPPED_ENGINE=rust, on the same fixtures the Python
        // engine passes.
        //
        // `compose` joined on the parity evidence: every registered
        // composed source with staged bytes reproduces the Python
        // engine's composed answer byte for byte through this binary --
        // the frames, the alignment receipt across all three terrain
        // clock rules, and the per-binding contributing-source records
        // across both cross-source shapes.  It declares the same three
        // formats as `decode` and cannot outrun it: the manifest a
        // preparation seals asks the capability table about `decode`
        // while the composition asks about `compose`, so a format
        // declared for one and not the other would seal one decoder
        // inventory and verify against another.
        // `inventory` is the raw per-record product-identity surface: it
        // renders section octets and does not decode a field, so it is a
        // GRIB2 surface by construction -- the subprocess tool it
        // replaces (`grib2_inventory`) reads the same edition and nothing
        // else, and the archive-contract gates that consume it pin GRIB2
        // octet vocabulary (PDT, GDT, DRT).
        "subcommands": {
            "decode": ["grib1", "grib2", "netcdf"],
            "inspect": ["grib1", "grib2", "netcdf"],
            "compose": ["grib1", "grib2", "netcdf"],
            "inventory": ["grib2"],
        },
    })
}

impl Invocation {
    pub fn parse(arguments: &[String]) -> Result<Self> {
        let mut invocation = Invocation::default();
        let Some(subcommand) = arguments.first() else {
            return Err(usage("unknown or missing subcommand"));
        };
        if !matches!(
            subcommand.as_str(),
            "decode" | "compose" | "inspect" | "inventory" | "capabilities"
        ) {
            return Err(usage(format!("unknown subcommand '{subcommand}'")));
        }
        invocation.subcommand = subcommand.clone();
        // `capabilities` answers from the binary alone: it is what a
        // caller runs to find out WHICH of the others this build can do,
        // so requiring a mapping and an input list to ask would defeat
        // the question.
        if subcommand == "capabilities" {
            return Ok(invocation);
        }
        let mut position = 1usize;
        while position < arguments.len() {
            let flag = arguments[position].as_str();
            let value = || -> Result<String> {
                arguments
                    .get(position + 1)
                    .cloned()
                    .ok_or_else(|| usage(format!("{flag} needs a value")))
            };
            match flag {
                "--mapping" => invocation.mapping = value()?,
                "--input-list" => invocation.input_list = value()?,
                "--output" => invocation.output = Some(value()?),
                "--composition" => invocation.composition = Some(value()?),
                "--input-manifest" => invocation.input_manifest = Some(value()?),
                "--input-manifest-sha256" => invocation.input_manifest_sha256 = Some(value()?),
                "--supplement" | "--provenance" | "--contributing-mapping" => {
                    let binding = value()?;
                    let (role, path) = binding.split_once('=').ok_or_else(|| {
                        usage(format!("{flag} takes ROLE=PATH; got '{binding}'"))
                    })?;
                    let entry = (role.to_owned(), path.to_owned());
                    match flag {
                        "--supplement" => invocation.supplements.push(entry),
                        "--provenance" => invocation.provenance.push(entry),
                        // The third role-bound binding a cross-source
                        // composition carries: each contributing
                        // source's own sealed mapping document, whose
                        // bytes the composition pins by SHA-256.
                        // `compose` resolves every `field_sources`
                        // binding through it.
                        _ => invocation.contributing_mappings.push(entry),
                    }
                }
                other => return Err(usage(format!("unknown option '{other}'"))),
            }
            position += 2;
        }
        // `inventory` reads raw section octets; there is no mapping to
        // resolve selectors against and no frameset to write, so it takes
        // the input list alone -- demanding a mapping would make a
        // product-identity question depend on a document it never reads.
        if invocation.mapping.is_empty() && invocation.subcommand != "inventory" {
            return Err(usage("--mapping is required"));
        }
        if invocation.subcommand == "inventory" && !invocation.mapping.is_empty() {
            return Err(usage(
                "inventory reads raw record identity and takes no --mapping",
            ));
        }
        if invocation.input_list.is_empty() {
            return Err(usage("--input-list is required"));
        }
        if !matches!(invocation.subcommand.as_str(), "inspect" | "inventory")
            && invocation.output.is_none()
        {
            return Err(usage("--output is required for decode and compose"));
        }
        if invocation.input_manifest.is_some() != invocation.input_manifest_sha256.is_some() {
            return Err(usage(
                "--input-manifest and --input-manifest-sha256 are an atomic pair",
            ));
        }
        Ok(invocation)
    }
}

/// `mapped_source.read_input_list`: one UTF-8 path per line, blank lines
/// dropped, duplicates refused.
pub fn read_input_list(path: &str) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| missing_input(format!("cannot read input list {path}: {error}")))?;
    let mut files: Vec<String> = Vec::new();
    for line in text.lines() {
        // Verbatim, exactly as Python's `read_input_list` takes it: a line
        // is skipped when it is whitespace-only, and otherwise used AS
        // WRITTEN.  Trimming the surviving lines would be a divergence, not
        // a courtesy — the path is caller data, and the two engines must
        // open the same bytes.  (Both refuse identically on a list written
        // with a UTF-8 BOM, because both keep the BOM on the first line.)
        if line.trim().is_empty() {
            continue;
        }
        files.push(line.to_owned());
    }
    if files.is_empty() {
        return Err(missing_input(format!(
            "input list {path} names no files"
        )));
    }
    let mut seen = std::collections::BTreeSet::new();
    for entry in &files {
        if !seen.insert(entry.clone()) {
            return Err(usage(format!(
                "mapped source input list contains duplicates: {entry}"
            )));
        }
    }
    for entry in &files {
        if !PathBuf::from(entry).is_file() {
            return Err(missing_input(format!("no such input file: {entry}")));
        }
    }
    Ok(files)
}

/// `mapped_composition.INPUT_MANIFEST_SCHEMA`: the composition manifest,
/// which seals a mapping, a composition and every role-bound file.
pub const COMPOSITION_MANIFEST_SCHEMA: &str = "gpuwm-mapped-composition-inputs-v1";

/// `mapped_source._verify_input_manifest`, as far as the engine can see it:
/// the manifest's own digest, its schema, its mapping binding, and its
/// per-file identity.  Python still owns the authority-window recheck.
pub fn verify_input_manifest(
    manifest_path: &str,
    expected_sha256: &str,
    mapping: &Mapping,
    files: &[String],
    input_sha256: &BTreeMap<String, String>,
) -> Result<()> {
    let bytes = std::fs::read(manifest_path).map_err(|error| {
        missing_input(format!("cannot read input manifest {manifest_path}: {error}"))
    })?;
    let observed = crate::digest::bytes_sha256(&bytes);
    if observed != expected_sha256.to_ascii_lowercase() {
        return Err(manifest_mismatch(format!(
            "mapped input-manifest SHA mismatch: expected {expected_sha256}, got {observed}"
        )));
    }
    let document = Node::parse(&bytes).map_err(|error| {
        manifest_mismatch(format!("input manifest {manifest_path} is not JSON: {error}"))
    })?;
    let schema = document.get("schema").and_then(Node::as_str).unwrap_or("");
    if schema != "gpuwm-mapped-source-inputs-v1" {
        return Err(manifest_mismatch(format!(
            "unsupported mapped input manifest schema '{schema}'"
        )));
    }
    if document.get("mapping_sha256").and_then(Node::as_str) != Some(mapping.sha256.as_str()) {
        return Err(manifest_mismatch(
            "input manifest mapping SHA does not match mapping bytes",
        ));
    }
    let rows = document.get("files").map(Node::items).unwrap_or(&[]);
    if rows.len() != files.len() {
        return Err(manifest_mismatch(
            "input manifest file inventory differs from request",
        ));
    }
    for (index, (row, source)) in rows.iter().zip(files.iter()).enumerate() {
        let declared_bytes = row.get("bytes").and_then(Node::as_i64);
        let declared_sha = row.get("sha256").and_then(Node::as_str);
        let observed_size = std::fs::metadata(source)
            .map_err(|error| missing_input(format!("cannot stat {source}: {error}")))?
            .len() as i64;
        if declared_bytes != Some(observed_size) || declared_sha != input_sha256.get(source).map(String::as_str)
        {
            return Err(manifest_mismatch(format!(
                "input manifest identity differs for {source} (entry {index})"
            )));
        }
    }
    Ok(())
}

/// `mapped_composition._verify_manifest`, as far as the engine can see it.
///
/// The composition manifest is a DIFFERENT document from the decode
/// manifest above — schema `gpuwm-mapped-composition-inputs-v1`, with the
/// mapping and the composition both sealed, the primary inventory in
/// `primary_files`, and one row (or one list of rows) per declared
/// supplement, provenance and decoder role.  `_compose_through_engine`
/// already passes `--input-manifest`, so a compose that accepted the
/// argument and did not check it would report a seal it never read.
///
/// What is checked here is IDENTITY, exactly as the decode manifest is
/// checked: the manifest's own digest, the two contract digests against
/// the bytes this run loaded, and every named file's size and sha256
/// against the bytes this run is about to read.  The `path` strings are
/// deliberately not compared — they are written RELATIVE to the manifest
/// directory, and resolving them for comparison would canonicalize on
/// Windows and prepend the `\\?\` verbatim prefix this engine never
/// applies.  Python owns the path-equality half of the check and runs it
/// on the same manifest bytes moments before this call.
/// Returns the manifest's EXPLICIT ensemble-member binding when it
/// declares one: `(member, member_identity)`.  The pair is atomic --
/// gpuwm's `_verify_manifest` refuses a half-pair first, and this engine
/// refuses it again as defense for a hand-run exe -- and `compose` stamps
/// the member onto every canonical frame and into the alignment receipt,
/// which is how an archive whose product octets carry no ensemble
/// identity keeps the one its verified authority named.
#[allow(clippy::too_many_arguments)]
pub fn verify_composition_manifest(
    manifest_path: &str,
    expected_sha256: &str,
    mapping: &Mapping,
    composition_sha256: &str,
    primary: &[String],
    supplements: &BTreeMap<String, Vec<String>>,
    provenance: &BTreeMap<String, String>,
    digests: &BTreeMap<String, String>,
) -> Result<Option<(String, String)>> {
    let bytes = std::fs::read(manifest_path).map_err(|error| {
        missing_input(format!("cannot read input manifest {manifest_path}: {error}"))
    })?;
    let observed = crate::digest::bytes_sha256(&bytes);
    if observed != expected_sha256.to_ascii_lowercase() {
        return Err(manifest_mismatch(format!(
            "composition input-manifest SHA mismatch: expected {expected_sha256}, \
             got {observed}"
        )));
    }
    let document = Node::parse(&bytes).map_err(|error| {
        manifest_mismatch(format!("input manifest {manifest_path} is not JSON: {error}"))
    })?;
    let schema = document.get("schema").and_then(Node::as_str).unwrap_or("");
    if schema != COMPOSITION_MANIFEST_SCHEMA {
        return Err(manifest_mismatch(format!(
            "unsupported composition input manifest schema '{schema}'"
        )));
    }
    if document.get("mapping_sha256").and_then(Node::as_str) != Some(mapping.sha256.as_str()) {
        return Err(manifest_mismatch(
            "input manifest mapping SHA does not match mapping bytes",
        ));
    }
    if document.get("composition_sha256").and_then(Node::as_str) != Some(composition_sha256) {
        return Err(manifest_mismatch(
            "input manifest composition SHA does not match composition bytes",
        ));
    }
    let rows = document.get("primary_files").map(Node::items).unwrap_or(&[]);
    if rows.len() != primary.len() {
        return Err(manifest_mismatch(
            "manifest primary file inventory differs from the request",
        ));
    }
    for (index, (row, source)) in rows.iter().zip(primary.iter()).enumerate() {
        verify_manifest_identity(row, &format!("manifest.primary_files[{index}]"), source, digests)?;
    }
    let declared_roles = |section: &str| -> Vec<String> {
        document
            .get(section)
            .map(|node| {
                node.entries()
                    .iter()
                    .map(|(role, _)| role.clone())
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut supplement_roles = declared_roles("supplements");
    supplement_roles.sort();
    let requested: Vec<String> = supplements.keys().cloned().collect();
    if supplement_roles != requested {
        return Err(manifest_mismatch(format!(
            "manifest supplement role inventory {} differs from the request {}",
            crate::refusal::python_list_repr(&supplement_roles),
            crate::refusal::python_list_repr(&requested)
        )));
    }
    for (role, paths) in supplements {
        let node = document
            .get("supplements")
            .and_then(|section| section.get(role))
            .ok_or_else(|| manifest_mismatch(format!("manifest.supplements.{role} is absent")))?;
        // A single-file role may be written as ONE row rather than a
        // list of one; `_manifest_file_inventory` accepts both spellings
        // and manifests written before the list spelling carry the bare
        // row, so refusing it here would refuse a manifest Python
        // verifies.
        let rows: Vec<&Node> = if node.is_array() {
            node.items().iter().collect()
        } else {
            vec![node]
        };
        if rows.len() != paths.len() {
            return Err(manifest_mismatch(format!(
                "manifest.supplements.{role} file inventory differs from the request"
            )));
        }
        for (index, (row, source)) in rows.iter().zip(paths.iter()).enumerate() {
            verify_manifest_identity(
                row,
                &format!("manifest.supplements.{role}[{index}]"),
                source,
                digests,
            )?;
        }
    }
    let mut provenance_roles = declared_roles("provenance");
    provenance_roles.sort();
    let requested: Vec<String> = provenance.keys().cloned().collect();
    if provenance_roles != requested {
        return Err(manifest_mismatch(format!(
            "manifest provenance role inventory {} differs from the request {}",
            crate::refusal::python_list_repr(&provenance_roles),
            crate::refusal::python_list_repr(&requested)
        )));
    }
    for (role, source) in provenance {
        let row = document
            .get("provenance")
            .and_then(|section| section.get(role))
            .ok_or_else(|| manifest_mismatch(format!("manifest.provenance.{role} is absent")))?;
        verify_manifest_identity(row, &format!("manifest.provenance.{role}"), source, digests)?;
    }
    // The decoder section seals WHICH BINARY read the bytes.  This
    // engine decodes in process, so the run it seals has exactly one
    // decoder row and that row is this engine; a manifest naming the
    // subprocess pair was sealed for a decoder that is not running, and
    // replaying it here would produce evidence naming a binary that
    // never opened the file.
    let decoder_roles = declared_roles("decoders");
    if decoder_roles != vec![crate::ENGINE_NAME.to_owned()] {
        return Err(manifest_mismatch(format!(
            "manifest decoder inventory {} was sealed for a different decoder; \
             this run decodes in process as {}",
            crate::refusal::python_list_repr(&decoder_roles),
            crate::ENGINE_NAME
        )));
    }
    let member = document.get("member").and_then(Node::as_str);
    let member_identity = document.get("member_identity").and_then(Node::as_str);
    match (member, member_identity) {
        (None, None) => Ok(None),
        (Some(member), Some(identity))
            if !member.trim().is_empty() && !identity.trim().is_empty() =>
        {
            Ok(Some((member.to_owned(), identity.to_owned())))
        }
        _ => Err(manifest_mismatch(
            "manifest member and member_identity are an atomic pair of \
             non-empty strings",
        )),
    }
}

/// One `{path, bytes, sha256}` manifest row against the bytes in hand.
fn verify_manifest_identity(
    row: &Node,
    label: &str,
    source: &str,
    digests: &BTreeMap<String, String>,
) -> Result<()> {
    let declared_bytes = row.get("bytes").and_then(Node::as_i64);
    let declared_sha = row.get("sha256").and_then(Node::as_str);
    let observed_size = std::fs::metadata(source)
        .map_err(|error| missing_input(format!("cannot stat {source}: {error}")))?
        .len() as i64;
    if declared_bytes != Some(observed_size) || declared_sha != digests.get(source).map(String::as_str)
    {
        return Err(manifest_mismatch(format!(
            "input manifest identity differs for {source} ({label})"
        )));
    }
    Ok(())
}

/// The digests every frameset and inspection carries for its inputs.
///
/// Hashed concurrently: one file per slot, drained in document order.
/// The answer is a map keyed by path, so the schedule cannot reach it
/// even in principle, and the refusal for an unreadable file is still
/// the first one in the input list.
pub fn input_digests(files: &[String]) -> Result<BTreeMap<String, String>> {
    let slots: Vec<Result<(String, String)>> = crate::threads::install(|| {
        use rayon::prelude::*;
        files
            .par_iter()
            .map(|path| {
                crate::digest::file_sha256(path)
                    .map(|digest| (path.clone(), digest))
                    .map_err(|error| missing_input(format!("cannot hash {path}: {error}")))
            })
            .collect()
    });
    Ok(crate::threads::in_order(slots)?.into_iter().collect())
}

/// `mapped_source._decode_grib` / `_decode_netcdf`, dispatched by format.
pub fn decode_collection(
    mapping: &Mapping,
    files: &[String],
    progress: &mut dyn FnMut(Value),
) -> Result<DecodedCollection> {
    let format = mapping.format()?.to_owned();
    if format == "netcdf" {
        progress(json!({"event": "decode_netcdf", "files": files.len()}));
        return crate::ncdf::decode_netcdf(mapping, files);
    }
    if format == "grib1" {
        return decode_grib1_collection(mapping, files, progress);
    }
    let declaration = mapping.grid_declaration()?;
    let mut records: Vec<GribRecord> = Vec::new();
    let mut inventoried = 0usize;
    let mut identity_pins: Vec<crate::grib::RecordIdentity> = Vec::new();
    // One task per input object, into pre-assigned slots.  Files are
    // independent: each is read, staged through its acquisition codec,
    // parsed, inventoried and decoded on its own bytes, and the results
    // are concatenated in INPUT-LIST ORDER below -- record provenance
    // (`<path>:<index>`) and every later assembly step read that order,
    // so it is reproduced by the slot index rather than by the schedule.
    let outcomes: Vec<FileOutcome> = crate::threads::install(|| {
        use rayon::prelude::*;
        files
            .par_iter()
            .map(|source| decode_one_object(mapping, source, &declaration))
            .collect()
    });
    // Drained in order, so both the progress stream and the refusal are
    // exactly the serial engine's: an object that fails BEFORE it is
    // inventoried announces nothing (as the serial loop's `?` did), one
    // that fails after announces its inventory first.
    for (source, outcome) in files.iter().zip(outcomes) {
        let (identities, selected, decoded) = match outcome {
            FileOutcome::Unopened(refusal) => return Err(refusal),
            FileOutcome::Inventoried {
                identities,
                selected,
                records,
            } => (identities, selected, records),
        };
        inventoried += identities.len();
        progress(json!({
            "event": "inventory",
            "source": source,
            "messages": identities.len(),
            "selected": selected,
        }));
        identity_pins.extend(identities);
        records.extend(decoded?);
    }
    if records.is_empty() {
        // A total miss earns the identity diagnosis: two products under one
        // filename, separable only by the section-1 octets, must refuse by
        // naming them.
        return Err(crate::refusal::selector_unmatched(
            selector_identity_refusal(mapping, &identity_pins, files, inventoried)?,
        ));
    }
    assemble_grib(mapping, &records)
}

/// `mapped_source._decode_grib`'s edition-1 arm.
///
/// Deliberately NOT the GRIB2 arm above, and the differences are the
/// Python engine's, not simplifications:
///
///   * every usable message becomes a record, with no selector pre-filter.
///     `_grib1_records` hands `_assemble_grib` the whole object and lets
///     assembly do the matching, so an object none of whose messages match
///     earns assembly's "no GRIB messages match the mapping selectors"
///     rather than the GRIB2 arm's producer-identity diagnosis (which
///     reads section-1 octets edition 1 does not carry);
///   * there is no grid DECLARATION cross-check.  A GRIB1 mapping declares
///     `embedded_grid`; the lambert declaration path is GRIB2-only.
///
/// Objects are decoded one at a time, in input-list order — the messages
/// inside one object are what the work is parallel over — so record
/// provenance (`<path>:<index>`) and the first refusal are the serial
/// engine's by construction.
fn decode_grib1_collection(
    mapping: &Mapping,
    files: &[String],
    progress: &mut dyn FnMut(Value),
) -> Result<DecodedCollection> {
    let mut records: Vec<GribRecord> = Vec::new();
    for source in files {
        let raw = std::fs::read(source)
            .map_err(|error| missing_input(format!("cannot read {source}: {error}")))?;
        // Acquisition codec staging, as on the GRIB2 arm: the compressed
        // object is what the caller hashed, the decompressed twin is what
        // the parser reads, and provenance stays bound to the supplied path.
        let payload = crate::codec::decoded_payload(raw, source)?;
        let (messages, decoded) = crate::grib1::grib1_records(&payload, source)?;
        progress(json!({
            "event": "inventory",
            "source": source,
            "messages": messages,
            "selected": decoded.len(),
        }));
        records.extend(decoded);
    }
    assemble_grib(mapping, &records)
}

/// What one input object's decode task produced.
///
/// Two shapes, because the serial loop had two: an object that failed
/// before its inventory line was printed announced nothing, and an
/// object that failed after it had already announced.  Reproducing that
/// split is what keeps the progress stream identical under threads.
enum FileOutcome {
    /// Failed at read, codec staging, envelope hygiene, parse or
    /// selection — before anything was announced for this object.
    Unopened(Refusal),
    /// Inventoried.  The records either decoded or refused.
    Inventoried {
        identities: Vec<crate::grib::RecordIdentity>,
        selected: usize,
        records: Result<Vec<GribRecord>>,
    },
}

/// Read, stage, parse, inventory and decode ONE input object.
fn decode_one_object(
    mapping: &Mapping,
    source: &str,
    declaration: &crate::model::GridDeclaration,
) -> FileOutcome {
    let opened = (|| -> Result<(grib_core::grib2::Grib2File, Vec<crate::grib::RecordIdentity>, Vec<usize>)> {
        let raw = std::fs::read(source)
            .map_err(|error| missing_input(format!("cannot read {source}: {error}")))?;
        // Acquisition codec staging: the compressed object is what the
        // caller hashed, the decompressed twin is what the parser reads.
        let payload = crate::codec::decoded_payload(raw, source)?;
        validate_grib2_envelopes(&payload, source)?;
        let file = grib_core::grib2::Grib2File::from_bytes(&payload).map_err(|error| {
            crate::refusal::decode_failed(format!("GRIB2 parse failed for {source}: {error}"))
        })?;
        if file.messages.is_empty() {
            return Err(crate::refusal::decode_failed(format!(
                "GRIB2 input {source} contains no parsed fields"
            )));
        }
        let identities = grib2_identities(&file.messages);
        let wanted = wanted_indices(mapping, &identities)?;
        Ok((file, identities, wanted))
    })();
    match opened {
        Err(refusal) => FileOutcome::Unopened(refusal),
        Ok((file, identities, wanted)) => FileOutcome::Inventoried {
            selected: wanted.len(),
            identities,
            // The object is parsed ONCE and the selected messages are
            // decoded from it.  The serial path parsed the same bytes a
            // second time inside `grib2_records`, holding two parsed
            // copies of a half-gigabyte object at the peak.
            records: crate::grib::grib2_records(&file, source, &wanted, declaration),
        },
    }
}

/// `mapped_source._selector_identity_refusal`.
fn selector_identity_refusal(
    mapping: &Mapping,
    identities: &[crate::grib::RecordIdentity],
    files: &[String],
    inventoried: usize,
) -> Result<String> {
    let keys = [
        "center",
        "subcenter",
        "master_table_version",
        "local_table_version",
    ];
    let mut pins: BTreeMap<&str, std::collections::BTreeSet<i64>> = BTreeMap::new();
    for field in mapping.fields()? {
        for selector in field.selectors() {
            for key in keys {
                if let Some(value) = selector.field(key).and_then(Node::as_i64) {
                    pins.entry(key).or_default().insert(value);
                }
            }
        }
    }
    let mut mismatched: Vec<String> = Vec::new();
    for key in keys {
        let Some(pinned) = pins.get(key).filter(|values| values.len() == 1) else {
            continue;
        };
        let observed: std::collections::BTreeSet<i64> = identities
            .iter()
            .filter_map(|identity| match key {
                "center" => identity.center,
                "subcenter" => identity.subcenter,
                "master_table_version" => identity.master_table_version,
                _ => identity.local_table_version,
            })
            .collect();
        if !observed.is_empty() && observed.intersection(pinned).count() == 0 {
            let seen: Vec<String> = observed.iter().map(i64::to_string).collect();
            mismatched.push(format!(
                "every mapping selector pins {key}={} but every supplied message \
                 observes {key}={}",
                pinned.iter().next().expect("one pinned value"),
                seen.join("/")
            ));
        }
    }
    let names: Vec<&str> = files
        .iter()
        .map(|path| {
            path.rsplit(['/', '\\'])
                .next()
                .unwrap_or(path.as_str())
        })
        .collect();
    let base = format!(
        "0 of {inventoried} GRIB message(s) in {} match this mapping's selectors",
        names.join(", ")
    );
    if mismatched.is_empty() {
        return Ok(base);
    }
    Ok(format!(
        "{base}; the producer-identity octets explain it: {} -- these bytes are \
         a DIFFERENT product line published under the same file naming, and \
         decoding them here would silently mix model versions; the profile's \
         provenance document names the front door that serves the pinned identity",
        mismatched.join("; ")
    ))
}

/// `decode`: one mapping, N inputs, one frameset directory.
pub fn run_decode(invocation: &Invocation, progress: &mut dyn FnMut(Value)) -> Result<Value> {
    let mapping = Mapping::load(&invocation.mapping)?;
    let files = read_input_list(&invocation.input_list)?;
    let digests = input_digests(&files)?;
    if let (Some(manifest), Some(expected)) = (
        invocation.input_manifest.as_ref(),
        invocation.input_manifest_sha256.as_ref(),
    ) {
        verify_input_manifest(manifest, expected, &mapping, &files, &digests)?;
    }
    let collection = decode_collection(&mapping, &files, progress)?;
    progress(json!({"event": "assembled", "valid_times": collection.source_cycles.len()}));
    let frames = crate::frames::materialize_frames(&mapping, &collection)?;
    let output = PathBuf::from(invocation.output.as_ref().expect("decode requires --output"));
    let document =
        crate::frames::write_frameset(&output, &mapping, &collection, &frames, &digests)?;
    Ok(json!({
        "event": "receipt",
        "subcommand": "decode",
        "schema": crate::FRAMESET_SCHEMA,
        "frames": frames.len(),
        "output": output.display().to_string(),
        "grid_fingerprint": collection.grid_fingerprint,
        "stream_bytes": document
            .get("stream")
            .and_then(|stream| stream.get("bytes"))
            .cloned()
            .unwrap_or(Value::Null),
    }))
}

/// `inspect`: the `gpuwm-mapped-source-inspection-v1` document.
pub fn run_inspect(invocation: &Invocation, progress: &mut dyn FnMut(Value)) -> Result<Value> {
    let mapping = Mapping::load(&invocation.mapping)?;
    let files = read_input_list(&invocation.input_list)?;
    let digests = input_digests(&files)?;
    if let (Some(manifest), Some(expected)) = (
        invocation.input_manifest.as_ref(),
        invocation.input_manifest_sha256.as_ref(),
    ) {
        verify_input_manifest(manifest, expected, &mapping, &files, &digests)?;
    }
    let collection = decode_collection(&mapping, &files, progress)?;

    let mut direct_names: Vec<String> = Vec::new();
    for field in mapping.fields()? {
        if field.derivation().is_none() {
            direct_names.push(field.name.clone());
        }
    }
    direct_names.sort();
    let mut frame_rows = Vec::new();
    for (valid_time, member) in collection.source_cycles.keys() {
        let mut decoded: BTreeMap<&str, &crate::assemble::DirectValue> = BTreeMap::new();
        for ((time, member_value, name), value) in &collection.direct {
            if time == valid_time && member_value == member {
                decoded.insert(name.as_str(), value);
            }
        }
        let unresolved: Vec<&String> = direct_names
            .iter()
            .filter(|name| !decoded.contains_key(name.as_str()))
            .collect();
        // One entry per decoded field, measured CONCURRENTLY: the
        // sha256 and the extrema are per-field work over disjoint
        // arrays.  The indexed collect keeps entry i with field i, and
        // the map is then filled in the same sorted order the serial
        // loop used.  The extrema stream over the field rather than
        // collecting its finite cells into a second array first -- on a
        // 3-km frameset that copy was a gigabyte per field for two
        // numbers.
        let rows: Vec<(&str, Value)> = crate::threads::install(|| {
            use rayon::prelude::*;
            decoded
                .iter()
                .map(|(name, value)| (*name, *value))
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(name, value)| {
                    let flat = crate::array::contiguous(&value.values);
                    (
                        name,
                        json!({
                            "axes": value.axes,
                            "shape": value.values.shape(),
                            "minimum": flat
                                .iter()
                                .copied()
                                .filter(|item| item.is_finite())
                                .reduce(f64::min),
                            "maximum": flat
                                .iter()
                                .copied()
                                .filter(|item| item.is_finite())
                                .reduce(f64::max),
                            "missing": value.missing_count,
                            "sha256": crate::digest::array_sha256(value.values.shape(), &flat),
                            "source_references": value.references,
                        }),
                    )
                })
                .collect()
        });
        let mut fields = serde_json::Map::new();
        for (name, row) in rows {
            fields.insert(name.to_owned(), row);
        }
        frame_rows.push(json!({
            "valid_time": crate::frames::naive_isoformat(*valid_time),
            "source_cycle": crate::frames::naive_isoformat(
                collection.source_cycles[&(*valid_time, member.clone())]
            ),
            "member": member,
            "decoded_direct_fields": decoded.keys().collect::<Vec<_>>(),
            "unresolved_direct_fields": unresolved,
            "fields": Value::Object(fields),
        }));
    }

    let materialization = match crate::frames::materialize_frames(&mapping, &collection) {
        Ok(frames) => json!({
            "verdict": "PASS",
            "frame_count": frames.len(),
            "frame_header_sha256": frames
                .iter()
                .map(|frame| crate::digest::bytes_sha256(
                    canonical_json(&frame.header).as_bytes()
                ))
                .collect::<Vec<String>>(),
        }),
        Err(refusal) => json!({
            "verdict": "INCOMPLETE",
            "error_class": refusal.class,
            "error": refusal.message,
        }),
    };
    let status = if materialization["verdict"] == "PASS" {
        "CANONICAL_FRAMES_MATERIALIZED_NOT_STOCK_WRF_CERTIFIED"
    } else {
        "DECODED_INCOMPLETE_NOT_STOCK_WRF_CERTIFIED"
    };
    Ok(json!({
        "schema": crate::INSPECTION_SCHEMA,
        "status": status,
        "stock_wrf_certified": false,
        "mapping": {"path": mapping.path, "sha256": mapping.sha256},
        "inputs": files
            .iter()
            .map(|path| json!({
                "path": path,
                "bytes": std::fs::metadata(path).map(|data| data.len()).unwrap_or(0),
                "sha256": digests[path],
            }))
            .collect::<Vec<Value>>(),
        "decoders": {"engine": {"name": crate::ENGINE_NAME, "version": crate::ENGINE_VERSION}},
        "source_format": mapping.format()?,
        "grid": {
            "ny": collection.latitude.len(),
            "nx": collection.longitude.len(),
            "vertical_count": collection.vertical_values.len(),
            "fingerprint": collection.grid_fingerprint,
        },
        "frames": frame_rows,
        "materialization": materialization,
    }))
}

/// `inventory`: the raw per-record GRIB2 product identity of every input.
///
/// This is the engine's answer to the one question the subprocess
/// `grib2_inventory` used to be resolved for on a composed route: WHAT
/// product is in each file, octet for octet -- authority, process,
/// time semantics, level pair, member octet, grid definition, packing.
/// Nothing is decoded; identity sections only.
///
/// Every value is a STRING in the exact spelling the subprocess tool's
/// TSV rendered (`0x40` scan modes, `true`/`false` bitmaps, `-` for an
/// absent PDT octet, shortest-round-trip floats), so an archive-contract
/// gate that moved onto this surface reads one spelling whichever
/// instrument measured it, and the two instruments can be compared
/// column for column on the same bytes.
pub fn run_inventory(invocation: &Invocation) -> Result<Value> {
    use grib_core::grib2::{level_name, parameter_name};

    let files = read_input_list(&invocation.input_list)?;
    let digests = input_digests(&files)?;
    let mut file_rows: Vec<Value> = Vec::with_capacity(files.len());
    for source in &files {
        let raw = std::fs::read(source)
            .map_err(|error| missing_input(format!("cannot read {source}: {error}")))?;
        let delivered_bytes = raw.len();
        // Acquisition codec staging, exactly as decode stages it: the
        // delivered object is what the caller hashed, the decompressed
        // twin is what the parser reads.
        let payload = crate::codec::decoded_payload(raw, source)?;
        let envelopes = validate_grib2_envelopes(&payload, source)?;
        let file = grib_core::grib2::Grib2File::from_bytes(&payload).map_err(|error| {
            crate::refusal::decode_failed(format!("GRIB2 parse failed for {source}: {error}"))
        })?;
        if file.messages.is_empty() {
            return Err(crate::refusal::decode_failed(format!(
                "GRIB2 input {source} contains no parsed fields"
            )));
        }
        let mut records: Vec<Value> = Vec::with_capacity(file.messages.len());
        for (index, message) in file.messages.iter().enumerate() {
            let absent_or = |value: Option<u8>| {
                value.map_or_else(|| "-".to_owned(), |value| value.to_string())
            };
            records.push(json!({
                "index": index.to_string(),
                "discipline": message.discipline.to_string(),
                "category": message.product.parameter_category.to_string(),
                "parameter": message.product.parameter_number.to_string(),
                "center": message.identification.center_id.to_string(),
                "subcenter": message.identification.subcenter_id.to_string(),
                "master_table_version":
                    message.identification.master_table_version.to_string(),
                "local_table_version":
                    message.identification.local_table_version.to_string(),
                "name": parameter_name(
                    message.discipline,
                    message.product.parameter_category,
                    message.product.parameter_number,
                ),
                "reference_time": message.reference_time.to_string(),
                "forecast_unit": message.product.time_range_unit.to_string(),
                "forecast_time": message.product.forecast_time.to_string(),
                "pdt": message.product.template.to_string(),
                "level_type": message.product.level_type.to_string(),
                "level_value": message.product.level_value.to_string(),
                "second_level_type":
                    message.product.second_level_type.to_string(),
                "second_level_value":
                    message.product.second_level_value.to_string(),
                "member": absent_or(message.product.perturbation_number),
                "generating_process":
                    message.product.generating_process.to_string(),
                "forecast_generating_process_id":
                    message.product.forecast_generating_process_id.to_string(),
                "level_name": level_name(message.product.level_type),
                "gdt": message.grid.template.to_string(),
                "nx": message.grid.nx.to_string(),
                "ny": message.grid.ny.to_string(),
                "lat1": message.grid.lat1.to_string(),
                "lon1": message.grid.lon1.to_string(),
                "dx": message.grid.dx.to_string(),
                "dy": message.grid.dy.to_string(),
                "latin1": message.grid.latin1.to_string(),
                "latin2": message.grid.latin2.to_string(),
                "lov": message.grid.lov.to_string(),
                "scan_mode": format!("0x{:02x}", message.grid.scan_mode),
                "shape_of_earth": message.grid.shape_of_earth.to_string(),
                "resolution_flags":
                    format!("0x{:02x}", message.grid.resolution_flags),
                "drt": message.data_rep.template.to_string(),
                "bitmap": message.bitmap.is_some().to_string(),
                "ensemble_type": absent_or(message.product.ensemble_type),
                "ensemble_size":
                    absent_or(message.product.num_forecasts_in_ensemble),
                "derived_forecast":
                    absent_or(message.product.derived_forecast_type),
            }));
        }
        file_rows.push(json!({
            "path": source,
            "bytes": delivered_bytes,
            "sha256": digests[source],
            "envelopes": envelopes,
            "records": records,
        }));
    }
    Ok(json!({
        "schema": crate::RECORD_INVENTORY_SCHEMA,
        "engine": {"name": crate::ENGINE_NAME, "version": crate::ENGINE_VERSION},
        "files": file_rows,
    }))
}


/// `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)`.
pub fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(entries) => {
            let mut keys: Vec<&String> = entries.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .iter()
                .map(|key| {
                    format!(
                        "{}:{}",
                        Value::String((*key).clone()),
                        canonical_json(&entries[*key])
                    )
                })
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_pair_is_atomic() {
        let arguments: Vec<String> = [
            "decode",
            "--mapping",
            "m.json",
            "--input-list",
            "f.txt",
            "--output",
            "out",
            "--input-manifest",
            "mf.json",
        ]
        .iter()
        .map(|item| (*item).to_owned())
        .collect();
        let refusal = Invocation::parse(&arguments).unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::USAGE);
        assert!(refusal.message.contains("atomic pair"));
    }

    #[test]
    fn role_bindings_take_role_equals_path() {
        let arguments: Vec<String> = [
            "compose",
            "--mapping",
            "m.json",
            "--input-list",
            "f.txt",
            "--output",
            "out",
            "--supplement",
            "terrain=/tmp/t.nc",
            "--provenance",
            "orography=/tmp/o.json",
        ]
        .iter()
        .map(|item| (*item).to_owned())
        .collect();
        let invocation = Invocation::parse(&arguments).unwrap();
        assert_eq!(
            invocation.supplements,
            vec![("terrain".to_owned(), "/tmp/t.nc".to_owned())]
        );
        assert_eq!(
            invocation.provenance,
            vec![("orography".to_owned(), "/tmp/o.json".to_owned())]
        );
    }

    #[test]
    fn a_binding_without_an_equals_sign_refuses_with_the_grammar() {
        let arguments: Vec<String> = [
            "compose", "--mapping", "m.json", "--input-list", "f.txt", "--output", "out",
            "--supplement", "terrain",
        ]
        .iter()
        .map(|item| (*item).to_owned())
        .collect();
        let refusal = Invocation::parse(&arguments).unwrap_err();
        assert!(refusal.message.contains("ROLE=PATH"));
    }

    #[test]
    fn inspect_needs_no_output_directory() {
        let arguments: Vec<String> = ["inspect", "--mapping", "m.json", "--input-list", "f.txt"]
            .iter()
            .map(|item| (*item).to_owned())
            .collect();
        assert!(Invocation::parse(&arguments).is_ok());
    }

    #[test]
    fn inventory_takes_the_input_list_alone() {
        // Raw record identity has no mapping to resolve against and no
        // frameset to write; demanding either would make a
        // product-identity question depend on documents it never reads.
        let arguments: Vec<String> = ["inventory", "--input-list", "f.txt"]
            .iter()
            .map(|item| (*item).to_owned())
            .collect();
        assert!(Invocation::parse(&arguments).is_ok());
    }

    #[test]
    fn inventory_refuses_a_mapping() {
        let arguments: Vec<String> = [
            "inventory", "--input-list", "f.txt", "--mapping", "m.json",
        ]
        .iter()
        .map(|item| (*item).to_owned())
        .collect();
        let refusal = Invocation::parse(&arguments).unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::USAGE);
        assert!(refusal.message.contains("no --mapping"), "{refusal}");
    }

    #[test]
    fn the_capabilities_document_declares_the_inventory_surface() {
        let document = run_capabilities();
        assert_eq!(
            document["subcommands"]["inventory"],
            serde_json::json!(["grib2"])
        );
    }

    #[test]
    fn canonical_json_sorts_keys_and_drops_whitespace() {
        let value = json!({"b": 1, "a": [1, {"d": 2, "c": 3}]});
        assert_eq!(canonical_json(&value), r#"{"a":[1,{"c":3,"d":2}],"b":1}"#);
    }
}
