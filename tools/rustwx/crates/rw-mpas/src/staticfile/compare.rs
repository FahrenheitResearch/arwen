//! Grading a static against a known answer.
//!
//! The instrument, kept separate from the thing it measures. It reports every
//! shared variable's worst absolute and relative gap and the count of exactly
//! equal values, and it reports fields present in only one file rather than
//! quietly skipping them -- a comparison that silently ignores what it cannot
//! align is how a builder gets graded on the fields it happened to get right.
//!
//! There is no verdict in here on purpose. Two statics built from DIFFERENT
//! geography archives must differ in the geography group and must not differ in
//! the mesh group, and only a caller who knows which archives went in can say
//! which of those a number means. This module measures; the receipt states the
//! provenance beside it.

use std::path::Path;

use serde::Serialize;

use crate::error::{MpasError, MpasResult};

/// One variable's agreement.
#[derive(Debug, Clone, Serialize)]
pub struct FieldComparison {
    pub name: String,
    pub n: usize,
    /// Values that are bit-for-bit equal after both are read as `f64`.
    pub exact: usize,
    pub max_abs: f64,
    /// Worst |a-b| / max(|b|, tiny), with the reference magnitude beside it.
    pub max_rel: f64,
    pub max_rel_reference: f64,
    pub at_index: usize,
    pub mean_abs: f64,
    pub reference_min: f64,
    pub reference_max: f64,
    pub candidate_min: f64,
    pub candidate_max: f64,
}

/// The whole comparison.
#[derive(Debug, Clone, Serialize)]
pub struct StaticComparison {
    pub reference: String,
    pub candidate: String,
    pub reference_sha256: String,
    pub candidate_sha256: String,
    pub dimensions_agree: bool,
    pub reference_dimensions: Vec<(String, usize)>,
    pub candidate_dimensions: Vec<(String, usize)>,
    pub fields: Vec<FieldComparison>,
    pub only_in_reference: Vec<String>,
    pub only_in_candidate: Vec<String>,
    pub shape_mismatch: Vec<String>,
}

fn dims(f: &netcrust::File) -> Vec<(String, usize)> {
    let mut v: Vec<(String, usize)> = f
        .dimensions()
        .unwrap_or_default()
        .into_iter()
        .map(|d| (d.name().to_string(), d.len()))
        .collect();
    v.sort();
    v
}

fn names(f: &netcrust::File) -> Vec<String> {
    let mut v: Vec<String> = f
        .variables()
        .unwrap_or_default()
        .into_iter()
        .map(|x| x.name().to_string())
        .collect();
    v.sort();
    v
}

/// Compare `candidate` against `reference`, variable by variable.
pub fn compare(reference: &Path, candidate: &Path) -> MpasResult<StaticComparison> {
    let a = netcrust::File::open(reference)
        .map_err(|e| MpasError::Refusal(format!("{}: {e}", reference.display())))?;
    let b = netcrust::File::open(candidate)
        .map_err(|e| MpasError::Refusal(format!("{}: {e}", candidate.display())))?;

    let (da, db) = (dims(&a), dims(&b));
    let (na, nb) = (names(&a), names(&b));

    let mut fields = Vec::new();
    let mut only_a = Vec::new();
    let mut only_b = Vec::new();
    let mut shape_mismatch = Vec::new();

    for name in &na {
        if !nb.contains(name) {
            only_a.push(name.clone());
            continue;
        }
        let (va, vb) = match (a.read_f64(name), b.read_f64(name)) {
            (Ok(x), Ok(y)) => (x, y),
            // A variable a reader cannot turn into numbers (text, mostly) is
            // not a silent pass: it is reported as a shape mismatch so the
            // caller sees it was not graded.
            _ => {
                shape_mismatch.push(name.clone());
                continue;
            }
        };
        if va.len() != vb.len() {
            shape_mismatch.push(format!("{name} ({} vs {})", va.len(), vb.len()));
            continue;
        }
        let mut c = FieldComparison {
            name: name.clone(),
            n: va.len(),
            exact: 0,
            max_abs: 0.0,
            max_rel: 0.0,
            max_rel_reference: 0.0,
            at_index: 0,
            mean_abs: 0.0,
            reference_min: f64::INFINITY,
            reference_max: f64::NEG_INFINITY,
            candidate_min: f64::INFINITY,
            candidate_max: f64::NEG_INFINITY,
        };
        let mut total = 0.0f64;
        for i in 0..va.len() {
            let (x, y) = (va[i], vb[i]);
            c.reference_min = c.reference_min.min(x);
            c.reference_max = c.reference_max.max(x);
            c.candidate_min = c.candidate_min.min(y);
            c.candidate_max = c.candidate_max.max(y);
            if x.to_bits() == y.to_bits() {
                c.exact += 1;
                continue;
            }
            let d = (x - y).abs();
            total += d;
            if d > c.max_abs {
                c.max_abs = d;
                c.at_index = i;
            }
            let scale = x.abs().max(1e-30);
            let r = d / scale;
            if r > c.max_rel {
                c.max_rel = r;
                c.max_rel_reference = x;
            }
        }
        c.mean_abs = if va.is_empty() { 0.0 } else { total / va.len() as f64 };
        fields.push(c);
    }
    for name in &nb {
        if !na.contains(name) {
            only_b.push(name.clone());
        }
    }

    Ok(StaticComparison {
        reference: reference.display().to_string(),
        candidate: candidate.display().to_string(),
        reference_sha256: crate::sha256_file(reference)?,
        candidate_sha256: crate::sha256_file(candidate)?,
        dimensions_agree: da == db,
        reference_dimensions: da,
        candidate_dimensions: db,
        fields,
        only_in_reference: only_a,
        only_in_candidate: only_b,
        shape_mismatch,
    })
}
