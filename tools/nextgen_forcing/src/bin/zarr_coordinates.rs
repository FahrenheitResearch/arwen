// SPDX-License-Identifier: Apache-2.0
//! Setup only: sealed local Zarr-v2 one-dimensional coordinates, never weather fields.
use arwen_nextgen_forcing::Result;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io::Read, path::PathBuf, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    metadata: PathBuf,
    metadata_sha256: String,
    source_store: String,
    chunks_root: PathBuf,
    output_dir: PathBuf,
    axes: Vec<Axis>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Axis {
    name: String,
    bounds: Bounds,
    chunk_sha256: Vec<String>,
    #[serde(default)]
    selections: Vec<Selection>,
}

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Bounds {
    None,
    MidpointExtrapolated,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    name: String,
    min_inclusive: f64,
    max_inclusive: f64,
}

#[derive(Debug)]
struct Decoded {
    name: String,
    centres: Vec<u8>,
    bounds: Option<Vec<u8>>,
    manifest: Value,
}

fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(message.into().into())
}
fn safe_name(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}
fn size(v: &Value, key: &str) -> Result<usize> {
    let a = v[key].as_array().ok_or_else(|| format!("missing {key}"))?;
    if a.len() != 1 {
        return fail(format!("{key} must have rank one"));
    }
    let n = a[0].as_u64().ok_or("invalid axis extent")?;
    if n == 0 || n > 1_000_000 {
        return fail("axis/chunk extent must be 1..1000000");
    }
    Ok(n as usize)
}

fn decode(request: &Request, metadata: &Value, axis: &Axis) -> Result<Decoded> {
    let array = &metadata[format!("{}/.zarray", axis.name)];
    let attrs = &metadata[format!("{}/.zattrs", axis.name)];
    if array["zarr_format"] != 2
        || array["dtype"] != "<f8"
        || array["order"] != "C"
        || array["compressor"]["id"] != "zstd"
        || !array["filters"].is_null()
        || !attrs.is_object()
        || attrs["_ARRAY_DIMENSIONS"] != json!([axis.name])
        || attrs["units"].as_str().filter(|s| !s.is_empty()).is_none()
    {
        return fail(format!(
            "{}: requires rank-one C-order <f8 Zarr2/Zstd, no filters, named dimension and units",
            axis.name
        ));
    }
    for (key, identity) in [("scale_factor", 1.0), ("add_offset", 0.0)] {
        if !attrs[key].is_null() && attrs[key].as_f64() != Some(identity) {
            return fail(format!(
                "{}: packed coordinate {key} is unsupported",
                axis.name
            ));
        }
    }
    if axis.bounds == Bounds::MidpointExtrapolated && !attrs["bounds"].is_null() {
        return fail(format!(
            "{}: declared provider bounds must be read instead of inferred",
            axis.name
        ));
    }
    let n = size(array, "shape")?;
    if n < 2 {
        return fail("at least two axis centres are required");
    }
    let chunk = size(array, "chunks")?;
    let chunks = n.div_ceil(chunk);
    if axis.chunk_sha256.len() != chunks {
        return fail(format!("{}: wrong chunk hash count", axis.name));
    }
    let mut centres = Vec::with_capacity(n * 8);
    let mut inputs = Vec::with_capacity(chunks);
    for i in 0..chunks {
        let path = request.chunks_root.join(&axis.name).join(i.to_string());
        if fs::metadata(&path)?.len() > (chunk * 8 + 65536) as u64 {
            return fail(format!("{}: oversized compressed chunk {i}", axis.name));
        }
        let compressed = fs::read(&path)?;
        let sha = hash(&compressed);
        if sha != axis.chunk_sha256[i] {
            return fail(format!("{}: chunk {i} SHA256 mismatch", axis.name));
        }
        let mut raw = Vec::new();
        zstd::stream::read::Decoder::new(compressed.as_slice())?
            .take((chunk * 8 + 1) as u64)
            .read_to_end(&mut raw)?;
        // Zarr-v2 edge chunks retain the declared chunk extent, including padding.
        if raw.len() != chunk * 8 {
            return fail(format!("{}: chunk {i} decoded extent differs", axis.name));
        }
        let count = chunk.min(n - i * chunk);
        centres.extend_from_slice(&raw[..count * 8]);
        inputs.push(json!({"key":format!("{}/{i}",axis.name), "sha256":sha,
            "compressed_bytes":compressed.len(), "decoded_bytes":raw.len(), "retained_values":count}));
    }
    let values: Vec<f64> = centres
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if values.iter().any(|v| {
        !v.is_finite()
            || ["_FillValue", "missing_value"]
                .iter()
                .any(|k| attrs[*k].as_f64() == Some(*v))
            || array["fill_value"].as_f64() == Some(*v)
    }) {
        return fail(format!("{}: nonfinite or missing coordinate", axis.name));
    }
    let ascending = values[1] > values[0];
    let mut min_spacing = f64::INFINITY;
    let mut max_spacing: f64 = 0.0;
    for w in values.windows(2) {
        let delta = w[1] - w[0];
        if !delta.is_finite()
            || (if ascending {
                delta <= 0.0
            } else {
                delta >= 0.0
            })
        {
            return fail(format!(
                "{}: coordinates must be strictly monotone",
                axis.name
            ));
        }
        min_spacing = min_spacing.min(delta.abs());
        max_spacing = max_spacing.max(delta.abs());
    }
    let mut selection_names = BTreeSet::new();
    let mut selected = Vec::new();
    for s in &axis.selections {
        if !safe_name(&s.name)
            || !selection_names.insert(&s.name)
            || !s.min_inclusive.is_finite()
            || !s.max_inclusive.is_finite()
            || s.min_inclusive > s.max_inclusive
        {
            return fail("invalid or duplicate coordinate selection");
        }
        let indices: Vec<usize> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| **v >= s.min_inclusive && **v <= s.max_inclusive)
            .map(|(i, _)| i)
            .collect();
        let first = *indices.first().ok_or("coordinate selection is empty")?;
        let last = *indices.last().unwrap();
        selected.push(
            json!({"name":s.name,"min_inclusive":s.min_inclusive,"max_inclusive":s.max_inclusive,
            "start_index":first,"end_index_exclusive":last+1,"count":indices.len(),
            "first":values[first],"last":values[last]}),
        );
    }
    let bounds = if axis.bounds == Bounds::MidpointExtrapolated {
        let mut edges = Vec::with_capacity(n + 1);
        edges.push(values[0] - (values[1] - values[0]) * 0.5);
        edges.extend(values.windows(2).map(|w| w[0] + (w[1] - w[0]) * 0.5));
        edges.push(values[n - 1] + (values[n - 1] - values[n - 2]) * 0.5);
        if edges.iter().any(|v| !v.is_finite()) {
            return fail("nonfinite inferred edge");
        }
        if edges.windows(2).any(|w| {
            if ascending {
                w[1] <= w[0]
            } else {
                w[1] >= w[0]
            }
        }) {
            return fail("inferred bounds collapse at coordinate precision");
        }
        Some(
            edges
                .windows(2)
                .flat_map(|w| [w[0], w[1]])
                .flat_map(f64::to_le_bytes)
                .collect::<Vec<u8>>(),
        )
    } else {
        None
    };
    let bounds_manifest = bounds.as_ref().map(|b| {
        json!({"path":format!("{}.midpoint-bounds.f64le",axis.name),
        "dtype":"<f8","shape":[n,2],"sha256":hash(b),"bytes":b.len(),
        "authority":"inferred_midpoints_and_half_spacing_outer_extrapolation",
        "provider_published":false,"edge_order":"same direction as centres; do not sort"})
    });
    let manifest = json!({"name":axis.name,"coordinates":{"path":format!("{}.f64le",axis.name),
        "dtype":"<f8","shape":[n],"sha256":hash(&centres),"bytes":centres.len(),
        "first":values[0],"last":values[n-1],"order":if ascending {"ascending"} else {"descending"},
        "min_absolute_spacing":min_spacing,"max_absolute_spacing":max_spacing},
        "bounds":bounds_manifest,"source_array":array,"source_attributes":attrs,
        "chunks":inputs,"selections":selected});
    Ok(Decoded {
        name: axis.name.clone(),
        centres,
        bounds,
        manifest,
    })
}

fn run() -> Result<()> {
    let started = Instant::now();
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 2 {
        return fail("usage: forcing-zarr-coordinates REQUEST.json");
    }
    let request: Request = serde_json::from_slice(&fs::read(&args[1])?)?;
    if request.schema != "nextgen.forcing.zarr-coordinates-request/v1"
        || request.axes.is_empty()
        || request.axes.len() > 16
        || request.source_store.is_empty()
    {
        return fail("invalid request schema/axes/source store");
    }
    let mut names = BTreeSet::new();
    for axis in &request.axes {
        if !safe_name(&axis.name) || !names.insert(&axis.name) {
            return fail("unsafe or duplicate axis name");
        }
    }
    if request.output_dir.exists() {
        return fail("output directory already exists; use a fresh path");
    }
    if fs::metadata(&request.metadata)?.len() > 16 * 1024 * 1024 {
        return fail("oversized metadata");
    }
    let bytes = fs::read(&request.metadata)?;
    if hash(&bytes) != request.metadata_sha256 {
        return fail("metadata SHA256 mismatch");
    }
    let consolidated: Value = serde_json::from_slice(&bytes)?;
    if consolidated["zarr_consolidated_format"] != 1 {
        return fail("requires consolidated Zarr-v2 metadata v1");
    }
    let metadata = &consolidated["metadata"];
    let decoded: Vec<Decoded> = request
        .axes
        .par_iter()
        .map(|axis| decode(&request, metadata, axis))
        .collect::<Result<Vec<_>>>()?;
    let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
    fs::create_dir(&request.output_dir)?;
    let mut total = 0;
    for axis in &decoded {
        fs::write(
            request.output_dir.join(format!("{}.f64le", axis.name)),
            &axis.centres,
        )?;
        total += axis.centres.len();
        if let Some(b) = &axis.bounds {
            fs::write(
                request
                    .output_dir
                    .join(format!("{}.midpoint-bounds.f64le", axis.name)),
                b,
            )?;
            total += b.len();
        }
    }
    let manifest = json!({"schema":"nextgen.forcing.zarr-coordinates/v1",
        "source_store":request.source_store,"metadata_sha256":request.metadata_sha256,
        "source_root_attributes":metadata[".zattrs"],"axes":decoded.iter().map(|a| &a.manifest).collect::<Vec<_>>(),
        "decode_ms":decode_ms,"rayon_threads":rayon::current_num_threads(),"logical_bytes":total,
        "scope":"coordinate setup only; no weather values, catchment weights, native sampling or CRS inferred"});
    let path = request.output_dir.join("manifest.json");
    fs::write(&path, serde_json::to_vec_pretty(&manifest)?)?;
    println!(
        "{}",
        json!({"manifest":path,"manifest_sha256":hash(&fs::read(&path)?),
        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"decode_ms":decode_ms,
        "rayon_threads":rayon::current_num_threads(),"logical_bytes":total})
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Fixture {
        request: Request,
        metadata: Value,
    }
    impl Fixture {
        fn new(values: [f64; 5]) -> Self {
            let root = std::env::temp_dir().join(format!(
                "forcing-coordinate-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            fs::create_dir(root.join("x")).unwrap();
            let mut chunks = Vec::new();
            // The final source chunk includes missing padding outside the shape.
            for (i, v) in [
                values[..4].to_vec(),
                vec![values[4], f64::NAN, f64::NAN, f64::NAN],
            ]
            .iter()
            .enumerate()
            {
                let raw: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
                let packed = zstd::stream::encode_all(raw.as_slice(), 3).unwrap();
                chunks.push(hash(&packed));
                fs::write(root.join("x").join(i.to_string()), packed).unwrap();
            }
            let axis = Axis {
                name: "x".into(),
                bounds: Bounds::MidpointExtrapolated,
                chunk_sha256: chunks,
                selections: vec![Selection {
                    name: "inside".into(),
                    min_inclusive: 3.0,
                    max_inclusive: 7.0,
                }],
            };
            let request = Request {
                schema: "unused".into(),
                metadata: root.join("metadata"),
                metadata_sha256: "unused".into(),
                source_store: "test".into(),
                chunks_root: root,
                output_dir: "unused".into(),
                axes: vec![axis],
            };
            let metadata = json!({"x/.zarray":{"zarr_format":2,"shape":[5],"chunks":[4],"dtype":"<f8",
                "order":"C","compressor":{"id":"zstd"},"filters":null},
                "x/.zattrs":{"_ARRAY_DIMENSIONS":["x"],"units":"m"}});
            Self { request, metadata }
        }
        fn decode(&self) -> Result<Decoded> {
            super::decode(&self.request, &self.metadata, &self.request.axes[0])
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.request.chunks_root).unwrap();
        }
    }

    #[test]
    fn source_order_padding_bounds_and_selection() {
        for (values, expected) in [
            (
                [1., 3., 5., 7., 9.],
                vec![0., 2., 2., 4., 4., 6., 6., 8., 8., 10.],
            ),
            (
                [9., 7., 5., 3., 1.],
                vec![10., 8., 8., 6., 6., 4., 4., 2., 2., 0.],
            ),
        ] {
            let f = Fixture::new(values);
            let d = f.decode().unwrap();
            assert_eq!(
                d.centres,
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            );
            let expected: Vec<u8> = expected
                .iter()
                .flat_map(|v: &f64| v.to_le_bytes())
                .collect();
            assert_eq!(d.bounds.unwrap(), expected);
            assert_eq!(d.manifest["selections"][0]["start_index"], 1);
            assert_eq!(d.manifest["selections"][0]["end_index_exclusive"], 4);
            assert_eq!(d.manifest["bounds"]["provider_published"], false);
        }
    }

    #[test]
    fn reject_hash_extent_nonmonotone_packing_and_declared_bounds() {
        let mut f = Fixture::new([1., 3., 5., 7., 9.]);
        f.request.axes[0].chunk_sha256[0] = "bad".into();
        assert!(f
            .decode()
            .unwrap_err()
            .to_string()
            .contains("SHA256 mismatch"));
        let mut f = Fixture::new([1., 3., 5., 7., 9.]);
        f.metadata["x/.zarray"]["chunks"] = json!([3]);
        assert!(f
            .decode()
            .unwrap_err()
            .to_string()
            .contains("decoded extent"));
        let f = Fixture::new([1., 3., 3., 7., 9.]);
        assert!(f
            .decode()
            .unwrap_err()
            .to_string()
            .contains("strictly monotone"));
        let mut f = Fixture::new([1., 3., 5., 7., 9.]);
        f.metadata["x/.zattrs"]["scale_factor"] = json!(0.1);
        assert!(f
            .decode()
            .unwrap_err()
            .to_string()
            .contains("packed coordinate"));
        let mut f = Fixture::new([1., 3., 5., 7., 9.]);
        f.metadata["x/.zattrs"]["bounds"] = json!("x_bounds");
        assert!(f
            .decode()
            .unwrap_err()
            .to_string()
            .contains("declared provider bounds"));
        let e = f64::EPSILON;
        let mut f = Fixture::new([1., 1. + e, 1. + 2. * e, 1. + 3. * e, 1. + 4. * e]);
        f.request.axes[0].selections.clear();
        assert!(f
            .decode()
            .unwrap_err()
            .to_string()
            .contains("bounds collapse"));
    }
}
