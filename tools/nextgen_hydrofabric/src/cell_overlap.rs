//! Original Rust startup transpose of measured polygon/cell intersections.
//! Geometry bookkeeping only: no land-source or aquifer partition is selected.
use crate::{bundle::digest, manifest::PreparedManifest, Result};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, time::Instant};

#[derive(Deserialize)]
struct Config {
    fabric_manifest: String,
    fabric_sha256: String,
    destination_manifest: String,
    destination_sha256: String,
    output: String,
}
fn read(m: &PreparedManifest, name: &str, dtype: &str, count: usize) -> Result<Vec<u8>> {
    let f = m
        .document
        .files
        .get(name)
        .ok_or("missing overlap input field")?;
    if f.dtype != dtype || f.shape != [count] {
        return Err(format!("incorrect {name} type/shape").into());
    }
    let mut b = vec![0; f.bytes];
    m.read_field_into(name, &mut b)?;
    Ok(b)
}
fn f64s(b: &[u8]) -> Vec<f64> {
    b.par_chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn write(
    out: &Path,
    files: &mut BTreeMap<String, Value>,
    name: &str,
    dtype: &str,
    count: usize,
    b: Vec<u8>,
) -> Result<()> {
    let path = format!("{name}.bin");
    fs::write(out.join(&path), &b)?;
    files.insert(name.into(), json!({"path":path,"dtype":dtype,"shape":[count],"bytes":b.len(),"sha256":format!("{:x}",Sha256::digest(&b))}));
    Ok(())
}
pub fn prepare(config: &Path) -> Result<Value> {
    let started = Instant::now();
    let config_bytes = fs::read(config)?;
    let c: Config = serde_json::from_slice(&config_bytes)?;
    for (path, expected) in [
        (&c.fabric_manifest, &c.fabric_sha256),
        (&c.destination_manifest, &c.destination_sha256),
    ] {
        if digest(Path::new(path))? != *expected {
            return Err("cell overlap input manifest SHA256 mismatch".into());
        }
    }
    let m = PreparedManifest::open(Path::new(&c.fabric_manifest))?;
    let destination: Value = serde_json::from_slice(&fs::read(&c.destination_manifest)?)?;
    let grid = m
        .document
        .metadata
        .get("grid")
        .ok_or("fabric has no grid")?;
    if destination["schema"] != "arwen.nextgen.terrain-grid/v1"
        || destination["row_order"] != "north-first, x increasing east; y0 is southern grid edge"
        || grid["row_order"] != "north_to_south"
        || ["nx", "ny", "x0", "y0", "dx"]
            .iter()
            .any(|k| grid[k] != destination["grid"][k])
        || grid["dy"] != destination["grid"]["dx"]
        || grid["crs"].as_str()
            != Some(&format!(
                "EPSG:{}",
                destination["grid"]["epsg"]
                    .as_u64()
                    .ok_or("destination EPSG absent")?
            ))
    {
        return Err(
            "cell overlap requires exact destination grid/CRS/row order, including southern edge"
                .into(),
        );
    }
    let nx = usize::try_from(grid["nx"].as_u64().ok_or("invalid nx")?)?;
    let ny = usize::try_from(grid["ny"].as_u64().ok_or("invalid ny")?)?;
    let cells = nx.checked_mul(ny).ok_or("grid size overflow")?;
    let dx = grid["dx"].as_f64().ok_or("invalid dx")?;
    let area = dx * dx;
    if cells == 0 || !area.is_finite() || dx <= 0. || cells > i32::MAX as usize {
        return Err("invalid cell dimensions/area".into());
    }
    let cats = m.document.catchment_count;
    if cats == 0 || cats > i32::MAX as usize {
        return Err("catchment axis must be nonempty and fit i32".into());
    }
    let network_bytes = m.read_auxiliary("network.json")?;
    let network: Value = serde_json::from_slice(&network_bytes)?;
    let ids = network["catchment_ids"]
        .as_array()
        .ok_or("catchment roster absent")?;
    if ids.len() != cats {
        return Err("catchment roster length mismatch".into());
    }
    let offsets: Vec<u64> = read(
        &m,
        "weights_offsets",
        "<u8",
        cats.checked_add(1).ok_or("count overflow")?,
    )?
    .par_chunks_exact(8)
    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    .collect();
    if offsets.first() != Some(&0) || offsets.windows(2).any(|p| p[0] > p[1]) {
        return Err("invalid source CSR offsets".into());
    }
    let edges = usize::try_from(*offsets.last().ok_or("empty source CSR")?)?;
    i64::try_from(edges).map_err(|_| "overlap count exceeds signed CSR offsets")?;
    let indices: Vec<u32> = read(&m, "weights_indices", "<u4", edges)?
        .par_chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let overlaps = f64s(&read(&m, "overlap_m2", "<f8", edges)?);
    let cat_area = f64s(&read(&m, "catchment_area_m2", "<f8", cats)?);
    if indices.iter().any(|&i| i as usize >= cells)
        || overlaps.iter().any(|a| !a.is_finite() || *a <= 0.)
        || cat_area.iter().any(|a| !a.is_finite() || *a <= 0.)
    {
        return Err("invalid source index or intersection/catchment area".into());
    }
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let transpose = Instant::now();
    let parts: Result<Vec<Vec<(u32, i32, f64)>>> = (0..cats)
        .into_par_iter()
        .map(|cat| {
            let a = usize::try_from(offsets[cat])?;
            let b = usize::try_from(offsets[cat + 1])?;
            Ok((a..b)
                .map(|k| (indices[k], cat as i32, overlaps[k]))
                .collect())
        })
        .collect();
    let mut sorted: Vec<_> = parts?.into_iter().flatten().collect();
    sorted.par_sort_unstable_by_key(|&(cell, cat, _)| (cell, cat));
    if sorted
        .windows(2)
        .any(|p| (p[0].0, p[0].1) == (p[1].0, p[1].1))
    {
        return Err("duplicate source catchment/cell intersection".into());
    }
    let ranges: Vec<_> = (0..cells)
        .into_par_iter()
        .map(|cell| {
            let a = sorted.partition_point(|p| p.0 < cell as u32);
            let b = sorted.partition_point(|p| p.0 <= cell as u32);
            (a, b)
        })
        .collect();
    let covered: Vec<f64> = ranges
        .par_iter()
        // Canonical +0 for empty cells, matching the area accumulator.
        .map(|&(a, b)| sorted[a..b].iter().fold(0.0, |sum, p| sum + p.2))
        .collect();
    let fractions: Vec<f64> = covered.par_iter().map(|x| x / area).collect();
    let represented: Vec<f64> = (0..cats)
        .into_par_iter()
        .map(|cat| {
            overlaps[offsets[cat] as usize..offsets[cat + 1] as usize]
                .iter()
                .fold(0.0, |sum, area| sum + area)
        })
        .collect();
    let cat_fraction: Vec<f64> = represented
        .par_iter()
        .zip(&cat_area)
        .map(|(a, b)| a / b)
        .collect();
    let row_offsets: Vec<i64> = ranges
        .iter()
        .map(|p| p.0 as i64)
        .chain(std::iter::once(i64::try_from(edges)?))
        .collect();
    let row_indices: Vec<i32> = sorted.par_iter().map(|p| p.1).collect();
    let row_areas: Vec<f64> = sorted.par_iter().map(|p| p.2).collect();
    let weights: Vec<f64> = sorted.par_iter().map(|p| p.2 / area).collect();
    let weights32: Vec<f32> = weights.par_iter().map(|&x| x as f32).collect();
    if fractions
        .iter()
        .chain(&cat_fraction)
        .chain(&weights)
        .any(|x| !x.is_finite())
        || weights32.iter().any(|x| !x.is_finite())
    {
        return Err("overlap area normalization exceeds output precision".into());
    }
    let mask: Vec<i32> = covered.par_iter().map(|&a| i32::from(a > 0.)).collect();
    let transpose_ms = transpose.elapsed().as_secs_f64() * 1000.;
    let mut key = Sha256::new();
    key.update(&config_bytes);
    key.update(c.fabric_sha256.as_bytes());
    key.update(c.destination_sha256.as_bytes());
    key.update(crate::SOURCE_REVISION.as_bytes());
    let key = format!("{:x}", key.finalize());
    let out = Path::new(&c.output).join(&key[..20]);
    fs::create_dir_all(&out)?;
    let mut files = BTreeMap::new();
    let write_started = Instant::now();
    macro_rules! put {
        ($name:literal,$dtype:literal,$values:expr) => {
            write(
                &out,
                &mut files,
                $name,
                $dtype,
                $values.len(),
                $values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )?
        };
    }
    put!("cell_catchment_offsets", "<i8", row_offsets);
    put!("cell_catchment_indices", "<i4", row_indices);
    put!("cell_catchment_overlap_m2", "<f8", row_areas);
    put!("cell_catchment_weights", "<f8", weights);
    put!("cell_catchment_weights_f32", "<f4", weights32);
    put!("cell_covered_area_m2", "<f8", covered);
    put!("cell_coverage_fraction", "<f8", fractions);
    put!("cell_positive_overlap", "<i4", mask);
    put!("catchment_covered_area_m2", "<f8", represented);
    put!("catchment_coverage_fraction", "<f8", cat_fraction);
    let result = json!({"schema":"arwen.nextgen.catchment-cell-overlap.v1","source":crate::SOURCE_REVISION,"request_sha256":key,"output":out,"fabric_manifest":c.fabric_manifest,"fabric_sha256":c.fabric_sha256,"destination_manifest":c.destination_manifest,"destination_sha256":c.destination_sha256,"network_sha256":format!("{:x}",Sha256::digest(&network_bytes)),"data_license":m.document.input.data_license,"catchment_ids":ids,"destination_grid":grid,"cell_count":cells,"catchment_count":cats,"edge_count":edges,"files":files,"semantics":{"rows":"destination flat C-order cells; each row ascending catchment-axis index","weights":"polygon intersection area / full destination cell planar area; not normalized over the covered fraction","depth_mapping":"only if producer declares catchment depth spatially uniform: destination full-cell mean depth = sum(source depth * weight)","conservation":"covered source polygon volume only; uncovered source area is explicitly outside this rectangle","mask":"positive sum of polygon intersections; not geometric union, DEM-valid mask, physical active mask or drainage map","overlap":"raw source intersections preserved, no clamping or renormalization of overlapping polygons","parent":"not an executed ArWen parent mapping or a registered numerical producer"},"diagnostics":{"positive_cells":mask.iter().filter(|&&x|x!=0).count(),"overcovered_cells_exact":fractions.iter().filter(|&&x|x>1.).count(),"maximum_cell_coverage":fractions.iter().copied().fold(0.,f64::max),"minimum_catchment_coverage":cat_fraction.iter().copied().fold(f64::INFINITY,f64::min),"maximum_catchment_coverage":cat_fraction.iter().copied().fold(0.,f64::max),"source_intersection_sum_m2":overlaps.iter().sum::<f64>(),"destination_intersection_sum_m2":covered.iter().sum::<f64>()},"parallelism":{"rayon_threads":rayon::current_num_threads(),"work":"parallel LE decode, source rows, transpose sort, destination rows and derived areas; startup only","gpu":"none"},"timings_ms":{"load_hash_validate":load_ms,"transpose":transpose_ms,"write_hash":write_started.elapsed().as_secs_f64()*1000.,"total":started.elapsed().as_secs_f64()*1000.}});
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(result)
}
