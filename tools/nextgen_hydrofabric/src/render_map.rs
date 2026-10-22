//! Output-boundary map binding from the sealed, collapsed routing raster.
use crate::{
    bundle::digest,
    weights::{Grid, Projection, RowOrder},
    Result,
};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{fs, path::Path, time::Instant};

pub fn prepare(manifest: &Path, output: &Path) -> Result<Value> {
    let started = Instant::now();
    let m: Value = serde_json::from_slice(&fs::read(manifest)?)?;
    let grid: Grid = serde_json::from_value(m["grid"].clone())?;
    let cells = (grid.nx as usize)
        .checked_mul(grid.ny as usize)
        .ok_or("map grid overflow")?;
    if grid.nx < 2 || grid.ny < 2 {
        return Err("map needs a two-dimensional grid".into());
    }
    let field = &m["files"]["channel_node"];
    if field["dtype"] != "<i4" || field["shape"] != json!([grid.ny, grid.nx]) {
        return Err("map needs the exact routing-grid channel raster".into());
    }
    let channel_path = manifest
        .parent()
        .ok_or("manifest parent")?
        .join(field["path"].as_str().ok_or("channel path")?);
    if digest(&channel_path)? != field["sha256"].as_str().ok_or("channel seal")? {
        return Err("map channel raster SHA-256 differs".into());
    }
    let bytes = fs::read(&channel_path)?;
    if bytes.len() != cells * 4 {
        return Err("map channel raster extent differs".into());
    }
    let nodes = m["files"]["node_ids"]["shape"][0]
        .as_u64()
        .ok_or("node roster size")? as i64;
    if bytes.par_chunks_exact(4).any(|b| {
        let i = i32::from_le_bytes(b.try_into().unwrap()) as i64;
        i < -1 || i >= nodes
    }) {
        return Err("map channel raster exceeds routing roster".into());
    }
    // Each task owns a PROJ context; no context crosses worker threads.
    let coordinates = (0..cells)
        .into_par_iter()
        .chunks(4096)
        .map(|indices| -> Result<Vec<[f32; 2]>> {
            let p = Projection::new(&grid.crs, "EPSG:4326")?;
            indices
                .into_iter()
                .map(|i| {
                    let row = i / grid.nx as usize;
                    let south_row = match grid.row_order {
                        RowOrder::SouthToNorth => row,
                        RowOrder::NorthToSouth => grid.ny as usize - 1 - row,
                    };
                    let xy = [
                        grid.x0 + (i % grid.nx as usize) as f64 * grid.dx + grid.dx * 0.5,
                        grid.y0 + south_row as f64 * grid.dy + grid.dy * 0.5,
                    ];
                    let ll = p.transform(xy)?;
                    Ok([ll[1] as f32, ll[0] as f32])
                })
                .collect()
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    fs::create_dir(output)?;
    fs::write(output.join("channel_node.i32le"), bytes)?;
    for (axis, name) in [(0, "latitude.f32le"), (1, "longitude.f32le")] {
        fs::write(
            output.join(name),
            coordinates
                .iter()
                .flat_map(|ll| ll[axis].to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    }
    let buffer = |name: &str, dtype: &str| json!({"path":output.join(name),"dtype":dtype});
    Ok(json!({"terrain":{"nx":grid.nx,"ny":grid.ny,
        "latitude":buffer("latitude.f32le","<f4"),"longitude":buffer("longitude.f32le","<f4"),
        "channel_node":buffer("channel_node.i32le","<i4")},
        "source_manifest":manifest,"channel_node_sha256":field["sha256"],
        "scope":"rasterized prepared network, exact collapsed routing node order; no HAND or inundation inferred",
        "seconds":started.elapsed().as_secs_f64(),"parallelism":format!("{} Rayon workers, independent PROJ contexts",rayon::current_num_threads())}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sealed_channel_map_preserves_routing_indices_and_north_row_coordinates() {
        let root = std::env::temp_dir().join(format!("hydro-render-map-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let channel = [1i32, -1, 0, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(root.join("channel.bin"), &channel).unwrap();
        let m = json!({"grid":{"nx":2,"ny":2,"x0":0.,"y0":0.,"dx":1000.,"dy":1000.,"crs":"EPSG:3857","row_order":"north_to_south"},
            "files":{"node_ids":{"shape":[2]},"channel_node":{"path":"channel.bin","dtype":"<i4","shape":[2,2],"sha256":digest(&root.join("channel.bin")).unwrap()}}});
        let manifest = root.join("manifest.json");
        fs::write(&manifest, serde_json::to_vec(&m).unwrap()).unwrap();
        let receipt = prepare(&manifest, &root.join("map")).unwrap();
        assert_eq!(
            fs::read(root.join("map/channel_node.i32le")).unwrap(),
            channel
        );
        let values = |name| {
            fs::read(root.join(name))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let lat = values("map/latitude.f32le");
        let lon = values("map/longitude.f32le");
        assert!(lat[0] > lat[2] && lon[1] > lon[0]);
        assert!((lon[0] - 0.0044915764).abs() < 1e-8);
        assert_eq!(receipt["terrain"]["nx"], 2);
        fs::write(root.join("channel.bin"), [0u8; 16]).unwrap();
        assert!(prepare(&manifest, &root.join("bad"))
            .unwrap_err()
            .to_string()
            .contains("SHA-256"));
        fs::remove_dir_all(root).unwrap();
    }
}
