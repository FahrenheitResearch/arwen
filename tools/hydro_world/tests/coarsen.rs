//! Any box on Earth gets a working grid: past a rule's cell budget the
//! builder coarsens the row's own grid (block mean) instead of refusing.
//! Origin: a 12 km continental ArWen domain (86 x 46 degrees, 5.7 G cells
//! at GLO-90) failed hydro prep with "grid 55223 x 103303 exceeds the u32
//! cell index; split the domain".
use hydro_world::grid::{self, BBox, Grid};
use hydro_world::source;
use hydro_world::tiff::Raster;
use std::collections::HashMap;

fn table() -> source::SourceTable {
    source::load(concat!(env!("CARGO_MANIFEST_DIR"), "/data/sources.v1.json")).unwrap()
}

#[test]
fn native_auxiliary_grid_uses_the_continental_budget() {
    let t=table();
    let b=BBox::parse("-136,17,-55,53").unwrap();
    let (_,k)=hydro_world::build::resolve(&t,"conus-nhdplus-v21",&b).unwrap();
    assert!(Grid::for_box(&b,3.0*k as f64).unwrap().len()<=250_000_000);
    assert!(k>1);
    let b=BBox::parse("-106,33,-105,34").unwrap();
    assert_eq!(hydro_world::build::resolve(&t,"conus-nhdplus-v21",&b).unwrap().1,1);
}

#[test]
fn continental_and_global_boxes_resolve_inside_the_budget() {
    let t = table();
    let budget = t.auto.iter().find(|r| r.coarsen).and_then(|r| r.max_cells).unwrap();
    // The failing domain's footprint: 55223 x 103303 cells at 3".
    let continental = BBox::parse("-140.6,12.0,-54.5,58.0").unwrap();
    let globe = BBox::parse("-180,-90,180,90").unwrap();
    for b in [continental, globe] {
        let (row, k) = hydro_world::build::resolve(&t, "auto", &b).unwrap();
        assert_eq!(row, "global-copernicus-glo90");
        assert!(k > 1);
        let g = Grid::for_box(&b, 3.0 * k as f64).unwrap();
        assert!(g.len() <= budget, "{} cells over {budget}", g.len());
        // Smallest factor: one less does not fit.
        let (ny, nx) = Grid::dims(&b, 3.0 * (k - 1) as f64);
        assert!(ny * nx > budget);
    }
    // Boxes inside the budget keep the row's own grid, unchanged.
    let rhine = BBox::parse("4,46,12,52").unwrap();
    assert_eq!(hydro_world::build::resolve(&t, "auto", &rhine).unwrap(), ("global-copernicus-glo90".to_string(), 1));
    let small = BBox::parse("-101,30.5,-99,31.5").unwrap();
    assert_eq!(hydro_world::build::resolve(&t, "auto", &small).unwrap(), ("global-copernicus-glo30".to_string(), 1));
    // A named row coarsens only as far as the cell index needs.
    let (_, k) = hydro_world::build::resolve(&t, "global-copernicus-glo90", &continental).unwrap();
    let (ny, nx) = Grid::dims(&continental, 3.0 * k as f64);
    assert!(k >= 2 && ny * nx <= grid::MAX_CELLS);
}

/// One synthetic 1-degree tile at `res_as` arc-seconds, Copernicus layout
/// (rows from lat0+1 down to lat0+res, columns from lon0).
fn tile(lat0: i32, lon0: i32, res_as: f64, f: impl Fn(f64, f64) -> f32) -> Raster {
    let n = (3600.0 / res_as).round() as usize;
    let d = res_as / 3600.0;
    let (y0, x0) = (lat0 as f64 + 1.0, lon0 as f64);
    let mut data = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            data.push(f(y0 - r as f64 * d, x0 + c as f64 * d));
        }
    }
    Raster { width: n, height: n, data, x0, y0, dx: d, dy: d, nodata: None }
}

#[test]
fn coarsened_cells_are_block_means_of_the_row_grid() {
    let res = 60.0;
    // Land everywhere except a sea strip (value 0) and one missing tile.
    let height = |lat: f64, lon: f64| if lon < 10.25 { 0.0 } else { (100.0 + 7.0 * lat + 3.0 * lon).round() as f32 };
    let mut tiles = HashMap::new();
    for la in 40..42 {
        for lo in 10..13 {
            if (la, lo) != (41, 12) {
                tiles.insert((la, lo), tile(la, lo, res, height));
            }
        }
    }
    let b = BBox::parse("10,40,13,42").unwrap();
    let fine = Grid::for_box(&b, res).unwrap();
    let (fz, fm) = grid::mosaic(&fine, &tiles).unwrap();
    for k in [2u32, 3, 5] {
        let g = Grid::for_box(&b, res * k as f64).unwrap();
        let mut z = vec![0f32; g.len()];
        let mut m = vec![false; g.len()];
        grid::aggregate(&g, k, res, &tiles, &b, Some(0.0), 0..g.ny, &mut z, &mut m).unwrap();
        for row in 0..g.ny {
            for col in 0..g.nx {
                // The k x k fine cells of this coarse cell, by position.
                let (lat_c, lon_c) = (g.lat(row), g.lon(col));
                let (mut n, mut miss, mut sea, mut sum, mut land) = (0, 0, 0, 0f64, 0);
                for fr in 0..fine.ny {
                    let dr = ((fine.lat(fr) - lat_c) * 3600.0 / res).round() as i64;
                    if dr < -(((k - 1) / 2) as i64) || dr > (k / 2) as i64 {
                        continue;
                    }
                    for fc in 0..fine.nx {
                        let dlon = ((fine.lon(fc) - lon_c) * 3600.0 / res).round();
                        if dlon < -((k / 2) as f64) || dlon > ((k - 1) / 2) as f64 {
                            continue;
                        }
                        let i = fr * fine.nx + fc;
                        n += 1;
                        if fm[i] {
                            miss += 1;
                        } else if fz[i] == 0.0 {
                            sea += 1;
                        } else {
                            sum += fz[i] as f64;
                            land += 1;
                        }
                    }
                }
                let i = row * g.nx + col;
                if miss * 2 > n {
                    assert!(m[i], "k={k} ({row},{col}) should be missing");
                } else if (miss + sea) * 2 > n || land == 0 {
                    assert!(!m[i] && z[i] == 0.0, "k={k} ({row},{col}) should be sea");
                } else {
                    assert!(!m[i]);
                    assert_eq!(z[i], (sum / land as f64) as f32, "k={k} ({row},{col}) n={n}");
                }
            }
        }
    }
}
