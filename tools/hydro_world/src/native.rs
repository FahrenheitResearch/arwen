//! Native vector recipe: a reach/catchment id join and a hydrologic-sequence
//! downstream join. Source column names, archives and licences live in the
//! source table. No new runoff or routing physics lives here.
use crate::build::{load_dams, load_lakes, threads, Hydro, LakeIn, ReachIn, Receipt};
use crate::source::{BuildSpec, Input, SourceRow};
use crate::{geo, grid::BBox, shp, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn col<'a>(inp: &'a Input, meaning: &str) -> Result<&'a str> {
    inp.columns
        .iter()
        .find(|(_, v)| v.as_str() == meaning)
        .map(|(k, _)| k.as_str())
        .ok_or_else(|| format!("{} has no {meaning} column", inp.dataset).into())
}
fn paths(inp: &Input, root: &Path, regions: &[String]) -> Vec<PathBuf> {
    inp.archives
        .iter()
        .filter(|a| regions.contains(&a.region))
        .map(|a| root.join(&a.local))
        .collect()
}
#[derive(Clone, Copy)]
struct Attr {
    id: i64,
    seq: i64,
    down: i64,
    area: f64,
    upstream: f64,
}
struct Native {
    id: i64,
    line: Vec<(f64, f64)>,
    rings: Vec<Vec<(f64, f64)>>,
}

fn native_slope(slope: f64, upper_cm: f64, lower_cm: f64, length_km: f64) -> Option<f64> {
    if slope.is_finite() && slope >= 0.0 {
        return Some(slope);
    }
    // Some NHDPlus reaches carry SLOPE=-9998 but valid smoothed
    // elevations. Reconstruct the same endpoint gradient, in SI units.
    (upper_cm.is_finite()
        && lower_cm.is_finite()
        && upper_cm > -9998.0
        && lower_cm > -9998.0
        && length_km.is_finite()
        && length_km > 0.0)
        .then(|| ((upper_cm - lower_cm) / (100_000.0 * length_km)).max(0.0))
}

fn dem_points(row: &SourceRow, root: &Path, points: &[(f64, f64)]) -> Result<Vec<f64>> {
    let mut groups: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, &(x, y)) in points.iter().enumerate() {
        groups
            .entry((y.floor() as i32, x.floor() as i32))
            .or_default()
            .push(i);
    }
    let pattern = row.inputs["dem"]
        .local
        .as_deref()
        .ok_or("DEM has no local pattern")?;
    let chunks = groups
        .into_par_iter()
        .map(|((la, lo), ids)| -> Result<Vec<(usize, f64)>> {
            let path = root.join(crate::source::tile_path(pattern, la, lo));
            let absent = path.parent().unwrap().join("ABSENT.txt");
            if !path.exists()
                && std::fs::read_to_string(absent)
                    .unwrap_or_default()
                    .lines()
                    .any(|s| Some(s) == path.file_name().and_then(|n| n.to_str()))
            {
                return Ok(ids.into_iter().map(|i| (i, 0.0)).collect());
            }
            let t = crate::tiff::read(&path)?;
            ids.into_iter()
                .map(|i| {
                    let (x, y) = points[i];
                    let c = ((x - t.x0) / t.dx).round().clamp(0.0, (t.width - 1) as f64) as usize;
                    let r = ((t.y0 - y) / t.dy)
                        .round()
                        .clamp(0.0, (t.height - 1) as f64) as usize;
                    let z = t.data[r * t.width + c];
                    if !z.is_finite() || t.nodata == Some(z) {
                        return Err(format!("DEM has no elevation at {x},{y}").into());
                    }
                    Ok((i, z as f64))
                })
                .collect()
        })
        .collect::<Result<Vec<_>>>()?;
    let mut out = vec![0.0; points.len()];
    for (i, z) in chunks.into_iter().flatten() {
        out[i] = z;
    }
    Ok(out)
}

pub fn hydro(
    row: &SourceRow,
    spec: &BuildSpec,
    b: &BBox,
    root: &Path,
    parent: Option<&crate::parent_grid::ParentGrid>,
    rc: &mut Receipt,
) -> Result<Hydro> {
    let regions = crate::source::regions(row, root, b)?;
    if regions.is_empty() {
        return Err("no native processing unit intersects this domain".into());
    }
    rc.count("processing_units", &regions);
    let vaa = &row.inputs["network_attributes"];
    let elev = &row.inputs["elevation_attributes"];
    let riv = &row.inputs["river_network"];
    let cat = &row.inputs["catchments"];
    let attrs = rc.time(
        "1 native topology attributes",
        &threads(),
        || -> Result<Vec<Attr>> {
            let chunks = paths(vaa, root, &regions)
                .par_iter()
                .map(|p| -> Result<Vec<Attr>> {
                    let d = shp::Dbf::open(p)?;
                    let (id, seq, down) = (
                        d.i64s(col(vaa, "id")?)?,
                        d.i64s(col(vaa, "sequence")?)?,
                        d.i64s(col(vaa, "down_sequence")?)?,
                    );
                    let (area, upstream) = (
                        d.f64s(col(vaa, "area_km2")?)?,
                        d.f64s(col(vaa, "upstream_km2")?)?,
                    );
                    Ok((0..d.n)
                        .filter(|&i| seq[i] > 0)
                        .map(|i| Attr {
                            id: id[i],
                            seq: seq[i],
                            down: down[i],
                            area: area[i],
                            upstream: upstream[i],
                        })
                        .collect())
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(chunks.into_iter().flatten().collect())
        },
    )?;
    let by_id: HashMap<_, _> = attrs.iter().map(|a| (a.id, *a)).collect();
    let by_seq: HashMap<_, _> = attrs.iter().map(|a| (a.seq, *a)).collect();
    if by_id.len() != attrs.len() || by_seq.len() != attrs.len() {
        return Err("duplicate native id or hydrologic sequence across processing units".into());
    }
    if attrs.iter().any(|a| a.down >= a.seq) {
        return Err("native downstream sequence is not strictly decreasing".into());
    }
    let mut slopes = rc.time(
        "2 native elevation and slope attributes",
        &threads(),
        || -> Result<HashMap<i64, (f64, f64)>> {
            let chunks = paths(elev, root, &regions)
                .par_iter()
                .map(|p| -> Result<Vec<_>> {
                    let d = shp::Dbf::open(p)?;
                    let (id, slope, z) = (
                        d.i64s(col(elev, "id")?)?,
                        d.f64s(col(elev, "slope")?)?,
                        d.f64s(col(elev, "outlet_elevation_cm")?)?,
                    );
                    let upper = d.f64s(col(elev, "upper_elevation_cm")?)?;
                    let length = d.f64s(col(elev, "slope_length_km")?)?;
                    Ok((0..d.n)
                        .map(|i| {
                            (
                                id[i],
                                (
                                    native_slope(slope[i], upper[i], z[i], length[i])
                                        .unwrap_or(f64::NAN),
                                    z[i] / 100.0,
                                ),
                            )
                        })
                        .collect())
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(chunks.into_iter().flatten().collect())
        },
    )?;
    // A whole native catchment belongs to the domain holding its outlet.
    // Do not clip polygons at a national rectangle or discard their islands.
    let mut lines = rc.time(
        "3 read native flowlines (outlet ownership)",
        &threads(),
        || -> Result<HashMap<i64, Vec<(f64, f64)>>> {
            let chunks = paths(riv, root, &regions)
                .par_iter()
                .map(|p| -> Result<Vec<_>> {
                    let d = shp::Dbf::open(&p.with_extension("dbf"))?;
                    let ids = d.i64s(col(riv, "id")?)?;
                    let shapes = shp::read_shapes(p)?;
                    Ok(shapes
                        .into_par_iter()
                        .enumerate()
                        .filter_map(|(i, s)| match s {
                            shp::Shape::Parts(parts) if by_id.contains_key(&ids[i]) => {
                                let line: Vec<_> = parts.into_iter().flatten().collect();
                                (line.len() > 1
                                    && line.last().is_some_and(|p| b.contains(p.0, p.1)))
                                .then_some((ids[i], line))
                            }
                            _ => None,
                        })
                        .collect())
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(chunks.into_iter().flatten().collect())
        },
    )?;
    if let Some(parent) = parent {
        let excluded = rc.time("3b native outlets inside projected parent", &threads(), || -> Result<usize> {
            let ids: Vec<_> = lines.keys().copied().collect();
            let points: Vec<_> = ids.iter().map(|id| *lines[id].last().unwrap()).collect();
            let inside = parent.contains_outlets(&points)?;
            let mut excluded = 0;
            for (id, keep) in ids.into_iter().zip(inside) {
                if !keep { lines.remove(&id); excluded += 1; }
            }
            Ok(excluded)
        })?;
        rc.count("native_outlets_outside_projected_parent", excluded);
        rc.count("parent_grid_boundary_policy", "whole native catchment owned by its outlet inside the half-open projected parent grid");
    }
    let mut native = rc.time(
        "4 read native catchments and join flowlines",
        &threads(),
        || -> Result<Vec<Native>> {
            let chunks = paths(cat, root, &regions)
                .par_iter()
                .map(|p| -> Result<Vec<_>> {
                    let d = shp::Dbf::open(&p.with_extension("dbf"))?;
                    let ids = d.i64s(col(cat, "id")?)?;
                    let shapes = shp::read_shapes(p)?;
                    Ok(shapes
                        .into_par_iter()
                        .enumerate()
                        .filter_map(|(i, s)| {
                            let line = lines.get(&ids[i])?;
                            match s {
                                shp::Shape::Parts(rings) if !rings.is_empty() => Some(Native {
                                    id: ids[i],
                                    line: line.clone(),
                                    rings,
                                }),
                                _ => None,
                            }
                        })
                        .collect())
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(chunks.into_iter().flatten().collect())
        },
    )?;
    if let Some(parent) = parent {
        let excluded = rc.time("4a native polygons overlapping projected parent", &threads(), || -> Result<usize> {
            let rings: Vec<_> = native.iter().map(|r| r.rings.as_slice()).collect();
            let keep = parent.overlaps_divides(&rings)?;
            let excluded = keep.iter().filter(|&&k| !k).count();
            native = std::mem::take(&mut native).into_iter().zip(keep).filter_map(|(r,k)| {
                if k {Some(r)} else {lines.remove(&r.id);None}
            }).collect();
            Ok(excluded)
        })?;
        rc.count("native_outlet_outline_offsets_outside_parent", excluded);
    }
    native.par_sort_unstable_by_key(|r| r.id);
    if native.is_empty() {
        return Err("no native catchment with an outlet in the domain".into());
    }
    let missing: Vec<_> = native
        .par_iter()
        .filter(|r| slopes.get(&r.id).is_none_or(|(s, _)| !s.is_finite()))
        .collect();
    rc.count("native_slopes_recovered_from_dem", missing.len());
    let recovered = rc.time(
        "4b missing native slopes from DEM endpoints",
        &threads(),
        || -> Result<Vec<_>> {
            let points: Vec<_> = missing
                .iter()
                .flat_map(|r| [*r.line.first().unwrap(), *r.line.last().unwrap()])
                .collect();
            let z = dem_points(row, root, &points)?;
            Ok(missing
                .par_iter()
                .enumerate()
                .map(|(i, r)| {
                    let slope =
                        ((z[2 * i] - z[2 * i + 1]) / geo::polyline_geodesic_m(&r.line)).max(0.0);
                    (r.id, (slope, slopes.get(&r.id).map_or(f64::NAN, |s| s.1)))
                })
                .collect())
        },
    )?;
    slopes.extend(recovered);
    let index: HashMap<_, _> = native.iter().enumerate().map(|(i, r)| (r.id, i)).collect();
    if index.len() != native.len() {
        return Err("duplicate native catchment id".into());
    }
    rc.count(
        "native_flowlines_without_catchment",
        lines.len() - native.len(),
    );
    let reaches = rc.time(
        "5 join native downstream across processing units",
        &threads(),
        || -> Result<Vec<ReachIn>> {
            native
                .into_par_iter()
                .map(|r| {
                    let a = by_id[&r.id];
                    let mut next = a.down;
                    let mut down = None;
                    // Traverse connectors lacking catchments without breaking the river.
                    while let Some(n) = by_seq.get(&next) {
                        if let Some(&j) = index.get(&n.id) {
                            down = Some(j);
                            break;
                        }
                        // A reach with an outlet outside the forecast is a true exit;
                        // never jump outside and re-enter across the domain boundary.
                        if !lines.contains_key(&n.id) {
                            break;
                        }
                        next = n.down;
                    }
                    let (slope, _) = slopes
                        .get(&r.id)
                        .copied()
                        .ok_or_else(|| format!("native elevation/slope missing for {}", r.id))?;
                    if !slope.is_finite() || slope < 0.0 || !a.area.is_finite() || a.area <= 0.0 {
                        return Err(format!(
                            "invalid native slope ({slope}) or catchment area ({}) for {}",
                            a.area, r.id
                        )
                        .into());
                    }
                    let mut rings = r.rings.into_iter();
                    Ok(ReachIn {
                        down,
                        line: r.line,
                        divide: Some((rings.next().unwrap(), rings.collect())),
                        esri_parts: true,
                        local_area_km2: a.area,
                        slope,
                        hf_id: r.id as f64,
                        terrain: None,
                    })
                })
                .collect()
        },
    )?;
    let lakes = rc.time(
        "6 HydroLAKES to native network crosswalk",
        &threads(),
        || -> Result<Vec<LakeIn>> {
            let lakes = load_lakes(row, spec, b, root)?;
            let cell = 0.1;
            let mut bins: HashMap<(i64, i64), Vec<(f64, f64, usize)>> = HashMap::new();
            for (j, r) in reaches.iter().enumerate() {
                for &(x, y) in &r.line {
                    bins.entry(((x / cell).floor() as i64, (y / cell).floor() as i64))
                        .or_default()
                        .push((x, y, j));
                }
            }
            let matches = lakes
                .par_iter()
                .filter_map(|l| {
                    let cx = (l.lon / cell).floor() as i64;
                    let cy = (l.lat / cell).floor() as i64;
                    let dx = (spec.lake_snap_radius / (111.0 * l.lat.to_radians().cos() * cell))
                        .ceil() as i64;
                    let dy = (spec.lake_snap_radius / (111.0 * cell)).ceil() as i64;
                    let mut best: Option<(f64, usize)> = None;
                    for x in cx - dx..=cx + dx {
                        for y in cy - dy..=cy + dy {
                            for &(vx, vy, j) in bins.get(&(x, y)).into_iter().flatten() {
                                if reaches[j].down.is_none() {
                                    continue;
                                }
                                let area = by_id[&(reaches[j].hf_id as i64)].upstream;
                                let ratio = area / l.wshd_km2;
                                if l.wshd_km2 > 0.0
                                    && (ratio > spec.lake_snap_area_ratio
                                        || ratio < 1.0 / spec.lake_snap_area_ratio)
                                {
                                    continue;
                                }
                                let d = geo::geodesic_m(l.lon, l.lat, vx, vy) / 1000.0;
                                if d <= spec.lake_snap_radius
                                    && best.is_none_or(|(old, k)| d < old || (d == old && j < k))
                                {
                                    best = Some((d, j));
                                }
                            }
                        }
                    }
                    best.map(|(_, j)| LakeIn {
                        hylak: l.hylak,
                        area_km2: l.area_km2,
                        mean_depth_m: l.depth_m,
                        weir_e_m: slopes[&(reaches[j].hf_id as i64)].1,
                        outlet: *reaches[j].line.last().unwrap(),
                        reach: j,
                    })
                })
                .collect::<Vec<_>>();
            let mut unique: HashMap<usize, LakeIn> = HashMap::new();
            for l in matches {
                let replace = unique.get(&l.reach).is_none_or(|v| {
                    l.area_km2 > v.area_km2 || (l.area_km2 == v.area_km2 && l.hylak < v.hylak)
                });
                if replace {
                    unique.insert(l.reach, l);
                }
            }
            let mut out: Vec<_> = unique.into_values().collect();
            out.sort_by_key(|l| l.reach);
            // Reservoir elevations use the same EGM2008 DEM as the worldwide
            // recipe; native NAVD88 channel elevations are used only for gradients.
            let points: Vec<_> = out.iter().map(|l| l.outlet).collect();
            let z = dem_points(row, root, &points)?;
            for (l, z) in out.iter_mut().zip(z) {
                l.weir_e_m = z;
            }
            Ok(out)
        },
    )?;
    rc.count(
        "native_lake_crosswalk",
        lakes
            .iter()
            .map(|l| serde_json::json!({"hylak_id":l.hylak,"comid":reaches[l.reach].hf_id as i64}))
            .collect::<Vec<_>>(),
    );
    rc.count("reaches", reaches.len());
    rc.count("lakes_snapped_to_inner_reaches", lakes.len());
    let lake_by_id: HashMap<_, _> = lakes.iter().map(|l| (l.hylak, l)).collect();
    let nid = |k: usize| row.id_namespace * 10_000_000_000 + k as i64 + 1;
    let dams=rc.time("7 GDW reservoir crosswalk", &threads(), || -> Result<Vec<_>> {
        Ok(load_dams(row,b,root)?.into_par_iter().map(|d| {
            let lake=lake_by_id.get(&d.hylak);
            serde_json::json!({
                "gdw_id":d.id,"hylak_id":(d.hylak>0).then_some(d.hylak),"name":d.name,
                "dam_height_m":d.height_m,"dam_length_m":d.length_m,"capacity_mcm":d.capacity_mcm,
                "lon":d.lon,"lat":d.lat,"main_use":d.main_use,"year":d.year,"lake_control":d.lake_control,
                "flowpath_id":lake.map(|l|format!("wb-{}",nid(l.reach))),
                "fabric_lake_nexus":lake.and_then(|l|reaches[l.reach].down.map(|j|format!("nex-{}",nid(j)))),
                "snapped_upstream_area_km2":lake.map(|l|by_id[&(reaches[l.reach].hf_id as i64)].upstream),
                "gdw_catchment_km2":d.catchment_km2
            })
        }).collect())
    })?;
    rc.count("dams_in_box", dams.len());
    rc.count(
        "dams_on_fabric_lakes",
        dams.iter()
            .filter(|d| !d["fabric_lake_nexus"].is_null())
            .count(),
    );
    Ok(Hydro {
        reaches,
        lakes,
        dams,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_missing_slope_uses_smoothed_elevations_with_correct_units() {
        assert_eq!(native_slope(-9998., 10500., 10000., 2.), Some(0.0025));
        assert_eq!(native_slope(0.01, -9998., -9998., 0.), Some(0.01));
        assert_eq!(native_slope(-9998., -9998., 10000., 2.), None);
    }
}
