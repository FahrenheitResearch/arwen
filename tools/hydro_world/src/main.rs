use hydro_world::{build, check, compare, contract, geo, gpkg, grid, render, source};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

/// The commit this binary was built from; the release pin step reads it from the bytes.
const SOURCE_REV_STAMP: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() {
    std::hint::black_box(SOURCE_REV_STAMP);
    if let Err(e) = run() {
        eprintln!("rw_hydro_world: {e}");
        std::process::exit(1)
    }
}

const USAGE: &str = "usage: rw_hydro_world schema | check FABRIC.gpkg | roundtrip IN.gpkg OUT.gpkg | sources TABLE.json | admit TABLE.json ROW|auto [--accept-licence NAME]
  | tiles --sources TABLE.json --row ROW --bbox W,S,E,N --data-root DIR
  | build --sources TABLE.json --row ROW|auto --bbox W,S,E,N --data-root DIR --out FABRIC.gpkg [--device auto|cuda|cpu] [--parent-grid GRID.json] [--accept-licence NAME]... [--params PARAMS.json] [--receipt RECEIPT.json]
  | hydrography-compare --sources TABLE.json --row ROW|auto --bbox W,S,E,N --data-root DIR [--out OUT.json]
  (--table is accepted for --sources; --data-root defaults to $GPUWM_HYDRO_WORLD_TILES)
  | compare --fabric FABRIC.gpkg --reference NOAA_5070.gpkg [--gauges Gages-A,Gages-B] [--radius-km 2] [--out OUT.json]
  | render --fabric FABRIC.gpkg --png OUT.png [--width 1600] [--lonlat W,S,E,N] [--reference NOAA_5070.gpkg]";

/// `--name value` or `--name=value` (the latter for values starting with '-').
fn flags(a: &[String], name: &str) -> Vec<String> {
    let eq = format!("{name}=");
    a.iter()
        .enumerate()
        .filter_map(|(i, x)| if x == name { a.get(i + 1).cloned() } else { x.strip_prefix(&eq).map(str::to_string) })
        .collect()
}
fn flag(a: &[String], name: &str) -> Option<String> {
    flags(a, name).into_iter().next()
}
fn need(a: &[String], name: &str) -> hydro_world::Result<String> {
    flag(a, name).ok_or_else(|| format!("{name} is required\n{USAGE}").into())
}
/// The source table: `--sources`, or `--table` as the default-on door's
/// provider row passes it (gpuwm/data/hydro/fabric-providers.v1.json).
fn sources_flag(a: &[String]) -> hydro_world::Result<String> {
    flag(a, "--sources").or_else(|| flag(a, "--table")).ok_or_else(|| format!("--sources (or --table) is required\n{USAGE}").into())
}
/// The data root: `--data-root`, or the door's GPUWM_HYDRO_WORLD_TILES.
fn data_root_flag(a: &[String]) -> hydro_world::Result<PathBuf> {
    flag(a, "--data-root")
        .or_else(|| std::env::var("GPUWM_HYDRO_WORLD_TILES").ok().filter(|s| !s.is_empty()))
        .map(PathBuf::from)
        .ok_or_else(|| format!("--data-root (or GPUWM_HYDRO_WORLD_TILES) is required\n{USAGE}").into())
}
fn options(a: &[String]) -> hydro_world::Result<build::Options> {
    Ok(build::Options {
        sources: PathBuf::from(sources_flag(a)?),
        row: need(a, "--row")?,
        bbox: grid::BBox::parse(&need(a, "--bbox")?)?,
        data_root: data_root_flag(a)?,
        accept: flags(a, "--accept-licence"),
        params: flag(a, "--params").map(PathBuf::from),
        revision: option_env!("HYDRO_WORLD_REV").unwrap_or("unrecorded-source").to_string(),
        parent_grid: flag(a, "--parent-grid").map(|p| -> hydro_world::Result<_> {
            Ok(serde_json::from_slice(&std::fs::read(p)?)?)
        }).transpose()?,
        device: build::Device::parse(&flag(a, "--device").unwrap_or_else(|| "auto".into()))?,
    })
}

fn laea_of(f: &contract::Fabric) -> hydro_world::Result<geo::Laea> {
    let d = &f.srs.definition;
    let p = |name: &str| -> hydro_world::Result<f64> {
        let k = d.find(&format!("PARAMETER[\"{name}\",")).ok_or_else(|| format!("fabric CRS has no {name}"))?;
        let rest = &d[k + name.len() + 13..];
        Ok(rest[..rest.find(',').ok_or("bad WKT")?].parse()?)
    };
    Ok(geo::Laea::new(p("Latitude of natural origin")?, p("Longitude of natural origin")?))
}

fn run() -> hydro_world::Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    let arg = |i: usize| a.get(i).map(String::as_str).ok_or(USAGE);
    match a.get(1).map(String::as_str) {
        Some("schema") => println!("{}", serde_json::to_string_pretty(&json!({"contract": contract::CONTRACT, "tables": contract::TABLES, "metadata_keys": contract::METADATA_KEYS}))?),
        Some("check") => {
            let (f, read) = gpkg::read(arg(2)?)?;
            let r = check::check(&f);
            println!("{}", serde_json::to_string_pretty(&json!({"conforming": r.conforming(), "read": read, "check": r}))?);
            if !r.conforming() {
                std::process::exit(2)
            }
        }
        Some("roundtrip") => {
            let (f, read) = gpkg::read(arg(2)?)?;
            gpkg::write(&f, arg(3)?)?;
            let (g, _) = gpkg::read(arg(3)?)?;
            let same = f == g;
            println!("{}", serde_json::to_string_pretty(&json!({"lossless": same, "source_deviations": read.deviations}))?);
            if !same {
                std::process::exit(2)
            }
        }
        Some("sources") => {
            let t = source::load(arg(2)?)?;
            println!("{}", serde_json::to_string_pretty(&t)?);
        }
        Some("admit") => {
            let t = source::load(arg(2)?)?;
            let requested = arg(3)?;
            // The forecast door probes before it has built the domain fabric.
            // Admit every possible auto choice; building still selects by bbox.
            let ids = if requested == "auto" {
                if t.auto.is_empty() {
                    return Err("source table has no auto rules".into());
                }
                t.auto.iter().map(|rule| rule.row.as_str()).collect::<Vec<_>>()
            } else {
                vec![requested]
            };
            let accept = flags(&a, "--accept-licence");
            for id in &ids {
                let row = t.rows.iter().find(|r| r.id == *id).ok_or("no such source row")?;
                source::admit_with(row, &accept)?;
            }
            println!("{requested} admitted: {}", ids.join(", "));
        }
        Some("tiles") => {
            let t = source::load(&sources_flag(&a)?)?;
            let b = grid::BBox::parse(&need(&a, "--bbox")?)?;
            let rid = build::resolve_row(&t, &need(&a, "--row")?, &b)?;
            let row = t.rows.iter().find(|r| r.id == rid).ok_or("no such source row")?;
            let root = data_root_flag(&a)?;
            let mut out = Vec::new();
            for (name, inp) in &row.inputs {
                let (Some(url), Some(local)) = (inp.url.as_deref(), inp.local.as_deref()) else { continue };
                if !local.contains("{lat2}") {
                    continue;
                }
                for (la, lo) in b.tiles() {
                    let l = source::tile_path(local, la, lo);
                    out.push(json!({"input": name, "url": source::tile_path(url, la, lo), "local": l, "exists": root.join(&l).exists()}));
                }
            }
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Some("build") => {
            let t0 = Instant::now();
            let out = need(&a, "--out")?;
            let o = options(&a)?;
            let mut b = build::build(&o)?;
            let tw = Instant::now();
            if std::path::Path::new(&out).exists() {
                std::fs::remove_file(&out)?;
            }
            gpkg::write(&b.fabric, &out)?;
            b.receipt.stages.push(build::StageTime { stage: "9 write GeoPackage".into(), ms: tw.elapsed().as_secs_f64() * 1e3, parallelism: format!("row encoding {} threads; SQLite insert 1 thread (single-writer database, one transaction)", rayon::current_num_threads()) });
            let terrain_path = format!("{out}.terrain.json");
            std::fs::write(&terrain_path, serde_json::to_vec_pretty(&b.terrain)?)?;
            let dams_path = (!b.dams.is_empty()).then(|| format!("{out}.dams.json"));
            if let Some(p) = &dams_path {
                std::fs::write(p, serde_json::to_vec_pretty(&json!({"schema": "arwen.hydro-world.dams.v1", "fabric": out, "dams": b.dams}))?)?;
            }
            let tc = Instant::now();
            let (back, _) = gpkg::read(&out)?;
            let lossless = back == b.fabric;
            if !lossless {
                let (x, y) = (serde_json::to_value(&back)?, serde_json::to_value(&b.fabric)?);
                for (k, v) in x.as_object().unwrap() {
                    if y.get(k) != Some(v) {
                        eprintln!("read-back differs in {k}");
                    }
                }
            }
            let r = check::check(&back);
            b.receipt.stages.push(build::StageTime { stage: "9 read back + contract check".into(), ms: tc.elapsed().as_secs_f64() * 1e3, parallelism: format!("9 tables on parallel read-only connections, row decoding {} threads; check 1 thread", rayon::current_num_threads()) });
            b.receipt.count("routing_levels", build::level_count(&back));
            let receipt = json!({
                "schema": "arwen.hydro-world.build-receipt.v1",
                "command": a[1..].join(" "),
                "builder_revision": o.revision,
                "threads": rayon::current_num_threads(),
                "total_ms": t0.elapsed().as_secs_f64() * 1e3,
                "fabric": out,
                "terrain_sidecar": terrain_path,
                "dams_sidecar": dams_path,
                "stages": b.receipt.stages,
                "counts": b.receipt.counts,
                "readback_lossless": lossless,
                "check": {"errors": r.errors, "contract_gaps": r.contract_gaps, "counts": r.counts, "canonical_order": r.canonical_order},
            });
            let txt = serde_json::to_string_pretty(&receipt)?;
            if let Some(p) = flag(&a, "--receipt") {
                std::fs::write(p, &txt)?;
            }
            println!("{txt}");
            if !r.errors.is_empty() || !lossless {
                std::process::exit(2)
            }
        }
        Some("hydrography-compare") => {
            #[cfg(feature = "cuda")]
            {
                let o = options(&a)?;
                let mut v = build::compare_passes(&o)?;
                v["command"] = json!(a[1..].join(" "));
                v["builder_revision"] = json!(o.revision);
                let txt = serde_json::to_string_pretty(&v)?;
                if let Some(p) = flag(&a, "--out") {
                    std::fs::write(p, &txt)?;
                }
                println!("{txt}");
                if v["identical"] != json!(true) {
                    std::process::exit(2)
                }
            }
            #[cfg(not(feature = "cuda"))]
            return Err("hydrography-compare needs the cuda feature".into());
        }
        Some("compare") => {
            let (f, _) = gpkg::read(&need(&a, "--fabric")?)?;
            let (rf, _) = gpkg::read(&need(&a, "--reference")?)?;
            let gauges: Vec<String> = flag(&a, "--gauges").map(|s| s.split(',').map(str::to_string).collect()).unwrap_or_default();
            let radius = flag(&a, "--radius-km").map(|s| s.parse()).transpose()?.unwrap_or(2.0);
            let tol = compare::Tolerances { area_rel: 0.05, magnitude_rel: 0.25, mainstem_rel: 0.10 };
            let v = compare::compare(&f, &rf, &gauges, radius, &[10.0, 25.0, 50.0, 100.0], &tol)?;
            let txt = serde_json::to_string_pretty(&v)?;
            if let Some(p) = flag(&a, "--out") {
                std::fs::write(p, &txt)?;
            }
            println!("{txt}");
        }
        Some("render") => {
            let (f, _) = gpkg::read(&need(&a, "--fabric")?)?;
            let width = flag(&a, "--width").map(|s| s.parse()).transpose()?.unwrap_or(1600);
            let proj = laea_of(&f)?;
            let extent = match flag(&a, "--lonlat") {
                Some(s) => {
                    let b = grid::BBox::parse(&s)?;
                    let pts = [(b.west, b.south), (b.east, b.south), (b.east, b.north), (b.west, b.north), ((b.west + b.east) / 2.0, b.north), ((b.west + b.east) / 2.0, b.south)];
                    let mut e = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
                    for (x, y) in pts {
                        let (px, py) = proj.forward(x, y);
                        e = [e[0].min(px), e[1].min(py), e[2].max(px), e[3].max(py)];
                    }
                    Some(e)
                }
                None => None,
            };
            let reference = match flag(&a, "--reference") {
                Some(p) => Some(gpkg::read(&p)?.0),
                None => None,
            };
            let to_ours = |lo: f64, la: f64| proj.forward(lo, la);
            let v = render::render(&f, extent, width, reference.as_ref().map(|r| (r, &to_ours as &dyn Fn(f64, f64) -> (f64, f64))), &need(&a, "--png")?)?;
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
        _ => return Err(USAGE.into()),
    }
    Ok(())
}
