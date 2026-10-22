fn main() {
    if let Err(e) = run() {
        eprintln!("rw_hydrofabric: {e}");
        std::process::exit(1)
    }
}
fn run() -> nextgen_hydrofabric::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    std::hint::black_box(nextgen_hydrofabric::SOURCE_REVISION);
    match args.get(1).map(String::as_str) {
        Some("--abi") => println!(
            "arwen.nextgen.hydrofabric.v1 {}",
            nextgen_hydrofabric::SOURCE_REVISION
        ),
        Some("replay") => {
            if args.get(2).map(String::as_str) != Some("--strict") {
                return Err(
                    "native parser replay requires explicit --strict MANIFEST NATIVE.tsv".into(),
                );
            }
            let r = nextgen_hydrofabric::replay::compare(
                std::path::Path::new(args.get(3).ok_or("missing MANIFEST")?),
                std::path::Path::new(args.get(4).ok_or("missing NATIVE.tsv")?),
            )?;
            println!("{}", serde_json::to_string_pretty(&r)?);
            if r["pass"] != true {
                std::process::exit(2)
            }
        }
        Some("prepare-render-map") => println!("{}", serde_json::to_string_pretty(
            &nextgen_hydrofabric::render_map::prepare(
                std::path::Path::new(args.get(2).ok_or("missing routing manifest")?),
                std::path::Path::new(args.get(3).ok_or("missing fresh map directory")?))?)?),
        Some("prepare") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::bundle::prepare(
                std::path::Path::new(args.get(2).ok_or("prepare CONFIG.json")?)
            )?)?
        ),
        Some("prepare-cell-overlap") => println!("{}", serde_json::to_string_pretty(
            &nextgen_hydrofabric::cell_overlap::prepare(std::path::Path::new(
                args.get(2).ok_or("prepare-cell-overlap CONFIG.json")?
            ))?
        )?),
        Some("prepare-routing-replay") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::routing_capture::prepare(
                std::path::Path::new(args.get(2).ok_or("prepare-routing-replay CONFIG.json")?)
            )?)?
        ),
        Some("prepare-routing-world") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::world_routing::prepare(
                std::path::Path::new(args.get(2).ok_or("prepare-routing-world CONFIG.json")?)
            )?)?
        ),
        Some("reservoir-initial-from-spinup") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::world_routing::reservoir_initial(
                std::path::Path::new(args.get(2).ok_or("reservoir-initial-from-spinup CONFIG.json")?)
            )?)?
        ),
        Some("replay-collapse") => {
            if args.get(2).map(String::as_str) != Some("--strict") {
                return Err(
                    "replay-collapse requires --strict FABRIC_MANIFEST CAPTURE_MANIFEST OUTPUT"
                        .into(),
                );
            }
            let result = nextgen_hydrofabric::collapse::replay(
                std::path::Path::new(args.get(3).ok_or("missing FABRIC_MANIFEST")?),
                std::path::Path::new(args.get(4).ok_or("missing CAPTURE_MANIFEST")?),
                std::path::Path::new(args.get(5).ok_or("missing OUTPUT")?),
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            if result["pass"] != true {
                std::process::exit(2);
            }
        }
        Some("dump") => {
            let db = nextgen_hydrofabric::sqlite::Db::open(args.get(2).ok_or("dump GPKG TABLE")?)?;
            let rows = db.query(
                &format!(
                    "SELECT * FROM {}",
                    nextgen_hydrofabric::sqlite::ident(args.get(3).ok_or("dump GPKG TABLE")?)
                ),
                Some("geom"),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &rows.into_iter().map(|r| r.properties).collect::<Vec<_>>()
                )?
            );
        }
        Some("inspect-source-dbf") => println!(
            "{}",
            serde_json::to_string_pretty(
                &nextgen_hydrofabric::dbf::Table::open(std::path::Path::new(
                    args.get(2).ok_or("inspect-source-dbf TABLE.dbf")?
                ))?
                .inspect()?
            )?
        ),
        Some("subset-source-network") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::source_network::prepare(
                std::path::Path::new(args.get(2).ok_or("subset-source-network CONFIG.json")?)
            )?)?
        ),
        Some("join-source-waterbodies") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::source_waterbodies::prepare(
                std::path::Path::new(args.get(2).ok_or("join-source-waterbodies CONFIG.json")?)
            )?)?
        ),
        Some("source-waterbody-boundaries") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::source_waterbodies::boundaries(
                std::path::Path::new(
                    args.get(2)
                        .ok_or("source-waterbody-boundaries CONFIG.json")?
                )
            )?)?
        ),
        Some("select-source-domain") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::source_geometry::select_domain(
                std::path::Path::new(args.get(2).ok_or("select-source-domain CONFIG.json")?)
            )?)?
        ),
        Some("audit-source-divides") => println!("{}", serde_json::to_string_pretty(
            &nextgen_hydrofabric::source_divides::audit(std::path::Path::new(
                args.get(2).ok_or("audit-source-divides CONFIG.json")?
            ))?
        )?),
        Some("measure-source-lengths") => println!("{}", serde_json::to_string_pretty(
            &nextgen_hydrofabric::source_lengths::prepare(std::path::Path::new(
                args.get(2).ok_or("measure-source-lengths CONFIG.json")?
            ))?
        )?),
        Some("replay-source-lengths") => {
            if args.get(2).map(String::as_str) != Some("--strict") {
                return Err("length replay requires --strict LENGTHS.json NATIVE.csv".into());
            }
            let r=nextgen_hydrofabric::source_lengths::replay(
                std::path::Path::new(args.get(3).ok_or("missing LENGTHS.json")?),
                std::path::Path::new(args.get(4).ok_or("missing NATIVE.csv")?),
            )?;
            println!("{}",serde_json::to_string_pretty(&r)?);
            if r["pass"]!=true { std::process::exit(2); }
        }
        Some("group-source-polygons") => println!("{}", serde_json::to_string_pretty(
            &nextgen_hydrofabric::source_polygons::prepare(std::path::Path::new(
                args.get(2).ok_or("group-source-polygons CONFIG.json")?
            ))?
        )?),
        Some("replay-source-polygons") => {
            if args.get(2).map(String::as_str) != Some("--strict") {
                return Err("polygon replay requires --strict GROUPED.json NATIVE.csv".into());
            }
            let r = nextgen_hydrofabric::source_polygons::replay(
                std::path::Path::new(args.get(3).ok_or("missing GROUPED.json")?),
                std::path::Path::new(args.get(4).ok_or("missing NATIVE.csv")?),
            )?;
            println!("{}", serde_json::to_string_pretty(&r)?);
            if r["pass"] != true { std::process::exit(2); }
        }
        Some("project-source-geometry") => println!("{}", serde_json::to_string_pretty(
            &nextgen_hydrofabric::source_projection::prepare(std::path::Path::new(
                args.get(2).ok_or("project-source-geometry CONFIG.json")?
            ))?
        )?),
        Some("replay-source-projection") => {
            if args.get(2).map(String::as_str) != Some("--strict") {
                return Err("projection replay requires --strict PROJECTED.json NATIVE.csv".into());
            }
            let r = nextgen_hydrofabric::source_projection::replay(
                std::path::Path::new(args.get(3).ok_or("missing PROJECTED.json")?),
                std::path::Path::new(args.get(4).ok_or("missing NATIVE.csv")?),
            )?;
            println!("{}", serde_json::to_string_pretty(&r)?);
            if r["pass"] != true { std::process::exit(2); }
        }
        Some("subset-source-geometry") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::source_geometry::prepare(
                std::path::Path::new(args.get(2).ok_or("subset-source-geometry CONFIG.json")?)
            )?)?
        ),
        Some("replay-source-geometry") => {
            if args.get(2).map(String::as_str) != Some("--strict") {
                return Err(
                    "replay-source-geometry requires --strict MANIFEST.json NATIVE.csv".into(),
                );
            }
            let r = nextgen_hydrofabric::source_geometry::replay(
                std::path::Path::new(args.get(3).ok_or("missing MANIFEST.json")?),
                std::path::Path::new(args.get(4).ok_or("missing NATIVE.csv")?),
            )?;
            println!("{}", serde_json::to_string_pretty(&r)?);
            if r["pass"] != true {
                std::process::exit(2);
            }
        }
        Some("query-source-dbf") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::dbf::query(std::path::Path::new(
                args.get(2).ok_or("query-source-dbf CONFIG.json")?
            ))?)?
        ),
        Some("replay-source-network") => {
            if args.get(2).map(String::as_str) != Some("--strict") {
                return Err(
                    "replay-source-network requires --strict NETWORK.json NATIVE.csv".into(),
                );
            }
            let result = nextgen_hydrofabric::source_network::replay(
                std::path::Path::new(args.get(3).ok_or("missing NETWORK.json")?),
                std::path::Path::new(args.get(4).ok_or("missing NATIVE.csv")?),
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            if result["pass"] != true {
                std::process::exit(2);
            }
        }
        Some("inspect") => println!(
            "{}",
            serde_json::to_string_pretty(&nextgen_hydrofabric::inspect(
                args.get(2).ok_or("inspect GPKG")?
            )?)?
        ),
        _ => return Err("usage: rw_hydrofabric inspect GPKG".into()),
    }
    Ok(())
}
