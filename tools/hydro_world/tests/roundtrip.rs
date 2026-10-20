//! Round-trips the CONUS lanes' existing NextGen fabrics through the contract
//! types. Fixtures are real files under HYDRO_WORLD_DATA/fixtures (not bundled: data
//! terms are held separately). Override with HYDRO_WORLD_FIXTURES=path1:path2. A
//! missing fixture fails the test; it never skips.
use hydro_world::{check, contract::*, derive, gpkg, source};
use std::path::PathBuf;

const DEFAULT_FIXTURES: &[&str] = &[
    // case (a): ngen's shipped example at the referee pin be11a180.
    "fixtures/ngen-be11a180565a78ff069d855b0b5dbfc23845e7d8/data/gauge_01073000/gauge_01073000.gpkg",
    // case (b): t-route's shipped LowerColorado_TX HYFeatures fabric (6971 reaches).
    "fixtures/LowerColorado_TX_v4/domain/LowerColorado_NGEN_v201.gpkg",
];

fn fixtures() -> Vec<String> {
    let list = std::env::var("HYDRO_WORLD_FIXTURES")
        .map(|s| s.split(':').map(str::to_owned).collect::<Vec<_>>())
        .unwrap_or_else(|_| {
            let root = std::env::var("HYDRO_WORLD_DATA").unwrap_or_else(|_| "hydro-world-data".into());
            DEFAULT_FIXTURES.iter().map(|s| format!("{root}/{s}")).collect()
        });
    for p in &list {
        assert!(std::path::Path::new(p).is_file(), "fixture missing: {p}");
    }
    list
}

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn conus_fabrics_round_trip_losslessly_through_contract() {
    let dir = scratch("roundtrip");
    for (i, path) in fixtures().iter().enumerate() {
        let (f, read) = gpkg::read(path).unwrap();
        // JSON is the builders' interchange form: exact (f64 bits, geometry bytes).
        let json = serde_json::to_vec(&f).unwrap();
        let back: Fabric = serde_json::from_slice(&json).unwrap();
        assert!(back == f, "{path}: JSON round trip changed the fabric");
        // GeoPackage written with the contract DDL, then read back.
        let out = dir.join(format!("fabric-{i}.gpkg"));
        gpkg::write(&f, out.to_str().unwrap()).unwrap();
        let (g, again) = gpkg::read(out.to_str().unwrap()).unwrap();
        assert!(g == f, "{path}: GeoPackage round trip changed the fabric");
        assert!(again.deviations.is_empty(), "rewritten file deviates: {:?}", again.deviations);
        let r = check::check(&f);
        eprintln!(
            "{path}\n  read: {}\n  check: {}",
            serde_json::to_string(&read).unwrap(),
            serde_json::to_string(&r).unwrap()
        );
        assert!(r.errors.is_empty(), "{path}: topology/ID errors {:?}", r.errors);
    }
}

#[test]
fn canonicalize_yields_canonical_order() {
    let path = &fixtures()[0];
    let (mut f, _) = gpkg::read(path).unwrap();
    derive::canonicalize(&mut f);
    let r = check::check(&f);
    assert!(r.canonical_order, "{:?}", r.contract_gaps);
    assert!(r.errors.is_empty(), "{:?}", r.errors);
}

/// The ngen example CFE packet cat-11223 (data/gauge_01073000/CFE at be11a180).
const NGEN_CAT_11223: &str = "forcing_file=BMI
surface_partitioning_scheme=Schaake
soil_params.depth=2.0[m]
soil_params.b=6.0[]
soil_params.satdk=4.698e-06[m s-1]
soil_params.satpsi=0.141000003[m]
soil_params.slop=0.028126024[m/m]
soil_params.smcmax=0.444680541[m/m]
soil_params.wltsmc=0.046999998[m/m]
soil_params.expon=1.0[]
soil_params.expon_secondary=1.0[]
refkdt=1.0992179
max_gw_storage=0.012863189697[m]
Cgw=1.8e-05[m h-1]
expon=7.0[]
gw_storage=0.35[m/m]
alpha_fc=0.33
soil_storage=0.35[m/m]
K_nash=0.03[]
K_lf=0.01[]
nash_storage=0.0,0.0
num_timesteps=1
verbosity=1
DEBUG=0
giuh_ordinates=0.63,0.37
";

fn parse_ini(s: &str) -> Vec<(String, Vec<String>, String)> {
    s.lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let (k, v) = l.split_once('=').unwrap();
            let (num, unit) = match v.find('[') {
                Some(i) => (&v[..i], &v[i..]),
                None => (v, ""),
            };
            // Numbers compare by f64 bits, text verbatim.
            let vals = num
                .split(',')
                .map(|x| x.parse::<f64>().map(|f| format!("{:016x}", f.to_bits())).unwrap_or_else(|_| x.to_string()))
                .collect();
            (k.to_string(), vals, unit.to_string())
        })
        .collect()
}

#[test]
fn cfe_packet_from_model_attributes_matches_ngen_example() {
    let m = ModelAttributes {
        fid: 1,
        divide_id: Some("cat-11223".into()),
        elevation_mean_m: Some(0.0),
        slope_mean_deg: Some(0.0),
        aspect_mean_deg: Some(0.0),
        centroid_lat_deg: Some(43.15823607770501),
        centroid_lon_deg: Some(-70.95985077145316),
        soil_class: Some(3),
        veg_class: Some(15),
        soil_color: Some(4),
        impervious_fraction: Some(0.0),
        soil_depth_m: Some(2.0),
        soil_b: Some(6.0),
        soil_satdk_m_s: Some(4.698e-06),
        soil_satpsi_m: Some(0.141000003),
        soil_slop: Some(0.028126024),
        soil_smcmax: Some(0.444680541),
        soil_wltsmc: Some(0.046999998),
        refkdt: Some(1.0992179),
        alpha_fc: Some(0.33),
        max_gw_storage_m: Some(0.012863189697),
        cgw_m_h: Some(1.8e-05),
        gw_expon: Some(7.0),
        k_lf: Some(0.01),
        k_nash: Some(0.03),
        giuh_ordinates: Some("0.63,0.37".into()),
    };
    let ini = derive::cfe_ini(&m, derive::CfeInitial { soil_storage: 0.35, gw_storage: 0.35 }).unwrap();
    // The example's obsolete spelling is ignored by the pinned CFE reader.
    // Keep its numbers, but emit the canonical state-carry key.
    let corrected = NGEN_CAT_11223.replace("\nnash_storage=", "\nnash_storage_subsurface=");
    assert_eq!(parse_ini(&ini), parse_ini(&corrected));
    assert!(!ini.lines().any(|line| line.starts_with("nash_storage=")));
    let noah = derive::noahowp_overrides(&m).unwrap();
    assert_eq!(noah["isltyp"], 3);
    assert_eq!(noah["vegtyp"], 15);
}

#[test]
fn source_table_validates_and_licence_law_holds() {
    let t = source::load(concat!(env!("CARGO_MANIFEST_DIR"), "/data/sources.v1.json")).unwrap();
    assert_eq!(t.rows.len(), 4);
    let row = |id: &str| t.rows.iter().find(|r| r.id == id).unwrap();
    // Original CONUS input terms are verified; the physical build recipe and
    // same-network parameters remain separate admission requirements.
    source::admit(row("conus-nhdplus-v21")).unwrap();
    // Open default: every input verified at the source.
    source::admit(row("global-copernicus-glo90")).unwrap();
    source::admit(row("global-copernicus-glo30")).unwrap();
    // --row auto: GLO-30 for a small box, GLO-90 for a large one.
    let small = hydro_world::grid::BBox::parse("-101,30.5,-99,31.5").unwrap();
    let large = hydro_world::grid::BBox::parse("4,46,12,52").unwrap();
    assert_eq!(hydro_world::build::resolve_row(&t, "auto", &small).unwrap(), "global-copernicus-glo30");
    assert_eq!(hydro_world::build::resolve_row(&t, "auto", &large).unwrap(), "global-copernicus-glo90");
    // HydroSHEDS v1 is research-only: default admission always refuses it;
    // the named opt-in is only for a private standalone cross-check.
    let hs = row("global-hydrosheds-v1");
    let e = source::admit(hs).unwrap_err().to_string();
    assert!(e.contains("research-only") && e.contains("never bundled, fetched by default or shipped"), "{e}");
    assert!(e.contains("--accept-licence 'WWF HydroSHEDS v1 License Agreement'"), "{e}");
    assert!(source::admit_with(hs, &["CC BY 4.0".to_string()]).is_err());
    source::admit_with(hs, &["WWF HydroSHEDS v1 License Agreement".to_string()]).unwrap();
}

#[test]
fn schema_and_structs_agree() {
    // Every contract column has a struct field and vice versa.
    fn keys<T: serde::Serialize>(v: &T) -> Vec<String> {
        let mut k = serde_json::to_value(v).unwrap().as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        k.sort();
        k
    }
    let (f, _) = gpkg::read(&fixtures()[0]).unwrap();
    let cols = |n: &str| {
        let mut c = table(n).unwrap().columns.iter().map(|c| c.name.to_string()).collect::<Vec<_>>();
        c.sort();
        c
    };
    assert_eq!(keys(&f.divides[0]), cols("divides"));
    assert_eq!(keys(&f.flowpaths[0]), cols("flowpaths"));
    assert_eq!(keys(&f.nexus[0]), cols("nexus"));
    assert_eq!(keys(&f.network[0]), cols("network"));
    assert_eq!(keys(&f.flowpath_attributes[0]), cols("flowpath_attributes"));
    assert_eq!(keys(&f.hydrolocations[0]), cols("hydrolocations"));
    let lake = Lake { fid: 1, geom: None, id: None, toid: None, hl_id: None, hl_reference: None, hl_link: None, hl_uri: None, dam_length: None, ifd: None, lk_area: None, lk_mx_e: None, orifice_a: None, orifice_c: None, orifice_e: None, time: None, weir_c: None, weir_e: None, weir_l: None };
    assert_eq!(keys(&lake), cols("lakes"));
    let meta = MetadataRow { fid: 1, key: None, value: None };
    assert_eq!(keys(&meta), cols("fabric_metadata"));
}

/// The engine cannot tell a contract-written fabric from its source: run the
/// engine's own `rw_hydrofabric prepare` on both and compare every prepared
/// payload. Needs RW_HYDROFABRIC=/path/to/rw_hydrofabric (built from
/// tools/nextgen_hydrofabric). Run with `cargo test -- --ignored`.
#[test]
#[ignore]
fn engine_prepare_is_identical_on_contract_written_fabric() {
    let exe = std::env::var("RW_HYDROFABRIC").expect("set RW_HYDROFABRIC to the engine preparer");
    let dir = scratch("engine");
    for (i, path) in fixtures().iter().enumerate() {
        let (f, read) = gpkg::read(path).unwrap();
        let rewritten = dir.join(format!("fabric-{i}.gpkg"));
        gpkg::write(&f, rewritten.to_str().unwrap()).unwrap();
        let prepare = |gpkg: &str, tag: &str| -> serde_json::Value {
            let cfg = dir.join(format!("cfg-{i}-{tag}.json"));
            std::fs::write(&cfg, serde_json::to_vec(&serde_json::json!({"geopackage": gpkg, "output": dir.join(format!("out-{i}-{tag}")), "data_license": "contract equivalence test input; terms held by source owner"})).unwrap()).unwrap();
            let o = std::process::Command::new(&exe).arg("prepare").arg(&cfg).output().unwrap();
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
            serde_json::from_slice(&o.stdout).unwrap()
        };
        let a = prepare(path, "source");
        let b = prepare(rewritten.to_str().unwrap(), "contract");
        assert_eq!(a["files"], b["files"], "{path}: prepared payload descriptors differ");
        for name in ["network.json", "parsed_network.tsv", "gauges.json"] {
            assert_eq!(a["auxiliary"][name], b["auxiliary"][name], "{path}: {name} differs");
        }
        // attributes.json carries every column with its JSON type, so it is
        // identical exactly when the source already used the contract types.
        let attrs_equal = a["auxiliary"]["attributes.json"] == b["auxiliary"]["attributes.json"];
        let typed = !read.deviations.keys().any(|d| d.contains("stored"));
        assert_eq!(attrs_equal, typed, "{path}: attributes.json equality {attrs_equal} vs source typing {typed}: {:?}", read.deviations);
        eprintln!("{path}: {} payloads identical; attributes.json identical={attrs_equal}; deviations={:?}", a["files"].as_object().unwrap().len(), read.deviations);
    }
}

/// docs/dev/HYDRO-WORLD-CONTRACT.md section 7 must state every column exactly
/// as the code does, so the document and the contract cannot drift apart.
#[test]
fn contract_document_matches_code() {
    let doc = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/dev/HYDRO-WORLD-CONTRACT.md")).unwrap();
    let esc = |s: &str| s.replace('<', "&lt;").replace('>', "&gt;");
    for t in TABLES {
        assert!(doc.contains(&format!("### {}\n", t.name)), "doc lacks table {}", t.name);
        for c in t.columns {
            let line = format!("| `{}` | {} | {} | {} | {} |", c.name, c.decl, esc(c.unit), if c.nullable { "yes" } else { "no" }, esc(c.meaning));
            assert!(doc.contains(&line), "doc row differs from code: {line}");
        }
    }
    assert!(doc.contains(CONTRACT));
}

/// The builder end to end on the CONUS check box (Copernicus GLO-90 tiles
/// and HydroLAKES under HYDRO_WORLD_DATA, the lab data root by default). A
/// missing input fails, never skips.
#[test]
fn builder_conus_box_conforms_and_round_trips() {
    let root = std::env::var("HYDRO_WORLD_DATA").unwrap_or_else(|_| "hydro-world-data".into());
    let o = hydro_world::build::Options {
        sources: concat!(env!("CARGO_MANIFEST_DIR"), "/data/sources.v1.json").into(),
        row: "global-copernicus-glo90".into(),
        bbox: hydro_world::grid::BBox::parse("-101,30.5,-99,31.5").unwrap(),
        data_root: root.into(),
        accept: vec![],
        params: None,
        parent_grid: None,
        revision: "test".into(),
        // CPU here so the suite runs on any node; the GPU passes are proven
        // equal by gpu_passes_equal_cpu_passes_on_conus_box (GPU node only).
        device: hydro_world::build::Device::Cpu,
    };
    let b = hydro_world::build::build(&o).unwrap();
    let dir = std::env::temp_dir().join(format!("hydro-world-builder-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("conus.gpkg");
    gpkg::write(&b.fabric, out.to_str().unwrap()).unwrap();
    let (back, _) = gpkg::read(out.to_str().unwrap()).unwrap();
    assert!(back == b.fabric, "builder fabric does not round-trip through the GeoPackage");
    let r = check::check(&back);
    assert!(r.errors.is_empty(), "{:?}", r.errors);
    assert!(r.canonical_order);
    // Only gap: model_attributes, owned by hydro-world-params.
    assert_eq!(r.contract_gaps, vec!["model_attributes table absent".to_string()]);
    assert!(back.flowpaths.len() > 900 && back.flowpaths.len() < 1300, "{}", back.flowpaths.len());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn hydrosheds_row_is_refused_without_the_named_opt_in() {
    let o = hydro_world::build::Options {
        sources: concat!(env!("CARGO_MANIFEST_DIR"), "/data/sources.v1.json").into(),
        row: "global-hydrosheds-v1".into(),
        bbox: hydro_world::grid::BBox::parse("4,46,12,52").unwrap(),
        data_root: "/nonexistent".into(),
        accept: vec![],
        params: None,
        parent_grid: None,
        revision: "test".into(),
        device: hydro_world::build::Device::Auto,
    };
    let e = hydro_world::build::build(&o).err().unwrap().to_string();
    assert!(e.contains("research-only") && e.contains("WWF HydroSHEDS v1 License Agreement"), "{e}");
}

/// The GPU per-cell passes equal the CPU passes bit for bit on the CONUS
/// check box (filled surface, D8 directions, accumulated areas). Needs a CUDA
/// device: `cargo test -- --ignored` on a GPU node.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn gpu_passes_equal_cpu_passes_on_conus_box() {
    let root = std::env::var("HYDRO_WORLD_DATA").unwrap_or_else(|_| "hydro-world-data".into());
    let o = hydro_world::build::Options {
        // A GPU node may hold the crate data elsewhere: HYDRO_WORLD_SOURCES overrides.
        sources: std::env::var("HYDRO_WORLD_SOURCES").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/data/sources.v1.json").into()).into(),
        row: "global-copernicus-glo90".into(),
        bbox: hydro_world::grid::BBox::parse("-101,30.5,-99,31.5").unwrap(),
        data_root: root.into(),
        accept: vec![],
        params: None,
        parent_grid: None,
        revision: "test".into(),
        device: hydro_world::build::Device::Cuda,
    };
    let v = hydro_world::build::compare_passes(&o).unwrap();
    assert_eq!(v["identical"], serde_json::json!(true), "{}", v["differing"]);
}
