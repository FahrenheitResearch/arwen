//! Exercise the actual rw_mpas_static binary on signed, haloed WPS tiles.
//! The expected ten-field words come from pristine MPAS Fortran, including
//! pole-crossing boxes. No geography or field processing lives in Python.
use rw_mpas::native_gwd::{self, NX, NativeGwdConfig, NativeMesh};
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};
use std::collections::BTreeSet;
use std::path::Path;

fn cells() -> ([f32; 6], [f32; 6]) {
    (
        [0.0, 0.002, 0.4, -0.4, 1.5707, -1.5707],
        [0.0, -3.1415, 1.3, -1.7, 0.02, -2.6],
    )
}
fn grid(path: &Path) {
    let (lat, lon) = cells();
    let dims = vec![
        NcDim::fixed("nCells", 6),
        NcDim::fixed("nEdges", 6),
        NcDim::fixed("maxEdges", 1),
    ];
    let variables = vec![
        NcVarDef::new("latCell", NcType::Float, vec![0]),
        NcVarDef::new("lonCell", NcType::Float, vec![0]),
        NcVarDef::new("dcEdge", NcType::Float, vec![1]),
        NcVarDef::new("nEdgesOnCell", NcType::Int, vec![0]),
        NcVarDef::new("edgesOnCell", NcType::Int, vec![0, 2]),
        NcVarDef::new("cellsOnCell", NcType::Int, vec![0, 2]),
    ];
    let mut w = NcClassicWriter::create(
        path,
        NcFormat::Offset64,
        dims,
        vec![NcAttr::floats("sphere_radius", vec![native_gwd::RE])],
        variables,
        0,
    )
    .unwrap();
    w.put("latCell", NcData::Floats(&lat)).unwrap();
    w.put("lonCell", NcData::Floats(&lon)).unwrap();
    w.put("dcEdge", NcData::Floats(&[5000.0; 6])).unwrap();
    w.put("nEdgesOnCell", NcData::Ints(&[1; 6])).unwrap();
    w.put("edgesOnCell", NcData::Ints(&[1, 2, 3, 4, 5, 6]))
        .unwrap();
    w.put("cellsOnCell", NcData::Ints(&[2, 1, 4, 3, 6, 5]))
        .unwrap();
    w.finish().unwrap();
}
fn fixture(root: &Path, shift: i64, topo_name: &str) {
    let tdir = root.join(topo_name);
    let ldir = root.join("landuse_30s");
    std::fs::create_dir_all(&tdir).unwrap();
    std::fs::create_dir_all(&ldir).unwrap();
    let shared = "projection=regular_ll\ndx=0.00833333\ndy=0.00833333\nknown_x=1\nknown_y=1\nknown_lat=-89.995833\ntile_x=1200\ntile_y=1200\ntile_z=1\nendian=big\nscale_factor=1\n";
    std::fs::write(
        tdir.join("index"),
        format!(
            "{shared}type=continuous\nwordsize=2\nsigned=yes\ntile_bdr=3\nknown_lon={}\n",
            if shift == 0 { -179.995833 } else { 0.004166667 }
        ),
    )
    .unwrap();
    // The published NCAR USGS index has no iswater metadata. Native's
    // compiled WATER=16 still governs masks and smoothing.
    std::fs::write(ldir.join("index"),format!("{shared}type=categorical\nwordsize=1\nsigned=yes\ntile_bdr=0\nknown_lon=-179.995833\ncategory_min=1\ncategory_max=24\nmminlu=USGS\n")).unwrap();
    let (lat, lon) = cells();
    let mut topo_tiles = BTreeSet::new();
    let mut land_tiles = BTreeSet::new();
    for c in 0..6 {
        let (nx, ny) = native_gwd::box_size(lat[c], 5000.0);
        let (ic, jc) = native_gwd::centre(lat[c], lon[c]);
        for j in 1..=ny {
            for i in 1..=nx {
                let (ii, jj) = native_gwd::wrap(i - nx / 2 + ic, j - ny / 2 + jc);
                let topo_i = (ii - shift - 1).rem_euclid(NX) + 1;
                topo_tiles.insert(((topo_i - 1) / 1200 * 1200 + 1, (jj - 1) / 1200 * 1200 + 1));
                land_tiles.insert(((ii - 1) / 1200 * 1200 + 1, (jj - 1) / 1200 * 1200 + 1));
            }
        }
    }
    for (xs, ys) in topo_tiles {
        let mut bytes = Vec::with_capacity(1206 * 1206 * 2);
        for j in -3..1203 {
            for i in -3..1203 {
                let (ii, jj) = native_gwd::wrap(xs + i + shift, ys + j);
                let v = ((ii * 37 + jj * 101) % 6000 - 1000) as i16;
                bytes.extend_from_slice(&v.to_be_bytes());
            }
        }
        std::fs::write(
            tdir.join(format!("{xs:05}-{:05}.{ys:05}-{:05}", xs + 1199, ys + 1199)),
            bytes,
        )
        .unwrap();
    }
    for (xs, ys) in land_tiles {
        let mut bytes = Vec::with_capacity(1200 * 1200);
        for j in 0..1200 {
            for i in 0..1200 {
                let (ii, jj) = (xs + i, ys + j);
                bytes.push(if (ii * 3 + jj * 5) % 8 < 3 { 16 } else { 1 });
            }
        }
        std::fs::write(
            ldir.join(format!("{xs:05}-{:05}.{ys:05}-{:05}", xs + 1199, ys + 1199)),
            bytes,
        )
        .unwrap();
    }
}
#[test]
#[ignore = "needs MPAS_GWD_ORACLE and MPAS_GWD_TEST_DIR; writes disposable native-layout tiles"]
fn native_gwd_real_binary_matches_fortran_and_is_band_invariant() {
    let scratch = std::path::PathBuf::from(
        std::env::var("MPAS_GWD_TEST_DIR").expect("explicit lane scratch required"),
    );
    assert!(!scratch.exists(), "test scratch must be new");
    std::fs::create_dir_all(&scratch).unwrap();
    let grid_path = scratch.join("grid.nc");
    grid(&grid_path);
    let oracle = std::fs::read(format!(
        "{}.tiles",
        std::env::var("MPAS_GWD_ORACLE").unwrap()
    ))
    .unwrap();
    assert_eq!(oracle.len(), 6 * 10 * 4);
    let expected: Vec<u32> = oracle
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let binary = std::env::var("MPAS_GWD_BINARY")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_rw_mpas_static").into());
    let names = [
        "var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4",
    ];
    let mut all_outputs = Vec::new();
    for (source, folder, shift) in [
        ("GTOPO30", "topo_30s", 0),
        ("GMTED2010", "topo_gmted2010_30s", NX / 2),
    ] {
        let geog = scratch.join(source);
        fixture(&geog, shift, folder);
        let out = scratch.join(format!("{source}.nc"));
        let receipt = scratch.join(format!("{source}.json"));
        let status = std::process::Command::new(&binary)
            .args([
                "--grid",
                grid_path.to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
                "--geog",
                geog.to_str().unwrap(),
                "--native-gwd-static",
                "--gwd-only",
                "--topo-data",
                source,
                "--receipt",
                receipt.to_str().unwrap(),
            ])
            .env("RW_MPAS_TILE_WORKERS", "2")
            .status()
            .unwrap();
        assert!(status.success(), "native binary {source}");
        let f = netcrust::File::open(&out).unwrap();
        let mut output = Vec::new();
        for (k, name) in names.iter().enumerate() {
            let values = f.read_f64(name).unwrap();
            assert_eq!(values.len(), 6);
            for (c, &v) in values.iter().enumerate() {
                let word = (v as f32).to_bits();
                assert_eq!(word, expected[c * 10 + k], "{source} cell {c} {name}");
                output.push(word);
            }
        }
        assert_eq!(f.variables().unwrap().len(), 10);
        all_outputs.push(output);
        let cfg = NativeGwdConfig::under(&geog, source, 1.0).unwrap();
        let mesh = NativeMesh::read(&grid_path, 1.0).unwrap();
        let floor = mesh.minimum_band_rows().unwrap();
        let narrow = native_gwd::compute(&cfg, &mesh, floor, &|_| {}).unwrap();
        let wide = native_gwd::compute(&cfg, &mesh, 768, &|_| {}).unwrap();
        assert!(narrow.bands > wide.bands);
        assert_eq!(narrow.var2d, wide.var2d);
        assert_eq!(narrow.con, wide.con);
        assert_eq!(narrow.oa, wide.oa);
        assert_eq!(narrow.ol, wide.ol);
        // An explicit budget cannot be silently bypassed by the native pass.
        let constrained = NativeGwdConfig {
            host_memory_limit_bytes: Some(1024 * 1024),
            ..cfg.clone()
        };
        let refusal = native_gwd::compute(&constrained, &mesh, floor, &|_| {})
            .unwrap_err()
            .to_string();
        assert!(
            refusal.contains("host memory admission refused"),
            "{refusal}"
        );
        // A deliberately wrong terrain-frame shift must actually change results.
        let wrong = NativeGwdConfig {
            topo_shift: (shift + 1200).rem_euclid(NX),
            ..cfg.clone()
        };
        assert!(native_gwd::compute(&wrong, &mesh, floor, &|_| {}).is_err()); // missing antipodal tile is detected
    }
    assert_eq!(
        all_outputs[0], all_outputs[1],
        "the geography source frame moved native fields"
    );
    std::fs::remove_dir_all(&scratch).unwrap();
    println!(
        "NATIVE_GWD_DOOR fields=120 mismatches=0 source_frames=2 band_grants=2 signed_terrain=true halo=3 polar_boxes=true scratch_removed=true"
    );
}
