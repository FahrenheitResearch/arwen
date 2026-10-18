//! Native-free cold vertical contract, composed from the shipping static producer.
//! MPAS v8.4.2 core_init_atmosphere/mpas_init_atm_cases.F:3025-3443;
//! operators/mpas_vector_operations.F:652-771;
//! operators/mpas_vector_reconstruction.F:61-191;
//! core_init_atmosphere/Registry.xml:1088-1103,1286-1312,1398-1400.
//! This adapter reuses the complete numerical modules and keeps their words in
//! Rust through NetCDF emission. Native output files are never producer inputs.
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    path::PathBuf,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    static_path: PathBuf,
    spec_path: PathBuf,
    output: PathBuf,
    start_time: String,
    nfglevels: usize,
    nsoillevels: usize,
    regional_halo: bool,
    sphere: bool,
    periodic: bool,
    x_period_m: f32,
    y_period_m: f32,
}

/// Run the complete static-to-vertical driver boundary using explicit options.
/// Native mpas_init_atm_cases.F:3025-3443 and Registry.xml:1088-1103,1286-1312,1398-1400.
pub fn run(input: impl Read, mut output: impl Write) -> Result<(), String> {
    let r: Request = serde_json::from_reader(input).map_err(|e| e.to_string())?;
    let spec: Value =
        serde_json::from_slice(&std::fs::read(&r.spec_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let file = netcrust::File::open(&r.static_path).map_err(|e| e.to_string())?;
    let read = |name: &str| -> Result<Vec<f64>, String> {
        Ok(file
            .read_array_f64(name)
            .map_err(|e| e.to_string())?
            .into_values())
    };
    let dim = |name: &str| -> Result<usize, String> {
        file.dimension(name)
            .map(|d| d.len())
            .ok_or_else(|| format!("missing dimension {name}"))
    };
    let option = |name: &str| -> Result<Value, String> {
        spec.get(name)
            .cloned()
            .ok_or_else(|| format!("vertical spec must explicitly declare {name}"))
    };
    let nc = dim("nCells")?;
    let ne = dim("nEdges")?;
    let me = dim("maxEdges")?;
    let nz = option("n_vert_levels")?
        .as_u64()
        .ok_or("invalid n_vert_levels")? as usize;
    let indices = |name: &str| -> Result<Vec<i64>, String> {
        Ok(read(name)?.iter().map(|&v| v as i64 - 1).collect())
    };
    let rows = |name: &str| -> Result<Vec<Vec<i64>>, String> {
        Ok(indices(name)?
            .chunks_exact(me)
            .map(|v| v.to_vec())
            .collect())
    };
    let deriv = read("deriv_two")?;
    if ne == 0 || me == 0 || deriv.is_empty() || deriv.len() % (ne * 2) != 0 {
        return Err("deriv_two does not match the declared edge stencil".into());
    }
    let stride = deriv.len() / (ne * 2);
    let mut request = json!({"mode":"grid","width":4,"terrain":read("ter")?,
        "counts":read("nEdgesOnCell")?.iter().map(|&v|v as usize).collect::<Vec<_>>(),
        "edges_on_cell":rows("edgesOnCell")?,"cells_on_cell":rows("cellsOnCell")?,
        "cells_on_edge":indices("cellsOnEdge")?.chunks_exact(2).map(|v|[v[0],v[1]]).collect::<Vec<_>>(),
        "dc":read("dcEdge")?,"dv":read("dvEdge")?,"area":read("areaCell")?,"levels":nz,
        "top":option("ztop_m")?,"scheme":option("scheme")?,"specified":spec.get("specified_interfaces_m"),
        "projection":option("interface_projection")?,"terrain_passes":option("terrain_smoothing_passes")?,
        "smooth":option("smooth_surfaces")?,"surface_passes":option("surface_smoothing_passes")?,
        "minimum_fraction":option("minimum_layer_fraction")?,"hybrid":option("hybrid_coordinate")?,
        "transition":option("hybrid_transition_height_m")?,"damping":option("rayleigh_xnutr")?,
        "damping_start":option("rayleigh_damping_start_m")?,"regional_halo":r.regional_halo,"order":option("theta_adv_order")?,
        "deriv":deriv.chunks_exact(stride*2).map(|v|vec![v[..stride].to_vec(),v[stride..].to_vec()]).collect::<Vec<_>>(),"zgrid":[]});
    let mut bytes = Vec::new();
    super::vertical::run(
        serde_json::to_vec(&request)
            .map_err(|e| e.to_string())?
            .as_slice(),
        &mut bytes,
    )?;
    let words: Vec<f32> = bytes[..bytes.len() - 8]
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let mut variables = Vec::new();
    let mut field = |name: &str, dims: &[&str], values: Vec<f32>| {
        variables.push(json!({"name":name,"dimensions":dims,"values":values}));
    };
    let mut offset = 0;
    for (name, size, stored) in [
        ("zw", nz + 1, false),
        ("dzw", nz, false),
        ("rdzw", nz, true),
        ("zu", nz, false),
        ("dzu", nz, true),
        ("rdzu", nz, true),
        ("rdzwp", nz, false),
        ("rdzwm", nz, false),
        ("fzp", nz, true),
        ("fzm", nz, true),
        ("ah", nz + 1, false),
    ] {
        if stored {
            let mut values = words[offset..offset + size].to_vec();
            if name != "rdzw" {
                values[0] = 0.;
            }
            field(name, &["nVertLevels"], values);
        }
        offset += size;
    }
    let hx = &words[offset..offset + nc * (nz + 1)];
    field("ter", &["nCells"], hx[..nc].to_vec());
    offset += hx.len();
    let mut level_grid = Vec::new();
    for (name, levels, count, entity, level_dim) in [
        ("zgrid", nz + 1, nc, "nCells", "nVertLevelsP1"),
        ("zz", nz, nc, "nCells", "nVertLevels"),
        ("zxu", nz, ne, "nEdges", "nVertLevels"),
        ("dss", nz, nc, "nCells", "nVertLevels"),
    ] {
        let raw = &words[offset..offset + levels * count];
        let mut values = Vec::with_capacity(raw.len());
        for c in 0..count {
            for k in 0..levels {
                values.push(raw[k * count + c]);
            }
        }
        if name == "zgrid" {
            level_grid = raw
                .chunks_exact(nc)
                .map(|v| v.iter().map(|&x| x as f64).collect::<Vec<_>>())
                .collect::<Vec<_>>();
        }
        field(name, &[entity, level_dim], values);
        offset += raw.len();
    }
    for name in ["cf1", "cf2", "cf3"] {
        field(name, &[], vec![words[offset]]);
        offset += 1;
    }
    if offset != words.len() {
        return Err("vertical protocol output size changed".into());
    }
    request["zgrid"] = json!(level_grid);
    request["mode"] = json!("edge");
    bytes.clear();
    super::vertical::run(
        serde_json::to_vec(&request)
            .map_err(|e| e.to_string())?
            .as_slice(),
        &mut bytes,
    )?;
    let words: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    offset = 0;
    for name in ["zb", "zb3"] {
        let mut values = Vec::with_capacity(ne * 2 * (nz + 1));
        for e in 0..ne {
            for side in 0..2 {
                for k in 0..nz + 1 {
                    values.push(words[offset + (k * 2 + side) * ne + e]);
                }
            }
        }
        field(name, &["nEdges", "TWO", "nVertLevelsP1"], values);
        offset += ne * 2 * (nz + 1);
    }
    let mut input = Vec::new();
    input.extend_from_slice(b"HEXGEO1\0");
    for size in [nc, ne, me, nc] {
        input.extend_from_slice(&(size as u64).to_le_bytes());
    }
    input.extend_from_slice(&[4, u8::from(r.sphere), u8::from(r.periodic), 0]);
    input.extend_from_slice(&r.x_period_m.to_le_bytes());
    input.extend_from_slice(&r.y_period_m.to_le_bytes());
    for (names, count) in [
        (["xCell", "yCell", "zCell"], nc),
        (["xEdge", "yEdge", "zEdge"], ne),
    ] {
        let xyz = names.map(read);
        let xyz = xyz.into_iter().collect::<Result<Vec<_>, _>>()?;
        for i in 0..count {
            for axis in &xyz {
                input.extend_from_slice(&(axis[i] as f32).to_le_bytes());
            }
        }
    }
    for name in ["cellsOnEdge", "edgesOnCell"] {
        for value in indices(name)? {
            input.extend_from_slice(&value.to_le_bytes());
        }
    }
    for value in read("nEdgesOnCell")? {
        input.extend_from_slice(&(value as i64).to_le_bytes());
    }
    bytes.clear();
    crate::geometry::run(input.as_slice(), &mut bytes)?;
    let words: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    offset = 0;
    for (name, size, dims) in [
        ("edgeNormalVectors", ne * 3, vec!["nEdges", "R3"]),
        ("localVerticalUnitVectors", nc * 3, vec!["nCells", "R3"]),
        ("cellTangentPlane", nc * 6, vec!["nCells", "TWO", "R3"]),
        (
            "coeffs_reconstruct",
            nc * me * 3,
            vec!["nCells", "maxEdges", "R3"],
        ),
    ] {
        field(name, &dims, words[offset..offset + size].to_vec());
        offset += size;
    }
    // Native met-stage output declarations. Zero state is generated here;
    // the case-7 emitter replaces computed weather fields and records its ledger.
    // Registry.xml:1088-1103,1286-1312,1398-1400.
    for name in [
        "precipw",
        "q2",
        "rh2",
        "seaice",
        "sfc_albbck",
        "skintemp",
        "snow",
        "snowc",
        "snowh",
        "sst",
        "surface_pressure",
        "t2m",
        "tmn",
        "u10",
        "v10",
        "vegfra",
        "xice",
        "xland",
        "h_oml_initial",
    ] {
        field(name, &["Time", "nCells"], vec![]);
    }
    for name in [
        "qc",
        "qr",
        "qv",
        "relhum",
        "rho",
        "rho_base",
        "theta",
        "theta_base",
        "tke",
    ] {
        field(name, &["Time", "nCells", "nVertLevels"], vec![]);
    }
    for name in ["dzs", "sh2o", "smois", "tslb", "zs", "dz"] {
        field(name, &["Time", "nCells", "nSoilLevels"], vec![]);
    }
    field("u", &["Time", "nEdges", "nVertLevels"], vec![]);
    field("w", &["Time", "nCells", "nVertLevelsP1"], vec![]);
    for name in ["u_init", "v_init", "qv_init"] {
        field(name, &["nVertLevels"], vec![]);
    }
    field("t_init", &["nCells", "nVertLevels"], vec![]);
    drop(field);
    // An omitted payload requests codec-owned zero initialization.
    for variable in &mut variables {
        if variable["values"].as_array().is_some_and(Vec::is_empty) {
            variable.as_object_mut().unwrap().remove("values");
        }
    }
    variables.push(json!({"name":"Time","dimensions":["Time"],"values":[0.],"attributes":{"units":format!("seconds since {}",r.start_time.replace('_'," "))}}));
    variables.push(json!({"name":"initial_time","dimensions":["StrLen"],"text":""}));
    let attributes = BTreeMap::from([
        ("config_start_time", json!(r.start_time)),
        ("config_nvertlevels", json!(nz)),
        ("config_nfglevels", json!(r.nfglevels)),
        ("config_ztop", option("ztop_m")?),
        (
            "gpuwm_hex_vertical_source",
            json!("constructed-no-native-runtime-input"),
        ),
    ]);
    let codec = json!({"template":r.static_path,"output":r.output,"dimensions":{"Time":null,"nVertLevels":nz,"nVertLevelsP1":nz+1,"TWO":2,"R3":3,"nSoilLevels":r.nsoillevels,"StrLen":64},"variables":variables,"attributes":attributes});
    let receipt =
        super::artifact::materialize(serde_json::from_value(codec).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    serde_json::to_writer(&mut output, &receipt).map_err(|e| e.to_string())
}
