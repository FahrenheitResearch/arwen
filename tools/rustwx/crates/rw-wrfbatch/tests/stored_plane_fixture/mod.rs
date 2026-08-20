//! A minimal, wrfout-shaped classic NetCDF file written by
//! `netcdf-writer`, used by the stored-plane tests.
//!
//! It is deliberately NOT a physics-plausible forecast: it carries the
//! smallest set of variables `wrf-core` needs to open a raw wrfout (`T`
//! for the dimension probe, `Times` for the time axis, `XLAT`/`XLONG`
//! for the grid) plus a handful of ordinary surface planes and ONE plane
//! no catalog in the tree knows about.  That last plane is the point:
//! it stands for a variable a user added to their own WRF Registry, and
//! the `var:` product has to reach it without a line of new product
//! code.

use std::path::{Path, PathBuf};

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

/// The wrfout variable no product catalog in this tree knows about.
pub const USER_PLANE: &str = "MSLP_ANOM";

/// The store name the generic `var:` route must be able to name.
pub const USER_PLANE_STORE_NAME: &str = "wrf_mslp_anom";

/// Units carried on [`USER_PLANE`], so the panel legend can be checked.
pub const USER_PLANE_UNITS: &str = "Pa";

pub const NX: usize = 24;
pub const NY: usize = 18;
pub const NZ: usize = 4;

/// Value at cell `(y, x)` of the user plane; a deterministic ramp, so a
/// reader can prove it read THIS plane and not some neighbour's.
pub fn user_plane_value(y: usize, x: usize) -> f32 {
    (y * NX + x) as f32 - 120.0
}

/// Write the fixture as `wrfout_d01_<stamp>` inside `dir`, returning its path.
pub fn write(dir: &Path) -> PathBuf {
    let path = dir.join("wrfout_d01_2026-08-19_00_00_00");
    let cells = NX * NY;
    let volume = cells * NZ;

    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let bottom_top = schema.def_dim("bottom_top", NZ, false).unwrap();
    let bottom_top_stag = schema.def_dim("bottom_top_stag", NZ + 1, false).unwrap();
    let south_north = schema.def_dim("south_north", NY, false).unwrap();
    let south_north_stag = schema.def_dim("south_north_stag", NY + 1, false).unwrap();
    let west_east = schema.def_dim("west_east", NX, false).unwrap();
    let west_east_stag = schema.def_dim("west_east_stag", NX + 1, false).unwrap();

    for (name, value) in [
        ("TITLE", " OUTPUT FROM WRF V4.6.1 MODEL"),
        ("START_DATE", "2026-08-19_00:00:00"),
        ("SIMULATION_START_DATE", "2026-08-19_00:00:00"),
        ("GRIDTYPE", "C"),
    ] {
        schema
            .put_global_attr(name, AttrValue::Text(value.into()))
            .unwrap();
    }
    for (name, value) in [("MAP_PROJ", 1i32), ("GRID_ID", 1), ("PARENT_ID", 0)] {
        schema
            .put_global_attr(name, AttrValue::Ints(vec![value]))
            .unwrap();
    }
    for (name, value) in [
        ("DX", 3000.0f32),
        ("DY", 3000.0),
        ("TRUELAT1", 38.0),
        ("TRUELAT2", 38.0),
        ("STAND_LON", -95.0),
        ("CEN_LAT", 38.0),
        ("CEN_LON", -95.0),
        ("POLE_LAT", 90.0),
        ("POLE_LON", 0.0),
    ] {
        schema
            .put_global_attr(name, AttrValue::Floats(vec![value]))
            .unwrap();
    }

    let times = schema.def_var("Times", NcType::Char, &[time, strlen]).unwrap();

    let surface = |schema: &mut Schema, name: &str, units: &str| {
        let id = schema
            .def_var(name, NcType::Float, &[time, south_north, west_east])
            .unwrap();
        schema
            .put_var_attr(id, "units", AttrValue::Text(units.into()))
            .unwrap();
        schema
            .put_var_attr(id, "description", AttrValue::Text(name.into()))
            .unwrap();
        id
    };
    let xlat = surface(&mut schema, "XLAT", "degree_north");
    let xlong = surface(&mut schema, "XLONG", "degree_east");
    let t2 = surface(&mut schema, "T2", "K");
    let q2 = surface(&mut schema, "Q2", "kg kg-1");
    let psfc = surface(&mut schema, "PSFC", "Pa");
    let hgt = surface(&mut schema, "HGT", "m");
    let u10 = surface(&mut schema, "U10", "m s-1");
    let v10 = surface(&mut schema, "V10", "m s-1");
    // Identity grid rotation.  Without these two, every earth-relative
    // wind diagnostic (`uvmet10` and everything downstream of it) fails
    // and the file can draw no wind layer at all -- which is exactly the
    // layer the streamline front door has to be provable against.
    let sinalpha = surface(&mut schema, "SINALPHA", "1");
    let cosalpha = surface(&mut schema, "COSALPHA", "1");
    let user = surface(&mut schema, USER_PLANE, USER_PLANE_UNITS);

    let volume_var = |schema: &mut Schema, name: &str, dims: &[usize], units: &str| {
        let id = schema.def_var(name, NcType::Float, dims).unwrap();
        schema
            .put_var_attr(id, "units", AttrValue::Text(units.into()))
            .unwrap();
        id
    };
    let t = volume_var(
        &mut schema,
        "T",
        &[time, bottom_top, south_north, west_east],
        "K",
    );
    let p = volume_var(
        &mut schema,
        "P",
        &[time, bottom_top, south_north, west_east],
        "Pa",
    );
    let pb = volume_var(
        &mut schema,
        "PB",
        &[time, bottom_top, south_north, west_east],
        "Pa",
    );
    let qvapor = volume_var(
        &mut schema,
        "QVAPOR",
        &[time, bottom_top, south_north, west_east],
        "kg kg-1",
    );
    let ph = volume_var(
        &mut schema,
        "PH",
        &[time, bottom_top_stag, south_north, west_east],
        "m2 s-2",
    );
    let phb = volume_var(
        &mut schema,
        "PHB",
        &[time, bottom_top_stag, south_north, west_east],
        "m2 s-2",
    );
    let u = volume_var(
        &mut schema,
        "U",
        &[time, bottom_top, south_north, west_east_stag],
        "m s-1",
    );
    let v = volume_var(
        &mut schema,
        "V",
        &[time, bottom_top, south_north_stag, west_east],
        "m s-1",
    );

    let mut lat = Vec::with_capacity(cells);
    let mut lon = Vec::with_capacity(cells);
    let mut t2_values = Vec::with_capacity(cells);
    let mut user_values = Vec::with_capacity(cells);
    for y in 0..NY {
        for x in 0..NX {
            lat.push(36.0 + 0.05 * y as f32);
            lon.push(-98.0 + 0.05 * x as f32);
            t2_values.push(295.0 + 0.1 * (x + y) as f32);
            user_values.push(user_plane_value(y, x));
        }
    }
    let q2_values = vec![0.010f32; cells];
    let psfc_values = vec![97_000.0f32; cells];
    let hgt_values = vec![320.0f32; cells];
    let u10_values = vec![6.0f32; cells];
    let v10_values = vec![-4.0f32; cells];
    let sinalpha_values = vec![0.0f32; cells];
    let cosalpha_values = vec![1.0f32; cells];

    // Perturbation potential temperature is stored as theta - 300 K.
    let theta_perturbation = vec![10.0f32; volume];
    let qvapor_values = vec![0.008f32; volume];
    let mut base_pressure = Vec::with_capacity(volume);
    let mut pressure_perturbation = Vec::with_capacity(volume);
    for level in 0..NZ {
        let base = 97_000.0 - 12_000.0 * level as f32;
        for _ in 0..cells {
            base_pressure.push(base);
            pressure_perturbation.push(0.0);
        }
    }
    let mut base_geopotential = Vec::with_capacity(cells * (NZ + 1));
    for level in 0..=NZ {
        let value = 9.81 * (320.0 + 1_000.0 * level as f32);
        for _ in 0..cells {
            base_geopotential.push(value);
        }
    }
    let geopotential_perturbation = vec![0.0f32; cells * (NZ + 1)];
    let u_values = vec![7.0f32; (NX + 1) * NY * NZ];
    let v_values = vec![-3.0f32; NX * (NY + 1) * NZ];

    let mut writer = NcWriter::create(&path, schema).unwrap();
    writer
        .write_record(0, times, VarData::Char(b"2026-08-19_00:00:00"))
        .unwrap();
    for (id, values) in [
        (xlat, &lat),
        (xlong, &lon),
        (t2, &t2_values),
        (q2, &q2_values),
        (psfc, &psfc_values),
        (hgt, &hgt_values),
        (u10, &u10_values),
        (v10, &v10_values),
        (sinalpha, &sinalpha_values),
        (cosalpha, &cosalpha_values),
        (user, &user_values),
    ] {
        writer
            .write_record(0, id, VarData::F32(values.as_slice()))
            .unwrap();
    }
    for (id, values) in [
        (t, &theta_perturbation),
        (p, &pressure_perturbation),
        (pb, &base_pressure),
        (qvapor, &qvapor_values),
        (ph, &geopotential_perturbation),
        (phb, &base_geopotential),
        (u, &u_values),
        (v, &v_values),
    ] {
        writer
            .write_record(0, id, VarData::F32(values.as_slice()))
            .unwrap();
    }
    writer.finish().unwrap();
    path
}
