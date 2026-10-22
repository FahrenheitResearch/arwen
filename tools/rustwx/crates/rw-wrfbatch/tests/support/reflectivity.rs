//! Identical history input for the heating and converted-nowcast panel tests.
use std::path::Path;
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

pub const NX: usize = 24;
pub const NY: usize = 18;
pub const NZ: usize = 4;

/// The composite reflectivity every fixture holds at cell (y, x).
pub fn composite(_y: usize, x: usize) -> f32 {
    5.0 + 2.0 * x as f32
}

/// A wrfout-shaped file valid 2026-08-19 01Z from a 00Z start, on the
/// regular 0.05 degree lattice from 36 N, 98 W; its REFL_10CM column
/// maximum is `composite`.
pub fn write_history(path: &Path) {
    let cells = NX * NY;
    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let bt = schema.def_dim("bottom_top", NZ, false).unwrap();
    let bts = schema.def_dim("bottom_top_stag", NZ + 1, false).unwrap();
    let sn = schema.def_dim("south_north", NY, false).unwrap();
    let we = schema.def_dim("west_east", NX, false).unwrap();
    for (name, value) in [("START_DATE", "2026-08-19_00:00:00"), ("SIMULATION_START_DATE", "2026-08-19_00:00:00")] {
        schema.put_global_attr(name, AttrValue::Text(value.into())).unwrap();
    }
    for (name, value) in [("MAP_PROJ", 1i32), ("GRID_ID", 1)] {
        schema.put_global_attr(name, AttrValue::Ints(vec![value])).unwrap();
    }
    for (name, value) in [("DX", 3000.0f32), ("DY", 3000.0), ("TRUELAT1", 38.0), ("TRUELAT2", 38.0),
        ("STAND_LON", -95.0), ("CEN_LAT", 38.0), ("CEN_LON", -95.0)] {
        schema.put_global_attr(name, AttrValue::Floats(vec![value])).unwrap();
    }
    let times = schema.def_var("Times", NcType::Char, &[time, strlen]).unwrap();
    let var = |schema: &mut Schema, name: &str, dims: &[usize], units: &str| {
        let id = schema.def_var(name, NcType::Float, dims).unwrap();
        schema.put_var_attr(id, "units", AttrValue::Text(units.into())).unwrap();
        id
    };
    let xlat = var(&mut schema, "XLAT", &[time, sn, we], "degree_north");
    let xlong = var(&mut schema, "XLONG", &[time, sn, we], "degree_east");
    let t2 = var(&mut schema, "T2", &[time, sn, we], "K");
    let t = var(&mut schema, "T", &[time, bt, sn, we], "K");
    let qv = var(&mut schema, "QVAPOR", &[time, bt, sn, we], "kg kg-1");
    let qr = var(&mut schema, "QRAIN", &[time, bt, sn, we], "kg kg-1");
    let qs = var(&mut schema, "QSNOW", &[time, bt, sn, we], "kg kg-1");
    let qg = var(&mut schema, "QGRAUP", &[time, bt, sn, we], "kg kg-1");
    let refl = var(&mut schema, "REFL_10CM", &[time, bt, sn, we], "dBZ");
    let w = var(&mut schema, "W", &[time, bts, sn, we], "m s-1");
    let hgt = var(&mut schema, "HGT", &[time, sn, we], "m");
    let ph = var(&mut schema, "PH", &[time, bts, sn, we], "m2 s-2");
    let phb = var(&mut schema, "PHB", &[time, bts, sn, we], "m2 s-2");
    let mut writer = NcWriter::create(path, schema).unwrap();
    writer.write_record(0, times, VarData::Char(b"2026-08-19_01:00:00")).unwrap();
    let plane = |f: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
        (0..NY).flat_map(|y| (0..NX).map(move |x| (y, x))).map(|(y, x)| f(y, x)).collect()
    };
    writer.write_record(0, xlat, VarData::F32(&plane(&|y, _| 36.0 + 0.05 * y as f32))).unwrap();
    writer.write_record(0, xlong, VarData::F32(&plane(&|_, x| -98.0 + 0.05 * x as f32))).unwrap();
    writer.write_record(0, t2, VarData::F32(&vec![295.0; cells])).unwrap();
    let volume: Vec<f32> = (0..NZ)
        .flat_map(|k| (0..NY).flat_map(move |y| (0..NX).map(move |x| (k, y, x))))
        .map(|(k, y, x)| if k == 1 { composite(y, x) } else { composite(y, x) - 20.0 })
        .collect();
    writer.write_record(0, refl, VarData::F32(&volume)).unwrap();
    writer.write_record(0, t, VarData::F32(&vec![10.0; cells * NZ])).unwrap();
    writer.write_record(0, qv, VarData::F32(&vec![0.01; cells * NZ])).unwrap();
    for id in [qr, qs, qg] {
        writer.write_record(0, id, VarData::F32(&vec![0.0; cells * NZ])).unwrap();
    }
    writer.write_record(0, w, VarData::F32(&vec![0.0; cells * (NZ + 1)])).unwrap();
    writer.write_record(0, hgt, VarData::F32(&vec![320.0; cells])).unwrap();
    writer.write_record(0, ph, VarData::F32(&vec![0.0; cells * (NZ + 1)])).unwrap();
    let phb_values: Vec<f32> = (0..NZ + 1)
        .flat_map(|k| std::iter::repeat_n(9.81 * (320.0 + 1_000.0 * k as f32), cells))
        .collect();
    writer.write_record(0, phb, VarData::F32(&phb_values)).unwrap();
    writer.finish().unwrap();
}
