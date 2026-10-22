// Native-format bridge fixture. This is a file-format test, not forecast evidence.
use rw_store::netcdf_classic::{
    NcClassicWriter as Writer, NcData, NcDim, NcFormat, NcType, NcVarDef,
};
use std::path::Path;
fn main() {
    let root = std::env::args().nth(1).unwrap();
    let root = Path::new(&root);
    let cells: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let pressure = [
        95000., 85000., 70000., 50000., 40000., 30000., 25000., 20000., 10000.,
    ];
    let nz = pressure.len();
    let dims = vec![
        NcDim::record("Time"),
        NcDim::fixed("nCells", cells),
        NcDim::fixed("nVertLevels", nz),
        NcDim::fixed("nVertLevelsP1", nz + 1),
    ];
    let volume = [
        "theta",
        "pressure",
        "qv",
        "uReconstructZonal",
        "uReconstructMeridional",
    ];
    let surface = ["surface_pressure", "u10", "v10", "ter", "t2m", "q2", "tsk"];
    let mut vars = vec![];
    for name in volume {
        vars.push(NcVarDef::new(name, NcType::Float, vec![0, 1, 2]));
    }
    for name in surface {
        vars.push(NcVarDef::new(name, NcType::Float, vec![0, 1]));
    }
    let p = root.join("history.2020-01-01_00.00.00.nc");
    let mut w = Writer::create(&p, NcFormat::Data64, dims.clone(), vec![], vars, 1).unwrap();
    for name in volume {
        let values: Vec<f32> = (0..cells)
            .flat_map(|_| {
                (0..nz).map(move |k| match name {
                    "pressure" => pressure[k],
                    "theta" => 300. + k as f32 * 8.,
                    "qv" => 0.001,
                    "uReconstructZonal" => 5.,
                    _ => 2.,
                })
            })
            .collect();
        w.put_record(name, 0, NcData::Floats(&values)).unwrap();
    }
    for name in surface {
        let v = match name {
            "surface_pressure" => 100000.,
            "u10" => 5.,
            "v10" => 2.,
            "ter" => 0.,
            "q2" => 0.01,
            _ => 290.,
        };
        w.put_record(name, 0, NcData::Floats(&vec![v; cells]))
            .unwrap();
    }
    w.finish().unwrap();
    let vars = vec![
        NcVarDef::new("zgrid", NcType::Float, vec![1, 3]),
        NcVarDef::new("landmask", NcType::Float, vec![1]),
    ];
    let mut w = Writer::create(
        &root.join("init.nc"),
        NcFormat::Data64,
        dims,
        vec![],
        vars,
        0,
    )
    .unwrap();
    let z: Vec<f32> = (0..cells)
        .flat_map(|_| (0..=nz).map(|k| k as f32 * 2000.))
        .collect();
    w.put("zgrid", NcData::Floats(&z)).unwrap();
    w.put("landmask", NcData::Floats(&vec![0.; cells])).unwrap();
    w.finish().unwrap();
}
