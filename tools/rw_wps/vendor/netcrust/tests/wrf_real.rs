use std::path::PathBuf;
use std::{fs::File, io::Read};

use netcrust::{looks_like_hdf5, DataType, NcSliceInfo, NcSliceInfoElem};

fn record_selection(ndim: usize, index: u64) -> NcSliceInfo {
    let mut selections = Vec::with_capacity(ndim);
    selections.push(NcSliceInfoElem::Index(index));
    selections.extend((1..ndim).map(|_| NcSliceInfoElem::Slice {
        start: 0,
        end: u64::MAX,
        step: 1,
    }));
    NcSliceInfo { selections }
}

fn fixture_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("NETCRUST_WRF_FIXTURE").map(PathBuf::from) {
        if path.exists() {
            return Some(path);
        }
    }

    let default =
        PathBuf::from(r"F:\250m_master\20210216_00z_tx_freeze\wrfout_d01_2021-02-16_00_00_00");
    default.exists().then_some(default)
}

#[test]
fn opens_local_wrf_fixture_when_available() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping WRF fixture test; set NETCRUST_WRF_FIXTURE to a wrfout file");
        return;
    };

    let mut header = [0; 8];
    File::open(&path)
        .expect("open fixture")
        .read_exact(&mut header)
        .expect("read fixture signature");
    assert!(looks_like_hdf5(&header));

    let file = netcrust::open(&path).expect("open WRF NetCDF4 file");
    assert!(matches!(
        file.format(),
        netcrust::NcFormat::Nc4 | netcrust::NcFormat::Nc4Classic
    ));

    assert_eq!(file.dimension("Time").expect("Time dim").len(), 1);
    let ny = file
        .dimension("south_north")
        .expect("south_north dim")
        .len();
    let nx = file.dimension("west_east").expect("west_east dim").len();
    assert!(ny >= 100);
    assert!(nx >= 100);

    let dx = file.attribute("DX").and_then(|attr| attr.as_f64());
    let dy = file.attribute("DY").and_then(|attr| attr.as_f64());
    assert!(dx.is_some_and(|value| value > 0.0));
    assert!(dy.is_some_and(|value| value > 0.0));

    let t2 = file.variable("T2").expect("T2 variable");
    assert_eq!(t2.dtype(), &DataType::F32);
    assert_eq!(t2.shape(), vec![1, ny, nx]);

    let t2_values = t2
        .array_f64_first_record_or_all()
        .expect("read first T2 time record");
    assert_eq!(t2_values.shape(), &[ny, nx]);
    assert_eq!(t2_values.len(), ny * nx);
    assert!(t2_values.values().iter().any(|value| value.is_finite()));
}

#[test]
fn reads_core_wrf_metadata_when_available() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping WRF metadata test; set NETCRUST_WRF_FIXTURE to a wrfout file");
        return;
    };

    let file = netcrust::open(&path).expect("open WRF NetCDF4 file");

    for dim in [
        "Time",
        "south_north",
        "west_east",
        "bottom_top",
        "west_east_stag",
        "south_north_stag",
        "bottom_top_stag",
    ] {
        assert!(file.dimension(dim).is_some(), "missing dimension {dim}");
    }

    for attr in [
        "DX",
        "DY",
        "MAP_PROJ",
        "CEN_LAT",
        "CEN_LON",
        "TRUELAT1",
        "TRUELAT2",
        "STAND_LON",
    ] {
        assert!(file.attribute(attr).is_some(), "missing attribute {attr}");
    }

    for var in [
        "XLAT", "XLONG", "PSFC", "T2", "Q2", "U10", "V10", "HGT", "P", "PB", "T", "QVAPOR", "PH",
        "PHB", "U", "V", "W",
    ] {
        assert!(file.variable(var).is_some(), "missing variable {var}");
    }
}

#[test]
fn reads_wrf_four_dimensional_first_record_when_available() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping WRF 4-D test; set NETCRUST_WRF_FIXTURE to a wrfout file");
        return;
    };

    let Ok(metadata) = std::fs::metadata(&path) else {
        eprintln!("skipping WRF 4-D test; fixture metadata unavailable");
        return;
    };
    if metadata.len() > 200_000_000 {
        eprintln!("skipping WRF 4-D test on large fixture {}", path.display());
        return;
    }

    let file = netcrust::open(&path).expect("open WRF NetCDF4 file");
    let nz = file.dimension("bottom_top").expect("bottom_top dim").len();
    let ny = file
        .dimension("south_north")
        .expect("south_north dim")
        .len();
    let nx = file.dimension("west_east").expect("west_east dim").len();

    let qvapor = file.variable("QVAPOR").expect("QVAPOR variable");
    assert_eq!(qvapor.shape(), vec![1, nz, ny, nx]);

    let values = qvapor
        .array_f64_first_record_or_all()
        .expect("read first QVAPOR time record");
    assert_eq!(values.shape(), &[nz, ny, nx]);
    assert_eq!(values.len(), nz * ny * nx);
    assert!(values.values().iter().any(|value| value.is_finite()));
}

#[test]
fn reads_nonzero_wrf_record_with_hdf5_fallback_when_available() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping WRF slice test; set NETCRUST_WRF_FIXTURE to a wrfout file");
        return;
    };
    let file = netcrust::open(&path).expect("open WRF NetCDF4 file");
    let nt = file.dimension("Time").expect("Time dim").len();
    if nt < 2 {
        eprintln!("skipping nonzero-record test; fixture has only {nt} time record(s)");
        return;
    }
    let ny = file
        .dimension("south_north")
        .expect("south_north dim")
        .len();
    let nx = file.dimension("west_east").expect("west_east dim").len();

    // Some real CONUS-II files expose a dataset to the raw HDF5 lookup but
    // omit it from netcdf-reader's metadata index. Such a field proves the
    // arbitrary-record by-name HDF5 route. If this fixture indexes every
    // candidate, still exercise record 1 through the public slice API; the
    // unit test in lib.rs separately proves its Nc -> HDF selection mapping.
    for name in ["U10", "MUCAPE", "SRH03", "SWUPT", "ACEDIR"] {
        if file.variable(name).is_some() {
            continue;
        }
        let Ok(record) = file.read_array_f64_record_or_all(name, 1) else {
            continue;
        };
        assert_eq!(record.shape(), &[ny, nx]);
        assert_eq!(record.len(), ny * nx);
        assert!(record.values().iter().any(|value| value.is_finite()));
        let sliced = file
            .read_array_f64_slice(name, &record_selection(3, 1))
            .unwrap_or_else(|err| panic!("HDF5 fallback slice for {name}: {err}"));
        assert_eq!(sliced, record);
        return;
    }

    let t2 = file.variable("T2").expect("T2 variable");
    let record = file
        .read_array_f64_slice("T2", &record_selection(t2.ndim(), 1))
        .expect("read nonzero T2 record");
    assert_eq!(record.shape(), &t2.shape()[1..]);
    assert!(record.values().iter().any(|value| value.is_finite()));
}
