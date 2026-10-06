//! Tool routes must read arrays larger than the former 128 Mi-value limit.
//! Files are generated as sparse CDF-5 fixtures and removed after each test.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use rw_mpas::init::emit::write_init;
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};

const ROWS: usize = 8_193;
const COLUMNS: usize = 16_384;
const ELEMENTS: usize = ROWS * COLUMNS;
const FIRST: f32 = 24.125;
const LAST: f32 = -1.25;

struct Scratch(PathBuf);

impl Scratch {
    fn new(tool: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let serial = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rw-mpas-large-{tool}-{}-{serial}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("unique fixture directory");
        assert!(ELEMENTS > 134_217_728);
        Self(path)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        fn remove_created(path: &Path) {
            for entry in fs::read_dir(path).expect("fixture inventory") {
                let entry = entry.expect("fixture entry");
                let kind = entry.file_type().expect("fixture type");
                if kind.is_dir() {
                    remove_created(&entry.path());
                } else {
                    let bytes = entry.metadata().expect("fixture size").len();
                    println!(
                        "fixture_deleted {} {bytes}",
                        entry.file_name().to_string_lossy()
                    );
                    fs::remove_file(entry.path()).expect("remove generated fixture");
                }
            }
            fs::remove_dir(path).expect("remove generated fixture directory");
        }
        remove_created(&self.0);
    }
}

/// The last fixed variable is deliberately left as a zero-filled sparse
/// extent. The fixture supplies its endpoint bytes directly, so generating
/// the file never allocates a dense array. All other declared variables are
/// written normally. This constructs test data, not a production writer path.
fn sparse_last_variable(
    path: &Path,
    dims: Vec<NcDim>,
    attrs: Vec<NcAttr>,
    vars: Vec<NcVarDef>,
    small: &[(&str, NcData<'_>)],
    first: [u8; 4],
    last: [u8; 4],
) {
    let records = u64::from(dims.iter().any(|dimension| dimension.unlimited));
    let mut writer = NcClassicWriter::create(path, NcFormat::Data64, dims, attrs, vars, records)
        .expect("sparse CDF-5 layout");
    for (name, data) in small {
        writer.put(name, *data).expect("small fixture variable");
    }
    let size = writer.file_len();
    drop(writer);
    let start = size
        .checked_sub((ELEMENTS as u64) * 4)
        .expect("last variable offset");
    let mut file = OpenOptions::new()
        .write(true)
        .open(path)
        .expect("fixture endpoint writer");
    file.seek(SeekFrom::Start(start)).unwrap();
    file.write_all(&first).unwrap();
    file.seek(SeekFrom::Start(size - 4)).unwrap();
    file.write_all(&last).unwrap();
    file.sync_all().unwrap();
}

fn run(binary: &str, args: &[&str]) -> Output {
    Command::new(binary)
        .args(args)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("GPUWM_NO_LOCAL_GPU", "1")
        .output()
        .expect("built tool executes")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn rw_mpas_init_carries_every_float_bit_above_the_former_cap() {
    let scratch = Scratch::new("init-emit");
    let capsule = scratch.file("capsule.nc");
    let output = scratch.file("init.nc");
    sparse_last_variable(
        &capsule,
        vec![
            NcDim::record("Time"),
            NcDim::fixed("rows", ROWS),
            NcDim::fixed("columns", COLUMNS),
        ],
        vec![],
        vec![NcVarDef::new("u_init", NcType::Float, vec![1, 2])],
        &[],
        FIRST.to_be_bytes(),
        LAST.to_be_bytes(),
    );
    let ledger = write_init(
        &output,
        &capsule,
        BTreeMap::new(),
        "largearray",
        &[],
        "large array fixture",
        &rw_mpas::init::Lineage::default(),
    )
    .expect("the init emitter carries an above-cap float array");
    assert_eq!(ledger.carried, vec!["u_init".to_string()]);
    let file = netcrust::File::open(&output).unwrap();
    assert_eq!(file.variable("u_init").unwrap().shape(), [ROWS, COLUMNS]);
    let values = file.read_array::<f32>("u_init").unwrap();
    assert_eq!(values.len(), ELEMENTS);
    for (index, &value) in values.iter().enumerate() {
        let expected = if index == 0 {
            FIRST
        } else if index == ELEMENTS - 1 {
            LAST
        } else {
            0.0
        };
        assert_eq!(value.to_bits(), expected.to_bits(), "value {index}");
    }
}

#[test]
fn rw_mpas_init_reads_above_cap_connectivity_before_reporting_its_shape() {
    let scratch = Scratch::new("init-cli");
    let statics = scratch.file("static.nc");
    sparse_last_variable(
        &statics,
        vec![
            NcDim::fixed("nCells", 3),
            NcDim::fixed("nEdges", 1),
            NcDim::fixed("maxEdges", 3),
            NcDim::fixed("rows", ROWS),
            NcDim::fixed("columns", COLUMNS),
        ],
        vec![],
        vec![NcVarDef::new("cellsOnEdge", NcType::Int, vec![3, 4])],
        &[],
        1i32.to_be_bytes(),
        3i32.to_be_bytes(),
    );
    let result = run(
        env!("CARGO_BIN_EXE_rw_mpas_init"),
        &[
            "--met",
            "unused-met",
            "--static",
            statics.to_str().unwrap(),
            "--capsule",
            "unused-capsule",
            "--reference",
            "unused-reference",
            "--out",
            scratch.file("unused-out.nc").to_str().unwrap(),
            "--start-time",
            "2026-01-01_00:00:00",
            "--nfglevels",
            "2",
            "--nfgsoillevels",
            "4",
            "--extrap-airtemp",
            "constant",
            "--use-spechumd",
            "yes",
            "--theta-adv-order",
            "2",
            "--coef-3rd-order",
            "0.25",
            "--virtual-factor",
            "consistent",
            "--deep-soil-moisture",
            "corrected",
            "--landuse-table",
            "USGS",
            "--frac-seaice",
            "no",
            "--tsk-seaice-threshold",
            "100",
            "--oned-underflow",
            "preserve",
        ],
    );
    assert!(!result.status.success());
    let message = stderr(&result);
    assert!(
        message.contains(&format!(
            "cellsOnEdge holds {ELEMENTS} value(s) for 1 edges"
        )),
        "{message}"
    );
    assert!(!message.contains("supported maximum"), "{message}");
}

#[test]
fn rw_mpas_mesh_reads_above_cap_centres_before_reporting_component_mismatch() {
    let scratch = Scratch::new("mesh-cli");
    let input = scratch.file("centres.nc");
    sparse_last_variable(
        &input,
        vec![
            NcDim::fixed("small", 3),
            NcDim::fixed("rows", ROWS),
            NcDim::fixed("columns", COLUMNS),
        ],
        vec![NcAttr::doubles("sphere_radius", vec![1.0])],
        vec![
            NcVarDef::new("yCell", NcType::Float, vec![0]),
            NcVarDef::new("zCell", NcType::Float, vec![0]),
            NcVarDef::new("xCell", NcType::Float, vec![1, 2]),
        ],
        &[
            ("yCell", NcData::Floats(&[0.0; 3])),
            ("zCell", NcData::Floats(&[1.0; 3])),
        ],
        FIRST.to_be_bytes(),
        LAST.to_be_bytes(),
    );
    let result = run(
        env!("CARGO_BIN_EXE_rw_mpas_mesh"),
        &["--from-centres", input.to_str().unwrap(), "--dry-run"],
    );
    assert!(!result.status.success());
    let message = stderr(&result);
    assert!(
        message.contains(&format!("{ELEMENTS} xCell, 3 yCell and 3 zCell values")),
        "{message}"
    );
    assert!(!message.contains("supported maximum"), "{message}");
}

#[test]
fn rw_mpas_convert_writes_a_small_window_after_reading_an_above_cap_profile() {
    let scratch = Scratch::new("convert-cli");
    let history = scratch.file("history.2026-01-01_00.00.00.nc");
    let out_dir = scratch.file("out");
    let lat = vec![0.0f64; ROWS];
    let lon = vec![0.0f64; ROWS];
    let area = vec![1.0f64; ROWS];
    let rain = vec![1.25f32; ROWS];
    sparse_last_variable(
        &history,
        vec![
            NcDim::fixed("nCells", ROWS),
            NcDim::fixed("nVertLevels", COLUMNS),
        ],
        vec![NcAttr::doubles("sphere_radius", vec![1.0])],
        vec![
            NcVarDef::new("latCell", NcType::Double, vec![0]),
            NcVarDef::new("lonCell", NcType::Double, vec![0]),
            NcVarDef::new("areaCell", NcType::Double, vec![0]),
            NcVarDef::new("rainnc", NcType::Float, vec![0]),
            NcVarDef::new("theta", NcType::Float, vec![0, 1]),
        ],
        &[
            ("latCell", NcData::Doubles(&lat)),
            ("lonCell", NcData::Doubles(&lon)),
            ("areaCell", NcData::Doubles(&area)),
            ("rainnc", NcData::Floats(&rain)),
        ],
        FIRST.to_be_bytes(),
        LAST.to_be_bytes(),
    );
    let result = run(
        env!("CARGO_BIN_EXE_rw_mpas_convert"),
        &[
            "--history",
            history.to_str().unwrap(),
            "--mesh",
            history.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--field-set",
            "surface",
            "--window",
            "lambert:0,0,1000,2,2,30,60,0",
            "--format",
            "cdf5",
        ],
    );
    assert!(result.status.success(), "{}", stderr(&result));
    let output = out_dir.join("wrfout_d01_2026-01-01_00_00_00");
    let file = netcrust::File::open(&output).unwrap();
    assert_eq!(file.dimension("bottom_top").unwrap().len(), COLUMNS);
    assert_eq!(file.variable("T").unwrap().shape(), [1, COLUMNS, 2, 2]);
    let actual = file.read_array::<f32>("RAINNC").unwrap();
    assert!(!actual.is_empty());
    assert!(actual
        .iter()
        .all(|&value| value.to_bits() == 1.25f32.to_bits()));
}

#[test]
fn rw_mpas_init_answers_the_declared_contract_without_inputs() {
    let result = run(env!("CARGO_BIN_EXE_rw_mpas_init"), &["--abi"]);
    assert!(result.status.success(), "{}", stderr(&result));
    let contract = String::from_utf8(result.stdout).unwrap();
    assert!(contract.starts_with("rw_mpas_init --met MET --static STATIC.nc --capsule CAPSULE.nc"));
    assert!(result.stderr.is_empty());
}
