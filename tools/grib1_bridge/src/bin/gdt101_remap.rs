//! Native GDT-101 remapping executable. Cargo discovers this binary in the
//! existing grib1_bridge package; no new decoder fork or third-party
//! dependency. Nothing here names a model and nothing here compiles in one
//! producer's code numbers: the caller supplies the mesh identity (cell
//! count and originating centre), the target window, and the seven selector
//! octets of EVERY record it wants read -- the three the plan is built from
//! (latitude, longitude, land fraction) exactly as much as the fields the
//! plan is applied to. Coordinate records are numbered in local tables, so
//! compiling a coordinate parameter number in here would be compiling in a
//! producer.
#[path = "../gdt101_remap_core.rs"]
mod core101;
use core101::{Plan, Target, Method, Result};
use grib_core::grib2::{Grib2File, unpack_message};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

const MAX_OBJECT: u64 = 256 * 1024 * 1024;
/// The mesh a caller declares: how many cells it has, and which centre
/// publishes it. Both are table data on the caller's side.
#[derive(Clone, Copy, Debug)]
struct Mesh { cells: usize, centre: u16 }
#[derive(Clone, Copy, Debug)]
struct Spec { discipline: u8, category: u8, parameter: u8,
              first_type: u8, first: f64, second_type: u8, second: f64 }
struct Decoded { raw: Vec<u8>, grid: Vec<u8>, values: Vec<f64> }
fn same_value(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a-b).abs() <= 1e-9 * b.abs().max(1.)
}
fn decode(path: &Path, mesh: Mesh, spec: Spec, cycle: &str, lead_seconds: u64) -> Result<Decoded> {
    let metadata = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if metadata.len() > MAX_OBJECT { return Err("native source object exceeds the expansion bound".into()); }
    let raw = fs::read(path).map_err(|e| e.to_string())?;
    let envelope = core101::envelope(&raw)?;
    let section3 = envelope.sections[3].unwrap();
    let grid = section3.to_vec();
    if section3.len() < 30
        || u16::from_be_bytes(section3[12..14].try_into().unwrap()) != 101
        || u32::from_be_bytes(section3[6..10].try_into().unwrap()) as usize != mesh.cells
    {
        return Err(format!("expected a GDT 101 unstructured mesh of {} cells", mesh.cells));
    }
    let parsed = Grib2File::from_bytes(&raw).map_err(|e| e.to_string())?;
    if parsed.messages.len() != 1 { return Err("a source object must contain exactly one field".into()); }
    let msg = &parsed.messages[0];
    let p = &msg.product;
    if msg.identification.center_id != mesh.centre || p.template != 0
        || msg.identification.processed_data_type > 2
        || msg.grid.template != 101 || msg.grid.num_data_points as usize != mesh.cells
    {
        return Err(format!(
            "not a deterministic centre-{} product on the declared {}-cell mesh",
            mesh.centre, mesh.cells));
    }
    // chrono is vendored without default features here, so there is no
    // strftime formatter: read the fields the Datelike/Timelike traits give.
    if !same_reference_time(&msg.reference_time, cycle)? {
        return Err(format!("reference-time mismatch in {}", path.display()));
    }
    let multiplier: u64 = match p.time_range_unit {
        0 => 60, 1 => 3600, 2 => 86400, 10 => 10800,
        11 => 21600, 12 => 43200, 13 => 1,
        _ => return Err(format!("unsupported forecast time unit {}", p.time_range_unit)),
    };
    if (p.forecast_time as u64).checked_mul(multiplier) != Some(lead_seconds) {
        return Err(format!("forecast lead mismatch in {}", path.display()));
    }
    if (msg.discipline, p.parameter_category, p.parameter_number, p.level_type, p.second_level_type)
        != (spec.discipline, spec.category, spec.parameter, spec.first_type, spec.second_type)
        // An unbounded surface (type 1, no second surface) has no
        // meaningful numeric height. Match the packaged selector: do not
        // require a zero where a producer may encode the missing sentinel.
        // Pressure, height, soil, and HSURF's bounded layer stay exact.
        || ((spec.first_type != 1 || spec.second_type != 255)
            && !same_value(p.level_value, spec.first))
        || (spec.second_type != 255 && !same_value(p.second_level_value, spec.second))
    {
        return Err(format!("field or vertical-level metadata disagrees with the requested object: {}", path.display()));
    }
    // No call to grid_latlon: GDT-101 coordinates are external by definition.
    // The existing decoder can unpack the native values by Section-3 count.
    let values = unpack_message(msg).map_err(|e| e.to_string())?;
    if values.len() != mesh.cells { return Err("unpacked native point count differs from the grid identity".into()); }
    Ok(Decoded { raw, grid, values })
}
/// Compare a decoded reference time against a ``YYYYMMDDHH`` cycle token.
fn same_reference_time(reference: &chrono::NaiveDateTime, cycle: &str) -> Result<bool> {
    use chrono::{Datelike, Timelike};
    if cycle.len() != 10 || !cycle.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("cycle must be YYYYMMDDHH, not {cycle:?}"));
    }
    let number = |a: usize, b: usize| cycle[a..b].parse::<u32>().unwrap();
    Ok(reference.year() == number(0, 4) as i32
        && reference.month() == number(4, 6)
        && reference.day() == number(6, 8)
        && reference.hour() == number(8, 10)
        && reference.minute() == 0
        && reference.second() == 0)
}

/// One record's seven selector values, in the order a job row spells them:
/// discipline, category, parameter, first level type, first level value,
/// second level type, second level value.
fn spec_from(values: &[&str]) -> Result<Spec> {
    if values.len() != 7 {
        return Err("a selector is seven values: discipline, category, parameter, first level type, first level value, second level type, second level value".into());
    }
    Ok(Spec { discipline: token(values[0],"discipline")?, category: token(values[1],"category")?,
              parameter: token(values[2],"parameter")?, first_type: token(values[3],"first level type")?,
              first: token(values[4],"first level value")?, second_type: token(values[5],"second level type")?,
              second: token(values[6],"second level value")? })
}
/// A selector as one command-line token: the same seven values, comma
/// separated. The coordinate and land-fraction records are numbered by the
/// producer -- the coordinates in a local table, in every unstructured
/// publication read so far -- so which codes to look for is the caller's
/// declaration here exactly as it is for every remapped field.
fn selector(text: &str) -> Result<Spec> {
    let values: Vec<&str> = text.split(',').collect();
    spec_from(&values)
}
fn plan(args: &[String]) -> Result<()> {
    if args.len() != 11 {
        return Err("plan requires LATITUDE LAT_SELECTOR LONGITUDE LON_SELECTOR LAND_FRACTION LAND_SELECTOR TARGET.txt PLAN.bin CYCLE CELLS CENTRE".into());
    }
    let mesh = Mesh { cells: token(&args[9], "mesh cell count")?,
                      centre: token(&args[10], "originating centre")? };
    if !(4..=core101::MAX_SOURCE).contains(&mesh.cells) {
        return Err("declared mesh cell count is outside the supported envelope".into());
    }
    let lat = decode(Path::new(&args[0]), mesh, selector(&args[1])?, &args[8],0)?;
    let lon = decode(Path::new(&args[2]), mesh, selector(&args[3])?, &args[8],0)?;
    let land = decode(Path::new(&args[4]), mesh, selector(&args[5])?, &args[8],0)?;
    if lat.grid != lon.grid || lat.grid != land.grid {
        return Err("the coordinate and land-fraction records have different full GDT-101 identities/UUIDs".into());
    }
    // The GRIB coordinate parameters are DEGREES; a mesh file's NetCDF
    // radians are not interchangeable with them, and converting on a guess
    // from small magnitudes would silently relocate the whole grid. A radian
    // array cannot leave the +/-pi/2 band, so requiring the latitudes to
    // reach past it refuses one by measurement rather than by suspicion.
    let min_lat = lat.values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_lat = lat.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_lon = lon.values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_lon = lon.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(min_lat.is_finite() && max_lat.is_finite() && min_lon.is_finite() && max_lon.is_finite())
        || min_lat < -90. || max_lat > 90. || min_lon < -360. || max_lon > 360.
        || max_lat.max(-min_lat) <= std::f64::consts::FRAC_PI_2
    {
        return Err("the coordinate records are not geographical degrees reaching past the radian band".into());
    }
    let target_text = fs::read_to_string(&args[6]).map_err(|e|e.to_string())?;
    let target = Target::parse(&target_text)?;
    let p = core101::build_plan(&lat.values, &lon.values, &land.values, lat.grid.clone(),
                                mesh.centre, target, 80_000.)?;
    let file = OpenOptions::new().write(true).create_new(true).open(&args[7]).map_err(|e|e.to_string())?;
    p.write(BufWriter::new(&file))?;
    file.sync_all().map_err(|e|e.to_string())?;
    Ok(())
}
fn token<T: std::str::FromStr>(s: &str, label: &str) -> Result<T> {
    s.parse().map_err(|_|format!("invalid {label} in remap job"))
}
/// WMO Code Table 4.5 fixes what a level type MEANS, and two of its
/// entries constrain any producer's choice of remapping. Type 100 is an
/// isobaric surface in the free atmosphere, where a cell's land/water class
/// says nothing about the value, so a class-aware method would choose its
/// donors by a property the field does not have. Type 106 is a depth below
/// the LAND surface, published only where there is land, so a method free
/// to average a water column would read the missing sentinel as data.
/// Everything else -- which surface field is nearest-neighbour and which is
/// interpolated -- is the caller's declaration, not this binary's.
fn level_type_admits(first_type: u8, method: Method) -> bool {
    match first_type {
        100 => matches!(method, Method::Idw4 | Method::Nearest),
        106 => method == Method::SoilNearest,
        _ => true,
    }
}
fn level_type_reason(first_type: u8) -> &'static str {
    match first_type {
        100 => "an isobaric surface has no land/water class to restrict donors by",
        106 => "a depth below the land surface is published on land only",
        _ => "the level admits any implemented method",
    }
}
fn job(plan: &Plan, mesh: Mesh, line: &str) -> Result<()> {
    let p: Vec<_> = line.split('\t').collect();
    if p.len() != 12 { return Err("a remap job must have input, output, seven selector values, method, cycle and lead seconds".into()); }
    let spec = spec_from(&p[2..9])?;
    let method = Method::parse(p[9])?;
    if !level_type_admits(spec.first_type, method) {
        return Err(format!(
            "remap method {} cannot be used on level type {}: {}; declare a method the level admits",
            p[9], spec.first_type, level_type_reason(spec.first_type)));
    }
    let decoded = decode(Path::new(p[0]),mesh,spec,p[10],token(p[11],"forecast seconds")?)?;
    if decoded.grid != plan.source_grid {
        return Err("native field grid identity/UUID does not match the remapping plan".into());
    }
    let values = core101::apply(plan,&decoded.values,method)?;
    let output = core101::encode_regular(&decoded.raw,plan.target,&values)?;
    // The caller stages in a temporary directory and publishes atomically.
    // Never overwrite an arbitrary destination supplied in a job.
    let mut file = OpenOptions::new().write(true).create_new(true).open(p[1]).map_err(|e|e.to_string())?;
    file.write_all(&output).map_err(|e|e.to_string())?;
    file.sync_all().map_err(|e|e.to_string())?;
    Ok(())
}
fn apply(args: &[String]) -> Result<()> {
    if args.len() != 2 { return Err("apply requires PLAN.bin JOBS.tsv".into()); }
    let plan_file = File::open(&args[0]).map_err(|e|e.to_string())?;
    if plan_file.metadata().map_err(|e|e.to_string())?.len() > 128 * 1024 * 1024 {
        return Err("oversized remap plan file".into());
    }
    let p = Plan::read(BufReader::new(plan_file))?;
    // The plan IS the mesh identity: it was built from the coordinate records
    // of this exact grid, and every field is compared to its Section 3 bytes.
    let mesh = Mesh { cells: p.source_count, centre: p.centre };
    let jobs = File::open(&args[1]).map_err(|e|e.to_string())?;
    if jobs.metadata().map_err(|e|e.to_string())?.len() > 16 * 1024 * 1024 {
        return Err("oversized remap job list".into());
    }
    let mut count=0;
    for line in BufReader::new(jobs).lines() {
        let line=line.map_err(|e|e.to_string())?;
        if line.is_empty() { return Err("blank remap job row".into()); }
        job(&p,mesh,&line)?;count+=1;
        if count>10_000 {return Err("too many remap jobs".into());}
    }
    if count==0 {return Err("empty remap job list".into());}
    Ok(())
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--contract") if args.len()==1 => {println!("{}",core101::CONTRACT);Ok(())},
        Some("plan") => plan(&args[1..]),
        Some("apply") => apply(&args[1..]),
        _ => Err(format!("{}\nusage: gdt101_remap plan LATITUDE LAT_SELECTOR LONGITUDE LON_SELECTOR LAND_FRACTION LAND_SELECTOR TARGET PLAN CYCLE CELLS CENTRE\n       gdt101_remap apply PLAN JOBS\n       a SELECTOR is discipline,category,parameter,first_level_type,first_level_value,second_level_type,second_level_value",core101::CONTRACT)),
    }
}
fn main() {
    // Keep the release owner's build.rs stamp in the executable even under
    // LTO/section stripping, from the crate's one declaration of it rather
    // than a second copy of the same concat. Dirty development builds may
    // say "unknown"; the release bundle gate, not the scientific decoder,
    // enforces the pin.
    let _ = std::hint::black_box(gpuwm_preprocess_cpu::SOURCE_REV_STAMP);
    if let Err(error)=run() {eprintln!("gdt101_remap: {error}");std::process::exit(2);}
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    fn section(number: u8, length: usize) -> Vec<u8> {
        let mut bytes = vec![0; length];
        bytes[..4].copy_from_slice(&(length as u32).to_be_bytes());
        bytes[4] = number;
        bytes
    }
    // A synthetic WMO-valid constant field on an unstructured mesh, with
    // every identity octet a caller declares left as an argument. No
    // fixture claims to be an observed download; the test exercises the
    // shared decoder.
    fn constant_record(count: usize, value: f32, discipline: u8, category: u8,
                       parameter: u8, level_type: u8, level_value: u32) -> Vec<u8> {
        let mut s1 = section(1, 21);
        s1[5..7].copy_from_slice(&78u16.to_be_bytes());
        s1[9] = 30;
        s1[12..14].copy_from_slice(&2026u16.to_be_bytes());
        s1[14] = 9; s1[15] = 15; s1[20] = 1;
        let mut s3 = section(3, 35);
        s3[6..10].copy_from_slice(&(count as u32).to_be_bytes());
        s3[12..14].copy_from_slice(&101u16.to_be_bytes());
        let mut s4 = section(4, 34);
        s4[9] = category; s4[10] = parameter;
        s4[11] = 2; s4[17] = 1; s4[22] = level_type;
        s4[24..28].copy_from_slice(&level_value.to_be_bytes());
        s4[28] = 255; s4[29..34].fill(255);
        let mut s5 = section(5, 21);
        s5[5..9].copy_from_slice(&(count as u32).to_be_bytes());
        s5[11..15].copy_from_slice(&value.to_be_bytes());
        let mut s6 = section(6, 6); s6[5] = 255;
        let mut bytes = vec![0; 16];
        bytes[..4].copy_from_slice(b"GRIB"); bytes[6] = discipline; bytes[7] = 2;
        for s in [s1, s3, s4, s5, s6, section(7, 5)] { bytes.extend(s); }
        bytes.extend(b"7777"); let n = bytes.len() as u64;
        bytes[8..16].copy_from_slice(&n.to_be_bytes());
        bytes
    }
    fn constant_native(count: usize) -> Vec<u8> {
        constant_record(count, 285., 0, 0, 0, 100, 85000)
    }
    fn target() -> Target {
        Target { west: -100., south: 30., dx: 0.125, dy: 0.125, nx: 2, ny: 2 }
    }
    fn mesh(cells: usize) -> Mesh { Mesh { cells, centre: 78 } }
    fn spec() -> Spec {
        Spec { discipline: 0, category: 0, parameter: 0,
               first_type: 100, first: 85000., second_type: 255, second: 0. }
    }
    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new(bytes: &[u8]) -> Self {
            let tick = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let path = std::env::temp_dir().join(format!("gdt101-record-{}-{tick}.grib2", std::process::id()));
            let mut file = OpenOptions::new().create_new(true).write(true).open(&path).unwrap();
            file.write_all(bytes).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture { fn drop(&mut self) { let _ = fs::remove_file(&self.0); } }

    fn scratch(name: &str) -> std::path::PathBuf {
        let tick = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_nanos();
        std::env::temp_dir().join(format!("gdt101-{name}-{}-{tick}", std::process::id()))
    }
    /// The three plan records are addressed by the selectors the CALLER
    /// declares. These code numbers are not the ones the shipped document
    /// uses -- coordinates on a producer-local category, land fraction on a
    /// different discipline entirely -- so a binary that compiled any of
    /// them in could not build this plan at all.
    #[test]
    fn a_plan_reads_coordinate_records_numbered_the_producer_s_own_way() {
        let lat = Fixture::new(&constant_record(4, 30., 0, 200, 7, 1, 0));
        let lon = Fixture::new(&constant_record(4, -100., 0, 200, 8, 1, 0));
        let land = Fixture::new(&constant_record(4, 1., 1, 3, 9, 1, 0));
        let target_path = scratch("target.txt");
        fs::write(&target_path, "-100 30 0.125 0.125 2 2").unwrap();
        let plan_path = scratch("weights.bin");
        let args: Vec<String> = [
            lat.0.to_str().unwrap(), "0,200,7,1,0,255,0",
            lon.0.to_str().unwrap(), "0,200,8,1,0,255,0",
            land.0.to_str().unwrap(), "1,3,9,1,0,255,0",
            target_path.to_str().unwrap(), plan_path.to_str().unwrap(),
            "2026091500", "4", "78",
        ].iter().map(|s| s.to_string()).collect();
        plan(&args).expect("a plan over caller-declared coordinate records");
        let built = Plan::read(BufReader::new(File::open(&plan_path).unwrap())).unwrap();
        assert_eq!((built.source_count, built.centre), (4, 78));
        // The same three files with one selector changed are not found.
        let mut wrong = args.clone();
        wrong[1] = "0,191,1,1,0,255,0".to_string();
        wrong[7] = scratch("weights-2.bin").to_str().unwrap().to_string();
        assert!(plan(&wrong).is_err());
        let _ = fs::remove_file(&target_path);
        let _ = fs::remove_file(&plan_path);
    }
    /// Code Table 4.5, not a field list: an isobaric level refuses a
    /// land-class method and a soil depth refuses a class-free one, and
    /// every other level takes the method the caller declared.
    #[test]
    fn the_level_type_constrains_the_method_and_nothing_else_does() {
        assert!(level_type_admits(100, Method::Idw4));
        assert!(level_type_admits(100, Method::Nearest));
        assert!(!level_type_admits(100, Method::SurfaceIdw4));
        assert!(!level_type_admits(100, Method::SoilNearest));
        assert!(level_type_admits(106, Method::SoilNearest));
        assert!(!level_type_admits(106, Method::Idw4));
        // A surface categorical field may be declared nearest OR
        // interpolated; the binary has no opinion about which field it is.
        for method in [Method::Nearest, Method::SurfaceIdw4, Method::Idw4,
                       Method::SeaIceNearest] {
            assert!(level_type_admits(1, method));
            assert!(level_type_admits(103, method));
        }
    }
    #[test]
    fn shared_decoder_reads_native_count_without_grid_latlon() {
        const CELLS: usize = 2_949_120;
        let fixture = Fixture::new(&constant_native(CELLS));
        let result = decode(&fixture.0, mesh(CELLS), spec(), "2026091500", 0).unwrap();
        assert_eq!(result.values.len(), CELLS);
        assert!(result.values.iter().all(|&x| x == 285.));
    }
    #[test]
    fn wrong_time_field_native_count_or_centre_are_rejected() {
        const CELLS: usize = 2_949_120;
        let fixture = Fixture::new(&constant_native(CELLS));
        assert!(decode(&fixture.0, mesh(CELLS), spec(), "2026091506", 0).is_err());
        assert!(decode(&fixture.0, mesh(CELLS), spec(), "2026091500", 10800).is_err());
        let mut wrong = spec(); wrong.first = 70000.;
        assert!(decode(&fixture.0, mesh(CELLS), wrong, "2026091500", 0).is_err());
        // A caller declaring a different mesh or a different centre is refused
        // rather than served whatever the bytes happen to carry.
        assert!(decode(&fixture.0, mesh(4), spec(), "2026091500", 0).is_err());
        assert!(decode(&fixture.0, Mesh { cells: CELLS, centre: 7 }, spec(), "2026091500", 0).is_err());
        let fixture = Fixture::new(&constant_native(4));
        assert!(decode(&fixture.0, mesh(CELLS), spec(), "2026091500", 0).is_err());
    }
    #[test]
    fn regular_output_roundtrips_through_production_decoder() {
        let values = [270., 280., 290., 300.];
        let bytes = core101::encode_regular(&constant_native(4), target(), &values).unwrap();
        let file = Grib2File::from_bytes(&bytes).unwrap();
        let msg = &file.messages[0];
        assert_eq!((msg.grid.template, msg.grid.nx, msg.grid.ny, msg.grid.scan_mode), (0,2,2,0x40));
        assert_eq!(msg.product.level_value, 85000.);
        assert_eq!(unpack_message(msg).unwrap(), values);
        let (lat, lon) = grib_core::grib2::grid_latlon(&msg.grid).unwrap();
        assert_eq!(lat.len(), 4); assert_eq!(lon.len(), 4);
        assert!((lat[0] - 30.).abs() < 1e-8);
        // grib-core may retain the GRIB 0..360 branch.
        assert!((lon[0].rem_euclid(360.) - 260.).abs() < 1e-8);
        // Both the 0/360 meridian and the signed dateline branch must
        // survive the actual shared structured-grid decoder, not only
        // the spherical search or the GRIB header writer.
        for west in [-0.125, 179.875] {
            let shifted = Target { west, ..target() };
            let bytes = core101::encode_regular(&constant_native(4), shifted, &values).unwrap();
            let file = Grib2File::from_bytes(&bytes).unwrap();
            let (_, longitudes) = grib_core::grib2::grid_latlon(&file.messages[0].grid).unwrap();
            for (i, lon) in longitudes.iter().enumerate() {
                let expected = west + (i % 2) as f64 * 0.125;
                let error = (*lon - expected + 180.).rem_euclid(360.) - 180.;
                assert!(error.abs() < 1e-8);
            }
        }
    }
    #[test]
    fn partial_bitmap_survives_production_decoder() {
        let bytes = core101::encode_regular(&constant_native(4), target(), &[270., f64::NAN, 280., f64::NAN]).unwrap();
        let file = Grib2File::from_bytes(&bytes).unwrap();
        let decoded = unpack_message(&file.messages[0]).unwrap();
        assert_eq!(decoded[0], 270.); assert!(decoded[1].is_nan());
        assert_eq!(decoded[2], 280.); assert!(decoded[3].is_nan());
    }
    #[test]
    fn all_water_soil_bitmap_survives_production_decoder() {
        let bytes = core101::encode_regular(&constant_native(4), target(), &[f64::NAN; 4]).unwrap();
        let file = Grib2File::from_bytes(&bytes).unwrap();
        let decoded = unpack_message(&file.messages[0]).unwrap();
        assert_eq!(decoded.len(), 4); assert!(decoded.iter().all(|x| x.is_nan()));
    }
}
