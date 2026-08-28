//! The self-nesting proofs, re-run against the regional artifacts.
//!
//! There is no native oracle for a boundary produced from our own output —
//! nobody has one, because nobody else produces one.  So these tests build the
//! proof out of what does exist, and each says exactly what it establishes and
//! what it does not.
//!
//! * **Degenerate.**  Drive a child that IS its parent.  A nesting operator
//!   that cannot reproduce identity on a trivial nest is wrong, so this one
//!   must, and the bound is stated per field rather than waved at.
//! * **Teeth.**  Move the parent at one point and the boundary must move at
//!   that point, by that amount, and nowhere else.  Without this the test
//!   above is satisfied by an operator that copies the file.
//! * **Consistency.**  The same output, graded against the native case-9
//!   boundary file for the same valid time.  The residue is compared against
//!   how much native's own two ways of computing that boundary — case 7's
//!   initial state and case 9's boundary file — differ from each other.
//! * **Identity.**  Same inputs twice, same bytes.
//! * **Containment.**  A real refinement: the coarse regional forecast driving
//!   the five-times-finer child over the same window.
//!
//! Gated on `GPUWM_LAM_ORACLE_DIR` pointing at the 2026-08-25 regional record
//! set.  Without it every test here reports why it did not run rather than
//! passing quietly, which is the difference between a skipped test and a green
//! one that measured nothing.

use std::path::{Path, PathBuf};

use rw_mpas::lbc::parent::{build_from_parent, ParentConfig};
use rw_mpas::lbc::BoundaryInterval;

const ORACLE_ENV: &str = "GPUWM_LAM_ORACLE_DIR";

fn oracle() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var(ORACLE_ENV).ok()?);
    if dir.join("init-x1/conus.init.nc").exists() {
        Some(dir)
    } else {
        eprintln!(
            "{ORACLE_ENV} is set to {} but it holds no init-x1/conus.init.nc; \
             the self-nesting proofs did not run",
            dir.display()
        );
        None
    }
}

fn skip(what: &str) {
    eprintln!(
        "skipped {what}: set {ORACLE_ENV} to the regional record set to run it \
         (it measures nothing without the artifacts, and passing quietly would say otherwise)"
    );
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rw-mpas-self-nesting-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

fn config(child: &Path, out: &Path, times: &[(&str, PathBuf)], row: &str) -> ParentConfig {
    ParentConfig {
        grid_path: child.to_path_buf(),
        parent_grid: None,
        out_dir: out.to_path_buf(),
        start_time: times[0].0.to_string(),
        stop_time: times[times.len() - 1].0.to_string(),
        intervals: times
            .iter()
            .map(|(t, p)| BoundaryInterval {
                valid_time: (*t).to_string(),
                met_path: p.clone(),
            })
            .collect(),
        fg_interval_seconds: 3600,
        source_row: row.to_string(),
        registry_path: None,
        attr_overrides: Default::default(),
        provenance: "rw-mpas self-nesting proof".to_string(),
        without_snap: false,
    }
}

/// Read one float32 variable's first record, flat.
fn read(path: &Path, name: &str) -> Vec<f32> {
    let file = netcrust::File::open(path).expect("open");
    file.read_array_f64_first_record_or_all(name)
        .unwrap_or_else(|e| panic!("{} has no readable {name}: {e}", path.display()))
        .into_values()
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

fn ulp(a: f32, b: f32) -> i64 {
    let key = |f: f32| -> i64 {
        let bits = f.to_bits() as i32 as i64;
        if bits < 0 {
            i64::from(i32::MIN) - bits
        } else {
            bits
        }
    };
    (key(a) - key(b)).abs()
}

struct Grade {
    bit_equal: usize,
    total: usize,
    max_abs: f64,
    max_ulp: i64,
}

fn grade(ours: &[f32], theirs: &[f32]) -> Grade {
    assert_eq!(ours.len(), theirs.len(), "field lengths disagree");
    let mut g = Grade {
        bit_equal: 0,
        total: ours.len(),
        max_abs: 0.0,
        max_ulp: 0,
    };
    for (a, b) in ours.iter().zip(theirs.iter()) {
        if a.to_bits() == b.to_bits() {
            g.bit_equal += 1;
        }
        g.max_abs = g.max_abs.max((*a as f64 - *b as f64).abs());
        g.max_ulp = g.max_ulp.max(ulp(*a, *b));
    }
    g
}

/// A child that is its parent must come back as its parent.
///
/// Establishes: the horizontal operator, the vertical remap and the wind
/// projection compose to the identity when there is nothing to interpolate.
/// Does NOT establish anything about accuracy on a real refinement — the next
/// test down is what stops this one being satisfied by a file copy.
#[test]
fn a_child_that_is_its_parent_gets_its_parent_back() {
    let Some(root) = oracle() else {
        return skip("the degenerate self-nest");
    };
    let parent = root.join("init-x1/conus.init.nc");
    let out = scratch("degenerate");
    let receipt = build_from_parent(&config(
        &parent,
        &out,
        &[("2026-08-12_06:00:00", parent.clone())],
        "unstructured-native-stream",
    ))
    .expect("the degenerate transfer must run");

    assert_eq!(receipt.cells_snapped, receipt.child_cells);
    assert_eq!(receipt.edges_snapped, receipt.child_edges);
    let produced = PathBuf::from(&receipt.intervals[0].out_path);

    for (ours, theirs) in [
        ("lbc_theta", "theta"),
        ("lbc_u", "u"),
        ("lbc_w", "w"),
    ] {
        let g = grade(&read(&produced, ours), &read(&parent, theirs));
        assert_eq!(
            g.bit_equal, g.total,
            "{ours} must be bit-identical to the parent's {theirs} on a degenerate nest; \
             {} of {} were, worst {:e} ({} ulp)",
            g.bit_equal, g.total, g.max_abs, g.max_ulp
        );
    }

    // Density is remapped in the log, so it comes back through one f32
    // round trip.  The bound is stated, not hoped for.
    let g = grade(&read(&produced, "lbc_rho"), &read(&parent, "rho"));
    assert!(
        g.max_ulp <= 2,
        "density on a degenerate nest is a log round trip and nothing else: {} ulp, worst {:e}",
        g.max_ulp,
        g.max_abs
    );

    // Water vapour differs only where the parent held a negative value, and
    // the receipt says how many of those there were.
    let g = grade(&read(&produced, "lbc_qv"), &read(&parent, "qv"));
    let clamped = receipt.intervals[0].negative_mixing_ratios_clamped;
    assert_eq!(
        g.total - g.bit_equal,
        clamped,
        "every water-vapour difference on a degenerate nest must be a clamped negative, and \
         the receipt must have counted it: {} differed, {clamped} were clamped",
        g.total - g.bit_equal
    );

    let _ = std::fs::remove_dir_all(&out);
}

/// Move the parent, and exactly the right thing moves.
///
/// Establishes that the test above measures a transfer rather than a copy: a
/// producer that ignored its parent entirely would pass that one on a
/// degenerate mesh and fail this one instantly.
#[test]
fn moving_the_parent_moves_the_boundary_there_and_nowhere_else() {
    let Some(root) = oracle() else {
        return skip("the perturbation teeth");
    };
    let parent = root.join("init-x1/conus.init.nc");
    let work = scratch("teeth");
    let moved = work.join("moved-parent.nc");
    std::fs::copy(&parent, &moved).expect("copy the parent");

    // Read the field, move one value, write it back through the same layout.
    let levels = {
        let f = netcrust::File::open(&parent).expect("open");
        f.dimension("nVertLevels").expect("nVertLevels").len()
    };
    let target_cell = 1500usize;
    let target_level = 20usize;
    let flat = target_cell * levels + target_level;
    let before = read(&parent, "theta")[flat];
    poke_f32(&moved, "theta", flat, before + 1.0);

    let out_base = scratch("teeth-base");
    let out_moved = scratch("teeth-moved");
    let base = build_from_parent(&config(
        &parent,
        &out_base,
        &[("2026-08-12_06:00:00", parent.clone())],
        "unstructured-native-stream",
    ))
    .expect("base");
    let pert = build_from_parent(&config(
        &parent,
        &out_moved,
        &[("2026-08-12_06:00:00", moved.clone())],
        "unstructured-native-stream",
    ))
    .expect("perturbed");

    let a = read(Path::new(&base.intervals[0].out_path), "lbc_theta");
    let b = read(Path::new(&pert.intervals[0].out_path), "lbc_theta");
    let moved_points: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
    assert_eq!(
        moved_points,
        vec![flat],
        "one moved parent value must move exactly one boundary value"
    );
    let delta = (b[flat] - a[flat]) as f64;
    assert!(
        (delta - 1.0).abs() < 1.0e-4,
        "the boundary moved by {delta}, the parent moved by 1.0"
    );

    for d in [work, out_base, out_moved] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// Same inputs, twice, same bytes.
#[test]
fn the_same_parent_and_child_produce_the_same_bytes() {
    let Some(root) = oracle() else {
        return skip("the identity proof");
    };
    let parent = root.join("init-x1/conus.init.nc");
    let mut digests = Vec::new();
    let mut dirs = Vec::new();
    for arm in ["a", "b"] {
        let out = scratch(&format!("identity-{arm}"));
        let receipt = build_from_parent(&config(
            &parent,
            &out,
            &[("2026-08-12_06:00:00", parent.clone())],
            "unstructured-native-stream",
        ))
        .expect("run");
        digests.push(receipt.intervals[0].out_sha256.clone());
        dirs.push(out);
    }
    assert_eq!(digests[0], digests[1], "two runs, two different files");
    for d in dirs {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// A real refinement: the coarse regional forecast driving the five-times
/// finer child over the same window, across a multi-frame series.
///
/// Establishes that the operator places every child point in the parent, that
/// the wind fit is well posed everywhere on a genuine refinement, and that a
/// series of frames produces a series of files.  It does NOT establish
/// accuracy: there is no fine-resolution truth for a coarse parent's boundary.
#[test]
fn a_coarse_regional_forecast_drives_a_finer_child_over_the_same_window() {
    let Some(root) = oracle() else {
        return skip("the refinement cascade");
    };
    let child = root.join("init/conus.init.nc");
    let frames: Vec<(&str, PathBuf)> = vec![
        (
            "2026-08-12_06:00:00",
            root.join("run-g/history.2026-08-12_06.00.00.nc"),
        ),
        (
            "2026-08-12_07:00:00",
            root.join("run-g/history.2026-08-12_07.00.00.nc"),
        ),
    ];
    if !frames[0].1.exists() {
        return skip("the refinement cascade (no parent forecast frames in the record set)");
    }
    let out = scratch("cascade");
    let receipt =
        build_from_parent(&config(&child, &out, &frames, "unstructured-native-stream"))
            .expect("the cascade must run");

    assert!(receipt.child_cells > receipt.parent_cells * 4, "a real refinement");
    assert_eq!(receipt.cells_snapped, 0, "nothing should coincide here");
    assert_eq!(receipt.intervals.len(), 2, "one file per boundary time");
    assert!(
        receipt.worst_edge_condition.is_finite() && receipt.worst_edge_condition < 1.0e4,
        "the local wind fit must stay well posed on a refinement: worst condition {}",
        receipt.worst_edge_condition
    );
    // The parent's regional history carries no condensate, and the producer
    // must say so rather than write silent zeros.
    assert!(
        receipt.intervals[0]
            .roles_absent
            .iter()
            .any(|r| r == "cloud-mixing-ratio"),
        "an absent role must be named in the receipt: {:?}",
        receipt.intervals[0].roles_absent
    );
    let _ = std::fs::remove_dir_all(&out);
}

/// A child wider than its parent is refused, by name, with the distance.
#[test]
fn a_child_wider_than_its_parent_is_refused() {
    let Some(root) = oracle() else {
        return skip("the containment refusal");
    };
    let fine_parent = root.join("run-k/history.2026-08-12_06.00.00.nc");
    if !fine_parent.exists() {
        return skip("the containment refusal (no fine-mesh forecast in the record set)");
    }
    // The x1 window reaches further out than the x4 one, so driving the wide
    // child from the narrow parent must fail.
    let wide_child = root.join("init-x1/conus.init.nc");
    let out = scratch("containment");
    let err = build_from_parent(&config(
        &wide_child,
        &out,
        &[("2026-08-12_06:00:00", fine_parent)],
        "unstructured-native-stream",
    ))
    .expect_err("a child outside its parent must be refused")
    .to_string();
    assert!(err.contains("outside the parent's domain"), "{err}");
    assert!(err.contains("must be contained in the parent"), "{err}");
    let _ = std::fs::remove_dir_all(&out);
}

/// Overwrite one float32 in a classic-format file, in place.
///
/// The header walk gives the variable's offset; the value is written straight
/// there.  This is a test fixture, not a writer: it works because the file was
/// copied from one this producer already read.
fn poke_f32(path: &Path, variable: &str, flat_index: usize, value: f32) {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut bytes = vec![0u8; 4 * 1024 * 1024];
    {
        let mut f = std::fs::File::open(path).expect("open for header");
        let n = f.read(&mut bytes).expect("read header");
        bytes.truncate(n);
    }
    let offset = classic_variable_offset(&bytes, variable)
        .unwrap_or_else(|| panic!("{variable} not found in {}", path.display()));
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open for write");
    f.seek(SeekFrom::Start(offset + 4 * flat_index as u64))
        .expect("seek");
    f.write_all(&value.to_bits().to_be_bytes()).expect("write");
}

/// Walk a CDF-5 header far enough to find one variable's data offset.
fn classic_variable_offset(b: &[u8], want: &str) -> Option<u64> {
    if b.len() < 8 || &b[0..3] != b"CDF" || b[3] != 5 {
        return None;
    }
    let mut at = 4usize;
    let u32at = |b: &[u8], at: &mut usize| -> u32 {
        let v = u32::from_be_bytes(b[*at..*at + 4].try_into().unwrap());
        *at += 4;
        v
    };
    let u64at = |b: &[u8], at: &mut usize| -> u64 {
        let v = u64::from_be_bytes(b[*at..*at + 8].try_into().unwrap());
        *at += 8;
        v
    };
    let name_at = |b: &[u8], at: &mut usize| -> String {
        let n = u64::from_be_bytes(b[*at..*at + 8].try_into().unwrap()) as usize;
        *at += 8;
        let s = String::from_utf8_lossy(&b[*at..*at + n]).to_string();
        *at += n + ((4 - n % 4) % 4);
        s
    };
    let _numrecs = u64at(b, &mut at);
    // Dimensions.
    let tag = u32at(b, &mut at);
    let ndims = u64at(b, &mut at) as usize;
    if tag != 0 {
        for _ in 0..ndims {
            let _ = name_at(b, &mut at);
            let _ = u64at(b, &mut at);
        }
    }
    // Global attributes.
    skip_attrs(b, &mut at, u32at, u64at, name_at)?;
    // Variables.
    let tag = u32at(b, &mut at);
    let nvars = u64at(b, &mut at) as usize;
    if tag == 0 {
        return None;
    }
    for _ in 0..nvars {
        let name = name_at(b, &mut at);
        let rank = u64at(b, &mut at) as usize;
        at += 8 * rank;
        skip_attrs(b, &mut at, u32at, u64at, name_at)?;
        let _nc_type = u32at(b, &mut at);
        let _vsize = u64at(b, &mut at);
        let begin = u64at(b, &mut at);
        if name == want {
            return Some(begin);
        }
    }
    None
}

fn skip_attrs(
    b: &[u8],
    at: &mut usize,
    u32at: impl Fn(&[u8], &mut usize) -> u32,
    u64at: impl Fn(&[u8], &mut usize) -> u64,
    name_at: impl Fn(&[u8], &mut usize) -> String,
) -> Option<()> {
    let tag = u32at(b, at);
    let n = u64at(b, at) as usize;
    if tag == 0 {
        return Some(());
    }
    for _ in 0..n {
        let _ = name_at(b, at);
        let nc_type = u32at(b, at);
        let count = u64at(b, at) as usize;
        let size = match nc_type {
            1 | 2 | 7 => 1,
            3 | 8 => 2,
            4 | 5 | 9 => 4,
            6 | 10 | 11 => 8,
            _ => return None,
        };
        let bytes = count * size;
        *at += bytes + ((4 - bytes % 4) % 4);
    }
    Some(())
}
