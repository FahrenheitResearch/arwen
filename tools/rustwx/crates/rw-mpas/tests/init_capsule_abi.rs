//! The emit ABI the native-free init door builds against, pinned.
//!
//! `gpuwm-hex`'s init door constructs an init-class capsule with no native
//! lineage: mesh/statics carried from the generated static file, the vertical
//! contract built from a declarative spec, and every met-state variable
//! PRE-DECLARED as a float32 zero slot so this emitter's capsule-derived
//! schema has a landing site for each computed value.  That door was measured
//! against this binary on real x1.40962 assets (2026-08-24); these tests pin
//! the four behaviours it keys on, so an engine change that breaks any of
//! them fails here first instead of at a user's native-free mint:
//!
//! 1. a capsule that pre-declares the landing sites is accepted -- computed
//!    values land in them, declared-but-uncomputed floats carry through;
//! 2. a computed value with no landing site refuses and names the variable
//!    ("...the init file has no variable for...");
//! 3. a Double-typed capsule variable refuses by type name ("...is Double,
//!    which this emitter does not copy") -- the reason the door writes its
//!    whole vertical contract as float32;
//! 4. an unsupplied char variable refuses as an identity label rather than
//!    being copied out of the capsule.
//!
//! Written with `rw_store::netcdf_classic` and read back with `netcrust`,
//! the same two codebases the emitter itself stands on.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rw_mpas::init::emit::{write_init, Computed};
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};

const N_CELLS: usize = 3;
const N_LEV: usize = 2;
const STRLEN: usize = 64;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("rw_mpas_init_abi_tests");
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir.join(name)
}

/// A minimal init-class capsule shaped the way the native-free door shapes
/// one: fixed mesh dims, an unlimited Time, zero-filled met-state slots.
struct CapsuleSpec {
    /// Pre-declare the `theta` landing site.
    with_theta_slot: bool,
    /// Declare a Double-typed scalar, which the emitter must refuse to carry.
    with_double_scalar: bool,
    /// Declare `xCell` as Double, the way a static for a mesh with no native
    /// MPAS-A counterpart stores it. The emitter must carry it AT ITS OWN
    /// WIDTH, which is the one exception to the Double refusal above.
    with_double_coordinate: bool,
    /// Declare an `xtime` char variable the caller does not compute.
    with_unsupplied_char: bool,
    /// Carry a `model_name` of the capsule's own, which the writer must not
    /// overwrite: a capsule that already names a model has an answer.
    carried_model_name: Option<&'static str>,
}

fn write_capsule(path: &PathBuf, spec: &CapsuleSpec) {
    let dims = vec![
        NcDim::record("Time"),
        NcDim::fixed("nCells", N_CELLS),
        NcDim::fixed("nVertLevels", N_LEV),
        NcDim::fixed("StrLen", STRLEN),
    ];
    let mut gattrs = vec![
        NcAttr::text("file_id", "donorabcde"),
        NcAttr::text("parent_id", ""),
    ];
    if let Some(name) = spec.carried_model_name {
        gattrs.push(NcAttr::text("model_name", name));
    }
    let mut vars = vec![
        // Carried statics: a fixed float and a carried record float.
        NcVarDef::new("ter", NcType::Float, vec![1]),
        NcVarDef::new("Time", NcType::Float, vec![0]),
        // A declared-but-never-computed profile, carried as zeros.
        NcVarDef::new("u_init", NcType::Float, vec![2]),
        // The one char identity the caller computes.
        NcVarDef::new("initial_time", NcType::Char, vec![3]),
    ];
    if spec.with_theta_slot {
        vars.push(NcVarDef::new("theta", NcType::Float, vec![0, 1, 2]));
    }
    if spec.with_double_scalar {
        vars.push(NcVarDef::new("cf1", NcType::Double, vec![]));
    }
    if spec.with_double_coordinate {
        vars.push(NcVarDef::new("xCell", NcType::Double, vec![1]));
    }
    if spec.with_unsupplied_char {
        vars.push(NcVarDef::new("xtime", NcType::Char, vec![0, 3]));
    }
    let mut writer = NcClassicWriter::create(path, NcFormat::Data64, dims, gattrs, vars, 1)
        .expect("capsule layout");
    writer
        .put("ter", NcData::Floats(&[10.0, 20.0, 30.0]))
        .expect("ter");
    writer
        .put_record("Time", 0, NcData::Floats(&[0.0]))
        .expect("Time");
    writer
        .put("u_init", NcData::Floats(&vec![0.0; N_LEV]))
        .expect("u_init");
    let blank = vec![b' '; STRLEN];
    writer
        .put("initial_time", NcData::Chars(&blank))
        .expect("initial_time");
    if spec.with_theta_slot {
        writer
            .put_record("theta", 0, NcData::Floats(&vec![0.0; N_CELLS * N_LEV]))
            .expect("theta slot");
    }
    if spec.with_double_scalar {
        writer
            .put("cf1", NcData::Doubles(&[1.5]))
            .expect("cf1");
    }
    if spec.with_double_coordinate {
        // Three values no f32 can hold, so a silent narrowing is visible in
        // the bytes rather than only in the type.
        writer
            .put(
                "xCell",
                NcData::Doubles(&[
                    6_371_229.000_000_001,
                    -1_234_567.891_011_12,
                    3_141_592.653_589_79,
                ]),
            )
            .expect("xCell");
    }
    if spec.with_unsupplied_char {
        writer
            .put_record("xtime", 0, NcData::Chars(&blank))
            .expect("xtime");
    }
    writer.finish().expect("capsule finish");
}

fn theta_values() -> Vec<f32> {
    (0..N_CELLS * N_LEV).map(|i| 300.0 + i as f32).collect()
}

fn computed_with_theta() -> BTreeMap<String, Computed> {
    let mut computed = BTreeMap::new();
    computed.insert("theta".to_string(), Computed::Floats(theta_values()));
    computed.insert(
        "initial_time".to_string(),
        Computed::Text("2026-08-12_06:00:00".to_string()),
    );
    computed
}

#[test]
fn a_capsule_that_predeclares_the_landing_sites_is_accepted() {
    let capsule = scratch("predeclared.capsule.nc");
    let out = scratch("predeclared.init.nc");
    write_capsule(
        &capsule,
        &CapsuleSpec {
            with_theta_slot: true,
            with_double_scalar: false,
            with_double_coordinate: false,
            with_unsupplied_char: false,
            carried_model_name: None,
        },
    );

    let ledger = write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &["donorabcde".to_string()],
        "abi pin test",
        &rw_mpas::init::Lineage::default(),
    )
    .unwrap_or_else(|e| panic!("a pre-declared landing site must be accepted: {e}"));

    assert!(
        ledger.computed.contains(&"theta".to_string()),
        "theta landed computed, not carried: {:?}",
        ledger.computed
    );
    for name in ["ter", "Time", "u_init"] {
        assert!(
            ledger.carried.contains(&name.to_string()),
            "{name} must carry through; carried = {:?}",
            ledger.carried
        );
    }

    // Read back through the independent reader: computed values in the slot,
    // carried zeros still zero, the identity label this file's own.
    let file = netcrust::File::open(&out).expect("minted init opens");
    let theta: Vec<f64> = file
        .read_array_f64("theta")
        .expect("theta readable")
        .into_values();
    let expected: Vec<f64> = theta_values().into_iter().map(f64::from).collect();
    assert_eq!(theta, expected, "computed theta must land verbatim");
    let u_init: Vec<f64> = file
        .read_array_f64("u_init")
        .expect("u_init readable")
        .into_values();
    assert!(
        u_init.iter().all(|&v| v == 0.0),
        "a declared-but-uncomputed slot carries its zeros"
    );
}

#[test]
fn a_computed_value_with_no_landing_site_refuses_and_names_it() {
    let capsule = scratch("noslot.capsule.nc");
    let out = scratch("noslot.init.nc");
    write_capsule(
        &capsule,
        &CapsuleSpec {
            with_theta_slot: false,
            with_double_scalar: false,
            with_double_coordinate: false,
            with_unsupplied_char: false,
            carried_model_name: None,
        },
    );

    let err = write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &[],
        "abi pin test",
        &rw_mpas::init::Lineage::default(),
    )
    .err()
    .expect("a computed value with no variable must refuse");
    let message = err.to_string();
    assert!(
        message.contains("no variable for"),
        "the door keys on this phrase; got: {message}"
    );
    assert!(
        message.contains("theta"),
        "the refusal must name the orphaned variable; got: {message}"
    );
}

#[test]
fn a_double_typed_capsule_variable_refuses_by_type_name() {
    let capsule = scratch("double.capsule.nc");
    let out = scratch("double.init.nc");
    write_capsule(
        &capsule,
        &CapsuleSpec {
            with_theta_slot: true,
            with_double_scalar: true,
            with_double_coordinate: false,
            with_unsupplied_char: false,
            carried_model_name: None,
        },
    );

    let err = write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &[],
        "abi pin test",
        &rw_mpas::init::Lineage::default(),
    )
    .err()
    .expect("a Double carried variable must refuse");
    let message = err.to_string();
    assert!(
        message.contains("cf1") && message.contains("Double"),
        "the door writes float32 because this refusal names Double; got: {message}"
    );
}

#[test]
fn an_unsupplied_char_variable_refuses_as_an_identity_label() {
    let capsule = scratch("char.capsule.nc");
    let out = scratch("char.init.nc");
    write_capsule(
        &capsule,
        &CapsuleSpec {
            with_theta_slot: true,
            with_double_scalar: false,
            with_double_coordinate: false,
            with_unsupplied_char: true,
            carried_model_name: None,
        },
    );

    let err = write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &[],
        "abi pin test",
        &rw_mpas::init::Lineage::default(),
    )
    .err()
    .expect("an unsupplied char variable must refuse, never copy");
    let message = err.to_string();
    assert!(
        message.contains("xtime") && message.contains("identity"),
        "char variables are identity labels; got: {message}"
    );
}

/// The four lineage attributes `rw_mpas_lbc` reads off its `--grid`.
///
/// THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26): `write_init` re-emitted
/// the capsule's global attributes and added `file_id`, `parent_id` and
/// `gpuwm_provenance`.  A capsule carries the MESH's lineage, not the
/// MODEL's, so no init this program produced carried `model_name`,
/// `core_name`, `version` or `git_version` -- and `rw_mpas_lbc` refuses a
/// `--grid` that lacks any of them rather than invent one, with "inventing a
/// value here would stamp the stream with an identity its mesh never had".
/// Every init this program minted was therefore unusable as a boundary
/// source, regional or global, and the limited-area chain went through a
/// hand-written attribute patcher instead.
#[test]
fn an_init_carries_the_lineage_a_boundary_producer_reads() {
    let capsule = scratch("lineage_capsule.nc");
    write_capsule(
        &capsule,
        &CapsuleSpec {
            with_theta_slot: true,
            with_double_scalar: false,
            with_double_coordinate: false,
            with_unsupplied_char: false,
            carried_model_name: None,
        },
    );
    let out = scratch("lineage_init.nc");
    let _ = std::fs::remove_file(&out);

    let lineage = rw_mpas::init::Lineage {
        model_name: "gpuwm-hex".to_string(),
        core_name: "atmosphere".to_string(),
        version: "9.9.9".to_string(),
        git_version: "gpuwm-hex-9.9.9".to_string(),
    };
    write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &["donorabcde".to_string()],
        "lineage test",
        &lineage,
    )
    .expect("an init with a lineage must be written");

    let file = netcrust::File::open(&out).expect("read the init back");
    let read = |name: &str| {
        file.attribute(name)
            .and_then(|a| a.as_string().map(str::to_string))
            .unwrap_or_else(|| panic!("{name} is absent from the written init"))
    };
    assert_eq!(read("model_name"), "gpuwm-hex");
    assert_eq!(read("core_name"), "atmosphere");
    assert_eq!(read("version"), "9.9.9");
    assert_eq!(read("git_version"), "gpuwm-hex-9.9.9");
    // It is never `mpas`: a file carrying this port's numerics claiming
    // MPAS's identity is the thing the consumer's refusal exists to stop.
    assert_ne!(read("model_name"), "mpas");
    // The engine's default is at least true about who wrote it, so a caller
    // that says nothing still produces a usable boundary source.
    let fallback = rw_mpas::init::Lineage::default();
    assert_eq!(fallback.core_name, "atmosphere");
    assert!(!fallback.model_name.is_empty());
    assert!(!fallback.version.is_empty());
    assert!(!fallback.git_version.is_empty());
    assert_ne!(fallback.model_name, "mpas");
}

/// A capsule that already names a model keeps its own answer.
#[test]
fn a_capsule_that_carries_a_lineage_is_not_overwritten() {
    let capsule = scratch("lineage_carried_capsule.nc");
    write_capsule(
        &capsule,
        &CapsuleSpec {
            with_theta_slot: true,
            with_double_scalar: false,
            with_double_coordinate: false,
            with_unsupplied_char: false,
            carried_model_name: Some("the-capsules-own-model"),
        },
    );
    let out = scratch("lineage_carried_init.nc");
    let _ = std::fs::remove_file(&out);
    write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &[],
        "lineage test",
        &rw_mpas::init::Lineage {
            model_name: "the-callers-model".to_string(),
            core_name: "atmosphere".to_string(),
            version: "2.0".to_string(),
            git_version: "caller-2.0".to_string(),
        },
    )
    .expect("a capsule with a carried lineage must still be written");

    let file = netcrust::File::open(&out).expect("read the init back");
    let read = |name: &str| {
        file.attribute(name)
            .and_then(|a| a.as_string().map(str::to_string))
            .unwrap_or_else(|| panic!("{name} is absent"))
    };
    assert_eq!(
        read("model_name"),
        "the-capsules-own-model",
        "a capsule that already names a model keeps its own answer"
    );
    // The three the capsule does NOT carry still come from the caller, so a
    // partially-labelled capsule still produces a usable boundary source.
    assert_eq!(read("core_name"), "atmosphere");
    assert_eq!(read("version"), "2.0");
    assert_eq!(read("git_version"), "caller-2.0");
}


/// THE ONE DOUBLE THE EMITTER CARRIES, and it carries it at full width.
///
/// A static built for a mesh with no native MPAS-A counterpart stores its
/// fifteen coordinate arrays as binary64 (`staticfile::coordframe`), because
/// no byte-identity anchor binds the storage precision of a mesh native
/// MPAS-A cannot produce. Narrowing those to f32 on the way into the init
/// file would put the 0.5 m coordinate quantum straight back into the file
/// the dycore runs -- the whole of what the representation removes -- so this
/// pins that they arrive as Double and keep every bit.
#[test]
fn a_double_coordinate_array_is_carried_at_its_own_width() {
    let capsule = scratch("coord64.capsule.nc");
    let out = scratch("coord64.init.nc");
    write_capsule(
        &capsule,
        &CapsuleSpec {
            with_theta_slot: true,
            with_double_scalar: false,
            with_double_coordinate: true,
            with_unsupplied_char: false,
            carried_model_name: None,
        },
    );

    let ledger = write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &["donorabcde".to_string()],
        "abi pin test",
        &rw_mpas::init::Lineage::default(),
    )
    .unwrap_or_else(|e| panic!("a binary64 coordinate array must be carried: {e}"));
    assert!(
        ledger.carried.iter().any(|n| n == "xCell"),
        "xCell was not carried: {:?}",
        ledger.carried
    );

    let file = netcrust::File::open(&out).expect("read the init back");
    let variable = file.variable("xCell").expect("xCell in the init file");
    assert!(
        matches!(variable.dtype(), netcrust::DataType::F64),
        "xCell was narrowed to {:?}; the coordinate quantum is back in the file",
        variable.dtype()
    );
    let values = file
        .read_array_f64("xCell")
        .expect("xCell values")
        .into_values();
    let expected = [
        6_371_229.000_000_001_f64,
        -1_234_567.891_011_12,
        3_141_592.653_589_79,
    ];
    for (got, want) in values.iter().zip(expected.iter()) {
        assert_eq!(got, want, "a coordinate lost bits on the way through");
    }
}
