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
    /// Declare an `xtime` char variable the caller does not compute.
    with_unsupplied_char: bool,
}

fn write_capsule(path: &PathBuf, spec: &CapsuleSpec) {
    let dims = vec![
        NcDim::record("Time"),
        NcDim::fixed("nCells", N_CELLS),
        NcDim::fixed("nVertLevels", N_LEV),
        NcDim::fixed("StrLen", STRLEN),
    ];
    let gattrs = vec![
        NcAttr::text("file_id", "donorabcde"),
        NcAttr::text("parent_id", ""),
    ];
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
            with_unsupplied_char: false,
        },
    );

    let ledger = write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &["donorabcde".to_string()],
        "abi pin test",
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
            with_unsupplied_char: false,
        },
    );

    let err = write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &[],
        "abi pin test",
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
            with_unsupplied_char: false,
        },
    );

    let err = write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &[],
        "abi pin test",
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
            with_unsupplied_char: true,
        },
    );

    let err = write_init(
        &out,
        &capsule,
        computed_with_theta(),
        "mintedfile",
        &[],
        "abi pin test",
    )
    .err()
    .expect("an unsupplied char variable must refuse, never copy");
    let message = err.to_string();
    assert!(
        message.contains("xtime") && message.contains("identity"),
        "char variables are identity labels; got: {message}"
    );
}
