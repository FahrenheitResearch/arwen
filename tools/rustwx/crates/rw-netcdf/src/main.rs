//! `rw_netcdf` -- the NetCDF front door.
//!
//! Two passes, deliberately the same shape as the `grib2_inventory` /
//! `grib2_dump` pair that `gpuwm.mapped_source` already drives for
//! GRIB2, so one module reads both formats through one idiom:
//!
//! * `rw_netcdf inventory FILE` prints a JSON description of the file --
//!   format, dimensions, global attributes, and every variable with its
//!   dimensions, shape, dtype and attributes.  Selector resolution
//!   (name, `standard_name`, units checks) happens on the Python side
//!   against this document; no values are read.
//! * `rw_netcdf dump FILE OUTDIR VAR...` decodes the named variables and
//!   writes one flat little-endian `f64` file each, plus a
//!   `metadata.json` giving each variable's shape and filename.  numpy
//!   maps those with `np.fromfile(..., dtype="<f8")`, exactly as it does
//!   for `grib2_dump` output.
//!
//! CF reference-time decoding lives HERE rather than in Python, because
//! turning `hours since 1970-01-01` plus a number into an instant is
//! decoding meteorological data, not plumbing.  A variable whose `units`
//! parses as a CF time gets an extra `times` array of RFC-3339 strings
//! in the dump metadata; Python reads strings and never runs a calendar.
//!
//! Everything is read through `netcrust`, the vendored pure-Rust
//! NetCDF/NetCDF-4 facade already used by rw-wrfbatch, rw-sat and
//! rw-fieldcmp.  There is no C NetCDF library anywhere in this path.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use serde::Serialize;

/// The contract marker. Printed by `--abi`, and -- because it is a
/// literal in the binary -- readable straight out of the bytes by the
/// release cut, the same convention every other bundled artifact uses.
const ABI: &str = concat!(
    "gpuwm-rw-netcdf-inventory-v1\tformat\tdimensions\tglobal_attributes\tvariables",
    "\tgpuwm-rw-netcdf-dump-v1\tvariables\tfilename\tshape\ttimes",
);

const INVENTORY_SCHEMA: &str = "gpuwm-rw-netcdf-inventory-v1";
const DUMP_SCHEMA: &str = "gpuwm-rw-netcdf-dump-v1";

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded as a literal so the release cut can
/// read it straight out of the bytes without executing anything.  A
/// bundled artifact the cut cannot prove must not ship, and this is what
/// makes "built from the release commit" a property of the file itself.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const USAGE: &str = "\
usage: rw_netcdf inventory FILE
       rw_netcdf dump [--raw] FILE OUTPUT_DIR VARIABLE [VARIABLE...]
       rw_netcdf --abi | --help

  inventory  print a JSON description of FILE (no values are read)
  dump       decode VARIABLEs into OUTPUT_DIR as flat little-endian f64
             files plus metadata.json
  --raw      skip CF decoding: no _FillValue/missing_value masking and no
             scale_factor/add_offset, so stored sentinels survive.  This
             is what netCDF4's set_auto_mask(False) asks for, and some
             products (gridded radar/satellite) read their own fill
             sentinel deliberately rather than a mask.
";

#[derive(Serialize)]
struct Inventory {
    schema: &'static str,
    format: String,
    metadata: MetadataProvenance,
    dimensions: Vec<DimensionRecord>,
    global_attributes: BTreeMap<String, serde_json::Value>,
    variables: Vec<VariableRecord>,
}

#[derive(Serialize)]
struct DimensionRecord {
    name: String,
    len: usize,
    unlimited: bool,
}

#[derive(Serialize)]
struct VariableRecord {
    name: String,
    dimensions: Vec<String>,
    shape: Vec<usize>,
    dtype: String,
    attributes: BTreeMap<String, serde_json::Value>,
    /// True when this variable was recovered from the raw HDF5 index
    /// because the NetCDF-4 variable table omitted it -- see
    /// [`recover_dimension_scales`].  Its attributes are limited to the
    /// CF string attributes that can be read by name, so a consumer that
    /// needs numeric attributes on it must say so rather than assume.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    recovered_dimension_scale: bool,
}

/// CF string attributes probed by name on a recovered dimension scale.
///
/// The raw-HDF5 escape hatch reads ONE named string attribute at a time
/// -- there is no enumeration -- so the set has to be written down.
/// These are the attributes the coordinate contract actually consults:
/// `units` and `standard_name` decide selector resolution, `calendar`
/// and `axis` describe the time axis, `long_name` is for diagnostics.
const RECOVERED_ATTRIBUTES: [&str; 5] =
    ["units", "standard_name", "long_name", "calendar", "axis"];

/// The netCDF-4 marker on an HDF5 dimension scale that has no coordinate
/// variable behind it.  The C library writes this exact sentence into the
/// scale's `NAME` attribute and hides such datasets from `variables`.
const PHONY_DIMENSION_PREFIX: &str =
    "This is a netCDF dimension but not a netCDF variable.";

#[derive(Serialize)]
struct DumpMetadata {
    schema: &'static str,
    metadata: MetadataProvenance,
    variables: Vec<DumpRecord>,
}

#[derive(Serialize)]
struct DumpRecord {
    name: String,
    filename: String,
    shape: Vec<usize>,
    dimensions: Vec<String>,
    dtype: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    units: Option<String>,
    /// RFC-3339 instants, present only when `units` parses as a CF
    /// reference time. Decoding a calendar is this binary's job.
    #[serde(skip_serializing_if = "Option::is_none")]
    times: Option<Vec<String>>,
    /// What CF decoding was applied, so a reader can audit it rather
    /// than infer it. `missing_count` is the number of elements written
    /// as NaN because they matched `_FillValue`/`missing_value`.
    cf: CfApplied,
}

#[derive(Serialize, Default)]
struct CfApplied {
    #[serde(skip_serializing_if = "Option::is_none")]
    scale_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    add_offset: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fill_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    missing_value: Option<f64>,
    missing_count: usize,
    /// False under `--raw`: the attributes above are reported but were
    /// deliberately not acted on, so stored sentinels survive.
    applied: bool,
}

fn main() {
    // Keep the stamp in the binary: an unreferenced static can be
    // dropped by the linker, and a stamp that optimises away is a
    // release the cut refuses for a reason nobody can see.
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprint!("{USAGE}");
        std::process::exit(2);
    }
    match args[0].as_str() {
        "--abi" => {
            println!("{ABI}");
        }
        "--help" | "-h" => {
            print!("{USAGE}");
        }
        "inventory" => {
            if args.len() != 2 {
                fail("inventory takes exactly one FILE");
            }
            if let Err(error) = inventory(Path::new(&args[1])) {
                fail(&error);
            }
        }
        "dump" => {
            // Two switches, because netCDF4-python has two.  `--raw` is
            // `set_auto_maskandscale(False)`: no masking and no scaling.
            // `--no-mask` is `set_auto_mask(False)`: no masking, but
            // scale_factor/add_offset STILL applied.  Collapsing the two
            // silently unpacked every packed variable a caller asked to
            // see unmasked, which is a value change, not a policy change.
            let raw = args.iter().any(|a| a == "--raw");
            let no_mask = raw || args.iter().any(|a| a == "--no-mask");
            let apply_scale = !raw;
            let positional: Vec<&String> = args[1..]
                .iter()
                .filter(|a| a.as_str() != "--raw" && a.as_str() != "--no-mask")
                .collect();
            if positional.len() < 3 {
                fail("dump takes FILE, OUTPUT_DIR and at least one VARIABLE");
            }
            let names: Vec<String> =
                positional[2..].iter().map(|s| (*s).clone()).collect();
            if let Err(error) = dump(
                Path::new(positional[0]),
                Path::new(positional[1]),
                &names,
                !no_mask,
                apply_scale,
            ) {
                fail(&error);
            }
        }
        other => fail(&format!("unknown subcommand {other:?}\n{USAGE}")),
    }
}

fn fail(message: &str) -> ! {
    eprintln!("rw_netcdf: {message}");
    std::process::exit(2);
}

fn attribute_json(value: &netcrust::AttributeValue) -> serde_json::Value {
    use netcrust::AttributeValue as V;
    fn numbers<T: Copy + Into<f64>>(values: &[T]) -> serde_json::Value {
        let mapped: Vec<serde_json::Value> = values
            .iter()
            .map(|&value| {
                serde_json::Number::from_f64(value.into())
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            })
            .collect();
        if mapped.len() == 1 {
            mapped.into_iter().next().unwrap()
        } else {
            serde_json::Value::Array(mapped)
        }
    }
    match value {
        V::Chars(text) => serde_json::Value::String(text.clone()),
        V::Strings(values) if values.len() == 1 => serde_json::Value::String(values[0].clone()),
        V::Strings(values) => serde_json::Value::Array(
            values
                .iter()
                .map(|value| serde_json::Value::String(value.clone()))
                .collect(),
        ),
        V::Bytes(values) => numbers(values),
        V::Shorts(values) => numbers(values),
        V::Ints(values) => numbers(values),
        V::Floats(values) => numbers(values),
        V::Doubles(values) => numbers(values),
        V::UBytes(values) => numbers(values),
        V::UShorts(values) => numbers(values),
        V::UInts(values) => numbers(values),
        // i64/u64 beyond f64's exact range would be silently rounded by
        // the numeric path; emit them as integers instead.
        V::Int64s(values) if values.len() == 1 => serde_json::Value::from(values[0]),
        V::Int64s(values) => serde_json::Value::Array(
            values.iter().map(|&v| serde_json::Value::from(v)).collect(),
        ),
        V::UInt64s(values) if values.len() == 1 => serde_json::Value::from(values[0]),
        V::UInt64s(values) => serde_json::Value::Array(
            values.iter().map(|&v| serde_json::Value::from(v)).collect(),
        ),
    }
}

fn attributes_map(attributes: &[netcrust::Attribute]) -> BTreeMap<String, serde_json::Value> {
    attributes
        .iter()
        .map(|attribute| {
            (
                attribute.name().to_string(),
                attribute_json(attribute.value()),
            )
        })
        .collect()
}

/// Open strictly, and fall back to size-inferred NetCDF-4 metadata only
/// when strict reconstruction is impossible -- reporting which happened.
///
/// Strict mode requires every NetCDF-4 variable to carry a
/// `DIMENSION_LIST` attribute naming its dimension scales.  Real files
/// from major producers do not always have one: ERA5 as delivered by the
/// Copernicus CDS omits it on `number` and `expver`, and the RRTMGP
/// coefficient tables omit it throughout.  Refusing those outright would
/// be refusing the project's own canonical forcing source.
///
/// The fallback resolves a variable's dimensions BY SIZE, which is a
/// heuristic and can mis-name axes whenever two dimensions share a
/// length.  So it is never silent: the mode travels in the output
/// document, and `dimension_lengths_ambiguous` says whether the file
/// contains a length collision that could make the inference wrong.  A
/// consumer that cares about axis identity can then refuse, instead of
/// being handed a guess it cannot distinguish from a fact.
fn open(path: &Path) -> Result<(netcrust::File, MetadataProvenance), String> {
    // The open itself is lazy: a NetCDF-4 file whose metadata cannot be
    // reconstructed still opens, and only reports it when the dimension
    // or variable tables are first built.  So strictness is decided by
    // PROBING those tables, not by whether the handle came back.
    let probe = |file: &netcrust::File| -> Result<(), String> {
        file.dimensions()
            .map_err(|error| error.to_string())
            .and_then(|_| file.variables().map_err(|error| error.to_string()))
            .map(|_| ())
    };

    let strict_error = match netcrust::File::open(path) {
        Ok(file) => match probe(&file) {
            Ok(()) => return Ok((file, MetadataProvenance::strict())),
            Err(error) => error,
        },
        Err(error) => error.to_string(),
    };

    let options = netcrust::NcOpenOptions {
        metadata_mode: netcrust::NcMetadataMode::Lossy,
        ..Default::default()
    };
    let file = netcrust::File::open_with_options(path, options).map_err(|lossy_error| {
        format!(
            "cannot open {}: {strict_error} (and size-inferred metadata \
             also failed: {lossy_error})",
            path.display()
        )
    })?;
    probe(&file).map_err(|lossy_error| {
        format!(
            "cannot read {}: {strict_error} (and size-inferred metadata \
             also failed: {lossy_error})",
            path.display()
        )
    })?;
    let ambiguous = has_duplicate_dimension_lengths(&file);
    Ok((
        file,
        MetadataProvenance {
            mode: "size-inferred",
            strict_error: Some(strict_error),
            dimension_lengths_ambiguous: ambiguous,
        },
    ))
}

/// Add back the NetCDF-4 coordinate variables the variable table omits.
///
/// In NetCDF-4 a coordinate variable is stored as an HDF5 *dimension
/// scale*, and the vendored reader's `variables()` does not report those
/// -- so `latitude`, `longitude`, `pressure_level` and `valid_time` are
/// simply absent from an ERA5 file, as are `time` and the `xh/yh/zh`
/// axes of a CM1 history file.  Those are exactly the variables a
/// coordinate contract resolves against, so without this the reader
/// lists every field and none of the axes.
///
/// Recovery uses only public netcrust API -- `hdf5_root_datasets` for
/// the names and shapes, `hdf5_dataset_attribute_string` for the CF
/// string attributes, and `read_array_f64`, which already falls back to
/// a raw-HDF5 read by name for values.  Nothing in the vendored tree is
/// modified; this belongs upstream and is reported there, but a reader
/// that silently drops the axes cannot be shipped in the meantime.
///
/// A 1-D dataset whose length matches a declared dimension is given that
/// dimension. Ambiguity is not guessed at: a dataset whose own name is a
/// dimension takes that dimension (the CF coordinate-variable
/// convention), and anything else that stays ambiguous is reported with
/// an empty dimension list rather than a plausible-looking wrong one.
fn recover_dimension_scales(
    file: &netcrust::File,
    dimensions: &[DimensionRecord],
    variables: &mut Vec<VariableRecord>,
) {
    let Ok(datasets) = file.hdf5_root_datasets() else {
        return;
    };
    let known: std::collections::BTreeSet<String> =
        variables.iter().map(|v| v.name.clone()).collect();
    for dataset in datasets {
        let name = dataset.name().to_string();
        if known.contains(&name) {
            continue;
        }
        // A dimension with no coordinate variable is still stored as an
        // HDF5 dimension scale, and the netCDF-4 convention marks it:
        // its NAME attribute begins with the sentence below.  netCDF4
        // hides exactly those, so recovering them would invent
        // variables the C library does not report -- which broke a
        // consumer that compares `set(dataset.variables)` against an
        // expected inventory and saw `Time`, `bottom_top`, `south_north`
        // appear as fields.  This is the convention's own answer, not a
        // heuristic about attributes.
        if file
            .hdf5_dataset_attribute_string(&name, "NAME")
            .is_some_and(|value| value.starts_with(PHONY_DIMENSION_PREFIX))
        {
            continue;
        }
        let shape: Vec<usize> = dataset.shape().iter().map(|&n| n as usize).collect();
        let mut axes = Vec::with_capacity(shape.len());
        for (index, &length) in shape.iter().enumerate() {
            // The coordinate-variable convention first: a 1-D dataset
            // named after a dimension IS that dimension.
            let named = (shape.len() == 1)
                .then(|| dimensions.iter().find(|d| d.name == name && d.len == length))
                .flatten();
            let resolved = named.or_else(|| {
                let mut matches = dimensions.iter().filter(|d| d.len == length);
                let first = matches.next();
                // Only accept a length match when it is unique; two
                // dimensions of equal length cannot be told apart here.
                match (first, matches.next()) {
                    (Some(only), None) => Some(only),
                    _ => None,
                }
            });
            match resolved {
                Some(dimension) => axes.push(dimension.name.clone()),
                None => {
                    axes.clear();
                    let _ = index;
                    break;
                }
            }
        }
        let mut attributes = BTreeMap::new();
        for key in RECOVERED_ATTRIBUTES {
            if let Some(value) = file.hdf5_dataset_attribute_string(&name, key) {
                attributes.insert(key.to_string(), serde_json::Value::String(value));
            }
        }
        variables.push(VariableRecord {
            name,
            dimensions: axes,
            shape,
            // Values come back promoted; the stored type is not exposed
            // by the raw index, and saying F64 would be a claim about
            // storage that this path cannot make.
            dtype: "Promoted".to_string(),
            attributes,
            recovered_dimension_scale: true,
        });
    }
}

fn has_duplicate_dimension_lengths(file: &netcrust::File) -> bool {
    let Ok(dimensions) = file.dimensions() else {
        return true;
    };
    let mut seen = std::collections::BTreeSet::new();
    dimensions
        .iter()
        .any(|dimension| !seen.insert(dimension.len()))
}

#[derive(Serialize, Clone)]
struct MetadataProvenance {
    /// `"strict"` when every dimension came from an explicit
    /// `DIMENSION_LIST`; `"size-inferred"` when it was reconstructed.
    mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict_error: Option<String>,
    /// True when two dimensions share a length, which is exactly when
    /// size inference can silently attach the wrong axis name.
    dimension_lengths_ambiguous: bool,
}

impl MetadataProvenance {
    fn strict() -> Self {
        Self {
            mode: "strict",
            strict_error: None,
            dimension_lengths_ambiguous: false,
        }
    }
}

fn inventory(path: &Path) -> Result<(), String> {
    let (file, metadata) = open(path)?;
    let dimensions: Vec<DimensionRecord> = file
        .dimensions()
        .map_err(|error| format!("cannot read dimensions of {}: {error}", path.display()))?
        .into_iter()
        .map(|dimension| DimensionRecord {
            name: dimension.name().to_string(),
            len: dimension.len(),
            unlimited: dimension.is_unlimited(),
        })
        .collect();
    let global_attributes = attributes_map(
        &file
            .attributes()
            .map_err(|error| format!("cannot read global attributes: {error}"))?,
    );
    let mut variables: Vec<VariableRecord> = file
        .variables()
        .map_err(|error| format!("cannot read variables of {}: {error}", path.display()))?
        .into_iter()
        .map(|variable| VariableRecord {
            name: variable.name().to_string(),
            dimensions: variable
                .dimensions()
                .iter()
                .map(|dimension| dimension.name().to_string())
                .collect(),
            shape: variable.shape(),
            dtype: format!("{:?}", variable.dtype()),
            attributes: attributes_map(variable.attributes()),
            recovered_dimension_scale: false,
        })
        .collect();
    recover_dimension_scales(&file, &dimensions, &mut variables);

    let record = Inventory {
        schema: INVENTORY_SCHEMA,
        format: format!("{:?}", file.format()),
        metadata,
        dimensions,
        global_attributes,
        variables,
    };
    let text = serde_json::to_string(&record)
        .map_err(|error| format!("cannot serialize inventory: {error}"))?;
    println!("{text}");
    Ok(())
}

fn dump(path: &Path, out_dir: &Path, names: &[String], apply_mask: bool,
        apply_scale: bool) -> Result<(), String> {
    let (file, metadata) = open(path)?;
    fs::create_dir_all(out_dir)
        .map_err(|error| format!("cannot create {}: {error}", out_dir.display()))?;

    let mut records = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        // A NetCDF-4 coordinate variable is an HDF5 dimension scale and
        // is absent from `variable()` -- see `recover_dimension_scales`.
        // `read_array_f64` already falls back to a raw-HDF5 read by
        // name, so the values are reachable either way; only the
        // attribute surface differs, which is why the CF lookups below
        // go through a closure that knows which path it is on.
        let variable = file.variable(name);
        if let Some(variable) = variable.as_ref() {
            // Named refusal rather than a confusing type error from deep
            // in the reader.  netcrust promotes numeric variables to f64
            // and exposes no string/char read at all, so a character
            // variable (WRF's `Times`, ERA5's `expver`) cannot be dumped
            // here.  Say which variable and why.
            if matches!(
                variable.dtype(),
                netcrust::DataType::Char | netcrust::DataType::String
            ) {
                return Err(format!(
                    "{name} is a {:?} variable; rw_netcdf dumps numeric \
                     variables only, because the vendored netcrust reader \
                     promotes to f64 and exposes no character read.  Select \
                     a numeric variable, or read this one another way.",
                    variable.dtype()
                ));
            }
        } else if !file.has_hdf5_dataset(name) {
            return Err(format!("variable not found in {}: {name}", path.display()));
        }
        let array = file
            .read_array_f64(name)
            .map_err(|error| format!("cannot decode {name}: {error}"))?;
        let shape = array.shape().to_vec();
        let mut values = array.into_values();

        // CF unpacking, applied HERE so Python receives decoded numbers.
        // Order matters and follows CF-1.x and netCDF4-python: a value is
        // tested against the fill markers in its STORED representation,
        // before scale_factor/add_offset are applied.  Applying the
        // packing first would compare a physical quantity against a
        // sentinel that was never in that space, which is how packed
        // fields silently keep their fill values as real data.
        // A recovered dimension scale exposes only the CF STRING
        // attributes reachable by name, so numeric CF attributes are
        // absent there rather than assumed to be zero/one.  Coordinate
        // axes are not packed in any format this reads, so that is a
        // real limitation and not a silent one: it is stated in the
        // inventory by `recovered_dimension_scale`.
        let number = |key: &str| -> Option<f64> {
            variable
                .as_ref()
                .and_then(|v| v.attribute(key))
                .and_then(|a| a.as_f64())
        };
        // `--raw` reports the attributes it did NOT act on, so the
        // record still says what the file declares while `applied`
        // says whether anything was done about it.
        let cf = CfApplied {
            scale_factor: number("scale_factor"),
            add_offset: number("add_offset"),
            fill_value: number("_FillValue"),
            missing_value: number("missing_value"),
            missing_count: 0,
            applied: apply_mask || apply_scale,
        };
        let mut missing_count = 0usize;
        for value in values.iter_mut() {
            if apply_mask {
                let is_missing = cf.fill_value.is_some_and(|fill| *value == fill)
                    || cf.missing_value.is_some_and(|marker| *value == marker);
                if is_missing || !value.is_finite() {
                    if is_missing {
                        missing_count += 1;
                    }
                    *value = f64::NAN;
                    continue;
                }
            }
            // Scaling is independent of masking, exactly as it is in
            // netCDF4-python: with masking off the sentinel itself comes
            // back, and it comes back scaled like every other element.
            if apply_scale {
                if let Some(scale) = cf.scale_factor {
                    *value *= scale;
                }
                if let Some(offset) = cf.add_offset {
                    *value += offset;
                }
            }
        }
        let cf = CfApplied {
            missing_count,
            ..cf
        };

        // The index prefix keeps the filename unique and filesystem-safe
        // whatever the variable is called: NetCDF names may differ only
        // by case, which collides on Windows and macOS.
        let filename = format!("{index:04}.f64");
        let target: PathBuf = out_dir.join(&filename);
        let mut handle = fs::File::create(&target)
            .map_err(|error| format!("cannot write {}: {error}", target.display()))?;
        let mut buffer = Vec::with_capacity(values.len() * 8);
        for value in &values {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        handle
            .write_all(&buffer)
            .map_err(|error| format!("cannot write {}: {error}", target.display()))?;

        let units = match variable.as_ref() {
            Some(v) => v
                .attribute("units")
                .and_then(|attribute| attribute.as_string().map(str::to_string)),
            None => file.hdf5_dataset_attribute_string(name, "units"),
        };
        let times = units
            .as_deref()
            .and_then(parse_cf_units)
            .map(|reference| decode_times(&reference, &values))
            .transpose()?;

        records.push(DumpRecord {
            name: name.clone(),
            filename,
            shape,
            dimensions: variable
                .as_ref()
                .map(|v| {
                    v.dimensions()
                        .iter()
                        .map(|dimension| dimension.name().to_string())
                        .collect()
                })
                .unwrap_or_default(),
            dtype: "<f8",
            units,
            times,
            cf,
        });
    }

    let metadata = DumpMetadata {
        schema: DUMP_SCHEMA,
        metadata,
        variables: records,
    };
    let text = serde_json::to_string(&metadata)
        .map_err(|error| format!("cannot serialize dump metadata: {error}"))?;
    let target = out_dir.join("metadata.json");
    fs::write(&target, text)
        .map_err(|error| format!("cannot write {}: {error}", target.display()))?;
    Ok(())
}

/// A CF reference time: the unit the values are counted in, and the
/// instant they are counted from.
struct CfReference {
    unit: CfUnit,
    epoch: DateTime<Utc>,
}

enum CfUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
}

/// Parse `"<unit> since <timestamp>"`, or None when the units string is
/// not a CF reference time (which is the common case -- `K`, `m s-1`).
///
/// Deliberately strict about the unit and forgiving about the timestamp
/// spelling, because that is where real files vary: `1970-01-01`,
/// `1970-01-01 00:00:0.0`, `1970-01-01T00:00:00Z`, and a trailing
/// `+00:00` or ` UTC` all appear in published archives.  A non-UTC
/// offset is REFUSED rather than silently shifted.
fn parse_cf_units(units: &str) -> Option<CfReference> {
    let lowered = units.trim().to_ascii_lowercase();
    let (unit_text, rest) = lowered.split_once(" since ")?;
    let unit = match unit_text.trim() {
        "s" | "sec" | "secs" | "second" | "seconds" => CfUnit::Seconds,
        "min" | "mins" | "minute" | "minutes" => CfUnit::Minutes,
        "h" | "hr" | "hrs" | "hour" | "hours" => CfUnit::Hours,
        "d" | "day" | "days" => CfUnit::Days,
        _ => return None,
    };
    let epoch = parse_cf_epoch(rest.trim())?;
    Some(CfReference { unit, epoch })
}

fn parse_cf_epoch(text: &str) -> Option<DateTime<Utc>> {
    let mut stamp = text.trim();
    for suffix in [" utc", "z", " gmt", "+00:00", "+0000", " +00:00"] {
        if let Some(head) = stamp.strip_suffix(suffix) {
            stamp = head.trim();
        }
    }
    // A remaining explicit non-UTC offset must not be guessed at.
    let body = stamp.replace('t', " ");
    let body = body.trim();
    if body.len() > 10 {
        let tail = &body[10..];
        if tail.contains('+') || tail.matches('-').count() > 0 {
            return None;
        }
    }
    let (date_text, time_text) = match body.split_once(' ') {
        Some((date, time)) => (date, time.trim()),
        None => (body, ""),
    };
    let date = NaiveDate::parse_from_str(date_text, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(date_text, "%Y-%-m-%-d"))
        .ok()?;
    let time = if time_text.is_empty() {
        NaiveTime::from_hms_opt(0, 0, 0)?
    } else {
        ["%H:%M:%S%.f", "%H:%M:%S", "%H:%M", "%H"]
            .iter()
            .find_map(|format| NaiveTime::parse_from_str(time_text, format).ok())?
    };
    Some(NaiveDateTime::new(date, time).and_utc())
}

fn decode_times(reference: &CfReference, values: &[f64]) -> Result<Vec<String>, String> {
    values
        .iter()
        .map(|&value| {
            if !value.is_finite() {
                return Err(format!("time coordinate is not finite: {value}"));
            }
            // Nanosecond arithmetic keeps sub-second offsets exact for
            // the fractional day/hour spellings that appear in the wild.
            let nanos = match reference.unit {
                CfUnit::Seconds => value * 1e9,
                CfUnit::Minutes => value * 60.0 * 1e9,
                CfUnit::Hours => value * 3600.0 * 1e9,
                CfUnit::Days => value * 86_400.0 * 1e9,
            };
            if nanos.abs() >= i64::MAX as f64 {
                return Err(format!("time coordinate out of representable range: {value}"));
            }
            let instant = reference
                .epoch
                .checked_add_signed(Duration::nanoseconds(nanos.round() as i64))
                .ok_or_else(|| format!("time coordinate out of range: {value}"))?;
            Ok(instant.format("%Y-%m-%dT%H:%M:%S%.fZ").to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_units_are_not_reference_times() {
        assert!(parse_cf_units("K").is_none());
        assert!(parse_cf_units("m s-1").is_none());
        assert!(parse_cf_units("kg kg-1").is_none());
    }

    #[test]
    fn hours_since_epoch_decodes() {
        let reference = parse_cf_units("hours since 1970-01-01 00:00:00").expect("parsed");
        let times = decode_times(&reference, &[0.0, 1.5]).expect("decoded");
        assert_eq!(times[0], "1970-01-01T00:00:00Z");
        assert_eq!(times[1], "1970-01-01T01:30:00Z");
    }

    #[test]
    fn iso_and_zulu_spellings_decode() {
        for spelling in [
            "hours since 2024-03-01T00:00:00Z",
            "hours since 2024-03-01 00:00:00",
            "hours since 2024-03-01",
            "hours since 2024-03-01 00:00:0.0",
        ] {
            let reference = parse_cf_units(spelling).unwrap_or_else(|| panic!("{spelling}"));
            let times = decode_times(&reference, &[6.0]).expect("decoded");
            assert_eq!(times[0], "2024-03-01T06:00:00Z", "{spelling}");
        }
    }

    #[test]
    fn non_utc_offsets_are_refused_rather_than_shifted() {
        assert!(parse_cf_units("hours since 2024-03-01 00:00:00+05:30").is_none());
    }

    #[test]
    fn days_and_seconds_decode() {
        let days = parse_cf_units("days since 2000-01-01").expect("parsed");
        assert_eq!(
            decode_times(&days, &[1.5]).expect("decoded")[0],
            "2000-01-02T12:00:00Z"
        );
        let seconds = parse_cf_units("seconds since 2000-01-01").expect("parsed");
        assert_eq!(
            decode_times(&seconds, &[90.0]).expect("decoded")[0],
            "2000-01-01T00:01:30Z"
        );
    }
}
