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
            .map(parse_cf_units)
            .transpose()?
            .flatten()
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
#[derive(Debug)]
struct CfReference {
    unit: CfUnit,
    epoch: DateTime<Utc>,
}

#[derive(Debug)]
enum CfUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
}

/// Parse `"<unit> since <timestamp>"`.  `Ok(None)` when the units
/// string is not a CF reference time at all (which is the common case
/// -- `K`, `m s-1`); `Err` when it plainly is one but its UTC-offset
/// designator is malformed, because dumping such a variable without its
/// time axis would silently strip the calendar off a file whose author
/// wrote one down.
///
/// Deliberately strict about the unit and forgiving about the timestamp
/// spelling, because that is where real files vary: `1970-01-01`,
/// `1970-01-01 00:00:0.0`, `1970-01-01T00:00:00Z`, and a trailing
/// `+05:30` or ` UTC` all appear in published archives.  A well-formed
/// UTC offset is APPLIED -- the epoch converts to UTC -- so the decoded
/// instants land where the equivalent UTC-spelled units would put them.
fn parse_cf_units(units: &str) -> Result<Option<CfReference>, String> {
    let lowered = units.trim().to_ascii_lowercase();
    let Some((unit_text, rest)) = lowered.split_once(" since ") else {
        return Ok(None);
    };
    let unit = match unit_text.trim() {
        "s" | "sec" | "secs" | "second" | "seconds" => CfUnit::Seconds,
        "min" | "mins" | "minute" | "minutes" => CfUnit::Minutes,
        "h" | "hr" | "hrs" | "hour" | "hours" => CfUnit::Hours,
        "d" | "day" | "days" => CfUnit::Days,
        _ => return Ok(None),
    };
    Ok(parse_cf_epoch(rest.trim())?.map(|epoch| CfReference { unit, epoch }))
}

/// The widest UTC-offset designator accepted, in seconds: 18 hours
/// either side.  The widest civil offset on Earth is +14:00 (the Line
/// Islands) and the udunits/`java.time` grammars both cap the field at
/// +/-18:00; a larger number is a mistyped timestamp, not a timezone,
/// and applying it would move every decoded instant by most of a day.
const MAX_UTC_OFFSET_SECONDS: i64 = 18 * 3600;

/// `Ok(None)` when the text is not a timestamp this parser reads;
/// `Err` when its UTC-offset designator cannot be trusted, saying why.
fn parse_cf_epoch(text: &str) -> Result<Option<DateTime<Utc>>, String> {
    let mut stamp = text.trim();
    // Textual zone designators, every one of which NAMES the zero
    // offset.  udunits reads them; so do published archives.
    let mut named_utc = false;
    for suffix in [" utc", "z", " gmt"] {
        if let Some(head) = stamp.strip_suffix(suffix) {
            stamp = head.trim();
            named_utc = true;
        }
    }
    let body = stamp.replace('t', " ");
    let mut tokens = body.split_whitespace();
    let Some(date_text) = tokens.next() else {
        return Ok(None);
    };
    // A signed token is a numeric UTC offset -- a time of day is never
    // signed.  The offset may ride attached to the tail of the time
    // token (`00:00:00+05:30`) or stand alone after the date or time
    // (`1992-10-8 15:15:42.5 -6:00`, the canonical udunits spelling).
    // It is never recognised attached to a bare date, whose own `-`
    // separators make that spelling unreadable without guessing.
    let (time_text, offset_text) = match (tokens.next(), tokens.next(), tokens.next()) {
        (None, _, _) => ("", None),
        (Some(second), None, _) => match second.find(['+', '-']) {
            Some(at) => (&second[..at], Some(&second[at..])),
            None => (second, None),
        },
        (Some(second), Some(third), None) => {
            if !third.starts_with(['+', '-']) {
                return Ok(None);
            }
            if second.contains(['+', '-']) {
                return Err(format!(
                    "reference time {text:?} carries two UTC offsets; \
                     one epoch cannot be shifted twice"
                ));
            }
            (second, Some(third))
        }
        _ => return Ok(None),
    };
    let offset_seconds = match offset_text {
        Some(token) => {
            if named_utc {
                return Err(format!(
                    "reference time {text:?} names UTC and also carries \
                     the numeric offset {token:?}; an epoch with two zone \
                     designators cannot be placed on the timeline"
                ));
            }
            parse_utc_offset(token, text)?
        }
        None => 0,
    };
    let date = match NaiveDate::parse_from_str(date_text, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(date_text, "%Y-%-m-%-d"))
    {
        Ok(date) => date,
        Err(_) => return Ok(None),
    };
    let time = if time_text.is_empty() {
        NaiveTime::from_hms_opt(0, 0, 0).expect("midnight is a valid time")
    } else {
        match ["%H:%M:%S%.f", "%H:%M:%S", "%H:%M", "%H"]
            .iter()
            .find_map(|format| NaiveTime::parse_from_str(time_text, format).ok())
        {
            Some(time) => time,
            None => return Ok(None),
        }
    };
    // A timestamp at +05:00 reads five hours AHEAD of UTC, so the UTC
    // instant it names is the naive reading minus the offset.
    let epoch = NaiveDateTime::new(date, time) - Duration::seconds(offset_seconds);
    Ok(Some(epoch.and_utc()))
}

/// One numeric UTC-offset designator -- a sign, then `hh:mm`, `hhmm`,
/// or a bare hour count `h`/`hh` -- as signed seconds east of UTC.
/// These are the spellings udunits reads.  Anything else is refused
/// with the reason, because a mis-read offset does not crash: it
/// silently moves every instant decoded from the file.
fn parse_utc_offset(token: &str, timestamp: &str) -> Result<i64, String> {
    let malformed = |why: String| {
        format!(
            "UTC offset {token:?} in reference time {timestamp:?} is \
             malformed: {why}"
        )
    };
    let (sign, magnitude) = if let Some(rest) = token.strip_prefix('+') {
        (1i64, rest)
    } else if let Some(rest) = token.strip_prefix('-') {
        (-1i64, rest)
    } else {
        return Err(malformed("it does not start with a sign".into()));
    };
    if magnitude.is_empty() {
        return Err(malformed("the sign has no digits behind it".into()));
    }
    let digits = |text: &str| text.bytes().all(|byte| byte.is_ascii_digit());
    let (hour_text, minute_text) = match magnitude.split_once(':') {
        Some((hours, minutes)) => {
            if minutes.len() != 2 {
                return Err(malformed(format!(
                    "minutes must be exactly two digits, got {minutes:?}; \
                     an offset carries no seconds field"
                )));
            }
            (hours, minutes)
        }
        None => match magnitude.len() {
            1 | 2 => (magnitude, "0"),
            4 => magnitude.split_at(2),
            _ => {
                return Err(malformed(format!(
                    "a colonless offset must be one or two hour digits or \
                     exactly four digits (hhmm); {magnitude:?} cannot be \
                     split into hours and minutes without guessing"
                )));
            }
        },
    };
    if hour_text.is_empty() || hour_text.len() > 2 || !digits(hour_text) || !digits(minute_text) {
        return Err(malformed(format!(
            "{hour_text:?} hours and {minute_text:?} minutes are not one- \
             or two-digit numbers"
        )));
    }
    let hours: i64 = hour_text.parse().expect("checked ascii digits");
    let minutes: i64 = minute_text.parse().expect("checked ascii digits");
    if minutes >= 60 {
        return Err(malformed(format!(
            "{minutes} is not a minute count below 60"
        )));
    }
    let magnitude_seconds = hours * 3600 + minutes * 60;
    if magnitude_seconds > MAX_UTC_OFFSET_SECONDS {
        return Err(malformed(
            "its magnitude passes 18:00, and no timezone is that far from \
             UTC (the widest civil offset on Earth is +14:00)"
            .into(),
        ));
    }
    Ok(sign * magnitude_seconds)
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

    /// A parsed CF reference, for units that must be well-formed ones.
    fn reference(units: &str) -> CfReference {
        parse_cf_units(units)
            .unwrap_or_else(|error| panic!("{units}: {error}"))
            .unwrap_or_else(|| panic!("{units} is a CF reference time"))
    }

    #[test]
    fn plain_units_are_not_reference_times() {
        for units in ["K", "m s-1", "kg kg-1"] {
            assert!(
                parse_cf_units(units).expect("nothing to malform").is_none(),
                "{units}"
            );
        }
    }

    #[test]
    fn hours_since_epoch_decodes() {
        let reference = reference("hours since 1970-01-01 00:00:00");
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
            let reference = reference(spelling);
            let times = decode_times(&reference, &[6.0]).expect("decoded");
            assert_eq!(times[0], "2024-03-01T06:00:00Z", "{spelling}");
        }
    }

    /// The one exact equivalence the offset handling promises: units
    /// carrying a UTC-offset designator decode to the SAME instants as
    /// the same epoch respelled in UTC by hand.  Every offset spelling
    /// udunits reads is here -- `+hh:mm`, `+hhmm`, bare `-h`/`-hh`,
    /// attached and spaced, and the zero-offset designators.
    #[test]
    fn offset_references_decode_like_their_utc_respelling() {
        for (offset_spelling, utc_respelling) in [
            // +05:00 reads five hours EAST: the epoch is EARLIER in UTC.
            (
                "hours since 2020-01-01 00:00:00 +05:00",
                "hours since 2019-12-31 19:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00+0500",
                "hours since 2019-12-31 19:00:00",
            ),
            // Negative forms read WEST: the epoch is LATER in UTC.
            (
                "hours since 2020-01-01 00:00:00 -06:00",
                "hours since 2020-01-01 06:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00-0600",
                "hours since 2020-01-01 06:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00 -6",
                "hours since 2020-01-01 06:00:00",
            ),
            // A half-hour zone exercises both halves of h*3600 + m*60.
            (
                "hours since 2020-01-01 00:00:00 +05:30",
                "hours since 2019-12-31 18:30:00",
            ),
            (
                "hours since 2020-01-01 00:00:00+0530",
                "hours since 2019-12-31 18:30:00",
            ),
            // Zero-offset designators are exactly UTC.
            (
                "hours since 2020-01-01 00:00:00Z",
                "hours since 2020-01-01 00:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00 +00:00",
                "hours since 2020-01-01 00:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00-0000",
                "hours since 2020-01-01 00:00:00",
            ),
            // A date-only stamp still takes a spaced offset (midnight).
            (
                "hours since 2020-01-01 +01:00",
                "hours since 2019-12-31 23:00:00",
            ),
        ] {
            let with_offset = reference(offset_spelling);
            let respelled = reference(utc_respelling);
            assert_eq!(
                with_offset.epoch, respelled.epoch,
                "{offset_spelling} vs {utc_respelling}"
            );
            assert_eq!(
                decode_times(&with_offset, &[0.0, 1.5]).expect("decoded"),
                decode_times(&respelled, &[0.0, 1.5]).expect("decoded"),
                "{offset_spelling}"
            );
        }
    }

    /// Ground truth for the shift direction, pinned as absolute strings
    /// so a sign error that broke both sides of the equivalence test the
    /// same way would still be caught.
    #[test]
    fn a_positive_offset_epoch_lands_earlier_on_the_utc_timeline() {
        let reference = reference("hours since 2020-01-01 00:00:00 +05:30");
        assert_eq!(
            decode_times(&reference, &[0.0, 5.5]).expect("decoded"),
            vec!["2019-12-31T18:30:00Z", "2020-01-01T00:00:00Z"],
        );
    }

    #[test]
    fn a_negative_offset_epoch_lands_later_on_the_utc_timeline() {
        let reference = reference("seconds since 2020-06-01 12:00:00 -06");
        assert_eq!(
            decode_times(&reference, &[0.0]).expect("decoded")[0],
            "2020-06-01T18:00:00Z",
        );
    }

    /// The designator bounds: 59 minutes and 18:00 are the last values
    /// inside the grammar, and the first value past each is refused by
    /// name.  udunits and java.time both cap the field at +/-18:00; the
    /// widest civil offset on Earth is +14:00.
    #[test]
    fn offset_bounds_admit_18_hours_and_59_minutes_and_nothing_past_them() {
        assert_eq!(
            reference("hours since 2020-01-01 00:00:00 +18:00").epoch,
            reference("hours since 2019-12-31 06:00:00").epoch,
        );
        assert_eq!(
            reference("hours since 2020-01-01 00:59:00 +00:59").epoch,
            reference("hours since 2020-01-01 00:00:00").epoch,
        );
        for (units, names) in [
            ("hours since 2020-01-01 00:00:00 +18:01", "18:00"),
            ("hours since 2020-01-01 00:00:00 -19:00", "18:00"),
            ("hours since 2020-01-01 00:00:00 +05:60", "below 60"),
            ("hours since 2020-01-01 00:00:00 +05:61", "below 60"),
        ] {
            let error = match parse_cf_units(units) {
                Err(error) => error,
                Ok(parsed) => panic!("{units} was not refused: {parsed:?}"),
            };
            assert!(error.contains("malformed"), "{units}: {error}");
            assert!(error.contains(names), "{units}: {error}");
        }
    }

    /// Every refusal names what is wrong; nothing is guessed at and
    /// nothing is silently dropped.  Silently dropping was the old
    /// behaviour, and it stripped the time axis off any file whose
    /// author spelled a real timezone.
    #[test]
    fn malformed_offsets_are_refused_with_the_reason() {
        for (units, names) in [
            // Three packed digits cannot be split into hours and
            // minutes without guessing between 5:30 and 53:0.
            ("hours since 2020-01-01 00:00:00 +530", "guessing"),
            ("hours since 2020-01-01 00:00:00 +05300", "guessing"),
            // udunits offsets carry no seconds field.
            ("hours since 2020-01-01 00:00:00 +05:30:00", "two digits"),
            ("hours since 2020-01-01 00:00:00 +05:3", "two digits"),
            // A sign with nothing behind it, a missing hours field, and
            // letters where digits go.
            ("hours since 2020-01-01 00:00:00 +", "no digits"),
            ("hours since 2020-01-01 00:00:00 +:30", "hours"),
            ("hours since 2020-01-01 00:00:00 +0a", "hours"),
            // Two zone designators cannot both place the epoch.
            ("hours since 2020-01-01 00:00:00+05:00Z", "names utc"),
            (
                "hours since 2020-01-01 00:00:00+05:00 +06:00",
                "two utc offsets",
            ),
        ] {
            let error = match parse_cf_units(units) {
                Err(error) => error,
                Ok(parsed) => panic!("{units} was not refused: {parsed:?}"),
            };
            assert!(
                error.to_ascii_lowercase().contains(names),
                "{units}: {error}"
            );
        }
    }

    /// `minutes since` is a spelling this decoder claimed and never
    /// checked.
    ///
    /// Both halves of `value * 60.0 * 1e9` and the whole
    /// `"min" | "mins" | "minute" | "minutes"` match arm could be
    /// deleted or turned into another operator without a single test
    /// noticing, which is a history file whose valid times are wrong by
    /// a factor of sixty and an initial condition read off the wrong
    /// frame.  The 90-minute value is the one that separates them: it
    /// decodes to 01:30 under a multiply, to 00:01:00.00000009 under an
    /// add, and to 00:00:00.0000000015 under a divide.
    #[test]
    fn minutes_since_decodes_as_minutes_and_not_as_some_other_arithmetic() {
        let minutes = reference("minutes since 2024-03-01 00:00:00");
        let times = decode_times(&minutes, &[0.0, 90.0]).expect("decoded");
        assert_eq!(times[0], "2024-03-01T00:00:00Z");
        assert_eq!(times[1], "2024-03-01T01:30:00Z");
        for spelling in ["min", "mins", "minute", "minutes"] {
            let spelled = reference(&format!("{spelling} since 2024-03-01"));
            assert_eq!(
                decode_times(&spelled, &[90.0]).expect("decoded")[0],
                "2024-03-01T01:30:00Z",
                "{spelling}"
            );
        }
    }

    /// The unpadded date spelling, which the epoch parser carries a
    /// second format for and no test ever reached.
    #[test]
    fn unpadded_month_and_day_spellings_decode() {
        let days = reference("days since 2000-1-1");
        assert_eq!(
            decode_times(&days, &[1.5]).expect("decoded")[0],
            "2000-01-02T12:00:00Z"
        );
        let hours = reference("hours since 2000-1-1 6:30");
        assert_eq!(
            decode_times(&hours, &[0.0]).expect("decoded")[0],
            "2000-01-01T06:30:00Z"
        );
    }

    #[test]
    fn days_and_seconds_decode() {
        let days = reference("days since 2000-01-01");
        assert_eq!(
            decode_times(&days, &[1.5]).expect("decoded")[0],
            "2000-01-02T12:00:00Z"
        );
        let seconds = reference("seconds since 2000-01-01");
        assert_eq!(
            decode_times(&seconds, &[90.0]).expect("decoded")[0],
            "2000-01-01T00:01:30Z"
        );
    }

    // ------------------------------------------------------------------
    // JSON attribute encoding.  The single/plural distinction below is a
    // CONTRACT with the Python side: netCDF4-python hands back a scalar
    // for a one-element attribute and an array otherwise, and selector
    // resolution compares `units` and `standard_name` against strings.
    // An attribute that flips shape stops resolving without an error.

    #[test]
    fn char_and_single_string_attributes_encode_as_json_strings() {
        use netcrust::AttributeValue as V;
        assert_eq!(
            attribute_json(&V::Chars("degrees_east".into())),
            serde_json::json!("degrees_east")
        );
        assert_eq!(
            attribute_json(&V::Strings(vec!["longitude".into()])),
            serde_json::json!("longitude")
        );
    }

    #[test]
    fn plural_string_attributes_encode_as_json_arrays() {
        use netcrust::AttributeValue as V;
        assert_eq!(
            attribute_json(&V::Strings(vec!["a".into(), "b".into()])),
            serde_json::json!(["a", "b"])
        );
    }

    #[test]
    fn single_numbers_are_scalars_and_plural_numbers_are_arrays() {
        use netcrust::AttributeValue as V;
        assert_eq!(attribute_json(&V::Ints(vec![7])), serde_json::json!(7.0));
        assert_eq!(
            attribute_json(&V::Floats(vec![1.5, 2.5])),
            serde_json::json!([1.5, 2.5])
        );
        assert_eq!(
            attribute_json(&V::Doubles(vec![0.5])),
            serde_json::json!(0.5)
        );
        // JSON has no NaN; the numeric path must emit null, not panic.
        assert_eq!(
            attribute_json(&V::Doubles(vec![f64::NAN])),
            serde_json::Value::Null
        );
    }

    /// i64/u64 take their own arms because f64 rounds them beyond 2^53;
    /// each arm has the same single/plural guard as the string arm.
    #[test]
    fn wide_integers_encode_exactly_with_the_same_shape_rule() {
        use netcrust::AttributeValue as V;
        let big = (1i64 << 53) + 1;
        assert_eq!(
            attribute_json(&V::Int64s(vec![big])),
            serde_json::json!(9007199254740993i64)
        );
        assert_eq!(
            attribute_json(&V::Int64s(vec![1, 2])),
            serde_json::json!([1, 2])
        );
        assert_eq!(
            attribute_json(&V::UInt64s(vec![u64::MAX])),
            serde_json::json!(18446744073709551615u64)
        );
        assert_eq!(
            attribute_json(&V::UInt64s(vec![3, 4])),
            serde_json::json!([3, 4])
        );
    }

    // ------------------------------------------------------------------
    // File-backed decode contracts.  The classic fixtures are BUILT here
    // by the workspace's own writer (an independent implementation from
    // the netcrust read path under test); the NetCDF-4/HDF5 fixtures are
    // checked in under tests/fixtures because nothing in the Rust stack
    // writes HDF5, with tests/fixtures/make_fixtures.py to regenerate.

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    /// A scratch directory unique to one test, wiped at entry so a
    /// failed earlier run cannot leak state into this one.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("rw-netcdf-suite-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// One classic (CDF-1) file carrying the whole CF unpack surface:
    /// a packed short with both fill markers, a CF time coordinate, a
    /// char variable (the named refusal), and two global attributes.
    ///
    /// Data layout of `packed`, stored as i16:
    ///   [-32768 (_FillValue), -32767 (missing_value), 4, 6]
    /// so with scale_factor 0.5 and add_offset 100 the decoded truth is
    ///   [NaN, NaN, 102.0, 103.0]      (mask on, scale on)
    ///   [-32768.0, -32767.0, 4.0, 6.0] (raw)
    ///   [-16284.0, -16283.5, 102.0, 103.0] (mask off, scale on)
    fn write_packed_classic(path: &Path) {
        use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
        let mut schema = Schema::new(NcFormat::Classic);
        let x = schema.def_dim("x", 4, false).expect("dim");
        schema
            .put_global_attr("title", AttrValue::Text("packed fixture".into()))
            .expect("gattr");
        schema
            .put_global_attr("version", AttrValue::Ints(vec![3]))
            .expect("gattr");
        let packed = schema.def_var("packed", NcType::Short, &[x]).expect("var");
        schema
            .put_var_attr(packed, "scale_factor", AttrValue::Doubles(vec![0.5]))
            .expect("attr");
        schema
            .put_var_attr(packed, "add_offset", AttrValue::Doubles(vec![100.0]))
            .expect("attr");
        schema
            .put_var_attr(packed, "_FillValue", AttrValue::Shorts(vec![-32768]))
            .expect("attr");
        schema
            .put_var_attr(packed, "missing_value", AttrValue::Shorts(vec![-32767]))
            .expect("attr");
        let hours = schema.def_var("hours", NcType::Double, &[x]).expect("var");
        schema
            .put_var_attr(
                hours,
                "units",
                AttrValue::Text("hours since 2024-03-01 00:00:00".into()),
            )
            .expect("attr");
        let label = schema.def_var("label", NcType::Char, &[x]).expect("var");
        let mut writer = NcWriter::create(path, schema).expect("create");
        writer
            .write_var(packed, VarData::I16(&[-32768, -32767, 4, 6]))
            .expect("write packed");
        writer
            .write_var(hours, VarData::F64(&[0.0, 1.5, 24.0, 36.0]))
            .expect("write hours");
        writer
            .write_var(label, VarData::Char(b"abcd"))
            .expect("write label");
        writer.finish().expect("finish");
    }

    fn read_f64_plane(path: &Path) -> Vec<f64> {
        let bytes = fs::read(path).expect("read plane");
        assert_eq!(bytes.len() % 8, 0, "not a whole number of f64s");
        bytes
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn read_metadata(out_dir: &Path) -> serde_json::Value {
        let text = fs::read_to_string(out_dir.join("metadata.json"))
            .expect("read metadata.json");
        serde_json::from_str(&text).expect("parse metadata.json")
    }

    fn record<'a>(metadata: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
        metadata["variables"]
            .as_array()
            .expect("variables array")
            .iter()
            .find(|record| record["name"] == name)
            .unwrap_or_else(|| panic!("{name} not in metadata"))
    }

    #[test]
    fn default_dump_masks_in_stored_space_then_scales() {
        let dir = scratch("dump-default");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let out = dir.join("out");
        dump(&source, &out, &["packed".into()], true, true).expect("dump");

        let values = read_f64_plane(&out.join("0000.f64"));
        assert!(values[0].is_nan(), "fill value must mask to NaN");
        assert!(values[1].is_nan(), "missing value must mask to NaN");
        assert_eq!(&values[2..], &[102.0, 103.0]);

        let metadata = read_metadata(&out);
        assert_eq!(metadata["schema"], "gpuwm-rw-netcdf-dump-v1");
        let packed = record(&metadata, "packed");
        assert_eq!(packed["filename"], "0000.f64");
        assert_eq!(packed["shape"], serde_json::json!([4]));
        assert_eq!(packed["dimensions"], serde_json::json!(["x"]));
        assert_eq!(packed["dtype"], "<f8");
        assert_eq!(packed["cf"]["scale_factor"], 0.5);
        assert_eq!(packed["cf"]["add_offset"], 100.0);
        assert_eq!(packed["cf"]["fill_value"], -32768.0);
        assert_eq!(packed["cf"]["missing_value"], -32767.0);
        assert_eq!(packed["cf"]["missing_count"], 2);
        assert_eq!(packed["cf"]["applied"], true);
        assert!(
            packed.get("times").is_none(),
            "a packed physical variable has no time axis"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn raw_dump_preserves_stored_sentinels_and_reports_not_applied() {
        let dir = scratch("dump-raw");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let out = dir.join("out");
        dump(&source, &out, &["packed".into()], false, false).expect("dump");

        assert_eq!(
            read_f64_plane(&out.join("0000.f64")),
            &[-32768.0, -32767.0, 4.0, 6.0]
        );
        let metadata = read_metadata(&out);
        let packed = record(&metadata, "packed");
        // The declared attributes still travel; `applied` says nothing
        // was done about them, and nothing was masked.
        assert_eq!(packed["cf"]["scale_factor"], 0.5);
        assert_eq!(packed["cf"]["missing_count"], 0);
        assert_eq!(packed["cf"]["applied"], false);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_mask_dump_scales_the_surviving_sentinels_like_netcdf4_python() {
        let dir = scratch("dump-nomask");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let out = dir.join("out");
        dump(&source, &out, &["packed".into()], false, true).expect("dump");

        assert_eq!(
            read_f64_plane(&out.join("0000.f64")),
            &[-16284.0, -16283.5, 102.0, 103.0]
        );
        let metadata = read_metadata(&out);
        let packed = record(&metadata, "packed");
        assert_eq!(packed["cf"]["missing_count"], 0);
        assert_eq!(packed["cf"]["applied"], true);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cf_time_variable_gets_decoded_instants_in_the_metadata() {
        let dir = scratch("dump-times");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let out = dir.join("out");
        dump(&source, &out, &["hours".into()], true, true).expect("dump");

        assert_eq!(
            read_f64_plane(&out.join("0000.f64")),
            &[0.0, 1.5, 24.0, 36.0]
        );
        let hours = read_metadata(&out);
        let hours = record(&hours, "hours");
        assert_eq!(hours["units"], "hours since 2024-03-01 00:00:00");
        assert_eq!(
            hours["times"],
            serde_json::json!([
                "2024-03-01T00:00:00Z",
                "2024-03-01T01:30:00Z",
                "2024-03-02T00:00:00Z",
                "2024-03-02T12:00:00Z",
            ])
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A classic file whose single `hours` variable carries *units*.
    fn write_hours_classic(path: &Path, units: &str, values: &[f64]) {
        use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
        let mut schema = Schema::new(NcFormat::Classic);
        let x = schema.def_dim("x", values.len(), false).expect("dim");
        let hours = schema.def_var("hours", NcType::Double, &[x]).expect("var");
        schema
            .put_var_attr(hours, "units", AttrValue::Text(units.into()))
            .expect("attr");
        let mut writer = NcWriter::create(path, schema).expect("create");
        writer.write_var(hours, VarData::F64(values)).expect("write");
        writer.finish().expect("finish");
    }

    /// The writer-path round trip for an offset-bearing reference: the
    /// units travel through the workspace's own writer, back through
    /// the netcrust read path, and decode to the same UTC instants the
    /// UTC respelling of that epoch would give.
    #[test]
    fn a_written_offset_reference_dumps_utc_instants() {
        let dir = scratch("dump-offset");
        let source = dir.join("offset.nc");
        write_hours_classic(
            &source,
            "hours since 2024-03-01 05:30:00 +05:30",
            &[0.0, 12.0],
        );
        let out = dir.join("out");
        dump(&source, &out, &["hours".into()], true, true).expect("dump");

        let metadata = read_metadata(&out);
        let hours = record(&metadata, "hours");
        assert_eq!(hours["units"], "hours since 2024-03-01 05:30:00 +05:30");
        assert_eq!(
            hours["times"],
            serde_json::json!(["2024-03-01T00:00:00Z", "2024-03-01T12:00:00Z"])
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A malformed offset fails the whole dump with its sentence, so a
    /// mis-spelled zone can never silently strip a file's time axis.
    #[test]
    fn a_malformed_offset_fails_the_dump_by_name() {
        let dir = scratch("dump-badoffset");
        let source = dir.join("bad.nc");
        write_hours_classic(&source, "hours since 2024-03-01 00:00:00 +19:00", &[0.0]);
        let out = dir.join("out");
        let error = dump(&source, &out, &["hours".into()], true, true)
            .expect_err("a malformed offset must fail the dump");
        assert!(error.contains("+19:00"), "{error}");
        assert!(error.contains("18:00"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn char_variables_are_refused_by_name_and_missing_ones_by_name() {
        let dir = scratch("dump-refusals");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let out = dir.join("out");

        let refusal = dump(&source, &out, &["label".into()], true, true)
            .expect_err("char variables cannot be dumped");
        assert!(refusal.contains("label"), "{refusal}");
        assert!(refusal.contains("numeric"), "{refusal}");

        let missing = dump(&source, &out, &["absent".into()], true, true)
            .expect_err("unknown variables are refused");
        assert!(missing.contains("variable not found"), "{missing}");
        assert!(missing.contains("absent"), "{missing}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A recovered dimension scale has no entry in the variable table,
    /// so its dump takes the raw-HDF5 path end to end: existence via
    /// `has_hdf5_dataset`, units via the by-name attribute read, and an
    /// empty dimension list rather than an invented one.
    #[test]
    fn a_recovered_coordinate_dumps_through_the_raw_hdf5_path() {
        let dir = scratch("dump-recovered");
        let out = dir.join("out");
        dump(&fixture("axes.nc4"), &out, &["time".into()], true, true)
            .expect("dump recovered scale");

        assert_eq!(read_f64_plane(&out.join("0000.f64")), &[0.0, 1.0, 2.0, 3.0]);
        let metadata = read_metadata(&out);
        let time = record(&metadata, "time");
        assert_eq!(time["units"], "hours since 2024-03-01");
        assert_eq!(
            time["times"],
            serde_json::json!([
                "2024-03-01T00:00:00Z",
                "2024-03-01T01:00:00Z",
                "2024-03-01T02:00:00Z",
                "2024-03-01T03:00:00Z",
            ])
        );
        assert_eq!(time["dimensions"], serde_json::json!([]));

        let missing = dump(&fixture("axes.nc4"), &out, &["absent".into()], true, true)
            .expect_err("unknown names are refused on HDF5 files too");
        assert!(missing.contains("variable not found"), "{missing}");
        let _ = fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // Metadata provenance: strict against size-inferred.

    #[test]
    fn a_classic_file_opens_strict_with_nothing_to_confess() {
        let dir = scratch("open-strict");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let (_, provenance) = open(&source).expect("open");
        assert_eq!(provenance.mode, "strict");
        assert_eq!(provenance.strict_error, None);
        assert!(!provenance.dimension_lengths_ambiguous);
        let _ = fs::remove_dir_all(&dir);
    }

    /// sizes.nc4 is a bare HDF5 file with no DIMENSION_LIST anywhere:
    /// strict reconstruction must fail, the size-inferred fallback must
    /// open it, and the provenance must carry the strict error so a
    /// consumer can refuse the guess.
    #[test]
    fn a_bare_hdf5_file_falls_back_to_size_inferred_metadata() {
        let (file, provenance) = open(&fixture("sizes.nc4")).expect("open");
        assert_eq!(provenance.mode, "size-inferred");
        let error = provenance.strict_error.expect("strict error is reported");
        assert!(error.contains("DIMENSION_LIST"), "{error}");
        let names: Vec<String> = file
            .variables()
            .expect("variables")
            .into_iter()
            .map(|variable| variable.name().to_string())
            .collect();
        assert!(names.contains(&"first".to_string()), "{names:?}");
        assert!(names.contains(&"second".to_string()), "{names:?}");
    }

    /// mixed.nc4 keeps its real dimension table (row and col, both 3)
    /// into the fallback, so the length collision that can mis-name an
    /// axis under size inference MUST be confessed.
    #[test]
    fn equal_dimension_lengths_are_confessed_as_ambiguous() {
        let (_, provenance) = open(&fixture("mixed.nc4")).expect("open");
        assert_eq!(provenance.mode, "size-inferred");
        assert!(provenance.strict_error.is_some());
        assert!(
            provenance.dimension_lengths_ambiguous,
            "row and col share a length; size inference cannot tell them apart"
        );
    }

    #[test]
    fn duplicate_dimension_lengths_are_detected_and_absence_is_clean() {
        use netcdf_writer::{NcFormat, NcType, NcWriter, Schema, VarData};
        let dir = scratch("dup-dims");

        let twins = dir.join("twins.nc");
        let mut schema = Schema::new(NcFormat::Classic);
        let a = schema.def_dim("a", 3, false).expect("dim");
        let b = schema.def_dim("b", 3, false).expect("dim");
        let va = schema.def_var("va", NcType::Double, &[a]).expect("var");
        let vb = schema.def_var("vb", NcType::Double, &[b]).expect("var");
        let mut writer = NcWriter::create(&twins, schema).expect("create");
        writer.write_var(va, VarData::F64(&[1.0, 2.0, 3.0])).expect("write");
        writer.write_var(vb, VarData::F64(&[4.0, 5.0, 6.0])).expect("write");
        writer.finish().expect("finish");
        let file = netcrust::File::open(&twins).expect("open");
        assert!(has_duplicate_dimension_lengths(&file));

        let distinct = dir.join("distinct.nc");
        let mut schema = Schema::new(NcFormat::Classic);
        let a = schema.def_dim("a", 3, false).expect("dim");
        let b = schema.def_dim("b", 4, false).expect("dim");
        let va = schema.def_var("va", NcType::Double, &[a]).expect("var");
        let vb = schema.def_var("vb", NcType::Double, &[b]).expect("var");
        let mut writer = NcWriter::create(&distinct, schema).expect("create");
        writer.write_var(va, VarData::F64(&[1.0, 2.0, 3.0])).expect("write");
        writer
            .write_var(vb, VarData::F64(&[4.0, 5.0, 6.0, 7.0]))
            .expect("write");
        writer.finish().expect("finish");
        let file = netcrust::File::open(&distinct).expect("open");
        assert!(!has_duplicate_dimension_lengths(&file));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn attributes_map_carries_every_attribute_by_its_own_name() {
        let dir = scratch("attrs-map");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let file = netcrust::File::open(&source).expect("open");
        let map = attributes_map(&file.attributes().expect("attributes"));
        assert_eq!(map.len(), 2, "{map:?}");
        assert_eq!(map["title"], serde_json::json!("packed fixture"));
        assert_eq!(map["version"], serde_json::json!(3.0));
        let _ = fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // Dimension-scale recovery against the checked-in NetCDF-4 fixture.

    fn recovered_axes() -> Vec<VariableRecord> {
        let (file, provenance) = open(&fixture("axes.nc4")).expect("open");
        assert_eq!(provenance.mode, "strict", "{:?}", provenance.strict_error);
        let dimensions: Vec<DimensionRecord> = file
            .dimensions()
            .expect("dimensions")
            .into_iter()
            .map(|dimension| DimensionRecord {
                name: dimension.name().to_string(),
                len: dimension.len(),
                unlimited: dimension.is_unlimited(),
            })
            .collect();
        let mut variables: Vec<VariableRecord> = file
            .variables()
            .expect("variables")
            .into_iter()
            .map(|variable| VariableRecord {
                name: variable.name().to_string(),
                dimensions: Vec::new(),
                shape: variable.shape(),
                dtype: String::new(),
                attributes: BTreeMap::new(),
                recovered_dimension_scale: false,
            })
            .collect();
        assert_eq!(
            variables.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
            vec!["t2"],
            "the reader's variable table must omit the dimension scales, \
             or this fixture no longer tests the recovery at all"
        );
        recover_dimension_scales(&file, &dimensions, &mut variables);
        variables
    }

    #[test]
    fn coordinate_variables_are_recovered_with_their_own_dimensions() {
        let variables = recovered_axes();
        let mut names: Vec<&str> = variables
            .iter()
            .filter(|v| v.recovered_dimension_scale)
            .map(|v| v.name.as_str())
            .collect();
        names.sort_unstable();
        assert_eq!(names, vec!["corners", "lat", "lon", "time", "zvals"]);

        // The name rule: lon and lat share length 3, so only the
        // coordinate-variable convention can attach each scale to its
        // OWN dimension rather than the first one of matching length.
        for name in ["time", "lon", "lat", "zvals"] {
            let record = variables
                .iter()
                .find(|v| v.name == name)
                .unwrap_or_else(|| panic!("{name} not recovered"));
            assert_eq!(
                record.dimensions,
                vec![name.to_string()],
                "{name} must resolve to its own dimension"
            );
            assert_eq!(record.dtype, "Promoted");
        }

        let time = variables.iter().find(|v| v.name == "time").expect("time");
        assert_eq!(time.shape, vec![4]);
        assert_eq!(
            time.attributes.get("units"),
            Some(&serde_json::json!("hours since 2024-03-01"))
        );
        assert_eq!(
            time.attributes.get("standard_name"),
            Some(&serde_json::json!("time"))
        );
        assert_eq!(
            time.attributes.get("calendar"),
            Some(&serde_json::json!("standard"))
        );
        assert_eq!(time.attributes.get("axis"), Some(&serde_json::json!("T")));
        assert_eq!(
            time.attributes.get("long_name"),
            None,
            "the fixture writes no long_name; inventing one would mean \
             the probe list stopped being read by name"
        );
    }

    /// The unique-length rule: a 2-D dataset can never take the name
    /// rule, so each axis of `corners` (5 x 6) resolves only because
    /// exactly one dimension has that length.
    #[test]
    fn a_two_dimensional_scale_resolves_axes_by_unique_length() {
        let variables = recovered_axes();
        let corners = variables
            .iter()
            .find(|v| v.name == "corners")
            .expect("corners recovered");
        assert_eq!(corners.shape, vec![5, 6]);
        assert_eq!(corners.dimensions, vec!["corners", "zvals"]);
        assert_eq!(
            corners.attributes.get("long_name"),
            Some(&serde_json::json!("cell corner offsets"))
        );
    }

    #[test]
    fn phony_dimensions_and_known_variables_are_not_recovered() {
        let variables = recovered_axes();
        assert!(
            !variables.iter().any(|v| v.name == "bnds"),
            "bnds has no coordinate variable; recovering it would invent \
             a variable netCDF4-python does not report"
        );
        assert_eq!(
            variables.iter().filter(|v| v.name == "t2").count(),
            1,
            "t2 is already in the variable table and must not be doubled"
        );
    }
}
