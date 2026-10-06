//! CPU replay of the resident diagnostic product kernels.
//!
//! Inputs are already diagnosed float32 planes, including compound events.
//! Member order and every arithmetic operation follow `batch_products._SOURCE`.
//! No headline diagnostic, forecast state, or forecast CUDA stream is used here.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use netcrust::{DataType, File};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const REQUEST_SCHEMA: &str = "gpuwm-ensemble-diagnostic-reduce.request.v1";
pub const ABI: &str =
    "gpuwm-ensemble-diagnostic-reduce.v1\tf32-words\tf64-two-pass\tmember-order\tsha256";
const RESPONSE_SCHEMA: &str = "gpuwm-ensemble-diagnostic-reduce.v1";
const PACK_CONTRACT: &str = "gpuwm-ensemble-diagnostic-spool.v1";
const MISSING: f32 = f32::from_bits(0x7fc00000);

/// CuPy compiles the diagnostic kernels with `--ftz=true`. PTX flushes
/// subnormal f32 comparison operands and f32 endpoints of float conversions
/// to signed zero. Stored words remain unchanged until a numeric operation.
fn cuda_float(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f800000 == 0 {
        f32::from_bits(bits & 0x80000000)
    } else {
        value
    }
}

fn cuda_promote(value: f32) -> f64 {
    f64::from(cuda_float(value))
}

fn cuda_demote(value: f64) -> f32 {
    cuda_float(value as f32)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    member_order: Vec<u64>,
    shape: [usize; 2],
    valid_time: String,
    packs: Vec<PackRequest>,
    fields: Vec<FieldRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackRequest {
    path: PathBuf,
    member_ids: Vec<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldRequest {
    field: String,
    threshold_bits: Vec<u32>,
    comparison: Comparison,
    paintball: bool,
    spaghetti: bool,
    postage_stamp: bool,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Comparison {
    Ge,
    Gt,
    Le,
    Lt,
}

impl Comparison {
    fn selected(self, value: f32, threshold: f32) -> bool {
        let value = cuda_float(value);
        let threshold = cuda_float(threshold);
        match self {
            Self::Ge => value >= threshold,
            Self::Gt => value > threshold,
            Self::Le => value <= threshold,
            Self::Lt => value < threshold,
        }
    }
}

struct Pack {
    file: File,
    /// Local pack slot to its position in the complete ordered roster.
    positions: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct BufferReceipt {
    name: String,
    path: PathBuf,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: usize,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct Receipt {
    schema: &'static str,
    members: usize,
    shape: [usize; 2],
    buffers: Vec<BufferReceipt>,
}

#[derive(Clone, Copy, Debug)]
struct Statistics {
    mean: f32,
    spread: f32,
    minimum: f32,
    maximum: f32,
    finite_count: u32,
}

fn checked_elements(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |elements, &extent| {
        if extent == 0 {
            return Err("diagnostic buffer extents must be positive".into());
        }
        elements
            .checked_mul(extent)
            .ok_or_else(|| "diagnostic buffer shape overflows addressable memory".into())
    })
}

fn safe_field(field: &str) -> bool {
    let mut bytes = field.bytes();
    bytes
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn validate_request(request: &Request) -> Result<(), String> {
    if request.schema != REQUEST_SCHEMA {
        return Err(format!("diagnostic reducer requires {REQUEST_SCHEMA}"));
    }
    if request.member_order.is_empty()
        || request.member_order.len() > i32::MAX as usize
        || request.member_order.iter().collect::<HashSet<_>>().len() != request.member_order.len()
    {
        return Err(
            "member_order must be a nonempty unique roster within the kernel member range".into(),
        );
    }
    checked_elements(&[
        request.member_order.len(),
        request.shape[0],
        request.shape[1],
    ])?
    .checked_mul(4)
    .ok_or("diagnostic float32 input size overflows addressable memory")?;
    if request.valid_time.is_empty() || request.packs.is_empty() || request.fields.is_empty() {
        return Err("valid_time, packs and fields must be nonempty".into());
    }
    let mut names = HashSet::new();
    for field in &request.fields {
        if !safe_field(&field.field) || !names.insert(&field.field) {
            return Err("diagnostic fields must be unique safe NetCDF identifiers".into());
        }
        let thresholds: Vec<_> = field
            .threshold_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect();
        for (index, &threshold) in thresholds.iter().enumerate() {
            if !threshold.is_finite() || thresholds[..index].contains(&threshold) {
                return Err(format!(
                    "{} thresholds must be unique finite float32 values",
                    field.field
                ));
            }
        }
        if (field.paintball || field.spaghetti) && thresholds.is_empty() {
            return Err(format!("{} contour products need thresholds", field.field));
        }
        if field.spaghetti && request.shape.iter().any(|&extent| extent < 2) {
            return Err(format!(
                "{} spaghetti needs two points per side",
                field.field
            ));
        }
        let cells = checked_elements(&request.shape)?;
        let members = request.member_order.len();
        if !thresholds.is_empty() {
            checked_elements(&[thresholds.len(), cells])?
                .checked_mul(4)
                .ok_or("diagnostic probability size overflows addressable memory")?;
        }
        if field.paintball {
            checked_elements(&[thresholds.len(), members.div_ceil(64), cells])?
                .checked_mul(8)
                .ok_or("diagnostic paintball size overflows addressable memory")?;
        }
        if field.spaghetti {
            checked_elements(&[
                thresholds.len(),
                members,
                request.shape[0] - 1,
                request.shape[1] - 1,
            ])?;
        }
    }
    Ok(())
}

fn string_attribute(file: &File, name: &str) -> Result<String, String> {
    file.attribute(name)
        .and_then(|attribute| attribute.as_string().map(str::to_owned))
        .ok_or_else(|| format!("diagnostic pack is missing string attribute {name}"))
}

fn typed_shape(
    file: &File,
    name: &str,
    dtype: DataType,
    dimensions: &[&str],
    shape: &[usize],
) -> Result<(), String> {
    let variable = file
        .variable(name)
        .ok_or_else(|| format!("missing diagnostic variable {name}"))?;
    let observed: Vec<_> = variable
        .dimensions()
        .iter()
        .map(|dimension| dimension.name())
        .collect();
    if variable.dtype() != &dtype || observed != dimensions || variable.shape() != shape {
        return Err(format!("{name} has type {:?}, dimensions {observed:?} and shape {:?}; expected {dtype:?}, {dimensions:?} and {shape:?}", variable.dtype(), variable.shape()));
    }
    Ok(())
}

fn load_packs(request: &Request) -> Result<Vec<Pack>, String> {
    validate_request(request)?;
    let positions: HashMap<_, _> = request
        .member_order
        .iter()
        .copied()
        .enumerate()
        .map(|(position, member)| (member, position))
        .collect();
    let mut seen = HashSet::new();
    let mut packs = Vec::with_capacity(request.packs.len());
    for pack in &request.packs {
        if !pack.path.is_absolute() || pack.member_ids.is_empty() {
            return Err("diagnostic packs need absolute paths and nonempty member_ids".into());
        }
        let mut placement = Vec::with_capacity(pack.member_ids.len());
        for member in &pack.member_ids {
            let position = positions.get(member).ok_or_else(|| {
                format!("diagnostic pack member {member} is outside member_order")
            })?;
            if !seen.insert(*member) {
                return Err(format!(
                    "diagnostic member {member} occurs in more than one pack slot"
                ));
            }
            placement.push(*position);
        }
        let file = netcrust::open(&pack.path)
            .map_err(|error| format!("open {}: {error}", pack.path.display()))?;
        if string_attribute(&file, "ensemble_diagnostic_contract")? != PACK_CONTRACT {
            return Err(format!(
                "{} is not a {PACK_CONTRACT} pack",
                pack.path.display()
            ));
        }
        if string_attribute(&file, "valid_time")? != request.valid_time {
            return Err(format!(
                "{} valid_time differs from the requested hour",
                pack.path.display()
            ));
        }
        typed_shape(
            &file,
            "member_ids",
            DataType::U64,
            &["pack_member"],
            &[pack.member_ids.len()],
        )?;
        let stored = file
            .read_array::<u64>("member_ids")
            .map_err(|error| format!("read member_ids: {error}"))?;
        if stored.iter().copied().ne(pack.member_ids.iter().copied()) {
            return Err(format!(
                "{} stored member_ids differ from the requested pack roster",
                pack.path.display()
            ));
        }
        let shape = [pack.member_ids.len(), request.shape[0], request.shape[1]];
        for field in &request.fields {
            typed_shape(
                &file,
                &field.field,
                DataType::F32,
                &["pack_member", "south_north", "west_east"],
                &shape,
            )?;
        }
        packs.push(Pack {
            file,
            positions: placement,
        });
    }
    if seen.len() != request.member_order.len() {
        return Err("diagnostic packs do not cover the complete member_order roster".into());
    }
    Ok(packs)
}

fn member_values(
    packs: &[Pack],
    field: &str,
    members: usize,
    cells: usize,
) -> Result<Vec<f32>, String> {
    let mut values = vec![MISSING; members * cells];
    for pack in packs {
        let array = pack
            .file
            .read_array::<f32>(field)
            .map_err(|error| format!("read {field}: {error}"))?;
        let source = array
            .as_slice()
            .ok_or_else(|| format!("{field} diagnostic is not contiguous"))?;
        for (local, &position) in pack.positions.iter().enumerate() {
            values[position * cells..(position + 1) * cells]
                .copy_from_slice(&source[local * cells..(local + 1) * cells]);
        }
    }
    Ok(values)
}

fn statistics(values: &[f32], cell: usize, cells: usize, members: usize) -> Statistics {
    let mut finite = 0u32;
    let mut total = 0.0f64;
    let mut minimum = values[cell];
    let mut maximum = minimum;
    for member in 0..members {
        let value = values[member * cells + cell];
        if value.is_finite() {
            finite += 1;
            total += cuda_promote(value);
            // PTX compares flushed operands, then stores the original word.
            // Strict comparisons retain the first signed zero/subnormal tie.
            if cuda_float(value) < cuda_float(minimum) {
                minimum = value;
            }
            if cuda_float(value) > cuda_float(maximum) {
                maximum = value;
            }
        }
    }
    if finite != members as u32 {
        return Statistics {
            mean: MISSING,
            spread: MISSING,
            minimum: MISSING,
            maximum: MISSING,
            finite_count: finite,
        };
    }
    let average = total / members as f64;
    let mut deviations = 0.0f64;
    for member in 0..members {
        let delta = cuda_promote(values[member * cells + cell]) - average;
        let squared = delta * delta;
        deviations += squared;
    }
    Statistics {
        mean: cuda_demote(average),
        spread: if members == 1 {
            0.0
        } else {
            cuda_demote((deviations / (members - 1) as f64).sqrt())
        },
        minimum,
        maximum,
        finite_count: finite,
    }
}

fn probability(
    values: &[f32],
    cell: usize,
    cells: usize,
    members: usize,
    threshold: f32,
    relation: Comparison,
    complete: bool,
) -> f32 {
    if !complete {
        return MISSING;
    }
    let mut count = 0u32;
    for member in 0..members {
        let value = values[member * cells + cell];
        if value.is_finite() && relation.selected(value, threshold) {
            count += 1;
        }
    }
    cuda_demote(f64::from(count) / members as f64)
}

fn paintball(
    values: &[f32],
    cell: usize,
    cells: usize,
    members: usize,
    word: usize,
    threshold: f32,
    relation: Comparison,
) -> u64 {
    let mut mask = 0u64;
    for member in word * 64..members.min((word + 1) * 64) {
        let value = values[member * cells + cell];
        if value.is_finite() && relation.selected(value, threshold) {
            mask |= 1u64 << (member - word * 64);
        }
    }
    mask
}

fn spaghetti(
    values: &[f32],
    cell: usize,
    member: usize,
    shape: [usize; 2],
    threshold: f32,
    relation: Comparison,
) -> u8 {
    let [ny, nx] = shape;
    let top = member * ny * nx + (cell / (nx - 1)) * nx + cell % (nx - 1);
    let corners = [
        values[top],
        values[top + 1],
        values[top + nx + 1],
        values[top + nx],
    ];
    if corners.iter().any(|value| !value.is_finite()) {
        return 255;
    }
    corners
        .iter()
        .enumerate()
        .fold(0, |code, (corner, &value)| {
            code | if relation.selected(value, threshold) {
                1 << corner
            } else {
                0
            }
        })
}

trait StoredWord: Copy {
    const DTYPE: &'static str;
    const BYTES: usize;
    fn append(self, buffer: &mut Vec<u8>);
}
impl StoredWord for f32 {
    const DTYPE: &'static str = "<f4";
    const BYTES: usize = 4;
    fn append(self, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&self.to_bits().to_le_bytes());
    }
}
impl StoredWord for u32 {
    const DTYPE: &'static str = "<u4";
    const BYTES: usize = 4;
    fn append(self, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&self.to_le_bytes());
    }
}
impl StoredWord for u64 {
    const DTYPE: &'static str = "<u8";
    const BYTES: usize = 8;
    fn append(self, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&self.to_le_bytes());
    }
}
impl StoredWord for u8 {
    const DTYPE: &'static str = "|u1";
    const BYTES: usize = 1;
    fn append(self, buffer: &mut Vec<u8>) {
        buffer.push(self);
    }
}

fn write_buffer<T: StoredWord>(
    directory: &Path,
    field: &str,
    kind: &str,
    shape: Vec<usize>,
    values: impl IntoIterator<Item = T>,
) -> Result<BufferReceipt, String> {
    let elements = checked_elements(&shape)?;
    let bytes = elements
        .checked_mul(T::BYTES)
        .ok_or("diagnostic buffer byte count overflow")?;
    let path = directory.join(format!("{field}-{kind}.bin"));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut hash = Sha256::new();
    let mut chunk = Vec::with_capacity(65536);
    let mut count = 0usize;
    for value in values {
        count += 1;
        if count > elements {
            return Err(format!("{field}:{kind} has more values than its shape"));
        }
        value.append(&mut chunk);
        if chunk.len() == 65536 {
            writer
                .write_all(&chunk)
                .map_err(|error| error.to_string())?;
            hash.update(&chunk);
            chunk.clear();
        }
    }
    if count != elements {
        return Err(format!(
            "{field}:{kind} has {count} values, expected {elements}"
        ));
    }
    writer
        .write_all(&chunk)
        .map_err(|error| error.to_string())?;
    hash.update(&chunk);
    writer.flush().map_err(|error| error.to_string())?;
    Ok(BufferReceipt {
        name: format!("{field}:{kind}"),
        path,
        dtype: T::DTYPE,
        shape,
        bytes,
        sha256: format!("{:x}", hash.finalize()),
    })
}

fn reduce(request: &Request, output: &Path) -> Result<Receipt, String> {
    // Validate every pack and requested plane before creating any output.
    let packs = load_packs(request)?;
    fs::create_dir(output).map_err(|error| {
        format!(
            "create new reducer output directory {}: {error}",
            output.display()
        )
    })?;
    let output = output.canonicalize().map_err(|error| error.to_string())?;
    let members = request.member_order.len();
    let cells = checked_elements(&request.shape)?;
    let mut buffers = Vec::new();
    for field in &request.fields {
        let values = member_values(&packs, &field.field, members, cells)?;
        let stats: Vec<_> = (0..cells)
            .into_par_iter()
            .map(|cell| statistics(&values, cell, cells, members))
            .collect();
        let shape = request.shape.to_vec();
        buffers.push(write_buffer(
            &output,
            &field.field,
            "mean",
            shape.clone(),
            stats.iter().map(|stat| stat.mean),
        )?);
        buffers.push(write_buffer(
            &output,
            &field.field,
            "spread",
            shape.clone(),
            stats.iter().map(|stat| stat.spread),
        )?);
        buffers.push(write_buffer(
            &output,
            &field.field,
            "min",
            shape.clone(),
            stats.iter().map(|stat| stat.minimum),
        )?);
        buffers.push(write_buffer(
            &output,
            &field.field,
            "max",
            shape.clone(),
            stats.iter().map(|stat| stat.maximum),
        )?);
        buffers.push(write_buffer(
            &output,
            &field.field,
            "finite_count",
            shape,
            stats.iter().map(|stat| stat.finite_count),
        )?);
        let thresholds: Vec<_> = field
            .threshold_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect();
        if !thresholds.is_empty() {
            buffers.push(write_buffer(
                &output,
                &field.field,
                "thresholds",
                vec![thresholds.len()],
                thresholds.iter().copied(),
            )?);
            let probabilities: Vec<_> = (0..thresholds.len() * cells)
                .into_par_iter()
                .map(|index| {
                    let cell = index % cells;
                    probability(
                        &values,
                        cell,
                        cells,
                        members,
                        thresholds[index / cells],
                        field.comparison,
                        stats[cell].finite_count == members as u32,
                    )
                })
                .collect();
            buffers.push(write_buffer(
                &output,
                &field.field,
                "probability",
                vec![thresholds.len(), request.shape[0], request.shape[1]],
                probabilities,
            )?);
        }
        if field.paintball {
            let words = members.div_ceil(64);
            let masks: Vec<_> = (0..thresholds.len() * words * cells)
                .into_par_iter()
                .map(|index| {
                    paintball(
                        &values,
                        index % cells,
                        cells,
                        members,
                        (index / cells) % words,
                        thresholds[index / (words * cells)],
                        field.comparison,
                    )
                })
                .collect();
            buffers.push(write_buffer(
                &output,
                &field.field,
                "paintball",
                vec![thresholds.len(), words, request.shape[0], request.shape[1]],
                masks,
            )?);
        }
        if field.spaghetti {
            let contour_cells = (request.shape[0] - 1) * (request.shape[1] - 1);
            let crossings: Vec<_> = (0..thresholds.len() * members * contour_cells)
                .into_par_iter()
                .map(|index| {
                    spaghetti(
                        &values,
                        index % contour_cells,
                        (index / contour_cells) % members,
                        request.shape,
                        thresholds[index / (members * contour_cells)],
                        field.comparison,
                    )
                })
                .collect();
            buffers.push(write_buffer(
                &output,
                &field.field,
                "spaghetti",
                vec![
                    thresholds.len(),
                    members,
                    request.shape[0] - 1,
                    request.shape[1] - 1,
                ],
                crossings,
            )?);
        }
        if field.postage_stamp {
            buffers.push(write_buffer(
                &output,
                &field.field,
                "members",
                vec![members, request.shape[0], request.shape[1]],
                values,
            )?);
        }
    }
    Ok(Receipt {
        schema: RESPONSE_SCHEMA,
        members,
        shape: request.shape,
        buffers,
    })
}

/// Shared `rw_ensbatch` and `rw_wrfbatch` diagnostic CLI dispatch.
pub fn cli(args: &[String]) -> Result<(), String> {
    if args.len() != 4 || args[0] != "--ensemble-diagnostic-reduce" || args[2] != "--out-dir" {
        return Err(
            "usage: --ensemble-diagnostic-reduce REQUEST.json --out-dir NEW_DIRECTORY".into(),
        );
    }
    let bytes = fs::read(&args[1]).map_err(|error| format!("read diagnostic request: {error}"))?;
    let request: Request = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse diagnostic request: {error}"))?;
    let receipt = reduce(&request, Path::new(&args[3]))?;
    println!(
        "{}",
        serde_json::to_string(&receipt).map_err(|error| error.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        root: PathBuf,
        files: Vec<PathBuf>,
        directories: Vec<PathBuf>,
    }

    impl Fixture {
        fn new() -> Self {
            let root = std::env::current_dir().unwrap().join(format!(
                ".ensemble-reduce-test-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir(&root).unwrap();
            Self {
                root,
                files: Vec::new(),
                directories: Vec::new(),
            }
        }

        fn pack(
            &mut self,
            name: &str,
            ids: &[u64],
            values: &[f32],
            numeric_time: bool,
            double_field: bool,
        ) -> PathBuf {
            use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
            let path = self.root.join(name);
            let mut schema = Schema::new(NcFormat::Cdf5);
            let member = schema.def_dim("pack_member", ids.len(), false).unwrap();
            let y = schema.def_dim("south_north", 2, false).unwrap();
            let x = schema.def_dim("west_east", 2, false).unwrap();
            schema
                .put_global_attr(
                    "ensemble_diagnostic_contract",
                    AttrValue::Text(PACK_CONTRACT.into()),
                )
                .unwrap();
            let time = if numeric_time {
                AttrValue::Ints(vec![1])
            } else {
                AttrValue::Text("2026-10-04T00:00:00Z".into())
            };
            schema.put_global_attr("valid_time", time).unwrap();
            let roster = schema
                .def_var("member_ids", NcType::UInt64, &[member])
                .unwrap();
            let field = schema
                .def_var(
                    "temperature2",
                    if double_field {
                        NcType::Double
                    } else {
                        NcType::Float
                    },
                    &[member, y, x],
                )
                .unwrap();
            let mut writer = NcWriter::create(&path, schema).unwrap();
            writer.write_var(roster, VarData::U64(ids)).unwrap();
            if double_field {
                let promoted: Vec<_> = values.iter().copied().map(f64::from).collect();
                writer.write_var(field, VarData::F64(&promoted)).unwrap();
            } else {
                writer.write_var(field, VarData::F32(values)).unwrap();
            }
            writer.finish().unwrap();
            self.files.push(path.clone());
            path
        }

        fn outputs(&mut self, directory: PathBuf, receipt: &Receipt) {
            self.files
                .extend(receipt.buffers.iter().map(|buffer| buffer.path.clone()));
            self.directories.push(directory);
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            // Only exact files and directories created by this fixture are removed.
            for path in self.files.iter().rev() {
                fs::remove_file(path).unwrap();
            }
            for path in self.directories.iter().rev() {
                fs::remove_dir(path).unwrap();
            }
            fs::remove_dir(&self.root).unwrap();
        }
    }

    fn request() -> Request {
        serde_json::from_value(serde_json::json!({
            "schema": REQUEST_SCHEMA, "member_order": [9, 3], "shape": [2, 2],
            "valid_time": "2026-10-04T00:00:00Z", "packs": [{"path": "/pack.nc", "member_ids": [3, 9]}],
            "fields": [{"field": "temperature2", "threshold_bits": [1.0f32.to_bits()],
                "comparison": "ge", "paintball": true, "spaghetti": true, "postage_stamp": true}]
        })).unwrap()
    }

    #[test]
    fn statistics_use_serial_double_accumulation_and_sample_spread() {
        let values = [16777216.0, 1.0, -16777216.0];
        let stat = statistics(&values, 0, 1, 3);
        assert_eq!(stat.mean.to_bits(), ((1.0f64 / 3.0) as f32).to_bits());
        let average = 1.0f64 / 3.0;
        let mut deviations = 0.0f64;
        for value in values {
            let delta = f64::from(value) - average;
            deviations += delta * delta;
        }
        assert_eq!(
            stat.spread.to_bits(),
            ((deviations / 2.0).sqrt() as f32).to_bits()
        );
        assert_eq!(
            (stat.minimum, stat.maximum, stat.finite_count),
            (-16777216.0, 16777216.0, 3)
        );
    }

    #[test]
    fn statistics_mask_any_nonfinite_member_with_the_canonical_nan() {
        for missing in [f32::from_bits(0x7fa01234), f32::INFINITY, f32::NEG_INFINITY] {
            let stat = statistics(&[1.0, missing, 2.0], 0, 1, 3);
            for value in [stat.mean, stat.spread, stat.minimum, stat.maximum] {
                assert_eq!(value.to_bits(), 0x7fc00000);
            }
            assert_eq!(stat.finite_count, 2);
            assert_eq!(
                probability(&[1.0, missing, 2.0], 0, 1, 3, 0.0, Comparison::Ge, false).to_bits(),
                0x7fc00000
            );
            assert_eq!(
                paintball(&[1.0, missing, 2.0], 0, 1, 3, 0, 0.0, Comparison::Ge),
                5
            );
        }
        let stat = statistics(&[f32::NAN, f32::from_bits(0xffc03456)], 0, 1, 2);
        assert_eq!(stat.finite_count, 0);
        for value in [stat.mean, stat.spread, stat.minimum, stat.maximum] {
            assert_eq!(value.to_bits(), 0x7fc00000);
        }
        assert_eq!(
            paintball(
                &[f32::NAN, f32::NEG_INFINITY],
                0,
                1,
                2,
                0,
                0.0,
                Comparison::Lt
            ),
            0
        );
    }

    #[test]
    fn extreme_finite_inputs_accumulate_in_double_without_overflow() {
        let maximum = f32::MAX;
        let stat = statistics(&[maximum, maximum], 0, 1, 2);
        assert_eq!(stat.mean.to_bits(), maximum.to_bits());
        assert_eq!(stat.spread.to_bits(), 0);
        let stat = statistics(&[maximum, -maximum], 0, 1, 2);
        assert_eq!(stat.mean.to_bits(), 0);
        // The variance is finite in f64. Only its final f32 conversion overflows.
        assert_eq!(stat.spread, f32::INFINITY);
        let tiny = f32::from_bits(1);
        let stat = statistics(&[tiny, tiny], 0, 1, 2);
        assert_eq!(stat.mean.to_bits(), 0);
        assert_eq!(stat.spread.to_bits(), 0);
    }

    #[test]
    fn cuda_ftz_operations_preserve_stored_subnormal_words_and_conversion_signs() {
        let tiny = f32::from_bits(1);
        let values = [tiny, f32::from_bits(2), -tiny, -0.0];
        let stat = statistics(&values, 0, 1, 4);
        assert_eq!(stat.mean.to_bits(), 0);
        assert_eq!(stat.spread.to_bits(), 0);
        assert_eq!(stat.minimum.to_bits(), 1);
        assert_eq!(stat.maximum.to_bits(), 1);
        assert_eq!(stat.finite_count, 4);
        assert_eq!(
            probability(&values, 0, 1, 4, 0.0, Comparison::Lt, true).to_bits(),
            0
        );
        assert!(Comparison::Ge.selected(-tiny, tiny));
        assert!(!Comparison::Gt.selected(tiny, -tiny));
        assert!(Comparison::Le.selected(tiny, -tiny));
        assert!(!Comparison::Lt.selected(-tiny, tiny));
        let minimum_normal = f32::from_bits(0x00800000);
        let maximum_subnormal = f32::from_bits(0x007fffff);
        let stat = statistics(
            &[
                minimum_normal,
                minimum_normal,
                maximum_subnormal,
                maximum_subnormal,
            ],
            0,
            1,
            4,
        );
        assert_eq!(stat.mean.to_bits(), 0);
        assert_eq!(stat.spread.to_bits(), 0);
        assert_eq!(stat.minimum.to_bits(), 0x007fffff);
        assert_eq!(stat.maximum.to_bits(), 0x00800000);
        let negatives = [0x80000001, 0x80000002, 0x80000003, 0x80000004].map(f32::from_bits);
        let stat = statistics(&negatives, 0, 1, 4);
        assert_eq!(stat.mean.to_bits(), 0);
        assert_eq!(stat.spread.to_bits(), 0);
        assert_eq!(stat.minimum.to_bits(), 0x80000001);
        assert_eq!(stat.maximum.to_bits(), 0x80000001);
        assert_eq!(cuda_demote(-f64::from(tiny)).to_bits(), 0x80000000);
        let stat = statistics(&[-minimum_normal, 0.0, 0.0, 0.0], 0, 1, 4);
        assert_eq!(stat.mean.to_bits(), 0x80000000);
        assert_eq!(stat.spread.to_bits(), 0);
    }

    #[test]
    fn strict_extrema_keep_first_signed_zero_and_one_member_spread_is_positive_zero() {
        for values in [[-0.0f32, 0.0], [0.0f32, -0.0]] {
            let stat = statistics(&values, 0, 1, 2);
            assert_eq!(stat.minimum.to_bits(), values[0].to_bits());
            assert_eq!(stat.maximum.to_bits(), values[0].to_bits());
            assert_eq!(stat.mean.to_bits(), 0);
            assert_eq!(stat.spread.to_bits(), 0);
        }
        assert_eq!(statistics(&[-0.0], 0, 1, 1).spread.to_bits(), 0);
    }

    #[test]
    fn probability_relations_and_second_paintball_word_match_the_kernel() {
        let values = [0.0, 1.0, 2.0];
        for (relation, count) in [
            (Comparison::Ge, 2),
            (Comparison::Gt, 1),
            (Comparison::Le, 2),
            (Comparison::Lt, 1),
        ] {
            assert_eq!(
                probability(&values, 0, 1, 3, 1.0, relation, true).to_bits(),
                ((count as f64 / 3.0) as f32).to_bits()
            );
        }
        let mut values = vec![0.0; 67];
        for member in [0, 53, 63, 64, 66] {
            values[member] = 1.0;
        }
        assert_eq!(
            paintball(&values, 0, 1, 67, 0, 0.5, Comparison::Ge),
            1 | (1u64 << 53) | (1u64 << 63)
        );
        assert_eq!(paintball(&values, 0, 1, 67, 1, 0.5, Comparison::Ge), 5);
    }

    #[test]
    fn spaghetti_uses_clockwise_corner_bits_and_255_for_nonfinite_corners() {
        let values = [1.0, 0.0, 0.0, 1.0, f32::NAN, 0.0, 0.0, 1.0];
        assert_eq!(spaghetti(&values, 0, 0, [2, 2], 0.5, Comparison::Ge), 5);
        assert_eq!(spaghetti(&values, 0, 1, [2, 2], 0.5, Comparison::Ge), 255);
        assert_eq!(spaghetti(&[1.0; 4], 0, 0, [2, 2], 1.0, Comparison::Gt), 0);
    }

    #[test]
    fn stored_float_words_do_not_promote_nan_payloads_signed_zero_or_subnormals() {
        let words = [0x80000000, 0x7fa01234, 0x7fc05678, 0x00000001, 0x80000001];
        let mut bytes = Vec::new();
        for word in words {
            f32::from_bits(word).append(&mut bytes);
        }
        assert_eq!(
            bytes,
            words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn request_rejects_duplicate_rosters_fields_thresholds_and_unsafe_names() {
        assert!(validate_request(&request()).is_ok());
        let mut invalid = request();
        invalid.member_order[1] = invalid.member_order[0];
        assert!(validate_request(&invalid).is_err());
        let mut invalid = request();
        invalid.fields[0].field = "../temperature2".into();
        assert!(validate_request(&invalid).is_err());
        let mut invalid = request();
        invalid.fields.push(request().fields.pop().unwrap());
        assert!(validate_request(&invalid).is_err());
        let mut invalid = request();
        invalid.fields[0].threshold_bits = vec![0.0f32.to_bits(), (-0.0f32).to_bits()];
        assert!(validate_request(&invalid).is_err());
        let mut invalid = request();
        invalid.fields[0].threshold_bits = vec![f32::INFINITY.to_bits()];
        assert!(validate_request(&invalid).is_err());
        let mut invalid = request();
        invalid.shape = [1, 2];
        assert!(validate_request(&invalid).is_err());
    }

    #[test]
    fn request_does_not_coerce_nonnumeric_member_or_threshold_metadata() {
        let mut value = serde_json::to_value(serde_json::json!({
            "schema": REQUEST_SCHEMA, "member_order": ["9", 3], "shape": [2, 2], "valid_time": "hour", "packs": [], "fields": []
        })).unwrap();
        assert!(serde_json::from_value::<Request>(value.clone()).is_err());
        value["member_order"] = serde_json::json!([9, 3]);
        value["valid_time"] = serde_json::json!(5);
        assert!(serde_json::from_value::<Request>(value).is_err());
    }

    #[test]
    fn typed_packs_restore_sparse_global_member_order_and_emit_verified_raw_buffers() {
        let mut fixture = Fixture::new();
        let high_member = (1u64 << 60) + 3;
        let first_words = [0x80000000u32, 0x7fa01234, 0x00000001, 0x3f800000];
        let second_words = [0x00000000u32, 0x3f800000, 0x80000001, 0x40000000];
        let first = first_words.map(f32::from_bits);
        let second = second_words.map(f32::from_bits);
        let second_path = fixture.pack("second.nc", &[high_member], &second, false, false);
        let first_path = fixture.pack("first.nc", &[9], &first, false, false);
        let mut request = request();
        request.member_order = vec![9, high_member];
        request.packs = vec![
            PackRequest {
                path: second_path,
                member_ids: vec![high_member],
            },
            PackRequest {
                path: first_path,
                member_ids: vec![9],
            },
        ];
        let packs = load_packs(&request).unwrap();
        let values = member_values(&packs, "temperature2", 2, 4).unwrap();
        let wanted: Vec<_> = first_words.into_iter().chain(second_words).collect();
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            wanted
        );
        drop(packs);
        let output = fixture.root.join("buffers");
        let receipt = reduce(&request, &output).unwrap();
        fixture.outputs(output.clone(), &receipt);
        let kinds: Vec<_> = receipt
            .buffers
            .iter()
            .map(|buffer| buffer.name.as_str())
            .collect();
        assert_eq!(
            kinds,
            [
                "temperature2:mean",
                "temperature2:spread",
                "temperature2:min",
                "temperature2:max",
                "temperature2:finite_count",
                "temperature2:thresholds",
                "temperature2:probability",
                "temperature2:paintball",
                "temperature2:spaghetti",
                "temperature2:members"
            ]
        );
        for buffer in &receipt.buffers {
            assert!(buffer.path.is_absolute());
            let bytes = fs::read(&buffer.path).unwrap();
            assert_eq!(bytes.len(), buffer.bytes);
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), buffer.sha256);
        }
        let words = |kind: &str| -> Vec<u32> {
            let buffer = receipt
                .buffers
                .iter()
                .find(|buffer| buffer.name.ends_with(kind))
                .unwrap();
            fs::read(&buffer.path)
                .unwrap()
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        };
        assert_eq!(words(":members"), wanted);
        assert_eq!(words(":mean"), [0, 0x7fc00000, 0, 1.5f32.to_bits()]);
        assert_eq!(words(":min")[0], 0x80000000);
        assert_eq!(words(":max")[0], 0x80000000);
        assert_eq!(words(":finite_count"), [2, 1, 2, 2]);
        assert_eq!(words(":probability"), [0, 0x7fc00000, 0, 1.0f32.to_bits()]);
        assert!(reduce(&request, &output)
            .unwrap_err()
            .contains("create new reducer output directory"));
    }

    #[test]
    fn typed_pack_validation_rejects_duplicate_incomplete_mismatched_or_promoted_inputs() {
        let mut fixture = Fixture::new();
        let good = fixture.pack("good.nc", &[9, 3], &[1.0; 8], false, false);
        let numeric = fixture.pack("numeric.nc", &[9, 3], &[1.0; 8], true, false);
        let double = fixture.pack("double.nc", &[9, 3], &[1.0; 8], false, true);
        let mut request = request();
        request.packs = vec![PackRequest {
            path: good.clone(),
            member_ids: vec![9, 3],
        }];
        assert!(load_packs(&request).is_ok());
        request.packs.push(PackRequest {
            path: good.clone(),
            member_ids: vec![9, 3],
        });
        assert!(load_packs(&request)
            .err()
            .unwrap()
            .contains("occurs in more than one pack slot"));
        request.packs = vec![PackRequest {
            path: good,
            member_ids: vec![3, 9],
        }];
        assert!(load_packs(&request)
            .err()
            .unwrap()
            .contains("stored member_ids differ"));
        request.packs = vec![PackRequest {
            path: numeric,
            member_ids: vec![9, 3],
        }];
        assert!(load_packs(&request)
            .err()
            .unwrap()
            .contains("missing string attribute valid_time"));
        request.packs = vec![PackRequest {
            path: double,
            member_ids: vec![9, 3],
        }];
        assert!(load_packs(&request).err().unwrap().contains("expected F32"));
        let partial = fixture.pack("partial.nc", &[9], &[1.0; 4], false, false);
        request.packs = vec![PackRequest {
            path: partial,
            member_ids: vec![9],
        }];
        assert!(load_packs(&request)
            .err()
            .unwrap()
            .contains("do not cover the complete"));
    }
}
