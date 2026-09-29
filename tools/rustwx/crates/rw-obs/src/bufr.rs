//! FM 94 BUFR, editions 3 and 4, decoded against the WMO master tables
//! vendored in `tables/bufr/` (Table B `element.table`, Table D
//! `sequence.def`, master table version 46; see `tables/bufr/NOTICE.md`).
//!
//! What this reader does: the six sections, the descriptor expansion
//! (Table D sequences, fixed and delayed replication including the
//! 031000 to 031002 factors and 031011/031012 repetition), the data
//! description operators 201 (width), 202 (scale), 203 (reference
//! values), 204 (associated fields with 031021), 206 (skip a local
//! descriptor of stated width), 207 (scale, reference and width
//! together), 208 (character width), the no-op markers 221 to 225,
//! 232, 235 to 237 and 242 to 243 where the data carry no bitmap values
//! this reader needs, and both data layouts: the plain subset-by-subset
//! packing and the compressed one (a minimum plus per-subset increments
//! per element, the six-bit increment width, character strings with a
//! byte count).  Every element comes back as a typed value: a number
//! (reference and scale applied), a code or flag as its raw integer, a
//! string, or missing (all ones).
//!
//! What it refuses, by name: a descriptor these tables do not carry (a
//! centre's local descriptor not skipped by 206), the marker operators
//! 223255 to 225255 (they need the bitmap machinery a surface or
//! upper-air report never uses), an edition other than 3 or 4, a message
//! whose sections run past its stated length, and a subset whose bits run
//! past section 4.  A refusal names the descriptor and the subset so the
//! payload can be looked at; nothing is guessed.
//!
//! Fail-closed and bounded: a replication factor is bounded by the data,
//! a delayed count past 65,535 is refused, and the expansion depth is
//! bounded so a sequence that names itself cannot recurse.

use std::collections::HashMap;
use std::error::Error;
use std::sync::OnceLock;

use crate::err;

const ELEMENT_TABLE: &str = include_str!("../tables/bufr/element.table");
const SEQUENCE_DEF: &str = include_str!("../tables/bufr/sequence.def");
pub const MASTER_TABLE_VERSION: u8 = 46;
const MAX_EXPANSION_DEPTH: usize = 64;
const MAX_REPLICATION: u64 = 65_535;

/// One Table B entry.
#[derive(Debug, Clone)]
pub struct Element {
    pub code: u32,
    pub abbreviation: &'static str,
    pub kind: ElementKind,
    pub name: &'static str,
    pub unit: &'static str,
    pub scale: i32,
    pub reference: i64,
    pub width: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementKind {
    Number,
    Code,
    Flag,
    String,
}

pub struct Tables {
    elements: HashMap<u32, Element>,
    sequences: HashMap<u32, Vec<u32>>,
}

fn parse_descriptor(text: &str) -> Option<u32> {
    let t = text.trim();
    if t.len() != 6 || !t.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    t.parse().ok()
}

/// `fxxyyy` as a number: f (0 to 3), x (0 to 63), y (0 to 255).
pub fn fxy(code: u32) -> (u32, u32, u32) {
    (code / 100_000, (code / 1000) % 100, code % 1000)
}

pub fn descriptor_text(code: u32) -> String {
    format!("{:06}", code)
}

impl Tables {
    fn load() -> Tables {
        let mut elements = HashMap::new();
        for line in ELEMENT_TABLE.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('|').collect();
            if fields.len() < 8 {
                continue;
            }
            let Some(code) = parse_descriptor(fields[0]) else { continue };
            let kind = match fields[2] {
                "string" => ElementKind::String,
                "table" => ElementKind::Code,
                "flag" => ElementKind::Flag,
                _ => ElementKind::Number,
            };
            let (Ok(scale), Ok(reference), Ok(width)) = (
                fields[5].trim().parse::<i32>(),
                fields[6].trim().parse::<i64>(),
                fields[7].trim().parse::<u32>(),
            ) else {
                continue;
            };
            elements.insert(code, Element {
                code,
                abbreviation: fields[1],
                kind,
                name: fields[3],
                unit: fields[4],
                scale,
                reference,
                width,
            });
        }
        let mut sequences = HashMap::new();
        // `"307096" = [  301090, 301089, ... ]`, an entry may wrap lines.
        let mut current: Option<(u32, Vec<u32>)> = None;
        for line in SEQUENCE_DEF.lines() {
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            if text.starts_with('"') {
                if let Some((code, list)) = current.take() {
                    sequences.insert(code, list);
                }
                let close = text.find('"').and_then(|_| text[1..].find('"')).map(|i| i + 1);
                let Some(close) = close else { continue };
                let Some(code) = parse_descriptor(&text[1..close]) else { continue };
                let rest = text[close + 1..].trim_start_matches(|c: char| c == ' ' || c == '=' || c == '[');
                current = Some((code, Vec::new()));
                if let Some((_, list)) = current.as_mut() {
                    push_codes(rest, list);
                }
            } else if let Some((_, list)) = current.as_mut() {
                push_codes(text, list);
            }
            if text.ends_with(']') {
                if let Some((code, list)) = current.take() {
                    sequences.insert(code, list);
                }
            }
        }
        if let Some((code, list)) = current.take() {
            sequences.insert(code, list);
        }
        Tables { elements, sequences }
    }

    pub fn get() -> &'static Tables {
        static TABLES: OnceLock<Tables> = OnceLock::new();
        TABLES.get_or_init(Tables::load)
    }

    pub fn element(&self, code: u32) -> Option<&Element> {
        self.elements.get(&code)
    }

    pub fn sequence(&self, code: u32) -> Option<&Vec<u32>> {
        self.sequences.get(&code)
    }

    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }
}

fn push_codes(text: &str, list: &mut Vec<u32>) {
    for token in text.split(|c: char| c == ',' || c == ']' || c == '[' || c.is_whitespace()) {
        if let Some(code) = parse_descriptor(token) {
            list.push(code);
        }
    }
}

// ------------------------------------------------------------------ bits

struct BitReader<'a> {
    bytes: &'a [u8],
    /// The absolute bit position.
    pos: usize,
    /// One past the last bit this reader may touch.
    end: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8], start_bit: usize, end_bit: usize) -> Self {
        Self { bytes, pos: start_bit, end: end_bit }
    }

    fn read(&mut self, width: u32, what: &str) -> Result<u64, Box<dyn Error>> {
        if width == 0 {
            return Ok(0);
        }
        if width > 64 {
            return Err(err(format!("{what}: a {width}-bit field is wider than this reader's word")));
        }
        if self.pos + width as usize > self.end {
            return Err(err(format!(
                "{what}: the data run past the end of section 4 (bit {} + {width} > {})",
                self.pos, self.end
            )));
        }
        let mut value: u64 = 0;
        for _ in 0..width {
            let byte = self.bytes[self.pos / 8];
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            value = (value << 1) | u64::from(bit);
            self.pos += 1;
        }
        Ok(value)
    }

    fn read_bytes(&mut self, count: usize, what: &str) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.read(8, what)? as u8);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------- values

/// One decoded element of one subset.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    /// A code-table or flag-table entry, the raw integer.
    Code(u64),
    Text(String),
    Missing,
}

impl Value {
    pub fn number(&self) -> Option<f64> {
        match self {
            Value::Number(v) => Some(*v),
            Value::Code(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn code(&self) -> Option<u64> {
        match self {
            Value::Code(v) => Some(*v),
            Value::Number(v) if v.fract() == 0.0 && *v >= 0.0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn text(&self) -> Option<&str> {
        match self {
            Value::Text(t) => Some(t),
            _ => None,
        }
    }
}

/// One expanded descriptor with its value: the code, the value, and
/// which replication instance it sits in (the path of replication
/// indices from the outermost loop inward), so a template's level loop
/// can be walked without re-parsing the descriptor list.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub code: u32,
    pub value: Value,
    pub path: Vec<u32>,
}

/// The decoded message: the header of section 1 and the subsets.
#[derive(Debug, Clone)]
pub struct Message {
    pub edition: u8,
    pub master_table: u8,
    pub centre: u16,
    pub subcentre: u16,
    pub update_sequence: u8,
    pub data_category: u8,
    pub international_subcategory: Option<u8>,
    pub local_subcategory: u8,
    pub master_table_version: u8,
    pub local_table_version: u8,
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub observed: bool,
    pub compressed: bool,
    pub descriptors: Vec<u32>,
    pub subsets: Vec<Vec<Item>>,
    pub total_bytes: usize,
}

// ------------------------------------------------------------- expansion

/// The state the data description operators keep while a subset (or a
/// compressed message) is walked.
#[derive(Debug, Clone, Default)]
struct OperatorState {
    width_delta: i32,
    scale_delta: i32,
    /// 207yyy: scale + yyy, reference * 10^yyy, width + floor((10 yyy + 2) / 3).
    increase: Option<u32>,
    /// 203yyy: new reference values, keyed by descriptor.
    reference_overrides: HashMap<u32, i64>,
    /// 203yyy in progress: the width the new references are read with.
    reading_references: Option<u32>,
    /// 208yyy: character width in bytes.
    char_width: Option<u32>,
    /// 204yyy: the associated field width (bits) preceding every element
    /// except class 31.
    associated_width: u32,
}

struct Decoder<'a> {
    tables: &'static Tables,
    reader: BitReader<'a>,
    compressed: bool,
    subsets: usize,
    /// The plain layout fills one subset at a time; the compressed one
    /// fills every subset per element.
    values: Vec<Vec<Item>>,
    state: OperatorState,
    /// The 206yyy width waiting for its descriptor.
    skip_width: Option<u32>,
}

impl<'a> Decoder<'a> {
    fn element_of(&self, code: u32) -> Result<Element, Box<dyn Error>> {
        match self.tables.element(code) {
            Some(e) => Ok(e.clone()),
            None => Err(err(format!(
                "descriptor {} is not in the WMO master tables (version {MASTER_TABLE_VERSION}); a centre's local \
                 descriptor is decodable only where the message skips it with operator 206",
                descriptor_text(code)
            ))),
        }
    }

    /// The width, scale and reference of an element under the operators
    /// in force.
    fn effective(&self, element: &Element) -> (u32, i32, i64) {
        if element.kind == ElementKind::String {
            let width = self.state.char_width.map(|bytes| bytes * 8).unwrap_or(element.width);
            return (width, 0, 0);
        }
        let mut width = element.width;
        let mut scale = element.scale;
        let mut reference = element.reference;
        if let Some(&r) = self.state.reference_overrides.get(&element.code) {
            reference = r;
        }
        // Class 31 (replication factors) and the code/flag classes are
        // never touched by 201/202/207 (the regulation's exceptions).
        let (_, x, _) = fxy(element.code);
        let exempt = x == 31 || matches!(element.kind, ElementKind::Code | ElementKind::Flag);
        if !exempt {
            if let Some(yyy) = self.state.increase {
                // Regulation 94.6.3.6 (207YYY): scale + YYY, reference x 10^YYY,
                // width + floor((10 YYY + 2) / 3).
                scale += yyy as i32;
                reference = reference.saturating_mul(10_i64.saturating_pow(yyy));
                width += (10 * yyy + 2) / 3;
            } else {
                width = (width as i32 + self.state.width_delta).max(1) as u32;
                scale += self.state.scale_delta;
            }
        }
        (width, scale, reference)
    }

    fn read_one(&mut self, element: &Element, path: &[u32], what: &str) -> Result<(), Box<dyn Error>> {
        let (width, scale, reference) = self.effective(element);
        let (_, x, _) = fxy(element.code);
        let associated = if self.state.associated_width > 0 && x != 31 { self.state.associated_width } else { 0 };
        if self.compressed {
            self.read_compressed(element, width, scale, reference, associated, path, what)
        } else {
            let subset = self.values.len() - 1;
            if associated > 0 {
                let _ = self.reader.read(associated, what)?;
            }
            let value = self.read_value(element, width, scale, reference, what)?;
            self.values[subset].push(Item { code: element.code, value, path: path.to_vec() });
            Ok(())
        }
    }

    fn read_value(&mut self, element: &Element, width: u32, scale: i32, reference: i64, what: &str) -> Result<Value, Box<dyn Error>> {
        if element.kind == ElementKind::String {
            let bytes = self.reader.read_bytes((width / 8) as usize, what)?;
            return Ok(text_value(&bytes));
        }
        let raw = self.reader.read(width, what)?;
        Ok(scalar_value(element, raw, width, scale, reference))
    }

    #[allow(clippy::too_many_arguments)]
    fn read_compressed(&mut self, element: &Element, width: u32, scale: i32, reference: i64, associated: u32,
                       path: &[u32], what: &str) -> Result<(), Box<dyn Error>> {
        let n = self.subsets;
        if associated > 0 {
            // The associated field is compressed like a number of its width.
            let _min = self.reader.read(associated, what)?;
            let inc = self.reader.read(6, what)? as u32;
            for _ in 0..n {
                let _ = self.reader.read(inc, what)?;
            }
        }
        if element.kind == ElementKind::String {
            let base = self.reader.read_bytes((width / 8) as usize, what)?;
            let inc_bytes = self.reader.read(6, what)? as usize;
            for subset in 0..n {
                let value = if inc_bytes == 0 {
                    text_value(&base)
                } else {
                    let bytes = self.reader.read_bytes(inc_bytes, what)?;
                    text_value(&bytes)
                };
                self.values[subset].push(Item { code: element.code, value, path: path.to_vec() });
            }
            return Ok(());
        }
        let minimum = self.reader.read(width, what)?;
        let inc_width = self.reader.read(6, what)? as u32;
        let all_ones = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
        for subset in 0..n {
            let value = if inc_width == 0 {
                scalar_value(element, minimum, width, scale, reference)
            } else {
                let inc = self.reader.read(inc_width, what)?;
                let inc_missing = inc_width < 64 && inc == (1u64 << inc_width) - 1;
                if minimum == all_ones || inc_missing {
                    Value::Missing
                } else {
                    scalar_value(element, minimum + inc, width, scale, reference)
                }
            };
            self.values[subset].push(Item { code: element.code, value, path: path.to_vec() });
        }
        Ok(())
    }

    /// Read a replication factor (class 31) for the current subset (plain
    /// layout) or for every subset (compressed: it must be the same).
    fn read_factor(&mut self, code: u32, what: &str) -> Result<u32, Box<dyn Error>> {
        let element = self.element_of(code)?;
        let width = element.width;
        let (_, _, y) = fxy(code);
        let count = if self.compressed {
            let minimum = self.reader.read(width, what)?;
            let inc_width = self.reader.read(6, what)? as u32;
            if inc_width != 0 {
                // Every subset must share the factor; increments would say otherwise.
                let mut first: Option<u64> = None;
                for _ in 0..self.subsets {
                    let inc = self.reader.read(inc_width, what)?;
                    match first {
                        None => first = Some(inc),
                        Some(f) if f != inc => {
                            return Err(err(format!(
                                "{what}: the subsets disagree on a delayed replication factor ({}), which the \
                                 compressed layout forbids", descriptor_text(code)
                            )))
                        }
                        _ => {}
                    }
                }
                minimum + first.unwrap_or(0)
            } else {
                minimum
            }
        } else {
            self.reader.read(width, what)?
        };
        // 031011/031012 (repetition factors) mean the same count here.
        let _ = y;
        if count > MAX_REPLICATION {
            return Err(err(format!("{what}: a delayed replication factor of {count} is past the ceiling")));
        }
        let item = Item { code, value: Value::Number(count as f64), path: Vec::new() };
        if self.compressed {
            for subset in 0..self.subsets {
                self.values[subset].push(item.clone());
            }
        } else {
            let subset = self.values.len() - 1;
            self.values[subset].push(item);
        }
        Ok(count as u32)
    }

    fn walk(&mut self, descriptors: &[u32], path: &mut Vec<u32>, depth: usize, what: &str) -> Result<(), Box<dyn Error>> {
        if depth > MAX_EXPANSION_DEPTH {
            return Err(err(format!("{what}: the descriptor expansion is deeper than {MAX_EXPANSION_DEPTH}; a sequence names itself")));
        }
        let mut i = 0;
        while i < descriptors.len() {
            let code = descriptors[i];
            let (f, x, y) = fxy(code);
            match f {
                0 => {
                    if let Some(width) = self.skip_width.take() {
                        if self.tables.element(code).is_none() {
                            // A local descriptor of the stated width: skipped.
                            let what2 = format!("{what}: skipping local descriptor {}", descriptor_text(code));
                            if self.compressed {
                                let _ = self.reader.read(width, &what2)?;
                                let inc = self.reader.read(6, &what2)? as u32;
                                for _ in 0..self.subsets {
                                    let _ = self.reader.read(inc, &what2)?;
                                }
                            } else {
                                let _ = self.reader.read(width, &what2)?;
                            }
                            i += 1;
                            continue;
                        }
                    }
                    let element = self.element_of(code)?;
                    if let Some(width) = self.state.reading_references {
                        // 203yyy: this element's new reference value.
                        let raw = self.reader.read(width, what)?;
                        let reference = if raw >> (width - 1) & 1 == 1 {
                            -((raw & ((1u64 << (width - 1)) - 1)) as i64)
                        } else {
                            raw as i64
                        };
                        self.state.reference_overrides.insert(code, reference);
                        if self.compressed {
                            let inc = self.reader.read(6, what)? as u32;
                            for _ in 0..self.subsets {
                                let _ = self.reader.read(inc, what)?;
                            }
                        }
                        i += 1;
                        continue;
                    }
                    self.read_one(&element, path, what)?;
                    i += 1;
                }
                1 => {
                    // Replication: x descriptors, y times (0: delayed, the
                    // factor is the next descriptor, class 31).
                    let count = x as usize;
                    let mut next = i + 1;
                    let times = if y == 0 {
                        let factor = *descriptors.get(next).ok_or_else(|| err(format!("{what}: a delayed replication without its factor")))?;
                        let (ff, fx, _) = fxy(factor);
                        if ff != 0 || fx != 31 {
                            return Err(err(format!("{what}: delayed replication followed by {} instead of a class 31 factor", descriptor_text(factor))));
                        }
                        next += 1;
                        self.read_factor(factor, what)?
                    } else {
                        y
                    };
                    let body: Vec<u32> = descriptors.get(next..next + count)
                        .ok_or_else(|| err(format!("{what}: a replication of {count} descriptors runs past the template")))?
                        .to_vec();
                    for instance in 0..times {
                        path.push(instance);
                        self.walk(&body, path, depth + 1, what)?;
                        path.pop();
                    }
                    i = next + count;
                }
                2 => {
                    self.apply_operator(x, y, what)?;
                    i += 1;
                }
                3 => {
                    let body = self.tables.sequence(code).cloned().ok_or_else(|| err(format!(
                        "sequence descriptor {} is not in the WMO master tables (version {MASTER_TABLE_VERSION})",
                        descriptor_text(code)
                    )))?;
                    self.walk(&body, path, depth + 1, what)?;
                    i += 1;
                }
                _ => return Err(err(format!("{what}: descriptor {} has an F outside 0 to 3", descriptor_text(code)))),
            }
        }
        Ok(())
    }

    fn apply_operator(&mut self, x: u32, y: u32, what: &str) -> Result<(), Box<dyn Error>> {
        match x {
            1 => self.state.width_delta = if y == 0 { 0 } else { y as i32 - 128 },
            2 => self.state.scale_delta = if y == 0 { 0 } else { y as i32 - 128 },
            3 => {
                if y == 0 {
                    self.state.reference_overrides.clear();
                    self.state.reading_references = None;
                } else if y == 255 {
                    self.state.reading_references = None;
                } else {
                    self.state.reading_references = Some(y);
                }
            }
            4 => {
                self.state.associated_width = y;
            }
            5 => {
                // Character data of y bytes inline.
                let what2 = format!("{what}: operator 205{y:03}");
                if self.compressed {
                    let base = self.reader.read_bytes(y as usize, &what2)?;
                    let inc = self.reader.read(6, &what2)? as usize;
                    for subset in 0..self.subsets {
                        let value = if inc == 0 { text_value(&base) } else { text_value(&self.reader.read_bytes(inc, &what2)?) };
                        self.values[subset].push(Item { code: 205_000 + y, value, path: Vec::new() });
                    }
                } else {
                    let bytes = self.reader.read_bytes(y as usize, &what2)?;
                    let subset = self.values.len() - 1;
                    self.values[subset].push(Item { code: 205_000 + y, value: text_value(&bytes), path: Vec::new() });
                }
            }
            6 => self.skip_width = Some(y),
            7 => self.state.increase = if y == 0 { None } else { Some(y) },
            8 => self.state.char_width = if y == 0 { None } else { Some(y) },
            21 | 22 | 32 | 35 | 36 | 37 | 42 | 43 => {
                // Data-not-present / quality-information / bitmap markers: the
                // values that follow are read as ordinary elements (031031,
                // 001031, 001032, class 33), which the generic expansion does.
            }
            23 | 24 | 25 => {
                if y == 255 {
                    return Err(err(format!(
                        "{what}: operator 2{x:02}255 (a substituted, first-order-statistics or difference \
                         marker) needs the bitmap machinery this reader does not carry; the report's own \
                         values were read before it"
                    )));
                }
            }
            _ => {
                return Err(err(format!("{what}: data description operator 2{x:02}{y:03} is not one this reader carries")));
            }
        }
        Ok(())
    }
}

fn text_value(bytes: &[u8]) -> Value {
    if bytes.iter().all(|&b| b == 0xFF) {
        return Value::Missing;
    }
    let text: String = bytes.iter().map(|&b| if (32..127).contains(&b) { b as char } else { ' ' }).collect();
    let trimmed = text.trim().to_string();
    if trimmed.is_empty() {
        Value::Missing
    } else {
        Value::Text(trimmed)
    }
}

fn scalar_value(element: &Element, raw: u64, width: u32, scale: i32, reference: i64) -> Value {
    let all_ones = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
    if raw == all_ones && width > 1 {
        return Value::Missing;
    }
    match element.kind {
        ElementKind::Code | ElementKind::Flag => Value::Code(raw),
        ElementKind::String => Value::Missing,
        ElementKind::Number => {
            let value = (raw as f64 + reference as f64) / 10f64.powi(scale);
            Value::Number(value)
        }
    }
}

// ----------------------------------------------------------------- decode

type Header = (u8, u16, u16, u8, bool, u8, Option<u8>, u8, u8, u8, u16, u8, u8, u8, u8, u8);

fn be24(bytes: &[u8], at: usize) -> Option<usize> {
    bytes.get(at..at + 3).map(|b| ((b[0] as usize) << 16) | ((b[1] as usize) << 8) | b[2] as usize)
}

/// Decode one BUFR message (the bytes may carry a WMO GTS header before
/// `BUFR`; the first `BUFR` is taken, and the message ends at its stated
/// length).  `what` names the payload for the refusals.
pub fn decode(bytes: &[u8], what: &str) -> Result<Message, Box<dyn Error>> {
    decode_with_partial(bytes, what).map_err(|(e, _)| e)
}

/// :func:`decode`, handing back beside a refusal the items decoded before
/// it (the diagnostic route: which element the data ran out under).
pub fn decode_with_partial(bytes: &[u8], what: &str) -> Result<Message, (Box<dyn Error>, Vec<Vec<Item>>)> {
    let fail = |e: Box<dyn Error>| (e, Vec::new());
    let start = bytes.windows(4).position(|w| w == b"BUFR")
        .ok_or_else(|| err(format!("{what}: no BUFR indicator in {} bytes", bytes.len()))).map_err(fail)?;
    let b = &bytes[start..];
    if b.len() < 8 {
        return Err(fail(err(format!("{what}: {} bytes after BUFR is no message", b.len()))));
    }
    let total = be24(b, 4).unwrap();
    let edition = b[7];
    if total > b.len() {
        return Err(fail(err(format!("{what}: section 0 states {total} bytes and {} are here", b.len()))));
    }
    let b = &b[..total];
    let s1 = 8;
    let l1 = be24(b, s1).ok_or_else(|| err(format!("{what}: section 1 is cut"))).map_err(fail)?;
    let need = |at: usize, name: &str| -> Result<u8, Box<dyn Error>> {
        b.get(at).copied().ok_or_else(|| err(format!("{what}: section 1 lacks its {name}")))
    };
    let header = (|| -> Result<Header, Box<dyn Error>> { match edition {
        4 => {
            Ok((
                need(s1 + 3, "master table")?,
                u16::from_be_bytes([need(s1 + 4, "centre")?, need(s1 + 5, "centre")?]),
                u16::from_be_bytes([need(s1 + 6, "subcentre")?, need(s1 + 7, "subcentre")?]),
                need(s1 + 8, "update sequence")?,
                need(s1 + 9, "optional flag")? & 0x80 != 0,
                need(s1 + 10, "category")?,
                Some(need(s1 + 11, "international subcategory")?),
                need(s1 + 12, "local subcategory")?,
                need(s1 + 13, "master table version")?,
                need(s1 + 14, "local table version")?,
                u16::from_be_bytes([need(s1 + 15, "year")?, need(s1 + 16, "year")?]),
                need(s1 + 17, "month")?, need(s1 + 18, "day")?, need(s1 + 19, "hour")?, need(s1 + 20, "minute")?,
                need(s1 + 21, "second")?,
            ))
        }
        3 => {
            let yy = need(s1 + 12, "year")?;
            let year = if yy > 100 { 1900 + yy as u16 } else if yy == 100 { 2000 } else { 2000 + yy as u16 };
            Ok((
                need(s1 + 3, "master table")?,
                u16::from(need(s1 + 5, "centre")?),
                u16::from(need(s1 + 4, "subcentre")?),
                need(s1 + 6, "update sequence")?,
                need(s1 + 7, "optional flag")? & 0x80 != 0,
                need(s1 + 8, "category")?,
                None,
                need(s1 + 9, "local subcategory")?,
                need(s1 + 10, "master table version")?,
                need(s1 + 11, "local table version")?,
                year,
                need(s1 + 13, "month")?, need(s1 + 14, "day")?, need(s1 + 15, "hour")?, need(s1 + 16, "minute")?,
                0,
            ))
        }
        other => Err(err(format!("{what}: BUFR edition {other} is not one this reader carries (3 and 4)"))),
    } })().map_err(fail)?;
    let (master_table, centre, subcentre, update_sequence, optional, data_category, international_subcategory,
         local_subcategory, master_table_version, local_table_version, year, month, day, hour, minute, second) = header;
    let mut s3 = s1 + l1;
    if optional {
        let l2 = be24(b, s3).ok_or_else(|| err(format!("{what}: section 2 is cut"))).map_err(fail)?;
        s3 += l2;
    }
    let l3 = be24(b, s3).ok_or_else(|| err(format!("{what}: section 3 is cut"))).map_err(fail)?;
    if s3 + l3 > b.len() || l3 < 7 {
        return Err(fail(err(format!("{what}: section 3 runs past the message"))));
    }
    let subsets = u16::from_be_bytes([b[s3 + 4], b[s3 + 5]]) as usize;
    let flags = b[s3 + 6];
    let observed = flags & 0x80 != 0;
    let compressed = flags & 0x40 != 0;
    let mut descriptors = Vec::new();
    let mut at = s3 + 7;
    while at + 1 < s3 + l3 {
        let raw = u16::from_be_bytes([b[at], b[at + 1]]);
        if raw == 0 && at + 2 >= s3 + l3 {
            break;
        }
        let (f, x, y) = ((raw >> 14) as u32, ((raw >> 8) & 0x3F) as u32, (raw & 0xFF) as u32);
        descriptors.push(f * 100_000 + x * 1000 + y);
        at += 2;
    }
    let s4 = s3 + l3;
    let l4 = be24(b, s4).ok_or_else(|| err(format!("{what}: section 4 is cut"))).map_err(fail)?;
    if s4 + l4 > b.len() || l4 < 4 {
        return Err(fail(err(format!("{what}: section 4 runs past the message"))));
    }
    if subsets == 0 {
        return Err(fail(err(format!("{what}: section 3 states zero subsets"))));
    }
    let data_start_bit = (s4 + 4) * 8;
    let data_end_bit = (s4 + l4) * 8;
    let tables = Tables::get();
    let mut decoder = Decoder {
        tables,
        reader: BitReader::new(b, data_start_bit, data_end_bit),
        compressed,
        subsets,
        values: Vec::new(),
        state: OperatorState::default(),
        skip_width: None,
    };
    let mut path = Vec::new();
    if compressed {
        decoder.values = (0..subsets).map(|_| Vec::new()).collect();
        if let Err(e) = decoder.walk(&descriptors, &mut path, 0, &format!("{what} (compressed, {subsets} subsets)")) {
            return Err((e, decoder.values));
        }
    } else {
        for subset in 0..subsets {
            decoder.values.push(Vec::new());
            decoder.state = OperatorState::default();
            decoder.skip_width = None;
            if let Err(e) = decoder.walk(&descriptors, &mut path, 0, &format!("{what} subset {}", subset + 1)) {
                return Err((e, decoder.values));
            }
        }
    }
    Ok(Message {
        edition,
        master_table,
        centre,
        subcentre,
        update_sequence,
        data_category,
        international_subcategory,
        local_subcategory,
        master_table_version,
        local_table_version,
        year,
        month,
        day,
        hour,
        minute,
        second,
        observed,
        compressed,
        descriptors,
        subsets: decoder.values,
        total_bytes: total,
    })
}

/// Every BUFR message in a payload, in order (a file may carry several,
/// each with its own GTS header); a payload without one is an error.
pub fn decode_all(bytes: &[u8], what: &str) -> Result<Vec<Message>, Box<dyn Error>> {
    let mut messages = Vec::new();
    let mut at = 0;
    while let Some(rel) = bytes[at..].windows(4).position(|w| w == b"BUFR") {
        let start = at + rel;
        let message = decode(&bytes[start..], what)?;
        at = start + message.total_bytes.max(8);
        messages.push(message);
        if at >= bytes.len() {
            break;
        }
    }
    if messages.is_empty() {
        return Err(err(format!("{what}: no BUFR indicator in {} bytes", bytes.len())));
    }
    Ok(messages)
}

/// First value of `code` in a subset, in template order.
pub fn first(items: &[Item], code: u32) -> Option<&Value> {
    items.iter().find(|i| i.code == code).map(|i| &i.value)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A test-only encoder: the plain layout, one subset, from a
    /// descriptor list and the values in expansion order (numbers as
    /// f64, strings, missing), so the reader is held to a message whose
    /// bits are known.
    pub struct Encoder {
        bits: Vec<u8>,
        pos: usize,
    }

    impl Encoder {
        pub fn new() -> Self {
            Self { bits: Vec::new(), pos: 0 }
        }

        pub fn put(&mut self, value: u64, width: u32) {
            for i in (0..width).rev() {
                let bit = (value >> i) & 1;
                if self.pos % 8 == 0 {
                    self.bits.push(0);
                }
                if bit == 1 {
                    let last = self.bits.len() - 1;
                    self.bits[last] |= 1 << (7 - (self.pos % 8));
                }
                self.pos += 1;
            }
        }

        pub fn put_number(&mut self, element: &Element, value: f64) {
            let raw = (value * 10f64.powi(element.scale)).round() as i64 - element.reference;
            self.put(raw as u64, element.width);
        }

        pub fn put_missing(&mut self, width: u32) {
            self.put(if width >= 64 { u64::MAX } else { (1u64 << width) - 1 }, width);
        }

        pub fn put_text(&mut self, text: &str, bytes: usize) {
            let mut padded = text.as_bytes().to_vec();
            padded.resize(bytes, b' ');
            for b in padded {
                self.put(b as u64, 8);
            }
        }

        pub fn finish(mut self) -> Vec<u8> {
            while self.pos % 8 != 0 {
                self.put(0, 1);
            }
            self.bits
        }
    }

    pub fn message(descriptors: &[u32], data: &[u8], subsets: u16, compressed: bool) -> Vec<u8> {
        let mut s1 = vec![0u8; 22];
        s1[3] = 0;
        s1[4..6].copy_from_slice(&65535u16.to_be_bytes());
        s1[10] = 0;
        s1[11] = 2;
        s1[13] = 30;
        s1[15..17].copy_from_slice(&2026u16.to_be_bytes());
        s1[17] = 9;
        s1[18] = 6;
        s1[19] = 2;
        s1[20] = 0;
        let l1 = s1.len() as u32;
        s1[0..3].copy_from_slice(&l1.to_be_bytes()[1..]);
        let mut s3 = vec![0u8; 7];
        for &d in descriptors {
            let (f, x, y) = fxy(d);
            let raw = ((f as u16) << 14) | ((x as u16) << 8) | y as u16;
            s3.extend_from_slice(&raw.to_be_bytes());
        }
        if s3.len() % 2 == 1 {
            s3.push(0);
        }
        let l3 = s3.len() as u32;
        s3[0..3].copy_from_slice(&l3.to_be_bytes()[1..]);
        s3[4..6].copy_from_slice(&subsets.to_be_bytes());
        s3[6] = 0x80 | if compressed { 0x40 } else { 0 };
        let mut s4 = vec![0u8; 4];
        s4.extend_from_slice(data);
        if s4.len() % 2 == 1 {
            s4.push(0);
        }
        let l4 = s4.len() as u32;
        s4[0..3].copy_from_slice(&l4.to_be_bytes()[1..]);
        let total = 8 + s1.len() + s3.len() + s4.len() + 4;
        let mut out = b"BUFR".to_vec();
        out.extend_from_slice(&(total as u32).to_be_bytes()[1..]);
        out.push(4);
        out.extend(s1);
        out.extend(s3);
        out.extend(s4);
        out.extend_from_slice(b"7777");
        out
    }

    #[test]
    fn the_tables_load_with_the_templates_the_broker_carries() {
        let t = Tables::get();
        assert!(t.element_count() > 1500, "{}", t.element_count());
        assert!(t.sequence_count() > 600, "{}", t.sequence_count());
        for code in [307080, 307096, 309052, 308009, 301150, 301090, 303054, 302031] {
            assert!(t.sequence(code).is_some(), "sequence {code}");
        }
        assert_eq!(t.sequence(301150).unwrap(), &vec![1125, 1126, 1127, 1128]);
        // the wrapped 309052 entry reads whole
        let temp = t.sequence(309052).unwrap();
        assert_eq!(temp.first(), Some(&301111));
        assert_eq!(temp.last(), Some(&303051));
        assert_eq!(temp.len(), 11);
        let t2 = t.element(12101).unwrap();
        assert_eq!((t2.scale, t2.reference, t2.width), (2, 0, 16));
        assert_eq!(t.element(1128).unwrap().kind, ElementKind::String);
        assert_eq!(t.element(8042).unwrap().kind, ElementKind::Flag);
    }

    #[test]
    fn a_plain_message_round_trips_numbers_strings_missing_and_delayed_replication() {
        let t = Tables::get();
        // 301011 (year, month, day) then a delayed replication of two
        // elements: 102000 031001 012101 011002, then a string 001015.
        let descriptors = [301011, 102000, 31001, 12101, 11002, 1015];
        let mut e = Encoder::new();
        e.put_number(t.element(4001).unwrap(), 2026.0);
        e.put_number(t.element(4002).unwrap(), 9.0);
        e.put_number(t.element(4003).unwrap(), 6.0);
        e.put(2, 8); // 031001: two levels
        e.put_number(t.element(12101).unwrap(), 288.15);
        e.put_number(t.element(11002).unwrap(), 5.5);
        e.put_missing(16);
        e.put_number(t.element(11002).unwrap(), 12.0);
        e.put_text("STATION NAME", 20);
        let bytes = message(&descriptors, &e.finish(), 1, false);
        let m = decode(&bytes, "test").unwrap();
        assert_eq!(m.edition, 4);
        assert_eq!((m.year, m.month, m.day, m.hour), (2026, 9, 6, 2));
        assert!(!m.compressed && m.observed);
        let s = &m.subsets[0];
        assert_eq!(first(s, 4001), Some(&Value::Number(2026.0)));
        assert_eq!(first(s, 31001), Some(&Value::Number(2.0)));
        let temps: Vec<&Item> = s.iter().filter(|i| i.code == 12101).collect();
        assert_eq!(temps.len(), 2);
        assert_eq!(temps[0].value, Value::Number(288.15));
        assert_eq!(temps[0].path, vec![0]);
        assert_eq!(temps[1].value, Value::Missing);
        assert_eq!(temps[1].path, vec![1]);
        let winds: Vec<f64> = s.iter().filter(|i| i.code == 11002).map(|i| i.value.number().unwrap()).collect();
        assert_eq!(winds, vec![5.5, 12.0]);
        assert_eq!(first(s, 1015), Some(&Value::Text("STATION NAME".into())));
    }

    #[test]
    fn a_compressed_message_reads_every_subset_and_a_missing_increment() {
        let t = Tables::get();
        let descriptors = [12101, 11002, 1015];
        let mut e = Encoder::new();
        // 012101: min 280.00 K (raw 28000), 6-bit inc width 8, increments 0, 5, 255 (missing)
        e.put_number(t.element(12101).unwrap(), 280.0);
        e.put(8, 6);
        e.put(0, 8);
        e.put(5, 8);
        e.put(255, 8);
        // 011002: min 3.0 (raw 30), inc width 0 -> every subset 3.0
        e.put_number(t.element(11002).unwrap(), 3.0);
        e.put(0, 6);
        // 001015: base 20 bytes, then per-subset 4 bytes
        e.put_text("", 20);
        e.put(4, 6);
        e.put_text("AAAA", 4);
        e.put_text("BBBB", 4);
        e.put_text("CCCC", 4);
        let bytes = message(&descriptors, &e.finish(), 3, true);
        let m = decode(&bytes, "test").unwrap();
        assert!(m.compressed && m.subsets.len() == 3);
        let temps: Vec<Value> = m.subsets.iter().map(|s| first(s, 12101).unwrap().clone()).collect();
        assert_eq!(temps, vec![Value::Number(280.0), Value::Number(280.05), Value::Missing]);
        assert!(m.subsets.iter().all(|s| first(s, 11002) == Some(&Value::Number(3.0))));
        let names: Vec<String> = m.subsets.iter().map(|s| first(s, 1015).unwrap().text().unwrap().to_string()).collect();
        assert_eq!(names, vec!["AAAA", "BBBB", "CCCC"]);
    }

    #[test]
    fn operators_change_widths_and_references_and_local_descriptors_are_skipped_or_refused() {
        let t = Tables::get();
        // 201133 (width +5), 025065 (numeric, width 8 by table) read at 13 bits; 201000 resets;
        // 206008 skips a local descriptor 063001 of 8 bits; 203014 sets a new 14-bit reference for
        // 007030 then 203255 ends it; 007030 read with the new reference.
        let e25065 = t.element(25065).unwrap();
        let descriptors = [201133, 25065, 201000, 206008, 63001, 203014, 7030, 203255, 7030];
        let mut e = Encoder::new();
        let raw = (7.0f64 * 10f64.powi(e25065.scale)).round() as i64 - e25065.reference;
        e.put(raw as u64, e25065.width + 5);
        e.put(0xAB, 8); // the local descriptor's bits
        e.put(1000, 14); // the new reference value for 007030
        e.put(250, 17); // 007030 raw: value = (250 + 1000) / 10 = 125.0 m
        let bytes = message(&descriptors, &e.finish(), 1, false);
        let m = decode(&bytes, "test").unwrap();
        let s = &m.subsets[0];
        assert_eq!(first(s, 25065).unwrap().number(), Some(7.0));
        assert!(first(s, 63001).is_none());
        assert_eq!(first(s, 7030), Some(&Value::Number(125.0)));
        // an unskipped local descriptor is refused by name
        let bytes = message(&[63001], &Encoder::new().finish(), 1, false);
        let e = decode(&bytes, "local").unwrap_err().to_string();
        assert!(e.contains("063001") && e.contains("not in the WMO master tables"), "{e}");
        // a marker operator this reader does not carry is refused by name
        let bytes = message(&[223255], &Encoder::new().finish(), 1, false);
        assert!(decode(&bytes, "marker").unwrap_err().to_string().contains("223255"));
    }

    #[test]
    fn operator_207_widens_scales_and_rescales_the_reference_per_the_regulation() {
        // 303056's level: 207001 007004 010009 207000.  007004 (scale -1,
        // width 14) becomes scale 0, width 18: 101325 Pa exact; 010009 (ref
        // -1000, width 17) becomes ref -10000, scale 1, width 21: 1234.5 gpm.
        let descriptors = [207001, 7004, 10009, 207000, 7004];
        let mut e = Encoder::new();
        e.put(101325, 18);
        e.put((12345i64 + 10000) as u64, 21);
        e.put(1013, 14); // back to the table's 10 Pa resolution
        let m = decode(&message(&descriptors, &e.finish(), 1, false), "207").unwrap();
        let s = &m.subsets[0];
        let p: Vec<f64> = s.iter().filter(|i| i.code == 7004).map(|i| i.value.number().unwrap()).collect();
        assert_eq!(p, vec![101325.0, 10130.0]);
        assert_eq!(first(s, 10009), Some(&Value::Number(1234.5)));
    }

    #[test]
    fn a_message_whose_data_run_past_section_4_is_refused_and_a_gts_header_is_skipped() {
        let descriptors = [12101, 12101];
        let mut e = Encoder::new();
        e.put(1, 16);
        let bytes = message(&descriptors, &e.finish(), 1, false);
        let error = decode(&bytes, "short").unwrap_err().to_string();
        assert!(error.contains("past the end of section 4"), "{error}");
        let mut with_header = b"\x01\r\r\n123\r\r\nISMD01 EGRR 060000\r\r\n".to_vec();
        let mut e = Encoder::new();
        e.put_number(Tables::get().element(12101).unwrap(), 290.0);
        with_header.extend(message(&[12101], &e.finish(), 1, false));
        let m = decode_all(&with_header, "gts").unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(first(&m[0].subsets[0], 12101), Some(&Value::Number(290.0)));
    }
}
