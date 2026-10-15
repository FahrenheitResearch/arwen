//! Grading a static against a known answer.
//!
//! The instrument, kept separate from the thing it measures. It reports every
//! shared variable's worst absolute and relative gap and the count of exactly
//! equal values, and it reports fields present in only one file rather than
//! quietly skipping them -- a comparison that silently ignores what it cannot
//! align is how a builder gets graded on the fields it happened to get right.
//!
//! There is no verdict in here on purpose. Two statics built from DIFFERENT
//! geography archives must differ in the geography group and must not differ in
//! the mesh group, and only a caller who knows which archives went in can say
//! which of those a number means. This module measures; the receipt states the
//! provenance beside it.

use std::path::Path;

use serde::Serialize;

use crate::error::{MpasError, MpasResult};

/// One variable's agreement.
#[derive(Debug, Clone, Serialize)]
pub struct FieldComparison {
    pub name: String,
    pub n: usize,
    /// Values equal in their stored type, including NaN payloads and text bytes.
    pub exact: usize,
    pub stored_type: String,
    pub max_abs: f64,
    /// Worst |a-b| / max(|b|, tiny), with the reference magnitude beside it.
    pub max_rel: f64,
    pub max_rel_reference: f64,
    pub at_index: usize,
    pub mean_abs: f64,
    pub reference_min: f64,
    pub reference_max: f64,
    pub candidate_min: f64,
    pub candidate_max: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rw_store::netcdf_classic::{NcClassicWriter,NcDim,NcFormat,NcVarDef,NcType,NcData};

    #[test]
    fn static_comparison_observes_text_padding_nan_bits_and_dimension_names() {
        let root=std::env::temp_dir().join(format!("rw-mpas-static-compare-{}",std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let write=|path:&Path,word:u32,text:&[u8],dim:&str| {
            let mut writer=NcClassicWriter::create(path,NcFormat::Offset64,
                vec![NcDim::fixed(dim,1),NcDim::fixed("StrLen",4)],vec![],
                vec![NcVarDef::new("value",NcType::Float,vec![0]),NcVarDef::new("label",NcType::Char,vec![1])],0).unwrap();
            writer.put("value",NcData::Floats(&[f32::from_bits(word)])).unwrap();
            writer.put("label",NcData::Chars(text)).unwrap();
            writer.finish().unwrap();
        };
        let (a,b)=(root.join("reference.nc"),root.join("candidate.nc"));
        write(&a,0x7fc0_0001,b"A \0\0","nCells");
        write(&b,0x7fc0_0001,b"A \0\0","nCells");
        assert!(compare(&a,&b).unwrap().word_identical);
        write(&b,0x7fc0_0002,b"A  \0","nCells");
        let changed=compare(&a,&b).unwrap();
        assert!(!changed.word_identical);
        assert_eq!(changed.fields.iter().find(|f|f.name=="value").unwrap().exact,0);
        assert_eq!(changed.fields.iter().find(|f|f.name=="label").unwrap().exact,3);
        let witness=differing_word_witness(&a,&b).unwrap();
        assert_eq!(witness.fields.iter().find(|f|f.name=="value").unwrap().words[0].index,0);
        let word=&witness.fields.iter().find(|f|f.name=="value").unwrap().words[0];
        assert_eq!((word.reference,word.candidate),(0x7fc0_0001,0x7fc0_0002));
        assert_eq!(witness.fields.iter().find(|f|f.name=="label").unwrap().words[0].index,2);
        write(&b,0x7fc0_0001,b"A \0\0","nEdges");
        assert_eq!(compare(&a,&b).unwrap().shape_mismatch,["value"]);
        assert!(differing_word_witness(&a,&b).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn static_comparison_checks_all_variable_attributes_without_coercion() {
        use rw_store::netcdf_classic::NcAttr;
        let root = std::env::temp_dir().join(format!("rw-mpas-static-attributes-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let (a,b) = (root.join("native.nc"),root.join("hex.nc"));
        let write = |path: &Path, attrs: Vec<NcAttr>, label_attrs: Vec<NcAttr>| {
            let mut writer = NcClassicWriter::create(path, NcFormat::Offset64,
                vec![NcDim::fixed("nCells",1)], vec![], vec![
                    NcVarDef::new("albedo12m",NcType::Float,vec![0]).with_attrs(attrs),
                    NcVarDef::new("soilcomp",NcType::Int,vec![0]).with_attrs(label_attrs)],0).unwrap();
            writer.put("albedo12m",NcData::Floats(&[20.0])).unwrap();
            writer.put("soilcomp",NcData::Ints(&[100])).unwrap();
            writer.finish().unwrap();
        };
        let base = || vec![NcAttr::text("units","percent"),NcAttr::text("long_name","native description"),
            NcAttr::floats("other",vec![f32::from_bits(0x7fc0_0001),-0.0])];
        let label = || vec![NcAttr::text("units","percent")];
        write(&a,base(),label());
        let mut reversed=base(); reversed.reverse();
        write(&b,reversed,label());
        let same=compare(&a,&b).unwrap();
        assert!(same.word_identical && same.attributes_identical);
        assert_eq!(same.attributes_compared,4);
        for change in 0..7 {
            let mut attrs=base(); let mut second=label();
            match change {
                0 => attrs[0]=NcAttr::text("units","-"),
                1 => attrs[1]=NcAttr::text("long_name","wrong description"),
                2 => { attrs.remove(1); },
                3 => attrs.push(NcAttr::text("extra","unexpected")),
                4 => attrs[2]=NcAttr::floats("other",vec![f32::from_bits(0x7fc0_0002),-0.0]),
                5 => attrs[2]=NcAttr::floats("other",vec![f32::from_bits(0x7fc0_0001),0.0]),
                _ => second[0]=NcAttr::int("units",1),
            }
            write(&b,attrs,second);
            let changed=compare(&a,&b).unwrap();
            assert!(changed.word_identical, "attributes never alter stored-word verdict");
            assert!(!changed.attributes_identical,"mutation {change}");
            assert_eq!(changed.attribute_mismatch.len(),1);
            serde_json::to_string(&changed).unwrap();
        }
        std::fs::remove_dir_all(root).unwrap();
    }

}

/// Exact stored-word locations for native defect witnesses. This instrument
/// never excludes a word or changes the comparison's bitwise verdict.
#[derive(Debug, Serialize)]
pub struct WordDifference {
    pub index: usize,
    pub reference: u64,
    pub candidate: u64,
}

#[derive(Debug, Serialize)]
pub struct FieldWordWitness {
    pub name: String,
    pub dimensions: Vec<(String,usize)>,
    pub stored_type: String,
    pub words: Vec<WordDifference>,
}

#[derive(Debug, Serialize)]
pub struct StoredWordWitness {
    pub schema: &'static str,
    pub index_order: &'static str,
    pub reference_sha256: String,
    pub candidate_sha256: String,
    pub fields: Vec<FieldWordWitness>,
}

pub fn differing_word_witness(reference: &Path,candidate: &Path)->MpasResult<StoredWordWitness> {
    let open=|path:&Path|netcrust::File::open(path).map_err(|e|MpasError::Refusal(format!("{}: {e}",path.display())));
    let (a,b)=(open(reference)?,open(candidate)?);
    if dims(&a)!=dims(&b) || names(&a)!=names(&b) {
        return Err(MpasError::Refusal("exact word locations require matching field and dimension inventories".into()));
    }
    let mut fields=Vec::new();
    for name in names(&a) {
        let (ma,mb)=(a.variable(&name).unwrap(),b.variable(&name).unwrap());
        let dimensions=|v:&netcrust::Variable|v.dimensions().iter().map(|d|(d.name().to_owned(),d.len())).collect::<Vec<_>>();
        if ma.dtype()!=mb.dtype() || dimensions(&ma)!=dimensions(&mb) {
            return Err(MpasError::Refusal(format!("{name}: exact word locations require matching stored type and dimensions")));
        }
        let (va,vb)=(words(&a,&name)?,words(&b,&name)?);
        let words=va.iter().zip(&vb).enumerate().filter_map(|(index,(a,b))|
            (a.0!=b.0).then_some(WordDifference{index,reference:a.0,candidate:b.0})).collect::<Vec<_>>();
        if !words.is_empty() {
            fields.push(FieldWordWitness{name,dimensions:dimensions(&ma),stored_type:format!("{:?}",ma.dtype()),words});
        }
    }
    Ok(StoredWordWitness{schema:"mpas-static-stored-word-witness/v1",index_order:"zero-based flattened file dimension order",reference_sha256:crate::sha256_file(reference)?,candidate_sha256:crate::sha256_file(candidate)?,fields})
}

/// Attribute values retain their stored type; floats are represented by bits
/// so NaN payloads and signed zero survive both comparison and JSON receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "stored_type", content = "values")]
pub enum StoredAttribute {
    Bytes(Vec<i8>), Chars(String), Shorts(Vec<i16>), Ints(Vec<i32>),
    Floats(Vec<u32>), Doubles(Vec<u64>), UBytes(Vec<u8>), UShorts(Vec<u16>),
    UInts(Vec<u32>), Int64s(Vec<i64>), UInt64s(Vec<u64>), Strings(Vec<String>),
}

impl From<&netcrust::AttributeValue> for StoredAttribute {
    fn from(value: &netcrust::AttributeValue) -> Self {
        use netcrust::AttributeValue as A;
        match value {
            A::Bytes(v) => Self::Bytes(v.clone()), A::Chars(v) => Self::Chars(v.clone()),
            A::Shorts(v) => Self::Shorts(v.clone()), A::Ints(v) => Self::Ints(v.clone()),
            A::Floats(v) => Self::Floats(v.iter().map(|v| v.to_bits()).collect()),
            A::Doubles(v) => Self::Doubles(v.iter().map(|v| v.to_bits()).collect()),
            A::UBytes(v) => Self::UBytes(v.clone()), A::UShorts(v) => Self::UShorts(v.clone()),
            A::UInts(v) => Self::UInts(v.clone()), A::Int64s(v) => Self::Int64s(v.clone()),
            A::UInt64s(v) => Self::UInt64s(v.clone()), A::Strings(v) => Self::Strings(v.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AttributeDifference {
    pub variable: String,
    pub attribute: String,
    pub reference: Option<StoredAttribute>,
    pub candidate: Option<StoredAttribute>,
}

fn attributes(variable: &netcrust::Variable) -> std::collections::BTreeMap<String, StoredAttribute> {
    variable.attributes().iter().map(|a| (a.name().to_owned(), a.value().into())).collect()
}

/// The whole comparison.
#[derive(Debug, Clone, Serialize)]
pub struct StaticComparison {
    pub reference: String,
    pub candidate: String,
    pub reference_sha256: String,
    pub candidate_sha256: String,
    pub dimensions_agree: bool,
    pub reference_dimensions: Vec<(String, usize)>,
    pub candidate_dimensions: Vec<(String, usize)>,
    pub fields: Vec<FieldComparison>,
    pub only_in_reference: Vec<String>,
    pub only_in_candidate: Vec<String>,
    pub shape_mismatch: Vec<String>,
    pub type_mismatch: Vec<String>,
    pub word_identical: bool,
    /// Independent of scientific words and any declared word residue.
    pub attributes_identical: bool,
    pub attributes_compared: usize,
    pub attribute_mismatch: Vec<AttributeDifference>,
}

/// Read stored words. Numeric promotion is only for diagnostic magnitudes.
fn words(f: &netcrust::File, name: &str) -> MpasResult<Vec<(u64,f64)>> {
    let variable=f.variable(name).ok_or_else(||MpasError::Refusal(format!("missing {name}")))?;
    macro_rules! read {
        ($ty:ty,$bits:expr) => {{
            let values=f.read_array::<$ty>(name).map_err(|e|MpasError::Refusal(format!("{name}: {e}")))?;
            values.iter().map(|&v|(($bits)(v),v as f64)).collect()
        }};
    }
    use netcrust::DataType::*;
    Ok(match variable.dtype() {
        F32=>read!(f32,|v:f32|v.to_bits() as u64),
        F64=>read!(f64,|v:f64|v.to_bits()),
        I8=>read!(i8,|v:i8|v as u8 as u64), I16=>read!(i16,|v:i16|v as u16 as u64),
        I32=>read!(i32,|v:i32|v as u32 as u64), I64=>read!(i64,|v:i64|v as u64),
        U8=>read!(u8,|v:u8|v as u64), U16=>read!(u16,|v:u16|v as u64),
        U32=>read!(u32,|v:u32|v as u64), U64=>read!(u64,|v:u64|v),
        Char=>{
            let strings=f.read_strings(name).map_err(|e|MpasError::Refusal(format!("{name}: {e}")))?;
            let width=*variable.shape().last().unwrap_or(&1);
            let mut bytes=Vec::new();
            for string in strings {
                // netcrust trims trailing NUL bytes. Restore them using the
                // stored dimension; refuse lossy decoding of non-ASCII text.
                if !string.is_ascii() || string.len()>width {
                    return Err(MpasError::Refusal(format!("{name}: non-ASCII text cannot be graded after netcrust's lossy UTF-8 decode")));
                }
                bytes.extend(string.bytes().chain(std::iter::repeat(0)).take(width).map(|b|(b as u64,b as f64)));
            }
            bytes
        },
        dtype=>return Err(MpasError::Refusal(format!("{name}: unsupported stored type {dtype:?} in strict static comparison"))),
    })
}

fn dims(f: &netcrust::File) -> Vec<(String, usize)> {
    let mut v: Vec<(String, usize)> = f
        .dimensions()
        .unwrap_or_default()
        .into_iter()
        .map(|d| (d.name().to_string(), d.len()))
        .collect();
    v.sort();
    v
}

fn names(f: &netcrust::File) -> Vec<String> {
    let mut v: Vec<String> = f
        .variables()
        .unwrap_or_default()
        .into_iter()
        .map(|x| x.name().to_string())
        .collect();
    v.sort();
    v
}

/// Compare `candidate` against `reference`, variable by variable.
pub fn compare(reference: &Path, candidate: &Path) -> MpasResult<StaticComparison> {
    let a = netcrust::File::open(reference)
        .map_err(|e| MpasError::Refusal(format!("{}: {e}", reference.display())))?;
    let b = netcrust::File::open(candidate)
        .map_err(|e| MpasError::Refusal(format!("{}: {e}", candidate.display())))?;

    let (da, db) = (dims(&a), dims(&b));
    let (na, nb) = (names(&a), names(&b));

    let mut fields = Vec::new();
    let mut only_a = Vec::new();
    let mut only_b = Vec::new();
    let mut shape_mismatch = Vec::new();
    let mut type_mismatch = Vec::new();
    let mut attribute_mismatch = Vec::new();
    let mut attributes_compared = 0;

    for name in &na {
        if !nb.contains(name) {
            only_a.push(name.clone());
            continue;
        }
        let (ma,mb)=(a.variable(name).unwrap(),b.variable(name).unwrap());
        let (aa, ab) = (attributes(&ma), attributes(&mb));
        let keys: std::collections::BTreeSet<_> = aa.keys().chain(ab.keys()).collect();
        attributes_compared += keys.len();
        for key in keys {
            if aa.get(key) != ab.get(key) {
                attribute_mismatch.push(AttributeDifference {
                    variable: name.clone(), attribute: key.clone(),
                    reference: aa.get(key).cloned(), candidate: ab.get(key).cloned(),
                });
            }
        }
        if ma.dtype()!=mb.dtype() {type_mismatch.push(name.clone());continue;}
        let dimension_names=|v:&netcrust::Variable|v.dimensions().iter().map(|d|(d.name().to_owned(),d.len())).collect::<Vec<_>>();
        if dimension_names(&ma)!=dimension_names(&mb) {shape_mismatch.push(name.clone());continue;}
        let (va,vb)=(words(&a,name)?,words(&b,name)?);
        if va.len() != vb.len() {
            shape_mismatch.push(format!("{name} ({} vs {})", va.len(), vb.len()));
            continue;
        }
        let mut c = FieldComparison {
            name: name.clone(),
            n: va.len(),
            exact: 0,
            stored_type: format!("{:?}",ma.dtype()),
            max_abs: 0.0,
            max_rel: 0.0,
            max_rel_reference: 0.0,
            at_index: 0,
            mean_abs: 0.0,
            reference_min: f64::INFINITY,
            reference_max: f64::NEG_INFINITY,
            candidate_min: f64::INFINITY,
            candidate_max: f64::NEG_INFINITY,
        };
        let mut total = 0.0f64;
        for i in 0..va.len() {
            let ((wx,x),(wy,y))=(va[i],vb[i]);
            c.reference_min = c.reference_min.min(x);
            c.reference_max = c.reference_max.max(x);
            c.candidate_min = c.candidate_min.min(y);
            c.candidate_max = c.candidate_max.max(y);
            if wx == wy {
                c.exact += 1;
                continue;
            }
            let d = (x - y).abs();
            total += d;
            if d > c.max_abs {
                c.max_abs = d;
                c.at_index = i;
            }
            let scale = x.abs().max(1e-30);
            let r = d / scale;
            if r > c.max_rel {
                c.max_rel = r;
                c.max_rel_reference = x;
            }
        }
        c.mean_abs = if va.is_empty() { 0.0 } else { total / va.len() as f64 };
        fields.push(c);
    }
    for name in &nb {
        if !na.contains(name) {
            only_b.push(name.clone());
        }
    }

    let word_identical=da==db && only_a.is_empty() && only_b.is_empty()
        && shape_mismatch.is_empty() && type_mismatch.is_empty()
        && fields.iter().all(|f|f.exact==f.n);
    let attributes_identical = only_a.is_empty() && only_b.is_empty() && attribute_mismatch.is_empty();
    Ok(StaticComparison {
        reference: reference.display().to_string(),
        candidate: candidate.display().to_string(),
        reference_sha256: crate::sha256_file(reference)?,
        candidate_sha256: crate::sha256_file(candidate)?,
        dimensions_agree: da == db,
        reference_dimensions: da,
        candidate_dimensions: db,
        fields,
        only_in_reference: only_a,
        only_in_candidate: only_b,
        shape_mismatch,
        type_mismatch,
        word_identical,
        attributes_identical,
        attributes_compared,
        attribute_mismatch,
    })
}
