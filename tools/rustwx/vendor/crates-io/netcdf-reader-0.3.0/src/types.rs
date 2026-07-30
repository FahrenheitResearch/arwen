/// A NetCDF dimension.
#[derive(Debug, Clone)]
pub struct NcDimension {
    pub name: String,
    pub size: u64,
    pub is_unlimited: bool,
}

/// A field within a compound (struct) type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcCompoundField {
    pub name: String,
    pub offset: u64,
    pub dtype: NcType,
}

/// NetCDF data types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NcType {
    /// NC_BYTE (i8)
    Byte,
    /// NC_CHAR (u8/char)
    Char,
    /// NC_SHORT (i16)
    Short,
    /// NC_INT (i32)
    Int,
    /// NC_FLOAT (f32)
    Float,
    /// NC_DOUBLE (f64)
    Double,
    /// NC_UBYTE (u8, CDF-5)
    UByte,
    /// NC_USHORT (u16, CDF-5)
    UShort,
    /// NC_UINT (u32, CDF-5)
    UInt,
    /// NC_INT64 (i64, CDF-5)
    Int64,
    /// NC_UINT64 (u64, CDF-5)
    UInt64,
    /// NetCDF-4 only (variable-length string)
    String,
    /// NetCDF-4 compound type (struct with named fields).
    Compound {
        size: u32,
        fields: Vec<NcCompoundField>,
    },
    /// NetCDF-4 opaque type (uninterpreted byte blob).
    Opaque { size: u32, tag: String },
    /// NetCDF-4 array type (fixed-size array of a base type).
    Array { base: Box<NcType>, dims: Vec<u64> },
    /// NetCDF-4 variable-length type.
    VLen { base: Box<NcType> },
}

impl NcType {
    /// Size of a single element in bytes.
    pub fn size(&self) -> usize {
        match self {
            NcType::Byte | NcType::Char | NcType::UByte => 1,
            NcType::Short | NcType::UShort => 2,
            NcType::Int | NcType::UInt | NcType::Float => 4,
            NcType::Int64 | NcType::UInt64 | NcType::Double => 8,
            // Variable-length string; no fixed element size, but pointer-sized in memory.
            NcType::String => std::mem::size_of::<usize>(),
            NcType::Compound { size, .. } => *size as usize,
            NcType::Opaque { size, .. } => *size as usize,
            NcType::Array { base, dims } => {
                base.size() * dims.iter().map(|&d| d as usize).product::<usize>()
            }
            NcType::VLen { .. } => std::mem::size_of::<usize>(), // pointer-sized
        }
    }

    /// The numeric type code used in CDF-1/2/5 headers.
    pub fn classic_type_code(&self) -> Option<u32> {
        match self {
            NcType::Byte => Some(1),
            NcType::Char => Some(2),
            NcType::Short => Some(3),
            NcType::Int => Some(4),
            NcType::Float => Some(5),
            NcType::Double => Some(6),
            NcType::UByte => Some(7),
            NcType::UShort => Some(8),
            NcType::UInt => Some(9),
            NcType::Int64 => Some(10),
            NcType::UInt64 => Some(11),
            // Extended types are not valid in classic format.
            NcType::String
            | NcType::Compound { .. }
            | NcType::Opaque { .. }
            | NcType::Array { .. }
            | NcType::VLen { .. } => None,
        }
    }

    /// Returns true if this is a primitive numeric or string type.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            NcType::Byte
                | NcType::Char
                | NcType::Short
                | NcType::Int
                | NcType::Float
                | NcType::Double
                | NcType::UByte
                | NcType::UShort
                | NcType::UInt
                | NcType::Int64
                | NcType::UInt64
                | NcType::String
        )
    }
}

/// A NetCDF attribute value.
#[derive(Debug, Clone)]
pub enum NcAttrValue {
    Bytes(Vec<i8>),
    Chars(String),
    Shorts(Vec<i16>),
    Ints(Vec<i32>),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    UBytes(Vec<u8>),
    UShorts(Vec<u16>),
    UInts(Vec<u32>),
    Int64s(Vec<i64>),
    UInt64s(Vec<u64>),
    Strings(Vec<String>),
}

impl NcAttrValue {
    /// Get the value as a string (for Chars or single-element Strings).
    pub fn as_string(&self) -> Option<String> {
        match self {
            NcAttrValue::Chars(s) => Some(s.clone()),
            NcAttrValue::Strings(v) if v.len() == 1 => Some(v[0].clone()),
            _ => None,
        }
    }

    /// Get the value as f64 (with numeric promotion from the first element).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            NcAttrValue::Bytes(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Shorts(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Ints(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Floats(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Doubles(v) => v.first().copied(),
            NcAttrValue::UBytes(v) => v.first().map(|&x| x as f64),
            NcAttrValue::UShorts(v) => v.first().map(|&x| x as f64),
            NcAttrValue::UInts(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Int64s(v) => v.first().map(|&x| x as f64),
            NcAttrValue::UInt64s(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Chars(_) | NcAttrValue::Strings(_) => None,
        }
    }

    /// Get the value as a vector of f64 (with numeric promotion).
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            NcAttrValue::Bytes(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Shorts(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Ints(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Floats(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Doubles(v) => Some(v.clone()),
            NcAttrValue::UBytes(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::UShorts(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::UInts(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Int64s(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::UInt64s(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Chars(_) | NcAttrValue::Strings(_) => None,
        }
    }
}

/// A NetCDF attribute.
#[derive(Debug, Clone)]
pub struct NcAttribute {
    pub name: String,
    pub value: NcAttrValue,
}

/// A NetCDF variable (metadata only -- data is read on demand).
#[derive(Debug, Clone)]
pub struct NcVariable {
    pub name: String,
    pub dimensions: Vec<NcDimension>,
    pub dtype: NcType,
    pub attributes: Vec<NcAttribute>,
    /// For classic: file byte offset to the start of this variable's data.
    /// For nc4: HDF5 dataset object header address.
    pub(crate) data_offset: u64,
    /// Total data size in bytes (for non-record variables).
    pub(crate) _data_size: u64,
    /// Whether this variable uses the unlimited (record) dimension.
    pub(crate) is_record_var: bool,
    /// Size of one record slice in bytes (only meaningful for record variables).
    pub(crate) record_size: u64,
}

impl NcVariable {
    /// Variable name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Variable dimensions.
    pub fn dimensions(&self) -> &[NcDimension] {
        &self.dimensions
    }

    /// Variable data type.
    pub fn dtype(&self) -> &NcType {
        &self.dtype
    }

    /// Shape of the variable as a vector of dimension sizes.
    pub fn shape(&self) -> Vec<u64> {
        self.dimensions.iter().map(|d| d.size).collect()
    }

    /// Variable attributes.
    pub fn attributes(&self) -> &[NcAttribute] {
        &self.attributes
    }

    /// Find an attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&NcAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dimensions.len()
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> u64 {
        if self.dimensions.is_empty() {
            return 1; // scalar
        }
        self.dimensions.iter().map(|d| d.size).product()
    }

    pub(crate) fn checked_num_elements(&self) -> crate::Result<u64> {
        checked_shape_elements(&self.shape(), "variable element count")
    }
}

/// A NetCDF group (NetCDF-4 only; classic files have one implicit root group).
#[derive(Debug, Clone)]
pub struct NcGroup {
    pub name: String,
    pub dimensions: Vec<NcDimension>,
    pub variables: Vec<NcVariable>,
    pub attributes: Vec<NcAttribute>,
    pub groups: Vec<NcGroup>,
}

impl NcGroup {
    /// Find a variable by name in this group.
    pub fn variable(&self, name: &str) -> Option<&NcVariable> {
        let (group_path, variable_name) = split_parent_path(name)?;
        let group = self.group(group_path)?;
        group.variables.iter().find(|v| v.name == variable_name)
    }

    /// Find a dimension by name in this group.
    pub fn dimension(&self, name: &str) -> Option<&NcDimension> {
        let (group_path, dimension_name) = split_parent_path(name)?;
        let group = self.group(group_path)?;
        group.dimensions.iter().find(|d| d.name == dimension_name)
    }

    /// Find an attribute by name in this group.
    pub fn attribute(&self, name: &str) -> Option<&NcAttribute> {
        let (group_path, attribute_name) = split_parent_path(name)?;
        let group = self.group(group_path)?;
        group.attributes.iter().find(|a| a.name == attribute_name)
    }

    /// Find a child group by relative path.
    pub fn group(&self, name: &str) -> Option<&NcGroup> {
        let trimmed = name.trim_matches('/');
        if trimmed.is_empty() {
            return Some(self);
        }

        let mut group = self;
        for component in trimmed.split('/').filter(|part| !part.is_empty()) {
            group = group.groups.iter().find(|child| child.name == component)?;
        }

        Some(group)
    }
}

fn split_parent_path(path: &str) -> Option<(&str, &str)> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return None;
    }

    match trimmed.rsplit_once('/') {
        Some((group_path, leaf_name)) if !leaf_name.is_empty() => Some((group_path, leaf_name)),
        Some(_) => None,
        None => Some(("", trimmed)),
    }
}

pub(crate) fn checked_usize_from_u64(value: u64, context: &str) -> crate::Result<usize> {
    usize::try_from(value)
        .map_err(|_| crate::Error::InvalidData(format!("{context} exceeds platform usize")))
}

pub(crate) fn checked_mul_u64(lhs: u64, rhs: u64, context: &str) -> crate::Result<u64> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| crate::Error::InvalidData(format!("{context} exceeds u64 capacity")))
}

pub(crate) fn checked_shape_elements(shape: &[u64], context: &str) -> crate::Result<u64> {
    shape
        .iter()
        .try_fold(1u64, |acc, &dim| checked_mul_u64(acc, dim, context))
}

/// Hyperslab selection for reading slices of NetCDF variables.
///
/// Each element corresponds to one dimension of the variable.
#[derive(Debug, Clone)]
pub struct NcSliceInfo {
    pub selections: Vec<NcSliceInfoElem>,
}

/// A single dimension's selection within a hyperslab.
#[derive(Debug, Clone)]
pub enum NcSliceInfoElem {
    /// Select a single index (reduces dimensionality).
    Index(u64),
    /// Select a range with stride.
    Slice { start: u64, end: u64, step: u64 },
}

impl NcSliceInfo {
    /// Create a selection that reads everything for an `ndim`-dimensional variable.
    pub fn all(ndim: usize) -> Self {
        NcSliceInfo {
            selections: vec![
                NcSliceInfoElem::Slice {
                    start: 0,
                    end: u64::MAX,
                    step: 1,
                };
                ndim
            ],
        }
    }
}

#[cfg(feature = "netcdf4")]
impl NcSliceInfo {
    /// Convert to hdf5_reader::SliceInfo for NC4 delegation.
    pub(crate) fn to_hdf5_slice_info(&self) -> hdf5_reader::SliceInfo {
        hdf5_reader::SliceInfo {
            selections: self
                .selections
                .iter()
                .map(|s| match s {
                    NcSliceInfoElem::Index(idx) => hdf5_reader::SliceInfoElem::Index(*idx),
                    NcSliceInfoElem::Slice { start, end, step } => {
                        hdf5_reader::SliceInfoElem::Slice {
                            start: *start,
                            end: *end,
                            step: *step,
                        }
                    }
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_group_tree() -> NcGroup {
        NcGroup {
            name: "/".to_string(),
            dimensions: vec![NcDimension {
                name: "root_dim".to_string(),
                size: 2,
                is_unlimited: false,
            }],
            variables: vec![NcVariable {
                name: "root_var".to_string(),
                dimensions: vec![],
                dtype: NcType::Int,
                attributes: vec![],
                data_offset: 0,
                _data_size: 0,
                is_record_var: false,
                record_size: 4,
            }],
            attributes: vec![NcAttribute {
                name: "title".to_string(),
                value: NcAttrValue::Chars("root".to_string()),
            }],
            groups: vec![NcGroup {
                name: "obs".to_string(),
                dimensions: vec![NcDimension {
                    name: "time".to_string(),
                    size: 3,
                    is_unlimited: false,
                }],
                variables: vec![NcVariable {
                    name: "temperature".to_string(),
                    dimensions: vec![],
                    dtype: NcType::Float,
                    attributes: vec![],
                    data_offset: 0,
                    _data_size: 0,
                    is_record_var: false,
                    record_size: 4,
                }],
                attributes: vec![],
                groups: vec![NcGroup {
                    name: "surface".to_string(),
                    dimensions: vec![],
                    variables: vec![NcVariable {
                        name: "pressure".to_string(),
                        dimensions: vec![],
                        dtype: NcType::Double,
                        attributes: vec![],
                        data_offset: 0,
                        _data_size: 0,
                        is_record_var: false,
                        record_size: 8,
                    }],
                    attributes: vec![NcAttribute {
                        name: "units".to_string(),
                        value: NcAttrValue::Chars("hPa".to_string()),
                    }],
                    groups: vec![],
                }],
            }],
        }
    }

    #[test]
    fn test_group_path_lookup() {
        let root = sample_group_tree();

        let surface = root.group("obs/surface").unwrap();
        assert_eq!(surface.name, "surface");
        assert!(root.group("/obs/surface").is_some());
        assert!(root.group("missing").is_none());
    }

    #[test]
    fn test_variable_path_lookup() {
        let root = sample_group_tree();

        assert_eq!(root.variable("root_var").unwrap().name(), "root_var");
        assert_eq!(
            root.variable("obs/temperature").unwrap().dtype(),
            &NcType::Float
        );
        assert_eq!(
            root.variable("/obs/surface/pressure").unwrap().dtype(),
            &NcType::Double
        );
        assert!(root.variable("pressure").is_none());
    }

    #[test]
    fn test_dimension_and_attribute_path_lookup() {
        let root = sample_group_tree();

        assert_eq!(root.dimension("root_dim").unwrap().size, 2);
        assert_eq!(root.dimension("obs/time").unwrap().size, 3);
        assert_eq!(
            root.attribute("title").unwrap().value.as_string().unwrap(),
            "root"
        );
        assert_eq!(
            root.attribute("obs/surface/units")
                .unwrap()
                .value
                .as_string()
                .unwrap(),
            "hPa"
        );
    }

    #[test]
    fn test_checked_shape_elements_overflow() {
        let err = checked_shape_elements(&[u64::MAX, 2], "test overflow").unwrap_err();
        assert!(matches!(err, crate::Error::InvalidData(_)));
    }
}
