//! Writing netCDF-4 files: the netCDF data model (groups, dimensions,
//! variables, attributes) stored the way netCDF-C stores it in HDF5
//! ("NetCDF-4 Format", netCDF-C documentation; `libhdf5/nc4hdf.c`), so
//! netCDF-C, netCDF4-python, xarray (either engine) and this crate's
//! [`crate::netcdf4::NcFile`] read it back:
//!
//! - every dimension is an HDF5 dimension scale in the group that defines
//!   it: the coordinate variable's dataset when the group has a variable of
//!   the same name over just that dimension, else a dataset holding no data
//!   whose `NAME` is `This is a netCDF dimension but not a netCDF variable.`
//!   and the length; each carries `CLASS = "DIMENSION_SCALE"`, `NAME`, a
//!   file-wide unique `_Netcdf4Dimid` and a `REFERENCE_LIST` of the axes
//!   attached to it;
//! - every other variable with dimensions has a `DIMENSION_LIST` (one
//!   object reference per axis); a variable named like a dimension it is
//!   not the coordinate of is stored as `_nc4_non_coord_<name>`;
//! - `char` attributes are scalar NUL-terminated fixed-length strings as
//!   long as the text (empty text: a null dataspace), `string` attributes
//!   and variables are variable-length strings, numeric attributes are
//!   one-dimensional;
//! - a variable's `_FillValue` attribute is also its dataset's fill value;
//! - the root carries `_NCProperties` naming this writer.
//!
//! Dimensions are fixed-size (no unlimited dimension).

use std::collections::HashMap;

use super::{Data, Layout, NewDataset, ObjectId, Shape, Value, WriteError, Writer};

/// `NAME` prefix of a dimension scale that is only a dimension.
const DIM_WITHOUT_VARIABLE: &str = "This is a netCDF dimension but not a netCDF variable.";
/// Dataset name prefix of a variable named like a dimension it does not
/// coordinate.
const NON_COORD_PREFIX: &str = "_nc4_non_coord_";
/// Attributes netCDF-C reserves; content attributes of these names are
/// refused.
const RESERVED: &[&str] = &[
    "CLASS",
    "NAME",
    "REFERENCE_LIST",
    "DIMENSION_LIST",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "_nc3_strict",
    "_NCProperties",
];

/// A group of an [`NcWriter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupId(usize);

/// A netCDF attribute value.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum NcAttr {
    /// `char` text.
    Text(String),
    /// `string` values (variable-length strings).
    Strings(Vec<String>),
    /// Numbers of one netCDF type (`Data::I8` .. `Data::F64`), one or more.
    Numbers(Data),
}

/// How a variable's data is stored.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NcStorage {
    /// Contiguous.
    Contiguous,
    /// Chunked, optionally shuffled and deflated.
    Chunked {
        /// Chunk extent per dimension.
        chunk: Vec<u64>,
        /// Byte shuffle before deflate.
        shuffle: bool,
        /// zlib level.
        deflate: Option<u32>,
    },
}

/// A variable to add.
#[derive(Clone, Debug, PartialEq)]
pub struct NcVariable {
    /// Variable name.
    pub name: String,
    /// Dimension names, outermost first; each defined in the variable's
    /// group or an ancestor. Empty for a scalar.
    pub dims: Vec<String>,
    /// Values, row-major (`Data::VarStrings` for a `string` variable).
    pub data: Data,
    /// Attributes in order.
    pub attrs: Vec<(String, NcAttr)>,
    /// Storage.
    pub storage: NcStorage,
}

struct Dim {
    name: String,
    len: u64,
}

struct Group {
    name: String,
    parent: Option<GroupId>,
    dims: Vec<Dim>,
    variables: Vec<NcVariable>,
    attrs: Vec<(String, NcAttr)>,
    children: Vec<GroupId>,
}

/// Builds a netCDF-4 file. See the [module documentation](self).
pub struct NcWriter {
    groups: Vec<Group>,
    properties: String,
}

impl Default for NcWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// netCDF-C's name rule (`NC_check_name`): not empty, no `/`, no control
/// characters, no trailing space, and a first character that is a letter,
/// a digit, `_` or non-ASCII.
pub fn is_valid_name(name: &str) -> bool {
    let Some(first) = name.chars().next() else {
        return false;
    };
    (first.is_ascii_alphanumeric() || first == '_' || !first.is_ascii())
        && !name.contains('/')
        && !name.chars().any(|c| c.is_control())
        && !name.ends_with(' ')
}

fn check_name(name: &str) -> Result<(), WriteError> {
    if is_valid_name(name) {
        Ok(())
    } else {
        Err(WriteError::InvalidName(name.to_owned()))
    }
}

fn check_attr(name: &str, attrs: &[(String, NcAttr)]) -> Result<(), WriteError> {
    if !is_valid_name(name) {
        return Err(WriteError::InvalidAttributeName(name.to_owned()));
    }
    if RESERVED.contains(&name) {
        return Err(WriteError::InvalidAttributeName(format!(
            "{name} (reserved by netCDF-4)"
        )));
    }
    if attrs.iter().any(|(existing, _)| existing == name) {
        return Err(WriteError::Duplicate(name.to_owned()));
    }
    Ok(())
}

fn attr_value(attr: &NcAttr) -> Result<Value, WriteError> {
    Ok(match attr {
        NcAttr::Text(text) => Value::text(text),
        NcAttr::Strings(values) => Value::var_texts(values.clone()),
        NcAttr::Numbers(data) => {
            if !matches!(
                data,
                Data::I8(_)
                    | Data::U8(_)
                    | Data::I16(_)
                    | Data::U16(_)
                    | Data::I32(_)
                    | Data::U32(_)
                    | Data::I64(_)
                    | Data::U64(_)
                    | Data::F32(_)
                    | Data::F64(_)
            ) {
                return Err(WriteError::Invalid(
                    "numeric attribute holds non-numeric data".to_owned(),
                ));
            }
            if data.is_empty() {
                return Err(WriteError::Invalid(
                    "numeric attribute without values".to_owned(),
                ));
            }
            Value::vector(data.clone())
        }
    })
}

impl NcWriter {
    /// A writer with an empty root group.
    pub fn new() -> Self {
        Self {
            groups: vec![Group {
                name: String::new(),
                parent: None,
                dims: Vec::new(),
                variables: Vec::new(),
                attrs: Vec::new(),
                children: Vec::new(),
            }],
            properties: format!("version=2,recast-radar-hdf5={}", env!("CARGO_PKG_VERSION")),
        }
    }

    /// The root group.
    pub fn root(&self) -> GroupId {
        GroupId(0)
    }

    fn group(&self, id: GroupId) -> Result<&Group, WriteError> {
        self.groups.get(id.0).ok_or(WriteError::UnknownObject(id.0))
    }

    fn group_mut(&mut self, id: GroupId) -> Result<&mut Group, WriteError> {
        self.groups
            .get_mut(id.0)
            .ok_or(WriteError::UnknownObject(id.0))
    }

    /// `true` when `name` is taken in group `id` by a child group or a
    /// variable.
    fn taken(&self, id: GroupId, name: &str) -> Result<bool, WriteError> {
        let group = self.group(id)?;
        Ok(group.variables.iter().any(|var| var.name == name)
            || group
                .children
                .iter()
                .any(|child| self.groups.get(child.0).is_some_and(|c| c.name == name)))
    }

    /// Add a child group.
    pub fn add_group(&mut self, parent: GroupId, name: &str) -> Result<GroupId, WriteError> {
        check_name(name)?;
        if self.taken(parent, name)? {
            return Err(WriteError::Duplicate(name.to_owned()));
        }
        let id = GroupId(self.groups.len());
        self.groups.push(Group {
            name: name.to_owned(),
            parent: Some(parent),
            dims: Vec::new(),
            variables: Vec::new(),
            attrs: Vec::new(),
            children: Vec::new(),
        });
        self.group_mut(parent)?.children.push(id);
        Ok(id)
    }

    /// Define a dimension in a group.
    pub fn add_dim(&mut self, group: GroupId, name: &str, len: u64) -> Result<(), WriteError> {
        check_name(name)?;
        let group = self.group_mut(group)?;
        if group.dims.iter().any(|dim| dim.name == name) {
            return Err(WriteError::Duplicate(name.to_owned()));
        }
        group.dims.push(Dim {
            name: name.to_owned(),
            len,
        });
        Ok(())
    }

    /// Add a group attribute.
    pub fn add_attr(
        &mut self,
        group: GroupId,
        name: &str,
        value: NcAttr,
    ) -> Result<(), WriteError> {
        let group = self.group_mut(group)?;
        check_attr(name, &group.attrs)?;
        attr_value(&value)?;
        group.attrs.push((name.to_owned(), value));
        Ok(())
    }

    /// The length of dimension `name` as seen from `group`.
    fn dim_len(&self, group: GroupId, name: &str) -> Option<u64> {
        let mut at = Some(group);
        while let Some(id) = at {
            let group = self.groups.get(id.0)?;
            if let Some(dim) = group.dims.iter().find(|dim| dim.name == name) {
                return Some(dim.len);
            }
            at = group.parent;
        }
        None
    }

    /// Add a variable. Its dimensions must be defined, its data must fill
    /// them, and its attribute names must be valid and unreserved.
    pub fn add_variable(&mut self, group: GroupId, variable: NcVariable) -> Result<(), WriteError> {
        check_name(&variable.name)?;
        if self.taken(group, &variable.name)? {
            return Err(WriteError::Duplicate(variable.name.clone()));
        }
        let mut expected = 1u64;
        for dim in &variable.dims {
            let len = self.dim_len(group, dim).ok_or_else(|| {
                WriteError::Invalid(format!(
                    "variable {}: dimension {dim} is not defined",
                    variable.name
                ))
            })?;
            expected = expected
                .checked_mul(len)
                .ok_or_else(|| WriteError::TooLarge(format!("variable {}", variable.name)))?;
        }
        if variable.data.len() as u64 != expected {
            return Err(WriteError::Invalid(format!(
                "variable {}: {} values for {expected} elements",
                variable.name,
                variable.data.len()
            )));
        }
        let mut seen: Vec<(String, NcAttr)> = Vec::with_capacity(variable.attrs.len());
        for (name, value) in &variable.attrs {
            check_attr(name, &seen)?;
            attr_value(value)?;
            seen.push((name.clone(), value.clone()));
        }
        if let NcStorage::Chunked { chunk, .. } = &variable.storage
            && chunk.len() != variable.dims.len()
        {
            return Err(WriteError::Invalid(format!(
                "variable {}: chunk rank {} for {} dimensions",
                variable.name,
                chunk.len(),
                variable.dims.len()
            )));
        }
        self.group_mut(group)?.variables.push(variable);
        Ok(())
    }

    /// Lay the file out and return its bytes.
    pub fn finish(self) -> Result<Vec<u8>, WriteError> {
        let mut writer = Writer::new();
        let root = writer.root();
        writer.add_attribute(root, "_NCProperties", Value::text(&self.properties))?;

        // HDF5 objects of every group, in pre-order (parents first).
        let mut order = Vec::with_capacity(self.groups.len());
        let mut stack = vec![GroupId(0)];
        while let Some(id) = stack.pop() {
            order.push(id);
            if let Some(group) = self.groups.get(id.0) {
                stack.extend(group.children.iter().rev().copied());
            }
        }
        let mut group_objects: HashMap<usize, ObjectId> = HashMap::new();
        group_objects.insert(0, root);
        // Dimension scale of (group, dimension name).
        let mut scales: HashMap<(usize, String), ObjectId> = HashMap::new();
        // Axes attached to each scale: (variable dataset, axis).
        let mut attached: HashMap<ObjectId, Vec<(ObjectId, i32)>> = HashMap::new();
        // Dimension ids in definition order, file-wide.
        let mut dimids: HashMap<(usize, String), i32> = HashMap::new();
        for id in &order {
            for dim in &self.groups[id.0].dims {
                let next = dimids.len() as i32;
                dimids.insert((id.0, dim.name.clone()), next);
            }
        }
        let dimid = |group: usize, name: &str| -> Value {
            let id = dimids.get(&(group, name.to_owned())).copied().unwrap_or(-1);
            Value::scalar(Data::I32(vec![id]))
        };
        // Variable datasets to give a DIMENSION_LIST: dataset, group and
        // dimension names (resolved once every scale exists).
        let mut dimension_lists: Vec<(ObjectId, GroupId, &[String], &str)> = Vec::new();

        for id in &order {
            let group = &self.groups[id.0];
            let object = match group.parent {
                None => root,
                Some(parent) => {
                    let parent_object = *group_objects.get(&parent.0).ok_or_else(|| {
                        WriteError::Invalid("group laid out before its parent".to_owned())
                    })?;
                    writer.add_group(parent_object, &group.name)?
                }
            };
            group_objects.insert(id.0, object);

            // Coordinate variables: a variable over exactly its own-named
            // dimension of this group.
            let is_coordinate = |variable: &NcVariable| {
                variable.dims.len() == 1
                    && variable.dims[0] == variable.name
                    && group.dims.iter().any(|dim| dim.name == variable.name)
            };
            // Dimension scales without a variable come first.
            for dim in &group.dims {
                if group
                    .variables
                    .iter()
                    .any(|var| var.name == dim.name && is_coordinate(var))
                {
                    continue;
                }
                let scale = writer.add_dataset(
                    object,
                    &dim.name,
                    NewDataset::new(
                        Value {
                            data: Data::F32(Vec::new()),
                            shape: Shape::Simple(vec![dim.len]),
                        },
                        Layout::Unallocated,
                    ),
                )?;
                writer.add_attribute(scale, "CLASS", Value::text_nul("DIMENSION_SCALE"))?;
                writer.add_attribute(
                    scale,
                    "NAME",
                    Value::text_nul(&format!("{DIM_WITHOUT_VARIABLE}{:10}", dim.len)),
                )?;
                writer.add_attribute(scale, "_Netcdf4Dimid", dimid(id.0, &dim.name))?;
                scales.insert((id.0, dim.name.clone()), scale);
            }
            for variable in &group.variables {
                let coordinate = is_coordinate(variable);
                let dataset_name =
                    if !coordinate && group.dims.iter().any(|d| d.name == variable.name) {
                        format!("{NON_COORD_PREFIX}{}", variable.name)
                    } else {
                        variable.name.clone()
                    };
                let shape = if variable.dims.is_empty() {
                    Shape::Scalar
                } else {
                    let mut dims = Vec::with_capacity(variable.dims.len());
                    for dim in &variable.dims {
                        dims.push(self.dim_len(*id, dim).unwrap_or(0));
                    }
                    Shape::Simple(dims)
                };
                let layout = match &variable.storage {
                    NcStorage::Contiguous => Layout::Contiguous,
                    NcStorage::Chunked {
                        chunk,
                        shuffle,
                        deflate,
                    } => Layout::Chunked {
                        chunk: chunk.clone(),
                        shuffle: *shuffle,
                        deflate: *deflate,
                    },
                };
                let mut dataset = NewDataset::new(
                    Value {
                        data: variable.data.clone(),
                        shape,
                    },
                    layout,
                );
                // The `_FillValue` attribute is the dataset's fill value too.
                if let Some((_, NcAttr::Numbers(fill))) =
                    variable.attrs.iter().find(|(name, _)| name == "_FillValue")
                    && fill.len() == 1
                    && std::mem::discriminant(fill) == std::mem::discriminant(&variable.data)
                {
                    dataset = dataset.with_fill_value(fill.clone());
                }
                let dataset_id = writer.add_dataset(object, &dataset_name, dataset)?;
                if coordinate {
                    writer.add_attribute(
                        dataset_id,
                        "CLASS",
                        Value::text_nul("DIMENSION_SCALE"),
                    )?;
                    writer.add_attribute(dataset_id, "NAME", Value::text_nul(&variable.name))?;
                    writer.add_attribute(
                        dataset_id,
                        "_Netcdf4Dimid",
                        dimid(id.0, &variable.name),
                    )?;
                    scales.insert((id.0, variable.name.clone()), dataset_id);
                } else if !variable.dims.is_empty() {
                    dimension_lists.push((dataset_id, *id, &variable.dims, &variable.name));
                }
                for (name, value) in &variable.attrs {
                    writer.add_attribute(dataset_id, name, attr_value(value)?)?;
                }
            }
            for (name, value) in &group.attrs {
                writer.add_attribute(object, name, attr_value(value)?)?;
            }
        }
        for (dataset, group, dims, name) in dimension_lists {
            let mut axis_scales = Vec::with_capacity(dims.len());
            for (axis, dim) in dims.iter().enumerate() {
                let scale = self.find_scale(&scales, group, dim).ok_or_else(|| {
                    WriteError::Invalid(format!("variable {name}: no scale for dimension {dim}"))
                })?;
                attached
                    .entry(scale)
                    .or_default()
                    .push((dataset, axis as i32));
                axis_scales.push(scale);
            }
            writer.add_attribute(
                dataset,
                "DIMENSION_LIST",
                Value::vector(Data::ObjectRefLists(
                    axis_scales.into_iter().map(|scale| vec![scale]).collect(),
                )),
            )?;
        }
        let mut attached: Vec<(ObjectId, Vec<(ObjectId, i32)>)> = attached.into_iter().collect();
        attached.sort_by_key(|(scale, _)| scale.0);
        for (scale, axes) in attached {
            writer.add_attribute(
                scale,
                "REFERENCE_LIST",
                Value::vector(Data::DimensionScaleRefs(axes)),
            )?;
        }
        writer.finish()
    }

    /// The scale of dimension `name` visible from `group` (the group's own,
    /// else the nearest ancestor's).
    fn find_scale(
        &self,
        scales: &HashMap<(usize, String), ObjectId>,
        group: GroupId,
        name: &str,
    ) -> Option<ObjectId> {
        let mut at = Some(group);
        while let Some(id) = at {
            if let Some(scale) = scales.get(&(id.0, name.to_owned())) {
                return Some(*scale);
            }
            at = self.groups.get(id.0)?.parent;
        }
        None
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::netcdf4::{NcFile, NcType};
    use crate::write::CharSet;

    fn sample() -> Vec<u8> {
        let mut nc = NcWriter::new();
        let root = nc.root();
        nc.add_dim(root, "sweep", 2).expect("dim");
        nc.add_attr(root, "Conventions", NcAttr::Text("CF-1.8".into()))
            .expect("attr");
        nc.add_attr(root, "empty", NcAttr::Text(String::new()))
            .expect("attr");
        nc.add_variable(
            root,
            NcVariable {
                name: "sweep_group_name".into(),
                dims: vec!["sweep".into()],
                data: Data::VarStrings {
                    values: vec!["sweep_0".into(), "sweep_1".into()],
                    charset: CharSet::Utf8,
                },
                attrs: Vec::new(),
                storage: NcStorage::Contiguous,
            },
        )
        .expect("var");
        let sweep = nc.add_group(root, "sweep_0").expect("group");
        nc.add_dim(sweep, "time", 3).expect("dim");
        nc.add_dim(sweep, "range", 4).expect("dim");
        nc.add_variable(
            sweep,
            NcVariable {
                name: "time".into(),
                dims: vec!["time".into()],
                data: Data::F64(vec![0.0, 1.0, 2.0]),
                attrs: vec![(
                    "units".into(),
                    NcAttr::Text("seconds since 2024-03-15T00:02:17Z".into()),
                )],
                storage: NcStorage::Contiguous,
            },
        )
        .expect("var");
        nc.add_variable(
            sweep,
            NcVariable {
                name: "DBZH".into(),
                dims: vec!["time".into(), "range".into()],
                data: Data::U8((0..12).collect()),
                attrs: vec![
                    ("_FillValue".into(), NcAttr::Numbers(Data::U8(vec![0]))),
                    ("scale_factor".into(), NcAttr::Numbers(Data::F64(vec![0.5]))),
                ],
                storage: NcStorage::Chunked {
                    chunk: vec![3, 4],
                    shuffle: true,
                    deflate: Some(4),
                },
            },
        )
        .expect("var");
        nc.add_variable(
            sweep,
            NcVariable {
                name: "sweep_mode".into(),
                dims: Vec::new(),
                data: Data::VarStrings {
                    values: vec!["azimuth_surveillance".into()],
                    charset: CharSet::Utf8,
                },
                attrs: Vec::new(),
                storage: NcStorage::Contiguous,
            },
        )
        .expect("var");
        // A variable named like a dimension it is not the coordinate of.
        nc.add_variable(
            sweep,
            NcVariable {
                name: "range".into(),
                dims: vec!["time".into()],
                data: Data::F32(vec![1.0, 2.0, 3.0]),
                attrs: Vec::new(),
                storage: NcStorage::Contiguous,
            },
        )
        .expect("var");
        nc.finish().expect("finish")
    }

    #[test]
    fn netcdf4_model_reads_back() {
        let bytes = sample();
        let file = NcFile::open(&bytes).expect("open");
        assert!(
            file.nc_properties()
                .is_some_and(|p| p.contains("recast-radar-hdf5"))
        );
        let root = file.root();
        assert_eq!(root.dims.len(), 1);
        assert_eq!(file.dim(root.dims[0]).map(|d| d.len), Some(2));
        let names = root.variable("sweep_group_name").expect("var");
        assert_eq!(names.nc_type, NcType::String);
        let group = file.group("/sweep_0").expect("group");
        let dims: Vec<(String, usize)> = group
            .dims
            .iter()
            .filter_map(|id| file.dim(*id).map(|d| (d.name.clone(), d.len)))
            .collect();
        assert_eq!(dims, [("time".to_owned(), 3), ("range".to_owned(), 4)]);
        let dbzh = group.variable("DBZH").expect("DBZH");
        let dim_names: Vec<&str> = dbzh
            .dims
            .iter()
            .filter_map(|id| file.dim(*id).map(|d| d.name.as_str()))
            .collect();
        assert_eq!(dim_names, ["time", "range"]);
        assert!(dbzh.attribute("DIMENSION_LIST").is_none());
        assert_eq!(
            dbzh.attributes.iter().map(|a| a.name()).collect::<Vec<_>>(),
            ["_FillValue", "scale_factor"]
        );
        let range = group.variable("range").expect("non-coordinate range");
        assert_eq!(range.path, "/sweep_0/_nc4_non_coord_range");
        assert_eq!(
            range
                .dims
                .iter()
                .filter_map(|id| file.dim(*id).map(|d| d.name.as_str()))
                .collect::<Vec<_>>(),
            ["time"]
        );
        let time = group.variable("time").expect("time");
        assert_eq!(time.attributes.len(), 1);
    }

    #[test]
    fn reserved_and_invalid_names_are_refused() {
        let mut nc = NcWriter::new();
        let root = nc.root();
        assert!(
            nc.add_attr(root, "CLASS", NcAttr::Text("x".into()))
                .is_err()
        );
        assert!(
            nc.add_attr(root, "_NCProperties", NcAttr::Text("x".into()))
                .is_err()
        );
        assert!(nc.add_group(root, "a/b").is_err());
        assert!(nc.add_dim(root, "", 1).is_err());
        assert!(
            nc.add_variable(
                root,
                NcVariable {
                    name: "v".into(),
                    dims: vec!["missing".into()],
                    data: Data::F32(vec![1.0]),
                    attrs: Vec::new(),
                    storage: NcStorage::Contiguous,
                },
            )
            .is_err()
        );
        assert!(is_valid_name("how.beamwH"));
        assert!(is_valid_name("1st"));
        assert!(!is_valid_name(".hidden"));
        assert!(!is_valid_name("trailing "));
    }
}
