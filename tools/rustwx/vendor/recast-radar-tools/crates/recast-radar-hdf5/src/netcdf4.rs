//! The netCDF-4 data model over an HDF5 file.
//!
//! netCDF-4 files are HDF5 files written with netCDF-C's conventions
//! ("NetCDF-4 Format", netCDF-C documentation, file format
//! specifications; netCDF-C `libhdf5/hdf5open.c` for how the library reads
//! them back). This module rebuilds what netCDF-C (and so netCDF4-python,
//! xarray and Py-ART) show for such a file:
//!
//! - **Groups** are HDF5 groups, walked from the root in link order
//!   (creation order when the file tracks it, which netCDF-C always asks
//!   for).
//! - **Dimensions** are HDF5 dimension scales: 1-D datasets whose `CLASS`
//!   attribute is `DIMENSION_SCALE`. A scale whose `NAME` attribute starts
//!   with `This is a netCDF dimension but not a netCDF variable` is only a
//!   dimension; any other scale is also the dimension's coordinate
//!   variable. The dimension is unlimited when the scale's maximum size is
//!   unlimited, and then its length is the largest extent any variable has
//!   along it. Dimensions are ordered by their `_Netcdf4Dimid`.
//! - **Variables** are the other datasets (plus coordinate variables). A
//!   variable's dimensions come from its `DIMENSION_LIST` attribute (one
//!   object reference per axis, to the scale), from `_Netcdf4Coordinates`
//!   (dimension ids) for a multi-dimensional coordinate variable, or are
//!   the coordinate variable's own dimension. A dataset without dimension
//!   scales gets netCDF-C's phony dimensions: an existing dimension of the
//!   group with the same length and unlimitedness that no other axis of the
//!   dataset uses, else a new `phony_dim_<id>` (any HDF5 file, ODIM_H5
//!   included, reads this way). A
//!   dataset netCDF-C renamed with its `_nc4_non_coord_` prefix (a variable
//!   that shares a dimension's name without being its coordinate) gets its
//!   name back.
//! - **Attributes** are the HDF5 attributes minus the ones netCDF-C
//!   reserves and hides (`CLASS`, `NAME`, `REFERENCE_LIST`,
//!   `DIMENSION_LIST`, `_Netcdf4Dimid`, `_Netcdf4Coordinates`,
//!   `_nc3_strict`, `_NCProperties`). `_NCProperties` (the writing library
//!   versions) is [`NcFile::nc_properties`]; `_nc3_strict` marks the
//!   netCDF-4 classic model ([`NcFile::is_classic_model`]).
//! - **Types**: [`NcType`] maps each HDF5 datatype to the netCDF external
//!   type netCDF-C reports. Fixed-length strings are `char` (a text
//!   attribute is one `char` array; a `char` variable has one byte per
//!   element), variable-length strings are `string`, and compound, enum,
//!   opaque and variable-length sequence types are user-defined.
//!
//! Values are read in their stored type through [`H5File::dataset`]
//! ([`NcFile::read`]); every structure goes through the HDF5 reader's
//! limits. The data model adds at most one dimension per dataset axis and
//! one variable per dataset, both bounded by the reader's object limit.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::UNLIMITED;
use crate::datatype::Datatype;
use crate::error::{Error, Result, invalid, limit};
use crate::file::{Attribute, H5File, ObjectKind};
use crate::limits::{MAX_GROUP_DEPTH, MAX_OBJECTS};
use crate::link::LinkTarget;
use crate::values::Values;

/// Prefix of the `NAME` attribute of a dimension scale that is only a
/// dimension (netCDF-C `DIM_WITHOUT_VARIABLE`).
const DIM_WITHOUT_VARIABLE: &str = "This is a netCDF dimension but not a netCDF variable";

/// Prefix netCDF-C gives the dataset of a variable that has a dimension's
/// name but is not its coordinate variable (`NON_COORD_PREPEND`).
const NON_COORD_PREFIX: &str = "_nc4_non_coord_";

/// Attributes netCDF-C reserves and hides from the attribute lists
/// (netCDF-C `NC_reserved`, the entries flagged hidden, plus `_nc3_strict`,
/// which the library turns into the classic-model mode).
const HIDDEN_ATTRIBUTES: &[&str] = &[
    "CLASS",
    "NAME",
    "REFERENCE_LIST",
    "DIMENSION_LIST",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "_nc3_strict",
    "_NCProperties",
];

/// netCDF external type of a variable or attribute (netCDF-C `nc_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NcType {
    /// `byte`: 8-bit signed integer.
    Byte,
    /// `ubyte`: 8-bit unsigned integer.
    UByte,
    /// `char`: 8-bit character (text).
    Char,
    /// `short`: 16-bit signed integer.
    Short,
    /// `ushort`: 16-bit unsigned integer.
    UShort,
    /// `int`: 32-bit signed integer.
    Int,
    /// `uint`: 32-bit unsigned integer.
    UInt,
    /// `int64`: 64-bit signed integer.
    Int64,
    /// `uint64`: 64-bit unsigned integer.
    UInt64,
    /// `float`: IEEE binary32.
    Float,
    /// `double`: IEEE binary64.
    Double,
    /// `string`: variable-length string.
    String,
    /// A user-defined type (compound, enum, opaque, variable-length
    /// sequence) or an HDF5 type netCDF has no atomic type for.
    UserDefined,
}

impl NcType {
    /// The netCDF type of an HDF5 datatype.
    pub fn of(datatype: &Datatype) -> Self {
        match datatype {
            Datatype::Integer { size, signed, .. } => match (size, signed) {
                (1, true) => Self::Byte,
                (1, false) => Self::UByte,
                (2, true) => Self::Short,
                (2, false) => Self::UShort,
                (4, true) => Self::Int,
                (4, false) => Self::UInt,
                (8, true) => Self::Int64,
                (8, false) => Self::UInt64,
                _ => Self::UserDefined,
            },
            Datatype::Float { size: 4, .. } => Self::Float,
            Datatype::Float { size: 8, .. } => Self::Double,
            Datatype::FixedString { .. } => Self::Char,
            Datatype::VarLenString { .. } => Self::String,
            _ => Self::UserDefined,
        }
    }

    /// The CDL name (`byte`, `ubyte`, `char`, `short`, `ushort`, `int`,
    /// `uint`, `int64`, `uint64`, `float`, `double`, `string`), or
    /// `user-defined`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Byte => "byte",
            Self::UByte => "ubyte",
            Self::Char => "char",
            Self::Short => "short",
            Self::UShort => "ushort",
            Self::Int => "int",
            Self::UInt => "uint",
            Self::Int64 => "int64",
            Self::UInt64 => "uint64",
            Self::Float => "float",
            Self::Double => "double",
            Self::String => "string",
            Self::UserDefined => "user-defined",
        }
    }
}

/// A netCDF dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcDim {
    /// Dimension name.
    pub name: String,
    /// Current length (for an unlimited dimension, the largest extent of
    /// any variable along it).
    pub len: usize,
    /// True for an unlimited (record) dimension.
    pub unlimited: bool,
    /// Path of the group that defines the dimension.
    pub group: String,
}

/// A netCDF variable.
#[derive(Clone, Debug, PartialEq)]
pub struct NcVariable {
    /// Variable name.
    pub name: String,
    /// HDF5 path of the dataset that stores it.
    pub path: String,
    /// Dimension ids ([`NcFile::dim`]), one per axis; empty for a scalar.
    pub dims: Vec<usize>,
    /// Stored extent per axis (the HDF5 dataset's current dimensions). It
    /// equals the dimension lengths except along an unlimited dimension
    /// another variable has grown further.
    pub stored_shape: Vec<usize>,
    /// netCDF type.
    pub nc_type: NcType,
    /// HDF5 datatype as stored.
    pub datatype: Datatype,
    /// Attributes, in storage order, without the hidden ones.
    pub attributes: Vec<Attribute>,
}

impl NcVariable {
    /// The attribute called `name`.
    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name() == name)
    }
}

/// A netCDF group.
#[derive(Clone, Debug, PartialEq)]
pub struct NcGroup {
    /// Group name (`/` for the root group).
    pub name: String,
    /// Absolute path (`/`, `/sweep_0`, ...).
    pub path: String,
    /// Dimension ids defined in this group, in netCDF id order.
    pub dims: Vec<usize>,
    /// Variables in creation order.
    pub variables: Vec<NcVariable>,
    /// Attributes, in storage order, without the hidden ones.
    pub attributes: Vec<Attribute>,
    /// Paths of the child groups, in link order.
    pub groups: Vec<String>,
}

impl NcGroup {
    /// The variable called `name`.
    pub fn variable(&self, name: &str) -> Option<&NcVariable> {
        self.variables.iter().find(|variable| variable.name == name)
    }

    /// The attribute called `name`.
    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name() == name)
    }
}

/// A netCDF-4 view of an HDF5 file.
pub struct NcFile<'a> {
    h5: H5File<'a>,
    groups: Vec<NcGroup>,
    by_path: BTreeMap<String, usize>,
    dims: Vec<NcDim>,
    properties: Option<String>,
    classic_model: bool,
}

/// A dimension scale found while walking the groups.
struct Scale {
    /// Index into the file's dimension list.
    dim: usize,
    /// The scale's `_Netcdf4Dimid`, when stored.
    netcdf_id: Option<i64>,
}

impl<'a> NcFile<'a> {
    /// Open an HDF5 byte buffer and build its netCDF-4 view.
    pub fn open(bytes: &'a [u8]) -> Result<Self> {
        Self::from_hdf5(H5File::open(bytes)?)
    }

    /// Build the netCDF-4 view of an opened HDF5 file.
    pub fn from_hdf5(h5: H5File<'a>) -> Result<Self> {
        let mut builder = Builder {
            h5: &h5,
            dims: Vec::new(),
            scales: HashMap::new(),
            groups: Vec::new(),
            order: Vec::new(),
            visited: HashSet::new(),
        };
        builder.walk("/", "/", 0)?;
        builder.variables()?;
        let Builder { dims, groups, .. } = builder;
        let root_attributes = h5.attrs("/");
        let properties = root_attributes
            .iter()
            .find(|attribute| attribute.name() == "_NCProperties")
            .and_then(Attribute::as_str);
        let classic_model = root_attributes
            .iter()
            .any(|attribute| attribute.name() == "_nc3_strict");
        let by_path = groups
            .iter()
            .enumerate()
            .map(|(index, group)| (group.path.clone(), index))
            .collect();
        Ok(Self {
            h5,
            groups,
            by_path,
            dims,
            properties,
            classic_model,
        })
    }

    /// The HDF5 file underneath.
    pub fn hdf5(&self) -> &H5File<'a> {
        &self.h5
    }

    /// Give the HDF5 file back.
    pub fn into_hdf5(self) -> H5File<'a> {
        self.h5
    }

    /// The root group.
    pub fn root(&self) -> &NcGroup {
        // The walk always records the root group first.
        &self.groups[0]
    }

    /// The group at an absolute path.
    pub fn group(&self, path: &str) -> Option<&NcGroup> {
        self.by_path.get(path).map(|index| &self.groups[*index])
    }

    /// Every group, root first, depth first in link order.
    pub fn groups(&self) -> &[NcGroup] {
        &self.groups
    }

    /// The dimension with id `id`.
    pub fn dim(&self, id: usize) -> Option<&NcDim> {
        self.dims.get(id)
    }

    /// Every dimension; ids index this slice.
    pub fn dims(&self) -> &[NcDim] {
        &self.dims
    }

    /// The dimension ids visible in the group at `path`: its own, then its
    /// parent's, up to the root's (the order netCDF-C searches them).
    pub fn visible_dims(&self, path: &str) -> Vec<usize> {
        let mut out = Vec::new();
        let mut current = Some(path.to_owned());
        while let Some(path) = current {
            if let Some(group) = self.group(&path) {
                out.extend(group.dims.iter().copied());
            }
            current = parent_path(&path);
        }
        out
    }

    /// The dimension lengths of a variable (its netCDF shape).
    pub fn shape(&self, variable: &NcVariable) -> Vec<usize> {
        variable
            .dims
            .iter()
            .map(|id| self.dims.get(*id).map_or(0, |dim| dim.len))
            .collect()
    }

    /// Read a variable's values in their stored type. A variable stored
    /// shorter than an unlimited dimension comes back at its stored extent
    /// ([`NcVariable::stored_shape`]).
    pub fn read(&self, variable: &NcVariable) -> Result<Values> {
        Ok(self.h5.dataset(&variable.path)?.values)
    }

    /// The root `_NCProperties` attribute (writer library versions,
    /// `version=2,netcdf=4.9.2,hdf5=1.14.3`), when the file has one
    /// (netCDF-C 4.4.1 and later write it).
    pub fn nc_properties(&self) -> Option<&str> {
        self.properties.as_deref()
    }

    /// True for the netCDF-4 classic model (the root has `_nc3_strict`).
    pub fn is_classic_model(&self) -> bool {
        self.classic_model
    }
}

/// True when an HDF5 file carries netCDF-4 markers: a root `_NCProperties`
/// or `_nc3_strict` attribute, or a dataset with `_Netcdf4Dimid` or a
/// `DIMENSION_SCALE` class.
pub fn is_netcdf4(h5: &H5File<'_>) -> bool {
    let root = h5.attrs("/");
    if root
        .iter()
        .any(|attribute| matches!(attribute.name(), "_NCProperties" | "_nc3_strict"))
    {
        return true;
    }
    h5.objects().any(|(_, object)| {
        object.kind() == ObjectKind::Dataset
            && (object.attribute("_Netcdf4Dimid").is_some()
                || object
                    .attribute("CLASS")
                    .and_then(Attribute::as_str)
                    .is_some_and(|class| class == "DIMENSION_SCALE"))
    })
}

fn parent_path(path: &str) -> Option<String> {
    if path == "/" {
        return None;
    }
    let trimmed = path.trim_end_matches('/');
    match trimmed.rfind('/') {
        Some(0) => Some("/".to_owned()),
        Some(index) => Some(trimmed[..index].to_owned()),
        None => Some("/".to_owned()),
    }
}

fn child_path(parent: &str, name: &str) -> String {
    if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

fn visible(attributes: &[Attribute]) -> Vec<Attribute> {
    attributes
        .iter()
        .filter(|attribute| !HIDDEN_ATTRIBUTES.contains(&attribute.name()))
        .cloned()
        .collect()
}

struct Builder<'h, 'a> {
    h5: &'h H5File<'a>,
    dims: Vec<NcDim>,
    /// Scale dataset header address -> its dimension.
    scales: HashMap<u64, Scale>,
    groups: Vec<NcGroup>,
    /// Per group: (dataset link name, path) of every dataset child, in link
    /// order, for the variable pass.
    order: Vec<Vec<(String, String)>>,
    visited: HashSet<u64>,
}

impl Builder<'_, '_> {
    /// First pass: groups, their attributes and their dimension scales.
    fn walk(&mut self, path: &str, name: &str, depth: usize) -> Result<()> {
        if depth > MAX_GROUP_DEPTH {
            return Err(limit(format!(
                "netCDF-4 group nesting is deeper than {MAX_GROUP_DEPTH} levels (limit)"
            )));
        }
        let object = self
            .h5
            .object(path)
            .ok_or_else(|| Error::NotFound(path.to_owned()))?;
        if !self.visited.insert(object.address()) {
            return Ok(());
        }
        let group_index = self.groups.len();
        self.groups.push(NcGroup {
            name: name.to_owned(),
            path: path.to_owned(),
            dims: Vec::new(),
            variables: Vec::new(),
            attributes: visible(object.attributes()),
            groups: Vec::new(),
        });
        self.order.push(Vec::new());
        let mut scales: Vec<(Option<i64>, usize)> = Vec::new();
        let mut subgroups = Vec::new();
        for link in object.links() {
            if !matches!(link.target, LinkTarget::Hard(_)) {
                continue;
            }
            let child = child_path(path, &link.name);
            let Some(target) = self.h5.object(&child) else {
                continue;
            };
            match target.kind() {
                ObjectKind::Group => subgroups.push((child, link.name.clone())),
                ObjectKind::Dataset => {
                    let is_scale = target
                        .attribute("CLASS")
                        .and_then(Attribute::as_str)
                        .is_some_and(|class| class == "DIMENSION_SCALE");
                    if is_scale && !self.scales.contains_key(&target.address()) {
                        let info = self.h5.dataset_info(&child)?;
                        if let [len] = info.dims.as_slice() {
                            if self.dims.len() >= MAX_OBJECTS {
                                return Err(limit(format!(
                                    "netCDF-4 file defines more than {MAX_OBJECTS} dimensions (limit)"
                                )));
                            }
                            let unlimited = info
                                .max_dims
                                .as_ref()
                                .and_then(|max| max.first())
                                .is_some_and(|max| *max == UNLIMITED);
                            let dim = self.dims.len();
                            self.dims.push(NcDim {
                                name: link
                                    .name
                                    .strip_prefix(NON_COORD_PREFIX)
                                    .unwrap_or(&link.name)
                                    .to_owned(),
                                len: *len,
                                unlimited,
                                group: path.to_owned(),
                            });
                            let netcdf_id = target
                                .attribute("_Netcdf4Dimid")
                                .and_then(Attribute::as_i64);
                            self.scales
                                .insert(target.address(), Scale { dim, netcdf_id });
                            scales.push((netcdf_id, dim));
                        }
                    }
                    self.order[group_index].push((link.name.clone(), child));
                }
                _ => {}
            }
        }
        // netCDF-C lists a group's dimensions by id; files without
        // `_Netcdf4Dimid` keep link order.
        if scales.iter().all(|(id, _)| id.is_some()) {
            scales.sort_by_key(|(id, _)| *id);
        }
        self.groups[group_index].dims = scales.into_iter().map(|(_, dim)| dim).collect();
        for (child, child_name) in subgroups {
            self.groups[group_index].groups.push(child.clone());
            self.walk(&child, &child_name, depth + 1)?;
        }
        Ok(())
    }

    /// Second pass: variables with their dimensions, then the lengths of
    /// the unlimited dimensions.
    fn variables(&mut self) -> Result<()> {
        let netcdf_ids: HashMap<i64, usize> = self
            .scales
            .values()
            .filter_map(|scale| scale.netcdf_id.map(|id| (id, scale.dim)))
            .collect();
        for group_index in 0..self.groups.len() {
            let entries = std::mem::take(&mut self.order[group_index]);
            for (name, path) in entries {
                let Some(object) = self.h5.object(&path) else {
                    continue;
                };
                let own_scale = self.scales.get(&object.address()).map(|scale| scale.dim);
                let only_dimension = object
                    .attribute("NAME")
                    .and_then(Attribute::as_str)
                    .is_some_and(|text| text.starts_with(DIM_WITHOUT_VARIABLE));
                if own_scale.is_some() && only_dimension {
                    continue;
                }
                let info = self.h5.dataset_info(&path)?;
                let rank = if info.null { 0 } else { info.dims.len() };
                let dims = match own_scale {
                    Some(dim) => match object
                        .attribute("_Netcdf4Coordinates")
                        .map(|attribute| attribute.values())
                    {
                        Some(values) if values.len() == rank && rank > 1 => (0..rank)
                            .map(|axis| {
                                values
                                    .get_i64(axis)
                                    .and_then(|id| netcdf_ids.get(&id).copied())
                                    .ok_or_else(|| {
                                        invalid(
                                            0,
                                            format!(
                                                "netCDF-4 variable '{path}' names an unknown dimension id in _Netcdf4Coordinates"
                                            ),
                                        )
                                    })
                            })
                            .collect::<Result<Vec<_>>>()?,
                        _ => vec![dim],
                    },
                    None => match self.dimension_list(object.attribute("DIMENSION_LIST"), rank)
                    {
                        Some(dims) => dims,
                        None => self.phony_dims(group_index, &info.dims, info.max_dims.as_deref(), rank)?,
                    },
                };
                if dims.len() != rank {
                    return Err(invalid(
                        0,
                        format!(
                            "netCDF-4 variable '{path}' has {} dimensions for a rank-{rank} dataset",
                            dims.len()
                        ),
                    ));
                }
                let stored_shape = if info.null {
                    Vec::new()
                } else {
                    info.dims.clone()
                };
                for (dim, extent) in dims.iter().zip(&stored_shape) {
                    if let Some(dim) = self.dims.get_mut(*dim)
                        && dim.unlimited
                    {
                        dim.len = dim.len.max(*extent);
                    }
                }
                let name = name
                    .strip_prefix(NON_COORD_PREFIX)
                    .unwrap_or(&name)
                    .to_owned();
                self.groups[group_index].variables.push(NcVariable {
                    name,
                    path,
                    dims,
                    stored_shape,
                    nc_type: NcType::of(&info.datatype),
                    datatype: info.datatype,
                    attributes: visible(object.attributes()),
                });
            }
        }
        Ok(())
    }

    /// The dimensions a `DIMENSION_LIST` attribute names: one sequence of
    /// object references per axis, each to a dimension scale. `None` when
    /// the attribute is missing or does not resolve.
    fn dimension_list(&self, attribute: Option<&Attribute>, rank: usize) -> Option<Vec<usize>> {
        let Values::Sequences(axes) = attribute?.values() else {
            return None;
        };
        if axes.len() != rank {
            return None;
        }
        axes.iter()
            .map(|axis| match axis {
                Values::References(targets) => targets
                    .iter()
                    .flatten()
                    .find_map(|address| self.scales.get(address).map(|scale| scale.dim)),
                _ => None,
            })
            .collect()
    }

    /// netCDF-C's phony dimensions for a dataset without dimension scales:
    /// per axis, the group's first dimension with the same length and
    /// unlimitedness, else a new `phony_dim_<id>`.
    fn phony_dims(
        &mut self,
        group_index: usize,
        extents: &[usize],
        max_dims: Option<&[u64]>,
        rank: usize,
    ) -> Result<Vec<usize>> {
        let mut out = Vec::with_capacity(rank);
        for (axis, len) in extents.iter().take(rank).enumerate() {
            let unlimited = max_dims
                .and_then(|max| max.get(axis))
                .is_some_and(|max| *max == UNLIMITED);
            // A dimension serves one axis per variable: a 500 x 500 plane
            // gets two dimensions.
            let existing = self.groups[group_index].dims.iter().copied().find(|id| {
                let dim = &self.dims[*id];
                dim.len == *len && dim.unlimited == unlimited && !out.contains(id)
            });
            let id = match existing {
                Some(id) => id,
                None => {
                    if self.dims.len() >= MAX_OBJECTS {
                        return Err(limit(format!(
                            "netCDF-4 file defines more than {MAX_OBJECTS} dimensions (limit)"
                        )));
                    }
                    let id = self.dims.len();
                    let path = self.groups[group_index].path.clone();
                    self.dims.push(NcDim {
                        name: format!("phony_dim_{id}"),
                        len: *len,
                        unlimited,
                        group: path,
                    });
                    self.groups[group_index].dims.push(id);
                    id
                }
            };
            out.push(id);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_paths() {
        assert_eq!(parent_path("/"), None);
        assert_eq!(parent_path("/sweep_0").as_deref(), Some("/"));
        assert_eq!(parent_path("/a/b").as_deref(), Some("/a"));
    }

    #[test]
    fn netcdf_types_of_hdf5_datatypes() {
        use crate::datatype::{ByteOrder, CharSet, StringPadding};
        let int = |size, signed| Datatype::Integer {
            size,
            signed,
            order: ByteOrder::LittleEndian,
            bit_offset: 0,
            precision: (size * 8) as u16,
        };
        assert_eq!(NcType::of(&int(1, true)), NcType::Byte);
        assert_eq!(NcType::of(&int(8, false)), NcType::UInt64);
        assert_eq!(NcType::of(&int(3, true)), NcType::UserDefined);
        assert_eq!(
            NcType::of(&Datatype::FixedString {
                size: 1,
                padding: StringPadding::NullTerminate,
                charset: CharSet::Ascii
            }),
            NcType::Char
        );
        assert_eq!(NcType::Short.name(), "short");
    }
}
