//! The ODIM_H5 attribute tables (EUMETNET OPERA WD_2008_03, v2.4) that the
//! reader and the writer share: which attribute names the specification
//! puts in `what` and in `where` at each level of a polar volume.
//!
//! A kept attribute (one no typed slot holds) reaches the model under its
//! bare name when that name alone tells its group: a `what` or `where`
//! name of these tables, or a `how` name outside them. Any other kept
//! attribute is `<group>.<name>` (a `what` attribute the tables do not
//! list, FMI's `what/type`; a `how` attribute named like a table one), so
//! the writer puts every attribute back in the group it came from.

/// ODIM_H5 top-level `what` attributes (Table 1).
pub(crate) const ROOT_WHAT: &[&str] = &["object", "version", "date", "time", "source"];
/// ODIM_H5 top-level `where` attributes of polar data (Table 4).
pub(crate) const ROOT_WHERE: &[&str] = &["lon", "lat", "height"];
/// ODIM_H5 dataset `what` attributes (Table 13).
pub(crate) const DATASET_WHAT: &[&str] = &[
    "product",
    "prodpar",
    "quantity",
    "startdate",
    "starttime",
    "enddate",
    "endtime",
    "gain",
    "offset",
    "nodata",
    "undetect",
];
/// ODIM_H5 dataset `where` attributes of polar data (Table 4).
pub(crate) const DATASET_WHERE: &[&str] = &[
    "elangle", "nbins", "rstart", "rscale", "nrays", "a1gate", "startaz", "stopaz", "startel",
    "stopel",
];
/// ODIM_H5 plane `what` attributes (Table 13).
pub(crate) const PLANE_WHAT: &[&str] = &["quantity", "gain", "offset", "nodata", "undetect"];

/// The level of a polar volume an attribute group belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Level {
    /// The file's top-level groups.
    Root,
    /// A `datasetN` group (a sweep).
    Dataset,
    /// A `dataM` or `qualityK` group (a field).
    Plane,
}

impl Level {
    /// The `what` and `where` tables of this level.
    pub(crate) fn tables(self) -> (&'static [&'static str], &'static [&'static str]) {
        match self {
            Level::Root => (ROOT_WHAT, ROOT_WHERE),
            Level::Dataset => (DATASET_WHAT, DATASET_WHERE),
            Level::Plane => (PLANE_WHAT, &[]),
        }
    }

    /// The group (`what` or `where`) the tables put `name` in at this
    /// level; `None` for a name they do not list.
    pub(crate) fn table_group(self, name: &str) -> Option<&'static str> {
        let (what, where_) = self.tables();
        if what.contains(&name) {
            Some("what")
        } else if where_.contains(&name) {
            Some("where")
        } else {
            None
        }
    }

    /// `true` when an attribute `name` of `group` (`what`, `where` or
    /// `how`) is kept under its bare name: its name alone places it back
    /// in `group` (module documentation). A `how` subgroup's attribute
    /// (`radar_system.name`) names its subgroup already.
    pub(crate) fn bare_name_places(self, group: &str, name: &str) -> bool {
        match (group, self.table_group(name)) {
            ("how", table) => table.is_none(),
            (group, Some(table)) => group == table,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_names_place_only_their_own_group() {
        assert!(Level::Plane.bare_name_places("what", "gain"));
        // FMI's plane `what/type` is not an ODIM_H5 plane `what` attribute.
        assert!(!Level::Plane.bare_name_places("what", "type"));
        assert!(Level::Plane.bare_name_places("how", "task"));
        // A `how` attribute named like a table attribute.
        assert!(!Level::Plane.bare_name_places("how", "gain"));
        assert!(Level::Dataset.bare_name_places("where", "a1gate"));
        assert!(!Level::Dataset.bare_name_places("where", "range"));
        assert!(!Level::Root.bare_name_places("what", "NAME"));
        assert!(Level::Root.bare_name_places("how", "beamwH"));
        assert!(Level::Root.bare_name_places("how", "radar_system.name"));
    }
}
