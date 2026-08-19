//! Map overlays and panel annotations -- re-exported, not redefined.
//!
//! The types live in [`rustwx_products::geographic_overlays`] because the
//! DIRECT render lane has to apply them too (`rw_wrfbatch --overlays`
//! reaches every named product through `DirectBatchRequest`), and a
//! second definition here would be a second schema that drifts.  This
//! module exists so the three binaries in this crate name one import
//! path.

pub use rustwx_products::geographic_overlays::{
    LabelSpec, LineSpec, MapOverlays, PanelAnnotations, PointSpec, RingSpec, parse_color,
};
