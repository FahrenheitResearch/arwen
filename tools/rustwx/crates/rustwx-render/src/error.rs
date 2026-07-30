use thiserror::Error;

#[derive(Debug, Error)]
pub enum RustwxRenderError {
    #[error("invalid grid shape: nx={nx}, ny={ny}")]
    InvalidGridShape { nx: usize, ny: usize },
    #[error(
        "field grid shape does not match render layer length for {layer}: expected {expected}, got {actual}"
    )]
    LayerShapeMismatch {
        layer: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("render request overlay grid does not match base field grid for {layer}")]
    OverlayGridMismatch { layer: &'static str },
    #[error("render request requires at least two x/y points per dimension")]
    DegenerateProjectedGrid,
    #[error("render request requires matching projected x/y arrays")]
    InvalidProjectedGrid,
    #[error("failed to build projected contour topology: {0}")]
    ContourTopology(String),
    #[error(
        "invalid panel layout: rows={rows}, columns={columns}, panel_width={panel_width}, panel_height={panel_height}"
    )]
    InvalidPanelLayout {
        rows: u32,
        columns: u32,
        panel_width: u32,
        panel_height: u32,
    },
    #[error("panel layout dimensions overflowed u32 canvas bounds")]
    PanelLayoutOverflow,
    #[error("panel count {actual} exceeds layout capacity {capacity}")]
    TooManyPanels { actual: usize, capacity: usize },
    #[error(
        "panel {index} had size {actual_width}x{actual_height}, expected {expected_width}x{expected_height}"
    )]
    PanelSizeMismatch {
        index: usize,
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    #[error("failed to decode rendered PNG into an RGBA image: {source}")]
    DecodeRenderedPng {
        #[source]
        source: image::ImageError,
    },
    #[error("failed to copy panel {index} into the composed canvas: {source}")]
    ComposePanel {
        index: usize,
        #[source]
        source: image::ImageError,
    },
    #[error("failed to write rendered PNG to {path}: {source}")]
    WriteFile {
        path: String,
        #[source]
        source: std::io::Error,
    },
}
