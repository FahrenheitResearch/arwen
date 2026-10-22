use crate::geometry::{LineSegment, Polygon};
use crate::grid::CellIndex;
use crate::levels::{LevelBin, LevelBins};

#[derive(Clone, Debug, PartialEq)]
pub struct ContourTopology {
    pub layers: Vec<ContourLayer>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContourLayer {
    pub level: f64,
    pub segments: Vec<ContourSegment>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContourSegment {
    pub cell: CellIndex,
    pub geometry: LineSegment,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FillTopology {
    pub bins: LevelBins,
    pub polygons: Vec<BandPolygon>,
}

impl FillTopology {
    pub fn bin(&self, index: usize) -> Option<&LevelBin> {
        self.bins.bins().get(index)
    }


}

#[derive(Clone, Debug, PartialEq)]
pub struct BandPolygon {
    pub bin_index: usize,
    pub cell: CellIndex,
    pub polygon: Polygon,
}
