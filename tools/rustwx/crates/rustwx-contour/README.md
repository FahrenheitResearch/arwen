# rustwx-contour

`rustwx-contour` provides renderer-agnostic contour and fill topology for the
`rustwx` plotting stack.

It is intentionally separate from `rustwx-render`: this crate owns scalar-field
grids, level bins, contour/fill geometry, and extraction algorithms, while the
renderer can later decide how to stroke or rasterize the resulting topology.

## Current scope

- rectilinear and projected scalar-field grids
- validated contour levels and contour-fill bins
- marching-squares isoline extraction with saddle resolution
- per-cell filled-band polygon extraction with consistent threshold assignment

## Integration shape

`rustwx-render` can integrate later by:

1. building a `ScalarField2D<ProjectedGrid>` from projected domain points
2. extracting isolines and filled-band polygons with `ContourEngine`
3. mapping line segments and polygon rings into the renderer's existing overlay
   and rasterization pipeline

The current fill path merges triangle-clipped fragments into cell-local
polygons. That gives stable band topology now without forcing a global polygon
stitcher into the first version.
