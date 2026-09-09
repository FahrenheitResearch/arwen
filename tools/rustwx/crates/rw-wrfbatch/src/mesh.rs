//! `mesh:` and `meshdiff:` products: an unstructured model mesh drawn CELL BY
//! CELL, straight from a history frame and its grid file.
//!
//! WHAT BREAKAGE THIS PREVENTS (gate law): a mesh forecast
//! resampled onto a lat/lon frame before it is drawn.  Every other family
//! here starts from a structured `(ny, nx)` plane, so a Voronoi mesh reaches
//! them through a regrid, and the regrid invents values between cell centres,
//! erases the mesh's own refinement, and smears a forty-cell response across
//! its neighbours.  A pair difference exists to show exactly that response.
//!
//! So this family never regrids.  It reads the history frame's own
//! `nCells`-shaped field, reads the cell boundaries out of the grid file
//! (`verticesOnCell`, `nEdgesOnCell`, `latVertex`/`lonVertex`), projects each
//! ring into the SAME projected space the panel's basemap is built in, and
//! hands the renderer one filled polygon per cell with a hairline along every
//! edge.  Basemap, chrome, colorbar and provenance label are the ones every
//! other product carries.
//!
//! ## Product grammar
//!
//! ```text
//! mesh:<field>[:colmax|:colmin|:level=K][@LO[..HI]][~log]
//! meshdiff:<field>[:colmax|:colmin|:level=K][@LO[..HI]][~log]
//! ```
//!
//! * `<field>` is a variable name in the history file, exactly as the file
//!   spells it (`qi`, `qc`, `refl10cm`, a carried tracer, ...).  Nothing in
//!   this module knows any model's variable list: a field this build has
//!   never heard of is a name in a file, not a code path.
//! * A 3-D `(nCells, nVertLevels)` field needs a level selector and is
//!   refused by name without one; `:colmax` and `:colmin` take the column
//!   extreme, `:level=K` takes one level, counted the way the model counts
//!   them (`level=1` is the first).
//! * `~log` draws `log10` of the field, with zero and negative cells taking
//!   the theme's empty fill, for quantities that span decades.
//! * `@LO[..HI]` fixes the fill's range in the field's OWN units and gives
//!   every cell below `LO` the theme's empty fill.  Absent, the ramp spans
//!   the frame's full finite range, which is right for a field whose zero
//!   means zero and wrong for one carrying a no-echo sentinel: simulated
//!   reflectivity is -95 dBZ wherever there is nothing, and a ramp from
//!   -95 to +50 spends nine tenths of its colours on empty air.  A user
//!   naming `@5` is naming a threshold, not adding a code path, so a field
//!   this build has never seen still gets a readable panel.  The two ends
//!   are separated by `..`, never a comma: `--products` is a comma-separated
//!   list and a range written with one would be read as two products.
//!
//! `meshdiff:` subtracts a reference frame cell by cell and draws the result
//! on a zero-centred diverging scale.  The reference comes from
//! `--mesh-reference PATH` the way `--minus-store-root` gives the store
//! lane's `diff:` family its reference: a directory holding the other leg's
//! frames (matched by file name) or a single file when a single frame is
//! rendered.  `--mesh-labels A,B` names the two legs; the default is
//! `TREATMENT,CONTROL`.
//!
//! The grid file is `--mesh-grid FILE.nc` and is required: a history frame
//! carries cell CENTRES and no boundaries at all, so without it there are no
//! polygons to draw and the family refuses rather than falling back to the
//! regrid it exists to avoid.

use std::path::{Path, PathBuf};

use rustwx_render::{
    ColorScale, CoreField2D, CoreGridShape, CoreLatLonGrid, CoreProductKey, DiscreteColorScale,
    ExtendMode, MapRenderRequest, MeshCell, MeshCellsLayer, PngWriteOptions, ProductVisualMode,
    ProjectedDomain, ProjectedMapBuildOptions, RenderTheme, build_projected_map_with_options,
    map_frame_aspect_ratio_for_mode, project_geographic_points_with_options,
    resolved_projection_for_options, save_png_profile_with_options,
};

/// Earth's radius in metres, for turning the grid file's unit-sphere
/// `areaCell` into a spacing a caption can carry.
const EARTH_RADIUS_M: f64 = 6_371_229.0;

/// The two product-family prefixes.
pub const PREFIX: &str = "mesh:";
pub const DIFF_PREFIX: &str = "meshdiff:";

/// How a 3-D field becomes one value per cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelSelect {
    /// A 2-D field: nothing to select.
    Plane,
    ColumnMax,
    ColumnMin,
    /// One level, counted from 1 the way the model counts them.
    Level(usize),
}

impl LevelSelect {
    fn label(self) -> Option<String> {
        match self {
            Self::Plane => None,
            Self::ColumnMax => Some("column maximum".to_string()),
            Self::ColumnMin => Some("column minimum".to_string()),
            Self::Level(k) => Some(format!("level {k}")),
        }
    }

    fn slug(self) -> Option<&'static str> {
        match self {
            Self::Plane => None,
            Self::ColumnMax => Some("colmax"),
            Self::ColumnMin => Some("colmin"),
            Self::Level(_) => Some("level"),
        }
    }
}

/// A parsed `mesh:` / `meshdiff:` product.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshProduct {
    /// The token as typed, for event lines.
    pub token: String,
    pub field: String,
    pub level: LevelSelect,
    pub log: bool,
    /// `@LO[..HI]`: the fill's range in the field's own units.  `LO` is also
    /// the mask: a cell below it takes the theme's empty fill.
    pub range: Option<(f64, Option<f64>)>,
    /// True for `meshdiff:`.
    pub difference: bool,
}

impl MeshProduct {
    /// `mesh_qi_colmax` / `meshdiff_qi_colmax`: the product half of the
    /// output filename, request-safe and stable across runs.
    pub fn slug(&self) -> String {
        let head = if self.difference { "meshdiff" } else { "mesh" };
        let mut slug = format!("{head}_{}", safe_component(&self.field));
        if let LevelSelect::Level(k) = self.level {
            slug.push_str(&format!("_level{k}"));
        } else if let Some(tag) = self.level.slug() {
            slug.push('_');
            slug.push_str(tag);
        }
        if self.log {
            slug.push_str("_log");
        }
        if let Some((lo, hi)) = self.range {
            slug.push_str(&format!("_from{}", safe_component(&format!("{lo}"))));
            if let Some(hi) = hi {
                slug.push_str(&format!("_to{}", safe_component(&format!("{hi}"))));
            }
        }
        slug
    }

    /// The panel headline: what was drawn, and how the column was reduced.
    pub fn title(&self, units: &str, labels: (&str, &str)) -> String {
        let mut title = self.field.clone();
        if let Some(reduction) = self.level.label() {
            title.push_str(": ");
            title.push_str(&reduction);
        }
        if self.log {
            title = format!("log10 {title}");
        }
        if self.difference {
            title.push_str(&format!(", {} minus {}", labels.0, labels.1));
        }
        if !units.trim().is_empty() {
            title.push_str(&format!(" [{}]", units.trim()));
        }
        title
    }
}

fn safe_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_underscore = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

/// Split a `--products` value into the rest of the spec and the mesh
/// products.  Group keywords carry no mesh products; a comma-separated list
/// may mix families.
pub fn split_product_spec(spec: &str) -> Result<(String, Vec<MeshProduct>), String> {
    let trimmed = spec.trim();
    if trimmed.eq_ignore_ascii_case("all")
        || ["direct", "derived", "heavy", "windowed"]
            .iter()
            .any(|group| trimmed.eq_ignore_ascii_case(group))
    {
        return Ok((trimmed.to_string(), Vec::new()));
    }
    let mut rest = Vec::new();
    let mut mesh = Vec::new();
    for token in trimmed.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if token.starts_with(PREFIX) || token.starts_with(DIFF_PREFIX) {
            mesh.push(parse_mesh_product(token)?);
        } else {
            rest.push(token.to_string());
        }
    }
    Ok((rest.join(","), mesh))
}

/// Parse one `mesh:` / `meshdiff:` token.
pub fn parse_mesh_product(token: &str) -> Result<MeshProduct, String> {
    let (difference, body) = match token.strip_prefix(DIFF_PREFIX) {
        Some(body) => (true, body),
        None => (
            false,
            token
                .strip_prefix(PREFIX)
                .ok_or_else(|| format!("'{token}' is not a mesh: or meshdiff: product"))?,
        ),
    };
    let (body, log) = match body.strip_suffix("~log") {
        Some(head) => (head, true),
        None => (body, false),
    };
    let (body, range) = match body.split_once('@') {
        None => (body, None),
        Some((head, tail)) => {
            let mut bounds = tail.split("..").map(str::trim);
            let lo_text = bounds.next().unwrap_or("");
            let lo: f64 = lo_text.parse().map_err(|_| {
                format!("'{token}': range floor '{lo_text}' is not a number")
            })?;
            let hi = match bounds.next() {
                None | Some("") => None,
                Some(text) => {
                    let hi: f64 = text.parse().map_err(|_| {
                        format!("'{token}': range top '{text}' is not a number")
                    })?;
                    if hi <= lo {
                        return Err(format!(
                            "'{token}': range top {hi} is not above its floor {lo}"
                        ));
                    }
                    Some(hi)
                }
            };
            if bounds.next().is_some() {
                return Err(format!(
                    "'{token}': a range is @LO or @LO..HI, not three values"
                ));
            }
            (head, Some((lo, hi)))
        }
    };
    let mut parts = body.split(':');
    let field = parts
        .next()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            format!("'{token}': name a history variable after the prefix, e.g. mesh:qi:colmax")
        })?
        .to_string();
    if field.contains(char::is_whitespace) || field.chars().any(char::is_control) {
        return Err(format!("'{token}': '{field}' is not a variable name"));
    }
    let mut level = LevelSelect::Plane;
    for modifier in parts {
        let modifier = modifier.trim();
        level = match modifier {
            "colmax" => LevelSelect::ColumnMax,
            "colmin" => LevelSelect::ColumnMin,
            other => match other.strip_prefix("level=") {
                Some(index) => {
                    let index: usize = index.parse().map_err(|_| {
                        format!("'{token}': level '{index}' is not a whole number")
                    })?;
                    if index == 0 {
                        return Err(format!(
                            "'{token}': levels are counted from 1, the way the model counts them"
                        ));
                    }
                    LevelSelect::Level(index)
                }
                None => {
                    return Err(format!(
                        "'{token}': '{other}' is not a mesh modifier; the modifiers are \
                         colmax, colmin and level=K"
                    ));
                }
            },
        };
    }
    Ok(MeshProduct {
        token: token.to_string(),
        field,
        level,
        log,
        range,
        difference,
    })
}

/// The cell geometry of one mesh, read from a grid file.
pub struct MeshGeometry {
    pub n_cells: usize,
    /// Cell centres, degrees.
    pub lat_cell_deg: Vec<f32>,
    pub lon_cell_deg: Vec<f32>,
    /// One boundary ring per DRAWN polygon, `(lat, lon)` in degrees,
    /// longitudes unwrapped so a ring never spans the whole map.  A mesh
    /// that wraps the globe carries a second ring for each cell the
    /// antimeridian cuts, shifted by 360 degrees, so both halves draw.
    pub rings_deg: Vec<Vec<(f64, f64)>>,
    /// Which cell each ring belongs to; the identity plus the seam copies.
    pub ring_cell: Vec<usize>,
    /// `bdyMaskCell` when the file carries one: non-zero marks a regional
    /// mesh's relaxation ring.
    pub boundary_mask: Option<Vec<i32>>,
    /// Cell-centre spacing in km, smallest and largest, from `areaCell`.
    pub spacing_km: (f64, f64),
    /// The mesh's own `nominalMinDc` stamp in km, when the file carries one.
    /// Preferred for the headline spacing: it is the number the mesh
    /// registry pairs a grid file to its static file on, so it is the
    /// number the mesh is KNOWN by, while the areaCell minimum is whatever
    /// the finest cell happened to come out at.
    pub nominal_min_dc_km: Option<f64>,
}

impl MeshGeometry {
    /// `3.75 km limited-area hex mesh` / `0.94-60.0 km global hex mesh`:
    /// derived from the file, never from a case name.
    pub fn description(&self) -> String {
        let (area_min, max) = self.spacing_km;
        let min = self.nominal_min_dc_km.unwrap_or(area_min);
        // Each end carries its OWN unit.  A variable-resolution global mesh
        // runs from sub-kilometre to a hundred kilometres, and one unit for
        // both ends printed the coarse end as 124805 m.
        // Both ends in one unit state it once ("3.75-7.13 km"); a mesh that
        // runs from sub-kilometre to a hundred kilometres needs both.
        let spacing = match (spacing_label(min), spacing_label(max)) {
            (low, _) if max <= min * 1.2 => low,
            (low, high) => match (low.rsplit_once(' '), high.rsplit_once(' ')) {
                (Some((low_value, low_unit)), Some((_, high_unit)))
                    if low_unit == high_unit =>
                {
                    format!("{low_value}-{high}")
                }
                _ => format!("{low}-{high}"),
            },
        };
        let extent = match self.boundary_mask.as_ref() {
            Some(mask) if mask.iter().any(|value| *value != 0) => "limited-area",
            _ => "global",
        };
        format!("{spacing} {extent} hex mesh")
    }

    /// `d01-3.75km`: the domain token in the output filename, the shape the
    /// render layout already places by.
    pub fn domain_slug(&self) -> String {
        let min = self.nominal_min_dc_km.unwrap_or(self.spacing_km.0);
        let token = if min >= 1.0 {
            format!("{}km", spacing_number(min, false))
        } else {
            format!("{}m", spacing_number(min, true))
        };
        format!("d01-{token}")
    }

    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let mut west = f64::INFINITY;
        let mut east = f64::NEG_INFINITY;
        let mut south = f64::INFINITY;
        let mut north = f64::NEG_INFINITY;
        for ring in &self.rings_deg {
            for &(lat, lon) in ring {
                west = west.min(lon);
                east = east.max(lon);
                south = south.min(lat);
                north = north.max(lat);
            }
        }
        if !west.is_finite() {
            return (-180.0, 180.0, -90.0, 90.0);
        }
        (west, east, south.max(-90.0), north.min(90.0))
    }
}

/// `3.75` / `938`: a spacing number without its unit, trailing zeros gone.
fn spacing_number(km: f64, metres: bool) -> String {
    if metres {
        return format!("{:.0}", km * 1000.0);
    }
    let text = format!("{km:.2}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// `938 m` / `3.75 km` / `125 km`: a spacing with the unit that suits it.
fn spacing_label(km: f64) -> String {
    if km < 1.0 {
        format!("{} m", spacing_number(km, true))
    } else if km >= 100.0 {
        format!("{:.0} km", km)
    } else {
        format!("{} km", spacing_number(km, false))
    }
}

fn read_indices(file: &netcrust::File, name: &str) -> Result<Vec<f64>, String> {
    file.read_f64(name)
        .map_err(|err| format!("{name}: {err}"))
}

/// Read the cell boundaries out of an MPAS-shaped grid file.
pub fn read_mesh_geometry(path: &Path) -> Result<MeshGeometry, String> {
    let file = netcrust::File::open(path)
        .map_err(|err| format!("{} is not a netCDF file this reader can open: {err}", path.display()))?;
    let n_cells = file
        .dimension("nCells")
        .ok_or_else(|| {
            format!(
                "{} has no nCells dimension, so it is not the grid file for a mesh forecast",
                path.display()
            )
        })?
        .len();
    let max_edges = file
        .dimension("maxEdges")
        .ok_or_else(|| format!("{} has no maxEdges dimension", path.display()))?
        .len();

    let lat_cell = read_indices(&file, "latCell")?;
    let lon_cell = read_indices(&file, "lonCell")?;
    let lat_vertex = read_indices(&file, "latVertex")?;
    let lon_vertex = read_indices(&file, "lonVertex")?;
    let n_edges_on_cell = read_indices(&file, "nEdgesOnCell")?;
    let vertices_on_cell = read_indices(&file, "verticesOnCell")?;
    if lat_cell.len() != n_cells || lon_cell.len() != n_cells {
        return Err(format!(
            "{}: latCell/lonCell carry {}/{} values for {n_cells} cells",
            path.display(),
            lat_cell.len(),
            lon_cell.len()
        ));
    }
    if vertices_on_cell.len() != n_cells * max_edges {
        return Err(format!(
            "{}: verticesOnCell carries {} values, not {n_cells} x {max_edges}",
            path.display(),
            vertices_on_cell.len()
        ));
    }
    let boundary_mask = file
        .read_f64("bdyMaskCell")
        .ok()
        .filter(|values| values.len() == n_cells)
        .map(|values| values.iter().map(|value| *value as i32).collect::<Vec<_>>());

    // areaCell is a unit-sphere area on every published mesh (the file's
    // sphere_radius says which); the cell-centre spacing of a hexagon of
    // area A is sqrt(2A/sqrt(3)).
    let sphere_radius = file
        .attribute("sphere_radius")
        .and_then(|attribute| attribute.as_f64())
        .filter(|radius| radius.is_finite() && *radius > 0.0)
        .unwrap_or(1.0);
    let spacing_km = match file.read_f64("areaCell") {
        Ok(areas) if areas.len() == n_cells => {
            let scale = if sphere_radius > 1.5 {
                1.0
            } else {
                EARTH_RADIUS_M * EARTH_RADIUS_M
            };
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for area in &areas {
                let metres = (2.0 * area * scale / 3.0_f64.sqrt()).max(0.0).sqrt();
                if metres.is_finite() && metres > 0.0 {
                    min = min.min(metres / 1000.0);
                    max = max.max(metres / 1000.0);
                }
            }
            if min.is_finite() { (min, max) } else { (0.0, 0.0) }
        }
        _ => (0.0, 0.0),
    };

    let to_deg = |radians: f64| radians.to_degrees();
    let lat_cell_deg: Vec<f32> = lat_cell.iter().map(|v| to_deg(*v) as f32).collect();
    let lon_cell_deg: Vec<f32> = lon_cell
        .iter()
        .map(|v| normalize_longitude(to_deg(*v)) as f32)
        .collect();

    let nominal_min_dc_km = file
        .read_f64("nominalMinDc")
        .ok()
        .and_then(|values| values.first().copied())
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| {
            let unit = value / sphere_radius;
            unit * EARTH_RADIUS_M / 1000.0
        });

    let mut rings_deg = Vec::with_capacity(n_cells);
    let mut ring_cell: Vec<usize> = Vec::with_capacity(n_cells);
    let mut seam: Vec<(usize, Vec<(f64, f64)>)> = Vec::new();
    for cell in 0..n_cells {
        let count = (n_edges_on_cell.get(cell).copied().unwrap_or(0.0) as usize).min(max_edges);
        let centre_lon = f64::from(lon_cell_deg[cell]);
        let mut ring = Vec::with_capacity(count);
        for slot in 0..count {
            // MPAS connectivity is 1-based and 0 marks an absent neighbour.
            let raw = vertices_on_cell[cell * max_edges + slot];
            if raw < 1.0 {
                continue;
            }
            let index = raw as usize - 1;
            if index >= lat_vertex.len() {
                continue;
            }
            let lat = to_deg(lat_vertex[index]);
            // Unwrap toward the cell's own centre: a cell that straddles the
            // date line otherwise draws a ring across the entire map, which
            // is a bar of colour through every other cell it crosses.
            let lon = unwrap_longitude(to_deg(lon_vertex[index]), centre_lon);
            ring.push((lat, lon));
        }
        // A cell the antimeridian cuts has vertices past +/-180 after the
        // unwrap.  On a frame that wraps the globe the far half of that cell
        // is a real part of the map, so it is drawn a second time shifted by
        // a full turn; on a regional frame the copy falls outside the clip
        // and costs one bounding-box test.
        let past_east = ring.iter().any(|(_, lon)| *lon > 180.0);
        let past_west = ring.iter().any(|(_, lon)| *lon < -180.0);
        if past_east {
            seam.push((
                cell,
                ring.iter().map(|(lat, lon)| (*lat, lon - 360.0)).collect(),
            ));
        } else if past_west {
            seam.push((
                cell,
                ring.iter().map(|(lat, lon)| (*lat, lon + 360.0)).collect(),
            ));
        }
        rings_deg.push(ring);
        ring_cell.push(cell);
    }
    for (cell, ring) in seam {
        rings_deg.push(ring);
        ring_cell.push(cell);
    }

    Ok(MeshGeometry {
        n_cells,
        lat_cell_deg,
        lon_cell_deg,
        rings_deg,
        ring_cell,
        boundary_mask,
        spacing_km,
        nominal_min_dc_km,
    })
}

fn normalize_longitude(value: f64) -> f64 {
    let mut lon = value % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    }
    if lon < -180.0 {
        lon += 360.0;
    }
    lon
}

fn unwrap_longitude(value: f64, reference: f64) -> f64 {
    let mut lon = normalize_longitude(value);
    while lon - reference > 180.0 {
        lon -= 360.0;
    }
    while reference - lon > 180.0 {
        lon += 360.0;
    }
    lon
}

/// One history frame's values for one product, plus what the file called
/// them.
pub struct MeshField {
    pub values: Vec<f64>,
    pub units: String,
    pub valid_label: String,
}

/// Read one `nCells`-shaped field and reduce it to one value per cell.
pub fn read_mesh_field(
    path: &Path,
    product: &MeshProduct,
    n_cells: usize,
) -> Result<MeshField, String> {
    let file = netcrust::File::open(path)
        .map_err(|err| format!("{}: {err}", path.display()))?;
    let variable = file.variable(&product.field).ok_or_else(|| {
        format!(
            "{} has no variable '{}'; a mesh: product names a variable in the history file \
             exactly as the file spells it",
            path.display(),
            product.field
        )
    })?;
    let has_time = variable
        .dimensions()
        .first()
        .map(|d| d.name() == "Time")
        .unwrap_or(false);
    let shape: Vec<usize> = variable.shape();
    let array = file
        .read_array_f64(&product.field)
        .map_err(|err| format!("{}: {}: {err}", path.display(), product.field))?;
    let all = array.into_values();
    let trailing: usize = if has_time {
        shape[1..].iter().product()
    } else {
        shape.iter().product()
    };
    if all.len() < trailing || trailing == 0 {
        return Err(format!(
            "{}: {} read back {} value(s) for a {trailing}-value record",
            path.display(),
            product.field,
            all.len()
        ));
    }
    let slice = &all[all.len() - trailing..];
    if slice.len() % n_cells != 0 {
        return Err(format!(
            "{}: {} has {} value(s), not a whole number of levels over {n_cells} cells; the \
             grid file and the history frame describe different meshes",
            path.display(),
            product.field,
            slice.len()
        ));
    }
    let levels = slice.len() / n_cells;
    let values = match (levels, product.level) {
        (1, LevelSelect::Level(1)) | (1, LevelSelect::Plane) => slice.to_vec(),
        (1, LevelSelect::ColumnMax) | (1, LevelSelect::ColumnMin) => slice.to_vec(),
        (1, LevelSelect::Level(k)) => {
            return Err(format!(
                "'{}': {} carries one level; level {k} is past it",
                product.token, product.field
            ));
        }
        (_, LevelSelect::Plane) => {
            return Err(format!(
                "'{}': {} carries {levels} levels; add :colmax, :colmin or :level=K",
                product.token, product.field
            ));
        }
        (_, LevelSelect::ColumnMax) => (0..n_cells)
            .map(|cell| {
                let column = &slice[cell * levels..(cell + 1) * levels];
                column.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            })
            .collect(),
        (_, LevelSelect::ColumnMin) => (0..n_cells)
            .map(|cell| {
                let column = &slice[cell * levels..(cell + 1) * levels];
                column.iter().copied().fold(f64::INFINITY, f64::min)
            })
            .collect(),
        (_, LevelSelect::Level(k)) => {
            if k > levels {
                return Err(format!(
                    "'{}': level {k} is past the {levels} levels {} carries",
                    product.token, product.field
                ));
            }
            (0..n_cells).map(|cell| slice[cell * levels + (k - 1)]).collect()
        }
    };
    let units = variable
        .attribute("units")
        .and_then(|attribute| attribute.as_string().map(str::to_string))
        .unwrap_or_default();
    let valid_label = read_valid_label(&file, path);
    Ok(MeshField {
        values,
        units,
        valid_label,
    })
}

/// The frame's valid time: the file's own `xtime` when it has one, else the
/// stamp in its name.  Never invented -- an unreadable time is reported as
/// unreadable rather than stamped with the clock.
fn read_valid_label(file: &netcrust::File, path: &Path) -> String {
    if let Ok(times) = file.read_strings("xtime") {
        if let Some(stamp) = times.first().map(|value| value.trim().to_string()) {
            if stamp.len() >= 19 {
                return stamp[..19].to_string();
            }
        }
    }
    stamp_from_name(path).unwrap_or_else(|| "unreadable".to_string())
}

/// `YYYY-MM-DD_HH.MM.SS`, `:` or `T` separators, anywhere in the name.
pub fn stamp_from_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let chars: Vec<char> = name.chars().collect();
    if chars.len() < 19 {
        return None;
    }
    for start in 0..=chars.len() - 19 {
        let window: String = chars[start..start + 19].iter().collect();
        let b: Vec<char> = window.chars().collect();
        let digit = |i: usize| b[i].is_ascii_digit();
        let ok = (0..4).all(digit)
            && b[4] == '-'
            && digit(5)
            && digit(6)
            && b[7] == '-'
            && digit(8)
            && digit(9)
            && matches!(b[10], '_' | 'T')
            && digit(11)
            && digit(12)
            && matches!(b[13], '.' | ':')
            && digit(14)
            && digit(15)
            && matches!(b[16], '.' | ':')
            && digit(17)
            && digit(18);
        if ok {
            return Some(window);
        }
    }
    None
}

/// `2026-08-12 09:00Z` from a `2026-08-12_09.00.00` stamp.
pub fn caption_time(stamp: &str) -> String {
    if stamp.len() < 16 {
        return stamp.to_string();
    }
    let bytes: Vec<char> = stamp.chars().collect();
    format!(
        "{} {}:{}Z",
        bytes[..10].iter().collect::<String>(),
        bytes[11..13].iter().collect::<String>(),
        bytes[14..16].iter().collect::<String>()
    )
}

/// Everything one render pass needs that is not the product itself.
pub struct MeshRenderConfig<'a> {
    pub inputs: &'a [PathBuf],
    pub out_dir: &'a Path,
    pub grid: &'a MeshGeometry,
    /// A directory of the other leg's frames, or a single file.
    pub reference: Option<&'a Path>,
    pub labels: (String, String),
    pub width: u32,
    pub height: u32,
    pub source_label: String,
    pub theme: &'a RenderTheme,
    /// Ordinal frame index across the inputs, or all.
    pub frame: Option<usize>,
    /// `--mesh-bounds W,E,S,N`: the frame to draw, when the caller wants a
    /// window rather than the whole mesh.  Cells outside it are clipped by
    /// the map rectangle, so the mesh is never re-read for a zoom.
    pub bounds: Option<(f64, f64, f64, f64)>,
    /// Extra caption fields the caller set; the mesh row and the valid time
    /// are filled from the mesh and the frame when they are absent.
    pub footer: rustwx_render::FooterFields,
}

/// One product on one frame: the slug and where it went, or why not.
pub struct MeshOutcome {
    pub slug: String,
    pub result: Result<PathBuf, String>,
    pub render_ms: u128,
    pub cells: usize,
}

/// Resolve the reference frame for `input`.
fn reference_for(reference: &Path, input: &Path, single: bool) -> Result<PathBuf, String> {
    if reference.is_dir() {
        let name = input
            .file_name()
            .ok_or_else(|| format!("{} has no file name", input.display()))?;
        let candidate = reference.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
        // The two legs may name their frames differently; fall back to the
        // one file in the directory carrying the same timestamp.
        if let Some(stamp) = stamp_from_name(input) {
            if let Ok(entries) = std::fs::read_dir(reference) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() && stamp_from_name(&path).as_deref() == Some(stamp.as_str()) {
                        return Ok(path);
                    }
                }
            }
        }
        return Err(format!(
            "--mesh-reference {} holds no frame matching {}",
            reference.display(),
            input.display()
        ));
    }
    if !single {
        return Err(format!(
            "--mesh-reference {} is one file but several frames are being rendered; point it at \
             the other leg's frame DIRECTORY",
            reference.display()
        ));
    }
    Ok(reference.to_path_buf())
}

/// Render every mesh product over every selected frame.
pub fn render_mesh_products(
    products: &[MeshProduct],
    config: &MeshRenderConfig<'_>,
    mut emit: impl FnMut(MeshOutcome),
) -> Result<(usize, usize), String> {
    if products.iter().any(|product| product.difference) && config.reference.is_none() {
        return Err(
            "meshdiff: products need the other leg: --mesh-reference DIR (or FILE for one frame)"
                .to_string(),
        );
    }
    let frames: Vec<&PathBuf> = match config.frame {
        None => config.inputs.iter().collect(),
        Some(index) => vec![config.inputs.get(index).ok_or_else(|| {
            format!(
                "--frames {index} out of range; the inputs hold {} frame(s)",
                config.inputs.len()
            )
        })?],
    };
    if frames.is_empty() {
        return Err("no history frames were given for the mesh: family".to_string());
    }
    std::fs::create_dir_all(config.out_dir)
        .map_err(|err| format!("create {}: {err}", config.out_dir.display()))?;

    let bounds = config.bounds.unwrap_or_else(|| config.grid.bounds());
    let target_ratio = map_frame_aspect_ratio_for_mode(
        ProductVisualMode::FilledMeteorology,
        config.width,
        config.height,
        true,
        true,
    );
    let map_options = ProjectedMapBuildOptions::from_bounds(bounds, target_ratio);
    let projected = build_projected_map_with_options(
        &config.grid.lat_cell_deg,
        &config.grid.lon_cell_deg,
        &map_options,
    )
    .map_err(|err| format!("project the mesh: {err}"))?;

    // Every ring vertex through the SAME projector the basemap was built
    // with: a second, hand-rolled projection is how a mesh ends up drawn on
    // a map of somewhere else.
    let mut flat: Vec<(f64, f64)> = Vec::new();
    let mut ring_starts: Vec<(usize, usize)> = Vec::with_capacity(config.grid.rings_deg.len());
    for ring in &config.grid.rings_deg {
        ring_starts.push((flat.len(), ring.len()));
        flat.extend(ring.iter().copied());
    }
    let projected_points = project_geographic_points_with_options(
        &config.grid.lat_cell_deg,
        &config.grid.lon_cell_deg,
        &map_options,
        &flat,
    )
    .map_err(|err| format!("project the mesh cells: {err}"))?;
    let resolved = resolved_projection_for_options(
        &config.grid.lat_cell_deg,
        &config.grid.lon_cell_deg,
        &map_options.domain,
    )
    .map_err(|err| format!("resolve the mesh projection: {err}"))?;

    // The renderer pairs `projected_domain` with the FIELD's shape, and this
    // family's field is a 2x2 placeholder the raster never draws.  So the
    // domain is the frame's four corners through the same projector the
    // cells went through -- the extent, which is what the map is actually
    // drawn in, still comes from the full-mesh build above.
    let corner_points = [
        (bounds.2, bounds.0),
        (bounds.2, bounds.1),
        (bounds.3, bounds.0),
        (bounds.3, bounds.1),
    ];
    let corners = project_geographic_points_with_options(
        &config.grid.lat_cell_deg,
        &config.grid.lon_cell_deg,
        &map_options,
        &corner_points,
    )
    .map_err(|err| format!("project the frame corners: {err}"))?;
    let placeholder_domain = ProjectedDomain {
        x: corners.iter().map(|(x, _)| *x).collect(),
        y: corners.iter().map(|(_, y)| *y).collect(),
        extent: projected.extent.clone(),
    };

    let single = frames.len() == 1;
    let mut rendered = 0usize;
    let mut failed = 0usize;

    for input in frames {
        for product in products {
            let started = std::time::Instant::now();
            let slug = product.slug();
            let outcome = (|| -> Result<PathBuf, String> {
                let field = read_mesh_field(input, product, config.grid.n_cells)?;
                let mut values = field.values;
                if product.difference {
                    let reference = reference_for(
                        config.reference.expect("checked above"),
                        input,
                        single,
                    )?;
                    let other = read_mesh_field(&reference, product, config.grid.n_cells)?;
                    for (value, minus) in values.iter_mut().zip(other.values.iter()) {
                        *value -= *minus;
                    }
                }
                if product.log {
                    for value in values.iter_mut() {
                        *value = if value.is_finite() && *value > 0.0 {
                            value.log10()
                        } else {
                            f64::NAN
                        };
                    }
                }
                render_one(
                    product,
                    config,
                    &values,
                    &field.units,
                    &field.valid_label,
                    &projected_points,
                    &ring_starts,
                    &projected,
                    &placeholder_domain,
                    &resolved,
                    bounds,
                    &slug,
                )
            })();
            match &outcome {
                Ok(_) => rendered += 1,
                Err(_) => failed += 1,
            }
            emit(MeshOutcome {
                slug,
                result: outcome,
                render_ms: started.elapsed().as_millis(),
                cells: config.grid.n_cells,
            });
        }
    }
    Ok((rendered, failed))
}

#[allow(clippy::too_many_arguments)]
fn render_one(
    product: &MeshProduct,
    config: &MeshRenderConfig<'_>,
    values: &[f64],
    units: &str,
    valid_label: &str,
    projected_points: &[(f64, f64)],
    ring_starts: &[(usize, usize)],
    projected: &rustwx_render::ProjectedMap,
    placeholder_domain: &ProjectedDomain,
    resolved: &rustwx_render::georeference::ResolvedProjection,
    bounds: (f64, f64, f64, f64),
    slug: &str,
) -> Result<PathBuf, String> {
    let boundary = config.grid.boundary_mask.as_deref();
    let mut finite_min = f64::INFINITY;
    let mut finite_max = f64::NEG_INFINITY;
    let mut cells = Vec::with_capacity(ring_starts.len());
    for (ring_index, (start, count)) in ring_starts.iter().enumerate() {
        let index = config.grid.ring_cell[ring_index];
        let ring: Vec<(f64, f64)> = projected_points[*start..*start + *count].to_vec();
        // A regional mesh's outer relaxation ring is not a forecast: it is
        // the lateral boundary being blended in.  Drawing its values as if
        // they were the model's answer is the one thing a limited-area panel
        // must not do, so those cells take the empty fill and stay visible as
        // the mesh they are.
        let on_boundary = boundary
            .map(|mask| mask.get(index).copied().unwrap_or(0) != 0)
            .unwrap_or(false);
        let value = values.get(index).copied().filter(|value| value.is_finite());
        if let Some(value) = value {
            if !on_boundary {
                finite_min = finite_min.min(value);
                finite_max = finite_max.max(value);
            }
        }
        cells.push(MeshCell {
            ring,
            value: if on_boundary { None } else { value },
        });
    }
    if !finite_min.is_finite() || !finite_max.is_finite() {
        return Err(format!(
            "'{}': every cell of {} is masked or not finite on this frame",
            product.token, product.field
        ));
    }

    // A colorbar whose every tick reads `0` measures nothing.  The tick
    // formatter carries one decimal, and a hydrometeor mixing ratio is
    // 1e-4 kg kg-1, so the whole ladder printed `0` and the panel reported
    // no number at all.  The remedy is generic and driven by the RANGE, not
    // by any variable's name: the values move onto a power-of-a-thousand
    // scale that puts the largest of them in 1-1000, and the decade is
    // stated in the units so the reader can put it back.
    let exponent = display_exponent(finite_min.abs().max(finite_max.abs()));
    let factor = 10f64.powi(-exponent);
    if exponent != 0 {
        for cell in cells.iter_mut() {
            cell.value = cell.value.map(|value| value * factor);
        }
        finite_min *= factor;
        finite_max *= factor;
    }
    let display_units = scaled_units(units, exponent);

    // A named range is in the FIELD's units, so it takes the same decade
    // shift the values did; and its floor is the mask, so a cell below it
    // shows the mesh rather than the ramp's bottom colour.
    if let Some((lo, hi)) = product.range {
        let lo = lo * factor;
        let hi = hi.map(|hi| hi * factor);
        for cell in cells.iter_mut() {
            if let Some(value) = cell.value {
                if value < lo {
                    cell.value = None;
                }
            }
        }
        finite_min = lo;
        finite_max = hi.unwrap_or(finite_max.max(lo * 1.000_001 + 1.0e-9));
    }

    let scale = if product.difference {
        diverging_scale(finite_min.abs().max(finite_max.abs()))
    } else {
        let style = rustwx_products::viewer::generic_style_for_store_variable(
            &product.field,
            &display_units,
            Some((finite_min as f32, finite_max as f32)),
        );
        style.scale
    };

    // The renderer's raster pass needs a structured field; the mesh carries
    // every value, so the placeholder is a 2x2 of NaN at the frame's corners.
    // NaN maps to transparent, so nothing of it reaches a pixel.
    let placeholder_grid = CoreLatLonGrid::new(
        CoreGridShape::new(2, 2).map_err(|err| format!("placeholder grid: {err}"))?,
        vec![
            bounds.2 as f32,
            bounds.2 as f32,
            bounds.3 as f32,
            bounds.3 as f32,
        ],
        vec![
            bounds.0 as f32,
            bounds.1 as f32,
            bounds.0 as f32,
            bounds.1 as f32,
        ],
    )
    .map_err(|err| format!("placeholder grid: {err}"))?;
    let field = CoreField2D::new(
        CoreProductKey::named(slug.to_string()),
        display_units.clone(),
        placeholder_grid,
        vec![f32::NAN; 4],
    )
    .map_err(|err| format!("placeholder field: {err}"))?;

    let labels = (config.labels.0.as_str(), config.labels.1.as_str());
    let mut request = MapRenderRequest::from_core_field(field, scale);
    request.width = config.width;
    request.height = config.height;
    request.visual_mode = ProductVisualMode::FilledMeteorology;
    request.title = Some(product.title(&display_units, labels));
    request.cbar_tick_step = None;
    request.subtitle_left = Some(format!("valid {}", caption_time(valid_label)));
    request.subtitle_right = Some(
        config
            .theme
            .source_label
            .clone()
            .unwrap_or_else(|| format!("source: {}", config.source_label)),
    );
    request.mesh_cells = Some(MeshCellsLayer::new(cells));
    // The domain frame is derived from the RASTER, and this family's raster
    // is a placeholder; a frame drawn from it would outline nothing.  The
    // mesh's own edges are its outline.
    request.domain_frame = None;
    request.projected_domain = Some(placeholder_domain.clone());
    request.projected_lines = projected.lines.clone();
    request.projected_polygons = projected.polygons.clone();
    request.resolved_projection = Some(resolved.clone());
    request.geographic_bounds = Some(bounds);

    let mut footer = config.footer.clone();
    if footer.product_title.is_none() {
        footer.product_title = request.title.clone();
    }
    if footer.valid_time.is_none() {
        footer.valid_time = Some(caption_time(valid_label));
    }
    if footer.mesh_or_grid.is_none() {
        footer.mesh_or_grid = Some(config.grid.description());
    }
    if product.difference && footer.leg.is_none() {
        footer.leg = Some(format!("{} minus {}", labels.0, labels.1));
    }
    rustwx_render::set_footer_fields(footer);

    let (date, hour, stamp) = filename_stamp(valid_label);
    let name = format!(
        "rustwx_wrf_{date}_{hour}z_f000_{}_{}_valid_{stamp}.png",
        config.grid.domain_slug(),
        slug
    );
    let output = config.out_dir.join(name);
    save_png_profile_with_options(&request, &output, &PngWriteOptions::default())
        .map_err(|err| format!("render {}: {err}", output.display()))?;
    Ok(output)
}

/// The power of a thousand that puts `max_abs` in 1-1000, or 0 when it is
/// already there.  Powers of a thousand, not of ten, so the decade in the
/// units is one a reader recognises (1e-3, 1e-6, 1e3) rather than an
/// arbitrary shift.
pub fn display_exponent(max_abs: f64) -> i32 {
    if !max_abs.is_finite() || max_abs <= 0.0 {
        return 0;
    }
    if (1.0..1000.0).contains(&max_abs) {
        return 0;
    }
    ((max_abs.log10() / 3.0).floor() * 3.0) as i32
}

/// `1e-6 kg kg^{-1}`: the units with the decade the values were moved onto.
pub fn scaled_units(units: &str, exponent: i32) -> String {
    if exponent == 0 {
        return units.to_string();
    }
    let units = units.trim();
    if units.is_empty() {
        format!("1e{exponent}")
    } else {
        format!("1e{exponent} {units}")
    }
}

/// A zero-centred diverging scale over `max_abs`, on the theme's diverging
/// ramp when it names one.
///
/// The one ramp a difference panel invents, and therefore one of the two a
/// theme may replace wholesale.  Symmetric by construction: a difference
/// plot whose zero is not the ramp's midpoint reports a sign that is not
/// there.
pub fn diverging_scale(max_abs: f64) -> ColorScale {
    const BANDS: usize = 16;
    const FALLBACK: [[u8; 3]; 9] = [
        [5, 48, 97],
        [33, 102, 172],
        [67, 147, 195],
        [146, 197, 222],
        [247, 247, 247],
        [244, 165, 130],
        [214, 96, 77],
        [178, 24, 43],
        [103, 0, 31],
    ];
    let span = nice_ceiling(max_abs);
    let levels: Vec<f64> = (0..=BANDS)
        .map(|index| -span + 2.0 * span * index as f64 / BANDS as f64)
        .collect();
    let colors = rustwx_render::active_theme()
        .diverging_colors(BANDS)
        .unwrap_or_else(|| {
            rustwx_render::theme::resample(
                &FALLBACK
                    .iter()
                    .map(|[r, g, b]| rustwx_render::Rgba::with_alpha(*r, *g, *b, 255))
                    .collect::<Vec<_>>(),
                BANDS,
            )
        });
    ColorScale::Discrete(DiscreteColorScale {
        levels,
        colors,
        extend: ExtendMode::Both,
        mask_below: None,
    })
}

/// `(YYYYMMDD, HH, YYYYMMDD_HHMMSS)` from a valid-time label, for the output
/// filename the render layout places by.  A label that could not be read
/// keeps its own text rather than being stamped with an invented time.
fn filename_stamp(valid_label: &str) -> (String, String, String) {
    let digits: String = valid_label.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() < 14 {
        let safe = safe_component(valid_label);
        return (safe.clone(), "00".to_string(), safe);
    }
    (
        digits[..8].to_string(),
        digits[8..10].to_string(),
        format!("{}_{}", &digits[..8], &digits[8..14]),
    )
}

/// The next 1-2-5 step at or above `value`, so two frames of one case share
/// a colorbar range instead of each getting its own arbitrary maximum.
fn nice_ceiling(value: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 {
        return 1.0;
    }
    let exponent = value.log10().floor();
    let base = 10f64.powf(exponent);
    for step in [1.0, 2.0, 5.0, 10.0] {
        if value <= step * base * 1.000_001 {
            return step * base;
        }
    }
    10.0 * base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grammar_parses_fields_reductions_levels_and_the_log_modifier() {
        let plain = parse_mesh_product("mesh:qi").expect("parses");
        assert_eq!(plain.field, "qi");
        assert_eq!(plain.level, LevelSelect::Plane);
        assert!(!plain.difference);
        assert!(!plain.log);
        assert_eq!(plain.slug(), "mesh_qi");

        let colmax = parse_mesh_product("mesh:qi:colmax").expect("parses");
        assert_eq!(colmax.level, LevelSelect::ColumnMax);
        assert_eq!(colmax.slug(), "mesh_qi_colmax");

        let ranged = parse_mesh_product("mesh:refl10cm:colmax@5..60").expect("parses");
        assert_eq!(ranged.range, Some((5.0, Some(60.0))));
        assert_eq!(ranged.slug(), "mesh_refl10cm_colmax_from5_to60");
        let floor = parse_mesh_product("mesh:refl10cm:colmax@5").expect("parses");
        assert_eq!(floor.range, Some((5.0, None)));
        let err = parse_mesh_product("mesh:refl10cm@60..5").expect_err("inverted");
        assert!(err.contains("not above its floor"), "{err}");
        // A comma would be read as the --products separator, so the range
        // never uses one and a token carrying one is refused by name.
        let err = parse_mesh_product("mesh:refl10cm@5,60").expect_err("comma");
        assert!(err.contains("not a number"), "{err}");

        let level = parse_mesh_product("meshdiff:qcloud:level=12~log").expect("parses");
        assert!(level.difference);
        assert!(level.log);
        assert_eq!(level.level, LevelSelect::Level(12));
        assert_eq!(level.slug(), "meshdiff_qcloud_level12_log");
        assert_eq!(
            level.title("kg kg-1", ("TREATMENT", "CONTROL")),
            "log10 qcloud: level 12, TREATMENT minus CONTROL [kg kg-1]"
        );
    }

    #[test]
    fn a_bad_modifier_and_a_zero_level_are_refused_by_name() {
        let err = parse_mesh_product("mesh:qi:colavg").expect_err("unknown modifier");
        assert!(err.contains("colavg"), "{err}");
        assert!(err.contains("colmax"), "{err}");
        let err = parse_mesh_product("mesh:qi:level=0").expect_err("zero level");
        assert!(err.contains("counted from 1"), "{err}");
        let err = parse_mesh_product("var:qi").expect_err("not this family");
        assert!(err.contains("mesh:"), "{err}");
    }

    #[test]
    fn the_spec_splitter_keeps_the_other_families_intact() {
        let (rest, mesh) = split_product_spec(
            "composite_reflectivity,mesh:qi:colmax,var:wrf_qice_colmax,meshdiff:qc:colmax",
        )
        .expect("splits");
        assert_eq!(rest, "composite_reflectivity,var:wrf_qice_colmax");
        assert_eq!(mesh.len(), 2);
        assert!(mesh[1].difference);
        let (rest, mesh) = split_product_spec("all").expect("splits");
        assert_eq!(rest, "all");
        assert!(mesh.is_empty());
    }

    #[test]
    fn longitudes_unwrap_toward_their_own_cell_instead_of_crossing_the_map() {
        assert_eq!(unwrap_longitude(-179.0, 179.0), 181.0);
        assert_eq!(unwrap_longitude(179.0, -179.0), -181.0);
        assert_eq!(unwrap_longitude(10.0, 12.0), 10.0);
        assert_eq!(normalize_longitude(190.0), -170.0);
    }

    #[test]
    fn the_diverging_scale_is_symmetric_about_zero_on_a_nice_step() {
        let ColorScale::Discrete(scale) = diverging_scale(0.00083) else {
            panic!("a discrete scale");
        };
        assert_eq!(scale.levels.len(), 17);
        assert!((scale.levels[0] + 0.001).abs() < 1.0e-12, "{:?}", scale.levels[0]);
        assert!((scale.levels[16] - 0.001).abs() < 1.0e-12);
        let middle = scale.levels[8];
        assert!(middle.abs() < 1.0e-12, "the midpoint is zero, not {middle}");
        assert_eq!(nice_ceiling(0.0), 1.0);
        assert_eq!(nice_ceiling(3.0), 5.0);
        assert_eq!(nice_ceiling(12.0), 20.0);
    }

    #[test]
    fn tiny_and_huge_ranges_move_onto_a_decade_the_colorbar_can_print() {
        // A hydrometeor mixing ratio: every tick printed 0 before this.
        assert_eq!(display_exponent(8.3e-4), -6);
        assert!((8.3e-4 * 10f64.powi(6) - 830.0).abs() < 1.0e-9);
        assert_eq!(scaled_units("kg kg^{-1}", -6), "1e-6 kg kg^{-1}");
        // Reflectivity and a cloud-water bank are already readable.
        assert_eq!(display_exponent(45.0), 0);
        assert_eq!(display_exponent(2.0), 0);
        assert_eq!(scaled_units("dBZ", 0), "dBZ");
        // A big one moves the other way.
        assert_eq!(display_exponent(4.2e7), 6);
        assert_eq!(display_exponent(0.0), 0);
        assert_eq!(scaled_units("", -3), "1e-3");
    }

    #[test]
    fn a_timestamp_is_read_from_a_frame_name_and_spelled_for_a_caption() {
        let stamp = stamp_from_name(Path::new("cuda-history.2026-08-12_09.00.00.nc"))
            .expect("a stamp");
        assert_eq!(stamp, "2026-08-12_09.00.00");
        assert_eq!(caption_time(&stamp), "2026-08-12 09:00Z");
        assert!(stamp_from_name(Path::new("history.nc")).is_none());
        assert_eq!(
            filename_stamp("2026-08-12_09.00.00"),
            (
                "20260812".to_string(),
                "09".to_string(),
                "20260812_090000".to_string()
            )
        );
        assert_eq!(
            filename_stamp("unreadable"),
            (
                "unreadable".to_string(),
                "00".to_string(),
                "unreadable".to_string()
            )
        );
    }
}
