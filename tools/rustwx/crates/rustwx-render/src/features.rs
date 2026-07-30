use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use shapefile::{Shape, ShapeReader};

use crate::color::Rgba;
use crate::presentation::{LineworkRole, PolygonRole};

pub type LonLatLine = Vec<(f64, f64)>;

/// Basemap presentation preset.
///
/// - `Filled`: cool-beige land + pale-blue ocean (weathermodels.com look).
/// - `White`: NWS-style white land + white ocean with US county lines drawn on
///   top; heavier state borders make the political grid read well.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BasemapStyle {
    Filled,
    White,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BasemapDetail {
    Regional,
    Broad,
    Global,
}

#[derive(Clone, Debug)]
pub struct StyledLonLatLayer {
    pub lines: Vec<LonLatLine>,
    pub color: Rgba,
    pub width: u32,
    pub role: LineworkRole,
}

/// A single closed polygon in lon/lat with optional holes. Outer ring first,
/// subsequent rings punch holes.
pub type LonLatPolygon = Vec<Vec<(f64, f64)>>;

#[derive(Clone, Debug)]
pub struct StyledLonLatPolygonLayer {
    pub polygons: Vec<LonLatPolygon>,
    pub color: Rgba,
    pub role: PolygonRole,
}

pub fn cartopy_natural_earth_root() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)?;
    let root = home
        .join(".local")
        .join("share")
        .join("cartopy")
        .join("shapefiles")
        .join("natural_earth");
    root.exists().then_some(root)
}

pub fn checked_in_natural_earth_110m_root() -> Option<PathBuf> {
    workspace_basemap_subdir("natural_earth_110m")
}

/// Checked-in 10m (high-res) Natural Earth assets inside the repo. Preferred
/// over the 110m set for CONUS-scale maps because coastlines stay crisp when
/// the frame covers only the Lower 48.
pub fn checked_in_natural_earth_10m_root() -> Option<PathBuf> {
    workspace_basemap_subdir("natural_earth_10m")
}

/// Checked-in US Census Cartographic Boundary counties at 1:5,000,000 — the
/// detail level NWS-style CONUS maps draw county borders at. Public-domain
/// TIGER/Line data (`cb_2023_us_county_5m.*`).
pub fn checked_in_us_counties_5m_root() -> Option<PathBuf> {
    workspace_basemap_subdir("us_counties_5m")
}

fn workspace_basemap_subdir(name: &str) -> Option<PathBuf> {
    basemap_root_candidates()
        .into_iter()
        .map(|root| root.join(name))
        .find(|path| path.exists())
}

fn basemap_root_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut push = |path: PathBuf| {
        if !candidates.iter().any(|existing| existing == &path) {
            candidates.push(path);
        }
    };

    if let Some(root) = std::env::var_os("RUSTWX_BASEMAP_DIR").map(PathBuf::from) {
        push(root);
    }
    if let Some(root) = std::env::var_os("RUSTWX_ASSETS_DIR").map(PathBuf::from) {
        push(root.join("basemap"));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            for ancestor in parent.ancestors().take(8) {
                push(ancestor.join("assets").join("basemap"));
                push(ancestor.join("Resources").join("assets").join("basemap"));
            }
        }
    }

    if let Ok(current_dir) = std::env::current_dir() {
        for ancestor in current_dir.ancestors().take(6) {
            push(ancestor.join("assets").join("basemap"));
            push(ancestor.join("basemap"));
        }
    }

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .map(Path::to_path_buf);
    if let Some(workspace_root) = workspace_root {
        push(workspace_root.join("assets").join("basemap"));
        push(
            workspace_root
                .join("rustbox-fresh")
                .join("assets")
                .join("basemap"),
        );
    }

    candidates
}

/// Load US county boundary line segments (from the 5m TIGER cartographic
/// boundary shapefile). Used only when the white basemap style is active.
pub fn load_us_county_lines() -> Vec<LonLatLine> {
    static CACHE: OnceLock<Vec<LonLatLine>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let Some(root) = checked_in_us_counties_5m_root() else {
                return Vec::new();
            };
            let path = root.join("cb_2023_us_county_5m.shp");
            // The TIGER shapefile is a polygon layer; load_lines_from_shapefile
            // returns ring-edges, which is exactly what we want for borders.
            load_lines_from_shapefile(&path).unwrap_or_default()
        })
        .clone()
}

pub fn default_conus_feature_paths() -> Vec<PathBuf> {
    let Some(root) = cartopy_natural_earth_root() else {
        return vec![];
    };
    vec![
        root.join("physical").join("ne_10m_coastline.shp"),
        root.join("cultural")
            .join("ne_10m_admin_0_boundary_lines_land.shp"),
        root.join("cultural")
            .join("ne_50m_admin_1_states_provinces_lines.shp"),
    ]
}

/// Load multi-ring polygons from a shapefile. Each returned polygon is a list
/// of rings — the first ring is the outer boundary, subsequent rings are
/// holes. Ring winding direction from the shapefile is preserved; the fill
/// routine uses an even-odd rule so orientation doesn't matter.
pub fn load_polygons_from_shapefile(path: &Path) -> Result<Vec<LonLatPolygon>, shapefile::Error> {
    let mut reader = ShapeReader::from_path(path)?;
    let mut polygons = Vec::new();

    for shape in reader.iter_shapes() {
        match shape? {
            Shape::Polygon(polygon) => {
                let mut rings_for_this_poly: Vec<Vec<(f64, f64)>> = Vec::new();
                for ring in polygon.rings() {
                    let pts: Vec<(f64, f64)> = ring.points().iter().map(|p| (p.x, p.y)).collect();
                    if pts.len() >= 3 {
                        rings_for_this_poly.push(pts);
                    }
                }
                if !rings_for_this_poly.is_empty() {
                    polygons.push(rings_for_this_poly);
                }
            }
            _ => {}
        }
    }
    Ok(polygons)
}

pub fn load_lines_from_shapefile(path: &Path) -> Result<Vec<LonLatLine>, shapefile::Error> {
    let mut reader = ShapeReader::from_path(path)?;
    let mut lines = Vec::new();

    for shape in reader.iter_shapes() {
        match shape? {
            Shape::Polyline(polyline) => {
                for part in polyline.parts() {
                    let points: LonLatLine = part.iter().map(|p| (p.x, p.y)).collect();
                    if points.len() >= 2 {
                        lines.push(points);
                    }
                }
            }
            Shape::Polygon(polygon) => {
                for ring in polygon.rings() {
                    let points: LonLatLine = ring.points().iter().map(|p| (p.x, p.y)).collect();
                    if points.len() >= 2 {
                        lines.push(points);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(lines)
}

pub fn load_default_conus_features() -> Vec<LonLatLine> {
    static CACHE: OnceLock<Vec<LonLatLine>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let mut all = Vec::new();
            for path in default_conus_feature_paths() {
                if let Ok(lines) = load_lines_from_shapefile(&path) {
                    all.extend(lines);
                }
            }
            all
        })
        .clone()
}

/// Load filled basemap polygons (ocean → land → lakes, bottom to top) from
/// the checked-in Natural Earth assets, colored for the requested style.
/// Prefers the repo's 10m set, falls back to 110m, then cartopy cache.
///
/// - `Filled`: beige land + pale-blue ocean (weathermodels.com look).
/// - `White`: white land + white ocean (NWS look — political lines carry the
///   composition instead of fill color).
/// Back-compat no-arg entry point. Defaults to `BasemapStyle::Filled`, which
/// is what existing product/CLI callers expect.
pub fn load_styled_conus_polygons() -> Vec<StyledLonLatPolygonLayer> {
    load_styled_conus_polygons_for(BasemapStyle::Filled)
}

pub fn load_styled_basemap_polygons() -> Vec<StyledLonLatPolygonLayer> {
    load_styled_conus_polygons()
}

pub fn load_styled_conus_polygons_for(style: BasemapStyle) -> Vec<StyledLonLatPolygonLayer> {
    static FILLED: OnceLock<Vec<StyledLonLatPolygonLayer>> = OnceLock::new();
    static WHITE: OnceLock<Vec<StyledLonLatPolygonLayer>> = OnceLock::new();
    let cache = match style {
        BasemapStyle::Filled => &FILLED,
        BasemapStyle::White => &WHITE,
    };
    cache.get_or_init(|| build_conus_polygons(style)).clone()
}

pub fn load_styled_basemap_polygons_for(style: BasemapStyle) -> Vec<StyledLonLatPolygonLayer> {
    load_styled_conus_polygons_for(style)
}

pub fn load_styled_basemap_polygons_for_detail(
    style: BasemapStyle,
    detail: BasemapDetail,
) -> Vec<StyledLonLatPolygonLayer> {
    match detail {
        BasemapDetail::Regional => load_styled_conus_polygons_for(style),
        BasemapDetail::Broad => load_styled_broad_polygons_for(style),
        BasemapDetail::Global => load_styled_global_polygons_for(style),
    }
}

fn load_styled_broad_polygons_for(style: BasemapStyle) -> Vec<StyledLonLatPolygonLayer> {
    static FILLED: OnceLock<Vec<StyledLonLatPolygonLayer>> = OnceLock::new();
    static WHITE: OnceLock<Vec<StyledLonLatPolygonLayer>> = OnceLock::new();
    let cache = match style {
        BasemapStyle::Filled => &FILLED,
        BasemapStyle::White => &WHITE,
    };
    cache.get_or_init(|| build_broad_polygons(style)).clone()
}

fn load_styled_global_polygons_for(style: BasemapStyle) -> Vec<StyledLonLatPolygonLayer> {
    static FILLED: OnceLock<Vec<StyledLonLatPolygonLayer>> = OnceLock::new();
    static WHITE: OnceLock<Vec<StyledLonLatPolygonLayer>> = OnceLock::new();
    let cache = match style {
        BasemapStyle::Filled => &FILLED,
        BasemapStyle::White => &WHITE,
    };
    cache.get_or_init(|| build_global_polygons(style)).clone()
}

fn build_conus_polygons(style: BasemapStyle) -> Vec<StyledLonLatPolygonLayer> {
    // Each candidate is (root, resolution_tag). We try 10m first — crisper
    // lines at CONUS zoom — then 110m, then cartopy's cache as a last resort.
    let candidates: Vec<(PathBuf, &'static str)> = [
        checked_in_natural_earth_10m_root().map(|r| (r, "10m")),
        checked_in_natural_earth_110m_root().map(|r| (r, "110m")),
        cartopy_natural_earth_root().map(|r| (r.join("physical"), "110m")),
    ]
    .into_iter()
    .flatten()
    .collect();

    let (land_fill, ocean_fill, lakes_fill) = match style {
        BasemapStyle::Filled => (BASEMAP_LAND_FILL, BASEMAP_OCEAN_FILL, BASEMAP_OCEAN_FILL),
        BasemapStyle::White => (WHITE_LAND_FILL, WHITE_OCEAN_FILL, WHITE_LAKES_FILL),
    };

    for (root, tag) in candidates {
        let ocean_path = root.join(format!("ne_{tag}_ocean.shp"));
        let land_path = root.join(format!("ne_{tag}_land.shp"));
        let lakes_path = root.join(format!("ne_{tag}_lakes.shp"));

        let ocean = load_polygons_from_shapefile(&ocean_path).unwrap_or_default();
        let land = load_polygons_from_shapefile(&land_path).unwrap_or_default();
        let lakes = load_polygons_from_shapefile(&lakes_path).unwrap_or_default();
        if land.is_empty() && ocean.is_empty() {
            continue;
        }

        let mut out = Vec::with_capacity(3);
        if !ocean.is_empty() {
            out.push(StyledLonLatPolygonLayer {
                polygons: ocean,
                color: ocean_fill,
                role: PolygonRole::Ocean,
            });
        }
        if !land.is_empty() {
            out.push(StyledLonLatPolygonLayer {
                polygons: land,
                color: land_fill,
                role: PolygonRole::Land,
            });
        }
        if !lakes.is_empty() {
            out.push(StyledLonLatPolygonLayer {
                polygons: lakes,
                color: lakes_fill,
                role: PolygonRole::Lake,
            });
        }
        return out;
    }

    Vec::new()
}

/// Back-compat no-arg entry point. Defaults to `BasemapStyle::Filled`, which
/// is what existing product/CLI callers expect.
pub fn load_styled_conus_features() -> Vec<StyledLonLatLayer> {
    load_styled_conus_features_for(BasemapStyle::Filled)
}

pub fn load_styled_basemap_features() -> Vec<StyledLonLatLayer> {
    load_styled_conus_features()
}

pub fn load_styled_conus_features_for(style: BasemapStyle) -> Vec<StyledLonLatLayer> {
    static FILLED: OnceLock<Vec<StyledLonLatLayer>> = OnceLock::new();
    static WHITE: OnceLock<Vec<StyledLonLatLayer>> = OnceLock::new();
    let cache = match style {
        BasemapStyle::Filled => &FILLED,
        BasemapStyle::White => &WHITE,
    };
    cache.get_or_init(|| build_conus_features(style)).clone()
}

pub fn load_styled_basemap_features_for(style: BasemapStyle) -> Vec<StyledLonLatLayer> {
    load_styled_conus_features_for(style)
}

pub fn load_styled_basemap_features_for_detail(
    style: BasemapStyle,
    detail: BasemapDetail,
) -> Vec<StyledLonLatLayer> {
    match detail {
        BasemapDetail::Regional => load_styled_conus_features_for(style),
        BasemapDetail::Broad => load_styled_broad_features_for(style),
        BasemapDetail::Global => load_styled_global_features_for(style),
    }
}

fn load_styled_broad_features_for(style: BasemapStyle) -> Vec<StyledLonLatLayer> {
    static FILLED: OnceLock<Vec<StyledLonLatLayer>> = OnceLock::new();
    static WHITE: OnceLock<Vec<StyledLonLatLayer>> = OnceLock::new();
    let cache = match style {
        BasemapStyle::Filled => &FILLED,
        BasemapStyle::White => &WHITE,
    };
    cache.get_or_init(|| build_broad_features(style)).clone()
}

fn build_conus_features(style: BasemapStyle) -> Vec<StyledLonLatLayer> {
    // Design goal: basemap must stay readable on top of saturated colormap fills
    // without being so heavy that an empty CONUS looks cluttered. Political
    // linework is drawn in hierarchy order (weakest → strongest): counties →
    // state → national → coast.
    //
    // Same precedence as polygon loading: 10m (checked-in) > 110m (checked-in)
    // > cartopy cache.
    let (root, tag): (PathBuf, &'static str) = if let Some(r) = checked_in_natural_earth_10m_root()
    {
        (r, "10m")
    } else if let Some(r) = checked_in_natural_earth_110m_root() {
        (r, "110m")
    } else if let Some(cartopy) = cartopy_natural_earth_root() {
        let _ = cartopy;
        return vec![StyledLonLatLayer {
            lines: load_default_conus_features(),
            color: feature_colors(style).coast,
            width: BASEMAP_COAST_WIDTH,
            role: LineworkRole::Coast,
        }];
    } else {
        return vec![StyledLonLatLayer {
            lines: load_default_conus_features(),
            color: feature_colors(style).coast,
            width: BASEMAP_COAST_WIDTH,
            role: LineworkRole::Coast,
        }];
    };

    let coast_path = root.join(format!("ne_{tag}_coastline.shp"));
    let lakes_path = root.join(format!("ne_{tag}_lakes.shp"));
    let nat_path = root.join(format!("ne_{tag}_admin_0_boundary_lines_land.shp"));
    let state_path = root.join(format!("ne_{tag}_admin_1_states_provinces_lines.shp"));

    let coast = load_lines_from_shapefile(&coast_path).unwrap_or_default();
    let lakes = load_lines_from_shapefile(&lakes_path).unwrap_or_default();
    let nat = load_lines_from_shapefile(&nat_path).unwrap_or_default();
    let state = load_lines_from_shapefile(&state_path).unwrap_or_default();

    let colors = feature_colors(style);
    let widths = feature_widths(style);
    let mut layers = Vec::with_capacity(7);

    // Counties come first (weakest) so everything else paints on top. The
    // presentation layer decides whether the current plot style should show
    // them, which lets operational filled maps opt into faint county texture
    // without forcing it into softer atlas maps.
    let counties = load_us_county_lines();
    if !counties.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: counties,
            color: colors.county,
            width: widths.county,
            role: LineworkRole::County,
        });
    }

    if !lakes.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: lakes,
            color: colors.lake,
            width: widths.lake,
            role: LineworkRole::Lake,
        });
    }
    if !state.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: state,
            color: colors.state,
            width: widths.state,
            role: LineworkRole::State,
        });
    }
    if !nat.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: nat,
            color: colors.nat,
            width: widths.nat,
            role: LineworkRole::International,
        });
    }
    if !coast.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: coast,
            color: colors.coast,
            width: widths.coast,
            role: LineworkRole::Coast,
        });
    }

    if layers.is_empty() {
        return vec![StyledLonLatLayer {
            lines: load_default_conus_features(),
            color: colors.coast,
            width: widths.coast,
            role: LineworkRole::Coast,
        }];
    }

    layers
}

fn build_broad_polygons(style: BasemapStyle) -> Vec<StyledLonLatPolygonLayer> {
    let (land_fill, ocean_fill, lakes_fill) = match style {
        BasemapStyle::Filled => (BASEMAP_LAND_FILL, BASEMAP_OCEAN_FILL, BASEMAP_OCEAN_FILL),
        BasemapStyle::White => (WHITE_LAND_FILL, WHITE_OCEAN_FILL, WHITE_LAKES_FILL),
    };

    if let Some(root) = checked_in_natural_earth_10m_root() {
        let ocean =
            load_polygons_from_shapefile(&root.join("ne_10m_ocean.shp")).unwrap_or_default();
        let land = load_polygons_from_shapefile(&root.join("ne_10m_land.shp")).unwrap_or_default();
        let lakes =
            load_polygons_from_shapefile(&root.join("ne_10m_lakes.shp")).unwrap_or_default();
        if !land.is_empty() || !ocean.is_empty() {
            let mut out = Vec::with_capacity(3);
            if !ocean.is_empty() {
                out.push(StyledLonLatPolygonLayer {
                    polygons: ocean,
                    color: ocean_fill,
                    role: PolygonRole::Ocean,
                });
            }
            if !land.is_empty() {
                out.push(StyledLonLatPolygonLayer {
                    polygons: land,
                    color: land_fill,
                    role: PolygonRole::Land,
                });
            }
            if !lakes.is_empty() {
                out.push(StyledLonLatPolygonLayer {
                    polygons: lakes,
                    color: lakes_fill,
                    role: PolygonRole::Lake,
                });
            }
            return out;
        }
    }

    if let Some(root) = checked_in_natural_earth_110m_root() {
        let countries = load_polygons_from_shapefile(&root.join("ne_110m_admin_0_countries.shp"))
            .unwrap_or_default();
        if !countries.is_empty() {
            return vec![StyledLonLatPolygonLayer {
                polygons: countries,
                color: land_fill,
                role: PolygonRole::Land,
            }];
        }
    }

    let mut broad = build_conus_polygons(style)
        .into_iter()
        .filter(|layer| !matches!(layer.role, PolygonRole::Lake))
        .collect::<Vec<_>>();
    if broad.is_empty() {
        broad.push(StyledLonLatPolygonLayer {
            polygons: Vec::new(),
            color: ocean_fill,
            role: PolygonRole::Ocean,
        });
    }
    let _ = lakes_fill;
    broad
}

fn build_global_polygons(style: BasemapStyle) -> Vec<StyledLonLatPolygonLayer> {
    let (land_fill, _, _) = match style {
        BasemapStyle::Filled => (BASEMAP_LAND_FILL, BASEMAP_OCEAN_FILL, BASEMAP_OCEAN_FILL),
        BasemapStyle::White => (WHITE_LAND_FILL, WHITE_OCEAN_FILL, WHITE_LAKES_FILL),
    };

    if let Some(root) = checked_in_natural_earth_110m_root() {
        let countries = load_polygons_from_shapefile(&root.join("ne_110m_admin_0_countries.shp"))
            .unwrap_or_default();
        if !countries.is_empty() {
            return vec![StyledLonLatPolygonLayer {
                polygons: countries,
                color: land_fill,
                role: PolygonRole::Land,
            }];
        }
    }

    build_broad_polygons(style)
}

fn build_broad_features(style: BasemapStyle) -> Vec<StyledLonLatLayer> {
    let Some((root, tag)) = checked_in_natural_earth_10m_root()
        .map(|root| (root, "10m"))
        .or_else(|| checked_in_natural_earth_110m_root().map(|root| (root, "110m")))
    else {
        return build_conus_features(style)
            .into_iter()
            .filter(|layer| !matches!(layer.role, LineworkRole::County | LineworkRole::State))
            .map(|mut layer| {
                layer.width = 1;
                layer
            })
            .collect();
    };

    let coast = load_lines_from_shapefile(&root.join(format!("ne_{tag}_coastline.shp")))
        .unwrap_or_default();
    let lakes =
        load_lines_from_shapefile(&root.join(format!("ne_{tag}_lakes.shp"))).unwrap_or_default();
    let nat =
        load_lines_from_shapefile(&root.join(format!("ne_{tag}_admin_0_boundary_lines_land.shp")))
            .unwrap_or_default();
    let state = load_lines_from_shapefile(
        &root.join(format!("ne_{tag}_admin_1_states_provinces_lines.shp")),
    )
    .unwrap_or_default();
    let colors = feature_colors(style);
    let mut layers = Vec::with_capacity(4);
    if !lakes.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: lakes,
            color: Rgba::with_alpha(colors.lake.r, colors.lake.g, colors.lake.b, 70),
            width: 1,
            role: LineworkRole::Lake,
        });
    }
    if !state.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: state,
            color: Rgba::with_alpha(colors.state.r, colors.state.g, colors.state.b, 210),
            width: 1,
            role: LineworkRole::State,
        });
    }
    if !nat.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: nat,
            color: Rgba::with_alpha(colors.nat.r, colors.nat.g, colors.nat.b, 155),
            width: 1,
            role: LineworkRole::International,
        });
    }
    if !coast.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: coast,
            color: Rgba::with_alpha(colors.coast.r, colors.coast.g, colors.coast.b, 180),
            width: 1,
            role: LineworkRole::Coast,
        });
    }
    layers
}

fn load_styled_global_features_for(style: BasemapStyle) -> Vec<StyledLonLatLayer> {
    static FILLED: OnceLock<Vec<StyledLonLatLayer>> = OnceLock::new();
    static WHITE: OnceLock<Vec<StyledLonLatLayer>> = OnceLock::new();
    let cache = match style {
        BasemapStyle::Filled => &FILLED,
        BasemapStyle::White => &WHITE,
    };
    cache.get_or_init(|| build_global_features(style)).clone()
}

fn build_global_features(style: BasemapStyle) -> Vec<StyledLonLatLayer> {
    let Some((root, tag)) = checked_in_natural_earth_110m_root()
        .map(|root| (root, "110m"))
        .or_else(|| checked_in_natural_earth_10m_root().map(|root| (root, "10m")))
    else {
        return build_broad_features(style)
            .into_iter()
            .filter(|layer| !matches!(layer.role, LineworkRole::County | LineworkRole::State))
            .collect();
    };

    let coast = load_lines_from_shapefile(&root.join(format!("ne_{tag}_coastline.shp")))
        .unwrap_or_default();
    let lakes =
        load_lines_from_shapefile(&root.join(format!("ne_{tag}_lakes.shp"))).unwrap_or_default();
    let nat =
        load_lines_from_shapefile(&root.join(format!("ne_{tag}_admin_0_boundary_lines_land.shp")))
            .unwrap_or_default();
    let colors = feature_colors(style);
    let mut layers = Vec::with_capacity(3);
    if !lakes.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: lakes,
            color: Rgba::with_alpha(colors.lake.r, colors.lake.g, colors.lake.b, 42),
            width: 1,
            role: LineworkRole::Lake,
        });
    }
    if !nat.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: nat,
            color: Rgba::with_alpha(colors.nat.r, colors.nat.g, colors.nat.b, 90),
            width: 1,
            role: LineworkRole::International,
        });
    }
    if !coast.is_empty() {
        layers.push(StyledLonLatLayer {
            lines: coast,
            color: Rgba::with_alpha(colors.coast.r, colors.coast.g, colors.coast.b, 115),
            width: 1,
            role: LineworkRole::Coast,
        });
    }
    layers
}

struct FeatureColors {
    coast: Rgba,
    lake: Rgba,
    nat: Rgba,
    state: Rgba,
    county: Rgba,
}

struct FeatureWidths {
    coast: u32,
    lake: u32,
    nat: u32,
    state: u32,
    county: u32,
}

fn feature_colors(style: BasemapStyle) -> FeatureColors {
    match style {
        BasemapStyle::Filled => FeatureColors {
            coast: BASEMAP_COAST_CORE,
            lake: Rgba::with_alpha(118, 136, 154, 220),
            nat: BASEMAP_NAT_CORE,
            state: BASEMAP_STATE_CORE,
            county: BASEMAP_COUNTY_CORE,
        },
        BasemapStyle::White => FeatureColors {
            coast: WHITE_COAST_CORE,
            lake: Rgba::new(124, 138, 152),
            nat: WHITE_NAT_CORE,
            state: WHITE_STATE_CORE,
            county: WHITE_COUNTY_CORE,
        },
    }
}

fn feature_widths(style: BasemapStyle) -> FeatureWidths {
    match style {
        BasemapStyle::Filled => FeatureWidths {
            coast: BASEMAP_COAST_WIDTH,
            lake: 1,
            nat: BASEMAP_NAT_WIDTH,
            state: BASEMAP_STATE_WIDTH,
            county: 1,
        },
        BasemapStyle::White => FeatureWidths {
            coast: 2,
            lake: 1,
            nat: 2,
            state: 2,
            county: 1,
        },
    }
}

// Basemap palette. Kept as module constants so a styling pass has one obvious dial.
//
// Design target: weathermodels.com ECMWF / NOAA Blend reference look — cool-beige
// land, pale cool-blue ocean, thin crisp dark linework.
pub const BASEMAP_LAND_FILL: Rgba = Rgba {
    r: 238,
    g: 237,
    b: 230,
    a: 255,
};
pub const BASEMAP_OCEAN_FILL: Rgba = Rgba {
    r: 224,
    g: 234,
    b: 242,
    a: 255,
};
const BASEMAP_COAST_CORE: Rgba = Rgba {
    r: 32,
    g: 40,
    b: 50,
    a: 255,
};
const BASEMAP_NAT_CORE: Rgba = Rgba {
    r: 74,
    g: 82,
    b: 96,
    a: 255,
};
// State borders should read as black linework without becoming as dominant as
// coastlines or national borders.
const BASEMAP_STATE_CORE: Rgba = Rgba {
    r: 20,
    g: 24,
    b: 30,
    a: 255,
};
const BASEMAP_COUNTY_CORE: Rgba = Rgba {
    r: 108,
    g: 116,
    b: 128,
    a: 170,
};
const BASEMAP_COAST_WIDTH: u32 = 1;
const BASEMAP_NAT_WIDTH: u32 = 1;
const BASEMAP_STATE_WIDTH: u32 = 1;

// ---- White / NWS-style palette ---------------------------------------------
//
// Target: the white-background SPC/NWS composition where political boundaries
// (counties, states, national) carry the composition and data overlays in
// saturated colors on top. Fills are pure white; linework is dark enough to
// read over white without being heavy.
pub const WHITE_LAND_FILL: Rgba = Rgba {
    r: 255,
    g: 255,
    b: 255,
    a: 255,
};
pub const WHITE_OCEAN_FILL: Rgba = Rgba {
    r: 255,
    g: 255,
    b: 255,
    a: 255,
};
const WHITE_LAKES_FILL: Rgba = Rgba {
    r: 240,
    g: 244,
    b: 248,
    a: 255,
};
const WHITE_COAST_CORE: Rgba = Rgba {
    r: 24,
    g: 30,
    b: 40,
    a: 255,
};
const WHITE_NAT_CORE: Rgba = Rgba {
    r: 44,
    g: 50,
    b: 62,
    a: 255,
};
const WHITE_STATE_CORE: Rgba = Rgba {
    r: 20,
    g: 24,
    b: 30,
    a: 255,
};
// County lines are the most common element in NWS-style CONUS maps: thin,
// medium-light gray so the density reads as texture rather than emphasis.
const WHITE_COUNTY_CORE: Rgba = Rgba {
    r: 164,
    g: 172,
    b: 184,
    a: 220,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn role_index(layers: &[StyledLonLatLayer], role: LineworkRole) -> Option<usize> {
        layers.iter().position(|layer| layer.role == role)
    }

    #[test]
    fn conus_linework_draws_lakes_below_coasts() {
        let layers = build_conus_features(BasemapStyle::Filled);
        let lake = role_index(&layers, LineworkRole::Lake).expect("lakes should be present");
        let coast = role_index(&layers, LineworkRole::Coast).expect("coasts should be present");

        assert!(lake < coast);
    }

    #[test]
    fn broad_linework_includes_admin1_context_below_country_and_coast() {
        let layers = build_broad_features(BasemapStyle::Filled);
        let state =
            role_index(&layers, LineworkRole::State).expect("state/province lines should load");
        let international =
            role_index(&layers, LineworkRole::International).expect("country lines should load");
        let coast = role_index(&layers, LineworkRole::Coast).expect("coasts should load");

        assert!(state < international);
        assert!(state < coast);
    }

    #[test]
    fn global_linework_draws_lakes_below_coasts() {
        let layers = build_global_features(BasemapStyle::Filled);
        let lake = role_index(&layers, LineworkRole::Lake);
        let coast = role_index(&layers, LineworkRole::Coast);

        if let (Some(lake), Some(coast)) = (lake, coast) {
            assert!(lake < coast);
        }
    }
}
