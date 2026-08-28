//! Reading generators back out of an existing MPAS grid file.
//!
//! Factored out of `rw_mpas_mesh --from-centres` so the measurement probe
//! and the CLI read a grid the SAME way: one reader, one set of refusals,
//! and an instrument that cannot quietly disagree with the door about what
//! a file says. Every name is the MPAS spelling, so a mesh this crate has
//! never seen is a file rather than a code path.

use crate::mesh::geom::V3;

/// Generators read off a grid file, with the numbers that let a caller check
/// the read rather than trust it.
pub struct Generators {
    pub points: Vec<V3>,
    pub mesh_density: Vec<f64>,
    pub nominal_min_dc: f64,
    pub sphere_radius: f64,
    /// The largest `|r / sphere_radius - 1|` over every centre.
    pub max_radius_departure: f64,
}

/// How far the stored radii may scatter before the points are not one sphere.
///
/// The published meshes measure 6.7e-16, three double ULP. 1e-9 of a unit sphere
/// is 6.4 mm on Earth, and nothing that writes a grid file misses by that much
/// unless its points are on different spheres.
pub const RADIUS_SCATTER_LIMIT: f64 = 1e-9;

/// Read the cell centres, `meshDensity` and `nominalMinDc` out of a grid file.
pub fn read_grid_generators(path: &std::path::Path) -> Result<Generators, String> {
    let file = netcrust::File::open(path)
        .map_err(|e| format!("{} is not a netCDF file this reader can open: {e}", path.display()))?;

    // UNITS. An MPAS grid file carries sphere_radius, and on both published
    // meshes that radius is 1.0: xCell/yCell/zCell are unit vectors and
    // areaCell, dcEdge, dvEdge and nominalMinDc are unit-sphere quantities
    // despite their m and m^2 units attributes. A reader that took those for
    // metres prints spacings of 0.0 km.
    let sphere_radius = match file.attribute("sphere_radius").and_then(|a| a.as_f64()) {
        Some(r) if r > 0.0 && r.is_finite() => r,
        Some(r) => {
            return Err(format!(
                "{} declares sphere_radius = {r}; every length in the file is divided by that radius to reach the unit sphere this crate derives on, and a non-positive radius would put every generator at infinity or reflect the mesh through the origin",
                path.display()
            ));
        }
        None => {
            return Err(format!(
                "{} carries no sphere_radius global attribute, so nothing tells a reader whether xCell is a unit vector or a length in metres. Guessing wrong scales every derived dcEdge and areaCell by 6.4e6 or its inverse, and the mesh would validate clean at the wrong size",
                path.display()
            ));
        }
    };

    let read = |name: &str| -> Result<Vec<f64>, String> {
        file.read_f64(name)
            .map_err(|e| format!("{} has no readable {name}: {e}", path.display()))
    };
    let x = read("xCell")?;
    let y = read("yCell")?;
    let z = read("zCell")?;
    if x.len() != y.len() || y.len() != z.len() {
        return Err(format!(
            "{} carries {} xCell, {} yCell and {} zCell values; three components of one point list have to be the same length or the centres pair up component by component into positions no cell ever had",
            path.display(),
            x.len(),
            y.len(),
            z.len()
        ));
    }
    let n_cells = x.len();

    let mut points = Vec::with_capacity(n_cells);
    let mut max_radius_departure = 0.0f64;
    for i in 0..n_cells {
        let p = [x[i] / sphere_radius, y[i] / sphere_radius, z[i] / sphere_radius];
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if !(r > 0.0) || !r.is_finite() {
            return Err(format!(
                "{}: cell centre {i} is {:?}, which has no direction; a point at the origin has no place on the sphere and the hull's orientation predicate is meaningless for every facet touching it",
                path.display(),
                [x[i], y[i], z[i]]
            ));
        }
        max_radius_departure = max_radius_departure.max((r - 1.0).abs());
        points.push([p[0] / r, p[1] / r, p[2] / r]);
    }
    if max_radius_departure > RADIUS_SCATTER_LIMIT {
        return Err(format!(
            "{}: cell centres depart from sphere_radius = {sphere_radius} by up to {max_radius_departure:.3e} relative, past the {RADIUS_SCATTER_LIMIT:.0e} this reader allows. A mesh built from a mixture of radii is not a spherical Voronoi tessellation of its own generators: the circumcentres sit off the surface and every kite area is taken on a different sphere from the cell it belongs to",
            path.display()
        ));
    }

    let mesh_density = match read("meshDensity") {
        Ok(v) if v.len() == n_cells => v,
        Ok(v) => {
            return Err(format!(
                "{} carries {} meshDensity values for {n_cells} cells; MPAS scales its horizontal mixing length by meshDensity, so a mismatched table applies one cell's diffusion to another",
                path.display(),
                v.len()
            ));
        }
        Err(_) => {
            return Err(format!(
                "{} has no meshDensity variable. It records what resolution function produced these centres and MPAS scales horizontal mixing by it; inventing 1.0 everywhere would silently claim a uniform mesh and give a refined region the background diffusion length",
                path.display()
            ));
        }
    };

    let nominal_min_dc = match read("nominalMinDc") {
        Ok(v) if v.len() == 1 => v[0] / sphere_radius,
        Ok(v) => {
            return Err(format!(
                "{} carries {} nominalMinDc values; it is a single scalar stamp and a reader cannot tell which of several the matching static file was built against",
                path.display(),
                v.len()
            ));
        }
        Err(_) => {
            return Err(format!(
                "{} has no nominalMinDc variable. The mesh registry matches a grid file to its static file on an FP32-bit-exact nominalMinDc, so a made-up stamp produces a grid file no static file can ever be paired with",
                path.display()
            ));
        }
    };

    Ok(Generators {
        points,
        mesh_density,
        nominal_min_dc,
        sphere_radius,
        max_radius_departure,
    })
}
