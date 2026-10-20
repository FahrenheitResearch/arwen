//! Setup-only projected mass-grid contract for hydrofabric area weights.
//! Geometry comes from the executed grid handle, including translated extents.
//! This exports metadata; no atmosphere or hydrology state is advanced here.

use super::{ERR, clear_error, guard, set_error, with_grid};
use crate::error::{Result, StaticError};
use crate::projection::{ProjectedGrid, ProjectionKind};
use crate::{EARTH_RADIUS_M, highres};

fn render(grid: &ProjectedGrid) -> Result<String> {
    let spec = &grid.spec;
    // Reuse the native static-field footprint, rather than reconstructing
    // projected coordinates from geographical bounds in the Python shell.
    let (_, transform, (ny, nx)) = highres::raster_geometry(spec)?;
    let nx = u32::try_from(nx).map_err(|_| invalid("mass-grid nx exceeds u32"))?;
    let ny = u32::try_from(ny).map_err(|_| invalid("mass-grid ny exceeds u32"))?;
    let x0 = transform[2];
    let y0 = transform[5] + transform[4] * f64::from(ny);
    let dx = transform[0];
    let dy = -transform[4];
    if nx == 0
        || ny == 0
        || nx.checked_mul(ny).is_none()
        || ![x0, y0, dx, dy].iter().all(|v| v.is_finite())
        || dx <= 0.0
        || dy <= 0.0
    {
        return Err(invalid(
            "hydrofabric requires finite projected cell edges, positive metre spacing and <=u32::MAX mass cells",
        ));
    }
    // CRS metadata mirrors Crs::ModelSphere's existing projection choices.
    // Mercator is anchored at ref_lon, not stand_lon; polar hemisphere follows
    // truelat1. No datum, ellipsoid or EPSG code is inferred from a bounding box.
    let projection = match spec.kind {
        ProjectionKind::Lambert => format!(
            "+proj=lcc +lat_1={} +lat_2={} +lat_0={} +lon_0={}",
            spec.truelat1, spec.truelat2, spec.truelat1, spec.stand_lon
        ),
        ProjectionKind::Mercator => format!(
            "+proj=merc +lat_ts={} +lon_0={}",
            spec.truelat1, spec.ref_lon
        ),
        ProjectionKind::Polar => format!(
            "+proj=stere +lat_0={} +lat_ts={} +lon_0={}",
            if spec.truelat1 >= 0.0 { 90 } else { -90 },
            spec.truelat1,
            spec.stand_lon
        ),
        ProjectionKind::Rows => {
            return Err(invalid(
                "a latitude-row grid is not a rectangular projected-metre hydrofabric grid",
            ));
        }
    };
    let crs =
        format!("{projection} +a={EARTH_RADIUS_M} +b={EARTH_RADIUS_M} +units=m +no_defs +type=crs");
    Ok(serde_json::to_string(&serde_json::json!({
        "schema": "arwen.nextgen.parent-grid.v1",
        "grid": {"nx": nx, "ny": ny, "x0": x0, "y0": y0, "dx": dx, "dy": dy,
                 "crs": crs, "row_order": "south_to_north"},
        "native_spec": spec,
        "layout": "C [south_north,west_east]; x-fast; southwest cell edge",
        "geometry_method": "static_fields::highres::raster_geometry; native mass-grid row order"
    }))
    .map_err(|error| invalid(&format!("hydrofabric grid JSON: {error}")))?)
}

fn invalid(message: &str) -> StaticError {
    StaticError::Invalid(message.into())
}

/// Export the projected mass-grid footprint of an existing native handle.
/// Returns full UTF-8 length (no trailing NUL), or -1 with last_error on failure.
/// Native mass-grid rows run south to north, even though raster_geometry's
/// affine transform describes a north-first raster. Only the affine origin is
/// converted here; no array is flipped or materialized.
///
/// # Safety
/// `buf` must be writable for `cap` bytes, or null with cap zero for sizing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_grid_hydrofabric_json(
    handle: u64,
    buf: *mut u8,
    cap: usize,
) -> i64 {
    guard(ERR as i64, || {
        clear_error();
        if buf.is_null() && cap != 0 {
            set_error("hydrofabric grid output buffer is null with nonzero capacity");
            return ERR as i64;
        }
        match with_grid(handle, render) {
            None => {
                set_error(format!("unknown grid handle {handle}"));
                ERR as i64
            }
            Some(Err(err)) => {
                set_error(err.to_string());
                ERR as i64
            }
            Some(Ok(json)) => {
                if cap != 0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(json.as_ptr(), buf, cap.min(json.len()))
                    };
                }
                json.len() as i64
            }
        }
    })
}
