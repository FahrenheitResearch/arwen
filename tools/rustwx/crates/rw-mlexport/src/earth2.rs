//! Generic profile coordinates and Earth2Studio's per-channel IO layout.
//! Consumer identities never enter the data path.
use crate::error::{refuse, Result};
use crate::request::Request;
use crate::state::{self, DomainState};
use crate::zarr::{self, ArrayMeta, Dtype};
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::path::Path;

pub fn nside(request: &Request) -> Option<usize> {
    request
        .grid
        .target
        .as_ref()
        .and_then(|p| p["nside"].as_u64())
        .map(|n| n as usize)
}

pub struct Coordinates {
    pub ny: usize,
    pub nx: usize,
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    pub axes: bool,
}

pub fn target_coordinates(p: &Value) -> Result<Coordinates> {
    let kind = p["kind"].as_str().unwrap_or("");
    let shape: Vec<usize> = p["shape"]
        .as_array()
        .ok_or_else(|| refuse("target profile lacks shape"))?
        .iter()
        .map(|s| s.as_u64().unwrap_or(0) as usize)
        .collect();
    if shape.len() < 2 || shape.iter().any(|&s| s == 0) {
        return Err(refuse("target profile has an empty spatial axis"));
    }
    let nx = *shape.last().unwrap();
    let ny = shape[..shape.len() - 1].iter().product();
    let number = |key: &str| {
        p[key]
            .as_f64()
            .ok_or_else(|| refuse(format!("target profile lacks {key}")))
    };
    if kind == "latlon" {
        let (lat0, lon0, dlat, dlon) = (
            number("lat0")?,
            number("lon0")?,
            number("dlat")?,
            number("dlon")?,
        );
        return Ok(Coordinates {
            ny,
            nx,
            lat: (0..ny).map(|j| lat0 + j as f64 * dlat).collect(),
            lon: (0..nx).map(|i| lon0 + i as f64 * dlon).collect(),
            axes: true,
        });
    }
    let pairs: Vec<(f64, f64)> = match kind {
        "healpix" => {
            let n = p["nside"].as_u64().unwrap_or(0) as usize;
            if n == 0
                || !n.is_power_of_two()
                || shape != [12, n, n]
                || p["ordering"] != "HEALPIX_PAD_XY"
            {
                return Err(refuse("HEALPix profile requires shape [12,nside,nside], power-of-two nside and HEALPIX_PAD_XY"));
            }
            (0..12 * n * n)
                .into_par_iter()
                .map(|k| {
                    let f = k / (n * n);
                    let (px, py) = (k % n, (k / n) % n);
                    // Earth2Grid XY(origin=N, clockwise=True) -> default XY(S,CCW).
                    let (x, y) = ((n - 1 - py) as f64, (n - 1 - px) as f64);
                    let jr =
                        [2., 2., 2., 2., 3., 3., 3., 3., 4., 4., 4., 4.][f] * n as f64 - x - y - 1.;
                    let (nr, z) = if jr < n as f64 {
                        (jr, 1. - jr * jr / (3. * (n * n) as f64))
                    } else if jr > (3 * n) as f64 {
                        let nr = (4 * n) as f64 - jr;
                        (nr, -1. + nr * nr / (3. * (n * n) as f64))
                    } else {
                        (n as f64, ((2 * n) as f64 - jr) * 2. / (3 * n) as f64)
                    };
                    let lon =
                        ([1., 3., 5., 7., 0., 2., 4., 6., 1., 3., 5., 7.][f] + (x - y) / nr) * 45.;
                    (z.asin().to_degrees(), lon.rem_euclid(360.))
                })
                .collect()
        }
        "lambert" => {
            let (phi0, lambda0, phi1, phi2, lambda_c) = (
                number("lat0")?.to_radians(),
                number("lon0")?.to_radians(),
                number("truelat1")?.to_radians(),
                number("truelat2")?.to_radians(),
                number("stand_lon")?.to_radians(),
            );
            let (dx, dy, radius) = (number("dx")?, number("dy")?, number("radius")?);
            let tanhalf = |phi: f64| (std::f64::consts::FRAC_PI_4 + phi / 2.).tan();
            let n = if (phi1 - phi2).abs() < 1e-12 {
                phi1.sin()
            } else {
                (phi1.cos() / phi2.cos()).ln() / (tanhalf(phi2) / tanhalf(phi1)).ln()
            };
            let a = radius * phi1.cos() * tanhalf(phi1).powf(n) / n;
            let rho0 = a / tanhalf(phi0).powf(n);
            let theta0 = n * (lambda0 - lambda_c);
            let (x0, y0) = (rho0 * theta0.sin(), rho0 * theta0.cos());
            (0..ny * nx)
                .into_par_iter()
                .map(|k| {
                    let x = x0 + (k % nx) as f64 * dx;
                    let y = y0 - (k / nx) as f64 * dy;
                    let rho = x.hypot(y).copysign(n);
                    let phi = 2. * (a / rho).powf(1. / n).atan() - std::f64::consts::FRAC_PI_2;
                    (
                        phi.to_degrees(),
                        (lambda_c + x.atan2(y) / n).to_degrees().rem_euclid(360.),
                    )
                })
                .collect()
        }
        _ => return Err(refuse(format!("no target coordinate operator for {kind}"))),
    };
    let (lat, lon) = pairs.into_iter().unzip();
    Ok(Coordinates {
        ny,
        nx,
        lat,
        lon,
        axes: false,
    })
}

fn small(
    store: &Path,
    name: &str,
    dims: Vec<String>,
    shape: Vec<usize>,
    dtype: Dtype,
    attrs: Value,
    bytes: Vec<u8>,
) -> Result<u64> {
    let attrs = attrs.as_object().cloned().unwrap_or_default();
    zarr::write_small(
        store,
        name,
        &ArrayMeta {
            chunks: shape.clone(),
            shape,
            dtype,
            dims,
            attrs,
        },
        &bytes,
    )
}

pub fn close_domain(
    out: &Path,
    d: &DomainState,
    r: &Request,
) -> Result<(u64, Vec<(String, String)>)> {
    if d.latlon_versions != 1 {
        return Err(refuse("Earth2Studio fixed-grid stores cannot represent a moving native grid; select a fixed target profile"));
    }
    let store = out.join(format!("{}.zarr", d.id));
    let profile = r
        .grid
        .target
        .as_ref()
        .ok_or_else(|| refuse("Earth2 layout requires a resolved grid profile"))?;
    let native = profile["kind"] == "native";
    let shape: Vec<usize> = if native {
        vec![d.out_ny, d.out_nx]
    } else {
        profile["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };
    let dims: Vec<String> = profile["dims"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let (cy, cx) = d.chunk_yx();
    let spatial_chunks = if nside(r).is_some() {
        vec![1, cy, cx]
    } else {
        vec![cy, cx]
    };
    let mut bytes = 0;
    bytes += small(
        &store,
        "time",
        vec!["time".into()],
        vec![1],
        Dtype::I64,
        json!({"units":"seconds since 1970-01-01 00:00:00","calendar":"proleptic_gregorian","standard_name":"forecast_reference_time"}),
        zarr::i64_bytes(&[d.init_time]),
    )?;
    bytes += small(
        &store,
        "lead_time",
        vec!["lead_time".into()],
        vec![d.times.len()],
        Dtype::I64,
        json!({"units":"seconds","dtype":"timedelta64[ns]","standard_name":"forecast_period"}),
        zarr::i64_bytes(&d.times.iter().map(|t| t - d.init_time).collect::<Vec<_>>()),
    )?;
    let (lat, lon, axes) = if let Some(g) = &d.regrid {
        (g.lat.clone(), g.lon.clone(), profile["kind"] == "latlon")
    } else if let Some(g) = &d.native.regular {
        (g.lat.clone(), g.lon.clone(), true)
    } else {
        (
            state::load_plane(out, &format!("{}-xlat-0.f64", d.id))?.unwrap(),
            state::load_plane(out, &format!("{}-xlong-0.f64", d.id))?.unwrap(),
            false,
        )
    };
    for (name, values, unit) in [("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")] {
        let (cdims, cshape) = if axes {
            (
                vec![if name == "lat" {
                    dims[0].clone()
                } else {
                    dims[1].clone()
                }],
                vec![values.len()],
            )
        } else {
            (dims.clone(), shape.clone())
        };
        bytes += small(
            &store,
            name,
            cdims,
            cshape,
            Dtype::F64,
            json!({"units":unit,"standard_name":if name=="lat"{"latitude"}else{"longitude"}}),
            zarr::f64_bytes(&values),
        )?;
    }
    for (dim, &size) in dims.iter().zip(&shape) {
        if dim == "lat" || dim == "lon" {
            continue;
        }
        if let Some(axis) = profile["axis_coordinates"].get(dim) {
            let start = axis["start"]
                .as_f64()
                .ok_or_else(|| refuse("profile axis lacks start"))?;
            let step = axis["step"]
                .as_f64()
                .ok_or_else(|| refuse("profile axis lacks step"))?;
            let values: Vec<f64> = (0..size)
                .into_par_iter()
                .map(|i| start + i as f64 * step)
                .collect();
            bytes += small(
                &store,
                dim,
                vec![dim.clone()],
                vec![size],
                Dtype::F64,
                json!({"units":axis["units"]}),
                zarr::f64_bytes(&values),
            )?;
            continue;
        }
        bytes += small(
            &store,
            dim,
            vec![dim.clone()],
            vec![size],
            Dtype::I32,
            json!({}),
            zarr::i32_bytes(&(0..size as i32).collect::<Vec<_>>()),
        )?;
    }
    let mut omitted = Vec::new();
    for var in &d.variables {
        if let Some(why) = &var.omitted {
            omitted.push((var.id.clone(), why.clone()));
            continue;
        }
        let row = r.variables.iter().find(|row| row.id == var.id).unwrap();
        let mut vdims = vec!["time".into(), "lead_time".into()];
        vdims.extend(dims.clone());
        let mut vshape = vec![1, d.times.len()];
        vshape.extend(&shape);
        let mut chunks = vec![1, 1];
        chunks.extend(&spatial_chunks);
        let mut attrs = Map::new();
        attrs.insert("units".into(), json!(row.units));
        attrs.insert("long_name".into(), json!(row.long_name));
        attrs.insert("coordinates".into(), json!("lat lon"));
        if let Some(p) = row.pressure_hpa {
            attrs.insert("pressure_hpa".into(), json!(p));
            attrs.insert(
                "vertical_interpolation".into(),
                json!(crate::finalize::VERTICAL_INTERPOLATION),
            );
            attrs.insert("below_ground_rule".into(), json!(row.below_ground));
        }
        if row.op.starts_with("accumulation:") {
            attrs.insert(
                "accumulation".into(),
                json!("metres of water; trailing period; unavailable initial windows are NaN"),
            );
        }
        zarr::write_meta(
            &store,
            &var.name,
            &ArrayMeta {
                shape: vshape,
                chunks,
                dtype: Dtype::F32,
                dims: vdims,
                attrs,
            },
        )?;
    }
    zarr::write_group(&store,&json!({"Conventions":"CF-1.8","earth2studio_version":"0.19.0","grid_profile":profile,"initialization_time":crate::times::iso(d.init_time),"source":r.provenance.engine,"exporter":r.provenance.exporter_version}).as_object().unwrap().clone())?;
    zarr::consolidate(&store)?;
    zarr::promote_v3(&store)?;
    Ok((bytes, omitted))
}
