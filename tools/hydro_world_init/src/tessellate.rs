//! Proof fixture only: an equal-area square tessellation of a lon/lat box,
//! written as a contract GeoPackage (divides only, every square terminal) in a
//! domain-centred Lambert Azimuthal Equal Area CRS (contract section 4, ARWEN
//! srs row, WKT2). It is NOT hydrography: it stands in for the real basin
//! fabric so the initial-state path (custom-WKT CRS, area weights, state
//! mapping, spin-up) can be exercised before the domain's fabric exists.
use crate::geom;
use crate::proj::Transform;
use crate::Result;
use hydro_world::contract::{Divide, Fabric, Geometry, MetadataRow, Srs, CONTRACT};
use std::collections::BTreeMap;

pub fn laea_wkt(name: &str, lat0: f64, lon0: f64) -> String {
    format!(
        "PROJCRS[\"{name}\",BASEGEOGCRS[\"WGS 84\",DATUM[\"World Geodetic System 1984\",ELLIPSOID[\"WGS 84\",6378137,298.257223563,LENGTHUNIT[\"metre\",1]]],PRIMEM[\"Greenwich\",0,ANGLEUNIT[\"degree\",0.0174532925199433]]],CONVERSION[\"Lambert Azimuthal Equal Area\",METHOD[\"Lambert Azimuthal Equal Area\",ID[\"EPSG\",9820]],PARAMETER[\"Latitude of natural origin\",{lat0},ANGLEUNIT[\"degree\",0.0174532925199433]],PARAMETER[\"Longitude of natural origin\",{lon0},ANGLEUNIT[\"degree\",0.0174532925199433]],PARAMETER[\"False easting\",0,LENGTHUNIT[\"metre\",1]],PARAMETER[\"False northing\",0,LENGTHUNIT[\"metre\",1]]],CS[Cartesian,2],AXIS[\"easting (X)\",east,ORDER[1],LENGTHUNIT[\"metre\",1]],AXIS[\"northing (Y)\",north,ORDER[2],LENGTHUNIT[\"metre\",1]]]"
    )
}

pub fn build(name: &str, south: f64, west: f64, north: f64, east: f64, cell_m: f64) -> Result<Fabric> {
    let (lat0, lon0) = (0.5 * (south + north), 0.5 * (west + east));
    let wkt = laea_wkt(&format!("ARWEN LAEA {name}"), lat0, lon0);
    let fwd = Transform::new("EPSG:4326", &wkt)?;
    let inv = Transform::new(&wkt, "EPSG:4326")?;
    let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..=100 {
        let t = i as f64 / 100.0;
        for (lo, la) in [(west + t * (east - west), south), (west + t * (east - west), north), (west, south + t * (north - south)), (east, south + t * (north - south))] {
            let p = fwd.apply(lo, la)?;
            x0 = x0.min(p[0]);
            x1 = x1.max(p[0]);
            y0 = y0.min(p[1]);
            y1 = y1.max(p[1]);
        }
    }
    let (i0, i1) = ((x0 / cell_m).floor() as i64, (x1 / cell_m).ceil() as i64);
    let (j0, j1) = ((y0 / cell_m).floor() as i64, (y1 / cell_m).ceil() as i64);
    let srs_id = 100_001;
    let mut divides = Vec::new();
    let mut n = 0i64;
    for j in j0..j1 {
        for i in i0..i1 {
            let (xa, ya) = (i as f64 * cell_m, j as f64 * cell_m);
            let c = inv.apply(xa + 0.5 * cell_m, ya + 0.5 * cell_m)?;
            if c[0] < west || c[0] > east || c[1] < south || c[1] > north {
                continue;
            }
            n += 1;
            let poly = vec![vec![[xa, ya], [xa + cell_m, ya], [xa + cell_m, ya + cell_m], [xa, ya + cell_m], [xa, ya]]];
            let area_km2 = geom::planar_area(&poly[0][..4]) / 1e6;
            // id namespace 9 (proof fixtures): n = 9*10^10 + local id
            let gid = 90_000_000_000 + n;
            divides.push(Divide {
                fid: n,
                geom: Some(Geometry(geom::encode_gpkg_polygon(&poly, srs_id as i32))),
                divide_id: Some(format!("cat-{gid}")),
                toid: Some(format!("tnx-{gid}")),
                kind: Some("terminal".into()),
                ds_id: None,
                areasqkm: Some(area_km2),
                id: None,
                lengthkm: None,
                tot_drainage_areasqkm: Some(area_km2),
                has_flowline: Some(false),
            });
        }
    }
    let meta = [
        ("contract", CONTRACT.to_string()),
        ("builder", "rw_hydro_init tessellate (proof fixture, not hydrography)".into()),
        ("domain", format!("{name}: lat {south}..{north}, lon {west}..{east}, {cell_m} m squares")),
        ("horizontal_crs", "laea-domain-centroid".into()),
        ("id_namespace", "9".into()),
    ];
    Ok(Fabric {
        contract: CONTRACT.into(),
        srs: Srs {
            srs_name: format!("ARWEN LAEA {name}"),
            srs_id,
            organization: "ARWEN".into(),
            organization_coordsys_id: srs_id,
            definition: wkt,
            description: Some(format!("Lambert Azimuthal Equal Area on WGS84 centred at {lat0}N {lon0}E")),
        },
        contents: BTreeMap::new(),
        divides,
        flowpaths: vec![],
        nexus: vec![],
        network: vec![],
        flowpath_attributes: vec![],
        hydrolocations: vec![],
        lakes: vec![],
        model_attributes: None,
        fabric_metadata: Some(meta.iter().enumerate().map(|(i, (k, v))| MetadataRow { fid: i as i64 + 1, key: Some(k.to_string()), value: Some(v.clone()) }).collect()),
    })
}
