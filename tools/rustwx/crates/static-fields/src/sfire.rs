//! Refined SFIRE static fields from provenance-bound terrain and fuel rasters.
//! Fuel uses nearest source pixels, terrain uses the four surrounding pixels.
//! Terrain is rounded to WRF REAL before the centered fine-grid derivatives.

use crate::error::{Result, StaticError};
use crate::highres::BoundRasterSpec;
use crate::projection::{GridSpec, ProjectedGrid};
use crate::raster::{geotiff::TiffReader, transform_points, Crs};
use crate::types::{Field, FieldSet, Grid2};
use std::collections::BTreeMap;
use sha2::{Digest,Sha256};

const FIELD_NAMES:[&str;6]=["DZDXF","DZDYF","FXLAT","FXLONG","NFUEL_CAT","ZSF"];

#[derive(Debug,Clone,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FireStaticMetadata {
    pub schema:String,
    pub grid_spec:GridSpec,
    pub sr_x:usize,
    pub sr_y:usize,
    pub fine_shape:[usize;2],
    pub fine_dx_m:f64,
    pub fine_dy_m:f64,
    pub source_sha256:BTreeMap<String,String>,
    pub field_sha256:BTreeMap<String,String>,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub observed_initialization:Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FireStaticLoadRequest {
    pub path:std::path::PathBuf,
    #[serde(default)]
    pub expected_sha256:Option<String>,
    #[serde(default)]
    pub expected_grid_spec:Option<GridSpec>,
}

/// Configuration values cross this seam unchanged. Rust resolves the root
/// anchor and each WPS parent/start/refinement placement.
#[derive(Debug,Clone,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FireGridDomain {
    pub grid_id:i64,pub parent_id:i64,pub i_parent_start:i64,
    pub j_parent_start:i64,pub parent_grid_ratio:i64,
    pub nx:i64,pub ny:i64,pub dx:f64,pub dy:f64,
}

#[derive(Debug,Clone,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FireGridProjection {
    pub map_proj:crate::projection::ProjectionKind,
    pub ref_lat:f64,pub ref_lon:f64,pub truelat1:f64,
    pub truelat2:f64,pub stand_lon:f64,
}

#[derive(Debug,Clone,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FireGridRequest {
    pub projection:FireGridProjection,pub domains:Vec<FireGridDomain>,
}

pub fn experiment_grids(request:&FireGridRequest) -> Result<BTreeMap<i64,GridSpec>> {
    let mut grids:BTreeMap<i64,ProjectedGrid>=BTreeMap::new();
    for domain in &request.domains {
        if domain.grid_id<=0 || grids.contains_key(&domain.grid_id) || domain.nx<1 || domain.ny<1
            || !domain.dx.is_finite() || !domain.dy.is_finite() || domain.dx<=0. || domain.dy<=0. {
            return Err(invalid("Fire experiment grid needs unique positive IDs, nonempty cells and positive finite spacing"));
        }
        let e_we=domain.nx.checked_add(1).ok_or_else(||invalid("Fire experiment x size overflows"))?;
        let e_sn=domain.ny.checked_add(1).ok_or_else(||invalid("Fire experiment y size overflows"))?;
        let grid=if domain.parent_id==0 {
            if !grids.is_empty() {return Err(invalid("Fire experiment has more than one root grid"));}
            let p=&request.projection;
            ProjectedGrid::new(GridSpec {kind:p.map_proj,ref_lat:p.ref_lat,ref_lon:p.ref_lon,
                truelat1:p.truelat1,truelat2:p.truelat2,stand_lon:p.stand_lon,
                dx:domain.dx,dy:domain.dy,e_we,e_sn,known_x:e_we as f64/2.,known_y:e_sn as f64/2.,
                moad_cen_lat:p.ref_lat,moad_cen_lon:p.ref_lon,lat_deg:vec![],lon0_deg:0.,dlon_deg:0.})?
        } else {
            let parent=grids.get(&domain.parent_id).ok_or_else(||invalid("Fire experiment parent must precede its child"))?;
            if domain.parent_grid_ratio<1 || domain.i_parent_start<1 || domain.j_parent_start<1
                || (domain.i_parent_start-1) as f64+domain.nx as f64/domain.parent_grid_ratio as f64>parent.spec.e_we as f64-1.
                || (domain.j_parent_start-1) as f64+domain.ny as f64/domain.parent_grid_ratio as f64>parent.spec.e_sn as f64-1. {
                return Err(invalid("Fire experiment nest leaves its parent grid"));
            }
            parent.nest(domain.i_parent_start,domain.j_parent_start,domain.parent_grid_ratio,
                e_we,e_sn,Some(domain.dx),Some(domain.dy))?
        };
        grids.insert(domain.grid_id,grid);
    }
    if grids.is_empty() {return Err(invalid("Fire experiment has no atmospheric grid"));}
    Ok(grids.into_iter().map(|(id,grid)|(id,grid.spec)).collect())
}

fn grid_location_matches(actual:&GridSpec,expected:&GridSpec) -> Result<bool> {
    let mut normalized=actual.clone();
    // Nest reference coordinates are derived through host projection math.
    // Every supplied parameter and grid dimension remains exact.
    normalized.ref_lat=expected.ref_lat;normalized.ref_lon=expected.ref_lon;
    if normalized!=*expected {return Ok(false);}
    let grid=ProjectedGrid::new(expected.clone())?;
    let (x,y)=grid.latlon_to_ij(actual.ref_lat,actual.ref_lon);
    Ok(x.is_finite() && y.is_finite() && (x-expected.known_x).abs()<=1e-3
        && (y-expected.known_y).abs()<=1e-3)
}

fn field_digest(field:&Field) -> String {
    let mut hash=Sha256::new();
    for value in field.data() {hash.update(value.to_le_bytes());}
    format!("{:x}",hash.finalize())
}

fn metadata(request:&FireStaticRequest,fields:&FieldSet) -> FireStaticMetadata {
    let (_,ny,nx)=fields.fields["ZSF"].dims();
    FireStaticMetadata {schema:"gpuwm-sfire-static-bundle-v1".into(),grid_spec:request.grid_spec.clone(),
        sr_x:request.sr_x,sr_y:request.sr_y,fine_shape:[ny,nx],
        fine_dx_m:((request.grid_spec.dx as f32)/(request.sr_x as f32)) as f64,
        fine_dy_m:((request.grid_spec.dy as f32)/(request.sr_y as f32)) as f64,
        source_sha256:[("terrain".into(),request.terrain.raster.sha256.clone()),
            ("fuel".into(),request.fuel.raster.sha256.clone())].into(),
        observed_initialization:fields.coverage_reports.get("SFIRE_OBSERVED").and_then(|text|serde_json::from_str(text).ok()),
        field_sha256:fields.fields.iter().map(|(name,field)|(name.clone(),field_digest(field))).collect()}
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct FireRasterSpec {
    #[serde(flatten)]
    pub raster: BoundRasterSpec,
    /// A public WCS request uses a fixed geographic pixel-edge lattice.
    /// Only transport roundoff within one millionth of a pixel is removed.
    #[serde(default)]
    pub lattice_resolution_degrees: Option<f64>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct FireStaticRequest {
    pub grid_spec: GridSpec,
    pub sr_x: usize,
    pub sr_y: usize,
    pub terrain: FireRasterSpec,
    pub fuel: FireRasterSpec,
    #[serde(default)]
    pub observed_perimeter:Option<FirePerimeterRequest>,
    #[serde(default)]
    pub output: Option<std::path::PathBuf>,
}

/// Observed geometry has no burn-age measurement. The caller must declare
/// the assumed age explicitly before it becomes an ignition-time field.
#[derive(Debug,Clone,serde::Deserialize,serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct FirePerimeterRequest {
    pub grid_spec:GridSpec,
    pub sr_x:usize,
    pub sr_y:usize,
    pub geojson:std::path::PathBuf,
    pub source_contract:std::path::PathBuf,
    pub observed_utc_epoch_ms:i64,
    pub model_elapsed_seconds:f64,
    pub assumed_burn_age_seconds:f64,
    #[serde(default)]
    pub output:Option<std::path::PathBuf>,
}

type FireRing=Vec<[f64;2]>;
type FirePolygon=Vec<FireRing>;

fn perimeter_rings(geometry:&serde_json::Value,grid:&ProjectedGrid) -> Result<Vec<FirePolygon>> {
    use serde_json::Value;
    let ring=|v:&Value|->Result<FireRing> {
        let points=v.as_array().ok_or_else(||invalid("Observed fire ring is not an array"))?;
        let mut result=Vec::with_capacity(points.len());
        for p in points {
            let lon=p.get(0).and_then(Value::as_f64).ok_or_else(||invalid("Observed fire longitude is missing"))?;
            let lat=p.get(1).and_then(Value::as_f64).ok_or_else(||invalid("Observed fire latitude is missing"))?;
            if !lon.is_finite() || !lat.is_finite() || !(-180.0..=180.0).contains(&lon) || !(-90.0..=90.0).contains(&lat) {
                return Err(invalid("Observed fire coordinates leave finite EPSG:4326"));
            }
            let (x,y)=grid.latlon_to_ij(lat,lon);
            if !x.is_finite() || !y.is_finite() {return Err(invalid("Observed fire ring leaves the atmospheric projection"));}
            result.push([(x-0.5)*grid.spec.dx,(y-0.5)*grid.spec.dy]);
        }
        if result.len()<4 || result.first()!=result.last() {return Err(invalid("Observed fire ring needs three vertices and explicit closure"));}
        Ok(result)
    };
    let polygon=|v:&Value|->Result<FirePolygon> {
        let rings=v.as_array().ok_or_else(||invalid("Observed fire polygon is not an array"))?;
        if rings.is_empty() {return Err(invalid("Observed fire polygon has no outer ring"));}
        rings.iter().map(&ring).collect()
    };
    let coordinates=geometry.get("coordinates").ok_or_else(||invalid("Observed fire geometry lacks coordinates"))?;
    match geometry.get("type").and_then(Value::as_str) {
        Some("Polygon")=>Ok(vec![polygon(coordinates)?]),
        Some("MultiPolygon")=>coordinates.as_array().ok_or_else(||invalid("Observed fire multipolygon is not an array"))?
            .iter().map(polygon).collect(),
        _=>Err(invalid("Observed fire geometry must be Polygon or MultiPolygon")),
    }
}

fn in_ring(point:[f64;2],ring:&FireRing) -> bool {
    let mut inside=false;
    for pair in ring.windows(2) {
        let [a,b]=[pair[0],pair[1]];
        if (a[1]>point[1])!=(b[1]>point[1]) {
            let crossing=(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0];
            if point[0]<crossing {inside=!inside;}
        }
    }
    inside
}

fn perimeter_distance(point:[f64;2],polygons:&[FirePolygon]) -> f64 {
    let inside=polygons.iter().any(|poly|in_ring(point,&poly[0]) && !poly[1..].iter().any(|ring|in_ring(point,ring)));
    let mut distance=f64::INFINITY;
    for ring in polygons.iter().flatten() {
        for pair in ring.windows(2) {
            let a=pair[0];let b=pair[1];let dx=b[0]-a[0];let dy=b[1]-a[1];
            let length2=dx*dx+dy*dy;
            let fraction=if length2==0. {0.} else {((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length2}.clamp(0.,1.);
            distance=distance.min((point[0]-a[0]-fraction*dx).hypot(point[1]-a[1]-fraction*dy));
        }
    }
    if inside {-distance} else {distance}
}

/// Native source validation, coordinate transform and signed-distance raster.
/// Positive is unburned; outer-ring interior is negative and holes are positive.
pub fn observed_perimeter(request:&FirePerimeterRequest) -> Result<FieldSet> {
    use serde_json::{Value,json};
    if !request.model_elapsed_seconds.is_finite() || !request.assumed_burn_age_seconds.is_finite()
        || request.assumed_burn_age_seconds<0. || request.model_elapsed_seconds<0. {
        return Err(invalid("Observed fire initialization needs finite nonnegative model time and explicitly assumed burn age"));
    }
    let (lat,_) = coordinates(&request.grid_spec,request.sr_x,request.sr_y,0)?;
    let contract:Value=serde_json::from_slice(&std::fs::read(&request.source_contract)?).map_err(|e|invalid(e.to_string()))?;
    let bytes=std::fs::read(&request.geojson)?;
    let digest=format!("{:x}",Sha256::digest(&bytes));
    if contract.get("input_sha256").and_then(Value::as_str)!=Some(digest.as_str())
        || contract.get("coordinate_reference_system").and_then(Value::as_str)!=Some("EPSG:4326")
        || contract.get("time_format").and_then(Value::as_str)!=Some("unix_ms") {
        return Err(invalid("Observed fire source contract must match its bytes, EPSG:4326 and UTC unix_ms"));
    }
    let key=contract.get("time_property").and_then(Value::as_str).ok_or_else(||invalid("Observed fire source contract lacks timestamp property"))?;
    let data:Value=serde_json::from_slice(&bytes).map_err(|e|invalid(e.to_string()))?;
    if data.get("type").and_then(Value::as_str)!=Some("FeatureCollection") {return Err(invalid("Observed fire source is not a GeoJSON FeatureCollection"));}
    let features=data.get("features").and_then(Value::as_array).ok_or_else(||invalid("Observed fire source lacks features"))?;
    let selected:Vec<_>=features.iter().filter(|f|f.get("properties").and_then(|v|v.get(key)).and_then(Value::as_i64)==Some(request.observed_utc_epoch_ms)).collect();
    if selected.len()!=1 {return Err(invalid("Observed fire timestamp must select exactly one captured geometry"));}
    let geometry=selected[0].get("geometry").ok_or_else(||invalid("Observed fire feature lacks geometry"))?;
    let grid=ProjectedGrid::new(request.grid_spec.clone())?;
    let polygons=perimeter_rings(geometry,&grid)?;
    let (ny,nx)=(lat.ny,lat.nx);
    let dx=(request.grid_spec.dx as f32/request.sr_x as f32) as f64;
    let dy=(request.grid_spec.dy as f32/request.sr_y as f32) as f64;
    let mut lfn=Grid2::filled(ny,nx,0.);let mut tign=Grid2::filled(ny,nx,1.0e10);
    let ignition=request.model_elapsed_seconds-request.assumed_burn_age_seconds;
    let mut burnt=0_usize;
    for j in 0..ny {for i in 0..nx {
        let distance=perimeter_distance([(i as f64+0.5)*dx,(j as f64+0.5)*dy],&polygons);
        if !distance.is_finite() {return Err(invalid("Observed fire signed distance is nonfinite"));}
        lfn.set(j,i,distance);
        if distance<=0. {tign.set(j,i,ignition);burnt+=1;}
    }}
    if burnt==0 || burnt==nx*ny {return Err(invalid("Observed perimeter must contain burned and unburned cells within this fire grid"));}
    let mut fields=FieldSet {fields:[("LFN_HIST".into(),Field::Plane(lfn)),("HISTORICAL_TIGN".into(),Field::Plane(tign))].into(),..Default::default()};
    let metadata=json!({"schema":"gpuwm-sfire-observed-initialization-v1","source_sha256":digest,
            "observed_utc_epoch_ms":request.observed_utc_epoch_ms,"model_elapsed_seconds":request.model_elapsed_seconds,
            "burn_age_policy":"uniform_explicit_assumption","assumed_burn_age_seconds":request.assumed_burn_age_seconds,
            "fine_shape":[ny,nx],"fine_dx_m":dx,"fine_dy_m":dy,"burned_cell_count":burnt,
            "cell_center_burned_area_m2":burnt as f64*dx*dy,"source_contract":contract});
    fields.coverage_reports.insert("SFIRE_OBSERVED".into(),serde_json::to_string(&metadata).map_err(|e|invalid(e.to_string()))?);
    if let Some(path)=&request.output {
        crate::npz::write_deterministic_npz_extra(path,&fields,&[("PERIMETER.json".into(),serde_json::to_vec(&metadata).map_err(|e|invalid(e.to_string()))?)].into())?;
        std::fs::write(path.with_extension("json"),serde_json::to_vec_pretty(&metadata).map_err(|e|invalid(e.to_string()))?)?;
    }
    Ok(fields)
}

fn invalid(message: impl Into<String>) -> StaticError {
    StaticError::Invalid(message.into())
}

fn coordinates(spec: &GridSpec, sr_x: usize, sr_y: usize, halo: usize)
    -> Result<(Grid2, Grid2)> {
    if sr_x == 0 || sr_y == 0 || spec.e_we < 2 || spec.e_sn < 2 {
        return Err(invalid("SFIRE grid needs positive refinement and nonempty atmospheric cells"));
    }
    let grid = ProjectedGrid::new(spec.clone())?;
    let nx = (spec.e_we as usize - 1).checked_mul(sr_x)
        .and_then(|n| n.checked_add(2*halo)).ok_or_else(|| invalid("SFIRE x size overflows"))?;
    let ny = (spec.e_sn as usize - 1).checked_mul(sr_y)
        .and_then(|n| n.checked_add(2*halo)).ok_or_else(|| invalid("SFIRE y size overflows"))?;
    let mut lat=Grid2::filled(ny,nx,0.0);
    let mut lon=Grid2::filled(ny,nx,0.0);
    for j in 0..ny {
        for i in 0..nx {
            let x=0.5+(i as f64-halo as f64+0.5)/sr_x as f64;
            let y=0.5+(j as f64-halo as f64+0.5)/sr_y as f64;
            let (la,lo)=grid.ij_to_latlon(x,y);
            if !la.is_finite() || !lo.is_finite() {
                return Err(invalid("SFIRE projection produces nonfinite fine-grid coordinates"));
            }
            lat.set(j,i,la);lon.set(j,i,lo);
        }
    }
    Ok((lat,lon))
}

/// Geographic bounds including one fine-cell stencil halo.
pub fn bounds(spec: &GridSpec, sr_x: usize, sr_y: usize) -> Result<[f64;4]> {
    let (lat,lon)=coordinates(spec,sr_x,sr_y,2)?;
    let min=|v:&[f64]| v.iter().copied().fold(f64::INFINITY,f64::min);
    let max=|v:&[f64]| v.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    let result=[min(&lon.data),min(&lat.data),max(&lon.data),max(&lat.data)];
    if result[2]-result[0]>=180.0 {
        return Err(invalid("SFIRE acquisition rectangle crosses the longitude cut; use explicit bound rasters covering the fire grid"));
    }
    Ok(result)
}

/// Fixed geographic source lattice with two pixels for interpolation support.
pub fn coverage_bounds(bounds:[f64;4],resolution:f64) -> Result<[f64;4]> {
    projected_coverage_bounds(bounds,resolution,"EPSG:4326",0.)
}

/// Project an acquisition envelope into the route's declared source CRS and
/// snap it to native pixel edges before asking the coverage service.
pub fn projected_coverage_bounds(bounds:[f64;4],resolution:f64,source_crs:&str,pixel_edge_offset:f64) -> Result<[f64;4]> {
    if !resolution.is_finite() || resolution<=0. || bounds.iter().any(|v|!v.is_finite())
        || bounds[0]>=bounds[2] || bounds[1]>=bounds[3] || !pixel_edge_offset.is_finite() {
        return Err(invalid("SFIRE coverage bounds need a finite geographic rectangle and positive source resolution"));
    }
    let crs=if source_crs.eq_ignore_ascii_case("EPSG:5070") {
        Crs::AlbersConusNad83 {lat_1:29.5,lat_2:45.5,lat_0:23.,lon_0:-96.,false_easting:0.,false_northing:0.}
    } else {Crs::parse_override(source_crs)?};
    let mut x=Vec::with_capacity(164);let mut y=Vec::with_capacity(164);
    for p in 0..=40 {
        let fraction=p as f64/40.;let lon=bounds[0]+fraction*(bounds[2]-bounds[0]);
        let lat=bounds[1]+fraction*(bounds[3]-bounds[1]);
        x.extend([lon,lon,bounds[0],bounds[2]]);y.extend([bounds[1],bounds[3],lat,lat]);
    }
    transform_points(&Crs::Geographic,&crs,&mut x,&mut y)?;
    if x.iter().chain(&y).any(|value|!value.is_finite()) {
        return Err(invalid("SFIRE acquisition envelope leaves the source projection"));
    }
    let min=|v:&[f64]|v.iter().copied().fold(f64::INFINITY,f64::min);
    let max=|v:&[f64]|v.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    let lo=|value:f64| ((value-pixel_edge_offset)/resolution).floor()*resolution+pixel_edge_offset-2.*resolution;
    let hi=|value:f64| ((value-pixel_edge_offset)/resolution).ceil()*resolution+pixel_edge_offset+2.*resolution;
    Ok([lo(min(&x)),lo(min(&y)),hi(max(&x)),hi(max(&y))])
}

/// WPS geogrid four_pt arithmetic in REAL precision. Pixel fractions are
/// derived before clipping the source window so moving the window cannot
/// change the interpolation weights of the same ground point.
fn four_point(u:f32,v:f32,a:f32,b:f32,c:f32,d:f32) -> f32 {
    if u==0.0 && v==0.0 {return a;}
    if u==0.0 {return a*(1.0-v)+c*v;}
    if v==0.0 {return a*(1.0-u)+b*u;}
    v*(c*(1.0-u)+d*u)+(1.0-v)*(a*(1.0-u)+b*u)
}

/// Read only the raster window reached by the refined grid and its stencil.
fn sample(source: &FireRasterSpec,lat:&Grid2,lon:&Grid2,bilinear:bool,allow_missing:bool) -> Result<Grid2> {
    let bound=&source.raster;
    bound.verify()?;
    let mut reader=TiffReader::open(&bound.path)?;
    let crs=reader.crs.clone().or(bound.crs_override.as_deref()
        .map(Crs::parse_override).transpose()?).ok_or_else(|| invalid("SFIRE source raster has no CRS"))?;
    let mut x=lon.data.clone();let mut y=lat.data.clone();
    transform_points(&Crs::Geographic,&crs,&mut x,&mut y)?;
    let t=reader.transform;
    let determinant=t[0]*t[4]-t[1]*t[3];
    if !determinant.is_finite() || determinant==0.0 {
        return Err(invalid("SFIRE source raster has a singular affine transform"));
    }
    let mut cols=Vec::with_capacity(x.len());let mut rows=Vec::with_capacity(y.len());
    if let Some(resolution)=source.lattice_resolution_degrees {
        if !matches!(crs,Crs::Geographic) || !resolution.is_finite() || resolution<=0.0 {
            return Err(invalid("SFIRE WCS source lattice requires a geographic raster and finite positive resolution"));
        }
        let origin_x=(t[2]/resolution).round();let origin_y=(t[5]/resolution).round();
        if ((t[0]/resolution)-1.).abs()>1e-6 || ((t[4]/resolution)+1.).abs()>1e-6
            || (t[1]/resolution).abs()>1e-6 || (t[3]/resolution).abs()>1e-6
            || ((t[2]/resolution)-origin_x).abs()>1e-6
            || ((t[5]/resolution)-origin_y).abs()>1e-6 {
            return Err(invalid("SFIRE WCS raster geometry differs from the acquisition lattice by more than transport roundoff"));
        }
        for (px,py) in x.iter().zip(&y) {
            cols.push(px/resolution-origin_x);rows.push(origin_y-py/resolution);
        }
    } else {
        for (px,py) in x.iter().zip(&y) {
            let dx=px-t[2];let dy=py-t[5];
            cols.push((t[4]*dx-t[1]*dy)/determinant);
            rows.push((-t[3]*dx+t[0]*dy)/determinant);
        }
    }
    if cols.iter().chain(&rows).any(|v|!v.is_finite()) {
        return Err(invalid("SFIRE source projection produces nonfinite pixel coordinates"));
    }
    let shift=if bilinear {0.5} else {0.0};
    let cmin=cols.iter().map(|v|(v-shift).floor()).fold(f64::INFINITY,f64::min);
    let rmin=rows.iter().map(|v|(v-shift).floor()).fold(f64::INFINITY,f64::min);
    let cmax=cols.iter().map(|v|if bilinear {(v-shift).ceil()} else {(v-shift).floor()}).fold(f64::NEG_INFINITY,f64::max);
    let rmax=rows.iter().map(|v|if bilinear {(v-shift).ceil()} else {(v-shift).floor()}).fold(f64::NEG_INFINITY,f64::max);
    if !cmin.is_finite() || !rmin.is_finite() || cmin<0.0 || rmin<0.0 || cmax>=reader.width as f64 || rmax>=reader.height as f64 {
        return Err(invalid(format!("SFIRE {} source does not cover the fine grid and terrain stencil",if bilinear {"terrain"} else {"fuel"})));
    }
    let c0=cmin as usize;let r0=rmin as usize;
    let width=cmax as usize-c0+1;let height=rmax as usize-r0+1;
    let raster=reader.read_window(c0,r0,width,height,Some(crs),bound.nodata_override,bound.scale_factor)?;
    let mut out=Grid2::filled(lat.ny,lat.nx,f64::NAN);
    for p in 0..out.data.len() {
        let c=cols[p]-c0 as f64-shift;let r=rows[p]-r0 as f64-shift;
        let i=c.floor() as usize;let j=r.floor() as usize;
        let value=if bilinear {
            let u=(c-i as f64) as f32;let v=(r-j as f64) as f32;
            let i1=if u==0.0 {i} else {i+1};let j1=if v==0.0 {j} else {j+1};
            let a=raster.values[j*width+i] as f32;let b=raster.values[j*width+i1] as f32;
            let c=raster.values[j1*width+i] as f32;let d=raster.values[j1*width+i1] as f32;
            four_point(u,v,a,b,c,d) as f64
        } else {raster.values[j*width+i]};
        if !value.is_finite() && !allow_missing {
            return Err(invalid(format!("SFIRE {} source has missing data at fine-grid cell {}",if bilinear {"terrain"} else {"fuel"},p)));
        }
        out.data[p]=value;
    }
    Ok(out)
}

/// LANDFIRE's NoData value, which its WCS GeoTIFFs carry as a pixel value
/// (the coverage's offshore and out-of-CONUS cells).
const LANDFIRE_NODATA: f64 = -9999.0;

fn fuel_code(value:f64) -> bool {
    if value.fract()!=0.0 {return false;}
    let n=value as i64;
    (1..=14).contains(&n)||(91..=99).contains(&n)||(101..=109).contains(&n)
        ||(121..=124).contains(&n)||(141..=149).contains(&n)||(161..=165).contains(&n)
        ||(181..=189).contains(&n)||(201..=204).contains(&n)
}

/// Construct the WRF fire-grid fields on the same atmospheric-cell footprint.
pub fn build(request:&FireStaticRequest) -> Result<FieldSet> {
    let spec=&request.grid_spec;
    if !spec.dx.is_finite() || !spec.dy.is_finite() || spec.dx<=0.0 || spec.dy<=0.0 {
        return Err(invalid("SFIRE spacing must be finite and positive to define terrain gradients"));
    }
    let (lat_h,lon_h)=coordinates(spec,request.sr_x,request.sr_y,1)?;
    let mut fuel_h=sample(&request.fuel,&lat_h,&lon_h,false,true)?;
    let mut terrain=sample(&request.terrain,&lat_h,&lon_h,true,true)?;
    // LANDFIRE covers the land and a coastal water margin (code 98); beyond
    // that margin the raster holds its NoData value (-9999), and a coastal
    // fire domain always reaches it.  A fuel cell with no source value is
    // open water only where the terrain source agrees there is no land (no
    // value, or at or below sea level).  Fuel missing over land stays a
    // refusal: an unknown fuel there would burn as whatever we guessed.
    for p in 0..fuel_h.data.len() {
        let fuel=fuel_h.data[p];
        if fuel.is_finite() && fuel!=LANDFIRE_NODATA {continue;}
        let height=terrain.data[p];
        if !height.is_finite() || height<=0.0 {fuel_h.data[p]=98.0;}
        else {
            return Err(invalid(format!("SFIRE fuel source has missing data at land fire cell or stencil cell {p} (terrain {height} m)")));
        }
    }
    if let Some((p,value))=fuel_h.data.iter().enumerate().find(|(_,v)|!fuel_code(**v)) {
        return Err(invalid(format!("NFUEL_CAT contains unmapped fuel code {value} at fire cell or stencil cell {p}")));
    }
    for (p,value) in terrain.data.iter_mut().enumerate() {
        if !value.is_finite() {
            // LANDFIRE 98 explicitly identifies open water. Unknown source
            // gaps and nonwater fuel classes must never become flat terrain.
            if fuel_h.data[p]==98.0 {*value=0.0;}
            else {return Err(invalid(format!("SFIRE terrain source has missing data at nonwater fire cell or stencil cell {p}")));}
        }
    }
    let ny=lat_h.ny-2;let nx=lat_h.nx-2;
    let mut lat=Grid2::filled(ny,nx,0.0);let mut lon=lat.clone();
    let mut zsf=lat.clone();let mut gx=lat.clone();let mut gy=lat.clone();let mut fuel=lat.clone();
    // WRF DX/DY and the refined spacing are REAL, so division follows the
    // atmospheric spacing's single-precision rounding.
    let fdx=(spec.dx as f32)/(request.sr_x as f32);
    let fdy=(spec.dy as f32)/(request.sr_y as f32);
    if !fdx.is_finite() || !fdy.is_finite() || fdx<=0.0 || fdy<=0.0 {
        return Err(invalid("SFIRE spacing cannot be represented as positive WRF REAL for terrain gradients"));
    }
    for j in 0..ny {for i in 0..nx {
        lat.set(j,i,lat_h.at(j+1,i+1));lon.set(j,i,lon_h.at(j+1,i+1));
        zsf.set(j,i,terrain.at(j+1,i+1) as f32 as f64);
        fuel.set(j,i,fuel_h.at(j+1,i+1));
        gx.set(j,i,((terrain.at(j+1,i+2) as f32-terrain.at(j+1,i) as f32)/(2.0_f32*fdx)) as f64);
        gy.set(j,i,((terrain.at(j+2,i+1) as f32-terrain.at(j,i+1) as f32)/(2.0_f32*fdy)) as f64);
    }}
    let mut fields=FieldSet::default();
    for (name,field) in [("ZSF",zsf),("DZDXF",gx),("DZDYF",gy),("NFUEL_CAT",fuel),("FXLAT",lat),("FXLONG",lon)] {
        fields.fields.insert(name.to_string(),Field::Plane(field));
    }
    if let Some(perimeter)=&request.observed_perimeter {
        if perimeter.sr_x!=request.sr_x || perimeter.sr_y!=request.sr_y
            || serde_json::to_value(&perimeter.grid_spec).unwrap()!=serde_json::to_value(&request.grid_spec).unwrap() {
            return Err(invalid("Observed perimeter grid/refinement differs from the static fire bundle"));
        }
        let observed=observed_perimeter(perimeter)?;
        fields.fields.extend(observed.fields);
        fields.coverage_reports.extend(observed.coverage_reports);
    }
    if let Some(path)=&request.output {
        let bytes=serde_json::to_vec(&metadata(request,&fields)).map_err(|e|invalid(e.to_string()))?;
        crate::npz::write_deterministic_npz_extra(path,&fields,&[("SFIRE.json".into(),bytes)].into())?;
    }
    Ok(fields)
}

/// Read the produced STORED classic ZIP contract, checking every payload CRC
/// and the directory offsets before interpreting any numerical array bytes.
fn read_members(bytes:&[u8]) -> Result<BTreeMap<String,Vec<u8>>> {
    let u16_at=|p:usize|bytes.get(p..p+2).map(|v|u16::from_le_bytes(v.try_into().unwrap()))
        .ok_or_else(||invalid("SFIRE bundle ZIP header is truncated"));
    let u32_at=|p:usize|bytes.get(p..p+4).map(|v|u32::from_le_bytes(v.try_into().unwrap()))
        .ok_or_else(||invalid("SFIRE bundle ZIP header is truncated"));
    let end=bytes.len().checked_sub(22).ok_or_else(||invalid("SFIRE bundle ZIP end record is missing"))?;
    if u32_at(end)?!=0x0605_4b50 || u16_at(end+4)?!=0 || u16_at(end+6)?!=0 || u16_at(end+20)?!=0
        || u16_at(end+8)?!=u16_at(end+10)? {
        return Err(invalid("SFIRE bundle needs the produced single-disk classic ZIP end record"));
    }
    let central=u32_at(end+16)? as usize;
    if central.checked_add(u32_at(end+12)? as usize)!=Some(end) {return Err(invalid("SFIRE bundle ZIP directory extent is inconsistent"));}
    let mut members=BTreeMap::new();let mut headers=Vec::new();let mut p=0;
    while p<central {
        if u32_at(p)?!=0x0403_4b50 || u16_at(p+6)?!=0 || u16_at(p+8)?!=0 || u32_at(p+18)?!=u32_at(p+22)? {
            return Err(invalid("SFIRE bundle needs the produced uncompressed ZIP members without descriptors"));
        }
        let start=p.checked_add(30).and_then(|v|v.checked_add(u16_at(p+26).ok()? as usize))
            .and_then(|v|v.checked_add(u16_at(p+28).ok()? as usize)).ok_or_else(||invalid("SFIRE ZIP member extent overflows"))?;
        let stop=start.checked_add(u32_at(p+22)? as usize).ok_or_else(||invalid("SFIRE ZIP member extent overflows"))?;
        if stop>central {return Err(invalid("SFIRE ZIP member extends into its directory"));}
        let name=std::str::from_utf8(bytes.get(p+30..p+30+u16_at(p+26)? as usize).ok_or_else(||invalid("SFIRE ZIP name is truncated"))?)
            .map_err(|_|invalid("SFIRE ZIP member name is not UTF-8"))?.to_string();
        let data=&bytes[start..stop];let crc=u32_at(p+14)?;
        if crate::npz::crc32(data)!=crc {return Err(invalid(format!("SFIRE bundle member {name} fails CRC validation")));}
        if members.insert(name.clone(),data.to_vec()).is_some() {return Err(invalid(format!("SFIRE bundle repeats member {name}")));}
        headers.push((name,p,crc,data.len()));p=stop;
    }
    if p!=central || headers.len()!=u16_at(end+10)? as usize {return Err(invalid("SFIRE ZIP member count or directory start is inconsistent"));}
    for (name,offset,crc,size) in headers {
        if u32_at(p)?!=0x0201_4b50 || u16_at(p+8)?!=0 || u16_at(p+10)?!=0
            || u32_at(p+16)?!=crc || u32_at(p+20)? as usize!=size || u32_at(p+24)? as usize!=size
            || u32_at(p+42)? as usize!=offset || u16_at(p+34)?!=0 {
            return Err(invalid("SFIRE ZIP directory disagrees with its array payload headers"));
        }
        let name_end=p+46+u16_at(p+28)? as usize;
        if bytes.get(p+46..name_end)!=Some(name.as_bytes()) {return Err(invalid("SFIRE ZIP directory member name differs from its payload"));}
        p=name_end+u16_at(p+30)? as usize+u16_at(p+32)? as usize;
    }
    if p!=end {return Err(invalid("SFIRE ZIP directory has unclaimed bytes"));}
    Ok(members)
}

fn read_plane(bytes:&[u8],ny:usize,nx:usize,name:&str) -> Result<Field> {
    if bytes.get(..8)!=Some(b"\x93NUMPY\x01\x00") {return Err(invalid(format!("SFIRE {name} needs produced NPY 1.0 data")));}
    let raw=bytes.get(8..10).ok_or_else(||invalid("SFIRE NPY header is truncated"))?;
    let start=10+u16::from_le_bytes(raw.try_into().unwrap()) as usize;
    let header=std::str::from_utf8(bytes.get(10..start).ok_or_else(||invalid("SFIRE NPY header is truncated"))?).map_err(|_|invalid("SFIRE NPY header is not UTF-8"))?;
    let expected=format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({ny}, {nx}), }}");
    if header.trim()!=expected {return Err(invalid(format!("SFIRE {name} dtype, order or shape differs from the grid/refinement metadata")));}
    let count=ny.checked_mul(nx).ok_or_else(||invalid("SFIRE field size overflows"))?;
    if count.checked_mul(8).and_then(|n|n.checked_add(start))!=Some(bytes.len()) {return Err(invalid(format!("SFIRE {name} array byte extent differs from its shape")));}
    let values=bytes[start..].chunks_exact(8).map(|v|f64::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
    if values.iter().any(|v|!v.is_finite()) {return Err(invalid(format!("SFIRE {name} contains missing or nonfinite static values")));}
    Ok(Field::Plane(Grid2 {ny,nx,data:values}))
}

/// Restore six fields word for word, with their portable mesh metadata.
pub fn load(request:&FireStaticLoadRequest) -> Result<(FieldSet,FireStaticMetadata)> {
    let bytes=std::fs::read(&request.path)?;
    if let Some(expected)=&request.expected_sha256 {
        if format!("{:x}",Sha256::digest(&bytes))!=*expected {return Err(invalid("SFIRE bundle hash differs from the expected artifact"));}
    }
    let members=read_members(&bytes)?;
    let json=members.get("SFIRE.json").ok_or_else(||invalid("SFIRE bundle lacks grid/refinement metadata; prepare it with gpuwm-sfire-static"))?;
    let metadata:FireStaticMetadata=serde_json::from_slice(json).map_err(|e|invalid(format!("SFIRE bundle metadata is invalid: {e}")))?;
    let spec=&metadata.grid_spec;
    if let Some(expected)=&request.expected_grid_spec {
        if !grid_location_matches(spec,expected)? {
            return Err(invalid("SFIRE bundle atmospheric grid location or projection differs from the requested stage grid"));
        }
    }
    if metadata.schema!="gpuwm-sfire-static-bundle-v1" || metadata.sr_x==0 || metadata.sr_y==0 || spec.e_we<2 || spec.e_sn<2 {
        return Err(invalid("SFIRE bundle schema or grid/refinement metadata is invalid"));
    }
    let nx=(spec.e_we as usize-1).checked_mul(metadata.sr_x).ok_or_else(||invalid("SFIRE bundle x refinement overflows"))?;
    let ny=(spec.e_sn as usize-1).checked_mul(metadata.sr_y).ok_or_else(||invalid("SFIRE bundle y refinement overflows"))?;
    let fdx=((spec.dx as f32)/(metadata.sr_x as f32)) as f64;
    let fdy=((spec.dy as f32)/(metadata.sr_y as f32)) as f64;
    if metadata.fine_shape!=[ny,nx] || !fdx.is_finite() || !fdy.is_finite() || fdx<=0. || fdy<=0.
        || fdx.to_bits()!=metadata.fine_dx_m.to_bits() || fdy.to_bits()!=metadata.fine_dy_m.to_bits() {
        return Err(invalid("SFIRE bundle fine shape or spacing differs from its atmospheric grid and refinement"));
    }
    ProjectedGrid::new(spec.clone())?;
    let mut names=FIELD_NAMES.to_vec();
    if let Some(observed)=&metadata.observed_initialization {
        use serde_json::Value;
        let hash=observed.get("source_sha256").and_then(Value::as_str).unwrap_or("");
        let valid_time=observed.get("model_elapsed_seconds").and_then(Value::as_f64).is_some_and(|v|v.is_finite() && v>=0.);
        let valid_age=observed.get("assumed_burn_age_seconds").and_then(Value::as_f64).is_some_and(|v|v.is_finite() && v>=0.);
        if observed.get("schema").and_then(Value::as_str)!=Some("gpuwm-sfire-observed-initialization-v1")
            || observed.get("burn_age_policy").and_then(Value::as_str)!=Some("uniform_explicit_assumption")
            || observed.get("observed_utc_epoch_ms").and_then(Value::as_i64).is_none()
            || hash.len()!=64 || !hash.bytes().all(|v|v.is_ascii_hexdigit()) || !valid_time || !valid_age {
            return Err(invalid("SFIRE observed initialization lacks its source hash, UTC timestamp or explicit burn-age assumption"));
        }
        names.extend(["LFN_HIST","HISTORICAL_TIGN"]);
    }
    if members.len()!=names.len()+1 || metadata.field_sha256.len()!=names.len() {
        return Err(invalid("SFIRE bundle requires six static fields and the declared optional observed-perimeter pair"));
    }
    if metadata.source_sha256.len()!=2 || ["terrain","fuel"].iter().any(|name|
        metadata.source_sha256.get(*name).is_none_or(|hash|hash.len()!=64 || !hash.bytes().all(|v|v.is_ascii_hexdigit()))) {
        return Err(invalid("SFIRE bundle requires its captured terrain and fuel source hashes"));
    }
    let mut fields=FieldSet::default();
    for name in names {
        let field=read_plane(members.get(&format!("{name}.npy")).ok_or_else(||invalid(format!("SFIRE bundle lacks {name}")))?,ny,nx,name)?;
        if metadata.field_sha256.get(name)!=Some(&field_digest(&field)) {return Err(invalid(format!("SFIRE {name} words differ from the sealed field hash")));}
        fields.fields.insert(name.into(),field);
    }
    if fields.fields["NFUEL_CAT"].data().iter().any(|value|!fuel_code(*value)) {return Err(invalid("SFIRE bundle contains unmapped fuel categories"));}
    if fields.fields["FXLAT"].data().iter().any(|v|!(-90.0..=90.0).contains(v))
        || fields.fields["FXLONG"].data().iter().any(|v|!(-180.0..=180.0).contains(v)) {return Err(invalid("SFIRE bundle coordinates leave geographic ranges"));}
    let grid=ProjectedGrid::new(spec.clone())?;
    for (index,(lat,lon)) in fields.fields["FXLAT"].data().iter().zip(fields.fields["FXLONG"].data()).enumerate() {
        let (x,y)=grid.latlon_to_ij(*lat,*lon);
        let expected_x=0.5+(index%nx) as f64/metadata.sr_x as f64+0.5/metadata.sr_x as f64;
        let expected_y=0.5+(index/nx) as f64/metadata.sr_y as f64+0.5/metadata.sr_y as f64;
        if !x.is_finite() || !y.is_finite() || (x-expected_x).abs()>1e-3 || (y-expected_y).abs()>1e-3 {
            return Err(invalid("SFIRE fine coordinates differ from the declared atmospheric grid location"));
        }
    }
    Ok((fields,metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declared_experiment_chain_resolves_native_nests_and_checks_location() {
        let request:FireGridRequest=serde_json::from_value(serde_json::json!({
            "projection":{"map_proj":"lambert","ref_lat":40.75,"ref_lon":-119.75,
                "truelat1":30.,"truelat2":60.,"stand_lon":-119.75},
            "domains":[
                {"grid_id":1,"parent_id":0,"i_parent_start":1,"j_parent_start":1,"parent_grid_ratio":1,"nx":360,"ny":640,"dx":3000.,"dy":3000.},
                {"grid_id":2,"parent_id":1,"i_parent_start":270,"j_parent_start":15,"parent_grid_ratio":3,"nx":96,"ny":96,"dx":1000.,"dy":1000.},
                {"grid_id":3,"parent_id":2,"i_parent_start":38,"j_parent_start":38,"parent_grid_ratio":4,"nx":96,"ny":96,"dx":250.,"dy":250.}]})).unwrap();
        let grids=experiment_grids(&request).unwrap();let spec=&grids[&3];
        assert_eq!((grids[&1].known_x,grids[&1].known_y),(180.5,320.5));
        assert_eq!((spec.dx,spec.e_we,spec.known_x),(250.,97,1.));
        assert!((spec.ref_lat-32.59049158227738).abs()<1e-11);
        assert!((spec.ref_lon+116.46850532652721).abs()<1e-11);
        assert!(grid_location_matches(spec,spec).unwrap());
        let grid=ProjectedGrid::new(spec.clone()).unwrap();
        let mut roundoff=spec.clone();let (lat,lon)=grid.ij_to_latlon(1.+0.0005,1.);
        roundoff.ref_lat=lat;roundoff.ref_lon=lon;
        assert!(grid_location_matches(&roundoff,spec).unwrap());
        let (lat,lon)=grid.ij_to_latlon(2.,1.);roundoff.ref_lat=lat;roundoff.ref_lon=lon;
        assert!(!grid_location_matches(&roundoff,spec).unwrap());
        roundoff=spec.clone();roundoff.stand_lon+=0.001;
        assert!(!grid_location_matches(&roundoff,spec).unwrap());
        let mut absent=request.clone();absent.domains.swap(1,2);
        assert!(experiment_grids(&absent).unwrap_err().to_string().contains("precede"));
        let mut outside=request.clone();outside.domains[2].i_parent_start=95;
        assert!(experiment_grids(&outside).unwrap_err().to_string().contains("leaves"));
    }
    #[test]
    fn signed_distance_preserves_polygon_holes_and_separate_islands() {
        let outer=vec![[0.,0.],[10.,0.],[10.,10.],[0.,10.],[0.,0.]];
        let hole=vec![[4.,4.],[6.,4.],[6.,6.],[4.,6.],[4.,4.]];
        let island=vec![[20.,0.],[22.,0.],[22.,2.],[20.,2.],[20.,0.]];
        let geometry=vec![vec![outer,hole],vec![island]];
        for (point,expected) in [([2.,2.],-2.),([5.,5.],1.),([21.,1.],-1.),
            ([12.,5.],2.),([10.,5.],0.),([30.,2.],8.)] {
            assert_eq!(perimeter_distance(point,&geometry),expected);
        }
    }
    const WPS_CASES:[(u32,u32,u32,u32);12]=[
        (0,0,1140690944,1140690944),(1065353216,1065353216,1140787200,1140787200),
        (1048576000,0,1140690944,1140622336),(0,1056964608,1140889600,1140809728),
        (1056964608,1056964608,1140787200,1140705792),
        (1036831952,1060320052,1140889600,1140839118),
        (1039980256,1061417820,1140889600,1140851086),
        (1065353214,872415232,1140416512,1140416512),
        (1048576000,1056964608,1140889600,1140757760),
        (1061158912,1056964608,1140787200,1140653824),
        (1056964608,1048576000,1140416512,1140629760),
        (1056964608,1061158912,1140787200,1140781824),
    ];
    #[test]
    fn compiled_wps_four_point_fixture_is_word_identical() {
        // Unmodified WPS v4.6.0 geogrid interp_module.F, gfortran 15.2.0
        // -O0 -ffp-contract=off. Source SHA-256:
        // cd3bf205f4870fb1deceefefd511bbc950ae97f78b08e43f68b7221333f6f424
        let corners=[507.125,498.75,514.375,510.0625];
        for (u,v,_,expected) in WPS_CASES {
            assert_eq!(four_point(f32::from_bits(u),f32::from_bits(v),
                corners[0],corners[1],corners[2],corners[3]).to_bits(),expected);
        }
        assert_eq!(four_point(0.,0.,507.125,f32::NAN,f32::NAN,f32::NAN),507.125);
        assert!(four_point(0.5,0.5,507.125,f32::NAN,514.375,510.0625).is_nan());
    }
    #[test]
    fn clipped_geotiff_samples_match_native_wps_including_last_pixel_center() {
        use crate::raster::{Raster,geotiff::{self,SampleType}};
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path=std::env::temp_dir().join(format!("sfire-wps-{nonce}.tif"));
        let raster=Raster {ny:2,nx:2,values:vec![507.125,498.75,514.375,510.0625],
            transform:[1.,0.,0.,0.,-1.,2.],crs:Crs::Geographic};
        geotiff::write_band1(&path,&raster,SampleType::F32,None).unwrap();
        let bound=FireRasterSpec {raster:BoundRasterSpec {path:path.clone(),sha256:crate::highres::sha256_file(&path).unwrap(),
            expected_bytes:None,crs_override:None,nodata_override:None,scale_factor:1.},lattice_resolution_degrees:Some(1.)};
        let lat=Grid2 {ny:1,nx:WPS_CASES.len(),data:WPS_CASES.iter().map(|(_,v,_,_)|1.5-f32::from_bits(*v) as f64).collect()};
        let lon=Grid2 {ny:1,nx:WPS_CASES.len(),data:WPS_CASES.iter().map(|(u,_,_,_)|0.5+f32::from_bits(*u) as f64).collect()};
        for (bilinear,slot) in [(false,2),(true,3)] {
            let field=sample(&bound,&lat,&lon,bilinear,false).unwrap();
            for (point,value) in field.data.iter().enumerate() {
                let case=WPS_CASES[point];let expected=if slot==2 {case.2} else {case.3};
                assert_eq!((*value as f32).to_bits(),expected,"point {point}, four_pt {bilinear}");
            }
        }
        println!("deleted owned test artifact {} bytes {}",path.file_name().unwrap().to_string_lossy(),std::fs::metadata(&path).unwrap().len());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn overlapping_wcs_crops_keep_identical_pixels_and_weights() {
        use crate::raster::{Raster,geotiff::{self,SampleType}};
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let full=std::env::temp_dir().join(format!("sfire-wcs-full-{nonce}.tif"));
        let crop=std::env::temp_dir().join(format!("sfire-wcs-crop-{nonce}.tif"));
        let values=(0..16).map(|p|{let j=p/4;let i=p%4;(200+j*3+i*5+j*i*7) as f64}).collect::<Vec<_>>();
        let a=Raster {ny:4,nx:4,values:values.clone(),transform:[1.,0.,0.,0.,-1.,4.],crs:Crs::Geographic};
        let mut b=Raster {ny:2,nx:2,values:vec![values[5],values[6],values[9],values[10]],
            transform:[1.+1e-8,0.,1.+1e-8,0.,-1.-1e-8,3.+1e-8],crs:Crs::Geographic};
        geotiff::write_band1(&full,&a,SampleType::F32,None).unwrap();
        geotiff::write_band1(&crop,&b,SampleType::F32,None).unwrap();
        let bound=|path:&std::path::Path| FireRasterSpec {raster:BoundRasterSpec {path:path.to_owned(),
            sha256:crate::highres::sha256_file(path).unwrap(),expected_bytes:None,
            crs_override:None,nodata_override:None,scale_factor:1.},lattice_resolution_degrees:Some(1.)};
        let lat=Grid2 {ny:1,nx:2,data:vec![2.25,1.75]};let lon=Grid2 {ny:1,nx:2,data:vec![1.75,2.25]};
        for bilinear in [false,true] {
            let first=sample(&bound(&full),&lat,&lon,bilinear,false).unwrap();
            let second=sample(&bound(&crop),&lat,&lon,bilinear,false).unwrap();
            assert_eq!(first.data,second.data);
        }
        b.transform[2]+=1e-3;
        geotiff::write_band1(&crop,&b,SampleType::F32,None).unwrap();
        assert!(sample(&bound(&crop),&lat,&lon,true,false).unwrap_err().to_string().contains("geometry differs"));
        for path in [full,crop] {
            println!("deleted owned test artifact {} bytes {}",path.file_name().unwrap().to_string_lossy(),std::fs::metadata(&path).unwrap().len());
            std::fs::remove_file(path).unwrap();
        }
    }
    #[test]
    fn category_ranges_preserve_raw_nonburnable_and_fuel_codes() {
        assert!(fuel_code(204.));assert!(fuel_code(91.));assert!(fuel_code(13.));
        assert!(!fuel_code(-9999.));assert!(!fuel_code(100.));assert!(!fuel_code(1.5));
    }
    #[test]
    fn acquisition_windows_keep_source_lattice_and_stencil_margin() {
        let a=coverage_bounds([-1.001,0.123,2.005,3.456],0.01).unwrap();
        let b=coverage_bounds([-1.009,0.121,2.001,3.459],0.01).unwrap();
        assert_eq!(a,b);
        assert!((a[0]+1.03).abs()<1e-12);
        assert!((a[3]-3.48).abs()<1e-12);
        let native=projected_coverage_bounds([-123.,38.,-122.,39.],30.,"EPSG:5070",15.).unwrap();
        assert!(native.iter().all(|value|((value-15.)/30.).fract()==0.));
        assert!(native[0]<native[2] && native[1]<native[3]);
    }
    #[test]
    fn independent_refinement_stays_centered_on_atmospheric_cells() {
        let spec=GridSpec {kind:crate::projection::ProjectionKind::Mercator,
            ref_lat:0.,ref_lon:0.,truelat1:0.,truelat2:0.,stand_lon:0.,dx:120.,dy:120.,
            e_we:3,e_sn:3,known_x:1.,known_y:1.,moad_cen_lat:0.,moad_cen_lon:0.,
            lat_deg:vec![],lon0_deg:0.,dlon_deg:0.};
        let (lat,lon)=coordinates(&spec,4,2,0).unwrap();
        assert_eq!((lat.ny,lat.nx),(4,8));
        let factor=crate::EARTH_RADIUS_M*std::f64::consts::PI/180.;
        assert!((lon.at(0,0)*factor+45.).abs()<1e-6);
        assert!((lon.at(0,3)*factor-45.).abs()<1e-6);
        assert!((lat.at(0,0)*factor+30.).abs()<1e-4);
    }
    #[test]
    fn bound_geotiffs_build_refined_slopes_and_deterministic_artifact() {
        use crate::raster::{Raster,geotiff::{self,SampleType}};
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let folder=std::env::temp_dir().join(format!("sfire-static-{nonce}"));
        std::fs::create_dir(&folder).unwrap();
        let terrain_path=folder.join("terrain.tif");let fuel_path=folder.join("fuel.tif");
        let output=folder.join("fire.npz");
        let metres_per_degree=crate::EARTH_RADIUS_M*std::f64::consts::PI/180.;
        let mut terrain=Raster {ny:64,nx:64,values:vec![0.;4096],
            transform:[10./metres_per_degree,0.,-320./metres_per_degree,0.,-10./metres_per_degree,320./metres_per_degree],
            crs:Crs::Geographic};
        for j in 0..64 {for i in 0..64 {
            let x=-320.+(i as f64+0.5)*10.;let y=320.-(j as f64+0.5)*10.;
            terrain.values[j*64+i]=500.+0.1*x+0.2*y;
        }}
        let mut fuel=terrain.clone();fuel.values.fill(201.);
        geotiff::write_band1(&terrain_path,&terrain,SampleType::F32,None).unwrap();
        geotiff::write_band1(&fuel_path,&fuel,SampleType::I16,None).unwrap();
        let bound=|path:&std::path::Path| FireRasterSpec {raster:BoundRasterSpec {path:path.to_owned(),
            sha256:crate::highres::sha256_file(path).unwrap(),expected_bytes:None,
            crs_override:None,nodata_override:None,scale_factor:1.},lattice_resolution_degrees:None};
        let spec=GridSpec {kind:crate::projection::ProjectionKind::Mercator,
            ref_lat:0.,ref_lon:0.,truelat1:0.,truelat2:0.,stand_lon:0.,dx:120.,dy:120.,
            e_we:3,e_sn:3,known_x:1.,known_y:1.,moad_cen_lat:0.,moad_cen_lon:0.,
            lat_deg:vec![],lon0_deg:0.,dlon_deg:0.};
        let request=FireStaticRequest {grid_spec:spec,sr_x:4,sr_y:2,
            terrain:bound(&terrain_path),fuel:bound(&fuel_path),observed_perimeter:None,output:Some(output.clone())};
        let fields=build(&request).unwrap();
        assert_eq!(fields.fields.len(),6);
        for (name,expected) in [("DZDXF",0.1),("DZDYF",0.2),("NFUEL_CAT",201.)] {
            assert_eq!(fields.fields[name].dims(),(1,4,8));
            for value in fields.fields[name].data() {assert!((value-expected).abs()<1e-6,"{name}: {value}");}
        }
        let first=std::fs::read(&output).unwrap();
        build(&request).unwrap();assert_eq!(first,std::fs::read(&output).unwrap());
        let load_request=FireStaticLoadRequest {path:output.clone(),expected_sha256:Some(format!("{:x}",Sha256::digest(&first))),expected_grid_spec:None};
        let (restored,meta)=load(&load_request).unwrap();
        assert_eq!(meta.fine_shape,[4,8]);assert_eq!((meta.sr_x,meta.sr_y),(4,2));
        for name in FIELD_NAMES {assert_eq!(fields.fields[name].data(),restored.fields[name].data());}
        let mut expected=request.grid_spec.clone();expected.ref_lon+=0.01;
        let location=FireStaticLoadRequest {path:output.clone(),expected_sha256:None,expected_grid_spec:Some(expected)};
        assert!(load(&location).unwrap_err().to_string().contains("location or projection"));
        let mut shifted=fields.clone();
        if let Field::Plane(plane)=shifted.fields.get_mut("FXLAT").unwrap() {plane.data[0]+=0.01;}
        let shifted_metadata=metadata(&request,&shifted);
        crate::npz::write_deterministic_npz_extra(&output,&shifted,&[("SFIRE.json".into(),serde_json::to_vec(&shifted_metadata).unwrap())].into()).unwrap();
        let unchecked=FireStaticLoadRequest {path:output.clone(),expected_sha256:None,expected_grid_spec:None};
        assert!(load(&unchecked).unwrap_err().to_string().contains("coordinates differ"));
        let mut corrupt=first.clone();corrupt[180]^=1;
        std::fs::write(&output,&corrupt).unwrap();
        assert!(load(&unchecked).unwrap_err().to_string().contains("CRC"));
        let mut wrong=metadata(&request,&fields);wrong.sr_y=3;
        crate::npz::write_deterministic_npz_extra(&output,&fields,&[("SFIRE.json".into(),serde_json::to_vec(&wrong).unwrap())].into()).unwrap();
        assert!(load(&unchecked).unwrap_err().to_string().contains("shape or spacing"));
        crate::npz::write_deterministic_npz(&output,&fields).unwrap();
        assert!(load(&unchecked).unwrap_err().to_string().contains("lacks grid/refinement metadata"));
        let mut bad=request.clone();bad.fuel.raster.sha256="0".repeat(64);
        assert!(build(&bad).unwrap_err().to_string().contains("hash mismatch"));
        let mut missing_terrain=terrain.clone();missing_terrain.values.fill(f64::NAN);
        geotiff::write_band1(&terrain_path,&missing_terrain,SampleType::F32,Some(-9999.)).unwrap();
        let mut coast=request.clone();coast.terrain=bound(&terrain_path);
        assert!(build(&coast).unwrap_err().to_string().contains("nonwater"));
        fuel.values.fill(98.);
        geotiff::write_band1(&fuel_path,&fuel,SampleType::I16,None).unwrap();
        coast.fuel=bound(&fuel_path);
        let water=build(&coast).unwrap();
        for name in ["ZSF","DZDXF","DZDYF"] {assert!(water.fields[name].data().iter().all(|v|*v==0.));}
        assert!(water.fields["NFUEL_CAT"].data().iter().all(|v|*v==98.));
        // Offshore beyond the LANDFIRE margin: no fuel value and no terrain
        // value is open water, with or without a NoData tag.
        for tag in [Some(-9999.), None] {
            fuel.values.fill(-9999.);
            geotiff::write_band1(&fuel_path,&fuel,SampleType::I16,tag).unwrap();
            coast.fuel=bound(&fuel_path);
            let sea=build(&coast).unwrap();
            assert!(sea.fields["NFUEL_CAT"].data().iter().all(|v|*v==98.));
            for name in ["ZSF","DZDXF","DZDYF"] {assert!(sea.fields[name].data().iter().all(|v|*v==0.));}
        }
        // Missing fuel over land still refuses, naming the land cell.
        geotiff::write_band1(&terrain_path,&terrain,SampleType::F32,None).unwrap();
        coast.terrain=bound(&terrain_path);
        assert!(build(&coast).unwrap_err().to_string().contains("missing data at land fire cell"));
        for path in [terrain_path,fuel_path,output] {
            println!("deleted owned test artifact {} bytes {}",path.file_name().unwrap().to_string_lossy(),std::fs::metadata(&path).unwrap().len());
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_dir(folder).unwrap();
    }
}
