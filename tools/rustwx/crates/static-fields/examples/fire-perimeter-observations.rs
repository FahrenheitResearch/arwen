//! Decode provenance-bound observed perimeters and measure their actual rings.
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use static_fields::raster::{Crs,transform_points};

type Ring=Vec<[f64;2]>;
type Polygon=Vec<Ring>;

fn ring(value:&Value) -> Result<Ring,String> {
    let points=value.as_array().ok_or("perimeter ring is not an array")?;
    let mut result=Vec::with_capacity(points.len());
    for point in points {
        let x=point.get(0).and_then(Value::as_f64).ok_or("longitude is missing")?;
        let y=point.get(1).and_then(Value::as_f64).ok_or("latitude is missing")?;
        if !x.is_finite() || !y.is_finite() || !(-180.0..=180.0).contains(&x) || !(-90.0..=90.0).contains(&y) {
            return Err("observed perimeter leaves finite geographic coordinates".into());
        }
        result.push([x,y]);
    }
    if result.len()<4 || result.first()!=result.last() {
        return Err("observed perimeter needs a closed ring with at least three vertices".into());
    }
    Ok(result)
}

fn polygon(value:&Value) -> Result<Polygon,String> {
    let rings=value.as_array().ok_or("perimeter polygon is not an array")?;
    if rings.is_empty() {return Err("observed perimeter polygon has no outer ring".into());}
    rings.iter().map(ring).collect()
}

fn polygons(geometry:&Value) -> Result<Vec<Polygon>,String> {
    let coordinates=geometry.get("coordinates").ok_or("perimeter has no coordinates")?;
    match geometry.get("type").and_then(Value::as_str) {
        Some("Polygon")=>Ok(vec![polygon(coordinates)?]),
        Some("MultiPolygon")=>coordinates.as_array().ok_or("multipolygon is not an array")?
            .iter().map(polygon).collect(),
        _=>Err("observed perimeter must be a Polygon or MultiPolygon".into()),
    }
}

fn measure(ring:&Ring,crs:&Crs) -> Result<(f64,f64),String> {
    let mut x=ring.iter().map(|point|point[0]).collect::<Vec<_>>();
    let mut y=ring.iter().map(|point|point[1]).collect::<Vec<_>>();
    transform_points(&Crs::Geographic,crs,&mut x,&mut y).map_err(|e|e.to_string())?;
    if x.iter().chain(&y).any(|value|!value.is_finite()) {return Err("observed perimeter leaves analysis projection".into());}
    let ox=x[0];let oy=y[0];let mut twice_area=0.;let mut length=0.;
    for p in 0..x.len()-1 {
        twice_area+=(x[p]-ox)*(y[p+1]-oy)-(x[p+1]-ox)*(y[p]-oy);
        length+=(x[p+1]-x[p]).hypot(y[p+1]-y[p]);
    }
    Ok((0.5*twice_area.abs(),length))
}

fn run() -> Result<(),String> {
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=4 {return Err("usage: fire-perimeter-observations INPUT_GEOJSON SOURCE_CONTRACT OUTPUT_JSON".into());}
    let contract:Value=serde_json::from_slice(&std::fs::read(&args[2]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let bytes=std::fs::read(&args[1]).map_err(|e|e.to_string())?;
    let digest=format!("{:x}",Sha256::digest(&bytes));
    if contract.get("input_sha256").and_then(Value::as_str)!=Some(&digest) {return Err("observed perimeter source hash differs from the captured receipt".into());}
    let data:Value=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
    if data.get("type").and_then(Value::as_str)!=Some("FeatureCollection") {return Err("observed source is not a GeoJSON FeatureCollection".into());}
    if contract.get("time_format").and_then(Value::as_str)!=Some("unix_ms") {return Err("observed source must declare the UTC unix_ms timestamp format".into());}
    let time_property=contract.get("time_property").and_then(Value::as_str).ok_or("source contract lacks its observed-time property")?;
    let crs=Crs::AlbersConusNad83 {lat_1:29.5,lat_2:45.5,lat_0:23.,lon_0:-96.,false_easting:0.,false_northing:0.};
    let features=data.get("features").and_then(Value::as_array).ok_or("observed source lacks feature records")?;
    if features.is_empty() {return Err("observed source has no perimeters to compare".into());}
    let mut output=Vec::with_capacity(features.len());
    let mut west=f64::INFINITY;let mut south=f64::INFINITY;let mut east=f64::NEG_INFINITY;let mut north=f64::NEG_INFINITY;
    for feature in features {
        let properties=feature.get("properties").ok_or("observed feature has no timestamp properties")?;
        let time=properties.get(time_property).and_then(Value::as_i64).ok_or("observed feature has no UTC unix_ms timestamp")?;
        let geometry=feature.get("geometry").ok_or("observed feature has no geometry")?;
        let polys=polygons(geometry)?;let mut area=0.;let mut length=0.;let mut vertices=0;
        for poly in &polys {
            let mut polygon_area=0.;
            for (index,ring) in poly.iter().enumerate() {
                let (part,perimeter)=measure(ring,&crs)?;
                polygon_area+=if index==0 {part} else {-part};length+=perimeter;vertices+=ring.len();
                for point in ring {west=west.min(point[0]);east=east.max(point[0]);south=south.min(point[1]);north=north.max(point[1]);}
            }
            if polygon_area<=0. {return Err("observed perimeter holes exhaust its outer-ring area".into());}
            area+=polygon_area;
        }
        let geometry_sha256=format!("{:x}",Sha256::digest(serde_json::to_vec(geometry).map_err(|e|e.to_string())?));
        output.push(json!({"observed_utc_epoch_ms":time,"properties":properties,"geometry":geometry,
            "geometry_sha256":geometry_sha256,"area_m2":area,"area_acres":area/4046.8564224,
            "perimeter_m":length,"polygon_count":polys.len(),"vertex_count":vertices}));
    }
    output.sort_by_key(|feature|feature["observed_utc_epoch_ms"].as_i64().unwrap());
    let document=json!({"schema":"sfire-observed-perimeters-v1","source_contract":contract,"input_sha256":digest,
        "analysis_crs":"EPSG:5070","geographic_bounds":[west,south,east,north],"features":output});
    std::fs::write(&args[3],serde_json::to_vec_pretty(&document).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    println!("{} records decoded in Rust; geographic bounds {}",features.len(),document["geographic_bounds"]);
    Ok(())
}

fn main() {if let Err(error)=run() {eprintln!("{error}");std::process::exit(1);}}
