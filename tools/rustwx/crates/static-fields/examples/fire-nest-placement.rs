//! Place declared nests around a supplied geographic point with Rust WPS grids.
use serde::Deserialize;
use serde_json::json;
use static_fields::projection::{GridSpec,ProjectedGrid};
#[derive(Deserialize)] struct Nest {ratio:i64,nx:i64,ny:i64}
#[derive(Deserialize)] struct Request {parent_spec:GridSpec,point:[f64;2],nests:Vec<Nest>,sr_x:usize,sr_y:usize}
fn run() -> Result<(),String> {
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=3 {return Err("usage: fire-nest-placement REQUEST_JSON OUTPUT_JSON".into());}
    let request:Request=serde_json::from_slice(&std::fs::read(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let mut parent=ProjectedGrid::new(request.parent_spec.clone()).map_err(|e|e.to_string())?;
    let mut domains=Vec::new();
    for (index,nest) in request.nests.iter().enumerate() {
        if nest.ratio<2 || nest.nx<2 || nest.ny<2 {return Err("nested weather grid needs refinement and nonempty cells".into());}
        let (x,y)=parent.latlon_to_ij(request.point[0],request.point[1]);
        let i=(x+0.5-nest.nx as f64/(2.*nest.ratio as f64)).round() as i64;
        let j=(y+0.5-nest.ny as f64/(2.*nest.ratio as f64)).round() as i64;
        if i<1 || j<1 || i as f64+nest.nx as f64/nest.ratio as f64>parent.spec.e_we as f64
            || j as f64+nest.ny as f64/nest.ratio as f64>parent.spec.e_sn as f64 {return Err("requested fire-weather nest leaves its parent footprint".into());}
        let child=parent.nest(i,j,nest.ratio,nest.nx+1,nest.ny+1,None,None).map_err(|e|e.to_string())?;
        let center=child.ij_to_latlon((nest.nx+1) as f64/2.,(nest.ny+1) as f64/2.);
        domains.push(json!({"grid_id":index+2,"parent_id":index+1,"i_parent_start":i,"j_parent_start":j,
            "parent_grid_ratio":nest.ratio,"nx":nest.nx,"ny":nest.ny,"dx":child.spec.dx,
            "grid_spec":child.spec,"actual_center":[center.0,center.1]}));
        parent=child;
    }
    let output=json!({"schema":"sfire-nest-placement-v1","parent_spec":request.parent_spec,"point":request.point,
        "domains":domains,"fire_static":{"schema":"gpuwm-sfire-static-v1","grid_spec":parent.spec,
        "sr_x":request.sr_x,"sr_y":request.sr_y,"sources":{"fuel_year":2024}}});
    std::fs::write(&args[2],serde_json::to_vec_pretty(&output).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    println!("{} nested domains placed in Rust",request.nests.len());Ok(())
}
fn main() {if let Err(error)=run() {eprintln!("{error}");std::process::exit(1);}}
