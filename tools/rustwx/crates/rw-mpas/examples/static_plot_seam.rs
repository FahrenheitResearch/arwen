//! Visualization-only antimeridian clipping for the existing Rust mesh renderer.
//! No scientific output or parity input is altered. Every source cell is retained.
use rw_store::netcdf_classic::{NcAttr,NcClassicWriter,NcData,NcDim,NcFormat,NcType,NcVarDef};
use std::{env,path::Path};
fn clip(ring:&[(f64,f64)], edge:f64, lower:bool)->Vec<(f64,f64)> {
    let inside=|p:(f64,f64)| if lower {p.1>=edge} else {p.1<=edge};
    let mut out=Vec::new();
    for i in 0..ring.len() {
        let a=ring[i]; let b=ring[(i+1)%ring.len()];
        if inside(a) {out.push(a);}
        if inside(a)!=inside(b) {
            let t=(edge-a.1)/(b.1-a.1); out.push((a.0+t*(b.0-a.0),edge));
        }
    }
    out
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let a:Vec<_>=env::args().skip(1).collect();
    assert_eq!(a.len(),5,"static_plot_seam GRID FIELD_FILE FIELD SELECTOR OUT");
    let g=netcrust::File::open(&a[0])?; let f=netcrust::File::open(&a[1])?;
    let n=g.dimension("nCells").unwrap().len(); let m=g.dimension("maxEdges").unwrap().len();
    let lc=g.read_f64("lonCell")?; let lv=g.read_f64("latVertex")?; let ov=g.read_f64("lonVertex")?;
    let ne=g.read_f64("nEdgesOnCell")?; let vo=g.read_f64("verticesOnCell")?;
    let variable=f.variable(&a[2]).unwrap();
    let vd=variable.dimensions();
    let first=vd.iter().position(|d|d.name()=="nCells").unwrap();
    assert!(vd[..first].iter().all(|d|d.len()==1),"multiple time records unsupported");
    assert_eq!(vd[first].len(),n);
    let stamp=f.read_strings("xtime").ok().and_then(|v|v.first().cloned()).unwrap_or_default();
    let units=variable.attribute("units").and_then(|v|v.as_string().map(str::to_owned)).unwrap_or_default();
    let values=f.read_f64(&a[2])?; let levels=values.len()/n; assert_eq!(values.len(),levels*n);
    let select=|c:usize| {let v=&values[c*levels..(c+1)*levels];
        if a[3]=="colmax" {v.iter().copied().fold(f64::NEG_INFINITY,f64::max)}
        else {v[a[3].strip_prefix("level=").unwrap().parse::<usize>().unwrap()-1]}};
    let mut rings=Vec::new(); let mut data=Vec::new(); let mut origins=Vec::new();
    for c in 0..n {
        let center=(lc[c].to_degrees()+180.0).rem_euclid(360.0)-180.0;
        let ring:Vec<_>=(0..ne[c] as usize).map(|k| {
            let v=vo[c*m+k] as usize-1;
            (lv[v].to_degrees(),center+(ov[v].to_degrees()-center+180.0).rem_euclid(360.0)-180.0)
        }).collect();
        let mut kept=0;
        for shift in [-360.0,0.0,360.0] {
            let r:Vec<_>=ring.iter().map(|p|(p.0,p.1+shift)).collect();
            let r=clip(&clip(&r,-179.999999,true),179.999999,false);
            if r.len()>=3 {rings.push(r);data.push(select(c));origins.push(c as i32);kept+=1;}
        }
        assert!(kept>0,"source cell omitted {c}");
    }
    let cells=rings.len(); let edges=rings.iter().map(Vec::len).max().unwrap();
    let vertices:usize=rings.iter().map(Vec::len).sum();
    let dims=vec![NcDim::fixed("nCells",cells),NcDim::fixed("maxEdges",edges),NcDim::fixed("nVertices",vertices),NcDim::fixed("DateStrLen",stamp.len().max(1))];
    let vars=vec![NcVarDef::new("latCell",NcType::Double,vec![0]),NcVarDef::new("lonCell",NcType::Double,vec![0]),
        NcVarDef::new("latVertex",NcType::Double,vec![2]),NcVarDef::new("lonVertex",NcType::Double,vec![2]),
        NcVarDef::new("verticesOnCell",NcType::Int,vec![0,1]),NcVarDef::new("nEdgesOnCell",NcType::Int,vec![0]),
        NcVarDef::new("sourceCellIndex",NcType::Int,vec![0]),NcVarDef::new(a[2].clone(),NcType::Double,vec![0]).with_attrs(vec![NcAttr::text("units",units)]),NcVarDef::new("xtime",NcType::Char,vec![3])];
    let mut w=NcClassicWriter::create(Path::new(&a[4]),NcFormat::Classic,dims,
        vec![NcAttr::text("visualization_only","antimeridian-clipped geometry; all source cells retained")],vars,0)?;
    let mut lat=Vec::new();let mut lon=Vec::new();let mut latv=Vec::new();let mut lonv=Vec::new();
    let mut counts=Vec::new();let mut conn=vec![0i32;cells*edges];
    for (c,r) in rings.iter().enumerate() {
        lat.push(r.iter().map(|p|p.0).sum::<f64>()/r.len() as f64);lon.push(r.iter().map(|p|p.1).sum::<f64>()/r.len() as f64);
        counts.push(r.len() as i32);
        for (k,p) in r.iter().enumerate() {conn[c*edges+k]=(latv.len()+1) as i32;latv.push(p.0.to_radians());lonv.push(p.1.to_radians());}
    }
    for x in &mut lat {*x=x.to_radians();} for x in &mut lon {*x=x.to_radians();}
    w.put("latCell",NcData::Doubles(&lat))?;w.put("lonCell",NcData::Doubles(&lon))?;
    w.put("latVertex",NcData::Doubles(&latv))?;w.put("lonVertex",NcData::Doubles(&lonv))?;
    w.put("verticesOnCell",NcData::Ints(&conn))?;w.put("nEdgesOnCell",NcData::Ints(&counts))?;
    w.put("sourceCellIndex",NcData::Ints(&origins))?;w.put(&a[2],NcData::Doubles(&data))?;w.put("xtime",NcData::Chars(stamp.as_bytes()))?;w.finish()?;
    println!("all {n} source cells retained in {cells} seam-clipped polygons; selector {}",a[3]); Ok(())
}
