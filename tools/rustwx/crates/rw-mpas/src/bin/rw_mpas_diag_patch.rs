//! Compact real-column diagnostic replay capture; netCDF decode/remap stays Rust.
use std::collections::{BTreeMap,BTreeSet};
use std::path::Path;
use std::io::Read;
use sha2::{Digest,Sha256};
fn file_sha(path:&str)->std::io::Result<String> {
 let mut file=std::fs::File::open(path)?;let mut hash=Sha256::new();let mut buffer=[0u8;65536];
 loop {let count=file.read(&mut buffer)?;if count==0{break}hash.update(&buffer[..count]);}
 Ok(format!("{:x}",hash.finalize()))
}
fn main()->Result<(),Box<dyn std::error::Error>> {
 let args:Vec<String>=std::env::args().collect();
 if args.len()<5 {return Err("usage: rw_mpas_diag_patch initial.nc native.nc directory cell1...".into())}
 let initial=netcrust::open(&args[1])?;let native=netcrust::open(&args[2])?;let dir=Path::new(&args[3]);std::fs::create_dir_all(dir)?;
 let owned:Vec<usize>=args[4..].iter().map(|s|s.parse::<usize>().map(|c|c-1)).collect::<Result<_,_>>()?;
 let nc=initial.dimension("nCells").ok_or("nCells")?.len();
 let me=initial.dimension("maxEdges").ok_or("maxEdges")?.len();
 let nz=initial.dimension("nVertLevels").ok_or("nVertLevels")?.len();
 let nec:Vec<i32>=initial.read_array::<i32>("nEdgesOnCell")?.iter().copied().collect();
 let coc:Vec<i32>=initial.read_array::<i32>("cellsOnCell")?.iter().copied().collect();
 let eoc:Vec<i32>=initial.read_array::<i32>("edgesOnCell")?.iter().copied().collect();
 let voc:Vec<i32>=initial.read_array::<i32>("verticesOnCell")?.iter().copied().collect();
 let mut cells=owned.clone();let mut edges=BTreeSet::new();let mut vertices=BTreeSet::new();
 for &c in &owned {if c>=nc{return Err("owned cell out of range".into())}for i in 0..nec[c] as usize {let n=(coc[c*me+i]-1) as usize;if n<nc&&!cells.contains(&n){cells.push(n)}edges.insert((eoc[c*me+i]-1) as usize);vertices.insert((voc[c*me+i]-1) as usize);}}
 // Friction curl needs the third edge at each retained vertex as well as
 // the edges touching each owned cell. Keep native global edge order.
 let eov:Vec<i32>=initial.read_array::<i32>("edgesOnVertex")?.iter().copied().collect();
 for &v in &vertices {for i in 0..3 {edges.insert((eov[v*3+i]-1) as usize);}}
 let edges:Vec<usize>=edges.into_iter().collect();let vertices:Vec<usize>=vertices.into_iter().collect();
 let maps: BTreeMap<&str,BTreeMap<usize,i32>>=[("nCells",&cells),("nEdges",&edges),("nVertices",&vertices)].into_iter().map(|(name,ids)|(name,ids.iter().enumerate().map(|(i,id)|(*id,i as i32+1)).collect())).collect();
 let entities:BTreeMap<&str,&Vec<usize>>=[("nCells",&cells),("nEdges",&edges),("nVertices",&vertices)].into_iter().collect();
 let names=["nEdgesOnCell","cellsOnCell","edgesOnCell","cellsOnEdge","verticesOnCell","cellsOnVertex","kiteAreasOnVertex","areaCell","dvEdge","edgeNormalVectors","cellTangentPlane","localVerticalUnitVectors","zgrid","latCell","lonCell","zz","verticesOnEdge","dcEdge","areaTriangle","coeffs_reconstruct","theta","rho","uReconstructX","uReconstructY","uReconstructZ","vorticity","w","theta_m","rho_zz","qv","exner","pressure_base","pressure_p","uReconstructZonal","uReconstructMeridional","ertel_pv","iLev_DT","u_pv","v_pv","theta_pv","vort_pv","cldfrac","cldfrac_low_UPP","cldfrac_mid_UPP","cldfrac_high_UPP","cldfrac_tot_UPP"];
 let mut manifest=Vec::new();
 for name in names {
  // Initialization files may allocate these fields without filling them.
  // A duplicate-time append can silently retain an older stream schema.
  if matches!(name,"edgeNormalVectors"|"cellTangentPlane"|"localVerticalUnitVectors"|"coeffs_reconstruct") && native.variable(name).is_none() {
   return Err(format!("native output lacks initialized {name}; capture a fresh output file containing geometry").into());
  }
  let file=if native.variable(name).is_some(){&native}else{&initial};let Some(var)=file.variable(name)else{continue};
  let dims:Vec<String>=var.dimensions().iter().map(|d|d.name().to_string()).collect();let original=var.shape();
  let axis=dims.iter().position(|d|entities.contains_key(d.as_str())).ok_or_else(||format!("no entity dimension: {name}"))?;
  let ids=entities[dims[axis].as_str()];let stride:usize=original[axis+1..].iter().product();
  let integer=matches!(name,"nEdgesOnCell"|"cellsOnCell"|"edgesOnCell"|"cellsOnEdge"|"verticesOnCell"|"cellsOnVertex"|"verticesOnEdge"|"iLev_DT");
  let remap=match name {"cellsOnCell"|"cellsOnEdge"|"cellsOnVertex"=>Some("nCells"),"edgesOnCell"=>Some("nEdges"),"verticesOnCell"|"verticesOnEdge"=>Some("nVertices"),_=>None};
  let source:Vec<u32>=if integer {file.read_array::<i32>(name)?.iter().map(|v|*v as u32).collect()}else{file.read_array::<f32>(name)?.iter().map(|v|v.to_bits()).collect()};
  let sentinel=dims[axis]!="nEdges";let mut words=Vec::new();
  for &id in ids {for &word in &source[id*stride..(id+1)*stride] {let w=if let Some(entity)=remap {let map=&maps[entity];*map.get(&(word.wrapping_sub(1) as usize)).unwrap_or(&(entities[entity].len() as i32+1)) as u32}else{word};words.push(w);}}
  if sentinel {words.resize(words.len()+stride,0)}
  let shape:Vec<usize>=original[axis+1..].iter().rev().copied().chain(std::iter::once(ids.len()+usize::from(sentinel))).collect();
  let bytes:Vec<u8>=words.iter().flat_map(|w|w.to_le_bytes()).collect();let raw_sha=format!("{:x}",Sha256::digest(&bytes));
  std::fs::write(dir.join(format!("{name}.raw")),bytes)?;
  manifest.push(serde_json::json!({"field":name,"sha256":raw_sha,"shape":shape,"dtype":if integer{"int32"}else{"float32"},"source":if std::ptr::eq(file,&native){"native-output"}else{"initial-state"}}));
 }
 std::fs::write(dir.join("patch.json"),serde_json::to_vec_pretty(&serde_json::json!({"initial_sha256":file_sha(&args[1])?,"native_output_sha256":file_sha(&args[2])?,"capture_kind":"initialized-native-netcdf-column-stencils","nCells":cells.len(),"nCellsSolve":owned.len(),"nEdges":edges.len(),"nVertices":vertices.len(),"maxEdges":me,"nVertLevels":nz,"global_cell_ids":cells.iter().map(|c|c+1).collect::<Vec<_>>(),"fields":manifest}))?)?;
 Ok(())
}
