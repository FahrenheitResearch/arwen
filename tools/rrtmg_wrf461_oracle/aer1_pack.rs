//! Convert the WRF formatted climatology to little-endian f32 tables.
use std::{env,fs,io::{self,Write},path::Path};
fn main()->io::Result<()> {
 let args:Vec<_>=env::args().collect();let root=Path::new(&args[1]);let out=Path::new(&args[2]);
 fs::create_dir_all(out)?;
 for (name,count) in [("aerosol.formatted",72*12*46*12*6),("aerosol_lat.formatted",46),("aerosol_lon.formatted",72),("aerosol_plev.formatted",12)] {
  let text=fs::read_to_string(root.join(name))?;let mut bytes=Vec::with_capacity(count*4);
  for word in text.split_whitespace(){let x:f32=word.parse().map_err(|_|io::Error::new(io::ErrorKind::InvalidData,"bad WRF float"))?;if !x.is_finite(){return Err(io::Error::new(io::ErrorKind::InvalidData,"nonfinite WRF value"))} bytes.extend(x.to_le_bytes());}
  if bytes.len()!=count*4{return Err(io::Error::new(io::ErrorKind::InvalidData,"wrong WRF table count"))}
  fs::File::create(out.join(format!("{name}.bin")))?.write_all(&bytes)?;
 }
 Ok(())
}
