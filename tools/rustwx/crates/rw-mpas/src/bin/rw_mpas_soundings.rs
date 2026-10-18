//! MPAS-A v8.4.2 sounding bridge. stdin/stdout carry raw RKIND words.
use std::io::{Read, Write};
use std::path::Path;
use rw_mpas::soundings::{self, Location};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Metadata { exists: bool, opened: bool, names: Vec<String>, #[serde(default)] names_hex: Vec<String>, #[serde(default)] levels: usize }
fn packet(meta: &Metadata, words: impl Iterator<Item=u32>) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_vec(meta)?;
    let mut out = std::io::stdout().lock();
    out.write_all(&(json.len() as u32).to_le_bytes())?; out.write_all(&json)?;
    for word in words { out.write_all(&word.to_le_bytes())?; }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("probe") => { println!("{}",Path::new(args.get(2).ok_or("missing location path")?).exists()); }
        Some("read") => {
            let path = Path::new(args.get(2).ok_or("missing location path")?);
            let exists = path.exists();
            match std::fs::read(path) {
                Ok(text) => {
                    let locations = soundings::read_locations_bytes(&text).map_err(std::io::Error::other)?;
                    let names = locations.iter().map(|l| String::from_utf8_lossy(&l.name).into_owned()).collect();
                    let names_hex=locations.iter().map(|l|l.name.iter().map(|b|format!("{b:02x}")).collect()).collect();
                    let meta = Metadata { exists, opened: true, names, names_hex, levels: 0 };
                    packet(&meta,locations.iter().map(|l|l.latitude.to_bits()).chain(locations.iter().map(|l|l.longitude.to_bits())))?;
                }
                Err(_) => { packet(&Metadata {exists,opened:false,names:vec![],names_hex:vec![],levels:0},std::iter::empty())?; }
            }
        }
        Some("write") => {
            let time = args.get(2).ok_or("missing timestamp")?;
            let dir = Path::new(args.get(3).ok_or("missing output directory")?);
            let mut input = Vec::new(); std::io::stdin().read_to_end(&mut input)?;
            if input.len() < 4 { return Err("truncated metadata prefix".into()); }
            let len = u32::from_le_bytes(input[..4].try_into()?) as usize;
            if input.len() < 4+len { return Err("truncated metadata".into()); }
            let meta: Metadata = serde_json::from_slice(&input[4..4+len])?;
            let n = meta.names.len();
            let bytes = &input[4+len..];
            if bytes.len() != 4*n*(3+6*meta.levels) { return Err("sounding payload extents disagree".into()); }
            let words: Vec<u32> = bytes.chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect();
            let mut locations=Vec::new();
            if !meta.names_hex.is_empty() && meta.names_hex.len()!=n{return Err("native name extents disagree".into())}
            for (i,name) in meta.names.iter().enumerate() {
                let bytes=if meta.names_hex.is_empty(){name.as_bytes().to_vec()}else{
                    let h=meta.names_hex[i].as_bytes();if h.len()%2!=0{return Err("partial native name byte".into())}
                    h.chunks_exact(2).map(|b|Ok(u8::from_str_radix(std::str::from_utf8(b)?,16)?)).collect::<Result<Vec<_>,Box<dyn std::error::Error>>>()?
                };
                locations.push(Location {latitude:f32::from_bits(words[i]),longitude:f32::from_bits(words[n+i]),name:bytes});
            }
            let owned: Vec<i32> = words[2*n..3*n].iter().map(|w|*w as i32).collect();
            let rows: Vec<f32> = words[3*n..].iter().map(|w|f32::from_bits(*w)).collect();
            println!("{}",serde_json::to_string(&soundings::write_soundings(dir,time,&locations,&owned,meta.levels,&rows)?)?);
        }
        Some("clock") => {
            let request: soundings::clock::Request=serde_json::from_reader(std::io::stdin())?;
            println!("{}",serde_json::to_string(&soundings::clock::query(request).map_err(std::io::Error::other)?)?);
        }
        Some("format") => {
            let mut input = Vec::new(); std::io::stdin().read_to_end(&mut input)?;
            if input.len()%4 != 0 { return Err("partial RKIND word".into()); }
            let mut out = std::io::BufWriter::new(std::io::stdout().lock());
            for word in input.chunks_exact(4) {
                let value = f32::from_bits(u32::from_le_bytes(word.try_into()?));
                for width in [3,4,6,7,9,10] { writeln!(out,"{}",soundings::fixed(value,width))?; }
            }
        }
        _ => return Err("usage: rw_mpas_soundings probe/read <locations> | write <timestamp> <directory> | format".into()),
    }
    Ok(())
}
