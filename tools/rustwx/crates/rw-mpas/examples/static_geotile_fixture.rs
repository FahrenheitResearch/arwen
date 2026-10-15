//! Split an owned complete tile into smaller tiles and remove/empty one.
//! Encoded weather bytes stay in Rust. MPAS: mpas_geotile_manager.F:505-644,665-708.
use std::{fs,path::PathBuf};
use rw_mpas::static_geog::GeogDataset;
fn main() {
    let args:Vec<_>=std::env::args_os().skip(1).collect();
    assert_eq!(args.len(),6,"SOURCE DEST TILE_X TILE_Y UNAVAILABLE_INDEX missing|empty");
    let source=PathBuf::from(&args[0]);let target=PathBuf::from(&args[1]);
    let tx:usize=args[2].to_str().unwrap().parse().unwrap();
    let ty:usize=args[3].to_str().unwrap().parse().unwrap();
    let absent:usize=args[4].to_str().unwrap().parse().unwrap();
    let mode=args[5].to_str().unwrap();assert!(matches!(mode,"missing"|"empty"));
    let ds=GeogDataset::open(&source).unwrap();assert_eq!(ds.tiles.len(),1);
    let tile=&ds.tiles[0];assert_eq!((tile.xs,tile.ys),(1,1));
    let sx=ds.index.tile_x as usize;let sy=ds.index.tile_y as usize;
    assert!(tx>0 && ty>0 && sx%tx==0 && sy%ty==0);
    let b=ds.index.tile_bdr as usize;let ws=ds.index.wordsize as usize;
    let bytes=fs::read(&tile.path).unwrap();assert_eq!(bytes.len(),ds.index.plane_words()*ds.index.nz()*ws);
    fs::create_dir(&target).unwrap();
    let mut index=String::new();
    for line in fs::read_to_string(source.join("index")).unwrap().lines() {
        match line.split_once('=').map(|(key,_)| key.trim()) {
            Some("tile_x")=>index.push_str(&format!("tile_x={tx}\n")),
            Some("tile_y")=>index.push_str(&format!("tile_y={ty}\n")),
            _=>{index.push_str(line);index.push('\n');}
        }
    }
    fs::write(target.join("index"),index).unwrap();
    for y in 0..sy/ty {for x in 0..sx/tx {
        let i=y*(sx/tx)+x;
        if i==absent && mode=="missing" {continue;}
        let name=format!("{:05}-{:05}.{:05}-{:05}",x*tx+1,(x+1)*tx,y*ty+1,(y+1)*ty);
        let mut encoded=Vec::new();
        if i!=absent {for z in 0..ds.index.nz() {for j in 0..ty+2*b {
            let start=((z*(sy+2*b)+y*ty+j)*(sx+2*b)+x*tx)*ws;
            encoded.extend_from_slice(&bytes[start..start+(tx+2*b)*ws]);
        }}}
        fs::write(target.join(name),encoded).unwrap();
    }}
}
