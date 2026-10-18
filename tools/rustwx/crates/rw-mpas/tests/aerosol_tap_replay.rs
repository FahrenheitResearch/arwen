//! Real-input init/LBC boundary replay with f1's ORCL0001 native transport.
//! MPAS: mpas_init_atm_thompson_aerosols.F:36-81,722-865.
use rw_mpas::init::aerosols::{self, Climatology};
use std::{fs, io::Write, path::{Path, PathBuf}};

struct Record { header: Vec<i32>, ids: Vec<i32>, payload: Vec<u8> }
fn record(folder: &Path, key: &str, event: usize, phase: &str, name: &str) -> Record {
    let path = folder.join(format!("{key}.s0000000000.e{event:010}.r000000.{phase}.{name}.bin"));
    let data = fs::read(&path).unwrap();
    assert_eq!(&data[..8], b"ORCL0001");
    let h: Vec<_> = data[8..92].chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    assert_eq!((h[13], h[14], h[15], h[19], h[20]), (0, event as i32, 0, 256, 0));
    for (i, expected) in [key, phase, name].iter().enumerate() {
        assert_eq!(std::str::from_utf8(&data[92+i*256..92+(i+1)*256]).unwrap().trim_end(), *expected);
    }
    let offset = 860 + usize::try_from(h[16]).unwrap()*4;
    let ids = data[860..offset].chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let n: usize = h[3..8].iter().map(|&v| usize::try_from(v).unwrap()).product();
    assert_eq!(data.len(), offset+n*usize::try_from(h[1]).unwrap(), "{}", path.display());
    Record { header:h, ids, payload:data[offset..].to_vec() }
}
fn floats(r: &Record, shape: &[usize], ids: &[i32], cell: usize, level: usize) -> Vec<f32> {
    assert_eq!((r.header[0], r.header[1], r.header[2]), (1,4,shape.len() as i32));
    for (i, &n) in shape.iter().enumerate() { assert_eq!(r.header[3+i], n as i32); }
    assert!(r.header[8..13].iter().all(|&v|v==1));
    assert_eq!((r.header[17],r.header[18]),(cell as i32,level as i32));
    assert_eq!(r.ids, ids);
    r.payload.chunks_exact(4).map(|b|f32::from_bits(u32::from_le_bytes(b.try_into().unwrap()))).collect()
}
fn integer(r: &Record) -> usize {
    assert_eq!((r.header[0],r.header[1],r.header[2]),(3,4,0));
    i32::from_le_bytes(r.payload.as_slice().try_into().unwrap()).try_into().unwrap()
}
fn text(r: &Record) -> String {
    assert_eq!((r.header[0],r.header[1],r.header[2]),(5,1,0));
    std::str::from_utf8(&r.payload).unwrap().trim_end().to_owned()
}
fn words(name: &str, got: &[f32], expected: &[f32]) -> usize {
    assert_eq!(got.len(),expected.len());
    for (i,(a,b)) in got.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(),b.to_bits(),"{name} word {i}: {a:?} != {b:?}");
    }
    got.len()
}
fn monthly(a: &[Vec<f32>], nc: usize, ng: usize) -> Vec<f32> {
    (0..nc).flat_map(|c|(0..ng).flat_map(move|k|(0..12).map(move|m|a[m][c*ng+k]))).collect()
}

#[test]
fn prepare_real_tap_inputs() {
    let Some(manifest)=std::env::var_os("MPAS_AEROSOL_TAP_INPUTS") else {return};
    let rows: serde_json::Value=serde_json::from_slice(&fs::read(manifest).unwrap()).unwrap();
    for row in rows.as_array().unwrap() {
        let input=netcrust::File::open(row["source"].as_str().unwrap()).unwrap();
        let lat=input.read_array::<f32>("latCell").unwrap();
        let lon=input.read_array::<f32>("lonCell").unwrap();
        let land=input.read_array::<i32>("landmask").unwrap();
        let z=input.read_array::<f32>("zgrid").unwrap();
        let metric=input.read_array::<f32>("zz").unwrap();
        let rho=input.read_array::<f32>("rho_base").unwrap();
        let ids=input.read_array::<i32>("indexToCellID").unwrap();
        let nc=lat.len();let nv=z.shape()[1]-1;
        assert_eq!(lon.len(),nc);assert_eq!(land.len(),nc);assert_eq!(ids.len(),nc);
        assert_eq!(metric.shape(),[nc,nv]);assert_eq!(rho.shape(),[1,nc,nv]);
        let mut wire=fs::File::create(row["wire"].as_str().unwrap()).unwrap();
        for n in [nc,nv] {wire.write_all(&(n as i32).to_be_bytes()).unwrap();}
        for a in [&lat,&lon] {for v in a {wire.write_all(&v.to_bits().to_be_bytes()).unwrap();}}
        for v in &land {wire.write_all(&v.to_be_bytes()).unwrap();}
        for a in [&z,&metric,&rho] {for v in a {wire.write_all(&v.to_bits().to_be_bytes()).unwrap();}}
        let mut output=fs::File::create(row["ids"].as_str().unwrap()).unwrap();
        for v in &ids {assert!(*v>0);output.write_all(&v.to_be_bytes()).unwrap();}
    }
}

#[test]
#[ignore = "requires sealed real native init/LBC tap captures; receipts/replay.sh runs this explicitly"]
fn real_native_tap_replay() {
    let root=PathBuf::from(std::env::var_os("MPAS_AEROSOL_TAP_ROOT").expect("native tap capture directory required"));
    let mut rows=Vec::new();
    let mut folders:Vec<_>=fs::read_dir(&root).unwrap().map(|p|p.unwrap().path()).filter(|p|p.is_dir()).collect();
    folders.sort();assert!(!folders.is_empty());
    for folder in folders {
        let base=record(&folder,"i6_init",1,"pre","lat");let ids=base.ids.clone();let nc=ids.len();
        let ng=integer(&record(&folder,"i6_init",1,"pre","ng"));
        let nv=integer(&record(&folder,"i6_init",1,"pre","nv"));
        assert_eq!(integer(&record(&folder,"i6_init",1,"pre","nc")),nc);
        let lat=floats(&base,&[nc],&ids,1,0);
        let lon=floats(&record(&folder,"i6_init",1,"pre","lon"),&[nc],&ids,1,0);
        let land_record=record(&folder,"i6_init",1,"pre","land");
        assert_eq!((land_record.header[0],land_record.header[1],land_record.header[2]),(3,4,1));
        assert_eq!(land_record.ids,ids);
        let land:Vec<_>=land_record.payload.chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap())).collect();
        let date=text(&record(&folder,"i6_init",1,"pre","date"));
        let pressure=floats(&record(&folder,"i6_init",1,"pre","pressure"),&[nv,nc],&ids,2,1);
        let table=folder.join(aerosols::NATIVE_FILENAME);
        let (clim,_) = aerosols::horizontal(&table,&lat,&lon,&land,ng).unwrap();
        let (initial,_)=aerosols::initialize(&table,&lat,&lon,&land,ng,&date,&pressure,nv).unwrap();
        let mut cache:Option<Climatology>=None;let mut outputs=0;
        for (key,event) in [("i6_init",1),("i6_lbc",1),("i6_lbc",2)] {
            let stamp=text(&record(&folder,key,event,"pre","date"));
            let start=text(&record(&folder,key,event,"pre","timestart"));
            assert_eq!(start,date);
            for name in ["nc","ng","nv","timestart","lat","lon","land","pressure"] {
                let entry=record(&folder,key,event,"pre",name);
                let original=record(&folder,"i6_init",1,"pre",name);
                assert_eq!(entry.payload,original.payload,"changed input {name}");
                assert_eq!(entry.ids,original.ids);
            }
            for (name,a) in [("nifa_clim",&clim.nifa),("nwfa_clim",&clim.nwfa),("pwif_clim",&clim.pressure)] {
                let entry=floats(&record(&folder,key,event,"pre",name),&[12,ng,nc],&ids,3,2);
                if key=="i6_init" { assert!(entry.iter().all(|v|v.to_bits()==0)); }
                else { words("entry cache",&monthly(a,nc,ng),&entry); }
            }
            let entry=floats(&record(&folder,key,event,"pre","scalars"),&[2,nv,nc],&ids,3,2);
            if key=="i6_init" || event==1 { assert!(entry.iter().all(|&v|v.to_bits()==(-999.0_f32).to_bits())); }
            else {
                let previous=aerosols::initialize_lbc(&table,&date,&date,&lat,&lon,&land,ng,&pressure,nv,&mut None).unwrap().unwrap();
                let scalar:Vec<_>=previous.nifa.iter().zip(&previous.nwfa).flat_map(|(&a,&b)|[a,b]).collect();
                words("entry LBC scalars",&scalar,&entry);
            }
            let state = if key=="i6_init" {None} else {
                Some(aerosols::initialize_lbc(&table,&stamp,&start,&lat,&lon,&land,ng,&pressure,nv,&mut cache).unwrap().unwrap())
            };
            let state=state.as_ref().unwrap_or(&initial);
            for (name,a) in [("nifa_clim",&clim.nifa),("nwfa_clim",&clim.nwfa),("pwif_clim",&clim.pressure)] {
                outputs+=words(name,&monthly(a,nc,ng),&floats(&record(&folder,key,event,"post",name),&[12,ng,nc],&ids,3,2));
            }
            let scalar:Vec<_>=state.nifa.iter().zip(&state.nwfa).flat_map(|(&a,&b)|[a,b]).collect();
            outputs+=words("scalars",&scalar,&floats(&record(&folder,key,event,"post","scalars"),&[2,nv,nc],&ids,3,2));
            // Every other post record is read-only storage, and is compared too.
            for name in ["nc","ng","nv","date","timestart","lat","lon","land","pressure"] {
                let pre=record(&folder,key,event,"pre",name);let post=record(&folder,key,event,"post",name);
                assert_eq!(pre.header,post.header);assert_eq!(pre.ids,post.ids);assert_eq!(pre.payload,post.payload,"read-only {name}");
            }
        }
        rows.push(serde_json::json!({"case":folder.file_name().unwrap().to_str().unwrap(),"n_cells":nc,"n_vert_levels":nv,"n_gocart_levels":ng,"native_calls":3,"output_words":outputs,"differing_words":0,"read_only_post_records_compared":27}));
    }
    let out=std::env::var_os("MPAS_AEROSOL_TAP_RESULT").unwrap();
    fs::write(out,serde_json::to_vec_pretty(&serde_json::json!({"schema":"i6-f1-transport-real-module-replay-v1","process_count":1,"mpi_ranks":null,"rust_test_threads":1,"differing_words":0,"cases":rows})).unwrap()).unwrap();
}
