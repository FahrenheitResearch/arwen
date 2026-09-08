//! Exact domain/time and immutable bundle contract around the original WRF engine.
mod stored_plane_fixture;
use rw_wrfbatch::process_request::{ProcessRequest, REQUEST_SCHEMA, process, sha256_file};
use std::path::PathBuf;

struct Scratch(PathBuf);
impl Scratch {
    fn new()->Self {
        let root=std::env::temp_dir().join(format!("arwen-full-wrf-request-{}-{}",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&root).unwrap();Self(root)
    }
}
impl Drop for Scratch {fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
fn request(scratch:&Scratch)->ProcessRequest {
    let original=stored_plane_fixture::write_rain_frame(&scratch.0,900,2.);
    let sha=sha256_file(&original).unwrap();
    let path=scratch.0.join(format!("{sha}.wrf"));std::fs::rename(original,&path).unwrap();
    ProcessRequest {schema:REQUEST_SCHEMA.into(),path,source_sha256:sha,case_id:"subhour-case".into(),domain:"d01".into(),
        valid_utc:"2026-08-19T00:15:00Z".into(),store_root:scratch.0.join("stores"),lead_seconds:Some(900),heavy_ecape:false}
}
#[test]
fn immutable_subhour_request_retains_native_planes_profiles_and_verified_cache(){
    let scratch=Scratch::new();let request=request(&scratch);
    let result=process(&request,|_|{}).unwrap();
    assert!(!result.cache_hit);assert_eq!(result.frame.storage_slot,0);
    assert_eq!(result.frame.identity.lead_seconds,900);assert_eq!(result.frame.identity.model,"wrf-d01");
    assert_eq!(result.files.len(),4);assert_eq!(result.frame.identity.source_sha256,request.source_sha256);
    assert!(result.frame.variables.contains(&stored_plane_fixture::USER_PLANE_STORE_NAME.into()));
    for field in ["temperature_iso","dewpoint_iso","u_iso","v_iso","height_iso"] {assert!(result.frame.variables.contains(&field.into()),"missing {field}");}
    let reader=rw_store::reader::HourReader::open(&result.frame.hour_path).unwrap();
    let values=reader.read_full_2d(stored_plane_fixture::USER_PLANE_STORE_NAME).unwrap();
    assert_eq!(values[0],stored_plane_fixture::user_plane_value(0,0));
    drop(reader);
    let cached=process(&request,|_|{}).unwrap();assert!(cached.cache_hit);assert_eq!(result.frame.id,cached.frame.id);
    let mut changed=request.clone();changed.case_id="other-case".into();
    let other=process(&changed,|_|{}).unwrap();assert_ne!(result.frame.store_root,other.frame.store_root);
    let grid=std::fs::read(&result.grid_path).unwrap();let mut broken=grid.clone();broken[0]^=1;
    std::fs::write(&result.grid_path,broken).unwrap();assert!(process(&request,|_|{}).is_err());
    assert_eq!(sha256_file(&request.path).unwrap(),request.source_sha256);
}
#[test]
fn wrong_source_hash_domain_time_and_lead_are_refused_before_store_publication(){
    let scratch=Scratch::new();let request=request(&scratch);
    for wrong in 0..4 {
        let mut changed=request.clone();match wrong {
            0=>changed.source_sha256="0".repeat(64),1=>changed.domain="d02".into(),
            2=>changed.valid_utc="2026-08-19T00:16:00Z".into(),_=>changed.lead_seconds=Some(901),
        }
        assert!(process(&changed,|_|{}).is_err());assert!(!request.store_root.exists());
    }
}
