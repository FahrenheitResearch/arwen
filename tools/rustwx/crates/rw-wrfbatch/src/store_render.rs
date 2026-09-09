//! Render the existing compact native store without reading WRF or building volumes.
use super::{BatchHourScope, BatchRenderDomain, BatchRenderEvent, BatchRenderLimits,
    BatchRenderRequest, TitleProvenance, run_batch_render};
use rw_wrfbatch::process_request::{ProcessResult, verify_result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io::Read, path::{Path, PathBuf}, sync::atomic::AtomicBool};

const REQUEST_SCHEMA: &str = "arwen.native-store-render-request.v1";
const RESULT_SCHEMA: &str = "arwen.native-store-render-result.v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    process_result: PathBuf,
    expected_frame_id: String,
    expected_source_sha256: String,
    out_dir: PathBuf,
    products: Vec<String>,
    #[serde(default)] spacing_m: Option<f64>,
    #[serde(default="width")] width: u32,
    #[serde(default="height")] height: u32,
}
fn width()->u32 {1200}
fn height()->u32 {900}

#[derive(Debug, Serialize)]
struct Panel { slug: String, path: PathBuf, sha256: String, bytes: u64 }

fn sha256(path:&Path)->Result<String,String>{
    let mut file=fs::File::open(path).map_err(|e|e.to_string())?;
    let mut digest=Sha256::new();let mut buffer=vec![0;1024*1024];
    loop {let count=file.read(&mut buffer).map_err(|e|e.to_string())?;if count==0{break;}digest.update(&buffer[..count]);}
    Ok(format!("{:x}",digest.finalize()))
}

fn validate(request:&Request,result:&ProcessResult)->Result<(),String>{
    if request.schema!=REQUEST_SCHEMA || result.schema!="arwen.wrf-process-result.v2"
        || result.profile.as_deref()!=Some("viewer-2d-v1") {
        return Err("Native plot rendering requires a compact viewer v2 receipt".into());
    }
    if result.frame.id!=request.expected_frame_id || result.frame.identity.source_sha256!=request.expected_source_sha256
        || result.frame.identity.model!=format!("wrf-{}",result.domain) || result.frame.identity.source!="arwen" {
        return Err("Native plot request differs from the committed frame identity".into());
    }
    if request.width<256 || request.height<256 || request.width>4096 || request.height>4096
        || request.spacing_m.is_some_and(|dx| !dx.is_finite() || dx<=0.) {
        return Err("Native plot dimensions or spacing are invalid".into());
    }
    if request.products.is_empty() || request.products.len()>96
        || request.products.iter().collect::<BTreeSet<_>>().len()!=request.products.len()
        || request.products.iter().any(|slug|!result.products.iter().any(|p|p.slug==*slug&&p.available)) {
        return Err("Native plot products must be distinct available products in this compact store".into());
    }
    verify_result(result)
}

fn render(request:Request,output:&Path)->Result<(),String>{
    let metadata=fs::metadata(&request.process_result).map_err(|e|e.to_string())?;
    if metadata.len()>512*1024{return Err("Native store receipt exceeds its metadata bound".into());}
    let result:ProcessResult=serde_json::from_slice(&fs::read(&request.process_result).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    validate(&request,&result)?;
    if request.out_dir.exists() || output.exists() {return Err("Native plots need new output paths; existing files are preserved".into());}
    // create_dir is exclusive even when another process races this invocation.
    let parent=request.out_dir.parent().ok_or("Native plots need a parent directory")?;
    fs::create_dir_all(parent).map_err(|e|e.to_string())?;
    fs::create_dir(&request.out_dir).map_err(|e|e.to_string())?;
    let out_dir=fs::canonicalize(&request.out_dir).map_err(|e|e.to_string())?;
    let identity=super::GridIdentity{domain:Some(result.domain.clone()),spacing_m:request.spacing_m};
    let mut limits=BatchRenderLimits::default();
    limits.max_products_per_hour=96;limits.max_work_items=96;
    limits.max_output_width=request.width;limits.max_output_height=request.height;
    limits.max_output_pixels=u64::from(request.width)*u64::from(request.height);
    rustwx_render::install_theme(rustwx_render::RenderTheme::default_theme())?;
    let batch=BatchRenderRequest{store_root:result.frame.store_root.clone(),model_slug:result.frame.model_slug.clone(),
        run_slug:result.frame.run_slug.clone(),hours:BatchHourScope::Current(result.frame.storage_slot),
        product_spec:request.products.join(","),out_dir:out_dir.clone(),domain:BatchRenderDomain::NativeGrid,
        native_domain_slug:super::native_domain_slug(&identity),subtitle_spacing:request.spacing_m.and_then(super::spacing_subtitle),
        source_label:Some("ArWen".into()),title_provenance:TitleProvenance::LocalImport{grid_label:super::domain_title_label(&identity)},
        date_yyyymmdd:None,cycle_utc:None,source:None,output_width:request.width,output_height:request.height,limits,
        geographic_overlays:None,panel_annotations:None};
    let mut georefs=Vec::new();let mut panels=Vec::new();let mut failures=Vec::new();
    let summary=run_batch_render(batch,&AtomicBool::new(false),|event|match event{
        BatchRenderEvent::ItemRendered{slug,output_path,georeference,georeference_absent_reason,completed,total,..}=>{
            eprintln!("NATIVE_PLOTS {completed}/{total} {slug}");
            panels.push((slug,output_path.clone()));georefs.push((output_path,georeference,georeference_absent_reason));
        },
        BatchRenderEvent::ItemFailed{slug,error,..}=>failures.push(format!("{slug}: {error}")),
        BatchRenderEvent::ItemSkipped{slug,reason,..}=>failures.push(format!("{slug}: {reason}")),
        _=>{}
    })?;
    if summary.cancelled || summary.failed>0 || summary.skipped>0 || summary.rendered!=request.products.len(){
        return Err(format!("Native plots incomplete ({}/{}): {}",summary.rendered,request.products.len(),failures.join("; ")));
    }
    verify_result(&result)?;
    super::write_georef_manifest(&out_dir,&georefs)?;
    let panels=panels.into_iter().map(|(slug,path)|Ok(Panel{slug,sha256:sha256(&path)?,bytes:fs::metadata(&path).map_err(|e|e.to_string())?.len(),path}))
        .collect::<Result<Vec<_>,String>>()?;
    let value=serde_json::json!({"schema":RESULT_SCHEMA,"frame_id":result.frame.id,"identity":result.frame.identity,
        "domain":result.domain,"grid_sha256":result.frame.grid_sha256,"source_store_sha256":result.frame.rws_sha256,
        "width":request.width,"height":request.height,"panels":panels,"elapsed_ms":summary.elapsed_ms,
        "processing":"existing_compact_store","wrf_imported":false,"volume_store_created":false});
    let mut file=fs::OpenOptions::new().write(true).create_new(true).open(output).map_err(|e|e.to_string())?;
    use std::io::Write;
    file.write_all(&serde_json::to_vec_pretty(&value).map_err(|e|e.to_string())?).and_then(|_|file.sync_all()).map_err(|e|e.to_string())
}

pub fn try_cli(args:&[String])->Option<Result<(),String>>{
    if !args.iter().any(|a|a=="--render-store-request"||a=="--render-store-result"){return None;}
    Some((||{
        if args.len()!=4 || args[0]!="--render-store-request" || args[2]!="--render-store-result" {
            return Err("Use --render-store-request REQUEST.json --render-store-result RESULT.json".into());
        }
        let input=Path::new(&args[1]);let output=Path::new(&args[3]);
        if fs::metadata(input).map_err(|e|e.to_string())?.len()>128*1024{return Err("Native plot request exceeds its metadata bound".into());}
        let request:Request=serde_json::from_slice(&fs::read(input).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        render(request,output)
    })())
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn store_render_grammar_never_enters_the_wrf_import_cli(){
        assert!(try_cli(&["--help".into()]).is_none());
        assert!(try_cli(&["--render-store-request".into(),"missing.json".into(),"--input".into(),"source.nc".into()]).unwrap().is_err());
        assert!(serde_json::from_value::<Request>(serde_json::json!({"schema":REQUEST_SCHEMA,"process_result":"receipt.json","expected_frame_id":"id","expected_source_sha256":"hash","out_dir":"plots","products":["2m_temperature"],"wrf_input":"forbidden.nc"})).is_err());
    }
}
