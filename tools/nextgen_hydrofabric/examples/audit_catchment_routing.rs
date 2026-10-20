//! Preparation diagnostic, not an initializer or a coupled-physics admission.
//! Uses only the existing sealed prepared schema; never infers IDs or areas.
use nextgen_hydrofabric::{manifest::PreparedManifest, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path, time::Instant};

fn strings(value: &Value) -> Result<Vec<&str>> {
    value
        .as_array()
        .ok_or("expected string roster")?
        .iter()
        .map(|v| v.as_str().ok_or_else(|| "non-string roster entry".into()))
        .collect()
}

fn payload(prepared: &PreparedManifest, name: &str, dtype: &str, n: usize) -> Result<Vec<u8>> {
    let field = prepared
        .document
        .files
        .get(name)
        .ok_or("missing transfer field")?;
    if field.dtype != dtype || field.shape != [n] {
        return Err(format!("wrong dtype/shape: {name}").into());
    }
    let mut bytes = vec![0; field.bytes];
    prepared.read_field_into(name, &mut bytes)?;
    Ok(bytes)
}

fn run() -> Result<Value> {
    let start = Instant::now();
    let args = std::env::args().collect::<Vec<_>>();
    if !(2..=3).contains(&args.len()) {
        return Err("audit_catchment_routing FABRIC_MANIFEST [CFE_INITIAL_MANIFEST]".into());
    }
    let path = Path::new(&args[1]);
    let prepared = PreparedManifest::open(path)?;
    let document = &prepared.document;
    let network: Value = serde_json::from_slice(&prepared.read_auxiliary("network.json")?)?;
    let catchments = strings(&network["catchment_ids"])?;
    let nodes = strings(&network["node_ids"])?;
    if catchments.len() != document.catchment_count
        || nodes.len() != document.node_count
        || catchments.is_empty()
        || nodes.is_empty()
        || catchments.iter().collect::<HashSet<_>>().len() != catchments.len()
        || nodes.iter().collect::<HashSet<_>>().len() != nodes.len()
    {
        return Err("empty/duplicate/incorrect network roster".into());
    }
    let area_bytes = payload(&prepared, "catchment_area_m2", "<f8", catchments.len())?;
    let receiver_bytes = payload(&prepared, "catchment_to_node", "<i4", catchments.len())?;
    let network_receivers = network["catchment_to_node"]
        .as_array()
        .ok_or("missing network receivers")?;
    if network_receivers.len() != catchments.len() {
        return Err("network receiver count differs".into());
    }
    let mut minimum = f64::INFINITY;
    let mut maximum = 0f64;
    let mut sum = 0f64;
    let mut receivers = HashSet::new();
    for (i, (area, receiver)) in area_bytes
        .chunks_exact(8)
        .zip(receiver_bytes.chunks_exact(4))
        .enumerate()
    {
        let area = f64::from_le_bytes(area.try_into()?);
        let receiver = i32::from_le_bytes(receiver.try_into()?);
        if !area.is_finite() || area <= 0. {
            return Err(format!("nonpositive/nonfinite polygon area at row {i}").into());
        }
        if receiver < 0
            || receiver as usize >= nodes.len()
            || network_receivers[i].as_i64() != Some(receiver as i64)
        {
            return Err(format!("invalid or inconsistent catchment receiver at row {i}").into());
        }
        minimum = minimum.min(area);
        maximum = maximum.max(area);
        sum += area;
        receivers.insert(receiver);
    }
    if !sum.is_finite() {
        return Err("polygon area sum overflow".into());
    }
    let fields = ["catchment_area_m2", "catchment_to_node"].into_iter().map(|name| {
        let f = &document.files[name];
        (name.to_owned(), json!({"path":f.path,"dtype":f.dtype,"shape":f.shape,"bytes":f.bytes,"sha256":f.sha256}))
    }).collect::<serde_json::Map<_, _>>();
    let producer = if let Some(producer_path) = args.get(2) {
        let bytes = fs::read(producer_path)?;
        let packet: Value = serde_json::from_slice(&bytes)?;
        if packet["schema"] != "cfe.initial.pack/v1" {
            return Err("producer comparison requires cfe.initial.pack/v1".into());
        }
        let ids = strings(&packet["catchments"])?;
        if ids.is_empty() || ids.iter().collect::<HashSet<_>>().len() != ids.len() {
            return Err("empty/duplicate producer catchments".into());
        }
        let available = catchments.iter().copied().collect::<HashSet<_>>();
        let matched = ids.iter().filter(|id| available.contains(**id)).count();
        json!({"path":producer_path,"sha256":format!("{:x}",Sha256::digest(&bytes)),
            "scope":"roster metadata only; producer payloads/parameters/state/area convention not audited",
            "catchment_count":ids.len(),"matched_catchments":matched,
            "unmatched_catchments":ids.len()-matched,"exact_order_equal":ids==catchments,
            "time_origin":packet["time_origin"],"period_seconds":packet["period_seconds"]})
    } else {
        Value::Null
    };
    Ok(json!({"schema":"arwen.nextgen.catchment-routing-audit.v1",
        "source_revision":nextgen_hydrofabric::SOURCE_REVISION,
        "scope":"sealed polygon area and explicit receiver inventory only; not CFE parameters, disjoint source area, connected native phase, or physical forecast qualification",
        "manifest":{"path":path,"sha256":format!("{:x}",Sha256::digest(fs::read(path)?)),
            "request_sha256":document.request_sha256},
        "source":document.input.path,"source_sha256":document.input.sha256,"data_license":document.input.data_license,
        "grid":document.metadata.get("grid"),"replay_provenance":document.metadata.get("replay_provenance"),
        "catchment_count":catchments.len(),"node_count":nodes.len(),"unique_receiver_nodes":receivers.len(),
        "network_sha256":document.auxiliary["network.json"].sha256,"fields":fields,
        "polygon_area_m2":{"minimum":minimum,"maximum":maximum,"sum_in_roster_order":sum,
            "sum_scope":"sum of polygon areas; overlapping polygons are not removed; not a disjoint drainage area"},
        "network_receiver_words_different":0,"producer":producer,
        "inventory_valid":true,"physical_transfer_admitted":false,
        "timings_ms":{"total":start.elapsed().as_secs_f64()*1000.},"gpu":"none; preparation-only Rust"}))
}

fn main() {
    match run() {
        Ok(value) => println!("{}", serde_json::to_string_pretty(&value).unwrap()),
        Err(error) => {
            eprintln!("audit_catchment_routing: {error}");
            std::process::exit(1);
        }
    }
}
