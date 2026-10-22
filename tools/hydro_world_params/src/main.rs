const USAGE: &str = "usage: rw_hydro_params derive RUN.json | table PARAMS.json";

/// The commit this binary was built from; the release pin step reads it from the bytes.
const SOURCE_REV_STAMP: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() {
    std::hint::black_box(SOURCE_REV_STAMP);
    let a: Vec<String> = std::env::args().collect();
    let r = match (a.get(1).map(String::as_str), a.get(2)) {
        (Some("derive"), Some(p)) => hydro_world_params::run::run(p).map(|rec| {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "domain": rec["domain"], "catchments": rec["catchments"]["count"], "complete_and_bounded": rec["complete_and_bounded"],
                    "violations": rec["violations"], "fallbacks": rec["fallbacks"], "total_seconds": rec["total_seconds"], "out_dir": rec["config"]["out_dir"]
                }))
                .unwrap()
            );
            if rec["complete_and_bounded"] != serde_json::json!(true) {
                std::process::exit(2)
            }
        }),
        (Some("table"), Some(p)) => hydro_world_params::table::ParamTable::load(p).map(|t| println!("{}", serde_json::to_string_pretty(&t).unwrap())),
        _ => Err(USAGE.into()),
    };
    if let Err(e) = r {
        eprintln!("rw_hydro_params: {e}");
        std::process::exit(1)
    }
}
