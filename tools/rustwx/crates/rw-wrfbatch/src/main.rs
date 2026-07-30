//! Headless raw-WRF import plus production Rusty Weather batch rendering.
//!
//! This intentionally reuses the UI crate's hardened WRF processing modules;
//! it only replaces the egui orchestration with a bounded command-line job.
//!
//! gpuwm adaptations over the source workspace's rw_wrf_batch bin:
//! every stored frame renders by default (`--frames all`), a single frame
//! is selectable by stored slot (`--frames N`), and the work limits are
//! sized to the request instead of the GUI's one-hour click ceilings --
//! which is what lets exact-time (sub-hourly) imports render every frame.

#[path = "grib_import.rs"]
mod grib_import;
#[path = "local_import.rs"]
mod local_import;
#[path = "postproc_severe.rs"]
mod postproc_severe;
#[path = "wrf_process.rs"]
mod wrf_process;
#[path = "wrf_volumes.rs"]
mod wrf_volumes;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use rusty_weather::batch_render::{
    BatchHourScope, BatchRenderDomain, BatchRenderEvent, BatchRenderLimits, BatchRenderRequest,
    inspect_renderable_products, run_batch_render,
};
use wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};

#[derive(Debug)]
struct Args {
    store_root: PathBuf,
    out_dir: PathBuf,
    products: String,
    frames: Option<usize>,
    width: u32,
    height: u32,
    heavy: bool,
    list_products: bool,
    inputs: Vec<PathBuf>,
}

fn usage() -> &'static str {
    "usage: rw_wrfbatch --store-root DIR --out-dir DIR [--products all|SLUGS] \
[--frames all|N] [--width N] [--height N] [--heavy] [--list-products] wrfout..."
}

fn parse_args() -> Result<Args, String> {
    let mut store_root = None;
    let mut out_dir = None;
    let mut products = "all".to_string();
    let mut frames = None;
    let mut width = 1_200u32;
    let mut height = 900u32;
    let mut heavy = false;
    let mut list_products = false;
    let mut inputs = Vec::new();
    let mut raw = std::env::args().skip(1);

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--store-root" => {
                store_root = Some(PathBuf::from(
                    raw.next().ok_or("--store-root requires a directory")?,
                ));
            }
            "--out-dir" => {
                out_dir = Some(PathBuf::from(
                    raw.next().ok_or("--out-dir requires a directory")?,
                ));
            }
            "--products" => {
                products = raw.next().ok_or("--products requires a value")?;
            }
            "--frames" => {
                let value = raw.next().ok_or("--frames requires 'all' or an index")?;
                if !value.eq_ignore_ascii_case("all") {
                    frames = Some(
                        value
                            .parse::<usize>()
                            .map_err(|err| format!("invalid --frames: {err}"))?,
                    );
                }
            }
            "--width" => {
                width = raw
                    .next()
                    .ok_or("--width requires a value")?
                    .parse()
                    .map_err(|err| format!("invalid --width: {err}"))?;
            }
            "--height" => {
                height = raw
                    .next()
                    .ok_or("--height requires a value")?
                    .parse()
                    .map_err(|err| format!("invalid --height: {err}"))?;
            }
            "--heavy" => heavy = true,
            "--list-products" => list_products = true,
            "--help" | "-h" => return Err(usage().to_string()),
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}")),
            _ => inputs.push(PathBuf::from(arg)),
        }
    }

    if inputs.is_empty() {
        return Err("at least one wrfout input is required".to_string());
    }
    Ok(Args {
        store_root: store_root.ok_or("--store-root is required")?,
        out_dir: out_dir.ok_or("--out-dir is required")?,
        products,
        frames,
        width,
        height,
        heavy,
        list_products,
        inputs,
    })
}

fn run(args: Args) -> Result<(), String> {
    let options = WrfProcessOptions {
        heavy_ecape: args.heavy,
        ..WrfProcessOptions::default()
    };
    let task = spawn_process_paths(args.inputs, args.store_root.clone(), options);
    let import = loop {
        match task
            .rx
            .recv()
            .map_err(|err| format!("WRF processor exited without a result: {err}"))?
        {
            WrfProcessMessage::Progress(message) => println!("PROCESS {message}"),
            WrfProcessMessage::Done(result) => break result?,
        }
    };
    println!(
        "IMPORTED model={} run={} files={} hours={} variables={} notes={}",
        import.model,
        import.run,
        import.files_seen,
        import.hours_written,
        import.variables.len(),
        import.notes.len()
    );

    let run_manifest = args
        .store_root
        .join(&import.model)
        .join(&import.run)
        .join("run.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&run_manifest)
            .map_err(|err| format!("read {}: {err}", run_manifest.display()))?,
    )
    .map_err(|err| format!("parse {}: {err}", run_manifest.display()))?;
    let stored_slots: Vec<u16> = {
        let mut slots: Vec<u16> = manifest
            .get("hours")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("{} has no hours object", run_manifest.display()))?
            .keys()
            .filter_map(|key| key.parse::<u16>().ok())
            .collect();
        slots.sort_unstable();
        slots
    };
    let first_slot = stored_slots
        .first()
        .copied()
        .ok_or_else(|| format!("{} has no stored forecast slots", run_manifest.display()))?;
    if args.list_products {
        return list_products(
            &args.store_root,
            &import.model,
            &import.run,
            &stored_slots,
            args.heavy,
        );
    }

    // --frames N is an ordinal index into the ascending stored slots (the
    // Nth stored frame), matching `gpuwm render --timeidx`: within one
    // imported file the plan orders slots by valid time, so the index is
    // the frame index whether the axis is whole-hour (slot = forecast
    // hour) or exact-time (slot = ordinal).
    let hour_scope = match args.frames {
        None => BatchHourScope::AllStored,
        Some(index) => {
            let slot = stored_slots.get(index).copied().ok_or_else(|| {
                format!(
                    "--frames {index} out of range; the store has {} frame(s)",
                    stored_slots.len()
                )
            })?;
            BatchHourScope::Current(slot)
        }
    };
    let catalog =
        inspect_renderable_products(&args.store_root, &import.model, &import.run, first_slot)?;
    let product_spec = if args.products.eq_ignore_ascii_case("all") {
        catalog
            .products
            .iter()
            .map(|product| product.slug.as_str())
            .collect::<Vec<_>>()
            .join(",")
    } else {
        args.products
    };
    println!(
        "CATALOG products={} stored_hours={:?}",
        catalog.products.len(),
        catalog.stored_hours
    );

    // Size the limits to the request: this is a command-line job whose
    // work is exactly frames x products, not a GUI guarding against an
    // accidental unbounded click.
    let selected_frames = match hour_scope {
        BatchHourScope::AllStored => stored_slots.len().max(1),
        BatchHourScope::Current(_) => 1,
    };
    let per_frame_products = product_spec
        .split(',')
        .filter(|slug| !slug.trim().is_empty())
        .count();
    let mut limits = BatchRenderLimits::default();
    limits.max_hours = limits.max_hours.max(selected_frames);
    limits.max_products_per_hour = limits.max_products_per_hour.max(per_frame_products);
    limits.max_work_items = limits.max_work_items.max(
        selected_frames
            .saturating_mul(per_frame_products)
            .saturating_add(per_frame_products),
    );
    limits.max_output_width = args.width.max(limits.max_output_width);
    limits.max_output_height = args.height.max(limits.max_output_height);
    limits.max_output_pixels = u64::from(args.width) * u64::from(args.height);
    let request = BatchRenderRequest {
        store_root: args.store_root,
        model_slug: import.model,
        run_slug: import.run,
        hours: hour_scope,
        product_spec,
        out_dir: args.out_dir,
        domain: BatchRenderDomain::NativeGrid,
        date_yyyymmdd: None,
        cycle_utc: None,
        source: None,
        output_width: args.width,
        output_height: args.height,
        limits,
    };
    let cancel = AtomicBool::new(false);
    let summary = run_batch_render(request, &cancel, |event| match event {
        BatchRenderEvent::Started {
            planned_items,
            output_dir,
            ..
        } => println!(
            "RENDER planned={planned_items} out={}",
            output_dir.display()
        ),
        BatchRenderEvent::ItemRendered {
            slug, output_path, ..
        } => println!("RENDERED {slug} {}", output_path.display()),
        BatchRenderEvent::ItemSkipped { slug, reason, .. } => {
            println!("SKIPPED {slug} {reason}")
        }
        BatchRenderEvent::ItemFailed { slug, error, .. } => {
            eprintln!("FAILED {slug} {error}")
        }
        BatchRenderEvent::Finished(summary) => println!(
            "FINISHED rendered={} skipped={} failed={} elapsed_ms={}",
            summary.rendered, summary.skipped, summary.failed, summary.elapsed_ms
        ),
        _ => {}
    })?;
    if summary.rendered == 0 || summary.failed > 0 {
        return Err(format!(
            "batch render incomplete: rendered={} skipped={} failed={}",
            summary.rendered, summary.skipped, summary.failed
        ));
    }
    Ok(())
}

/// One PRODUCT row per catalog entry, with per-store availability:
/// `PRODUCT\t<slug>\t<kind>\t<status>\t<detail>`, then one CATALOG
/// summary line.  Statuses: `renderable` (proven against the imported
/// store), `missing-fields` (direct recipe whose required fields are
/// not all stored), `blocked` (windowed compute reported an honest
/// per-product blocker), and `excluded` (recipe cannot be realized by
/// this lane, reason given).  Every decision routes through stored
/// FIELD availability; there is no model-identity gate and no `gated`
/// status -- a product that cannot render always names the fields it
/// is missing (gpuwm architectural rule; the listing test rejects any
/// identity-gated row).
fn list_products(
    store_root: &std::path::Path,
    model_slug: &str,
    run_slug: &str,
    stored_slots: &[u16],
    heavy_imported: bool,
) -> Result<(), String> {
    use rusty_weather::render_all::StoreFieldSource;
    use rusty_weather::render_all::windowed_store;

    let first_slot = stored_slots
        .first()
        .copied()
        .ok_or("catalog listing needs at least one stored frame")?;
    let store = StoreFieldSource::open(store_root, model_slug, run_slug, first_slot)
        .map_err(|err| err.to_string())?;
    let stored_derived: std::collections::HashSet<&str> = store
        .derived_slugs()
        .iter()
        .map(String::as_str)
        .collect();

    let mut rows: Vec<(String, &str, &str, String)> = Vec::new();

    for spec in rustwx_products::spec::direct_product_specs() {
        if rustwx_products::direct::direct_recipe_requires_explicit_opt_in(&spec.slug) {
            continue;
        }
        match rustwx_models::plot_recipe_store_requirements(&spec.slug) {
            Err(err) => rows.push((
                spec.slug,
                "direct",
                "excluded",
                format!("catalog spec has no plot recipe: {err}"),
            )),
            Ok(requirements) => {
                let missing: Vec<String> = requirements
                    .iter()
                    .filter_map(|requirement| match requirement.selector {
                        Some(selector) => store
                            .resolve(&selector)
                            .is_none()
                            .then(|| selector.key()),
                        None => Some(format!(
                            "{} (no canonical store selector exists for this field)",
                            requirement.field_key
                        )),
                    })
                    .collect();
                if missing.is_empty() {
                    rows.push((spec.slug, "direct", "renderable", spec.title));
                } else {
                    rows.push((
                        spec.slug,
                        "direct",
                        "missing-fields",
                        format!("not stored: {}", missing.join(", ")),
                    ));
                }
            }
        }
    }

    for entry in rustwx_products::derived::supported_derived_recipe_inventory() {
        let kind = if entry.heavy { "heavy" } else { "derived" };
        if stored_derived.contains(entry.slug) {
            rows.push((entry.slug.to_string(), kind, "renderable", entry.title.to_string()));
        } else if entry.heavy {
            let reason = if heavy_imported {
                "the wrfout lane's heavy (ECAPE) diagnostics do not produce \
                 this recipe's grid (ml/mu parcels and the CAPE-ratio pairs \
                 need import-side plumbing wrf-core does not expose yet)"
            } else {
                "heavy grid not computed at import; re-run with --heavy \
                 to compute the ECAPE family"
            };
            rows.push((entry.slug.to_string(), kind, "excluded", reason.to_string()));
        } else {
            rows.push((
                entry.slug.to_string(),
                kind,
                "excluded",
                "not realized by the wrfout import lane (no matching \
                 wrf-core diagnostic is stored under this recipe slug)"
                    .to_string(),
            ));
        }
    }
    for entry in rustwx_products::derived::blocked_derived_recipe_inventory() {
        rows.push((
            entry.slug.to_string(),
            "derived",
            "excluded",
            entry.reason.to_string(),
        ));
    }

    let windowed_ready = windowed_store::windowed_axis_ready(store_root, model_slug, run_slug)
        .map_err(|err| err.to_string())?;
    let windowed_slugs: Vec<String> =
        rustwx_products::windowed::HrrrWindowedProduct::supported_products()
            .iter()
            .map(|product| product.slug().to_string())
            .collect();
    if !windowed_ready {
        let reason = if stored_slots.len() <= 1 {
            "windowed accumulations need more than one stored whole-hour frame"
        } else {
            "exact-time ordinal axis; fixed-hour windows are undefined on it"
        };
        for slug in &windowed_slugs {
            rows.push((slug.clone(), "windowed", "excluded", reason.to_string()));
        }
    } else {
        match windowed_store::compute_windowed_products(
            store_root,
            model_slug,
            run_slug,
            stored_slots,
            &windowed_slugs,
        ) {
            Ok(outcome) => {
                let blocked: std::collections::HashMap<String, String> =
                    outcome.blockers.into_iter().collect();
                for grid in &outcome.grids {
                    rows.push((
                        grid.slug.clone(),
                        "windowed",
                        "renderable",
                        grid.strategy.clone(),
                    ));
                }
                for slug in &windowed_slugs {
                    if let Some(reason) = blocked.get(slug) {
                        rows.push((slug.clone(), "windowed", "blocked", reason.clone()));
                    }
                }
            }
            Err(err) => {
                let reason = format!("windowed compute unavailable: {err}");
                for slug in &windowed_slugs {
                    rows.push((slug.clone(), "windowed", "excluded", reason.clone()));
                }
            }
        }
    }

    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for (slug, kind, status, detail) in &rows {
        *counts.entry(status).or_default() += 1;
        println!("PRODUCT\t{slug}\t{kind}\t{status}\t{detail}");
    }
    let summary = counts
        .iter()
        .map(|(status, count)| format!("{status}={count}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!("CATALOG total={} {summary}", rows.len());
    Ok(())
}

fn main() -> ExitCode {
    match parse_args().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            if error == usage() {
                ExitCode::SUCCESS
            } else {
                eprintln!("{}", usage());
                ExitCode::FAILURE
            }
        }
    }
}
