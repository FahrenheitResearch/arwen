//! Convert MPAS history frames into wrfout-shaped frames the renderer takes.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rw_mpas::composite::{CompositePlan, CompositeSource, COMPOSITE_CONTEXT_FACTOR};
use rw_mpas::convert::{
    convert_frame, convert_frame_composite, CompositeGather, CompositeLayer, ConvertOptions,
};
use rw_mpas::history::{assert_mesh_agrees, read_history, MpasFrame, Timestamp};
use rw_mpas::weights::{
    build_weights, read_mesh_cell_spacing, read_mesh_coordinates, ON_MESH_SPACING_FACTOR,
};
use rw_mpas::window::Window;
use rw_store::netcdf_classic::NcFormat;

const USAGE: &str = "\
usage: rw_mpas_convert --history FILE... --mesh FILE --out-dir DIR
                       [--init FILE] [--mesh-sha256 HEX] [--init-sha256 HEX]
                       [--window focus|global|mesh|composite]
                       [--compose SOURCES.json] [--compose-base-only]
                       [--field-set full|surface]
                       [--simulation-start YYYY-MM-DD_HH:MM:SS]
                       [--prefix NAME] [--format cdf2|cdf5] [--clobber]
                       [--json FILE]
       rw_mpas_convert --help | --abi

windows:
  focus   a fixed 240x150 Lambert box at 22 km over the CONUS. Frozen
          geometry: its weights digests are already recorded in evidence.
  global  a fixed 0.25 degree lat-lon overview of the whole sphere.
  mesh    DERIVED from --mesh: the cells within 2x the mesh's minimum
          spacing are its refined region, and the window is centred on
          them, at their median spacing, covering their extent. A mesh
          refined anywhere renders its own fine region with this, and no
          new box has to be added for the next placement.
  composite  DERIVED from --compose: one frame covering every fine region
          in the source list, at the finest spacing among them, pulled out
          to 3x their extent so the coarse field around them is visible.

--compose SOURCES.json composes several runs of the same hours into one
frame -- the coarse run everywhere, each fine run inside the ground it
actually resolves:

  {\"base\":     {\"label\": \"coarse\", \"mesh\": \"g96.static.nc\",
                \"history\": [\"...06.nc\", \"...09.nc\"]},
   \"overlays\": [{\"label\": \"s01\", \"mesh\": \"s01.static.nc\",
                 \"history\": [\"...06.nc\", \"...09.nc\"]}]}

A point takes the finest source whose refined region it falls inside, and
the base everywhere else. The change of source is a hard switch: nothing is
blended, and the frame carries COMPOSITE_SOURCE and COMPOSITE_DX_KM so the
seam can be drawn and audited.

--compose-base-only frames the same window from the same sources and then
converts the BASE alone, so the composite has a coarse counterpart over
exactly the same ground to be compared against.";

/// The window name that is derived rather than served from the catalogue.
const MESH_WINDOW: &str = "mesh";

/// The tape contract: the output schema name and the stdout progress
/// tokens, which is the literal a stale build gets wrong. It stays `v1`
/// across the derived window because the stdout grammar is untouched --
/// `--window mesh` reports what it derived on STDERR, the way the mesh
/// reader already advises about inferred angle units. `gpuwm.bridges` and
/// `gpuwm.mpas_mesh` both carry this literal; a change here is a change
/// there.
const ABI: &str = "gpuwm-rw-mpas-convert-v1\tCONVERTED\tWINDOW\tWEIGHTS\tFINISHED";

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes alone
/// (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs` injects
/// the value; `main` references the constant so the linker cannot discard
/// it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

struct Args {
    history: Vec<PathBuf>,
    mesh: Option<PathBuf>,
    mesh_sha256: Option<String>,
    init: Option<PathBuf>,
    init_sha256: Option<String>,
    out_dir: Option<PathBuf>,
    window: String,
    field_set: String,
    simulation_start: Option<String>,
    prefix: String,
    format: NcFormat,
    clobber: bool,
    json: Option<PathBuf>,
    compose: Option<PathBuf>,
    compose_base_only: bool,
}

/// One run named in a `--compose` source list.
#[derive(Debug, serde::Deserialize)]
struct ComposeEntry {
    label: String,
    mesh: PathBuf,
    history: Vec<PathBuf>,
}

#[derive(Debug, serde::Deserialize)]
struct ComposeSpec {
    base: ComposeEntry,
    #[serde(default)]
    overlays: Vec<ComposeEntry>,
}

/// One source, read and resampled onto the shared window.
struct LoadedSource {
    label: String,
    source: CompositeSource,
    frames: Vec<MpasFrame>,
}

fn parse() -> Result<Args, String> {
    let mut args = Args {
        history: Vec::new(),
        mesh: None,
        mesh_sha256: None,
        init: None,
        init_sha256: None,
        out_dir: None,
        window: "focus".to_string(),
        field_set: "full".to_string(),
        simulation_start: None,
        prefix: "wrfout_d01".to_string(),
        format: NcFormat::Offset64,
        clobber: false,
        json: None,
        compose: None,
        compose_base_only: false,
    };
    let mut raw = std::env::args().skip(1).peekable();
    while let Some(flag) = raw.next() {
        let mut value = || -> Result<String, String> {
            raw.next()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--history" => {
                while let Some(next) = raw.peek() {
                    if next.starts_with("--") {
                        break;
                    }
                    args.history.push(PathBuf::from(raw.next().unwrap()));
                }
                if args.history.is_empty() {
                    return Err("--history needs at least one file".to_string());
                }
            }
            "--mesh" => args.mesh = Some(PathBuf::from(value()?)),
            "--mesh-sha256" => args.mesh_sha256 = Some(value()?),
            "--init" => args.init = Some(PathBuf::from(value()?)),
            "--init-sha256" => args.init_sha256 = Some(value()?),
            "--out-dir" => args.out_dir = Some(PathBuf::from(value()?)),
            "--window" => args.window = value()?,
            "--field-set" => args.field_set = value()?,
            "--simulation-start" => args.simulation_start = Some(value()?),
            "--prefix" => args.prefix = value()?,
            "--format" => {
                args.format = match value()?.as_str() {
                    "cdf2" => NcFormat::Offset64,
                    "cdf5" => NcFormat::Data64,
                    other => return Err(format!("--format '{other}' is not cdf2 or cdf5")),
                }
            }
            "--clobber" => args.clobber = true,
            "--json" => args.json = Some(PathBuf::from(value()?)),
            "--compose" => args.compose = Some(PathBuf::from(value()?)),
            "--compose-base-only" => args.compose_base_only = true,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(args)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Read one source's mesh, resample it onto `window`, and read its frames.
fn load_source(
    entry: &ComposeEntry,
    window: &Window,
    window_name: &str,
    is_base: bool,
) -> Result<LoadedSource, String> {
    if entry.history.is_empty() {
        return Err(format!(
            "composite source '{}' names no history file, so it has nothing to contribute at \
             any hour",
            entry.label
        ));
    }
    let mesh = read_mesh_coordinates(&entry.mesh, None).map_err(|e| e.to_string())?;
    let spacing = read_mesh_cell_spacing(&entry.mesh).map_err(|e| e.to_string())?;
    if spacing.spacing_metres.len() != mesh.n_cells() {
        return Err(format!(
            "composite source '{}' mesh {} carries {} cell centre(s) but {} spacing value(s)",
            entry.label,
            entry.mesh.display(),
            mesh.n_cells(),
            spacing.spacing_metres.len()
        ));
    }
    let weights = build_weights(&mesh, window_name, window).map_err(|e| e.to_string())?;
    let source = if is_base {
        CompositeSource::base(entry.label.clone(), &mesh, weights, spacing.spacing_metres)
    } else {
        CompositeSource::overlay(entry.label.clone(), &mesh, weights, spacing.spacing_metres)
    }
    .map_err(|e| e.to_string())?;
    let mut frames = Vec::with_capacity(entry.history.len());
    for history in &entry.history {
        let frame = read_history(history, None, None, None, None).map_err(|e| e.to_string())?;
        assert_mesh_agrees(&frame, &mesh, 1.0e-4).map_err(|e| e.to_string())?;
        frames.push(frame);
    }
    Ok(LoadedSource {
        label: entry.label.clone(),
        source,
        frames,
    })
}

/// Read the refined cells of every overlay and size one window over all of
/// them.  It reads the meshes itself rather than taking loaded sources
/// because the weights cannot be built until the window exists, and the
/// window is what this returns.
fn composite_window(overlays: &[&ComposeEntry]) -> Result<Window, String> {
    if overlays.is_empty() {
        return Err("--window composite sizes itself to the fine regions in --compose; the \
                    source list names no overlay, so there is no fine region to frame and \
                    nothing to composite. Use --window global for the base alone"
            .to_string());
    }
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    let mut spacing = Vec::new();
    for entry in overlays {
        let mesh = read_mesh_coordinates(&entry.mesh, None).map_err(|e| e.to_string())?;
        let spacings = read_mesh_cell_spacing(&entry.mesh).map_err(|e| e.to_string())?;
        let refined = Window::refined_region(&spacings.spacing_metres).map_err(|e| {
            format!(
                "composite source '{}' has no refined region to frame: {e}",
                entry.label
            )
        })?;
        for cell in refined {
            lat.push(mesh.latitude_degrees[cell]);
            lon.push(mesh.longitude_degrees[cell]);
            spacing.push(spacings.spacing_metres[cell]);
        }
    }
    Window::composite_focus(&lat, &lon, &spacing, COMPOSITE_CONTEXT_FACTOR)
        .map_err(|e| e.to_string())
}

fn run_compose(args: Args, spec_path: &Path) -> Result<(), String> {
    if !args.history.is_empty() || args.mesh.is_some() {
        return Err(
            "--compose owns the mesh and history list; passing --mesh or --history as well \
             would leave two source lists that can disagree about which run is the base"
                .to_string(),
        );
    }
    let out_dir = args.out_dir.clone().ok_or("--out-dir is required")?;
    let simulation_start = match &args.simulation_start {
        Some(text) => Some(Timestamp::parse(text).map_err(|e| e.to_string())?),
        None => None,
    };
    let bytes =
        std::fs::read(spec_path).map_err(|e| format!("read {}: {e}", spec_path.display()))?;
    let spec: ComposeSpec = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse {}: {e}", spec_path.display()))?;

    let started = std::time::Instant::now();
    let overlay_refs: Vec<&ComposeEntry> = spec.overlays.iter().collect();
    let window = match args.window.as_str() {
        "composite" => composite_window(&overlay_refs)?,
        MESH_WINDOW => {
            return Err(
                "--window mesh derives from ONE mesh's refined region; a composite has \
                 several. Use --window composite, which frames all of them at once"
                    .to_string(),
            )
        }
        other => Window::named(other).map_err(|e| e.to_string())?,
    };
    eprintln!(
        "rw_mpas_convert: advisory: composite window: {}",
        window.description()
    );

    let base = load_source(&spec.base, &window, &args.window, true)?;
    let mut overlays = Vec::with_capacity(spec.overlays.len());
    for entry in &spec.overlays {
        overlays.push(load_source(entry, &window, &args.window, false)?);
    }
    // --compose-base-only builds the SAME window from the same overlays and
    // then converts the base alone. It exists so the composite can be
    // compared against what the coarse run shows over the identical ground:
    // without it the only coarse counterpart is on a different frame, and a
    // comparison across two frames cannot separate what the fine data added
    // from what the reframing did.
    let overlay_sources: Vec<&CompositeSource> = if args.compose_base_only {
        Vec::new()
    } else {
        overlays.iter().map(|o| &o.source).collect()
    };
    let plan = CompositePlan::build(&base.source, &overlay_sources).map_err(|e| e.to_string())?;

    println!(
        "WINDOW\t{}\t{}x{}\tcells={}",
        args.window,
        plan.ny,
        plan.nx,
        base.source.weights.n_cells
    );
    for (index, label) in plan.labels.iter().enumerate() {
        println!(
            "COMPOSITE\t{index}\t{label}\tpoints={}\tshare={:.4}",
            plan.counts[index],
            plan.counts[index] as f64 / plan.points() as f64
        );
    }
    println!(
        "COMPOSITE\tseam\tboundary_points={}",
        plan.boundary_slots().len()
    );

    // Only hours every source carries can be composed. A frame missing
    // from one overlay is dropped by name rather than composed without it:
    // silently falling back to the base for that hour would put a coarse
    // patch in the middle of a fine sequence with nothing saying so.
    let mut shared: Vec<Timestamp> = Vec::new();
    for frame in &base.frames {
        let mut everywhere = true;
        for overlay in &overlays {
            if !overlay
                .frames
                .iter()
                .any(|f| f.valid_time.label() == frame.valid_time.label())
            {
                everywhere = false;
                eprintln!(
                    "rw_mpas_convert: advisory: dropping {} -- composite source '{}' has no \
                     frame at that time",
                    frame.valid_time.label(),
                    overlay.label
                );
            }
        }
        if everywhere {
            shared.push(frame.valid_time);
        }
    }
    if shared.is_empty() {
        return Err(format!(
            "no valid time is carried by all {} source(s); the base has {} frame(s) and none \
             of them line up",
            overlays.len() + 1,
            base.frames.len()
        ));
    }

    let mut records: Vec<String> = Vec::new();
    for valid in &shared {
        let base_frame = base
            .frames
            .iter()
            .find(|f| f.valid_time.label() == valid.label())
            .expect("shared times come from the base");
        let mut layers: Vec<CompositeLayer<'_>> = vec![CompositeLayer {
            label: &base.label,
            frame: base_frame,
            weights: &base.source.weights,
        }];
        if !args.compose_base_only {
            for overlay in &overlays {
                let frame = overlay
                    .frames
                    .iter()
                    .find(|f| f.valid_time.label() == valid.label())
                    .expect("shared times are carried by every source");
                layers.push(CompositeLayer {
                    label: &overlay.label,
                    frame,
                    weights: &overlay.source.weights,
                });
            }
        }
        let gather = CompositeGather {
            plan: &plan,
            layers: &layers,
        };
        let stamp = valid.label().replace(':', "_");
        let destination = Path::new(&out_dir).join(format!("{}_{stamp}", args.prefix));
        let options = ConvertOptions {
            simulation_start,
            field_set: args.field_set.clone(),
            format: args.format,
            clobber: args.clobber,
            ..ConvertOptions::default()
        };
        let emitted = convert_frame_composite(
            base_frame,
            &base.source.weights,
            Some(&gather),
            &destination,
            &options,
        )
        .map_err(|e| e.to_string())?;
        println!(
            "CONVERTED\t{}\t{:.3}s\t{} bytes\t{} written\t{} absent",
            emitted.path.display(),
            emitted.seconds,
            emitted.bytes,
            emitted.written.len(),
            emitted.absent.len()
        );
        for stat in &emitted.seam {
            println!(
                "SEAM\t{}\t{}\tpoints={}\tmean={:.4}\tp95={:.4}\tmax={:.4}\tframe_range={:.4}",
                emitted.valid_time.label(),
                stat.field,
                stat.points,
                stat.mean_abs,
                stat.p95_abs,
                stat.max_abs,
                stat.frame_range
            );
        }
        records.push(format!(
            "{{\"seam\": [{}], \"output\": \"{}\", \"output_bytes\": {}, \"valid_time\": \"{}\", \"convert_seconds\": {}, \"written_variables\": [{}], \"absent_wrf_fields\": [{}]}}",
            emitted
                .seam
                .iter()
                .map(|s| s.json())
                .collect::<Vec<_>>()
                .join(", "),
            json_escape(&emitted.path.display().to_string()),
            emitted.bytes,
            emitted.valid_time.iso(),
            emitted.seconds,
            emitted
                .written
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(", "),
            emitted
                .absent
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }

    if let Some(path) = &args.json {
        let body = format!(
            "{{\n  \"schema\": \"{}\",\n  \"engine\": \"rw-mpas {} (rust)\",\n  \"window\": \"{}\",\n  \"window_spec\": {},\n  \"composite\": {},\n  \"resample_method\": \"{}\",\n  \"field_set\": \"{}\",\n  \"frames\": [\n    {}\n  ]\n}}\n",
            rw_mpas::convert::BRIDGE_SCHEMA,
            env!("CARGO_PKG_VERSION"),
            args.window,
            window.spec_json(),
            plan.spec_json(),
            rw_mpas::weights::RESAMPLE_METHOD,
            args.field_set,
            records.join(",\n    ")
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, body).map_err(|e| e.to_string())?;
        println!("REPORT\t{}", path.display());
    }
    println!(
        "FINISHED\tframes={}\telapsed_s={:.3}",
        shared.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse()?;
    if let Some(spec) = args.compose.clone() {
        return run_compose(args, &spec);
    }
    let mesh_path = args.mesh.ok_or("--mesh is required")?;
    let out_dir = args.out_dir.ok_or("--out-dir is required")?;
    if args.history.is_empty() {
        return Err("--history is required".to_string());
    }
    let simulation_start = match &args.simulation_start {
        Some(text) => Some(Timestamp::parse(text).map_err(|e| e.to_string())?),
        None => None,
    };

    let started = std::time::Instant::now();
    let mesh = read_mesh_coordinates(&mesh_path, args.mesh_sha256.as_deref())
        .map_err(|e| e.to_string())?;
    let mesh_seconds = started.elapsed().as_secs_f64();

    // The window is built AFTER the mesh, and the order is the point: a
    // derived window is a property OF the mesh, so it cannot be named before
    // the mesh has been read. The two fixed windows do not care when they are
    // built, and their geometry is untouched by the move.
    let window = if args.window == MESH_WINDOW {
        let spacing = read_mesh_cell_spacing(&mesh_path).map_err(|e| e.to_string())?;
        if spacing.spacing_metres.len() != mesh.n_cells() {
            return Err(format!(
                "mesh {} carries {} cell centre(s) but {} spacing value(s); a window derived \
                 from mismatched arrays would pair one cell's position with another cell's \
                 resolution",
                mesh_path.display(),
                mesh.n_cells(),
                spacing.spacing_metres.len()
            ));
        }
        let derived = Window::mesh_focus(
            &mesh.latitude_degrees,
            &mesh.longitude_degrees,
            &spacing.spacing_metres,
        )
        .map_err(|e| e.to_string())?;
        eprintln!(
            "rw_mpas_convert: advisory: --window mesh derived from {} using {}: {}",
            mesh_path.display(),
            spacing.source.label(),
            derived.description()
        );
        derived
    } else {
        Window::named(&args.window).map_err(|e| e.to_string())?
    };

    let mut weights =
        build_weights(&mesh, &args.window, &window).map_err(|e| e.to_string())?;
    // A mesh that does not cover the sphere has a FOOTPRINT, and a window is
    // a rectangle: the points outside the footprint have to be marked, or the
    // nearest-cell resample fills them with the furthest cell it can reach.
    //
    // MEASURED (2026-08-26, r4.75.11020): the first weather field ever
    // rendered from a limited-area forecast came out with radial streaks of
    // the outermost ring's reflectivity across three corners of the frame, at
    // a maximum nearest-cell distance of 195.8 km on a 4.6 km mesh.  A reader
    // cannot tell a streak from a squall line.
    //
    // A GLOBAL mesh covers every window point by construction, so its
    // footprint marks nothing and its output bytes are unchanged -- which is
    // why this is unconditional rather than a regional switch.
    let footprint_spacing = read_mesh_cell_spacing(&mesh_path).map_err(|e| e.to_string())?;
    if footprint_spacing.spacing_metres.len() != mesh.n_cells() {
        return Err(format!(
            "mesh {} carries {} cell centre(s) but {} spacing value(s); a \
             footprint derived from mismatched arrays would mask one cell's \
             window points using another cell's resolution",
            mesh_path.display(),
            mesh.n_cells(),
            footprint_spacing.spacing_metres.len()
        ));
    }
    let off_mesh = weights
        .mark_footprint(&footprint_spacing.spacing_metres, ON_MESH_SPACING_FACTOR)
        .map_err(|e| e.to_string())?;
    println!(
        "WINDOW\t{}\t{}x{}\tcells={}",
        args.window,
        weights.ny,
        weights.nx,
        weights.n_cells
    );
    println!(
        "WEIGHTS\t{}\tbuild_s={:.3}\tmesh_read_s={:.3}\tmax_nn_km={:.3}\tmean_nn_km={:.3}",
        weights.weights_sha256,
        weights.build_seconds,
        mesh_seconds,
        weights.max_distance_km(),
        weights.mean_distance_km()
    );
    println!(
        "FOOTPRINT\toff_mesh_points={}\tof={}\ton_mesh_max_nn_km={:.3}\tfactor={:.1}",
        off_mesh,
        weights.ny * weights.nx,
        weights.max_on_mesh_distance_km(),
        ON_MESH_SPACING_FACTOR
    );

    let mut records: Vec<String> = Vec::new();
    for history in &args.history {
        let frame = read_history(
            history,
            args.init.as_deref(),
            None,
            None,
            args.init_sha256.as_deref(),
        )
        .map_err(|e| e.to_string())?;
        let (dlat, dlon) =
            assert_mesh_agrees(&frame, &mesh, 1.0e-4).map_err(|e| e.to_string())?;
        let stamp = frame.valid_time.label().replace(':', "_");
        let destination = Path::new(&out_dir).join(format!("{}_{stamp}", args.prefix));
        let options = ConvertOptions {
            simulation_start,
            field_set: args.field_set.clone(),
            format: args.format,
            clobber: args.clobber,
            ..ConvertOptions::default()
        };
        let emitted =
            convert_frame(&frame, &weights, &destination, &options).map_err(|e| e.to_string())?;
        println!(
            "CONVERTED\t{}\t{:.3}s\t{} bytes\t{} written\t{} absent",
            emitted.path.display(),
            emitted.seconds,
            emitted.bytes,
            emitted.written.len(),
            emitted.absent.len()
        );
        records.push(format!(
            "{{\"output\": \"{}\", \"output_sha256\": \"{}\", \"output_bytes\": {}, \"valid_time\": \"{}\", \"simulation_start\": \"{}\", \"lead_seconds\": {}, \"convert_seconds\": {}, \"history_sha256\": \"{}\", \"init_sha256\": \"{}\", \"max_latitude_error_degrees\": {:e}, \"max_longitude_error_degrees\": {:e}, \"written_variables\": [{}], \"absent_wrf_fields\": [{}]}}",
            json_escape(&emitted.path.display().to_string()),
            frame.history_sha256,
            emitted.bytes,
            emitted.valid_time.iso(),
            emitted.simulation_start.iso(),
            emitted.lead_seconds,
            emitted.seconds,
            frame.history_sha256,
            frame.init_sha256.clone().unwrap_or_default(),
            dlat,
            dlon,
            emitted
                .written
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(", "),
            emitted
                .absent
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }

    if let Some(path) = &args.json {
        let body = format!(
            "{{\n  \"schema\": \"{}\",\n  \"engine\": \"rw-mpas {} (rust)\",\n  \"window\": \"{}\",\n  \"window_spec\": {},\n  \"mesh_sha256\": \"{}\",\n  \"weights_sha256\": \"{}\",\n  \"resample_method\": \"{}\",\n  \"field_set\": \"{}\",\n  \"weights_build_seconds\": {},\n  \"frames\": [\n    {}\n  ]\n}}\n",
            rw_mpas::convert::BRIDGE_SCHEMA,
            env!("CARGO_PKG_VERSION"),
            args.window,
            window.spec_json(),
            weights.mesh_sha256,
            weights.weights_sha256,
            rw_mpas::weights::RESAMPLE_METHOD,
            args.field_set,
            weights.build_seconds,
            records.join(",\n    ")
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, body).map_err(|e| e.to_string())?;
        println!("REPORT\t{}", path.display());
    }
    println!(
        "FINISHED\tframes={}\telapsed_s={:.3}",
        args.history.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if argv.iter().any(|a| a == "--abi") {
        println!("{ABI}");
        return ExitCode::SUCCESS;
    }
    if argv.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("rw_mpas_convert: {message}");
            ExitCode::FAILURE
        }
    }
}
