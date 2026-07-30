use std::path::Path;
use std::time::Instant;

use rustwx_core::{CanonicalBundleDescriptor, CycleSpec, ModelId, ModelRunRequest, SourceId};
use rustwx_io::{FetchRequest, fetch_bytes_with_cache};
use rustwx_models::{ResolvedCanonicalBundleProduct, resolve_canonical_bundle_product};

use super::{FetchedModelFile, pressure_optional_decode_enabled};

pub(super) fn thermo_bundles(
    model: ModelId,
    surface_product_override: Option<&str>,
    pressure_product_override: Option<&str>,
) -> (
    ResolvedCanonicalBundleProduct,
    ResolvedCanonicalBundleProduct,
) {
    (
        resolve_canonical_bundle_product(
            model,
            CanonicalBundleDescriptor::SurfaceAnalysis,
            surface_product_override,
        ),
        resolve_canonical_bundle_product(
            model,
            CanonicalBundleDescriptor::PressureAnalysis,
            pressure_product_override,
        ),
    )
}

pub(crate) fn fetch_family_file(
    model: ModelId,
    cycle: CycleSpec,
    forecast_hour: u16,
    source: SourceId,
    bundle: &ResolvedCanonicalBundleProduct,
    cache_root: &Path,
    use_cache: bool,
) -> Result<FetchedModelFile, Box<dyn std::error::Error>> {
    fetch_family_file_with_patterns(
        model,
        cycle,
        forecast_hour,
        source,
        bundle,
        bundle_fetch_variable_patterns(model, bundle.bundle, &bundle.native_product),
        cache_root,
        use_cache,
    )
}

pub(super) fn fetch_surface_pressure_files_parallel(
    model: ModelId,
    cycle: CycleSpec,
    forecast_hour: u16,
    source: SourceId,
    surface_bundle: &ResolvedCanonicalBundleProduct,
    pressure_bundle: &ResolvedCanonicalBundleProduct,
    cache_root: &Path,
    use_cache: bool,
) -> (
    (Result<FetchedModelFile, std::io::Error>, u128),
    (Result<FetchedModelFile, std::io::Error>, u128),
) {
    let surface_cycle = cycle.clone();
    let pressure_cycle = cycle;
    rayon::join(
        || {
            let start = Instant::now();
            let result = fetch_family_file(
                model,
                surface_cycle,
                forecast_hour,
                source,
                surface_bundle,
                cache_root,
                use_cache,
            )
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()));
            (result, start.elapsed().as_millis())
        },
        || {
            let start = Instant::now();
            let result = fetch_family_file(
                model,
                pressure_cycle,
                forecast_hour,
                source,
                pressure_bundle,
                cache_root,
                use_cache,
            )
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()));
            (result, start.elapsed().as_millis())
        },
    )
}

pub(crate) fn fetch_family_file_with_patterns(
    model: ModelId,
    cycle: CycleSpec,
    forecast_hour: u16,
    source: SourceId,
    bundle: &ResolvedCanonicalBundleProduct,
    variable_patterns: Vec<String>,
    cache_root: &Path,
    use_cache: bool,
) -> Result<FetchedModelFile, Box<dyn std::error::Error>> {
    let request = FetchRequest {
        request: ModelRunRequest::new(model, cycle, forecast_hour, &bundle.native_product)?,
        source_override: Some(source),
        variable_patterns,
    };
    let fetched = fetch_bytes_with_cache(&request, cache_root, use_cache)?;
    Ok(FetchedModelFile {
        request,
        bytes: fetched.result.bytes.clone(),
        fetched,
    })
}

pub(crate) fn bundle_fetch_variable_patterns(
    model: ModelId,
    bundle: CanonicalBundleDescriptor,
    native_product: &str,
) -> Vec<String> {
    match (bundle, native_product) {
        (CanonicalBundleDescriptor::SurfaceAnalysis, _) if model_has_grib_surface_bundle(model) => {
            surface_analysis_fetch_patterns(model)
        }
        (CanonicalBundleDescriptor::PressureAnalysis, _)
            if model_has_grib_pressure_bundle(model) =>
        {
            pressure_analysis_fetch_patterns(model)
        }
        (CanonicalBundleDescriptor::NativeAnalysis, "nat-na") => vec![
            "CAPE:surface",
            "CIN:surface",
            "LFTX:500-1000 mb",
            "CAPE:90-0 mb above ground",
            "CIN:90-0 mb above ground",
            "CAPE:255-0 mb above ground",
            "CIN:255-0 mb above ground",
            "HGT:cloud base",
            "PRES:cloud base",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        (CanonicalBundleDescriptor::NativeAnalysis, "sfc") if matches!(model, ModelId::Hrrr) => {
            vec![
                "PRES:surface",
                "APCP:surface",
                "TMP:2 m above ground",
                "DPT:2 m above ground",
                "RH:2 m above ground",
                "UGRD:10 m above ground",
                "VGRD:10 m above ground",
                "GUST:surface",
                "GUST:10 m above ground",
                "MSLMA:mean sea level",
                "PRMSL:mean sea level",
                "MSLET:mean sea level",
                "LCDC:low cloud layer",
                "MCDC:middle cloud layer",
                "HCDC:high cloud layer",
                "MXUPHL:5000-2000 m above ground",
                "WIND:10 m above ground",
                "VIS:surface",
            ]
            .into_iter()
            .map(str::to_string)
            .collect()
        }
        _ => Vec::new(),
    }
}

fn model_has_grib_surface_bundle(model: ModelId) -> bool {
    matches!(
        model,
        ModelId::Hrrr
            | ModelId::HrrrAk
            | ModelId::Gfs
            | ModelId::Gdas
            | ModelId::Gefs
            | ModelId::Aigfs
            | ModelId::Aigefs
            | ModelId::Rap
            | ModelId::Nam
            | ModelId::Hiresw
            | ModelId::Sref
            | ModelId::Rtma
            | ModelId::Urma
            | ModelId::Nbm
            | ModelId::RrfsA
    )
}

fn model_has_grib_pressure_bundle(model: ModelId) -> bool {
    matches!(
        model,
        ModelId::Hrrr
            | ModelId::HrrrAk
            | ModelId::Gfs
            | ModelId::Gdas
            | ModelId::Gefs
            | ModelId::Aigfs
            | ModelId::Aigefs
            | ModelId::Rap
            | ModelId::Nam
            | ModelId::Hiresw
            | ModelId::Sref
            | ModelId::RrfsA
    )
}

pub(super) fn surface_analysis_fetch_patterns(model: ModelId) -> Vec<String> {
    let mut patterns = vec![
        "PRES:surface",
        "HGT:surface",
        "GP:surface",
        "TMP:2 m above ground",
        "SPFH:2 m above ground",
        "UGRD:10 m above ground",
        "VGRD:10 m above ground",
    ];
    if matches!(
        model,
        ModelId::Gfs
            | ModelId::Gdas
            | ModelId::Gefs
            | ModelId::Aigfs
            | ModelId::Aigefs
            | ModelId::Rap
            | ModelId::Nam
            | ModelId::Hiresw
            | ModelId::Sref
            | ModelId::Rtma
            | ModelId::Urma
            | ModelId::Nbm
            | ModelId::RrfsA
    ) {
        patterns.extend(["DPT:2 m above ground", "RH:2 m above ground"]);
    }
    patterns.into_iter().map(str::to_string).collect()
}

pub(super) fn pressure_analysis_fetch_patterns(model: ModelId) -> Vec<String> {
    let patterns = match model {
        ModelId::Hrrr | ModelId::HrrrAk => {
            hrrr_pressure_analysis_fetch_patterns(pressure_optional_decode_enabled())
        }
        // RAP pressure products stay on the full family fetch in production;
        // old subset attempts could omit pressure-level wind records required
        // by the generic pressure decoder.
        ModelId::Rap => Vec::new(),
        ModelId::Gfs
        | ModelId::Gdas
        | ModelId::Gefs
        | ModelId::Aigfs
        | ModelId::Aigefs
        | ModelId::Nam
        | ModelId::Hiresw
        | ModelId::Sref => vec!["HGT", "TMP", "RH", "UGRD", "VGRD"],
        ModelId::RrfsA => vec!["HGT", "GP", "TMP", "SPFH", "DPT", "RH", "UGRD", "VGRD"],
        _ => Vec::new(),
    };
    patterns.into_iter().map(str::to_string).collect()
}

pub(super) fn hrrr_pressure_analysis_fetch_patterns(include_optional: bool) -> Vec<&'static str> {
    let mut patterns = vec!["HGT", "TMP", "SPFH", "UGRD", "VGRD"];
    if include_optional {
        patterns.extend([
            "VVEL", "ABSV", "CLWMR", "CIMIXR", "ICMR", "RWMR", "SNMR", "GRLE",
        ]);
    }
    patterns
}

pub(super) fn merge_variable_patterns(
    pattern_groups: impl IntoIterator<Item = Vec<String>>,
) -> Vec<String> {
    let mut merged = Vec::new();
    for group in pattern_groups {
        if group.is_empty() {
            return Vec::new();
        }
        for pattern in group {
            if !merged.contains(&pattern) {
                merged.push(pattern);
            }
        }
    }
    merged
}
