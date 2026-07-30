use super::types::{
    HrrrNonEcapeHourSummary, NonEcapeBuildDomainTiming, NonEcapeBuildProductTiming,
    NonEcapeMultiDomainReport,
};
use crate::derived::HrrrDerivedBatchReport;
use crate::direct::HrrrDirectBatchReport;
use crate::windowed::HrrrWindowedBatchReport;

pub(super) fn build_summary(
    direct: &Option<HrrrDirectBatchReport>,
    derived: &Option<HrrrDerivedBatchReport>,
    windowed: &Option<HrrrWindowedBatchReport>,
) -> HrrrNonEcapeHourSummary {
    let mut output_paths = Vec::new();
    let mut runner_count = 0usize;
    let mut direct_rendered_count = 0usize;
    let mut derived_rendered_count = 0usize;
    let mut windowed_rendered_count = 0usize;
    let mut windowed_blocker_count = 0usize;

    if let Some(report) = direct {
        runner_count += 1;
        direct_rendered_count = report.recipes.len();
        output_paths.extend(
            report
                .recipes
                .iter()
                .map(|recipe| recipe.output_path.clone()),
        );
    }

    if let Some(report) = derived {
        runner_count += 1;
        derived_rendered_count = report.recipes.len();
        output_paths.extend(
            report
                .recipes
                .iter()
                .map(|recipe| recipe.output_path.clone()),
        );
    }

    if let Some(report) = windowed {
        runner_count += 1;
        windowed_rendered_count = report.products.len();
        windowed_blocker_count = report.blockers.len();
        output_paths.extend(
            report
                .products
                .iter()
                .map(|product| product.output_path.clone()),
        );
    }

    HrrrNonEcapeHourSummary {
        runner_count,
        direct_rendered_count,
        derived_rendered_count,
        windowed_rendered_count,
        windowed_blocker_count,
        output_count: output_paths.len(),
        output_paths,
    }
}

pub(super) fn build_static_domain_timings(
    report: &NonEcapeMultiDomainReport,
) -> Vec<NonEcapeBuildDomainTiming> {
    report
        .domains
        .iter()
        .map(|domain| NonEcapeBuildDomainTiming {
            domain_slug: domain.domain.slug.clone(),
            total_ms: domain.total_ms,
            direct_total_ms: domain.direct.as_ref().map(|direct| direct.total_ms),
            derived_total_ms: domain.derived.as_ref().map(|derived| derived.total_ms),
            windowed_total_ms: domain.windowed.as_ref().map(|windowed| windowed.total_ms),
            output_count: domain.summary.output_count,
            direct_count: domain.summary.direct_rendered_count,
            derived_count: domain.summary.derived_rendered_count,
            windowed_count: domain.summary.windowed_rendered_count,
            windowed_blocker_count: domain.summary.windowed_blocker_count,
        })
        .collect()
}

pub(super) fn build_static_product_timings(
    report: &NonEcapeMultiDomainReport,
) -> Vec<NonEcapeBuildProductTiming> {
    let mut timings = Vec::new();
    for domain in &report.domains {
        let domain_slug = domain.domain.slug.clone();
        if let Some(direct) = &domain.direct {
            timings.extend(
                direct
                    .recipes
                    .iter()
                    .map(|recipe| NonEcapeBuildProductTiming {
                        domain_slug: domain_slug.clone(),
                        lane: "direct".to_string(),
                        product_slug: recipe.recipe_slug.clone(),
                        title: Some(recipe.title.clone()),
                        output_path: recipe.output_path.clone(),
                        source_route: Some(recipe.source_route),
                        render_ms: recipe.timing.render_ms,
                        total_ms: recipe.timing.total_ms,
                        render_to_image_ms: Some(recipe.timing.render_to_image_ms),
                        data_layer_draw_ms: Some(recipe.timing.data_layer_draw_ms),
                        overlay_draw_ms: Some(recipe.timing.overlay_draw_ms),
                        png_encode_ms: Some(recipe.timing.png_encode_ms),
                        file_write_ms: Some(recipe.timing.file_write_ms),
                    }),
            );
        }
        if let Some(derived) = &domain.derived {
            timings.extend(
                derived
                    .recipes
                    .iter()
                    .map(|recipe| NonEcapeBuildProductTiming {
                        domain_slug: domain_slug.clone(),
                        lane: "derived".to_string(),
                        product_slug: recipe.recipe_slug.clone(),
                        title: Some(recipe.title.clone()),
                        output_path: recipe.output_path.clone(),
                        source_route: Some(recipe.source_route),
                        render_ms: recipe.timing.render_ms,
                        total_ms: recipe.timing.total_ms,
                        render_to_image_ms: Some(recipe.timing.render_to_image_ms),
                        data_layer_draw_ms: Some(recipe.timing.data_layer_draw_ms),
                        overlay_draw_ms: Some(recipe.timing.overlay_draw_ms),
                        png_encode_ms: Some(recipe.timing.png_encode_ms),
                        file_write_ms: Some(recipe.timing.file_write_ms),
                    }),
            );
        }
        if let Some(windowed) = &domain.windowed {
            timings.extend(
                windowed
                    .products
                    .iter()
                    .map(|product| NonEcapeBuildProductTiming {
                        domain_slug: domain_slug.clone(),
                        lane: "windowed".to_string(),
                        product_slug: format!("{:?}", product.product),
                        title: None,
                        output_path: product.output_path.clone(),
                        source_route: None,
                        render_ms: product.timing.render_ms,
                        total_ms: product.timing.total_ms,
                        render_to_image_ms: None,
                        data_layer_draw_ms: None,
                        overlay_draw_ms: None,
                        png_encode_ms: None,
                        file_write_ms: None,
                    }),
            );
        }
    }
    timings
}
