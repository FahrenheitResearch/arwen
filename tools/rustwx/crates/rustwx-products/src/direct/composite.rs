use super::types::{
    CLOUD_LEVEL_COMPONENT_SLUGS, DirectBatchRequest, OUTPUT_HEIGHT, OUTPUT_WIDTH,
    PRECIPITATION_TYPE_COMPONENT_SLUGS,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct CompositePanelSpec {
    pub(super) rows: u32,
    pub(super) columns: u32,
    pub(super) panel_width: u32,
    pub(super) panel_height: u32,
    pub(super) top_padding: u32,
    pub(super) component_slugs: &'static [&'static str],
}

impl CompositePanelSpec {
    pub(super) fn scaled_for_output(self, output_width: u32, output_height: u32) -> Self {
        let scale_x = output_width as f64 / OUTPUT_WIDTH as f64;
        let scale_y = output_height as f64 / OUTPUT_HEIGHT as f64;
        Self {
            rows: self.rows,
            columns: self.columns,
            panel_width: ((self.panel_width as f64) * scale_x).round().max(1.0) as u32,
            panel_height: ((self.panel_height as f64) * scale_y).round().max(1.0) as u32,
            top_padding: ((self.top_padding as f64) * scale_y).round().max(1.0) as u32,
            component_slugs: self.component_slugs,
        }
    }

    pub(super) fn scaled_for_request(self, request: &DirectBatchRequest) -> Self {
        self.scaled_for_output(request.output_width, request.output_height)
    }
}

pub(super) fn composite_panel_spec(slug: &str) -> Option<CompositePanelSpec> {
    match slug {
        "cloud_cover_levels" => Some(CompositePanelSpec {
            rows: 1,
            columns: 3,
            panel_width: 420,
            panel_height: 320,
            top_padding: 64,
            component_slugs: CLOUD_LEVEL_COMPONENT_SLUGS,
        }),
        "precipitation_type" => Some(CompositePanelSpec {
            rows: 2,
            columns: 2,
            panel_width: 600,
            panel_height: 415,
            top_padding: 70,
            component_slugs: PRECIPITATION_TYPE_COMPONENT_SLUGS,
        }),
        _ => None,
    }
}
