use image::{GenericImage, Rgba, RgbaImage};

use crate::{Color, MapRenderRequest, RustwxRenderError, render_image};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PanelPadding {
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    pub left: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelGridLayout {
    pub rows: u32,
    pub columns: u32,
    pub panel_width: u32,
    pub panel_height: u32,
    pub gap_x: u32,
    pub gap_y: u32,
    pub padding: PanelPadding,
    pub background: Color,
}

impl PanelGridLayout {
    pub fn new(
        rows: u32,
        columns: u32,
        panel_width: u32,
        panel_height: u32,
    ) -> Result<Self, RustwxRenderError> {
        if rows == 0 || columns == 0 || panel_width == 0 || panel_height == 0 {
            return Err(RustwxRenderError::InvalidPanelLayout {
                rows,
                columns,
                panel_width,
                panel_height,
            });
        }

        Ok(Self {
            rows,
            columns,
            panel_width,
            panel_height,
            gap_x: 0,
            gap_y: 0,
            padding: PanelPadding::default(),
            background: Color::WHITE,
        })
    }

    pub fn two_by_two(panel_width: u32, panel_height: u32) -> Result<Self, RustwxRenderError> {
        Self::new(2, 2, panel_width, panel_height)
    }

    pub fn two_by_four(panel_width: u32, panel_height: u32) -> Result<Self, RustwxRenderError> {
        Self::new(2, 4, panel_width, panel_height)
    }

    pub fn with_gaps(mut self, gap_x: u32, gap_y: u32) -> Self {
        self.gap_x = gap_x;
        self.gap_y = gap_y;
        self
    }

    pub fn with_padding(mut self, padding: PanelPadding) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_background(mut self, background: Color) -> Self {
        self.background = background;
        self
    }

    pub fn capacity(self) -> usize {
        (self.rows as usize) * (self.columns as usize)
    }

    pub fn canvas_size(self) -> Result<(u32, u32), RustwxRenderError> {
        let width = axis_span(
            self.padding.left,
            self.columns,
            self.panel_width,
            self.gap_x,
            self.padding.right,
        )?;
        let height = axis_span(
            self.padding.top,
            self.rows,
            self.panel_height,
            self.gap_y,
            self.padding.bottom,
        )?;

        Ok((width, height))
    }

    pub fn panel_origin(self, index: usize) -> Result<(u32, u32), RustwxRenderError> {
        let capacity = self.capacity();
        if index >= capacity {
            return Err(RustwxRenderError::TooManyPanels {
                actual: index + 1,
                capacity,
            });
        }

        let row = index as u32 / self.columns;
        let column = index as u32 % self.columns;
        let x_stride = self
            .panel_width
            .checked_add(self.gap_x)
            .ok_or(RustwxRenderError::PanelLayoutOverflow)?;
        let y_stride = self
            .panel_height
            .checked_add(self.gap_y)
            .ok_or(RustwxRenderError::PanelLayoutOverflow)?;
        let x = self
            .padding
            .left
            .checked_add(
                column
                    .checked_mul(x_stride)
                    .ok_or(RustwxRenderError::PanelLayoutOverflow)?,
            )
            .ok_or(RustwxRenderError::PanelLayoutOverflow)?;
        let y = self
            .padding
            .top
            .checked_add(
                row.checked_mul(y_stride)
                    .ok_or(RustwxRenderError::PanelLayoutOverflow)?,
            )
            .ok_or(RustwxRenderError::PanelLayoutOverflow)?;
        Ok((x, y))
    }
}

pub fn compose_panel_images(
    layout: &PanelGridLayout,
    panels: &[RgbaImage],
) -> Result<RgbaImage, RustwxRenderError> {
    if panels.len() > layout.capacity() {
        return Err(RustwxRenderError::TooManyPanels {
            actual: panels.len(),
            capacity: layout.capacity(),
        });
    }

    let (canvas_width, canvas_height) = layout.canvas_size()?;
    let mut canvas = RgbaImage::from_pixel(
        canvas_width,
        canvas_height,
        Rgba([
            layout.background.r,
            layout.background.g,
            layout.background.b,
            layout.background.a,
        ]),
    );

    for (index, panel) in panels.iter().enumerate() {
        validate_panel_size(layout, index, panel.width(), panel.height())?;
        let (x, y) = layout.panel_origin(index)?;
        canvas
            .copy_from(panel, x, y)
            .map_err(|source| RustwxRenderError::ComposePanel { index, source })?;
    }

    Ok(canvas)
}

pub fn render_panel_grid(
    layout: &PanelGridLayout,
    requests: &[MapRenderRequest],
) -> Result<RgbaImage, RustwxRenderError> {
    if requests.len() > layout.capacity() {
        return Err(RustwxRenderError::TooManyPanels {
            actual: requests.len(),
            capacity: layout.capacity(),
        });
    }

    let mut panels = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        validate_panel_size(layout, index, request.width, request.height)?;
        panels.push(render_image(request)?);
    }

    compose_panel_images(layout, &panels)
}

fn validate_panel_size(
    layout: &PanelGridLayout,
    index: usize,
    actual_width: u32,
    actual_height: u32,
) -> Result<(), RustwxRenderError> {
    if actual_width != layout.panel_width || actual_height != layout.panel_height {
        return Err(RustwxRenderError::PanelSizeMismatch {
            index,
            expected_width: layout.panel_width,
            expected_height: layout.panel_height,
            actual_width,
            actual_height,
        });
    }
    Ok(())
}

fn axis_span(
    start_padding: u32,
    count: u32,
    item_size: u32,
    gap: u32,
    end_padding: u32,
) -> Result<u32, RustwxRenderError> {
    let item_total = count
        .checked_mul(item_size)
        .ok_or(RustwxRenderError::PanelLayoutOverflow)?;
    let gap_total = count
        .checked_sub(1)
        .ok_or(RustwxRenderError::PanelLayoutOverflow)?
        .checked_mul(gap)
        .ok_or(RustwxRenderError::PanelLayoutOverflow)?;

    start_padding
        .checked_add(item_total)
        .and_then(|value| value.checked_add(gap_total))
        .and_then(|value| value.checked_add(end_padding))
        .ok_or(RustwxRenderError::PanelLayoutOverflow)
}

#[cfg(test)]
mod tests;
