//! One calm weather-workspace palette, shared by local and node screens.
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, BorderType, Borders},
    Frame,
};

pub const BACK: Color = Color::Rgb(8, 25, 29);
pub const SURFACE: Color = Color::Rgb(15, 40, 43);
pub const RAISED: Color = Color::Rgb(23, 56, 59);
pub const BORDER: Color = Color::Rgb(57, 102, 101);
pub const INK: Color = Color::Rgb(231, 246, 239);
pub const MUTED: Color = Color::Rgb(151, 186, 179);
pub const SKY: Color = Color::Rgb(136, 224, 239);
pub const TEAL: Color = Color::Rgb(111, 220, 189);
pub const LEAF: Color = Color::Rgb(184, 221, 145);
pub const AMBER: Color = Color::Rgb(248, 207, 126);
pub const ERROR: Color = Color::Rgb(255, 159, 145);
pub const ERROR_SURFACE: Color = Color::Rgb(64, 34, 35);

pub fn heading() -> Style {
    Style::default().fg(SKY).add_modifier(Modifier::BOLD)
}

pub fn selected() -> Style {
    Style::default()
        .fg(BACK)
        .bg(SKY)
        .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
}

pub fn monochrome(buffer: &mut Buffer) {
    for cell in &mut buffer.content {
        cell.fg = Color::Reset;
        cell.bg = Color::Reset;
        cell.set_style(Style::default().underline_color(Color::Reset));
    }
}

pub fn finish(frame: &mut Frame) {
    if std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()) {
        // Normalize before the backend emits colors: its NO_COLOR reset would
        // otherwise cancel the bold/underline modifiers on the same cell.
        monochrome(frame.buffer_mut());
    }
}

pub fn button(primary: bool) -> Style {
    if primary {
        Style::default().fg(BACK).bg(LEAF).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(SKY).bg(RAISED).add_modifier(Modifier::BOLD)
    }
}

pub fn notice(error: bool) -> Style {
    Style::default()
        .fg(if error { ERROR } else { AMBER })
        .bg(if error { ERROR_SURFACE } else { SURFACE })
}

pub fn panel(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .style(Style::default().fg(INK).bg(SURFACE))
        .border_style(Style::default().fg(BORDER))
        .title(Span::styled(format!(" {} ", title.trim()), heading()))
}

pub fn backdrop(frame: &mut Frame, area: Rect) {
    // A restrained sky-to-evergreen wash uses real terminal cells. Cards
    // retain solid backgrounds so selections and text remain unambiguous.
    for row in 0..area.height {
        let sky = area.height.saturating_sub(row).min(20) as u8;
        let background = Color::Rgb(8 + sky / 5, 25 + sky / 3, 29 + sky / 2);
        frame.render_widget(
            Block::default().style(Style::default().fg(INK).bg(background)),
            Rect::new(area.x, area.y + row, area.width, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monochrome_keeps_non_color_selection_cues() {
        let mut buffer = Buffer::empty(Rect::new(0, 0, 2, 1));
        buffer[(0, 0)].set_symbol(">").set_style(selected());
        monochrome(&mut buffer);
        let cell = &buffer[(0, 0)];
        assert_eq!(cell.fg, Color::Reset);
        assert_eq!(cell.bg, Color::Reset);
        assert!(cell.modifier.contains(Modifier::BOLD | Modifier::UNDERLINED));
        assert_eq!(cell.symbol(), ">");
    }

    fn luminance(color: Color) -> f64 {
        let Color::Rgb(r, g, b) = color else { panic!("theme must use explicit RGB"); };
        let linear = |value: u8| {
            let s = f64::from(value) / 255.0;
            if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
        };
        0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
    }

    #[test]
    fn body_muted_selection_and_attention_text_keep_readable_contrast() {
        for (foreground, background) in [
            (INK, SURFACE), (MUTED, SURFACE), (SKY, RAISED),
            (BACK, SKY), (BACK, LEAF), (AMBER, SURFACE), (ERROR, ERROR_SURFACE),
        ] {
            let one = luminance(foreground);
            let two = luminance(background);
            let ratio = (one.max(two) + 0.05) / (one.min(two) + 0.05);
            assert!(ratio >= 4.5, "{foreground:?} on {background:?}: {ratio}");
        }
    }
}
