//! # sharppyrs
//!
//! A Rust/egui port of the SPC-style Skew-T sounding plot from
//! [SHARPpy-Reimagined-vRust](https://github.com/FahrenheitResearch/SHARPpy-Reimagined-vRust)
//! (itself built on [SHARPpy](https://github.com/sharppy/SHARPpy)).
//!
//! ```no_run
//! # fn demo(ui: &mut egui::Ui, data: sharppyrs::SoundingData) {
//! let profile = sharppyrs::Profile::new(data).unwrap();
//! ui.add(
//!     sharppyrs::SkewT::new(&profile)
//!         .title("Forecast profile")
//!         .style(sharppyrs::SkewTStyle::space_grotesk()),
//! );
//! # }
//! ```
//!
//! Call [`install_fonts`] once at startup to register the bundled
//! Space Grotesk font (the face the original renderer uses); pass
//! [`SkewTStyle::space_grotesk`] to the widget to select it.

pub mod barbs;
pub mod derived;
pub mod diagnostic_table;
pub mod extras;
pub mod panels;
pub mod profile;
pub mod skewt;
pub mod utils;
pub mod window;

pub use derived::DerivedParams;
pub use diagnostic_table::{
    DiagnosticTableBoard, DiagnosticTablePanel, DiagnosticTablePanelKind, DiagnosticTableRow,
    DiagnosticTableSection, NativeDiagnosticPatch, NativeDiagnosticPatchBoard,
    NativeDiagnosticSlotPatch, native_diagnostic_slot_ids,
};
pub use profile::{LocationFootprint, ParcelType, Profile, SoundingData};
pub use sharprs;
pub use sharprs::params::cape::ParcelResult as Parcel;
pub use skewt::{SkewT, SkewTStyle, SoundingFontPreset};
pub use window::{
    CornerPanel, PanelKind, SoundingLayout, SoundingView, store_layout, stored_layout,
};

/// Name of the bundled regular font family registered by [`install_fonts`].
pub const FONT_FAMILY: &str = "SpaceGrotesk";
/// Name of the bundled bold font family registered by [`install_fonts`].
pub const FONT_FAMILY_BOLD: &str = "SpaceGrotesk-Bold";

/// Add the bundled Space Grotesk fonts (OFL licensed) to an existing
/// [`egui::FontDefinitions`] — use this when your app builds its own font set
/// (call it before your own `ctx.set_fonts`).
pub fn add_fonts(fonts: &mut egui::FontDefinitions) {
    fonts.font_data.insert(
        "SpaceGrotesk-Regular".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/SpaceGrotesk-Regular.ttf"))
            .into(),
    );
    fonts.font_data.insert(
        "SpaceGrotesk-Bold".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/SpaceGrotesk-Bold.ttf")).into(),
    );
    fonts.families.insert(
        egui::FontFamily::Name(FONT_FAMILY.into()),
        vec!["SpaceGrotesk-Regular".to_owned()],
    );
    fonts.families.insert(
        egui::FontFamily::Name(FONT_FAMILY_BOLD.into()),
        vec!["SpaceGrotesk-Bold".to_owned()],
    );
}

/// Register the bundled Space Grotesk fonts with egui, replacing the font set
/// with the egui defaults plus the [`FONT_FAMILY`] / [`FONT_FAMILY_BOLD`]
/// families. If your app installs its own fonts, use [`add_fonts`] on your
/// own `FontDefinitions` instead so they are not clobbered.
pub fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    add_fonts(&mut fonts);
    ctx.set_fonts(fonts);
}
