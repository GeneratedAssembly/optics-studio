//! Task-10.1 shell dioxus components (wasm-only).
//!
//! Mounted by [`crate::ui::App`] as the three shell regions (spec §3):
//! [`ShellColumn`] (file panel + text editor + prescription table +
//! carrier sliders), [`ShellPreview`] (live PSF + Beamline strip +
//! sampling banner), and [`ShellSetup`] (paraxial card + wavelength /
//! source / engine pickers). The 10.2 viewport (`crate::app::Studio`) and
//! the 10.3 run console (`crate::ui::run_console`) are composed by the
//! root; the shell only adds the 10.1 pieces.

pub mod panels;
pub mod strip;
pub mod studio;
pub mod style;
pub mod viewport;

use dioxus::prelude::*;

pub use studio::{AppMode, Shell};

use panels::{
    CarrierPanel, EnginePanel, FilePanel, ParaxialCard, PrescriptionTable, SourcePanel, TextEditor,
    WavelengthPanel,
};

/// Mode switch for the shell column header (spec §3: the mode selects
/// which panels drive the live preview).
#[component]
pub fn ModeSwitch() -> Element {
    let shell = use_context::<Shell>();
    let mode = shell.mode();
    rsx! {
        div { class: "mode-switch",
            button {
                class: if mode == AppMode::Forward { "active" } else { "" },
                onclick: move |_| {
                    let mut s = shell;
                    s.mode.set(AppMode::Forward);
                    s.run_preview();
                },
                "Forward"
            }
            button {
                class: if mode == AppMode::Inverse { "active" } else { "" },
                onclick: move |_| {
                    let mut s = shell;
                    s.mode.set(AppMode::Inverse);
                    s.run_preview();
                },
                "Inverse"
            }
        }
    }
}

/// Left editor column (spec §3.1–§3.2): import, text mode, table mode,
/// carrier sliders.
#[component]
pub fn ShellColumn() -> Element {
    let shell = use_context::<Shell>();
    let mode = shell.mode();
    rsx! {
        ModeSwitch {}
        FilePanel {}
        TextEditor {}
        PrescriptionTable {}
        if mode == AppMode::Inverse {
            CarrierPanel {}
        }
    }
}

/// Center live-preview panel: PSF frame + Beamline schematic strip +
/// sampling banner (§3.1 "Beamline preview", §3.2 "Live Beamline preview").
#[component]
pub fn ShellPreview() -> Element {
    rsx! {
        viewport::Viewport {}
    }
}

/// Right setup column: paraxial card + wavelength / source / engine
/// pickers (§3.1).
#[component]
pub fn ShellSetup() -> Element {
    rsx! {
        ParaxialCard {}
        WavelengthPanel {}
        SourcePanel {}
        EnginePanel {}
    }
}

/// Dismissible typed-error toasts (spec §3.1 flow, §3.5).
#[component]
pub fn ShellToasts() -> Element {
    rsx! {
        viewport::Toasts {}
    }
}
