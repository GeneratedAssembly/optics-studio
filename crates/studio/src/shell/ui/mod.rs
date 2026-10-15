//! Task-10.1 shell dioxus components (wasm-only), reorganized for the
//! GEN-57 Zemax-style layout: a System Explorer navigation column on the
//! left, a tabbed document workspace in the center, and an always-visible
//! summary column on the right.
//!
//! Mounted by [`crate::ui::App`]: [`ShellColumn`] (file actions + explorer
//! tree), [`WorkspaceTabs`] (the tab strip; the root renders the active
//! tab's content), and [`ShellSetup`] (system summary + selected-surface
//! properties). The 10.2 viewport (`crate::app::Studio`) and the 10.3 run
//! console (`crate::ui::run_console`) are composed by the root into the
//! Analysis / Run tabs.

pub mod explorer;
pub mod layout2d;
pub mod panels;
pub mod strip;
pub mod studio;
pub mod style;
pub mod viewport;

use dioxus::prelude::*;

pub use studio::{AppMode, Shell, WorkspaceTab};

use panels::{FilePanel, SummaryPanel};

/// Mode switch for the workspace header (spec §3: the mode selects which
/// panels drive the live preview).
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

/// Left column (GEN-57): file actions on top, System Explorer tree below.
#[component]
pub fn ShellColumn() -> Element {
    rsx! {
        FilePanel {}
        explorer::SystemExplorer {}
    }
}

/// The center workspace tab strip: one tab per document/editor/viewer,
/// like the OpticStudio document tabs.
#[component]
pub fn WorkspaceTabs() -> Element {
    let shell = use_context::<Shell>();
    let active = shell.tab();
    rsx! {
        div { class: "wtab-bar",
            for (tab, label) in WorkspaceTab::ALL {
                button {
                    key: "{label}",
                    class: if tab == active { "wtab active" } else { "wtab" },
                    onclick: move |_| {
                        let mut s = shell;
                        s.set_tab(tab);
                    },
                    "{label}"
                }
            }
        }
    }
}

/// Center live-preview panel (Layout tab): PSF frame + Beamline schematic
/// strip + sampling banner (spec §3.1, §3.2).
#[component]
pub fn ShellPreview() -> Element {
    rsx! {
        viewport::Viewport {}
    }
}

/// Right column: the always-visible system summary and selected-surface
/// properties (replaces the old stacked setup panels, which now live in
/// the System Setup tab).
#[component]
pub fn ShellSetup() -> Element {
    rsx! {
        SummaryPanel {}
    }
}

/// Dismissible typed-error toasts (spec §3.1 flow, §3.5).
#[component]
pub fn ShellToasts() -> Element {
    rsx! {
        viewport::Toasts {}
    }
}
