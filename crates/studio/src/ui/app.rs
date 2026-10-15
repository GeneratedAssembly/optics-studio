//! Application root. GEN-57 layout: Zemax-style three-region workspace —
//! System Explorer navigation column (left), tabbed document workspace
//! (center: Lens Data Editor, Prescription Text, Layout, System Setup,
//! PSF Analysis, Inverse Run), and a system summary / properties column
//! (right). The task-10.2 viewport ([`crate::app::Studio`]) and the
//! task-10.3 run console ([`RunConsole`]) mount inside workspace tabs.

use dioxus::prelude::*;

use crate::shell::ui::panels::{CarrierPanel, PrescriptionTable, SetupPanels, TextEditor};
use crate::shell::ui::studio::Shell;
use crate::shell::ui::style as shell_style;
use crate::shell::ui::{
    ModeSwitch, ShellColumn, ShellPreview, ShellSetup, ShellToasts, WorkspaceTab, WorkspaceTabs,
};

use super::run_console::RunConsole;

const CSS: &str = r#"
body { margin: 0; background: #12151c; color: #dbe2ea; font: 14px/1.4 system-ui, sans-serif; }
.app { padding: 0.5rem 0.9rem; max-width: none; }
.app-top { display: flex; align-items: center; gap: 14px; padding: 0.35rem 0 0.6rem; border-bottom: 1px solid #232a36; margin-bottom: 0.7rem; }
.app-top .brand { font-size: 15px; font-weight: 650; letter-spacing: 0.02em; }
.app-top .brand em { color: #7fd1ff; font-style: normal; }
.doc-name { color: #93a1b3; font-size: 12.5px; }
.app-top .spacer { flex: 1; }
.app > h2 { font-size: 16px; font-weight: 600; margin: 1.25rem 0 0.25rem; }
.subtitle { color: #7c8898; font-size: 12px; margin: 0 0 1rem; }
.console { display: flex; gap: 1.5rem; align-items: flex-start; flex-wrap: wrap; }
.config, .run { flex: 1 1 420px; min-width: 380px; }
.run { background: #171b24; border: 1px solid #232a36; border-radius: 8px; padding: 0.75rem 1rem; }
.console h3 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; color: #8b98a9; margin: 1rem 0 0.4rem; }
.field { display: flex; gap: 0.5rem; align-items: center; margin: 0.2rem 0; }
.field label { flex: 1 1 auto; color: #93a1b3; }
.field input:not([type="checkbox"]), .field select {
    width: 12rem; background: #0f1218; color: #dbe2ea; border: 1px solid #2a3342; border-radius: 4px; padding: 2px 6px;
}
.field.toggle input { width: auto; }
.buttons { display: flex; gap: 0.5rem; margin: 0.75rem 0; }
button { background: #2a3342; color: #dbe2ea; border: 0; border-radius: 5px; padding: 6px 14px; cursor: pointer; }
button.start { background: #1d7a4f; }
button.stop { background: #a33c3c; }
button:disabled { opacity: 0.45; cursor: default; }
.badge { background: #23304a; border-radius: 4px; padding: 1px 8px; font-size: 12px; margin-right: 6px; }
.status { font-size: 12px; color: #9fb0c3; }
.error { color: #ff8f8f; font-size: 12px; }
.note { color: #7c8898; font-size: 12px; }
.plot svg { background: #0c0f15; border-radius: 6px; border: 1px solid #232a36; }
.plot.empty { background: #0c0f15; border-radius: 6px; border: 1px dashed #2a3342; color: #66727f; padding: 2rem; text-align: center; }
.plot-meta { font-size: 11px; color: #7c8898; margin-top: 2px; }
details pre { background: #0c0f15; padding: 0.6rem; border-radius: 6px; font-size: 11px; overflow-x: auto; }
"#;

/// The optics-studio app root: explorer + tabbed workspace + summary.
#[component]
pub fn App() -> Element {
    // Task-10.4 round-trip seam: the viewport model is shared context at
    // the root — the shell replaces it with every live PSF (analyze the
    // frame you simulated), and the run console captures frames from it
    // as MSE/Strehl merit targets.
    let viewport_model = use_context_provider(|| {
        Signal::new(std::rc::Rc::new(
            crate::viewmodel::demo::StudioModel::demo()
                .unwrap_or_else(|e| panic!("demo model failed to build: {e}")),
        ))
    });
    let shell = use_context_provider(|| Shell::new(viewport_model));
    let tab = shell.tab();
    let doc = shell.doc_name();
    let workspace = match tab {
        WorkspaceTab::LensData => rsx! { PrescriptionTable {} },
        WorkspaceTab::Text => rsx! { TextEditor {} },
        WorkspaceTab::Layout => rsx! { ShellPreview {} },
        WorkspaceTab::Setup => rsx! { SetupPanels {} },
        WorkspaceTab::Analysis => rsx! { crate::app::Studio {} },
        WorkspaceTab::Run => rsx! {
            CarrierPanel {}
            RunConsole {}
        },
    };
    rsx! {
        style { "{CSS}" }
        style { {shell_style::CSS} }
        div { class: "app",
            header { class: "app-top",
                span { class: "brand", "optics-" em { "studio" } }
                span { class: "doc-name", if doc.is_empty() { "no document" } else { "{doc}" } }
                span { class: "spacer" }
                ModeSwitch {}
            }
            div { class: "studio-grid",
                aside { class: "grid-left", ShellColumn {} }
                main { class: "grid-center",
                    WorkspaceTabs {}
                    div { class: "workspace", {workspace} }
                }
                aside { class: "grid-right", ShellSetup {} }
            }
        }
        ShellToasts {}
    }
}
