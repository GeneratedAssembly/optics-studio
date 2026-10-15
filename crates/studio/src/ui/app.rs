//! Application root. Composes the task-10.1 shell (prescription /
//! slider column + live preview, [`crate::shell::ui`]) with the
//! task-10.2 PSF/metrics viewport ([`crate::app::Studio`]) and the
//! task-10.3 inverse run console ([`RunConsole`]) in the spec §3
//! three-region layout: editor column left, viewport center, setup
//! column right, run console bottom.

use dioxus::prelude::*;

use crate::shell::ui::studio::Shell;
use crate::shell::ui::style as shell_style;
use crate::shell::ui::{ShellColumn, ShellPreview, ShellSetup, ShellToasts};

use super::run_console::RunConsole;

const CSS: &str = r#"
body { margin: 0; background: #12151c; color: #dbe2ea; font: 14px/1.4 system-ui, sans-serif; }
.app { padding: 1rem 1.5rem; max-width: 1400px; margin: 0 auto; }
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

/// The optics-studio app root: shell + viewport + run console.
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
    let _shell = use_context_provider(|| Shell::new(viewport_model));
    rsx! {
        style { "{CSS}" }
        style { {shell_style::CSS} }
        div { class: "app",
            div { class: "studio-grid",
                aside { class: "grid-left", ShellColumn {} }
                main { class: "grid-center",
                    ShellPreview {}
                    crate::app::Studio {}
                }
                aside { class: "grid-right", ShellSetup {} }
            }
            h2 { "Inverse run controls" }
            p { class: "subtitle",
                "task 10.3 — start/stop solvers, convergence plot. Solver runs \
                execute in a dedicated web worker (gloo-worker, ADR-10.0 Q3) with \
                an inline fallback. optics-inverse 0.1.0 solvers; no physics in the UI crate."
            }
            RunConsole {}
        }
        ShellToasts {}
    }
}
