//! Dioxus components for the PSF/metrics viewport (task 10.2).
//!
//! Wasm-only module (`cfg(target_arch = "wasm32")` in `lib.rs`): the whole
//! dioxus dependency tree stays off the native test path. Components bind
//! the pure [`crate::viewmodel`] to the DOM:
//!
//! - [`Studio`] — the root: multi-band tabs, intensity canvas + controls,
//!   cross-sections, metric tab column, sampling banner.
//! - The canvas paints via `web-sys` 2D `ImageData` (ADR-10.0 Q4 resolved:
//!   2D `ImageData` — the ≤2048² display budget never needs WebGL, and
//!   browser GPU compute is out of scope, spec §6).
//! - Line charts render as SVG polylines (spec §4), built by
//!   [`crate::viewmodel::charts`].

mod banner;
mod canvas;
mod panels;
mod viewport;

use std::rc::Rc;

use dioxus::prelude::*;

use crate::viewmodel::demo::StudioModel;
use crate::viewmodel::state::ViewportState;

/// The viewport root component.
///
/// Frames come from [`StudioModel::demo`] until task 10.1 (the shell)
/// supplies imported PSFs — the component tree is identical either way.
#[component]
pub fn Studio() -> Element {
    let model = use_context_provider(|| {
        Rc::new(StudioModel::demo().unwrap_or_else(|e| panic!("demo model failed to build: {e}")))
    });
    let view = use_signal(ViewportState::default);

    rsx! {
        div { class: "studio", style: STUDIO_CSS,
            header {
                h1 { "optics-studio — PSF / metrics viewport" }
                span { class: "phase", "task 10.2 · dioxus 0.7.10 · demo data" }
            }
            banner::BannerView { banner: model.banner.clone() }
            div { class: "columns", style: "display:flex; gap:16px; flex-wrap:wrap;",
                viewport::IntensityColumn { view }
                panels::MetricColumn { view }
            }
            footer {
                style: "font-size:11px; color:#666; margin-top:12px;",
                "Analysis APIs only (optics-core::analysis) — no physics in the UI crate."
            }
        }
    }
}

/// Minimal inline stylesheet (the shell task 10.1 will own global styling).
const STUDIO_CSS: &str = r#"
    font-family: system-ui, sans-serif; color: #e8e8ee; background: #14141b;
"#;
