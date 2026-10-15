//! Task-10.1 application shell: prescription view/editor + parameter
//! sliders → live preview (docs/ux/spec.md §3.1–§3.2).
//!
//! The pure, DOM-free half lives here natively (unit-tested with
//! `cargo test -p optics-studio`); the dioxus components live in
//! [`ui`] (wasm-only). The shell feeds its live traces into the task-10.2
//! viewport through `viewmodel::metrics::make_frame` +
//! `viewmodel::demo::StudioModel`, and reports diagnostics through
//! `viewmodel::banner` — no rendering or metric logic is duplicated.
//!
//! Boundary rules (spec §6): API plumbing, unit conversion, and display
//! normalization only — every number comes from a headless crate.
//!
//! Known overlap (flagged for 10.4): [`dispatch`] is the shell's five-way
//! engine picker (spec §3.1 includes Rayleigh–Sommerfeld, which the core
//! `Engine` enum does not model, so `viewmodel::engine`'s four-way
//! dispatch cannot cover it). Unifying both dispatchers into `viewmodel`
//! is a deliberate follow-up.

pub mod carrier;
pub mod dispatch;
pub mod prescription;
pub mod preview;
pub mod source;

#[cfg(target_arch = "wasm32")]
pub mod ui;
