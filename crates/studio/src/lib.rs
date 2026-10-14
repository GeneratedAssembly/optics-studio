//! `optics-studio` — the Phase-10 web UI over the headless optics APIs.
//!
//! Framework: dioxus `=0.7.10`, web target, CSR-only
//! ([`docs/adr-10.0-ui-framework.md`](../../docs/adr-10.0-ui-framework.md)).
//! The dioxus dependency tree is target-gated to `wasm32`
//! (`Cargo.toml`), so this crate's **view model** compiles and unit-tests
//! natively while only the `app` module needs the browser.
//!
//! Crate map (Phase-10 tasks):
//!
//! - [`viewmodel`] — pure, DOM-free view model: colormaps, intensity
//!   mapping, downsampling, cross-sections, metric frames, chart paths,
//!   sampling banner, engine dispatch, reference-PSF recipe, demo data.
//!   Task 10.2 lives here plus the `app` viewport components.
//! - `app` — dioxus components (wasm only): the PSF/metrics viewport
//!   (intensity canvas + EE/MTF/Strehl/spot/wavefront panels + sampling
//!   banner + multi-band tabs).
//!
//! Constraints (docs/ux/spec.md §6): **no physics in this crate** — only
//! calls into `optics-core` public APIs, unit conversion, and display
//! normalization; the crate is an opt-in workspace member excluded from
//! `default-members`, so repo gates never build it. The wasm check is an
//! explicit target: `cargo check -p optics-studio --target
//! wasm32-unknown-unknown`.

pub mod viewmodel;

#[cfg(target_arch = "wasm32")]
pub mod app;
