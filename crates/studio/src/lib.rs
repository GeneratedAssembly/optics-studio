//! `optics-studio` — the Phase-10 web UI over the headless optics APIs.
//!
//! Framework: dioxus `=0.7.10`, web target, CSR-only
//! ([`docs/adr-10.0-ui-framework.md`](../../docs/adr-10.0-ui-framework.md)).
//! The dioxus dependency tree is target-gated to `wasm32` (`Cargo.toml`), so
//! this crate's **view model** and **run core** compile and unit-test
//! natively while only the UI modules need the browser. The crate is an
//! opt-in workspace member excluded from `default-members`, so the default
//! gates never build dioxus; check it explicitly with
//! `cargo check -p optics-studio --target wasm32-unknown-unknown`.
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
//! - [`run`] — host-testable inverse-run core (task 10.3): serializable run
//!   jobs, the chunked solver session (gap G-2 workaround,
//!   `docs/ux/spec.md` §3.4), and the start/stop controller. No dioxus, no
//!   wasm here; unit-tested on the host with the real `optics-inverse`
//!   solvers.
//! - [`bridge`] — wasm only: the dedicated-web-worker transport (ADR-10.0
//!   Q3) with a synchronous inline fallback, via `gloo-worker`.
//! - [`shell`] — the task-10.1 application shell: prescription view/editor
//!   (`.zmx` import + text/table modes + parameter sliders), the source /
//!   engine picker models, the G-8 five-way dispatch adapter (the shell
//!   picker includes Rayleigh-Sommerfeld, outside the core `Engine` enum),
//!   the Beamline schematic strip model, and the demo carrier; its dioxus
//!   components live in [`shell::ui`] (wasm only) and are mounted by
//!   [`ui::App`].
//! - [`ui`] — wasm only: dioxus components (run console + convergence plot).
//!
//! Constraints (docs/ux/spec.md §6): **no physics in this crate** — only
//! calls into the headless crates' public APIs, unit conversion, and display
//! normalization. The run console owns its problem definition because `solve`
//! needs a complete carrier/renderer/merit description, which the worker
//! reconstructs from the serializable [`run::job::RunJob`] (solver state
//! cannot cross the worker boundary as closures).

pub mod run;
pub mod shell;
pub mod viewmodel;

#[cfg(target_arch = "wasm32")]
pub mod app;
#[cfg(target_arch = "wasm32")]
pub mod bridge;
#[cfg(target_arch = "wasm32")]
pub mod ui;
