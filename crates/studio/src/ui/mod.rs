//! Dioxus UI for the run console (task 10.3). The shell (10.1) and the
//! PSF viewport (10.2) add their own modules here.

pub mod app;
pub mod convergence;
pub mod run_console;

pub use app::App;

/// Launch the dioxus app on the main thread. Called from the module start
/// hook ([`crate::bridge`]) when the wasm instance runs in a page scope —
/// the single `#[wasm_bindgen(start)]` in the crate dispatches by context,
/// because a `start` export makes the loader skip `main()`.
pub fn launch() {
    dioxus::launch(App);
}
