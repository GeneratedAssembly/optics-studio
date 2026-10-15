//! optics-studio binary entry (dioxus CSR launch).
//!
//! On wasm32 this launches [`optics_studio::ui::App`], the combined root:
//! the task-10.2 viewport (`app::Studio`) above the task-10.3 inverse run
//! console. The solver worker is bootstrapped by
//! `optics_studio::bridge::worker_bootstrap` (`#[wasm_bindgen(start)]`)
//! whenever the module loads inside a worker scope. Native builds are a
//! no-op stub: the app is wasm-only (ADR-10.0), and the pure view model and
//! run core are exercised natively via `cargo test -p optics-studio`. The
//! wasm binary is built with `dx build --platform web` (or `dx serve` for
//! the dev loop); `cargo check -p optics-studio --target
//! wasm32-unknown-unknown` is the leak guard.

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        // On wasm the module `start` hook (`bridge::worker_bootstrap`) owns
        // bootstrap: it launches the app in a page scope and registers the
        // solver worker in a worker scope. The loader skips `main()` when a
        // `start` export exists, so this branch is a safety net only.
        optics_studio::ui::launch();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // No native UI target: the crate's logic is exercised by unit tests.
        eprintln!("optics-studio is a wasm-only dioxus app; run `dx serve`.");
    }
}
