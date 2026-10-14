//! optics-studio web entrypoint (dioxus CSR launch).
//!
//! Native builds are a no-op stub: the app is wasm-only (ADR-10.0), and the
//! pure view model is tested natively via `cargo test -p optics-studio`. The
//! wasm binary is built with `dx build --platform web` (or `dx serve` for the
//! dev loop); `cargo check -p optics-studio --target wasm32-unknown-unknown`
//! is the leak guard that keeps the framework tree inside this crate.

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        dioxus::launch(optics_studio::app::Studio);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // No native UI target: the crate's logic is exercised by unit tests.
        eprintln!("optics-studio is a wasm-only dioxus app; run `dx serve`.");
    }
}
