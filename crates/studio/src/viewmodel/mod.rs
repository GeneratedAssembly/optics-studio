//! Pure view model for the PSF/metrics viewport (task 10.2).
//!
//! Everything here is DOM-free and framework-free: it turns core types
//! (`Psf`, `EncircledEnergy`, `Mtf`, sampling reports) into display-ready
//! data (RGBA buffers, SVG path strings, banner severities). The `app`
//! module binds these to dioxus components; the split lets the whole
//! viewport logic unit-test natively (`cargo test -p optics-studio`)
//! without a wasm build.
//!
//! Boundary rules (docs/ux/spec.md §4, §6):
//!
//! - Analysis numbers come from `optics_core::analysis` only — this crate
//!   never recomputes a metric, a sampling criterion, or a propagation.
//! - Display transforms (colormaps, autoscale/lock, log floor, box-average
//!   downsampling, unit conversion for labels) are the only arithmetic here.

pub mod banner;
pub mod charts;
pub mod colormap;
pub mod crosscut;
pub mod demo;
pub mod engine;
pub mod intensity;
pub mod metrics;
pub mod reference;
pub mod state;
