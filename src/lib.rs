//! # optics-rs — Diffractive Optics Engineering Suite
//!
//! Wave-optics propagation engine for designing and analyzing diffractive
//! optical elements: scalar complex fields sampled on 2D grids, advanced
//! through free space with selectable engines — exact **Angular Spectrum**
//! (`propagation::asm`), paraxial **Fresnel** in transfer-function and
//! convolution forms (`propagation::fresnel`), and the **Fraunhofer**
//! far-field limit (`propagation::fraunhofer`) — with sampling guards,
//! regime diagnostics (`propagation::regime`), and locally acting
//! [`element::Element`]s (apertures, and in later phases lenses, phase
//! masks, DOEs, gratings).
//!
//! Ray tracing / geometric optics is deliberately out of scope (see
//! `agents.md`). The physics models, formulas, and validity regimes are
//! detailed in `docs/physics.md`; module structure and API specs in
//! `docs/architecture.md`.
//!
//! ## Conventions
//!
//! - **Units:** SI throughout — meters, radians, cycles per meter; every
//!   public item documents its unit expectations.
//! - **Precision:** `f64` everywhere (approved decision 2026-10-11).
//! - **Field layers:** [`field::Field`] is a pure spatial grid;
//!   [`field::BandField`] binds it to one wavelength and is the unit of
//!   propagation and element interaction. Incoherent multiband sources are
//!   later modeled as collections of `BandField`s (intensities sum).
//! - **Data vs algorithms:** field/element types are data + local
//!   transformations; propagators are free functions in [`propagation`]
//!   depending on field types, never the reverse.
//! - **Numerics:** FFT sign/normalization/bin-order conventions are pinned
//!   in `propagation::fft` docs and unit tests before dependent code.
//! - **Sampling safety:** every propagator exposes and enforces
//!   spatial-sampling criteria (`propagation::asm::sampling_report`,
//!   `propagation::fresnel::sampling_report`,
//!   `propagation::fraunhofer::regime_report`); undersampled transfers are
//!   mitigated automatically where possible before an
//!   `AliasingRisk`/`RegimeViolation` is raised, and `propagation::regime`
//!   recommends an engine per geometry.
//!
//! ## Quick start
//!
//! ```
//! use optics_rs::element::{CircularAperture, Element};
//! use optics_rs::field::BandField;
//! use optics_rs::propagation::asm::{propagate, AsmConfig};
//!
//! // Uniform 500 nm band on a 64 x 64 grid, 20 um pitch (1.28 mm window).
//! let band = BandField::uniform(1.0, 0.0, 64, 64, 20e-6, 20e-6, 500e-9).unwrap();
//! // Hard-edge circular aperture, radius 0.3 mm, centered: a local mask.
//! let aperture = CircularAperture::new(0.3e-3, (0.0, 0.0)).unwrap();
//! let masked = aperture.apply(&band).unwrap();
//! // Diffractive spreading appears only after propagation: the edge
//! // structure advances 5 cm through the angular spectrum transfer function.
//! let far = propagate(&masked, 0.05, &AsmConfig::default()).unwrap();
//! assert_eq!(far.field().dims(), (64, 64)); // same grid, same wavelength
//! ```

pub mod common;
pub mod element;
pub mod field;
pub mod propagation;
pub mod wave;
