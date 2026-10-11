//! Wave-optics propagation engines.
//!
//! This module owns the numerical machinery that advances a
//! [`crate::field::BandField`] along the optical ($z$) axis. Propagation
//! algorithms depend on the field representation, never the reverse: the
//! [`crate::field`] types stay pure data so multiple engines (angular spectrum,
//! Fresnel/Fraunhofer, Rayleigh-Sommerfeld) can operate on the same grids.
//!
//! Public API:
//!
//! - [`asm`]: Angular Spectrum Method propagator (exact, Phase 1).
//! - [`fresnel`]: Fresnel transfer-function propagator (paraxial, Phase 2).
//!
//! Internal:
//!
//! - `fft`: private NumPy-convention FFT (unnormalized forward, $1/N$
//!   inverse) and frequency-axis helpers shared by the
//!   propagators; not part of the public API.
//!
//! # Composition with elements
//!
//! Propagators model **free space only**: they advance a
//! [`crate::field::BandField`] from
//! one transverse plane to the next and never couple to optical elements.
//! An [`crate::element::Element`] is a *local* transmission-function
//! transformation sitting AT a plane. System behavior emerges from
//! caller-side composition — element, distance, element, distance…:
//!
//! ```text
//! source plane ──apply()──▶ plane z0 ──propagate(z1−z0)──▶ plane z1 ──apply()──▶ …
//! ```
//!
//! (e.g. `aperture.apply(&band)?` then `fresnel::propagate(&masked, z, cfg)?`).
//! A sugar-level beamline API (element list + spacings) is deferred to the
//! Phase 5 ergonomics decision point. Diffraction structure (light bending
//! around aperture edges into the geometric shadow) is produced by the
//! propagator advancing the edge-injected angular components — never by
//! `apply` itself; see `docs/physics.md` §6.

pub mod asm;
mod fft;
pub mod fresnel;

/// Padded grid size for one axis: `ceil(n * padding)`, at least `n`.
///
/// Shared by the transfer-function engines; `padding <= 1.0` means no
/// padding.
pub(crate) fn padded_len(n: usize, padding: f64) -> usize {
    if padding <= 1.0 {
        n
    } else {
        ((n as f64) * padding).ceil() as usize
    }
}
