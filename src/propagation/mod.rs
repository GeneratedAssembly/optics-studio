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
//! - [`asm`]: Angular Spectrum Method propagator (Phase 1).
//!
//! Internal:
//!
//! - `fft`: private NumPy-convention FFT (unnormalized forward, $1/N$
//!   inverse) and frequency-axis helpers shared by the
//!   propagators; not part of the public API.

pub mod asm;
mod fft;
