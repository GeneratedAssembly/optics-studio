//! Wave-optics propagation engines.
//!
//! This module owns the numerical machinery that advances a
//! [`crate::field::BandField`] along the optical ($z$) axis. Propagation
//! algorithms depend on the field representation, never the reverse: the
//! [`crate::field`] types stay pure data so multiple engines (angular spectrum,
//! Fresnel/Fraunhofer, Rayleigh-Sommerfeld) can operate on the same grids.
//!
//! Planned contents:
//!
//! - `asm`: Angular Spectrum Method propagator (Phase 1, task 4).
//! - `fft`: private orthonormal FFT and frequency-axis helpers shared by the
//!   propagators; not part of the public API.

// `fft` is consumed by `asm` (task 4); the allow is removed when that lands.
#[allow(dead_code)]
mod fft;
