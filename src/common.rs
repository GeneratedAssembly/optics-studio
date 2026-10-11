//! Shared scalar primitives: the [`Angle`] newtype and the imaginary unit.
//!
//! Angles are SI radians internally; degree helpers are explicit at the
//! boundary. Complex amplitudes use `num_complex::Complex64` (f64 precision,
//! approved decision 2026-10-11).

use std::f64::consts::PI;
use std::ops::{Add, Deref, Sub};

use num_complex::Complex64;

/// The imaginary unit $i$, for typed complex literals (`J / norm` etc.).
pub const J: Complex64 = Complex64::I;

/// An angle stored in **radians** (SI).
///
/// The newtype prevents accidental mixing of degrees and radians at API
/// boundaries; `Deref<Target = f64>` exposes the radian value for math.
#[derive(Clone, Copy, Debug)]
pub struct Angle {
    radians: f64,
}

impl Angle {
    /// Angle of `radians` (rad). No range normalization is applied.
    pub fn new(radians: f64) -> Self {
        Self { radians }
    }

    /// Zero angle.
    pub fn zero() -> Self {
        Self { radians: 0.0 }
    }

    /// $\pi/2$ rad.
    pub fn pi_half() -> Self {
        Self { radians: PI / 2.0 }
    }

    /// $\pi$ rad.
    pub fn pi() -> Self {
        Self { radians: PI }
    }

    /// 90 degrees (alias of [`Angle::pi_half`]).
    pub fn deg_90() -> Self {
        Angle::pi_half()
    }

    /// 180 degrees (alias of [`Angle::pi`]).
    pub fn deg_180() -> Self {
        Angle::pi()
    }

    /// $\pi + $ `radians` (rad): offset from the anti-axis direction.
    pub fn from_pi(radians: f64) -> Self {
        Angle::pi() + Angle::new(radians)
    }

    /// $\pi/2 + $ `radians` (rad): offset from the vertical direction.
    pub fn from_pi_half(radians: f64) -> Self {
        Angle::pi_half() + Angle::new(radians)
    }

    /// 90 degrees + `degrees` (deg).
    pub fn from_90(degrees: f64) -> Self {
        Angle::pi_half() + Angle::from_degrees(degrees)
    }

    /// 180 degrees + `degrees` (deg).
    pub fn from_180(degrees: f64) -> Self {
        Angle::pi() + Angle::from_degrees(degrees)
    }

    /// Convert `degrees` (deg) to an angle: $180° = \pi$ rad.
    pub fn from_degrees(degrees: f64) -> Self {
        let radians = PI / 180.0 * degrees;
        Self { radians }
    }

    /// The angle in degrees.
    pub fn degrees(&self) -> f64 {
        180.0 / PI * self.radians
    }
}

impl Deref for Angle {
    type Target = f64;

    /// Derefs to the radian value.
    fn deref(&self) -> &Self::Target {
        &self.radians
    }
}

impl Add for Angle {
    type Output = Angle;

    fn add(self, rhs: Self) -> Self::Output {
        Angle {
            radians: *self + *rhs,
        }
    }
}

impl Sub for Angle {
    type Output = Angle;

    fn sub(self, rhs: Self) -> Self::Output {
        Angle {
            radians: *self - *rhs,
        }
    }
}
