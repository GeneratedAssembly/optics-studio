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
/// Every constructor and arithmetic op **wraps implicitly** into
/// $[0, 2\pi)$ (non-finite inputs wrap to NaN and are the caller's to
/// reject); trig is insensitive, and equal-modulo angles become equal
/// values. Degree conversion is explicit at the boundary
/// ([`Angle::from_degrees`], [`Angle::degrees`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Angle {
    radians: f64,
}

/// Wrap a radian value into $[0, 2\pi)$.
fn wrap(radians: f64) -> f64 {
    radians.rem_euclid(2.0 * PI)
}

impl Angle {
    /// Angle of `radians` (rad), wrapped into $[0, 2\pi)$.
    pub fn new(radians: f64) -> Self {
        Self {
            radians: wrap(radians),
        }
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

    /// Convert `degrees` (deg) to an angle: $180° = \pi$ rad, wrapped into
    /// $[0, 2\pi)$.
    pub fn from_degrees(degrees: f64) -> Self {
        Angle::new(PI / 180.0 * degrees)
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
        Angle::new(*self + *rhs)
    }
}

impl Sub for Angle {
    type Output = Angle;

    fn sub(self, rhs: Self) -> Self::Output {
        Angle::new(*self - *rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_wrap_into_two_pi() {
        // Negative and multi-turn inputs wrap to [0, 2pi).
        assert_eq!(Angle::new(-PI), Angle::pi());
        assert_eq!(Angle::new(3.0 * PI), Angle::pi());
        assert_eq!(Angle::new(2.0 * PI), Angle::zero());
        assert_eq!(Angle::new(-2.0 * PI), Angle::zero());
        assert_eq!(Angle::new(0.25), Angle::new(0.25));
        assert_eq!(Angle::from_degrees(-90.0), Angle::from_degrees(270.0));
        assert_eq!(Angle::from_degrees(360.0), Angle::zero());
        // Non-finite inputs wrap to NaN (callers reject them).
        assert!(Angle::new(f64::NAN).is_nan());
        assert!(Angle::new(f64::INFINITY).is_nan());
    }

    #[test]
    fn arithmetic_wraps_and_round_trips() {
        let a = Angle::new(1.5 * PI);
        let b = Angle::new(0.5 * PI);
        assert_eq!(a + b, Angle::zero());
        assert_eq!(a - b, Angle::pi());
        assert_eq!(Angle::from_pi(-PI / 2.0), Angle::pi_half());
        assert_eq!(Angle::from_90(-180.0), Angle::from_degrees(-90.0));
        // Degree conversion round trip on canonical values.
        for deg in [0.0, 45.0, 90.0, 180.0, 270.0, 359.0] {
            let a = Angle::from_degrees(deg);
            assert!((a.degrees() - deg).abs() < 1e-12);
        }
    }
}
