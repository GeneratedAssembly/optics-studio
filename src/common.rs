use std::f64::consts::PI;
use std::ops::{Add, Deref, Sub};

use num_complex::Complex64;

pub const J: Complex64 = Complex64::I;

#[derive(Clone, Copy, Debug)]
pub struct Angle {
    radians: f64,
}

impl Angle {
    pub fn new(radians: f64) -> Self {
        Self { radians }
    }

    pub fn zero() -> Self {
        Self { radians: 0.0 }
    }

    pub fn pi_half() -> Self {
        Self { radians: PI / 2.0 }
    }

    pub fn pi() -> Self {
        Self { radians: PI }
    }

    pub fn deg_90() -> Self {
        Angle::pi_half()
    }

    pub fn deg_180() -> Self {
        Angle::pi()
    }

    pub fn from_pi(radians: f64) -> Self {
        Angle::pi() + Angle::new(radians)
    }

    pub fn from_pi_half(radians: f64) -> Self {
        Angle::pi_half() + Angle::new(radians)
    }

    pub fn from_90(degrees: f64) -> Self {
        Angle::pi_half() + Angle::from_degrees(degrees)
    }

    pub fn from_180(degrees: f64) -> Self {
        Angle::pi() + Angle::from_degrees(degrees)
    }

    pub fn from_degrees(degrees: f64) -> Self {
        let radians = PI / 180.0 * degrees;
        Self { radians }
    }

    pub fn degrees(&self) -> f64 {
        180.0 / PI * self.radians
    }
}

impl Deref for Angle {
    type Target = f64;

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
