use std::f32::consts::PI;
use std::ops::{Add, Deref, Sub};

use num_complex::Complex32;

pub const J: Complex32 = Complex32::I;

#[derive(Clone, Copy, Debug)]
pub struct Angle {
    radians: f32,
}

impl Angle {
    pub fn new(radians: f32) -> Self {
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

    pub fn from_pi(radians: f32) -> Self {
        Angle::pi() + Angle::new(radians)
    }

    pub fn from_pi_half(radians: f32) -> Self {
        Angle::pi_half() + Angle::new(radians)
    }

    pub fn from_90(degrees: f32) -> Self {
        Angle::pi_half() + Angle::from_degrees(degrees)
    }

    pub fn from_180(degrees: f32) -> Self {
        Angle::pi() + Angle::from_degrees(degrees)
    }

    pub fn from_degrees(degrees: f32) -> Self {
        let radians = PI / 180.0 * degrees;
        Self { radians }
    }

    pub fn degrees(&self) -> f32 {
        180.0 / PI * self.radians
    }
}

impl Deref for Angle {
    type Target = f32;

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
