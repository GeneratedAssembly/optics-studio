use std::fmt::Display;

use dfdx::prelude::*;
use num_traits::{Float, FromPrimitive};

pub trait FloatDtype: Dtype + Float + FromPrimitive + Display {}

impl<T> FloatDtype for T where T: Dtype + Float + FromPrimitive + Display {}

pub fn float<E: FloatDtype>(value: f64) -> E {
    E::from_f64(value).expect("physical scalar must fit in dtype")
}

pub(crate) fn factorial(value: usize) -> f64 {
    (1..=value).fold(1.0, |acc, item| acc * item as f64)
}
