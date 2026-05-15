use std::fmt::{Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use crate::core::{GridSpacing, OpticalElement};
use crate::elements::masks::{apply_phase, apply_thickness_phase};
use crate::utils::grid::tensor_from_fn_2d;
use crate::utils::scalar::{FloatDtype, float};

#[derive(Debug, Clone)]
pub struct ThinLens<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub focal_length: E,
    pub wavelength: E,
    pub phase: Tensor<S, E, D, NoneTape>,
}

impl<S, E, D> ThinLens<S, E, D>
where
    S: Shape<Concrete = [usize; 2]>,
    E: FloatDtype,
    D: TensorFromVec<E>,
{
    pub fn new(
        device: &D,
        shape: S,
        spacing: GridSpacing<E>,
        focal_length: E,
        wavelength: E,
    ) -> Self {
        let pi = float::<E>(std::f64::consts::PI);
        let scale = -pi / (wavelength * focal_length);
        let phase = tensor_from_fn_2d(device, shape, spacing, |x, y| scale * (x * x + y * y));
        Self {
            focal_length,
            wavelength,
            phase,
        }
    }
}

impl<S, E, D> Display for ThinLens<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "ThinLens(focal_length={}, wavelength={}, shape={:?})",
            self.focal_length,
            self.wavelength,
            self.phase.shape()
        )
    }
}

impl<F, S, E, D> OpticalElement<F> for ThinLens<S, E, D>
where
    F: crate::core::Field<Shape = S, Elem = E, Device = D>,
    S: Shape,
    E: Dtype + Display,
    D: Device<E>,
{
    fn apply(self, field: F) -> F {
        apply_phase(field, &self.phase)
    }
}

#[derive(Debug, Clone)]
pub struct ThickLens<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub thickness: Tensor<S, E, D, NoneTape>,
    pub refractive_index: E,
    pub ambient_index: E,
    pub wavelength: E,
}

impl<S, E, D> ThickLens<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub fn new(
        thickness: Tensor<S, E, D, NoneTape>,
        refractive_index: E,
        ambient_index: E,
        wavelength: E,
    ) -> Self {
        Self {
            thickness,
            refractive_index,
            ambient_index,
            wavelength,
        }
    }
}

impl<S, E, D> Display for ThickLens<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "ThickLens(index={}, ambient={}, wavelength={}, shape={:?})",
            self.refractive_index,
            self.ambient_index,
            self.wavelength,
            self.thickness.shape()
        )
    }
}

impl<F, S, E, D> OpticalElement<F> for ThickLens<S, E, D>
where
    F: crate::core::Field<Shape = S, Elem = E, Device = D>,
    S: Shape,
    E: FloatDtype,
    D: Device<E>,
{
    fn apply(self, field: F) -> F {
        let phase_per_meter = float::<E>(std::f64::consts::TAU)
            * (self.refractive_index - self.ambient_index)
            / self.wavelength;
        apply_thickness_phase(field, &self.thickness, phase_per_meter)
    }
}
