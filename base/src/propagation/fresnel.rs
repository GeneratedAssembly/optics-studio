use std::fmt::{Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use crate::core::{GridSpacing, OpticalElement};
use crate::elements::masks::apply_phase;
use crate::utils::grid::tensor_from_fn_2d;
use crate::utils::scalar::{FloatDtype, float};

#[derive(Debug, Clone)]
pub struct FreeSpace<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub distance: E,
    pub wavelength: E,
    pub phase: Tensor<S, E, D, NoneTape>,
}

impl<S, E, D> FreeSpace<S, E, D>
where
    S: Shape<Concrete = [usize; 2]>,
    E: FloatDtype,
    D: TensorFromVec<E>,
{
    pub fn paraxial_phase(
        device: &D,
        shape: S,
        spacing: GridSpacing<E>,
        distance: E,
        wavelength: E,
    ) -> Self {
        let phase_delay = float::<E>(std::f64::consts::TAU) * distance / wavelength;
        let phase = tensor_from_fn_2d(device, shape, spacing, |_x, _y| phase_delay);
        Self {
            distance,
            wavelength,
            phase,
        }
    }
}

impl<S, E, D> Display for FreeSpace<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "FreeSpace(distance={}, wavelength={}, shape={:?})",
            self.distance,
            self.wavelength,
            self.phase.shape()
        )
    }
}

impl<F, S, E, D> OpticalElement<F> for FreeSpace<S, E, D>
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
