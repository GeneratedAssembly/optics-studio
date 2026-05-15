use std::fmt::{Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use crate::core::{CoherenceState, ComplexField, GridSpacing, Source, Spectrum};
use crate::utils::grid::centered_grid_points;
use crate::utils::scalar::FloatDtype;

#[derive(Debug, Clone, Copy)]
pub struct PumpSource<S, E>
where
    S: Shape,
    E: Dtype,
{
    pub shape: S,
    pub spacing: GridSpacing<E>,
    pub wavelength: E,
    pub power: E,
    pub waist: E,
}

impl<S, E> PumpSource<S, E>
where
    S: Shape,
    E: Dtype,
{
    pub fn new(shape: S, spacing: GridSpacing<E>, wavelength: E, power: E, waist: E) -> Self {
        Self {
            shape,
            spacing,
            wavelength,
            power,
            waist,
        }
    }
}

impl<S, E> Display for PumpSource<S, E>
where
    S: Shape,
    E: Dtype + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "PumpSource(shape={:?}, wavelength={}, power={}, waist={})",
            self.shape, self.wavelength, self.power, self.waist
        )
    }
}

impl<S, E, D> Source<E, D> for PumpSource<S, E>
where
    S: Shape<Concrete = [usize; 2]>,
    E: FloatDtype,
    D: TensorFromVec<E>,
{
    type Shape = S;

    fn emit(self, device: &D) -> ComplexField<Self::Shape, E, D, NoneTape> {
        let waist2 = self.waist * self.waist;
        let amplitude = self.power.sqrt();
        let real_values = centered_grid_points(self.shape, self.spacing)
            .into_iter()
            .map(|(x, y)| amplitude * (-(x * x + y * y) / waist2).exp())
            .collect();
        let imag_values = vec![E::zero(); self.shape.num_elements()];
        let real = device.tensor((real_values, self.shape));
        let imag = device.tensor((imag_values, self.shape));
        ComplexField::new(
            real,
            imag,
            self.spacing,
            Spectrum::Monochromatic(self.wavelength),
            CoherenceState::Coherent,
        )
    }
}
