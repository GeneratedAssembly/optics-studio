use std::fmt::{Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use crate::core::{CoherenceState, ComplexField, GridSpacing, Source, Spectrum};
use crate::utils::grid::centered_grid_points;
use crate::utils::scalar::{FloatDtype, float};

#[derive(Debug, Clone)]
pub struct GaussianBeam<S, E>
where
    S: Shape,
    E: Dtype,
{
    pub shape: S,
    pub spacing: GridSpacing<E>,
    pub spectrum: Spectrum<E>,
    pub waist: E,
    pub amplitude: E,
}

impl<S, E> GaussianBeam<S, E>
where
    S: Shape,
    E: Dtype,
{
    pub fn new(
        shape: S,
        spacing: GridSpacing<E>,
        spectrum: Spectrum<E>,
        waist: E,
        amplitude: E,
    ) -> Self {
        Self {
            shape,
            spacing,
            spectrum,
            waist,
            amplitude,
        }
    }
}

impl<S, E> Display for GaussianBeam<S, E>
where
    S: Shape,
    E: Dtype + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "GaussianBeam(shape={:?}, waist={}, spectrum={})",
            self.shape, self.waist, self.spectrum
        )
    }
}

impl<S, E, D> Source<E, D> for GaussianBeam<S, E>
where
    S: Shape<Concrete = [usize; 2]>,
    E: FloatDtype,
    D: TensorFromVec<E>,
{
    type Shape = S;

    fn emit(self, device: &D) -> ComplexField<Self::Shape, E, D, NoneTape> {
        let waist2 = self.waist * self.waist;
        let real_values = centered_grid_points(self.shape, self.spacing)
            .into_iter()
            .map(|(x, y)| self.amplitude * (-(x * x + y * y) / waist2).exp())
            .collect();
        let imag_values = vec![E::zero(); self.shape.num_elements()];
        let real = device.tensor((real_values, self.shape));
        let imag = device.tensor((imag_values, self.shape));
        ComplexField::new(
            real,
            imag,
            self.spacing,
            self.spectrum,
            CoherenceState::Coherent,
        )
    }
}

#[derive(Debug, Clone)]
pub struct HermiteGaussianMode<S, E>
where
    S: Shape,
    E: Dtype,
{
    pub shape: S,
    pub spacing: GridSpacing<E>,
    pub spectrum: Spectrum<E>,
    pub waist: E,
    pub amplitude: E,
    pub mode_x: usize,
    pub mode_y: usize,
}

impl<S, E> HermiteGaussianMode<S, E>
where
    S: Shape,
    E: Dtype,
{
    pub fn new(
        shape: S,
        spacing: GridSpacing<E>,
        spectrum: Spectrum<E>,
        waist: E,
        amplitude: E,
        mode_x: usize,
        mode_y: usize,
    ) -> Self {
        Self {
            shape,
            spacing,
            spectrum,
            waist,
            amplitude,
            mode_x,
            mode_y,
        }
    }
}

impl<S, E> Display for HermiteGaussianMode<S, E>
where
    S: Shape,
    E: Dtype + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "HermiteGaussianMode(shape={:?}, mode=({}, {}), waist={}, spectrum={})",
            self.shape, self.mode_x, self.mode_y, self.waist, self.spectrum
        )
    }
}

impl<S, E, D> Source<E, D> for HermiteGaussianMode<S, E>
where
    S: Shape<Concrete = [usize; 2]>,
    E: FloatDtype,
    D: TensorFromVec<E>,
{
    type Shape = S;

    fn emit(self, device: &D) -> ComplexField<Self::Shape, E, D, NoneTape> {
        let waist2 = self.waist * self.waist;
        let root_two = float::<E>(2.0_f64.sqrt());
        let real_values = centered_grid_points(self.shape, self.spacing)
            .into_iter()
            .map(|(x, y)| {
                let hx = hermite(self.mode_x, root_two * x / self.waist);
                let hy = hermite(self.mode_y, root_two * y / self.waist);
                self.amplitude * hx * hy * (-(x * x + y * y) / waist2).exp()
            })
            .collect();
        let imag_values = vec![E::zero(); self.shape.num_elements()];
        let real = device.tensor((real_values, self.shape));
        let imag = device.tensor((imag_values, self.shape));
        ComplexField::new(
            real,
            imag,
            self.spacing,
            self.spectrum,
            CoherenceState::Coherent,
        )
    }
}

fn hermite<E: FloatDtype>(order: usize, x: E) -> E {
    match order {
        0 => E::one(),
        1 => float::<E>(2.0) * x,
        _ => {
            let mut h_nm2 = E::one();
            let mut h_nm1 = float::<E>(2.0) * x;
            for n in 2..=order {
                let h_n = float::<E>(2.0) * x * h_nm1 - float::<E>(2.0 * (n - 1) as f64) * h_nm2;
                h_nm2 = h_nm1;
                h_nm1 = h_n;
            }
            h_nm1
        }
    }
}
