use std::fmt::{Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use crate::core::{GridSpacing, OpticalElement};
use crate::elements::masks::apply_phase;
use crate::utils::grid::tensor_from_fn_2d;
use crate::utils::scalar::{FloatDtype, float};
use crate::utils::zernike;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZernikeTerm<E> {
    pub n: usize,
    pub m: isize,
    pub coefficient: E,
}

impl<E> ZernikeTerm<E> {
    pub const fn new(n: usize, m: isize, coefficient: E) -> Self {
        Self { n, m, coefficient }
    }
}

impl<E: Display> Display for ZernikeTerm<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Zernike(n={}, m={}, coefficient={})",
            self.n, self.m, self.coefficient
        )
    }
}

#[derive(Debug, Clone)]
pub struct ZernikeAberration<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub aperture_radius: E,
    pub terms: Vec<ZernikeTerm<E>>,
    pub phase: Tensor<S, E, D, NoneTape>,
}

impl<S, E, D> ZernikeAberration<S, E, D>
where
    S: Shape<Concrete = [usize; 2]>,
    E: FloatDtype,
    D: TensorFromVec<E>,
{
    pub fn new(
        device: &D,
        shape: S,
        spacing: GridSpacing<E>,
        aperture_radius: E,
        terms: Vec<ZernikeTerm<E>>,
    ) -> Self {
        let phase = tensor_from_fn_2d(device, shape, spacing, |x, y| {
            let rho = (x * x + y * y).sqrt() / aperture_radius;
            if rho > E::one() {
                return E::zero();
            }

            let rho_f64 = rho
                .to_f64()
                .expect("normalized pupil radius must fit in f64");
            let theta_f64 = y
                .atan2(x)
                .to_f64()
                .expect("azimuthal angle must fit in f64");

            terms.iter().fold(E::zero(), |acc, term| {
                acc + term.coefficient * float::<E>(zernike(term.n, term.m, rho_f64, theta_f64))
            })
        });

        Self {
            aperture_radius,
            terms,
            phase,
        }
    }
}

impl<S, E, D> Display for ZernikeAberration<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "ZernikeAberration(radius={}, terms={}, shape={:?})",
            self.aperture_radius,
            self.terms.len(),
            self.phase.shape()
        )
    }
}

impl<F, S, E, D> OpticalElement<F> for ZernikeAberration<S, E, D>
where
    F: crate::core::Field<Shape = S, Elem = E, Device = D>,
    S: Shape,
    E: FloatDtype,
    D: Device<E>,
{
    fn apply(self, field: F) -> F {
        apply_phase(field, &self.phase)
    }
}
