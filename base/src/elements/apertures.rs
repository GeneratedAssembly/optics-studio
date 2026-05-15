use std::fmt::{Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use crate::core::{GridSpacing, OpticalElement};
use crate::elements::masks::{AmplitudeMask, apply_amplitude};
use crate::utils::grid::tensor_from_fn_2d;
use crate::utils::scalar::FloatDtype;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApertureKind<E> {
    Circular { radius: E },
    Annular { inner_radius: E, outer_radius: E },
    Rectangular { width: E, height: E },
}

impl<E: Display> Display for ApertureKind<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Circular { radius } => write!(f, "circular(radius={})", radius),
            Self::Annular {
                inner_radius,
                outer_radius,
            } => write!(
                f,
                "annular(inner_radius={}, outer_radius={})",
                inner_radius, outer_radius
            ),
            Self::Rectangular { width, height } => {
                write!(f, "rectangular(width={}, height={})", width, height)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Aperture<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub kind: ApertureKind<E>,
    pub mask: AmplitudeMask<S, E, D>,
}

impl<S, E, D> Aperture<S, E, D>
where
    S: Shape<Concrete = [usize; 2]>,
    E: FloatDtype,
    D: TensorFromVec<E>,
{
    pub fn new(device: &D, shape: S, spacing: GridSpacing<E>, kind: ApertureKind<E>) -> Self {
        let amplitude = tensor_from_fn_2d(device, shape, spacing, |x, y| match kind {
            ApertureKind::Circular { radius } => {
                if x * x + y * y <= radius * radius {
                    E::one()
                } else {
                    E::zero()
                }
            }
            ApertureKind::Annular {
                inner_radius,
                outer_radius,
            } => {
                let r2 = x * x + y * y;
                if r2 >= inner_radius * inner_radius && r2 <= outer_radius * outer_radius {
                    E::one()
                } else {
                    E::zero()
                }
            }
            ApertureKind::Rectangular { width, height } => {
                if x.abs() <= width * E::from_f64(0.5).unwrap()
                    && y.abs() <= height * E::from_f64(0.5).unwrap()
                {
                    E::one()
                } else {
                    E::zero()
                }
            }
        });

        Self {
            kind,
            mask: AmplitudeMask::new(amplitude),
        }
    }
}

impl<S, E, D> Display for Aperture<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Aperture({}, shape={:?})",
            self.kind,
            self.mask.amplitude.shape()
        )
    }
}

impl<F, S, E, D> OpticalElement<F> for Aperture<S, E, D>
where
    F: crate::core::Field<Shape = S, Elem = E, Device = D>,
    S: Shape,
    E: Dtype + Display,
    D: Device<E>,
{
    fn apply(self, field: F) -> F {
        apply_amplitude(field, &self.mask.amplitude)
    }
}
