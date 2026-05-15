use std::fmt::{Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use crate::core::OpticalElement;
use crate::elements::{Aperture, ThinLens};

#[derive(Debug, Clone)]
pub struct FourFRelay<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub first_lens: ThinLens<S, E, D>,
    pub pupil: Option<Aperture<S, E, D>>,
    pub second_lens: ThinLens<S, E, D>,
}

impl<S, E, D> FourFRelay<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub fn new(
        first_lens: ThinLens<S, E, D>,
        pupil: Option<Aperture<S, E, D>>,
        second_lens: ThinLens<S, E, D>,
    ) -> Self {
        Self {
            first_lens,
            pupil,
            second_lens,
        }
    }
}

impl<S, E, D> Display for FourFRelay<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "FourFRelay(first={}, second={})",
            self.first_lens, self.second_lens
        )
    }
}

impl<F, S, E, D> OpticalElement<F> for FourFRelay<S, E, D>
where
    F: crate::core::Field<Shape = S, Elem = E, Device = D>,
    S: Shape,
    E: Dtype + Display,
    D: Device<E>,
    Aperture<S, E, D>: OpticalElement<F>,
{
    fn apply(self, field: F) -> F {
        let field = self.first_lens.apply(field);
        let field = match self.pupil {
            Some(pupil) => pupil.apply(field),
            None => field,
        };
        self.second_lens.apply(field)
    }
}
