use std::fmt::{Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use crate::core::{CoherenceState, ComplexField, GridSpacing, Source, Spectrum};
use crate::utils::scalar::FloatDtype;

#[derive(Debug, Clone)]
pub struct ContinuousWave<S, E>
where
    S: Shape,
    E: Dtype,
{
    pub shape: S,
    pub spacing: GridSpacing<E>,
    pub spectrum: Spectrum<E>,
    pub coherence: CoherenceState,
    pub amplitude: E,
    pub phase: E,
}

impl<S, E> ContinuousWave<S, E>
where
    S: Shape,
    E: Dtype,
{
    pub fn new(
        shape: S,
        spacing: GridSpacing<E>,
        spectrum: Spectrum<E>,
        coherence: CoherenceState,
        amplitude: E,
        phase: E,
    ) -> Self {
        Self {
            shape,
            spacing,
            spectrum,
            coherence,
            amplitude,
            phase,
        }
    }
}

impl<S, E> Display for ContinuousWave<S, E>
where
    S: Shape,
    E: Dtype + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "ContinuousWave(shape={:?}, spacing={}, spectrum={}, coherence={})",
            self.shape, self.spacing, self.spectrum, self.coherence
        )
    }
}

impl<S, E, D> Source<E, D> for ContinuousWave<S, E>
where
    S: Shape,
    E: FloatDtype,
    D: TensorFromVec<E>,
{
    type Shape = S;

    fn emit(self, device: &D) -> ComplexField<Self::Shape, E, D, NoneTape> {
        let count = self.shape.num_elements();
        let real_value = self.amplitude * self.phase.cos();
        let imag_value = self.amplitude * self.phase.sin();
        let real = device.tensor((vec![real_value; count], self.shape));
        let imag = device.tensor((vec![imag_value; count], self.shape));
        ComplexField::new(real, imag, self.spacing, self.spectrum, self.coherence)
    }
}
