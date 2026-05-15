use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use super::metadata::{CoherenceState, GridSpacing, Spectrum};
use super::traits::Field;

#[derive(Debug, Clone)]
pub struct FieldComponents<S, E, D, T = NoneTape>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
    T: Tape<E, D>,
{
    pub real: Tensor<S, E, D, T>,
    pub imag: Tensor<S, E, D, T>,
    pub spacing: GridSpacing<E>,
    pub spectrum: Spectrum<E>,
    pub coherence: CoherenceState,
}

#[derive(Debug, Clone)]
pub struct ComplexField<S, E, D, T = NoneTape>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
    T: Tape<E, D>,
{
    real: Tensor<S, E, D, T>,
    imag: Tensor<S, E, D, T>,
    spacing: GridSpacing<E>,
    spectrum: Spectrum<E>,
    coherence: CoherenceState,
}

impl<S, E, D, T> ComplexField<S, E, D, T>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
    T: Tape<E, D>,
{
    pub fn new(
        real: Tensor<S, E, D, T>,
        imag: Tensor<S, E, D, T>,
        spacing: GridSpacing<E>,
        spectrum: Spectrum<E>,
        coherence: CoherenceState,
    ) -> Self {
        assert_eq!(real.shape(), imag.shape());
        Self {
            real,
            imag,
            spacing,
            spectrum,
            coherence,
        }
    }

    pub fn real(&self) -> &Tensor<S, E, D, T> {
        &self.real
    }

    pub fn imag(&self) -> &Tensor<S, E, D, T> {
        &self.imag
    }

    pub fn spacing(&self) -> GridSpacing<E> {
        self.spacing
    }

    pub fn spectrum(&self) -> &Spectrum<E> {
        &self.spectrum
    }

    pub fn coherence(&self) -> CoherenceState {
        self.coherence
    }

    pub fn shape(&self) -> S {
        *self.real.shape()
    }

    pub fn map_complex(
        self,
        real: Tensor<S, E, D, T>,
        imag: Tensor<S, E, D, T>,
    ) -> ComplexField<S, E, D, T> {
        Self::new(real, imag, self.spacing, self.spectrum, self.coherence)
    }

    pub fn into_components(self) -> FieldComponents<S, E, D, T> {
        FieldComponents {
            real: self.real,
            imag: self.imag,
            spacing: self.spacing,
            spectrum: self.spectrum,
            coherence: self.coherence,
        }
    }

    pub fn from_components(components: FieldComponents<S, E, D, T>) -> Self {
        Self::new(
            components.real,
            components.imag,
            components.spacing,
            components.spectrum,
            components.coherence,
        )
    }

    pub fn intensity(self) -> Tensor<S, E, D, T>
    where
        D: Device<E>,
    {
        self.real.square() + self.imag.square()
    }
}

impl<S, E, D> ComplexField<S, E, D, NoneTape>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub fn leaky_traced(self) -> ComplexField<S, E, D, OwnedTape<E, D>> {
        ComplexField::new(
            self.real.leaky_traced(),
            self.imag.leaky_traced(),
            self.spacing,
            self.spectrum,
            self.coherence,
        )
    }

    pub fn traced(
        self,
        real_gradients: Gradients<E, D>,
        imag_gradients: Gradients<E, D>,
    ) -> ComplexField<S, E, D, OwnedTape<E, D>> {
        ComplexField::new(
            self.real.traced(real_gradients),
            self.imag.traced(imag_gradients),
            self.spacing,
            self.spectrum,
            self.coherence,
        )
    }
}

impl<S, E, D, T> Field for ComplexField<S, E, D, T>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
    T: Tape<E, D> + Debug,
{
    type Device = D;
    type Elem = E;
    type Shape = S;
    type Tape = T;

    fn real(&self) -> &Tensor<Self::Shape, Self::Elem, Self::Device, Self::Tape> {
        &self.real
    }

    fn imag(&self) -> &Tensor<Self::Shape, Self::Elem, Self::Device, Self::Tape> {
        &self.imag
    }

    fn spacing(&self) -> GridSpacing<Self::Elem> {
        self.spacing
    }

    fn spectrum(&self) -> &Spectrum<Self::Elem> {
        &self.spectrum
    }

    fn coherence(&self) -> CoherenceState {
        self.coherence
    }

    fn into_components(self) -> FieldComponents<Self::Shape, Self::Elem, Self::Device, Self::Tape> {
        self.into_components()
    }

    fn from_components(
        components: FieldComponents<Self::Shape, Self::Elem, Self::Device, Self::Tape>,
    ) -> Self {
        Self::from_components(components)
    }
}

impl<S, E, D, T> Display for ComplexField<S, E, D, T>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
    T: Tape<E, D>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "ComplexField(shape={:?}, spacing={}, spectrum={}, coherence={})",
            self.real.shape(),
            self.spacing,
            self.spectrum,
            self.coherence
        )
    }
}
