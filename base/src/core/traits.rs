use std::fmt::{Debug, Display};

use dfdx::prelude::*;

use super::field::{ComplexField, FieldComponents};
use super::metadata::{CoherenceState, GridSpacing, Spectrum};

pub trait Field: Sized + Debug {
    type Shape: Shape;
    type Elem: Dtype;
    type Device: Storage<Self::Elem>;
    type Tape: Tape<Self::Elem, Self::Device>;

    fn real(&self) -> &Tensor<Self::Shape, Self::Elem, Self::Device, Self::Tape>;
    fn imag(&self) -> &Tensor<Self::Shape, Self::Elem, Self::Device, Self::Tape>;
    fn spacing(&self) -> GridSpacing<Self::Elem>;
    fn spectrum(&self) -> &Spectrum<Self::Elem>;
    fn coherence(&self) -> CoherenceState;
    fn into_components(self) -> FieldComponents<Self::Shape, Self::Elem, Self::Device, Self::Tape>;
    fn from_components(
        components: FieldComponents<Self::Shape, Self::Elem, Self::Device, Self::Tape>,
    ) -> Self;
}

pub trait Source<E, D>: Debug + Clone + Display
where
    E: Dtype,
    D: Storage<E>,
{
    type Shape: Shape;

    fn emit(self, device: &D) -> ComplexField<Self::Shape, E, D, NoneTape>;
}

pub trait OpticalElement<F>: Debug + Clone + Display
where
    F: Field,
{
    fn apply(self, field: F) -> F;
}
