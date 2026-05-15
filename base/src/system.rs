use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use crate::core::{ComplexField, OpticalElement, Source};
use crate::propagation::FreeSpace;
use crate::utils::scalar::FloatDtype;

pub type SystemField<S, E, D> = ComplexField<S, E, D, NoneTape>;

#[derive(Debug, Clone, PartialEq)]
pub enum OpticalSystemError<E> {
    IndexOutOfBounds { index: usize, len: usize },
    NegativeDistance { distance: E },
    PositionBeforeSource { z: E },
    NoRepresentativeWavelength,
}

impl<E> Display for OpticalSystemError<E>
where
    E: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::IndexOutOfBounds { index, len } => write!(
                f,
                "element index {index} is out of bounds for system length {len}"
            ),
            Self::NegativeDistance { distance } => {
                write!(
                    f,
                    "element distance from previous plane must be non-negative, got {distance}"
                )
            }
            Self::PositionBeforeSource { z } => {
                write!(f, "requested z={z} is before the source plane at z=0")
            }
            Self::NoRepresentativeWavelength => {
                write!(
                    f,
                    "field spectrum does not contain a representative wavelength"
                )
            }
        }
    }
}

impl<E> std::error::Error for OpticalSystemError<E> where E: Debug + Display {}

trait SystemElementModel<S, E, D>: Debug + Display
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    fn apply(&self, field: SystemField<S, E, D>) -> SystemField<S, E, D>;
    fn clone_box(&self) -> Box<dyn SystemElementModel<S, E, D>>;
}

impl<S, E, D, T> SystemElementModel<S, E, D> for T
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
    T: OpticalElement<SystemField<S, E, D>> + 'static,
{
    fn apply(&self, field: SystemField<S, E, D>) -> SystemField<S, E, D> {
        self.clone().apply(field)
    }

    fn clone_box(&self) -> Box<dyn SystemElementModel<S, E, D>> {
        Box::new(self.clone())
    }
}

#[derive(Debug)]
pub struct PositionedElement<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    z: E,
    element: Box<dyn SystemElementModel<S, E, D>>,
}

impl<S, E, D> PositionedElement<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    fn new<T>(z: E, element: T) -> Self
    where
        T: OpticalElement<SystemField<S, E, D>> + 'static,
    {
        Self {
            z,
            element: Box::new(element),
        }
    }

    pub fn z(&self) -> E {
        self.z
    }

    pub fn element_description(&self) -> String {
        self.element.to_string()
    }

    fn apply(&self, field: SystemField<S, E, D>) -> SystemField<S, E, D> {
        self.element.apply(field)
    }
}

impl<S, E, D> Clone for PositionedElement<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    fn clone(&self) -> Self {
        Self {
            z: self.z,
            element: self.element.clone_box(),
        }
    }
}

impl<S, E, D> Display for PositionedElement<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "PositionedElement(z={}, {})", self.z, self.element)
    }
}

#[derive(Debug, Clone)]
pub struct OpticalSystem<S, E, D, Src>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
    Src: Source<E, D, Shape = S>,
{
    device: D,
    source: Src,
    elements: Vec<PositionedElement<S, E, D>>,
}

impl<S, E, D, Src> OpticalSystem<S, E, D, Src>
where
    S: Shape<Concrete = [usize; 2]>,
    E: FloatDtype,
    D: Device<E> + 'static,
    Src: Source<E, D, Shape = S> + 'static,
{
    pub fn new(device: D, source: Src) -> Self {
        Self {
            device,
            source,
            elements: Vec::new(),
        }
    }

    pub fn source(&self) -> &Src {
        &self.source
    }

    pub fn device(&self) -> &D {
        &self.device
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn elements(&self) -> &[PositionedElement<S, E, D>] {
        &self.elements
    }

    pub fn total_length(&self) -> E {
        self.elements
            .last()
            .map(PositionedElement::z)
            .unwrap_or_else(E::zero)
    }

    pub fn z_at(&self, index: usize) -> Result<E, OpticalSystemError<E>> {
        if index == 0 {
            return Ok(E::zero());
        }

        self.elements
            .get(index - 1)
            .map(PositionedElement::z)
            .ok_or(OpticalSystemError::IndexOutOfBounds {
                index,
                len: self.len(),
            })
    }

    pub fn insert<T>(
        &mut self,
        z_from_previous: E,
        element: T,
    ) -> Result<usize, OpticalSystemError<E>>
    where
        T: OpticalElement<SystemField<S, E, D>> + 'static,
    {
        if z_from_previous < E::zero() {
            return Err(OpticalSystemError::NegativeDistance {
                distance: z_from_previous,
            });
        }

        let z = self.total_length() + z_from_previous;
        self.elements.push(PositionedElement::new(z, element));
        Ok(self.len())
    }

    pub fn add_element<T>(
        &mut self,
        z_from_previous: E,
        element: T,
    ) -> Result<usize, OpticalSystemError<E>>
    where
        T: OpticalElement<SystemField<S, E, D>> + 'static,
    {
        self.insert(z_from_previous, element)
    }

    pub fn output(&self) -> Result<SystemField<S, E, D>, OpticalSystemError<E>> {
        self.propagate_to(self.len())
    }

    pub fn propagate_to(
        &self,
        element_index: usize,
    ) -> Result<SystemField<S, E, D>, OpticalSystemError<E>> {
        if element_index > self.len() {
            return Err(OpticalSystemError::IndexOutOfBounds {
                index: element_index,
                len: self.len(),
            });
        }

        let field = self.source.clone().emit(&self.device);
        self.propagate_through(field, self.elements.iter().take(element_index), None)
    }

    pub fn propagate_to_z(&self, z: E) -> Result<SystemField<S, E, D>, OpticalSystemError<E>> {
        if z < E::zero() {
            return Err(OpticalSystemError::PositionBeforeSource { z });
        }

        let field = self.source.clone().emit(&self.device);
        self.propagate_through(
            field,
            self.elements.iter().take_while(|element| element.z() <= z),
            Some(z),
        )
    }

    pub fn field_at(
        &self,
        element_index: usize,
    ) -> Result<SystemField<S, E, D>, OpticalSystemError<E>> {
        self.propagate_to(element_index)
    }

    pub fn field_at_z(&self, z: E) -> Result<SystemField<S, E, D>, OpticalSystemError<E>> {
        self.propagate_to_z(z)
    }

    fn propagate_through<'a, I>(
        &self,
        mut field: SystemField<S, E, D>,
        elements: I,
        final_z: Option<E>,
    ) -> Result<SystemField<S, E, D>, OpticalSystemError<E>>
    where
        I: IntoIterator<Item = &'a PositionedElement<S, E, D>>,
    {
        let mut current_z = E::zero();

        for element in elements {
            field = self.propagate_free_space(field, element.z() - current_z)?;
            field = element.apply(field);
            current_z = element.z();
        }

        if let Some(final_z) = final_z {
            field = self.propagate_free_space(field, final_z - current_z)?;
        }

        Ok(field)
    }

    fn propagate_free_space(
        &self,
        field: SystemField<S, E, D>,
        distance: E,
    ) -> Result<SystemField<S, E, D>, OpticalSystemError<E>> {
        if distance < E::zero() {
            return Err(OpticalSystemError::NegativeDistance { distance });
        }

        if distance == E::zero() {
            return Ok(field);
        }

        let wavelength = field
            .spectrum()
            .representative_wavelength()
            .ok_or(OpticalSystemError::NoRepresentativeWavelength)?;
        let shape = field.shape();
        let spacing = field.spacing();

        Ok(
            FreeSpace::paraxial_phase(&self.device, shape, spacing, distance, wavelength)
                .apply(field),
        )
    }
}

impl<S, E, D, Src> Display for OpticalSystem<S, E, D, Src>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
    Src: Source<E, D, Shape = S>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "OpticalSystem(source={}, elements={})",
            self.source,
            self.elements.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{CoherenceState, GridSpacing, Spectrum};
    use crate::elements::{Aperture, ApertureKind, PhaseMask};
    use crate::sources::ContinuousWave;

    fn test_source() -> ContinuousWave<(usize, usize), f64> {
        ContinuousWave::new(
            (5, 5),
            GridSpacing::new(10.0e-6, 10.0e-6),
            Spectrum::Monochromatic(560.0e-9),
            CoherenceState::Coherent,
            1.0,
            0.0,
        )
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1.0e-12);
    }

    #[test]
    fn inserted_element_positions_are_relative_to_previous_element() {
        let device: Cpu = Default::default();
        let mut system = OpticalSystem::new(device.clone(), test_source());
        let phase = device.tensor((vec![0.0; 25], (5, 5)));

        system.insert(0.10, PhaseMask::new(phase.clone())).unwrap();
        system.insert(0.20, PhaseMask::new(phase)).unwrap();

        assert_eq!(system.len(), 2);
        assert_close(system.z_at(0).unwrap(), 0.0);
        assert_close(system.z_at(1).unwrap(), 0.10);
        assert_close(system.z_at(2).unwrap(), 0.30);
    }

    #[test]
    fn index_and_z_propagation_match_at_element_plane() {
        let device: Cpu = Default::default();
        let mut system = OpticalSystem::new(device.clone(), test_source());
        let aperture = Aperture::new(
            &device,
            (5, 5),
            GridSpacing::new(10.0e-6, 10.0e-6),
            ApertureKind::Circular { radius: 15.0e-6 },
        );

        system.insert(0.10, aperture).unwrap();

        let indexed = system.propagate_to(1).unwrap();
        let positioned = system.propagate_to_z(0.10).unwrap();

        assert_eq!(indexed.real().as_vec(), positioned.real().as_vec());
        assert_eq!(indexed.imag().as_vec(), positioned.imag().as_vec());
    }

    #[test]
    fn arbitrary_z_can_be_after_last_element() {
        let device: Cpu = Default::default();
        let mut system = OpticalSystem::new(device.clone(), test_source());
        let phase = device.tensor((vec![0.0; 25], (5, 5)));

        system.insert(0.10, PhaseMask::new(phase)).unwrap();

        let field = system.propagate_to_z(0.25).unwrap();

        assert_eq!(field.shape(), (5, 5));
    }
}
