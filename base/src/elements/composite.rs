use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

use crate::core::{Field, OpticalElement};

#[derive(Debug, Clone)]
pub struct CompositeElement<Elements> {
    elements: Elements,
}

impl<Elements> CompositeElement<Elements> {
    pub const fn new(elements: Elements) -> Self {
        Self { elements }
    }

    pub fn elements(&self) -> &Elements {
        &self.elements
    }

    pub fn into_elements(self) -> Elements {
        self.elements
    }
}

impl<Elements> Display for CompositeElement<Elements>
where
    Elements: ElementChainDisplay,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "CompositeElement(len={}, elements=[",
            self.elements.len()
        )?;
        self.elements.fmt_chain(f)?;
        write!(f, "])")
    }
}

impl<F, Elements> OpticalElement<F> for CompositeElement<Elements>
where
    F: Field,
    Elements: OpticalElement<F> + ElementChainDisplay,
{
    fn apply(self, field: F) -> F {
        self.elements.apply(field)
    }
}

#[derive(Debug, Clone)]
pub struct ElementChain<Head, Tail> {
    head: Head,
    tail: Tail,
}

impl<Head, Tail> ElementChain<Head, Tail> {
    pub const fn new(head: Head, tail: Tail) -> Self {
        Self { head, tail }
    }

    pub fn head(&self) -> &Head {
        &self.head
    }

    pub fn tail(&self) -> &Tail {
        &self.tail
    }

    pub fn into_parts(self) -> (Head, Tail) {
        (self.head, self.tail)
    }
}

impl<Head, Tail> Display for ElementChain<Head, Tail>
where
    Self: ElementChainDisplay,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "[")?;
        self.fmt_chain(f)?;
        write!(f, "]")
    }
}

impl<F, Head, Tail> OpticalElement<F> for ElementChain<Head, Tail>
where
    F: Field,
    Head: OpticalElement<F>,
    Tail: OpticalElement<F> + ElementChainDisplay,
{
    fn apply(self, field: F) -> F {
        self.tail.apply(self.head.apply(field))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ElementChainEnd;

impl Display for ElementChainEnd {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "identity")
    }
}

impl<F> OpticalElement<F> for ElementChainEnd
where
    F: Field,
{
    fn apply(self, field: F) -> F {
        field
    }
}

#[doc(hidden)]
pub trait ElementChainDisplay {
    fn fmt_chain(&self, f: &mut Formatter<'_>) -> FmtResult;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ElementChainDisplay for ElementChainEnd {
    fn fmt_chain(&self, _f: &mut Formatter<'_>) -> FmtResult {
        Ok(())
    }

    fn len(&self) -> usize {
        0
    }
}

impl<Head, Tail> ElementChainDisplay for ElementChain<Head, Tail>
where
    Head: Display,
    Tail: ElementChainDisplay,
{
    fn fmt_chain(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.head)?;

        if !self.tail.is_empty() {
            write!(f, ", ")?;
            self.tail.fmt_chain(f)?;
        }

        Ok(())
    }

    fn len(&self) -> usize {
        1 + self.tail.len()
    }
}

/// Builds one composite optical element from zero or more element expressions.
///
/// The resulting value implements [`OpticalElement`] for any field type supported
/// by every element in the sequence.
#[macro_export]
macro_rules! make_element {
    () => {
        $crate::elements::CompositeElement::new($crate::elements::ElementChainEnd)
    };
    ($($element:expr),+ $(,)?) => {
        $crate::elements::CompositeElement::new($crate::__make_element_chain!($($element),+))
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __make_element_chain {
    () => {
        $crate::elements::ElementChainEnd
    };
    ($element:expr $(,)?) => {
        $crate::elements::ElementChain::new($element, $crate::elements::ElementChainEnd)
    };
    ($element:expr, $($rest:expr),+ $(,)?) => {
        $crate::elements::ElementChain::new(
            $element,
            $crate::__make_element_chain!($($rest),+),
        )
    };
}

#[cfg(test)]
mod tests {
    use dfdx::prelude::*;

    use crate::OpticalSystem;
    use crate::core::{CoherenceState, GridSpacing, OpticalElement, Source, Spectrum};
    use crate::elements::{Aperture, ApertureKind, PhaseMask};
    use crate::sources::ContinuousWave;

    #[test]
    fn macro_composes_heterogeneous_elements_in_order() {
        let device: Cpu = Default::default();
        let shape = (5, 5);
        let spacing = GridSpacing::new(10.0e-6, 10.0e-6);
        let source = ContinuousWave::new(
            shape,
            spacing,
            Spectrum::Monochromatic(560.0e-9),
            CoherenceState::Coherent,
            1.0,
            0.0,
        )
        .emit(&device);

        let phase = PhaseMask::new(device.tensor((vec![0.25; 25], shape)));
        let aperture = Aperture::new(
            &device,
            shape,
            spacing,
            ApertureKind::Circular { radius: 15.0e-6 },
        );

        let sequential = aperture.clone().apply(phase.clone().apply(source.clone()));
        let composite = crate::make_element!(phase, aperture).apply(source);

        assert_eq!(sequential.real().as_vec(), composite.real().as_vec());
        assert_eq!(sequential.imag().as_vec(), composite.imag().as_vec());
    }

    #[test]
    fn empty_macro_is_identity_element() {
        let device: Cpu = Default::default();
        let shape = (3, 3);
        let spacing = GridSpacing::new(10.0e-6, 10.0e-6);
        let source = ContinuousWave::new(
            shape,
            spacing,
            Spectrum::Monochromatic(560.0e-9),
            CoherenceState::Coherent,
            1.0,
            0.0,
        )
        .emit(&device);

        let identity = crate::make_element!();
        let output = identity.apply(source.clone());

        assert_eq!(source.real().as_vec(), output.real().as_vec());
        assert_eq!(source.imag().as_vec(), output.imag().as_vec());
    }

    #[test]
    fn composite_element_can_be_inserted_into_system() {
        let device: Cpu = Default::default();
        let shape = (5, 5);
        let spacing = GridSpacing::new(10.0e-6, 10.0e-6);
        let source = ContinuousWave::new(
            shape,
            spacing,
            Spectrum::Monochromatic(560.0e-9),
            CoherenceState::Coherent,
            1.0,
            0.0,
        );
        let phase = PhaseMask::new(device.tensor((vec![0.25; 25], shape)));
        let aperture = Aperture::new(
            &device,
            shape,
            spacing,
            ApertureKind::Circular { radius: 15.0e-6 },
        );

        let mut system = OpticalSystem::new(device, source);
        system
            .insert(0.0, crate::make_element!(phase, aperture))
            .unwrap();

        let output = system.output().unwrap();

        assert_eq!(output.shape(), shape);
    }
}
