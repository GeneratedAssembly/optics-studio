//! Minimal system composition: an ordered beamline of elements and
//! spacings (Phase 5 ergonomics decision — sugar over caller-side
//! composition, not a new abstraction layer).
//!
//! The locked architecture is unchanged: [`Element::apply`] is local to a
//! plane and propagators are free functions over [`BandField`]s; they never
//! call each other. A [`Beamline`] is the loop you would otherwise write by
//! hand — element, distance, element, distance… — with the engine left as a
//! caller-supplied closure:
//!
//! ```
//! use optics_rs::beamline::Beamline;
//! use optics_rs::element::{ClearAperture, Element, EllipticalAperture, ThinLens};
//! use optics_rs::field::BandField;
//! use optics_rs::propagation::asm::{propagate, AsmConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let band = BandField::uniform(1.0, 0.0, 64, 64, 1e-3, 1e-3, 500e-9)?;
//! let stop = EllipticalAperture::circle(4e-3, (0.0, 0.0))?;
//! let lens = ThinLens::fixed(0.2, None, ClearAperture::Field, (0.0, 0.0))?;
//! let cfg = AsmConfig::default();
//! let mut bl = Beamline::new();
//! bl.push(&stop, 0.0) // no spacing: element back-to-back
//!     .push(&lens, 0.2); // propagate 0.2 m after the lens
//! let focused = bl.trace(&band, |b, z| propagate(b, z, &cfg))?;
//! # Ok(())
//! # }
//! ```
//!
//! **Why a closure, not an engine enum:** each engine carries its own
//! config (`AsmConfig`, `FresnelConfig`, …) and its own error type — the
//! engine-error unification is an open API decision, and a closure keeps
//! this module stable as engines are added. Regime choices stay with the
//! caller (see [`crate::propagation::regime`]): the beamline never picks
//! an engine, guards, or resamples on its own.
//!
//! Spacing `z == 0.0` skips the propagation call entirely (no engine
//! round-trip, no `z = 0` policy differences between engines).

use thiserror::Error;

use crate::element::{Element, OpticsError};
use crate::field::BandField;

/// An ordered sequence of `(element, z_after)` steps over borrowed
/// elements. The beamline is a transient composition: elements stay
/// caller-owned (`&'a dyn Element`), and building or pushing never copies
/// them.
#[derive(Clone, Default)]
pub struct Beamline<'a> {
    steps: Vec<(&'a dyn Element, f64)>,
}

impl<'a> Beamline<'a> {
    /// An empty beamline (tracing it is the identity).
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `element` followed by a free-space distance `z_after`
    /// (meters; `0.0` means no spacing). Returns `self` for chaining.
    pub fn push(&mut self, element: &'a dyn Element, z_after: f64) -> &mut Self {
        self.steps.push((element, z_after));
        self
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// `true` when the beamline has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Applies each element in order, propagating `z_after` through the
    /// caller-supplied engine between planes. `propagate` is only called
    /// for nonzero spacings, in step order.
    ///
    /// # Errors
    /// [`BeamlineError::Element`] for the first failing `apply`
    /// ([`OpticsError`]), [`BeamlineError::Engine`] for the first failing
    /// propagation (the engine's own error type, untouched).
    pub fn trace<E>(
        &self,
        band: &BandField,
        mut propagate: impl FnMut(&BandField, f64) -> Result<BandField, E>,
    ) -> Result<BandField, BeamlineError<E>>
    where
        E: std::error::Error,
    {
        let mut current = band.clone();
        for (element, z) in &self.steps {
            current = element.apply(&current)?;
            if *z != 0.0 {
                current = propagate(&current, *z).map_err(BeamlineError::Engine)?;
            }
        }
        Ok(current)
    }
}

/// Beamline trace failures: element application or engine propagation.
#[derive(Debug, Error)]
pub enum BeamlineError<E: std::error::Error> {
    /// An [`Element::apply`] failed mid-trace.
    #[error(transparent)]
    Element(#[from] OpticsError),
    /// The caller-supplied propagation closure failed; the engine's own
    /// error is preserved untouched.
    #[error(transparent)]
    Engine(E),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{ClearAperture, EllipticalAperture, RectangularAperture, ThinLens};
    use crate::propagation::asm::{AsmConfig, propagate as asm_propagate};
    use std::cell::Cell;

    const LAMBDA: f64 = 500e-9;

    fn band() -> BandField {
        BandField::uniform(1.0, 0.0, 64, 64, 1e-3, 1e-3, LAMBDA).unwrap()
    }

    #[derive(Debug, Error)]
    #[error("boom")]
    struct Boom;

    #[test]
    fn trace_matches_manual_chain_bit_for_bit() {
        let stop = EllipticalAperture::circle(4e-3, (0.0, 0.0)).unwrap();
        let lens = ThinLens::fixed(0.2, None, ClearAperture::Field, (0.0, 0.0)).unwrap();
        let rect = RectangularAperture::new((3e-3, 2e-3), None, (0.0, 0.0)).unwrap();
        let cfg = AsmConfig::default();

        let mut bl = Beamline::new();
        bl.push(&stop, 0.1).push(&lens, 0.2).push(&rect, 0.0);
        let traced = bl.trace(&band(), |b, z| asm_propagate(b, z, &cfg)).unwrap();

        // Hand-rolled chain in the same order.
        let mut manual = band();
        manual = stop.apply(&manual).unwrap();
        manual = asm_propagate(&manual, 0.1, &cfg).unwrap();
        manual = lens.apply(&manual).unwrap();
        manual = asm_propagate(&manual, 0.2, &cfg).unwrap();
        manual = rect.apply(&manual).unwrap();

        for (a, b) in traced
            .field()
            .samples()
            .iter()
            .zip(manual.field().samples())
        {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn zero_spacing_skips_the_engine() {
        let stop = EllipticalAperture::circle(4e-3, (0.0, 0.0)).unwrap();
        let calls = Cell::new(0usize);
        let mut bl = Beamline::new();
        bl.push(&stop, 0.0).push(&stop, 0.05);
        let out = bl
            .trace(&band(), |b, _z| {
                calls.set(calls.get() + 1);
                asm_propagate(b, 0.05, &AsmConfig::default())
            })
            .unwrap();
        assert_eq!(calls.get(), 1, "only the nonzero spacing calls propagate");
        assert_eq!(bl.len(), 2);
        assert!(!bl.is_empty());
        assert_eq!(out.field().dims(), band().field().dims());
    }

    #[test]
    fn element_errors_surface_typed() {
        // Oversized stopped mask: OutsideField through the Element arm.
        let big = EllipticalAperture::circle(40e-3, (0.0, 0.0)).unwrap();
        let mut bl = Beamline::new();
        bl.push(&big, 0.0);
        let err = bl
            .trace(&band(), |b, z| asm_propagate(b, z, &AsmConfig::default()))
            .unwrap_err();
        assert!(matches!(
            err,
            BeamlineError::Element(OpticsError::OutsideField { .. })
        ));
    }

    #[test]
    fn engine_errors_surface_untouched() {
        let stop = EllipticalAperture::circle(4e-3, (0.0, 0.0)).unwrap();
        let mut bl = Beamline::new();
        bl.push(&stop, 0.1);
        let err = bl
            .trace(&band(), |_b, _z| Err::<BandField, Boom>(Boom))
            .unwrap_err();
        assert!(matches!(err, BeamlineError::Engine(Boom)));
    }

    #[test]
    fn empty_beamline_is_the_identity() {
        let bl = Beamline::new();
        assert!(bl.is_empty());
        let out = bl
            .trace(&band(), |_b, _z| Err::<BandField, Boom>(Boom))
            .unwrap();
        assert_eq!(out, band());
    }
}
