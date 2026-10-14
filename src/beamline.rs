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
//! use optics_core::beamline::Beamline;
//! use optics_core::element::{ClearAperture, Element, EllipticalAperture, ThinLens};
//! use optics_core::field::BandField;
//! use optics_core::propagation::asm::{propagate, AsmConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let band = BandField::uniform(1.0, 0.0, 64, 64, 1e-3, 1e-3, 500e-9)?;
//! let stop = EllipticalAperture::circle(4e-3, (0.0, 0.0))?;
//! let lens = ThinLens::fixed(0.2, None, ClearAperture::Field, (0.0, 0.0))?;
//! let cfg = AsmConfig::default();
//! let mut bl = Beamline::new();
//! bl.push(&stop, 0.0) // no spacing: element back-to-back
//!     .push(&lens, 0.2); // propagate 0.2 m after the lens
//! let focused = bl.trace(&band, |b, z, _n| propagate(b, z, &cfg))?;
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

pub use crate::field::Psf;

/// An ordered sequence of `(element, z_after, medium)` steps over borrowed
/// elements. The beamline is a transient composition: elements stay
/// caller-owned (`&'a dyn Element`), and building or pushing never copies
/// them.
#[derive(Clone, Default)]
pub struct Beamline<'a> {
    steps: Vec<(Option<&'a dyn Element>, f64, f64)>,
}

impl<'a> Beamline<'a> {
    /// An empty beamline (tracing it is the identity).
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `element` followed by a vacuum distance `z_after`
    /// (meters; `0.0` means no spacing). Returns `self` for chaining.
    pub fn push(&mut self, element: &'a dyn Element, z_after: f64) -> &mut Self {
        self.push_medium(element, z_after, 1.0)
    }

    /// Appends `element` followed by a distance `z_after` (meters) through
    /// a medium of refractive index `medium` (finite, > 0 — validated by
    /// the engine, not here). The medium reaches the propagation closure
    /// as its third argument (e.g. `AsmConfig::medium`).
    pub fn push_medium(
        &mut self,
        element: &'a dyn Element,
        z_after: f64,
        medium: f64,
    ) -> &mut Self {
        self.steps.push((Some(element), z_after, medium));
        self
    }

    /// Appends a pure propagation step: distance `z` (meters, `0.0`
    /// skipped) through medium `medium`, with no element. Composes slabs
    /// and thick-lens internals (surface → gap → surface).
    pub fn gap(&mut self, z: f64, medium: f64) -> &mut Self {
        self.steps.push((None, z, medium));
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
    /// caller-supplied engine between planes with the step's medium.
    /// `propagate(band, z, medium)` is only called for nonzero spacings,
    /// in step order.
    ///
    /// # Errors
    /// [`BeamlineError::Element`] for the first failing `apply`
    /// ([`OpticsError`]), [`BeamlineError::Engine`] for the first failing
    /// propagation (the engine's own error type, untouched).
    pub fn trace<E>(
        &self,
        band: &BandField,
        mut propagate: impl FnMut(&BandField, f64, f64) -> Result<BandField, E>,
    ) -> Result<BandField, BeamlineError<E>>
    where
        E: std::error::Error,
    {
        let mut current = band.clone();
        for (element, z, medium) in &self.steps {
            if let Some(element) = element {
                current = element.apply(&current)?;
            }
            if *z != 0.0 {
                current = propagate(&current, *z, *medium).map_err(BeamlineError::Engine)?;
            }
        }
        Ok(current)
    }

    /// Traces the beamline and returns the detector-plane intensity grid
    /// $I = |u|^2$ — the coherent PSF when `band` is a point-source
    /// ([`crate::sources::spherical`]) or tilted-plane-wave
    /// ([`crate::field::BandField::plane_wave`]) input. Same error taxonomy
    /// as [`Beamline::trace`].
    pub fn psf<E>(
        &self,
        band: &BandField,
        propagate: impl FnMut(&BandField, f64, f64) -> Result<BandField, E>,
    ) -> Result<Psf, BeamlineError<E>>
    where
        E: std::error::Error,
    {
        let out = self.trace(band, propagate)?;
        Ok(Psf::from_band(&out))
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
    use crate::sources::{GaussianBeam, gaussian};
    use num_complex::Complex64;
    use std::cell::Cell;
    use std::f64::consts::PI;

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
        let traced = bl
            .trace(&band(), |b, z, _n| asm_propagate(b, z, &cfg))
            .unwrap();

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
            .trace(&band(), |b, _z, _n| {
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
            .trace(&band(), |b, z, _n| {
                asm_propagate(b, z, &AsmConfig::default())
            })
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
            .trace(&band(), |_b, _z, _n| Err::<BandField, Boom>(Boom))
            .unwrap_err();
        assert!(matches!(err, BeamlineError::Engine(Boom)));
    }

    #[test]
    fn empty_beamline_is_the_identity() {
        let bl = Beamline::new();
        assert!(bl.is_empty());
        let out = bl
            .trace(&band(), |_b, _z, _n| Err::<BandField, Boom>(Boom))
            .unwrap();
        assert_eq!(out, band());
    }

    #[test]
    fn psf_is_traced_intensity_with_geometry() {
        // Gaussian input (unique central peak; a uniform beam at N_F ~ 300
        // would put the intensity max in the aperture-edge fringes instead).
        let beam = GaussianBeam {
            w0: 1.5e-3,
            amplitude: 1.0,
            center: (0.0, 0.0),
        };
        let src = gaussian(&beam, 0.0, 500e-9, 64, 64, 1e-3, 1e-3).unwrap();
        let stop = EllipticalAperture::circle(4e-3, (0.0, 0.0)).unwrap();
        let mut bl = Beamline::new();
        bl.push(&stop, 0.1);
        let cfg = AsmConfig::default();
        let psf = bl.psf(&src, |b, z, _n| asm_propagate(b, z, &cfg)).unwrap();
        let traced = bl
            .trace(&src, |b, z, _n| asm_propagate(b, z, &cfg))
            .unwrap();
        // I = |u|^2 sample-for-sample, geometry preserved.
        assert_eq!(psf.dims(), band().field().dims());
        assert_eq!(psf.pitch(), band().field().pitch());
        for (i, s) in traced.field().samples().iter().enumerate() {
            assert!((psf.samples()[i] - s.norm_sqr()).abs() <= s.norm_sqr() * 1e-15 + 1e-30);
        }
        // Power bookkeeping: Psf total_power == traced field power.
        let rel = (psf.total_power() - traced.field().power()).abs() / traced.field().power();
        assert!(rel < 1e-12);
        // Gaussian through a generous stop: peak at the origin bin-pair
        // (even grid: nearest sample is half a pitch off-axis), centroid
        // at origin by symmetry.
        let (px, py) = psf.peak_position();
        assert!(px.abs() <= 0.6e-3 && py.abs() <= 0.6e-3, "peak {px},{py}");
        let (cx, cy) = psf.centroid();
        assert!(cx.abs() < 1e-6 && cy.abs() < 1e-6, "centroid {cx},{cy}");
        let (x00, y00) = psf.coord(0, 0);
        assert!((x00 + 31.5e-3).abs() < 1e-15 && (y00 + 31.5e-3).abs() < 1e-15);
        // Zero field: centroid safely at origin.
        let empty = Psf::new(vec![0.0; 4], 2, 2, 1.0, 1.0).unwrap();
        assert_eq!(empty.centroid(), (0.0, 0.0));
    }

    #[test]
    fn medium_reaches_the_engine_closure_per_step() {
        // Slab composition anchor (Phase 6): a gap step carries its medium
        // to the closure, push() stays vacuum (1.0), and push_medium
        // carries its index. Uniform band = on-axis plane wave, unpadded:
        // each gap multiplies exactly by e^{i n k z}, so the product is
        // e^{i k (n1 z1 + n2 z2)}; the aperture step has zero distance and
        // transmits the probe sample (r = 95 um < 120 um), leaving the
        // ratio pure piston.
        let n = 16usize;
        let d = 20e-6;
        let band = BandField::uniform(1.0, 0.0, n, n, d, d, LAMBDA).unwrap();
        let mut seen: Vec<(f64, f64)> = Vec::new();
        let stop = EllipticalAperture::circle(120e-6, (0.0, 0.0)).unwrap();
        let mut bl = Beamline::new();
        bl.gap(1e-3, 1.5)
            .gap(0.5e-3, 1.0)
            .push_medium(&stop, 0.0, 1.5);
        let out = bl
            .trace(&band, |b, z, m| {
                seen.push((z, m));
                asm_propagate(
                    b,
                    z,
                    &AsmConfig {
                        padding: 1.0,
                        medium: m,
                        ..AsmConfig::default()
                    },
                )
            })
            .unwrap();
        assert_eq!(seen, vec![(1e-3, 1.5), (0.5e-3, 1.0)], "zero gaps skipped");
        let k = 2.0 * PI / LAMBDA;
        let expected = ((1.5 * 1e-3 + 1.0 * 0.5e-3) * k * Complex64::I).exp();
        let ratio = out.field().sample(3, 9) / band.field().sample(3, 9);
        assert!(
            (ratio - expected).norm() < 1e-9,
            "slab piston {ratio} vs {expected}"
        );
    }

    #[test]
    fn gap_only_beamline_and_len() {
        let mut bl = Beamline::new();
        assert!(bl.is_empty());
        bl.gap(0.0, 2.0); // zero distance: traced without engine call
        assert_eq!(bl.len(), 1);
        let calls = Cell::new(0usize);
        let out = bl
            .trace(&band(), |_b, _z, _n| {
                calls.set(calls.get() + 1);
                Ok::<BandField, Boom>(band())
            })
            .unwrap();
        assert_eq!(calls.get(), 0);
        assert_eq!(out, band());
    }
}
