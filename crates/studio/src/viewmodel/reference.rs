//! Synthetic diffraction-limited reference PSF (docs/ux/spec.md §3.3
//! "Strehl reference PSF", gap G-6).
//!
//! No core API generates a reference PSF, so the UI implements the
//! documented recipe: build a diffraction-limited stand-in with the same
//! aperture and focal geometry — `EllipticalAperture::circle` +
//! `ThinLens::fixed` traced with the **identical engine and grid** from
//! `sources::spherical` — and pass it as `reference` to
//! `analysis::strehl`. The panel must label the result
//! "reference is synthetic, same grid/engine".
//!
//! This is plumbing over core APIs (composition of source + elements +
//! engine dispatch), not new physics: every number comes from
//! `optics_core`.

use optics_core::beamline::{Beamline, BeamlineError};
use optics_core::element::{EllipticalAperture, ThinLens};
use optics_core::field::{BandField, FieldError, Psf};
use optics_core::sources::{SourcesError, spherical};

use crate::viewmodel::engine::{EngineError, EngineSettings, closure};

/// A reference-PSF trace failure.
#[derive(Debug, thiserror::Error)]
pub enum ReferenceError {
    /// The point-source grid was invalid.
    #[error(transparent)]
    Source(#[from] SourcesError),
    /// A synthetic element (aperture/lens) was built with invalid geometry.
    #[error(transparent)]
    Element(#[from] optics_core::element::OpticsError),
    /// The beamline trace failed (element application or engine).
    #[error(transparent)]
    Trace(#[from] BeamlineError<EngineError>),
    /// The traced field grid was invalid.
    #[error(transparent)]
    Field(#[from] FieldError),
    /// The geometry was invalid (a closed-form check failed; the core
    /// constructors would also reject it, but the message is clearer here).
    #[error("invalid reference geometry: {0}")]
    InvalidGeometry(String),
}

/// Geometry of the synthetic reference: the point source, the stop/lens,
/// and the detector grid — all SI (meters, radians) per crate convention.
#[derive(Clone, Debug)]
pub struct ReferenceGeometry {
    /// Lateral point-source position $(x_s, y_s)$ at the source plane (m).
    pub source: (f64, f64),
    /// Distance from the point source to the stop plane (m), $> 0$.
    pub source_distance: f64,
    /// Source amplitude $A$ (the $A/r$ field scale, `sources::spherical`).
    pub amplitude: f64,
    /// Wavelength (m).
    pub lambda: f64,
    /// Detector grid sample counts $(n_x, n_y)$.
    pub dims: (usize, usize),
    /// Detector grid pitch $(\Delta x, \Delta y)$ (m).
    pub pitch: (f64, f64),
    /// Clear circular aperture radius at the stop (m).
    pub aperture_radius: f64,
    /// Thin-lens focal length (m), non-zero.
    pub focal_length: f64,
    /// Distance from the lens to the detector plane (m).
    pub image_distance: f64,
}

impl ReferenceGeometry {
    /// Validate the geometry (cheap, closed-form checks only — the core
    /// constructors re-check the physics).
    pub fn validate(&self) -> Result<(), String> {
        if !self.source_distance.is_finite() || self.source_distance <= 0.0 {
            return Err("source_distance must be positive and finite".into());
        }
        if !self.lambda.is_finite() || self.lambda <= 0.0 {
            return Err("lambda must be positive and finite".into());
        }
        if self.dims.0 == 0 || self.dims.1 == 0 {
            return Err("detector dims must be non-zero".into());
        }
        if !self.pitch.0.is_finite()
            || self.pitch.0 <= 0.0
            || !self.pitch.1.is_finite()
            || self.pitch.1 <= 0.0
        {
            return Err("detector pitch must be positive and finite".into());
        }
        if !self.aperture_radius.is_finite() || self.aperture_radius <= 0.0 {
            return Err("aperture_radius must be positive and finite".into());
        }
        if !self.focal_length.is_finite() || self.focal_length == 0.0 {
            return Err("focal_length must be non-zero and finite".into());
        }
        if !self.image_distance.is_finite() {
            return Err("image_distance must be finite".into());
        }
        Ok(())
    }
}

/// Builds the synthetic reference PSF: spherical wave → circular stop →
/// thin lens → detector, traced with `settings` on the declared grid.
///
/// The same geometry fed to a real (possibly aberrated) trace yields the
/// measured PSF; `analysis::strehl(measured, reference)` then compares
/// peak-to-peak at equal total power.
pub fn build_reference(
    settings: &EngineSettings,
    geom: &ReferenceGeometry,
) -> Result<Psf, ReferenceError> {
    geom.validate().map_err(ReferenceError::InvalidGeometry)?;

    let band = spherical(
        geom.source,
        geom.source_distance,
        geom.amplitude,
        geom.lambda,
        geom.dims.0,
        geom.dims.1,
        geom.pitch.0,
        geom.pitch.1,
    )?;

    let aperture = EllipticalAperture::circle(geom.aperture_radius, (0.0, 0.0))?;
    let lens = ThinLens::fixed(
        geom.focal_length,
        None,
        optics_core::element::ClearAperture::Circle(geom.aperture_radius),
        (0.0, 0.0),
    )?;

    let mut line = Beamline::new();
    line.push(&aperture, 0.0);
    line.push(&lens, geom.image_distance);

    let psf = line.psf(&band, closure(settings.clone()))?;
    Ok(psf)
}

/// Convenience: a [`BandField`] of the same source (for callers that want
/// the pupil field, e.g. the wavefront panel's phase input).
pub fn build_pupil_band(
    settings: &EngineSettings,
    geom: &ReferenceGeometry,
) -> Result<BandField, ReferenceError> {
    let _ = settings; // source sampling is engine-independent
    geom.validate().map_err(ReferenceError::InvalidGeometry)?;
    Ok(spherical(
        geom.source,
        geom.source_distance,
        geom.amplitude,
        geom.lambda,
        geom.dims.0,
        geom.dims.1,
        geom.pitch.0,
        geom.pitch.1,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> ReferenceGeometry {
        ReferenceGeometry {
            source: (0.0, 0.0),
            source_distance: 0.05,
            amplitude: 1e-6,
            lambda: 550e-9,
            dims: (128, 128),
            pitch: (20e-6, 20e-6),
            aperture_radius: 1e-3,
            focal_length: 0.05,
            image_distance: 0.05,
        }
    }

    #[test]
    fn reference_builds_on_declared_grid() {
        let psf = build_reference(&EngineSettings::default(), &geom()).unwrap();
        assert_eq!(psf.dims(), (128, 128));
        assert_eq!(psf.pitch(), (20e-6, 20e-6));
        assert!(psf.total_power() > 0.0);
    }

    #[test]
    fn reference_against_itself_has_strehl_one() {
        use optics_core::analysis::strehl;
        let psf = build_reference(&EngineSettings::default(), &geom()).unwrap();
        assert!((strehl(&psf, &psf) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        let mut g = geom();
        g.aperture_radius = 0.0;
        assert!(build_reference(&EngineSettings::default(), &g).is_err());
        let mut g = geom();
        g.image_distance = f64::NAN;
        assert!(build_reference(&EngineSettings::default(), &g).is_err());
    }

    #[test]
    fn pupil_band_matches_grid() {
        let band = build_pupil_band(&EngineSettings::default(), &geom()).unwrap();
        assert_eq!(band.field().dims(), (128, 128));
        assert_eq!(band.lambda(), 550e-9);
    }
}
