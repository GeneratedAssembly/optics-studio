//! Demo frames for the viewport.
//!
//! Task 10.1 (the shell) supplies real imported PSFs; until it lands, the
//! viewport needs data to render and to exercise every panel. This module
//! builds a small synthetic bundle **from core APIs only** (the reference
//! recipe plus a defocused variant), so it stays within the no-physics-in-UI
//! constraint: it composes `optics_core` sources/elements/engines, it does
//! not implement them.
//!
//! The bundle is a [`DemoBundle`] with a diffraction-limited reference, a
//! defocused "measured" PSF, and the pupil phase for the wavefront panel.

use optics_core::analysis::zernike_modes;
use optics_core::field::Psf;
use optics_core::propagation::asm::{self, AsmConfig};

use crate::viewmodel::banner::{Banner, from_asm};
use crate::viewmodel::engine::EngineSettings;
use crate::viewmodel::metrics::{Frame, PupilPhaseGrid, make_frame};
use crate::viewmodel::reference::{
    ReferenceError, ReferenceGeometry, build_pupil_band, build_reference,
};

/// A synthetic multi-frame demo set for the viewport.
#[derive(Clone, Debug)]
pub struct DemoBundle {
    /// Diffraction-limited reference PSF (Strehl panel's `reference`).
    pub reference: Psf,
    /// Defocused measured PSF (Strehl < 1 vs the reference).
    pub measured: Psf,
    /// A second band (different wavelength) for the multi-band tabs.
    pub measured_b2: Psf,
    /// Pupil phase grid (radians) for the Zernike/wavefront panel, with its
    /// geometry, sampled from the measured band's field.
    pub pupil_phase: PupilPhaseGrid,
}

/// Base geometry for the demo (a fast f/25-ish point-source trace).
/// Base geometry for the demo: a finite-conjugate point-source trace
/// (source 60 mm before an f = 50 mm lens, in-focus image at 300 mm).
fn base_geometry(lambda: f64) -> ReferenceGeometry {
    ReferenceGeometry {
        source: (0.0, 0.0),
        source_distance: 0.06,
        amplitude: 1e-6,
        lambda,
        dims: (128, 128),
        pitch: (20e-6, 20e-6),
        aperture_radius: 1.0e-3,
        focal_length: 0.05,
        image_distance: 0.3,
    }
}

/// In-focus image distance for the base geometry (lens equation).
const IN_FOCUS: f64 = 0.3;
/// Defocus offset applied to the "measured" bands — ~3x the depth of focus
/// (NA = 1e-3/0.05 = 0.02, DOF = lambda/NA^2 ~ 1.4 mm), so the Strehl is
/// well below 1 while the spot stays on-grid.
const DEFOCUS: f64 = 4e-3;

impl DemoBundle {
    /// Builds the demo bundle with the default engine settings (ASM).
    ///
    /// # Errors
    /// [`DemoError`] wrapping any core [`ReferenceError`] from the traces.
    pub fn build() -> Result<Self, DemoError> {
        let settings = EngineSettings::default();

        // Reference: on-focus image distance (diffraction-limited Airy).
        let ref_geom = base_geometry(550e-9);
        let reference = build_reference(&settings, &ref_geom)?;

        // Measured: defocused past the depth of focus -> Strehl < 1.
        let mut meas_geom = base_geometry(550e-9);
        meas_geom.image_distance = IN_FOCUS + DEFOCUS;
        let measured = build_reference(&settings, &meas_geom)?;

        // Second band (650 nm) for the multi-band tabs.
        let mut b2_geom = base_geometry(650e-9);
        b2_geom.image_distance = IN_FOCUS + DEFOCUS;
        let measured_b2 = build_reference(&settings, &b2_geom)?;

        // Pupil phase: a synthetic defocus bowl (display fixture for the
        // wavefront panel; the Zernike fit produces the coefficients).
        let pupil = analytic_pupil_phase(&meas_geom);

        Ok(DemoBundle {
            reference,
            measured,
            measured_b2,
            pupil_phase: pupil,
        })
    }

    /// Zernike mode list up to the panel's truncation order, for the
    /// coefficient-bar labels. Each entry is `(n, m)` with `m` the signed
    /// azimuthal order (`ZernikeMode`).
    pub fn zernike_labels(n_max: u32) -> Vec<(u32, i32)> {
        zernike_modes(n_max)
            .into_iter()
            .map(|mode| (mode.n, mode.m))
            .collect()
    }

    /// Turns the bundle into viewport frames (multi-band tabs): the
    /// reference, then each measured band with its Strehl-vs-reference and
    /// wavefront metrics precomputed through [`make_frame`].
    pub fn frames(&self) -> Vec<Frame> {
        let mut frames = vec![make_frame(
            "reference (synthetic)",
            self.reference.clone(),
            None,
            None,
        )];
        for (label, psf) in [
            ("550 nm (defocused)", &self.measured),
            ("650 nm (defocused)", &self.measured_b2),
        ] {
            frames.push(make_frame(
                label,
                psf.clone(),
                Some(&self.reference),
                Some(&self.pupil_phase),
            ));
        }
        frames
    }
}

/// A demo error.
#[derive(Debug, thiserror::Error)]
pub enum DemoError {
    /// A core trace failed.
    #[error(transparent)]
    Reference(#[from] ReferenceError),
}

/// The full viewport model: frames + the sampling banner for the last run.
///
/// Built from [`DemoBundle`] until task 10.1 supplies imported prescriptions;
/// the banner is the real ASM [`optics_core::propagation::asm::sampling_report`]
/// for the measured geometry
/// (pure diagnostics, spec §5.1 — nothing here re-derives a criterion).
#[derive(Clone, Debug)]
pub struct StudioModel {
    /// Multi-band frames (tabs); index 0 is the synthetic reference.
    pub frames: Vec<Frame>,
    /// Sampling banner from the demo run's diagnostics.
    pub banner: Banner,
}

impl StudioModel {
    /// Builds the demo model (see [`DemoBundle::build`]).
    ///
    /// # Errors
    /// [`DemoError`] if any core trace fails.
    pub fn demo() -> Result<Self, DemoError> {
        let bundle = DemoBundle::build()?;
        let frames = bundle.frames();

        let settings = EngineSettings::default();
        let mut geom = base_geometry(550e-9);
        geom.image_distance = IN_FOCUS + DEFOCUS;
        let band = build_pupil_band(&settings, &geom)?;
        let report = asm::sampling_report(&band, geom.image_distance, &AsmConfig::default());
        let banner = from_asm(&report);

        Ok(Self { frames, banner })
    }

    /// The active frame, clamped to the frame list.
    pub fn active(&self, index: usize) -> &Frame {
        let i = index.min(self.frames.len().saturating_sub(1));
        &self.frames[i]
    }

    /// Zernike mode labels up to `n_max` for the wavefront panel bars.
    pub fn zernike_labels(n_max: u32) -> Vec<(u32, i32)> {
        DemoBundle::zernike_labels(n_max)
    }
}

/// Analytic defocus pupil phase (radians) on the detector grid geometry.
///
/// A pure geometric display fixture for the wavefront panel: a parabolic
/// defocus bowl $\varphi(\rho) = \varphi_{pv}\,(\rho/a)^2$ reaching
/// $\varphi_{pv} = 4\pi$ rad (2 waves) at the pupil rim. The Zernike fit in
/// the panel — not this function — produces the coefficients, so no metric
/// is computed in the UI crate (spec §6).
fn analytic_pupil_phase(geom: &ReferenceGeometry) -> PupilPhaseGrid {
    let (nx, ny) = geom.dims;
    let (dx, dy) = geom.pitch;
    let a = geom.aperture_radius;
    // Peak-to-valley ~ 2 waves of defocus across the pupil at the rim.
    let pv = 4.0 * std::f64::consts::PI;
    let mut values = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dy;
        for ix in 0..nx {
            let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx;
            let rho2 = (x * x + y * y) / (a * a);
            values.push(pv * rho2.min(1.0));
        }
    }
    PupilPhaseGrid {
        values,
        dims: (nx, ny),
        pitch: (dx, dy),
        radius: a,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optics_core::analysis::strehl;

    #[test]
    fn bundle_builds() {
        let d = DemoBundle::build().unwrap();
        assert_eq!(d.reference.dims(), (128, 128));
        assert_eq!(d.measured.dims(), (128, 128));
        assert_eq!(d.measured_b2.dims(), (128, 128));
        assert!(d.measured.total_power() > 0.0);
        // The two measured bands differ (different lambda -> different
        // diffraction scale), so the multi-band tabs show distinct frames.
        assert_ne!(d.measured.samples(), d.measured_b2.samples());
    }

    #[test]
    fn measured_is_aberrated_vs_reference() {
        let d = DemoBundle::build().unwrap();
        let s = strehl(&d.measured, &d.reference);
        assert!(s < 1.0, "defocus should drop Strehl below 1 (got {s})");
        assert!(s > 0.0);
    }

    #[test]
    fn pupil_phase_is_finite_bowl() {
        let d = DemoBundle::build().unwrap();
        assert!(d.pupil_phase.values.iter().all(|v| v.is_finite()));
        assert_eq!(d.pupil_phase.dims, (128, 128));
        // Center is the minimum of the defocus bowl.
        let c = d.pupil_phase.values[(127 / 2) * 128 + 127 / 2];
        let edge = d.pupil_phase.values[0];
        assert!(c <= edge);
    }

    #[test]
    fn zernike_labels_present() {
        let labels = DemoBundle::zernike_labels(4);
        assert!(!labels.is_empty());
        assert_eq!(labels[0], (0, 0)); // piston first
    }

    #[test]
    fn frames_carry_metrics() {
        let d = DemoBundle::build().unwrap();
        let frames = d.frames();
        assert_eq!(frames.len(), 3);
        assert!(frames[0].metrics.strehl.is_none());
        assert!(frames[1].metrics.strehl.unwrap() < 1.0);
        let wf = frames[1]
            .metrics
            .wavefront
            .as_ref()
            .expect("wavefront fit ran");
        assert!(!wf.coefficients.is_empty());
        assert!(frames[1].metrics.marechal_strehl.is_some());
    }
}
