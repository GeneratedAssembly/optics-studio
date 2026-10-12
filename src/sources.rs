//! Field initializers: physical beam descriptions turned into [`BandField`]s.
//!
//! Sources are *data + pure sampling*: no propagation machinery lives here.
//! Each initializer evaluates a closed-form paraxial solution on the sample
//! grid and binds the result to a wavelength, so a multiband source is simply
//! a collection of `BandField`s built from the same beam record (approved
//! Phase-3 decision D3: `sources` is a top-level module returning
//! [`BandField`]s, multiband by construction).
//!
//! # Position is a system property, not a source property
//!
//! A [`GaussianBeam`] describes the beam *itself* (waist radius, amplitude,
//! transverse center); it holds **no longitudinal position**. Where the beam
//! is sampled along the optical axis — the signed distance `z` from the waist
//! to the sampling plane — is supplied at sampling time to [`gaussian`], just
//! as an element's plane placement is a system decision rather than a property
//! of the element. The beam radius and Gouy phase are therefore *functions of
//! `z`* ([`GaussianBeam::beam_radius`], [`GaussianBeam::gouy_phase`]): a
//! measurement of the beam at a particular plane, which matters when a system
//! places sources at distances other than the waist.
//!
//! # Gaussian beam convention
//!
//! The classical paraxial Gaussian beam (Siegman/Goodman; the project
//! textbook only uses Gaussian beams as inputs to beam shapers, Sec. 11.2,
//! and does not restate the $q(z)$ solution) is written here in the
//! $q$-form adapted to **this crate's $e^{+ikz}$ forward-propagation
//! convention** (ASM/Fresnel transfer functions are $e^{+ikz\sqrt{1-\lambda^2 f^2}}$;
//! the on-axis phase $kz - \zeta(z)$ of the propagated waist was pinned by
//! the Phase-2 validation suite, `tests/phase2.rs`):
//!
//! $$u(\rho; z) = A\,\frac{w_0}{w(z)}\,
//!   \exp\!\Big(-\frac{\rho^2}{w(z)^2}\Big)\,
//!   \exp\!\Big(i\,\frac{k\rho^2}{2R(z)}\Big)\,
//!   e^{\,i(kz - \zeta(z))}, \qquad \rho^2 = (x-x_c)^2 + (y-y_c)^2$$
//!
//! with
//!
//! $$z_R = \frac{\pi w_0^2}{\lambda}, \quad
//!   w(z) = w_0\sqrt{1 + (z/z_R)^2}, \quad
//!   \frac{1}{R(z)} = \frac{z}{z^2 + z_R^2}, \quad
//!   \zeta(z) = \arctan(z/z_R), \quad k = \frac{2\pi}{\lambda}.$$
//!
//! Equivalently $u \propto \exp\!\big(i k \rho^2 / (2q)\big)$ with
//! $q(z) = z - i z_R$ — the conjugate of the usual $e^{-ikz}$-textbook
//! $q = z + i z_R$. Here `z` is the **signed distance from the waist to the
//! sampling plane** (meters), a plane-placement argument to [`gaussian`] and
//! to the radius/Gouy accessors, not a stored beam field: $z = 0$ samples the
//! waist (pure real Gaussian), $z > 0$ a diverging beam, $z < 0$ a converging
//! one, and $u(\rho; -z) = \overline{u(\rho; z)}$ (pin test below). The
//! curvature $1/R$ form is used so $z = 0$ is exact, not a limit.
//!
//! Sign conventions (Gouy, curvature) are the historical bug source: they are
//! pinned by cross-checks against the crate's own engines — Fresnel-TF
//! propagation of a waist beam must reproduce `gaussian` at $z$ to
//! discretization accuracy, and ASM to paraxial accuracy
//! ([`gaussian`] unit tests).
//!
//! # Sampling guidance
//!
//! The window should cover the beam without truncation: half-extents of
//! several $w(z)$ keep the power loss below $e^{-2(\text{extent}/w)^2}$ per
//! side, and the pitch should resolve the waist, $\Delta x \lesssim w_0/10$,
//! or the sampled profile (and its far field) is biased.

use std::f64::consts::PI;

use num_complex::Complex64;
use thiserror::Error;

use crate::field::{BandField, Field, FieldError};

/// Errors produced by field initializers.
///
/// Beam-parameter problems are reported as [`SourcesError::InvalidBeam`]; a
/// non-finite sampling-plane distance as [`SourcesError::InvalidPlane`]; grid
/// and wavelength validity is delegated to [`FieldError`] (carried
/// transparently).
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum SourcesError {
    /// A Gaussian beam parameter violated its invariant: `w0` must be
    /// positive and finite; `amplitude` and the center coordinates must be
    /// finite (SI: meters, arbitrary coherent amplitude units).
    #[error(
        "invalid Gaussian beam: w0 = {w0} m (must be positive and finite), \
         amplitude = {amplitude}, center = ({cx}, {cy}) m (all must be finite)"
    )]
    InvalidBeam {
        /// Waist radius in meters.
        w0: f64,
        /// Peak field amplitude at the waist.
        amplitude: f64,
        /// x center of the waist in meters (field coordinates).
        cx: f64,
        /// y center of the waist in meters (field coordinates).
        cy: f64,
    },
    /// The signed waist-to-sampling-plane distance `z` was not finite
    /// (meters). Position is a sampling argument, not a beam field, so it is
    /// validated separately from the beam.
    #[error("invalid sampling plane: z = {z} m (must be finite)")]
    InvalidPlane {
        /// Signed waist-to-plane distance in meters.
        z: f64,
    },
    /// Grid or wavelength validation failed.
    #[error(transparent)]
    Field(#[from] FieldError),
}

/// Gaussian beam parameter record (SI units, meters).
///
/// A pure data record describing the paraxial fundamental-mode beam *as
/// such*: waist radius, peak amplitude, and transverse center. It carries no
/// longitudinal position — the waist-to-plane distance `z` is a system
/// placement argument passed to [`gaussian`] and to the radius/Gouy accessors
/// (see the module's "position is a system property" note). Pass a beam to
/// [`gaussian`] to sample a [`BandField`] at a chosen plane and wavelength;
/// one record can generate many bands — an incoherent multiband source is a
/// collection of the resulting `BandField`s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianBeam {
    /// Waist radius $w_0$ in meters (field $1/e$ radius at the waist);
    /// positive and finite.
    pub w0: f64,
    /// Peak field amplitude $A$ at the waist (arbitrary coherent units;
    /// $|u|^2$ is irradiance up to a constant). Finite.
    pub amplitude: f64,
    /// Waist center $(x_c, y_c)$ in meters, field coordinates relative to
    /// the grid origin. Finite.
    pub center: (f64, f64),
}

impl GaussianBeam {
    /// Checks the beam invariants (see [`GaussianBeam`] field docs).
    ///
    /// # Errors
    /// [`SourcesError::InvalidBeam`] if any parameter is non-finite or
    /// `w0 <= 0`.
    pub fn validate(&self) -> Result<(), SourcesError> {
        let (cx, cy) = self.center;
        if !self.w0.is_finite()
            || self.w0 <= 0.0
            || !self.amplitude.is_finite()
            || !cx.is_finite()
            || !cy.is_finite()
        {
            return Err(SourcesError::InvalidBeam {
                w0: self.w0,
                amplitude: self.amplitude,
                cx,
                cy,
            });
        }
        Ok(())
    }

    /// Rayleigh range $z_R = \pi w_0^2 / \lambda$ in meters for the given
    /// wavelength (meters). This is the beam's intrinsic length scale and is
    /// independent of the sampling plane.
    pub fn rayleigh_range(&self, lambda: f64) -> f64 {
        PI * self.w0 * self.w0 / lambda
    }

    /// Beam radius $w(z) = w_0 \sqrt{1 + (z/z_R)^2}$ in meters measured at
    /// the sampling plane a signed distance `z` (meters) from the waist, for
    /// wavelength `lambda` (meters).
    pub fn beam_radius(&self, z: f64, lambda: f64) -> f64 {
        let zr = self.rayleigh_range(lambda);
        self.w0 * (1.0 + (z / zr).powi(2)).sqrt()
    }

    /// Gouy phase $\zeta(z) = \arctan(z / z_R)$ in radians accumulated at the
    /// sampling plane a signed distance `z` (meters) from the waist, for
    /// wavelength `lambda` (meters).
    pub fn gouy_phase(&self, z: f64, lambda: f64) -> f64 {
        (z / self.rayleigh_range(lambda)).atan()
    }
}

/// Samples a [`GaussianBeam`] onto an `nx` x `ny` grid at the plane a signed
/// distance `z` from the waist, at wavelength `lambda`, returning the
/// [`BandField`] (module docs give the exact formula and sign conventions).
///
/// # Arguments
/// - `beam`: waist radius, amplitude, and transverse center (meters).
/// - `z`: signed waist-to-sampling-plane distance in meters ($0$ = waist,
///   $>0$ diverging, $<0$ converging); must be finite. This is a system
///   placement argument, not a beam property.
/// - `lambda`: wavelength in meters; positive and finite.
/// - `nx`, `ny`: sample counts along $x$, $y$; both > 0.
/// - `dx`, `dy`: sample pitch in meters; positive and finite.
///
/// # Errors
/// [`SourcesError::InvalidBeam`] for bad beam parameters, then
/// [`SourcesError::InvalidPlane`] for a non-finite `z`, then [`FieldError`]
/// (wrapped) for invalid wavelength, grid, or non-finite samples.
///
/// # Sampling
/// See the module-level guidance: resolve the waist (`dx <= w0/10`) and cover
/// several $w(z)$ per half-window, or the truncation/bias shows up in the
/// power and far field.
// The full grid specification (nx, ny, dx, dy) mirrors BandField constructors;
// a grid-spec struct is deferred until more initializers exist (Phase 5).
#[allow(clippy::too_many_arguments)]
pub fn gaussian(
    beam: &GaussianBeam,
    z: f64,
    lambda: f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Result<BandField, SourcesError> {
    beam.validate()?;
    if !z.is_finite() {
        return Err(SourcesError::InvalidPlane { z });
    }
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(FieldError::NonPositiveWavelength(lambda).into());
    }
    let z_r = beam.rayleigh_range(lambda);
    let w_z = beam.beam_radius(z, lambda);
    let k = 2.0 * PI / lambda;
    // 1/R(z) written as z/(z^2 + z_R^2): exact and finite at z = 0 (flat
    // wavefront), no division-by-infinity branch needed.
    let inv_r = z / (z * z + z_r * z_r);
    let carrier = k * z - beam.gouy_phase(z, lambda);
    let amp = beam.amplitude * beam.w0 / w_z;
    let mut samples = Vec::with_capacity(nx.saturating_mul(ny));
    for iy in 0..ny {
        let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dy - beam.center.1;
        for ix in 0..nx {
            let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx - beam.center.0;
            let rho2 = x * x + y * y;
            let envelope = amp * (-rho2 / (w_z * w_z)).exp();
            let phase = carrier + 0.5 * k * rho2 * inv_r;
            samples.push(envelope * (phase * Complex64::I).exp());
        }
    }
    let field = Field::new(samples, nx, ny, dx, dy)?;
    Ok(BandField::new(lambda, field)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::{asm, fresnel};

    const LAMBDA: f64 = 500e-9;

    fn beam(w0: f64) -> GaussianBeam {
        GaussianBeam {
            w0,
            amplitude: 1.0,
            center: (0.0, 0.0),
        }
    }

    /// Relative Frobenius norm ||a - b|| / ||b|| over the sample buffers.
    fn rel_frobenius(a: &BandField, b: &BandField) -> f64 {
        let diff: f64 = a
            .field()
            .samples()
            .iter()
            .zip(b.field().samples())
            .map(|(x, y)| (x - y).norm_sqr())
            .sum();
        let norm: f64 = b.field().samples().iter().map(|y| y.norm_sqr()).sum();
        diff.sqrt() / norm.sqrt()
    }

    fn wrap(phase: f64) -> f64 {
        (phase + PI).rem_euclid(2.0 * PI) - PI
    }

    #[test]
    fn waist_plane_is_real_gaussian_with_analytic_power() {
        // w0 = 1 mm, 20 samples per w0, half-window = 5 w0: truncation loss
        // ~ e^{-50} per side, discretization error tiny.
        let w0 = 1e-3;
        let d = w0 / 20.0;
        let n = 201usize;
        let band = gaussian(&beam(w0), 0.0, LAMBDA, n, n, d, d).unwrap();
        let f = band.field();
        // Sample values match A exp(-rho^2/w0^2) exactly (imaginary part 0).
        for (ix, iy) in [(0usize, 0usize), (100, 100), (200, 200), (37, 160)] {
            let (x, y) = f.coord(ix, iy);
            let expected = (-((x * x + y * y) / (w0 * w0))).exp();
            let s = f.sample(ix, iy);
            assert!((s.im).abs() < 1e-15);
            assert!((s.re - expected).abs() < 1e-12);
        }
        // P = A^2 * pi w0^2 / 2 (integral of exp(-2 rho^2/w0^2)).
        let expected_power = PI * w0 * w0 / 2.0;
        let rel = (f.power() - expected_power).abs() / expected_power;
        assert!(rel < 1e-3, "power rel. error {rel:e}");
    }

    #[test]
    fn on_axis_amplitude_and_gouy_match_q_form() {
        let w0 = 0.5e-3;
        let z = 0.25;
        let n = 129usize;
        let d = w0 / 20.0;
        let src = GaussianBeam {
            w0,
            amplitude: 2.0,
            center: (0.0, 0.0),
        };
        let band = gaussian(&src, z, LAMBDA, n, n, d, d).unwrap();
        let c = band.field().sample(n / 2, n / 2); // odd grid: origin sample
        let z_r = src.rayleigh_range(LAMBDA);
        let w_z = src.beam_radius(z, LAMBDA);
        let amp_exp = src.amplitude * w0 / w_z;
        assert!((c.norm() - amp_exp).abs() / amp_exp < 1e-12);
        let phase_exp = 2.0 * PI * z / LAMBDA - (z / z_r).atan();
        assert!(wrap(c.arg() - phase_exp).abs() < 1e-9);
        // The accessors are pure functions of z: w(0) = w0, gouy(0) = 0.
        assert!((src.beam_radius(0.0, LAMBDA) - w0).abs() < 1e-15);
        assert_eq!(src.gouy_phase(0.0, LAMBDA), 0.0);
    }

    #[test]
    fn analytic_beam_matches_own_engines() {
        // Convention pin: propagate a waist beam by z with the engines and
        // compare to the analytic initializer sampled at z. Fresnel TF is
        // exact for the paraxial Gaussian (its eigenfunction) up to
        // discretization; ASM differs by the (tiny) non-paraxial remainder.
        let w0 = 0.15e-3;
        let z = 0.1;
        let n = 255usize;
        let d = 1e-5;
        let analytic = gaussian(&beam(w0), z, LAMBDA, n, n, d, d).unwrap();
        let waist_band = gaussian(&beam(w0), 0.0, LAMBDA, n, n, d, d).unwrap();

        let fr = fresnel::propagate(
            &waist_band,
            z,
            &fresnel::FresnelConfig {
                padding: 2.0,
                strict: false,
                ..Default::default()
            },
        )
        .unwrap();
        let asm_out = asm::propagate(&waist_band, z, &asm::AsmConfig::default()).unwrap();

        let rel_fr = rel_frobenius(&fr, &analytic);
        let rel_asm = rel_frobenius(&asm_out, &analytic);
        assert!(rel_fr < 1e-3, "Fresnel TF vs analytic: {rel_fr:e}");
        assert!(rel_asm < 1e-3, "ASM vs analytic: {rel_asm:e}");
    }

    #[test]
    fn negative_z_is_complex_conjugate() {
        // u(rho; -z) = conj(u(rho; z)): converging is the mirror of diverging.
        let w0 = 0.5e-3;
        let z = 0.2;
        let at = |zz: f64| gaussian(&beam(w0), zz, LAMBDA, 64, 64, 1e-5, 1e-5).unwrap();
        let pos = at(z);
        let neg = at(-z);
        for (p, n) in pos.field().samples().iter().zip(neg.field().samples()) {
            assert!((p.conj() - n).norm() < 1e-12);
        }
    }

    #[test]
    fn off_center_waist_translates_by_integer_samples() {
        let w0 = 1e-3;
        let d = w0 / 20.0;
        let n = 161usize;
        let on_axis = gaussian(&beam(w0), 0.0, LAMBDA, n, n, d, d).unwrap();
        let shifted = gaussian(
            &GaussianBeam {
                w0,
                amplitude: 1.0,
                center: (3.0 * d, -2.0 * d),
            },
            0.0,
            LAMBDA,
            n,
            n,
            d,
            d,
        )
        .unwrap();
        // Beam moved +3 bins in x, -2 in y: shifted(ix) = on_axis(ix-3),
        // since u_shifted(x) = u(x - x_c) translates the profile along +x_c.
        for iy in 5..n - 5 {
            for ix in 5..n - 5 {
                let sx = ix as isize - 3;
                let sy = iy as isize + 2;
                if sx >= 0 && sy >= 0 && (sx as usize) < n && (sy as usize) < n {
                    let a = shifted.field().sample(ix, iy);
                    let b = on_axis.field().sample(sx as usize, sy as usize);
                    assert!((a - b).norm() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn validation_rejects_bad_parameters() {
        let bad = [
            GaussianBeam {
                w0: 0.0,
                ..beam(1e-3)
            },
            GaussianBeam {
                w0: -1e-3,
                ..beam(1e-3)
            },
            GaussianBeam {
                w0: f64::NAN,
                ..beam(1e-3)
            },
            GaussianBeam {
                amplitude: f64::INFINITY,
                ..beam(1e-3)
            },
            GaussianBeam {
                center: (0.0, f64::INFINITY),
                ..beam(1e-3)
            },
        ];
        for b in &bad {
            assert!(matches!(
                gaussian(b, 0.0, LAMBDA, 8, 8, 1e-4, 1e-4),
                Err(SourcesError::InvalidBeam { .. })
            ));
        }
        // Non-finite sampling-plane distance is rejected separately (z is a
        // placement argument, validated after the beam).
        for z in [f64::NAN, f64::INFINITY] {
            assert!(matches!(
                gaussian(&beam(1e-3), z, LAMBDA, 8, 8, 1e-4, 1e-4),
                Err(SourcesError::InvalidPlane { .. })
            ));
        }
        // Wavelength and grid errors come through as Field(_).
        assert!(matches!(
            gaussian(&beam(1e-3), 0.0, 0.0, 8, 8, 1e-4, 1e-4),
            Err(SourcesError::Field(FieldError::NonPositiveWavelength(0.0)))
        ));
        assert!(matches!(
            gaussian(&beam(1e-3), 0.0, LAMBDA, 0, 8, 1e-4, 1e-4),
            Err(SourcesError::Field(FieldError::EmptyGrid))
        ));
        assert!(matches!(
            gaussian(&beam(1e-3), 0.0, LAMBDA, 8, 8, f64::NAN, 1e-4),
            Err(SourcesError::Field(FieldError::NonPositivePitch { .. }))
        ));
    }
}
