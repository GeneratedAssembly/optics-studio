//! Propagation regime guidance — diagnostics only (Phase-2 decision).
//!
//! Given a band and a distance, which engine should you use? This module
//! answers without choosing for the caller (no auto-dispatch wrapper):
//!
//! - [`fresnel_number`] implements the regime criterion itself,
//!   $N_F = a^2/(\lambda |z|)$, for a *known* aperture half-width $a$
//!   (e.g. `EllipticalAperture::radius()`): $N_F \ll 1$ far field
//!   (Fraunhofer), $N_F \gtrsim 1$ near field (Fresnel/ASM).
//! - [`regime_report`] assesses a whole grid conservatively (window extent
//!   in place of an aperture radius, since the propagator cannot know the
//!   aperture) and reports every criterion the engines guard on, plus a
//!   recommendation.
//!
//! # Recommendation ladder
//!
//! 1. **Non-paraxial** (grid half-extent subtends $\theta \approx h/|z| >
//!    0.1$ rad): [`Engine::Asm`] — the only exact model; the binomial
//!    truncation of both Fresnel forms grows as $kz\,s^2/8$.
//! 2. **Far field** ($N_F \le 1$): [`Engine::Fraunhofer`] — valid and the
//!    cheapest (one forward transform; the Fresnel TF form costs forward +
//!    inverse). $N_F \le 1$ is strictly stronger than the convolution
//!    kernel's rule-2 guard ($\lambda z > Nd^2$), so the far field always
//!    finds [`Engine::FresnelConvolution`] well-sampled too — it is the
//!    right pick there when PSF/impulse-response access is wanted.
//! 3. **Near field, paraxial**: [`Engine::FresnelTransfer`] when its
//!    frequency-domain guard holds (the usual case — the guards of the two
//!    Fresnel forms are reciprocal, so one of them nearly always holds);
//!    [`Engine::FresnelConvolution`] when it is the sampled one.
//! 4. Anything else: [`Engine::Asm`] as the robust default.
//!
//! All SI units: meters, radians.

use std::fmt;

use super::fresnel;
use crate::field::BandField;

/// Which propagation engine a geometry points to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// Exact angular spectrum ([`super::asm`]) — valid at any distance and
    /// angle; costs forward + inverse transform on the padded grid.
    Asm,
    /// Paraxial transfer-function Fresnel ([`fresnel::propagate`]).
    FresnelTransfer,
    /// Paraxial impulse-response convolution ([`fresnel::propagate_convolution`]).
    FresnelConvolution,
    /// Far-field Fourier transform ([`super::fraunhofer::propagate`]).
    Fraunhofer,
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Engine::Asm => "asm",
            Engine::FresnelTransfer => "fresnel (transfer function)",
            Engine::FresnelConvolution => "fresnel (convolution)",
            Engine::Fraunhofer => "fraunhofer",
        };
        f.write_str(name)
    }
}

/// The Fresnel number $N_F = a^2 / (\lambda |z|)$.
///
/// The regime criterion for the distance from an aperture (or beam) of
/// half-width $a$ (m) at wavelength $\lambda$ (m) to the next plane
/// (distance $z$, m, sign ignored): $N_F \ll 1$ far field (Fraunhofer),
/// $N_F \gtrsim 1$ near field (Fresnel zones across the aperture),
/// $N_F \gg 1$ deep near field. Use the actual aperture radius
/// (e.g. [`crate::element::EllipticalAperture::radius`]) for a tight
/// estimate; [`regime_report`] falls back to the conservative window extent.
///
/// Returns `f64::INFINITY` for $z = 0$ and `f64::NAN` for invalid inputs
/// (negative/non-finite $a$, non-positive/non-finite $\lambda$,
/// non-finite $z$).
pub fn fresnel_number(a: f64, lambda: f64, z: f64) -> f64 {
    if !a.is_finite() || a < 0.0 || !lambda.is_finite() || lambda <= 0.0 || !z.is_finite() {
        return f64::NAN;
    }
    if z == 0.0 {
        return f64::INFINITY;
    }
    (a * a) / (lambda * z.abs())
}

/// Regime diagnostics for one propagation of `band` over `z`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegimeReport {
    /// Signed distance (m).
    pub z: f64,
    /// Conservative grid half-extents used as fallback source extent
    /// $((n_x - 1)\Delta x / 2, (n_y - 1)\Delta y / 2)$ (m).
    pub half_extent: (f64, f64),
    /// Source half-extent actually used (m): the declared aperture when
    /// valid, else the larger grid half-extent.
    pub extent_used: f64,
    /// `true` when [`RegimeReport::extent_used`] came from a declared
    /// aperture half-extent rather than the window.
    pub extent_is_aperture: bool,
    /// Fresnel number from the extent used, $h^2/(\lambda |z|)$.
    pub n_f: f64,
    /// `true` when $N_F \le 1$ (far field; Fraunhofer-eligible).
    pub far_field: bool,
    /// Largest transverse angle to the source extent used,
    /// $\theta \approx h/|z|$ (rad; `INFINITY` at $z = 0$).
    pub max_angle: f64,
    /// `true` when [`RegimeReport::max_angle`] $\le 0.1$ rad: the paraxial
    /// regime in which the Fresnel forms' binomial truncation is
    /// structurally justified for the full window.
    pub paraxial: bool,
    /// Fresnel transfer-function guard at default padding
    /// ([`fresnel::FresnelSamplingReport::nyquist_ok`]).
    pub tf_sampled: bool,
    /// Fresnel convolution kernel guard at default padding
    /// ([`fresnel::FresnelSamplingReport::kernel_nyquist_ok`]).
    pub kernel_sampled: bool,
    /// Ladder result (see module docs).
    pub recommended: Engine,
}

impl fmt::Display for RegimeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "regime: z={:.3e} m, extent={:.3e} m ({}) [window half=({:.3e}, {:.3e}) m], \
             N_F={:.3} (far_field={}), \
             theta_max={:.3} rad (paraxial={}), tf_sampled={}, kernel_sampled={} \
             -> recommend {}",
            self.z,
            self.extent_used,
            if self.extent_is_aperture {
                "aperture"
            } else {
                "window"
            },
            self.half_extent.0,
            self.half_extent.1,
            self.n_f,
            self.far_field,
            self.max_angle,
            self.paraxial,
            self.tf_sampled,
            self.kernel_sampled,
            self.recommended,
        )
    }
}

/// Assess `band` propagated a signed distance `z` (m) and recommend an
/// engine, exposing every criterion (see module docs for the ladder).
/// Pure diagnostics: nothing here executes a propagation.
///
/// The extent estimate is the conservative full window; when the physical
/// aperture is known (e.g. after [`crate::element::EllipticalAperture`]),
/// use [`regime_report_with`] with the true half-extent — the far-field
/// condition is a property of the source, not of the sampling window.
pub fn regime_report(band: &BandField, z: f64) -> RegimeReport {
    regime_report_with(band, z, None)
}

/// As [`regime_report`], with an optional true source half-extent `a`
/// (meters, from the axis; e.g. `EllipticalAperture::radius()`). A valid
/// positive finite `Some(a)` replaces the conservative window extent for
/// $N_F$ and the paraxial angle; `None` or degenerate values fall back.
pub fn regime_report_with(band: &BandField, z: f64, aperture: Option<f64>) -> RegimeReport {
    let (nx, ny) = band.field().dims();
    let (dx, dy) = band.field().pitch();
    let lambda = band.lambda();
    let hx = (nx as f64 - 1.0) / 2.0 * dx;
    let hy = (ny as f64 - 1.0) / 2.0 * dy;
    let window = hx.max(hy);
    let (extent, from_aperture) = match aperture.filter(|v| v.is_finite() && *v > 0.0) {
        Some(a) => (a, true),
        None => (window, false),
    };

    let n_f = fresnel_number(extent, lambda, z);
    let far_field = n_f <= 1.0;
    let max_angle = if z == 0.0 {
        f64::INFINITY
    } else {
        extent / z.abs()
    };
    let paraxial = max_angle <= 0.1;

    let fr = fresnel::sampling_report(
        band,
        z,
        &fresnel::FresnelConfig {
            strict: false,
            ..Default::default()
        },
    );
    let tf_sampled = fr.nyquist_ok;
    let kernel_sampled = fr.kernel_nyquist_ok;

    let recommended = if !paraxial {
        Engine::Asm
    } else if far_field {
        Engine::Fraunhofer
    } else if tf_sampled {
        Engine::FresnelTransfer
    } else if kernel_sampled {
        Engine::FresnelConvolution
    } else {
        Engine::Asm
    };

    RegimeReport {
        z,
        half_extent: (hx, hy),
        extent_used: extent,
        extent_is_aperture: from_aperture,
        n_f,
        far_field,
        max_angle,
        paraxial,
        tf_sampled,
        kernel_sampled,
        recommended,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAMBDA: f64 = 500e-9;

    fn band(n: usize, d: f64) -> BandField {
        BandField::uniform(1.0, 0.0, n, n, d, d, LAMBDA).unwrap()
    }

    #[test]
    fn fresnel_number_math_and_edge_cases() {
        // a = 1 mm, lambda = 500 nm, z = 2 m -> N_F = 1e-6/1e-6 = 1.
        assert!((fresnel_number(1e-3, LAMBDA, 2.0) - 1.0).abs() < 1e-12);
        // Sign of z irrelevant.
        let pos = fresnel_number(1e-3, LAMBDA, 2.0);
        let neg = fresnel_number(1e-3, LAMBDA, -2.0);
        assert!((pos - neg).abs() < 1e-12);
        assert!(fresnel_number(1e-3, LAMBDA, 0.0).is_infinite());
        assert!(fresnel_number(-1e-3, LAMBDA, 2.0).is_nan());
        assert!(fresnel_number(1e-3, 0.0, 2.0).is_nan());
        assert!(fresnel_number(f64::NAN, LAMBDA, 2.0).is_nan());
    }

    #[test]
    fn far_field_geometry_recommends_fraunhofer() {
        // n=64, d=10um, z=2 m: h=3.15e-4, N_F = h^2/(lambda z) = 0.099,
        // theta = 1.6e-4 rad -> far field, paraxial.
        let rep = regime_report(&band(64, 1e-5), 2.0);
        assert!(rep.far_field && rep.paraxial);
        assert_eq!(rep.recommended, Engine::Fraunhofer);
    }

    #[test]
    fn near_field_paraxial_recommends_fresnel_transfer() {
        // z = 0.01 m: N_F = 19.8 (near field), theta = 0.0315 (paraxial),
        // TF guard sampled -> Fresnel transfer function.
        let rep = regime_report(&band(64, 1e-5), 0.01);
        assert!(!rep.far_field && rep.paraxial && rep.tf_sampled);
        assert_eq!(rep.recommended, Engine::FresnelTransfer);
    }

    #[test]
    fn large_angle_recommends_asm() {
        // z = 1 mm: theta = 0.315 rad > 0.1 -> non-paraxial, exact model.
        let rep = regime_report(&band(64, 1e-5), 1e-3);
        assert!(!rep.paraxial);
        assert_eq!(rep.recommended, Engine::Asm);
        // z = 0 is likewise the identity/no-regime case.
        let rep0 = regime_report(&band(64, 1e-5), 0.0);
        assert_eq!(rep0.recommended, Engine::Asm);
        assert!(rep0.n_f.is_infinite());
    }

    #[test]
    fn guards_are_complementary_across_rule2_crossover() {
        // The guard crossover is rule 2: lambda z = (n-1) d^2 (kernel step
        // = pi). Below it the TF form is sampled (near field), above it the
        // kernel is sampled — and only much further out (lambda z > n^2 d^4
        // ... i.e. N_F <= 1) does the far field begin. One guard or the
        // other always holds.
        let b = band(64, 1e-5);
        let z_star = 63.0 * 1e-10 / LAMBDA; // (n-1) d^2 / lambda = 1.26e-2 m
        // Near but paraxial (theta = h/z = 0.063 rad): TF side of the pair.
        let z_near = 5e-3;
        let z_far = 100.0 * z_star;
        for &z in &[z_near, z_far] {
            let rep = regime_report(&b, z);
            assert!(
                rep.tf_sampled || rep.kernel_sampled,
                "both guards failed at z={z}: {rep}"
            );
        }
        let near = regime_report(&b, z_near);
        let far = regime_report(&b, z_far);
        assert!(near.tf_sampled && !near.kernel_sampled && !near.far_field);
        assert!(far.far_field && far.kernel_sampled && !far.tf_sampled);
        assert_eq!(far.recommended, Engine::Fraunhofer);
        assert_eq!(near.recommended, Engine::FresnelTransfer);
    }

    #[test]
    fn report_uses_window_extent_conservatively() {
        // A band whose actual content is a small centered aperture is still
        // judged on the full window; the tight N_F needs fresnel_number with
        // the true radius. Same geometry, two answers by design.
        let b = band(64, 1e-5);
        let rep = regime_report(&b, 0.05);
        let n_f_window = rep.n_f;
        let a = 1e-4; // a 0.1 mm aperture inside the 0.63 mm window
        let n_f_aperture = fresnel_number(a, LAMBDA, 0.05);
        assert!(n_f_aperture < n_f_window / 4.0);
        let expected = ((64 - 1) as f64 * 1e-5 / 2.0).powi(2) / (LAMBDA * 0.05);
        assert!((n_f_window - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn display_roundtrip_has_recommendation() {
        let rep = regime_report(&band(32, 2e-5), 1.0);
        let s = format!("{rep}");
        assert!(s.contains("recommend"));
        assert!(s.contains(&rep.recommended.to_string()));
    }

    #[test]
    fn declared_aperture_moves_the_regime_verdict() {
        // Same geometry as `report_uses_window_extent_conservatively`:
        // window says near field; the true 0.1 mm aperture says far.
        let b = band(64, 1e-5);
        let z = 0.05;
        let win = regime_report(&b, z);
        assert!(!win.far_field && !win.extent_is_aperture);
        // Fine grid at z = 0.05: TF guard fails (pi*3.8), kernel guard holds
        // (pi*0.25) -> convolution side of the complementary pair.
        assert!(!win.tf_sampled && win.kernel_sampled);
        assert_eq!(win.recommended, Engine::FresnelConvolution);
        let ap = regime_report_with(&b, z, Some(1e-4));
        assert!(ap.extent_is_aperture);
        assert!((ap.n_f - 0.4).abs() < 1e-9);
        assert!(ap.far_field);
        assert_eq!(ap.recommended, Engine::Fraunhofer);
        // Degenerate aperture falls back to the window.
        assert_eq!(regime_report_with(&b, z, Some(f64::NAN)), win);
    }
}
