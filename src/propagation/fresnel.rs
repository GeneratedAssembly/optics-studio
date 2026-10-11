//! Fresnel transfer-function propagation (paraxial Angular Spectrum).
//!
//! The Fresnel method is the paraxial approximation of the exact ASM
//! transfer function ([`super::asm`]): expanding the direction cosine
//! $\gamma = \sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2} \approx
//! 1 - \tfrac{1}{2}\big((\lambda f_x)^2 + (\lambda f_y)^2\big)$ in
//! $H = e^{ikz\gamma}$ gives
//!
//! $$H_F(f_x, f_y; z) = e^{\,ikz}\,
//!   \exp\!\Big(-i\pi\lambda z\,(f_x^2 + f_y^2)\Big), \qquad k = \frac{2\pi}{\lambda},$$
//!
//! and the propagation keeps the same diagonal-in-frequency structure:
//!
//! $$U(x, y, z) = \mathcal{F}^{-1}\!\Big\{\, H_F \cdot \mathcal{F}\{U(x, y, 0)\}\,\Big\}.$$
//!
//! The quadratic chirp phase is exactly the kernel of the Fresnel integral
//! (Rayleigh–Sommerfeld with the binomial expansion of $r$, textbook
//! Sec. 2.1–2.2); the FFT-scaling factors cancel as in ASM, so the
//! NumPy-convention layer of `super::fft` applies unchanged.
//!
//! # Validity
//!
//! Paraxial only: the binomial truncation error at bin $(f_x, f_y)$ is
//! $k z\big(1 - \tfrac{s}{2} - \sqrt{1 - s}\big)$ with $s = (\lambda f_x)^2 +
//! (\lambda f_y)^2$; [`sampling_report`] exposes its worst-case value over
//! the grid (`max_phase_error_vs_asm`, flagged when $\ge 0.01$ rad) plus the
//! adjacent-bin phase-step Nyquist guard below. Unlike ASM there is *no*
//! evanescent cutoff in $H_F$ — the model is simply invalid where $s$ grows;
//! use [`super::asm::propagate`] when the spectrum reaches large angles.
//!
//! # Sampling verification (textbook Sec. 2.2, rules 2–3)
//!
//! The phase of $H_F$ between adjacent frequency bins along $x$ is
//! $\pi\lambda z\,(f_{hi}^2 - f_{hi \to lo}^2)$ — separable in the axes, so
//! the worst step is closed-form at the largest-magnitude bins:
//!
//! $$\Delta\phi_x^{\max} = \pi\lambda z \, \delta f_x \,(2 f_{\text{nyq},x} - \delta f_x),
//!   \qquad \delta f_x = \frac{1}{n_{x,\text{pad}} \Delta x}.$$
//!
//! Requiring $\Delta\phi < \pi$ is algebraically $n_{\text{pad}} >
//! \lambda z / \Delta x^2$ — the same criterion as the textbook's sampling
//! rules $N > D^2/(Z\lambda)$ and $\Delta < \sqrt{Z\lambda}/D$ expressed on
//! the padded grid. `strict` mode (default) mitigates violations by doubling
//! `padding` up to `max_padding` before raising
//! [`FresnelError::AliasingRisk`], mirroring the ASM policy.

use std::f64::consts::PI;
use std::fmt;

use num_complex::Complex64;
use thiserror::Error;

use super::fft::{fft2, fftfreq, ifft2};
use super::padded_len;
use crate::field::{BandField, Field};

/// Tuning knobs for [`propagate`].
#[derive(Clone, Copy, Debug)]
pub struct FresnelConfig {
    /// Linear zero-padding factor applied to each axis before transform;
    /// values `< 1.0` are treated as `1.0` (no padding). Typical: `2.0`.
    pub padding: f64,
    /// When `true`, [`propagate`] mitigates an undersampled chirp transfer
    /// function by densifying the padded grid (doubling `padding` up to
    /// `max_padding`) and returns [`FresnelError::AliasingRisk`] only if the
    /// cap cannot restore Nyquist adequacy.
    pub strict: bool,
    /// Upper bound on automatic densification in `strict` mode
    /// (`padding` never exceeds this). `1.0` disables mitigation.
    pub max_padding: f64,
}

impl Default for FresnelConfig {
    /// Phase-2 defaults mirroring ASM: `padding = 2.0`, `strict = true`,
    /// `max_padding = 8.0`.
    fn default() -> Self {
        Self {
            padding: 2.0,
            strict: true,
            max_padding: 8.0,
        }
    }
}

/// Spatial-sampling and paraxial-validity diagnostics for one Fresnel TF
/// propagation step.
///
/// All quantities SI: meters, cycles per meter, radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FresnelSamplingReport {
    /// Operating wavelength (m).
    pub lambda: f64,
    /// Signed propagation distance (m).
    pub z: f64,
    /// Input grid sample counts `(nx, ny)`.
    pub input_dims: (usize, usize),
    /// Input sample pitch `(dx, dy)` (m).
    pub pitch: (f64, f64),
    /// Padding factor requested via [`FresnelConfig::padding`].
    pub requested_padding: f64,
    /// Padding factor actually used (equals `requested_padding` unless
    /// `strict` mode densified the grid).
    pub used_padding: f64,
    /// Padded grid sample counts `(nx_pad, ny_pad)` at `used_padding`.
    pub padded_dims: (usize, usize),
    /// Closed-form worst adjacent-bin phase step of $H_F$ along $f_x$ (rad).
    pub max_phase_step_x: f64,
    /// Worst adjacent-bin phase step along $f_y$ (rad).
    pub max_phase_step_y: f64,
    /// `true` when both max phase steps are below $\pi$: the chirp transfer
    /// function is Nyquist-sampled on the padded grid (textbook rules 2–3).
    pub nyquist_ok: bool,
    /// Worst-case binomial-truncation phase error of $H_F$ against the exact
    /// ASM root over the padded grid (rad). $0.5\,k|z|$ when the grid reaches
    /// evanescent territory ($s \ge 1$), where the comparison saturates.
    pub max_phase_error_vs_asm: f64,
    /// `true` when [`FresnelSamplingReport::max_phase_error_vs_asm`] < 0.01
    /// rad: the paraxial truncation is negligible at this distance.
    pub paraxial_ok: bool,
}

impl fmt::Display for FresnelSamplingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fresnel TF sampling: lambda={:.3e} m, z={:.3e} m, grid={}x{} pitch={:.3e}x{:.3e} m, \
             padding req={:.2} used={:.2}, padded={}x{}, max phase step x={:.3} y={:.3} rad, \
             nyquist_ok={}, paraxial error vs ASM={:.3} rad, paraxial_ok={}",
            self.lambda,
            self.z,
            self.input_dims.0,
            self.input_dims.1,
            self.pitch.0,
            self.pitch.1,
            self.requested_padding,
            self.used_padding,
            self.padded_dims.0,
            self.padded_dims.1,
            self.max_phase_step_x,
            self.max_phase_step_y,
            self.nyquist_ok,
            self.max_phase_error_vs_asm,
            self.paraxial_ok,
        )
    }
}

/// Failures of the Fresnel transfer-function propagator.
///
/// Mirrors [`super::asm::PropagationError`]'s taxonomy with a
/// [`FresnelSamplingReport`] payload; unifying the two into one engine-level
/// error type is a later API decision.
#[derive(Clone, Debug, Error)]
pub enum FresnelError {
    /// Propagation distance was not finite. (Signed distances are valid:
    /// negative $z$ back-propagates, $z = 0$ is the identity.)
    #[error("propagation distance must be finite (got {0} m)")]
    InvalidDistance(f64),
    /// Wavelength was not positive and finite (m).
    #[error("wavelength must be positive and finite (got {0} m)")]
    InvalidWavelength(f64),
    /// The field grid had zero samples.
    #[error("field grid is empty")]
    EmptyGrid,
    /// The field contained a non-finite sample.
    #[error("field contains non-finite samples")]
    NonFiniteSample,
    /// The chirp transfer function is undersampled even after `strict`-mode
    /// densification reached `FresnelConfig::max_padding`. Remedies: raise
    /// `padding`/`max_padding`, coarsen the pitch, shorten `z`, or set
    /// `strict = false` deliberately.
    #[error("sampling criteria not met: {report}")]
    AliasingRisk {
        /// Diagnostics describing exactly which criterion failed.
        report: FresnelSamplingReport,
    },
}

/// Fresnel transfer-function value at frequency `(fx, fy)`:
/// $e^{ikz} \exp\!\big(-i\pi\lambda z (f_x^2 + f_y^2)\big)$.
fn transfer_value(fx: f64, fy: f64, lambda: f64, z: f64) -> Complex64 {
    let k = 2.0 * PI / lambda;
    let phase = k * z - PI * lambda * z * (fx * fx + fy * fy);
    (phase * Complex64::I).exp()
}

/// Worst adjacent-bin phase step of the separable chirp along one axis:
/// $\pi\lambda |z| \, \delta f (2 f_{\text{nyq}} - \delta f)$ evaluated at the
/// largest-magnitude bin pair (exact maximum, no sweep).
fn worst_axis_step(n_pad: usize, d: f64, lambda: f64, z: f64) -> f64 {
    let df = 1.0 / (n_pad as f64 * d);
    let f_nyq = (n_pad / 2) as f64 * df;
    let hi = (f_nyq / df).floor().max(0.0) * df;
    let lo = (hi - df).max(0.0);
    PI * lambda * z.abs() * (hi * hi - lo * lo)
}

/// Worst binomial-truncation error against the exact ASM root, evaluated at
/// the grid corner bin $(f_{\text{nyq},x}, f_{\text{nyq},y})$ where
/// $s = (\lambda f_x)^2 + (\lambda f_y)^2$ is largest:
/// $k|z|\big(1 - \tfrac{s}{2} - \sqrt{\max(0, 1 - s)}\big)$. The Nyquist
/// limit $1/(2\Delta)$ is padding-invariant.
fn worst_paraxial_error(dx: f64, dy: f64, lambda: f64, z: f64) -> f64 {
    let fx = 1.0 / (2.0 * dx); // Nyquist magnitude along x
    let fy = 1.0 / (2.0 * dy);
    let s = (lambda * fx) * (lambda * fx) + (lambda * fy) * (lambda * fy);
    let k = 2.0 * PI / lambda;
    k * z.abs() * (1.0 - 0.5 * s - (1.0 - s).max(0.0).sqrt())
}

/// Internal diagnostics at a concrete padding factor (shared by
/// [`sampling_report`] and [`propagate`]).
fn analyze(band: &BandField, z: f64, cfg: &FresnelConfig, padding: f64) -> FresnelSamplingReport {
    let field = band.field();
    let (nx, ny) = field.dims();
    let (dx, dy) = field.pitch();
    let lambda = band.lambda();
    let nx_pad = padded_len(nx, padding);
    let ny_pad = padded_len(ny, padding);

    let step_x = worst_axis_step(nx_pad, dx, lambda, z);
    let step_y = worst_axis_step(ny_pad, dy, lambda, z);
    let err = worst_paraxial_error(dx, dy, lambda, z);

    FresnelSamplingReport {
        lambda,
        z,
        input_dims: (nx, ny),
        pitch: (dx, dy),
        requested_padding: cfg.padding,
        used_padding: padding,
        padded_dims: (nx_pad, ny_pad),
        max_phase_step_x: step_x,
        max_phase_step_y: step_y,
        nyquist_ok: step_x < PI && step_y < PI,
        max_phase_error_vs_asm: err,
        paraxial_ok: err < 0.01,
    }
}

/// Sampling/validity diagnostics for a Fresnel TF propagation of `band`
/// over `z` under `cfg`, without performing the propagation. Evaluated at
/// the configured `padding`; mitigation is a [`propagate`] behavior.
pub fn sampling_report(band: &BandField, z: f64, cfg: &FresnelConfig) -> FresnelSamplingReport {
    analyze(band, z, cfg, cfg.padding)
}

/// Propagates a [`BandField`] a signed distance `z` (meters) with the Fresnel
/// transfer-function method; see module docs for the formula, validity, and
/// sampling criteria.
///
/// The output grid geometry (dims, pitch, wavelength) equals the input; the
/// computation runs on the zero-padded grid and the central physical window
/// is extracted.
///
/// # Errors
/// - [`FresnelError::InvalidDistance`] if `z` is not finite.
/// - [`FresnelError::InvalidWavelength`] if the band wavelength is not
///   positive and finite.
/// - [`FresnelError::EmptyGrid`] / [`FresnelError::NonFiniteSample`] for
///   degenerate inputs (normally prevented by [`Field`] validation).
/// - [`FresnelError::AliasingRisk`] when `cfg.strict`, the closed-form chirp
///   sampling check fails even after automatic densification up to
///   `cfg.max_padding`.
pub fn propagate(band: &BandField, z: f64, cfg: &FresnelConfig) -> Result<BandField, FresnelError> {
    if !z.is_finite() {
        return Err(FresnelError::InvalidDistance(z));
    }
    let lambda = band.lambda();
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(FresnelError::InvalidWavelength(lambda));
    }
    let field = band.field();
    let (nx, ny) = field.dims();
    if nx == 0 || ny == 0 {
        return Err(FresnelError::EmptyGrid);
    }
    // Defensive boundary re-validation. `Field::new`/`BandField::new` already
    // enforce these invariants at construction, but `Field::field_mut()` is
    // public — custom initializers or downstream code can scribble non-finite
    // samples into an otherwise valid band. The engine is the last place a
    // silent NaN could be amplified, so it re-checks (O(N), cheap) rather
    // than trusting the caller.
    if field.samples().iter().any(|s| !s.is_finite()) {
        return Err(FresnelError::NonFiniteSample);
    }
    if z == 0.0 {
        return Ok(band.clone());
    }

    let mut report = analyze(band, z, cfg, cfg.padding);
    if cfg.strict && !report.nyquist_ok {
        let mut padding = cfg.padding;
        while !report.nyquist_ok && padding < cfg.max_padding {
            padding = (padding * 2.0).min(cfg.max_padding);
            report = analyze(band, z, cfg, padding);
        }
        if !report.nyquist_ok {
            return Err(FresnelError::AliasingRisk { report });
        }
    }

    let (nx_pad, ny_pad) = report.padded_dims;
    let (dx, dy) = field.pitch();
    let ox = (nx_pad - nx) / 2;
    let oy = (ny_pad - ny) / 2;

    // Embed the centered input window into the centered padded grid,
    // surroundings zeroed.
    //
    // Why padding at all: the DFT treats the sampled window as ONE PERIOD of
    // a periodic signal, so transform-based propagation is a *circular*
    // convolution, while free space performs a *linear* one. A spreading
    // field would otherwise wrap past the right edge and re-enter from the
    // left — unphysical replica contamination. The zero guard band separates
    // adjacent replicas; while the spread field stays inside the guard band,
    // circular convolution equals linear convolution in the central region.
    let mut buf = vec![Complex64::ZERO; nx_pad * ny_pad];
    for iy in 0..ny {
        for ix in 0..nx {
            buf[(iy + oy) * nx_pad + (ix + ox)] = field.sample(ix, iy);
        }
    }

    fft2(&mut buf, nx_pad, ny_pad);

    // Pointwise multiplication by H_F. By the convolution theorem this IS
    // the spatial convolution with the impulse response h(x, y; z) of
    // textbook Eq. 2.24 — frequency-domain multiplication is the cheap form
    // of it (the sampled-kernel variant arrives with the convolution-form
    // propagator). Each spectral bin acquires the quadratic chirp phase
    // exp(-i pi lambda z f^2) plus the global e^{ikz}; |H_F| = 1, so a fully
    // captured spectrum conserves power exactly.
    let fx_axis = fftfreq(nx_pad, dx);
    let fy_axis = fftfreq(ny_pad, dy);
    for (iy, &fy) in fy_axis.iter().enumerate() {
        let row = &mut buf[iy * nx_pad..(iy + 1) * nx_pad];
        for (ix, h) in row.iter_mut().enumerate() {
            *h *= transfer_value(fx_axis[ix], fy, lambda, z);
        }
    }

    ifft2(&mut buf, nx_pad, ny_pad);

    // Extract the central physical window: the output plane is sampled on
    // the SAME detector grid as the input (identical pitch and extent), so
    // the guard band is a computational artifact and is discarded — dims and
    // pitch are preserved across `propagate`. Consequence: energy that
    // spread past the window is genuinely lost (crop); size the input grid
    // so the plane of interest contains what you need.
    let mut out = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        for ix in 0..nx {
            out.push(buf[(iy + oy) * nx_pad + (ix + ox)]);
        }
    }
    let field_out = Field::new(out, nx, ny, dx, dy).map_err(|_| FresnelError::NonFiniteSample)?;
    BandField::new(lambda, field_out).map_err(|_| FresnelError::InvalidWavelength(lambda))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAMBDA: f64 = 500e-9;

    fn band_uniform(a: f64, nx: usize, ny: usize, dx: f64, dy: f64) -> BandField {
        BandField::uniform(a, 0.0, nx, ny, dx, dy, LAMBDA).unwrap()
    }

    #[test]
    fn defaults_mirror_asm_policy() {
        let cfg = FresnelConfig::default();
        assert_eq!(cfg.padding, 2.0);
        assert!(cfg.strict);
        assert_eq!(cfg.max_padding, 8.0);
    }

    #[test]
    fn transfer_function_matches_asm_at_low_frequencies() {
        // H_F = e^{ikz} e^{-ikz s/2} vs H_ASM = e^{ikz sqrt(1-s)}; the
        // truncation error scales as k z s^2 / 8.
        let z = 0.1;
        let k = 2.0 * PI / LAMBDA;
        let s = 1e-4f64; // (lambda f)^2
        let f = s.sqrt() / LAMBDA;
        let h_f = transfer_value(f, 0.0, LAMBDA, z);
        let h_asm = (k * z * (1.0 - s).sqrt() * Complex64::I).exp();
        let expected_err = k * z * s * s / 8.0;
        let actual = (h_f - h_asm).norm();
        assert!(
            actual < 2.0 * expected_err + 1e-12,
            "paraxial deviation {actual} not O(kz s^2/8) = {expected_err}"
        );
        // DC bin: exact global phase e^{ikz}.
        assert!(
            (transfer_value(0.0, 0.0, LAMBDA, z) - (k * z * Complex64::I).exp()).norm() < 1e-12
        );
    }

    #[test]
    fn closed_form_step_is_the_exact_sweep_maximum() {
        // Brute-force sweep over all adjacent bin pairs along x for a small
        // grid must match the closed-form worst step (separable quadratic).
        let (nx_pad, ny_pad, d) = (12usize, 8usize, 2e-6f64);
        let z = 0.037;
        let fx = fftfreq(nx_pad, d);
        let fy = fftfreq(ny_pad, d);
        let phase = |ix: usize, iy: usize| -PI * LAMBDA * z * (fx[ix] * fx[ix] + fy[iy] * fy[iy]);
        let mut sweep = 0.0_f64;
        for iy in 0..ny_pad {
            for ix in 0..nx_pad {
                let a = phase(ix, iy);
                let b = phase((ix + 1) % nx_pad, iy);
                sweep = sweep.max((a - b).abs());
            }
        }
        // The Nyquist-straddling pair carries the raw quadratic difference
        // (no wrapping) — identical to the closed form by separability.
        let closed = worst_axis_step(nx_pad, d, LAMBDA, z);
        assert!(
            (sweep - closed).abs() < 1e-9,
            "sweep {sweep} vs closed-form {closed}"
        );
    }

    #[test]
    fn z_zero_is_identity() {
        let band = band_uniform(1.5, 16, 16, 20e-6, 20e-6);
        let out = propagate(&band, 0.0, &FresnelConfig::default()).unwrap();
        assert_eq!(out, band);
    }

    #[test]
    fn plane_wave_bins_pick_up_exact_quadratic_phase() {
        // Unpadded periodic grid: a tilt at bin m has fx = m / (nx * dx),
        // so the output must be input * exp(i(kz - pi lambda z fx^2)).
        let (nx, ny, d) = (32usize, 32usize, 20e-6f64);
        let z = 1e-3;
        let m = 2.0f64;
        let alpha = m * LAMBDA / (nx as f64 * d); // direction cosine of an exact bin
        let band = BandField::plane_wave(alpha.asin(), 0.0, 1.0, nx, ny, d, d, LAMBDA).unwrap();
        let cfg = FresnelConfig {
            padding: 1.0,
            strict: false,
            max_padding: 1.0,
        };
        let out = propagate(&band, z, &cfg).unwrap();
        let k = 2.0 * PI / LAMBDA;
        let fx = m / (nx as f64 * d);
        let expected = ((k * z - PI * LAMBDA * z * fx * fx) * Complex64::I).exp();
        for iy in [0usize, 16, 31] {
            let s = out.field().sample(10, iy) / band.field().sample(10, iy);
            assert!((s - expected).norm() < 1e-9, "bin phase wrong at row {iy}");
        }
    }

    #[test]
    fn round_trip_on_periodic_grid_is_exact() {
        // |H_F| = 1 everywhere (no mask): +z then -z on the unpadded grid is
        // the identity to roundoff.
        let band = band_uniform(1.0, 16, 16, 20e-6, 20e-6);
        let cfg = FresnelConfig {
            padding: 1.0,
            strict: false,
            max_padding: 1.0,
        };
        let fwd = propagate(&band, 2e-3, &cfg).unwrap();
        let rt = propagate(&fwd, -2e-3, &cfg).unwrap();
        let err = rt
            .field()
            .samples()
            .iter()
            .zip(band.field().samples())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(err < 1e-9);
    }

    #[test]
    fn strict_mode_flags_undersampled_chirp_and_mitigates() {
        // Fine pitch, unpadded: worst step pi*lambda*z*df*(2 f_nyq - df) with
        // n=32, d=1um: step = pi * 5e-7 * z * 31250 * 968750 ~ pi * 1.51e4 * z.
        let band = band_uniform(1.0, 32, 32, 1e-6, 1e-6);
        let z = 3e-4; // step ~ 14 rad > pi at padding 1; ~1.8 rad < pi at padding 8
        let cfg_risk = FresnelConfig {
            padding: 1.0,
            strict: true,
            max_padding: 1.0,
        };
        let report = sampling_report(&band, z, &cfg_risk);
        assert!(!report.nyquist_ok);
        assert!(report.max_phase_step_x > PI);
        assert!(matches!(
            propagate(&band, z, &cfg_risk),
            Err(FresnelError::AliasingRisk { .. })
        ));

        // Non-strict proceeds with the flagged report.
        let cfg_loose = FresnelConfig {
            strict: false,
            ..cfg_risk
        };
        assert!(propagate(&band, z, &cfg_loose).is_ok());

        // Strict with headroom: doubling 1 -> 2 -> 4 -> 8 restores adequacy.
        let cfg_fix = FresnelConfig {
            max_padding: 8.0,
            ..cfg_risk
        };
        assert!(!sampling_report(&band, z, &cfg_fix).nyquist_ok);
        let out = propagate(&band, z, &cfg_fix).expect("mitigation must succeed");
        assert_eq!(out.field().dims(), band.field().dims());
        let at8 = sampling_report(
            &band,
            z,
            &FresnelConfig {
                padding: 8.0,
                ..cfg_fix
            },
        );
        assert!(at8.nyquist_ok && at8.max_phase_step_x < PI);
    }

    #[test]
    fn paraxial_flag_tracks_grid_angular_extent() {
        // Coarse pitch (1 mm): Nyquist is at lambda*f = 2.5e-4 -> paraxial.
        let coarse = band_uniform(1.0, 32, 32, 1e-3, 1e-3);
        let cfg = FresnelConfig::default();
        let rep = sampling_report(&coarse, 0.1, &cfg);
        assert!(rep.paraxial_ok);
        assert!(rep.max_phase_error_vs_asm < 0.01);

        // Fine pitch (1 um) at large z: truncation error k|z|(1 - s/2 - sqrt(1-s))
        // at s = 0.125 grows past the 0.01 rad threshold.
        let fine = band_uniform(1.0, 32, 32, 1e-6, 1e-6);
        let rep = sampling_report(&fine, 0.05, &cfg);
        assert!(!rep.paraxial_ok);
        assert!(rep.max_phase_error_vs_asm > 0.01);
    }

    #[test]
    fn invalid_inputs_rejected() {
        let band = band_uniform(1.0, 8, 8, 20e-6, 20e-6);
        assert!(matches!(
            propagate(&band, f64::INFINITY, &FresnelConfig::default()),
            Err(FresnelError::InvalidDistance(_))
        ));
    }
}
