//! Angular Spectrum Method (ASM) propagation.
//!
//! Propagates a [`BandField`] a signed distance $z$ (meters) along the optical
//! axis by decomposing the field into plane-wave components, advancing each by
//! its own phase, and recomposing. Per the textbook (Eqs. 3.5–3.13), a plane
//! wave with direction cosines $\alpha = \lambda f_x$, $\beta = \lambda f_y$,
//! $\gamma = \sqrt{1 - \alpha^2 - \beta^2}$ accumulates
//! $\exp\!\big(i \frac{2\pi}{\lambda} \gamma z\big)$ over distance $z$, so the
//! field at plane $z$ is
//!
//! $$U(x, y, z) = \mathcal{F}^{-2}\!\Big\{\, H(f_x, f_y)\cdot
//!   \mathcal{F}^{2}\!\{U(x, y, 0)\}\,\Big\}$$
//!
//! with the transfer function
//!
//! $$H(f_x, f_y) =
//!   \begin{cases}
//!     e^{\,i k z \sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2}}
//!       & (\lambda f_x)^2 + (\lambda f_y)^2 \le 1 \\\
//!     0 & \text{(evanescent, otherwise)}
//!   \end{cases},
//!   \qquad k = \frac{2\pi}{\lambda}.$$
//!
//! # Boundary and aliasing control (Phase 1 policy)
//!
//! The DFT implements *circular* convolution, so wraparound of the periodic
//! replicas contaminates the physical window. Phase 1 controls this with
//! **zero-padding** (default factor 2 in each axis) plus the **band-limited
//! transfer function** of Matsushima & Shimobaba (Opt. Express 17, 19731, 2009),
//! which additionally suppresses frequency components whose associated rays
//! could shift across more than the padded grid width over $z$:
//!
//! $$|f_x| \le f_{\lim,x} = \frac{1}{\lambda \sqrt{(2z / L_x)^2 + 1}},
//!   \qquad L_x = n_{x,\text{pad}}\, \Delta x$$
//!
//! (and identically along $y$ with $L_y = n_{y,\text{pad}} \Delta y$). Note the
//! band limit uses the **padded** grid width $L$, not the original window.
//!
//! # Sampling verification
//!
//! [`sampling_report`] exposes the automatic check mandated by the project
//! guidelines: a conservative **closed-form** upper bound on the phase step of
//! $H$ between adjacent frequency bins along each axis, evaluated at the
//! worst-case candidate bin pairs (band edges and the band-limit corner) — no
//! grid sweep. Nyquist-adequate sampling of the *transfer function itself*
//! requires that step $< \pi$. In `strict` mode a violation is **mitigated
//! automatically** by densifying the padded grid (smaller $\Delta f$), doubling
//! `padding` up to `max_padding`; only if the cap cannot satisfy the criterion
//! does [`propagate`] fail with [`PropagationError::AliasingRisk`].
//!
//! # Validity regime
//!
//! Paraxial and non-paraxial propagation is exact for the scalar Helmholtz
//! model *within the sampled band*: evanescent components ($\lambda|f| > 1$)
//! are dropped (they decay within ~$\lambda$ of the plane), and the sampling
//! report guards against transfer-function aliasing. Very large $z$ with fine
//! pitch still concentrates energy near low frequencies; no Fresnel
//! approximation is made. Negative $z$ back-propagates; $z = 0$ is the
//! identity.

use std::f64::consts::PI;
use std::fmt;

use num_complex::Complex64;
use thiserror::Error;

use super::fft::{fft2, fftfreq, ifft2};
use super::padded_len;
use crate::field::{BandField, Field};

/// Tuning knobs for [`propagate`].
///
/// Structurally extensible for future boundary-condition strategies (Phase 1
/// implements only zero-padding + band-limiting, per the approved decision);
/// new strategies will arrive as additional fields/variants without changing
/// existing behavior.
#[derive(Clone, Copy, Debug)]
pub struct AsmConfig {
    /// Linear zero-padding factor applied to each axis before transform;
    /// values `< 1.0` are treated as `1.0` (no padding). Typical: `2.0`.
    pub padding: f64,
    /// Apply the Matsushima band-limited mask to the transfer function.
    /// Recommended `true` whenever propagation distance is significant.
    pub band_limited: bool,
    /// When `true`, [`propagate`] mitigates an undersampled transfer function
    /// by densifying the padded grid (doubling `padding` up to `max_padding`),
    /// and returns [`PropagationError::AliasingRisk`] only if the cap cannot
    /// restore Nyquist adequacy. When `false`, the configured padding is used
    /// as-is and risk is only visible via [`sampling_report`].
    pub strict: bool,
    /// Upper bound on automatic densification in `strict` mode
    /// (`padding` never exceeds this). `1.0` disables mitigation.
    pub max_padding: f64,
}

impl Default for AsmConfig {
    /// Phase-1 defaults: `padding = 2.0`, `band_limited = true`,
    /// `strict = true`, `max_padding = 8.0`.
    fn default() -> Self {
        Self {
            padding: 2.0,
            band_limited: true,
            strict: true,
            max_padding: 8.0,
        }
    }
}

/// Spatial-sampling diagnostics for one ASM propagation step.
///
/// Produced by [`sampling_report`] and embedded in
/// [`PropagationError::AliasingRisk`]. All quantities SI: meters, cycles per
/// meter, radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingReport {
    /// Operating wavelength (m).
    pub lambda: f64,
    /// Signed propagation distance (m).
    pub z: f64,
    /// Input grid sample counts `(nx, ny)`.
    pub input_dims: (usize, usize),
    /// Input sample pitch `(dx, dy)` (m).
    pub pitch: (f64, f64),
    /// Zero-padding factor requested via [`AsmConfig::padding`].
    pub requested_padding: f64,
    /// Zero-padding factor actually used (equals `requested_padding` unless
    /// `strict` mode densified the grid).
    pub used_padding: f64,
    /// Zero-padded grid sample counts `(nx_pad, ny_pad)` at `used_padding`.
    pub padded_dims: (usize, usize),
    /// Effective per-axis frequency mask cutoff $f_{\lim}$ (cycles/m):
    /// Matsushima band limit when `band_limited`, else the evanescent
    /// cutoff $1/\lambda$.
    pub band_limit_fx: f64,
    /// See [`SamplingReport::band_limit_fx`] (y axis).
    pub band_limit_fy: f64,
    /// Conservative closed-form upper bound on the maximum adjacent-bin phase
    /// step of $H$ along $f_x$ (radians), evaluated at worst-case candidate
    /// bin pairs (band edge row and band-limit corner).
    pub max_phase_step_x: f64,
    /// As [`SamplingReport::max_phase_step_x`] along $f_y$.
    pub max_phase_step_y: f64,
    /// `true` when both max phase steps are below $\pi$: the transfer function
    /// itself is Nyquist-sampled on the padded grid.
    pub nyquist_ok: bool,
}

impl fmt::Display for SamplingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ASM sampling: lambda={:.3e} m, z={:.3e} m, grid={}x{} pitch={:.3e}x{:.3e} m, \
             padding req={:.2} used={:.2}, padded={}x{}, band limit fx={:.3e} fy={:.3e} cyc/m, \
             max phase step x={:.3} y={:.3} rad, nyquist_ok={}",
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
            self.band_limit_fx,
            self.band_limit_fy,
            self.max_phase_step_x,
            self.max_phase_step_y,
            self.nyquist_ok,
        )
    }
}

/// Failures of the ASM propagator.
#[derive(Clone, Debug, Error)]
pub enum PropagationError {
    /// Propagation distance was not finite. (Signed distances are valid:
    /// negative $z$ back-propagates, $z = 0$ is the identity.)
    #[error("propagation distance must be finite (got {0} m)")]
    InvalidDistance(f64),
    /// Wavelength was not positive and finite (m); [`BandField`] construction
    /// already rejects this, so it can only arise from a future unvalidated
    /// path.
    #[error("wavelength must be positive and finite (got {0} m)")]
    InvalidWavelength(f64),
    /// The field grid had zero samples.
    #[error("field grid is empty")]
    EmptyGrid,
    /// The field contained a non-finite sample.
    #[error("field contains non-finite samples")]
    NonFiniteSample,
    /// The transfer function is undersampled even after `strict`-mode
    /// automatic densification reached `AsmConfig::max_padding`. Inspect the
    /// embedded report; remedies: raise `padding`/`max_padding`, coarsen the
    /// pitch, shorten `z`, enable `band_limited`, or set `strict = false`
    /// deliberately.
    #[error("sampling criteria not met: {report}")]
    AliasingRisk {
        /// Diagnostics describing exactly which criterion failed.
        report: SamplingReport,
    },
}

/// Effective per-axis frequency cutoff (cycles/m) used by the mask.
///
/// Band-limited ASM (Matsushima & Shimobaba 2009):
/// $f_{\lim} = 1 / (\lambda \sqrt{(2 z / L)^2 + 1})$ with $L$ the **padded**
/// axis width; otherwise the evanescent cutoff $1/\lambda$ (the circle
/// $(\lambda f_x)^2 + (\lambda f_y)^2 \le 1$ is still applied in both cases).
fn axis_band_limit(lambda: f64, z: f64, grid_width: f64, band_limited: bool) -> f64 {
    if band_limited {
        1.0 / (lambda * ((2.0 * z / grid_width).powi(2) + 1.0).sqrt())
    } else {
        1.0 / lambda
    }
}

// `padded_len` lives in `super` so all transfer-function engines share the
// padding policy.

/// Transfer function value at frequency `(fx, fy)` for wavelength `lambda`,
/// distance `z`; returns `None` for masked (zero) components.
fn transfer_value(fx: f64, fy: f64, lambda: f64, z: f64) -> Option<Complex64> {
    let s = 1.0 - (lambda * fx).powi(2) - (lambda * fy).powi(2);
    if s < 0.0 {
        return None; // evanescent
    }
    let k = 2.0 * PI / lambda;
    Some((k * z * s.sqrt() * Complex64::I).exp())
}

/// Internal evaluation context for one ASM step at a concrete padding factor.
struct AsmPlan {
    lambda: f64,
    z: f64,
    nx_pad: usize,
    ny_pad: usize,
    dx: f64,
    dy: f64,
    lim_x: f64,
    lim_y: f64,
    band_limited: bool,
}

/// Conservative closed-form upper bound on the maximum adjacent-bin phase step
/// of $H$ along each axis for the padded grid described by `plan`.
///
/// Evaluated at the worst-case candidate bin pairs only — the largest
/// unmasked-then-next bin along each axis at $f_y = 0$ (or $f_x = 0$), and the
/// same at the band-limit edge row/column when band-limiting is active (the
/// mask corner carries the largest radius, hence the steepest phase
/// gradient). This replaces a full-grid sweep: a handful of `sqrt` calls.
///
/// Returns `(step_x, step_y)` in radians.
fn worst_phase_steps(plan: &AsmPlan) -> (f64, f64) {
    let AsmPlan {
        lambda,
        z,
        nx_pad,
        ny_pad,
        dx,
        dy,
        lim_x,
        lim_y,
        band_limited,
    } = *plan;
    let kz = (2.0 * PI / lambda) * z.abs();

    // Largest phase step between adjacent bins along x at fixed fy.
    let step_along_x = |fy: f64| -> f64 {
        let df = 1.0 / (nx_pad as f64 * dx);
        let nyq = ((nx_pad / 2) as f64) * df; // largest |bin| magnitude
        let f_circle = ((1.0 / lambda).powi(2) - fy.powi(2)).max(0.0).sqrt();
        let mut f_edge = f_circle.min(nyq);
        if band_limited {
            if fy.abs() > lim_y {
                return 0.0; // whole row masked by the band limit
            }
            f_edge = f_edge.min(lim_x);
        }
        let hi = (f_edge / df).floor().max(0.0) * df;
        let lo = (hi - df).max(0.0);
        let root = |f: f64| (1.0 - lambda * lambda * (f * f + fy * fy)).max(0.0).sqrt();
        kz * (root(hi) - root(lo)).abs()
    };

    // Symmetric counterpart along y at fixed fx.
    let step_along_y = |fx: f64| -> f64 {
        let df = 1.0 / (ny_pad as f64 * dy);
        let nyq = ((ny_pad / 2) as f64) * df;
        let f_circle = ((1.0 / lambda).powi(2) - fx.powi(2)).max(0.0).sqrt();
        let mut f_edge = f_circle.min(nyq);
        if band_limited {
            if fx.abs() > lim_x {
                return 0.0;
            }
            f_edge = f_edge.min(lim_y);
        }
        let hi = (f_edge / df).floor().max(0.0) * df;
        let lo = (hi - df).max(0.0);
        let root = |f: f64| (1.0 - lambda * lambda * (f * f + fx * fx)).max(0.0).sqrt();
        kz * (root(hi) - root(lo)).abs()
    };

    // Candidate rows/cols: the axis (fy = 0 / fx = 0, largest circle radius)
    // and, when band-limited, the mask edge (corner of the rectangle).
    let mut step_x = step_along_x(0.0);
    let mut step_y = step_along_y(0.0);
    if band_limited {
        step_x = step_x.max(step_along_x(lim_y));
        step_y = step_y.max(step_along_y(lim_x));
    }
    (step_x, step_y)
}

/// Core diagnostics shared by [`sampling_report`] and [`propagate`], evaluated
/// for an explicit `padding` factor (which may exceed `cfg.padding` when
/// `strict`-mode mitigation densifies the grid).
fn analyze(band: &BandField, z: f64, cfg: &AsmConfig, padding: f64) -> SamplingReport {
    let field = band.field();
    let (nx, ny) = field.dims();
    let (dx, dy) = field.pitch();
    let lambda = band.lambda();
    let nx_pad = padded_len(nx, padding);
    let ny_pad = padded_len(ny, padding);

    let lim_x = axis_band_limit(lambda, z, nx_pad as f64 * dx, cfg.band_limited);
    let lim_y = axis_band_limit(lambda, z, ny_pad as f64 * dy, cfg.band_limited);
    let plan = AsmPlan {
        lambda,
        z,
        nx_pad,
        ny_pad,
        dx,
        dy,
        lim_x,
        lim_y,
        band_limited: cfg.band_limited,
    };
    let (step_x, step_y) = worst_phase_steps(&plan);

    SamplingReport {
        lambda,
        z,
        input_dims: (nx, ny),
        pitch: (dx, dy),
        requested_padding: cfg.padding,
        used_padding: padding,
        padded_dims: (nx_pad, ny_pad),
        band_limit_fx: lim_x,
        band_limit_fy: lim_y,
        max_phase_step_x: step_x,
        max_phase_step_y: step_y,
        nyquist_ok: step_x < PI && step_y < PI,
    }
}

/// Spatial-sampling diagnostics for propagating `band` a distance `z` under
/// `cfg`, without performing the propagation. Evaluated at the configured
/// `padding` (mitigation is a [`propagate`] behavior); see the module docs for
/// the criteria it exposes.
pub fn sampling_report(band: &BandField, z: f64, cfg: &AsmConfig) -> SamplingReport {
    analyze(band, z, cfg, cfg.padding)
}

/// Propagates a [`BandField`] a signed distance `z` (meters) with the Angular
/// Spectrum Method; see module docs for the transfer function, zero-padding,
/// band-limiting, and sampling-mitigation formulas.
///
/// The output grid geometry (dims, pitch) equals the input; the computation
/// runs on the zero-padded grid and the central physical window is extracted.
///
/// # Errors
/// - [`PropagationError::InvalidDistance`] if `z` is not finite.
/// - [`PropagationError::InvalidWavelength`] if the band wavelength is not
///   positive and finite.
/// - [`PropagationError::EmptyGrid`] / [`PropagationError::NonFiniteSample`]
///   for degenerate inputs (normally prevented by [`Field`] validation).
/// - [`PropagationError::AliasingRisk`] when `cfg.strict`, the closed-form
///   transfer-function sampling check fails even after automatic densification
///   of the padded grid up to `cfg.max_padding` (see [`sampling_report`]).
pub fn propagate(band: &BandField, z: f64, cfg: &AsmConfig) -> Result<BandField, PropagationError> {
    if !z.is_finite() {
        return Err(PropagationError::InvalidDistance(z));
    }
    let lambda = band.lambda();
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(PropagationError::InvalidWavelength(lambda));
    }
    let field = band.field();
    let (nx, ny) = field.dims();
    if nx == 0 || ny == 0 {
        return Err(PropagationError::EmptyGrid);
    }
    // Defensive boundary re-validation: constructors enforce the invariants,
    // but `field_mut()` is public, so the engine re-checks rather than
    // trusting caller-side mutation (see `fresnel::propagate` for the full
    // rationale).
    if field.samples().iter().any(|s| !s.is_finite()) {
        return Err(PropagationError::NonFiniteSample);
    }
    if z == 0.0 {
        return Ok(band.clone());
    }

    let mut report = analyze(band, z, cfg, cfg.padding);
    if cfg.strict && !report.nyquist_ok {
        // Mitigate by densifying the padded grid (smaller delta-f) before
        // considering failure: doubling padding up to cfg.max_padding.
        let mut padding = cfg.padding;
        while !report.nyquist_ok && padding < cfg.max_padding {
            padding = (padding * 2.0).min(cfg.max_padding);
            report = analyze(band, z, cfg, padding);
        }
        if !report.nyquist_ok {
            return Err(PropagationError::AliasingRisk { report });
        }
    }

    let (nx_pad, ny_pad) = report.padded_dims;
    let (dx, dy) = field.pitch();
    let ox = (nx_pad - nx) / 2;
    let oy = (ny_pad - ny) / 2;

    // Embed the centered input window into the centered padded grid,
    // surroundings zeroed. The DFT treats the window as one period of a
    // periodic signal (circular convolution), so the zero guard band keeps
    // the spreading field's replicas from folding back into the physical
    // window; while the field stays inside the guard band the result equals
    // the linear (physical) convolution. See `fresnel::propagate` docs for
    // the same rationale in expanded form.
    let mut buf = vec![Complex64::ZERO; nx_pad * ny_pad];
    for iy in 0..ny {
        for ix in 0..nx {
            buf[(iy + oy) * nx_pad + (ix + ox)] = field.sample(ix, iy);
        }
    }

    fft2(&mut buf, nx_pad, ny_pad);

    // Pointwise multiplication by H: by the convolution theorem this is the
    // spatial convolution with the exact plane-wave-composition kernel; the
    // evanescent region and the Matsushima band-limit mask zero those bins.
    let fx_axis = fftfreq(nx_pad, dx);
    let fy_axis = fftfreq(ny_pad, dy);
    for (iy, &fy) in fy_axis.iter().enumerate() {
        let row = &mut buf[iy * nx_pad..(iy + 1) * nx_pad];
        for (ix, h) in row.iter_mut().enumerate() {
            let fx = fx_axis[ix];
            let masked = cfg.band_limited
                && (fx.abs() > report.band_limit_fx || fy.abs() > report.band_limit_fy);
            *h = if masked {
                Complex64::ZERO
            } else {
                match transfer_value(fx, fy, lambda, z) {
                    Some(t) => *h * t,
                    None => Complex64::ZERO,
                }
            };
        }
    }

    ifft2(&mut buf, nx_pad, ny_pad);

    // Extract the central physical window: same detector grid in, same grid
    // out (dims/pitch preserved); the guard band is a computational artifact.
    // Energy spread beyond the window is genuinely lost (crop).
    let mut out = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        for ix in 0..nx {
            out.push(buf[(iy + oy) * nx_pad + (ix + ox)]);
        }
    }
    let field_out =
        Field::new(out, nx, ny, dx, dy).map_err(|_| PropagationError::NonFiniteSample)?;
    BandField::new(lambda, field_out).map_err(|_| PropagationError::InvalidWavelength(lambda))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band_uniform(a: f64, nx: usize, ny: usize, dx: f64, dy: f64) -> BandField {
        BandField::uniform(a, 0.0, nx, ny, dx, dy, 500e-9).unwrap()
    }

    #[test]
    fn defaults_match_approved_phase1_policy() {
        let cfg = AsmConfig::default();
        assert_eq!(cfg.padding, 2.0);
        assert!(cfg.band_limited);
        assert!(cfg.strict);
        assert_eq!(cfg.max_padding, 8.0);
    }

    #[test]
    fn padding_rounds_up_and_clamps() {
        assert_eq!(padded_len(100, 2.0), 200);
        assert_eq!(padded_len(100, 1.5), 150);
        assert_eq!(padded_len(100, 1.0), 100);
        assert_eq!(padded_len(100, 0.5), 100);
    }

    #[test]
    fn band_limit_uses_padded_width_and_shrinks_with_z() {
        let lambda = 500e-9;
        let l_padded = 200.0 * 10e-6; // 2 mm
        let f0 = axis_band_limit(lambda, 0.0, l_padded, true);
        assert!((f0 - 1.0 / lambda).abs() < 1e-6);
        let f1 = axis_band_limit(lambda, 0.05, l_padded, true);
        assert!(f1 < f0);
        // Non-band-limited reports the evanescent cutoff.
        assert!((axis_band_limit(lambda, 1.0, l_padded, false) - 1.0 / lambda).abs() < 1e-9);
    }

    #[test]
    fn z_zero_is_identity() {
        let band = band_uniform(1.5, 16, 16, 20e-6, 20e-6);
        let out = propagate(&band, 0.0, &AsmConfig::default()).unwrap();
        assert_eq!(out, band);
    }

    #[test]
    fn on_axis_plane_wave_picks_up_global_phase_kz() {
        // A uniform field on the *unpadded* grid is exactly the on-axis plane
        // wave: every spectral bin is zero except DC, so ASM output must be
        // input * exp(i k z). (Padding would truncate the wave and make it
        // diffract at the window edges, so the exact invariant uses padding
        // 1.0; z is kept below the transfer-function Nyquist limit.)
        let z = 1e-3;
        let lambda = 500e-9;
        let band = band_uniform(1.0, 16, 16, 20e-6, 20e-6);
        let cfg = AsmConfig {
            padding: 1.0,
            ..AsmConfig::default()
        };
        let out = propagate(&band, z, &cfg).unwrap();
        let k = 2.0 * PI / lambda;
        let expected = (k * z * Complex64::I).exp();
        let s = out.field().sample(3, 9);
        assert!((s - expected).norm() < 1e-9);
    }

    #[test]
    fn back_propagation_inverts_forward() {
        // Unpadded periodic grid: H(f, z) * H(f, -z) = 1 on unmasked bins,
        // so +z then -z is the exact identity.
        let z = 1e-3;
        let band = band_uniform(1.0, 16, 16, 20e-6, 20e-6);
        let cfg = AsmConfig {
            padding: 1.0,
            ..AsmConfig::default()
        };
        let fwd = propagate(&band, z, &cfg).unwrap();
        let rt = propagate(&fwd, -z, &cfg).unwrap();
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
    fn invalid_inputs_rejected() {
        let band = band_uniform(1.0, 8, 8, 20e-6, 20e-6);
        assert!(matches!(
            propagate(&band, f64::NAN, &AsmConfig::default()),
            Err(PropagationError::InvalidDistance(_))
        ));
    }

    #[test]
    fn strict_mode_flags_undersampled_transfer_function() {
        // Band-limiting IS the aliasing fix, so the risk case is a fine pitch,
        // large z, unpadded grid with band limiting disabled: the phase of H
        // varies by >> pi between adjacent bins near the grid edge, and even
        // automatic densification to max_padding = 1.0 (mitigation disabled)
        // cannot recover.
        let band = band_uniform(1.0, 32, 32, 1e-6, 1e-6);
        let z = 0.5;
        let cfg = AsmConfig {
            padding: 1.0,
            band_limited: false,
            strict: true,
            max_padding: 1.0,
        };
        let report = sampling_report(&band, z, &cfg);
        assert!(!report.nyquist_ok);
        assert!(report.max_phase_step_x > PI);
        assert!(matches!(
            propagate(&band, z, &cfg),
            Err(PropagationError::AliasingRisk { .. })
        ));
        // Non-strict returns the (aliased) result with the report flags set.
        let lenient = AsmConfig {
            strict: false,
            ..cfg
        };
        assert!(propagate(&band, z, &lenient).is_ok());
    }

    #[test]
    fn strict_mode_mitigates_by_densifying_padded_grid() {
        // Same undersampled regime, but far enough below the cap that doubling
        // padding (1 -> 2 -> 4 -> 8) restores transfer-function Nyquist:
        // strict mode must SUCCEED via mitigation instead of erroring.
        let band = band_uniform(1.0, 32, 32, 1e-6, 1e-6);
        let z = 4e-4;
        let cfg = AsmConfig {
            padding: 1.0,
            band_limited: false,
            strict: true,
            max_padding: 8.0,
        };
        assert!(!sampling_report(&band, z, &cfg).nyquist_ok);
        let out = propagate(&band, z, &cfg).expect("mitigated by densification");
        assert!(out.field().samples().iter().all(|s| s.is_finite()));
    }
}
