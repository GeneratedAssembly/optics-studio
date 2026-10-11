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
//! Requiring $\Delta\phi < \pi$ is algebraically $n_{\text{pad}} \Delta x^2 >
//! \lambda z$ — the *negation* of the textbook Sec. 2.2 rule (2)
//! $N > D^2/(Z\lambda)$ evaluated at the padded width $D = n_{\text{pad}}
//! \Delta x$: the transfer-function form is the short-distance / coarse-pitch
//! side of the complementary pair (see below). `strict` mode (default)
//! mitigates violations by doubling `padding` up to `max_padding` before
//! raising [`FresnelError::AliasingRisk`], mirroring the ASM policy.
//!
//! # Convolution form ([`propagate_convolution`])
//!
//! Parallel planes make free-space propagation linear and shift-invariant,
//! so the field is the spatial convolution of the input with the impulse
//! response (Green's function) — textbook Sec. 2.2.2 and Ch. 3, with the
//! Eq. 2.24 kernel reduced to its paraxial form:
//!
//! $$h(x, y; z) = \frac{e^{ikz}}{i\lambda z}\,
//!   \exp\!\Big(i\pi\frac{x^2 + y^2}{\lambda z}\Big), \qquad
//!   U_z = \iint U_0(\xi, \eta)\, h(x - \xi, y - \eta)\, d\xi\, d\eta .$$
//!
//! The discrete approximation samples $h$ on the padded grid in DFT order
//! (kernel origin at bin $(0,0)$, required for circular convolution) and
//! evaluates $U_z \approx \mathcal{F}^{-1}\{\mathcal{F}\{U_0\}\cdot
//! \mathcal{F}\{h\}\}\,\Delta x\,\Delta y$ — exact use of the convolution
//! theorem, which is why the NumPy FFT scaling (Phase-2 decision) matters.
//!
//! **The two forms have reciprocal sampling conditions.** $H_F$'s phase step
//! grows with $z$ (guarded above), while $h$'s chirp phase step grows with
//! the *grid width*: worst adjacent-sample step
//! $\pi(n_{\text{pad}} - 1)\Delta x^2 / (\lambda |z|) < \pi$, i.e.
//! $n_{\text{pad}}\Delta x^2 \lesssim \lambda |z|$ — textbook rule (2)
//! $N > D^2/(Z\lambda)$ verbatim at the padded width. The forms are therefore
//! **complementary**: the convolution form is at home for long $z$ / fine
//! grids, the TF form for short $z$ / coarse grids, crossing at critical
//! sampling $n_{\text{pad}}\Delta x^2 = \lambda z$ where both steps are
//! $\approx\pi$. Densification cannot mitigate an undersampled *kernel* (it
//! makes the step worse — the remedy is a smaller kernel extent, coarser
//! pitch, or longer $|z|$), so `strict` mode raises
//! [`FresnelError::AliasingRisk`] immediately for the convolution form; the
//! crossover is surfaced by the Phase-2 regime-guidance task.
//!
//! **Translation invariance.** Both forms carry no transverse-origin
//! assumption — the TF form is diagonal in frequency, and the convolution
//! kernel $h$ depends only on the difference $(x - \xi)$ — so off-axis
//! sources propagate exactly as sampled, with no source-position parameter.
//! Among the Phase-2 engines, only the Fraunhofer prefactor references the
//! optical axis ([`super::fraunhofer::FraunhoferConfig::source_pos`]).

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
    /// Worst adjacent-sample phase step of the impulse response $h$ along $x$
    /// (rad): $\pi(n_{x,\text{pad}} - 1)\Delta x^2 / (\lambda |z|)$, the
    /// kernel-sampling criterion of [`propagate_convolution`] (reciprocal to
    /// [`FresnelSamplingReport::max_phase_step_x`]).
    pub max_kernel_step_x: f64,
    /// As [`FresnelSamplingReport::max_kernel_step_x`] along $y$.
    pub max_kernel_step_y: f64,
    /// `true` when both kernel steps are below $\pi$: the sampled impulse
    /// response is adequate for [`propagate_convolution`].
    pub kernel_nyquist_ok: bool,
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
             nyquist_ok={}, kernel step x={:.3} y={:.3} rad, kernel_ok={}, \
             paraxial error vs ASM={:.3} rad, paraxial_ok={}",
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
            self.max_kernel_step_x,
            self.max_kernel_step_y,
            self.kernel_nyquist_ok,
            self.max_phase_error_vs_asm,
            self.paraxial_ok,
        )?;
        if !self.nyquist_ok {
            write!(
                f,
                " | hint: transfer function undersampled — remedy: larger field \
                 (raise padding / max_padding to densify frequency sampling), \
                 coarsen pitch, or reduce |z|"
            )?;
        }
        if !self.kernel_nyquist_ok {
            write!(
                f,
                " | hint: impulse-response kernel undersampled — padding worsens \
                 this; remedy: smaller kernel extent (reduce padding), coarsen \
                 pitch, or increase |z|"
            )?;
        }
        Ok(())
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
    /// A sampling criterion failed under `strict` mode. The embedded report
    /// says which: for the TF form ([`propagate`]) the chirp *transfer
    /// function* stayed undersampled even after densification reached
    /// `max_padding` — remedy: larger field (more padding), coarser pitch,
    /// or shorter $|z|$; for the convolution form
    /// ([`propagate_convolution`]) the impulse-response *kernel* is
    /// undersampled and densification would worsen it — remedy: smaller
    /// kernel extent, coarser pitch, or longer $|z|$. `strict = false`
    /// proceeds with the flagged diagnostics.
    #[error("sampling criteria not met: {report}")]
    AliasingRisk {
        /// Diagnostics describing exactly which criterion failed. Boxed to
        /// keep the `Result` error arm small (review decision 2026-10-11).
        report: Box<FresnelSamplingReport>,
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

/// Worst adjacent-sample phase step of the sampled impulse response $h$
/// along one axis: the chirp $\pi x^2/(\lambda |z|)$ changes fastest at the
/// largest-magnitude sample, the Nyquist-straddling pair
/// $x_{hi} = \tfrac{n_{\text{pad}}}{2}\Delta$, $x_{lo} = x_{hi} - \Delta$:
/// $\pi(x_{hi}^2 - x_{lo}^2)/(\lambda|z|) = \pi(n_{\text{pad}} - 1)\Delta^2/(\lambda|z|)$.
/// Reciprocal in $z$ to [`worst_axis_step`] — see module docs.
fn worst_kernel_axis_step(n_pad: usize, d: f64, lambda: f64, z: f64) -> f64 {
    let hi = (n_pad / 2) as f64 * d;
    let lo = hi - d;
    PI * (hi * hi - lo * lo) / (lambda * z.abs())
}

/// Sample the paraxial impulse response (textbook Eq. 2.24 kernel, obliquity
/// factor reduced to its paraxial constant)
///
/// $$h(x, y; z) = \frac{e^{ikz}}{i\lambda z}\exp\!\Big(i\pi\frac{x^2+y^2}{\lambda z}\Big)$$
///
/// on the padded grid **in DFT order**: bin $(0, 0)$ carries the kernel
/// origin $h(0, 0)$, bins above half the length carry negative coordinates
/// (the chirp's $x^2$ symmetry makes the Nyquist-bin sign irrelevant). The
/// quadrature factor $\Delta x \Delta y$ of the convolution integral is
/// folded into the amplitude so that
/// $\text{ifft2}(\text{fft2}(u) \cdot \text{fft2}(h))$ approximates
/// $\iint u(\xi) h(x - \xi)\, d\xi$ directly.
fn impulse_response(
    nx_pad: usize,
    ny_pad: usize,
    dx: f64,
    dy: f64,
    lambda: f64,
    z: f64,
) -> Vec<Complex64> {
    let k = 2.0 * PI / lambda;
    let pref = ((k * z) * Complex64::I).exp() / (Complex64::I * lambda * z) * (dx * dy);
    let coord = |i: usize, n: usize, d: f64| -> f64 {
        if i <= n / 2 {
            i as f64 * d
        } else {
            (i as f64 - n as f64) * d
        }
    };
    let mut h = vec![Complex64::ZERO; nx_pad * ny_pad];
    for iy in 0..ny_pad {
        let y = coord(iy, ny_pad, dy);
        for ix in 0..nx_pad {
            let x = coord(ix, nx_pad, dx);
            h[iy * nx_pad + ix] = pref * (PI * (x * x + y * y) / (lambda * z) * Complex64::I).exp();
        }
    }
    h
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
    let kstep_x = worst_kernel_axis_step(nx_pad, dx, lambda, z);
    let kstep_y = worst_kernel_axis_step(ny_pad, dy, lambda, z);
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
        max_kernel_step_x: kstep_x,
        max_kernel_step_y: kstep_y,
        kernel_nyquist_ok: kstep_x < PI && kstep_y < PI,
        max_phase_error_vs_asm: err,
        paraxial_ok: err < 0.01,
    }
}

/// Sampling/validity diagnostics for a Fresnel propagation of `band`
/// over `z` under `cfg`, without performing the propagation. Covers both
/// forms: transfer-function bin steps (`max_phase_step_*`, `nyquist_ok`)
/// and impulse-response sample steps (`max_kernel_step_*`,
/// `kernel_nyquist_ok`); each propagator's `strict` gate uses its own
/// criterion. Evaluated at the configured `padding`; mitigation is a
/// [`propagate`] behavior.
pub fn sampling_report(band: &BandField, z: f64, cfg: &FresnelConfig) -> FresnelSamplingReport {
    analyze(band, z, cfg, cfg.padding)
}

/// Common engine-boundary validation shared by [`propagate`] and
/// [`propagate_convolution`].
///
/// Defensive re-validation: `Field::new`/`BandField::new` already enforce
/// these invariants at construction, but `Field::field_mut()` is public —
/// custom initializers or downstream code can scribble non-finite samples
/// into an otherwise valid band. The engine is the last place a silent NaN
/// could be amplified, so it re-checks (O(N), cheap) rather than trusting
/// the caller.
fn validate(band: &BandField, z: f64) -> Result<(), FresnelError> {
    if !z.is_finite() {
        return Err(FresnelError::InvalidDistance(z));
    }
    let lambda = band.lambda();
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(FresnelError::InvalidWavelength(lambda));
    }
    let (nx, ny) = band.field().dims();
    if nx == 0 || ny == 0 {
        return Err(FresnelError::EmptyGrid);
    }
    if band.field().samples().iter().any(|s| !s.is_finite()) {
        return Err(FresnelError::NonFiniteSample);
    }
    Ok(())
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
    validate(band, z)?;
    let lambda = band.lambda();
    let field = band.field();
    let (nx, ny) = field.dims();
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
            return Err(FresnelError::AliasingRisk {
                report: Box::new(report),
            });
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

/// Propagates a [`BandField`] a signed distance `z` (meters) with the Fresnel
/// **convolution (impulse-response) form**: the field convolved with the
/// sampled paraxial kernel $h$ of textbook Eq. 2.24 (see module docs for the
/// formula, the DFT-order sampling, and the reciprocal kernel-sampling
/// criterion).
///
/// Mathematically equivalent to [`propagate`] (same operator, related by the
/// convolution theorem); numerically it is the right form when the transfer
/// function would be undersampled — long $z$ / fine grids (textbook rule 2,
/// $N > D^2/(Z\lambda)$) — while the TF form is at home for short $z$ /
/// coarse grids; the crossover is critical sampling
/// $n_{\text{pad}}\Delta x^2 = \lambda z$.
///
/// # Errors
/// Same taxonomy as [`propagate`], but the `strict` gate uses
/// [`FresnelSamplingReport::kernel_nyquist_ok`]: an undersampled *kernel*
/// cannot be fixed by densification (it worsens the step), so
/// [`FresnelError::AliasingRisk`] is raised immediately; remedies are a
/// coarser pitch, a shorter $|z|$, or `strict = false`. `z = 0` returns the
/// input (the kernel is singular there; the limit is the identity).
pub fn propagate_convolution(
    band: &BandField,
    z: f64,
    cfg: &FresnelConfig,
) -> Result<BandField, FresnelError> {
    validate(band, z)?;
    let lambda = band.lambda();
    let field = band.field();
    let (nx, ny) = field.dims();
    if z == 0.0 {
        return Ok(band.clone());
    }

    let report = analyze(band, z, cfg, cfg.padding);
    if cfg.strict && !report.kernel_nyquist_ok {
        // No densification loop: the kernel step grows with the padded grid
        // width, so mitigation would make it worse (see module docs).
        return Err(FresnelError::AliasingRisk {
            report: Box::new(report),
        });
    }

    let (nx_pad, ny_pad) = report.padded_dims;
    let (dx, dy) = field.pitch();
    let ox = (nx_pad - nx) / 2;
    let oy = (ny_pad - ny) / 2;

    // Embed the centered input window into the centered padded grid
    // (guard-band rationale as in `propagate`: circular must equal linear).
    let mut buf = vec![Complex64::ZERO; nx_pad * ny_pad];
    for iy in 0..ny {
        for ix in 0..nx {
            buf[(iy + oy) * nx_pad + (ix + ox)] = field.sample(ix, iy);
        }
    }
    fft2(&mut buf, nx_pad, ny_pad);

    // Sampled impulse response in DFT order; its transform multiplies the
    // field spectrum. With NumPy-convention scaling,
    // ifft2(fft2(u) * fft2(h)) is exactly the circular convolution sum,
    // and the Delta^2 quadrature weight folded into `h` makes it the
    // Riemann approximation of the convolution integral.
    let mut hbuf = impulse_response(nx_pad, ny_pad, dx, dy, lambda, z);
    fft2(&mut hbuf, nx_pad, ny_pad);
    for (b, hh) in buf.iter_mut().zip(&hbuf) {
        *b *= hh;
    }

    ifft2(&mut buf, nx_pad, ny_pad);

    // Extract the central physical window (same grid in, same grid out).
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
        assert!(matches!(
            propagate_convolution(&band, f64::NAN, &FresnelConfig::default()),
            Err(FresnelError::InvalidDistance(_))
        ));
    }

    fn gaussian(nx: usize, ny: usize, d: f64, sigma: f64) -> BandField {
        let mut samples = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * d;
            for ix in 0..nx {
                let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * d;
                samples.push(Complex64::from(
                    (-((x * x + y * y) / (2.0 * sigma * sigma))).exp(),
                ));
            }
        }
        BandField::new(LAMBDA, Field::new(samples, nx, ny, d, d).unwrap()).unwrap()
    }

    #[test]
    fn convolution_of_delta_is_the_sampled_kernel() {
        // Absolute normalization anchor: u = delta at bin (0,0) on the
        // unpadded grid => output = Delta^2 * h in DFT order. |h| = 1/(lambda z)
        // is flat, and the phase ratio reproduces the chirp exactly.
        let (n, d, z) = (32usize, 1e-5f64, 1e-3f64);
        let mut samples = vec![Complex64::ZERO; n * n];
        samples[0] = Complex64::new(1.0, 0.0);
        let band = BandField::new(LAMBDA, Field::new(samples, n, n, d, d).unwrap()).unwrap();
        let cfg = FresnelConfig {
            padding: 1.0,
            strict: false,
            max_padding: 1.0,
        };
        let out = propagate_convolution(&band, z, &cfg).unwrap();
        let expected_amp = d * d / (LAMBDA * z); // 0.2
        let h0 = out.field().sample(0, 0);
        assert!((h0.norm() - expected_amp).abs() < 1e-12);
        // k = e^{ikz} / (i lambda z) * d^2: check the DC value's phase too.
        let k = 2.0 * PI / LAMBDA;
        let pref = ((k * z) * Complex64::I).exp() / (Complex64::I * LAMBDA * z) * (d * d);
        assert!((h0 - pref).norm() < 1e-12);
        // Flat magnitude everywhere and exact chirp phase ratios along x.
        for iy in [0usize, 11, 16] {
            for ix in [1usize, 5, 16, 31] {
                let s = out.field().sample(ix, iy);
                assert!((s.norm() - expected_amp).abs() < 1e-12, "amp not flat");
                let x = if ix <= n / 2 {
                    ix as f64
                } else {
                    ix as f64 - n as f64
                } * d;
                let y = if iy <= n / 2 {
                    iy as f64
                } else {
                    iy as f64 - n as f64
                } * d;
                let chirp = (PI * (x * x + y * y) / (LAMBDA * z) * Complex64::I).exp();
                assert!(
                    (s / h0 - chirp).norm() < 1e-9,
                    "kernel chirp wrong at {ix},{iy}"
                );
            }
        }
    }

    #[test]
    fn convolution_guard_is_reciprocal_to_transfer_guard() {
        // n=64, d=1um, z=5e-4: n d^2 = 6.4e-11 << lambda z = 2.5e-10, the
        // convolution form's home regime. Kernel step = pi*0.25 (fine), TF
        // step = pi*3.8 (undersampled): strict convolution succeeds where
        // strict TF fails at max_padding = 1.0, and TF succeeds once
        // densification (1 -> 2 -> 4) is allowed.
        let band = band_uniform(1.0, 64, 64, 1e-6, 1e-6);
        let z = 5e-4;
        let cfg = FresnelConfig {
            padding: 1.0,
            strict: true,
            max_padding: 1.0,
        };
        let rep = sampling_report(&band, z, &cfg);
        assert!(rep.kernel_nyquist_ok && rep.max_kernel_step_x < PI);
        assert!(!rep.nyquist_ok && rep.max_phase_step_x > PI);
        assert!(propagate_convolution(&band, z, &cfg).is_ok());
        assert!(matches!(
            propagate(&band, z, &cfg),
            Err(FresnelError::AliasingRisk { .. })
        ));
        // Densification fixes the TF form (step ~ 1/padding) and is
        // unnecessary for the convolution form.
        let tf_fix = FresnelConfig {
            max_padding: 8.0,
            ..cfg
        };
        assert!(propagate(&band, z, &tf_fix).is_ok());
    }

    #[test]
    fn convolution_matches_tf_at_critical_sampling() {
        // Critical sampling n*d^2 = lambda z makes the DFT of the sampled
        // (non-decaying) chirp kernel match the sampled transfer function:
        // the two forms must then agree. Compact Gaussian keeps spectral
        // replicas negligible so the comparison isolates the operator.
        let n = 256usize;
        let z = 2e-3;
        let d = (LAMBDA * z / n as f64).sqrt(); // n d^2 = lambda z exactly
        let sigma = 50.0 * d;
        let band = gaussian(n, n, d, sigma);
        let cfg = FresnelConfig {
            padding: 1.0,
            strict: true,
            max_padding: 1.0,
        };
        let rep = sampling_report(&band, z, &cfg);
        assert!(
            rep.nyquist_ok && rep.kernel_nyquist_ok,
            "critical sampling should sit just inside both guards: {rep}"
        );
        let tf = propagate(&band, z, &cfg).unwrap();
        let conv = propagate_convolution(&band, z, &cfg).unwrap();
        let diff: f64 = tf
            .field()
            .samples()
            .iter()
            .zip(conv.field().samples())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum();
        let norm: f64 = tf.field().samples().iter().map(|a| a.norm_sqr()).sum();
        let rel = (diff / norm).sqrt();
        assert!(
            rel < 1e-6,
            "TF vs convolution relative difference {rel:e} at critical sampling"
        );
    }

    #[test]
    fn convolution_z_zero_is_identity() {
        let band = band_uniform(2.0, 16, 16, 20e-6, 20e-6);
        let out = propagate_convolution(&band, 0.0, &FresnelConfig::default()).unwrap();
        assert_eq!(out, band);
    }
}
