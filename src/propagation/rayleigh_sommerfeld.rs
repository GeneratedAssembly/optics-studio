//! Rayleigh–Sommerfeld first-kind propagation (exact spherical kernel).
//!
//! The Rayleigh–Sommerfeld (RS) integral is the scalar-diffraction solution
//! of the Helmholtz equation for the half-space problem: given the field
//! sampled on a plane $z = 0$, evaluate it at a parallel plane at signed
//! distance $z$ (textbook Sec. 2.1, Eq. 2.8). The book writes it in its
//! $e^{-ikr}$ time convention; in this crate's $e^{+ikz}$ forward frame the
//! first-kind impulse response is the conjugate form
//!
//! $$h(x, y; z) = \frac{|z|}{2\pi r^2}\Big(\frac{1}{r} - i\,\mathrm{sgn}(z)\,k\Big)
//!   \exp\!\big(i\,\mathrm{sgn}(z)\,k r\big), \qquad
//!   r = \sqrt{x^2 + y^2 + z^2}, \qquad k = \frac{2\pi}{\lambda},$$
//!
//! (textbook Eq. 2.8 with the obliquity factor $(1/r - ik)$ from the
//! Dirichlet Green's function; assumptions 7–8: aperture and $|z|$ large
//! vs $\lambda$), and propagation is the convolution
//!
//! $$U_z(x, y) = \iint U_0(\xi, \eta)\, h(x - \xi, y - \eta; z)\, d\xi\, d\eta .$$
//!
//! # Why first kind, and why it is *not* new physics
//!
//! The exact Kirchhoff surface integral couples the boundary field $u$ with
//! its unknown normal derivative $\partial u/\partial n$. The two RS
//! variants break the deadlock with image Green's functions: the **first
//! kind** (antisymmetric Green, $G = 0$ on the plane) needs only $u$ — the
//! data a [`BandField`] actually carries — while the second kind needs
//! $\partial u/\partial n$, which the crate's plane-sampled model does not
//! provide (estimating it as $\approx -iku$ would reintroduce exactly the
//! inconsistent boundary assumption RS theory avoids). RS-II is
//! deliberately excluded.
//!
//! With the first-kind kernel, the transfer function is exactly the ASM's
//! $e^{ikz\gamma}$ (the book makes the identification in Sec. 2.2.2/3.2:
//! RS propagation *is* the plane-wave spectrum). The engine's value is a
//! **third discretization of the same physics**: the ASM advances the
//! spectral transfer function (guarded in [`super::asm`]), this engine
//! samples the spatial kernel — so agreement between them validates both
//! discretizations, and agreement/divergence vs [`super::fresnel`]
//! quantifies the paraxial truncation (see [`RayleighSommerfeldSamplingReport::max_phase_error_vs_fresnel`]).
//!
//! # Paraxial reduction (validated)
//!
//! With $r \approx |z|$ and $1/r \ll k$ the kernel becomes
//! $\frac{e^{ikz}}{i\lambda z}\exp\!\big(i\pi\frac{x^2+y^2}{\lambda z}\big)$ —
//! the Fresnel convolution kernel of [`super::fresnel::propagate_convolution`]
//! (textbook Eq. 2.24). The unit test `kernel_reduces_to_fresnel_paraxially`
//! pins the sampled kernels against each other in a small-angle geometry.
//!
//! # Sampling verification (kernel-side guard)
//!
//! The kernel phase is $\phi(\rho) = k r(\rho) - \arctan(kr)$ (the
//! $\arctan$ from the obliquity factor). Along one axis the geometric term
//! is convex in $|x|$, so the worst adjacent-sample step is closed-form at
//! the Nyquist-straddling pair, with the obliquity term added as a bound
//! (its $r$-derivative is $k/(1 + k^2r^2) \le k/(1 + k^2z^2)$):
//!
//! $$\Delta\phi_x^{\max} = k\big(r(x_{hi}) - r(x_{lo})\big) + \frac{k\,\Delta x}{1 + k^2 z^2},
//!   \qquad x_{hi} = \frac{n_{x,\text{pad}}}{2}\Delta x,\; x_{lo} = x_{hi} - \Delta x.$$
//!
//! Because $|x|/r \le 1$, this step is **always smaller than the Fresnel
//! convolution kernel step** $\pi(n_{\text{pad}} - 1)\Delta x^2/(\lambda|z|)$
//! (the paraxial flattening grows the phase gradient without bound; the
//! spherical $kr$ saturates at $k\sin\theta \le k$): the RS form tolerates
//! wider padded grids than its paraxial sibling. Like the Fresnel
//! convolution form, densification *worsens* kernel sampling (it extends
//! the grid to larger $|x|/r$), so `strict` mode raises
//! [`RayleighSommerfeldError::AliasingRisk`] immediately — remedies:
//! coarser pitch, larger $|z|$, smaller padding, or the transfer-function
//! engines. The limit $\Delta\phi < \pi$ reduces to the intuitive
//! $\Delta x < \lambda/2$ requirement only at the grid rim ($|x| \gg |z|$):
//! near-axis kernel sampling is far more forgiving.
//!
//! # Signed $z$ and $z = 0$
//!
//! Back-propagation is the exact inverse operator:
//! $h(\cdot, -z) = \overline{h(\cdot, z)}$ (bit-exact by construction —
//! the book's $e^{-ikr}$ form). $z = 0$ is **rejected**
//! ([`RayleighSommerfeldError::InvalidDistance`]): the kernel is a
//! distribution ($\to \delta$) there, not a sampleable function; use
//! [`super::asm::propagate`] (identity at $z = 0$) if a zero step must
//! flow through a pipeline.

use std::f64::consts::PI;
use std::fmt;

use num_complex::Complex64;
use thiserror::Error;

use super::fft::{fft2, ifft2};
use super::padded_len;
use crate::field::{BandField, Field};

/// Tuning knobs for [`propagate`].
#[derive(Clone, Copy, Debug)]
pub struct RayleighSommerfeldConfig {
    /// Linear zero-padding factor applied to each axis before transform;
    /// values `< 1.0` are treated as `1.0` (no padding). Typical: `2.0`.
    pub padding: f64,
    /// When `true`, [`propagate`] raises
    /// [`RayleighSommerfeldError::AliasingRisk`] if the sampled kernel is
    /// undersampled. There is no densification mitigation (it worsens
    /// kernel sampling — see module docs), unlike the ASM/Fresnel-TF
    /// policies.
    pub strict: bool,
}

impl Default for RayleighSommerfeldConfig {
    /// Engine defaults mirroring the convolution-form policy:
    /// `padding = 2.0`, `strict = true`.
    fn default() -> Self {
        Self {
            padding: 2.0,
            strict: true,
        }
    }
}

/// Spatial-sampling and paraxial-divergence diagnostics for one
/// Rayleigh–Sommerfeld propagation step.
///
/// All quantities SI: meters, cycles per meter, radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayleighSommerfeldSamplingReport {
    /// Operating wavelength (m).
    pub lambda: f64,
    /// Signed propagation distance (m).
    pub z: f64,
    /// Input grid sample counts `(nx, ny)`.
    pub input_dims: (usize, usize),
    /// Input sample pitch `(dx, dy)` (m).
    pub pitch: (f64, f64),
    /// Padding factor used for the padded grid (no mitigation ladder exists
    /// for this engine, so requested and used coincide).
    pub padding: f64,
    /// Padded grid sample counts `(nx_pad, ny_pad)`.
    pub padded_dims: (usize, usize),
    /// Closed-form worst adjacent-sample phase step of $h$ along $x$ (rad).
    pub max_kernel_step_x: f64,
    /// Worst adjacent-sample kernel phase step along $y$ (rad).
    pub max_kernel_step_y: f64,
    /// `true` when both kernel steps are below $\pi$: the sampled impulse
    /// response is adequate for [`propagate`].
    pub kernel_nyquist_ok: bool,
    /// Worst-case non-paraxial phase excess of the spherical kernel over
    /// the paraxial expansion, $k\big(r_{\text{corner}} - |z| -
    /// \rho_{\text{corner}}^2/(2|z|)\big)$ (rad) — the physics the Fresnel
    /// forms drop and this engine keeps. $0$ when the grid is small vs
    /// $|z|$.
    pub max_phase_error_vs_fresnel: f64,
    /// `true` when [`RayleighSommerfeldSamplingReport::max_phase_error_vs_fresnel`]
    /// < 0.01 rad: the geometry is effectively paraxial and
    /// [`super::fresnel::propagate_convolution`] would agree.
    pub paraxial_ok: bool,
}

impl fmt::Display for RayleighSommerfeldSamplingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RS-I sampling: lambda={:.3e} m, z={:.3e} m, grid={}x{} pitch={:.3e}x{:.3e} m, \
             padding={:.2}, padded={}x{}, kernel step x={:.3} y={:.3} rad, kernel_ok={}, \
             phase excess vs Fresnel={:.3} rad, paraxial_ok={}",
            self.lambda,
            self.z,
            self.input_dims.0,
            self.input_dims.1,
            self.pitch.0,
            self.pitch.1,
            self.padding,
            self.padded_dims.0,
            self.padded_dims.1,
            self.max_kernel_step_x,
            self.max_kernel_step_y,
            self.kernel_nyquist_ok,
            self.max_phase_error_vs_fresnel,
            self.paraxial_ok,
        )?;
        if !self.kernel_nyquist_ok {
            write!(
                f,
                " | hint: impulse-response kernel undersampled — padding worsens \
                 this; remedy: smaller kernel extent (reduce padding), coarsen \
                 pitch, increase |z|, or switch to a transfer-function engine \
                 (ASM / Fresnel TF)"
            )?;
        }
        Ok(())
    }
}

/// Failures of the Rayleigh–Sommerfeld first-kind propagator.
///
/// Mirrors [`super::fresnel::FresnelError`]'s convolution-side taxonomy with a
/// [`RayleighSommerfeldSamplingReport`] payload; unifying the engine error
/// enums is an open API decision.
#[derive(Clone, Debug, Error)]
pub enum RayleighSommerfeldError {
    /// Propagation distance was not finite, **or zero**: the RS-I impulse
    /// response is a distribution at $z = 0$ (it tends to $\delta$) and
    /// cannot be sampled. Negative $z$ is valid (back-propagation kernel
    /// $= \overline{h(\cdot, z)}$).
    #[error("RS propagation distance must be finite and nonzero (got {0} m)")]
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
    /// The kernel-sampling criterion failed under `strict` mode. The
    /// embedded report says how far the worst adjacent-sample phase step
    /// exceeds $\pi$; densification would worsen it, so remedies are a
    /// coarser pitch, a shorter grid, a longer $|z|$, or `strict = false`.
    #[error("sampling criteria not met: {report}")]
    AliasingRisk {
        /// Diagnostics describing exactly which criterion failed. Boxed to
        /// keep the `Result` error arm small (review decision 2026-10-11).
        report: Box<RayleighSommerfeldSamplingReport>,
    },
}

/// RS-I impulse response value at coordinate $(x, y)$ for signed distance
/// $z$: $h = \frac{|z|}{2\pi r^2}\big(\frac{1}{r} - i s k\big)e^{i s k r}$
/// with $s = \mathrm{sgn}(z)$, $r = \sqrt{x^2 + y^2 + z^2}$.
fn kernel_value(x: f64, y: f64, lambda: f64, z: f64) -> Complex64 {
    let k = 2.0 * PI / lambda;
    let s = if z > 0.0 { 1.0 } else { -1.0 };
    let r = (x * x + y * y + z * z).sqrt();
    let amp = z.abs() / (2.0 * PI * r * r);
    let obliquity = Complex64::new(1.0 / r, -s * k);
    amp * obliquity * ((s * k * r) * Complex64::I).exp()
}

/// Worst adjacent-sample kernel phase step along one axis: geometric term
/// $k(r_{hi} - r_{lo})$ at the Nyquist-straddling pair (convex in $|x|$,
/// worst row $y = 0$) plus the obliquity bound $k\Delta/(1 + k^2z^2)$.
fn worst_kernel_axis_step(n_pad: usize, d: f64, lambda: f64, z: f64) -> f64 {
    let k = 2.0 * PI / lambda;
    let x_hi = (n_pad / 2) as f64 * d;
    let x_lo = x_hi - d;
    let r_hi = (x_hi * x_hi + z * z).sqrt();
    let r_lo = (x_lo * x_lo + z * z).sqrt();
    let geom = k * (r_hi - r_lo);
    let obliq = k * d / (1.0 + k * k * z * z);
    geom + obliq
}

/// Worst non-paraxial phase excess of the spherical kernel over the
/// paraxial expansion, at the farthest grid corner:
/// $k\big(r_c - |z| - \rho_c^2/(2|z|)\big) \ge 0$.
fn worst_paraxial_excess(
    nx_pad: usize,
    ny_pad: usize,
    dx: f64,
    dy: f64,
    lambda: f64,
    z: f64,
) -> f64 {
    let k = 2.0 * PI / lambda;
    let x = (nx_pad / 2) as f64 * dx;
    let y = (ny_pad / 2) as f64 * dy;
    let az = z.abs();
    let r = (x * x + y * y + az * az).sqrt();
    let excess = k * (r - az - (x * x + y * y) / (2.0 * az));
    excess.max(0.0)
}

/// Sample the RS-I impulse response on the padded grid **in DFT order**
/// (kernel origin at bin $(0, 0)$, required for circular convolution; the
/// kernel is even in $x, y$ so the Nyquist-bin sign is irrelevant), with
/// the $\Delta x \Delta y$ quadrature weight folded in so that
/// `ifft2(fft2(u) * fft2(h))` approximates the convolution integral
/// directly (NumPy-convention scaling — Phase-2 decision).
fn impulse_response(
    nx_pad: usize,
    ny_pad: usize,
    dx: f64,
    dy: f64,
    lambda: f64,
    z: f64,
) -> Vec<Complex64> {
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
            h[iy * nx_pad + ix] = kernel_value(x, y, lambda, z) * (dx * dy);
        }
    }
    h
}

/// Internal diagnostics at a concrete padding factor.
fn analyze(band: &BandField, z: f64, padding: f64) -> RayleighSommerfeldSamplingReport {
    let field = band.field();
    let (nx, ny) = field.dims();
    let (dx, dy) = field.pitch();
    let lambda = band.lambda();
    let nx_pad = padded_len(nx, padding);
    let ny_pad = padded_len(ny, padding);

    let kstep_x = worst_kernel_axis_step(nx_pad, dx, lambda, z);
    let kstep_y = worst_kernel_axis_step(ny_pad, dy, lambda, z);
    let err = worst_paraxial_excess(nx_pad, ny_pad, dx, dy, lambda, z);

    RayleighSommerfeldSamplingReport {
        lambda,
        z,
        input_dims: (nx, ny),
        pitch: (dx, dy),
        padding,
        padded_dims: (nx_pad, ny_pad),
        max_kernel_step_x: kstep_x,
        max_kernel_step_y: kstep_y,
        kernel_nyquist_ok: kstep_x < PI && kstep_y < PI,
        max_phase_error_vs_fresnel: err,
        paraxial_ok: err < 0.01,
    }
}

/// Sampling diagnostics for a Rayleigh–Sommerfeld propagation of `band`
/// over `z` under `cfg`, without performing the propagation.
pub fn sampling_report(
    band: &BandField,
    z: f64,
    cfg: &RayleighSommerfeldConfig,
) -> RayleighSommerfeldSamplingReport {
    analyze(band, z, cfg.padding)
}

/// Engine-boundary validation. Defensive re-validation of samples follows
/// the same rationale as the Fresnel engines (public `field_mut()` makes
/// construction invariants necessary but not sufficient at the engine
/// boundary).
fn validate(band: &BandField, z: f64) -> Result<(), RayleighSommerfeldError> {
    if !z.is_finite() || z == 0.0 {
        return Err(RayleighSommerfeldError::InvalidDistance(z));
    }
    let lambda = band.lambda();
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(RayleighSommerfeldError::InvalidWavelength(lambda));
    }
    let (nx, ny) = band.field().dims();
    if nx == 0 || ny == 0 {
        return Err(RayleighSommerfeldError::EmptyGrid);
    }
    if band.field().samples().iter().any(|s| !s.is_finite()) {
        return Err(RayleighSommerfeldError::NonFiniteSample);
    }
    Ok(())
}

/// Propagates a [`BandField`] a signed, nonzero distance `z` (meters) with
/// the Rayleigh–Sommerfeld first-kind convolution form: the field convolved
/// with the sampled spherical impulse response $h$ (module docs for the
/// formula, the DFT-order sampling, and the kernel-sampling criterion).
///
/// The output grid geometry (dims, pitch, wavelength) equals the input; the
/// computation runs on the zero-padded grid and the central physical window
/// is extracted. The operator equals the ASM's transfer function in the
/// continuum — this form trades the spectral discretization for a spatial
/// one.
///
/// # Errors
/// - [`RayleighSommerfeldError::InvalidDistance`] if `z` is not finite or
///   zero (the kernel is a distribution at $z = 0$).
/// - [`RayleighSommerfeldError::InvalidWavelength`] if the band wavelength
///   is not positive and finite.
/// - [`RayleighSommerfeldError::EmptyGrid`] /
///   [`RayleighSommerfeldError::NonFiniteSample`] for degenerate inputs.
/// - [`RayleighSommerfeldError::AliasingRisk`] when `cfg.strict` and the
///   sampled kernel is undersampled (raised immediately — densification
///   would worsen it).
pub fn propagate(
    band: &BandField,
    z: f64,
    cfg: &RayleighSommerfeldConfig,
) -> Result<BandField, RayleighSommerfeldError> {
    validate(band, z)?;
    let lambda = band.lambda();
    let field = band.field();
    let (nx, ny) = field.dims();

    let report = analyze(band, z, cfg.padding);
    if cfg.strict && !report.kernel_nyquist_ok {
        // No densification loop: the kernel step grows with the padded grid
        // extent (see module docs).
        return Err(RayleighSommerfeldError::AliasingRisk {
            report: Box::new(report),
        });
    }

    let (nx_pad, ny_pad) = report.padded_dims;
    let (dx, dy) = field.pitch();
    let ox = (nx_pad - nx) / 2;
    let oy = (ny_pad - ny) / 2;

    // Embed the centered input window into the centered padded grid
    // (guard-band rationale as in the Fresnel convolution engine: the DFT
    // makes circular convolution out of what must be linear).
    let mut buf = vec![Complex64::ZERO; nx_pad * ny_pad];
    for iy in 0..ny {
        for ix in 0..nx {
            buf[(iy + oy) * nx_pad + (ix + ox)] = field.sample(ix, iy);
        }
    }
    fft2(&mut buf, nx_pad, ny_pad);

    // Sampled RS impulse response in DFT order; with NumPy-convention
    // scaling, ifft2(fft2(u) * fft2(h)) is exactly the circular convolution
    // sum, and the Delta^2 weight folded into `h` makes it the Riemann
    // approximation of the RS convolution integral.
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
    let field_out =
        Field::new(out, nx, ny, dx, dy).map_err(|_| RayleighSommerfeldError::NonFiniteSample)?;
    BandField::new(lambda, field_out)
        .map_err(|_| RayleighSommerfeldError::InvalidWavelength(lambda))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAMBDA: f64 = 500e-9;

    fn band_uniform(a: f64, nx: usize, ny: usize, dx: f64, dy: f64) -> BandField {
        BandField::uniform(a, 0.0, nx, ny, dx, dy, LAMBDA).unwrap()
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

    fn rel_frobenius(a: &BandField, b: &BandField) -> f64 {
        let diff: f64 = a
            .field()
            .samples()
            .iter()
            .zip(b.field().samples())
            .map(|(x, y)| (x - y).norm_sqr())
            .sum();
        let norm: f64 = b.field().samples().iter().map(|x| x.norm_sqr()).sum();
        (diff / norm).sqrt()
    }

    #[test]
    fn defaults_mirror_convolution_form_policy() {
        let cfg = RayleighSommerfeldConfig::default();
        assert_eq!(cfg.padding, 2.0);
        assert!(cfg.strict);
    }

    #[test]
    fn kernel_reduces_to_fresnel_paraxially() {
        // Small-angle geometry: n=64, d=10um, z=5m -> corner rho/|z| ~ 9e-5,
        // paraxial excess k*rho^4/(8 z^3) ~ 5e-10 rad. The RS-I kernel and
        // the Fresnel convolution kernel (textbook Eq. 2.24) must agree to
        // far below that bound.
        let (n, d, z) = (64usize, 10e-6f64, 5.0f64);
        let k = 2.0 * PI / LAMBDA;
        let h_f = |x: f64, y: f64| -> Complex64 {
            ((k * z) * Complex64::I).exp() / (Complex64::I * LAMBDA * z)
                * ((PI * (x * x + y * y) / (LAMBDA * z)) * Complex64::I).exp()
        };
        let coord = |i: usize| -> f64 {
            (if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            }) * d
        };
        let mut max_rel = 0.0_f64;
        for iy in [0usize, 16, 31, 32, 48, 63] {
            for ix in [0usize, 16, 31, 32, 48, 63] {
                let (x, y) = (coord(ix), coord(iy));
                let rs = kernel_value(x, y, LAMBDA, z);
                let fr = h_f(x, y);
                max_rel = max_rel.max((rs / fr - 1.0).norm());
            }
        }
        assert!(
            max_rel < 1e-6,
            "RS kernel deviates from paraxial limit by {max_rel:e}"
        );
    }

    #[test]
    fn negative_z_kernel_is_conjugate() {
        // Back-propagation kernel = conjugate, bit-exact by construction
        // (even grid coords; sin is odd, cos even; real prefactor).
        let (n, d, z) = (32usize, 10e-6f64, 2e-3f64);
        let fwd = impulse_response(n, n, d, d, LAMBDA, z);
        let bwd = impulse_response(n, n, d, d, LAMBDA, -z);
        for (f, b) in fwd.iter().zip(&bwd) {
            assert_eq!(*b, f.conj());
        }
    }

    #[test]
    fn delta_input_is_the_sampled_kernel() {
        // Absolute normalization anchor: u = delta at bin (0,0), unpadded
        // grid => output = Delta^2 * h in DFT order.
        let (n, d, z) = (32usize, 1e-5f64, 1e-3f64);
        let mut samples = vec![Complex64::ZERO; n * n];
        samples[0] = Complex64::new(1.0, 0.0);
        let band = BandField::new(LAMBDA, Field::new(samples, n, n, d, d).unwrap()).unwrap();
        let cfg = RayleighSommerfeldConfig {
            padding: 1.0,
            strict: false,
        };
        let out = propagate(&band, z, &cfg).unwrap();
        let h0 = out.field().sample(0, 0);
        let expected = kernel_value(0.0, 0.0, LAMBDA, z) * (d * d);
        assert!((h0 - expected).norm() < 1e-12);
        // A few off-origin DFT-order points reproduce the kernel exactly.
        for (ix, iy) in [(1usize, 0), (5, 3), (16, 16), (31, 31)] {
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
            let s = out.field().sample(ix, iy);
            let e = kernel_value(x, y, LAMBDA, z) * (d * d);
            assert!((s - e).norm() < 1e-12, "kernel wrong at {ix},{iy}");
        }
    }

    #[test]
    fn engine_agrees_with_asm_on_smooth_band() {
        // The RS-I operator IS the plane-wave spectrum (Sec. 2.2.2/3.2):
        // two discretizations of the same physics (spatial kernel vs
        // spectral transfer) must agree on a band whose spectrum is well
        // inside both guards. Geometry at critical-sampling scale:
        // n_pad*d^2 = 400*4e-12 = 1.6e-9 < lambda*z = 2.5e-9 (rim kernel
        // step ~2.0 rad < pi) and ~n_pad/4 = 100 kernel Fresnel zones
        // resolved inside the padded window; the Gaussian (sigma 30 um,
        // spread 46 um at z) stays deep inside the 400 um window.
        let (n, d) = (200usize, 2e-6f64);
        let z = 5e-3f64;
        let band = gaussian(n, n, d, 30e-6);
        let cfg = RayleighSommerfeldConfig {
            padding: 2.0,
            strict: true,
        };
        let rep = sampling_report(&band, z, &cfg);
        assert!(rep.kernel_nyquist_ok, "geometry must pass the guard: {rep}");
        let rs = propagate(&band, z, &cfg).unwrap();
        let asm = crate::propagation::asm::propagate(
            &band,
            z,
            &crate::propagation::asm::AsmConfig {
                padding: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
        let rel = rel_frobenius(&rs, &asm);
        assert!(
            rel < 5e-2,
            "RS kernel discretization vs ASM transfer: rel {rel:e}"
        );
    }

    #[test]
    fn kernel_step_closed_form_brackets_sweep() {
        // Brute-force sweep of adjacent-sample phase differences along x
        // over all rows must not exceed the closed form, and the closed form
        // must stay within 2x of it (conservative, not vacuous). The phase
        // is evaluated in UNWRAPPED analytic form
        // phi = s k r - s atan(k r) (arg of (1/r - i s k) e^{i s k r}) —
        // arg() of the complex value would wrap and fake 2pi jumps.
        let (n, d, z) = (64usize, 2e-6f64, 3e-4f64);
        let k = 2.0 * PI / LAMBDA;
        let coord = |i: usize| -> f64 {
            (if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            }) * d
        };
        let phase = |x: f64, y: f64| {
            let r = (x * x + y * y + z * z).sqrt();
            k * r - (k * r).atan()
        };
        let mut sweep = 0.0_f64;
        for iy in 0..n {
            for ix in 0..n {
                let a = phase(coord(ix), coord(iy));
                let b = phase(coord((ix + 1) % n), coord(iy));
                sweep = sweep.max((a - b).abs());
            }
        }
        let closed = worst_kernel_axis_step(n, d, LAMBDA, z);
        assert!(
            sweep <= closed + 1e-12,
            "sweep {sweep} exceeded closed form {closed}"
        );
        assert!(
            closed < 2.0 * sweep + 1e-12,
            "closed form {closed} vacuous vs sweep {sweep}"
        );
    }

    #[test]
    fn strict_guard_raises_immediately_and_padding_worsens_it() {
        // d=10um, z=1mm: rim samples reach |x|/r ~ 0.95, step ~ k*0.95*d
        // ~ 120 rad >> pi. Strict mode raises at once; doubling padding
        // pushes the rim further out (larger |x|/r) and cannot help.
        let band = band_uniform(1.0, 64, 64, 10e-6, 10e-6);
        let z = 1e-3;
        let cfg = RayleighSommerfeldConfig {
            padding: 1.0,
            strict: true,
        };
        let rep = sampling_report(&band, z, &cfg);
        assert!(!rep.kernel_nyquist_ok);
        assert!(rep.max_kernel_step_x > PI);
        assert!(matches!(
            propagate(&band, z, &cfg),
            Err(RayleighSommerfeldError::AliasingRisk { .. })
        ));

        let rep4 = sampling_report(
            &band,
            z,
            &RayleighSommerfeldConfig {
                padding: 4.0,
                ..cfg
            },
        );
        assert!(
            rep4.max_kernel_step_x > rep.max_kernel_step_x,
            "densification must worsen kernel sampling"
        );

        // Non-strict proceeds with the flagged diagnostics.
        let loose = RayleighSommerfeldConfig {
            strict: false,
            ..cfg
        };
        assert!(propagate(&band, z, &loose).is_ok());

        // Comfortable geometry passes the guard: coarse pitch at long z.
        let ok = sampling_report(
            &band,
            0.5,
            &RayleighSommerfeldConfig {
                padding: 2.0,
                strict: true,
            },
        );
        assert!(ok.kernel_nyquist_ok && ok.paraxial_ok);
    }

    #[test]
    fn z_zero_and_invalid_inputs_rejected() {
        let band = band_uniform(1.0, 8, 8, 20e-6, 20e-6);
        let cfg = RayleighSommerfeldConfig::default();
        assert!(matches!(
            propagate(&band, 0.0, &cfg),
            Err(RayleighSommerfeldError::InvalidDistance(0.0))
        ));
        assert!(matches!(
            propagate(&band, f64::INFINITY, &cfg),
            Err(RayleighSommerfeldError::InvalidDistance(_))
        ));
        assert!(matches!(
            propagate(&band, f64::NAN, &cfg),
            Err(RayleighSommerfeldError::InvalidDistance(_))
        ));
    }

    #[test]
    fn back_propagation_inverts_on_smooth_band() {
        // +z then -z with the conjugate kernel inverts the operator up to
        // the sampled-kernel truncation: a compact Gaussian (negligible
        // spectral replicas) at a guard-passing geometry must return to
        // itself. z=0.2 m keeps the rim tilt sin(theta) = x/r ~ 6e-3, so
        // the kernel step (~0.8 rad) passes the guard.
        let (n, d) = (128usize, 10e-6f64);
        let z = 0.2f64;
        let band = gaussian(n, n, d, 80e-6);
        let cfg = RayleighSommerfeldConfig {
            padding: 2.0,
            strict: true,
        };
        let rep = sampling_report(&band, z, &cfg);
        assert!(rep.kernel_nyquist_ok, "geometry must pass the guard: {rep}");
        assert!(rep.paraxial_ok);
        let fwd = propagate(&band, z, &cfg).unwrap();
        let rt = propagate(&fwd, -z, &cfg).unwrap();
        let rel = rel_frobenius(&rt, &band);
        // Not machine-exact: the sampled kernel's DFT has window-truncation
        // ripple (~few Fresnel zones resolved at this geometry), so |H_d|
        // deviates from 1 by ~1e-3. A sign error in the conjugate kernel
        // would fail catastrophically (O(1)), which is what this pins.
        assert!(rel < 1e-2, "round trip rel error {rel:e}");
    }
}
