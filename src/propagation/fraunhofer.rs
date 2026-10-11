//! Fraunhofer (far-field) propagation.
//!
//! When the observation distance is large compared with the source extent
//! (textbook assumption 9, "the distance from the aperture plane to the
//! evaluation plane is very large compared with the aperture size"), the
//! Fresnel integral degenerates to a pure Fourier transform of the aperture
//! field, evaluated at spatial frequencies mapped to physical far-field
//! coordinates and wrapped in a quadratic phase (textbook Sec. 2.2,
//! Eq. 2.19):
//!
//! $$U(x, y; z) = \frac{e^{ikz}}{i\lambda z}\,
//!   e^{\,i\pi (x^2 + y^2)/(\lambda z)}\,
//!   \widetilde{U}_0\!\Big(\frac{x}{\lambda z}, \frac{y}{\lambda z}\Big),
//!   \qquad
//!   \widetilde{U}_0(f_x, f_y) = \iint U_0(\xi, \eta)\,
//!     e^{-i 2\pi (f_x \xi + f_y \eta)}\, d\xi\, d\eta.$$
//!
//! # Regime gate (the Fresnel number)
//!
//! The dropped Fresnel quadratic inside the integral contributes a phase
//! error $\pi r^2/(\lambda z)$ at source radius $r$; with the conservative
//! bound $r = D/2$ (source extent $D$ measured from the optical axis — see
//! [`FraunhoferConfig::source_pos`] — to the farthest window-edge sample)
//! this is $\varphi_{\max} = \pi N_F / 4$ in terms of the **Fresnel number**
//!
//! $$N_F = \frac{D^2}{\lambda |z|}.$$
//!
//! $N_F \lesssim 1$ is the far field (the criterion the element docs record:
//! regime is judged at propagation time from aperture, wavelength, and
//! distance). `strict` mode refuses to propagate when
//! $N_F > \texttt{max\_n\_f}$ with [`FraunhoferError::RegimeViolation`]
//! carrying the full report; remedies are a larger $|z|$, a smaller source
//! window, or — if the near field is genuinely being asked for — the Fresnel
//! or ASM engines, which are valid at any distance. `strict = false`
//! propagates anyway with `regime_ok` flagged.
//!
//! # Off-axis modeling (`source_pos`)
//!
//! The Fraunhofer formula is the only propagation form whose *prefactor*
//! references the optical axis: the chirp $e^{i\pi r_\perp^2/\lambda z}$ and
//! the frequency map $f = r_\perp/(\lambda z)$ are measured from it.
//! [`FraunhoferConfig::source_pos`] sets that axis explicitly (meters, field
//! coordinates; `None` = grid origin). The axis must be given whenever the
//! source is displaced — e.g. `CircularAperture::new(a, (x_s, y_s))` pairs
//! with `source_pos: Some((x_s, y_s))` — and is honored exactly: an input
//! linear-phase ramp $e^{i2\pi(f_s \xi + g_s \eta)}$, $f_s = x_s/(\lambda z)$,
//! re-centers the evaluated frequency grid on the axis (no interpolation),
//! and the prefactor chirp is taken about $(x_s, y_s)$. The output window
//! itself stays grid-centered; ensure it covers the axis-centered pattern.
//!
//! ASM and Fresnel need no such parameter: their operators are
//! translation-invariant (functions of frequency content or of the
//! difference $x - \xi$), so displaced sources propagate exactly as sampled.
//!
//! # Output grid
//!
//! The far-field plane is sampled at $\Delta X = \lambda |z| / (n_x \Delta
//! \xi)$, extent $n_x \Delta X = \lambda |z| / \Delta \xi$ — the DFT
//! frequency grid mapped through $x = \lambda z f$. Output dims match the
//! input dims; the pitch is recomputed (a BandField on a new grid).
//! Negative $z$ mirrors the pattern ($x = \lambda z f$ flips sign); $z = 0$
//! is rejected (the far-field map is singular).
//!
//! # Exact off-bin alignment
//!
//! Field sample coordinates are origin-centered, $(i - (n-1)/2)\Delta$,
//! which for **even** $n$ sits at *half-integer* multiples of the DFT bin
//! spacing, while raw DFT bins are integer multiples. Rather than shift by
//! half a pixel (or restrict to odd grids), the propagator evaluates the
//! offset DFT
//!
//! $$\widetilde{U}_0(f_m) = \Delta^2\,
//!   e^{\,i 2\pi c (m - c)/n}\,
//!   \mathrm{DFT}\big\{u_i\, e^{\,i 2\pi c\, i / n}\big\}_m,
//!   \qquad c = \tfrac{n-1}{2},$$
//!
//! which is algebraically the exact discrete sum
//! $\Delta^2 \sum_i u_i e^{-i 2\pi f_m \xi_i}$ at the *physical* output
//! frequencies $f_m = (m - c)/(n\Delta)$ — unit-modulus pre/post phase
//! factors, no interpolation, all parities. Concretely, the pre-phase is a
//! shift by $c$ bins (multiplication by $e^{i2\pi c i/n}$ turns bin $k$ into
//! bin $k + c$), so output sample $(i_x, i_y)$ reads bin
//! $m = \operatorname{sgn}(z)\,(i - c) + c$ of the raw transform: identity
//! for $z > 0$, mirror $n - 1 - i$ for $z < 0$. Parseval then makes the
//! far-field discrete power $\sum |U|^2 \Delta X \Delta Y$ equal the input
//! power exactly (pinned by test).
//!
//! # Notation (math ↔ code)
//!
//! Formulas are written per-axis in 1D; the code instantiates them once per
//! axis with `x`/`y` suffixes.
//!
//! | Symbol | Code | Meaning |
//! |---|---|---|
//! | $n$ | `nx`, `ny` | samples along the axis |
//! | $i$ | `ix`, `iy` | sample index (as in `Field::sample(ix, iy)`) |
//! | $c = \frac{n-1}{2}$ | `cx`, `cy` | **index of the physical origin** — the Field convention places $x = 0$ at $c$ (a sample for odd $n$, midway for even) |
//! | $m$ | `mx`, `my` | raw DFT bin index, $0 \ldots n-1$ |
//! | $\Delta$ | `dx`, `dy` | input pitch (m) |
//! | $\Delta X$ | `dxo`, `dyo` | far-field output pitch $= \lambda\|z\|/(n\Delta)$ (m) |
//! | $\xi_i = (i-c)\Delta$ | via `Field::coord` | physical input coordinate |
//! | $f_m = \frac{m-c}{n\Delta}$ | — | physical frequency held by output sample $m$ after the offset-DFT shift |

use std::f64::consts::PI;
use std::fmt;

use num_complex::Complex64;
use thiserror::Error;

use super::fft::fft2;
use crate::field::{BandField, Field};

/// Tuning knobs for [`propagate`].
#[derive(Clone, Copy, Debug)]
pub struct FraunhoferConfig {
    /// When `true` (default), refuse non-far-field geometries with
    /// [`FraunhoferError::RegimeViolation`] instead of computing a
    /// physically invalid pattern.
    pub strict: bool,
    /// Regime threshold on the Fresnel number $N_F = D^2/(\lambda |z|)$
    /// ($D$ = conservative source extent measured from the axis; see
    /// [`FraunhoferConfig::source_pos`]). Default `1.0`; tighten (e.g.
    /// `0.1`) for quantitative far-field work.
    pub max_n_f: f64,
    /// Physical position of the optical axis in field coordinates (meters).
    /// `None` places the axis at the grid origin (the default).
    ///
    /// The Fraunhofer prefactor chirp and the frequency map
    /// $f = r_\perp/(\lambda z)$ are centered on the axis, so the axis
    /// reference must be explicit whenever the source is off-axis — e.g.
    /// `CircularAperture::new(a, (x_s, y_s))` pairs with
    /// `source_pos: Some((x_s, y_s))`. The pattern is then computed about
    /// $(x_s, y_s)$: frequencies $f = (X - x_s)/(\lambda z)$ (exactly, via
    /// a linear-phase input ramp — no interpolation) and chirp
    /// $e^{i\pi((X-x_s)^2 + (Y-y_s)^2)/\lambda z}$.
    ///
    /// ASM and Fresnel need no such parameter: their operators are
    /// translation-invariant (functions of frequency differences or of
    /// $x - \xi$), so displaced sources propagate exactly as sampled.
    pub source_pos: Option<(f64, f64)>,
}

impl Default for FraunhoferConfig {
    /// Phase-2 defaults: `strict = true`, `max_n_f = 1.0`,
    /// `source_pos = None` (axis at grid origin).
    fn default() -> Self {
        Self {
            strict: true,
            max_n_f: 1.0,
            source_pos: None,
        }
    }
}

/// Regime and grid diagnostics for one Fraunhofer propagation.
///
/// All quantities SI: meters, radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FraunhoferReport {
    /// Operating wavelength (m).
    pub lambda: f64,
    /// Signed propagation distance (m).
    pub z: f64,
    /// Input grid sample counts `(nx, ny)`.
    pub input_dims: (usize, usize),
    /// Input sample pitch `(dx, dy)` (m).
    pub input_pitch: (f64, f64),
    /// Output (far-field) grid sample counts — equal to the input's.
    pub output_dims: (usize, usize),
    /// Output sample pitch `(lambda |z| / (nx dx), lambda |z| / (ny dy))` (m).
    pub output_pitch: (f64, f64),
    /// Fresnel number along x: $(2 r_x)^2 / (\lambda |z|)$ with $r_x$ the
    /// largest distance from the axis to a window edge sample.
    pub n_f_x: f64,
    /// As [`FraunhoferReport::n_f_x`] along y.
    pub n_f_y: f64,
    /// Optical-axis position used (meters, field coordinates; resolved from
    /// [`FraunhoferConfig::source_pos`], `(0, 0)` when `None`).
    pub source_pos: (f64, f64),
    /// Worst dropped-quadratic phase error bound $\pi N_F^{\max}/4$ (rad):
    /// the Fresnel curvature inside the transform integral at the window
    /// edge farthest from the axis.
    pub quad_phase_error_rad: f64,
    /// Configured threshold.
    pub max_n_f: f64,
    /// `true` when both Fresnel numbers are within `max_n_f`.
    pub regime_ok: bool,
}

impl fmt::Display for FraunhoferReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fraunhofer regime: lambda={:.3e} m, z={:.3e} m, grid={}x{} pitch={:.3e}x{:.3e} m \
             -> far pitch={:.3e}x{:.3e} m, axis=({:.3e}, {:.3e}) m, N_F x={:.3} y={:.3} (max {:.3}), \
             dropped-quadratic phase bound={:.3} rad, regime_ok={}",
            self.lambda,
            self.z,
            self.input_dims.0,
            self.input_dims.1,
            self.input_pitch.0,
            self.input_pitch.1,
            self.output_pitch.0,
            self.output_pitch.1,
            self.source_pos.0,
            self.source_pos.1,
            self.n_f_x,
            self.n_f_y,
            self.max_n_f,
            self.quad_phase_error_rad,
            self.regime_ok,
        )?;
        if !self.regime_ok {
            write!(
                f,
                " | hint: not the far field (N_F > max_n_f) — remedy: increase |z|, \
                 use a smaller source window, or switch to the Fresnel/ASM engines \
                 (valid at any distance); strict=false forces this geometry anyway"
            )?;
        }
        Ok(())
    }
}

/// Failures of the Fraunhofer propagator.
#[derive(Clone, Debug, Error)]
pub enum FraunhoferError {
    /// Propagation distance was not finite, or was zero — the far-field map
    /// $x = \lambda z f$ is singular at $z = 0$ (use the Fresnel/ASM
    /// identity or a nonzero distance).
    #[error("propagation distance must be finite and nonzero (got {0} m)")]
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
    /// The geometry is not far-field (`strict` mode). The embedded report
    /// gives the Fresnel numbers and remedies.
    #[error("far-field regime criteria not met: {report}")]
    RegimeViolation {
        /// Diagnostics describing exactly which criterion failed. Boxed to
        /// keep the `Result` error arm small (review decision 2026-10-11).
        report: Box<FraunhoferReport>,
    },
}

fn validate(band: &BandField, z: f64) -> Result<(), FraunhoferError> {
    // Engine-boundary re-validation (see `fresnel::validate` for the
    // rationale): constructors enforce invariants, `field_mut()` does not.
    if !z.is_finite() || z == 0.0 {
        return Err(FraunhoferError::InvalidDistance(z));
    }
    let lambda = band.lambda();
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(FraunhoferError::InvalidWavelength(lambda));
    }
    let (nx, ny) = band.field().dims();
    if nx == 0 || ny == 0 {
        return Err(FraunhoferError::EmptyGrid);
    }
    if band.field().samples().iter().any(|s| !s.is_finite()) {
        return Err(FraunhoferError::NonFiniteSample);
    }
    Ok(())
}

/// Regime diagnostics for a Fraunhofer propagation of `band` over `z` under
/// `cfg`, without performing the propagation.
pub fn regime_report(band: &BandField, z: f64, cfg: &FraunhoferConfig) -> FraunhoferReport {
    let (nx, ny) = band.field().dims();
    let (dx, dy) = band.field().pitch();
    let lambda = band.lambda();
    let az = z.abs();
    let (xs, ys) = cfg.source_pos.unwrap_or((0.0, 0.0));
    // Conservative source extent measured from the axis: outermost window
    // sample on each side, $r = \max(|x_{\text{edge}} - x_s|)$; the grid is
    // origin-centered so the edges are $\mp c\Delta$ and $r = c\Delta + |s|$.
    let hx = (nx as f64 - 1.0) / 2.0 * dx;
    let hy = (ny as f64 - 1.0) / 2.0 * dy;
    let rx = hx + xs.abs();
    let ry = hy + ys.abs();
    let n_f_x = (2.0 * rx).powi(2) / (lambda * az);
    let n_f_y = (2.0 * ry).powi(2) / (lambda * az);
    let n_f = n_f_x.max(n_f_y);
    FraunhoferReport {
        lambda,
        z,
        input_dims: (nx, ny),
        input_pitch: (dx, dy),
        output_dims: (nx, ny),
        output_pitch: (
            lambda * az / (nx as f64 * dx),
            lambda * az / (ny as f64 * dy),
        ),
        n_f_x,
        n_f_y,
        source_pos: (xs, ys),
        quad_phase_error_rad: PI * n_f / 4.0,
        max_n_f: cfg.max_n_f,
        regime_ok: n_f <= cfg.max_n_f,
    }
}

/// Propagates a [`BandField`] to the far-field plane at signed distance `z`
/// (meters) with the Fraunhofer (scaled Fourier transform) model; see module
/// docs for the formula, regime gate, and output grid.
///
/// The returned band lives on the far-field grid: same sample counts,
/// pitch $\lambda |z| / (n \Delta)$ (the input grid is *not* preserved —
/// this is the one propagator that resamples, because the far-field plane
/// is physically a different sampling).
///
/// # Errors
/// - [`FraunhoferError::InvalidDistance`] if `z` is not finite or is zero.
/// - [`FraunhoferError::InvalidWavelength`] / `EmptyGrid` / `NonFiniteSample`
///   as in the other engines.
/// - [`FraunhoferError::RegimeViolation`] when `cfg.strict` and the Fresnel
///   number exceeds `cfg.max_n_f`.
pub fn propagate(
    band: &BandField,
    z: f64,
    cfg: &FraunhoferConfig,
) -> Result<BandField, FraunhoferError> {
    validate(band, z)?;
    let report = regime_report(band, z, cfg);
    if cfg.strict && !report.regime_ok {
        return Err(FraunhoferError::RegimeViolation {
            report: Box::new(report),
        });
    }

    let lambda = band.lambda();
    let field = band.field();
    let (nx, ny) = field.dims();
    let (dx, dy) = field.pitch();
    let (dxo, dyo) = report.output_pitch;
    let (xs, ys) = report.source_pos;
    let k = 2.0 * PI / lambda;
    // Axis offset expressed as a frequency shift: evaluating
    // f = (X - x_s)/(lambda z) on the sample grid equals the on-grid
    // frequencies minus f_s = x_s/(lambda z) (signed z), realized exactly by
    // ramping the input with e^{+i 2 pi (f_s xi + g_s eta)}.
    let f_s = xs / (lambda * z);
    let g_s = ys / (lambda * z);

    // Offset-DFT pre-phase per axis: v_i = u_i * exp(i 2 pi c i / n),
    // with i = sample index (ix/iy) and c = origin index (cx/cy) — see the
    // Notation table in the module docs. By the modulation theorem the raw
    // DFT bin m of v holds
    // the spectrum at physical frequency f_m = (m - c)/(n Delta) — the
    // shift is built in, so bin m maps directly to output sample m and no
    // fftshift is needed. Exact for even and odd n.
    let mut buf: Vec<Complex64> = Vec::with_capacity(nx * ny);
    let cx = (nx as f64 - 1.0) / 2.0;
    let cy = (ny as f64 - 1.0) / 2.0;
    for iy in 0..ny {
        let eta = (iy as f64 - cy) * dy;
        let pre_y = (2.0 * PI * cy * iy as f64 / ny as f64 * Complex64::I).exp();
        let ramp_y = (2.0 * PI * g_s * eta * Complex64::I).exp();
        for ix in 0..nx {
            let xi = (ix as f64 - cx) * dx;
            let pre = pre_y
                * ramp_y
                * (2.0 * PI * (cx * ix as f64 / nx as f64 + f_s * xi) * Complex64::I).exp();
            buf.push(field.sample(ix, iy) * pre);
        }
    }

    fft2(&mut buf, nx, ny);

    // Map bins to the origin-centered far-field grid. Output sample (ix, iy)
    // sits at X = (ix - cx) * dxo; the frequency asked for is
    // f = (X - xs)/(lambda z) = sgn * (ix - cx)/(nx dx) - f_s — handled by
    // the input ramp — so bin m = sgn * (ix - cx) + cx: an exact integer,
    // identity for z > 0, mirror (n-1-ix) for z < 0.
    let sgn = if z > 0.0 { 1.0 } else { -1.0 };
    let quad = (k * z * Complex64::I).exp() / (Complex64::I * lambda * z);
    let mut out: Vec<Complex64> = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        let dyp = (iy as f64 - cy) * dyo - ys;
        let my = (sgn * (iy as f64 - cy) + cy).round() as usize;
        let post_y = (2.0 * PI * cy * (my as f64 - cy) / ny as f64 * Complex64::I).exp();
        for ix in 0..nx {
            let dxp = (ix as f64 - cx) * dxo - xs;
            let mx = (sgn * (ix as f64 - cx) + cx).round() as usize;
            // Post-phase completes the exact sum
            // Delta^2 * sum_i u_i e^{-i 2 pi (fx xi + fy eta)}.
            let post = post_y * (2.0 * PI * cx * (mx as f64 - cx) / nx as f64 * Complex64::I).exp();
            // Prefactor chirp is centered on the axis at (xs, ys).
            let chirp = (PI * (dxp * dxp + dyp * dyp) / (lambda * z) * Complex64::I).exp();
            out.push(quad * chirp * post * buf[my * nx + mx] * (dx * dy));
        }
    }

    let field_out =
        Field::new(out, nx, ny, dxo, dyo).map_err(|_| FraunhoferError::NonFiniteSample)?;
    BandField::new(lambda, field_out).map_err(|_| FraunhoferError::InvalidWavelength(lambda))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAMBDA: f64 = 500e-9;

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
    fn delta_at_origin_gives_flat_amplitude_quadratic_phase() {
        // FT of delta at the physical origin is the constant Delta^2, so the
        // far field is exactly the prefactor: |U| = dx*dy/(lambda z) flat,
        // phase = kz - pi/2 + pi r^2/(lambda z). Odd grid puts a sample at 0.
        let n = 15usize;
        let d = 2e-5;
        let z = 4.0;
        let mut samples = vec![Complex64::ZERO; n * n];
        samples[(n / 2) * n + n / 2] = Complex64::new(1.0, 0.0);
        let band = BandField::new(LAMBDA, Field::new(samples, n, n, d, d).unwrap()).unwrap();
        let out = propagate(&band, z, &FraunhoferConfig::default()).unwrap();
        let k = 2.0 * PI / LAMBDA;
        let amp = d * d / (LAMBDA * z);
        let (dxo, dyo) = (LAMBDA * z / (n as f64 * d), LAMBDA * z / (n as f64 * d));
        for iy in 0..n {
            for ix in 0..n {
                let s = out.field().sample(ix, iy);
                assert!((s.norm() - amp).abs() < 1e-12, "amplitude not flat");
                let x = (ix as f64 - n as f64 / 2.0 + 0.5) * dxo;
                let y = (iy as f64 - n as f64 / 2.0 + 0.5) * dyo;
                let expected =
                    ((k * z - PI / 2.0 + PI * (x * x + y * y) / (LAMBDA * z)) * Complex64::I).exp()
                        * amp;
                assert!((s - expected).norm() < 1e-9, "prefactor wrong at {ix},{iy}");
            }
        }
    }

    #[test]
    fn uniform_input_concentrates_at_dc() {
        // Windowed plane wave: its FT samples hit the rect's exact zeros at
        // all nonzero harmonics -> single spike at X = Y = 0, magnitude
        // A * (n d)^2 / (lambda z) (window area over lambda z).
        let n = 15usize;
        let d = 2e-5;
        let z = 4.0;
        let a = 0.75;
        let band = BandField::uniform(a, 0.0, n, n, d, d, LAMBDA).unwrap();
        let out = propagate(&band, z, &FraunhoferConfig::default()).unwrap();
        let center = (n / 2) * n + n / 2;
        let expected = a * (n as f64 * d).powi(2) / (LAMBDA * z);
        assert!((out.field().samples()[center].norm() - expected).abs() < 1e-9);
        for (i, s) in out.field().samples().iter().enumerate() {
            if i != center {
                assert!(s.norm() < 1e-9, "leakage at bin {i}: {}", s.norm());
            }
        }
    }

    #[test]
    fn far_field_power_equals_input_power() {
        // Discrete Parseval through the offset DFT + unit-modulus phases +
        // the Delta^2 / (lambda z) scaling makes the far-field grid power
        // integral exactly the input power.
        let n = 64usize;
        let d = 1e-5;
        let z = 2.0; // N_F = (n d)^2/(lambda z) = 4.1e-7/1e-6 = 0.41 < 1
        let band = gaussian(n, n, d, 5.0 * d);
        let rep = regime_report(&band, z, &FraunhoferConfig::default());
        assert!(rep.regime_ok);
        let out = propagate(&band, z, &FraunhoferConfig::default()).unwrap();
        let rel = ((out.field().power() - band.field().power()) / band.field().power()).abs();
        assert!(rel < 1e-9, "far-field power rel. error {rel:e}");
    }

    #[test]
    fn regime_gate_blocks_near_field_geometry() {
        let n = 64usize;
        let d = 1e-5;
        let band = gaussian(n, n, d, 5.0 * d);
        let z = 0.01; // N_F = 4.1e-7 / 5e-9 = 82 >> 1
        let cfg = FraunhoferConfig::default();
        match propagate(&band, z, &cfg) {
            Err(FraunhoferError::RegimeViolation { report }) => {
                assert!(!report.regime_ok);
                assert!(report.n_f_x > 1.0);
            }
            other => panic!("expected RegimeViolation, got {other:?}"),
        }
        // Non-strict proceeds with the flag set.
        let loose = FraunhoferConfig {
            strict: false,
            ..cfg
        };
        let out = propagate(&band, z, &loose).unwrap();
        assert_eq!(out.field().dims(), band.field().dims());
        assert!(!regime_report(&band, z, &loose).regime_ok);
        // Threshold is configurable: N_F = 0.41 at z = 2 m fails a tightened
        // cap of 0.2 but passes at z = 5 m (N_F = 0.164).
        let tight = FraunhoferConfig {
            max_n_f: 0.2,
            ..cfg
        };
        assert!(!regime_report(&band, 2.0, &tight).regime_ok);
        assert!(regime_report(&band, 5.0, &tight).regime_ok);
    }

    #[test]
    fn negative_z_mirrors_the_pattern() {
        let n = 64usize;
        let d = 1e-5;
        let z = 2.0;
        let band = gaussian(n, n, d, 5.0 * d);
        let fwd = propagate(&band, z, &FraunhoferConfig::default()).unwrap();
        let back = propagate(&band, -z, &FraunhoferConfig::default()).unwrap();
        for iy in 0..n {
            for ix in 0..n {
                let a = fwd.field().sample(ix, iy);
                let b = back.field().sample(n - 1 - ix, n - 1 - iy);
                // Mirror with conjugate chirp phase: magnitudes equal.
                assert!((a.norm() - b.norm()) < 1e-12);
            }
        }
    }

    #[test]
    fn invalid_inputs_rejected() {
        let band = BandField::uniform(1.0, 0.0, 8, 8, 20e-6, 20e-6, LAMBDA).unwrap();
        let cfg = FraunhoferConfig {
            strict: false,
            ..Default::default()
        };
        assert!(matches!(
            propagate(&band, 0.0, &cfg),
            Err(FraunhoferError::InvalidDistance(0.0))
        ));
        assert!(matches!(
            propagate(&band, f64::NAN, &cfg),
            Err(FraunhoferError::InvalidDistance(_))
        ));
    }

    fn gaussian_at(nx: usize, ny: usize, d: f64, sigma: f64, c: (f64, f64)) -> BandField {
        let mut samples = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * d - c.1;
            for ix in 0..nx {
                let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * d - c.0;
                samples.push(Complex64::from(
                    (-((x * x + y * y) / (2.0 * sigma * sigma))).exp(),
                ));
            }
        }
        BandField::new(LAMBDA, Field::new(samples, nx, ny, d, d).unwrap()).unwrap()
    }

    #[test]
    fn source_pos_none_equals_explicit_origin() {
        let band = gaussian(64, 64, 1e-5, 5e-5);
        let a = propagate(
            &band,
            2.0,
            &FraunhoferConfig {
                source_pos: None,
                ..Default::default()
            },
        )
        .unwrap();
        let b = propagate(
            &band,
            2.0,
            &FraunhoferConfig {
                source_pos: Some((0.0, 0.0)),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(a.field(), b.field());
    }

    #[test]
    fn off_axis_propagation_conserves_power() {
        // The axis ramp e^{i 2 pi (f_s xi + g_s eta)} and the prefactor chirp
        // are unit-modulus, and the bin map is a bijection, so far-field
        // power is conserved for any source_pos — an exact check that the
        // off-axis machinery preserves spectral magnitude while re-centering
        // the frequency grid on the axis.
        let s = (1e-4, -5e-5);
        let z = 2.0;
        let cfg_axis = FraunhoferConfig {
            source_pos: Some(s),
            ..Default::default()
        };
        let shifted = gaussian_at(64, 64, 1e-5, 5e-5, s);
        let centered = gaussian(64, 64, 1e-5, 5e-5);
        let rep = regime_report(&shifted, z, &cfg_axis);
        assert!(rep.regime_ok); // N_F = (2*(3.15e-4 + 1e-4))^2 / 1e-6 = 0.689
        let out_s = propagate(&shifted, z, &cfg_axis).unwrap();
        let rel =
            ((out_s.field().power() - shifted.field().power()) / shifted.field().power()).abs();
        assert!(rel < 1e-9, "off-axis power rel. error {rel:e}");
        // Inputs have equal power (both ~ fully inside the window), so the
        // far-field patterns do too — while the complex fields differ.
        let out_c = propagate(&centered, z, &FraunhoferConfig::default()).unwrap();
        let rel2 = ((out_s.field().power() - out_c.field().power()) / out_c.field().power()).abs();
        assert!(rel2 < 1e-9, "shifted vs centered far-field power {rel2:e}");
    }

    #[test]
    fn regime_extent_is_measured_from_the_axis() {
        // Same geometry; moving the axis toward a window edge grows the
        // conservative extent and can push N_F past the threshold.
        let band = gaussian(64, 64, 1e-5, 5e-5);
        let cfg = FraunhoferConfig {
            max_n_f: 0.5,
            ..Default::default()
        };
        assert!(regime_report(&band, 2.0, &cfg).regime_ok); // N_F = 0.397
        let off = FraunhoferConfig {
            source_pos: Some((2e-4, 0.0)),
            ..cfg
        };
        let rep = regime_report(&band, 2.0, &off);
        assert!(!rep.regime_ok); // N_F = (2*(3.15e-4+2e-4))^2/1e-6 = 1.06
        assert_eq!(rep.source_pos, (2e-4, 0.0));
        assert!(matches!(
            propagate(&band, 2.0, &off),
            Err(FraunhoferError::RegimeViolation { .. })
        ));
        // The displaced source is also off-center in the window: its energy
        // still propagates exactly under ASM/Fresnel (translation invariance)
        // — cross-checked in the Phase-2 integration suite.
    }
}
