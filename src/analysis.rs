//! Detector-plane analysis metrics for intensity grids (Phase 7 task 7.3).
//!
//! Pure functions over [`Psf`] — no propagation, no elements (data/algorithm
//! split holds: this module consumes the same intensity grid the engines and
//! `beamline::psf` produce, and incoherent multiband sums via
//! [`crate::field::sum_intensity`]).
//!
//! Metrics (imaging-analysis anchors, cf. `docs/physics.md` §9.6–9.7):
//!
//! - [`encircled_energy`]: cumulative power fraction $EE(r)$ with inverse
//!   lookup `radius_for_fraction` — Airy anchor: $EE(r_1) \approx 0.84$ at
//!   the first dark ring $r_1 = 1.22\,\lambda z/2a$.
//! - [`spot_rms_radius`]: intensity-weighted second moment
//!   $\sqrt{\langle \rho^2 \rangle}$ (Gaussian anchor: $= w$ for
//!   $I = e^{-\rho^2/w^2}$).
//! - [`mtf`]: modulation transfer function $|F\{I\}|/F\{I\}(0)$ along $+x$
//!   (Gaussian anchor: $e^{-\pi^2 w^2 f^2}$; a diffraction-limited system's
//!   MTF is the pupil autocorrelation).
//! - [`strehl`]: peak-to-peak ratio of two same-power PSFs (measured vs
//!   reference — 1.0 for identical grids).
//! - [`marechal_strehl`]: $\exp(-(2\pi\sigma_{WV})^2)$ from wavefront RMS
//!   in waves (pairs with `zernike` decomposition, task 7.4; valid
//!   $\sigma \lesssim \lambda/14$).

use crate::field::Psf;
use crate::propagation::fft::fft2;
use num_complex::Complex64;

/// Cumulative encircled-energy curve $EE(r)$ of a [`Psf`].
///
/// Built from a radius-sorted pixel walk (exact discrete sums — no
/// interpolation of the pixels themselves); [`EncircledEnergy::value_at`]
/// and [`EncircledEnergy::radius_for_fraction`] linearly interpolate the
/// curve.
#[derive(Clone, Debug, PartialEq)]
pub struct EncircledEnergy {
    radii: Vec<f64>,
    fraction: Vec<f64>,
}

impl EncircledEnergy {
    /// Radii (m, ascending) and cumulative fractions (0..=1).
    pub fn curve(&self) -> (&[f64], &[f64]) {
        (&self.radii, &self.fraction)
    }

    /// Fraction of total power within radius `r` (m).
    pub fn value_at(&self, r: f64) -> f64 {
        interp(&self.radii, &self.fraction, r)
    }

    /// Radius (m) at which the cumulative fraction first reaches `f`
    /// (0..=1) — e.g. `radius_for_fraction(0.84)` locates the Airy dark
    /// ring, `radius_for_fraction(0.5)` the median spot radius.
    pub fn radius_for_fraction(&self, f: f64) -> f64 {
        interp(&self.fraction, &self.radii, f.clamp(0.0, 1.0))
    }
}

fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.is_empty() || x <= xs[0] {
        return ys.first().copied().unwrap_or(0.0);
    }
    if x >= *xs.last().unwrap() {
        return *ys.last().unwrap();
    }
    let k = match xs.binary_search_by(|v| v.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(k) => return ys[k],
        Err(k) => k,
    };
    let (x0, x1, y0, y1) = (xs[k - 1], xs[k], ys[k - 1], ys[k]);
    if x1 == x0 {
        return y1;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Encircled-energy curve of `psf` out to the grid corner radius.
pub fn encircled_energy(psf: &Psf) -> EncircledEnergy {
    let (nx, ny) = psf.dims();
    let mut pixels: Vec<(f64, f64)> = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        for ix in 0..nx {
            let (x, y) = psf.coord(ix, iy);
            pixels.push(((x * x + y * y).sqrt(), psf.samples()[iy * nx + ix]));
        }
    }
    pixels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let total: f64 = pixels.iter().map(|(_, i)| *i).sum();
    let (mut radii, mut fraction) = (
        Vec::with_capacity(pixels.len()),
        Vec::with_capacity(pixels.len()),
    );
    if total > 0.0 {
        let mut acc = 0.0;
        for idx in 0..pixels.len() {
            acc += pixels[idx].1;
            // Emit at the END of each tied-radius group so the cumulative
            // sum covers every pixel at r <= this radius.
            if idx + 1 == pixels.len() || pixels[idx + 1].0 != pixels[idx].0 {
                radii.push(pixels[idx].0);
                fraction.push(acc / total);
            }
        }
    }
    EncircledEnergy { radii, fraction }
}

/// Intensity-weighted second-moment spot radius
/// $\sqrt{\sum I \rho^2 / \sum I}$ (meters).
pub fn spot_rms_radius(psf: &Psf) -> f64 {
    let (nx, ny) = psf.dims();
    let (mut sw, mut swr2) = (0.0f64, 0.0f64);
    for iy in 0..ny {
        for ix in 0..nx {
            let (x, y) = psf.coord(ix, iy);
            let i = psf.samples()[iy * nx + ix];
            sw += i;
            swr2 += i * (x * x + y * y);
        }
    }
    if sw > 0.0 { (swr2 / sw).sqrt() } else { 0.0 }
}

/// One-dimensional MTF sample $(f, \mathrm{MTF})$ along $+f_x$ (cycles/m,
/// normalized to 1 at DC).
#[derive(Clone, Debug, PartialEq)]
pub struct Mtf {
    pub freqs: Vec<f64>,
    pub values: Vec<f64>,
}

/// Modulation transfer function of `psf` along $x$: $|F\{I\}(f_x, 0)|$
/// normalized by the DC value $\sum I$. Frequencies up to Nyquist
/// ($1 / (2\Delta x)$); the zero-frequency bin is first.
pub fn mtf(psf: &Psf) -> Mtf {
    let (nx, ny) = psf.dims();
    let (dx, _dy) = psf.pitch();
    let mut buf: Vec<Complex64> = psf
        .samples()
        .iter()
        .map(|v| Complex64::new(*v, 0.0))
        .collect();
    fft2(&mut buf, nx, ny);
    let dc = buf[0].norm().max(f64::MIN_POSITIVE);
    let half = nx / 2;
    let freqs = (0..=half).map(|k| k as f64 / (nx as f64 * dx)).collect();
    // fy = 0 is DFT row 0 (unshifted bin order); bins 0..nx/2 are +fx.
    let values = (0..=half).map(|k| (buf[k].norm() / dc).min(1.0)).collect();
    Mtf { freqs, values }
}

/// Strehl ratio of `measured` against a `reference` PSF: peak ratio after
/// equalizing total power (1.0 for identical shapes; < 1 when `measured`
/// is broader/aberrated).
///
/// For a wavefront-error estimate without a reference grid, use
/// [`marechal_strehl`] with the Zernike RMS from task 7.4.
pub fn strehl(measured: &Psf, reference: &Psf) -> f64 {
    let (pm, pr) = (measured.total_power(), reference.total_power());
    if pm <= 0.0 || pr <= 0.0 {
        return 0.0;
    }
    let peak = |p: &Psf| p.samples().iter().cloned().fold(0.0f64, f64::max);
    (peak(measured) / pm) / (peak(reference) / pr)
}

/// Maréchal approximation $\mathrm{Strehl} \approx e^{-(2\pi\sigma)^2}$
/// from wavefront RMS $\sigma$ **in waves** (valid $\sigma \lesssim 1/14$).
pub fn marechal_strehl(rms_waves: f64) -> f64 {
    (-(2.0 * std::f64::consts::PI * rms_waves) * (2.0 * std::f64::consts::PI * rms_waves)).exp()
}

/// Errors of [`zernike_fit`].
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum AnalysisError {
    /// Pupil radius was not positive and finite (m).
    #[error("pupil radius must be positive and finite (got {0} m)")]
    InvalidPupil(f64),
    /// Sample buffer length did not match the declared grid.
    #[error("sample buffer length {len} does not match grid size {expected}")]
    DimensionMismatch {
        /// Provided length.
        len: usize,
        /// Expected `nx * ny`.
        expected: usize,
    },
    /// No grid samples fell inside the pupil.
    #[error("pupil of radius {0} m contains no grid samples")]
    EmptyPupil(f64),
    /// The normal equations were singular (mode list too large for the
    /// sampled pupil).
    #[error("Zernike normal equations singular (n_max too high for this pupil)")]
    Singular,
}

/// One Zernike mode: radial order $n \ge 0$, azimuthal order $m$
/// ($|m| \le n$, $n - |m|$ even).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZernikeMode {
    /// Radial order.
    pub n: u32,
    /// Azimuthal order (negative selects the sin branch).
    pub m: i32,
}

/// Radial polynomial via the Jacobi representation
/// $R_n^m(r) = (-1)^{(n-m)/2} r^m P_{(n-m)/2}^{(m,0)}(1 - 2r^2)$
/// (exact for all valid $n, m$; golden-checked below).
pub fn zernike_radial(n: u32, m: u32, rho: f64) -> f64 {
    debug_assert!(m <= n && (n - m).is_multiple_of(2));
    let k = (n - m) / 2;
    let x = 1.0 - 2.0 * rho * rho;
    let p = jacobi_p(k, m, x);
    let sign = if k.is_multiple_of(2) { 1.0 } else { -1.0 };
    sign * rho.powi(m as i32) * p
}

/// Jacobi $P_k^{(\alpha,0)}(x)$ for integer $\alpha \ge 0$ via the
/// exact binomial sum $2^{-k}\sum_j \binom{k+\alpha}{j}\binom{k}{k-j}
/// (x-1)^{k-j}(x+1)^j$ (golden-checked against the classical radial
/// polynomials; no recurrence coefficients to misremember).
fn jacobi_p(k: u32, alpha: u32, x: f64) -> f64 {
    let mut s = 0.0;
    for j in 0..=k {
        s += binom(k + alpha, j)
            * binom(k, k - j)
            * (x - 1.0).powi((k - j) as i32)
            * (x + 1.0).powi(j as i32);
    }
    s / (1u64 << k) as f64
}

/// Exact integer binomial coefficient (small indices, f64 result).
fn binom(n: u32, k: u32) -> f64 {
    if k > n {
        return 0.0;
    }
    let kk = k.min(n - k);
    let mut acc = 1.0;
    for i in 0..kk {
        acc *= (n - i) as f64;
        acc /= (i + 1) as f64;
    }
    acc
}

/// Orthonormal-on-the-unit-disk Zernike value ($\int Z_i Z_j\,dA = \pi
/// \delta_{ij}$ with the $\sqrt{2(n+1)}$ normalization): radial part
/// times $\cos(m\theta)$ for $m \ge 0$, $\sin(|m|\theta)$ for $m < 0$.
pub fn zernike(mode: ZernikeMode, rho: f64, theta: f64) -> f64 {
    let m = mode.m.unsigned_abs();
    // sqrt(n+1) on-axis / sqrt(2(n+1)) off-axis: every mode has unit RMS
    // over the unit disk (int Z_i Z_j dA = pi * delta_ij).
    let norm = if m == 0 {
        (mode.n as f64 + 1.0).sqrt()
    } else {
        (2.0 * (mode.n as f64 + 1.0)).sqrt()
    };
    let r = zernike_radial(mode.n, m, rho);
    if mode.m >= 0 {
        norm * r * (m as f64 * theta).cos()
    } else {
        norm * r * (m as f64 * theta).sin()
    }
}

/// Mode list up to radial order `n_max`, ordered by $n$ then $m$ from
/// $-n$ to $+n$ in steps of 2 (explicit, not Noll — the list is returned
/// with the fit so the mapping is self-documenting).
pub fn zernike_modes(n_max: u32) -> Vec<ZernikeMode> {
    let mut out = Vec::new();
    for n in 0..=n_max {
        let mut m = -(n as i32);
        while m <= n as i32 {
            out.push(ZernikeMode { n, m });
            m += 2;
        }
    }
    out
}

/// Wavefront decomposition result: coefficients (radians) per mode,
/// residual RMS, and the Maréchal-convenient residual in waves.
#[derive(Clone, Debug, PartialEq)]
pub struct WavefrontDecomp {
    /// Fitted modes (same order as `coefficients`).
    pub modes: Vec<ZernikeMode>,
    /// Coefficients (radians): phase $\approx \sum_k c_k Z_k$.
    pub coefficients: Vec<f64>,
    /// RMS of the fit residual over the pupil (radians).
    pub residual_rms: f64,
}

impl WavefrontDecomp {
    /// Residual wavefront RMS **in waves** ($\sigma/2\pi$) — the
    /// [`marechal_strehl`] input.
    pub fn residual_rms_waves(&self) -> f64 {
        self.residual_rms / (2.0 * std::f64::consts::PI)
    }
}

/// Least-squares Zernike fit of a phase grid (radians) over a circular
/// pupil of radius `pupil` (m) centered on the grid origin. Solves the
/// normal equations with Cholesky (mode count $\le (n+1)(n+2)/2$, tiny).
///
/// # Errors
/// [`AnalysisError::InvalidPupil`], [`AnalysisError::DimensionMismatch`],
/// [`AnalysisError::EmptyPupil`], [`AnalysisError::Singular`].
pub fn zernike_fit(
    phase: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    pupil: f64,
    n_max: u32,
) -> Result<WavefrontDecomp, AnalysisError> {
    if !pupil.is_finite() || pupil <= 0.0 {
        return Err(AnalysisError::InvalidPupil(pupil));
    }
    if phase.len() != nx * ny {
        return Err(AnalysisError::DimensionMismatch {
            len: phase.len(),
            expected: nx * ny,
        });
    }
    let modes = zernike_modes(n_max);
    let k = modes.len();
    let mut ata = vec![vec![0.0f64; k]; k];
    let mut atb = vec![0.0f64; k];
    let mut npix = 0usize;
    let mut b2sum = 0.0f64; // residual accumulator base: sum of phase^2
    for iy in 0..ny {
        let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dy;
        for ix in 0..nx {
            let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx;
            let rho2 = x * x + y * y;
            if rho2 > pupil * pupil {
                continue;
            }
            npix += 1;
            let rho = rho2.sqrt() / pupil;
            let theta = y.atan2(x);
            let p = phase[iy * nx + ix];
            b2sum += p * p;
            let zs: Vec<f64> = modes.iter().map(|&m| zernike(m, rho, theta)).collect();
            for a in 0..k {
                atb[a] += zs[a] * p;
                for b in 0..k {
                    ata[a][b] += zs[a] * zs[b];
                }
            }
        }
    }
    if npix == 0 {
        return Err(AnalysisError::EmptyPupil(pupil));
    }
    let ata_saved = ata.clone();
    let c = cholesky_solve(&mut ata, &atb).ok_or(AnalysisError::Singular)?;
    // Residual squared: ||A c - b||^2 = b.b - 2 c.(A^T b) + c^T (A^T A) c
    let mut quad = b2sum;
    for a in 0..k {
        quad -= 2.0 * c[a] * atb[a];
        for b in 0..k {
            quad += c[a] * c[b] * ata_saved[a][b];
        }
    }
    let residual_rms = (quad.max(0.0) / npix as f64).sqrt();
    Ok(WavefrontDecomp {
        modes,
        coefficients: c,
        residual_rms,
    })
}

fn cholesky_solve(a: &mut [Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    // in-place Cholesky (lower)
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            s -= a[i][..j]
                .iter()
                .zip(&a[j][..j])
                .map(|(x, y)| x * y)
                .sum::<f64>();
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                a[i][i] = s.sqrt();
            } else {
                a[i][j] = s / a[j][j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        s -= a[i][..i]
            .iter()
            .zip(&y[..i])
            .map(|(x, y)| x * y)
            .sum::<f64>();
        y[i] = s / a[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        s -= a[i + 1..]
            .iter()
            .map(|row| row[i])
            .zip(&x[i + 1..])
            .map(|(v, x)| v * x)
            .sum::<f64>();
        x[i] = s / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, EllipticalAperture};
    use crate::field::BandField;
    use crate::propagation::fraunhofer::{FraunhoferConfig, propagate};
    use std::f64::consts::PI;

    const LAMBDA: f64 = 500e-9;

    fn gaussian_psf(w: f64, n: usize, d: f64) -> Psf {
        let mut samples = Vec::with_capacity(n * n);
        for iy in 0..n {
            let y = (iy as f64 - (n as f64 - 1.0) / 2.0) * d;
            for ix in 0..n {
                let x = (ix as f64 - (n as f64 - 1.0) / 2.0) * d;
                samples.push((-(x * x + y * y) / (w * w)).exp());
            }
        }
        Psf::new(samples, n, n, d, d).unwrap()
    }

    #[test]
    fn encircled_energy_of_airy_hits_084_at_first_dark_ring() {
        // Aperture a = 0.5 mm, Fraunhofer z = 0.5 m: r1 = 1.22 lambda z / 2a
        // = 305 um; grid n=128 d=50 um (window +-3.175 mm), output pitch
        // lambda z / (n d) = 39 um.
        let a = 0.5e-3;
        let z = 0.5;
        let band = BandField::uniform(1.0, 0.0, 128, 128, 50e-6, 50e-6, LAMBDA).unwrap();
        let ap = EllipticalAperture::circle(a, (0.0, 0.0)).unwrap();
        let masked = ap.apply(&band).unwrap();
        let cfg = FraunhoferConfig {
            aperture_extent: Some(a),
            ..Default::default()
        };
        let far = propagate(&masked, z, &cfg).unwrap();
        let psf = Psf::from_band(&far);
        let ee = encircled_energy(&psf);
        let r1 = 1.22 * LAMBDA * z / (2.0 * a);
        assert!(
            (ee.value_at(r1) - 0.84).abs() < 0.02,
            "EE(r1) = {}",
            ee.value_at(r1)
        );
        assert!(
            (ee.radius_for_fraction(0.84) - r1).abs() < 2.0 * psf.pitch().0,
            "r(0.84) = {:e} vs {r1:e}",
            ee.radius_for_fraction(0.84)
        );
        assert!((ee.value_at(f64::INFINITY) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_anchors_spot_rms_and_mtf() {
        // I = exp(-rho^2/w^2): <rho^2> = w^2 (2D); MTF = exp(-pi^2 w^2 f^2).
        let (w, d) = (100e-6, 10e-6);
        let psf = gaussian_psf(w, 512, d);
        let rms = spot_rms_radius(&psf);
        assert!((rms / w - 1.0).abs() < 0.01, "rms {rms:e} vs w {w:e}");

        let m = mtf(&psf);
        assert!((m.values[0] - 1.0).abs() < 1e-12);
        let mut max_err = 0.0f64;
        for (&f, &v) in m.freqs.iter().zip(&m.values).skip(1) {
            if f > 800e3 {
                break; // beyond 0.8 f_nyq the sampled-Gaussian tail is noise
            }
            let expect = (-PI * PI * w * w * f * f).exp();
            max_err = max_err.max((v - expect).abs());
        }
        assert!(max_err < 0.02, "MTF deviation {max_err}");
    }

    #[test]
    fn strehl_self_is_one_and_broader_is_less() {
        let narrow = gaussian_psf(80e-6, 256, 10e-6);
        let wide = gaussian_psf(160e-6, 256, 10e-6);
        assert!((strehl(&narrow, &narrow) - 1.0).abs() < 1e-12);
        let s = strehl(&wide, &narrow);
        assert!(s > 0.2 && s < 0.6, "wide-vs-narrow {s}"); // (80/160)^2 = 0.25
        assert!(marechal_strehl(0.1) > 0.67 && marechal_strehl(0.1) < 0.68);
        assert!(marechal_strehl(0.0) == 1.0);
    }

    #[test]
    fn zernike_radial_golden_values() {
        // Classical low-order radial polynomials.
        for rho in [0.0f64, 0.3, 0.7, 1.0] {
            assert!((zernike_radial(0, 0, rho) - 1.0).abs() < 1e-14);
            assert!((zernike_radial(1, 1, rho) - rho).abs() < 1e-14);
            assert!((zernike_radial(2, 0, rho) - (2.0 * rho * rho - 1.0)).abs() < 1e-14);
            assert!((zernike_radial(2, 2, rho) - rho * rho).abs() < 1e-14);
            assert!((zernike_radial(3, 1, rho) - (3.0 * rho.powi(3) - 2.0 * rho)).abs() < 1e-14);
            assert!(
                (zernike_radial(4, 2, rho) - (4.0 * rho.powi(4) - 3.0 * rho * rho)).abs() < 1e-14
            );
            assert!(
                (zernike_radial(4, 0, rho) - (6.0 * rho.powi(4) - 6.0 * rho * rho + 1.0)).abs()
                    < 1e-14
            );
        }
    }

    fn disk_grid(n: usize, d: f64, pupil: f64, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        let mut v = Vec::with_capacity(n * n);
        for iy in 0..n {
            let y = (iy as f64 - (n as f64 - 1.0) / 2.0) * d;
            for ix in 0..n {
                let x = (ix as f64 - (n as f64 - 1.0) / 2.0) * d;
                let r = x.hypot(y);
                v.push(if r <= pupil {
                    f(r / pupil, y.atan2(x))
                } else {
                    0.0
                });
            }
        }
        v
    }

    #[test]
    fn zernike_modes_are_discretely_orthonormal() {
        // int Zi Zj dA over the pupil = pi * pupil^2 * delta_ij (sqrt(2(n+1))
        // normalization): check the Gram matrix on a 512^2 grid.
        let (n, d, pupil) = (512usize, 40e-6, 10e-3);
        let modes = zernike_modes(4);
        let mut gram = vec![vec![0.0f64; modes.len()]; modes.len()];
        let mut zs_cache = vec![0.0f64; modes.len()];
        for iy in 0..n {
            let y = (iy as f64 - (n as f64 - 1.0) / 2.0) * d;
            for ix in 0..n {
                let x = (ix as f64 - (n as f64 - 1.0) / 2.0) * d;
                let r = x.hypot(y);
                if r > pupil {
                    continue;
                }
                let (rho, th) = (r / pupil, y.atan2(x));
                for (k, &m) in modes.iter().enumerate() {
                    zs_cache[k] = zernike(m, rho, th);
                }
                for a in 0..modes.len() {
                    for b in 0..=a {
                        gram[a][b] += zs_cache[a] * zs_cache[b] * d * d;
                    }
                }
            }
        }
        let area_norm = PI * pupil * pupil;
        for a in 0..modes.len() {
            for b in 0..=a {
                let expect = if a == b { 1.0 } else { 0.0 };
                let got = gram[a][b] / area_norm;
                assert!(
                    (got - expect).abs() < 2e-3,
                    "gram[{}][{}] = {got} vs {expect}",
                    modes[a].n,
                    modes[b].n
                );
            }
        }
    }

    #[test]
    fn zernike_fit_recovers_synthetic_wavefront() {
        // phase = 0.5 Z(2,0) + 0.3 Z(3,-1) + 0.1 Z(4,2) (radians),
        // n_max = 4 recovers coefficients exactly, residual ~ 0.
        let (n, d, pupil) = (256usize, 100e-6, 10e-3);
        let modes = [
            (ZernikeMode { n: 2, m: 0 }, 0.5),
            (ZernikeMode { n: 3, m: -1 }, 0.3),
            (ZernikeMode { n: 4, m: 2 }, 0.1),
        ];
        let phase = disk_grid(n, d, pupil, |rho, th| {
            modes.iter().map(|(m, a)| a * zernike(*m, rho, th)).sum()
        });
        let wf = zernike_fit(&phase, n, n, d, d, pupil, 4).unwrap();
        for (mode, amp) in modes {
            let k = wf
                .modes
                .iter()
                .position(|&m| m == mode)
                .expect("mode present");
            assert!(
                (wf.coefficients[k] - amp).abs() < 1e-9,
                "coeff for {mode:?}: {} vs {amp}",
                wf.coefficients[k]
            );
        }
        assert!(wf.residual_rms < 1e-9);
        // Marechal pairing: residual ~ 0 -> Strehl ~ 1.
        assert!((marechal_strehl(wf.residual_rms_waves()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zernike_fit_residual_tracks_unmodeled_spherical() {
        // Fit only up to n_max = 2 a wavefront containing 0.02 waves of
        // Z(4,0): residual RMS (waves) = 0.02 (unit-RMS modes), Maréchal
        // Strehl = exp(-(2 pi 0.02)^2) ~ 0.9843.
        let (n, d, pupil) = (256usize, 100e-6, 10e-3);
        let s4 = ZernikeMode { n: 4, m: 0 };
        let phase = disk_grid(n, d, pupil, |rho, th| {
            2.0 * PI
                * (0.05 * zernike(ZernikeMode { n: 2, m: 0 }, rho, th)
                    + 0.02 * zernike(s4, rho, th))
        });
        let wf = zernike_fit(&phase, n, n, d, d, pupil, 2).unwrap();
        assert!((wf.residual_rms_waves() - 0.02).abs() < 1e-3);
        let st = marechal_strehl(wf.residual_rms_waves());
        assert!((st - 0.9843).abs() < 2e-3, "marechal {st}");
    }

    #[test]
    fn zernike_validation() {
        let phase = vec![0.0; 16];
        assert!(matches!(
            zernike_fit(&phase, 4, 4, 1e-3, 1e-3, -1.0, 2),
            Err(AnalysisError::InvalidPupil(-1.0))
        ));
        assert!(matches!(
            zernike_fit(&phase, 4, 5, 1e-3, 1e-3, 1e-3, 2),
            Err(AnalysisError::DimensionMismatch { .. })
        ));
        // Pupil smaller than the nearest off-center sample: only the
        // origin-column pixels... use a pupil that catches zero samples.
        let big = vec![0.0; 4 * 4];
        assert!(matches!(
            zernike_fit(&big, 4, 4, 1.0, 1.0, 1e-9, 2),
            Err(AnalysisError::EmptyPupil(_) | AnalysisError::Singular)
        ));
        // One pixel of support cannot condition a 6-mode fit.
        let tiny = disk_grid(9, 1.0, 0.4, |_, _| 1.0);
        assert!(matches!(
            zernike_fit(&tiny, 9, 9, 1.0, 1.0, 0.4, 3),
            Err(AnalysisError::EmptyPupil(_) | AnalysisError::Singular)
        ));
    }

    #[test]
    fn mtf_dc_and_zero_field_safety() {
        let flat = Psf::new(vec![1.0; 64], 8, 8, 1e-3, 1e-3).unwrap();
        let m = mtf(&flat);
        // Flat field: DC only, all higher bins ~0.
        assert!(m.values[1..].iter().all(|v| *v < 1e-9));
        let zero = Psf::new(vec![0.0; 64], 8, 8, 1e-3, 1e-3).unwrap();
        assert_eq!(spot_rms_radius(&zero), 0.0);
        assert_eq!(strehl(&zero, &flat), 0.0);
        let ee = encircled_energy(&zero);
        assert_eq!(ee.value_at(1.0), 0.0);
    }
}
