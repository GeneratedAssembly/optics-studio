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
