//! Private 2D FFT (NumPy scaling convention) and frequency-axis helpers for
//! propagation engines.
//!
//! Conventions (deliberate, because shift/normalization bugs are the main
//! failure mode of spectral propagators):
//!
//! - **Scaling (NumPy convention).** [`fft2`] is unnormalized (raw rustfft);
//!   [`ifft2`] divides by $N = n_x n_y$. Round-trip is the identity, Parseval
//!   reads $\sum |U|^2 = N \sum |u|^2$, and the circular convolution theorem
//!   is **exact**: $u \circledast h = \mathcal{F}^{-1}\{\mathcal{F}\{u\}
//!   \cdot \mathcal{F}\{h\}\}$ with no stray factors — the reason Phase 2
//!   adopts this convention for convolution-form propagators and PSF analysis
//!   (check-in 2026-10-11, superseding the Phase-1 unitary scaling). Note
//!   rustfft is unnormalized in **both** directions (its raw round-trip
//!   returns $N u$), so `ifft2` adds the $1/N$ that NumPy's inverse applies.
//!   Transfer-function propagation (ASM, Fresnel TF form) is unaffected: the
//!   factors cancel in $\mathcal{F}^{-1}\{H \cdot \mathcal{F}\{u\}\}$.
//! - **Sign.** The forward kernel is $e^{-i 2\pi k n / N}$ (rustfft and NumPy
//!   convention), so a discrete plane wave $x_n = e^{+i 2\pi k_0 n/N}$ has its
//!   spectral mass in bin $+k_0$.
//! - **Bin order.** [`fft2`] output is in standard (unshifted) DFT bin order:
//!   bin $k$ corresponds to frequency $f_k = k / (N \Delta)$ with $k$ wrapped
//!   negative above the Nyquist index — see [`fftfreq`]. [`fftfreq_shifted`]
//!   returns the same axis in ascending `fftshift` order for building shifted
//!   transfer functions.
//! - **Shift semantics.** [`fftshift1d`] matches NumPy `fftshift` (roll by
//!   $\lfloor n/2 \rfloor$), moving the zero-frequency bin to the array center;
//!   [`ifftshift1d`] is its exact inverse (identical for even `n`).
//!
//! All buffers are row-major flat slices of length `nx * ny` where `nx` runs
//! along x (columns) and `ny` along y (rows); this layer is representation-
//! independent and knows nothing about [`crate::field::Field`].

use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

/// Unnormalized forward 2D DFT of a row-major buffer, in place (NumPy
/// convention: forward carries no scaling).
///
/// `nx` samples per row (x), `ny` rows (y). Requires `buf.len() == nx * ny`.
pub(crate) fn fft2(buf: &mut [Complex64], nx: usize, ny: usize) {
    let mut planner = FftPlanner::new();
    let plan_x = planner.plan_fft_forward(nx);
    let plan_y = planner.plan_fft_forward(ny);
    transform2(buf, nx, ny, plan_x.as_ref(), plan_y.as_ref(), None);
}

/// Inverse 2D DFT of a row-major buffer, in place, scaled by $1 / (n_x n_y)$
/// (NumPy convention) — the exact inverse of [`fft2`].
///
/// With this scaling the circular convolution theorem is exact:
/// $\mathcal{F}^{-1}\{\mathcal{F}\{u\} \cdot \mathcal{F}\{h\}\} = u \circledast h$.
pub(crate) fn ifft2(buf: &mut [Complex64], nx: usize, ny: usize) {
    let mut planner = FftPlanner::new();
    let plan_x = planner.plan_fft_inverse(nx);
    let plan_y = planner.plan_fft_inverse(ny);
    transform2(
        buf,
        nx,
        ny,
        plan_x.as_ref(),
        plan_y.as_ref(),
        Some(1.0 / (nx * ny) as f64),
    );
}

/// Applies the two 1D plans (x over contiguous rows, y over strided columns),
/// optionally scaling the whole buffer afterward (the $1/N$ of the NumPy
/// convention inverse).
///
/// rustfft is unnormalized in BOTH directions: raw inverse(raw forward(x)) =
/// N * x (unlike NumPy, whose inverse divides by N). `ifft2` therefore passes
/// the 1/N here; `fft2` passes `None`.
fn transform2(
    buf: &mut [Complex64],
    nx: usize,
    ny: usize,
    plan_x: &dyn Fft<f64>,
    plan_y: &dyn Fft<f64>,
    scale: Option<f64>,
) {
    assert_eq!(buf.len(), nx * ny, "fft buffer length mismatch");
    assert!(nx > 0 && ny > 0, "fft dimensions must be positive");

    // Rows (x axis) are contiguous; transform in place.
    for iy in 0..ny {
        let row = &mut buf[iy * nx..(iy + 1) * nx];
        plan_x.process(row);
    }

    // Columns (y axis) are strided; gather, transform, scatter.
    let mut col: Vec<Complex64> = vec![Complex64::ZERO; ny];
    for ix in 0..nx {
        for (r, v) in col.iter_mut().enumerate() {
            *v = buf[r * nx + ix];
        }
        plan_y.process(&mut col);
        for (r, v) in col.iter().enumerate().take(ny) {
            buf[r * nx + ix] = *v;
        }
    }

    if let Some(s) = scale {
        let s = Complex64::from(s);
        for v in buf.iter_mut() {
            *v *= s;
        }
    }
}

/// Frequency axis (cycles per meter) of a length-`n` axis sampled with pitch
/// `d` (meters), in standard unshifted DFT bin order matching [`fft2`] output.
///
/// $f_k = \dfrac{k}{n\,\Delta}$ for $k = 0 \ldots \left\lfloor \frac{n-1}{2} \right\rfloor$,
/// then negative wrap-around $k < 0$ for the remaining bins.
pub(crate) fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let kk = if k <= (n - 1) / 2 {
                k as f64
            } else {
                k as f64 - n as f64
            };
            kk / (n as f64 * d)
        })
        .collect()
}

/// Frequency axis in ascending `fftshift` order (zero frequency at the center
/// sample for odd `n`, just right of center for even `n`).
// Consumed by spectrum reporting/visualization paths (Phase 1+ public surface); intentionally unused by the ASM transfer function, which works in unshifted bin order.
#[allow(dead_code)]
pub(crate) fn fftfreq_shifted(n: usize, d: f64) -> Vec<f64> {
    let half = n / 2; // floor; shifted order starts at k = -half
    (0..n)
        .map(|i| (i as f64 - half as f64) / (n as f64 * d))
        .collect()
}

/// NumPy-`fftshift` semantics on a 1D slice: roll right by `n / 2` (floor),
/// moving index 0 (DC bin) to the center.
// Consumed by spectrum reporting/visualization paths (Phase 1+ public surface); intentionally unused by the ASM transfer function, which works in unshifted bin order.
#[allow(dead_code)]
pub(crate) fn fftshift1d(v: &[Complex64]) -> Vec<Complex64> {
    let n = v.len();
    let s = n / 2;
    (0..n).map(|i| v[(i + n - s) % n]).collect()
}

/// Exact inverse of [`fftshift1d`] (identical to it for even `n`).
// Consumed by spectrum reporting/visualization paths (Phase 1+ public surface); intentionally unused by the ASM transfer function, which works in unshifted bin order.
#[allow(dead_code)]
pub(crate) fn ifftshift1d(v: &[Complex64]) -> Vec<Complex64> {
    let n = v.len();
    let s = n / 2;
    (0..n).map(|i| v[(i + s) % n]).collect()
}

/// [`fftshift1d`] applied along both axes of a row-major `nx` x `ny` buffer.
// Consumed by spectrum reporting/visualization paths (Phase 1+ public surface); intentionally unused by the ASM transfer function, which works in unshifted bin order.
#[allow(dead_code)]
pub(crate) fn fftshift2(buf: &[Complex64], nx: usize, ny: usize) -> Vec<Complex64> {
    shift2(buf, nx, ny, false)
}

/// [`ifftshift1d`] applied along both axes of a row-major `nx` x `ny` buffer.
// Consumed by spectrum reporting/visualization paths (Phase 1+ public surface); intentionally unused by the ASM transfer function, which works in unshifted bin order.
#[allow(dead_code)]
pub(crate) fn ifftshift2(buf: &[Complex64], nx: usize, ny: usize) -> Vec<Complex64> {
    shift2(buf, nx, ny, true)
}

fn shift2(buf: &[Complex64], nx: usize, ny: usize, inverse: bool) -> Vec<Complex64> {
    assert_eq!(buf.len(), nx * ny);
    let (sx, sy) = (nx / 2, ny / 2);
    let mut out = vec![Complex64::ZERO; nx * ny];
    for iy in 0..ny {
        for ix in 0..nx {
            // fftshift: out[i] = x[(i + n - s) % n]; ifftshift: out[i] = x[(i + s) % n]
            let (fx, fy) = if inverse {
                ((ix + sx) % nx, (iy + sy) % ny)
            } else {
                ((ix + nx - sx) % nx, (iy + ny - sy) % ny)
            };
            out[iy * nx + ix] = buf[fy * nx + fx];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Deterministic pseudo-random complex buffer (no external rng).
    fn noise(len: usize) -> Vec<Complex64> {
        let mut state = 0x2545F491_u64;
        (0..len)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let re = (state as f64 / u64::MAX as f64) - 0.5;
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let im = (state as f64 / u64::MAX as f64) - 0.5;
                Complex64::new(re, im)
            })
            .collect()
    }

    fn max_abs_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn roundtrip_is_identity_even_and_odd() {
        for (nx, ny) in [(8, 8), (6, 7), (1, 5), (5, 1)] {
            let u = noise(nx * ny);
            let mut buf = u.clone();
            fft2(&mut buf, nx, ny);
            ifft2(&mut buf, nx, ny);
            assert!(
                max_abs_diff(&buf, &u) < 1e-12,
                "roundtrip failed for {nx}x{ny}"
            );
        }
    }

    #[test]
    fn parseval_energy_scales_by_n() {
        // NumPy convention => sum |fft2(u)|^2 == N * sum |u|^2.
        let (nx, ny) = (9, 6);
        let n = (nx * ny) as f64;
        let u = noise(nx * ny);
        let energy_in: f64 = u.iter().map(|s| s.norm_sqr()).sum();
        let mut buf = u;
        fft2(&mut buf, nx, ny);
        let energy_out: f64 = buf.iter().map(|s| s.norm_sqr()).sum();
        let expected = n * energy_in;
        assert!((energy_out - expected).abs() < 1e-9 * expected.max(1.0));
    }

    #[test]
    fn delta_gives_flat_unit_spectrum() {
        // Delta at index (0, 0): unnormalized forward DFT => every bin = 1.
        let (nx, ny) = (8, 4);
        let mut buf = vec![Complex64::ZERO; nx * ny];
        buf[0] = Complex64::new(1.0, 0.0);
        fft2(&mut buf, nx, ny);
        assert!(
            buf.iter()
                .all(|s| (s - Complex64::new(1.0, 0.0)).norm() < 1e-12)
        );
    }

    #[test]
    fn convolution_theorem_is_exact() {
        // NumPy scaling pins F^{-1}{F{u} * F{h}} = circular convolution of
        // u and h (textbook Eq. 2.23-2.24 form) with NO stray sqrt(N) factor.
        let (nx, ny) = (5, 4);
        let u = noise(nx * ny);
        let h = noise(nx * ny);
        let mut fu = u.clone();
        let mut fh = h.clone();
        fft2(&mut fu, nx, ny);
        fft2(&mut fh, nx, ny);
        let mut prod: Vec<Complex64> = fu.iter().zip(&fh).map(|(a, b)| *a * *b).collect();
        ifft2(&mut prod, nx, ny);
        // Reference 2D circular convolution, direct sum.
        for iy in 0..ny {
            for ix in 0..nx {
                let mut acc = Complex64::ZERO;
                for jy in 0..ny {
                    for jx in 0..nx {
                        let ux = u[jy * nx + jx];
                        let hx = h[((iy + ny - jy) % ny) * nx + ((ix + nx - jx) % nx)];
                        acc += ux * hx;
                    }
                }
                let diff = (prod[iy * nx + ix] - acc).norm();
                assert!(
                    diff < 1e-9 * acc.norm().max(1.0),
                    "convolution mismatch at {ix},{iy}"
                );
            }
        }
    }

    #[test]
    fn plane_wave_energy_sits_in_positive_bin() {
        // Sign convention: x_n = e^{+i 2 pi k0 n / N} concentrates at bin +k0
        // of the unshifted forward transform (e^{-i 2 pi k n / N} kernel).
        // Unnormalized forward: unit-amplitude ramp of length N lands with
        // magnitude N in that single bin.
        let (nx, ny) = (16, 4);
        let k0 = 3usize;
        let mut buf = vec![Complex64::ZERO; nx * ny];
        for (i, v) in buf.iter_mut().enumerate() {
            let n = (i % nx) as f64;
            *v = (2.0 * PI * k0 as f64 * n / nx as f64 * Complex64::I).exp();
        }
        fft2(&mut buf, nx, ny);
        // y axis is constant => energy only at the y-DC row (iy = 0).
        assert!((buf[k0] - Complex64::from(nx as f64 * ny as f64)).norm() < 1e-9);
        let total: f64 = buf.iter().map(|s| s.norm()).sum::<f64>() - nx as f64 * ny as f64;
        assert!(total.abs() < 1e-6, "spectral mass leaked out of bin +k0");
    }

    #[test]
    fn fftfreq_matches_numpy_semantics() {
        let f4 = fftfreq(4, 1.0);
        assert_eq!(f4, vec![0.0, 0.25, -0.5, -0.25]);
        let f5 = fftfreq(5, 1.0);
        assert_eq!(f5, vec![0.0, 0.2, 0.4, -0.4, -0.2]);
        let s4 = fftfreq_shifted(4, 1.0);
        assert_eq!(s4, vec![-0.5, -0.25, 0.0, 0.25]);
        let s5 = fftfreq_shifted(5, 1.0);
        assert_eq!(s5, vec![-0.4, -0.2, 0.0, 0.2, 0.4]);
        // Shifted axis is exactly fftshift of the unshifted axis.
        let shifted_from_unshifted: Vec<f64> = {
            let f = fftfreq(5, 1.0);
            let n = f.len();
            let s = n / 2;
            (0..n).map(|i| f[(i + n - s) % n]).collect()
        };
        assert_eq!(shifted_from_unshifted, s5);
    }

    #[test]
    fn shifts_follow_numpy_roll_convention() {
        let v: Vec<Complex64> = (0..4).map(|i| Complex64::new(i as f64, 0.0)).collect();
        // fftshift even: [2,3,0,1]; ifftshift identical for even n.
        assert_eq!(fftshift1d(&v)[0].re, 2.0);
        assert_eq!(fftshift1d(&v), ifftshift1d(&v));
        let w: Vec<Complex64> = (0..5).map(|i| Complex64::new(i as f64, 0.0)).collect();
        // fftshift odd (roll +2): [3,4,0,1,2]; ifftshift (roll -2): [2,3,4,0,1].
        let fs = fftshift1d(&w);
        let mut re = [0.0; 5];
        for (i, r) in re.iter_mut().enumerate() {
            *r = fs[i].re;
        }
        assert_eq!(re, [3.0, 4.0, 0.0, 1.0, 2.0]);
        let is = ifftshift1d(&w);
        let mut re = [0.0; 5];
        for (i, r) in re.iter_mut().enumerate() {
            *r = is[i].re;
        }
        assert_eq!(re, [2.0, 3.0, 4.0, 0.0, 1.0]);
        // Exact inverses for both parities.
        for src in [&v, &w] {
            assert_eq!(&ifftshift1d(&fftshift1d(src)), src);
            assert_eq!(&fftshift1d(&ifftshift1d(src)), src);
        }
    }

    #[test]
    fn shift2_roundtrips_and_moves_dc_to_center() {
        let (nx, ny) = (4, 3);
        let u = noise(nx * ny);
        assert_eq!(ifftshift2(&fftshift2(&u, nx, ny), nx, ny), u);
        // DC bin (index 0) lands at (nx/2, ny/2) = (2, 1) after fftshift2.
        let mut spec = vec![Complex64::ZERO; nx * ny];
        spec[0] = Complex64::new(7.0, 0.0);
        let shifted = fftshift2(&spec, nx, ny);
        let center = (ny / 2) * nx + nx / 2; // (ix=2, iy=1)
        assert_eq!(shifted[center], Complex64::new(7.0, 0.0));
    }
}
