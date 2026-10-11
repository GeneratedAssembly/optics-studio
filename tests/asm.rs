//! Physical validation of the ASM propagator against analytical optics.
//!
//! These tests verify *physics*, not just numerics, per the project
//! guidelines. Every expected value is derived from the textbook
//! (`diffractive_optics_design_frabication_and_test.pdf`):
//!
//! - Sec. 1.4 (Huygens/Fresnel zones): truncation at an aperture makes light
//!   spread — the diffraction structure must emerge from *propagation*, not
//!   from the element mask.
//! - Sec. 2.2 (far field = Fourier transform of the aperture function):
//!   circular aperture → Airy profile $[2J_1(v)/v]^2$, first zero at
//!   $1.22\,\lambda z / (2a)$, valid in the regime $N_F = a^2/(\lambda z) \ll 1$.
//! - Eqs. 3.5–3.13 (plane-wave basis, direction cosines): the invariance and
//!   tilt tests.
//!
//! Conventions (see `src/propagation/asm.rs` docs): all quantities SI; NumPy
//! FFT normalization; $H$ applied in unshifted DFT bin order.

use std::f64::consts::PI;

use num_complex::Complex64;
use optics_rs::element::{CircularAperture, Element};
use optics_rs::field::{BandField, Field};
use optics_rs::propagation::asm::{AsmConfig, PropagationError, propagate, sampling_report};

const LAMBDA: f64 = 500e-9; // meters (green)

/// Wrap a phase difference (rad) into $[-\pi, \pi)$.
fn wrap_phase(d: f64) -> f64 {
    (d + PI).rem_euclid(2.0 * PI) - PI
}

/// Gaussian envelope $u = \exp(-(x^2 + y^2) / 2\sigma^2)$ on an origin-centered grid.
fn gaussian(nx: usize, ny: usize, dx: f64, dy: f64, sigma: f64) -> Field {
    let mut samples = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dy;
        for ix in 0..nx {
            let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx;
            samples.push(Complex64::from(
                (-((x * x + y * y) / (2.0 * sigma * sigma))).exp(),
            ));
        }
    }
    Field::new(samples, nx, ny, dx, dy).expect("valid gaussian grid")
}

/// Bessel $J_1$ via Simpson integration of
/// $J_1(x) = \frac{1}{\pi} \int_0^\pi \cos(x \sin t - t)\, dt$
/// (test-only helper: avoids adding a special-functions dependency).
fn bessel_j1(x: f64) -> f64 {
    let n = 20_000usize; // even, Simpson requires an even interval count
    let h = PI / n as f64;
    let mut sum = 0.0f64;
    for i in 0..=n {
        let t = i as f64 * h;
        let f = (x * t.sin() - t).cos();
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * f;
    }
    sum * h / 3.0 / PI
}

/// Normalized Airy intensity $I(v)/I(0) = [2 J_1(v) / v]^2$, with the
/// analytic limit $I(0) = 1$.
fn airy(v: f64) -> f64 {
    if v.abs() < 1e-8 {
        1.0 - v * v / 3.0
    } else {
        let j = 2.0 * bessel_j1(v) / v;
        j * j
    }
}

/// Test 1 — energy conservation.
///
/// A compact Gaussian band, propagated a modest distance with the default
/// configuration (padding 2, band-limited), must preserve total power in the
/// extracted central window: the FFT scalings cancel in the transfer-function
/// form and $|H| = 1$
/// on the pass band, so loss is only possible through crop or band clipping,
/// both negligible here (window half-extent = 8.5 sigma; spectrum ~ 1/(2 pi sigma)
/// << band limit). Relative tolerance 1e-9.
#[test]
fn energy_conservation() {
    let band = BandField::new(LAMBDA, gaussian(256, 256, 10e-6, 10e-6, 0.15e-3)).unwrap();
    let out = propagate(&band, 0.1, &AsmConfig::default()).unwrap();
    let p_in = band.field().power();
    let p_out = out.field().power();
    let rel = ((p_out - p_in) / p_in).abs();
    assert!(rel <= 1e-9, "relative power change {rel:e} exceeds 1e-9");
}

/// Test 2 — plane-wave invariance.
///
/// An on-axis plane wave is a single spectral bin (DC), so ASM must return it
/// unchanged except for the global phase $e^{ikz}$ (textbook Eqs. 3.5–3.13).
/// Requires `padding = 1.0`: the DFT grid is periodic, and with zero padding
/// the truncated plane wave legitimately diffracts at the window edges.
/// Intensity and phase tolerance 1e-6.
#[test]
fn plane_wave_invariance() {
    let cfg = AsmConfig {
        padding: 1.0,
        band_limited: true,
        strict: false,
        max_padding: 1.0,
    };
    let z = 0.01;
    let band = BandField::uniform(1.0, 0.0, 128, 128, 10e-6, 10e-6, LAMBDA).unwrap();
    let out = propagate(&band, z, &cfg).unwrap();
    let expected = (2.0 * PI / LAMBDA) * z; // k z, radians
    for iy in 0..128 {
        for ix in 0..128 {
            let s = out.field().sample(ix, iy);
            assert!(
                (s.norm() - 1.0).abs() < 1e-6,
                "intensity changed: {}",
                s.norm()
            );
            let err = wrap_phase(s.arg() - expected).abs();
            assert!(err < 1e-6, "global phase error {err} rad exceeds 1e-6");
        }
    }
}

/// Test 3 — tilted plane wave.
///
/// A plane wave with direction cosine $\alpha$ carries the phase ramp
/// $e^{ik\alpha x}$; propagation multiplies its single spectral bin by a
/// constant $e^{ik\gamma z}$, so the ramp slope $k\alpha$ must be unchanged
/// (textbook Eq. 3.5). The tilt is chosen as an exact DFT bin
/// ($\alpha = m\lambda / L_x$) so the periodic grid represents the wave
/// without seam diffraction; `padding = 1.0`.
#[test]
fn tilted_plane_wave_ramp() {
    let (nx, ny) = (128usize, 128usize);
    let dx = 10e-6;
    let m = 3.0f64; // spectral bin index of the tilt
    let alpha = m * LAMBDA / (nx as f64 * dx);
    let theta = alpha.asin();
    let band = BandField::plane_wave(theta, 0.0, 1.0, nx, ny, dx, dx, LAMBDA).unwrap();
    let cfg = AsmConfig {
        padding: 1.0,
        band_limited: true,
        strict: false,
        max_padding: 1.0,
    };
    let out = propagate(&band, 0.01, &cfg).unwrap();
    let k = 2.0 * PI / LAMBDA;
    let expected_step = (k * alpha * dx * Complex64::I).exp();
    for iy in [0usize, 64, 127] {
        let ratio = out.field().sample(60, iy) / out.field().sample(59, iy);
        let err = (ratio - expected_step).norm();
        assert!(err < 1e-6, "x-ramp slope error {err:e} at row {iy}");
    }
    // No y-dependence introduced: identical ramp ratio on every probed row.
    let r0 = out.field().sample(60, 0) / out.field().sample(59, 0);
    let r1 = out.field().sample(60, 127) / out.field().sample(59, 127);
    assert!((r0 - r1).norm() < 1e-8, "ramp varies along y");
}

/// Test 4 — Airy pattern (circular aperture, far field).
///
/// Uniform illumination masked by a hard-edge circular aperture of radius $a$,
/// propagated to $z$ with $N_F = a^2/(\lambda z) = 0.025 \ll 1$ (Fraunhofer
/// regime — the regime judgment the element docs mandate). The textbook far
/// field is the Fourier transform of the aperture function; for a circle that
/// is the Airy profile
///
/// $$I(v)/I(0) = \left[\frac{2 J_1(v)}{v}\right]^2, \qquad v = k a \sin\theta
///   \approx \frac{k a r}{z},$$
///
/// with first zero at $v = 3.8317 \Leftrightarrow r_1 = 1.22\,\lambda z / (2a)$.
/// The ring-averaged numerical profile is compared to the analytic one
/// (absolute error <= 2% of peak for $r \le 12.6$ mm) and the first dark-ring
/// radius to 2%.
#[test]
fn airy_pattern_far_field() {
    let n = 1536usize;
    let dx = 20e-6; // window 30.72 mm, half-extent 15.36 mm
    let a = 1.0e-3; // aperture radius, m (50 samples across the radius)
    let z = 40.0; // m -> N_F = a^2/(lambda z) = 0.025
    assert!(
        a * a / (LAMBDA * z) <= 0.05,
        "test must sit in the Fraunhofer regime"
    );

    let band = BandField::uniform(1.0, 0.0, n, n, dx, dx, LAMBDA).unwrap();
    let ap = CircularAperture::new(a, (0.0, 0.0)).unwrap();
    let masked = ap.apply(&band).unwrap();
    let far = propagate(&masked, z, &AsmConfig::default()).unwrap();

    // Ring-average the intensity profile (bin width 2*dy = 40 um).
    let dr = 2.0 * dx;
    let r_max = 12.6e-3;
    let nbins = (r_max / dr) as usize; // 315
    let mut sum = vec![0.0f64; nbins];
    let mut cnt = vec![0usize; nbins];
    let field = far.field();
    for iy in 0..n {
        for ix in 0..n {
            let (x, y) = field.coord(ix, iy);
            let r = (x * x + y * y).sqrt();
            let b = (r / dr) as usize;
            if b < nbins {
                sum[b] += field.sample(ix, iy).norm_sqr();
                cnt[b] += 1;
            }
        }
    }
    let profile: Vec<f64> = sum
        .iter()
        .zip(&cnt)
        .map(|(s, c)| if *c > 0 { s / *c as f64 } else { f64::NAN })
        .collect();
    let i0 = profile[..3].iter().sum::<f64>() / 3.0; // on-axis estimate (v ~ 0)
    assert!(i0 > 0.0, "on-axis intensity must be positive");

    // Analytic scale: v(r) = k a r / z.
    let v_of_r = |r: f64| (2.0 * PI / LAMBDA) * a * r / z;

    // Compare bright-field profile: absolute error <= 2% of peak.
    let mut worst = (0.0f64, 0.0f64);
    for (b, &raw) in profile.iter().enumerate() {
        let r = (b as f64 + 0.5) * dr;
        if r > 12.6e-3 - dr {
            continue;
        }
        let num = raw / i0;
        if num.is_nan() {
            continue;
        }
        let err = (num - airy(v_of_r(r))).abs();
        if err > worst.0 {
            worst = (err, r);
        }
    }
    assert!(
        worst.0 <= 0.02,
        "Airy profile deviation {:.3} (peak units) at r = {:.3} mm exceeds 2%",
        worst.0,
        worst.1 * 1e3
    );

    // First dark ring: parabolic refinement of the binned minimum.
    let r1_expected = 1.22 * LAMBDA * z / (2.0 * a); // 12.2 mm
    let b_min = ((8.0e-3 / dr) as usize..(12.4e-3 / dr) as usize)
        .min_by(|&x, &y| profile[x].partial_cmp(&profile[y]).unwrap())
        .unwrap();
    let (ym, y0, y1) = (profile[b_min], profile[b_min - 1], profile[b_min + 1]);
    let d2 = y0 - 2.0 * ym + y1;
    let frac = if d2.abs() > 0.0 {
        0.5 * (y0 - y1) / d2
    } else {
        0.0
    };
    let r_zero = (b_min as f64 + frac.clamp(-0.5, 0.5)) * dr;
    let rel = ((r_zero - r1_expected) / r1_expected).abs();
    assert!(
        rel <= 0.02,
        "first zero at {r_zero:e} m vs analytic {r1_expected:e} m ({:.1}% off)",
        rel * 100.0
    );
}

/// Test 5 — sampling guard: detection, bypass, and automatic mitigation.
///
/// On a fine-pitch grid ($\Delta x = 1\mu$m, $n = 64$) at $z = 0.768$ mm
/// with the band-limit disabled, the closed-form worst adjacent-bin phase step
/// of $H$ is ~19 rad >> $\pi$. The step scales as $1/\text{padding}$, so:
/// - `strict = true`, `max_padding = 1.0` (mitigation disabled) ->
///   `AliasingRisk` with the failing report embedded;
/// - `strict = false` -> propagation proceeds and `sampling_report` flags it;
/// - `strict = true`, `max_padding = 8.0` -> densification 1 -> 2 -> 4 -> 8
///   restores Nyquist adequacy (step ~2.4 rad < pi) and succeeds.
#[test]
fn sampling_guard_and_mitigation() {
    let band = BandField::uniform(1.0, 0.0, 64, 64, 1e-6, 1e-6, LAMBDA).unwrap();
    let z = 7.68e-4;

    // (a) Undersampled transfer function, mitigation disabled -> error.
    let cfg_risk = AsmConfig {
        padding: 1.0,
        band_limited: false,
        strict: true,
        max_padding: 1.0,
    };
    match propagate(&band, z, &cfg_risk) {
        Err(PropagationError::AliasingRisk { report }) => {
            assert!(!report.nyquist_ok);
            assert!(report.max_phase_step_x > PI);
            assert_eq!(report.used_padding, 1.0);
            assert_eq!(report.requested_padding, 1.0);
        }
        other => panic!("expected AliasingRisk, got {other:?}"),
    }

    // (b) Non-strict mode: proceeds, but the diagnostics still flag the risk.
    let cfg_loose = AsmConfig {
        strict: false,
        ..cfg_risk
    };
    let out = propagate(&band, z, &cfg_loose).expect("strict=false must not gate");
    assert_eq!(out.field().dims(), band.field().dims());
    let report = sampling_report(&band, z, &cfg_loose);
    assert!(!report.nyquist_ok && report.max_phase_step_x > PI);

    // (c) Strict mode with mitigation headroom: densify 1 -> 2 -> 4 -> 8.
    let cfg_fix = AsmConfig {
        max_padding: 8.0,
        ..cfg_risk
    };
    let out = propagate(&band, z, &cfg_fix).expect("mitigation must succeed at padding 8");
    assert_eq!(out.field().dims(), band.field().dims());
    // The remedy is exactly what the automatic search found: at padding 8 the
    // closed-form step is Nyquist-adequate.
    let at8 = sampling_report(
        &band,
        z,
        &AsmConfig {
            padding: 8.0,
            ..cfg_fix
        },
    );
    assert!(at8.nyquist_ok && at8.max_phase_step_x < PI);
    // And padding 4 is still insufficient — the search must reach 8.
    let at4 = sampling_report(
        &band,
        z,
        &AsmConfig {
            padding: 4.0,
            ..cfg_fix
        },
    );
    assert!(!at4.nyquist_ok);
}

/// Test 6 — round trip.
///
/// Propagating $+z$ then $-z$ applies $H$ and $H^*$ (same band-limit mask,
/// since $f_{\lim}$ depends on $|z|$), so a Gaussian-envelope band — compact
/// in space and spectrum, negligible crop loss — must recover the input field
/// to a relative Frobenius error of 1e-6. Signed distances are valid by the
/// task-4 decision.
#[test]
fn round_trip_recovers_field() {
    let band = BandField::new(LAMBDA, gaussian(256, 256, 10e-6, 10e-6, 0.15e-3)).unwrap();
    let cfg = AsmConfig::default();
    let z = 0.1;
    let fwd = propagate(&band, z, &cfg).unwrap();
    let back = propagate(&fwd, -z, &cfg).unwrap();

    let diff: f64 = band
        .field()
        .samples()
        .iter()
        .zip(back.field().samples())
        .map(|(a, b)| (a - b).norm_sqr())
        .sum();
    let norm: f64 = band.field().samples().iter().map(|a| a.norm_sqr()).sum();
    let rel = (diff / norm).sqrt();
    assert!(rel < 1e-6, "round-trip relative error {rel:e} exceeds 1e-6");
}
