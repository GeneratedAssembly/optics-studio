//! Phase 2 physical validation: Fresnel/Fraunhofer engines and regime
//! guidance checked against textbook analytics.
//!
//! Companion to `tests/asm.rs` (Phase 1). Every expected curve is
//! analytic (Gaussian beam $q(z)$ solution, straight-edge Fresnel
//! integrals, rectangular-aperture sinc², circular-aperture Airy);
//! helpers live in `tests/common/mod.rs`.

mod common;

use common::*;
use num_complex::Complex64;
use optics_rs::element::{Element, EllipticalAperture};
use optics_rs::field::{BandField, Field};
use optics_rs::propagation::{asm, fraunhofer, fresnel, regime};
use std::f64::consts::PI;

/// Intensity profile averaged over `2*half_rows+1` center rows:
/// `(r, mean |U|^2, analytic(r))` for pixels with $|x| \le x_{\max}$.
/// Row sampling uses each engine's native pitch (ring binning is too
/// coarse on the Fraunhofer grid); the few-row average suppresses
/// azimuthal speckle from aperture pixelation.
fn row_profile(
    field: &Field,
    x_max: f64,
    half_rows: usize,
    analytic: &dyn Fn(f64) -> f64,
) -> Vec<(f64, f64, f64)> {
    let (nx, ny) = field.dims();
    let iy0 = ny / 2;
    let mut out = Vec::new();
    for ix in 0..nx {
        let (x, _) = field.coord(ix, iy0);
        if x.abs() <= x_max {
            let mut acc = 0.0f64;
            for dy in 0..=2 * half_rows {
                acc += field.sample(ix, iy0 - half_rows + dy).norm_sqr();
            }
            acc /= (2 * half_rows + 1) as f64;
            out.push((x.abs(), acc, analytic(x.abs())));
        }
    }
    out
}

/// Test A — Gaussian beam analytic solution (q-parameter propagation).
///
/// An input waist $u = e^{-r^2/2\sigma^2}$ (waist radius
/// $w_0 = \sqrt{2}\sigma$, Rayleigh range $z_R = \pi w_0^2/\lambda$) must
/// broaden as $w(z) = w_0\sqrt{1 + (z/z_R)^2}$, shrink its on-axis
/// amplitude as $w_0/w(z)$, and accumulate the Gouy phase
/// $-{\arctan}(z/z_R)$ on top of $e^{ikz}$. Checked for ASM (exact) and
/// Fresnel TF (paraxial — the Gaussian beam is its canonical solution),
/// plus ASM-vs-Fresnel cross-model consistency.
#[test]
fn gaussian_beam_width_and_gouy_match_analytics() {
    let n = 255usize;
    let d = 1e-5;
    let sigma = 0.15e-3;
    let z = 0.1;
    let band = gaussian_band(n, n, d, sigma, (0.0, 0.0));
    let w0 = (2.0f64).sqrt() * sigma;
    let z_r = PI * w0 * w0 / LAMBDA;
    let w_z = w0 * (1.0 + (z / z_r).powi(2)).sqrt();
    let gouy = (z / z_r).atan();
    let k = 2.0 * PI / LAMBDA;

    let out_asm = asm::propagate(&band, z, &asm::AsmConfig::default()).unwrap();
    let out_fr = fresnel::propagate(&band, z, &fresnel::FresnelConfig::default()).unwrap();

    for (label, out) in [("asm", &out_asm), ("fresnel", &out_fr)] {
        let f = out.field();
        let (mut s0, mut s2) = (0.0f64, 0.0f64);
        for iy in 0..n {
            for ix in 0..n {
                let (x, _) = f.coord(ix, iy);
                let i = f.sample(ix, iy).norm_sqr();
                s0 += i;
                s2 += i * x * x;
            }
        }
        let sigma_meas = (2.0 * s2 / s0).sqrt();
        let sigma_exp = w_z / (2.0f64).sqrt();
        let rel = ((sigma_meas - sigma_exp) / sigma_exp).abs();
        assert!(rel < 1e-3, "{label}: beam width rel. error {rel:e}");

        let c = f.sample(n / 2, n / 2); // origin sample (odd grid)
        let amp_exp = w0 / w_z;
        assert!(
            ((c.norm() - amp_exp) / amp_exp).abs() < 1e-3,
            "{label}: on-axis amplitude {}",
            c.norm()
        );
        let phase_err = wrap_phase(c.arg() - (k * z - gouy)).abs();
        assert!(
            phase_err < 1e-3,
            "{label}: Gouy phase error {phase_err:e} rad"
        );
    }

    let diff: f64 = out_asm
        .field()
        .samples()
        .iter()
        .zip(out_fr.field().samples())
        .map(|(a, b)| (a - b).norm_sqr())
        .sum();
    let nrm: f64 = out_asm.field().samples().iter().map(|a| a.norm_sqr()).sum();
    let rel = (diff / nrm).sqrt();
    assert!(
        rel < 1e-3,
        "ASM vs Fresnel TF cross-model rel. error {rel:e}"
    );
}

/// Test B — straight-edge Fresnel diffraction.
///
/// A half-plane ($x < 0$ illuminated) propagated a short distance must
/// reproduce the Fresnel-integral profile (textbook Sec. 1.4 half-period
/// zones, evaluated in closed form):
///
/// $$I(u)/I_0 = \tfrac12\left[\left(\tfrac12 - C(u)\right)^2 +
///   \left(\tfrac12 - S(u)\right)^2\right], \qquad
///   u = x\sqrt{2/(\lambda z)}.$$
///
/// Engine: Fresnel TF. The grid must resolve the *angles* the fringes
/// need: the edge wave reaching observation $x$ rides on spectral
/// components $f \approx x/(\lambda z)$, so the pitch must satisfy
/// $\Delta < \lambda z / (2 x_{\max})$ — at $d = 1\mu$m the sampled band
/// covers $|u| \lesssim 45$, far beyond the compared $|u| \le 9$.
/// Tolerance 2.5% of $I_0$ over $|x| \le 0.2$ mm (window truncation and
/// the finite $y$ extent contribute $\lesssim 1.5\%$ ripple there).
#[test]
fn straight_edge_matches_fresnel_integrals() {
    let n = 2048usize;
    let d = 1e-6; // window +/-1.024 mm
    let z = 2e-3;
    let mut samples = vec![Complex64::new(1.0, 0.0); n * n];
    for iy in 0..n {
        for ix in 0..n {
            let x = (ix as f64 - (n as f64 - 1.0) / 2.0) * d;
            if x > 0.0 {
                samples[iy * n + ix] = Complex64::ZERO;
            }
        }
    }
    let band = BandField::new(LAMBDA, Field::new(samples, n, n, d, d).unwrap()).unwrap();
    let out = fresnel::propagate(&band, z, &fresnel::FresnelConfig::default()).unwrap();

    let ft = FresnelTable::new(20.0, 1e-4);
    let scale = (2.0 / (LAMBDA * z)).sqrt(); // u per meter of x
    let row = n / 2;
    let mut worst = (0.0f64, 0.0f64);
    for ix in 0..n {
        let x = (ix as f64 - (n as f64 - 1.0) / 2.0) * d;
        if x.abs() > 0.2e-3 {
            continue;
        }
        let u = x * scale;
        let i_exp = 0.5 * ((0.5 - ft.c(u)).powi(2) + (0.5 - ft.s(u)).powi(2));
        let i_num = out.field().sample(ix, row).norm_sqr();
        let err = (i_num - i_exp).abs();
        if err > worst.0 {
            worst = (err, x);
        }
    }
    assert!(
        worst.0 < 0.025,
        "straight-edge profile deviation {:.4} (I0 units) at x={:.3e} m",
        worst.0,
        worst.1
    );
}

/// Test C — rectangular aperture far field: separable sinc².
///
/// A square of half-width $a$ illuminated uniformly, propagated to the far
/// field with the Fraunhofer engine (aperture extent declared), must give
/// $I(X)/I_{\text{peak}} = \operatorname{sinc}^2(\pi \cdot 2a \cdot
/// X/(\lambda z))$ along a center row (textbook Sec. 2.2/2.3: far field =
/// FT of the aperture function; $\operatorname{rect} \to \operatorname{sinc}$).
/// Tolerance 3% of peak out to the second zero region. Pixelation of the
/// hard edge (20 samples across the side) dominates the residual.
#[test]
fn rectangular_aperture_sinc2_far_field() {
    let n = 2048usize;
    let d = 1e-5; // window +/-10.24 mm
    let a = 1e-4; // half-width: 10 samples each side
    let z = 1.0;
    let w = 2.0 * a; // full side
    let mut samples = vec![Complex64::ZERO; n * n];
    for iy in 0..n {
        let y = (iy as f64 - (n as f64 - 1.0) / 2.0) * d;
        for ix in 0..n {
            let x = (ix as f64 - (n as f64 - 1.0) / 2.0) * d;
            if x.abs() <= a && y.abs() <= a {
                samples[iy * n + ix] = Complex64::new(1.0, 0.0);
            }
        }
    }
    let band = BandField::new(LAMBDA, Field::new(samples, n, n, d, d).unwrap()).unwrap();
    let cfg = fraunhofer::FraunhoferConfig {
        aperture_extent: Some(a),
        ..Default::default()
    };
    let out = fraunhofer::propagate(&band, z, &cfg).unwrap();
    let dxo = LAMBDA * z / (n as f64 * d);
    let row = n / 2;
    let peak = out
        .field()
        .intensities()
        .iter()
        .fold(0.0f64, |a, &b| a.max(b));
    let mut worst = (0.0f64, 0.0f64);
    for ix in 0..n {
        let x = (ix as f64 - (n as f64 - 1.0) / 2.0) * dxo;
        if x.abs() > 4.0e-3 {
            continue;
        }
        let s = sinc(PI * w * x / (LAMBDA * z));
        let err = (out.field().sample(ix, row).norm_sqr() / peak - s * s).abs();
        if err > worst.0 {
            worst = (err, x);
        }
    }
    assert!(
        worst.0 < 0.03,
        "sinc2 profile deviation {:.4} of peak at X={:.3e} m",
        worst.0,
        worst.1
    );
}

/// Test D — Airy pattern across the engines that own this regime.
///
/// The masked aperture (radius $a = 1$ mm) at $z = 40$ m gives
/// $N_F = a^2/\lambda z = 0.05$ — deep far field (at $N_F = 0.25$ the true
/// pattern legitimately deviates from the Fraunhofer limit by ~3% in the
/// dark rings). Three engines must reproduce $[2J_1(v)/v]^2$,
/// $v = k a r / z$:
///
/// - Fraunhofer (its own far-field grid, 3%),
/// - Fresnel **convolution** form (input grid, padding 1 — the kernel's
///   rule-2 guard holds at this long distance, 2%),
/// - ASM (band-limited, exact, 2%).
///
/// The Fresnel **TF** form is deliberately *not* run here: at $\lambda z =
/// 2\times10^{-5} \gg n\Delta^2$, its frequency-domain chirp is
/// undersampled beyond any padding cap — the test asserts this
/// complementarity (TF guard fails, kernel guard holds), which is exactly
/// what `regime` reports for the geometry.
#[test]
fn airy_pattern_across_all_three_engines() {
    let n = 2048usize;
    let d = 2e-5; // window +/-20.5 mm
    let a = 1e-3;
    let z = 40.0;
    let band = BandField::uniform(1.0, 0.0, n, n, d, d, LAMBDA).unwrap();
    let ap = EllipticalAperture::circle(a, (0.0, 0.0)).unwrap();
    let masked = ap.apply(&band).unwrap();
    let k = 2.0 * PI / LAMBDA;
    let r_max = 12.6e-3; // v up to 3.96: core, first dark ring, first bright ring

    // Regime complementarity at this geometry (textbook rules, task-3 docs).
    let rep = fresnel::sampling_report(&masked, z, &fresnel::FresnelConfig::default());
    assert!(
        !rep.nyquist_ok && rep.kernel_nyquist_ok,
        "expected undersampled TF chirp but sampled kernel: {rep}"
    );

    let ff = fraunhofer::propagate(
        &masked,
        z,
        &fraunhofer::FraunhoferConfig {
            aperture_extent: Some(a),
            ..Default::default()
        },
    )
    .unwrap();
    let fc = fresnel::propagate_convolution(
        &masked,
        z,
        &fresnel::FresnelConfig {
            padding: 1.0,
            strict: true,
            max_padding: 1.0,
        },
    )
    .unwrap();
    let asm_out = asm::propagate(&masked, z, &asm::AsmConfig::default()).unwrap();

    let runs = [
        ("fraunhofer", &ff, 0.03),
        ("fresnel-conv", &fc, 0.02),
        ("asm", &asm_out, 0.02),
    ];

    for (label, out, tol) in runs {
        let f = out.field();
        let profile = row_profile(f, r_max, 2, &|r| airy(k * a * r / z));
        let peak = profile.iter().map(|(_, i, _)| *i).fold(0.0f64, f64::max);
        let a_peak = profile.iter().map(|(_, _, e)| *e).fold(0.0f64, f64::max);
        let mut worst = (0.0f64, 0.0f64);
        for (r, i, a_exp) in &profile {
            let err = (i / peak - a_exp / a_peak).abs();
            if err > worst.0 {
                worst = (err, *r);
            }
        }
        assert!(
            worst.0 < tol,
            "{label}: Airy profile deviation {:.4} of peak at r={:.3e} m (tol {tol})",
            worst.0,
            worst.1
        );
    }
}

/// Test E — the regime ladder's recommendations execute cleanly.
///
/// For each Phase-2 geometry, `regime`'s recommended engine must propagate
/// it without error (smoke test tying task 5 to the engines; the
/// recommendation semantics themselves are unit-tested in `regime`).
#[test]
fn regime_recommendations_execute_cleanly() {
    // A's Gaussian waist at z = 0.1: paraxial near field, TF guard holds.
    let g = gaussian_band(255, 255, 1e-5, 0.15e-3, (0.0, 0.0));
    let rep = regime::regime_report(&g, 0.1);
    assert_eq!(rep.recommended, regime::Engine::FresnelTransfer);
    fresnel::propagate(&g, 0.1, &fresnel::FresnelConfig::default()).unwrap();

    // B's half-plane at z = 2e-3: window edge subtends 0.5 rad -> non-
    // paraxial verdict -> ASM.
    let mut samples = vec![Complex64::new(1.0, 0.0); 256 * 256];
    for iy in 0..256 {
        for ix in 0..256 {
            let x = (ix as f64 - 127.5) * 5e-6;
            if x > 0.0 {
                samples[iy * 256 + ix] = Complex64::ZERO;
            }
        }
    }
    let hp = BandField::new(LAMBDA, Field::new(samples, 256, 256, 5e-6, 5e-6).unwrap()).unwrap();
    let rep = regime::regime_report(&hp, 2e-3);
    assert_eq!(rep.recommended, regime::Engine::Asm);
    asm::propagate(&hp, 2e-3, &asm::AsmConfig::default()).unwrap();

    // D's aperture at z = 4 with declared extent: far field -> Fraunhofer.
    let band = BandField::uniform(1.0, 0.0, 256, 256, 2e-5, 2e-5, LAMBDA).unwrap();
    let ap = EllipticalAperture::circle(1e-3, (0.0, 0.0)).unwrap();
    let masked = ap.apply(&band).unwrap();
    let rep = regime::regime_report_with(&masked, 4.0, Some(1e-3));
    assert_eq!(rep.recommended, regime::Engine::Fraunhofer);
    fraunhofer::propagate(
        &masked,
        4.0,
        &fraunhofer::FraunhoferConfig {
            aperture_extent: Some(1e-3),
            ..Default::default()
        },
    )
    .unwrap();
}
