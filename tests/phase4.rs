//! Phase-4 physical validation: Rayleigh–Sommerfeld first-kind propagation
//! cross-checked against its two sibling discretizations of the same
//! physics (textbook Sec. 2.2.2/3.2: RS propagation *is* the plane-wave
//! spectrum) and against a direct O(n⁴) quadrature of the RS-I integral
//! (Eq. 2.8 kernel, linear convolution — reference truth).
//!
//! Headline statements:
//! - the FFT-sampled-kernel engine equals the direct linear sum up to the
//!   circular wraparound the guard + padding control;
//! - RS, ASM, and direct quadrature agree on a paraxial Gaussian (three
//!   discretizations, one operator);
//! - the Fresnel TF form diverges from the RS truth *exactly where its own
//!   reported binomial-truncation error says it should* — the non-paraxial
//!   physics the paraxial models drop and RS-I keeps;
//! - |H_RS| = 1 on band: power conservation;
//! - the conjugate back-propagation kernel inverts the operator.

mod common;

use common::{LAMBDA, gaussian_band, rs_direct};
use optics_core::element::{Element, EllipticalAperture};
use optics_core::field::BandField;
use optics_core::propagation::asm::{AsmConfig, propagate as asm_propagate};
use optics_core::propagation::fresnel::{
    FresnelConfig, propagate as fresnel_propagate, sampling_report as fresnel_report,
};
use optics_core::propagation::rayleigh_sommerfeld::{
    RayleighSommerfeldConfig, propagate as rs_propagate, sampling_report as rs_report,
};

fn rel_frobenius(a: &BandField, b: &BandField) -> f64 {
    let diff: f64 = a
        .field()
        .samples()
        .iter()
        .zip(b.field().samples())
        .map(|(x, y)| (x - y).norm_sqr())
        .sum();
    let norm: f64 = b.field().samples().iter().map(|y| y.norm_sqr()).sum();
    (diff / norm).sqrt()
}

// --- Geometry A: paraxial Gaussian, three discretizations must agree. ---
//
// n=48, d=2 um (96 um window), sigma=10 um: beam half-width reaches 24.5 um
// at z = 2 mm — deep inside the window, crop negligible. Padded rim
// (x_hi = 96 um at padding 2): kernel step = k*d*x_hi/r ~ 1.2 rad < pi and
// R^2/(lambda*z) ~ 9 kernel Fresnel zones resolved. ASM default (padding 2,
// band-limited, strict) passes its own guard at this distance.
fn band_a() -> BandField {
    gaussian_band(48, 48, 2e-6, 10e-6, (0.0, 0.0))
}
const ZA: f64 = 2e-3;

fn rs_cfg_a() -> RayleighSommerfeldConfig {
    RayleighSommerfeldConfig {
        padding: 2.0,
        strict: true,
    }
}

#[test]
fn engine_matches_direct_quadrature() {
    // The FFT engine and the direct linear sum share the kernel domain
    // (source window) — their difference is pure circular wraparound,
    // controlled by the guard + padding.
    let band = band_a();
    let rep = rs_report(&band, ZA, &rs_cfg_a());
    assert!(
        rep.kernel_nyquist_ok,
        "geometry A must pass the guard: {rep}"
    );
    let fast = rs_propagate(&band, ZA, &rs_cfg_a()).unwrap();
    let slow = rs_direct(&band, ZA);
    let rel = rel_frobenius(&fast, &slow);
    assert!(
        rel < 1e-2,
        "FFT circular vs direct linear convolution: rel {rel:e}"
    );
}

#[test]
fn rs_asm_and_direct_agree_paraxial() {
    // The Phase-4 headline: spatial-kernel (FFT), spatial-kernel (direct),
    // and spectral-transfer discretizations of the SAME operator agree on a
    // smooth band within their shared discretization error.
    let band = band_a();
    let rs = rs_propagate(&band, ZA, &rs_cfg_a()).unwrap();
    let asm = asm_propagate(&band, ZA, &AsmConfig::default()).unwrap();
    let direct = rs_direct(&band, ZA);
    let e_rs_asm = rel_frobenius(&rs, &asm);
    let e_dir_asm = rel_frobenius(&direct, &asm);
    let e_rs_dir = rel_frobenius(&rs, &direct);
    assert!(e_rs_asm < 2e-2, "RS engine vs ASM: rel {e_rs_asm:e}");
    assert!(
        e_dir_asm < 2e-2,
        "direct quadrature vs ASM: rel {e_dir_asm:e}"
    );
    assert!(e_rs_dir < 1e-2, "RS engine vs direct: rel {e_rs_dir:e}");
}

#[test]
fn back_propagation_recovers_field() {
    // +z then -z with the conjugate kernel inverts the operator up to the
    // sampled-kernel window ripple: |H_d| deviates from 1 by the kernel
    // truncation, ~1/zones scale — geometry A resolves ~9 kernel zones,
    // so expect ~2e-2, not machine zero. A sign error in the conjugate
    // kernel would fail catastrophically (O(1)), which is what this pins.
    let band = band_a();
    let cfg = rs_cfg_a();
    let fwd = rs_propagate(&band, ZA, &cfg).unwrap();
    let rt = rs_propagate(&fwd, -ZA, &cfg).unwrap();
    let rel = rel_frobenius(&rt, &band);
    assert!(rel < 3e-2, "round trip +z/-z: rel {rel:e}");
}

// --- Geometry B: hard-edge aperture at an angle-rich distance. ---
//
// Circular aperture a=20 um on a uniform band, n=32, d=4 um (128 um
// window), z = 3 mm: Airy first zero at lambda*z/(2a) = 37.5 um (9.4 bins)
// — the core pattern is inside the window (the ~11% tail crop past it is
// physical spreading, shared by every engine, so no power statements are
// made on this geometry). The aperture's broad
// spectrum reaches the corner bins where the Fresnel binomial truncation
// is ~0.29 rad (its report says so), while RS-I and ASM keep the exact
// spherical/spectral phase.
fn band_b() -> BandField {
    let flat = BandField::uniform(1.0, 0.0, 32, 32, 4e-6, 4e-6, LAMBDA).unwrap();
    EllipticalAperture::circle(20e-6, (0.0, 0.0))
        .unwrap()
        .apply(&flat)
        .unwrap()
}
const ZB: f64 = 3e-3;

fn rs_cfg_b() -> RayleighSommerfeldConfig {
    RayleighSommerfeldConfig {
        padding: 2.0,
        strict: true,
    }
}

#[test]
fn fresnel_diverges_where_reported_error_says() {
    let band = band_b();
    assert!(
        rs_report(&band, ZB, &rs_cfg_b()).kernel_nyquist_ok,
        "geometry B must pass the RS guard"
    );
    let truth = rs_direct(&band, ZB);
    let rs = rs_propagate(&band, ZB, &rs_cfg_b()).unwrap();

    // The Fresnel TF guard is a sampling condition and passes after its
    // standard densification; the PARAXIAL condition is a property of the
    // geometry and stays violated — and that is what shows up against the
    // RS truth.
    let fcfg = FresnelConfig {
        padding: 2.0,
        strict: true,
        max_padding: 8.0,
    };
    let rep = fresnel_report(&band, ZB, &fcfg);
    assert!(
        !rep.paraxial_ok,
        "corner truncation must be significant: {rep}"
    );
    assert!(
        rep.max_phase_error_vs_asm > 0.1 && rep.max_phase_error_vs_asm < 0.6,
        "reported truncation out of expected band: {rep}"
    );
    let fr = fresnel_propagate(&band, ZB, &fcfg).unwrap();

    let e_rs = rel_frobenius(&rs, &truth);
    let e_fr = rel_frobenius(&fr, &truth);
    assert!(e_rs < 2e-2, "RS vs direct truth: rel {e_rs:e}");
    assert!(
        e_fr > 4.0 * e_rs,
        "Fresnel error {e_fr:e} must exceed 4x the RS error {e_rs:e} \
         (truncation ~0.29 rad at corner bins vs RS exactness)"
    );
    assert!(e_fr < 0.3, "Fresnel error must stay bounded: {e_fr:e}");
}

#[test]
fn rs_conserves_power() {
    // |H_RS| = 1 on the propagating band. On geometry A the crop is
    // negligible (spread 24.5 um vs 48 um half-window) and the Gaussian
    // spectrum lies ~10x below the ASM band-limit, so absolute power is
    // conserved: input vs RS output, and vs the exact ASM.
    let band = band_a();
    let out = rs_propagate(&band, ZA, &rs_cfg_a()).unwrap();
    let asm = asm_propagate(&band, ZA, &AsmConfig::default()).unwrap();
    let p0 = band.field().power();
    let rel_rs = ((out.field().power() - p0) / p0).abs();
    let rel_asm = ((asm.field().power() - p0) / p0).abs();
    assert!(rel_rs < 1e-3, "RS power drift {rel_rs:e}");
    assert!(rel_asm < 1e-3, "ASM power drift {rel_asm:e}");
}
