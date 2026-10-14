//! Phase-6 physical validation: refractive surfaces, thick-lens
//! composition, hybrid achromats, and wedge deflection.
//!
//! Headline statements:
//! - a `RefractiveSurface -> medium gap -> RefractiveSurface` stack traced
//!   with the medium-aware ASM focuses at the FULL thick-lens back focal
//!   length — textbook Eq. 4.1, whose (n-1) bracket multiplies the
//!   thickness term (equation re-verified against the rendered page, not
//!   the text-extraction layout); the thin-lens law is measurably wrong;
//! - a refractive lens paired with a wavelength-independent diffractive
//!   phase screen (Sec. 4.4.1 hybrid correction) is achromatic to first
//!   order at the design wavelength: the DOE power Phi_d ~ lambda has the
//!   OPPOSITE chromatic slope of the refractive power, so same-sign powers
//!   cancel color while an uncorrected lens shifts ~10x more;
//! - a wedge deflects a plane wave to sin(theta) = -(n-1)tan(alpha) toward
//!   the base, measured through the Fraunhofer engine.

use optics_core::beamline::Beamline;
use optics_core::element::{
    ClearAperture, Element, Material, PhaseMask, RefractiveSurface, SurfaceProfile, ThinLens, Wedge,
};
use optics_core::field::BandField;
use optics_core::propagation::asm::{AsmConfig, propagate as asm_propagate};
use optics_core::propagation::fraunhofer::{FraunhoferConfig, propagate as fraunhofer_propagate};
use std::f64::consts::PI;

const L0: f64 = 550e-9; // design wavelength
const N_G: f64 = 1.5; // crown-like constant index (stack test)

fn glass(n: f64) -> Material {
    Material::new(move |_l, _a| n, move |_l, _a| 0.0)
}

/// Unwrapped phase (rad) along +x from the axis sample, bins 1..=kmax.
fn radial_phase(band: &BandField, kmax: usize) -> Vec<f64> {
    let c = band.field().dims().0 / 2;
    let mut out = Vec::with_capacity(kmax);
    let mut prev = band.field().sample(c, c).arg();
    let mut acc = 0.0f64;
    for j in 1..=kmax {
        let a = band.field().sample(c + j, c).arg();
        acc += (a - prev + PI).rem_euclid(2.0 * PI) - PI;
        prev = a;
        out.push(acc);
    }
    out
}

/// Least-squares power Phi of phi(rho) = -k Phi rho^2 / 2 through the
/// origin from radial phases at pitch d.
fn fit_power(phases: &[f64], d: f64, k: f64) -> f64 {
    let (mut sxx, mut sxp) = (0.0f64, 0.0f64);
    for (j, p) in phases.iter().enumerate() {
        let x = ((j + 1) as f64 * d).powi(2);
        sxx += x * x;
        sxp += x * p;
    }
    -2.0 * (sxp / sxx) / k
}

#[test]
fn thick_stack_focuses_at_book_eq_41_back_focal_length() {
    // Asymmetric biconvex stack: R1 = +25 mm (Phi1 = (n-1)c1 = 20 1/m),
    // R2 = -100 mm (Phi2 = 5 1/m), n = 1.5, center thickness t = 5 mm,
    // stop a = 1 mm, ODD grid (n=1021, d=4 um: exact axis sample). The
    // gap-propagated converging wave carries truncation-diffraction phase
    // ripple of scale a^2/(lambda_n (s-t)) = 20 half-period zones across
    // the aperture, so the wavefront curvature is fit over the central
    // 96 um (24 bins, step 0.6 rad) where the geometric chirp dominates.
    // Matrix power Phi = Phi1 + Phi2 - (t/n)Phi1 Phi2 = 24.6667;
    // BFL = (1 - (t/n) Phi1)/Phi = 0.0378364 m — versus the thin-lens law
    // 1/(Phi1+Phi2) = 0.040 m: a 5.4% shift, measured as the exit
    // wavefront curvature (Eq. 4.1's bracketed thickness term).
    let (c1, c2, t, n) = (40.0f64, -10.0f64, 5e-3f64, N_G);
    let phi1 = (n - 1.0) * c1;
    let phi2 = (1.0 - n) * c2;
    let big_phi = phi1 + phi2 - t / n * phi1 * phi2;
    let bfl = (1.0 - t / n * phi1) / big_phi;
    let thin = 1.0 / (phi1 + phi2);

    let (n_grid, d, a) = (1021usize, 4e-6, 1e-3);
    let band = BandField::uniform(1.0, 0.0, n_grid, n_grid, d, d, L0).unwrap();
    let s1 = RefractiveSurface::new(
        SurfaceProfile::Spherical { radius: 1.0 / c1 },
        glass(1.0),
        glass(n),
        None,
        None,
        ClearAperture::Circle(a),
        (0.0, 0.0),
    )
    .unwrap();
    let s2 = RefractiveSurface::new(
        SurfaceProfile::Spherical { radius: 1.0 / c2 },
        glass(n),
        glass(1.0),
        None,
        None,
        ClearAperture::Circle(a),
        (0.0, 0.0),
    )
    .unwrap();
    let mut bl = Beamline::new();
    bl.push_medium(&s1, t, n).push(&s2, 0.0);
    let out = bl
        .trace(&band, |b, z, m| {
            asm_propagate(
                b,
                z,
                &AsmConfig {
                    medium: m,
                    ..AsmConfig::default()
                },
            )
        })
        .unwrap();

    let k = 2.0 * PI / L0;
    let measured = fit_power(&radial_phase(&out, 24), d, k).recip();
    assert!(
        (measured - bfl).abs() < 0.01 * bfl,
        "BFL {measured:e} vs Eq. 4.1 {bfl:e} (thin law {thin:e})"
    );
    assert!(
        (measured - thin).abs() > 0.5 * (thin - bfl),
        "stack must NOT land at the thin-lens law: {measured:e}"
    );
}

#[test]
fn hybrid_achromat_cancels_first_order_color() {
    // Refractive ThinLens with Sellmeier-like closure n(lam) = 1.4 +
    // 4e-14/lam^2 (power rises toward blue, Sec. 4.1.3) + fixed-phase
    // DOE screen (PhaseMask): phi_DOE = -A rho^2 with A = pi*Phi_d0/L0 so
    // its equivalent power is Phi_d(lam) = Phi_d0 * lam/L0 — slope of the
    // OPPOSITE sign. Achromacy at L0: Phi_d0 = -L0 * dPhi_r/dlam.
    // Measured total power must shift < 1% over 500..600 nm while the
    // refractive element alone shifts > 4%.
    let c_r = 0.25f64; // vertex curvature, weak for grid sampling
    let a = 400e-6f64;
    let n_of = move |lam: f64| 1.4 + 4e-14 / (lam * lam);
    let phi_r = move |lam: f64| (n_of(lam) - 1.0) * c_r;
    let dphi_r = -2.0 * c_r * 4e-14 / L0.powi(3); // dPhi_r/dlam at L0
    let phi_d0 = -L0 * dphi_r; // cancels the linear term
    assert!(phi_d0 > 0.0, "DOE power must be same-sign here");

    let doe = PhaseMask::new(
        move |x, y| -(PI * phi_d0 / L0) * (x * x + y * y),
        None,
        ClearAperture::Circle(a),
        (0.0, 0.0),
    )
    .unwrap();

    let measure = |lam: f64, with_doe: bool| -> f64 {
        let lens = ThinLens::new(
            move |l| 1.0 / phi_r(l),
            None,
            ClearAperture::Field,
            (0.0, 0.0),
        )
        .unwrap();
        let band = BandField::uniform(1.0, 0.0, 128, 128, 8e-6, 8e-6, lam).unwrap();
        let mut field = lens.apply(&band).unwrap();
        if with_doe {
            field = doe.apply(&field).unwrap();
        }
        let k = 2.0 * PI / lam;
        fit_power(&radial_phase(&field, 50), 8e-6, k)
    };

    let p0 = measure(L0, true);
    let p500 = measure(500e-9, true);
    let p600 = measure(600e-9, true);
    let r500 = measure(500e-9, false);
    let r0 = measure(L0, false);
    let shift = |a: f64, b: f64| ((a - b) / b).abs();
    assert!(
        shift(p500, p0) < 0.01 && shift(p600, p0) < 0.01,
        "corrected shifts {} / {}",
        shift(p500, p0),
        shift(p600, p0)
    );
    assert!(
        shift(r500, r0) > 0.04,
        "uncorrected refractive shift {} not > 4%",
        shift(r500, r0)
    );
    // First-order cancellation: corrected shift ratio vs uncorrected.
    assert!(shift(p500, p0) < 0.25 * shift(r500, r0));
}

#[test]
fn wedge_deflects_toward_the_base() {
    // alpha = 1 mrad, n = 1.5: sin(theta) = -(n-1)tan(alpha) = -5e-4.
    // Fraunhofer at z = 2 m maps the tilt to x = -1 mm. Grid n = 128,
    // d = 20 um: output pitch lambda*z/(n*d) = 430 um -> the spot lands at
    // -2.33 bins; stop a = 400 um gives a 3.2-bin spot (N_F = 0.073).
    let alpha = 1e-3f64;
    let z = 2.0f64;
    let a = 400e-6f64;
    let w = Wedge::new(
        alpha,
        glass(1.5),
        1e-3,
        optics_core::common::Angle::zero(),
        ClearAperture::Circle(a),
        (0.0, 0.0),
    )
    .unwrap();
    let band = BandField::uniform(1.0, 0.0, 128, 128, 20e-6, 20e-6, L0).unwrap();
    let fcfg = FraunhoferConfig {
        aperture_extent: Some(a),
        ..Default::default()
    };
    let mut bl = Beamline::new();
    bl.push(&w, z);
    let psf = bl
        .psf(&band, |b, zz, _n| fraunhofer_propagate(b, zz, &fcfg))
        .unwrap();
    let (cx, cy) = psf.centroid();
    let expect = -(1.5 - 1.0) * alpha.tan() * z; // -1.000000e-3 m
    assert!(
        (cx - expect).abs() < 1e-4,
        "deflection centroid x {cx:e} vs {expect:e}"
    );
    assert!(cy.abs() < 1e-6, "off-axis tilt leaked into y: {cy:e}");
}
