//! Phase-3 physical validation: the blazed grating through the Fraunhofer
//! engine against the textbook grating equation (Eqs. 1.1, 2.26, 5.1):
//!
//! $$\sin\theta_m = \frac{m\lambda}{\Lambda}
//!   \;\;\longleftrightarrow\;\; x_m = \frac{m\lambda z}{\Lambda}$$
//!
//! (small-angle far-field mapping, exact under the engine's $f =
//! X/(\lambda z)$ frequency map).
//!
//! Scope note (task-e decision 2026-10-12): the element carries a
//! grating-vector orientation, but this suite validates the x-direction
//! ($\psi = 0$) orders and blaze concentration only; rotated-order and
//! binary-amplitude efficiency (Eqs. 2.27–2.29) coverage arrives when those
//! features are exercised.

mod common;

use common::LAMBDA;
use optics_rs::element::{BlazedGrating, ClearAperture, Element};
use optics_rs::field::BandField;
use optics_rs::propagation::fraunhofer::{self, FraunhoferConfig};
use std::f64::consts::PI;

// Geometry: input window n*PITCH = 10.24 mm; grating period chosen so the
// far-field order spacing lambda*z/Lambda equals EXACTLY `BINS_PER_ORDER`
// output bins (output pitch = lambda*|z|/(n*PITCH)). The circular stop
// (5 periods, radius STOP) keeps N_F = STOP^2/(lambda*Z) = 0.87 <= 1, so
// the strict Fraunhofer gate passes with `aperture_extent` declared.
const N: usize = 512;
const PITCH: f64 = 20e-6;
const PERIOD: f64 = 102.4e-6;
const STOP: f64 = 512e-6;
const Z: f64 = 0.6;
const BINS_PER_ORDER: f64 = (N as f64 * PITCH) / PERIOD; // = 100

fn order_bins(m: f64) -> f64 {
    (N as f64 - 1.0) / 2.0 + m * BINS_PER_ORDER
}

fn propagate_blaze(depth: f64) -> BandField {
    let band = BandField::uniform(1.0, 0.0, N, N, PITCH, PITCH, LAMBDA).unwrap();
    let grating =
        BlazedGrating::new(PERIOD, depth, 0.0, ClearAperture::Circle(STOP), (0.0, 0.0)).unwrap();
    let masked = grating.apply(&band).unwrap();
    let cfg = FraunhoferConfig {
        aperture_extent: Some(STOP),
        ..Default::default()
    };
    fraunhofer::propagate(&masked, Z, &cfg).unwrap()
}

/// Column-integrated intensity (far-field pattern along x).
fn column_power(out: &BandField) -> Vec<f64> {
    let (nx, ny) = out.field().dims();
    (0..nx)
        .map(|ix| {
            (0..ny)
                .map(|iy| out.field().sample(ix, iy).norm_sqr())
                .sum()
        })
        .collect()
}

fn power_in_window(profile: &[f64], center_bin: f64, halfwidth: f64) -> f64 {
    let lo = (center_bin - halfwidth).ceil().max(0.0) as usize;
    let hi = (center_bin + halfwidth).floor() as usize;
    profile[lo..=hi.min(profile.len() - 1)].iter().sum()
}

#[test]
fn kinoform_blaze_steers_the_first_order() {
    // Depth 2pi: e^{i phi} == e^{i 2pi x/Lambda} — a pure plane wave tilted
    // to order +1 (Sec. 2.4.2: blaze efficiency "approaches 100%").
    let out = propagate_blaze(2.0 * PI);
    let profile = column_power(&out);
    let total: f64 = profile.iter().sum();
    let peak = profile
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    // Grating equation: order +1 at x_1 = lambda*Z/Lambda = 100 bins.
    let ideal = order_bins(1.0);
    assert!(
        (peak as f64 - ideal).abs() <= 2.0,
        "order +1 peak at bin {peak}, expected near {ideal}"
    );
    let x_peak = out.field().coord(peak, N / 2).0;
    let x_1 = LAMBDA * Z / PERIOD;
    assert!(
        (x_peak - x_1).abs() <= 2.0 * (x_1 / BINS_PER_ORDER),
        "sin theta_1: peak {x_peak} vs grating equation {x_1}"
    );
    let frac = |m: f64| power_in_window(&profile, order_bins(m), 40.0) / total;
    assert!(frac(1.0) > 0.85, "blaze concentration: {}", frac(1.0));
    assert!(frac(0.0) < 0.03, "zero order leaked: {}", frac(0.0));
    assert!(frac(-1.0) < 0.03, "order -1 leaked: {}", frac(-1.0));
}

#[test]
fn double_depth_blaze_steers_the_second_order() {
    // Depth 4pi: pure order +2 (x_2 = 2 lambda z / Lambda, Eq. 2.26).
    let out = propagate_blaze(4.0 * PI);
    let profile = column_power(&out);
    let total: f64 = profile.iter().sum();
    let peak = profile
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let ideal = order_bins(2.0);
    assert!(
        (peak as f64 - ideal).abs() <= 2.0,
        "order +2 peak at bin {peak}, expected near {ideal}"
    );
    let frac_2 = power_in_window(&profile, order_bins(2.0), 40.0) / total;
    assert!(frac_2 > 0.8, "order +2 concentration: {frac_2}");
}

#[test]
fn half_wave_sawtooth_splits_orders_zero_and_one() {
    // Depth pi (non-integer): sawtooth Fourier coefficients
    // |c_m|^2 = 4/(pi^2 (1-2m)^2) -> ~40% each into orders 0 and +1,
    // ~4.5% into -1; peaks still sit on m*lambda*z/Lambda (Eq. 5.1).
    let out = propagate_blaze(PI);
    let profile = column_power(&out);
    let total: f64 = profile.iter().sum();
    let frac = |m: f64| power_in_window(&profile, order_bins(m), 40.0) / total;
    let (f0, f1, fm1) = (frac(0.0), frac(1.0), frac(-1.0));
    assert!(f0 > 0.30 && f0 < 0.55, "order 0 fraction {f0}");
    assert!(f1 > 0.30 && f1 < 0.55, "order +1 fraction {f1}");
    assert!(
        (f0 - f1).abs() < 0.05,
        "orders 0/+1 should be comparable: {f0} vs {f1}"
    );
    assert!(fm1 < 0.10, "order -1 should be weak: {fm1}");
}
