//! Deep-surrogate harness end-to-end (task 9.14, feature `dl`).
//!
//! Problem: a Gaussian beam through a thin lens whose focal length and
//! center offsets are the tunables — three parameters, one band, one
//! scene. The focused spot's size encodes `f`, its displacement encodes
//! `(cx, cy)`, so the observation↔parameter map is learnable in both
//! directions while every render is a real ASM trace. Offsets stay inside
//! ±25 µm and `f` below the observation distance: wider sweeps push the
//! converging beam toward the grid edge and the f ↔ 2z−f intensity
//! mirror, where block-averaged features become nearly blind to `f`.
//!
//! Acceptance is deliberately statistical: validation MSE in normalized
//! space (the network learning the renderer) plus held-out relative
//! feature and parameter error. Relative L2 on raw features weights the
//! near-zero background like the spot, so its thresholds bound the mean
//! and worst loosely; the normalized MSE bound is the strict one.

use optics_core::element::{ClearAperture, ThinLens};
use optics_core::propagation::asm::AsmConfig;
use optics_core::sources::{GaussianBeam, gaussian};

use optics_inverse::carrier::{Carrier, Design};
use optics_inverse::dl::dataset::{self, FeatureSpec};
use optics_inverse::dl::{Direction, Surrogate, TrainConfig, warm_start};
use optics_inverse::error::InverseError;
use optics_inverse::observe::Renderer;
use optics_inverse::params::{Param, Params};

const LAMBDA: f64 = 500e-9;
const N: usize = 64;
const D: f64 = 10e-6;
const Z_OBS: f64 = 10e-3;
const W0: f64 = 60e-6;

fn carrier() -> Carrier {
    let params = Params::from_slots([
        Param::new("f", 6e-3, 8.5e-3, 7.25e-3),
        Param::new("cx", -25e-6, 25e-6, 0.0),
        Param::new("cy", -25e-6, 25e-6, 0.0),
    ]);
    Carrier::analytic(params, move |v, _lambda| {
        let f = v[0];
        let lens =
            ThinLens::new(move |_| f, None, ClearAperture::Field, (v[1], v[2])).map_err(|e| {
                InverseError::Carrier {
                    reason: e.to_string(),
                }
            })?;
        let mut d = Design::new();
        d.push(lens, Z_OBS, 1.0);
        Ok(d)
    })
}

fn renderer(carrier: &Carrier) -> Renderer<'_> {
    let beam = GaussianBeam {
        w0: W0,
        amplitude: 1.0,
        center: (0.0, 0.0),
    };
    let mut r = Renderer::new(carrier, move |x0, y0, lambda| {
        // Single scene at the origin: source-position arguments unused.
        let _ = (x0, y0);
        gaussian(&beam, 0.0, lambda, N, N, D, D).map_err(|e| InverseError::Render {
            reason: e.to_string(),
        })
    });
    r.bands = vec![LAMBDA];
    r.scenes = vec![(0.0, 0.0)];
    r.asm = AsmConfig {
        band_limited: false,
        strict: false,
        ..Default::default()
    };
    r
}

/// Feature plan: 4× block averaging → 256 features per scene.
fn spec() -> FeatureSpec {
    FeatureSpec { downsample: 4 }
}

fn cfg() -> TrainConfig {
    TrainConfig {
        hidden: vec![64, 64],
        epochs: 300,
        lr: 0.02,
        batch: 16,
        seed: 5,
    }
}

#[test]
fn dataset_generation_is_deterministic_and_shaped() {
    let carrier = carrier();
    let renderer = renderer(&carrier);
    let a = dataset::generate(&renderer, &spec(), 12, 42).unwrap();
    let b = dataset::generate(&renderer, &spec(), 12, 42).unwrap();
    assert_eq!(a, b, "same seed must render the same dataset");
    assert_eq!(a.dim_params, 3);
    assert_eq!(a.dim_features, 256); // (64/4)^2 blocks, one scene
    assert_eq!(a.len(), 12);

    let dir = std::env::temp_dir().join("optics-dl-gen-test");
    let path = dir.join("ds.json");
    std::fs::create_dir_all(&dir).unwrap();
    a.save_json(&path).unwrap();
    let back = dataset::Dataset::load_json(&path).unwrap();
    assert_eq!(a, back, "JSON roundtrip must be bit-exact");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn forward_surrogate_learns_the_renderer() {
    let carrier = carrier();
    let renderer = renderer(&carrier);
    let ds = dataset::generate(&renderer, &spec(), 384, 7).unwrap();
    let (train, val) = ds.split(0.25, 11).unwrap();

    let (fwd, report) =
        optics_inverse::dl::train(&train, &val, Direction::Forward, &cfg()).unwrap();

    let first = report.history.first().unwrap().val_loss;
    let last = report.history.last().unwrap().val_loss;
    assert!(last < first * 0.1, "val loss must fall: {first} -> {last}");
    assert!(
        report.best_val_loss < 0.005,
        "normalized val MSE should be small, got {}",
        report.best_val_loss
    );

    // Relative L2 error against the raw rendered features.
    let mut errors = Vec::new();
    for s in &val.samples {
        let pred = fwd.predict(&s.params).unwrap();
        let err = pred
            .iter()
            .zip(&s.features)
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f64>()
            .sqrt();
        let scale = s.features.iter().map(|t| t * t).sum::<f64>().sqrt();
        errors.push(err / scale);
    }
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let worst = errors.iter().copied().fold(0.0f64, f64::max);
    assert!(mean < 0.3, "mean relative feature error {mean}");
    assert!(worst < 0.8, "worst relative feature error {worst}");
}

#[test]
fn inverse_surrogate_warms_up_the_solver() {
    let carrier = carrier();
    let renderer = renderer(&carrier);
    let ds = dataset::generate(&renderer, &spec(), 384, 7).unwrap();
    let (train, val) = ds.split(0.25, 11).unwrap();

    let (inv, report) =
        optics_inverse::dl::train(&train, &val, Direction::Inverse, &cfg()).unwrap();
    assert!(
        report.best_val_loss < 0.05,
        "inverse val MSE {}",
        report.best_val_loss
    );

    // Parameter error in units of each bound span.
    let spans = [2.5e-3, 50e-6, 50e-6];
    let mut worst = 0.0f64;
    for s in &val.samples {
        let pred = inv.predict(&s.features).unwrap();
        for (p, (t, sp)) in pred.iter().zip(s.params.iter().zip(spans)) {
            worst = worst.max((p - t).abs() / sp);
        }
    }
    assert!(worst < 0.25, "worst normalized parameter error {worst}");

    // warm_start on a fresh render: bounded, sane, and usable as an Adam
    // initial point.
    let truth = vec![7.2e-3, 20e-6, -10e-6];
    let obs = renderer.render(&truth).unwrap();
    let start = warm_start(&renderer, &spec(), &obs, &inv).unwrap();
    assert_eq!(start.len(), 3);
    for (v, (lo, hi)) in start
        .iter()
        .zip([(6e-3, 8.5e-3), (-25e-6, 25e-6), (-25e-6, 25e-6)])
    {
        assert!(*v >= lo && *v <= hi, "warm start {v} outside [{lo}, {hi}]");
    }
    for (v, (t, sp)) in start.iter().zip(truth.iter().zip(spans)) {
        assert!((v - t).abs() < 0.35 * sp, "warm start {v} vs truth {t}");
    }
}

#[test]
fn surrogate_artifact_roundtrip() {
    let carrier = carrier();
    let renderer = renderer(&carrier);
    let ds = dataset::generate(&renderer, &spec(), 24, 3).unwrap();
    let (train, val) = ds.split(0.25, 1).unwrap();
    let cfg = TrainConfig {
        hidden: vec![16],
        epochs: 5,
        lr: 0.02,
        batch: 8,
        seed: 9,
    };
    let (fwd, _) = optics_inverse::dl::train(&train, &val, Direction::Forward, &cfg).unwrap();

    let dir = std::env::temp_dir().join("optics-dl-save-test");
    let _ = std::fs::remove_dir_all(&dir);
    fwd.save(&dir).unwrap();
    let back = Surrogate::load(&dir).unwrap();
    assert_eq!(back.direction(), Direction::Forward);
    assert_eq!(back.dims(), fwd.dims());
    let probe = &train.samples[0];
    let a = fwd.predict(&probe.params).unwrap();
    let b = back.predict(&probe.params).unwrap();
    for (x, y) in a.iter().zip(&b) {
        let scale = x.abs().max(y.abs()).max(1e-30);
        assert!(
            (x - y).abs() < 1e-5 * scale,
            "burnpack roundtrip must match: {x} vs {y}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
