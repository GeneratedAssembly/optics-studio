//! Deep-surrogate harness worked example (task 9.14, feature `dl`).
//!
//! Headless dataset generation → forward + inverse burn surrogates →
//! surrogate warm start feeding Adam on a real ASM objective.
//!
//! Run: `cargo run --release -p optics-inverse --features dl --example
//! surrogate_doe`
//!
//! Physics: Gaussian beam (w0 = 60 µm) through a thin lens at the input
//! plane, observed 10 mm downstream on a 64×64 / 10 µm ASM grid. Tunables:
//! focal length f ∈ [6, 8.5] mm and lens-center offsets (cx, cy) ∈
//! [−25, 25] µm — spot size reads `f`, displacement reads the offsets, so
//! the observation↔parameter map is learnable in both directions.

use optics_core::element::{ClearAperture, ThinLens};
use optics_core::propagation::asm::AsmConfig;
use optics_core::sources::{GaussianBeam, gaussian};

use optics_inverse::carrier::{Carrier, Design};
use optics_inverse::dl::dataset::{self, FeatureSpec};
use optics_inverse::dl::{self as dl, Direction, TrainConfig, warm_start};
use optics_inverse::error::InverseError;
use optics_inverse::fd::AdamConfig;
use optics_inverse::merit::{Merit, MseToTarget, Selector, objective};
use optics_inverse::observe::{Channel, Renderer};
use optics_inverse::params::{Param, Params};
use optics_inverse::solve::{self, Solver, StochasticConfig};

const LAMBDA: f64 = 500e-9;
const N: usize = 64;
const D: f64 = 10e-6;
const Z_OBS: f64 = 10e-3;

fn main() {
    let carrier = build_carrier();
    let renderer = build_renderer(&carrier);
    let spec = FeatureSpec { downsample: 4 };

    // 1) Headless dataset: 256 renders, deterministic seed.
    println!("generating dataset (384 ASM renders)...");
    let ds = dataset::generate(&renderer, &spec, 384, 20261014).expect("dataset");
    let (train, val) = ds.split(0.25, 1).expect("split");
    println!(
        "  {} train / {} val, {} → {} features",
        train.len(),
        val.len(),
        ds.dim_params,
        ds.dim_features
    );

    // 2) Forward surrogate: params → features.
    let cfg = TrainConfig {
        hidden: vec![64, 64],
        epochs: 300,
        lr: 0.02,
        batch: 16,
        seed: 5,
    };
    let (fwd, rep_f) = dl::train(&train, &val, Direction::Forward, &cfg).expect("forward train");
    println!(
        "forward surrogate: val MSE {:.4} → {:.5} (best {:.5})",
        rep_f.history[0].val_loss,
        rep_f.history[rep_f.history.len() - 1].val_loss,
        rep_f.best_val_loss
    );

    // 3) Inverse surrogate: features → params.
    let (inv, rep_i) = dl::train(&train, &val, Direction::Inverse, &cfg).expect("inverse train");
    println!(
        "inverse surrogate: val MSE {:.4} → {:.5} (best {:.5})",
        rep_i.history[0].val_loss,
        rep_i.history[rep_i.history.len() - 1].val_loss,
        rep_i.best_val_loss
    );

    // 4) Warm start + solver refinement on a held-out design.
    let truth = vec![7.2e-3, 20e-6, -10e-6];
    let obs = renderer.render(&truth).expect("render truth");
    let start = warm_start(&renderer, &spec, &obs, &inv).expect("warm start");
    println!(
        "warm start: f={:.3} mm cx={:.1} um cy={:.1} um",
        start[0] * 1e3,
        start[1] * 1e6,
        start[2] * 1e6
    );

    let target = obs.band_sum(Channel::Intensity).expect("band sum");
    let merit = MseToTarget {
        target,
        selector: Selector::intensity(None, None),
    };
    let obj = objective(&renderer, &merit as &dyn Merit);
    let mut params = carrier.params().clone();
    params.set_values(&start).expect("warm start in bounds");
    let sol = solve::solve(
        &params,
        &obj,
        Solver::Adam(AdamConfig {
            lr_frac: 0.05,
            ..AdamConfig::default()
        }),
        &StochasticConfig {
            budget: 30,
            seed: Some(1),
            ..Default::default()
        },
        false,
    )
    .expect("Adam run");
    println!(
        "Adam from warm start: MSE {:.3e} → {:.3e}, best f={:.3} mm cx={:.1} um cy={:.1} um",
        obj(&start).unwrap(),
        sol.value,
        sol.best[0] * 1e3,
        sol.best[1] * 1e6,
        sol.best[2] * 1e6
    );

    // 5) Artifacts: dataset + both surrogates on disk.
    let out = std::env::temp_dir().join("optics-dl-example");
    std::fs::create_dir_all(&out).expect("artifact dir");
    ds.save_json(&out.join("dataset.json"))
        .expect("save dataset");
    fwd.save(&out.join("forward")).expect("save forward");
    inv.save(&out.join("inverse")).expect("save inverse");
    println!("artifacts in {}", out.display());
}

fn build_carrier() -> Carrier {
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

fn build_renderer(carrier: &Carrier) -> Renderer<'_> {
    let beam = GaussianBeam {
        w0: 60e-6,
        amplitude: 1.0,
        center: (0.0, 0.0),
    };
    let mut r = Renderer::new(carrier, move |_x0, _y0, lambda| {
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
