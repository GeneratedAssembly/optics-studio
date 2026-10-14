//! CRLB A-optimal DOE window — custom-merit worked example (task 9.12).
//!
//! The merit is **user code**: `examples/crlb_merit.py` scores each
//! observation with the A-optimal Cramer–Rao trace for source-position
//! estimation (Poisson + gaussian read-noise weighting), expressed over
//! the numpy-ready `Observation` layout through the `PyMerit` frontend
//! (feature `py`, ADR-9.11). The scene stencil `[(0,0),(+h,0),(-h,0),
//! (0,+h),(0,-h)]` puts ∂I/∂x₀ and ∂I/∂y₀ in the *intensity* channel
//! only, so the 9.9 adjoint can seed the backward pass (ADR-9.9 limits
//! adjoint seeds to channel 0; the derivative channels would not have an
//! adjoint path).
//!
//! Three gradient layers are demonstrated in one run:
//!
//! 1. **User-supplied gradient** (`merit_grad`) — validated against
//!    central differences on a small pixel window with
//!    `gradcheck::assert_close` (the ADR's shipped check helper).
//! 2. **Adjoint descent** — `grid_gradient` (1 forward render + one
//!    adjoint run per frame per epoch) drives Adam on the 24×24 phase
//!    window; the analytic seed makes each epoch ~6 renders regardless
//!    of the 576 design parameters.
//! 3. **FD-on-obs fallback** — the event-camera log-ratio variant
//!    (`merit_event`, no Python gradient) is wrapped in `FdObsMerit`,
//!    so the same adjoint driver runs at window scale (2N merit calls
//!    per gradient, zero extra renders).
//!
//! Run: `cargo run --release -p optics-inverse --features py --example crlb_doe`

use std::error::Error;
use std::f64::consts::PI;

use optics_core::element::{ClearAperture, PhaseMask};
use optics_core::propagation::asm::AsmConfig;
use optics_core::sources::{GaussianBeam, gaussian};
use optics_inverse::InverseError;
use optics_inverse::adjoint::grid_gradient;
use optics_inverse::carrier::{Carrier, Design};
use optics_inverse::gradcheck::{self, FdObsMerit, GradCheckConfig};
use optics_inverse::merit::Merit;
use optics_inverse::observe::{Channel, Renderer};
use optics_inverse::params::Params;
use optics_inverse::py::PyMerit;

/// Grid samples per side of the DOE window.
const N: usize = 24;
/// Sample pitch (m).
const D: f64 = 15e-6;
/// Design wavelength (m).
const L: f64 = 500e-9;
/// Input plane -> DOE distance (m).
const Z_IN: f64 = 2.5e-3;
/// DOE -> detector distance (m).
const Z_OUT: f64 = 2.5e-3;
/// Gaussian waist at the input plane (m).
const W0: f64 = 60e-6;
/// Scene-stencil half step (m) — reported to the merit as `fd_steps[0]`.
const H_SCENE: f64 = 1e-6;
/// Adam learning rate as a fraction of the phase span per epoch.
const LR_FRAC: f64 = 0.12;

fn doe_carrier() -> Result<Carrier, InverseError> {
    // Phase lattice = parameter vector directly (identity chain rule), so
    // `grid_gradient`'s per-plane dM/dphi is exactly dM/dtheta (ADR-9.9).
    let params = Params::uniform(N * N, -PI, PI, 0.0, "phi");
    Ok(Carrier::analytic(params, move |v, _lambda| {
        let mask = PhaseMask::sampled(
            v.to_vec(),
            (N, N),
            (D, D),
            None,
            ClearAperture::Field,
            (0.0, 0.0),
        )
        .map_err(|e| InverseError::Carrier {
            reason: e.to_string(),
        })?;
        let mut design = Design::new();
        design.gap(Z_IN, 1.0);
        design.push(mask, Z_OUT, 1.0);
        Ok(design)
    }))
}

fn renderer<'c>(carrier: &'c Carrier) -> Renderer<'c> {
    let mut r = Renderer::new(carrier, move |x0, y0, lambda| {
        let beam = GaussianBeam {
            w0: W0,
            amplitude: 1.0,
            center: (x0, y0),
        };
        gaussian(&beam, 0.0, lambda, N, N, D, D).map_err(|e| InverseError::Render {
            reason: e.to_string(),
        })
    });
    r.bands = vec![L];
    r.scenes = vec![
        (0.0, 0.0),
        (H_SCENE, 0.0),
        (-H_SCENE, 0.0),
        (0.0, H_SCENE),
        (0.0, -H_SCENE),
    ];
    r.scene_step = H_SCENE;
    r.asm = AsmConfig {
        band_limited: true,
        strict: false,
        ..Default::default()
    };
    r
}

/// Adam descent whose gradients come from the 9.9 adjoint. `dim`-independent
/// cost per epoch: one forward render inside `grid_gradient` plus one
/// adjoint run per (band, scene) frame.
fn adam_on_adjoint(
    carrier: &Carrier,
    renderer: &Renderer,
    merit: &dyn Merit,
    epochs: usize,
    label: &str,
) -> Result<(Vec<f64>, f64), Box<dyn Error>> {
    let bounds = carrier
        .params()
        .slots()
        .iter()
        .map(|s| (s.lo, s.hi))
        .collect::<Vec<_>>();
    let mut values = carrier.params().values();
    let n = values.len();
    let (b1, b2, eps) = (0.9, 0.999, 1e-8);
    let mut m = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];
    let mut best = f64::INFINITY;
    let mut best_values = values.clone();

    for epoch in 1..=epochs {
        let planes = grid_gradient(renderer, &values, merit)?;
        let grad = &planes
            .first()
            .ok_or_else(|| InverseError::Adjoint {
                reason: "no element planes in the design".into(),
            })?
            .dphi;
        let value = merit.evaluate(&renderer.render(&values)?)?;
        if value < best {
            best = value;
            best_values = values.clone();
        }
        for i in 0..n {
            m[i] = b1 * m[i] + (1.0 - b1) * grad[i];
            v[i] = b2 * v[i] + (1.0 - b2) * grad[i] * grad[i];
            let mh = m[i] / (1.0 - b1.powi(epoch as i32));
            let vh = v[i] / (1.0 - b2.powi(epoch as i32));
            let step = LR_FRAC * (bounds[i].1 - bounds[i].0);
            let next = values[i] - step * mh / (vh.sqrt() + eps);
            values[i] = next.clamp(bounds[i].0, bounds[i].1);
        }
        if epoch == 1 || epoch % 10 == 0 || epoch == epochs {
            println!("  [{label}] epoch {epoch:3}: merit {value:.6e}");
        }
    }
    let final_value = merit.evaluate(&renderer.render(&values)?)?;
    if final_value < best {
        best = final_value;
        best_values = values;
    }
    Ok((best_values, best))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("crlb_doe — user-merit CRLB A-optimal DOE window (task 9.12)");
    println!(
        "  grid {N}x{N} @ {:.0} um, lambda {:.0} nm, stencil h = {:.2} um",
        D * 1e6,
        L * 1e9,
        H_SCENE * 1e6
    );

    let carrier = doe_carrier()?;
    let renderer = renderer(&carrier);
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/crlb_merit.py"
    ))?;

    // --- Layer 1: user gradient, validated by the shipped check helper.
    let crlb = PyMerit::from_source("crlb_aopt", &source)?;
    let obs0 = renderer.render(&carrier.params().values())?;
    let window = gradcheck::window_indices(&obs0, Channel::Intensity, 0, 0, 8..12, 8..12);
    let report = gradcheck::assert_close(
        &crlb,
        &obs0,
        &GradCheckConfig::default().with_window(window),
    )?;
    println!(
        "  gradcheck on 4x4 window: worst abs err {:.2e} over {} entries (pass)",
        report.max_abs_err, report.checked
    );

    let m0 = crlb.evaluate(&obs0)?;
    println!("  initial A-optimal trace: {m0:.6e} um^2");

    // --- Layer 2: adjoint descent with the analytic user gradient.
    println!("  optimizing (adjoint + user gradient, analytic seed)...");
    let (values, best) = adam_on_adjoint(&carrier, &renderer, &crlb, 120, "crlb")?;
    println!(
        "  final A-optimal trace: {best:.6e} um^2 ({:.3}x better)",
        m0 / best
    );

    // --- Layer 3: event log-ratio variant through the FD-on-obs fallback.
    // The FD window is the interior of the beam: absolute steps on
    // 1e-12-class tails would swamp the gradient in cancellation noise
    // (exactly the ADR cost-model caveat — window scale, not full frame).
    let event = PyMerit::from_source_named("crlb_event", &source, "merit_event", None)?;
    let mut fd_cfg = GradCheckConfig {
        eps: 1e-9,
        relative: false,
        ..Default::default()
    };
    let mut window = Vec::new();
    for scene in 0..obs0.scenes.len() {
        window.extend(gradcheck::window_indices(
            &obs0,
            Channel::Intensity,
            0,
            scene,
            6..18,
            6..18,
        ));
    }
    fd_cfg.indices = Some(window.clone());
    println!(
        "  optimizing event variant (FD-on-obs fallback, {} merit calls/gradient)...",
        2 * window.len()
    );
    let wrapped = FdObsMerit {
        inner: Box::new(event),
        cfg: fd_cfg,
    };
    let (event_values, event_best) = adam_on_adjoint(&carrier, &renderer, &wrapped, 8, "event")?;
    println!("  event-variant best trace: {event_best:.6e} um^2");
    println!(
        "  note: the event path demonstrates the FD-on-obs fallback plumbing \
         (validated against autograd in tests); its landscape from a flat DOE \
         is nearly flat — attach an analytic `merit_grad` for production runs."
    );

    // Persist the standard-path design (phase in radians, row-major).
    let out = std::path::Path::new("target/crlb_doe_phase.txt");
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut text = format!("# crlb_doe phase window, {N}x{N} rad, row-major\n");
    for row in values.chunks(N) {
        let line = row
            .iter()
            .map(|p| format!("{p:.6} "))
            .collect::<String>()
            .trim_end()
            .to_string();
        text.push_str(&line);
        text.push('\n');
    }
    std::fs::write(out, text)?;
    println!("  wrote target/crlb_doe_phase.txt (standard path)");
    let _ = event_values; // event design not persisted — demo of the fallback only
    Ok(())
}
