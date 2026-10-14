//! Coded DOE design — Gerchberg–Saxton top-hat beam shaper (textbook
//! §5.5, `optics-inverse` task 9.8/9.10).
//!
//! A phase-only DOE (`Carrier::grid_phase`) sits at the midpoint of a
//! coherent relay from a Gaussian input plane to a target plane, and GS
//! retrieves the DOE phase that flattens the target-plane intensity
//! into a top hat: forward half-transform, Fourier-plane amplitude
//! projection ($|U_t| \to T$), back-propagation (the exact ASM adjoint,
//! §13 of `docs/physics.md`), DOE-plane illumination projection, repeat.
//!
//! The target amplitude is scaled to the input power — a phase-only DOE
//! conserves power, so $T^2$ summed over the disk equals $\sum|U_{in}|^2$
//! and the demand is reachable by construction.
//!
//! Prints the convergence trace (merit, Eq. 5.18 amplitude MSE, phase
//! update RMS) and the achieved flatness (interior coefficient of
//! variation), then writes the DOE phase and the delivered target-plane
//! frame to `target/coded_doe_phase.pgm` and
//! `target/coded_doe_target.pgm`.
//!
//! Run: `cargo run --release -p optics-inverse --example coded_doe`

use std::f64::consts::PI;

use optics_core::element::ClearAperture;
use optics_core::propagation::asm::AsmConfig;
use optics_core::sources::{GaussianBeam, gaussian};
use optics_inverse::Carrier;
use optics_inverse::gs::{self, GsConfig, GsInit, GsProblem, GsVariant};
use optics_inverse::merit::{MseToTarget, Selector};

/// Design wavelength (m) — single band, fully coherent.
const L: f64 = 500e-9;
/// Grid samples per side.
const N: usize = 128;
/// Sample pitch (m).
const D: f64 = 8e-6;
/// Input→target separation (m); the DOE sits at $z_{\text{total}}/2$.
const Z_TOTAL: f64 = 8e-3;
/// Gaussian waist at the input plane (m).
const W0: f64 = 150e-6;
/// Top-hat target radius (m).
const TARGET_R: f64 = 250e-6;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("coded_doe — Gerchberg–Saxton top-hat shaper");
    println!(
        "  grid {N}x{N} at {:.0} um pitch, lambda {:.0} nm, z_total {:.0} mm",
        D * 1e6,
        L * 1e9,
        Z_TOTAL * 1e3
    );

    // Illumination: centered Gaussian at the input plane.
    let beam = GaussianBeam {
        w0: W0,
        amplitude: 1.0,
        center: (0.0, 0.0),
    };
    let input = gaussian(&beam, 0.0, L, N, N, D, D)?;

    // Top-hat target amplitude, power-scaled to the illumination:
    // T^2 * (disk pixels) = sum |U_in|^2.
    let power: f64 = input.field().samples().iter().map(|s| s.norm_sqr()).sum();
    let disk = |ix: usize, iy: usize| {
        let (x, y) = input.field().coord(ix, iy);
        x * x + y * y <= TARGET_R * TARGET_R
    };
    let disk_pixels = (0..N)
        .flat_map(|iy| (0..N).map(move |ix| (ix, iy)))
        .filter(|&(ix, iy)| disk(ix, iy))
        .count() as f64;
    let amp = (power / disk_pixels).sqrt();
    let target: Vec<f64> = (0..N)
        .flat_map(|iy| (0..N).map(move |ix| (ix, iy)))
        .map(|(ix, iy)| if disk(ix, iy) { amp } else { 0.0 })
        .collect();
    println!(
        "  target: r = {:.0} um flat-top, amplitude {:.3} (power {:.3} arb)",
        TARGET_R * 1e6,
        amp,
        power
    );

    // Phase-only DOE carrier: full 2*pi depth, flat start.
    let carrier = Carrier::grid_phase(
        (N, N),
        (D, D),
        ClearAperture::Field,
        (0.0, 0.0),
        -PI,
        PI,
        0.0,
    )?;

    // Stopping runs through the Merit seam on each iteration's
    // Observation (MSE-to-target intensity is the natural choice).
    let merit = MseToTarget {
        target: target.iter().map(|t| t * t).collect(),
        selector: Selector::intensity(None, None),
    };

    let problem = GsProblem {
        carrier: &carrier,
        input: input.clone(),
        z_total: Z_TOTAL,
        target_amplitude: target,
        merit: &merit,
        asm: AsmConfig {
            band_limited: false,
            strict: false,
            ..Default::default()
        },
        config: GsConfig {
            max_iterations: 120,
            patience: 10,
            merit_tolerance: 1e-4,
            variant: GsVariant::Classic,
            init: GsInit::Random { seed: 7 },
        },
    };
    let sol = gs::run(&problem)?;

    // Convergence trace, every 10th iteration plus the best.
    println!(
        "  {:>4}  {:>10}  {:>10}  {:>12}  {:>10}",
        "iter", "merit", "best", "ampl. MSE", "dphase RMS"
    );
    for r in sol
        .history
        .iter()
        .filter(|r| r.iteration % 10 == 0 || r.iteration == sol.best_iteration)
    {
        println!(
            "  {:>4}  {:>10.4}  {:>10.4}  {:>12.5}  {:>10.4}",
            r.iteration, r.merit, r.best_merit, r.amplitude_mse, r.phase_update_rms
        );
    }
    println!(
        "  stopped after {} iterations ({}), best at iter {}",
        sol.iterations,
        if sol.converged { "patience" } else { "cap" },
        sol.best_iteration
    );

    // Achieved flatness: coefficient of variation inside 0.7*R.
    let interior_r = 0.7 * TARGET_R;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let mut cnt = 0usize;
    for iy in 0..N {
        for ix in 0..N {
            let (x, y) = input.field().coord(ix, iy);
            if x * x + y * y <= interior_r * interior_r {
                let v = sol.target_intensity[iy * N + ix];
                sum += v;
                sum2 += v * v;
                cnt += 1;
            }
        }
    }
    let mean = sum / cnt as f64;
    let cv = (sum2 / cnt as f64 - mean * mean).max(0.0).sqrt() / mean;
    println!(
        "  interior CV (r < {:.0} um): {cv:.3}  (flat top; GS golden bound 0.35)",
        interior_r * 1e6
    );

    // The returned phase is a legal GridPhase parameter vector: the
    // design can be rebuilt and forwarded through the core renderer.
    let design = carrier.build(sol.carrier_values(), L)?;
    println!(
        "  design rebuilt via Carrier::build: {} elements, {} steps",
        design.elements().len(),
        design.steps().len()
    );

    write_pgm("target/coded_doe_phase.pgm", &sol.phase, true)?;
    write_pgm("target/coded_doe_target.pgm", &sol.target_intensity, false)?;
    println!(
        "  wrote target/coded_doe_phase.pgm (phase -pi..pi) and target/coded_doe_target.pgm (log intensity)"
    );
    Ok(())
}

/// 8-bit PGM dump: linear over $[-\pi, \pi)$ for the phase map, log
/// scale for the intensity frame (drism precedent).
fn write_pgm(path: &str, img: &[f64], phase_map: bool) -> std::io::Result<()> {
    use std::io::Write;
    let mut buf = Vec::with_capacity(img.len());
    if phase_map {
        for v in img {
            buf.push(((*v + PI) / (2.0 * PI) * 255.0) as u8);
        }
    } else {
        let peak = img.iter().cloned().fold(0.0f64, f64::max).max(1e-30);
        for v in img {
            let g = (1.0 + 999.0 * v / peak).ln() / (1000.0f64).ln();
            buf.push((255.0 * g) as u8);
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "P5\n{N} {N}\n255")?;
    f.write_all(&buf)?;
    f.flush()
}
