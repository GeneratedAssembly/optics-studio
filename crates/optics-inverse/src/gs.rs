//! Gerchberg–Saxton (GS) phase retrieval and DOE design — textbook §5.5.
//!
//! ## Algorithm (bidirectional loop, Eqs. 5.14–5.18)
//!
//! A phase-only DOE [`Carrier::GridPhase`] sits **between** an input plane
//! and a target plane separated by $z_{\text{total}}$, at the midpoint, so
//! the coherent relay is **two ASM half-transforms**:
//!
//! ```text
//! U_in ──ASM(+z/2)──▶ A·e^{iψ} (DOE plane) ──ASM(+z/2)──▶ U_t (target plane)
//! ```
//!
//! The illumination arriving at the DOE plane,
//! $A\,e^{i\psi} = \mathcal{A}(z/2)\{U_{\text{in}}\}$, is computed once;
//! its amplitude $A$ (masked by the carrier's clear aperture) is the
//! DOE-plane energy constraint. Each iteration:
//!
//! 1. **Forward half-transform**:
//!    $U_t = \mathcal{A}(z/2)\{A\,\Pi\,e^{i\varphi_k}\}$ ($\Pi$ = aperture
//!    indicator).
//! 2. **Error signal**: $e = |U_t| - T$ (target-plane amplitude mismatch,
//!    the residual of Eq. 5.18) and, after step 4, the DOE-plane phase
//!    residual $\Delta\varphi = \operatorname{wrap}(\arg V - \varphi_k)$.
//! 3. **Fourier-plane energy projection** (Eq. 5.16): the target-plane
//!    amplitude is replaced by $T$ and the phase is kept — with the `Fpr`
//!    variant the replacement is relaxed:
//!    $|U_t| \to (1-\beta)\,|U_t| + \beta\,T$.
//! 4. **Back-propagation** $\mathcal{A}(-z/2)$ (the ASM adjoint on the
//!    passband) to the DOE plane, giving $V$.
//! 5. **DOE-plane energy projection**: the arriving illumination amplitude
//!    $A$ is re-imposed and only the phase is updated — with the `Dvr`
//!    variant the update is damped: $\varphi_{k+1} = \varphi_k +
//!    \beta\,\Delta\varphi$.
//!
//! The phase is encoded into the carrier's slot bounds every iteration, so
//! the iterate is always a legal `GridPhase` parameter vector
//! ([`GsSolution::phase`]).
//!
//! ## Variants (iteration options)
//!
//! [`GsVariant::Classic`] is the textbook algorithm: hard Fourier-plane
//! replacement, full phase update. The adaptive options named in the task
//! (DVR/FPR — the acronyms are not defined in §5.5 or the public GS
//! literature, so this module states its reading):
//!
//! - [`GsVariant::Dvr`] — **D**amped **V**ariable **R**elaxation: the DOE
//!   phase update is relaxed by $\beta \in (0, 2)$; $\beta < 1$ damps
//!   oscillation near local minima, $\beta > 1$ over-relaxes (textbook
//!   over-relaxation).
//! - [`GsVariant::Fpr`] — **F**ourier-**P**lane **R**elaxation: the
//!   target-plane amplitude replacement is softened by
//!   $\beta \in (0, 1]$; $\beta = 1$ is the hard textbook replacement.
//!
//! Both reduce to `Classic` at $\beta = 1$.
//!
//! ## Conventions (pinned by tests — agents.md §4 numerical-conventions)
//!
//! - FFT scaling/sign/bin order are inherited from
//!   `optics_core::propagation::fft` (NumPy convention: unnormalized
//!   forward, $1/N$ inverse, $e^{-i2\pi kn/N}$ kernel, unshifted bins).
//!   ASM is transfer-function form, so the scaling factors cancel between
//!   $\mathcal{F}$ and $\mathcal{F}^{-1}$; the sign and shift semantics are
//!   pinned end-to-end by the module tests `asm_roundtrip_is_identity`,
//!   `two_half_transforms_compose_to_full_transform`,
//!   `converging_lens_phase_focuses_at_positive_z` and
//!   `blaze_phase_steers_spot_to_positive_x`.
//! - Grids are the core center-origin row-major [`Field`] lattice: sample
//!   $(i, j)$ sits at $x = (i - (n_x-1)/2)\,\Delta x$. The carrier lattice
//!   (dims, pitch) must match the input field grid — validated in [`run`].
//! - Phase sign: transmission is $e^{+i\varphi}$; a converging thin lens is
//!   $\varphi(r) = -k r^2 / (2f)$ and focuses at $+z = f$ (test-pinned).
//! - Back-propagation uses $\mathcal{A}(-z/2)$, the adjoint of the forward
//!   half-transform on propagating content (plan §2; pinned by
//!   `asm_roundtrip_is_identity`).
//! - Single band, fully coherent: GS is the monochromatic design loop
//!   (multiband shaping is out of scope for task 9.8).
//!
//! ## Seams
//!
//! The convergence trace and stopping test run through the 9.5 [`Merit`]
//! seam: each iteration wraps the current target-plane intensity into a
//! single-band/single-scene [`Observation`] (identical layout to the
//! renderer's) and evaluates the user merit on it; `MseToTarget` is the
//! natural choice. [`GsSolution::history`] carries per-iteration merit,
//! best-so-far merit, the Eq. 5.18 amplitude MSE, and error-signal RMS;
//! [`GsSolution`] keeps the final error-signal vectors themselves, not
//! just the retrieved field.

use std::f64::consts::PI;

use num_complex::Complex64;
use rand::SeedableRng;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;

use optics_core::field::{BandField, Field};
use optics_core::propagation::asm::{AsmConfig, propagate};

use crate::Result;
use crate::carrier::Carrier;
use crate::error::InverseError;
use crate::merit::Merit;
use crate::observe::{Channel, Observation};

const TAU: f64 = 2.0 * PI;

/// Wrap a phase difference into $[-\pi, \pi)$.
fn wrap_pi(v: f64) -> f64 {
    (v + PI).rem_euclid(TAU) - PI
}

/// Encode an unconstrained phase into the carrier's slot bounds: wrap into
/// $[\text{lo}, \text{lo}+2\pi)$ when the span covers $2\pi$, else clamp
/// (quantized/limited-depth DOEs keep their bounds honored).
fn encode_bounds(phi: f64, lo: f64, hi: f64) -> f64 {
    if hi - lo >= TAU {
        lo + (phi - lo).rem_euclid(TAU)
    } else {
        phi.clamp(lo, hi)
    }
}

/// Iteration scheme for [`run`]. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GsVariant {
    /// Textbook §5.5 GS: hard Fourier-plane replacement, full phase update.
    Classic,
    /// Damped Variable Relaxation: textbook Fourier replacement, damped
    /// DOE phase update $\varphi \mathrel{+}= \beta\,\Delta\varphi$,
    /// $\beta \in (0, 2)$.
    Dvr {
        /// Phase-update relaxation factor.
        beta: f64,
    },
    /// Fourier-Plane Relaxation: relaxed target-plane amplitude
    /// replacement with weight $\beta \in (0, 1]$, full phase update.
    Fpr {
        /// Fourier-plane replacement weight ($1.0$ = hard).
        beta: f64,
    },
}

impl GsVariant {
    /// Fourier-plane replacement weight.
    fn fourier_beta(&self) -> f64 {
        match self {
            Self::Classic | Self::Dvr { .. } => 1.0,
            Self::Fpr { beta } => *beta,
        }
    }

    /// DOE-plane phase-update relaxation weight.
    fn phase_beta(&self) -> f64 {
        match self {
            Self::Classic | Self::Fpr { .. } => 1.0,
            Self::Dvr { beta } => *beta,
        }
    }

    fn validate(&self) -> std::result::Result<(), String> {
        match *self {
            Self::Classic => Ok(()),
            Self::Dvr { beta } => {
                if beta.is_finite() && beta > 0.0 && beta < 2.0 {
                    Ok(())
                } else {
                    Err(format!("Dvr beta must be in (0, 2), got {beta}"))
                }
            }
            Self::Fpr { beta } => {
                if beta.is_finite() && beta > 0.0 && beta <= 1.0 {
                    Ok(())
                } else {
                    Err(format!("Fpr beta must be in (0, 1], got {beta}"))
                }
            }
        }
    }
}

/// Initial phase estimate. The textbook starts from a random draw
/// (§5.5.1); zero phase is the natural start for retrieval problems whose
/// answer is a smooth profile (a lens).
#[derive(Debug, Clone, PartialEq)]
pub enum GsInit {
    /// Flat wavefront: $\varphi_0 = $ the carrier's current values.
    Zero,
    /// Uniform random draw over the carrier's phase range, seeded.
    Random {
        /// RNG seed (kept for reproducible designs).
        seed: u64,
    },
    /// Explicit starting phase (radians, row-major, grid length); encoded
    /// into the carrier bounds before the first iteration.
    Given(Vec<f64>),
}

/// Iteration and stopping configuration for [`run`].
#[derive(Debug, Clone, PartialEq)]
pub struct GsConfig {
    /// Hard cap on iterations (each iteration = two ASM half-transforms).
    pub max_iterations: usize,
    /// Stopping test: converged when the best-so-far merit has improved
    /// by less than `merit_tolerance` (relative) over the last
    /// `patience` iterations.
    pub patience: usize,
    /// Relative merit improvement counted as progress (see `patience`).
    pub merit_tolerance: f64,
    /// Iteration scheme.
    pub variant: GsVariant,
    /// Initial phase estimate.
    pub init: GsInit,
}

impl Default for GsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            patience: 10,
            merit_tolerance: 1e-4,
            variant: GsVariant::Classic,
            init: GsInit::Zero,
        }
    }
}

/// Everything [`run`] needs: the DOE carrier, the coherent relay geometry,
/// the target constraint, and the merit seam for the stopping test.
pub struct GsProblem<'a> {
    /// Phase-only DOE carrier; must be a `GridPhase` whose lattice matches
    /// [`GsProblem::input`] (validated).
    pub carrier: &'a Carrier,
    /// Complex field at the input plane (single band).
    pub input: BandField,
    /// Input-plane → target-plane separation $z_{\text{total}} > 0$ (m);
    /// the DOE sits at the midpoint, giving two ASM half-transforms.
    pub z_total: f64,
    /// Target-plane amplitude $T$ (row-major, one frame); finite and
    /// non-negative. This array drives the Fourier-plane projection and
    /// the Eq. 5.18 diagnostic; the [`Merit`] receives the intensity
    /// frames through the [`Observation`] seam.
    pub target_amplitude: Vec<f64>,
    /// Merit seam (task 9.5) evaluated on each iteration's rendered
    /// [`Observation`]; drives the convergence trace and stopping test.
    pub merit: &'a dyn Merit,
    /// ASM configuration for the half-transforms (forward and adjoint).
    pub asm: AsmConfig,
    /// Iteration / stopping configuration.
    pub config: GsConfig,
}

/// One iteration's convergence diagnostics (see [`GsSolution::history`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GsRecord {
    /// Iteration index (0-based, forward pass of the current phase).
    pub iteration: usize,
    /// Merit evaluated on this iteration's target-plane intensity.
    pub merit: f64,
    /// Best-so-far merit through this iteration (monotone non-increasing).
    pub best_merit: f64,
    /// Textbook Eq. 5.18 amplitude MSE: $\frac1N \sum (|U_t| - T)^2$.
    pub amplitude_mse: f64,
    /// RMS of the target-plane amplitude error signal $|U_t| - T$.
    pub target_error_rms: f64,
    /// RMS of the DOE-plane phase update $\Delta\varphi$ (before variant
    /// relaxation) — the fixed-point residual; tends to 0 as GS converges.
    pub phase_update_rms: f64,
}

/// Result of a [`run`]: retrieved/design phase plus full diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct GsSolution {
    /// Best-so-far DOE phase (radians, row-major, encoded into the
    /// carrier's slot bounds). Feed directly to `Carrier::build` as the
    /// `GridPhase` parameter vector.
    pub phase: Vec<f64>,
    /// Merit value at the best iteration.
    pub merit: f64,
    /// Iteration index that achieved [`GsSolution::merit`].
    pub best_iteration: usize,
    /// Eq. 5.18 amplitude MSE at the best iteration.
    pub amplitude_mse: f64,
    /// `true` when the patience-based stopping test triggered before the
    /// iteration cap.
    pub converged: bool,
    /// Iterations actually run.
    pub iterations: usize,
    /// Per-iteration diagnostics, oldest first.
    pub history: Vec<GsRecord>,
    /// Target-plane intensity at the best iteration (what the design
    /// actually produces).
    pub target_intensity: Vec<f64>,
    /// Final-pass error signal at the target plane: $|U_t| - T$.
    pub target_amplitude_error: Vec<f64>,
    /// Final-pass error signal at the DOE plane: the wrapped phase
    /// residual $\Delta\varphi$ that drove the last update — the negative
    /// gradient direction of the Eq. 5.18 amplitude MSE, preconditioned by
    /// the illumination intensity: $\nabla_\varphi M = -\frac{2}{N} A^2
    /// \odot \Delta\varphi$ to first order (pinned by ADR-9.9 and
    /// `adjoint::gs_phase_error_is_adjoint_mse_negative_gradient`).
    pub phase_error: Vec<f64>,
    /// Illumination amplitude $A$ arriving at the DOE plane (the DOE-plane
    /// energy constraint used by every projection).
    pub illumination_amplitude: Vec<f64>,
}

impl GsSolution {
    /// The best phase as a `GridPhase` parameter vector (same data; named
    /// for the carrier seam).
    pub fn carrier_values(&self) -> &[f64] {
        &self.phase
    }
}

/// Map a core construction/propagation failure onto the crate error type
/// (renderer precedent, `observe.rs`).
fn render_err(e: impl std::fmt::Display) -> InverseError {
    InverseError::Render {
        reason: e.to_string(),
    }
}

/// Propagate a sample buffer a signed distance through the core ASM,
/// preserving grid geometry.
fn propagate_samples(
    samples: &[Complex64],
    lambda: f64,
    dims: (usize, usize),
    pitch: (f64, f64),
    z: f64,
    asm: &AsmConfig,
) -> Result<Vec<Complex64>> {
    let field =
        Field::new(samples.to_vec(), dims.0, dims.1, pitch.0, pitch.1).map_err(render_err)?;
    let band = BandField::new(lambda, field).map_err(render_err)?;
    let out = propagate(&band, z, asm).map_err(render_err)?;
    Ok(out.field().samples().to_vec())
}

/// Run the Gerchberg–Saxton loop on a [`GsProblem`].
///
/// # Errors
/// [`InverseError`] for: non-`GridPhase` carrier, carrier/input lattice
/// mismatch, wrong-length or invalid target, non-positive/non-finite
/// `z_total`, out-of-range variant $\beta$, zero iterations, or an ASM
/// propagation failure (strict-mode aliasing included).
pub fn run(problem: &GsProblem) -> Result<GsSolution> {
    let geom = problem
        .carrier
        .grid_geom()
        .ok_or_else(|| InverseError::Carrier {
            reason: "GS requires a GridPhase carrier (phase-only DOE)".into(),
        })?;
    let (lo, hi) = problem
        .carrier
        .slot_bounds()
        .ok_or_else(|| InverseError::Carrier {
            reason: "GS carrier has no parameter slots".into(),
        })?;

    let (nx, ny) = problem.input.field().dims();
    let (dx, dy) = problem.input.field().pitch();
    let n = nx * ny;
    if geom.dims != (nx, ny) {
        return Err(InverseError::Carrier {
            reason: format!(
                "carrier lattice {:?} does not match input grid {:?}",
                geom.dims,
                (nx, ny)
            ),
        });
    }
    if (geom.pitch.0 - dx).abs() > 1e-12 * dx.max(1.0)
        || (geom.pitch.1 - dy).abs() > 1e-12 * dy.max(1.0)
    {
        return Err(InverseError::Carrier {
            reason: format!(
                "carrier pitch {:?} does not match input pitch {:?}",
                geom.pitch,
                (dx, dy)
            ),
        });
    }
    if problem.target_amplitude.len() != n {
        return Err(InverseError::WrongDimension {
            expected: n,
            actual: problem.target_amplitude.len(),
        });
    }
    for (k, &t) in problem.target_amplitude.iter().enumerate() {
        if !t.is_finite() {
            return Err(InverseError::NotFinite {
                name: format!("target_amplitude[{k}]"),
                value: t,
            });
        }
        if t < 0.0 {
            return Err(InverseError::OutOfBounds {
                name: format!("target_amplitude[{k}]"),
                value: t,
                lo: 0.0,
                hi: f64::INFINITY,
            });
        }
    }
    if !(problem.z_total.is_finite() && problem.z_total > 0.0) {
        return Err(InverseError::Render {
            reason: format!(
                "z_total must be positive and finite, got {}",
                problem.z_total
            ),
        });
    }
    problem
        .config
        .variant
        .validate()
        .map_err(|reason| InverseError::Render { reason })?;
    if problem.config.max_iterations == 0 {
        return Err(InverseError::Render {
            reason: "gs max_iterations must be > 0".into(),
        });
    }

    let lambda = problem.input.lambda();
    let half_z = problem.z_total / 2.0;
    let aperture_open: Vec<bool> = (0..ny)
        .flat_map(|iy| (0..nx).map(move |ix| (ix, iy)))
        .map(|(ix, iy)| {
            let (x, y) = problem.input.field().coord(ix, iy);
            geom.aperture.contains(x - geom.center.0, y - geom.center.1)
        })
        .collect();

    // Illumination arriving at the DOE plane: fixed energy constraint.
    let arriving = propagate_samples(
        problem.input.field().samples(),
        lambda,
        (nx, ny),
        (dx, dy),
        half_z,
        &problem.asm,
    )?;
    let illum: Vec<f64> = arriving.iter().map(|s| s.norm()).collect();

    // Initial phase estimate, encoded into carrier bounds.
    let mut phase: Vec<f64> = match &problem.config.init {
        GsInit::Zero => problem.carrier.params().values(),
        GsInit::Random { seed } => {
            let mut rng = StdRng::seed_from_u64(*seed);
            let span = (hi - lo).min(TAU).max(0.0);
            if span > 0.0 {
                let u = Uniform::new(0.0, span);
                (0..n).map(|_| lo + u.sample(&mut rng)).collect()
            } else {
                vec![lo; n]
            }
        }
        GsInit::Given(v) => {
            if v.len() != n {
                return Err(InverseError::WrongDimension {
                    expected: n,
                    actual: v.len(),
                });
            }
            v.to_vec()
        }
    };
    for p in phase.iter_mut() {
        *p = encode_bounds(*p, lo, hi);
    }

    let beta_f = problem.config.variant.fourier_beta();
    let beta_p = problem.config.variant.phase_beta();

    let mut best: Option<(f64, usize, Vec<f64>, Vec<f64>)> = None;
    let mut history: Vec<GsRecord> = Vec::new();
    let mut last_target_error = vec![0.0f64; n];
    let mut last_phase_error = vec![0.0f64; n];
    let mut converged = false;

    for it in 0..problem.config.max_iterations {
        // Forward half-transform through the current DOE.
        let mask: Vec<Complex64> = (0..n)
            .map(|k| {
                if aperture_open[k] {
                    Complex64::from_polar(illum[k], phase[k])
                } else {
                    Complex64::ZERO
                }
            })
            .collect();
        let u_t = propagate_samples(&mask, lambda, (nx, ny), (dx, dy), half_z, &problem.asm)?;

        // Error signal + Eq. 5.18 amplitude MSE at the target plane.
        let amp_t: Vec<f64> = u_t.iter().map(|s| s.norm()).collect();
        let intensity: Vec<f64> = amp_t.iter().map(|a| a * a).collect();
        let err: Vec<f64> = amp_t
            .iter()
            .zip(problem.target_amplitude.iter())
            .map(|(a, t)| a - t)
            .collect();
        let amp_mse = err.iter().map(|e| e * e).sum::<f64>() / n as f64;

        // Merit seam: single-band/single-scene Observation of this frame.
        let obs = Observation {
            channels: vec![Channel::Intensity],
            bands: vec![lambda],
            scenes: vec![(0.0, 0.0)],
            dims: (nx, ny),
            pitch: (dx, dy),
            data: intensity.clone(),
            fd_steps: (0.0, 0.0),
        };
        let merit = problem.merit.evaluate(&obs)?;

        let improved = match &best {
            Some((bm, ..)) => merit < *bm,
            None => true,
        };
        let best_merit = if improved {
            best = Some((merit, it, phase.clone(), intensity.clone()));
            merit
        } else {
            let (bm, ..) = best.as_ref().expect("best set on first iteration");
            *bm
        };
        history.push(GsRecord {
            iteration: it,
            merit,
            best_merit,
            amplitude_mse: amp_mse,
            target_error_rms: amp_mse.sqrt(),
            phase_update_rms: 0.0, // filled after the backward pass
        });
        last_target_error = err;

        // Fourier-plane energy projection (Eq. 5.16), relaxed by beta_f.
        let u_t_proj: Vec<Complex64> = u_t
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let a = (1.0 - beta_f) * amp_t[k] + beta_f * problem.target_amplitude[k];
                let m = s.norm();
                if m > 0.0 {
                    s * (a / m)
                } else {
                    Complex64::new(a, 0.0)
                }
            })
            .collect();

        // Adjoint half-transform back to the DOE plane.
        let v = propagate_samples(&u_t_proj, lambda, (nx, ny), (dx, dy), -half_z, &problem.asm)?;

        // DOE-plane energy projection: keep phase, relax the update.
        let dphi: Vec<f64> = v
            .iter()
            .zip(phase.iter())
            .map(|(s, p)| wrap_pi(s.arg() - p))
            .collect();
        for (p, d) in phase.iter_mut().zip(dphi.iter()) {
            *p = encode_bounds(*p + beta_p * *d, lo, hi);
        }
        last_phase_error = dphi;
        let pu_rms = last_phase_error.iter().map(|d| d * d).sum::<f64>().sqrt() / (n as f64).sqrt();
        history.last_mut().expect("just pushed").phase_update_rms = pu_rms;

        // Patience-based stopping on the best-so-far trace.
        if it + 1 > problem.config.patience {
            let earlier = history[history.len() - 1 - problem.config.patience].best_merit;
            let scale = earlier.abs().max(1e-30);
            if (earlier - best_merit) / scale < problem.config.merit_tolerance {
                converged = true;
                break;
            }
        }
    }

    let (merit, best_iteration, best_phase, best_intensity) =
        best.ok_or_else(|| InverseError::Render {
            reason: "gs loop ran zero iterations".into(),
        })?;
    Ok(GsSolution {
        phase: best_phase,
        merit,
        best_iteration,
        amplitude_mse: history[best_iteration].amplitude_mse,
        converged,
        iterations: history.len(),
        history,
        target_intensity: best_intensity,
        target_amplitude_error: last_target_error,
        phase_error: last_phase_error,
        illumination_amplitude: illum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carrier::Design;
    use crate::merit::{MseToTarget, Selector};
    use crate::params::Params;
    use optics_core::element::ClearAperture;
    use optics_core::sources::{GaussianBeam, gaussian};

    const L: f64 = 500e-9;

    fn gaussian_band(w0: f64, n: usize, d: f64) -> BandField {
        let beam = GaussianBeam {
            w0,
            amplitude: 1.0,
            center: (0.0, 0.0),
        };
        gaussian(&beam, 0.0, L, n, n, d, d).unwrap()
    }

    fn plane_wave(n: usize, d: f64) -> BandField {
        BandField::plane_wave(0.0, 0.0, 1.0, n, n, d, d, L).unwrap()
    }

    fn asm_plain() -> AsmConfig {
        AsmConfig {
            band_limited: false,
            strict: false,
            ..Default::default()
        }
    }

    fn grid_carrier(n: usize, d: f64, aperture: ClearAperture) -> Carrier {
        Carrier::grid_phase((n, n), (d, d), aperture, (0.0, 0.0), -PI, PI, 0.0).unwrap()
    }

    fn mse_merit(target: &[f64]) -> MseToTarget {
        MseToTarget {
            target: target.iter().map(|t| t * t).collect(),
            selector: Selector::intensity(None, None),
        }
    }

    /// Row-major (ix, iy) index pairs for a square grid.
    fn grid(n: usize) -> impl Iterator<Item = (usize, usize)> {
        (0..n).flat_map(move |iy| (0..n).map(move |ix| (ix, iy)))
    }

    // -------------------------------------------------------------------
    // Convention pins (agents.md §4 numerical-conventions)
    // -------------------------------------------------------------------

    #[test]
    fn asm_roundtrip_is_identity() {
        // A(+z) then A(−z) = identity on propagating content: pins the
        // adjoint/conjugation convention GS relies on for back-propagation.
        let (n, d, z) = (64, 10e-6, 2e-3);
        let band = gaussian_band(60e-6, n, d);
        let out =
            propagate_samples(band.field().samples(), L, (n, n), (d, d), z, &asm_plain()).unwrap();
        let back = propagate_samples(&out, L, (n, n), (d, d), -z, &asm_plain()).unwrap();
        let mut worst = 0.0f64;
        for (a, b) in band.field().samples().iter().zip(&back) {
            worst = worst.max((a - b).norm());
        }
        assert!(worst < 1e-9, "roundtrip max abs error {worst}");
    }

    #[test]
    fn two_half_transforms_compose_to_full_transform() {
        // The GS relay (z/2 + z/2) equals a single full ASM step: pins the
        // half-transform split convention (padding/crop differ only at
        // guard-band level).
        let (n, d, z) = (64, 10e-6, 2e-3);
        let band = gaussian_band(60e-6, n, d);
        let asm = asm_plain();
        let full =
            propagate_samples(band.field().samples(), L, (n, n), (d, d), 2.0 * z, &asm).unwrap();
        let h1 = propagate_samples(band.field().samples(), L, (n, n), (d, d), z, &asm).unwrap();
        let half = propagate_samples(&h1, L, (n, n), (d, d), z, &asm).unwrap();
        let scale: f64 = full.iter().map(|s| s.norm()).sum::<f64>() / (n * n) as f64;
        let mut worst = 0.0f64;
        for (a, b) in full.iter().zip(&half) {
            worst = worst.max((a - b).norm());
        }
        assert!(
            worst < 1e-6 * scale,
            "half-transform composition error {worst} vs scale {scale}"
        );
    }

    #[test]
    fn converging_lens_phase_focuses_at_positive_z() {
        // φ(r) = −k r²/(2f) with transmission e^{+iφ} concentrates the
        // target-plane intensity at z = +f: pins the phase sign and the
        // propagation direction convention.
        let (n, d, f) = (128, 5e-6, 12e-3);
        let k = TAU / L;
        let pw = plane_wave(n, d);
        let masked: Vec<Complex64> = grid(n)
            .map(|(ix, iy)| {
                let (x, y) = pw.field().coord(ix, iy);
                let r2 = x * x + y * y;
                pw.field().sample(ix, iy) * (-(k * r2 / (2.0 * f)) * Complex64::I).exp()
            })
            .collect();
        let focused = propagate_samples(&masked, L, (n, n), (d, d), f, &asm_plain()).unwrap();
        let bare =
            propagate_samples(pw.field().samples(), L, (n, n), (d, d), f, &asm_plain()).unwrap();
        let peak = |buf: &[Complex64]| buf.iter().map(|s| s.norm_sqr()).fold(0.0f64, f64::max);
        // A focused spot has a much higher peak than the bare plane wave.
        assert!(
            peak(&focused) > 5.0 * peak(&bare),
            "lens did not concentrate intensity: {} vs {}",
            peak(&focused),
            peak(&bare)
        );
        // And its centroid sits at the grid center (converging, not
        // diverging, lens).
        let mut sx = 0.0;
        let mut sw = 0.0;
        for (ix, iy) in grid(n) {
            let w = focused[iy * n + ix].norm_sqr();
            let (x, _) = pw.field().coord(ix, iy);
            sx += w * x;
            sw += w;
        }
        assert!((sx / sw).abs() < 10e-6, "focus off-axis: {}", sx / sw);
    }

    #[test]
    fn blaze_phase_steers_spot_to_positive_x() {
        // φ(x) = +2π x/Λ (a kinoform grating) deflects the beam toward
        // +x: the target-plane centroid shifts by ≈ +λ z/Λ. Pins the FFT
        // sign convention end-to-end through the ASM chain.
        let (n, d, z) = (128, 5e-6, 4e-3);
        let period = 60e-6;
        // Gaussian envelope (not a hard-window plane wave): the steered
        // beam must stay inside the grid so the crop cannot bias the
        // centroid back toward zero.
        let pw = gaussian_band(120e-6, n, d);
        let masked: Vec<Complex64> = grid(n)
            .map(|(ix, iy)| {
                let (x, _) = pw.field().coord(ix, iy);
                pw.field().sample(ix, iy) * (TAU * x / period * Complex64::I).exp()
            })
            .collect();
        let out = propagate_samples(&masked, L, (n, n), (d, d), z, &asm_plain()).unwrap();
        let mut sx = 0.0;
        let mut sw = 0.0;
        for (ix, iy) in grid(n) {
            let w = out[iy * n + ix].norm_sqr();
            let (x, _) = pw.field().coord(ix, iy);
            sx += w * x;
            sw += w;
        }
        let shift = sx / sw;
        let expected = L * z / period; // sin θ ≈ λ/Λ
        assert!(
            (shift - expected).abs() < 0.15 * expected,
            "blaze steering sign/magnitude wrong: shift {shift}, expected {expected}"
        );
    }

    #[test]
    fn phase_encoding_wraps_and_clamps_to_bounds() {
        // Full-depth DOE: span ≥ 2π wraps modulo 2π.
        for &p in &[PI, 3.0 * PI, -PI - 0.5, 7.0] {
            let e = encode_bounds(p, -PI, PI);
            assert!(e >= -PI && e < PI, "{p} -> {e}");
            assert!(wrap_pi(e - p).abs() < 1e-12, "{p} -> {e} not congruent");
        }
        // Limited-depth DOE: span < 2π clamps (fabrication constraint).
        assert_eq!(encode_bounds(4.0, 0.0, PI), PI);
        assert_eq!(encode_bounds(-1.0, 0.0, PI), 0.0);
        assert_eq!(encode_bounds(1.0, 0.0, PI), 1.0);
    }

    // -------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------

    #[test]
    fn run_rejects_non_grid_carrier_and_mismatch() {
        let analytic = Carrier::analytic(Params::new(), |_v, _l| Ok(Design::new()));
        let band = plane_wave(16, 10e-6);
        let merit = mse_merit(&[1.0; 16 * 16]);
        let problem = GsProblem {
            carrier: &analytic,
            input: band.clone(),
            z_total: 1e-3,
            target_amplitude: vec![1.0; 16 * 16],
            merit: &merit,
            asm: asm_plain(),
            config: GsConfig::default(),
        };
        assert!(matches!(run(&problem), Err(InverseError::Carrier { .. })));

        // Lattice mismatch: carrier dims differ from the input grid.
        let small = grid_carrier(8, 10e-6, ClearAperture::Field);
        let problem = GsProblem {
            carrier: &small,
            input: band,
            z_total: 1e-3,
            target_amplitude: merit.target.clone(),
            merit: &merit,
            asm: asm_plain(),
            config: GsConfig::default(),
        };
        assert!(matches!(run(&problem), Err(InverseError::Carrier { .. })));
    }

    #[test]
    fn run_rejects_invalid_setup() {
        let carrier = grid_carrier(16, 10e-6, ClearAperture::Field);
        let band = plane_wave(16, 10e-6);
        let merit = mse_merit(&[1.0; 16 * 16]);
        let mut problem = GsProblem {
            carrier: &carrier,
            input: band,
            z_total: 1e-3,
            target_amplitude: vec![1.0; 16 * 16],
            merit: &merit,
            asm: asm_plain(),
            config: GsConfig::default(),
        };
        // wrong target length
        problem.target_amplitude = vec![1.0; 10];
        assert!(matches!(
            run(&problem),
            Err(InverseError::WrongDimension { .. })
        ));
        // negative target amplitude
        problem.target_amplitude = vec![-0.1; 16 * 16];
        assert!(matches!(
            run(&problem),
            Err(InverseError::OutOfBounds { .. })
        ));
        // non-positive z_total
        problem.target_amplitude = vec![1.0; 16 * 16];
        problem.z_total = 0.0;
        assert!(run(&problem).is_err());
        // out-of-range variant beta
        problem.z_total = 1e-3;
        problem.config.variant = GsVariant::Dvr { beta: 3.0 };
        assert!(run(&problem).is_err());
        problem.config.variant = GsVariant::Fpr { beta: 0.0 };
        assert!(run(&problem).is_err());
        // zero iterations
        problem.config.variant = GsVariant::Classic;
        problem.config.max_iterations = 0;
        assert!(run(&problem).is_err());
    }

    // -------------------------------------------------------------------
    // Golden 1: top-hat shaper (textbook §5.5 beam-shaping box)
    // -------------------------------------------------------------------

    #[test]
    fn golden_tophat_shaper_is_flat_and_mse_monotone() {
        let (n, d) = (128, 8e-6);
        let z_total = 8e-3;
        let target_r = 250e-6;
        let carrier = grid_carrier(n, d, ClearAperture::Field);
        let input = gaussian_band(150e-6, n, d);

        // Flat-top target with amplitude scaled to the input power: a
        // phase-only DOE conserves power, so a target demanding more
        // energy than the illumination carries is unreachable by
        // construction. T²·(disk pixels) = Σ|U_in|².
        let power: f64 = input.field().samples().iter().map(|s| s.norm_sqr()).sum();
        let disk_pixels = grid(n)
            .filter(|&(ix, iy)| {
                let (x, y) = input.field().coord(ix, iy);
                x * x + y * y <= target_r * target_r
            })
            .count() as f64;
        let amp = (power / disk_pixels).sqrt();
        let target: Vec<f64> = grid(n)
            .map(|(ix, iy)| {
                let (x, y) = input.field().coord(ix, iy);
                if x * x + y * y <= target_r * target_r {
                    amp
                } else {
                    0.0
                }
            })
            .collect();
        let merit = mse_merit(&target);
        let problem = GsProblem {
            carrier: &carrier,
            input,
            z_total,
            target_amplitude: target.clone(),
            merit: &merit,
            asm: asm_plain(),
            config: GsConfig {
                max_iterations: 120,
                patience: 10,
                merit_tolerance: 1e-4,
                variant: GsVariant::Classic,
                init: GsInit::Random { seed: 7 },
            },
        };
        let sol = run(&problem).unwrap();

        // 1. Best-so-far merit trace is non-increasing (acceptance).
        for w in sol.history.windows(2) {
            assert!(
                w[1].best_merit <= w[0].best_merit + 1e-15,
                "best-so-far trace not monotone: {:?}",
                sol.history.iter().map(|r| r.best_merit).collect::<Vec<_>>()
            );
        }
        // 2. The design materially improves on the random start.
        let first = sol.history.first().unwrap().merit;
        assert!(
            sol.merit < 0.5 * first,
            "GS did not converge: best {} vs first {}",
            sol.merit,
            first
        );
        // 3. Flat target within the aperture: interior uniformity.
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        let mut cnt = 0usize;
        let interior_r = 0.7 * target_r;
        for (ix, iy) in grid(n) {
            let (x, y) = problem.input.field().coord(ix, iy);
            if x * x + y * y <= interior_r * interior_r {
                let v = sol.target_intensity[iy * n + ix];
                sum += v;
                sum2 += v * v;
                cnt += 1;
            }
        }
        let mean = sum / cnt as f64;
        let var = (sum2 / cnt as f64 - mean * mean).max(0.0);
        let cv = var.sqrt() / mean;
        assert!(cv < 0.35, "top-hat interior not flat: CV {cv}");
        // 4. Error signal + diagnostics exposed, not just the final field.
        assert_eq!(sol.target_amplitude_error.len(), n * n);
        assert_eq!(sol.phase_error.len(), n * n);
        assert_eq!(sol.illumination_amplitude.len(), n * n);
        assert!(sol.history.iter().all(|r| r.target_error_rms.is_finite()));
        assert!(sol.history.iter().all(|r| r.phase_update_rms.is_finite()));
        // 5. Returned phase is a legal GridPhase parameter vector.
        assert!(carrier.build(sol.carrier_values(), L).is_ok());
    }

    #[test]
    fn dvr_variant_converges_on_tophat() {
        // Adaptive iteration option: damped phase update must still reach
        // a flat top and keep a monotone best-so-far trace.
        let (n, d) = (96, 8e-6);
        let target_r = 220e-6;
        let carrier = grid_carrier(n, d, ClearAperture::Field);
        let input = gaussian_band(140e-6, n, d);
        let power: f64 = input.field().samples().iter().map(|s| s.norm_sqr()).sum();
        let disk_pixels = grid(n)
            .filter(|&(ix, iy)| {
                let (x, y) = input.field().coord(ix, iy);
                x * x + y * y <= target_r * target_r
            })
            .count() as f64;
        let amp = (power / disk_pixels).sqrt();
        let target: Vec<f64> = grid(n)
            .map(|(ix, iy)| {
                let (x, y) = input.field().coord(ix, iy);
                if x * x + y * y <= target_r * target_r {
                    amp
                } else {
                    0.0
                }
            })
            .collect();
        let merit = mse_merit(&target);
        let problem = GsProblem {
            carrier: &carrier,
            input,
            z_total: 8e-3,
            target_amplitude: target,
            merit: &merit,
            asm: asm_plain(),
            config: GsConfig {
                max_iterations: 150,
                patience: 10,
                merit_tolerance: 1e-4,
                variant: GsVariant::Dvr { beta: 0.7 },
                init: GsInit::Random { seed: 11 },
            },
        };
        let sol = run(&problem).unwrap();
        for w in sol.history.windows(2) {
            assert!(w[1].best_merit <= w[0].best_merit + 1e-15);
        }
        let first = sol.history.first().unwrap().merit;
        assert!(
            sol.merit < 0.5 * first,
            "DVR did not converge: {} vs {}",
            sol.merit,
            first
        );
    }

    #[test]
    fn fpr_variant_stops_on_merit_seam() {
        // Adaptive iteration option + the 9.5-merit stopping test: relaxed
        // Fourier-plane replacement must run, converge via the patience
        // rule, and expose the stopping flag.
        let (n, d) = (64, 10e-6);
        let carrier = grid_carrier(n, d, ClearAperture::Field);
        let input = gaussian_band(120e-6, n, d);
        let target: Vec<f64> = grid(n)
            .map(|(ix, iy)| {
                let (x, y) = input.field().coord(ix, iy);
                if x * x + y * y <= 180e-6f64.powi(2) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let merit = mse_merit(&target);
        let problem = GsProblem {
            carrier: &carrier,
            input,
            z_total: 6e-3,
            target_amplitude: target,
            merit: &merit,
            asm: asm_plain(),
            config: GsConfig {
                max_iterations: 120,
                patience: 5,
                merit_tolerance: 1e-3,
                variant: GsVariant::Fpr { beta: 0.8 },
                init: GsInit::Random { seed: 3 },
            },
        };
        let sol = run(&problem).unwrap();
        assert!(sol.converged, "Fpr run should hit the patience stop");
        assert!(sol.iterations <= 120);
    }

    // -------------------------------------------------------------------
    // Golden 2: recover a known phase-only lens from its target intensity
    // -------------------------------------------------------------------

    #[test]
    fn golden_lens_phase_recovered_from_target_intensity() {
        let (n, d) = (128, 5e-6);
        let f = 12e-3;
        let radius = 150e-6;
        let k = TAU / L;
        let carrier = grid_carrier(n, d, ClearAperture::Circle(radius));
        let input = plane_wave(n, d);

        // Truth: wrapped phase-only lens.
        let true_phase: Vec<f64> = grid(n)
            .map(|(ix, iy)| {
                let (x, y) = input.field().coord(ix, iy);
                wrap_pi(-k * (x * x + y * y) / (2.0 * f))
            })
            .collect();

        // Target: intensity of the true DOE one half-transform away (DOE
        // at z_total/2 = f from the target plane).
        let mask: Vec<Complex64> = grid(n)
            .map(|(ix, iy)| {
                let (x, y) = input.field().coord(ix, iy);
                let in_ap = x * x + y * y <= radius * radius;
                if in_ap {
                    Complex64::from_polar(1.0, true_phase[iy * n + ix])
                } else {
                    Complex64::ZERO
                }
            })
            .collect();
        let u_true = propagate_samples(&mask, L, (n, n), (d, d), f, &asm_plain()).unwrap();
        // Physical target amplitude (unnormalized): the GS fixed point is
        // the true DOE only if the target carries the same energy the
        // unit-illumination DOE actually delivers.
        let target: Vec<f64> = u_true.iter().map(|s| s.norm()).collect();

        let merit = mse_merit(&target);
        let problem = GsProblem {
            carrier: &carrier,
            input: input.clone(),
            z_total: 2.0 * f,
            target_amplitude: target.clone(),
            merit: &merit,
            asm: asm_plain(),
            config: GsConfig {
                max_iterations: 60,
                patience: 10,
                merit_tolerance: 1e-5,
                variant: GsVariant::Classic,
                init: GsInit::Zero,
            },
        };
        let sol = run(&problem).unwrap();

        // The retrieved design reproduces the measured intensity.
        let first = sol.history.first().unwrap().merit;
        assert!(
            sol.merit < 0.1 * first.max(1e-30),
            "lens retrieval MSE not low enough: {} vs first {}",
            sol.merit,
            first
        );

        // Phase recovery: modulo piston, the wrapped phase matches the
        // truth inside the aperture.
        let mut diff: Vec<f64> = Vec::new();
        for (ix, iy) in grid(n) {
            let (x, y) = input.field().coord(ix, iy);
            if x * x + y * y <= (0.9 * radius) * (0.9 * radius) {
                diff.push(wrap_pi(sol.phase[iy * n + ix] - true_phase[iy * n + ix]));
            }
        }
        let mean = diff.iter().sum::<f64>() / diff.len() as f64;
        let score: f64 = diff.iter().map(|dd| (dd - mean).cos()).sum::<f64>() / diff.len() as f64;
        assert!(
            score > 0.9,
            "recovered lens phase diverges from truth: cos-score {score}"
        );
    }
}
