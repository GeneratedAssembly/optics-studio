//! Finite-difference gradients over `Params` and the Adam descent solver
//! (task 9.7).
//!
//! [`gradient`] approximates $\partial M/\partial x_i$ by central
//! differences along each parameter axis:
//!
//! $$g_i = \frac{M(x + h_i e_i) - M(x - h_i e_i)}{2 h_i},$$
//!
//! consuming `2·dim` objective evaluations (renders) per step. The steps
//! $h_i$ come from [`Params`] — each slot's explicit `fd_step`, else the
//! documented default of $10^{-4}$ of the bound span ([`crate::params::Param::fd_step`])
//! — so the FD scale tracks the design space instead of a global epsilon.
//!
//! [`AdamConfig`] + [`Solver::Adam`](crate::solve::Solver::Adam) implement
//! Adam (Kingma & Ba, 2015) with the hypercam notebook's descent
//! discipline: **gradient-norm clipping** before the moment updates and a
//! **best-epoch snapshot** — the returned [`Solution`] reports the best
//! merit seen and the per-epoch convergence history, exactly like the 9.6
//! derivative-free solvers.
//!
//! ## Bounds and failure policy
//!
//! Same prevention discipline as `solve`: every probe, and every updated
//! point, is clamped into `[lo, hi]` before evaluation, so the carrier
//! never sees an invalid vector and nothing panics. At a bound the
//! central difference degrades gracefully to the one-sided slope over the
//! *effective* separation (the clamped points define the denominator).
//! A zero-width slot (`lo == hi`) contributes gradient 0 and never moves.
//! Non-finite objective values or gradient components surface as
//! [`InverseError::Render`].
//!
//! ## Adam update
//!
//! With per-coordinate moments $m, v$ and bias corrections
//! $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$:
//!
//! $$x_i \leftarrow \mathrm{clamp}\!\left(x_i - \eta_i \frac{\hat m_{i,t}}{\sqrt{\hat v_{i,t}} + \epsilon}\right),
//! \qquad \eta_i = \texttt{lr\_frac} \cdot (\mathrm{hi}_i - \mathrm{lo}_i),$$
//!
//! i.e. the learning rate is expressed as a *fraction of each slot's
//! bound span* per epoch — the same scale-free convention as the FD step
//! defaults. Since $\hat m/\sqrt{\hat v}$ is roughly sign-normalized,
//! $\eta_i$ is the maximum move per epoch; Adam is deterministic (no RNG
//! unless `random_init` starts the search at a seeded draw).
//!
//! For `Solver::Adam`, `StochasticConfig::budget` counts **epochs**
//! (each epoch costs `2·dim + 1` objective evaluations: the central
//! differences plus the post-update merit recorded in `history`).
//!
//! Physics anchor for curvature tuning (the golden problem): the textbook
//! lensmaker relation $\varphi = (n-1)c$ (§4.1.2, Eq. 4.3) — lens bending
//! is the classical merit-minimization over curvatures.

use rand::rngs::StdRng;

use crate::Result;
use crate::error::InverseError;
use crate::params::Params;
use crate::solve::{Solution, clamp_into, start};

/// Adam hyperparameters. All fields scalar → the config is `Copy` and can
/// ride inside [`crate::solve::Solver::Adam`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamConfig {
    /// Learning rate as a fraction of each slot's bound span per epoch
    /// (default 0.02 — at most 2% of the span per step).
    pub lr_frac: f64,
    /// First-moment (mean) decay (default 0.9).
    pub beta1: f64,
    /// Second-moment (variance) decay (default 0.999).
    pub beta2: f64,
    /// Denominator floor added to $\sqrt{\hat v}$ (default 1e-8, in
    /// gradient units).
    pub eps: f64,
    /// Gradient L2-norm clip (default `Some(1.0)`, objective-per-parameter
    /// units). Adam's per-coordinate normalization makes the trajectory
    /// insensitive to overall gradient scale, so clipping mainly guards
    /// against transient spikes from noisy renders.
    pub clip_norm: Option<f64>,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr_frac: 0.02,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            clip_norm: Some(1.0),
        }
    }
}

impl AdamConfig {
    fn validate(&self) -> Result<()> {
        let bad = |what: String| InverseError::Render {
            reason: format!("invalid Adam config: {what}"),
        };
        if !self.lr_frac.is_finite() || self.lr_frac <= 0.0 {
            return Err(bad(format!(
                "lr_frac must be > 0 and finite, got {}",
                self.lr_frac
            )));
        }
        for (name, b) in [("beta1", self.beta1), ("beta2", self.beta2)] {
            if !b.is_finite() || !(0.0..1.0).contains(&b) {
                return Err(bad(format!("{name} must be in [0, 1), got {b}")));
            }
        }
        if !self.eps.is_finite() || self.eps < 0.0 {
            return Err(bad(format!(
                "eps must be >= 0 and finite, got {}",
                self.eps
            )));
        }
        if let Some(c) = self.clip_norm
            && (!c.is_finite() || c <= 0.0)
        {
            return Err(bad(format!("clip_norm must be > 0 or None, got {c}")));
        }
        Ok(())
    }
}

/// Central-difference gradient of `objective` at `x` over the `params`
/// design space.
///
/// Returns `(g, evaluations)` with `g.len() == params.dim()` and
/// `evaluations == 2 · (number of non-degenerate slots)`. `x` is
/// validated against the bounds first (apply-time policy); probe points
/// are clamped into bounds, so slots at a bound yield the one-sided slope
/// over the effective probe separation. Zero-width slots contribute 0.
///
/// # Errors
/// [`InverseError::WrongDimension`], [`InverseError::NotFinite`],
/// [`InverseError::OutOfBounds`] for a bad `x`;
/// [`InverseError::Render`] if any probe objective fails or returns a
/// non-finite merit, or a difference quotient is non-finite.
pub fn gradient(
    params: &Params,
    objective: &dyn Fn(&[f64]) -> Result<f64>,
    x: &[f64],
) -> Result<(Vec<f64>, usize)> {
    params.validate_values(x)?;
    let dim = params.dim();
    let steps = params.fd_steps();
    let mut g = vec![0.0f64; dim];
    let mut evaluations = 0usize;
    for i in 0..dim {
        let slot = &params.slots()[i];
        if slot.lo == slot.hi {
            continue; // degenerate slot: no room to probe, gradient 0
        }
        let mut up = x.to_vec();
        up[i] = (x[i] + steps[i]).clamp(slot.lo, slot.hi);
        let mut dn = x.to_vec();
        dn[i] = (x[i] - steps[i]).clamp(slot.lo, slot.hi);
        let fu = objective(&up)?;
        let fd = objective(&dn)?;
        evaluations += 2;
        let d = up[i] - dn[i];
        if !(d > 0.0) {
            return Err(InverseError::Render {
                reason: format!(
                    "fd step for slot {} collapses to zero separation (step {})",
                    slot.name, steps[i]
                ),
            });
        }
        let gi = (fu - fd) / d;
        if !gi.is_finite() {
            return Err(InverseError::Render {
                reason: format!("non-finite gradient component for slot {}", slot.name),
            });
        }
        g[i] = gi;
    }
    Ok((g, evaluations))
}

/// Run Adam descent for `epochs` epochs from `params`' current values (or
/// a seeded random draw when `random_init`).
///
/// Entry point for [`crate::solve::Solver::Adam`]; see the [module
/// docs](crate::fd) for the update rule and bounds policy.
pub(crate) fn adam(
    params: &Params,
    objective: &dyn Fn(&[f64]) -> Result<f64>,
    cfg: &AdamConfig,
    epochs: usize,
    random_init: bool,
    rng: &mut StdRng,
) -> Result<Solution> {
    cfg.validate()?;
    let dim = params.dim();
    let spans: Vec<f64> = params.slots().iter().map(|p| p.hi - p.lo).collect();
    let mut x = start(params, random_init, rng);
    clamp_into(params, &mut x);
    let mut val = objective(&x)?;
    if !val.is_finite() {
        return Err(InverseError::Render {
            reason: "objective returned a non-finite value at the start point".into(),
        });
    }
    let mut best = x.clone();
    let mut best_val = val;
    let mut history = vec![val];
    let mut evaluations = 1usize;
    let mut m = vec![0.0f64; dim];
    let mut v = vec![0.0f64; dim];

    for t in 1..=epochs {
        let (g, evals) = gradient(params, objective, &x)?;
        evaluations += evals;

        // Gradient-norm clipping (notebook discipline).
        let norm = g.iter().map(|q| q * q).sum::<f64>().sqrt();
        let scale = match cfg.clip_norm {
            Some(cn) if norm > cn && norm > 0.0 => cn / norm,
            _ => 1.0,
        };

        let bc1 = 1.0 - cfg.beta1.powf(t as f64);
        let bc2 = 1.0 - cfg.beta2.powf(t as f64);
        for i in 0..dim {
            let gi = g[i] * scale;
            m[i] = cfg.beta1 * m[i] + (1.0 - cfg.beta1) * gi;
            v[i] = cfg.beta2 * v[i] + (1.0 - cfg.beta2) * gi * gi;
            let mh = m[i] / bc1;
            let vh = v[i] / bc2;
            let denom = vh.sqrt() + cfg.eps;
            if denom > 0.0 {
                x[i] -= cfg.lr_frac * spans[i] * mh / denom;
            }
        }
        clamp_into(params, &mut x);

        val = objective(&x)?;
        if !val.is_finite() {
            return Err(InverseError::Render {
                reason: "objective returned a non-finite value during descent".into(),
            });
        }
        evaluations += 1;
        history.push(val);
        if val < best_val {
            best_val = val;
            best = x.clone();
        }
    }
    Ok(Solution {
        best,
        value: best_val,
        history,
        evaluations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Param;
    use crate::solve::{Solver, StochasticConfig, solve};

    fn unit_params(n: usize) -> Params {
        Params::from_slots((0..n).map(|i| Param::new(format!("x{i}"), -1.0, 1.0, 0.0)))
    }

    /// Quadratic bowl with a cross term: `f = Σ (x_i − t_i)² + x0·x1`.
    /// Closed-form gradient: `2(x_i − t_i)` plus `(x1, x0)`.
    fn bowl(target: &[f64]) -> impl Fn(&[f64]) -> Result<f64> {
        let t = target.to_vec();
        move |v: &[f64]| -> Result<f64> {
            let mut s = 0.0;
            for (x, ti) in v.iter().zip(&t) {
                s += (x - ti) * (x - ti);
            }
            s += v[0] * v[1];
            Ok(s)
        }
    }

    fn closed_form(v: &[f64], t: &[f64]) -> Vec<f64> {
        let mut g: Vec<f64> = v.iter().zip(t).map(|(x, ti)| 2.0 * (x - ti)).collect();
        g[0] += v[1];
        g[1] += v[0];
        g
    }

    #[test]
    fn fd_gradient_matches_closed_form() {
        let p = unit_params(3);
        let target = [0.3, -0.2, 0.5];
        let obj = bowl(&target);
        let x = [0.1, 0.05, -0.4];
        let (g, evals) = gradient(&p, &obj, &x).unwrap();
        assert_eq!(evals, 6); // 2·dim renders
        let want = closed_form(&x, &target);
        for i in 0..3 {
            assert!(
                (g[i] - want[i]).abs() < 1e-6 * (1.0 + want[i].abs()),
                "g[{i}] = {} vs closed form {}",
                g[i],
                want[i]
            );
        }
    }

    #[test]
    fn fd_gradient_uses_params_steps_and_clamps_probes() {
        // Slot at its upper bound: the +h probe clamps to the bound and
        // the quotient becomes the one-sided slope over h (still exact for
        // the quadratic below since the denominator tracks the clamp).
        let p = Params::from_slots([Param::new("x", 0.0, 1.0, 1.0).with_step(0.25)]);
        let obj = |v: &[f64]| -> Result<f64> { Ok(v[0] * v[0]) };
        let (g, evals) = gradient(&p, &obj, &[1.0]).unwrap();
        assert_eq!(evals, 2);
        // (1² − 0.75²)/0.25 = 1.75 — the documented boundary behaviour.
        assert!((g[0] - 1.75).abs() < 1e-12, "g = {}", g[0]);
    }

    #[test]
    fn zero_width_slot_is_ignored() {
        let p = Params::from_slots([
            Param::new("k", 2.0, 2.0, 2.0),
            Param::new("x", 0.0, 1.0, 0.5),
        ]);
        let obj = |v: &[f64]| -> Result<f64> { Ok(v[0] + v[1] * v[1]) };
        let (g, evals) = gradient(&p, &obj, &[2.0, 0.5]).unwrap();
        assert_eq!(evals, 2); // only the live slot probed
        assert_eq!(g[0], 0.0);
        assert!((g[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn gradient_rejects_bad_inputs() {
        let p = unit_params(2);
        let obj = |_v: &[f64]| -> Result<f64> { Ok(1.0) };
        assert!(matches!(
            gradient(&p, &obj, &[0.0]),
            Err(InverseError::WrongDimension { .. })
        ));
        assert!(matches!(
            gradient(&p, &obj, &[f64::NAN, 0.0]),
            Err(InverseError::NotFinite { .. })
        ));
        assert!(matches!(
            gradient(&p, &obj, &[0.0, 2.0]),
            Err(InverseError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn adam_reduces_a_quadratic_bowl() {
        let p = unit_params(3);
        let target = [0.3, -0.2, 0.5];
        let obj = bowl(&target);
        let cfg = StochasticConfig {
            budget: 200, // epochs for Adam
            ..Default::default()
        };
        let sol = solve(&p, &obj, Solver::Adam(AdamConfig::default()), &cfg, false).unwrap();
        // f(start) = 0.38 (bowl at origin, cross term 0 there).
        assert!(sol.history[0] > 0.3, "start {}", sol.history[0]);
        // True minimizer of Σ(x−t)² + x0·x1: x* = (0.5333, −0.4667, 0.5),
        // f(x*) = −0.12333.
        let star = [8.0 / 15.0, -7.0 / 15.0, 0.5];
        let f_min = (star[0] - target[0]).powi(2)
            + (star[1] - target[1]).powi(2)
            + (star[2] - target[2]).powi(2)
            + star[0] * star[1];
        assert!(
            sol.value < f_min + 0.01,
            "final {} vs f_min {f_min}",
            sol.value
        );
        for (x, s) in sol.best.iter().zip(&star) {
            assert!(
                (x - s).abs() < 0.08,
                "best {:?} vs minimizer {:?}",
                sol.best,
                star
            );
        }
        // Best-so-far trace: monotone non-increasing, strictly better at
        // the end.
        let mut prev = f64::INFINITY;
        let mut improving = false;
        for v in &sol.history {
            if *v < prev {
                improving = true;
            }
            prev = prev.min(*v);
        }
        let mut r = f64::INFINITY;
        let mut best_so_far = Vec::with_capacity(sol.history.len());
        for v in &sol.history {
            r = r.min(*v);
            best_so_far.push(r);
        }
        for w in best_so_far.windows(2) {
            assert!(w[1] <= w[0], "best-so-far trace not monotone: {w:?}");
        }
        assert!(improving, "no epoch beat the running best");
        assert!(best_so_far[best_so_far.len() - 1] < best_so_far[0]);
        assert_eq!(sol.evaluations, 1 + 200 * (2 * 3 + 1));
    }

    #[test]
    fn adam_clips_and_stays_in_bounds() {
        // lr_frac 1.0 would jump the full span per epoch; clipping keeps
        // the trajectory sane and clamping keeps every evaluation inside
        // bounds.
        let seen = std::cell::RefCell::new(Vec::<Vec<f64>>::new());
        let p = Params::from_slots([Param::new("x", 0.0, 1.0, 0.9)]);
        let obj = {
            let seen = &seen;
            move |v: &[f64]| -> Result<f64> {
                seen.borrow_mut().push(v.to_vec());
                Ok((v[0] - 0.2) * (v[0] - 0.2) * 1e6) // steep bowl
            }
        };
        let adam = AdamConfig {
            lr_frac: 0.5,
            clip_norm: Some(1.0),
            ..AdamConfig::default()
        };
        let cfg = StochasticConfig {
            budget: 100,
            ..Default::default()
        };
        let sol = solve(&p, &obj, Solver::Adam(adam), &cfg, false).unwrap();
        for v in seen.borrow().iter() {
            assert!(v[0] >= 0.0 && v[0] <= 1.0, "evaluated out of bounds: {v:?}");
        }
        // Start f = 4.9e5; clipping + clamping still descend three
        // orders of magnitude (the coarse lr_frac 0.5 limit cycle keeps
        // the last bit of the basin).
        assert!(sol.value < 1e3, "steep bowl not descended: {}", sol.value);
        assert!(sol.best[0] > 0.0 && sol.best[0] < 1.0);
    }

    #[test]
    fn adam_deterministic_and_reproducible() {
        let p = unit_params(2);
        let obj = bowl(&[0.2, -0.6]);
        let cfg = StochasticConfig {
            budget: 50,
            seed: Some(42),
            ..Default::default()
        };
        let a = solve(&p, &obj, Solver::Adam(AdamConfig::default()), &cfg, true).unwrap();
        let b = solve(&p, &obj, Solver::Adam(AdamConfig::default()), &cfg, true).unwrap();
        assert_eq!(a.best, b.best);
        assert_eq!(a.history, b.history);
        // And with a fixed start it is fully deterministic (no RNG at all).
        let c = solve(
            &p,
            &obj,
            Solver::Adam(AdamConfig::default()),
            &StochasticConfig {
                budget: 50,
                ..Default::default()
            },
            false,
        )
        .unwrap();
        assert!((c.history[0] - 0.4).abs() < 1e-12); // f(0,0) = 0.04 + 0.36
        let d = solve(
            &p,
            &obj,
            Solver::Adam(AdamConfig::default()),
            &StochasticConfig {
                budget: 50,
                ..Default::default()
            },
            false,
        )
        .unwrap();
        assert_eq!(c.best, d.best);
    }

    #[test]
    fn adam_config_validation_errors() {
        let p = unit_params(1);
        let obj = |_v: &[f64]| -> Result<f64> { Ok(0.0) };
        let cfg = StochasticConfig {
            budget: 4,
            ..Default::default()
        };
        for bad in [
            AdamConfig {
                lr_frac: 0.0,
                ..Default::default()
            },
            AdamConfig {
                lr_frac: f64::NAN,
                ..Default::default()
            },
            AdamConfig {
                beta1: 1.0,
                ..Default::default()
            },
            AdamConfig {
                beta2: -0.1,
                ..Default::default()
            },
            AdamConfig {
                eps: f64::INFINITY,
                ..Default::default()
            },
            AdamConfig {
                clip_norm: Some(0.0),
                ..Default::default()
            },
        ] {
            let err = solve(&p, &obj, Solver::Adam(bad), &cfg, false).unwrap_err();
            assert!(matches!(err, InverseError::Render { .. }), "{err}");
        }
    }

    #[test]
    fn adam_reports_objective_errors() {
        // An objective that fails on a probe propagates as an error, not
        // a panic.
        let p = unit_params(2);
        let obj = |v: &[f64]| -> Result<f64> {
            if v[1] > 0.5 {
                Err(InverseError::Carrier {
                    reason: "boom".into(),
                })
            } else {
                Ok(v[0] * v[0] + v[1] * v[1])
            }
        };
        let mut ps = p;
        ps.set_values(&[0.0, 0.9]).unwrap();
        let cfg = StochasticConfig {
            budget: 10,
            ..Default::default()
        };
        let err = solve(&ps, &obj, Solver::Adam(AdamConfig::default()), &cfg, false).unwrap_err();
        assert!(matches!(err, InverseError::Carrier { .. }));
    }

    #[test]
    fn sign_normalized_step_scales_with_span() {
        // Two slots with different spans: with a consistent gradient sign
        // each epoch moves ~lr_frac of that slot's own span.
        let p = Params::from_slots([
            Param::new("big", -1000.0, 1000.0, 500.0),
            Param::new("small", -0.01, 0.01, 0.005),
        ]);
        let obj = |v: &[f64]| -> Result<f64> { Ok(v[0] * v[0] + v[1] * v[1]) };
        let adam = AdamConfig {
            lr_frac: 0.1,
            beta1: 0.0, // pure per-coordinate sign step
            beta2: 0.0,
            clip_norm: None,
            ..Default::default()
        };
        let (g, _) = gradient(&p, &obj, &p.values()).unwrap();
        assert!(g[0] > 0.0 && g[1] > 0.0);
        let sol = solve(
            &p,
            &obj,
            Solver::Adam(adam),
            &StochasticConfig {
                budget: 1,
                ..Default::default()
            },
            false,
        )
        .unwrap();
        // One epoch = one lr_frac·span step toward the minimum on each
        // axis: big 200 m, small 2 µm (eps slightly shrinks each step).
        assert!((sol.best[0] - 300.0).abs() < 1e-4, "best {}", sol.best[0]);
        assert!((sol.best[1] - 0.003).abs() < 1e-7, "best {}", sol.best[1]);
    }
}
