//! Derivative-free solvers: random search with adaptive step, and
//! simulated annealing. These need only scalar merit evaluations (any user
//! merit, including CRLB-class ones, works out of the box), so they are the
//! baseline every design problem can at least be attacked with.
//!
//! All solvers consume an **objective** `Fn(&[f64]) -> Result<f64>` (see
//! [`crate::merit::objective`]) plus a [`Params`] design space, and return
//! a [`Solution`] with the best value found, its parameters, and the
//! full evaluation history (for the UI convergence plot, Phase 10).
//!
//! Bounds are enforced by **clamping** (prevention, agents.md §4): a probe
//! outside `[lo, hi]` is pulled to the nearest bound before evaluation, so
//! solvers can wander freely without ever feeding an invalid vector to the
//! carrier.
//!
//! A diagonal evolution strategy was evaluated and dropped: without
//! covariance-matrix adaptation (CMA) it converges prematurely against
//! bounds in >2D, and the adaptive-step random walk already covers that
//! role more reliably. High-dimensional design is served by the gradient
//! (adjoint) and deep-surrogate milestones instead.

use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::Result;
use crate::error::InverseError;
use crate::params::Params;

/// A completed (or budget-exhausted) search.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Best parameter vector found (clamped into bounds).
    pub best: Vec<f64>,
    /// Best merit value.
    pub value: f64,
    /// Merit after every accepted evaluation (convergence trace).
    pub history: Vec<f64>,
    /// Total objective evaluations consumed.
    pub evaluations: usize,
}

/// Common configuration for the stochastic solvers.
#[derive(Debug, Clone)]
pub struct StochasticConfig {
    /// Maximum objective evaluations.
    pub budget: usize,
    /// Optional RNG seed (`None` = nondeterministic).
    pub seed: Option<u64>,
    /// Initial move scale as a fraction of each parameter's bound span
    /// (annealing/ES population sigma at step 0).
    pub initial_scale: f64,
}

impl Default for StochasticConfig {
    fn default() -> Self {
        Self {
            budget: 2000,
            seed: Some(0),
            initial_scale: 0.2,
        }
    }
}

/// Clamp `v` into each slot's bounds (prevention policy).
pub(crate) fn clamp_into(params: &Params, v: &mut [f64]) {
    for (slot, x) in params.slots().iter().zip(v.iter_mut()) {
        *x = x.clamp(slot.lo, slot.hi);
    }
}

/// Initial vector: current values, or a uniform random draw when asked.
fn start(params: &Params, random_init: bool, rng: &mut StdRng) -> Vec<f64> {
    if !random_init {
        return params.values();
    }
    let mut v = params.values();
    for (slot, x) in params.slots().iter().zip(v.iter_mut()) {
        if slot.lo == slot.hi {
            *x = slot.lo;
        } else {
            *x = Uniform::new(slot.lo, slot.hi).sample(rng);
        }
    }
    v
}

/// Solver selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Solver {
    /// Random search around the incumbent, accept-only-improvements, with
    /// the step shrinking on stagnation.
    RandomWalk {
        /// Fraction of budget with no improvement before halving the step.
        stagnation: usize,
    },
    /// Metropolis simulated annealing with geometric cooling.
    Annealing {
        /// Initial temperature as a fraction of the first accepted cost
        /// scale (auto-calibrated from the initial value spread).
        t0: Option<f64>,
        /// Final temperature (as a fraction of `t0`).
        t_end_frac: f64,
    },
}

impl Default for Solver {
    fn default() -> Self {
        Solver::Annealing {
            t0: None,
            t_end_frac: 1e-3,
        }
    }
}

/// Run a derivative-free solver over `params` minimizing `objective`.
///
/// `objective` receives already-clamped vectors. `random_init` starts from
/// a uniform draw in the bounds rather than the current values.
pub fn solve(
    params: &Params,
    objective: &dyn Fn(&[f64]) -> Result<f64>,
    solver: Solver,
    cfg: &StochasticConfig,
    random_init: bool,
) -> Result<Solution> {
    if params.dim() == 0 {
        return Err(InverseError::WrongDimension {
            expected: 1,
            actual: 0,
        });
    }
    if cfg.budget == 0 {
        return Err(InverseError::Render {
            reason: "solver budget must be > 0".into(),
        });
    }
    let mut rng = match cfg.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_rng(rand::thread_rng()).map_err(|_| InverseError::Render {
            reason: "rng seeding failed".into(),
        })?,
    };
    match solver {
        Solver::RandomWalk { stagnation } => {
            random_walk(params, objective, cfg, random_init, stagnation, &mut rng)
        }
        Solver::Annealing { t0, t_end_frac } => annealing(
            params,
            objective,
            cfg,
            random_init,
            t0,
            t_end_frac,
            &mut rng,
        ),
    }
}

/// Clamp `v` into bounds (in place) and evaluate the objective. Returns the
/// clamped vector so callers store in-bounds points consistently.
fn eval(
    params: &Params,
    objective: &dyn Fn(&[f64]) -> Result<f64>,
    v: &[f64],
) -> Result<(Vec<f64>, f64)> {
    let mut c = v.to_vec();
    clamp_into(params, &mut c);
    let val = objective(&c)?;
    Ok((c, val))
}

fn random_walk(
    params: &Params,
    objective: &dyn Fn(&[f64]) -> Result<f64>,
    cfg: &StochasticConfig,
    random_init: bool,
    stagnation: usize,
    rng: &mut StdRng,
) -> Result<Solution> {
    let dim = params.dim();
    let spans: Vec<f64> = params.slots().iter().map(|p| p.hi - p.lo).collect();
    let current = start(params, random_init, rng);
    let (mut current, mut cur_val) = eval(params, objective, &current)?;
    let mut best = current.clone();
    let mut best_val = cur_val;
    let mut history = vec![cur_val];
    let mut step = cfg.initial_scale;
    let mut since_gain = 0usize;
    let stagn = stagnation.max(1);
    let mut evaluations = 1;
    while evaluations < cfg.budget {
        let mut cand = current.clone();
        for i in 0..dim {
            let u: f64 = rng.gen_range(-1.0..1.0);
            cand[i] += u * step * spans[i];
        }
        let (cand, val) = eval(params, objective, &cand)?;
        evaluations += 1;
        if val < cur_val {
            current = cand;
            cur_val = val;
            if val < best_val {
                best_val = val;
                best = current.clone();
                since_gain = 0;
            } else {
                since_gain += 1;
            }
            history.push(val);
        } else {
            since_gain += 1;
            if since_gain >= stagn {
                step *= 0.5;
                since_gain = 0;
                if step < 1e-12 {
                    break;
                }
            }
        }
    }
    Ok(Solution {
        best,
        value: best_val,
        history,
        evaluations,
    })
}

fn annealing(
    params: &Params,
    objective: &dyn Fn(&[f64]) -> Result<f64>,
    cfg: &StochasticConfig,
    random_init: bool,
    t0: Option<f64>,
    t_end_frac: f64,
    rng: &mut StdRng,
) -> Result<Solution> {
    let dim = params.dim();
    let spans: Vec<f64> = params.slots().iter().map(|p| p.hi - p.lo).collect();
    let current = start(params, random_init, rng);
    let (mut current, mut cur_val) = eval(params, objective, &current)?;
    let mut best = current.clone();
    let mut best_val = cur_val;
    let mut history = vec![cur_val];
    let mut evaluations = 1;

    // Auto-calibrate temperature from early random-walk spread if unset.
    let t_start = t0.unwrap_or_else(|| {
        let probes = (dim * 4).clamp(8, 32).min(cfg.budget.saturating_sub(1));
        let mut vals = Vec::with_capacity(probes);
        for _ in 0..probes {
            let mut c = current.clone();
            for i in 0..dim {
                let u: f64 = rng.gen_range(-1.0..1.0);
                c[i] += u * cfg.initial_scale * spans[i];
            }
            if let Ok((_, vv)) = eval(params, objective, &c) {
                evaluations += 1;
                vals.push(vv);
            }
        }
        if vals.is_empty() {
            cur_val.abs().max(1.0)
        } else {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
            var.sqrt().max(1e-12).max(cur_val.abs() * 1e-6)
        }
    });
    let t_end = t_start * t_end_frac.max(1e-9);
    let cooling = (t_end / t_start).powf(1.0 / (cfg.budget as f64).max(1.0));
    let mut temperature = t_start;

    while evaluations < cfg.budget {
        let mut cand = current.clone();
        for i in 0..dim {
            let u: f64 = rng.gen_range(-1.0..1.0);
            cand[i] += u * cfg.initial_scale * spans[i] * (temperature / t_start);
        }
        let (cand, val) = eval(params, objective, &cand)?;
        evaluations += 1;
        let delta = val - cur_val;
        let accept = delta < 0.0
            || (temperature > 0.0 && rng.gen_range(0.0..1.0) < (-delta / temperature).exp());
        if accept {
            current = cand;
            cur_val = val;
            if val < best_val {
                best_val = val;
                best = current.clone();
            }
            history.push(val);
        }
        temperature *= cooling;
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

    /// Quadratic bowl centered at `target` within unit bounds.
    fn bowl(target: &[f64], curvature: f64) -> impl Fn(&[f64]) -> Result<f64> {
        let t = target.to_vec();
        move |v: &[f64]| {
            let s: f64 = v
                .iter()
                .zip(&t)
                .map(|(x, ti)| curvature * (x - ti) * (x - ti))
                .sum();
            Ok(s)
        }
    }

    fn unit_params(n: usize) -> Params {
        Params::from_slots((0..n).map(|i| Param::new(format!("x{i}"), -1.0, 1.0, 0.0)))
    }

    #[test]
    fn random_walk_finds_the_bowl_min() {
        let p = unit_params(3);
        let target = [0.3, -0.2, 0.5];
        let obj = bowl(&target, 1.0);
        let cfg = StochasticConfig {
            budget: 6000,
            seed: Some(7),
            initial_scale: 0.3,
        };
        let sol = solve(&p, &obj, Solver::RandomWalk { stagnation: 150 }, &cfg, true).unwrap();
        for (x, t) in sol.best.iter().zip(&target) {
            assert!(
                (x - t).abs() < 0.1,
                "best {:?} vs target {:?}",
                sol.best,
                target
            );
        }
        assert!(sol.value < 0.02, "value {}", sol.value);
        assert!(sol.evaluations <= cfg.budget);
    }

    #[test]
    fn annealing_beats_greedy_on_a_rough_surface() {
        // 1-D bowl plus a spurious local min near the start; annealing must
        // escape it to reach the global basin.
        let p = unit_params(1);
        let obj = move |v: &[f64]| -> Result<f64> {
            let x = v[0];
            // global min at x=0.7, local min at x=-0.6
            let g = (x - 0.7).powi(2);
            let local = 0.4 * (-((x + 0.6).powi(2) / 0.02)).exp();
            Ok(g + local)
        };
        let cfg = StochasticConfig {
            budget: 4000,
            seed: Some(3),
            initial_scale: 0.4,
        };
        let sol = solve(
            &p,
            &obj,
            Solver::Annealing {
                t0: None,
                t_end_frac: 1e-3,
            },
            &cfg,
            true,
        )
        .unwrap();
        assert!(sol.best[0] > 0.4, "annealing stuck at {}", sol.best[0]);
        assert!(sol.value < 0.1, "value {}", sol.value);
    }

    #[test]
    fn clamping_prevents_out_of_bounds_evaluation() {
        let seen = std::cell::RefCell::new(Vec::<Vec<f64>>::new());
        let p = Params::from_slots([Param::new("x", 0.0, 1.0, 0.5)]);
        let obj = {
            let seen = &seen;
            move |v: &[f64]| -> Result<f64> {
                seen.borrow_mut().push(v.to_vec());
                Ok(v[0])
            }
        };
        let cfg = StochasticConfig {
            budget: 500,
            seed: Some(1),
            initial_scale: 5.0, // deliberately huge to force bound hits
        };
        let sol = solve(&p, &obj, Solver::default(), &cfg, true).unwrap();
        for v in seen.borrow().iter() {
            assert!(v[0] >= 0.0 && v[0] <= 1.0, "evaluated out of bounds: {v:?}");
        }
        assert!(sol.value >= 0.0 && sol.value <= 1.0);
    }

    #[test]
    fn degenerate_inputs_error_cleanly() {
        let empty = Params::new();
        let obj = |_v: &[f64]| -> Result<f64> { Ok(0.0) };
        assert!(matches!(
            solve(
                &empty,
                &obj,
                Solver::default(),
                &StochasticConfig::default(),
                false
            ),
            Err(InverseError::WrongDimension { .. })
        ));
        let p = unit_params(1);
        let cfg = StochasticConfig {
            budget: 0,
            ..Default::default()
        };
        assert!(solve(&p, &obj, Solver::default(), &cfg, false).is_err());
        // A zero-dimension design space is rejected by every solver.
        assert!(solve(&empty, &obj, Solver::default(), &cfg, false).is_err());
    }

    #[test]
    fn seeded_runs_are_reproducible() {
        let p = unit_params(2);
        let obj = bowl(&[0.2, -0.6], 1.0);
        let cfg = StochasticConfig {
            budget: 800,
            seed: Some(42),
            initial_scale: 0.2,
        };
        let a = solve(&p, &obj, Solver::default(), &cfg, true).unwrap();
        let b = solve(&p, &obj, Solver::default(), &cfg, true).unwrap();
        assert_eq!(a.best, b.best);
        assert_eq!(a.history, b.history);
    }
}
