//! Gradient check helpers and the FD-on-observation fallback (ADR-9.11 §6).
//!
//! User merits (the 0.2.0 Python frontend, `crate::py` behind feature
//! `py`) may supply an analytic $\partial M/\partial\text{data}$ through
//! [`Merit::obs_gradient`]. Analytic gradients are a correctness trap, so this module ships the
//! check the ADR promises: central differences **on the observation data
//! vector** compared against the analytic gradient, on a small index
//! window ([`check`], [`assert_close`]).
//!
//! ## Cost model (ADR-9.11, "FD on obs")
//!
//! FD on the observation costs $2N$ merit evaluations per gradient (N =
//! `obs.data.len()`) and **zero extra renders** — the renders happen inside
//! the adjoint driver, not here. That makes it a *window-scale* tool: fine
//! for the small scenes the worked examples optimize, impractical for full
//! detector frames. [`FdObsMerit`] wraps any opaque merit so the adjoint
//! path (task 9.9) can still run: its [`Merit::obs_gradient`] is the FD
//! estimate.
//!
//! ## Convention
//!
//! The FD step for entry $k$ is $\varepsilon_k = \texttt{eps}\cdot
//! \max(1, |d_k|)$ when [`GradCheckConfig::relative`] is set, else the
//! constant `eps`. Central differences:
//!
//! $$g_k = \frac{M(d + \varepsilon_k e_k) - M(d - \varepsilon_k e_k)}
//!   {2\varepsilon_k}$$

use crate::Result;
use crate::error::InverseError;
use crate::merit::Merit;
use crate::observe::{Channel, Observation};

/// Tuning for [`fd_obs_gradient`] / [`check`] / [`assert_close`].
#[derive(Debug, Clone, PartialEq)]
pub struct GradCheckConfig {
    /// Base central-difference step on data entries (default 1e-7).
    pub eps: f64,
    /// Scale each entry's step by $\max(1, |d_k|)$ (default `true`).
    pub relative: bool,
    /// Data indices to difference (default: all of `obs.data`). A window
    /// from [`window_indices`] keeps the cost bounded for large frames.
    pub indices: Option<Vec<usize>>,
    /// Relative tolerance for [`check`]/[`assert_close`] (default 1e-4).
    pub rel_tol: f64,
    /// Absolute tolerance floor for [`check`]/[`assert_close`] (default 1e-8).
    pub abs_tol: f64,
}

impl Default for GradCheckConfig {
    fn default() -> Self {
        Self {
            eps: 1e-7,
            relative: true,
            indices: None,
            rel_tol: 1e-4,
            abs_tol: 1e-8,
        }
    }
}

impl GradCheckConfig {
    /// Restrict the check to an index window (see [`window_indices`]).
    pub fn with_window(mut self, indices: Vec<usize>) -> Self {
        self.indices = Some(indices);
        self
    }

    /// The FD step to use for a data entry with value `d`.
    pub fn step_for(&self, d: f64) -> f64 {
        if self.relative {
            self.eps * d.abs().max(1.0)
        } else {
            self.eps
        }
    }
}

/// Data indices of a pixel crop of one `(channel, band, scene)` frame —
/// the "small window" the ADR check helper runs on. `rows`/`cols` are
/// half-open ranges into the frame's row-major grid.
pub fn window_indices(
    obs: &Observation,
    channel: Channel,
    band: usize,
    scene: usize,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) -> Vec<usize> {
    let ci = obs
        .channel_index(channel)
        .expect("window_indices: channel not present in observation");
    let frames = obs.bands.len() * obs.scenes.len();
    let n = obs.frame_len();
    let (nx, _ny) = obs.dims;
    let start = (ci * frames + band * obs.scenes.len() + scene) * n;
    let mut out = Vec::with_capacity(rows.clone().count() * cols.clone().count());
    for iy in rows {
        for ix in cols.clone() {
            out.push(start + iy * nx + ix);
        }
    }
    out
}

/// Central-difference FD of `merit` **on the observation data vector**.
///
/// Entries outside `cfg.indices` (when set) are left at `0.0` — the result
/// is a sparse full-length gradient, which is exactly what an FD-on-obs
/// adjoint seed needs when the merit only reads a window.
///
/// # Errors
/// [`InverseError::GradCheck`] if an index is out of range or a merit
/// evaluation fails; merit evaluation errors surface as their own variant.
pub fn fd_obs_gradient(
    merit: &dyn Merit,
    obs: &Observation,
    cfg: &GradCheckConfig,
) -> Result<Vec<f64>> {
    let idx: Vec<usize> = match &cfg.indices {
        Some(v) => {
            if v.iter().any(|&k| k >= obs.data.len()) {
                return Err(InverseError::GradCheck {
                    reason: "window index out of observation range".into(),
                });
            }
            v.clone()
        }
        None => (0..obs.data.len()).collect(),
    };
    let mut g = vec![0.0f64; obs.data.len()];
    let mut probe = obs.clone();
    for &k in &idx {
        let eps = cfg.step_for(obs.data[k]);
        if eps <= 0.0 || !eps.is_finite() {
            return Err(InverseError::GradCheck {
                reason: format!("invalid FD step {eps} at index {k}"),
            });
        }
        probe.data = obs.data.clone();
        probe.data[k] = obs.data[k] + eps;
        let up = merit.evaluate(&probe)?;
        probe.data[k] = obs.data[k] - eps;
        let dn = merit.evaluate(&probe)?;
        g[k] = (up - dn) / (2.0 * eps);
    }
    Ok(g)
}

/// Outcome of one [`check`] run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientCheck {
    /// Number of data entries compared.
    pub checked: usize,
    /// Largest absolute mismatch.
    pub max_abs_err: f64,
    /// Largest relative mismatch (against the FD magnitude + 1 floor).
    pub max_rel_err: f64,
    /// Index of the worst absolute mismatch.
    pub worst_index: usize,
    /// `true` when every entry satisfied `cfg`'s tolerances.
    pub passed: bool,
}

/// Compare `merit`'s analytic [`Merit::obs_gradient`] against
/// [`fd_obs_gradient`] on `cfg`'s index window. Never panics; reports.
///
/// # Errors
/// [`InverseError::GradCheck`] if the merit has no analytic gradient to
/// check (that is the FD fallback case, nothing to compare), or if a
/// merit evaluation fails.
pub fn check(merit: &dyn Merit, obs: &Observation, cfg: &GradCheckConfig) -> Result<GradientCheck> {
    let analytic = merit
        .obs_gradient(obs)
        .ok_or_else(|| InverseError::GradCheck {
            reason: format!(
                "merit '{}' has no analytic obs_gradient to check",
                merit.name()
            ),
        })?;
    if analytic.len() != obs.data.len() {
        return Err(InverseError::GradCheck {
            reason: format!(
                "analytic gradient length {} != observation length {}",
                analytic.len(),
                obs.data.len()
            ),
        });
    }
    let fd = fd_obs_gradient(merit, obs, cfg)?;
    let idx: &[usize] = match &cfg.indices {
        Some(v) => v,
        None => &[],
    };
    let all: Vec<usize> = if idx.is_empty() {
        (0..obs.data.len()).collect()
    } else {
        idx.to_vec()
    };
    let mut out = GradientCheck {
        checked: all.len(),
        max_abs_err: 0.0,
        max_rel_err: 0.0,
        worst_index: all.first().copied().unwrap_or(0),
        passed: true,
    };
    for &k in &all {
        let err = (analytic[k] - fd[k]).abs();
        let rel = err / (1.0 + fd[k].abs());
        if err > out.max_abs_err {
            out.max_abs_err = err;
            out.worst_index = k;
        }
        out.max_rel_err = out.max_rel_err.max(rel);
        if err > cfg.abs_tol.max(cfg.rel_tol * fd[k].abs()) {
            out.passed = false;
        }
    }
    Ok(out)
}

/// [`check`] plus the ADR's `assert_close`: returns the report when the
/// analytic gradient matches FD within tolerance, and a descriptive
/// [`InverseError::GradCheck`] when it does not.
///
/// # Errors
/// As [`check`], plus [`InverseError::GradCheck`] on tolerance violation.
pub fn assert_close(
    merit: &dyn Merit,
    obs: &Observation,
    cfg: &GradCheckConfig,
) -> Result<GradientCheck> {
    let report = check(merit, obs, cfg)?;
    if !report.passed {
        return Err(InverseError::GradCheck {
            reason: format!(
                "{} vs FD-on-obs: worst abs err {:.3e} at index {} (rel {:.3e}, checked {})",
                merit.name(),
                report.max_abs_err,
                report.worst_index,
                report.max_rel_err,
                report.checked
            ),
        });
    }
    Ok(report)
}

/// Wrap an opaque merit (no analytic `obs_gradient`) so the adjoint path
/// can consume it: [`Merit::obs_gradient`] becomes [`fd_obs_gradient`].
///
/// Cost: $2N$ merit evaluations per gradient call (ADR cost model —
/// window-scale only). Evaluation and naming pass through to the inner
/// merit. FD errors surface as `None` (the documented fallback), so use
/// [`fd_obs_gradient`] directly when errors must propagate.
pub struct FdObsMerit {
    /// The wrapped merit.
    pub inner: Box<dyn Merit>,
    /// FD configuration (index window, step, tolerances unused here).
    pub cfg: GradCheckConfig,
}

impl FdObsMerit {
    /// Wrap `merit` with full-data FD (window it via [`Self::cfg`]).
    pub fn new(merit: impl Merit + 'static) -> Self {
        Self {
            inner: Box::new(merit),
            cfg: GradCheckConfig::default(),
        }
    }
}

impl Merit for FdObsMerit {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn evaluate(&self, obs: &Observation) -> Result<f64> {
        self.inner.evaluate(obs)
    }

    fn obs_gradient(&self, obs: &Observation) -> Option<Vec<f64>> {
        fd_obs_gradient(self.inner.as_ref(), obs, &self.cfg).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merit::{MseToTarget, Selector};

    fn obs4() -> Observation {
        Observation {
            channels: vec![Channel::Intensity],
            bands: vec![500e-9],
            scenes: vec![(0.0, 0.0)],
            dims: (4, 4),
            pitch: (10e-6, 10e-6),
            data: (0..16).map(|k| k as f64 * 0.5 - 2.0).collect(),
            fd_steps: (1e-6, 1e-9),
        }
    }

    #[test]
    fn fd_matches_analytic_mse_on_window() {
        let obs = obs4();
        let target = vec![1.0; 16];
        let m = MseToTarget {
            target: target.clone(),
            selector: Selector::intensity(Some(0), Some(0)),
        };
        let win = window_indices(&obs, Channel::Intensity, 0, 0, 1..3, 1..3);
        let cfg = GradCheckConfig::default().with_window(win.clone());
        let fd = fd_obs_gradient(&m, &obs, &cfg).unwrap();
        let g = m.obs_gradient(&obs).unwrap();
        for &k in &win {
            assert!(
                (fd[k] - g[k]).abs() < 1e-5 * (1.0 + g[k].abs()),
                "k {k}: fd {} vs g {}",
                fd[k],
                g[k]
            );
        }
        // Sparse outside the window.
        assert_eq!(fd[0], 0.0);
    }

    #[test]
    fn assert_close_passes_for_analytic_merits() {
        let obs = obs4();
        let m = MseToTarget {
            target: vec![1.0; 16],
            selector: Selector::intensity(Some(0), Some(0)),
        };
        let rep = assert_close(&m, &obs, &GradCheckConfig::default())
            .expect("analytic MSE matches its own FD");
        assert!(rep.passed);
        assert_eq!(rep.checked, 16);
    }

    #[test]
    fn check_errors_without_analytic_gradient() {
        struct NoGrad;
        impl Merit for NoGrad {
            fn name(&self) -> &str {
                "no_grad"
            }
            fn evaluate(&self, obs: &Observation) -> Result<f64> {
                Ok(obs.data.iter().sum())
            }
        }
        let obs = obs4();
        let err = check(&NoGrad, &obs, &GradCheckConfig::default()).unwrap_err();
        assert!(err.to_string().contains("no analytic obs_gradient"));
    }

    #[test]
    fn fd_obs_merit_wrapper_feeds_adjoint_seed() {
        let obs = obs4();
        let m = FdObsMerit::new(MseToTarget {
            target: vec![1.0; 16],
            selector: Selector::intensity(Some(0), Some(0)),
        });
        let g = m.obs_gradient(&obs).expect("FD gradient");
        let want = MseToTarget {
            target: vec![1.0; 16],
            selector: Selector::intensity(Some(0), Some(0)),
        }
        .obs_gradient(&obs)
        .unwrap();
        for (a, b) in g.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5 * (1.0 + b.abs()));
        }
        assert_eq!(m.evaluate(&obs).unwrap(), want_value(&obs));
    }

    fn want_value(obs: &Observation) -> f64 {
        obs.data.iter().map(|a| (a - 1.0) * (a - 1.0)).sum::<f64>() / obs.data.len() as f64
    }

    #[test]
    fn window_out_of_range_errors() {
        let obs = obs4();
        let cfg = GradCheckConfig::default().with_window(vec![0, 99]);
        assert!(
            fd_obs_gradient(
                &MseToTarget {
                    target: vec![1.0; 16],
                    selector: Selector::intensity(Some(0), Some(0)),
                },
                &obs,
                &cfg
            )
            .is_err()
        );
    }
}
