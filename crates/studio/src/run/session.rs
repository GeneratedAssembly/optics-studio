//! The chunked solver session (spec §3.4, gap G-2 workaround).
//!
//! `optics_inverse::solve` is synchronous with no progress callback, so
//! the UI runs it in **chunks**: each [`RunSession::run_chunk`] consumes at
//! most `job.chunk` solver units and returns a [`ChunkReport`]. The session
//! keeps the incumbent `Solution::best` in its `Params` between chunks
//! (`Params::set_values`, exactly the continuation the spec prescribes),
//! so Stop between chunks is "do not launch the next chunk" — no
//! cancellation plumbing needed inside the solver.
//!
//! Determinism: chunk *k* runs with `seed = job.seed + k` (wrapping). One
//! seed per run keeps the whole chunked run reproducible while giving each
//! chunk a fresh random stream (reusing one seed would replay the same
//! stream every chunk).

use serde::{Deserialize, Serialize};

use optics_inverse::{Merit, Params, Solution, StochasticConfig, objective, solve::solve};

use super::job::RunJob;

/// What one chunk of a run reports back to the UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkReport {
    /// Zero-based index of the chunk that produced this report.
    pub chunk_index: usize,
    /// Merit-after-each-accepted-evaluation trace for this chunk.
    pub history: Vec<f64>,
    /// Best parameter vector so far (clamped into bounds).
    pub best: Vec<f64>,
    /// Best merit value so far.
    pub value: f64,
    /// Objective evaluations consumed by this chunk.
    pub chunk_evaluations: usize,
    /// Objective evaluations consumed by the whole run so far.
    pub total_evaluations: usize,
    /// True when the budget is exhausted after this chunk.
    pub finished: bool,
}

/// One live chunked run: owns the carrier, the merit, the incumbent
/// parameters, and the budget accounting. Runs on the worker thread (wasm)
/// or inline on the host (tests, fallback transport).
pub struct RunSession {
    job: RunJob,
    carrier: optics_inverse::Carrier,
    merit: Box<dyn Merit>,
    params: Params,
    chunk_index: usize,
    units_done: usize,
    total_evaluations: usize,
    best: Vec<f64>,
    value: f64,
    finished: bool,
}

impl RunSession {
    /// Validate the job and prepare the run state.
    pub fn start(job: RunJob) -> Result<Self, String> {
        job.validate()?;
        let mut carrier = job.build_carrier()?;
        let merit = job.build_merit();
        let mut params = carrier.params().clone();
        if let Some(v) = &job.start_values {
            params
                .set_values(v)
                .map_err(|e| format!("start_values rejected: {e}"))?;
            carrier
                .params_mut()
                .set_values(v)
                .map_err(|e| format!("start_values rejected: {e}"))?;
        }
        let best = params.values();
        Ok(Self {
            job,
            carrier,
            merit,
            params,
            chunk_index: 0,
            units_done: 0,
            total_evaluations: 0,
            best,
            value: f64::NAN,
            finished: false,
        })
    }

    /// Run at most `job.chunk` solver units. Errors from the solver are
    /// reported as `Err(String)` (typed `InverseError` display text) — the
    /// UI renders them per spec §3.5 (no string-only error *source*; the
    /// text comes from the typed enum).
    pub fn run_chunk(&mut self) -> Result<ChunkReport, String> {
        if self.finished {
            return Err("session already finished".into());
        }
        let units = self.job.chunk.min(self.job.budget - self.units_done);
        let random_init = self.chunk_index == 0 && self.job.random_init;
        // The renderer borrows the carrier; scope it so the carrier can be
        // updated with the new incumbent after the solve.
        let solution: Solution = {
            let renderer = self.job.build_renderer(&self.carrier);
            let obj = objective(&renderer, self.merit.as_ref());
            let cfg = StochasticConfig {
                budget: units,
                seed: Some(self.job.seed.wrapping_add(self.chunk_index as u64)),
                initial_scale: self.job.initial_scale,
            };
            solve(
                &self.params,
                &obj,
                self.job.solver_enum(),
                &cfg,
                random_init,
            )
            .map_err(|e| e.to_string())?
        };

        let report = ChunkReport {
            chunk_index: self.chunk_index,
            history: solution.history.clone(),
            best: solution.best.clone(),
            value: solution.value,
            chunk_evaluations: solution.evaluations,
            total_evaluations: self.total_evaluations + solution.evaluations,
            finished: self.units_done + units >= self.job.budget,
        };

        // Continue-from-best: the next chunk starts at this chunk's best.
        self.params
            .set_values(&solution.best)
            .map_err(|e| e.to_string())?;
        self.carrier
            .params_mut()
            .set_values(&solution.best)
            .map_err(|e| e.to_string())?;
        self.best = solution.best;
        self.value = solution.value;
        self.units_done += units;
        self.total_evaluations = report.total_evaluations;
        self.chunk_index += 1;
        self.finished = report.finished;
        Ok(report)
    }

    /// True once the budget is exhausted.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Incumbent best parameter vector.
    pub fn best(&self) -> &[f64] {
        &self.best
    }

    /// Incumbent best merit value (`NaN` before the first chunk).
    pub fn value(&self) -> f64 {
        self.value
    }

    /// The design space as it stands between chunks (tests, run card).
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The job this session runs.
    pub fn job(&self) -> &RunJob {
        &self.job
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_job() -> RunJob {
        let mut job = RunJob::demo_ee_focus();
        job.design = super::super::job::DesignSpec::GridPhaseRelay {
            dims: (8, 8),
            pitch: (20e-6, 20e-6),
            aperture: super::super::job::ApertureSpec::Circle { radius: 150e-6 },
            center: (0.0, 0.0),
            lo: -std::f64::consts::PI,
            hi: std::f64::consts::PI,
            init: 0.0,
            z_after: 2e-3,
            medium: 1.0,
        };
        job.grid = super::super::job::GridUi {
            nx: 16,
            ny: 16,
            dx: 20e-6,
            dy: 20e-6,
        };
        job.budget = 120;
        job.chunk = 60;
        job.seed = 7;
        job
    }

    #[test]
    fn chunks_continue_from_incumbent_best() {
        let mut s = RunSession::start(tiny_job()).expect("valid job starts");
        let r0 = s.run_chunk().expect("chunk 0");
        assert_eq!(r0.chunk_index, 0);
        assert!(!r0.finished);
        assert!(!r0.history.is_empty());
        // The session's design space now sits at chunk 0's best.
        assert_eq!(s.params().values(), r0.best);

        let r1 = s.run_chunk().expect("chunk 1");
        assert_eq!(r1.chunk_index, 1);
        assert!(r1.finished);
        // Chunk 1 started from chunk 0's best (random_init is chunk-0 only).
        assert_eq!(r1.total_evaluations, 120);
        assert_eq!(s.best(), &r1.best[..]);
        assert_eq!(s.value(), r1.value);
        assert!(s.is_finished());
        assert!(s.run_chunk().is_err());
    }

    #[test]
    fn chunked_run_is_deterministic_per_seed() {
        let run = || {
            let mut s = RunSession::start(tiny_job()).unwrap();
            let mut hist = Vec::new();
            while !s.is_finished() {
                hist.extend(s.run_chunk().unwrap().history);
            }
            hist
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn start_values_seed_the_first_chunk() {
        let mut job = tiny_job();
        let params = job.build_params().unwrap();
        let mut v = params.values();
        for x in v.iter_mut() {
            *x = 0.5;
        }
        job.start_values = Some(v.clone());
        let s = RunSession::start(job).expect("continuation start accepted");
        assert_eq!(s.best(), &v[..]);
    }

    #[test]
    fn invalid_jobs_are_rejected_before_any_render() {
        let mut job = tiny_job();
        job.chunk = 10;
        assert!(RunSession::start(job.clone()).is_err());
        job.chunk = 60;
        job.bands = vec![];
        assert!(RunSession::start(job.clone()).is_err());
        job.bands = vec![500e-9];
        job.start_values = Some(vec![0.0; 3]);
        assert!(RunSession::start(job).is_err());
    }
}
