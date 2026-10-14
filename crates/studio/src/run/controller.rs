//! The UI-side start/stop controller (spec §3.4 run flow).
//!
//! Pure state machine — no dioxus, no wasm, no timers. It mirrors the
//! chunked session from the UI's point of view: phases, the accumulated
//! convergence trace, progress accounting, and the stop semantics the spec
//! prescribes ("Stop = do not launch the next chunk"). Timestamps are
//! supplied by the caller (`js_sys::Date::now()` on wasm) so this stays
//! host-testable.

use serde::{Deserialize, Serialize};

use super::session::ChunkReport;

/// Run lifecycle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunPhase {
    /// Nothing running; waiting for configuration + Start.
    Idle,
    /// Chunks are being launched.
    Running,
    /// Stop pressed; the in-flight chunk may still land, no next launch.
    Stopping,
    /// Budget exhausted.
    Finished,
    /// Validation, solver, or session error ended the run.
    Failed,
}

/// Accumulated run state shown by the run console.
#[derive(Debug, Clone, PartialEq)]
pub struct RunController {
    phase: RunPhase,
    history: Vec<f64>,
    best: Vec<f64>,
    value: f64,
    total_evaluations: usize,
    chunks_done: usize,
    stop_requested: bool,
    started_ms: f64,
    error: Option<String>,
}

impl Default for RunController {
    fn default() -> Self {
        Self::new()
    }
}

impl RunController {
    /// A fresh idle controller.
    pub fn new() -> Self {
        Self {
            phase: RunPhase::Idle,
            history: Vec::new(),
            best: Vec::new(),
            value: f64::NAN,
            total_evaluations: 0,
            chunks_done: 0,
            stop_requested: false,
            started_ms: 0.0,
            error: None,
        }
    }

    /// Begin a run at wall-clock `now_ms` (epoch millis).
    pub fn begin(&mut self, now_ms: f64) {
        self.phase = RunPhase::Running;
        self.history.clear();
        self.best.clear();
        self.value = f64::NAN;
        self.total_evaluations = 0;
        self.chunks_done = 0;
        self.stop_requested = false;
        self.started_ms = now_ms;
        self.error = None;
    }

    /// Fold one chunk report into the trace. Returns `true` when the UI
    /// should launch the next chunk (Running and no stop requested).
    pub fn absorb(&mut self, report: &ChunkReport) -> bool {
        if !matches!(self.phase, RunPhase::Running | RunPhase::Stopping) {
            return false;
        }
        self.history.extend(report.history.iter().copied());
        self.best = report.best.clone();
        self.value = report.value;
        self.total_evaluations = report.total_evaluations;
        self.chunks_done += 1;
        if report.finished {
            self.phase = RunPhase::Finished;
        } else if self.stop_requested {
            self.phase = RunPhase::Idle;
        }
        self.phase == RunPhase::Running
    }

    /// Stop pressed: the current chunk may land, the next one is not
    /// launched (spec §3.4).
    pub fn request_stop(&mut self) {
        if self.phase == RunPhase::Running {
            self.stop_requested = true;
            self.phase = RunPhase::Stopping;
        }
    }

    /// Record a failure (validation, worker, or solver error text).
    pub fn fail(&mut self, message: impl Into<String>) {
        self.phase = RunPhase::Failed;
        self.error = Some(message.into());
    }

    /// Return to Idle (after a finished or failed run).
    pub fn reset(&mut self) {
        self.phase = RunPhase::Idle;
        self.error = None;
        self.stop_requested = false;
    }

    /// Current phase.
    pub fn phase(&self) -> RunPhase {
        self.phase
    }

    /// Accumulated convergence trace (merit after each accepted eval).
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Incumbent best parameter vector.
    pub fn best(&self) -> &[f64] {
        &self.best
    }

    /// Incumbent best merit value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Objective evaluations consumed so far (client accounting, gap G-2).
    pub fn total_evaluations(&self) -> usize {
        self.total_evaluations
    }

    /// Chunks completed.
    pub fn chunks_done(&self) -> usize {
        self.chunks_done
    }

    /// Elapsed wall time in milliseconds at `now_ms`.
    pub fn elapsed_ms(&self, now_ms: f64) -> f64 {
        (now_ms - self.started_ms).max(0.0)
    }

    /// Error text when the phase is `Failed`.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// "Apply result" (spec §3.4): feed the incumbent best back as the
    /// next run's continuation start.
    pub fn apply_best_to(&self, job: &mut super::job::RunJob) {
        if !self.best.is_empty() {
            job.start_values = Some(self.best.clone());
        }
    }

    /// Reproducibility run card (spec §3.4): solver + config verbatim,
    /// seed, merit name, band/scene lists, final value/evaluations.
    pub fn run_card(&self, job: &super::job::RunJob) -> String {
        use std::fmt::Write as _;
        let mut card = String::new();
        let _ = writeln!(card, "solver: {}", job.solver_label());
        let _ = writeln!(card, "solver config: {:?}", job.solver);
        let _ = writeln!(
            card,
            "budget: {}  chunk: {}  seed: {}",
            job.budget, job.chunk, job.seed
        );
        let _ = writeln!(
            card,
            "initial_scale: {}  random_init: {}",
            job.initial_scale, job.random_init
        );
        let _ = writeln!(card, "merit: {}", job.merit.label());
        let _ = writeln!(card, "bands (m): {:?}", job.bands);
        let _ = writeln!(card, "scenes (m): {:?}", job.scenes);
        let _ = writeln!(card, "evaluations: {}", self.total_evaluations);
        let _ = writeln!(card, "final value: {:?}", self.value);
        card
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(chunk: usize, finished: bool, evals: usize) -> ChunkReport {
        ChunkReport {
            chunk_index: chunk,
            history: vec![1.0 - chunk as f64 * 0.1],
            best: vec![0.25; 4],
            value: 1.0 - chunk as f64 * 0.1,
            chunk_evaluations: evals,
            total_evaluations: (chunk + 1) * evals,
            finished,
        }
    }

    #[test]
    fn stop_waits_for_the_in_flight_chunk() {
        let mut c = RunController::new();
        c.begin(0.0);
        assert!(c.absorb(&report(0, false, 60)));
        c.request_stop();
        assert_eq!(c.phase(), RunPhase::Stopping);
        // The next report lands but no further chunk is launched.
        assert!(!c.absorb(&report(1, false, 60)));
        assert_eq!(c.phase(), RunPhase::Idle);
        assert_eq!(c.chunks_done(), 2);
        assert_eq!(c.total_evaluations(), 120);
        assert_eq!(c.history().len(), 2);
    }

    #[test]
    fn finished_budget_ends_the_run() {
        let mut c = RunController::new();
        c.begin(0.0);
        assert!(c.absorb(&report(0, false, 60)));
        assert!(!c.absorb(&report(1, true, 60)));
        assert_eq!(c.phase(), RunPhase::Finished);
        assert_eq!(c.value(), 0.9);
    }

    #[test]
    fn absorb_outside_a_run_is_ignored() {
        let mut c = RunController::new();
        assert!(!c.absorb(&report(0, false, 60)));
        c.fail("boom");
        assert_eq!(c.phase(), RunPhase::Failed);
        assert_eq!(c.error(), Some("boom"));
        c.reset();
        assert_eq!(c.phase(), RunPhase::Idle);
    }
}
