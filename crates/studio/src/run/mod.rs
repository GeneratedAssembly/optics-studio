//! Host-testable run-control core for the inverse console (task 10.3).
//!
//! - [`job`] — the serializable run job: design, source, observation,
//!   merit, solver, and chunking configuration, plus builders that turn it
//!   into the real `optics-inverse` objects.
//! - [`session`] — the chunked `solve` loop (spec §3.4, gap G-2): one
//!   budget slice per call, continuing from the incumbent best.
//! - [`controller`] — the UI-side start/stop state machine and the
//!   accumulated convergence trace.

pub mod controller;
pub mod job;
pub mod session;

pub use controller::{RunController, RunPhase};
pub use job::RunJob;
pub use session::{ChunkReport, RunSession};
