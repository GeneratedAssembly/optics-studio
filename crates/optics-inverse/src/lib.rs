//! # optics-inverse — inverse design for `optics-core` systems
//!
//! Inverse design flips the forward model: instead of choosing a surface
//! and simulating the image it makes, a **design** is parameterized by a
//! bounded vector, a **renderer** produces an `Observation` (intensity
//! stacks over wavelength/field, optionally with derivative channels), a
//! **merit function** scores that observation, and a **solver** searches
//! the parameter space. See `tasks/plan.md` decision 7 and
//! `.kilo/plans/1792240000000-phase9-plan.md` for the full design.
//!
//! The crate is organized as three milestones:
//!
//! - **0.1.0 (framework + solvers)** — parameters, carriers (analytic
//!   element parameters and sampled phase grids), the observation model,
//!   simple built-in merits (MSE-to-target, encircled energy, Strehl,
//!   MTF, spot RMS), derivative-free solvers (random walk, annealing,
//!   evolution strategies), finite-difference Adam, Gerchberg–Saxton
//!   phase retrieval, and adjoint gradients (the adjoint of an ASM
//!   propagation is propagation by −z, so gradients need no autodiff).
//! - **0.2.0 (custom merits)** — a frontend for user-defined merit
//!   functions: `PyMerit` (feature `py`) embeds CPython via pyo3 and
//!   presents `Observation` as a read-only numpy array, per ADR-9.11;
//!   [`gradcheck`] ships the FD-on-observation check helper and the
//!   [`gradcheck::FdObsMerit`] adjoint fallback. Estimation-theoretic
//!   merits such as Cramer–Rao trace minimization are *user* merits, not
//!   built-ins — see `examples/crlb_doe.rs` for the worked example.
//! - **0.3.0 (deep surrogates)** — dataset generation and network
//!   surrogates on `burn` (feature-gated; GPU never in core).
//!
//! ## Layout conventions
//!
//! `Observation` stores flat row-major `f64` stacks with documented
//! strides so a merit (Rust or Python) can view them as arrays without
//! copies. Carrier parameters are always a flat `Vec<f64>`; optimizers
//! never see design structure. All validation surfaces as
//! `InverseError` at apply time — never a panic.

pub mod adjoint;
pub mod carrier;
pub mod error;
pub mod fd;
pub mod gradcheck;
pub mod gs;
pub mod merit;
pub mod observe;
pub mod params;
pub mod solve;

#[cfg(feature = "py")]
pub mod py;

pub use adjoint::{PlaneGradient, asm_adjoint, grid_gradient, push_back, transmission};
pub use carrier::{AnalyticBuilder, Carrier, Design, GridGeom, GridSpec, Step};
pub use error::InverseError;
pub use fd::{AdamConfig, gradient};
pub use gradcheck::{FdObsMerit, GradCheckConfig, GradientCheck, fd_obs_gradient, window_indices};
pub use gs::{GsConfig, GsInit, GsProblem, GsRecord, GsSolution, GsVariant};
pub use merit::{
    EncircledEnergyMerit, Merit, MseToTarget, MtfMerit, Selector, SpotRmsMerit, StrehlMerit,
    objective,
};
pub use observe::{Channel, Derivatives, Observation, Renderer};
pub use params::{Param, Params};
pub use solve::{Solution, Solver, StochasticConfig};

#[cfg(feature = "py")]
pub use py::PyMerit;

/// Crate result alias.
pub type Result<T> = std::result::Result<T, InverseError>;
