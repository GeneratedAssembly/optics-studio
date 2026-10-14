//! Inverse-design failures: invalid parameter vectors and render errors.
//!
//! Convention (agents.md §4): validation surfaces as an error at *apply*
//! time, never a panic — mirrors `optics_core::element::OpticsError`.

use thiserror::Error;

/// Errors raised while building a design from a parameter vector or while
/// rendering an observation for a merit evaluation.
#[derive(Debug, Error)]
pub enum InverseError {
    /// A parameter value is NaN or infinite.
    #[error("parameter {name} is not finite ({value})")]
    NotFinite {
        /// Parameter name (or index label when unnamed).
        name: String,
        /// The offending value.
        value: f64,
    },
    /// A parameter value lies outside its declared bounds.
    #[error("parameter {name} out of bounds: {value} not in [{lo}, {hi}]")]
    OutOfBounds {
        /// Parameter name (or index label when unnamed).
        name: String,
        /// The offending value.
        value: f64,
        /// Lower bound.
        lo: f64,
        /// Upper bound.
        hi: f64,
    },
    /// A parameter vector has the wrong length for the carrier.
    #[error("expected {expected} parameters, got {actual}")]
    WrongDimension {
        /// Dimension the carrier requires.
        expected: usize,
        /// Dimension supplied.
        actual: usize,
    },
    /// The carrier could not build a system/field from valid parameters.
    #[error("carrier build failed: {reason}")]
    Carrier {
        /// Human-readable cause from the carrier closure.
        reason: String,
    },
    /// Rendering (propagation/element application) failed.
    #[error("render failed: {reason}")]
    Render {
        /// Human-readable cause from the propagation layer.
        reason: String,
    },
    /// The adjoint gradient pass (task 9.9) could not run: the merit has no
    /// `obs_gradient`, the seed touches a channel without an adjoint path,
    /// or a backward propagation/apply failed.
    #[error("adjoint gradient failed: {reason}")]
    Adjoint {
        /// Human-readable cause.
        reason: String,
    },
    /// A user-defined Python merit (feature `py`, ADR-9.11) raised an
    /// exception or returned a value the frontend could not use. `reason`
    /// carries the Python exception text.
    #[error("python merit failed: {reason}")]
    Python {
        /// Python-side diagnostic (exception type and message).
        reason: String,
    },
    /// The gradient check helper (ADR-9.11 §6) compared a merit's analytic
    /// `obs_gradient` against central differences on the observation and
    /// they disagreed beyond the configured tolerance.
    #[error("obs-gradient check failed: {reason}")]
    GradCheck {
        /// Summary of the worst mismatch (index, values, tolerance).
        reason: String,
    },
}
