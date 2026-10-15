//! Deep surrogates — the `burn` harness (task 9.14, plan decision 3).
//!
//! Milestone C of `optics-inverse` 0.3.0, behind the optional `dl`
//! feature: the *headless* forward model becomes a data factory, and a
//! small MLP learns both directions of the design↔observation map.
//!
//! ## Workflow
//!
//! 1. **Generate** — [`dataset::generate`] draws parameter vectors
//!    uniformly from the carrier's bound box, renders them through the
//!    borrowed [`crate::observe::Renderer`], and reduces each observation
//!    with a [`dataset::FeatureSpec`]. Datasets persist as JSON
//!    ([`dataset::Dataset::save_json`]) so the render sweep runs once.
//! 2. **Train** — [`surrogate::train`] fits a [`surrogate::Direction::Forward`]
//!    surrogate (params → features, a cheap stand-in for the renderer) or a
//!    [`surrogate::Direction::Inverse`] one (features → params, a warm
//!    start for the solvers) with mini-batch Adam and best-epoch
//!    snapshots. GPU is allowed here — never in core — via
//!    [`surrogate::train_on`]; the default is the pure-Rust Flex CPU
//!    backend so workspace gates stay light.
//! 3. **Use** — [`Surrogate::predict`] scores new designs;
//!    [`warm_start`] turns an observation into a bounded solver starting
//!    point. Artifacts round-trip through [`Surrogate::save`] /
//!    [`Surrogate::load`].
//!
//! ## Physical scope
//!
//! The surrogate approximates the *renderer*, not new physics: features
//! are band-summed, block-averaged detector intensities, so the network
//! inherits the sampling guarantees of the ASM chain it imitates (see
//! `docs/physics.md` §15). Fidelity is measured against fresh renders —
//! [`warm_start`] output is always clamped into the carrier's bounds and
//! remains an *initial guess* for a solver run, never a delivered design.

pub mod dataset;
pub mod surrogate;

pub use dataset::{DATASET_SCHEMA, Dataset, FeatureSpec, Norm, Sample, generate};
pub use surrogate::{Direction, EpochRecord, Surrogate, TrainConfig, TrainReport, train, train_on};

use crate::Result;
use crate::observe::Observation;
use crate::observe::Renderer;

/// Predict a bounded starting point for a solver from an observation.
///
/// Runs `inverse` (features → params) on the [`FeatureSpec`] reduction of
/// `obs`, then clamps the result into the carrier's declared bounds — the
/// prevention-over-detection policy of agents.md §4: a surrogate can
/// extrapolate slightly outside the box, and the clamp is the cheap
/// closed-form correction before the solver ever sees the value.
///
/// # Errors
/// Propagates [`FeatureSpec`] geometry errors and surrogate input-length
/// mismatches ([`crate::InverseError::Dl`]).
pub fn warm_start(
    renderer: &Renderer<'_>,
    spec: &dataset::FeatureSpec,
    obs: &Observation,
    inverse: &Surrogate,
) -> Result<Vec<f64>> {
    let features = spec.extract(obs)?;
    let predicted = inverse.predict(&features)?;
    let slots = renderer.carrier.params().slots();
    Ok(predicted
        .iter()
        .zip(slots)
        .map(|(v, p)| v.clamp(p.lo, p.hi))
        .collect())
}
