//! Custom-merit Python frontend — feature `py` (ADR-9.11, task 9.12).
//!
//! [`PyMerit`] lets a **user** define merit functions in Python and plugs
//! them into the 0.1.0 solver stack through the existing [`Merit`] trait —
//! nothing else in the crate changes (decision 7's "plugs in without
//! redesign" constraint). The Rust binary embeds the interpreter (ADR
//! option A1, `auto-initialize`); merits are plain Python functions:
//!
//! ```python
//! # merit(array, meta) -> float   (minimized)
//! # merit_grad(array, meta) -> float64 array, same shape  (optional)
//! def merit(array, meta):
//!     I = array[meta["channels"].index("intensity")]
//!     return float((I * I).sum())
//! ```
//!
//! ## The bridge
//!
//! [`PyMerit::evaluate`] presents the [`Observation`] as a **read-only
//! float64 ndarray** shaped `(C, B, S, Ny, Nx)` — the documented
//! `[channel][band][scene][row][col]` layout viewed, not transposed — plus
//! a metadata dict (`channels`, `bands`, `scenes`, `dims`, `pitch`,
//! `fd_steps`, `frame_len`). The buffer is moved into the Python array in
//! one contiguous memcpy per call (no per-element conversion); the merit
//! must not retain the array past the call.
//!
//! ## Gradients (layered, ADR §6)
//!
//! - `PyMerit::new(call)` — no analytic gradient: `obs_gradient()` returns
//!   `None`, the documented FD fallback (see [`crate::gradcheck::FdObsMerit`]
//!   to make it adjoint-consumable).
//! - `PyMerit::with_gradient(call, grad)` — user-supplied
//!   $\partial M/\partial\text{data}$; validate it with
//!   [`crate::gradcheck::assert_close`] (the shipped check helper).
//!
//! ## GIL and threading
//!
//! Every call takes the GIL for its duration, so merit evaluations
//! serialize. Harmless while the solvers are single-threaded; any future
//! parallel-render path must keep merit calls serialized or move to
//! sub-interpreters (ADR-9.11 consequence, documented here).
//!
//! ## Errors
//!
//! Python exceptions and unusable return values surface as
//! [`InverseError::Python`] with the exception text; they never panic
//! across the FFI boundary.

use std::ffi::CString;
use std::path::Path;

use numpy::ndarray::ArrayD;
use numpy::{IxDyn, PyArray, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::Result;
use crate::error::InverseError;
use crate::merit::Merit;
use crate::observe::{Channel, Observation};

/// Python-side name of an observation channel (metadata `channels`).
pub fn channel_name(ch: Channel) -> &'static str {
    match ch {
        Channel::Intensity => "intensity",
        Channel::DxDyX => "dx",
        Channel::DxDyY => "dy",
        Channel::Dlambda => "dlambda",
    }
}

fn python_err(e: PyErr) -> InverseError {
    InverseError::Python {
        reason: e.to_string(),
    }
}

/// A user merit defined in Python (ADR-9.11 §8).
///
/// Built from source strings ([`PyMerit::from_source`],
/// [`PyMerit::from_file`]) or from already-live callables
/// ([`PyMerit::new`] + [`PyMerit::with_gradient`]). Implements [`Merit`]:
/// every 0.1.0 solver consumes it unchanged, and the 9.9 adjoint consumes
/// the optional gradient callable.
pub struct PyMerit {
    name: String,
    call: Py<PyAny>,
    grad: Option<Py<PyAny>>,
}

impl std::fmt::Debug for PyMerit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyMerit")
            .field("name", &self.name)
            .field("has_gradient", &self.grad.is_some())
            .finish()
    }
}

impl PyMerit {
    /// Wrap a live callable `merit(array, meta) -> float`.
    ///
    /// # Errors
    /// [`InverseError::Python`] if `call` is not callable.
    pub fn new(name: impl Into<String>, call: Bound<'_, PyAny>) -> Result<Self> {
        if !call.is_callable() {
            return Err(InverseError::Python {
                reason: "merit callable is not callable".into(),
            });
        }
        Ok(Self {
            name: name.into(),
            call: call.unbind(),
            grad: None,
        })
    }

    /// Attach the analytic `merit_grad(array, meta) -> float64 array`
    /// callable (flat or `(C,B,S,Ny,Nx)` shaped; float64).
    ///
    /// # Errors
    /// [`InverseError::Python`] if `grad` is not callable.
    pub fn with_gradient(mut self, grad: Bound<'_, PyAny>) -> Result<Self> {
        if !grad.is_callable() {
            return Err(InverseError::Python {
                reason: "gradient callable is not callable".into(),
            });
        }
        self.grad = Some(grad.unbind());
        Ok(self)
    }

    /// Execute `source` in a fresh namespace and bind `merit` (required)
    /// plus `merit_grad` (optional).
    ///
    /// # Errors
    /// [`InverseError::Python`] if the source fails to compile/execute,
    /// `merit` is missing or not callable.
    pub fn from_source(name: impl Into<String>, source: &str) -> Result<Self> {
        Self::from_source_named(name, source, "merit", Some("merit_grad"))
    }

    /// [`PyMerit::from_source`] with explicit function names.
    ///
    /// # Errors
    /// As [`PyMerit::from_source`].
    pub fn from_source_named(
        name: impl Into<String>,
        source: &str,
        eval_fn: &str,
        grad_fn: Option<&str>,
    ) -> Result<Self> {
        let name = name.into();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            let code = CString::new(source).map_err(|_| InverseError::Python {
                reason: "merit source contains a NUL byte".into(),
            })?;
            py.run(&code, Some(&globals), Some(&globals))
                .map_err(python_err)?;
            let call = require_callable(py, &globals, eval_fn)?;
            let mut merit = PyMerit::new(name, call)?;
            if let Some(gname) = grad_fn
                && let Some(g) = globals.get_item(gname).map_err(python_err)?
            {
                merit = merit.with_gradient(g)?;
            }
            Ok(merit)
        })
    }

    /// Read a `.py` file and [`PyMerit::from_source`] it.
    ///
    /// # Errors
    /// As [`PyMerit::from_source`], plus I/O failures as
    /// [`InverseError::Python`].
    pub fn from_file(name: impl Into<String>, path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path).map_err(|e| InverseError::Python {
            reason: format!("cannot read merit file {}: {e}", path.display()),
        })?;
        Self::from_source(name, &source)
    }

    /// The merit name (for reports).
    pub fn merit_name(&self) -> &str {
        &self.name
    }

    /// `true` when an analytic gradient callable is attached.
    pub fn has_gradient(&self) -> bool {
        self.grad.is_some()
    }

    /// Evaluate, propagating Python errors (unlike [`Merit::evaluate`]
    /// through the trait, which also reports them).
    ///
    /// # Errors
    /// [`InverseError::Python`] on exception, non-numeric return, or a
    /// non-finite value.
    pub fn try_evaluate(&self, obs: &Observation) -> Result<f64> {
        Python::attach(|py| {
            let (arr, meta) = push_observation(py, obs)?;
            let call = self.call.bind(py);
            let value: f64 = call
                .call1((arr, meta))
                .map_err(python_err)?
                .extract()
                .map_err(|e| InverseError::Python {
                    reason: format!("merit must return a real scalar: {e}"),
                })?;
            if !value.is_finite() {
                return Err(InverseError::Python {
                    reason: format!("merit returned non-finite value {value}"),
                });
            }
            Ok(value)
        })
    }

    /// Call the attached gradient callable, flattening to the observation
    /// data order. `Ok(None)` when no gradient callable is attached.
    ///
    /// # Errors
    /// [`InverseError::Python`] on exception, wrong dtype, or wrong length.
    pub fn try_obs_gradient(&self, obs: &Observation) -> Result<Option<Vec<f64>>> {
        let Some(grad) = &self.grad else {
            return Ok(None);
        };
        Python::attach(|py| {
            let (arr, meta) = push_observation(py, obs)?;
            let result = grad.bind(py).call1((arr, meta)).map_err(python_err)?;
            let view = result
                .extract::<PyReadonlyArrayDyn<'_, f64>>()
                .map_err(|e| InverseError::Python {
                    reason: format!("merit_grad must return a float64 array: {e}"),
                })?;
            let flat = view.as_array();
            if flat.len() != obs.data.len() {
                return Err(InverseError::Python {
                    reason: format!(
                        "merit_grad returned {} entries, expected {}",
                        flat.len(),
                        obs.data.len()
                    ),
                });
            }
            Ok(Some(flat.iter().copied().collect()))
        })
    }
}

fn require_callable<'py>(
    _py: Python<'py>,
    globals: &Bound<'py, PyDict>,
    name: &str,
) -> Result<Bound<'py, PyAny>> {
    let obj = globals
        .get_item(name)
        .map_err(python_err)?
        .ok_or_else(|| InverseError::Python {
            reason: format!("merit source defines no `{name}` function"),
        })?;
    if !obj.is_callable() {
        return Err(InverseError::Python {
            reason: format!("`{name}` is not callable"),
        });
    }
    Ok(obj)
}

/// The `(array, meta)` pair passed to both user callables.
type ObsPair<'py> = (Bound<'py, PyArray<f64, IxDyn>>, Bound<'py, PyDict>);

/// Build the `(array, meta)` pair passed to the user callables.
fn push_observation<'py>(py: Python<'py>, obs: &Observation) -> Result<ObsPair<'py>> {
    let (c, b, s) = (obs.channels.len(), obs.bands.len(), obs.scenes.len());
    let (ny, nx) = (obs.dims.1, obs.dims.0);
    let array =
        ArrayD::from_shape_vec(IxDyn(&[c, b, s, ny, nx]), obs.data.clone()).map_err(|e| {
            InverseError::Python {
                reason: format!("observation buffer/shape mismatch: {e}"),
            }
        })?;
    let arr = PyArray::from_owned_array(py, array);
    // Read-only contract: numpy users get a WRITEABLE=False flag, matching
    // the ADR's "read-only ndarray view".
    arr.getattr("flags")
        .and_then(|f| f.setattr("writeable", false))
        .map_err(python_err)?;

    let meta = PyDict::new(py);
    meta.set_item(
        "channels",
        obs.channels
            .iter()
            .map(|ch| channel_name(*ch))
            .collect::<Vec<_>>(),
    )
    .map_err(python_err)?;
    meta.set_item("bands", obs.bands.clone())
        .map_err(python_err)?;
    meta.set_item(
        "scenes",
        obs.scenes.iter().map(|&(x, y)| (x, y)).collect::<Vec<_>>(),
    )
    .map_err(python_err)?;
    meta.set_item("dims", (obs.dims.0, obs.dims.1))
        .map_err(python_err)?;
    meta.set_item("pitch", (obs.pitch.0, obs.pitch.1))
        .map_err(python_err)?;
    meta.set_item("fd_steps", (obs.fd_steps.0, obs.fd_steps.1))
        .map_err(python_err)?;
    meta.set_item("frame_len", obs.frame_len())
        .map_err(python_err)?;
    Ok((arr, meta))
}

impl Merit for PyMerit {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, obs: &Observation) -> Result<f64> {
        self.try_evaluate(obs)
    }

    /// The attached gradient callable, if any. Python errors are folded
    /// into `None` here (the documented FD fallback); use
    /// [`PyMerit::try_obs_gradient`] to propagate them.
    fn obs_gradient(&self, obs: &Observation) -> Option<Vec<f64>> {
        self.try_obs_gradient(obs).ok().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs3() -> Observation {
        Observation {
            channels: vec![Channel::Intensity, Channel::DxDyX],
            bands: vec![500e-9],
            scenes: vec![(0.0, 0.0)],
            dims: (2, 2),
            pitch: (10e-6, 10e-6),
            data: vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.25, 0.125, 0.0625],
            fd_steps: (1e-6, 1e-9),
        }
    }

    #[test]
    fn evaluate_uses_intensity_frames() {
        let m = PyMerit::from_source(
            "sumsq",
            r"
def merit(array, meta):
    ci = meta['channels'].index('intensity')
    return float((array[ci] ** 2).sum())
",
        )
        .unwrap();
        let obs = obs3();
        // Intensity channel = first 4 entries: 1+4+9+16 = 30.
        assert_eq!(m.evaluate(&obs).unwrap(), 30.0);
    }

    #[test]
    fn metadata_matches_observation() {
        let m = PyMerit::from_source(
            "meta_check",
            r"
def merit(array, meta):
    assert meta['channels'] == ['intensity', 'dx']
    assert meta['bands'] == [500e-9]
    assert meta['scenes'] == [(0.0, 0.0)]
    assert meta['dims'] == (2, 2)
    assert meta['pitch'] == (10e-6, 10e-6)
    assert meta['fd_steps'] == (1e-6, 1e-9)
    assert meta['frame_len'] == 4
    assert array.shape == (2, 1, 1, 2, 2)
    return 0.0
",
        )
        .unwrap();
        assert_eq!(m.evaluate(&obs3()).unwrap(), 0.0);
    }

    #[test]
    fn gradient_callable_flattens_any_shape() {
        let m = PyMerit::from_source(
            "lin",
            r"
def merit(array, meta):
    ci = meta['channels'].index('intensity')
    return float(array[ci].sum())

def merit_grad(array, meta):
    import numpy as np
    g = np.zeros(array.shape)
    g[meta['channels'].index('intensity')] = 1.0
    return g
",
        )
        .unwrap();
        let g = m.try_obs_gradient(&obs3()).unwrap().unwrap();
        // Only the intensity channel (first 4 entries) has gradient 1.
        assert_eq!(g, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn missing_gradient_is_none() {
        let m = PyMerit::from_source("no_grad", "def merit(array, meta): return 1.0").unwrap();
        assert!(!m.has_gradient());
        assert!(m.obs_gradient(&obs3()).is_none());
        assert!(m.try_obs_gradient(&obs3()).unwrap().is_none());
    }

    #[test]
    fn python_exception_surfaces_as_inverse_error() {
        let m = PyMerit::from_source("boom", "def merit(array, meta): raise ValueError('bad')")
            .unwrap();
        let err = m.evaluate(&obs3()).unwrap_err();
        assert!(matches!(err, InverseError::Python { .. }));
        assert!(err.to_string().contains("bad"));
    }

    #[test]
    fn non_numeric_return_is_wrapped() {
        let m = PyMerit::from_source("str_ret", "def merit(array, meta): return 'nope'").unwrap();
        assert!(m.evaluate(&obs3()).is_err());
    }

    #[test]
    fn source_without_merit_errors() {
        let err = PyMerit::from_source("empty", "x = 1").unwrap_err();
        assert!(err.to_string().contains("no `merit` function"));
    }

    #[test]
    fn array_is_read_only() {
        let m = PyMerit::from_source(
            "ro",
            r"
def merit(array, meta):
    return 1.0 if array.flags.writeable is False else 0.0
",
        )
        .unwrap();
        assert_eq!(m.evaluate(&obs3()).unwrap(), 1.0);
    }
}
