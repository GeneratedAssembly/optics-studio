//! Integration tests for the custom-merit Python frontend (feature `py`,
//! ADR-9.11 task 9.12). Run with `cargo test -p optics-inverse --features py`.

use optics_inverse::InverseError;
use optics_inverse::gradcheck::{self, GradCheckConfig};
use optics_inverse::merit::{Merit, MseToTarget, Selector};
use optics_inverse::observe::{Channel, Observation};
use optics_inverse::py::PyMerit;

fn obs() -> Observation {
    Observation {
        channels: vec![Channel::Intensity],
        bands: vec![500e-9],
        scenes: vec![(0.0, 0.0)],
        dims: (4, 4),
        pitch: (10e-6, 10e-6),
        data: (0..16).map(|k| (k as f64) * 0.25).collect(),
        fd_steps: (1e-6, 1e-9),
    }
}

#[test]
fn py_mse_matches_rust_builtin() {
    let target = vec![0.5; 16];
    let m = PyMerit::from_source(
        "py_mse",
        r"
import numpy as np

TARGET = np.full(16, 0.5)

def merit(array, meta):
    ci = meta['channels'].index('intensity')
    frame = array[ci].reshape(-1)
    return float(((frame - TARGET) ** 2).mean())

def merit_grad(array, meta):
    ci = meta['channels'].index('intensity')
    g = np.zeros(array.shape)
    n = array[ci].size
    g[ci] = ((array[ci].reshape(-1) - TARGET) * (2.0 / n)).reshape(array[ci].shape)
    return g
",
    )
    .unwrap();
    let rust = MseToTarget {
        target: target.clone(),
        selector: Selector::intensity(Some(0), Some(0)),
    };
    let o = obs();
    let a = m.evaluate(&o).unwrap();
    let b = rust.evaluate(&o).unwrap();
    assert!((a - b).abs() < 1e-12, "py {a} vs rust {b}");

    // The shipped check helper validates the user gradient against FD.
    let rep = gradcheck::assert_close(&m, &o, &GradCheckConfig::default()).unwrap();
    assert!(rep.passed, "gradcheck {rep:?}");
}

#[test]
fn missing_gradient_falls_back_to_fd_wrapper() {
    let m = PyMerit::from_source(
        "opaque",
        "def merit(array, meta): return float(array.sum())",
    )
    .unwrap();
    let o = obs();
    assert!(m.obs_gradient(&o).is_none());
    let fd = optics_inverse::gradcheck::FdObsMerit::new(m);
    let g = fd.obs_gradient(&o).expect("FD gradient");
    // FD on a sum: eps*max(1,|d|) steps, cancellation-limited precision.
    assert!(g.iter().all(|&v| (v - 1.0).abs() < 1e-6));
}

#[test]
fn exception_and_bad_length_surface_as_errors() {
    let m = PyMerit::from_source_named(
        "bad_grad_len",
        r"
import numpy as np

def merit(array, meta):
    return float(array.sum())

def merit_grad(array, meta):
    return np.zeros(3)
",
        "merit",
        Some("merit_grad"),
    )
    .unwrap();
    let o = obs();
    let err = m.try_obs_gradient(&o).unwrap_err();
    assert!(matches!(err, InverseError::Python { .. }));
    assert!(err.to_string().contains("expected 16"));

    let m2 = PyMerit::from_source("raise", "def merit(array, meta): return array.nope").unwrap();
    assert!(m2.evaluate(&o).is_err());
}

#[test]
fn from_source_named_binds_custom_functions() {
    let m = PyMerit::from_source_named(
        "named",
        r"
def other_merit(array, meta):
    return 42.0
",
        "other_merit",
        None,
    )
    .unwrap();
    assert_eq!(m.evaluate(&obs()).unwrap(), 42.0);
}
