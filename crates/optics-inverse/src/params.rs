//! Bounded parameter vectors — the design space every solver searches.
//!
//! A [`Params`] is a fixed-length vector of [`Param`] slots, each with a
//! name, closed bounds `[lo, hi]`, a current value, and a finite-difference
//! step. It is the *only* structure optimizers see: analytic element
//! parameters and flattened grid degrees of freedom both live here, so a
//! solver never needs to know what it is tuning.
//!
//! Validation policy (agents.md §4 — prefer prevention): values are checked
//! at *apply* time (`Params::validate`) and surface as [`InverseError`],
//! never a panic. Solvers that must stay inside bounds use [`Params::clamp`]
//! (prevention) rather than relying on rejection.

use rand::Rng;
use rand::distributions::{Distribution, Uniform};

use crate::error::InverseError;

/// One tunable scalar with closed bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    /// Human-readable label (used in diagnostics and error messages).
    pub name: String,
    /// Lower bound (inclusive).
    pub lo: f64,
    /// Upper bound (inclusive).
    pub hi: f64,
    /// Current value.
    pub value: f64,
    /// Finite-difference step. When `None`, [`Param::fd_step`] falls back
    /// to a fraction of the bound span.
    pub fd_step: Option<f64>,
}

impl Param {
    /// Build a bounded parameter with an explicit initial value.
    ///
    /// `lo > hi` is a construction bug (structural), checked by
    /// `debug_assert!` like the core's mode-validity convention; a solver
    /// pushing `value` out of range is a *data* error and surfaces via
    /// [`Params::validate`], never a panic.
    pub fn new(name: impl Into<String>, lo: f64, hi: f64, value: f64) -> Self {
        debug_assert!(lo <= hi, "Param: lo ({lo}) > hi ({hi})");
        Self {
            name: name.into(),
            lo,
            hi,
            value,
            fd_step: None,
        }
    }

    /// Set an explicit finite-difference step (builder).
    pub fn with_step(mut self, step: f64) -> Self {
        self.fd_step = Some(step);
        self
    }

    /// Effective finite-difference step: the explicit value, else 1e-4 of
    /// the bound span (never zero — a degenerate point bound uses a small
    /// absolute floor).
    pub fn fd_step(&self) -> f64 {
        match self.fd_step {
            Some(s) => s,
            None => {
                let span = self.hi - self.lo;
                let rel = span * 1.0e-4;
                if rel > 0.0 { rel } else { 1.0e-9 }
            }
        }
    }

    /// Clamp the current value into `[lo, hi]` in place.
    pub fn clamp(&mut self) {
        self.value = self.value.clamp(self.lo, self.hi);
    }

    /// Validate this slot (finite bounds and value, value within bounds).
    pub fn validate(&self) -> Result<(), InverseError> {
        if !self.lo.is_finite() || !self.hi.is_finite() {
            return Err(InverseError::NotFinite {
                name: self.name.clone(),
                value: if self.lo.is_finite() {
                    self.hi
                } else {
                    self.lo
                },
            });
        }
        if !self.value.is_finite() {
            return Err(InverseError::NotFinite {
                name: self.name.clone(),
                value: self.value,
            });
        }
        if self.value < self.lo || self.value > self.hi {
            return Err(InverseError::OutOfBounds {
                name: self.name.clone(),
                value: self.value,
                lo: self.lo,
                hi: self.hi,
            });
        }
        Ok(())
    }
}

/// A fixed-length design-space vector of [`Param`] slots.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Params {
    slots: Vec<Param>,
}

impl Params {
    /// Empty parameter vector (dimension 0).
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Build from an iterable of slots.
    pub fn from_slots(iter: impl IntoIterator<Item = Param>) -> Self {
        Self {
            slots: iter.into_iter().collect(),
        }
    }

    /// Number of tunable degrees of freedom.
    pub fn dim(&self) -> usize {
        self.slots.len()
    }

    /// Slice view of the slots.
    pub fn slots(&self) -> &[Param] {
        &self.slots
    }

    /// Mutable slice view of the slots.
    pub fn slots_mut(&mut self) -> &mut [Param] {
        &mut self.slots
    }

    /// Current values as a flat `Vec<f64>` (the form carriers consume).
    pub fn values(&self) -> Vec<f64> {
        self.slots.iter().map(|p| p.value).collect()
    }

    /// Effective finite-difference steps as a flat `Vec<f64>`.
    pub fn fd_steps(&self) -> Vec<f64> {
        self.slots.iter().map(|p| p.fd_step()).collect()
    }

    /// Write a flat value vector into the slots by index.
    ///
    /// Errors with [`InverseError::WrongDimension`] if lengths differ;
    /// does not validate bounds (call [`Params::validate`] after).
    pub fn set_values(&mut self, values: &[f64]) -> Result<(), InverseError> {
        if values.len() != self.dim() {
            return Err(InverseError::WrongDimension {
                expected: self.dim(),
                actual: values.len(),
            });
        }
        for (slot, &v) in self.slots.iter_mut().zip(values) {
            slot.value = v;
        }
        Ok(())
    }

    /// Validate every slot (see [`Params::set_values`] which does not).
    pub fn validate(&self) -> Result<(), InverseError> {
        for slot in &self.slots {
            slot.validate()?;
        }
        Ok(())
    }

    /// Clamp every value into its bounds (prevention policy for solvers).
    pub fn clamp_all(&mut self) {
        for slot in &mut self.slots {
            slot.clamp();
        }
    }

    /// Fill each slot's value uniformly at random within `[lo, hi]`.
    pub fn randomize<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        for slot in &mut self.slots {
            if slot.lo == slot.hi {
                slot.value = slot.lo;
            } else {
                slot.value = Uniform::new(slot.lo, slot.hi).sample(rng);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_step_default_is_relative_to_span() {
        let p = Param::new("x", 0.0, 1.0, 0.5);
        assert!((p.fd_step() - 1.0e-4).abs() < 1.0e-12);
        let q = Param::new("y", -2.0, 2.0, 0.0).with_step(0.01);
        assert!((q.fd_step() - 0.01).abs() < 1.0e-12);
    }

    #[test]
    fn fd_step_degenerate_bound_has_floor() {
        let p = Param::new("k", 3.0, 3.0, 3.0);
        assert!(p.fd_step() > 0.0);
    }

    #[test]
    fn validate_rejects_out_of_bounds() {
        let mut ps = Params::from_slots([Param::new("x", 0.0, 1.0, 0.5)]);
        assert!(ps.validate().is_ok());
        ps.slots_mut()[0].value = 1.5;
        let err = ps.validate().unwrap_err();
        assert!(matches!(err, InverseError::OutOfBounds { .. }));
    }

    #[test]
    fn validate_rejects_nan() {
        let ps = Params::from_slots([Param::new("x", 0.0, 1.0, f64::NAN)]);
        assert!(matches!(
            ps.validate().unwrap_err(),
            InverseError::NotFinite { .. }
        ));
    }

    #[test]
    fn clamp_recovers_out_of_range() {
        let mut ps = Params::from_slots([
            Param::new("a", 0.0, 1.0, -0.3),
            Param::new("b", 0.0, 1.0, 1.7),
        ]);
        ps.clamp_all();
        assert_eq!(ps.values(), vec![0.0, 1.0]);
        assert!(ps.validate().is_ok());
    }

    #[test]
    fn set_values_checks_dimension() {
        let mut ps = Params::from_slots([Param::new("x", 0.0, 1.0, 0.5)]);
        assert!(matches!(
            ps.set_values(&[0.2, 0.3]).unwrap_err(),
            InverseError::WrongDimension {
                expected: 1,
                actual: 2
            }
        ));
        ps.set_values(&[0.25]).unwrap();
        assert_eq!(ps.values(), vec![0.25]);
    }

    #[test]
    fn randomize_stays_in_bounds() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut ps = Params::from_slots([
            Param::new("a", -1.0, -0.5, -0.7),
            Param::new("b", 2.0, 3.0, 2.5),
        ]);
        for _ in 0..64 {
            ps.randomize(&mut rng);
            assert!(ps.validate().is_ok());
        }
    }
}
