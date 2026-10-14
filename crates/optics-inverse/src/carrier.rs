//! Design carriers: what a parameter vector *means*.
//!
//! A [`Carrier`] maps a flat `Vec<f64>` (plus the wavelength the renderer
//! is evaluating) to an owned [`Design`] — an element stack that borrows
//! itself into a [`optics_core::beamline::Beamline`] on demand. The
//! ownership pattern follows the zemax builder's two-phase plan: elements
//! are owned by the `Design`, steps reference them by index, so no
//! self-referential borrows appear in carrier closures.
//!
//! Three carriers cover the daily design space:
//!
//! - [`Carrier::analytic`] — a user closure: analytic element parameters
//!   (curvatures, gaps, focal lengths, blaze periods) build a whole
//!   system per call. The closure also receives the band wavelength so
//!   dispersive rebuilds are possible.
//! - [`Carrier::grid_phase`] — sampled **phase** (rad) grids: each
//!   parameter slot is one lattice sample of a
//!   [`optics_core::element::PhaseMask::sampled`] mask. Band-independent
//!   (the mask *is* the phase).
//! - [`Carrier::grid_thickness`] — sampled **relief thickness** (m) grids
//!   in a material: the DOE carrier of the coded-aperture workflow,
//!   $\varphi(x,y;\lambda) = \frac{2\pi}{\lambda}\,[n(\lambda)-1]\,h(x,y)$,
//!   rebuilt per band from `Material::n_at`.

use optics_core::beamline::Beamline;
use optics_core::element::{ClearAperture, Element, Material, PhaseMask};

use crate::Result;
use crate::error::InverseError;
use crate::params::Params;

/// One beamline step: an optional element index, the vacuum-equivalent
/// distance after it (m), and the medium index of that propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    /// Index into the owning [`Design`]'s element list; `None` is a pure
    /// gap (flat interface / spacing).
    pub element: Option<usize>,
    /// Distance propagated after this plane (m). `0.0` means no spacing.
    pub z_after: f64,
    /// Refractive index of the medium of `z_after` (1.0 = air/vacuum).
    pub medium: f64,
}

/// An owned element stack plus its step list — the artifact a carrier
/// builds and a renderer traces. [`Design::beamline`] borrows the elements
/// into a transient [`Beamline`] without copying.
#[derive(Default)]
pub struct Design {
    elements: Vec<Box<dyn Element>>,
    steps: Vec<Step>,
}

impl Design {
    /// Empty design (its beamline is the identity).
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an owned element followed by propagation `z_after` (m)
    /// through medium `n`.
    pub fn push(
        &mut self,
        element: impl Element + 'static,
        z_after: f64,
        medium: f64,
    ) -> &mut Self {
        self.elements.push(Box::new(element));
        self.steps.push(Step {
            element: Some(self.elements.len() - 1),
            z_after,
            medium,
        });
        self
    }

    /// Append a pure propagation step (no element).
    pub fn gap(&mut self, z_after: f64, medium: f64) -> &mut Self {
        self.steps.push(Step {
            element: None,
            z_after,
            medium,
        });
        self
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// `true` when there are no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Borrow this design as a [`Beamline`] (elements stay owned here).
    pub fn beamline(&self) -> Beamline<'_> {
        let mut bl = Beamline::new();
        for step in &self.steps {
            match step.element {
                Some(i) => bl.push_medium(&*self.elements[i], step.z_after, step.medium),
                None => bl.gap(step.z_after, step.medium),
            };
        }
        bl
    }
}

/// Sampled-grid carrier geometry shared by [`Carrier::grid_phase`] and
/// [`Carrier::grid_thickness`]. Fields are private; construct via the
/// [`Carrier`] grid builders.
#[derive(Clone)]
pub struct GridSpec {
    params: Params,
    dims: (usize, usize),
    pitch: (f64, f64),
    aperture: ClearAperture,
    center: (f64, f64),
}

/// Geometry + slot bounds for a grid carrier constructor.
#[derive(Clone, Copy, Debug)]
pub struct GridGeom {
    /// Lattice sample counts `(nx, ny)`.
    pub dims: (usize, usize),
    /// Lattice pitch (m).
    pub pitch: (f64, f64),
    /// Clear aperture (stop).
    pub aperture: ClearAperture,
    /// Grid center in field coordinates (m).
    pub center: (f64, f64),
}

impl GridSpec {
    fn new(geom: GridGeom, lo: f64, hi: f64, init: f64, prefix: &str) -> Result<Self> {
        let (nx, ny) = geom.dims;
        if nx == 0 || ny == 0 {
            return Err(InverseError::Carrier {
                reason: format!("grid dims must be positive, got ({nx}, {ny})"),
            });
        }
        Ok(Self {
            params: Params::uniform(nx * ny, lo, hi, init, prefix),
            dims: geom.dims,
            pitch: geom.pitch,
            aperture: geom.aperture,
            center: geom.center,
        })
    }

    /// Validate values, convert them to phase samples, and wrap the result
    /// in a single-element `Design`.
    fn build_mask(&self, values: &[f64], phase: Vec<f64>) -> Result<Design> {
        self.params.validate_values(values)?;
        let mask = PhaseMask::sampled(
            phase,
            self.dims,
            self.pitch,
            None,
            self.aperture,
            self.center,
        )
        .map_err(|e| InverseError::Carrier {
            reason: e.to_string(),
        })?;
        let mut design = Design::new();
        design.push(mask, 0.0, 1.0);
        Ok(design)
    }
}

/// Analytic builder closure: `(values, lambda_m) -> Result<Design>`.
pub type AnalyticBuilder = dyn Fn(&[f64], f64) -> Result<Design>;

/// What the parameter vector tunes. See the [module docs](crate::carrier).
pub enum Carrier {
    /// Analytic element parameters: closure `(values, lambda_m) -> Design`.
    Analytic {
        params: Params,
        build: Box<AnalyticBuilder>,
    },
    /// Sampled phase grid (radians per slot).
    GridPhase { grid: GridSpec },
    /// Sampled thickness grid (meters per slot) in a dispersive material.
    GridThickness { grid: GridSpec, material: Material },
}

impl Carrier {
    /// Analytic carrier from a builder closure over `params` (whose
    /// dimension the closure must respect).
    pub fn analytic(
        params: Params,
        build: impl Fn(&[f64], f64) -> Result<Design> + 'static,
    ) -> Self {
        Self::Analytic {
            params,
            build: Box::new(build),
        }
    }

    /// Sampled phase-mask carrier: `dims` lattice with `pitch` (m), all
    /// slots bounded `[lo, hi]` radians, initialized at `init`.
    pub fn grid_phase(
        dims: (usize, usize),
        pitch: (f64, f64),
        aperture: ClearAperture,
        center: (f64, f64),
        lo: f64,
        hi: f64,
        init: f64,
    ) -> Result<Self> {
        Ok(Self::GridPhase {
            grid: GridSpec::new(
                GridGeom {
                    dims,
                    pitch,
                    aperture,
                    center,
                },
                lo,
                hi,
                init,
                "phi",
            )?,
        })
    }

    /// Sampled relief-thickness carrier: `dims` lattice with `pitch` (m),
    /// all slots bounded `[0, t_max]` meters, initialized at `init`, in
    /// `material` (phase rebuilt per band as $2\pi (n(\lambda)-1) h/\lambda$).
    pub fn grid_thickness(
        dims: (usize, usize),
        pitch: (f64, f64),
        aperture: ClearAperture,
        center: (f64, f64),
        t_max: f64,
        init: f64,
        material: Material,
    ) -> Result<Self> {
        if !(0.0..f64::INFINITY).contains(&t_max) {
            return Err(InverseError::Carrier {
                reason: format!("t_max must be positive and finite, got {t_max}"),
            });
        }
        Ok(Self::GridThickness {
            grid: GridSpec::new(
                GridGeom {
                    dims,
                    pitch,
                    aperture,
                    center,
                },
                0.0,
                t_max,
                init,
                "h",
            )?,
            material,
        })
    }

    /// The design space (bounds, names, FD steps).
    pub fn params(&self) -> &Params {
        match self {
            Self::Analytic { params, .. } => params,
            Self::GridPhase { grid } | Self::GridThickness { grid, .. } => &grid.params,
        }
    }

    /// Number of tunable degrees of freedom.
    pub fn dim(&self) -> usize {
        self.params().dim()
    }

    /// Single-slot bounds helper for grid carriers (all slots share them).
    pub fn slot_bounds(&self) -> Option<(f64, f64)> {
        self.params().slots().first().map(|p| (p.lo, p.hi))
    }

    /// Build the design from a parameter vector at band `lambda` (meters).
    ///
    /// Validation is apply-time: dimension, finiteness, and bounds surface
    /// as [`InverseError`]; a failing carrier closure or mask construction
    /// maps to [`InverseError::Carrier`].
    pub fn build(&self, values: &[f64], lambda: f64) -> Result<Design> {
        match self {
            Self::Analytic { params, build } => {
                params.validate_values(values)?;
                build(values, lambda)
            }
            Self::GridPhase { grid } => {
                grid.params.validate_values(values)?;
                grid.build_mask(values, values.to_vec())
            }
            Self::GridThickness { grid, material } => {
                grid.params.validate_values(values)?;
                if !lambda.is_finite() || lambda <= 0.0 {
                    return Err(InverseError::Carrier {
                        reason: format!("wavelength must be positive and finite, got {lambda}"),
                    });
                }
                let n = material.n_at(lambda, 0.0);
                if !n.is_finite() {
                    return Err(InverseError::Carrier {
                        reason: format!("material index not finite at lambda {lambda}: {n}"),
                    });
                }
                let scale = 2.0 * std::f64::consts::PI * (n - 1.0) / lambda;
                let phase: Vec<f64> = values.iter().map(|h| scale * h).collect();
                grid.build_mask(values, phase)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Param;
    use num_complex::Complex64;
    use optics_core::element::ThinLens;
    use optics_core::field::{BandField, Field};

    fn unit_field(n: usize, d: f64) -> Field {
        Field::new(vec![Complex64::new(1.0, 0.0); n * n], n, n, d, d).unwrap()
    }

    fn band(n: usize, d: f64, lambda: f64) -> BandField {
        BandField::new(lambda, unit_field(n, d)).unwrap()
    }

    fn identity_propagate(b: &BandField, _z: f64, _m: f64) -> Result<BandField> {
        Ok(b.clone())
    }

    #[test]
    fn analytic_builds_deterministic_design() {
        let carrier = Carrier::analytic(
            Params::from_slots([Param::new("f", 0.01, 1.0, 0.05)]),
            |v, _lambda| {
                let f = v[0];
                let mut d = Design::new();
                let lens = ThinLens::new(move |_| f, None, ClearAperture::Field, (0.0, 0.0))
                    .map_err(|e| InverseError::Carrier {
                        reason: e.to_string(),
                    })?;
                d.push(lens, 0.0, 1.0);
                Ok(d)
            },
        );
        assert_eq!(carrier.dim(), 1);
        let b = band(16, 10e-6, 500e-9);
        let out1 = carrier
            .build(&[0.05], 500e-9)
            .unwrap()
            .beamline()
            .trace(&b, identity_propagate)
            .unwrap();
        let out2 = carrier
            .build(&[0.05], 500e-9)
            .unwrap()
            .beamline()
            .trace(&b, identity_propagate)
            .unwrap();
        assert_eq!(out1.field().samples(), out2.field().samples());
        // A different focal length changes the quadratic phase.
        let out3 = carrier
            .build(&[0.02], 500e-9)
            .unwrap()
            .beamline()
            .trace(&b, identity_propagate)
            .unwrap();
        assert_ne!(out1.field().samples(), out3.field().samples());
    }

    #[test]
    fn analytic_rejects_bad_values_at_apply() {
        let carrier = Carrier::analytic(
            Params::from_slots([Param::new("f", 0.01, 1.0, 0.05)]),
            |v, _| {
                let mut d = Design::new();
                d.gap(v[0], 1.0);
                Ok(d)
            },
        );
        assert!(matches!(
            carrier.build(&[5.0], 500e-9),
            Err(InverseError::OutOfBounds { .. })
        ));
        assert!(matches!(
            carrier.build(&[f64::NAN], 500e-9),
            Err(InverseError::NotFinite { .. })
        ));
        assert!(matches!(
            carrier.build(&[], 500e-9),
            Err(InverseError::WrongDimension { .. })
        ));
    }

    #[test]
    fn grid_phase_roundtrip_matches_manual_mask() {
        let dims = (4, 4);
        let pitch = (20e-6, 20e-6);
        let values: Vec<f64> = (0..16).map(|k| k as f64 * 0.2).collect();
        let carrier = Carrier::grid_phase(
            dims,
            pitch,
            ClearAperture::Field,
            (0.0, 0.0),
            0.0,
            std::f64::consts::PI,
            0.0,
        )
        .unwrap();
        assert_eq!(carrier.dim(), 16);
        assert_eq!(carrier.slot_bounds(), Some((0.0, std::f64::consts::PI)));
        let b = band(32, pitch.0, 500e-9);
        let out = carrier
            .build(&values, 500e-9)
            .unwrap()
            .beamline()
            .trace(&b, identity_propagate)
            .unwrap();
        let manual = PhaseMask::sampled(
            values.clone(),
            dims,
            pitch,
            None,
            ClearAperture::Field,
            (0.0, 0.0),
        )
        .unwrap()
        .apply(&b)
        .unwrap();
        assert_eq!(out.field().samples(), manual.field().samples());
    }

    #[test]
    fn grid_thickness_applies_chromatic_phase_law() {
        // n = 1.5 constant, h = 500 nm, lambda = 500 nm -> phase = pi
        // everywhere: the applied field is exactly -input.
        let material = Material::new(|_l, _a| 1.5, |_l, _a| 0.0);
        let carrier = Carrier::grid_thickness(
            (4, 4),
            (20e-6, 20e-6),
            ClearAperture::Field,
            (0.0, 0.0),
            10e-6,
            500e-9,
            material,
        )
        .unwrap();
        let values = vec![500e-9; 16];
        // Field = lattice extent: samples outside the lattice are
        // legitimately phase-0 (transparent), so compare on-grid.
        let b = band(4, 20e-6, 500e-9);
        let out = carrier
            .build(&values, 500e-9)
            .unwrap()
            .beamline()
            .trace(&b, identity_propagate)
            .unwrap();
        for s in out.field().samples() {
            assert!((s - Complex64::new(-1.0, 0.0)).norm() < 1e-12, "{s}");
        }
    }

    #[test]
    fn grid_thickness_is_band_dependent_via_dispersion() {
        // n(lambda) = 1.4 + 100 nm / lambda: phase law differs per band.
        let material = Material::new(|l, _a| 1.4 + 100e-9 / l, |_l, _a| 0.0);
        let carrier = Carrier::grid_thickness(
            (2, 2),
            (10e-6, 10e-6),
            ClearAperture::Field,
            (0.0, 0.0),
            10e-6,
            1e-6,
            material,
        )
        .unwrap();
        let values = vec![1e-6; 4];
        let b1 = band(2, 10e-6, 400e-9);
        let b2 = band(2, 10e-6, 700e-9);
        let o1 = carrier
            .build(&values, 400e-9)
            .unwrap()
            .beamline()
            .trace(&b1, identity_propagate)
            .unwrap();
        let o2 = carrier
            .build(&values, 700e-9)
            .unwrap()
            .beamline()
            .trace(&b2, identity_propagate)
            .unwrap();
        // phi = 2pi (n-1) h / lambda with n-1 = 0.4 + 1e-7/lambda.
        let expect =
            |lambda: f64| 2.0 * std::f64::consts::PI * (0.4 + 100e-9 / lambda) * 1e-6 / lambda;
        let s1 = o1.field().samples()[0];
        let s2 = o2.field().samples()[0];
        let e1 = expect(400e-9);
        let e2 = expect(700e-9);
        assert!((s1 - Complex64::from_polar(1.0, e1)).norm() < 1e-9);
        assert!((s2 - Complex64::from_polar(1.0, e2)).norm() < 1e-9);
        assert!((e1 - e2).abs() > 0.1, "bands must differ: {e1} vs {e2}");
    }

    #[test]
    fn grid_carriers_validate_before_building() {
        let carrier = Carrier::grid_phase(
            (2, 2),
            (1e-5, 1e-5),
            ClearAperture::Field,
            (0.0, 0.0),
            0.0,
            1.0,
            0.5,
        )
        .unwrap();
        assert!(matches!(
            carrier.build(&[0.5, 0.5, 0.5, 1.5], 500e-9),
            Err(InverseError::OutOfBounds { .. })
        ));
        assert!(matches!(
            carrier.build(&[0.5; 3], 500e-9),
            Err(InverseError::WrongDimension { .. })
        ));
        // Zero-dim grids are rejected at construction.
        assert!(
            Carrier::grid_phase(
                (0, 4),
                (1e-5, 1e-5),
                ClearAperture::Field,
                (0.0, 0.0),
                0.0,
                1.0,
                0.5
            )
            .is_err()
        );
    }
}
