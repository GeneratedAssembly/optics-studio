//! Scalar source/beam parameter records (pre-sampling).
//!
//! These types describe an idealized beam — wavelength, amplitude, phase,
//! polarization — *before* it is sampled onto a spatial grid. Propagation
//! and element interaction operate on [`crate::field::BandField`] grids;
//! field initializers that turn a [`Wave`] into a sampled grid arrive with
//! the source library (Phase 5). Units: wavelength in meters, angles in
//! radians, amplitudes in arbitrary coherent units.

use num_complex::Complex64;

use crate::common::{Angle, J};

/// Wavelength, stored in SI meters; constructed from nanometers.
#[derive(Copy, Clone, Debug)]
pub struct Wavelength {
    value: f64, // Stored in meters
}

impl Wavelength {
    /// From a whole-number wavelength in nanometers (converted to meters).
    pub fn new(nanometers: u32) -> Self {
        Self {
            value: nanometers as f64 * 1e-9, // u32 to f64 is safe - no loss of precision
        }
    }

    /// The wavelength in meters (SI).
    pub fn as_meters(&self) -> f64 {
        self.value
    }

    /// The wavelength in nanometers.
    pub fn as_nanometers(&self) -> f64 {
        self.value * 1e9
    }
}

/// Polarization as a normalized Jones vector $(e_x, e_y)$ (dimensionless,
/// $|e_x|^2 + |e_y|^2 = 1$).
///
/// Polarization stays a beam-level property for now: vector/polarized field
/// grids are out of Phase 1 scope (see field docs).
#[derive(Copy, Clone, Debug)]
pub struct Polarization {
    ex: Complex64,
    ey: Complex64,
}

impl Polarization {
    /// Linear polarization oriented at `angle` (rad) in the transverse plane.
    pub fn linear(angle: Angle) -> Self {
        Self {
            ex: Complex64::from(angle.cos()),
            ey: Complex64::from(angle.sin()),
        }
    }

    /// Horizontal linear polarization ($0$ rad).
    pub fn horizontal() -> Self {
        Self::linear(Angle::zero())
    }

    /// Vertical linear polarization ($\pi/2$ rad).
    pub fn vertical() -> Self {
        Self::linear(Angle::pi_half())
    }

    /// Circular polarization; `left_handed` selects the handedness
    /// ($e_y = \pm i\, e_x$, equal power in both components).
    pub fn circular(left_handed: bool) -> Self {
        let sign = if left_handed { J } else { -J };
        let norm = (2.0_f64).sqrt();
        Polarization {
            ex: Complex64::new(1.0 / norm, 0.0),
            ey: sign / norm,
        }
    }

    /// General elliptical polarization from a Jones vector `(ex, ey)`;
    /// normalized on construction. A zero vector falls back to horizontal.
    pub fn elliptical(ex: Complex64, ey: Complex64) -> Self {
        let norm = (ex.norm_sqr() + ey.norm_sqr()).sqrt();
        if norm > 0.0 {
            Polarization {
                ex: ex / norm,
                ey: ey / norm,
            }
        } else {
            Polarization {
                ex: Complex64::new(1.0, 0.0),
                ey: Complex64::new(0.0, 0.0),
            }
        }
    }

    /// $x$ component of the Jones vector (dimensionless complex amplitude).
    pub fn ex(&self) -> Complex64 {
        self.ex
    }

    /// $y$ component of the Jones vector (dimensionless complex amplitude).
    pub fn ey(&self) -> Complex64 {
        self.ey
    }
}

/// Scalar source/beam parameter record: wavelength, amplitude, phase, and
/// polarization describing a idealized beam *before* it is sampled onto a
/// grid.
///
/// `Wave` is not a field type — elements and propagators operate on
/// [`crate::field::BandField`] grids. Field initializers that build a
/// [`crate::field::Field`] from a `Wave` arrive with the source library
/// (Phase 5); vector/polarized field grids remain out of scope for now
/// (`Polarization` stays a `Wave`-level property).
pub struct Wave {
    wavelength: Wavelength,
    amplitude: f64,
    phase: Angle, // radians
    /// `None` for unpolarized light.
    polarization: Option<Polarization>,
}

impl Wave {
    /// Assemble a beam record from wavelength, amplitude (arbitrary coherent
    /// units), phase (rad), and optional polarization (`None` = unpolarized).
    pub fn new(
        wavelength: Wavelength,
        amplitude: f64,
        phase: Angle,
        polarization: Option<Polarization>,
    ) -> Self {
        Self {
            wavelength,
            amplitude,
            phase,
            polarization,
        }
    }

    /// The operating wavelength.
    pub fn wavelength(&self) -> Wavelength {
        self.wavelength
    }

    /// Real amplitude $A$ (arbitrary coherent units; $|A|^2 \propto$ irradiance).
    pub fn amplitude(&self) -> f64 {
        self.amplitude
    }

    /// Constant phase (rad).
    pub fn phase(&self) -> Angle {
        self.phase
    }

    /// `None` for unpolarized light.
    pub fn polarization(&self) -> Option<Polarization> {
        self.polarization
    }
}
