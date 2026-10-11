use num_complex::Complex64;

use crate::common::{Angle, J};

/// Wavelength, stored in SI meters; constructed from nanometers.
#[derive(Copy, Clone, Debug)]
pub struct Wavelength {
    value: f64, // Stored in meters
}

impl Wavelength {
    pub fn new(nanometers: u32) -> Self {
        Self {
            value: nanometers as f64 * 1e-9, // u32 to f64 is safe - no loss of precision
        }
    }

    pub fn as_meters(&self) -> f64 {
        self.value
    }

    pub fn as_nanometers(&self) -> f64 {
        self.value * 1e9
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Polarization {
    ex: Complex64,
    ey: Complex64,
}

impl Polarization {
    /// Linear polarization oriented at `angle` in the transverse plane.
    pub fn linear(angle: Angle) -> Self {
        Self {
            ex: Complex64::from(angle.cos()),
            ey: Complex64::from(angle.sin()),
        }
    }

    pub fn horizontal() -> Self {
        Self::linear(Angle::zero())
    }

    pub fn vertical() -> Self {
        Self::linear(Angle::pi_half())
    }

    /// Circular polarization; `left_handed` selects the handedness.
    pub fn circular(left_handed: bool) -> Self {
        let sign = if left_handed { J } else { -J };
        let norm = (2.0_f64).sqrt();
        Polarization {
            ex: Complex64::new(1.0 / norm, 0.0),
            ey: sign / norm,
        }
    }

    /// General elliptical polarization from a Jones vector (ex, ey).
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

    pub fn ex(&self) -> Complex64 {
        self.ex
    }

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

    pub fn wavelength(&self) -> Wavelength {
        self.wavelength
    }

    pub fn amplitude(&self) -> f64 {
        self.amplitude
    }

    pub fn phase(&self) -> Angle {
        self.phase
    }

    /// `None` for unpolarized light.
    pub fn polarization(&self) -> Option<Polarization> {
        self.polarization
    }
}
