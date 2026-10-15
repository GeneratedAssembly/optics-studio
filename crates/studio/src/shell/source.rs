//! Source picker model (spec §3.1): point source / tilted plane wave /
//! Gaussian beam, each with the grid spec the core source APIs take.
//!
//! Grid units are meters internally; the UI labels show mm/µm (§3.5).

use optics_core::field::BandField;
use optics_core::sources::{self, GaussianBeam};

/// The three source kinds the picker offers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SourceSpec {
    /// On-axis spherical wave sampled at distance `distance_m` (point
    /// source at the object).
    Point {
        /// Source→grid distance (m).
        distance_m: f64,
        /// Field amplitude.
        amplitude: f64,
        /// Grid size.
        nx: usize,
        /// Grid size.
        ny: usize,
        /// Grid pitch (m).
        dx: f64,
        /// Grid pitch (m).
        dy: f64,
    },
    /// Tilted plane wave at `theta_deg` field angles (degrees).
    PlaneWave {
        /// X tilt (degrees).
        theta_x_deg: f64,
        /// Y tilt (degrees).
        theta_y_deg: f64,
        /// Field amplitude.
        amplitude: f64,
        /// Grid size.
        nx: usize,
        /// Grid size.
        ny: usize,
        /// Grid pitch (m).
        dx: f64,
        /// Grid pitch (m).
        dy: f64,
    },
    /// Gaussian beam waist sampled at `z_m` downstream of the waist.
    Gaussian {
        /// Waist radius (m).
        w0: f64,
        /// Peak amplitude.
        amplitude: f64,
        /// Waist→grid distance (m).
        z_m: f64,
        /// Grid size.
        nx: usize,
        /// Grid size.
        ny: usize,
        /// Grid pitch (m).
        dx: f64,
        /// Grid pitch (m).
        dy: f64,
    },
}

impl SourceSpec {
    /// A small square grid shared by every kind (preview default).
    pub fn grid(self) -> (usize, usize, f64, f64) {
        match self {
            SourceSpec::Point { nx, ny, dx, dy, .. }
            | SourceSpec::PlaneWave { nx, ny, dx, dy, .. }
            | SourceSpec::Gaussian { nx, ny, dx, dy, .. } => (nx, ny, dx, dy),
        }
    }

    /// The traced band at `lambda_m` (meters).
    pub fn make_band(&self, lambda_m: f64) -> Result<BandField, SourceError> {
        match *self {
            SourceSpec::Point {
                distance_m,
                amplitude,
                nx,
                ny,
                dx,
                dy,
            } => Ok(sources::spherical(
                (0.0, 0.0),
                distance_m,
                amplitude,
                lambda_m,
                nx,
                ny,
                dx,
                dy,
            )?),
            SourceSpec::PlaneWave {
                theta_x_deg,
                theta_y_deg,
                amplitude,
                nx,
                ny,
                dx,
                dy,
            } => Ok(BandField::plane_wave(
                theta_x_deg.to_radians(),
                theta_y_deg.to_radians(),
                amplitude,
                nx,
                ny,
                dx,
                dy,
                lambda_m,
            )?),
            SourceSpec::Gaussian {
                w0,
                amplitude,
                z_m,
                nx,
                ny,
                dx,
                dy,
            } => Ok(sources::gaussian(
                &GaussianBeam {
                    w0,
                    amplitude,
                    center: (0.0, 0.0),
                },
                z_m,
                lambda_m,
                nx,
                ny,
                dx,
                dy,
            )?),
        }
    }
}

/// Source construction failures (typed `SourcesError`/`FieldError`).
#[derive(Clone, Debug, thiserror::Error)]
pub enum SourceError {
    /// Source geometry rejected.
    #[error(transparent)]
    Sources(#[from] optics_core::sources::SourcesError),
    /// Grid rejected by the field layer.
    #[error(transparent)]
    Field(#[from] optics_core::field::FieldError),
}
