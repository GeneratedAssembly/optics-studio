//! Complex scalar optical fields sampled on uniform 2D grids.
//!
//! Two layers live here:
//!
//! - [`Field`] — a purely spatial sample grid: complex amplitudes $u(x, y)$
//!   with a pitch and origin-centered coordinates, **independent of any
//!   wavelength**. Geometry, indexing, and intensity bookkeeping operate at
//!   this level.
//! - [`BandField`] — a [`Field`] bound to one monochromatic wavelength
//!   $\lambda$. This is the unit of propagation and element interaction,
//!   because phase effects (tilt, ASM transfer functions, thin-lens masks) are
//!   wavelength-dependent.
//!
//! The split exists so that future **incoherent multiband sources** can be
//! represented as a collection of `BandField`s (one per band, each sharing a
//! common [`Field`] grid geometry) whose intensities sum at detection. Nothing
//! at the [`Field`] level assumes a single band.
//!
//! # Grid coordinate convention
//!
//! Sample storage is row-major: the sample at column `ix` (x-direction) and
//! row `iy` (y-direction) is stored at flat index `iy * nx + ix`, so rows run
//! along $x$ and the grid has `ny` rows of `nx` samples. The physical
//! coordinate of sample `(ix, iy)` is
//!
//! $$x = \left(ix - \frac{n_x - 1}{2}\right)\Delta x, \qquad
//!   y = \left(iy - \frac{n_y - 1}{2}\right)\Delta y$$
//!
//! with the origin $(0, 0)$ at the **center** of the grid: an odd-dimensioned
//! grid has a sample exactly at the origin, and an even-dimensioned grid is
//! centered between samples. All spatial quantities are SI units (meters);
//! field amplitudes are in arbitrary coherent units whose squared modulus is
//! proportional to irradiance.

use std::f64::consts::PI;

use num_complex::Complex64;
use thiserror::Error;

/// Errors produced when constructing or validating a [`Field`] or [`BandField`].
///
/// A deliberate, minimal taxonomy: every variant corresponds to a grid or band
/// that cannot physically or numerically represent a sampled wavefront.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum FieldError {
    /// `nx` or `ny` was zero; a field must contain at least one sample.
    #[error("field grid is empty (nx and ny must be > 0)")]
    EmptyGrid,
    /// The flat sample buffer length does not equal `nx * ny`.
    #[error("sample buffer length {len} does not match grid size {expected} (nx * ny)")]
    DimensionMismatch {
        /// Number of samples provided.
        len: usize,
        /// Number of samples required (`nx * ny`).
        expected: usize,
    },
    /// A sample pitch (`dx` or `dy`) was not strictly positive and finite.
    #[error("sample pitch must be positive and finite (got dx = {dx} m, dy = {dy} m)")]
    NonPositivePitch {
        /// x-axis sample pitch in meters.
        dx: f64,
        /// y-axis sample pitch in meters.
        dy: f64,
    },
    /// Wavelength was not strictly positive and finite (meters).
    #[error("wavelength must be positive and finite (got {0} m)")]
    NonPositiveWavelength(f64),
    /// At least one complex sample had a non-finite real or imaginary part.
    #[error("field contains a non-finite sample")]
    NonFiniteSample,
}

fn valid_pitch(p: f64) -> bool {
    p.is_finite() && p > 0.0
}

fn valid_wavelength(lambda: f64) -> bool {
    lambda.is_finite() && lambda > 0.0
}

/// A complex scalar field $u(x, y)$ on a uniform, origin-centered grid,
/// independent of wavelength.
///
/// The field is fully described by its complex samples and the sample pitch
/// $(\Delta x, \Delta y)$ in meters. Intensity is $|u|^2$ and total discrete
/// power is $P = \sum_{ix, iy} |u|^2\, \Delta x\, \Delta y$
/// (units of $|u|^2\,\mathrm{m}^2$). To propagate or apply wavelength-dependent
/// elements, bind a `Field` to a wavelength with [`BandField`].
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    samples: Vec<Complex64>, // row-major: index = iy * nx + ix
    nx: usize,
    ny: usize,
    dx: f64, // sample pitch, meters
    dy: f64, // sample pitch, meters
}

impl Field {
    /// Builds a field from a row-major flat buffer of complex samples.
    ///
    /// # Arguments
    /// - `samples`: length must be exactly `nx * ny`; row-major with the
    ///   sample at `(ix, iy)` located at `samples[iy * nx + ix]`.
    /// - `nx`, `ny`: number of samples along $x$ and $y$; both must be > 0.
    /// - `dx`, `dy`: sample pitch in meters; both must be positive, finite.
    ///
    /// # Errors
    /// Returns [`FieldError`] describing the first violated invariant:
    /// [`FieldError::EmptyGrid`], [`FieldError::DimensionMismatch`],
    /// [`FieldError::NonPositivePitch`], or [`FieldError::NonFiniteSample`].
    pub fn new(
        samples: Vec<Complex64>,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
    ) -> Result<Self, FieldError> {
        if nx == 0 || ny == 0 {
            return Err(FieldError::EmptyGrid);
        }
        let expected = nx * ny;
        if samples.len() != expected {
            return Err(FieldError::DimensionMismatch {
                len: samples.len(),
                expected,
            });
        }
        if !valid_pitch(dx) || !valid_pitch(dy) {
            return Err(FieldError::NonPositivePitch { dx, dy });
        }
        if samples
            .iter()
            .any(|s| !s.re.is_finite() || !s.im.is_finite())
        {
            return Err(FieldError::NonFiniteSample);
        }
        Ok(Self {
            samples,
            nx,
            ny,
            dx,
            dy,
        })
    }

    /// Field of all-zero samples on an `nx` x `ny` grid with pitch `dx`, `dy`
    /// (meters).
    ///
    /// # Errors
    /// Same grid-validity errors as [`Field::new`] (no non-finite-sample case).
    pub fn zeros(nx: usize, ny: usize, dx: f64, dy: f64) -> Result<Self, FieldError> {
        Self::new(vec![Complex64::ZERO; nx.saturating_mul(ny)], nx, ny, dx, dy)
    }

    /// Uniform plane at normal incidence: every sample equals
    /// $u = A\, e^{i\varphi}$.
    ///
    /// # Arguments
    /// - `amplitude`: real amplitude $A$ (arbitrary coherent units).
    /// - `phase`: constant phase $\varphi$ in radians.
    /// - `nx`, `ny`: sample counts along $x$, $y$.
    /// - `dx`, `dy`: sample pitch in meters.
    ///
    /// # Errors
    /// Same grid-validity errors as [`Field::new`].
    pub fn uniform(
        amplitude: f64,
        phase: f64,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
    ) -> Result<Self, FieldError> {
        let value = Complex64::from(amplitude) * (phase * Complex64::I).exp();
        Self::new(vec![value; nx.saturating_mul(ny)], nx, ny, dx, dy)
    }

    /// Grid dimensions `(nx, ny)` in samples.
    pub fn dims(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Sample pitch `(dx, dy)` in meters.
    pub fn pitch(&self) -> (f64, f64) {
        (self.dx, self.dy)
    }

    /// Complex sample at column `ix` (x index, `0..nx`) and row `iy`
    /// (y index, `0..ny`).
    ///
    /// # Panics
    /// Panics if `ix >= nx` or `iy >= ny`.
    pub fn sample(&self, ix: usize, iy: usize) -> Complex64 {
        self.samples[iy * self.nx + ix]
    }

    /// Mutable reference to the complex sample at column `ix`, row `iy`.
    ///
    /// # Panics
    /// Panics if `ix >= nx` or `iy >= ny`.
    pub fn sample_mut(&mut self, ix: usize, iy: usize) -> &mut Complex64 {
        &mut self.samples[iy * self.nx + ix]
    }

    /// Physical $(x, y)$ coordinate in meters of sample `(ix, iy)`, relative
    /// to the grid center.
    pub fn coord(&self, ix: usize, iy: usize) -> (f64, f64) {
        (
            (ix as f64 - (self.nx as f64 - 1.0) / 2.0) * self.dx,
            (iy as f64 - (self.ny as f64 - 1.0) / 2.0) * self.dy,
        )
    }

    /// Nearest-neighbor sample lookup at physical coordinates `(x, y)` in
    /// meters relative to the grid center.
    ///
    /// Returns `None` when `(x, y)` falls outside the grid extent
    /// $|x| \le \frac{n_x - 1}{2}\Delta x$, $|y| \le \frac{n_y - 1}{2}\Delta y$.
    pub fn at(&self, x: f64, y: f64) -> Option<Complex64> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let half_x = (self.nx as f64 - 1.0) / 2.0 * self.dx;
        let half_y = (self.ny as f64 - 1.0) / 2.0 * self.dy;
        if x.abs() > half_x || y.abs() > half_y {
            return None;
        }
        let ix = (x / self.dx + (self.nx as f64 - 1.0) / 2.0).round();
        let iy = (y / self.dy + (self.ny as f64 - 1.0) / 2.0).round();
        Some(self.sample(
            ix.clamp(0.0, self.nx as f64 - 1.0) as usize,
            iy.clamp(0.0, self.ny as f64 - 1.0) as usize,
        ))
    }

    /// Amplitude $|u|$ of every sample, row-major.
    pub fn amplitudes(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.norm()).collect()
    }

    /// Phase $\arg u \in (-\pi, \pi]$ of every sample, radians, row-major.
    pub fn phases(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.arg()).collect()
    }

    /// Intensity $|u|^2$ of every sample (arbitrary irradiance units), row-major.
    pub fn intensities(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.norm_sqr()).collect()
    }

    /// Total discrete power $P = \sum_{ix, iy} |u|^2\, \Delta x\, \Delta y$.
    ///
    /// With intensity interpreted as irradiance (W/m^2), `power()` is in watts;
    /// it is the natural energy-conservation invariant for lossless propagation
    /// of a single [`BandField`]. Incoherent multiband total power is the sum
    /// of per-band powers.
    pub fn power(&self) -> f64 {
        self.intensities().iter().sum::<f64>() * self.dx * self.dy
    }

    /// Row-major complex samples (flat, length `nx * ny`).
    pub fn samples(&self) -> &[Complex64] {
        &self.samples
    }
}

/// A [`Field`] bound to a single monochromatic wavelength: the unit of
/// propagation and wavelength-dependent element interaction.
///
/// A monochromatic band field represents the complex amplitude of light at
/// wavelength $\lambda$ sampled on the grid. Future incoherent multiband
/// sources are modeled as a collection of `BandField`s — one per band —
/// sharing common grid geometry; their *intensities* (not amplitudes) sum at
/// detection, since different bands do not maintain a stable phase
/// relationship. Coherent addition is only valid between `BandField`s of equal
/// wavelength on identical grids.
#[derive(Clone, Debug, PartialEq)]
pub struct BandField {
    lambda: f64,
    field: Field,
}

impl BandField {
    /// Binds an existing grid to a wavelength.
    ///
    /// # Arguments
    /// - `lambda`: operating wavelength in meters; positive and finite.
    /// - `field`: the spatial sample grid.
    ///
    /// # Errors
    /// [`FieldError::NonPositiveWavelength`] if `lambda` is not positive and
    /// finite.
    pub fn new(lambda: f64, field: Field) -> Result<Self, FieldError> {
        if !valid_wavelength(lambda) {
            return Err(FieldError::NonPositiveWavelength(lambda));
        }
        Ok(Self { lambda, field })
    }

    /// Uniform plane at normal incidence, $u = A\, e^{i\varphi}$, at
    /// wavelength `lambda` (meters).
    ///
    /// # Errors
    /// Grid-validity errors from [`Field::uniform`] plus
    /// [`FieldError::NonPositiveWavelength`].
    pub fn uniform(
        amplitude: f64,
        phase: f64,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        lambda: f64,
    ) -> Result<Self, FieldError> {
        if !valid_wavelength(lambda) {
            return Err(FieldError::NonPositiveWavelength(lambda));
        }
        let field = Field::uniform(amplitude, phase, nx, ny, dx, dy)?;
        Ok(Self { lambda, field })
    }

    /// Tilted plane wave with direction cosines $\alpha = \sin\theta_x$,
    /// $\beta = \sin\theta_y$:
    ///
    /// $$u(x, y) = A \exp\!\Big(i \frac{2\pi}{\lambda}\,(\alpha x + \beta y)\Big)$$
    ///
    /// following the plane-wave basis $\exp(i\,\mathbf{k}\cdot\mathbf{r})$ with
    /// $\mathbf{k}\cdot\mathbf{r} = \frac{2\pi}{\lambda}(\alpha x + \beta y + \gamma z)$
    /// (textbook Eq. 3.5, direction cosines $\alpha^2+\beta^2+\gamma^2=1$).
    /// The field is sampled on the $z = 0$ plane.
    ///
    /// # Arguments
    /// - `theta_x`, `theta_y`: tilt angles from the $z$ axis in the $x$-$z$ and
    ///   $y$-$z$ planes, in radians. Requires
    ///   $\sin^2\theta_x + \sin^2\theta_y \le 1$ (propagating, not evanescent).
    /// - `amplitude`: real amplitude $A$.
    /// - `nx`, `ny`: sample counts along $x$, $y$.
    /// - `dx`, `dy`: sample pitch in meters.
    /// - `lambda`: wavelength in meters.
    ///
    /// # Errors
    /// [`FieldError::NonPositiveWavelength`] for an invalid `lambda` (checked
    /// before grid construction); same grid-validity errors as [`Field::new`].
    /// An evanescent direction cosine still yields a finite (physically
    /// meaningless) field — check $\sin^2\theta_x + \sin^2\theta_y \le 1$ at
    /// the call site.
    // The full grid specification (nx, ny, dx, dy) is required to place the
    // wavefront; grouping it into a separate struct is deferred until more
    // field initializers exist (Phase 5 sources).
    #[allow(clippy::too_many_arguments)]
    pub fn plane_wave(
        theta_x: f64,
        theta_y: f64,
        amplitude: f64,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        lambda: f64,
    ) -> Result<Self, FieldError> {
        if !valid_wavelength(lambda) {
            return Err(FieldError::NonPositiveWavelength(lambda));
        }
        let k = 2.0 * PI / lambda;
        let alpha = theta_x.sin();
        let beta = theta_y.sin();
        let mut samples = Vec::with_capacity(nx.saturating_mul(ny));
        for iy in 0..ny {
            let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dy;
            for ix in 0..nx {
                let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx;
                samples.push(amplitude * (k * (alpha * x + beta * y) * Complex64::I).exp());
            }
        }
        let field = Field::new(samples, nx, ny, dx, dy)?;
        Ok(Self { lambda, field })
    }

    /// Operating wavelength in meters.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The underlying spatial grid.
    pub fn field(&self) -> &Field {
        &self.field
    }

    /// Mutable access to the underlying spatial grid.
    pub fn field_mut(&mut self) -> &mut Field {
        &mut self.field
    }

    /// Unwrap into the spatial grid, discarding the band association.
    pub fn into_field(self) -> Field {
        self.field
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lambda() -> f64 {
        500e-9
    }

    #[test]
    fn new_validates_grid() {
        assert_eq!(Field::zeros(0, 4, 1e-3, 1e-3), Err(FieldError::EmptyGrid));
        assert_eq!(
            Field::new(vec![Complex64::ZERO; 5], 2, 2, 1e-3, 1e-3),
            Err(FieldError::DimensionMismatch {
                len: 5,
                expected: 4
            })
        );
        assert_eq!(
            Field::zeros(2, 2, 0.0, 1e-3),
            Err(FieldError::NonPositivePitch { dx: 0.0, dy: 1e-3 })
        );
        assert_eq!(
            Field::new(vec![Complex64::new(f64::NAN, 0.0); 4], 2, 2, 1e-3, 1e-3),
            Err(FieldError::NonFiniteSample)
        );
    }

    #[test]
    fn band_field_validates_wavelength() {
        let grid = Field::zeros(2, 2, 1e-3, 1e-3).unwrap();
        assert_eq!(
            BandField::new(0.0, grid.clone()),
            Err(FieldError::NonPositiveWavelength(0.0))
        );
        // NaN != NaN, so match the variant rather than compare values.
        assert!(matches!(
            BandField::new(f64::NAN, grid.clone()),
            Err(FieldError::NonPositiveWavelength(_))
        ));
        let band = BandField::new(lambda(), grid).unwrap();
        assert_eq!(band.lambda(), lambda());
        assert_eq!(band.field().dims(), (2, 2));
        // Invalid wavelength is caught before grid construction.
        assert_eq!(
            BandField::plane_wave(0.0, 0.0, 1.0, 4, 4, 1e-3, 1e-3, -1.0),
            Err(FieldError::NonPositiveWavelength(-1.0))
        );
    }

    #[test]
    fn row_major_index_mapping() {
        // 3 columns (x) by 2 rows (y); value encodes iy * 10 + ix.
        let samples: Vec<Complex64> = (0..6).map(|n| Complex64::new(n as f64, 0.0)).collect();
        let field = Field::new(samples, 3, 2, 1e-3, 2e-3).unwrap();
        assert_eq!(field.sample(0, 0).re, 0.0); // iy=0 row starts flat index 0
        assert_eq!(field.sample(2, 0).re, 2.0); // last column of row 0
        assert_eq!(field.sample(0, 1).re, 3.0); // iy=1 row starts flat index 3
        assert_eq!(field.sample(2, 1).re, 5.0);
        assert_eq!(field.samples().len(), 6);
    }

    #[test]
    fn coordinates_are_centered_on_origin() {
        let field = Field::zeros(5, 3, 1e-3, 1e-3).unwrap();
        assert_eq!(field.coord(2, 1), (0.0, 0.0)); // odd grid: exact center sample
        assert_eq!(field.coord(0, 0), (-2e-3, -1e-3));
        assert_eq!(field.coord(4, 2), (2e-3, 1e-3));
    }

    #[test]
    fn at_finds_nearest_sample_and_enforces_bounds() {
        let samples: Vec<Complex64> = (0..6).map(|n| Complex64::new(n as f64, 0.0)).collect();
        let field = Field::new(samples, 3, 2, 1e-3, 1e-3).unwrap();
        // Extent: |x| <= 1e-3, |y| <= 0.5e-3.
        assert_eq!(field.at(0.0, 0.0), Some(Complex64::new(4.0, 0.0))); // (ix=2, iy=1)
        assert_eq!(field.at(0.9e-3, 0.4e-3), Some(Complex64::new(5.0, 0.0))); // rounds to (2, 1)
        assert_eq!(field.at(1.1e-3, 0.0), None); // x beyond half-extent
        assert_eq!(field.at(0.0, 0.9e-3), None); // y beyond half-extent
        assert_eq!(field.at(f64::NAN, 0.0), None);
    }

    #[test]
    fn uniform_field_power() {
        // P = A^2 * (nx * ny) * dx * dy for a uniform field of amplitude A.
        let a = 2.0;
        let band = BandField::uniform(a, PI / 3.0, 8, 4, 1e-3, 5e-4, lambda()).unwrap();
        let expected = a * a * 8.0 * 4.0 * 1e-3 * 5e-4;
        assert!((band.field().power() - expected).abs() < 1e-6 * expected);
        for (amp, ph) in band.field().amplitudes().iter().zip(band.field().phases()) {
            assert!((amp - a).abs() < 1e-12);
            assert!((ph - PI / 3.0).abs() < 1e-12);
        }
        assert!(
            band.field()
                .intensities()
                .iter()
                .all(|i| (i - a * a).abs() < 1e-12)
        );
    }

    #[test]
    fn plane_wave_phase_ramp_matches_direction_cosines() {
        let theta_x = 0.05; // rad
        let band =
            BandField::plane_wave(theta_x, 0.0, 1.0, 16, 16, 10e-6, 10e-6, lambda()).unwrap();
        let field = band.field();
        let k = 2.0 * PI / band.lambda();
        let expected_step = (k * theta_x.sin() * 10e-6 * Complex64::I).exp();
        // Ratio of adjacent x samples is the constant phase step e^{i k alpha dx}.
        for iy in [0usize, 7, 15] {
            let ratio = field.sample(8, iy) / field.sample(7, iy);
            assert!((ratio - expected_step).norm() < 1e-9);
        }
        // On-axis (theta = 0) plane wave is spatially constant.
        let flat = BandField::plane_wave(0.0, 0.0, 3.0, 16, 16, 10e-6, 10e-6, lambda()).unwrap();
        assert!(
            flat.field()
                .samples()
                .iter()
                .all(|s| *s == Complex64::new(3.0, 0.0))
        );
    }

    #[test]
    fn grid_geometry_is_band_independent() {
        // The same Field grid can be bound to different wavelengths without
        // mutation — the invariant that enables multiband collections.
        let grid = Field::uniform(1.0, 0.0, 4, 4, 1e-3, 1e-3).unwrap();
        let red = BandField::new(650e-9, grid.clone()).unwrap();
        let blue = BandField::new(450e-9, grid.clone()).unwrap();
        assert_eq!(red.field(), blue.field());
        assert_ne!(red.lambda(), blue.lambda());
        assert_eq!(blue.clone().into_field(), grid);
    }
}
