//! Optical elements: **local** transformations of a sampled [`BandField`].
//!
//! An [`Element`] modifies the complex samples of a [`BandField`] on the same
//! grid — it never resamples, interpolates, or advances the field along $z$.
//! Distance-dependent physics belongs to the propagators in
//! [`crate::propagation`]. In particular, whether the diffraction pattern
//! formed downstream of an aperture of radius $a$ is in the **Fresnel**
//! (near-field) or **Fraunhofer** (far-field) regime is judged by the **Fresnel
//! number** at the distance $z$ to the next element or detector:
//!
//! $$N_F = \frac{a^2}{\lambda z}$$
//!
//! with $N_F \gg 1$ near field and $N_F \lesssim 1$ far field (Fraunhofer).
//! This is a *propagation* decision combining the element's aperture, the
//! wavelength, and the inter-element distance — so elements expose their
//! aperture geometry (e.g. [`CircularAperture::radius`]) for downstream regime
//! checks (Phase 2+ regime guidance). The Phase 1 ASM propagator is valid in
//! both regimes and does not need this number.
//!
//! All spatial quantities are SI units (meters); angles in radians.

use num_complex::Complex64;
use thiserror::Error;

use crate::field::BandField;

/// Errors produced when an element acts on a [`BandField`].
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum OpticsError {
    /// The element's own geometry is not physically meaningful: the radius
    /// must be positive and finite and the center coordinates finite.
    #[error(
        "aperture needs positive finite radius and finite center (got radius = {radius} m, center = ({center_x}, {center_y}) m)"
    )]
    InvalidAperture {
        /// Aperture radius in meters.
        radius: f64,
        /// Aperture center x-coordinate in meters.
        center_x: f64,
        /// Aperture center y-coordinate in meters.
        center_y: f64,
    },
    /// The element's footprint extends beyond the field grid it is applied to;
    /// the sampled plane cannot represent the full interaction.
    #[error(
        "aperture radius {radius} m at ({center_x}, {center_y}) m exceeds the field grid half-extents ({half_x} m x {half_y} m)"
    )]
    OutsideField {
        /// Aperture radius in meters.
        radius: f64,
        /// Aperture center x-coordinate in meters.
        center_x: f64,
        /// Aperture center y-coordinate in meters.
        center_y: f64,
        /// Field grid half-extent along x in meters, $(n_x - 1)\Delta x / 2$.
        half_x: f64,
        /// Field grid half-extent along y in meters, $(n_y - 1)\Delta y / 2$.
        half_y: f64,
    },
}

/// Physical outline of an element's clear aperture.
pub enum ElementShape {
    /// Circular aperture.
    Circle,
    /// Elliptical aperture.
    Ellipse,
    /// Rectangular aperture.
    Rectangle,
    /// Square aperture.
    Square,
    // Polygons are defined by n_lines, and angles between them
    /// Polygonal aperture: `(n_lines, angles)` with the vertex angles in
    /// radians.
    Poly((u8, Vec<f64>)),
}

/// Functional class of an optical element.
pub enum ElementKind {
    /// Binary or graded amplitude stop.
    Aperture,
    /// Refractive or diffractive focusing element.
    Lens,
    /// Reflecting surface.
    Mirror,
    /// Amplitude/phase splitter.
    Splitter,
    /// Emission plane.
    Source,
    /// Reference plane (no physical effect).
    Plane,
    /// Dispersive refractive element.
    Prism,
    /// Flat transmissive window.
    Window,
    /// Polarization-transforming plate.
    Waveplate,
    /// Diffractive optical element surface.
    Diffractive,
    /// Guided-wave structure.
    Waveguide,
}

/// Extent of an element, derived from its kind.
/// Optical elements only care about their clear aperture;
/// refractive elements additionally carry a central thickness.
pub enum Extent {
    /// Flat elements: clear-aperture radius (m).
    Aperture(f64),
    /// Refractive elements: clear-aperture radius and central thickness (m).
    Thick {
        /// Clear-aperture radius in meters.
        aperture: f64,
        /// Central thickness in meters.
        thickness: f64,
    },
}

/// All units are SI: wavelength in meters, angles in radians.
pub struct Material {
    /// Index of refraction as a function of wavelength (m) and incidence angle (rad).
    pub n: Box<dyn Fn(f64, f64) -> f64>,
    /// Absorption coefficient (1/m) as a function of wavelength (m) and incidence angle (rad).
    pub absorption: Box<dyn Fn(f64, f64) -> f64>,
}

impl Material {
    /// Builds a material from index and absorption functions of
    /// (wavelength in m, incidence angle in rad).
    pub fn new(
        n: impl Fn(f64, f64) -> f64 + 'static,
        absorption: impl Fn(f64, f64) -> f64 + 'static,
    ) -> Self {
        Self {
            n: Box::new(n),
            absorption: Box::new(absorption),
        }
    }

    /// Index of refraction at wavelength `lambda` (m) and angle `angle` (rad).
    pub fn n_at(&self, lambda: f64, angle: f64) -> f64 {
        (self.n)(lambda, angle)
    }

    /// Absorption coefficient at wavelength `lambda` (m) and angle `angle` (rad).
    pub fn absorption_at(&self, lambda: f64, angle: f64) -> f64 {
        (self.absorption)(lambda, angle)
    }
}

/// A wave-optical element acting on a monochromatic band field.
///
/// The interaction is local to the sampled plane: `apply` returns a field with
/// identical grid dimensions, pitch, and wavelength. Propagation to the next
/// plane (and the Fresnel-vs-Fraunhofer regime judgment via $N_F = a^2/(\lambda z)$,
/// see the module docs) is the propagators' responsibility.
pub trait Element {
    /// Aperture outline of the element.
    fn shape(&self) -> &ElementShape;
    /// Functional class of the element.
    fn kind(&self) -> ElementKind;
    /// Front and back interface materials.
    // front and back
    fn material(&self) -> (&Material, &Material);
    /// Clear-aperture extent (and thickness for refractive elements).
    // Extent is derived from the kind and owns the correct extent object
    fn extent(&self) -> Extent;
    /// Applies the element to `band` and returns the field immediately after
    /// it (same grid geometry and wavelength).
    ///
    /// # Errors
    /// [`OpticsError::OutsideField`] when the element footprint exceeds the
    /// field grid extent — a partial interaction the sampled plane cannot
    /// represent is rejected rather than silently truncated.
    fn apply(&self, band: &BandField) -> Result<BandField, OpticsError>;
}

/// Ideal hard-edge circular aperture: a binary amplitude mask
///
/// $$t(x, y) = \begin{cases} 1 & (x - x_c)^2 + (y - y_c)^2 \le a^2 \\ 0 &
/// \text{otherwise} \end{cases}$$
///
/// with radius $a$ (m) and center $(x_c, y_c)$ (m) in field coordinates
/// (origin at the grid center, see [`crate::field`]). A sample whose
/// coordinate lies exactly on the circle boundary is **inside** (transmitted).
///
/// # Regime note (Fresnel number)
///
/// The pattern formed downstream depends on the distance $z$ to the next
/// element or detector through the Fresnel number $N_F = a^2 / (\lambda z)$:
/// $N_F \gg 1$ → Fresnel near field (Fresnel number of half-period zones
/// across the aperture, cf. textbook Sec. 1.4); $N_F \lesssim 1$ → Fraunhofer
/// far field (Fourier-transform of the aperture, Airy profile). Phase 1 ASM
/// propagation is exact in both regimes; Phase 2+ propagator selection and
/// regime guidance consume [`CircularAperture::radius`], $\lambda$, and $z$.
// No `Clone`/`Debug`/`PartialEq` derives: the `Material` field holds boxed
// closures; element identity in Phase 1 is by value of its geometry accessors.
pub struct CircularAperture {
    radius: f64,        // meters
    center: (f64, f64), // meters, field coordinates
    shape: ElementShape,
    material: Material,
}

impl CircularAperture {
    /// Ideal hard-edge aperture of radius `radius` (m) centered at `center`
    /// (m, m) in field coordinates.
    ///
    /// The stored `material` is metadata for future refractive/absorptive
    /// element models; the Phase-1 hard-edge transmission model is the binary
    /// mask alone.
    ///
    /// # Errors
    /// [`OpticsError::InvalidAperture`] if `radius` is not positive and finite,
    /// or either center coordinate is not finite.
    pub fn new(radius: f64, center: (f64, f64)) -> Result<Self, OpticsError> {
        let (cx, cy) = center;
        if !radius.is_finite() || radius <= 0.0 || !cx.is_finite() || !cy.is_finite() {
            return Err(OpticsError::InvalidAperture {
                radius,
                center_x: cx,
                center_y: cy,
            });
        }
        Ok(Self {
            radius,
            center,
            shape: ElementShape::Circle,
            // Non-absorbing, index-1 placeholder: unused by the hard-edge mask.
            material: Material::new(|_lambda, _angle| 1.0, |_lambda, _angle| 0.0),
        })
    }

    /// Aperture radius $a$ in meters.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Aperture center $(x_c, y_c)$ in meters (field coordinates).
    pub fn center(&self) -> (f64, f64) {
        self.center
    }

    /// Half-extents $((n_x - 1)\Delta x / 2, (n_y - 1)\Delta y / 2)$ of a
    /// grid's physical coverage in meters.
    fn half_extents(band: &BandField) -> (f64, f64) {
        let (nx, ny) = band.field().dims();
        let (dx, dy) = band.field().pitch();
        ((nx as f64 - 1.0) / 2.0 * dx, (ny as f64 - 1.0) / 2.0 * dy)
    }
}

impl Element for CircularAperture {
    fn shape(&self) -> &ElementShape {
        &self.shape
    }

    fn kind(&self) -> ElementKind {
        ElementKind::Aperture
    }

    fn material(&self) -> (&Material, &Material) {
        (&self.material, &self.material)
    }

    fn extent(&self) -> Extent {
        Extent::Aperture(self.radius)
    }

    /// Masks the field with the hard-edge transmission function: samples whose
    /// physical coordinates fall outside the circle are set to zero, all
    /// others (including the boundary) are passed through unchanged.
    ///
    /// # Errors
    /// [`OpticsError::OutsideField`] if the aperture bounding box
    /// $|x_c| + a >$ half-extent$_x$ or $|y_c| + a >$ half-extent$_y$ does not
    /// fit inside the grid.
    fn apply(&self, band: &BandField) -> Result<BandField, OpticsError> {
        let (cx, cy) = self.center;
        let (half_x, half_y) = Self::half_extents(band);
        if cx.abs() + self.radius > half_x || cy.abs() + self.radius > half_y {
            return Err(OpticsError::OutsideField {
                radius: self.radius,
                center_x: cx,
                center_y: cy,
                half_x,
                half_y,
            });
        }
        let mut out = band.clone();
        let field = out.field_mut();
        let (nx, ny) = field.dims();
        let r2 = self.radius * self.radius;
        for iy in 0..ny {
            for ix in 0..nx {
                let (x, y) = field.coord(ix, iy);
                let dxc = x - cx;
                let dyc = y - cy;
                if dxc * dxc + dyc * dyc > r2 {
                    *field.sample_mut(ix, iy) = Complex64::ZERO;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;
    use std::f64::consts::PI;

    fn lambda() -> f64 {
        500e-9
    }

    // 101 x 101 grid, 1 mm pitch, half-extents exactly 50 mm-equivalents:
    // (101 - 1)/2 * 1e-3 = 50.0 * 1e-3.
    fn band_101() -> BandField {
        BandField::uniform(1.0, PI / 3.0, 101, 101, 1e-3, 1e-3, lambda()).unwrap()
    }

    fn r50() -> f64 {
        50.0 * 1e-3
    }

    #[test]
    fn constructor_validates_geometry() {
        assert!(matches!(
            CircularAperture::new(0.0, (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { radius: 0.0, .. })
        ));
        assert!(matches!(
            CircularAperture::new(f64::NAN, (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        assert!(matches!(
            CircularAperture::new(1.0, (f64::INFINITY, 0.0)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        let ap = CircularAperture::new(2.5e-3, (1e-3, -1e-3)).unwrap();
        assert_eq!(ap.radius(), 2.5e-3);
        assert_eq!(ap.center(), (1e-3, -1e-3));
    }

    #[test]
    fn aperture_exceeding_grid_extent_is_rejected() {
        let band = band_101();
        let big = CircularAperture::new(55e-3, (0.0, 0.0)).unwrap();
        match big.apply(&band) {
            Err(OpticsError::OutsideField {
                radius,
                center_x,
                center_y,
                half_x,
                half_y,
            }) => {
                assert_eq!(radius, 55e-3);
                assert_eq!((center_x, center_y), (0.0, 0.0));
                assert_eq!(half_x, r50());
                assert_eq!(half_y, r50());
            }
            other => panic!("expected OutsideField, got {other:?}"),
        }
        // Off-center: radius 45 mm at x = 10 mm reaches 55 mm > 50 mm half-extent.
        let offset = CircularAperture::new(45e-3, (10e-3, 0.0)).unwrap();
        assert!(matches!(
            offset.apply(&band),
            Err(OpticsError::OutsideField { .. })
        ));
        // Exactly filling the grid fits (boundary inclusive).
        assert!(
            CircularAperture::new(r50(), (0.0, 0.0))
                .unwrap()
                .apply(&band)
                .is_ok()
        );
    }

    #[test]
    fn mask_geometry_includes_boundary() {
        let band = band_101();
        let ap = CircularAperture::new(r50(), (0.0, 0.0)).unwrap();
        let out = ap.apply(&band).unwrap();
        let f = out.field();
        let pass = band.field().sample(50, 50);
        // Center passes; exact boundary sample (x = 50.0e-3, y = 0) passes.
        assert_eq!(f.sample(50, 50), pass);
        assert_eq!(f.sample(100, 50), pass); // (ix=100, iy=50): x = r, on circle
        assert_eq!(f.sample(50, 100), pass); // y = r, on circle
        // Just outside: (x = r, y = 1e-3) is off the circle.
        assert_eq!(f.sample(100, 51), Complex64::ZERO);
        assert_eq!(f.sample(99, 100), Complex64::ZERO); // x = 49e-3, y = 50e-3
        assert_eq!(f.sample(100, 100), Complex64::ZERO); // corner
        // Wavelength and grid geometry are preserved by the interaction.
        assert_eq!(out.lambda(), band.lambda());
        assert_eq!(f.dims(), band.field().dims());
        assert_eq!(f.pitch(), band.field().pitch());
    }

    #[test]
    fn power_reduction_matches_area_ratio() {
        // Uniform input: output power / input power -> pi a^2 / (W * H).
        let n = 512usize;
        let d = 1e-6;
        let band = BandField::uniform(2.0, -PI / 4.0, n, n, d, d, lambda()).unwrap();
        let a = 100e-6;
        let ap = CircularAperture::new(a, (0.0, 0.0)).unwrap();
        let out = ap.apply(&band).unwrap();
        let ratio = out.field().power() / band.field().power();
        let expected = PI * a * a / ((n as f64 * d) * (n as f64 * d));
        // Pixel-count discretization error is O(perimeter * d^2 / area) ~ d/a.
        let rel = ((ratio - expected) / expected).abs();
        assert!(rel < 0.03, "relative area error {rel} too large");
        // Amplitude and phase inside the aperture are untouched.
        assert_eq!(
            out.field().sample(n / 2, n / 2),
            band.field().sample(n / 2, n / 2)
        );
    }

    #[test]
    fn applies_to_nonuniform_fields_by_masking() {
        // A tilted plane wave keeps its phase ramp inside the circle.
        let band = BandField::plane_wave(0.02, 0.0, 1.0, 101, 101, 1e-3, 1e-3, lambda()).unwrap();
        let ap = CircularAperture::new(20e-3, (0.0, 0.0)).unwrap();
        let out = ap.apply(&band).unwrap();
        for iy in 0..101 {
            for ix in 0..101 {
                let (x, y) = band.field().coord(ix, iy);
                let inside = x * x + y * y <= (20e-3) * (20e-3);
                let s = out.field().sample(ix, iy);
                if inside {
                    assert_eq!(s, band.field().sample(ix, iy));
                } else {
                    assert_eq!(s, Complex64::ZERO);
                }
            }
        }
    }

    #[test]
    fn trait_object_dispatch() {
        let ap = CircularAperture::new(20e-3, (0.0, 0.0)).unwrap();
        let el: &dyn Element = &ap;
        assert!(matches!(el.shape(), ElementShape::Circle));
        assert!(matches!(el.kind(), ElementKind::Aperture));
        assert!(matches!(
            el.extent(),
            Extent::Aperture(r) if (r - 20e-3).abs() < 1e-18
        ));
        let (front, back) = el.material();
        assert_eq!(front.n_at(lambda(), 0.0), 1.0);
        assert_eq!(back.absorption_at(lambda(), 0.0), 0.0);
        let band = band_101();
        let out = el.apply(&band).unwrap();
        assert_eq!(out.field().sample(50, 50), band.field().sample(50, 50));
        assert_eq!(out.field().sample(100, 100), Complex64::ZERO);
    }

    #[test]
    fn field_grid_helpers_still_valid() {
        // Sanity: coord() used by the mask agrees with manual grid math.
        let field = Field::zeros(101, 101, 1e-3, 1e-3).unwrap();
        assert_eq!(field.coord(100, 50), (r50(), 0.0));
        assert_eq!(field.coord(0, 0), (-r50(), -r50()));
    }
}
