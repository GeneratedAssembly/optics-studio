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
//! aperture geometry (e.g. [`EllipticalAperture::radius`] or
//! [`EllipticalAperture::semi_axes`]) for downstream regime
//! checks (Phase 2+ regime guidance). The Phase 1 ASM propagator is valid in
//! both regimes and does not need this number.
//!
//! All spatial quantities are SI units (meters); angles in radians.

use num_complex::Complex64;
use thiserror::Error;

use crate::field::BandField;

/// Errors produced when an element acts on a [`BandField`].
///
/// Aperture validity is stated in **footprint half-extents** `(half_x,
/// half_y)`, shared by every aperture shape: a circle of radius $r$ fills
/// $(r, r)$, an ellipse its semi-axes, a rectangle its half-widths.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum OpticsError {
    /// The element's own geometry is not physically meaningful: both
    /// footprint half-extents must be positive and finite and the center
    /// coordinates finite.
    #[error(
        "aperture needs positive finite half-extents and finite center (got half_x = {half_x} m, half_y = {half_y} m, center = ({center_x}, {center_y}) m)"
    )]
    InvalidAperture {
        /// Footprint half-extent along x in meters.
        half_x: f64,
        /// Footprint half-extent along y in meters.
        half_y: f64,
        /// Aperture center x-coordinate in meters.
        center_x: f64,
        /// Aperture center y-coordinate in meters.
        center_y: f64,
    },
    /// The element's footprint extends beyond the field grid it is applied to;
    /// the sampled plane cannot represent the full interaction.
    #[error(
        "aperture half-extents ({half_x} m x {half_y} m) at ({center_x}, {center_y}) m exceed the field grid half-extents ({grid_half_x} m x {grid_half_y} m)"
    )]
    OutsideField {
        /// Footprint half-extent along x in meters.
        half_x: f64,
        /// Footprint half-extent along y in meters.
        half_y: f64,
        /// Aperture center x-coordinate in meters.
        center_x: f64,
        /// Aperture center y-coordinate in meters.
        center_y: f64,
        /// Field grid half-extent along x in meters, $(n_x - 1)\Delta x / 2$.
        grid_half_x: f64,
        /// Field grid half-extent along y in meters, $(n_y - 1)\Delta y / 2$.
        grid_half_y: f64,
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

/// Field-grid half-extents $((n_x - 1)\Delta x / 2, (n_y - 1)\Delta y / 2)$
/// in meters: the physical coverage of the sampled plane.
fn grid_half_extents(band: &BandField) -> (f64, f64) {
    let (nx, ny) = band.field().dims();
    let (dx, dy) = band.field().pitch();
    ((nx as f64 - 1.0) / 2.0 * dx, (ny as f64 - 1.0) / 2.0 * dy)
}

/// Validates aperture footprint half-extents and center (shared by all
/// aperture shapes): both half-extents positive and finite, both center
/// coordinates finite.
fn validate_footprint(half_x: f64, half_y: f64, center: (f64, f64)) -> Result<(), OpticsError> {
    let (cx, cy) = center;
    if !half_x.is_finite()
        || half_x <= 0.0
        || !half_y.is_finite()
        || half_y <= 0.0
        || !cx.is_finite()
        || !cy.is_finite()
    {
        return Err(OpticsError::InvalidAperture {
            half_x,
            half_y,
            center_x: cx,
            center_y: cy,
        });
    }
    Ok(())
}

/// Ideal hard-edge elliptical aperture: a binary amplitude mask
///
/// $$t(x, y) = \begin{cases} 1 & \displaystyle\frac{(x - x_c)^2}{a^2} +
/// \frac{(y - y_c)^2}{b^2} \le 1 \\ 0 & \text{otherwise} \end{cases}$$
///
/// with semi-axes $(a, b)$ (m) along $x$, $y$ and center $(x_c, y_c)$ (m) in
/// field coordinates (origin at the grid center, see [`crate::field`]). A
/// sample whose coordinate lies exactly on the ellipse boundary is **inside**
/// (transmitted).
///
/// The **circle is the axis-symmetric special case**: construct it with
/// [`EllipticalAperture::circle`] (or `new` with equal semi-axes), which
/// reports [`ElementShape::Circle`] and exposes [`EllipticalAperture::radius`]
/// (approved task-b decision 2026-10-12: one aperture type, no separate
/// circular element).
///
/// # Regime note (Fresnel number)
///
/// The pattern formed downstream depends on the distance $z$ to the next
/// element or detector through the Fresnel number $N_F = a^2 / (\lambda z)$:
/// $N_F \gg 1$ → Fresnel near field (Fresnel number of half-period zones
/// across the aperture, cf. textbook Sec. 1.4); $N_F \lesssim 1$ → Fraunhofer
/// far field (Fourier-transform of the aperture — a circle gives the Airy
/// profile $[2J_1(v)/v]^2$, an ellipse its axis-scaled version). Phase 1 ASM
/// propagation is exact in both regimes; Phase 2+ propagator selection and
/// regime guidance consume the aperture half-extent (for a circle,
/// [`EllipticalAperture::radius`]; for a general ellipse, the conservative
/// maximum from [`Element::extent`]), $\lambda$, and $z$.
// No `Clone`/`Debug`/`PartialEq` derives: the `Material` field holds boxed
// closures; element identity is by value of its geometry accessors.
pub struct EllipticalAperture {
    semi_axes: (f64, f64), // meters: (a, b)
    center: (f64, f64),    // meters, field coordinates
    shape: ElementShape,
    material: Material,
}

impl EllipticalAperture {
    /// Ideal hard-edge ellipse with semi-axes `semi_axes` $= (a, b)$ (m) and
    /// center `center` (m, m) in field coordinates.
    ///
    /// Equal semi-axes ($a == b$ exactly) report [`ElementShape::Circle`].
    ///
    /// The stored `material` is metadata for future refractive/absorptive
    /// element models; the hard-edge transmission model is the binary mask
    /// alone.
    ///
    /// # Errors
    /// [`OpticsError::InvalidAperture`] if either semi-axis is not positive
    /// and finite, or either center coordinate is not finite.
    pub fn new(semi_axes: (f64, f64), center: (f64, f64)) -> Result<Self, OpticsError> {
        let (a, b) = semi_axes;
        validate_footprint(a, b, center)?;
        let shape = if a == b {
            ElementShape::Circle
        } else {
            ElementShape::Ellipse
        };
        Ok(Self {
            semi_axes,
            center,
            shape,
            // Non-absorbing, index-1 placeholder: unused by the hard-edge mask.
            material: Material::new(|_lambda, _angle| 1.0, |_lambda, _angle| 0.0),
        })
    }

    /// Ideal hard-edge **circle** of radius `radius` (m) centered at `center`
    /// (m, m) — the axis-symmetric special case of [`EllipticalAperture::new`]
    /// with semi-axes $(r, r)$.
    ///
    /// # Errors
    /// [`OpticsError::InvalidAperture`] if `radius` is not positive and
    /// finite, or either center coordinate is not finite.
    pub fn circle(radius: f64, center: (f64, f64)) -> Result<Self, OpticsError> {
        Self::new((radius, radius), center)
    }

    /// Semi-axes $(a, b)$ in meters (along $x$, $y$).
    pub fn semi_axes(&self) -> (f64, f64) {
        self.semi_axes
    }

    /// Aperture center $(x_c, y_c)$ in meters (field coordinates).
    pub fn center(&self) -> (f64, f64) {
        self.center
    }

    /// Circle radius in meters — `Some(r)` for the axis-symmetric special
    /// case ($a == b$, including everything built with
    /// [`EllipticalAperture::circle`]), `None` for a genuine ellipse.
    /// Pairs with `FraunhoferConfig::aperture_extent` and
    /// `propagation::regime::fresnel_number` for circular sources.
    pub fn radius(&self) -> Option<f64> {
        match self.shape {
            ElementShape::Circle => Some(self.semi_axes.0),
            _ => None,
        }
    }
}

impl Element for EllipticalAperture {
    fn shape(&self) -> &ElementShape {
        &self.shape
    }

    fn kind(&self) -> ElementKind {
        ElementKind::Aperture
    }

    fn material(&self) -> (&Material, &Material) {
        (&self.material, &self.material)
    }

    /// Conservative single-number extent for regime gating: the **larger**
    /// semi-axis $\max(a, b)$ — the worst-case half-width for the Fresnel
    /// number $N_F = a^2/(\lambda z)$.
    fn extent(&self) -> Extent {
        Extent::Aperture(self.semi_axes.0.max(self.semi_axes.1))
    }

    /// Masks the field with the hard-edge transmission function: samples whose
    /// physical coordinates fall outside the ellipse are set to zero, all
    /// others (including the boundary) are passed through unchanged.
    ///
    /// The boundary test uses the axis-multiplied form
    /// $b^2\,\Delta x^2 + a^2\,\Delta y^2 > a^2 b^2$ (no divisions, exactly
    /// the ellipse inequality).
    ///
    /// # Errors
    /// [`OpticsError::OutsideField`] if the aperture bounding box
    /// $|x_c| + a >$ half-extent$_x$ or $|y_c| + b >$ half-extent$_y$ does not
    /// fit inside the grid.
    fn apply(&self, band: &BandField) -> Result<BandField, OpticsError> {
        let (a, b) = self.semi_axes;
        let (cx, cy) = self.center;
        let (grid_half_x, grid_half_y) = grid_half_extents(band);
        if cx.abs() + a > grid_half_x || cy.abs() + b > grid_half_y {
            return Err(OpticsError::OutsideField {
                half_x: a,
                half_y: b,
                center_x: cx,
                center_y: cy,
                grid_half_x,
                grid_half_y,
            });
        }
        let mut out = band.clone();
        let field = out.field_mut();
        let (nx, ny) = field.dims();
        let (a2, b2) = (a * a, b * b);
        let limit = a2 * b2;
        for iy in 0..ny {
            for ix in 0..nx {
                let (x, y) = field.coord(ix, iy);
                let dxc = x - cx;
                let dyc = y - cy;
                if b2 * dxc * dxc + a2 * dyc * dyc > limit {
                    *field.sample_mut(ix, iy) = Complex64::ZERO;
                }
            }
        }
        Ok(out)
    }
}

/// Ideal hard-edge rectangular aperture: a binary amplitude mask
///
/// $$t(x, y) = \begin{cases} 1 & |x - x_c| \le a \;\text{and}\; |y - y_c| \le b
/// \\ 0 & \text{otherwise} \end{cases}$$
///
/// with half-widths $(a, b)$ (m) along $x$, $y$ and center $(x_c, y_c)$ (m)
/// in field coordinates (origin at the grid center, see [`crate::field`]).
/// Samples whose coordinates lie exactly on an edge are **inside**
/// (transmitted). The full opening is $2a \times 2b$; a square is the
/// $a == b$ special case (still reported as [`ElementShape::Rectangle`]).
///
/// # Regime note (Fresnel number)
///
/// Downstream of the rectangle the far field is the separable
/// $\operatorname{sinc}^2$ pattern of the two slit edges (textbook Sec. 2.2,
/// Eq. 2.19: far field = Fourier transform of the aperture; the
/// rectangular-sinc² profile is validated in `tests/phase2.rs`). Regime
/// guidance consumes the conservative half-extent from [`Element::extent`]
/// ($\max(a, b)$), $\lambda$, and $z$.
// No `Clone`/`Debug`/`PartialEq` derives: the `Material` field holds boxed
// closures; element identity is by value of its geometry accessors.
pub struct RectangularAperture {
    half_width: (f64, f64), // meters: (a, b)
    center: (f64, f64),     // meters, field coordinates
    shape: ElementShape,
    material: Material,
}

impl RectangularAperture {
    /// Ideal hard-edge rectangle with half-widths `half_width` $= (a, b)$
    /// (m) and center `center` (m, m) in field coordinates.
    ///
    /// The stored `material` is metadata for future refractive/absorptive
    /// element models; the hard-edge transmission model is the binary mask
    /// alone.
    ///
    /// # Errors
    /// [`OpticsError::InvalidAperture`] if either half-width is not positive
    /// and finite, or either center coordinate is not finite.
    pub fn new(half_width: (f64, f64), center: (f64, f64)) -> Result<Self, OpticsError> {
        let (a, b) = half_width;
        validate_footprint(a, b, center)?;
        Ok(Self {
            half_width,
            center,
            shape: ElementShape::Rectangle,
            // Non-absorbing, index-1 placeholder: unused by the hard-edge mask.
            material: Material::new(|_lambda, _angle| 1.0, |_lambda, _angle| 0.0),
        })
    }

    /// Half-widths $(a, b)$ in meters (along $x$, $y$); full opening
    /// $2a \times 2b$.
    pub fn half_width(&self) -> (f64, f64) {
        self.half_width
    }

    /// Aperture center $(x_c, y_c)$ in meters (field coordinates).
    pub fn center(&self) -> (f64, f64) {
        self.center
    }
}

impl Element for RectangularAperture {
    fn shape(&self) -> &ElementShape {
        &self.shape
    }

    fn kind(&self) -> ElementKind {
        ElementKind::Aperture
    }

    fn material(&self) -> (&Material, &Material) {
        (&self.material, &self.material)
    }

    /// Conservative single-number extent for regime gating: the **larger**
    /// half-width $\max(a, b)$ — the worst-case half-width for the Fresnel
    /// number $N_F = a^2/(\lambda z)$.
    fn extent(&self) -> Extent {
        Extent::Aperture(self.half_width.0.max(self.half_width.1))
    }

    /// Masks the field with the hard-edge transmission function: samples
    /// outside the axis-aligned box are set to zero, all others (including
    /// the edges) are passed through unchanged.
    ///
    /// # Errors
    /// [`OpticsError::OutsideField`] if the rectangle footprint
    /// $|x_c| + a >$ half-extent$_x$ or $|y_c| + b >$ half-extent$_y$ does not
    /// fit inside the grid.
    fn apply(&self, band: &BandField) -> Result<BandField, OpticsError> {
        let (a, b) = self.half_width;
        let (cx, cy) = self.center;
        let (grid_half_x, grid_half_y) = grid_half_extents(band);
        if cx.abs() + a > grid_half_x || cy.abs() + b > grid_half_y {
            return Err(OpticsError::OutsideField {
                half_x: a,
                half_y: b,
                center_x: cx,
                center_y: cy,
                grid_half_x,
                grid_half_y,
            });
        }
        let mut out = band.clone();
        let field = out.field_mut();
        let (nx, ny) = field.dims();
        for iy in 0..ny {
            for ix in 0..nx {
                let (x, y) = field.coord(ix, iy);
                if (x - cx).abs() > a || (y - cy).abs() > b {
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
    fn elliptical_constructor_validates_geometry() {
        assert!(matches!(
            EllipticalAperture::new((0.0, 1e-3), (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { half_x: 0.0, .. })
        ));
        assert!(matches!(
            EllipticalAperture::new((2e-3, -1e-3), (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { half_y, .. }) if half_y == -1e-3
        ));
        assert!(matches!(
            EllipticalAperture::new((f64::NAN, 1e-3), (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        assert!(matches!(
            EllipticalAperture::new((1e-3, 1e-3), (f64::INFINITY, 0.0)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        assert!(matches!(
            EllipticalAperture::circle(0.0, (0.0, 0.0)),
            Err(OpticsError::InvalidAperture {
                half_x: 0.0,
                half_y: 0.0,
                ..
            })
        ));
        let ap = EllipticalAperture::new((2.5e-3, 1.5e-3), (1e-3, -1e-3)).unwrap();
        assert_eq!(ap.semi_axes(), (2.5e-3, 1.5e-3));
        assert_eq!(ap.center(), (1e-3, -1e-3));
        assert!(ap.radius().is_none());
    }

    #[test]
    fn circle_is_the_axis_symmetric_special_case() {
        let ap = EllipticalAperture::circle(2.5e-3, (1e-3, -1e-3)).unwrap();
        assert_eq!(ap.semi_axes(), (2.5e-3, 2.5e-3));
        assert_eq!(ap.radius(), Some(2.5e-3));
        assert!(matches!(ap.shape(), ElementShape::Circle));
        // Equal semi-axes through `new` are the same circle.
        let same = EllipticalAperture::new((2.5e-3, 2.5e-3), (1e-3, -1e-3)).unwrap();
        assert!(matches!(same.shape(), ElementShape::Circle));
        assert_eq!(same.radius(), ap.radius());
    }

    #[test]
    fn rectangular_constructor_validates_geometry() {
        assert!(matches!(
            RectangularAperture::new((0.0, 1e-3), (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { half_x: 0.0, .. })
        ));
        assert!(matches!(
            RectangularAperture::new((1e-3, f64::NAN), (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        assert!(matches!(
            RectangularAperture::new((1e-3, 1e-3), (0.0, f64::NAN)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        let ap = RectangularAperture::new((2.5e-3, 1.5e-3), (1e-3, -1e-3)).unwrap();
        assert_eq!(ap.half_width(), (2.5e-3, 1.5e-3));
        assert_eq!(ap.center(), (1e-3, -1e-3));
    }

    #[test]
    fn aperture_exceeding_grid_extent_is_rejected() {
        let band = band_101();
        let big = EllipticalAperture::new((55e-3, 10e-3), (0.0, 0.0)).unwrap();
        match big.apply(&band) {
            Err(OpticsError::OutsideField {
                half_x,
                half_y,
                center_x,
                center_y,
                grid_half_x,
                grid_half_y,
            }) => {
                assert_eq!((half_x, half_y), (55e-3, 10e-3));
                assert_eq!((center_x, center_y), (0.0, 0.0));
                assert_eq!((grid_half_x, grid_half_y), (r50(), r50()));
            }
            other => panic!("expected OutsideField, got {other:?}"),
        }
        // Off-center: semi-axis 45 mm at x = 10 mm reaches 55 mm > 50 mm.
        let offset = EllipticalAperture::circle(45e-3, (10e-3, 0.0)).unwrap();
        assert!(matches!(
            offset.apply(&band),
            Err(OpticsError::OutsideField { .. })
        ));
        // Exactly filling the grid fits (boundary inclusive).
        assert!(
            EllipticalAperture::new((r50(), r50()), (0.0, 0.0))
                .unwrap()
                .apply(&band)
                .is_ok()
        );
        // Rectangle: same bounding-box rule.
        let rect_big = RectangularAperture::new((10e-3, 55e-3), (0.0, 0.0)).unwrap();
        assert!(matches!(
            rect_big.apply(&band),
            Err(OpticsError::OutsideField { half_y: 55e-3, .. })
        ));
        assert!(
            RectangularAperture::new((r50(), r50()), (0.0, 0.0))
                .unwrap()
                .apply(&band)
                .is_ok()
        );
    }

    #[test]
    fn elliptical_mask_geometry_includes_boundary() {
        let band = band_101();
        // Ellipse a = 50 mm (fills x), b = 30 mm.
        let ap = EllipticalAperture::new((r50(), 30e-3), (0.0, 0.0)).unwrap();
        let out = ap.apply(&band).unwrap();
        let f = out.field();
        let pass = band.field().sample(50, 50);
        // Center passes; exact boundary samples pass.
        assert_eq!(f.sample(50, 50), pass);
        assert_eq!(f.sample(100, 50), pass); // (x = a, y = 0) on ellipse
        assert_eq!(f.sample(50, 80), pass); // (x = 0, y = b) on ellipse
        // Just outside: on the bounding box corner and one step off the tips.
        assert_eq!(f.sample(100, 80), Complex64::ZERO); // x = a, y = b: outside
        assert_eq!(f.sample(100, 51), Complex64::ZERO); // x = a, y = 1 mm
        assert_eq!(f.sample(100, 100), Complex64::ZERO); // grid corner
        // Wavelength and grid geometry are preserved by the interaction.
        assert_eq!(out.lambda(), band.lambda());
        assert_eq!(f.dims(), band.field().dims());
        assert_eq!(f.pitch(), band.field().pitch());
    }

    #[test]
    fn rectangular_mask_geometry_includes_boundary() {
        let band = band_101();
        let ap = RectangularAperture::new((r50(), 30e-3), (0.0, 0.0)).unwrap();
        let out = ap.apply(&band).unwrap();
        let f = out.field();
        let pass = band.field().sample(50, 50);
        // Center, edges, and corners of the box all pass (inclusive).
        assert_eq!(f.sample(50, 50), pass);
        assert_eq!(f.sample(100, 50), pass); // x = +a edge
        assert_eq!(f.sample(50, 80), pass); // y = +b edge
        assert_eq!(f.sample(100, 80), pass); // box corner (x = a, y = b)
        // Just outside the edges.
        assert_eq!(f.sample(100, 81), Complex64::ZERO); // y = 31 mm > b
        assert_eq!(f.sample(0, 50), pass); // x = -a edge: boundary inclusive
        assert_eq!(f.sample(0, 0), Complex64::ZERO); // grid corner, outside
        // Wavelength and grid geometry are preserved.
        assert_eq!(out.lambda(), band.lambda());
        assert_eq!(f.dims(), band.field().dims());
        assert_eq!(f.pitch(), band.field().pitch());
    }

    #[test]
    fn power_reduction_matches_area_ratio() {
        // Uniform input: output power / input power -> aperture area / W * H.
        let n = 512usize;
        let d = 1e-6;
        let band = BandField::uniform(2.0, -PI / 4.0, n, n, d, d, lambda()).unwrap();
        let (a, b) = (100e-6, 60e-6);
        let window = (n as f64 * d) * (n as f64 * d);

        let ellipse = EllipticalAperture::new((a, b), (0.0, 0.0)).unwrap();
        let ratio = ellipse.apply(&band).unwrap().field().power() / band.field().power();
        let expected = PI * a * b / window;
        // Pixel-count discretization error is O(perimeter * d^2 / area) ~ d/a.
        let rel = ((ratio - expected) / expected).abs();
        assert!(rel < 0.03, "ellipse area error {rel} too large");

        let rect = RectangularAperture::new((a, b), (0.0, 0.0)).unwrap();
        let ratio = rect.apply(&band).unwrap().field().power() / band.field().power();
        let expected = 4.0 * a * b / window;
        let rel = ((ratio - expected) / expected).abs();
        assert!(rel < 0.03, "rectangle area error {rel} too large");

        // Amplitude and phase inside the apertures are untouched.
        assert_eq!(
            ellipse.apply(&band).unwrap().field().sample(n / 2, n / 2),
            band.field().sample(n / 2, n / 2)
        );
    }

    #[test]
    fn applies_to_nonuniform_fields_by_masking() {
        // A tilted plane wave keeps its phase ramp inside the rectangle.
        let band = BandField::plane_wave(0.02, 0.0, 1.0, 101, 101, 1e-3, 1e-3, lambda()).unwrap();
        let ap = RectangularAperture::new((20e-3, 10e-3), (0.0, 0.0)).unwrap();
        let out = ap.apply(&band).unwrap();
        for iy in 0..101 {
            for ix in 0..101 {
                let (x, y) = band.field().coord(ix, iy);
                let inside = x.abs() <= 20e-3 && y.abs() <= 10e-3;
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
        let ell: &dyn Element = &EllipticalAperture::new((3e-3, 20e-3), (0.0, 0.0)).unwrap(); // a < b
        assert!(matches!(ell.shape(), ElementShape::Ellipse));
        assert!(matches!(ell.kind(), ElementKind::Aperture));
        // Conservative extent: the larger semi-axis.
        assert!(matches!(
            ell.extent(),
            Extent::Aperture(r) if (r - 20e-3).abs() < 1e-18
        ));
        let rect: &dyn Element = &RectangularAperture::new((20e-3, 3e-3), (0.0, 0.0)).unwrap();
        assert!(matches!(rect.shape(), ElementShape::Rectangle));
        assert!(matches!(
            rect.extent(),
            Extent::Aperture(r) if (r - 20e-3).abs() < 1e-18
        ));
        let (front, back) = rect.material();
        assert_eq!(front.n_at(lambda(), 0.0), 1.0);
        assert_eq!(back.absorption_at(lambda(), 0.0), 0.0);
        let band = band_101();
        let out = ell.apply(&band).unwrap();
        assert_eq!(out.field().sample(50, 50), band.field().sample(50, 50));
        assert_eq!(out.field().sample(100, 100), Complex64::ZERO);
    }

    #[test]
    fn field_grid_helpers_still_valid() {
        // Sanity: coord() used by the masks agrees with manual grid math.
        let field = Field::zeros(101, 101, 1e-3, 1e-3).unwrap();
        assert_eq!(field.coord(100, 50), (r50(), 0.0));
        assert_eq!(field.coord(0, 0), (-r50(), -r50()));
        assert_eq!(grid_half_extents(&band_101()), (r50(), r50()));
    }
}
