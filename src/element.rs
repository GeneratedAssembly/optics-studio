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
//! All spatial quantities are SI units (meters); angles are [`Angle`]
//! (radians).
//!
//! # Orientation convention
//!
//! An element's `orientation` ψ is its **in-plane rotation about the
//! optical axis $z$, measured counterclockwise (CCW) when viewed from
//! downstream — looking back along $-z$ toward the oncoming beam, with
//! $+x$ to the right and $+y$ up** (right-hand rule about $+z$, the
//! propagation direction). Viewed from upstream looking along $+z$ (with
//! the beam), the same positive rotation appears clockwise. The
//! element's intrinsic frame $(x', y')$ relates to field coordinates by
//! $\mathbf{r}' = R(-\psi)(\mathbf{r} - \mathbf{c})$ with
//! $R(\psi) = \begin{pmatrix}\cos\psi & -\sin\psi\\ \sin\psi &
//! \cos\psi\end{pmatrix}$: patterns (grating vectors, mask closures,
//! ellipse/rectangle axes) are defined along/within the element frame,
//! and the clear aperture rotates with the element.
//!
//! Elements with a **typical orientation** — flat, normal to $z$, their
//! intrinsic axes aligned with the lab axes — accept
//! `Option<Angle>`: `None` *is* that typical orientation (ψ = 0), and
//! passing `Some(Angle::zero())` behaves identically. Elements whose
//! direction is intrinsic design data rather than a mounting choice
//! (the [`BlazedGrating`] grating vector) take a mandatory `Angle`.
//! Orientation is always exposed by the `orientation()` accessor as the
//! effective `Angle` (wrapped into $[0, 2\pi)$).

use std::f64::consts::PI;

use num_complex::Complex64;
use thiserror::Error;

use crate::common::Angle;
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
    /// The optical axis (phase-center) coordinates were not finite.
    #[error("element axis needs finite coordinates (got ({center_x}, {center_y}) m)")]
    InvalidAxis {
        /// Axis x-coordinate in meters.
        center_x: f64,
        /// Axis y-coordinate in meters.
        center_y: f64,
    },
    /// The dispersive focal closure returned a focal length that cannot
    /// define a phase screen: zero (infinite power) or non-finite.
    #[error("lens focal length f({lambda} m) = {f} m is invalid (must be non-zero and finite)")]
    InvalidFocal {
        /// Wavelength (m) at which the closure was evaluated.
        lambda: f64,
        /// Focal length (m) the closure returned.
        f: f64,
    },
    /// The phase-mask closure returned a non-finite value at a field
    /// coordinate; the reported position is the first offending sample
    /// (element-frame coordinates, meters).
    #[error("phase mask returned non-finite value {phase} at ({x}, {y}) m (element frame)")]
    InvalidPhase {
        /// Center-relative x of the first offending sample (m).
        x: f64,
        /// Center-relative y of the first offending sample (m).
        y: f64,
        /// The non-finite phase value the closure returned (rad).
        phase: f64,
    },
    /// The grating geometry is not physically meaningful: the period must
    /// be positive and finite; the blaze depth (rad) and orientation (rad)
    /// must be finite.
    #[error(
        "grating needs positive finite period, finite depth and orientation \
         (got period = {period} m, depth = {depth} rad, orientation = {orientation} rad)"
    )]
    InvalidGrating {
        /// Grating period in meters.
        period: f64,
        /// Blaze depth in radians per period.
        depth: f64,
        /// Grating-vector orientation in radians from the +x axis.
        orientation: f64,
    },
    /// The element orientation was not finite. `Angle` wraps implicitly
    /// into $[0, 2\pi)$, so non-finite inputs arrive here as NaN.
    #[error("element orientation must be finite (got {radians} rad)")]
    InvalidOrientation {
        /// The non-finite orientation value in radians.
        radians: f64,
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
    /// No finite clear aperture: an unbounded pure-phase screen (e.g.
    /// [`ThinLens`], task-c decision 2026-10-12 — stops are composed
    /// separately by the caller).
    Unbounded,
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
    /// Unbounded pure-phase screen: no clear aperture, so no finite extent
    /// for regime gating (see [`ElementShape::Unbounded`]).
    Unbounded,
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

/// Clear aperture (stop) of a phase element — the shared geometry vocabulary
/// of [`ThinLens`] and [`PhaseMask`] (task-d revision 2026-10-12: rectangular
/// DOE windows need non-circular stops; a predicate-closure stop is a
/// deferred decision for later discussion).
///
/// Stops are **hard-edge and boundary inclusive**, all sizes in meters,
/// measured from the element center — the same predicates as the
/// [`EllipticalAperture`]/[`RectangularAperture`] elements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearAperture {
    /// No stop: the element acts on the whole sampled plane (reports
    /// `Unbounded` shape/extent; regime gating must then use the applied
    /// stop's half-extent).
    Field,
    /// Circular stop of radius `r` (m).
    Circle(f64),
    /// Elliptical stop with semi-axes `a`, `b` (m) along x, y.
    Ellipse {
        /// Semi-axis along x in meters.
        a: f64,
        /// Semi-axis along y in meters.
        b: f64,
    },
    /// Rectangular stop with half-widths (m) along x, y; full opening
    /// $2\,\text{half\_x} \times 2\,\text{half\_y}$.
    Rectangle {
        /// Half-width along x in meters.
        half_x: f64,
        /// Half-width along y in meters.
        half_y: f64,
    },
}

impl ClearAperture {
    /// Checks stop geometry (positive finite half-extents where stopped)
    /// and a finite center.
    ///
    /// # Errors
    /// [`OpticsError::InvalidAxis`] for `Field` with a non-finite center;
    /// [`OpticsError::InvalidAperture`] otherwise.
    fn validate(&self, center: (f64, f64)) -> Result<(), OpticsError> {
        match *self {
            ClearAperture::Field => {
                let (cx, cy) = center;
                if !cx.is_finite() || !cy.is_finite() {
                    return Err(OpticsError::InvalidAxis {
                        center_x: cx,
                        center_y: cy,
                    });
                }
            }
            ClearAperture::Circle(r) => validate_footprint(r, r, center)?,
            ClearAperture::Ellipse { a, b } => validate_footprint(a, b, center)?,
            ClearAperture::Rectangle { half_x, half_y } => {
                validate_footprint(half_x, half_y, center)?
            }
        }
        Ok(())
    }

    /// Outline for the element taxonomy.
    fn shape(&self) -> ElementShape {
        match self {
            ClearAperture::Field => ElementShape::Unbounded,
            ClearAperture::Circle(_) => ElementShape::Circle,
            ClearAperture::Ellipse { .. } => ElementShape::Ellipse,
            ClearAperture::Rectangle { .. } => ElementShape::Rectangle,
        }
    }

    /// Footprint half-extents `(fx, fy)` for the grid-fit check, or `None`
    /// for the field-wide stop.
    fn footprint(&self) -> Option<(f64, f64)> {
        match self {
            ClearAperture::Field => None,
            ClearAperture::Circle(r) => Some((*r, *r)),
            ClearAperture::Ellipse { a, b } => Some((*a, *b)),
            ClearAperture::Rectangle { half_x, half_y } => Some((*half_x, *half_y)),
        }
    }

    /// Half-extent along each axis, used by sampling diagnostics; `INF`
    /// along both for `Field` (the grid then bounds the reach).
    fn axis_halves(&self) -> (f64, f64) {
        match self {
            ClearAperture::Field => (f64::INFINITY, f64::INFINITY),
            ClearAperture::Circle(r) => (*r, *r),
            ClearAperture::Ellipse { a, b } => (*a, *b),
            ClearAperture::Rectangle { half_x, half_y } => (*half_x, *half_y),
        }
    }

    /// Conservative single-number clear-aperture radius for regime gating
    /// (larger half-side; worst-case $N_F$); `None` for `Field`.
    pub fn extent_radius(&self) -> Option<f64> {
        let (fx, fy) = self.footprint()?;
        Some(fx.max(fy))
    }

    /// Boundary-inclusive containment at center-relative coordinates
    /// (meters), in the stop's own (unrotated) frame.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match *self {
            ClearAperture::Field => true,
            ClearAperture::Circle(r) => x * x + y * y <= r * r,
            ClearAperture::Ellipse { a, b } => {
                let (a2, b2) = (a * a, b * b);
                b2 * x * x + a2 * y * y <= a2 * b2
            }
            ClearAperture::Rectangle { half_x, half_y } => x.abs() <= half_x && y.abs() <= half_y,
        }
    }

    /// As [`ClearAperture::contains`], for the stop rotated by `psi`
    /// (rad, CCW about z — module docs): center-relative **field**
    /// coordinates are mapped into the stop frame first.
    pub fn contains_at(&self, x: f64, y: f64, psi: f64) -> bool {
        let (fx, fy) = to_element_frame(x, y, psi);
        self.contains(fx, fy)
    }

    /// Axis-aligned bounding half-extents of the stop rotated by `psi`;
    /// `None` for the field-wide stop. A circle is rotation-invariant.
    fn footprint_at(&self, psi: f64) -> Option<(f64, f64)> {
        let (hx, hy) = self.footprint()?;
        match self {
            ClearAperture::Circle(_) => Some((hx, hy)),
            _ => Some(rotated_box(hx, hy, psi)),
        }
    }

    /// As [`ClearAperture::axis_halves`], for the stop rotated by `psi`.
    fn axis_halves_at(&self, psi: f64) -> (f64, f64) {
        match self {
            ClearAperture::Field => (f64::INFINITY, f64::INFINITY),
            ClearAperture::Circle(r) => (*r, *r),
            _ => {
                let (hx, hy) = self.axis_halves();
                rotated_box(hx, hy, psi)
            }
        }
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

/// Rotate center-relative field coordinates into the element frame:
/// $\mathbf{r}' = R(-\psi)\mathbf{r}$, with ψ the element orientation
/// (CCW about $z$ viewed from downstream — module docs).
fn to_element_frame(x: f64, y: f64, psi: f64) -> (f64, f64) {
    let (c, s) = (psi.cos(), psi.sin());
    (c * x + s * y, -s * x + c * y)
}

/// Axis-aligned bounding half-extents of a $(h_x, h_y)$ box rotated by
/// `psi` (rad): $h_x' = h_x|\cos| + h_y|\sin|$, $h_y' = h_x|\sin| +
/// h_y|\cos|$.
fn rotated_box(hx: f64, hy: f64, psi: f64) -> (f64, f64) {
    let (c, s) = (psi.cos().abs(), psi.sin().abs());
    (hx * c + hy * s, hx * s + hy * c)
}

/// Resolve a constructor's optional orientation: `None` is the typical
/// orientation of a flat element (ψ = 0: normal to $z$, intrinsic axes
/// aligned with the lab axes); `Some` must be finite.
fn resolve_orientation(orientation: Option<Angle>) -> Result<Angle, OpticsError> {
    let o = orientation.unwrap_or_else(Angle::zero);
    if !o.is_finite() {
        return Err(OpticsError::InvalidOrientation { radians: *o });
    }
    Ok(o)
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
/// $$t(x, y) = \begin{cases} 1 & \displaystyle\frac{x'^2}{a^2} +
/// \frac{y'^2}{b^2} \le 1 \\ 0 & \text{otherwise} \end{cases}
/// \qquad (x', y') = R(-\psi)\big((x, y) - (x_c, y_c)\big)$$
///
/// with semi-axes $(a, b)$ (m) in the element frame, orientation ψ (CCW
/// about $z$, module docs), and center $(x_c, y_c)$ (m) in field
/// coordinates (origin at the grid center, see [`crate::field`]). A sample
/// whose coordinate lies exactly on the ellipse boundary is **inside**
/// (transmitted).
///
/// The **circle is the axis-symmetric special case**: construct it with
/// [`EllipticalAperture::circle`] (or `new` with equal semi-axes), which
/// reports [`ElementShape::Circle`] and exposes [`EllipticalAperture::radius`]
/// (approved task-b decision 2026-10-12: one aperture type, no separate
/// circular element). A circle ignores its orientation (invariant).
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
    semi_axes: (f64, f64), // meters: (a, b) in the element frame
    orientation: Angle,    // rad, CCW about z (module docs)
    center: (f64, f64),    // meters, field coordinates
    shape: ElementShape,
    material: Material,
}

impl EllipticalAperture {
    /// Ideal hard-edge ellipse with semi-axes `semi_axes` $= (a, b)$ (m,
    /// element frame), orientation `orientation` (see the module docs:
    /// `None` = typical axis-aligned orientation), and center `center`
    /// (m, m) in field coordinates.
    ///
    /// Equal semi-axes ($a == b$ exactly) report [`ElementShape::Circle`]
    /// (orientation then irrelevant).
    ///
    /// The stored `material` is metadata for future refractive/absorptive
    /// element models; the hard-edge transmission model is the binary mask
    /// alone.
    ///
    /// # Errors
    /// [`OpticsError::InvalidOrientation`] for a non-finite `Some` angle;
    /// [`OpticsError::InvalidAperture`] if either semi-axis is not positive
    /// and finite, or either center coordinate is not finite.
    pub fn new(
        semi_axes: (f64, f64),
        orientation: Option<Angle>,
        center: (f64, f64),
    ) -> Result<Self, OpticsError> {
        let orientation = resolve_orientation(orientation)?;
        let (a, b) = semi_axes;
        validate_footprint(a, b, center)?;
        let shape = if a == b {
            ElementShape::Circle
        } else {
            ElementShape::Ellipse
        };
        Ok(Self {
            semi_axes,
            orientation,
            center,
            shape,
            // Non-absorbing, index-1 placeholder: unused by the hard-edge mask.
            material: Material::new(|_lambda, _angle| 1.0, |_lambda, _angle| 0.0),
        })
    }

    /// Ideal hard-edge **circle** of radius `radius` (m) centered at `center`
    /// (m, m) — the axis-symmetric special case of [`EllipticalAperture::new`]
    /// with semi-axes $(r, r)$; orientation is irrelevant (typical).
    ///
    /// # Errors
    /// [`OpticsError::InvalidAperture`] if `radius` is not positive and
    /// finite, or either center coordinate is not finite.
    pub fn circle(radius: f64, center: (f64, f64)) -> Result<Self, OpticsError> {
        Self::new((radius, radius), None, center)
    }

    /// Semi-axes $(a, b)$ in meters (element frame).
    pub fn semi_axes(&self) -> (f64, f64) {
        self.semi_axes
    }

    /// Effective orientation (rad, CCW about $z$; `Angle` wrapped).
    pub fn orientation(&self) -> Angle {
        self.orientation
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
    /// physical coordinates fall outside the (rotated) ellipse are set to
    /// zero, all others (including the boundary) are passed through
    /// unchanged.
    ///
    /// The boundary test uses the axis-multiplied form
    /// $b^2\,x'^2 + a^2\,y'^2 > a^2 b^2$ in the element frame (no
    /// divisions, exactly the ellipse inequality).
    ///
    /// # Errors
    /// [`OpticsError::OutsideField`] if the rotated aperture's bounding box
    /// (module docs) does not fit inside the grid.
    fn apply(&self, band: &BandField) -> Result<BandField, OpticsError> {
        let (a, b) = self.semi_axes;
        let (cx, cy) = self.center;
        let psi = *self.orientation;
        // A circle is rotation-invariant; a genuine ellipse bounds a rotated box.
        let (bx, by) = if a == b {
            (a, b)
        } else {
            rotated_box(a, b, psi)
        };
        let (grid_half_x, grid_half_y) = grid_half_extents(band);
        if cx.abs() + bx > grid_half_x || cy.abs() + by > grid_half_y {
            return Err(OpticsError::OutsideField {
                half_x: bx,
                half_y: by,
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
                let (fx, fy) = to_element_frame(x - cx, y - cy, psi);
                if b2 * fx * fx + a2 * fy * fy > limit {
                    *field.sample_mut(ix, iy) = Complex64::ZERO;
                }
            }
        }
        Ok(out)
    }
}

/// Ideal hard-edge rectangular aperture: a binary amplitude mask
///
/// $$t(x, y) = \begin{cases} 1 & |x'| \le a \;\text{and}\; |y'| \le b
/// \\ 0 & \text{otherwise} \end{cases}
/// \qquad (x', y') = R(-\psi)\big((x, y) - (x_c, y_c)\big)$$
///
/// with half-widths $(a, b)$ (m) in the element frame, orientation ψ (CCW
/// about $z$, module docs), and center $(x_c, y_c)$ (m) in field
/// coordinates (origin at the grid center, see [`crate::field`]).
/// Samples whose coordinates lie exactly on an edge are **inside**
/// (transmitted). The full opening is $2a \times 2b$; a square is the
/// $a == b$ special case (still reported as [`ElementShape::Rectangle`],
/// and rotating a square DOES change its footprint).
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
    half_width: (f64, f64), // meters: (a, b) in the element frame
    orientation: Angle,     // rad, CCW about z (module docs)
    center: (f64, f64),     // meters, field coordinates
    shape: ElementShape,
    material: Material,
}

impl RectangularAperture {
    /// Ideal hard-edge rectangle with half-widths `half_width` $= (a, b)$
    /// (m, element frame), orientation `orientation` (`None` = typical
    /// axis-aligned), and center `center` (m, m) in field coordinates.
    ///
    /// The stored `material` is metadata for future refractive/absorptive
    /// element models; the hard-edge transmission model is the binary mask
    /// alone.
    ///
    /// # Errors
    /// [`OpticsError::InvalidOrientation`] for a non-finite `Some` angle;
    /// [`OpticsError::InvalidAperture`] if either half-width is not
    /// positive and finite, or either center coordinate is not finite.
    pub fn new(
        half_width: (f64, f64),
        orientation: Option<Angle>,
        center: (f64, f64),
    ) -> Result<Self, OpticsError> {
        let orientation = resolve_orientation(orientation)?;
        let (a, b) = half_width;
        validate_footprint(a, b, center)?;
        Ok(Self {
            half_width,
            orientation,
            center,
            shape: ElementShape::Rectangle,
            // Non-absorbing, index-1 placeholder: unused by the hard-edge mask.
            material: Material::new(|_lambda, _angle| 1.0, |_lambda, _angle| 0.0),
        })
    }

    /// Half-widths $(a, b)$ in meters (element frame); full opening
    /// $2a \times 2b$.
    pub fn half_width(&self) -> (f64, f64) {
        self.half_width
    }

    /// Effective orientation (rad, CCW about $z$; `Angle` wrapped).
    pub fn orientation(&self) -> Angle {
        self.orientation
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
    /// outside the (rotated) box are set to zero, all others (including
    /// the edges) are passed through unchanged.
    ///
    /// # Errors
    /// [`OpticsError::OutsideField`] if the rotated rectangle's bounding
    /// box (module docs) does not fit inside the grid.
    fn apply(&self, band: &BandField) -> Result<BandField, OpticsError> {
        let (a, b) = self.half_width;
        let (cx, cy) = self.center;
        let psi = *self.orientation;
        let (bx, by) = rotated_box(a, b, psi);
        let (grid_half_x, grid_half_y) = grid_half_extents(band);
        if cx.abs() + bx > grid_half_x || cy.abs() + by > grid_half_y {
            return Err(OpticsError::OutsideField {
                half_x: bx,
                half_y: by,
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
                let (fx, fy) = to_element_frame(x - cx, y - cy, psi);
                if fx.abs() > a || fy.abs() > b {
                    *field.sample_mut(ix, iy) = Complex64::ZERO;
                }
            }
        }
        Ok(out)
    }
}

/// Ideal **thin lens**: a quadratic phase screen with dispersive focal
/// closure $f(\lambda)$ (approved Phase-3 decision D1) inside a **clear
/// aperture** of any supported shape (task-c revision 2026-10-12: `extent`
/// describes the lens, not the field — regime gating uses the lens's own
/// aperture).
///
/// At the sampled plane the lens acts as
///
/// $$t(x, y; \lambda) = \begin{cases}
///   \exp\!\Big(-i\,\frac{2\pi}{\lambda}\,
///   \frac{(x - x_c)^2 + (y - y_c)^2}{2 f(\lambda)}\Big) &
///   (x - x_c,\, y - y_c) \in A \\
///   0 & \text{otherwise}\end{cases}$$
///
/// with optical axis $(x_c, y_c)$ (m, field coordinates), clear aperture
/// $A$ (see [`ClearAperture`]), and focal length $f$ (m) evaluated at the
/// band's wavelength. Sign convention: $f > 0$ is **converging** — the negative
/// quadratic phase matches the converging curvature of the $q$-form beam
/// in [`crate::sources`] ($e^{+ikz}$ frame) and the textbook's lens
/// construction, Eq. 4.13: zone radii $r_p^2 = 2fp\lambda$ impose $2\pi$
/// phase steps, i.e. the same quadratic profile (Sec. 4.2.1). $f < 0$
/// diverges; $f = 0$ is rejected ([`OpticsError::InvalidFocal`]).
///
/// The **field-wide special case** [`ClearAperture::Field`] (the earlier
/// task-c form, kept for callers who compose their own stop) applies the
/// phase over the whole sampled plane; it reports `Unbounded` shape/extent
/// and regime gating must then use the *applied* stop.
///
/// Chromatic behavior (textbook Sec. 4.1.3): refractive power is linear in
/// $n(\lambda)$, so blue focuses shorter — encode any material/DOE law in
/// the closure; `fixed` is the non-dispersing sugar.
// No `Clone`/`Debug`/`PartialEq` derives: the focal closure and `Material`
// hold boxed functions; element identity is by value of its accessors.
pub struct ThinLens {
    f: Box<dyn Fn(f64) -> f64>, // focal length (m) as a function of wavelength (m)
    orientation: Angle,         // rad, CCW about z; only the stop depends on it
    aperture: ClearAperture,    // stop geometry, meters from center
    center: (f64, f64),         // optical axis, meters, field coordinates
    shape: ElementShape,
    material: Material,
}

impl ThinLens {
    /// Thin lens with focal closure `f` of wavelength (m), `orientation`
    /// (`None` = typical: normal to $z$; the quadratic phase is rotation-
    /// symmetric, so only a non-circular `aperture` depends on it), clear
    /// aperture `aperture` (see [`ClearAperture`]), and optical axis at
    /// `center` (m, m) in field coordinates.
    ///
    /// The closure is evaluated per band at `apply` time; results must be
    /// non-zero and finite (else [`OpticsError::InvalidFocal`]).
    ///
    /// # Errors
    /// [`OpticsError::InvalidOrientation`] for a non-finite `Some` angle;
    /// [`OpticsError::InvalidAperture`] for a stopped lens with a
    /// non-positive/non-finite half-extent; [`OpticsError::InvalidAxis`]
    /// for `Field` with a non-finite center.
    pub fn new(
        f: impl Fn(f64) -> f64 + 'static,
        orientation: Option<Angle>,
        aperture: ClearAperture,
        center: (f64, f64),
    ) -> Result<Self, OpticsError> {
        let orientation = resolve_orientation(orientation)?;
        aperture.validate(center)?;
        let shape = aperture.shape();
        Ok(Self {
            f: Box::new(f),
            orientation,
            aperture,
            center,
            shape,
            // Non-absorbing, index-1 placeholder: the phase is carried by `f`.
            material: Material::new(|_lambda, _angle| 1.0, |_lambda, _angle| 0.0),
        })
    }

    /// Non-dispersing thin lens of focal length `f0` (m): the constant
    /// closure $f(\lambda) \equiv f_0$. See [`ThinLens::new`] for the
    /// `orientation`/`aperture` semantics.
    ///
    /// # Errors
    /// As [`ThinLens::new`].
    pub fn fixed(
        f0: f64,
        orientation: Option<Angle>,
        aperture: ClearAperture,
        center: (f64, f64),
    ) -> Result<Self, OpticsError> {
        Self::new(move |_lambda| f0, orientation, aperture, center)
    }

    /// Focal length (m) at wavelength `lambda` (m) from the closure.
    pub fn focal_at(&self, lambda: f64) -> f64 {
        (self.f)(lambda)
    }

    /// Effective orientation (rad, CCW about $z$); affects only the stop.
    pub fn orientation(&self) -> Angle {
        self.orientation
    }

    /// The lens's clear aperture.
    pub fn aperture(&self) -> ClearAperture {
        self.aperture
    }

    /// Optical axis $(x_c, y_c)$ in meters (field coordinates).
    pub fn center(&self) -> (f64, f64) {
        self.center
    }

    /// Worst adjacent-sample phase step of the lens quadratic on `band`'s
    /// grid, in radians — a closed-form $O(1)$ sampling diagnostic (no
    /// sweep): with $k = 2\pi/\lambda$ and $r$ the farthest on-axis
    /// coordinate *inside the acting region* along the measured axis
    /// ($\min(\text{aperture half-extent},\, h + |c|)$; $h$ = grid
    /// half-extent, $c$ = axis coordinate),
    ///
    /// $$\Delta\varphi_{\max} = \frac{k\,\delta}{|f(\lambda)|}
    ///   \Big(r - \frac{\delta}{2}\Big)$$
    ///
    /// maximized over the two axes ($\delta$ = pitch along the axis; the
    /// extreme adjacent pair straddles $r$ minus one half-pitch — an exact
    /// maximum field-wide, an upper bound for a stopped lens whose rim
    /// does not land on a sample). The lens profile is resolved when
    /// $\Delta\varphi_{\max} < \pi$; the value is reported, not enforced
    /// (elements stay local — regime/sampling judgment is caller-side).
    /// Returns `f64::INFINITY` when the focal closure is invalid at
    /// `band`'s wavelength.
    pub fn worst_phase_step(&self, band: &BandField) -> f64 {
        let lambda = band.lambda();
        let f = (self.f)(lambda);
        if !f.is_finite() || f == 0.0 {
            return f64::INFINITY;
        }
        let (nx, ny) = band.field().dims();
        let (dx, dy) = band.field().pitch();
        let (cx, cy) = self.center;
        let half_x = (nx as f64 - 1.0) / 2.0 * dx;
        let half_y = (ny as f64 - 1.0) / 2.0 * dy;
        let (ax, ay) = self.aperture.axis_halves_at(*self.orientation);
        let rx = ax.min(half_x + cx.abs());
        let ry = ay.min(half_y + cy.abs());
        let k = 2.0 * PI / lambda;
        let step_x = k * dx * (rx - dx / 2.0).max(0.0) / f.abs();
        let step_y = k * dy * (ry - dy / 2.0).max(0.0) / f.abs();
        step_x.max(step_y)
    }
}

impl Element for ThinLens {
    fn shape(&self) -> &ElementShape {
        &self.shape
    }

    fn kind(&self) -> ElementKind {
        ElementKind::Lens
    }

    fn material(&self) -> (&Material, &Material) {
        (&self.material, &self.material)
    }

    /// The lens's own clear aperture (task-c revision: extent describes the
    /// element, not the grid) — the conservative larger half-extent for
    /// stopped forms, `Unbounded` for [`ClearAperture::Field`].
    fn extent(&self) -> Extent {
        match self.aperture.extent_radius() {
            Some(r) => Extent::Aperture(r),
            None => Extent::Unbounded,
        }
    }

    /// Applies the lens: inside the clear aperture (or the whole plane when
    /// `Field`) multiplies the quadratic phase; outside the stop the sample
    /// is set to zero (hard edge, boundary inclusive like the aperture
    /// elements).
    ///
    /// # Errors
    /// [`OpticsError::InvalidFocal`] if $f(\lambda)$ is zero or non-finite
    /// at the band's wavelength; [`OpticsError::OutsideField`] if a stopped
    /// lens's footprint exceeds the grid.
    fn apply(&self, band: &BandField) -> Result<BandField, OpticsError> {
        let lambda = band.lambda();
        let f = (self.f)(lambda);
        if !f.is_finite() || f == 0.0 {
            return Err(OpticsError::InvalidFocal { lambda, f });
        }
        let (cx, cy) = self.center;
        let psi = *self.orientation;
        if let Some((fx, fy)) = self.aperture.footprint_at(psi) {
            let (grid_half_x, grid_half_y) = grid_half_extents(band);
            if cx.abs() + fx > grid_half_x || cy.abs() + fy > grid_half_y {
                return Err(OpticsError::OutsideField {
                    half_x: fx,
                    half_y: fy,
                    center_x: cx,
                    center_y: cy,
                    grid_half_x,
                    grid_half_y,
                });
            }
        }
        let k = 2.0 * PI / lambda;
        let mut out = band.clone();
        let field = out.field_mut();
        let (nx, ny) = field.dims();
        for iy in 0..ny {
            for ix in 0..nx {
                let (x, y) = field.coord(ix, iy);
                let (xr, yr) = (x - cx, y - cy);
                let s = field.sample_mut(ix, iy);
                if !self.aperture.contains_at(xr, yr, psi) {
                    *s = Complex64::ZERO;
                    continue;
                }
                let phase = -k * (xr * xr + yr * yr) / (2.0 * f);
                *s *= (phase * Complex64::I).exp();
            }
        }
        Ok(out)
    }
}

/// General analytic **phase mask** — the diffractive-optics element
/// interface: an arbitrary profile $(x, y) \mapsto \varphi(x, y)$ (radians)
/// painted onto the sampled plane (approved Phase-3 decision D2: analytic
/// closure now; the sampled-profile variant arrives with Phase 5, since PSF
/// export needs grids anyway).
///
/// $$t(x, y) = \begin{cases} e^{i\varphi(x',\; y')} & (x',\, y') \in A \\
///   0 & \text{otherwise (when stopped)} \end{cases}
///   \qquad (x', y') = R(-\psi)\big((x, y) - (x_c, y_c)\big)$$
///
/// The closure receives **element-frame coordinates** in meters (center-
/// relative, rotated by $-\psi$) and returns phase in radians, so DOE
/// designs are translation- and rotation-invariant; the
/// mask itself imposes no wavelength dependence beyond $e^{i\varphi}$ —
/// dispersive profiles (blazed at a design wavelength, DOE lenses) encode
/// $\lambda$ by composing per-band masks or by closures curried with the
/// band's wavelength at construction. This is the textbook's premise: the
/// diffractive lens of Sec. 4.2.1 and the blazed grating of Sec. 2.4.2 are
/// exactly such profiles.
///
/// Clear aperture follows [`ThinLens`] via [`ClearAperture`]: a stopped
/// mask (circle, ellipse, or rectangle — rectangular DOE windows included)
/// is the primary form (hard edge, boundary inclusive, `extent` describes
/// the mask); [`ClearAperture::Field`] acts field-wide. The `orientation`
/// (`None` = typical, normal to $z$) rotates the **whole element frame**:
/// the closure receives element-frame coordinates
/// $(x', y') = R(-\psi)((x, y) - (x_c, y_c))$ and the stop rotates with
/// the pattern — DOE designs are stated once in their own frame.
///
/// **Sampling duty is caller-side:** no closed-form step bound exists for
/// an arbitrary profile, and an $O(N^2)$ sweep to emit warnings is against
/// the project guidelines — the profile must satisfy
/// $|\nabla\varphi| \cdot \max(\Delta x, \Delta y) < \pi$ on the grid.
/// Structured elements validate themselves analytically where possible
/// (see the linear-ramp bound in [`BlazedGrating`], Phase 3 task e).
// No `Clone`/`Debug`/`PartialEq` derives: the phase closure and `Material`
// hold boxed functions; element identity is by value of its accessors.
pub struct PhaseMask {
    phase: Box<dyn Fn(f64, f64) -> f64>, // element-frame (x', y') m -> phase rad
    orientation: Angle,                  // rad, CCW about z (module docs)
    aperture: ClearAperture,             // stop geometry, meters from center
    center: (f64, f64),                  // mask origin, meters, field coordinates
    shape: ElementShape,
    material: Material,
}

impl PhaseMask {
    /// Phase mask with analytic closure `phase` of element-frame
    /// coordinates (m, m) returning radians, `orientation` (`None` =
    /// typical, normal to $z$), clear aperture `aperture` (see
    /// [`ClearAperture`]), and origin at `center` (m, m) in field
    /// coordinates.
    ///
    /// # Errors
    /// [`OpticsError::InvalidOrientation`] for a non-finite `Some` angle;
    /// [`OpticsError::InvalidAperture`] for a stopped mask with a
    /// non-positive/non-finite half-extent; [`OpticsError::InvalidAxis`]
    /// for `Field` with a non-finite center.
    pub fn new(
        phase: impl Fn(f64, f64) -> f64 + 'static,
        orientation: Option<Angle>,
        aperture: ClearAperture,
        center: (f64, f64),
    ) -> Result<Self, OpticsError> {
        let orientation = resolve_orientation(orientation)?;
        aperture.validate(center)?;
        let shape = aperture.shape();
        Ok(Self {
            phase: Box::new(phase),
            orientation,
            aperture,
            center,
            shape,
            // Non-absorbing, index-1 placeholder: the phase is carried by the closure.
            material: Material::new(|_lambda, _angle| 1.0, |_lambda, _angle| 0.0),
        })
    }

    /// Closure probe: phase (rad) at element-frame coordinates `(x, y)`
    /// (m) — center-relative and rotated by $-\psi$. Not validated — use
    /// [`PhaseMask::new`] + `apply` for checked evaluation on the grid.
    pub fn phase_at(&self, x: f64, y: f64) -> f64 {
        (self.phase)(x, y)
    }

    /// Effective orientation (rad, CCW about $z$).
    pub fn orientation(&self) -> Angle {
        self.orientation
    }

    /// The mask's clear aperture.
    pub fn aperture(&self) -> ClearAperture {
        self.aperture
    }

    /// Mask origin $(x_c, y_c)$ in meters (field coordinates).
    pub fn center(&self) -> (f64, f64) {
        self.center
    }
}

impl Element for PhaseMask {
    fn shape(&self) -> &ElementShape {
        &self.shape
    }

    fn kind(&self) -> ElementKind {
        ElementKind::Diffractive
    }

    fn material(&self) -> (&Material, &Material) {
        (&self.material, &self.material)
    }

    /// The mask's own clear aperture — the conservative larger half-extent
    /// for stopped forms, `Unbounded` for [`ClearAperture::Field`].
    fn extent(&self) -> Extent {
        match self.aperture.extent_radius() {
            Some(r) => Extent::Aperture(r),
            None => Extent::Unbounded,
        }
    }

    /// Paints $e^{i\varphi}$ onto the plane (field-wide when `Field`;
    /// inside the clear aperture otherwise, hard zero outside — boundary
    /// inclusive like the aperture elements).
    ///
    /// # Errors
    /// [`OpticsError::InvalidPhase`] when the closure returns a non-finite
    /// value at any acted-on sample (first occurrence reported);
    /// [`OpticsError::OutsideField`] if a stopped mask's footprint exceeds
    /// the grid.
    fn apply(&self, band: &BandField) -> Result<BandField, OpticsError> {
        let (cx, cy) = self.center;
        let psi = *self.orientation;
        if let Some((fx, fy)) = self.aperture.footprint_at(psi) {
            let (grid_half_x, grid_half_y) = grid_half_extents(band);
            if cx.abs() + fx > grid_half_x || cy.abs() + fy > grid_half_y {
                return Err(OpticsError::OutsideField {
                    half_x: fx,
                    half_y: fy,
                    center_x: cx,
                    center_y: cy,
                    grid_half_x,
                    grid_half_y,
                });
            }
        }
        let mut out = band.clone();
        let field = out.field_mut();
        let (nx, ny) = field.dims();
        for iy in 0..ny {
            for ix in 0..nx {
                let (x, y) = field.coord(ix, iy);
                let (xr, yr) = (x - cx, y - cy);
                let (fx, fy) = to_element_frame(xr, yr, psi);
                let s = field.sample_mut(ix, iy);
                if !self.aperture.contains(fx, fy) {
                    *s = Complex64::ZERO;
                    continue;
                }
                let phi = (self.phase)(fx, fy);
                if !phi.is_finite() {
                    return Err(OpticsError::InvalidPhase {
                        x: fx,
                        y: fy,
                        phase: phi,
                    });
                }
                *s *= (phi * Complex64::I).exp();
            }
        }
        Ok(out)
    }
}

/// Ideal **blazed grating** (kinoform): a sawtooth phase ramp of period
/// $\Lambda$ (m), total blaze depth $\Phi$ (rad per period), and grating
/// vector along $\hat{g} = (\cos\psi, \sin\psi)$ — the orientation $\psi$
/// (rad, from the $+x$ axis) is carried now; rotated-order validation
/// arrives when the feature is exercised (task-e decision 2026-10-12).
///
/// $$\varphi(x, y) = \Phi \left(\frac{(\mathbf{r} - \mathbf{c}) \cdot
///   \hat{g}}{\Lambda} \bmod 1\right), \qquad t = e^{i\varphi} \text{ on } A$$
///
/// with hard-edge [`ClearAperture`] stop $A$ (boundary inclusive, like
/// [`ThinLens`] and [`PhaseMask`]). Key identity for depth $\Phi = 2\pi m$
/// (integer $m$): $e^{i\varphi} \equiv e^{\,i 2\pi m\,(\mathbf{r}-\mathbf{c})
///   \cdot\hat{g}/\Lambda}$ — the mod-1 wrap is invisible to the complex
/// exponential, so the grating is EXACTLY a plane wave tilted along
/// $\hat g$ at frequency $m/\Lambda$, and its far field is the single
/// diffraction order $m$ at the grating equation (textbook Eqs. 1.1, 2.26,
/// 5.1):
///
/// $$\sin\theta_m = \frac{m\lambda}{\Lambda}.$$
///
/// This is the Sec. 2.4.2 blazed grating whose efficiency "approaches
/// 100%" into the design order; non-integer depths split power across
/// orders per the Fourier series of the sawtooth (the square-wave
/// efficiency law of Eqs. 2.27–2.29 is the amplitude analogue, deferred
/// with its tests).
///
/// Sampling: the ramp gradient is $\Phi/\Lambda$, so the closed-form worst
/// adjacent-sample step ([`BlazedGrating::worst_phase_step`]) must stay
/// below $\pi$ — for the classic $\Phi = 2\pi$ kinoform this is exactly the
/// period-Nyquist rule $\delta < \Lambda/2$.
// No `Clone`/`Debug`/`PartialEq` derives: the `Material` field holds boxed
// closures; element identity is by value of its accessors.
pub struct BlazedGrating {
    period: f64,        // Lambda, meters
    phase_depth: f64,   // Phi, radians per period
    orientation: Angle, // psi, rad CCW about z — the grating vector (mandatory)
    aperture: ClearAperture,
    center: (f64, f64), // grating origin, meters, field coordinates
    shape: ElementShape,
    material: Material,
}

impl BlazedGrating {
    /// Blazed grating with period `period` (m, positive finite), blaze
    /// depth `phase_depth` (rad per period, finite), grating-vector
    /// orientation `orientation` — mandatory `Angle`, since the vector
    /// direction is intrinsic design data, not a mounting choice (module
    /// docs: CCW about $z$ viewed from downstream) — clear aperture
    /// `aperture` (rotating with the element frame), and origin `center`
    /// (m, m).
    ///
    /// # Errors
    /// [`OpticsError::InvalidGrating`] for a non-positive/non-finite period
    /// or non-finite depth/orientation; the stop is validated with the
    /// shared aperture rules (`InvalidAperture` / `InvalidAxis`).
    pub fn new(
        period: f64,
        phase_depth: f64,
        orientation: Angle,
        aperture: ClearAperture,
        center: (f64, f64),
    ) -> Result<Self, OpticsError> {
        if !period.is_finite()
            || period <= 0.0
            || !phase_depth.is_finite()
            || !orientation.is_finite()
        {
            return Err(OpticsError::InvalidGrating {
                period,
                depth: phase_depth,
                orientation: *orientation,
            });
        }
        aperture.validate(center)?;
        let shape = aperture.shape();
        Ok(Self {
            period,
            phase_depth,
            orientation,
            aperture,
            center,
            shape,
            material: Material::new(|_lambda, _angle| 1.0, |_lambda, _angle| 0.0),
        })
    }

    /// Classic **kinoform blaze**: depth $\Phi = 2\pi$ per period — a pure
    /// tilted plane wave steering all power into diffraction order $m = 1$
    /// (textbook Sec. 2.4.2, Fig. 1.17).
    ///
    /// # Errors
    /// As [`BlazedGrating::new`].
    pub fn kinoform(
        period: f64,
        orientation: Angle,
        aperture: ClearAperture,
        center: (f64, f64),
    ) -> Result<Self, OpticsError> {
        Self::new(period, 2.0 * PI, orientation, aperture, center)
    }

    /// Grating period $\Lambda$ in meters.
    pub fn period(&self) -> f64 {
        self.period
    }

    /// Blaze depth $\Phi$ in radians per period.
    pub fn phase_depth(&self) -> f64 {
        self.phase_depth
    }

    /// Grating-vector orientation (rad, CCW about $z$; `Angle` wrapped).
    pub fn orientation(&self) -> Angle {
        self.orientation
    }

    /// The grating's clear aperture.
    pub fn aperture(&self) -> ClearAperture {
        self.aperture
    }

    /// Grating origin $(x_c, y_c)$ in meters (field coordinates).
    pub fn center(&self) -> (f64, f64) {
        self.center
    }

    /// Sawtooth probe: blaze phase (rad) at element-frame coordinates
    /// `(x, y)` (m) — the ramp runs along the element-frame $x'$ (the
    /// grating vector), so $\varphi = \Phi\,(x/\Lambda \bmod 1)$.
    pub fn phase_at(&self, x: f64, y: f64) -> f64 {
        let _ = y; // the stripes are invariant along the element-frame y
        let u = x / self.period;
        self.phase_depth * (u - u.floor())
    }

    /// Worst adjacent-sample phase step of the ramp on `band`'s grid,
    /// radians — closed form $O(1)$: the gradient magnitude is
    /// $\Phi/\Lambda$ along $\hat g$, so steps are
    /// $(\Phi/\Lambda)|\cos\psi|\,\Delta x$ along $x$ and
    /// $(\Phi/\Lambda)|\sin\psi|\,\Delta y$ along $y$; the maximum must
    /// stay below $\pi$ (for $\Phi = 2\pi$ and $\psi = 0$: $\Delta x <$
    /// $\Lambda/2$, the period-Nyquist rule). Reports the ramp bound:
    /// for integer-$2\pi$ depths the sawtooth wrap is invisible to
    /// $e^{i\varphi}$; for other depths the physical discontinuity jump
    /// (magnitude $|\Phi \bmod 2\pi|$, principal value) is additional.
    /// Reported, not enforced.
    pub fn worst_phase_step(&self, band: &BandField) -> f64 {
        let (dx, dy) = band.field().pitch();
        let slope = self.phase_depth.abs() / self.period;
        slope * (self.orientation.cos().abs() * dx).max(self.orientation.sin().abs() * dy)
    }
}

impl Element for BlazedGrating {
    fn shape(&self) -> &ElementShape {
        &self.shape
    }

    fn kind(&self) -> ElementKind {
        ElementKind::Diffractive
    }

    fn material(&self) -> (&Material, &Material) {
        (&self.material, &self.material)
    }

    /// The grating's own clear aperture; `Unbounded` for
    /// [`ClearAperture::Field`].
    fn extent(&self) -> Extent {
        match self.aperture.extent_radius() {
            Some(r) => Extent::Aperture(r),
            None => Extent::Unbounded,
        }
    }

    /// Paints the sawtooth blaze phase onto the plane inside the clear
    /// aperture (hard zero outside, boundary inclusive).
    ///
    /// # Errors
    /// [`OpticsError::OutsideField`] if a stopped grating's footprint
    /// exceeds the grid.
    fn apply(&self, band: &BandField) -> Result<BandField, OpticsError> {
        let (cx, cy) = self.center;
        let psi = *self.orientation;
        if let Some((fx, fy)) = self.aperture.footprint_at(psi) {
            let (grid_half_x, grid_half_y) = grid_half_extents(band);
            if cx.abs() + fx > grid_half_x || cy.abs() + fy > grid_half_y {
                return Err(OpticsError::OutsideField {
                    half_x: fx,
                    half_y: fy,
                    center_x: cx,
                    center_y: cy,
                    grid_half_x,
                    grid_half_y,
                });
            }
        }
        let inv_period = 1.0 / self.period;
        let mut out = band.clone();
        let field = out.field_mut();
        let (nx, ny) = field.dims();
        for iy in 0..ny {
            for ix in 0..nx {
                let (x, y) = field.coord(ix, iy);
                let (fx, fy) = to_element_frame(x - cx, y - cy, psi);
                let s = field.sample_mut(ix, iy);
                if !self.aperture.contains(fx, fy) {
                    *s = Complex64::ZERO;
                    continue;
                }
                let u = fx * inv_period;
                let phi = self.phase_depth * (u - u.floor());
                *s *= (phi * Complex64::I).exp();
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;
    use crate::propagation::{asm, fresnel};
    use crate::sources::{self, GaussianBeam};
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
            EllipticalAperture::new((0.0, 1e-3), None, (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { half_x: 0.0, .. })
        ));
        assert!(matches!(
            EllipticalAperture::new((2e-3, -1e-3), None, (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { half_y, .. }) if half_y == -1e-3
        ));
        assert!(matches!(
            EllipticalAperture::new((f64::NAN, 1e-3), None, (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        assert!(matches!(
            EllipticalAperture::new((1e-3, 1e-3), None, (f64::INFINITY, 0.0)),
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
        let ap = EllipticalAperture::new((2.5e-3, 1.5e-3), None, (1e-3, -1e-3)).unwrap();
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
        let same = EllipticalAperture::new((2.5e-3, 2.5e-3), None, (1e-3, -1e-3)).unwrap();
        assert!(matches!(same.shape(), ElementShape::Circle));
        assert_eq!(same.radius(), ap.radius());
    }

    #[test]
    fn rectangular_constructor_validates_geometry() {
        assert!(matches!(
            RectangularAperture::new((0.0, 1e-3), None, (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { half_x: 0.0, .. })
        ));
        assert!(matches!(
            RectangularAperture::new((1e-3, f64::NAN), None, (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        assert!(matches!(
            RectangularAperture::new((1e-3, 1e-3), None, (0.0, f64::NAN)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        let ap = RectangularAperture::new((2.5e-3, 1.5e-3), None, (1e-3, -1e-3)).unwrap();
        assert_eq!(ap.half_width(), (2.5e-3, 1.5e-3));
        assert_eq!(ap.center(), (1e-3, -1e-3));
    }

    #[test]
    fn aperture_exceeding_grid_extent_is_rejected() {
        let band = band_101();
        let big = EllipticalAperture::new((55e-3, 10e-3), None, (0.0, 0.0)).unwrap();
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
            EllipticalAperture::new((r50(), r50()), None, (0.0, 0.0))
                .unwrap()
                .apply(&band)
                .is_ok()
        );
        // Rectangle: same bounding-box rule.
        let rect_big = RectangularAperture::new((10e-3, 55e-3), None, (0.0, 0.0)).unwrap();
        assert!(matches!(
            rect_big.apply(&band),
            Err(OpticsError::OutsideField { half_y: 55e-3, .. })
        ));
        assert!(
            RectangularAperture::new((r50(), r50()), None, (0.0, 0.0))
                .unwrap()
                .apply(&band)
                .is_ok()
        );
    }

    #[test]
    fn elliptical_mask_geometry_includes_boundary() {
        let band = band_101();
        // Ellipse a = 50 mm (fills x), b = 30 mm.
        let ap = EllipticalAperture::new((r50(), 30e-3), None, (0.0, 0.0)).unwrap();
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
        let ap = RectangularAperture::new((r50(), 30e-3), None, (0.0, 0.0)).unwrap();
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

        let ellipse = EllipticalAperture::new((a, b), None, (0.0, 0.0)).unwrap();
        let ratio = ellipse.apply(&band).unwrap().field().power() / band.field().power();
        let expected = PI * a * b / window;
        // Pixel-count discretization error is O(perimeter * d^2 / area) ~ d/a.
        let rel = ((ratio - expected) / expected).abs();
        assert!(rel < 0.03, "ellipse area error {rel} too large");

        let rect = RectangularAperture::new((a, b), None, (0.0, 0.0)).unwrap();
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
        let ap = RectangularAperture::new((20e-3, 10e-3), None, (0.0, 0.0)).unwrap();
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
        let ell: &dyn Element = &EllipticalAperture::new((3e-3, 20e-3), None, (0.0, 0.0)).unwrap(); // a < b
        assert!(matches!(ell.shape(), ElementShape::Ellipse));
        assert!(matches!(ell.kind(), ElementKind::Aperture));
        // Conservative extent: the larger semi-axis.
        assert!(matches!(
            ell.extent(),
            Extent::Aperture(r) if (r - 20e-3).abs() < 1e-18
        ));
        let rect: &dyn Element =
            &RectangularAperture::new((20e-3, 3e-3), None, (0.0, 0.0)).unwrap();
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

    #[test]
    fn focal_plane_reproduces_fourier_property() {
        // Gaussian waist w0 at the lens plane, lens f, propagate exactly f:
        // the focal plane holds the Fourier transform of the input — a
        // Gaussian waist w0' = lambda * f / (pi * w0), amplitude w0/w0'.
        let w0 = 0.15e-3;
        let f = 0.05;
        let n = 255usize;
        let d = 5e-6;
        let band = sources::gaussian(
            &GaussianBeam {
                w0,
                amplitude: 1.0,
                center: (0.0, 0.0),
            },
            0.0,
            lambda(),
            n,
            n,
            d,
            d,
        )
        .unwrap();
        let focused = ThinLens::fixed(f, None, ClearAperture::Field, (0.0, 0.0))
            .unwrap()
            .apply(&band)
            .unwrap();
        let w_focus = lambda() * f / (PI * w0);
        let sigma_exp = w_focus / 2.0f64.sqrt();
        let asm_out = asm::propagate(&focused, f, &asm::AsmConfig::default()).unwrap();
        let fr_out =
            fresnel::propagate_convolution(&focused, f, &fresnel::FresnelConfig::default())
                .unwrap();
        for (label, out) in [("asm", &asm_out), ("fresnel-conv", &fr_out)] {
            let field = out.field();
            let (mut s0, mut s2) = (0.0f64, 0.0f64);
            for iy in 0..n {
                for ix in 0..n {
                    let (x, _) = field.coord(ix, iy);
                    let i = field.sample(ix, iy).norm_sqr();
                    s0 += i;
                    s2 += i * x * x;
                }
            }
            let sigma = (2.0 * s2 / s0).sqrt();
            let rel = ((sigma - sigma_exp) / sigma_exp).abs();
            assert!(rel < 1e-2, "{label}: focal width rel {rel:e}");
            let amp_exp = w0 / w_focus; // energy-conserving focus amplitude
            let c = field.sample(n / 2, n / 2);
            let rel_a = ((c.norm() - amp_exp) / amp_exp).abs();
            assert!(rel_a < 1e-2, "{label}: focal amplitude rel {rel_a:e}");
        }
    }

    #[test]
    fn phase_profile_is_the_negative_quadratic() {
        let f = 0.1;
        let n = 128usize;
        let d = 1e-5;
        let band = BandField::uniform(2.0, 0.0, n, n, d, d, lambda()).unwrap();
        let out = ThinLens::fixed(f, None, ClearAperture::Field, (0.0, 0.0))
            .unwrap()
            .apply(&band)
            .unwrap();
        let k = 2.0 * PI / lambda();
        for (ix, iy) in [(0usize, 0usize), (63, 64), (127, 127), (10, 120)] {
            let (x, y) = band.field().coord(ix, iy);
            let expected = -k * (x * x + y * y) / (2.0 * f);
            let s = out.field().sample(ix, iy);
            assert!((s.norm() - 2.0).abs() < 1e-12); // pure phase: |t| = 1
            let err = (s.arg() - expected + PI).rem_euclid(2.0 * PI) - PI;
            assert!(err.abs() < 1e-9, "phase at ({ix},{iy}): {err}");
        }
    }

    #[test]
    fn dispersive_closure_evaluates_per_band() {
        // Refractive-like law (textbook Sec. 4.1.3): f grows with lambda.
        let l0 = 500e-9;
        let f0 = 0.1;
        let lens = ThinLens::new(
            move |l| f0 * (1.0 + 0.5 * (l - l0) / l0),
            None,
            ClearAperture::Field,
            (0.0, 0.0),
        )
        .unwrap();
        assert!((lens.focal_at(l0) - f0).abs() < 1e-15);
        assert!((lens.focal_at(1.2 * l0) - 1.1 * f0).abs() < 1e-12);
        let n = 64usize;
        let d = 1e-5;
        let (ix, iy) = (60usize, 32usize);
        for (lam, want_f) in [(l0, f0), (1.2 * l0, 1.1 * f0)] {
            let band = BandField::uniform(1.0, 0.0, n, n, d, d, lam).unwrap();
            let out = lens.apply(&band).unwrap();
            let (x, y) = band.field().coord(ix, iy);
            let expected = -(2.0 * PI / lam) * (x * x + y * y) / (2.0 * want_f);
            let err = (out.field().sample(ix, iy).arg() - expected + PI).rem_euclid(2.0 * PI) - PI;
            assert!(err.abs() < 1e-9, "lambda {lam}: phase err {err}");
        }
    }

    #[test]
    fn diverging_lens_conjugates_the_converging_phase() {
        let band = BandField::uniform(1.0, 0.0, 64, 64, 1e-5, 1e-5, lambda()).unwrap();
        let conv = ThinLens::fixed(0.1, None, ClearAperture::Field, (0.0, 0.0))
            .unwrap()
            .apply(&band)
            .unwrap();
        let div = ThinLens::fixed(-0.1, None, ClearAperture::Field, (0.0, 0.0))
            .unwrap()
            .apply(&band)
            .unwrap();
        for (c, dd) in conv.field().samples().iter().zip(div.field().samples()) {
            assert!((c.conj() - dd).norm() < 1e-12);
        }
    }

    #[test]
    fn lens_validation_and_taxonomy() {
        let band = BandField::uniform(1.0, 0.0, 8, 8, 1e-5, 1e-5, lambda()).unwrap();
        for bad in [0.0, f64::NAN, f64::INFINITY] {
            let lens = ThinLens::fixed(bad, None, ClearAperture::Field, (0.0, 0.0)).unwrap();
            assert!(matches!(
                lens.apply(&band),
                Err(OpticsError::InvalidFocal { .. })
            ));
            assert_eq!(lens.worst_phase_step(&band), f64::INFINITY);
        }
        assert!(matches!(
            ThinLens::new(|_| 0.1, None, ClearAperture::Field, (f64::NAN, 0.0)),
            Err(OpticsError::InvalidAxis { .. })
        ));
        for bad_r in [0.0, -1e-3, f64::NAN] {
            assert!(matches!(
                ThinLens::new(|_| 0.1, None, ClearAperture::Circle(bad_r), (0.0, 0.0)),
                Err(OpticsError::InvalidAperture { .. })
            ));
        }
        // Field-wide special case.
        let wide: &dyn Element =
            &ThinLens::fixed(0.1, None, ClearAperture::Field, (0.0, 0.0)).unwrap();
        assert!(matches!(wide.shape(), ElementShape::Unbounded));
        assert!(matches!(wide.kind(), ElementKind::Lens));
        assert!(matches!(wide.extent(), Extent::Unbounded));
        // Stopped lens: extent describes the lens, not the field.
        let stopped =
            ThinLens::fixed(0.1, None, ClearAperture::Circle(20e-3), (1e-3, -1e-3)).unwrap();
        assert_eq!(stopped.aperture(), ClearAperture::Circle(20e-3));
        assert_eq!(stopped.center(), (1e-3, -1e-3));
        let el: &dyn Element = &stopped;
        assert!(matches!(el.shape(), ElementShape::Circle));
        assert!(matches!(
            el.extent(),
            Extent::Aperture(r) if (r - 20e-3).abs() < 1e-18
        ));
    }

    #[test]
    fn stopped_lens_masks_outside_the_clear_aperture() {
        let band = band_101();
        let a = 30e-3;
        let lens = ThinLens::fixed(0.1, None, ClearAperture::Circle(a), (0.0, 0.0)).unwrap();
        let out = lens.apply(&band).unwrap();
        let f = out.field();
        let k = 2.0 * PI / lambda();
        // Boundary-inclusive stop: (x = a, y = 0) passes with the full lens
        // phase there; just outside is zero. Compare phase *difference* to
        // the input (band_101 carries a pi/3 constant phase).
        assert!((f.sample(80, 50).norm() - band.field().sample(80, 50).norm()).abs() < 1e-12);
        let ratio = f.sample(80, 50) / band.field().sample(80, 50);
        let err = (ratio.arg() - (-k * a * a / (2.0 * 0.1)) + PI).rem_euclid(2.0 * PI) - PI;
        assert!(err.abs() < 1e-9);
        assert_eq!(f.sample(81, 50), Complex64::ZERO); // x = 31 mm > a
        assert_eq!(f.sample(100, 100), Complex64::ZERO); // corner
        assert_eq!(f.sample(50, 50), band.field().sample(50, 50)); // axis: phase 0
        // Uniform input: transmitted power = aperture area / window area.
        let ratio = f.power() / band.field().power();
        let expected = PI * a * a / (100.0e-3 * 100.0e-3);
        let rel = ((ratio - expected) / expected).abs();
        assert!(rel < 0.03, "lens stop area error {rel}");
        // Footprint larger than the grid is rejected, not truncated.
        let big = ThinLens::fixed(0.1, None, ClearAperture::Circle(55e-3), (0.0, 0.0)).unwrap();
        assert!(matches!(
            big.apply(&band),
            Err(OpticsError::OutsideField { half_x: 55e-3, .. })
        ));
    }

    #[test]
    fn worst_phase_step_closed_form_matches_sweep() {
        // Geometry chosen so the true steps stay below pi (no wrap in the
        // sweep); the extreme adjacent pair sits at the grid end farthest
        // from the axis: |x'_a + x'_b| = 2(h + |c|) - d.
        let n = 64usize;
        let d = 1e-4;
        let center = (3e-4, -2e-4);
        let band = BandField::uniform(1.0, 0.0, n, n, d, d, lambda()).unwrap();
        let lens = ThinLens::fixed(2.0, None, ClearAperture::Field, center).unwrap();
        let out = lens.apply(&band).unwrap();
        let field = out.field();
        let mut max_step = 0.0f64;
        for iy in 0..n {
            for ix in 1..n {
                let step = (field.sample(ix, iy) / field.sample(ix - 1, iy))
                    .arg()
                    .abs();
                max_step = max_step.max(step);
            }
        }
        for iy in 1..n {
            for ix in 0..n {
                let step = (field.sample(ix, iy) / field.sample(ix, iy - 1))
                    .arg()
                    .abs();
                max_step = max_step.max(step);
            }
        }
        let closed = lens.worst_phase_step(&band);
        assert!(closed < PI, "geometry must stay unwrapped, got {closed}");
        assert!(
            (closed - max_step).abs() < 1e-9,
            "closed form {closed} vs sweep {max_step}"
        );
    }

    #[test]
    fn identity_mask_is_the_exact_identity() {
        let band = band_101();
        let mask = PhaseMask::new(|_x, _y| 0.0, None, ClearAperture::Field, (0.0, 0.0)).unwrap();
        let out = mask.apply(&band).unwrap();
        assert_eq!(out.field().samples(), band.field().samples());
    }

    #[test]
    fn linear_ramp_reproduces_a_tilted_plane_wave() {
        // Blazed-grating profile at small deflection: phi = k * alpha * x
        // with alpha = sin(theta) (textbook Eq. 3.5 direction cosines).
        let theta = 0.02f64;
        let k = 2.0 * PI / lambda();
        let alpha = theta.sin();
        let band = BandField::uniform(1.0, 0.0, 101, 101, 1e-3, 1e-3, lambda()).unwrap();
        let mask = PhaseMask::new(
            move |x, _y| k * alpha * x,
            None,
            ClearAperture::Field,
            (0.0, 0.0),
        )
        .unwrap();
        let out = mask.apply(&band).unwrap();
        let want = BandField::plane_wave(theta, 0.0, 1.0, 101, 101, 1e-3, 1e-3, lambda()).unwrap();
        for (a, b) in out.field().samples().iter().zip(want.field().samples()) {
            // Closure multiplies (k * alpha) * x vs plane_wave's k * (alpha * x):
            // operand order differs, so compare at a realistic rounding margin
            // for phase magnitudes ~1e4 rad.
            assert!((a - b).norm() < 1e-9);
        }
    }

    #[test]
    fn doe_lens_closure_equals_the_thin_lens() {
        // Textbook Sec. 4.2.1: the diffractive lens IS the quadratic phase
        // mask — a DOE-lens closure must reproduce ThinLens sample-exactly.
        let f = 0.1;
        let a = 20e-3;
        let k = 2.0 * PI / lambda();
        let band = band_101();
        let doe = PhaseMask::new(
            move |x, y| -k * (x * x + y * y) / (2.0 * f),
            None,
            ClearAperture::Circle(a),
            (0.0, 0.0),
        )
        .unwrap();
        let lens = ThinLens::fixed(f, None, ClearAperture::Circle(a), (0.0, 0.0)).unwrap();
        let out_doe = doe.apply(&band).unwrap();
        let out_lens = lens.apply(&band).unwrap();
        assert_eq!(out_doe.field().samples(), out_lens.field().samples());
    }

    #[test]
    fn spiral_phase_is_unit_modulus_wrapped() {
        // Optical-vortex mask: phi = l * atan2(y, x), l = 3.
        let l = 3.0;
        let band = band_101();
        let input_phase = PI / 3.0;
        let mask = PhaseMask::new(
            move |x, y| l * y.atan2(x),
            None,
            ClearAperture::Field,
            (0.0, 0.0),
        )
        .unwrap();
        let out = mask.apply(&band).unwrap();
        for iy in 0..101usize {
            for ix in 0..101 {
                let (x, y) = band.field().coord(ix, iy);
                let s = out.field().sample(ix, iy);
                // Pure phase: modulus equals the input amplitude everywhere.
                assert!((s.norm() - 1.0).abs() < 1e-12);
                let want = input_phase + l * y.atan2(x);
                let err = (s.arg() - want + PI).rem_euclid(2.0 * PI) - PI;
                assert!(err.abs() < 1e-9, "vortex phase at ({ix},{iy}): {err}");
            }
        }
    }

    #[test]
    fn non_finite_closure_is_rejected_with_position() {
        let band = BandField::uniform(1.0, 0.0, 16, 16, 1e-3, 1e-3, lambda()).unwrap();
        for bad in [f64::NAN, f64::INFINITY] {
            let mask =
                PhaseMask::new(move |_x, _y| bad, None, ClearAperture::Field, (0.0, 0.0)).unwrap();
            match mask.apply(&band) {
                Err(OpticsError::InvalidPhase { x, y, phase }) => {
                    assert!(!phase.is_finite());
                    // Row-major scan: first acted sample is (ix=0, iy=0).
                    assert_eq!((x, y), band.field().coord(0, 0));
                }
                other => panic!("expected InvalidPhase, got {other:?}"),
            }
        }
    }

    #[test]
    fn mask_taxonomy_and_aperture_rules() {
        // Stopped form: extent describes the mask (task-c precedent).
        let stopped: &dyn Element = &PhaseMask::new(
            |_x, _y| 0.0,
            None,
            ClearAperture::Circle(20e-3),
            (1e-3, -1e-3),
        )
        .unwrap();
        assert!(matches!(stopped.kind(), ElementKind::Diffractive));
        assert!(matches!(stopped.shape(), ElementShape::Circle));
        assert!(matches!(
            stopped.extent(),
            Extent::Aperture(r) if (r - 20e-3).abs() < 1e-18
        ));
        // Field-wide special case.
        let wide: &dyn Element =
            &PhaseMask::new(|_x, _y| 0.0, None, ClearAperture::Field, (0.0, 0.0)).unwrap();
        assert!(matches!(wide.shape(), ElementShape::Unbounded));
        assert!(matches!(wide.extent(), Extent::Unbounded));
        // Constructor validation.
        assert!(matches!(
            PhaseMask::new(|_x, _y| 0.0, None, ClearAperture::Circle(0.0), (0.0, 0.0)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        assert!(matches!(
            PhaseMask::new(|_x, _y| 0.0, None, ClearAperture::Field, (f64::NAN, 0.0)),
            Err(OpticsError::InvalidAxis { .. })
        ));
        // Oversized footprint rejected, not truncated.
        let band = band_101();
        let big =
            PhaseMask::new(|_x, _y| 0.5, None, ClearAperture::Circle(55e-3), (0.0, 0.0)).unwrap();
        assert!(matches!(
            big.apply(&band),
            Err(OpticsError::OutsideField { half_x: 55e-3, .. })
        ));
        // phase_at probe.
        let m = PhaseMask::new(|x, y| x - y, None, ClearAperture::Field, (0.0, 0.0)).unwrap();
        assert_eq!(m.phase_at(2.0, 0.5), 1.5);
    }

    #[test]
    fn clear_aperture_geometry_helpers() {
        // contains(): hard edge, boundary inclusive, every variant.
        assert!(ClearAperture::Field.contains(1e9, -1e9));
        assert!(ClearAperture::Circle(2.0).contains(2.0, 0.0));
        assert!(!ClearAperture::Circle(2.0).contains(2.1, 0.0));
        assert!(ClearAperture::Ellipse { a: 3.0, b: 2.0 }.contains(0.0, 2.0));
        assert!(!ClearAperture::Ellipse { a: 3.0, b: 2.0 }.contains(3.0, 2.0));
        assert!(
            ClearAperture::Rectangle {
                half_x: 3.0,
                half_y: 2.0
            }
            .contains(-3.0, 2.0)
        );
        assert!(
            !ClearAperture::Rectangle {
                half_x: 3.0,
                half_y: 2.0
            }
            .contains(3.0, 2.001)
        );
        // Conservative extent_radius for regime gating: larger half-side.
        assert_eq!(ClearAperture::Field.extent_radius(), None);
        assert_eq!(ClearAperture::Circle(1.5).extent_radius(), Some(1.5));
        assert_eq!(
            ClearAperture::Ellipse { a: 1.0, b: 2.5 }.extent_radius(),
            Some(2.5)
        );
        assert_eq!(
            ClearAperture::Rectangle {
                half_x: 3.0,
                half_y: 1.0
            }
            .extent_radius(),
            Some(3.0)
        );
        // Taxonomy mapping.
        assert!(matches!(
            ClearAperture::Field.shape(),
            ElementShape::Unbounded
        ));
        assert!(matches!(
            ClearAperture::Circle(1.0).shape(),
            ElementShape::Circle
        ));
        assert!(matches!(
            ClearAperture::Ellipse { a: 1.0, b: 2.0 }.shape(),
            ElementShape::Ellipse
        ));
        assert!(matches!(
            ClearAperture::Rectangle {
                half_x: 1.0,
                half_y: 2.0
            }
            .shape(),
            ElementShape::Rectangle
        ));
        // Validation split: Field checks only the axis; stopped forms the
        // full footprint.
        assert!(matches!(
            ClearAperture::Field.validate((f64::NAN, 0.0)),
            Err(OpticsError::InvalidAxis { .. })
        ));
        assert!(matches!(
            ClearAperture::Circle(-1.0).validate((0.0, 0.0)),
            Err(OpticsError::InvalidAperture { .. })
        ));
        assert!(matches!(
            ClearAperture::Rectangle {
                half_x: 1.0,
                half_y: 0.0
            }
            .validate((0.0, 0.0)),
            Err(OpticsError::InvalidAperture { half_y: 0.0, .. })
        ));
    }

    #[test]
    fn rectangular_stop_masks_a_doe_window() {
        let band = band_101();
        let rect = ClearAperture::Rectangle {
            half_x: 40e-3,
            half_y: 20e-3,
        };
        let mask = PhaseMask::new(|_x, _y| 0.5, None, rect, (0.0, 0.0)).unwrap();
        let out = mask.apply(&band).unwrap();
        let f = out.field();
        let pass = band.field().sample(50, 50);
        // Window corner (x = half_x, y = half_y) is inclusive: constant
        // 0.5 rad shift, modulus preserved.
        let inside = f.sample(90, 70);
        assert!((inside.norm() - pass.norm()).abs() < 1e-12);
        let err = (inside.arg() - (pass.arg() + 0.5) + PI).rem_euclid(2.0 * PI) - PI;
        assert!(err.abs() < 1e-12);
        // One step past either edge is zero.
        assert_eq!(f.sample(91, 70), Complex64::ZERO); // x = 41 mm > half_x
        assert_eq!(f.sample(90, 71), Complex64::ZERO); // y = 21 mm > half_y
        // Uniform input: transmitted power = window area / grid area.
        let ratio = f.power() / band.field().power();
        let expected = (80.0e-3 * 40.0e-3) / (100.0e-3 * 100.0e-3);
        let rel = ((ratio - expected) / expected).abs();
        assert!(rel < 0.03, "rectangular window area error {rel}");
        // Asymmetric oversize: half_x fits, half_y does not.
        let big = PhaseMask::new(
            |_x, _y| 0.0,
            None,
            ClearAperture::Rectangle {
                half_x: 40e-3,
                half_y: 55e-3,
            },
            (0.0, 0.0),
        )
        .unwrap();
        assert!(matches!(
            big.apply(&band),
            Err(OpticsError::OutsideField {
                half_x: 40e-3,
                half_y: 55e-3,
                ..
            })
        ));
        // Taxonomy: rectangle shape, conservative extent (larger half-side).
        let el: &dyn Element = &mask;
        assert!(matches!(el.shape(), ElementShape::Rectangle));
        assert!(matches!(
            el.extent(),
            Extent::Aperture(r) if (r - 40e-3).abs() < 1e-18
        ));
    }

    #[test]
    fn sawtooth_phase_wraps_mod_one() {
        // Kinoform (depth 2pi, psi=0): phi = 2pi*frac(x/Lambda). Check the
        // profile repeats with period Lambda and ramps 0 -> 2pi across it.
        let period = 100e-6;
        let g = BlazedGrating::kinoform(period, Angle::zero(), ClearAperture::Field, (0.0, 0.0))
            .unwrap();
        // Sample the ramp at x = 0, quarter, half, three-quarter period.
        let at = |x: f64| g.phase_at(x, 0.0);
        assert!(at(0.0).abs() < 1e-12);
        assert!((at(period / 4.0) - PI / 2.0).abs() < 1e-12);
        assert!((at(period / 2.0) - PI).abs() < 1e-12);
        assert!((at(3.0 * period / 4.0) - 3.0 * PI / 2.0).abs() < 1e-12);
        // Periodicity: one full period wraps back to 0 (mod 2pi).
        assert!(at(period).abs() < 1e-12);
        assert!((at(-period / 4.0) - 3.0 * PI / 2.0).abs() < 1e-12); // frac(-0.25)=0.75
    }

    #[test]
    fn orientation_rotates_the_grating_vector() {
        // psi = pi/2 (CCW about z): the element-frame x' axis is lab +y, so
        // the phase depends on lab y and is constant along lab x.
        let period = 100e-6;
        let band = BandField::uniform(1.0, 0.0, 64, 64, 5e-6, 5e-6, lambda()).unwrap();
        let g = BlazedGrating::kinoform(period, Angle::pi_half(), ClearAperture::Field, (0.0, 0.0))
            .unwrap();
        // phase_at is element-frame: the ramp runs along frame x'.
        assert!((g.phase_at(period / 4.0, 0.0) - PI / 2.0).abs() < 1e-12);
        assert!(g.phase_at(0.0, period / 4.0).abs() < 1e-12);
        // apply works in lab coordinates: frame (x', y') = R(-pi/2)(x, y) = (y, -x).
        let out = g.apply(&band).unwrap();
        for (ix, iy) in [(0usize, 0usize), (63, 17), (40, 5), (12, 63)] {
            let (_, y) = band.field().coord(ix, iy);
            let u = y / period;
            let want = 2.0 * PI * (u - u.floor());
            let err = (out.field().sample(ix, iy).arg() - want + PI).rem_euclid(2.0 * PI) - PI;
            assert!(err.abs() < 1e-9, "rotated blaze at ({ix},{iy}): {err}");
        }
    }

    #[test]
    fn blaze_integer_2pi_depth_is_a_pure_tilted_plane_wave() {
        // Depth Phi = 2 pi m => e^{i phi} == e^{i 2 pi m x/Lambda} exactly
        // (the mod-1 wrap is invisible). Compare a grating to a PhaseMask
        // carrying that plane-wave ramp, m = 1 and m = 2.
        let period = 100e-6;
        let band = BandField::uniform(1.0, 0.0, 64, 64, 5e-6, 5e-6, lambda()).unwrap();
        for m in [1.0f64, 2.0] {
            let g = BlazedGrating::new(
                period,
                2.0 * PI * m,
                Angle::zero(),
                ClearAperture::Field,
                (0.0, 0.0),
            )
            .unwrap();
            let out = g.apply(&band).unwrap();
            let k = 2.0 * PI * m / period;
            for iy in 0..64usize {
                for ix in 0..64 {
                    let (x, _) = band.field().coord(ix, iy);
                    let err =
                        (out.field().sample(ix, iy).arg() - k * x + PI).rem_euclid(2.0 * PI) - PI;
                    assert!(err.abs() < 1e-9, "m={m} at ({ix},{iy}): {err}");
                    assert!((out.field().sample(ix, iy).norm() - 1.0).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn grating_masks_and_validates() {
        let band = band_101();
        let period = 100e-3; // coarse so the stop dominates the check
        let g = BlazedGrating::kinoform(
            period,
            Angle::zero(),
            ClearAperture::Circle(30e-3),
            (0.0, 0.0),
        )
        .unwrap();
        let out = g.apply(&band).unwrap();
        // Hard-edge stop: (x = 30 mm, y = 0) inclusive; one past is zero.
        assert!((out.field().sample(80, 50).norm() - 1.0).abs() < 1e-12);
        assert_eq!(out.field().sample(81, 50), Complex64::ZERO);
        assert_eq!(out.field().sample(100, 100), Complex64::ZERO);
        // Oversized footprint rejected.
        let big = BlazedGrating::kinoform(
            period,
            Angle::zero(),
            ClearAperture::Circle(55e-3),
            (0.0, 0.0),
        )
        .unwrap();
        assert!(matches!(
            big.apply(&band),
            Err(OpticsError::OutsideField { half_x: 55e-3, .. })
        ));
        // Constructor validation: period/depth/orientation are grating
        // geometry; the stop keeps the shared aperture errors.
        for bad_period in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                BlazedGrating::new(bad_period, PI, Angle::zero(), ClearAperture::Field, (0.0, 0.0)),
                Err(OpticsError::InvalidGrating { period, .. })
                    if period == bad_period || bad_period.is_nan()
            ));
        }
        assert!(matches!(
            BlazedGrating::new(period, f64::NAN, Angle::zero(), ClearAperture::Field, (0.0, 0.0)),
            Err(OpticsError::InvalidGrating { depth, .. }) if depth.is_nan()
        ));
        assert!(matches!(
            BlazedGrating::new(period, PI, Angle::new(f64::INFINITY), ClearAperture::Field, (0.0, 0.0)),
            Err(OpticsError::InvalidGrating { orientation, .. })
                if orientation.is_nan() // Angle wraps non-finite to NaN
        ));
        assert!(matches!(
            BlazedGrating::new(
                period,
                PI,
                Angle::zero(),
                ClearAperture::Circle(0.0),
                (0.0, 0.0)
            ),
            Err(OpticsError::InvalidAperture { .. })
        ));
        // Taxonomy.
        let el: &dyn Element = &g;
        assert!(matches!(el.kind(), ElementKind::Diffractive));
        assert!(matches!(el.shape(), ElementShape::Circle));
        assert!(matches!(
            el.extent(),
            Extent::Aperture(r) if (r - 30e-3).abs() < 1e-18
        ));
    }

    #[test]
    fn grating_worst_phase_step_closed_form_matches_sweep() {
        // kinoform 2pi along x: ramp step = 2pi * dx / period (no wrap
        // because e^{i 2pi frac} = e^{i 2pi u}); keep it below pi.
        let period = 51.2e-6;
        let d = 20e-6; // 2pi * 20/51.2 = 2.45 rad < pi
        let band = BandField::uniform(1.0, 0.0, 64, 64, d, d, lambda()).unwrap();
        let g = BlazedGrating::kinoform(period, Angle::zero(), ClearAperture::Field, (0.0, 0.0))
            .unwrap();
        let out = g.apply(&band).unwrap();
        let field = out.field();
        let mut max_step = 0.0f64;
        for iy in 0..64usize {
            for ix in 1..64 {
                let s = (field.sample(ix, iy) / field.sample(ix - 1, iy))
                    .arg()
                    .abs();
                max_step = max_step.max(s);
            }
        }
        let closed = g.worst_phase_step(&band);
        assert!(closed < PI, "geometry must stay unwrapped, got {closed}");
        assert!(
            (closed - max_step).abs() < 1e-9,
            "closed {closed} vs sweep {max_step}"
        );
    }

    #[test]
    fn orientation_defaults_to_typical_and_validates() {
        // None == Some(zero) == axis-aligned typical orientation.
        let ell = EllipticalAperture::new((2e-3, 1e-3), None, (0.0, 0.0)).unwrap();
        assert_eq!(*ell.orientation(), 0.0);
        let same = EllipticalAperture::new((2e-3, 1e-3), Some(Angle::zero()), (0.0, 0.0)).unwrap();
        assert_eq!(*same.orientation(), *ell.orientation());
        // -pi/2 wraps to 3pi/2 (implicit normalization, equal-modulo).
        let w = RectangularAperture::new((2e-3, 1e-3), Some(Angle::new(-PI / 2.0)), (0.0, 0.0))
            .unwrap();
        assert!((*w.orientation() - 3.0 * PI / 2.0).abs() < 1e-12);
        // Non-finite orientation rejected by every oriented constructor
        // (Angle wraps non-finite inputs to NaN).
        let nan = Some(Angle::new(f64::NAN));
        assert!(matches!(
            EllipticalAperture::new((1e-3, 1e-3), nan, (0.0, 0.0)),
            Err(OpticsError::InvalidOrientation { .. })
        ));
        assert!(matches!(
            RectangularAperture::new((1e-3, 1e-3), nan, (0.0, 0.0)),
            Err(OpticsError::InvalidOrientation { .. })
        ));
        assert!(matches!(
            ThinLens::fixed(0.1, nan, ClearAperture::Field, (0.0, 0.0)),
            Err(OpticsError::InvalidOrientation { .. })
        ));
        assert!(matches!(
            PhaseMask::new(|_x, _y| 0.0, nan, ClearAperture::Field, (0.0, 0.0)),
            Err(OpticsError::InvalidOrientation { .. })
        ));
    }

    #[test]
    fn rotated_ellipse_aperture() {
        // Semi-axes (50, 20) mm at psi = pi/4: the major axis runs along
        // lab (cos45, sin45); the axis-aligned tip (50 mm, 0) is OUTSIDE.
        let band = band_101();
        let psi = Some(Angle::from_degrees(45.0));
        let ap = EllipticalAperture::new((r50(), 20e-3), psi, (0.0, 0.0)).unwrap();
        let out = ap.apply(&band).unwrap();
        let f = out.field();
        let pass = band.field().sample(50, 50);
        // Along the rotated major axis: (35 mm, 35 mm) projects to x' ~ 49.5,
        // y' = 0 -> inside (boundary-inclusive ellipse).
        assert_eq!(f.sample(85, 85), pass);
        // Old axis tip (50 mm, 0): frame (35.4, -35.4) -> outside the ellipse.
        assert_eq!(f.sample(100, 50), Complex64::ZERO);
        // Rotated bounding box (49.5 mm per side) still fits the 50 mm grid.
        assert!(matches!(ap.shape(), ElementShape::Ellipse));
        // A circle ignores orientation: same mask as the unrotated circle.
        // (Radius chosen with no exact grid-boundary samples, so f64 frame
        // rotation cannot flip a pixel.)
        let circ = EllipticalAperture::new((31.7e-3, 31.7e-3), Some(Angle::pi_half()), (0.0, 0.0))
            .unwrap();
        let plain = EllipticalAperture::circle(31.7e-3, (0.0, 0.0)).unwrap();
        assert_eq!(
            circ.apply(&band).unwrap().field().samples(),
            plain.apply(&band).unwrap().field().samples()
        );
    }

    #[test]
    fn rotated_rectangle_aperture() {
        // Half-widths (40, 20) mm at psi = pi/2: the box swaps axes in lab.
        let band = band_101();
        let ap =
            RectangularAperture::new((40e-3, 20e-3), Some(Angle::pi_half()), (0.0, 0.0)).unwrap();
        let out = ap.apply(&band).unwrap();
        let f = out.field();
        let pass = band.field().sample(50, 50);
        assert_eq!(f.sample(69, 89), pass); // lab (19, 39) mm: frame (39, -19) strictly inside
        assert_eq!(f.sample(91, 70), Complex64::ZERO); // lab (41, 20): frame y' = -41 > 20
        assert_eq!(f.sample(70, 91), Complex64::ZERO); // lab (20, 41): frame x' = 41 > 40
    }

    #[test]
    fn rotated_stop_outside_field_uses_bounding_box() {
        let band = band_101();
        // Rectangle (48, 30) mm fits axis-aligned ...
        let fits = PhaseMask::new(
            |_x, _y| 0.0,
            Some(Angle::zero()),
            ClearAperture::Rectangle {
                half_x: 48e-3,
                half_y: 30e-3,
            },
            (0.0, 0.0),
        )
        .unwrap();
        assert!(fits.apply(&band).is_ok());
        // ... but rotated 45 deg its bounding box reaches 55.2 mm > 50 mm.
        let tilted = PhaseMask::new(
            |_x, _y| 0.0,
            Some(Angle::from_degrees(45.0)),
            ClearAperture::Rectangle {
                half_x: 48e-3,
                half_y: 30e-3,
            },
            (0.0, 0.0),
        )
        .unwrap();
        match tilted.apply(&band) {
            Err(OpticsError::OutsideField { half_x, half_y, .. }) => {
                let b = (48e-3 + 30e-3) * (PI / 4.0).cos(); // (hx + hy) * cos45
                assert!((half_x - b).abs() < 1e-15 && (half_y - b).abs() < 1e-15);
            }
            other => panic!("expected OutsideField, got {other:?}"),
        }
    }

    #[test]
    fn lens_and_grating_apertures_rotate_with_the_element() {
        let band = band_101();
        // Elliptical stop (40, 15) mm on a lens, psi = pi/2: major axis
        // along lab +y. Point (0, 40 mm) passes, (40 mm, 0) is stopped.
        let lens = ThinLens::fixed(
            0.5,
            Some(Angle::pi_half()),
            ClearAperture::Ellipse { a: 40e-3, b: 15e-3 },
            (0.0, 0.0),
        )
        .unwrap();
        let out = lens.apply(&band).unwrap();
        assert!((out.field().sample(50, 90).norm() - 1.0).abs() < 1e-12); // (0, 40 mm)
        assert_eq!(out.field().sample(90, 50), Complex64::ZERO); // (40 mm, 0)
        // Grating rectangular stop aligned with the stripes (psi = pi/2):
        // stripes run along lab y; a tall window follows the rotation.
        let g = BlazedGrating::kinoform(
            100e-3,
            Angle::pi_half(),
            ClearAperture::Rectangle {
                half_x: 10e-3,
                half_y: 40e-3,
            },
            (0.0, 0.0),
        )
        .unwrap();
        let gout = g.apply(&band).unwrap();
        // Frame coords (x', y') = (y, -x): lab (0, 40 mm) -> (40, 0) mm is
        // OUTSIDE the frame box (|x'| = 40 > 10); lab (10 mm, 0) -> (0, -10)
        // is inside.
        assert_eq!(gout.field().sample(50, 90), Complex64::ZERO);
        assert!((gout.field().sample(60, 50).norm() - 1.0).abs() < 1e-12);
    }
}
