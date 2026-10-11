use crate::wave::Wave;

pub enum ElementShape {
    Circle,
    Ellipse,
    Rectangle,
    Square,
    // Polygons are defined by n_lines, and angles between them
    Poly((u8, Vec<f32>)),
}

pub enum ElementKind {
    Aperture,
    Lens,
    Mirror,
    Splitter,
    Source,
    Plane,
    Prism,
    Window,
    Waveplate,
    Diffractive,
    Waveguide,
}

/// Extent of an element, derived from its kind.
/// Optical elements only care about their clear aperture;
/// refractive elements additionally carry a central thickness.
pub enum Extent {
    /// Flat elements: clear-aperture radius.
    Aperture(f32),
    /// Refractive elements: clear-aperture radius and central thickness.
    Thick { aperture: f32, thickness: f32 },
}

/// All units are SI: wavelength in meters, angles in radians.
pub struct Material {
    /// Index of refraction as a function of wavelength (m) and incidence angle (rad).
    pub n: Box<dyn Fn(f32, f32) -> f32>,
    /// Absorption coefficient (1/m) as a function of wavelength (m) and incidence angle (rad).
    pub absorption: Box<dyn Fn(f32, f32) -> f32>,
}

impl Material {
    pub fn new(
        n: impl Fn(f32, f32) -> f32 + 'static,
        absorption: impl Fn(f32, f32) -> f32 + 'static,
    ) -> Self {
        Self {
            n: Box::new(n),
            absorption: Box::new(absorption),
        }
    }

    /// Index of refraction at wavelength `lambda` (m) and angle `angle` (rad).
    pub fn n_at(&self, lambda: f32, angle: f32) -> f32 {
        (self.n)(lambda, angle)
    }

    /// Absorption coefficient at wavelength `lambda` (m) and angle `angle` (rad).
    pub fn absorption_at(&self, lambda: f32, angle: f32) -> f32 {
        (self.absorption)(lambda, angle)
    }
}

pub trait Element {
    fn shape(&self) -> &ElementShape;
    fn kind(&self) -> ElementKind;
    // front and back
    fn material(&self) -> (&Material, &Material);
    // Extent is derived from the kind and owns the correct extent object
    fn extent(&self) -> Extent;
    fn transform(&self, wave: &Wave, angle: f32) -> Option<Wave>;
}
