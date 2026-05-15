pub mod field;
pub mod metadata;
pub mod traits;

pub use field::{ComplexField, FieldComponents};
pub use metadata::{CoherenceState, GridSpacing, SpectralSample, Spectrum};
pub use traits::{Field, OpticalElement, Source};
