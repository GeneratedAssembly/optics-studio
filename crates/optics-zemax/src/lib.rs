//! # optics-zemax — Zemax prescription import for `optics-core`
//!
//! Parses text-format Zemax prescriptions (`.zmx`, line-oriented ASCII,
//! CRLF-tolerant) into a surface/glass description, resolves glasses
//! through a built-in Sellmeier catalog, and builds `optics_core`
//! systems: per-surface [`optics_core::element::RefractiveSurface`]s
//! composed through a [`optics_core::beamline::Beamline`] with medium
//! steps (the Phase-6 composition model). Paraxial reports (EFL/BFL) let
//! prescriptions be checked against their own data (an imported file's
//! image distance IS the paraxial back focal length at the primary
//! wavelength — used as a golden validation).
//!
//! Scope (tasks/plan.md Phase 8): sequential-mode `STANDARD` surfaces,
//! circular apertures (semi-diameters), air/glass media from the built-in
//! catalog, wavelengths from `WAVM`/`PWAV`. Binary `.zar` archives,
//! coatings, aspheres beyond even terms, non-sequential modes, and
//! coordinate breaks are out of scope (documented rejections).
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let sys = optics_zemax::ImportedSystem::load(concat!(
//!     env!("CARGO_MANIFEST_DIR"),
//!     "/tests/data/ACA254-100-UV.zmx",
//! ))?;
//! assert_eq!(sys.system().surfaces.len(), 6);
//! // Thorlabs ACA254-100-UV: the file's own image distance (89.4896 mm)
//! // is the paraxial BFL at the design center wavelength (0.325 um).
//! let par = sys.paraxial_at(0.325)?;
//! assert!((par.bfl_mm - 89.4896).abs() < 0.2, "bfl {}", par.bfl_mm);
//! # Ok(())
//! # }
//! ```

pub mod builder;
pub mod catalog;
pub mod error;
pub mod zmx;

pub use builder::{ImportedSystem, ParaxialReport};
pub use error::ZemaxError;
pub use zmx::{ZmxSurface, ZmxSystem};
