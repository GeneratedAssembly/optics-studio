//! Prescription → `optics-core` system: paraxial analysis and wave-optics
//! tracing (Phase-6 composition model: `RefractiveSurface` screens +
//! medium-carrying `Beamline` gaps, all evaluated at the traced band's
//! wavelength).

use crate::catalog::Registry;
use crate::error::ZemaxError;
use crate::zmx::ZmxSystem;
use optics_core::beamline::{Beamline, BeamlineError, Psf};
use optics_core::element::{ClearAperture, RefractiveSurface, SurfaceProfile};
use optics_core::field::BandField;

/// Paraxial first-order report (millimeters).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParaxialReport {
    /// Effective focal length (1/system power).
    pub efl_mm: f64,
    /// Back focal length: last surface to the paraxial image of an
    /// object at infinity.
    pub bfl_mm: f64,
    /// Primary wavelength (µm) the report was evaluated at.
    pub lambda_um: f64,
}

/// One traced step: optional surface element (index into the plan's
/// surfaces), gap distance (m), gap refractive index.
type Step = (Option<usize>, f64, f64);
/// Two-phase assembly plan: owned surfaces + steps referencing them.
type Plan = (Vec<RefractiveSurface>, Vec<Step>);

/// An imported sequential system ready for analysis and tracing.
#[derive(Clone, Debug)]
pub struct ImportedSystem {
    sys: ZmxSystem,
    registry: Registry,
}

impl ImportedSystem {
    /// Parse + validate importability (surface types, mirrors, aspheric
    /// terms, glass resolvability) up front.
    pub fn parse(text: &str) -> Result<Self, ZemaxError> {
        Self::parse_with(text, Registry::builtin())
    }

    /// Parse against an explicit glass registry (AGF catalogs loaded).
    pub fn parse_with(text: &str, registry: Registry) -> Result<Self, ZemaxError> {
        let sys = ZmxSystem::parse(text)?;
        let s = Self { sys, registry };
        s.validate()?;
        Ok(s)
    }

    /// Read and parse a `.zmx` file (built-in catalog).
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ZemaxError> {
        Self::load_with(path, Registry::builtin())
    }

    /// Read and parse a `.zmx` file against an explicit registry.
    pub fn load_with(
        path: impl AsRef<std::path::Path>,
        registry: Registry,
    ) -> Result<Self, ZemaxError> {
        let p = path.as_ref();
        let text = std::fs::read_to_string(p).map_err(|e| ZemaxError::Io {
            path: p.display().to_string(),
            source: e,
        })?;
        Self::parse_with(&text, registry)
    }

    /// The parsed prescription.
    pub fn system(&self) -> &ZmxSystem {
        &self.sys
    }

    fn validate(&self) -> Result<(), ZemaxError> {
        for s in &self.sys.surfaces {
            if s.kind != "STANDARD" {
                return Err(ZemaxError::UnsupportedSurface {
                    surface: s.index,
                    reason: format!("TYPE {}", s.kind),
                });
            }
            if s.mirror {
                return Err(ZemaxError::UnsupportedSurface {
                    surface: s.index,
                    reason: "mirrors are out of scope (axis fold)".into(),
                });
            }
            if s.has_para {
                return Err(ZemaxError::UnsupportedSurface {
                    surface: s.index,
                    reason: "PARA aspheric terms pending golden data".into(),
                });
            }
            self.registry.validate_glass(s.glass.as_deref())?;
        }
        Ok(())
    }

    fn medium_index_after(&self, surface: usize, lambda_um: f64) -> Result<f64, ZemaxError> {
        self.registry
            .index_at(self.sys.surfaces[surface].glass.as_deref(), lambda_um)
    }

    /// Paraxial matrix report at `lambda_um` (thin-screen composition
    /// model: powers (n₂−n₁)c, reduced-thickness gaps).
    pub fn paraxial_at(&self, lambda_um: f64) -> Result<ParaxialReport, ZemaxError> {
        // (y, nu) ray vectors: R = [[1,0],[-phi,1]], T = [[1, d/n],[0,1]].
        let (mut m11, mut m12, mut m21, mut m22) = (1.0f64, 0.0, 0.0, 1.0);
        let n_before = |i: usize| -> Result<f64, ZemaxError> {
            if i == 0 {
                Ok(1.0)
            } else {
                self.medium_index_after(i - 1, lambda_um)
            }
        };
        // Surfaces 1..=N-2 refract (last surface is the image plane);
        // thickness of surface i is the gap to surface i+1.
        let last = self.sys.surfaces.len() - 1;
        for i in 1..last {
            let s = &self.sys.surfaces[i];
            let nb = n_before(i)?;
            let na = self.medium_index_after(i, lambda_um)?;
            let phi = s.curvature_mm * (na - nb);
            // M = R * M
            m21 -= phi * m11;
            m22 -= phi * m12;
            // Internal gaps only: the LAST refracting surface's thickness
            // is the image distance itself (BFL is measured from it).
            if i < last - 1
                && let Some(d) = s.thickness_mm
            {
                // M = T * M
                m11 += d / na * m21;
                m12 += d / na * m22;
            }
        }
        let power = -m21;
        Ok(ParaxialReport {
            efl_mm: 1.0 / power,
            bfl_mm: -m11 / m21,
            lambda_um,
        })
    }

    /// Paraxial report at the primary wavelength.
    pub fn paraxial(&self) -> Result<ParaxialReport, ZemaxError> {
        self.paraxial_at(self.sys.primary_um())
    }

    /// Trace a [`BandField`] through the system with the caller's engine
    /// (`FnMut(&BandField, z, medium) -> Result<_, E>`), evaluated at the
    /// band's own wavelength (dispersive glasses resolve consistently).
    ///
    /// # Errors
    /// [`TraceError::Setup`] for import/geometry failures,
    /// [`TraceError::Beamline`] for element/engine failures.
    pub fn trace<E>(
        &self,
        band: &BandField,
        engine: impl FnMut(&BandField, f64, f64) -> Result<BandField, E>,
    ) -> Result<BandField, TraceError<E>>
    where
        E: std::error::Error,
    {
        let (surfaces, steps) = self.plan(band.lambda() * 1e6, None)?;
        let mut bl = Beamline::new();
        for (k, z, medium) in &steps {
            match k {
                Some(i) => bl.push_medium(&surfaces[*i], *z, *medium),
                None => bl.gap(*z, *medium),
            };
        }
        bl.trace(band, engine).map_err(TraceError::Beamline)
    }

    /// Trace with every circular stop clamped to `semi_diam_mm` (stop-down
    /// sub-aperture testing; keeps imported stops inside small analysis
    /// grids).
    pub fn trace_subaperture<E>(
        &self,
        band: &BandField,
        semi_diam_mm: f64,
        engine: impl FnMut(&BandField, f64, f64) -> Result<BandField, E>,
    ) -> Result<BandField, TraceError<E>>
    where
        E: std::error::Error,
    {
        let (surfaces, steps) = self.plan(band.lambda() * 1e6, Some(semi_diam_mm))?;
        let mut bl = Beamline::new();
        for (k, z, medium) in &steps {
            match k {
                Some(i) => bl.push_medium(&surfaces[*i], *z, *medium),
                None => bl.gap(*z, *medium),
            };
        }
        bl.trace(band, engine).map_err(TraceError::Beamline)
    }

    /// Coherent PSF at the detector plane (after the last gap), traced
    /// with the caller's engine.
    pub fn psf<E>(
        &self,
        band: &BandField,
        engine: impl FnMut(&BandField, f64, f64) -> Result<BandField, E>,
    ) -> Result<Psf, TraceError<E>>
    where
        E: std::error::Error,
    {
        let (surfaces, steps) = self.plan(band.lambda() * 1e6, None)?;
        let mut bl = Beamline::new();
        for (k, z, medium) in &steps {
            match k {
                Some(i) => bl.push_medium(&surfaces[*i], *z, *medium),
                None => bl.gap(*z, *medium),
            };
        }
        bl.psf(band, engine).map_err(TraceError::Beamline)
    }

    /// PSF with every circular stop clamped to `semi_diam_mm`.
    pub fn psf_subaperture<E>(
        &self,
        band: &BandField,
        semi_diam_mm: f64,
        engine: impl FnMut(&BandField, f64, f64) -> Result<BandField, E>,
    ) -> Result<Psf, TraceError<E>>
    where
        E: std::error::Error,
    {
        let (surfaces, steps) = self.plan(band.lambda() * 1e6, Some(semi_diam_mm))?;
        let mut bl = Beamline::new();
        for (k, z, medium) in &steps {
            match k {
                Some(i) => bl.push_medium(&surfaces[*i], *z, *medium),
                None => bl.gap(*z, *medium),
            };
        }
        bl.psf(band, engine).map_err(TraceError::Beamline)
    }

    /// Two-phase plan: owned [`RefractiveSurface`]s (curved surfaces only)
    /// and steps referencing them by index (flat surfaces become pure
    /// gaps). Phase separation avoids any self-referential borrow.
    fn plan(&self, lambda_um: f64, sub_aperture_mm: Option<f64>) -> Result<Plan, ZemaxError> {
        let last = self.sys.surfaces.len() - 1;
        let mut surfaces = Vec::new();
        let mut steps = Vec::new();
        for i in 1..last {
            let s = &self.sys.surfaces[i];
            let ki = if s.curvature_mm != 0.0 {
                let semi = match (s.semi_diam_mm, sub_aperture_mm) {
                    (semi, Some(cap)) => semi.min(cap),
                    (semi, _) => semi,
                };
                let aperture = if semi > 0.0 {
                    ClearAperture::Circle(semi * 1e-3)
                } else {
                    ClearAperture::Field
                };
                let profile = if s.conic == 0.0 {
                    SurfaceProfile::Spherical {
                        radius: 1e-3 / s.curvature_mm,
                    }
                } else {
                    SurfaceProfile::Conic {
                        radius: 1e-3 / s.curvature_mm,
                        conic: s.conic,
                    }
                };
                let front_glass = if i == 1 {
                    None
                } else {
                    self.sys.surfaces[i - 1].glass.as_deref()
                };
                surfaces.push(RefractiveSurface::new(
                    profile,
                    self.registry.material(front_glass)?,
                    self.registry.material(s.glass.as_deref())?,
                    None,
                    None,
                    aperture,
                    (0.0, 0.0),
                )?);
                Some(surfaces.len() - 1)
            } else {
                None
            };
            let medium = self.medium_index_after(i, lambda_um)?;
            let z = s.thickness_mm.unwrap_or(0.0) * 1e-3;
            steps.push((ki, z, medium));
        }
        Ok((surfaces, steps))
    }
}

/// Trace/PSF failures: import setup or beamline execution.
#[derive(Debug, thiserror::Error)]
pub enum TraceError<E: std::error::Error> {
    /// The prescription could not be planned (glass/geometry).
    #[error(transparent)]
    Setup(#[from] ZemaxError),
    /// An element `apply` or the engine closure failed mid-trace.
    #[error(transparent)]
    Beamline(#[from] BeamlineError<E>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aca() -> ImportedSystem {
        ImportedSystem::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/ACA254-100-UV.zmx"
        ))
        .unwrap()
    }

    #[test]
    fn paraxial_matches_published_focal_length() {
        let sys = aca();
        let p = sys.paraxial().unwrap();
        // Thorlabs ACA254-100-UV: f = 100 mm nominal; built-in published
        // coefficient sets (Li 1980, Malitson) vs Zemax's proprietary
        // SCHOTTIR catalog leave ~3% EFL residual (AGF import pending).
        assert!((p.efl_mm - 100.0).abs() < 3.5, "EFL {} vs 100", p.efl_mm);
        // The file's own image distance is the paraxial BFL at the design
        // CENTER wavelength (0.325 um) — agreement to 0.15% validates the
        // parser, media assignment, signs, and matrix end-to-end.
        let file_bfl = sys.system().surfaces[4].thickness_mm.unwrap();
        let pc = sys.paraxial_at(0.325).unwrap();
        assert!(
            (pc.bfl_mm - file_bfl).abs() < 0.002 * file_bfl,
            "BFL {} vs file {}",
            pc.bfl_mm,
            file_bfl
        );
        // Achromat design intent: EFL spread across the three design
        // wavelengths far smaller than a singlet's (~5% in UV).
        let e1 = sys.paraxial_at(0.2825).unwrap().efl_mm;
        let e3 = sys.paraxial_at(0.3675).unwrap().efl_mm;
        assert!(
            (e1 - e3).abs() / e1 < 0.015,
            "achromatic spread {}%",
            100.0 * (e1 - e3).abs() / e1
        );
    }

    #[test]
    fn rejects_unsupported_surfaces_cleanly() {
        let mirror = "MODE SEQ\nWAVM 1 0.55 1\nPWAV 1\nSURF 0\nSURF 1\n  MIRR 1 0\nSURF 2\n";
        assert!(matches!(
            ImportedSystem::parse(mirror),
            Err(ZemaxError::UnsupportedSurface { .. })
        ));
        let glassy = "MODE SEQ\nWAVM 1 0.55 1\nPWAV 1\nSURF 0\nSURF 1\n  GLAS UNOBTAINIUM\n  CURV 0.01\nSURF 2\n";
        assert!(matches!(
            ImportedSystem::parse(glassy),
            Err(ZemaxError::UnknownGlass(_))
        ));
    }
}
