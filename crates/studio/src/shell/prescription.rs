//! Prescription view/editor model (spec §3.1–§3.2, task 10.1).
//!
//! The `.zmx` text is the authoritative source (text mode); table edits
//! work on a UI-owned `Vec<ZmxSurface>` clone and rebuild a *simulation*
//! beamline directly with core types — the mapping mirrors the
//! `optics-zemax` builder exactly (same units, same flat/curved split,
//! same medium assignment). This is allowed plumbing (spec §6): no new
//! physics, only the documented `RefractiveSurface` + `Beamline::gap`
//! composition the spec prescribes.

use optics_core::beamline::{Beamline, BeamlineError, Psf};
use optics_core::element::{ClearAperture, RefractiveSurface, SurfaceProfile};
use optics_core::field::BandField;
use optics_zemax::{Registry, ZmxSurface};

use super::dispatch::{self, DispatchError, EngineSettings};

/// Planning failures for the simulation beamline.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// Glass/registry failure (typed `ZemaxError` `Display`).
    #[error(transparent)]
    Zemax(#[from] optics_zemax::ZemaxError),
    /// Element construction failure (typed `OpticsError` `Display`).
    #[error(transparent)]
    Core(#[from] optics_core::element::OpticsError),
    /// Fewer than two surfaces: nothing to trace.
    #[error("prescription has {0} surfaces; need at least an object and an image plane")]
    TooShort(usize),
}

/// One schematic node: a refracting surface or a pure gap.
#[derive(Clone, Debug)]
pub enum StripNode {
    /// Refracting surface at `z_m` (m from the first surface).
    Surface {
        /// `SURF` index in the prescription.
        index: usize,
        /// z-position along the axis (m).
        z_m: f64,
        /// Semi-diameter (mm); 0.0 = unbounded (`DIAM 0`).
        semi_mm: f64,
        /// Refractive index just before the surface (λ-resolved).
        front_n: f64,
        /// Refractive index just after the surface (λ-resolved).
        back_n: f64,
        /// Vertex curvature (1/mm).
        curvature_per_mm: f64,
        /// `STOP` flag.
        stop: bool,
        /// `COMM` text.
        comment: String,
    },
    /// Gap of `z_m` meters through index `n` following a surface.
    Gap {
        /// Gap length (m).
        z_m: f64,
        /// Refractive index of the gap medium.
        n: f64,
    },
}

/// The schematic strip model for the Beamline preview panel (spec §3.1:
/// "Schematic strip of the element stack (apertures, surfaces, gaps,
/// media)").
#[derive(Clone, Debug, Default)]
pub struct StripModel {
    /// Nodes in axial order.
    pub nodes: Vec<StripNode>,
    /// Total track length (m): sum of all gaps after surface 1.
    pub track_m: f64,
    /// Largest semi-diameter (mm) across surfaces, for strip scaling.
    pub max_semi_mm: f64,
    /// Stop-surface index, when flagged.
    pub stop_index: Option<usize>,
}

/// A rebuilt simulation beamline: owned surfaces plus the step list
/// referencing them, ready to trace with the dispatch adapter.
///
/// `RefractiveSurface` holds boxed closures, so the plan is neither
/// `Clone` nor `Debug`; the app rebuilds it per preview run and keeps only
/// the strip model and the resulting PSF in state.
pub struct SimPlan {
    surfaces: Vec<RefractiveSurface>,
    steps: Vec<(Option<usize>, f64, f64)>,
    strip: StripModel,
    /// `(prescription surface index, element index)` for each curved
    /// surface, used to flag per-surface phase steps in the table (§5.3).
    curved: Vec<(usize, usize)>,
}

impl SimPlan {
    /// Number of refracting elements.
    pub fn element_count(&self) -> usize {
        self.surfaces.len()
    }

    /// The schematic model.
    pub fn strip(&self) -> &StripModel {
        &self.strip
    }

    /// Per-surface `worst_phase_step` for curved surfaces (spec §5.3
    /// shading column): `(SURF index, worst phase step rad)`.
    pub fn phase_steps(&self, band: &BandField) -> Vec<(usize, f64)> {
        self.curved
            .iter()
            .map(|&(surf, si)| (surf, self.surfaces[si].worst_phase_step(band)))
            .collect()
    }

    /// Trace the plan and return the detector-plane PSF.
    pub fn psf<E>(
        &self,
        band: &BandField,
        engine: impl FnMut(&BandField, f64, f64) -> Result<BandField, E>,
    ) -> Result<Psf, BeamlineError<E>>
    where
        E: std::error::Error,
    {
        let mut bl = Beamline::new();
        for (k, z, medium) in &self.steps {
            match k {
                Some(i) => bl.push_medium(&self.surfaces[*i], *z, *medium),
                None => bl.gap(*z, *medium),
            };
        }
        bl.psf(band, engine)
    }

    /// Trace through the dispatch adapter.
    pub fn psf_with(
        &self,
        band: &BandField,
        settings: &EngineSettings,
    ) -> Result<Psf, BeamlineError<DispatchError>> {
        self.psf(band, |b, z, m| dispatch::propagate(settings, b, z, m))
    }

    /// The final gap distance (m) and medium — the detector plane the PSF
    /// lands on, used for the sampling banner.
    pub fn last_step(&self) -> Option<(f64, f64)> {
        self.steps.last().map(|&(_, z, m)| (z, m))
    }
}

/// Rebuild the simulation beamline from the (possibly edited) surface
/// list, evaluated at `lambda_um`. Mirrors `optics-zemax`'s plan():
/// surfaces `1..last` refract, the last surface is the image plane, flat
/// surfaces become pure gaps, `CURV` 1/mm → meters radius, mm → m.
pub fn build_plan(
    surfaces: &[ZmxSurface],
    registry: &Registry,
    lambda_um: f64,
) -> Result<SimPlan, PlanError> {
    let last = match surfaces.len() {
        0..=1 => return Err(PlanError::TooShort(surfaces.len())),
        n => n - 1,
    };
    let mut plan = SimPlan {
        surfaces: Vec::new(),
        steps: Vec::new(),
        strip: StripModel::default(),
        curved: Vec::new(),
    };
    let mut z_cursor = 0.0f64;
    for i in 1..last {
        let s = &surfaces[i];
        let front_glass = if i == 1 {
            None
        } else {
            surfaces[i - 1].glass.as_deref()
        };
        let front_n = registry.index_at(front_glass, lambda_um)?;
        let back_n = registry.index_at(s.glass.as_deref(), lambda_um)?;
        let ki = if s.curvature_mm != 0.0 {
            let aperture = if s.semi_diam_mm > 0.0 {
                ClearAperture::Circle(s.semi_diam_mm * 1e-3)
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
            plan.surfaces.push(RefractiveSurface::new(
                profile,
                registry.material(front_glass)?,
                registry.material(s.glass.as_deref())?,
                None,
                None,
                aperture,
                (0.0, 0.0),
            )?);
            let si = plan.surfaces.len() - 1;
            plan.curved.push((s.index, si));
            Some(si)
        } else {
            None
        };
        let z = s.thickness_mm.unwrap_or(0.0) * 1e-3;
        plan.steps.push((ki, z, back_n));
        plan.strip.nodes.push(StripNode::Surface {
            index: s.index,
            z_m: z_cursor,
            semi_mm: s.semi_diam_mm,
            front_n,
            back_n,
            curvature_per_mm: s.curvature_mm,
            stop: s.stop,
            comment: s.comment.clone(),
        });
        plan.strip.nodes.push(StripNode::Gap { z_m: z, n: back_n });
        z_cursor += z;
        plan.strip.track_m += z;
        if s.semi_diam_mm > plan.strip.max_semi_mm {
            plan.strip.max_semi_mm = s.semi_diam_mm;
        }
        if s.stop {
            plan.strip.stop_index = Some(s.index);
        }
    }
    Ok(plan)
}

/// The file's own image distance (mm): the thickness of the last
/// refracting surface — the paraxial-card cross-check line (spec §3.1).
pub fn image_distance_mm(surfaces: &[ZmxSurface]) -> Option<f64> {
    if surfaces.len() < 3 {
        return None;
    }
    surfaces[surfaces.len() - 2].thickness_mm
}

/// Compact per-surface table row for the prescription view (spec §3.1:
/// index, type, stop, curvature, conic, semi-diam, thickness, glass,
/// comment) — display strings with the units the spec names.
#[derive(Clone, Debug, PartialEq)]
pub struct TableRow {
    /// `SURF` index.
    pub index: usize,
    /// `TYPE` keyword.
    pub kind: String,
    /// `STOP` flag.
    pub stop: bool,
    /// Curvature (1/mm), formatted.
    pub curvature: String,
    /// Conic constant, formatted.
    pub conic: String,
    /// Semi-diameter (mm), formatted.
    pub semi_diam: String,
    /// Thickness (mm) or `INF`.
    pub thickness: String,
    /// Glass name or `—`.
    pub glass: String,
    /// `COMM` text.
    pub comment: String,
}

impl TableRow {
    /// Format one prescription surface for the table.
    pub fn of(s: &ZmxSurface) -> Self {
        Self {
            index: s.index,
            kind: s.kind.clone(),
            stop: s.stop,
            curvature: format!("{:.*}", 6, s.curvature_mm),
            conic: format!("{:.4}", s.conic),
            semi_diam: format!("{:.4}", s.semi_diam_mm),
            thickness: match s.thickness_mm {
                Some(t) => format!("{:.4}", t),
                None => "INF".to_string(),
            },
            glass: s.glass.clone().unwrap_or_else(|| "—".to_string()),
            comment: s.comment.clone(),
        }
    }
}

/// Builds the schematic strip from an inverse-design [`Design`]: reads
/// each element's `kind`/`extent`/`material` through the core `Element`
/// trait (spec §3.1 "Beamline preview" row) — no engine execution.
pub fn strip_from_design(design: &optics_inverse::Design, lambda_m: f64) -> StripModel {
    use optics_core::element::Extent;
    let mut strip = StripModel::default();
    let mut z_cursor = 0.0f64;
    for (element, step) in design.elements().iter().zip(design.steps()) {
        let radius_mm = match element.extent() {
            Extent::Aperture(r) => r * 1e3,
            Extent::Thick { aperture, .. } => aperture * 1e3,
            Extent::Unbounded => 0.0,
        };
        let (front, back) = element.material();
        let label = match element.kind() {
            optics_core::element::ElementKind::Aperture => "stop",
            optics_core::element::ElementKind::Lens => "lens",
            optics_core::element::ElementKind::Mirror => "mirror",
            optics_core::element::ElementKind::Splitter => "splitter",
            optics_core::element::ElementKind::Source => "source",
            optics_core::element::ElementKind::Plane => "plane",
            optics_core::element::ElementKind::Prism => "prism",
            optics_core::element::ElementKind::Window => "window",
            optics_core::element::ElementKind::Waveplate => "waveplate",
            optics_core::element::ElementKind::Diffractive => "diffractive",
            optics_core::element::ElementKind::Waveguide => "waveguide",
        };
        strip.nodes.push(StripNode::Surface {
            index: strip.nodes.len(),
            z_m: z_cursor,
            semi_mm: radius_mm,
            front_n: front.n_at(lambda_m, 0.0),
            back_n: back.n_at(lambda_m, 0.0),
            curvature_per_mm: 0.0,
            stop: false,
            comment: label.to_string(),
        });
        z_cursor += step.z_after;
        strip.track_m += step.z_after;
        strip.nodes.push(StripNode::Gap {
            z_m: step.z_after,
            n: step.medium,
        });
        if radius_mm > strip.max_semi_mm {
            strip.max_semi_mm = radius_mm;
        }
    }
    strip
}
