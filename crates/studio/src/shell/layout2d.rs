//! 2D Layout cross-section (GEN-57): a Zemax-style side view of the
//! element stack — each surface drawn as its true conic sag profile,
//! glass gaps shaded, stop and image plane marked, an axial paraxial
//! focus reference, and a scale bar.
//!
//! Pure display plumbing (spec §6): every number comes from public APIs.
//! Surface heights use [`optics_core::element::SurfaceProfile::sag`] (the
//! same profile the simulation applies), refractive indices come from the
//! [`Registry`], and the dashed reference converges at the paraxial back
//! focal length from [`ImportedSystem::paraxial_at`]. This module draws
//! **no traced rays** — the reference viewer's colored fan is a geometric
//! ray trace, which `agents.md` §1 and spec §6 place out of scope pending
//! an explicit plan change.

use optics_core::element::SurfaceProfile;
use optics_zemax::{ImportedSystem, Registry, ZmxSurface};

/// A sampled surface curve for the layout: `z_mm` is the vertex position,
/// `sag_mm[i]` the height (mm) at radial sample `y_mm[i]` (mm, ascending,
/// symmetric about the axis).
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutSurface {
    /// `SURF` index.
    pub index: usize,
    /// Vertex axial position (mm).
    pub z_mm: f64,
    /// Semi-diameter (mm); `0.0` = unbounded (drawn at the field edge).
    pub semi_mm: f64,
    /// Sampled radial coordinates (mm), ascending, `0.0` first.
    pub y_mm: Vec<f64>,
    /// Sag heights (mm) at `y_mm`.
    pub sag_mm: Vec<f64>,
    /// `STOP` flag.
    pub stop: bool,
    /// Glass after this surface (`None` = air/vacuum).
    pub glass: Option<String>,
    /// Refractive index after this surface at the layout wavelength.
    pub n_after: f64,
    /// Comment text.
    pub comment: String,
}

/// A medium gap between consecutive surfaces.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutGap {
    /// Start vertex (mm).
    pub z0_mm: f64,
    /// End vertex (mm).
    pub z1_mm: f64,
    /// Refractive index in the gap at the layout wavelength.
    pub n: f64,
    /// Glass name, when not air.
    pub glass: Option<String>,
}

/// The 2D layout model: surfaces, gaps, extent, and the paraxial focus
/// reference. All positions are in millimeters along `z` from the first
/// refracting surface (surface 1), matching the schematic strip.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutModel {
    /// Refracting surfaces (`SURF 1 ..= last-1`), in order.
    pub surfaces: Vec<LayoutSurface>,
    /// Gaps between consecutive surfaces (length `surfaces.len()`, the
    /// last entry being the image-space gap).
    pub gaps: Vec<LayoutGap>,
    /// Total track length (mm).
    pub track_mm: f64,
    /// Largest semi-diameter (mm); drives the cross-section height.
    pub max_semi_mm: f64,
    /// Paraxial back focal length (mm) at the layout wavelength, when the
    /// system has one (object at infinity).
    pub bfl_mm: Option<f64>,
    /// Layout wavelength (µm).
    pub lambda_um: f64,
    /// `false` when a medium index could not be resolved (empty model).
    pub ok: bool,
}

impl Default for LayoutModel {
    fn default() -> Self {
        Self {
            surfaces: Vec::new(),
            gaps: Vec::new(),
            track_mm: 0.0,
            max_semi_mm: 0.0,
            bfl_mm: None,
            lambda_um: 0.55,
            ok: true,
        }
    }
}

/// Radial samples per surface curve: a handful of points captures the
/// conic sag shape at layout scale (SVG polyline, not a ray grid).
const SAMPLES: usize = 9;

fn sample_y(semi_mm: f64) -> Vec<f64> {
    let top = if semi_mm > 0.0 { semi_mm } else { 0.0 };
    (0..SAMPLES)
        .map(|i| top * i as f64 / (SAMPLES - 1) as f64)
        .collect()
}

/// Build the 2D layout from the prescription working copy. `imported`
/// supplies the paraxial reference (its own surfaces), `surfaces` is the
/// edited/inserted model actually drawn; `registry` resolves media at
/// `lambda_um`.
pub fn build_layout(
    surfaces: &[ZmxSurface],
    registry: &Registry,
    lambda_um: f64,
    imported: Option<&ImportedSystem>,
) -> LayoutModel {
    let last = match surfaces.len() {
        0..=1 => return LayoutModel::default(),
        n => n - 1,
    };
    let mut model = LayoutModel {
        lambda_um,
        ..LayoutModel::default()
    };
    let mut z_cursor = 0.0f64;
    for s in &surfaces[1..last] {
        let back_n = match registry.index_at(s.glass.as_deref(), lambda_um) {
            Ok(n) => n,
            Err(_) => {
                model.ok = false;
                return model;
            }
        };
        let profile = if s.curvature_mm != 0.0 {
            Some(if s.conic == 0.0 {
                SurfaceProfile::Spherical {
                    radius: 1e-3 / s.curvature_mm,
                }
            } else {
                SurfaceProfile::Conic {
                    radius: 1e-3 / s.curvature_mm,
                    conic: s.conic,
                }
            })
        } else {
            None
        };
        let y_mm = sample_y(s.semi_diam_mm);
        let sag_mm = match &profile {
            Some(p) => y_mm
                .iter()
                .map(|&y| p.sag(y * 1e-3) * 1e3)
                .collect::<Vec<f64>>(),
            None => vec![0.0; y_mm.len()],
        };
        model.surfaces.push(LayoutSurface {
            index: s.index,
            z_mm: z_cursor,
            semi_mm: s.semi_diam_mm,
            y_mm,
            sag_mm,
            stop: s.stop,
            glass: s.glass.clone(),
            n_after: back_n,
            comment: s.comment.clone(),
        });
        let z = s.thickness_mm.unwrap_or(0.0);
        model.gaps.push(LayoutGap {
            z0_mm: z_cursor,
            z1_mm: z_cursor + z,
            n: back_n,
            glass: s.glass.clone(),
        });
        z_cursor += z;
        if s.semi_diam_mm > model.max_semi_mm {
            model.max_semi_mm = s.semi_diam_mm;
        }
    }
    model.track_mm = z_cursor;
    model.bfl_mm = imported
        .and_then(|sys| sys.paraxial_at(lambda_um).ok())
        .map(|r| r.bfl_mm);
    model
}

/// The image-plane vertex (mm): where the last refracting surface's
/// thickness lands the detector.
pub fn image_z_mm(model: &LayoutModel) -> f64 {
    model.track_mm
}
