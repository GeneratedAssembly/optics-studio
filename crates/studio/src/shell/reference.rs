//! Synthetic reference PSF for **live** forward frames (task 10.4
//! round-trip seam).
//!
//! The viewport's Strehl panel needs a reference PSF (spec §3.3, gap G-6:
//! no core API generates one). The demo model builds a fixed synthetic
//! bundle; the live model builds the same recipe from the *current*
//! geometry: a diffraction-limited thin-lens stand-in with the imported
//! design's stop aperture and conjugates, traced on the measured frame's
//! grid and engine family. Aberrations and defocus of the real design then
//! show up as Strehl < 1 exactly as in the demo.
//!
//! This is plumbing over core APIs (`viewmodel::reference::build_reference`
//! — no new physics, spec §6). When the geometry cannot support the recipe
//! (plane-wave / Gaussian source, Rayleigh–Sommerfeld engine, immersion
//! medium, missing aperture) the frame ships without a reference and the
//! panel shows its no-reference state.

use optics_core::field::Psf;
use optics_core::propagation::regime::Engine;

use crate::shell::dispatch::{EnginePick, EngineSettings};
use crate::viewmodel::engine::EngineSettings as TraceSettings;
use crate::viewmodel::reference::{ReferenceGeometry, build_reference};

/// Map the shell's five-way picker to the viewport engine settings so the
/// reference is traced with the same engine family and knobs as the
/// measured frame. Rayleigh–Sommerfeld has no viewport counterpart
/// (`None` → the live frame carries no synthetic reference).
pub fn to_trace_settings(settings: &EngineSettings) -> Option<TraceSettings> {
    let engine = match settings.pick {
        EnginePick::Asm => Engine::Asm,
        EnginePick::FresnelTransfer => Engine::FresnelTransfer,
        EnginePick::FresnelConvolution => Engine::FresnelConvolution,
        EnginePick::Fraunhofer => Engine::Fraunhofer,
        EnginePick::RayleighSommerfeld => return None,
    };
    Some(TraceSettings {
        engine,
        asm: settings.asm,
        fresnel: settings.fresnel,
        fraunhofer: settings.fraunhofer,
    })
}

/// Best-effort synthetic G-6 reference for a live forward frame.
///
/// `psf` is the measured frame — its grid (`dims`/`pitch`) is reused so
/// `analysis::strehl` compares like-for-like. `point` is
/// `(source→stop distance, amplitude)` for a point-source illumination;
/// every other source kind returns `None`. `z`/`medium` are the final
/// gap; `aperture` the stop semi-diameter (m). The stand-in lens uses the
/// thin-lens conjugate $f = s z / (s + z)$, so a perfectly focused design
/// scores Strehl ≈ 1 and any residual aberration / defocus lowers it.
#[allow(clippy::too_many_arguments)]
pub fn forward_reference(
    settings: &EngineSettings,
    psf: &Psf,
    lambda_m: f64,
    point: Option<(f64, f64)>,
    z: f64,
    medium: f64,
    aperture: Option<f64>,
) -> Option<Psf> {
    let trace = to_trace_settings(settings)?;
    let (source_distance, amplitude) = point?;
    let aperture_radius = aperture?;
    if !medium.approx_eq_air() {
        return None;
    }
    if !source_distance.is_finite() || source_distance <= 0.0 {
        return None;
    }
    if !z.is_finite() || z <= 0.0 {
        return None;
    }
    let focal_length = (source_distance * z) / (source_distance + z);
    if !focal_length.is_finite() || focal_length <= 0.0 {
        return None;
    }
    // The synthetic focus must be resolvable on the measured grid, or the
    // peak ratio is aliasing noise, not a metric: guard with the Airy
    // first-null radius at the detector plane.
    let (px, py) = psf.pitch();
    let airy_radius = 1.22 * lambda_m * z / (2.0 * aperture_radius);
    if !airy_radius.is_finite() || airy_radius < 1.5 * px.max(py) {
        return None;
    }
    let geom = ReferenceGeometry {
        source: (0.0, 0.0),
        source_distance,
        amplitude,
        lambda: lambda_m,
        dims: psf.dims(),
        pitch: psf.pitch(),
        aperture_radius,
        focal_length,
        image_distance: z,
    };
    build_reference(&trace, &geom).ok()
}

trait ApproxEqAir {
    fn approx_eq_air(self) -> bool;
}
impl ApproxEqAir for f64 {
    fn approx_eq_air(self) -> bool {
        (self - 1.0).abs() < 1e-12
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optics_core::analysis::strehl;

    /// (settings, measured psf, λ, point source, z, medium, aperture)
    type LiveFixture = (
        EngineSettings,
        Psf,
        f64,
        Option<(f64, f64)>,
        f64,
        f64,
        Option<f64>,
    );

    fn demo_like() -> LiveFixture {
        // The demo base geometry: source 60 mm before an f = 50 mm lens,
        // in-focus image at 300 mm — the conjugate formula must reproduce
        // f = 0.05 from (s, z).
        let geom = ReferenceGeometry {
            source: (0.0, 0.0),
            source_distance: 0.06,
            amplitude: 1e-6,
            lambda: 550e-9,
            dims: (64, 64),
            pitch: (20e-6, 20e-6),
            aperture_radius: 0.5e-3,
            focal_length: 0.05,
            image_distance: 0.3,
        };
        let trace = TraceSettings::default();
        let psf = build_reference(&trace, &geom).expect("reference builds");
        (
            EngineSettings::default(),
            psf,
            550e-9,
            Some((geom.source_distance, geom.amplitude)),
            geom.image_distance,
            1.0,
            Some(geom.aperture_radius),
        )
    }

    #[test]
    fn picker_maps_except_rayleigh_sommerfeld() {
        let mut s = EngineSettings::default();
        assert!(to_trace_settings(&s).is_some());
        s.pick = EnginePick::RayleighSommerfeld;
        assert!(to_trace_settings(&s).is_none());
    }

    #[test]
    fn live_reference_reproduces_the_in_focus_air_airy() {
        let (settings, psf, lambda, point, z, medium, aperture) = demo_like();
        let reference =
            forward_reference(&settings, &psf, lambda, point, z, medium, aperture).expect("built");
        assert_eq!(reference.dims(), psf.dims());
        assert_eq!(reference.pitch(), psf.pitch());
        // Same geometry → the measured frame IS the reference: ratio ≈ 1.
        let s = strehl(&psf, &reference);
        assert!((s - 1.0).abs() < 1e-9, "in-focus Strehl {s}");
    }

    #[test]
    fn unsupported_geometry_yields_no_reference() {
        let (settings, psf, lambda, point, z, medium, aperture) = demo_like();
        // Plane wave / Gaussian illumination.
        assert!(forward_reference(&settings, &psf, lambda, None, z, medium, aperture).is_none());
        // Immersion medium.
        assert!(
            forward_reference(&settings, &psf, lambda, point, z, 1.5, aperture).is_none(),
            "medium != air"
        );
        // No stop flagged.
        assert!(forward_reference(&settings, &psf, lambda, point, z, medium, None).is_none());
        // Rayleigh–Sommerfeld has no viewport engine counterpart.
        let mut rs = settings;
        rs.pick = EnginePick::RayleighSommerfeld;
        assert!(forward_reference(&rs, &psf, lambda, point, z, medium, aperture).is_none());
    }

    #[test]
    fn undersampled_reference_focus_is_skipped() {
        // Coarse preview grid (0.5 mm pitch) vs a ~3 µm Airy: the ratio
        // would be aliasing noise — no reference ships.
        let (settings, psf, lambda, point, z, medium, aperture) = demo_like();
        let coarse = Psf::new(psf.samples().to_vec(), 64, 64, 0.5e-3, 0.5e-3).expect("coarse psf");
        assert!(
            forward_reference(&settings, &coarse, lambda, point, z, medium, aperture).is_none(),
            "undersampled synthetic focus must be skipped"
        );
    }
}
