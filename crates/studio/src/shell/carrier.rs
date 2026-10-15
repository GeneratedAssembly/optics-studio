//! Demo inverse-design carrier (spec §3.2 slider semantics) for task 10.1.
//!
//! Task 10.3 wires real solver runs to these sliders; 10.1 ships one
//! analytic carrier so the `Carrier::params → slider → build →
//! Design::beamline → Beamline::psf` live-preview path is exercised
//! end-to-end. The build closure is pure API composition (element
//! constructors + gaps) — no physics formulas (spec §6).

use optics_core::element::{ClearAperture, RefractiveSurface, SurfaceProfile};
use optics_inverse::{Carrier, Design, InverseError, Param, Params};
use optics_zemax::catalog;

/// Zemax errors surface as the typed carrier-construction failure.
fn carrier_err<E: std::fmt::Display>(e: E) -> InverseError {
    InverseError::Carrier {
        reason: e.to_string(),
    }
}

/// The demo carrier: a biconvex N-BK7 singlet with editable surface
/// curvatures (1/m), center thickness (mm), image distance (mm), and
/// clear aperture (mm).
pub fn demo_carrier() -> Carrier {
    let params = Params::from_slots([
        Param::new("front_curv_per_m", -30.0, 30.0, 12.0),
        Param::new("back_curv_per_m", -30.0, 30.0, -10.0),
        Param::new("thickness_mm", 1.0, 20.0, 6.0),
        Param::new("image_mm", 20.0, 300.0, 100.0),
        Param::new("aperture_mm", 2.0, 25.0, 12.5),
    ]);
    Carrier::analytic(params, |values, lambda| {
        let (c1, c2, t_mm, img_mm, ap_mm) = (values[0], values[1], values[2], values[3], values[4]);
        // A zero curvature would send the radius through infinity; clamp
        // to a near-flat 1 km radius (same sag to ~1e-9 m at these
        // apertures — display honesty, not new physics).
        let radius = |c: f64| {
            if c.abs() < 1e-3 {
                if c < 0.0 { -1e3 } else { 1e3 }
            } else {
                1.0 / c
            }
        };
        let aperture = ClearAperture::Circle(ap_mm * 0.5e-3);
        let glass = catalog::material(Some("N-BK7")).map_err(carrier_err)?;
        let n_glass = glass.n_at(lambda, 0.0);
        let mut design = Design::new();
        design.push(
            RefractiveSurface::new(
                SurfaceProfile::Spherical { radius: radius(c1) },
                catalog::material(None).map_err(carrier_err)?,
                catalog::material(Some("N-BK7")).map_err(carrier_err)?,
                None,
                None,
                aperture,
                (0.0, 0.0),
            )
            .map_err(carrier_err)?,
            t_mm * 1e-3,
            n_glass,
        );
        design.push(
            RefractiveSurface::new(
                SurfaceProfile::Spherical { radius: radius(c2) },
                glass,
                catalog::material(None).map_err(carrier_err)?,
                None,
                None,
                aperture,
                (0.0, 0.0),
            )
            .map_err(carrier_err)?,
            img_mm * 1e-3,
            1.0,
        );
        Ok(design)
    })
}
