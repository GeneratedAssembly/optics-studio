//! Golden (task 9.7): close the zemax → inverse loop. Import the Thorlabs
//! ACA254-100-UV air-spaced achromatic doublet with `optics-zemax` (a
//! **dev-dependency only** — the public API of `optics-inverse` stays
//! zemax-free), tune **one curvature** with the FD/Adam solver, and
//! minimize the spot RMS of the focused PSF.
//!
//! Physics anchors:
//!
//! - Textbook Eq. (4.3) (§4.1.2): surface power $\varphi = (n_2 - n_1)c$ —
//!   lens bending (curvature change) shifts focus linearly in $c$, so a
//!   curvature offset from the design value defocuses the image plane.
//! - The imported file's own image distance (89.4896 mm) is the paraxial
//!   BFL at the design center wavelength 0.325 µm (verified in the zemax
//!   crate's golden). Rendering at 325 nm therefore has its spot-RMS
//!   optimum *at* the imported curvature of the tuned surface: starting
//!   from an offset curvature, Adam must bend it back toward the design
//!   value while monotonically improving the best-so-far spot RMS.
//!
//! The system is stop-down to a 0.7 mm sub-aperture (the same
//! `trace_subaperture` discipline the zemax crate uses). The start point
//! carries ~16 mm of defocus — a geometric ring of ~90 µm radius at the
//! detector, clearly above the aperture-diffraction halo floor (~70 µm
//! spot RMS) — so the merit improvement is unambiguous at a tractable
//! 256² grid.

use optics_core::element::{ClearAperture, RefractiveSurface, SurfaceProfile};
use optics_core::field::BandField;
use optics_core::propagation::asm::AsmConfig;
use optics_inverse::carrier::{Carrier, Design};
use optics_inverse::error::InverseError;
use optics_inverse::fd::AdamConfig;
use optics_inverse::merit::{Selector, SpotRmsMerit, objective};
use optics_inverse::observe::Renderer;
use optics_inverse::params::{Param, Params};
use optics_inverse::solve::{Solver, StochasticConfig, solve};
use optics_zemax::{ImportedSystem, Registry, ZmxSystem};

/// Design center wavelength (m): the file's image distance is the paraxial
/// BFL here (zemax-crate golden).
const LAMBDA: f64 = 325e-9;
/// Analysis grid: 256 samples at 7.5 µm (half-field 0.956 mm, clear margin
/// around the 0.7 mm sub-aperture stop; the geometric ring of the
/// deliberately defocused start point is ~110 µm = 15 px).
const N: usize = 256;
const DX: f64 = 7.5e-6;
/// Sub-aperture stop-down (mm), applied to every circular stop.
const CAP_MM: f64 = 0.7;
/// Tuned surface: the CaF2 rear curvature (power coefficient
/// $n_{air} - n_{CaF_2} \approx -0.44$ — strong, smooth defocus lever).
const TUNED: usize = 2;
/// Curvature bounds: design value ± 3.5e-3 mm⁻¹ (~9% of the curvature).
const HALF_SPAN: f64 = 3.5e-3;
/// Deliberate start offset: ~16 mm of defocus, a geometric ring of ~110 µm
/// radius at the detector — well above the aperture-diffraction halo floor
/// (~70 µm spot RMS) so the merit gradient is unambiguous.
const START_OFFSET: f64 = 3.2e-3;

fn aca() -> ImportedSystem {
    ImportedSystem::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../optics-zemax/tests/data/ACA254-100-UV.zmx"
    ))
    .expect("ACA254 sample loads with the built-in catalog")
}

/// Mirror of the zemax builder's two-phase plan (surfaces 1..last, curved
/// ones become `RefractiveSurface`s, every step carries its medium), with
/// the tuned surface's curvature replaced and all stops clamped to
/// `CAP_MM`.
fn doublet(
    z: &ZmxSystem,
    reg: &Registry,
    c_tuned: f64,
    lambda_um: f64,
) -> Result<Design, InverseError> {
    let carrier_err = |e: optics_zemax::ZemaxError| InverseError::Carrier {
        reason: e.to_string(),
    };
    let last = z.surfaces.len() - 1;
    let mut d = Design::new();
    for i in 1..last {
        let s = &z.surfaces[i];
        let curv = if i == TUNED { c_tuned } else { s.curvature_mm };
        let semi = s.semi_diam_mm.min(CAP_MM);
        let aperture = if semi > 0.0 {
            ClearAperture::Circle(semi * 1e-3)
        } else {
            ClearAperture::Field
        };
        let front = if i == 1 {
            None
        } else {
            z.surfaces[i - 1].glass.as_deref()
        };
        let surf = RefractiveSurface::new(
            SurfaceProfile::Spherical {
                radius: 1e-3 / curv,
            },
            reg.material(front).map_err(carrier_err)?,
            reg.material(s.glass.as_deref()).map_err(carrier_err)?,
            None,
            None,
            aperture,
            (0.0, 0.0),
        )
        .map_err(|e| InverseError::Carrier {
            reason: e.to_string(),
        })?;
        let medium = reg
            .index_at(s.glass.as_deref(), lambda_um)
            .map_err(carrier_err)?;
        d.push(surf, s.thickness_mm.unwrap_or(0.0) * 1e-3, medium);
    }
    Ok(d)
}

#[test]
fn adam_bends_aca254_curvature_to_minimize_spot_rms() {
    let sys = aca();
    let z = sys.system().clone();
    let reg = Registry::builtin();
    let c0 = z.surfaces[TUNED].curvature_mm;

    let params = Params::from_slots([Param::new(
        "c2_caF2_rear",
        c0 - HALF_SPAN,
        c0 + HALF_SPAN,
        c0 + START_OFFSET, // start deliberately off the design curvature
    )]);
    let carrier = Carrier::analytic(params.clone(), move |v, lambda| {
        doublet(&z, &reg, v[0], lambda * 1e6)
    });

    let mut renderer = Renderer::new(&carrier, |_x0, _y0, lambda| {
        BandField::plane_wave(0.0, 0.0, 1.0, N, N, DX, DX, lambda).map_err(|e| {
            InverseError::Render {
                reason: e.to_string(),
            }
        })
    });
    renderer.bands = vec![LAMBDA];
    renderer.scenes = vec![(0.0, 0.0)];
    // Long converging beam: the band-limit would clip the spectrum at the
    // 89.5 mm final gap (same convention as the zemax-crate trace test).
    renderer.asm = AsmConfig {
        band_limited: false,
        strict: false,
        ..Default::default()
    };

    let spot = SpotRmsMerit {
        selector: Selector::intensity(None, None),
    };
    let obj = objective(&renderer, &spot);

    let sol = solve(
        &params,
        &obj,
        Solver::Adam(AdamConfig {
            lr_frac: 0.06,
            ..AdamConfig::default()
        }),
        &StochasticConfig {
            budget: 20, // epochs
            ..Default::default()
        },
        false,
    )
    .expect("Adam run completes");

    // Every epoch evaluation is inside bounds (prevention policy).
    assert!(sol.best[0] >= c0 - HALF_SPAN && sol.best[0] <= c0 + HALF_SPAN);

    // Convergence history recorded like the 9.6 solvers: one value per
    // epoch (start + 30 epochs); the best-so-far snapshot is the minimum
    // of the raw per-epoch trace (Adam's raw values may rise between
    // epochs — best-so-far is monotone by construction).
    assert_eq!(sol.history.len(), 21);
    assert_eq!(sol.evaluations, 1 + 20 * (2 * 1 + 1));
    let trace_min = sol.history.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    assert!(
        (sol.value - trace_min).abs() <= 1e-12 * trace_min.abs().max(1.0),
        "best {} != history min {trace_min}",
        sol.value
    );
    // Best-so-far strictly improves and ends well below the start.
    assert!(
        sol.value < 0.9 * sol.history[0],
        "spot RMS barely improved: start {:.3} um, best {:.3} um",
        sol.history[0] * 1e6,
        sol.value * 1e6
    );

    // Physics check: the optimum curvature is the design one (image plane
    // is paraxially focused at 325 nm) — Adam must bend the surface most
    // of the way back (Eq. 4.3: defocus ∝ curvature offset). Tolerance is
    // a few Adam oscillation steps (lr_frac·span = 4.2e-4).
    assert!(
        (sol.best[0] - c0).abs() < 8.0e-4,
        "best curvature {} vs design {c0} (start offset +{START_OFFSET:e})",
        sol.best[0]
    );

    // And the design curvature itself must be (near) the best found —
    // evaluating the merit there confirms the loop is anchoring on the
    // imported prescription, not on an artifact of the grid.
    let at_design = obj(&[c0]).expect("design point renders");
    assert!(
        sol.value <= at_design * 1.05,
        "best {:.3} um should sit at/below the design spot {:.3} um",
        sol.value * 1e6,
        at_design * 1e6
    );
}
