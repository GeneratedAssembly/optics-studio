//! End-to-end: import the Thorlabs ACA254-100-UV prescription, trace a
//! UV beam through the doublet with the medium-aware ASM, and verify the
//! imported image plane is the focal plane (on-axis intensity peak).
//!
//! Beam sigma 0.5 mm (far inside the 11.125 mm stop): the focused waist
//! is lambda*f/(pi*w) ~ 18 um, resolved by the 5 um grid; the Gaussian
//! tails vanish ~500x below the window edge, so the band-limit is off
//! (it would clip the converging rays' spectrum at this long distance).

use optics_core::field::BandField;
use optics_core::propagation::asm::{AsmConfig, propagate as asm_propagate};
use optics_core::sources::{GaussianBeam, gaussian};
use optics_zemax::ImportedSystem;

const LAMBDA: f64 = 282.5e-9; // primary design wavelength

fn sample() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/ACA254-100-UV.zmx"
    ))
    .unwrap()
}

fn engine()
-> impl Fn(&BandField, f64, f64) -> Result<BandField, optics_core::propagation::asm::PropagationError>
{
    |b: &BandField, z, n| {
        asm_propagate(
            b,
            z,
            &AsmConfig {
                medium: n,
                band_limited: false,
                strict: false,
                ..Default::default()
            },
        )
    }
}

#[test]
fn doublet_focuses_uv_beam_at_imported_image_plane() {
    let sys = ImportedSystem::parse(&sample()).unwrap();
    let band = gaussian(
        &GaussianBeam {
            w0: 0.5e-3,
            amplitude: 1.0,
            center: (0.0, 0.0),
        },
        0.0,
        LAMBDA,
        1024,
        1024,
        5e-6,
        5e-6,
    )
    .unwrap();
    let mut eng = engine();
    let out = sys.trace_subaperture(&band, 2.0, &mut eng).unwrap();
    let c = out.field().dims().0 / 2;
    let on_axis = out.field().sample(c, c).norm_sqr();
    // +20 mm past focus (z_R' ~ 3.8 mm): on-axis intensity collapses by
    // 1 + (20/3.8)^2 ~ 28x.
    let past = asm_propagate(
        &out,
        0.020,
        &AsmConfig {
            band_limited: false,
            strict: false,
            ..Default::default()
        },
    )
    .unwrap();
    let on_axis_past = past.field().sample(c, c).norm_sqr();
    assert!(
        on_axis > 8.0 * on_axis_past,
        "image plane not focal: {on_axis:e} vs {on_axis_past:e}"
    );
    // And -20 mm before: similar collapse.
    let before = asm_propagate(
        &out,
        -0.020,
        &AsmConfig {
            band_limited: false,
            strict: false,
            ..Default::default()
        },
    )
    .unwrap();
    let on_axis_before = before.field().sample(c, c).norm_sqr();
    assert!(
        on_axis > 8.0 * on_axis_before,
        "pre-focus ratio {on_axis:e} vs {on_axis_before:e}"
    );
    // Peak concentration: > 30% of power within 60 um of centroid.
    let psf = optics_core::field::Psf::from_band(&out);
    let (cx, cy) = psf.centroid();
    let (nx, ny) = psf.dims();
    let mut hit = 0.0;
    let mut all = 0.0;
    for iy in 0..ny {
        for ix in 0..nx {
            let (x, y) = psf.coord(ix, iy);
            let v = psf.samples()[iy * nx + ix];
            all += v;
            if (x - cx).hypot(y - cy) <= 60e-6 {
                hit += v;
            }
        }
    }
    assert!(hit / all > 0.3, "concentration {}", hit / all);
}
