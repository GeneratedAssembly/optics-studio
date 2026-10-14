//! Phase-5 physical validation: the PSF pipeline — `sources::spherical` /
//! plane-wave sources traced through a `Beamline` — against classical
//! anchors: the Airy pattern of a circular pupil (Eqs. 2.19 + the J1 form,
//! as in `tests/asm.rs` test 4), the focal shift of a tilted plane wave
//! through a lens, and the geometric image location of an off-axis point
//! source (paraxial imaging: m = -d_i/d_o).

mod common;

use common::airy;
use optics_core::beamline::Beamline;
use optics_core::element::{ClearAperture, EllipticalAperture, ThinLens};
use optics_core::field::BandField;
use optics_core::propagation::asm::{AsmConfig, propagate as asm_propagate};
use optics_core::propagation::fraunhofer::{FraunhoferConfig, propagate as fraunhofer_propagate};
use optics_core::sources::spherical;

const LAMBDA: f64 = 500e-9;

#[test]
fn psf_of_circular_aperture_is_airy() {
    // Uniform pupil, circular stop a = 0.5 mm, Fraunhofer to z = 2 m:
    // output pitch lambda*z/(n*delta) = 62.5 um, Airy first zero at
    // r1 = 1.22 lambda z / (2a) = 1.22 mm ~ 19.5 bins.
    let a = 0.5e-3;
    let z = 2.0;
    let n = 256usize; // stop radius = 20 bins: edge staircase negligible
    let d = 25e-6;
    let band = BandField::uniform(1.0, 0.0, n, n, d, d, LAMBDA).unwrap();
    let stop = EllipticalAperture::circle(a, (0.0, 0.0)).unwrap();
    let fcfg = FraunhoferConfig {
        aperture_extent: Some(a),
        ..Default::default()
    };
    let mut bl = Beamline::new();
    bl.push(&stop, z);
    let psf = bl
        .psf(&band, |b, zz, _n| fraunhofer_propagate(b, zz, &fcfg))
        .unwrap();

    let (onx, ony) = psf.dims();
    assert_eq!((onx, ony), (n, n));
    let (dx, _dy) = psf.pitch();
    assert!((dx - LAMBDA * z / (n as f64 * d)).abs() < 1e-12);
    // Central row: I(x)/I(0) vs the Airy profile [2 J1(v)/v]^2,
    // v = 2 pi a r / (lambda z).
    let iy = ony / 2;
    let i0 = psf.samples()[iy * onx + onx / 2];
    let mut max_err = 0.0f64;
    for ix in 0..onx {
        let (x, _) = psf.coord(ix, iy);
        let v = 2.0 * std::f64::consts::PI * a * x.abs() / (LAMBDA * z);
        let rel = (psf.samples()[iy * onx + ix] / i0 - airy(v)).abs();
        max_err = max_err.max(rel);
    }
    assert!(max_err < 0.03, "Airy profile deviation {max_err} (> 0.03)");
    // First dark ring: minimum search around r1 lands within 2 bins.
    let r1 = 1.22 * LAMBDA * z / (2.0 * a);
    let mut dark = (0usize, f64::MAX);
    for ix in 0..onx {
        let (x, _) = psf.coord(ix, iy);
        if x > 0.6 * r1 && x < 1.6 * r1 {
            let v = psf.samples()[iy * onx + ix];
            if v < dark.1 {
                dark = (ix, v);
            }
        }
    }
    let (xd, _) = psf.coord(dark.0, iy);
    assert!((xd - r1).abs() < 2.0 * dx, "first zero at {xd:e} vs {r1:e}");
    // Central-lobe profile is the tight statement; the far tail is dominated
    // by finite-grid sampling of the hard-edge aperture.
    let mut lobe_err = 0.0f64;
    for ix in 0..onx {
        let (x, _) = psf.coord(ix, iy);
        if x.abs() <= r1 {
            let v = 2.0 * std::f64::consts::PI * a * x.abs() / (LAMBDA * z);
            lobe_err = lobe_err.max((psf.samples()[iy * onx + ix] / i0 - airy(v)).abs());
        }
    }
    assert!(
        lobe_err < 0.02,
        "central-lobe deviation {lobe_err} (> 0.02)"
    );
}

#[test]
fn psf_peak_follows_field_angle() {
    // Resolved pupil (a = 1.25 mm = 50 bins) and focal spot (r1 ~ 5 bins):
    // a plane wave at theta = 1 mrad through a lens f = 0.5 m focuses at
    // x = f tan(theta) = 0.5 mm (20 bins); the intensity centroid is the
    // sub-bin estimator of the spot position.
    let a = 1.25e-3;
    let f = 0.5;
    let theta = 0.001;
    let n = 256usize;
    let d = 25e-6;
    let band = BandField::plane_wave(theta, 0.0, 1.0, n, n, d, d, LAMBDA).unwrap();
    let stop = EllipticalAperture::circle(a, (0.0, 0.0)).unwrap();
    let lens = ThinLens::fixed(f, None, ClearAperture::Field, (0.0, 0.0)).unwrap();
    let cfg = AsmConfig::default();
    let mut bl = Beamline::new();
    bl.push(&stop, 0.0).push(&lens, f);
    let psf = bl.psf(&band, |b, z, _n| asm_propagate(b, z, &cfg)).unwrap();

    let (cx, cy) = psf.centroid();
    let expect = f * theta.tan(); // 0.5 mm
    assert!(
        (cx - expect).abs() < 2e-5,
        "focal centroid x {cx:e} vs {expect:e}"
    );
    assert!(cy.abs() < 1e-6, "focal centroid y {cy:e}");
    // On-axis reference: same system, untilted wave -> centroid at origin.
    let band0 = BandField::plane_wave(0.0, 0.0, 1.0, n, n, d, d, LAMBDA).unwrap();
    let psf0 = bl
        .psf(&band0, |b, z, _n| asm_propagate(b, z, &cfg))
        .unwrap();
    let (c0x, c0y) = psf0.centroid();
    assert!(c0x.abs() < 1e-6 && c0y.abs() < 1e-6, "on-axis {c0x},{c0y}");
}

#[test]
fn point_source_focuses_at_geometric_image() {
    // Off-axis point at x_s = +0.5 mm, distance d_o = 0.3 m upstream;
    // lens f = 0.1 m with a 1 mm stop. Lens equation: d_i = f d_o/(d_o - f)
    // = 0.15 m; magnification m = -d_i/d_o = -0.5 -> image at -0.25 mm.
    // Grid: n = 256, delta = 10 um (window +-1.275 mm): the spherical-wave
    // rim phase step k*delta*rho/d_o ~ 0.53 rad is well below pi, and the
    // diffraction spot 1.22 lambda d_i/(2 a_stop) ~ 46 um spans ~5 bins.
    let x_s = 0.5e-3;
    let d_o = 0.3;
    let f = 0.1;
    let d_i = f * d_o / (d_o - f);
    let stop = 1.0e-3;
    let n = 256usize;
    let d = 1e-5;
    let src = spherical((x_s, 0.0), d_o, 1e-3, LAMBDA, n, n, d, d).unwrap();
    let lens = ThinLens::fixed(f, None, ClearAperture::Circle(stop), (0.0, 0.0)).unwrap();
    let cfg = AsmConfig::default();
    let mut bl = Beamline::new();
    bl.push(&lens, d_i);
    let psf = bl.psf(&src, |b, z, _n| asm_propagate(b, z, &cfg)).unwrap();

    let (cx, cy) = psf.centroid();
    let expect = -x_s * d_i / d_o; // -0.25 mm
    assert!(
        (cx - expect).abs() < 2e-5,
        "image centroid x {cx:e} vs {expect:e}"
    );
    assert!(cy.abs() < 2e-6, "image centroid y {cy:e}");
    // Concentration: the focused spot must stand far above the mean
    // (a defocused/aberrated field would not).
    let peak = psf.samples().iter().cloned().fold(0.0f64, f64::max);
    let mean = psf.samples().iter().sum::<f64>() / psf.samples().len() as f64;
    assert!(
        peak > 10.0 * mean,
        "peak {peak:e} vs mean {mean:e}: not focused"
    );
}
