//! Phase-5 example: diffraction-limited image blur via a PSF.
//!
//! Builds the coherent PSF of a stopped lens (pupil -> thin lens -> focal
//! plane, ASM engine through a `Beamline`), then blurs a synthetic scene of
//! three point sources by circular convolution with the normalized PSF —
//! the isoplanatic (small-field-angle) approximation: each scene point
//! images to a shifted copy of the PSF. The kernel is rolled from the
//! centered grid into wrap-around (DFT) order, truncated where intensity
//! falls below 1e-6 of peak, and renormalized so total power is preserved.
//!
//! Run: `cargo run --release --example psf_image_convolve`

use optics_core::beamline::Beamline;
use optics_core::element::{ClearAperture, EllipticalAperture, ThinLens};
use optics_core::field::BandField;
use optics_core::propagation::asm::{AsmConfig, propagate};

const LAMBDA: f64 = 500e-9;
const N: usize = 256;
const D: f64 = 25e-6; // focal-plane pitch (m)
const STOP: f64 = 1.25e-3; // pupil radius (m)
const F: f64 = 0.5; // focal length (m)

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- PSF of the imaging system: stop, then lens, then focal plane. ---
    let band = BandField::uniform(1.0, 0.0, N, N, D, D, LAMBDA)?;
    let stop = EllipticalAperture::circle(STOP, (0.0, 0.0))?;
    let lens = ThinLens::fixed(F, None, ClearAperture::Circle(STOP), (0.0, 0.0))?;
    let cfg = AsmConfig::default();
    let mut bl = Beamline::new();
    bl.push(&stop, 0.0).push(&lens, F);
    let psf = bl.psf(&band, |b, z, _n| propagate(b, z, &cfg))?;

    // Roll the centered kernel into wrap-around order (bin 0 = origin)
    // and truncate + renormalize: sum(kernel) = 1 exactly.
    let peak = psf
        .samples()
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(1e-30);
    let h = N / 2;
    let mut kernel = vec![0.0f64; N * N];
    for iy in 0..N {
        for ix in 0..N {
            // Centered bin (cx, cy) maps to wrap bin ((cx - h) mod N).
            let (cx, cy) = ((ix + h) % N, (iy + h) % N);
            let v = psf.samples()[cy * N + cx];
            kernel[iy * N + ix] = if v > 1e-6 * peak { v } else { 0.0 };
        }
    }
    let ksum: f64 = kernel.iter().sum();
    for v in kernel.iter_mut() {
        *v /= ksum;
    }

    // --- Synthetic scene: three point sources (unit flux each). ---
    let points = [(60usize, 60usize), (120, 150), (200, 90)];
    let mut scene = vec![0.0f64; N * N];
    for &(ix, iy) in &points {
        scene[iy * N + ix] = 1.0;
    }

    // --- Blur: out[i] = sum_k kernel[k] * scene[(i - k) mod N]. ---
    let mut blurred = vec![0.0f64; N * N];
    for ky in 0..N {
        for kx in 0..N {
            let w = kernel[ky * N + kx];
            if w == 0.0 {
                continue;
            }
            for iy in 0..N {
                let sy = (iy + N - ky) % N;
                for ix in 0..N {
                    blurred[iy * N + ix] += w * scene[sy * N + (ix + N - kx) % N];
                }
            }
        }
    }

    // --- Checks: power preserved, blobs unmoved (within the even-grid
    // half-bin kernel-center ambiguity), deltas actually blurred. ---
    let p_in: f64 = scene.iter().sum();
    let p_out: f64 = blurred.iter().sum();
    let rel = ((p_out - p_in) / p_in).abs();
    assert!(rel < 1e-9, "power {p_in} -> {p_out} (rel {rel:e})");

    for &(px, py) in &points {
        // Local centroid in a +-12-bin window around the source.
        let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
        for dy in -12..12 {
            for dx in -12..12 {
                let ix = (px as isize + dx).rem_euclid(N as isize) as usize;
                let iy = (py as isize + dy).rem_euclid(N as isize) as usize;
                let v = blurred[iy * N + ix];
                sx += v * dx as f64;
                sy += v * dy as f64;
                sw += v;
            }
        }
        let shift = ((sx / sw).hypot(sy / sw)) * D; // meters from the source bin
        assert!(
            shift < 2.0 * D,
            "blob at {px},{py} drifted {shift:e} m (> 2 bins)"
        );
    }

    // Global peak: the unit deltas are spread over the PSF footprint, so
    // the blurred max sits well below 1 but remains the brightest pixel.
    let blob_peak = blurred.iter().cloned().fold(0.0f64, f64::max);
    assert!(blob_peak < 1.0 && blob_peak > 0.02, "peak {blob_peak}");

    println!(
        "PSF blur: pupil r={:.2} mm, f={F:.1} m, lambda={}.0 nm -> \
         blob peak {blob_peak:.3} of unit flux, power preserved (rel {rel:.1})",
        STOP * 1e3,
        LAMBDA * 1e9
    );
    Ok(())
}
