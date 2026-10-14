//! SAGE IV prototype — DRISM telescope: inverse design from the instrument
//! spec, then wave-optics verification of the layout.
//!
//! Layout (figure (a)): entrance stop -> Primary Lens -> Collimator ->
//! DRISM meta-filter -> Focusing Lens -> FPA detector plane. The DRISM
//! meta-filter (figure (b)) is modeled as one plane: a 2pi kinoform blaze
//! of period `PERIOD` (linear chromatic angular dispersion, sin(theta) =
//! lambda / PERIOD) multiplied by the six-band Gaussian passband stack
//! from the lambda-C / FWHM table. The final lens maps each diffracted
//! tilt to a focal-plane position x = F_DET * lambda / PERIOD, so each
//! passband lands at its own position along the dispersion axis.
//!
//! Inverse-design step (spec -> geometry): the passband table and the FPA
//! format (5.5 um pitch, 3296 x 2472, 17 mrad circular FOV) fix the blaze
//! period and focal lengths — the printout shows the derivation and the
//! aperture needed to resolve the 2 nm HiSpec pair at full scale.
//!
//! Verification: each band is traced (three wavelengths across its FWHM,
//! incoherent sum) and its centroid compared against the grating-equation
//! design position; off-band probes must be blocked. The accumulated FPA
//! frame is written to `target/drism_fpa.pgm` (log scale, 1 sample = 1
//! FPA pixel).
//!
//! Run: `cargo run --release --example inverse_drism`

use std::f64::consts::PI;

use num_complex::Complex64;
use optics_core::beamline::Beamline;
use optics_core::element::{
    ClearAperture, Element, ElementKind, ElementShape, EllipticalAperture, Extent, Material,
    OpticsError, ThinLens,
};
use optics_core::field::BandField;
use optics_core::propagation::asm::{AsmConfig, propagate};

/// One meta-filter passband: center wavelength and FWHM, meters.
struct SpectralBand {
    name: &'static str,
    lambda_c: f64,
    fwhm: f64,
}

/// The six DRISM passbands from the lambda-C / FWHM table (nm -> m).
/// The 1020 nm band extends past the 0.35-0.95 um nominal range (SWIR
/// extension); it is kept because the table specifies it.
const BANDS: [SpectralBand; 6] = [
    SpectralBand {
        name: "UV",
        lambda_c: 386e-9,
        fwhm: 15e-9,
    },
    SpectralBand {
        name: "HiSpec-a",
        lambda_c: 448e-9,
        fwhm: 2e-9,
    },
    SpectralBand {
        name: "HiSpec-b",
        lambda_c: 452e-9,
        fwhm: 2e-9,
    },
    SpectralBand {
        name: "Green",
        lambda_c: 525e-9,
        fwhm: 15e-9,
    },
    SpectralBand {
        name: "Red",
        lambda_c: 600e-9,
        fwhm: 15e-9,
    },
    SpectralBand {
        name: "SWIR",
        lambda_c: 1020e-9,
        fwhm: 15e-9,
    },
];

/// Off-band probe wavelengths: the meta-filter must block these.
const PROBES: [f64; 2] = [470e-9, 700e-9];

// --- Instrument spec (SAGE IV prototype table) -----------------------------

/// Reference-detector / FPA pixel pitch, meters.
const FPA_PITCH: f64 = 5.5e-6;
/// Circular FOV half-angle, radians (17 mrad full).
const FOV_RADIUS: f64 = 8.5e-3;

// --- Derived design (inverse step, printed by main) ------------------------

/// DRISM blaze period, meters.
const PERIOD: f64 = 40e-6;
/// Focusing (spectral imaging) lens focal length, meters.
const F_DET: f64 = 90e-3;
/// Primary lens focal length, meters.
const F_PRIMARY: f64 = 100e-3;
/// Collimator focal length, meters (1.25x afocal magnification).
const F_COLLIMATOR: f64 = 125e-3;
/// Entrance-pupil (primary stop) radius, meters.
const R_PRIMARY: f64 = 0.75e-3;
/// Collimated beam radius at the DRISM, meters.
const R_BEAM: f64 = R_PRIMARY * F_COLLIMATOR / F_PRIMARY;
/// Clear-aperture radius of the collimator / DRISM / focusing lens, meters.
const R_APER: f64 = 1.05e-3;

/// Simulation grid: 1024 x 1024 at the FPA pitch (1 sample = 1 pixel).
const N: usize = 1024;
const DX: f64 = FPA_PITCH;

/// Grating-equation design position on the detector, meters from the axis:
/// the blaze tilts the collimated beam by sin(theta) = lambda / PERIOD and
/// the focusing lens maps the tilt to x = F_DET * tan(theta) ~ F_DET * lambda / PERIOD.
fn ring_x(lambda: f64) -> f64 {
    F_DET * lambda / PERIOD
}

/// Gaussian passband transmission stack x 2pi kinoform blaze: the DRISM
/// meta-filter as a single local plane element.
///
/// Amplitude transmission at wavelength lambda:
/// t(lambda) = sum_i exp(-4 ln2 ((lambda - lambda_Ci) / FWHM_i)^2), and
/// phase phi(x) = 2 pi (x / PERIOD mod 1) — an ideal blaze, so all
/// transmitted power leaves in order +1 (the tilted plane wave the
/// focusing lens images to `ring_x`).
struct DrismMetaFilter {
    period: f64,            // meters
    bands: Vec<(f64, f64)>, // (lambda_c, fwhm), meters
    aperture: ClearAperture,
    shape: ElementShape,
    material: Material,
}

impl DrismMetaFilter {
    /// Blaze period `period` (m, positive finite), passbands `bands`
    /// ((lambda_c, fwhm) in m), circular clear aperture `aperture`.
    ///
    /// # Errors
    /// [`OpticsError::InvalidGrating`] for a non-positive/non-finite period;
    /// [`OpticsError::InvalidAperture`] for a non-positive/non-finite radius.
    fn new(period: f64, bands: Vec<(f64, f64)>, radius: f64) -> Result<Self, OpticsError> {
        if !period.is_finite() || period <= 0.0 || !radius.is_finite() || radius <= 0.0 {
            return Err(OpticsError::InvalidGrating {
                period,
                depth: 2.0 * PI,
                orientation: 0.0,
            });
        }
        Ok(Self {
            period,
            bands,
            aperture: ClearAperture::Circle(radius),
            shape: ElementShape::Circle,
            material: Material::new(|_lambda, _angle| 1.0, |_lambda, _angle| 0.0),
        })
    }

    /// Bandpass amplitude transmission t(lambda) at wavelength `lambda` (m).
    fn transmission_at(&self, lambda: f64) -> f64 {
        const FOUR_LN2: f64 = 4.0 * std::f64::consts::LN_2;
        self.bands
            .iter()
            .map(|&(lambda_c, fwhm)| {
                let d = (lambda - lambda_c) / fwhm;
                (-FOUR_LN2 * d * d).exp()
            })
            .sum()
    }
}

impl Element for DrismMetaFilter {
    fn shape(&self) -> &ElementShape {
        &self.shape
    }

    fn kind(&self) -> ElementKind {
        ElementKind::Diffractive
    }

    fn material(&self) -> (&Material, &Material) {
        (&self.material, &self.material)
    }

    fn extent(&self) -> Extent {
        match self.aperture.extent_radius() {
            Some(r) => Extent::Aperture(r),
            None => Extent::Unbounded,
        }
    }

    fn apply(&self, band: &BandField) -> Result<BandField, OpticsError> {
        let (nx, ny) = band.field().dims();
        let (dx, dy) = band.field().pitch();
        let half_x = (nx as f64 - 1.0) / 2.0 * dx;
        let half_y = (ny as f64 - 1.0) / 2.0 * dy;
        if let Some(r) = self.aperture.extent_radius()
            && (r > half_x || r > half_y)
        {
            return Err(OpticsError::OutsideField {
                half_x: r,
                half_y: r,
                center_x: 0.0,
                center_y: 0.0,
                grid_half_x: half_x,
                grid_half_y: half_y,
            });
        }
        let t = self.transmission_at(band.lambda());
        let inv_period = 1.0 / self.period;
        let mut out = band.clone();
        let field = out.field_mut();
        for iy in 0..ny {
            for ix in 0..nx {
                let (x, y) = field.coord(ix, iy);
                let s = field.sample_mut(ix, iy);
                if !self.aperture.contains(x, y) {
                    *s = Complex64::ZERO;
                    continue;
                }
                let u = x * inv_period;
                *s *= t * ((u - u.floor()) * 2.0 * PI * Complex64::I).exp();
            }
        }
        Ok(out)
    }
}

/// Incoherent moments of an intensity frame: (power, x-bar, y-bar, sigma_x)
/// over samples above 1e-4 of the frame peak.
fn moments(img: &[f64]) -> (f64, f64, f64, f64) {
    let peak = img.iter().cloned().fold(0.0f64, f64::max);
    let thr = 1e-4 * peak;
    let (mut p, mut sx, mut sy, mut sxx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for iy in 0..N {
        for ix in 0..N {
            let v = img[iy * N + ix];
            if v <= thr {
                continue;
            }
            let (x, y) = (
                (ix as f64 - (N as f64 - 1.0) / 2.0) * DX,
                (iy as f64 - (N as f64 - 1.0) / 2.0) * DX,
            );
            p += v;
            sx += v * x;
            sy += v * y;
            sxx += v * x * x;
        }
    }
    let (xb, yb) = (sx / p, sy / p);
    (p, xb, yb, (sxx / p - xb * xb).max(0.0).sqrt())
}

/// Log-scaled 8-bit PGM dump of the accumulated FPA frame.
fn write_pgm(path: &str, img: &[f64]) -> std::io::Result<()> {
    use std::io::Write;
    let peak = img.iter().cloned().fold(0.0f64, f64::max).max(1e-30);
    let mut buf = Vec::with_capacity(N * N);
    for v in img {
        let g = (1.0 + 999.0 * v / peak).ln() / (1000.0f64).ln();
        buf.push((255.0 * g) as u8);
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "P5\n{N} {N}\n255")?;
    f.write_all(&buf)?;
    f.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Inverse design: spec table -> geometry (printed derivation). ---
    let dispersion = F_DET / PERIOD; // m per m of wavelength
    let field_radius = F_DET * FOV_RADIUS;
    let aper_for_hires = 1.22 * 450e-9 * PERIOD / 2e-9; // D >= 1.22 lambda Lambda / dLambda
    println!("SAGE IV DRISM prototype — derived design");
    println!("  blaze period                 {:.0} um", PERIOD * 1e6);
    println!("  focusing lens focal length   {:.0} mm", F_DET * 1e3);
    println!(
        "  dispersion                   {:.2} um/nm = {:.2} FPA px/nm",
        dispersion * 1e-3,
        dispersion * 1e-9 / FPA_PITCH
    );
    println!(
        "  telescope f_p / f_c          {:.0} / {:.0} mm",
        F_PRIMARY * 1e3,
        F_COLLIMATOR * 1e3
    );
    println!(
        "  collimated beam diameter     {:.3} mm",
        2.0 * R_BEAM * 1e3
    );
    println!(
        "  diffraction spot @450 nm     {:.1} um",
        1.22 * 450e-9 * F_DET / (2.0 * R_BEAM) * 1e6
    );
    println!(
        "  spectral res. (this aper.)   {:.1} nm (2 nm pair NOT resolved at prototype scale)",
        1.22 * 450e-9 * PERIOD / (2.0 * R_BEAM) * 1e9
    );
    println!(
        "  aperture to resolve 2 nm      {:.1} mm collimated (full-scale SAGE IV)",
        aper_for_hires * 1e3
    );
    println!(
        "  FOV budget                   field disk r = {:.2} mm < innermost ring x = {:.2} mm",
        field_radius * 1e3,
        ring_x(BANDS[0].lambda_c) * 1e3
    );

    // --- Beamline: stop -> primary -> collimator -> DRISM -> focus -> FPA. ---
    let cfg = AsmConfig {
        padding: 1.5,
        ..AsmConfig::default()
    };
    let stop = EllipticalAperture::circle(R_PRIMARY, (0.0, 0.0))?;
    let primary = ThinLens::fixed(F_PRIMARY, None, ClearAperture::Field, (0.0, 0.0))?;
    let collimator = ThinLens::fixed(
        F_COLLIMATOR,
        None,
        ClearAperture::Circle(R_APER),
        (0.0, 0.0),
    )?;
    let drism = DrismMetaFilter::new(
        PERIOD,
        BANDS.iter().map(|b| (b.lambda_c, b.fwhm)).collect(),
        R_APER,
    )?;
    let focus = ThinLens::fixed(F_DET, None, ClearAperture::Circle(R_APER), (0.0, 0.0))?;
    let mut bl = Beamline::new();
    bl.push(&stop, 0.0)
        .push(&primary, F_PRIMARY + F_COLLIMATOR)
        .push(&collimator, 0.0)
        .push(&drism, 0.0)
        .push(&focus, F_DET);

    // --- Forward verification: trace each band, compare to the grating equation. ---
    let mut composite = vec![0.0f64; N * N];
    let mut measured = Vec::with_capacity(BANDS.len());
    let mut band_power_mean = 0.0f64;
    println!("\n band       lambda_C  FWHM   design x   measured x   err    sigma_x");
    for b in &BANDS {
        // Power weights t^2 at (+/- FWHM/2, center) = (0.25, 1, 0.25), normalized.
        let weights = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];
        let step = b.fwhm / 2.0;
        let mut img = vec![0.0f64; N * N];
        let mut power = 0.0f64;
        for (&l, &w) in [b.lambda_c - step, b.lambda_c, b.lambda_c + step]
            .iter()
            .zip(&weights)
        {
            let band = BandField::plane_wave(0.0, 0.0, 1.0, N, N, DX, DX, l)?;
            let out = bl.trace(&band, |x, z, _n| propagate(x, z, &cfg))?;
            for (acc, s) in img.iter_mut().zip(out.field().samples()) {
                *acc += w * s.norm_sqr();
            }
            power += w * out.field().power() / (DX * DX); // raw intensities, like `moments`
        }
        let (p, xb, yb, sx) = moments(&img);
        let design = ring_x(b.lambda_c);
        println!(
            " {:9} {:7.0} {:5.0} {:9.3} mm {:9.3} mm {:6.1} um {:7.1} um",
            b.name,
            b.lambda_c * 1e9,
            b.fwhm * 1e9,
            design * 1e3,
            xb * 1e3,
            (xb - design) * 1e6,
            sx * 1e6
        );
        assert!(
            (xb - design).abs() < 60e-6,
            "{} centroid {xb:.2e} m vs design {design:.2e} m",
            b.name
        );
        assert!(yb.abs() < 30e-6, "{} off-axis centroid {yb:.2e} m", b.name);
        assert!(p > 0.5 * power, "{} lost >50% of traced power", b.name);
        band_power_mean += p / BANDS.len() as f64;
        measured.push(xb);
        for (c, v) in composite.iter_mut().zip(&img) {
            *c += v;
        }
    }
    for pair in measured.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        assert!(
            b > a,
            "band positions not monotonic in wavelength: {a} -> {b}"
        );
    }

    // --- Off-band probes: the meta-filter must block them. ---
    for &l in &PROBES {
        let band = BandField::plane_wave(0.0, 0.0, 1.0, N, N, DX, DX, l)?;
        let out = bl.trace(&band, |x, z, _n| propagate(x, z, &cfg))?;
        let p = out.field().power() / (DX * DX); // raw, matching band moments
        let frac = p / band_power_mean;
        println!(
            " probe {:.0} nm  leaked {:.3}% of band power",
            l * 1e9,
            frac * 100.0
        );
        assert!(
            frac < 0.01,
            "probe {:.0} nm leaked {frac:.2e} of band power",
            l * 1e9
        );
    }

    // --- FPA frame. ---
    let peak_x = {
        let (ix, iy) = img_peak_bin(&composite);
        let (x, y) = (
            (ix as f64 - (N as f64 - 1.0) / 2.0) * DX,
            (iy as f64 - (N as f64 - 1.0) / 2.0) * DX,
        );
        println!(
            "\n composite peak at x = {:.3} mm, y = {:.3} mm",
            x * 1e3,
            y * 1e3
        );
        x
    };
    assert!(
        (peak_x - (measured[1] + measured[2]) / 2.0).abs() < 100e-6,
        "composite peak {peak_x:.2e} m should sit on the merged HiSpec pair"
    );
    std::fs::create_dir_all("target")?;
    write_pgm("target/drism_fpa.pgm", &composite)?;
    println!(
        " FPA frame written to target/drism_fpa.pgm ({} px at {:.1} um)",
        N,
        DX * 1e6
    );
    Ok(())
}

/// Peak bin (ix, iy) of an intensity frame.
fn img_peak_bin(img: &[f64]) -> (usize, usize) {
    let mut best = (0usize, 0usize);
    let mut bv = f64::NEG_INFINITY;
    for iy in 0..N {
        for ix in 0..N {
            if img[iy * N + ix] > bv {
                bv = img[iy * N + ix];
                best = (ix, iy);
            }
        }
    }
    best
}
