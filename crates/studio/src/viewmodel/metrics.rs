//! Metric frames for the EE/MTF/Strehl/spot/wavefront panels
//! (docs/ux/spec.md §3.3).
//!
//! Every number here is produced by an `optics_core::analysis` call — the
//! UI crate never recomputes a metric (spec §6). This module only *binds*
//! the analysis outputs to display-ready structs and derives the shared
//! "radius/frequency scrubber" operating point the EE and MTF panels use.

use optics_core::analysis::{
    EncircledEnergy, Mtf, WavefrontDecomp, encircled_energy, marechal_strehl, mtf, spot_rms_radius,
    strehl as strehl_ratio, zernike_fit,
};
use optics_core::field::Psf;

/// Default Zernike truncation order for the wavefront panel (radial order
/// $n$; modes $\le (n+1)(n+2)/2$). Kept small — the fit is $O(N \cdot k^2)$
/// and the panel is a preview, not a full wavefront sensor.
pub const DEFAULT_ZERNIKE_N_MAX: u32 = 6;

/// Everything the metric panels show for one PSF frame.
#[derive(Clone, Debug)]
pub struct FrameMetrics {
    /// Encircled-energy curve + inverse lookup.
    pub ee: EncircledEnergy,
    /// 1-D MTF along $+f_x$ (cycles/m, normalized at DC).
    pub mtf: Mtf,
    /// Intensity-weighted RMS spot radius (m).
    pub spot_rms: f64,
    /// Peak centroid (m, m) — the spot panel's excursion readout.
    pub centroid: (f64, f64),
    /// Total power (W-proxy) for the locked-scale drift readout (spec §4).
    pub total_power: f64,
    /// Strehl ratio vs a reference PSF, when one is supplied.
    pub strehl: Option<f64>,
    /// Zernike decomposition of a pupil phase, when one is supplied.
    pub wavefront: Option<WavefrontDecomp>,
    /// Maréchal Strehl from the Zernike residual RMS (waves), when the
    /// wavefront fit ran.
    pub marechal_strehl: Option<f64>,
}

/// Optional analysis inputs that need a caller-supplied grid the `Psf`
/// alone cannot provide.
#[derive(Clone, Debug, Default)]
pub struct AnalysisInputs<'a> {
    /// Reference PSF for the peak-ratio Strehl (spec §3.3 "Strehl panel").
    pub reference: Option<&'a Psf>,
    /// Pupil phase grid (radians, row-major `nx`×`ny`) and its geometry for
    /// the Zernike fit + Maréchal estimate. `None` hides the wavefront
    /// panel's fit-derived rows.
    pub pupil_phase: Option<PupilPhase<'a>>,
    /// Zernike truncation order (default [`DEFAULT_ZERNIKE_N_MAX`]).
    pub zernike_n_max: Option<u32>,
}

/// A pupil phase grid with the geometry `zernike_fit` needs.
#[derive(Clone, Copy, Debug)]
pub struct PupilPhase<'a> {
    /// Phase samples (radians), row-major.
    pub values: &'a [f64],
    /// Grid sample counts.
    pub dims: (usize, usize),
    /// Grid pitch (m, m).
    pub pitch: (f64, f64),
    /// Pupil radius (m) centered on the grid origin.
    pub radius: f64,
}

/// An owned pupil phase grid (the shell/demo hands one to each frame so the
/// wavefront panel can run `zernike_fit` on construction).
#[derive(Clone, Debug)]
pub struct PupilPhaseGrid {
    /// Row-major phase samples (radians).
    pub values: Vec<f64>,
    /// Grid sample counts.
    pub dims: (usize, usize),
    /// Grid pitch (m, m).
    pub pitch: (f64, f64),
    /// Pupil radius (m).
    pub radius: f64,
}

impl PupilPhaseGrid {
    /// Borrow as the [`PupilPhase`] input [`compute`] takes.
    pub fn borrowed(&self) -> PupilPhase<'_> {
        PupilPhase {
            values: &self.values,
            dims: self.dims,
            pitch: self.pitch,
            radius: self.radius,
        }
    }
}

/// One viewport frame: a labeled PSF with its metrics precomputed.
///
/// Frames are built once (per run/import) by the shell or the demo; every
/// metric number comes from `optics_core::analysis` at build time, so panel
/// switching is pure display work.
#[derive(Clone, Debug)]
pub struct Frame {
    /// Tab label (band name, "sum", "reference", …).
    pub label: String,
    /// The detector-plane intensity grid.
    pub psf: Psf,
    /// Metrics computed from `psf` (analysis APIs only).
    pub metrics: FrameMetrics,
}

/// Builds a frame: runs [`compute`] with the optional reference PSF (Strehl
/// panel) and pupil phase (wavefront panel).
pub fn make_frame(
    label: impl Into<String>,
    psf: Psf,
    reference: Option<&Psf>,
    pupil: Option<&PupilPhaseGrid>,
) -> Frame {
    let inputs = AnalysisInputs {
        reference,
        pupil_phase: pupil.map(|p| p.borrowed()),
        zernike_n_max: None,
    };
    let metrics = compute(&psf, &inputs);
    Frame {
        label: label.into(),
        psf,
        metrics,
    }
}

/// Runs the analysis APIs for one frame.
///
/// `psf` is the detector-plane intensity grid; `inputs` add the reference
/// PSF and pupil phase. All core calls are the documented §3.3 bindings.
pub fn compute(psf: &Psf, inputs: &AnalysisInputs<'_>) -> FrameMetrics {
    let ee = encircled_energy(psf);
    let mtf_curve = mtf(psf);
    let spot_rms = spot_rms_radius(psf);
    let centroid = psf.centroid();
    let total_power = psf.total_power();

    let strehl = inputs.reference.map(|r| strehl_ratio(psf, r));

    let (wavefront, marechal_strehl) = match &inputs.pupil_phase {
        Some(p) => {
            let n_max = inputs.zernike_n_max.unwrap_or(DEFAULT_ZERNIKE_N_MAX);
            let (nx, ny) = p.dims;
            let (dx, dy) = p.pitch;
            match zernike_fit(p.values, nx, ny, dx, dy, p.radius, n_max) {
                Ok(w) => {
                    let rms = w.residual_rms_waves();
                    (Some(w), Some(marechal_strehl(rms)))
                }
                Err(_) => (None, None),
            }
        }
        None => (None, None),
    };

    FrameMetrics {
        ee,
        mtf: mtf_curve,
        spot_rms,
        centroid,
        total_power,
        strehl,
        wavefront,
        marechal_strehl,
    }
}

/// Readout strings and numbers for the EE panel: the Airy 84% radius, the
/// 50% (median) radius, and the curve value at the shared scrubber radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncircledEnergyReadout {
    /// Radius (m) enclosing 84% of the power (first Airy dark ring anchor).
    pub r84: f64,
    /// Radius (m) enclosing 50% of the power (median spot).
    pub r50: f64,
    /// Fraction within the scrubber radius.
    pub value_at_scrub: f64,
    /// The scrubber radius (m).
    pub scrub_radius: f64,
}

impl FrameMetrics {
    /// EE panel readouts at `scrub_radius` (m).
    pub fn ee_readout(&self, scrub_radius: f64) -> EncircledEnergyReadout {
        EncircledEnergyReadout {
            r84: self.ee.radius_for_fraction(0.84),
            r50: self.ee.radius_for_fraction(0.50),
            value_at_scrub: self.ee.value_at(scrub_radius),
            scrub_radius,
        }
    }
}

/// The operating point shared by the EE and MTF panels (spec §3.3
/// "radius/frequency scrubber"). A single normalized position $u \in [0, 1]$
/// maps to a radius (EE) and a frequency (MTF) so a design comparison keeps
/// one point; the panels interpolate their own curves for the display value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scrubber {
    /// Normalized position in $[0, 1]$.
    pub u: f64,
}

impl Scrubber {
    /// Clamps and builds a scrubber from a normalized position.
    pub fn new(u: f64) -> Self {
        Self {
            u: if u.is_finite() {
                u.clamp(0.0, 1.0)
            } else {
                0.0
            },
        }
    }

    /// Radius (m) at position `u` over `0..=r_max`.
    pub fn radius(&self, r_max: f64) -> f64 {
        self.u * r_max.max(0.0)
    }

    /// Frequency (cycles/m) at position `u` over `0..=f_max`.
    pub fn frequency(&self, f_max: f64) -> f64 {
        self.u * f_max.max(0.0)
    }
}

/// Largest radius the EE curve spans (its last sample), for the scrubber.
pub fn ee_max_radius(ee: &EncircledEnergy) -> f64 {
    let (r, _) = ee.curve();
    r.last().copied().unwrap_or(0.0)
}

/// Nyquist frequency the MTF curve spans (its last sample), for the scrubber.
pub fn mtf_max_freq(m: &Mtf) -> f64 {
    m.freqs.last().copied().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_psf(nx: usize, ny: usize, dx: f64, w: f64) -> Psf {
        // I = exp(-rho^2 / w^2); spot_rms anchor = w (docs/analysis module).
        let mut s = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dx;
            for ix in 0..nx {
                let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx;
                s.push((-(x * x + y * y) / (w * w)).exp());
            }
        }
        Psf::new(s, nx, ny, dx, dx).unwrap()
    }

    #[test]
    fn compute_binds_analysis_apis() {
        let w = 60e-6;
        let psf = gaussian_psf(201, 201, 10e-6, w);
        let m = compute(&psf, &AnalysisInputs::default());
        // Gaussian PSF: spot RMS ~= w.
        assert!((m.spot_rms - w) / w < 0.02, "spot_rms {}", m.spot_rms);
        // MTF normalized to 1 at DC.
        assert!((m.mtf.values[0] - 1.0).abs() < 1e-9);
        // EE reaches ~1 at the grid corner radius.
        let (_r, f) = m.ee.curve();
        assert!(*f.last().unwrap() > 0.99);
        assert!(m.strehl.is_none());
        assert!(m.wavefront.is_none());
    }

    #[test]
    fn strehl_vs_self_is_one() {
        let psf = gaussian_psf(64, 64, 10e-6, 50e-6);
        let m = compute(
            &psf,
            &AnalysisInputs {
                reference: Some(&psf),
                ..Default::default()
            },
        );
        assert!((m.strehl.unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ee_readout_anchors() {
        let psf = gaussian_psf(201, 201, 10e-6, 60e-6);
        let m = compute(&psf, &AnalysisInputs::default());
        let r = m.ee_readout(0.0);
        assert!(r.r50 < r.r84, "median inside 84% radius");
        assert!(r.r50 > 0.0);
    }

    #[test]
    fn wavefront_fit_flat_phase_zero_residual() {
        // A flat (zero) pupil phase fits to ~zero coefficients, ~zero
        // residual, so Maréchal Strehl ~= 1.
        let n = 64;
        let dx = 20e-6;
        let phase = vec![0.0; n * n];
        let inputs = AnalysisInputs {
            pupil_phase: Some(PupilPhase {
                values: &phase,
                dims: (n, n),
                pitch: (dx, dx),
                radius: (n as f64 * dx) / 2.0 * 0.8,
            }),
            ..Default::default()
        };
        let psf = gaussian_psf(n, n, dx, 60e-6);
        let m = compute(&psf, &inputs);
        let w = m.wavefront.expect("fit ran");
        assert!(w.residual_rms < 1e-9);
        assert!(m.marechal_strehl.unwrap() > 0.999);
    }

    #[test]
    fn scrubber_maps_and_clamps() {
        let s = Scrubber::new(0.5);
        assert_eq!(s.radius(100.0), 50.0);
        assert_eq!(s.frequency(2e6), 1e6);
        assert_eq!(Scrubber::new(2.0).u, 1.0);
        assert_eq!(Scrubber::new(-1.0).u, 0.0);
        assert_eq!(Scrubber::new(f64::NAN).u, 0.0);
    }

    #[test]
    fn curve_extrema_helpers() {
        let psf = gaussian_psf(64, 64, 10e-6, 40e-6);
        let m = compute(&psf, &AnalysisInputs::default());
        assert!(ee_max_radius(&m.ee) > 0.0);
        assert!(mtf_max_freq(&m.mtf) > 0.0);
    }
}
