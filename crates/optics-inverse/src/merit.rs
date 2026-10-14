//! Merit functions: scalar scores over an [`Observation`].
//!
//! Convention: **all merits are minimized**. "Maximize EE" is expressed
//! as the merit `-EE(r)`, "minimize spot" directly as spot RMS.
//!
//! The trait has the two halves the Phase-9 framework needs:
//!
//! - [`Merit::evaluate`] — the scalar. Derivative-free and FD solvers use
//!   only this, so *any* user-defined score (including CRLB-class merits
//!   via the 0.2.0 frontend) is immediately optimizable.
//! - [`Merit::obs_gradient`] — ∂M/∂data over the flat
//!   [`Observation::data`] vector, the adjoint source for task 9.9.
//!   `None` means "use finite differences on the observation" (fallback
//!   for opaque user merits).
//!
//! Built-ins wrap the core `analysis` metrics on a [`Selector`]-picked
//! frame (channel × band × scene, with band/scene aggregation).

use crate::Result;
use crate::error::InverseError;
use crate::observe::{Channel, Observation};

/// Which part of an observation a merit reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Selector {
    /// Channel to read (usually [`Channel::Intensity`]).
    pub channel: Channel,
    /// Band index; `None` sums over bands (incoherent).
    pub band: Option<usize>,
    /// Scene index; `None` sums over scenes.
    pub scene: Option<usize>,
}

impl Selector {
    /// Intensity of band `b`, scene `s`.
    pub fn intensity(band: Option<usize>, scene: Option<usize>) -> Self {
        Self {
            channel: Channel::Intensity,
            band,
            scene,
        }
    }

    /// Extract the selected frame (summed where `None`), same length as
    /// one observation frame.
    pub fn frame(&self, obs: &Observation) -> Result<Vec<f64>> {
        let ci = obs
            .channel_index(self.channel)
            .ok_or_else(|| InverseError::Render {
                reason: format!("merit needs channel {:?}, not rendered", self.channel),
            })?;
        let bands: Box<dyn Iterator<Item = usize>> = match self.band {
            Some(b) => Box::new(std::iter::once(b)),
            None => Box::new(0..obs.bands.len()),
        };
        let scenes: Box<dyn Iterator<Item = usize>> = match self.scene {
            Some(s) => Box::new(std::iter::once(s)),
            None => Box::new(0..obs.scenes.len()),
        };
        let scenes_v: Vec<usize> = scenes.collect();
        let mut acc = vec![0.0f64; obs.frame_len()];
        for bi in bands {
            if bi >= obs.bands.len() {
                return Err(InverseError::Render {
                    reason: format!("band index {bi} out of range"),
                });
            }
            for &si in &scenes_v {
                if si >= obs.scenes.len() {
                    return Err(InverseError::Render {
                        reason: format!("scene index {si} out of range"),
                    });
                }
                for (a, v) in acc.iter_mut().zip(obs.slice(ci, bi, si)) {
                    *a += v;
                }
            }
        }
        Ok(acc)
    }
}

/// A scalar merit over an observation (minimized by solvers).
pub trait Merit {
    /// Short label for reports.
    fn name(&self) -> &str;

    /// Merit value (lower is better by convention).
    fn evaluate(&self, obs: &Observation) -> Result<f64>;

    /// Analytic ∂M/∂data (flat, `obs.data.len()`), the adjoint source for
    /// task 9.9. `None` ⇒ solvers fall back to FD on the observation.
    fn obs_gradient(&self, _obs: &Observation) -> Option<Vec<f64>> {
        None
    }
}

/// Wrap a renderer + merit into the scalar objective solvers consume.
pub fn objective<'a, 'c>(
    renderer: &'a crate::observe::Renderer<'c>,
    merit: &'a dyn Merit,
) -> impl Fn(&[f64]) -> Result<f64> + 'a {
    move |values| merit.evaluate(&renderer.render(values)?)
}

// ---------------------------------------------------------------------------
// Built-in merits
// ---------------------------------------------------------------------------

/// Mean squared error of a selected intensity frame against a target
/// frame: `M = (1/n) Σ (I − T)²`. Analytic gradient provided.
pub struct MseToTarget {
    /// Target samples (one frame, row-major; length must match).
    pub target: Vec<f64>,
    /// Frame selector (channel should be intensity-like).
    pub selector: Selector,
}

impl Merit for MseToTarget {
    fn name(&self) -> &str {
        "mse_to_target"
    }

    fn evaluate(&self, obs: &Observation) -> Result<f64> {
        let frame = self.selector.frame(obs)?;
        if frame.len() != self.target.len() {
            return Err(InverseError::Render {
                reason: format!(
                    "target length {} != frame length {}",
                    self.target.len(),
                    frame.len()
                ),
            });
        }
        Ok(frame
            .iter()
            .zip(&self.target)
            .map(|(a, t)| (a - t) * (a - t))
            .sum::<f64>()
            / frame.len() as f64)
    }

    fn obs_gradient(&self, obs: &Observation) -> Option<Vec<f64>> {
        let ci = obs.channel_index(self.selector.channel)?;
        let n = obs.frame_len() as f64;
        let mut g = vec![0.0f64; obs.data.len()];
        let frames = obs.bands.len() * obs.scenes.len();
        for bi in 0..obs.bands.len() {
            if self.selector.band.is_some_and(|b| b != bi) {
                continue;
            }
            for si in 0..obs.scenes.len() {
                if self.selector.scene.is_some_and(|s| s != si) {
                    continue;
                }
                let start = (ci * frames + bi * obs.scenes.len() + si) * obs.frame_len();
                for k in 0..obs.frame_len() {
                    g[start + k] = 2.0 * (obs.data[start + k] - self.target[k]) / n;
                }
            }
        }
        Some(g)
    }
}

/// Negative encircled energy at radius `r`: minimizing this maximizes the
/// fraction of power inside the circle. Analytic gradient provided.
pub struct EncircledEnergyMerit {
    /// EE radius (meters, from the frame origin).
    pub radius: f64,
    /// Frame selector.
    pub selector: Selector,
}

impl EncircledEnergyMerit {
    /// Inside-mask for the frame geometry.
    fn mask(&self, obs: &Observation) -> Vec<bool> {
        let (nx, ny) = obs.dims;
        let (dx, dy) = obs.pitch;
        (0..ny)
            .flat_map(|iy| {
                let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dy;
                (0..nx).map(move |ix| {
                    let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx;
                    x * x + y * y <= self.radius * self.radius
                })
            })
            .collect()
    }
}

impl Merit for EncircledEnergyMerit {
    fn name(&self) -> &str {
        "neg_encircled_energy"
    }

    fn evaluate(&self, obs: &Observation) -> Result<f64> {
        let frame = self.selector.frame(obs)?;
        let mask = self.mask(obs);
        let total: f64 = frame.iter().sum();
        if total <= 0.0 {
            return Ok(0.0);
        }
        let inside: f64 = frame
            .iter()
            .zip(&mask)
            .map(|(v, m)| if *m { *v } else { 0.0 })
            .sum();
        Ok(-inside / total)
    }

    fn obs_gradient(&self, obs: &Observation) -> Option<Vec<f64>> {
        // EE = S_in/S_tot; dEE/dI_k = (m_k S_tot − S_in)/S_tot^2.
        let ci = obs.channel_index(self.selector.channel)?;
        let frame = self.selector.frame(obs).ok()?;
        let mask = self.mask(obs);
        let total: f64 = frame.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let inside: f64 = frame
            .iter()
            .zip(&mask)
            .map(|(v, m)| if *m { *v } else { 0.0 })
            .sum();
        let mut g = vec![0.0f64; obs.data.len()];
        let frames = obs.bands.len() * obs.scenes.len();
        for bi in 0..obs.bands.len() {
            if self.selector.band.is_some_and(|b| b != bi) {
                continue;
            }
            for si in 0..obs.scenes.len() {
                if self.selector.scene.is_some_and(|s| s != si) {
                    continue;
                }
                let start = (ci * frames + bi * obs.scenes.len() + si) * obs.frame_len();
                for (k, m) in mask.iter().enumerate() {
                    g[start + k] = -((if *m { total } else { 0.0 } - inside) / (total * total));
                }
            }
        }
        Some(g)
    }
}

/// Spot RMS radius $\sqrt{\sum I \rho^2 / \sum I}$ (meters) — directly
/// minimized. Analytic gradient provided.
pub struct SpotRmsMerit {
    /// Frame selector.
    pub selector: Selector,
}

impl SpotRmsMerit {
    fn coords(obs: &Observation) -> Vec<f64> {
        let (nx, ny) = obs.dims;
        let (dx, dy) = obs.pitch;
        (0..ny)
            .flat_map(|iy| {
                let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dy;
                (0..nx).map(move |ix| {
                    let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx;
                    x * x + y * y
                })
            })
            .collect()
    }
}

impl Merit for SpotRmsMerit {
    fn name(&self) -> &str {
        "spot_rms"
    }

    fn evaluate(&self, obs: &Observation) -> Result<f64> {
        let frame = self.selector.frame(obs)?;
        let r2 = Self::coords(obs);
        let sw: f64 = frame.iter().sum();
        if sw <= 0.0 {
            return Ok(0.0);
        }
        let swr2: f64 = frame.iter().zip(&r2).map(|(w, q)| w * q).sum();
        Ok((swr2 / sw).sqrt())
    }

    fn obs_gradient(&self, obs: &Observation) -> Option<Vec<f64>> {
        // R = sqrt(S_r2/S_w); dR/dI_k = (rho_k^2 - R^2) / (2 R S_w).
        let ci = obs.channel_index(self.selector.channel)?;
        let frame = self.selector.frame(obs).ok()?;
        let r2 = Self::coords(obs);
        let sw: f64 = frame.iter().sum();
        if sw <= 0.0 {
            return None;
        }
        let swr2: f64 = frame.iter().zip(&r2).map(|(w, q)| w * q).sum();
        let rr = swr2 / sw;
        let root = rr.sqrt();
        if root <= 0.0 {
            return None;
        }
        let mut g = vec![0.0f64; obs.data.len()];
        let frames = obs.bands.len() * obs.scenes.len();
        for bi in 0..obs.bands.len() {
            if self.selector.band.is_some_and(|b| b != bi) {
                continue;
            }
            for si in 0..obs.scenes.len() {
                if self.selector.scene.is_some_and(|s| s != si) {
                    continue;
                }
                let start = (ci * frames + bi * obs.scenes.len() + si) * obs.frame_len();
                for (k, q) in r2.iter().enumerate() {
                    g[start + k] = (q - rr) / (2.0 * root * sw);
                }
            }
        }
        Some(g)
    }
}

/// Negative Strehl ratio of the selected frame against a reference frame
/// (power-equalized peak ratio, core `analysis::strehl` semantics).
/// Gradient falls back to FD (`None`).
pub struct StrehlMerit {
    /// Reference intensity frame (same grid as the selected frame).
    pub reference: Vec<f64>,
    /// Frame selector.
    pub selector: Selector,
}

impl Merit for StrehlMerit {
    fn name(&self) -> &str {
        "neg_strehl"
    }

    fn evaluate(&self, obs: &Observation) -> Result<f64> {
        let frame = self.selector.frame(obs)?;
        if frame.len() != self.reference.len() {
            return Err(InverseError::Render {
                reason: "strehl reference length != frame length".into(),
            });
        }
        let pm: f64 = frame.iter().sum();
        let pr: f64 = self.reference.iter().sum();
        if pm <= 0.0 || pr <= 0.0 {
            return Ok(0.0);
        }
        let peak = |v: &[f64]| v.iter().cloned().fold(0.0f64, f64::max);
        Ok(-((peak(&frame) / pm) / (peak(&self.reference) / pr)))
    }
}

/// Negative MTF at spatial frequency `freq` (cycles/m) along x, computed
/// as |FT(I)|/ΣI on the frame grid (core `analysis::mtf` semantics,
/// evaluated at one frequency). Gradient falls back to FD.
pub struct MtfMerit {
    /// Frequency (cycles/m) — must be below Nyquist 1/(2 dx).
    pub freq: f64,
    /// Frame selector.
    pub selector: Selector,
}

impl Merit for MtfMerit {
    fn name(&self) -> &str {
        "neg_mtf"
    }

    fn evaluate(&self, obs: &Observation) -> Result<f64> {
        let frame = self.selector.frame(obs)?;
        let (nx, ny) = obs.dims;
        let dx = obs.pitch.0;
        if self.freq < 0.0 || 2.0 * self.freq * dx >= 1.0 {
            return Err(InverseError::Render {
                reason: format!("MTF frequency {} beyond Nyquist", self.freq),
            });
        }
        // k-th DFT bin along x for frequency f: k = f * nx * dx (integer).
        let k = self.freq * nx as f64 * dx;
        let k_int = k.round() as usize;
        if (k - k_int as f64).abs() > 1e-9 {
            return Err(InverseError::Render {
                reason: format!(
                    "MTF frequency must land on a DFT bin (f = m / (nx*dx)); got {freq}",
                    freq = self.freq
                ),
            });
        }
        // |sum_{ix} I(ix,iy) e^{-2pi i k ix/nx}| summed coherently over rows:
        // analysis::mtf reads the fy=0 DFT row, i.e. sum over iy too.
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (idx, v) in frame.iter().enumerate() {
            let ix = (idx % nx) as f64;
            let ang = -2.0 * std::f64::consts::PI * (k_int as f64) * ix / nx as f64;
            re += v * ang.cos();
            im += v * ang.sin();
        }
        let dc: f64 = frame.iter().sum();
        if dc <= 0.0 {
            return Ok(0.0);
        }
        Ok(-(re.hypot(im) / dc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_obs(nx: usize, f: &dyn Fn(usize, usize) -> f64) -> Observation {
        let mut data = Vec::with_capacity(nx * nx);
        for iy in 0..nx {
            for ix in 0..nx {
                data.push(f(ix, iy));
            }
        }
        Observation {
            channels: vec![Channel::Intensity],
            bands: vec![500e-9],
            scenes: vec![(0.0, 0.0)],
            dims: (nx, nx),
            pitch: (10e-6, 10e-6),
            data,
            fd_steps: (1e-6, 1e-9),
        }
    }

    fn gaussian_frame(nx: usize) -> Vec<f64> {
        let w = (nx as f64 * 10e-6) / 6.0;
        (0..nx)
            .flat_map(|iy| {
                let y = (iy as f64 - (nx as f64 - 1.0) / 2.0) * 10e-6;
                (0..nx).map(move |ix| {
                    let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * 10e-6;
                    (-2.0 * (x * x + y * y) / (w * w)).exp()
                })
            })
            .collect()
    }

    #[test]
    fn mse_matches_manual_and_gradient_fd() {
        let obs = synth_obs(8, &|ix, iy| (ix + iy) as f64);
        let target = vec![1.0; 64];
        let m = MseToTarget {
            target: target.clone(),
            selector: Selector::intensity(Some(0), Some(0)),
        };
        let want: f64 = obs
            .data
            .iter()
            .zip(&target)
            .map(|(a, t)| (a - t) * (a - t))
            .sum::<f64>()
            / 64.0;
        assert!((m.evaluate(&obs).unwrap() - want).abs() < 1e-12);

        // Gradient vs central differences on the data vector.
        let g = m.obs_gradient(&obs).unwrap();
        let eps = 1e-7;
        for k in [0usize, 17, 63] {
            let mut up = obs.clone();
            up.data[k] += eps;
            let mut dn = obs.clone();
            dn.data[k] -= eps;
            let fd = (m.evaluate(&up).unwrap() - m.evaluate(&dn).unwrap()) / (2.0 * eps);
            assert!(
                (fd - g[k]).abs() < 1e-5 * (1.0 + fd.abs()),
                "k {k}: fd {fd} vs g {}",
                g[k]
            );
        }
    }

    #[test]
    fn ee_and_spot_match_core_analysis() {
        use optics_core::field::Psf;
        let nx = 32usize;
        let frame = gaussian_frame(nx);
        let psf = Psf::new(frame.clone(), nx, nx, 10e-6, 10e-6).unwrap();
        let obs = synth_obs(nx, &|ix, iy| frame[iy * nx + ix]);

        // EE: exact cumulative fraction from the core curve at the last
        // knot <= 20 um (EncircledEnergy::value_at interpolates between
        // knots; this merit uses the pixel-exact mask).
        let ee = optics_core::analysis::encircled_energy(&psf);
        let (radii, frac) = ee.curve();
        let want = -frac
            .iter()
            .zip(radii)
            .filter(|(_, r)| **r <= 20e-6 + 1e-12)
            .last()
            .map(|(f, _)| *f)
            .unwrap();
        let got = EncircledEnergyMerit {
            radius: 20e-6,
            selector: Selector::intensity(None, None),
        }
        .evaluate(&obs)
        .unwrap();
        assert!((got - want).abs() < 1e-3, "EE merit {got} vs core {want}");

        // Spot RMS.
        let want = optics_core::analysis::spot_rms_radius(&psf);
        let got = SpotRmsMerit {
            selector: Selector::intensity(None, None),
        }
        .evaluate(&obs)
        .unwrap();
        assert!(
            (got - want).abs() / want < 1e-9,
            "spot {got} vs core {want}"
        );

        // Strehl vs itself = -1.
        let s = StrehlMerit {
            reference: frame.clone(),
            selector: Selector::intensity(None, None),
        };
        assert!((s.evaluate(&obs).unwrap() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn ee_and_spot_gradients_fd_check() {
        let nx = 16usize;
        let frame = gaussian_frame(nx);
        let obs = synth_obs(nx, &|ix, iy| frame[iy * nx + ix]);
        for merit in [
            Box::new(EncircledEnergyMerit {
                radius: 30e-6,
                selector: Selector::intensity(None, None),
            }) as Box<dyn Merit>,
            Box::new(SpotRmsMerit {
                selector: Selector::intensity(None, None),
            }) as Box<dyn Merit>,
        ] {
            let g = merit
                .obs_gradient(&obs)
                .unwrap_or_else(|| panic!("{} lacks gradient", merit.name()));
            let eps = 1e-9;
            for k in [0usize, 68, 255] {
                let mut up = obs.clone();
                up.data[k] += eps;
                let mut dn = obs.clone();
                dn.data[k] -= eps;
                let fd =
                    (merit.evaluate(&up).unwrap() - merit.evaluate(&dn).unwrap()) / (2.0 * eps);
                let tol = 1e-4 * (fd.abs() + 1.0);
                assert!(
                    (fd - g[k]).abs() < tol,
                    "{} k {k}: fd {fd} vs g {}",
                    merit.name(),
                    g[k]
                );
            }
        }
    }

    #[test]
    fn mtf_at_bin_matches_core_analysis_on_gaussian() {
        use optics_core::field::Psf;
        let nx = 32usize;
        let frame = gaussian_frame(nx);
        let psf = Psf::new(frame.clone(), nx, nx, 10e-6, 10e-6).unwrap();
        let obs = synth_obs(nx, &|ix, iy| frame[iy * nx + ix]);
        let core = optics_core::analysis::mtf(&psf);
        // Pick core.freqs[4] (a bin frequency).
        let f = core.freqs[4];
        let got = MtfMerit {
            freq: f,
            selector: Selector::intensity(None, None),
        }
        .evaluate(&obs)
        .unwrap();
        assert!(
            (got + core.values[4]).abs() < 2e-3,
            "MTF merit {got} vs core {}",
            -core.values[4]
        );
    }

    #[test]
    fn selector_aggregation_and_errors() {
        let mut obs = synth_obs(4, &|_, _| 1.0);
        obs.bands = vec![450e-9, 550e-9];
        obs.data = vec![1.0; 2 * 16]; // 2 bands, 1 scene
        obs.scenes = vec![(0.0, 0.0)];
        let m = EncircledEnergyMerit {
            radius: 1e-3,
            selector: Selector::intensity(None, None),
        };
        // radius covers whole frame => EE = 1 => merit = -1.
        assert!((m.evaluate(&obs).unwrap() + 1.0).abs() < 1e-12);
        // Out-of-range band errors.
        let bad = Selector::intensity(Some(5), None);
        assert!(bad.frame(&obs).is_err());
        // Missing channel errors.
        let miss = Selector {
            channel: Channel::DxDyX,
            band: None,
            scene: None,
        };
        assert!(miss.frame(&obs).is_err());
    }
}
