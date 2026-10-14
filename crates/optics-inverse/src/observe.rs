//! The observation model and renderer — how a design becomes data.
//!
//! [`Renderer`] turns a parameter vector into an [`Observation`]: flat
//! `f64` stacks of detector-plane intensity per (band, scene), optionally
//! with finite-difference derivative channels ∂I/∂x₀, ∂I/∂y₀, ∂I/∂λ.
//! The derivative channels are what make estimation-theoretic merits
//! (Fisher information / CRLB) expressible as pure functions of the
//! observation — see the module docs of `merit` and the Phase-9 plan.
//!
//! ## Layout convention (numpy-ready)
//!
//! `Observation.data` is row-major with strides
//! `[channel][band][scene][row][col]`, i.e. one flat array viewable as
//! shape `(C, B, S, Ny, Nx)` in numpy. Channel 0 is always intensity;
//! enabled derivative channels follow in x, y, λ order. All bands and
//! scenes must render on the same grid (validated).
//!
//! ## Physics notes (textbook anchors)
//!
//! - Scenes parameterize the **source position** (x₀, y₀): the renderer
//!   calls the source closure at FD-shifted positions, exploiting ASM
//!   translation invariance ($\partial I/\partial x_0 = -\partial I/\partial x$
//!   on the sampling grid — golden-tested).
//! - The λ channel uses central differences across the band, rebuilding
//!   the design at λ±Δ so chromatic carrier phase laws are included.
//! - A(δ)∘A(−δ) = identity for propagating content (the ASM transfer
//!   function conjugates under z → −z): the defocus/mask/defocus chain of
//!   a uniform-thickness relief is pure piston — collimation preserved
//!   (golden-tested; this fact is also the adjoint's foundation, task 9.9).

use optics_core::field::{BandField, Psf};
use optics_core::propagation::asm::{AsmConfig, propagate};

use crate::Result;
use crate::carrier::Carrier;
use crate::error::InverseError;

/// Which channels a renderer emits. Intensity is always channel 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Detector-plane intensity $I = |U|^2$.
    Intensity,
    /// $\partial I/\partial x_0$ (m⁻¹·I units) — source-position FD.
    DxDyX,
    /// $\partial I/\partial y_0$ (m⁻¹).
    DxDyY,
    /// $\partial I/\partial\lambda$ (m⁻¹).
    Dlambda,
}

/// Which derivative channels to compute (each costs two extra renders).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Derivatives {
    /// ∂I/∂x₀ via source-position central difference.
    pub x: bool,
    /// ∂I/∂y₀.
    pub y: bool,
    /// ∂I/∂λ via band central difference (design rebuilt at λ±Δ).
    pub lambda: bool,
}

/// A rendered design: flat stacks plus the metadata to interpret them.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Channel list (index = channel slot; `[0] == Channel::Intensity`).
    pub channels: Vec<Channel>,
    /// Band wavelengths (m), render order.
    pub bands: Vec<f64>,
    /// Scene source positions (x₀, y₀) in meters, render order.
    pub scenes: Vec<(f64, f64)>,
    /// Grid sample counts `(nx, ny)`.
    pub dims: (usize, usize),
    /// Grid pitch `(dx, dy)` in meters.
    pub pitch: (f64, f64),
    /// Flat data, layout `[channel][band][scene][row][col]`.
    pub data: Vec<f64>,
    /// FD steps actually used (m): `(scene_step, lambda_step)`.
    pub fd_steps: (f64, f64),
}

impl Observation {
    /// Number of pixels per frame.
    pub fn frame_len(&self) -> usize {
        self.dims.0 * self.dims.1
    }

    /// Channel slot index for a channel kind, if present.
    pub fn channel_index(&self, ch: Channel) -> Option<usize> {
        self.channels.iter().position(|c| *c == ch)
    }

    /// Row-major slice for one (channel, band, scene).
    ///
    /// # Panics
    /// Debug-asserts slot ranges; returns an empty view on out-of-range in
    /// release (callers iterate from `channels.len()`/`bands.len()`).
    pub fn slice(&self, ci: usize, bi: usize, si: usize) -> &[f64] {
        let n = self.frame_len();
        let frames = self.bands.len() * self.scenes.len();
        debug_assert!(ci < self.channels.len() && bi * self.scenes.len() + si < frames);
        let start = (ci * frames + bi * self.scenes.len() + si) * n;
        let end = (start + n).min(self.data.len());
        &self.data[start..end]
    }

    /// Write a frame slice (used by the renderer while filling).
    fn slice_mut(&mut self, ci: usize, bi: usize, si: usize) -> &mut [f64] {
        let n = self.frame_len();
        let frames = self.bands.len() * self.scenes.len();
        let start = (ci * frames + bi * self.scenes.len() + si) * n;
        &mut self.data[start..start + n]
    }

    /// Incoherent sum over bands of one channel (per-pixel, same grid) —
    /// the `field::sum_intensity` semantics applied to stored stacks.
    pub fn band_sum(&self, ch: Channel) -> Option<Vec<f64>> {
        let ci = self.channel_index(ch)?;
        let n = self.frame_len();
        let mut acc = vec![0.0; n];
        for bi in 0..self.bands.len() {
            for si in 0..self.scenes.len() {
                for (a, v) in acc.iter_mut().zip(self.slice(ci, bi, si)) {
                    *a += v;
                }
            }
        }
        Some(acc)
    }

    /// Rebuild a [`Psf`] for one (channel, band, scene) for the core
    /// analysis metrics (EE, MTF, Strehl, spot RMS).
    pub fn psf(&self, ch: Channel, bi: usize, si: usize) -> Result<Psf> {
        let ci = self.channel_index(ch).ok_or_else(|| InverseError::Render {
            reason: format!("channel {ch:?} not rendered"),
        })?;
        Psf::new(
            self.slice(ci, bi, si).to_vec(),
            self.dims.0,
            self.dims.1,
            self.pitch.0,
            self.pitch.1,
        )
        .map_err(|e| InverseError::Render {
            reason: e.to_string(),
        })
    }
}

/// Source initializer closure: `(x0, y0, lambda) -> input BandField`.
/// The renderer calls it at base and FD-shifted positions; it must return
/// the same grid geometry for every call.
pub type Source = dyn Fn(f64, f64, f64) -> Result<BandField>;

/// Renders designs into observations through the core ASM engine.
pub struct Renderer<'c> {
    /// The design carrier (borrowed; owns the parameter space).
    pub carrier: &'c Carrier,
    /// Source initializer (see [`Source`]).
    pub source: Box<Source>,
    /// Band wavelengths (m) — must be positive and finite.
    pub bands: Vec<f64>,
    /// Scene source positions (x₀, y₀) in meters.
    pub scenes: Vec<(f64, f64)>,
    /// ASM configuration (per-step medium is injected from the design).
    pub asm: AsmConfig,
    /// Derivative channels to compute.
    pub derivatives: Derivatives,
    /// Source-position FD step (m).
    pub scene_step: f64,
    /// Band FD step (m).
    pub lambda_step: f64,
}

impl<'c> Renderer<'c> {
    /// New renderer with defaults: no derivatives, 10 µm-class scene step,
    /// 1 nm band step, `AsmConfig::default()`.
    pub fn new(
        carrier: &'c Carrier,
        source: impl Fn(f64, f64, f64) -> Result<BandField> + 'static,
    ) -> Self {
        Self {
            carrier,
            source: Box::new(source),
            bands: vec![500e-9],
            scenes: vec![(0.0, 0.0)],
            asm: AsmConfig::default(),
            derivatives: Derivatives::default(),
            scene_step: 1e-6,
            lambda_step: 1e-9,
        }
    }

    /// One coherent render: build the design at `lambda`, trace the
    /// source through it, return the detector-plane intensity.
    fn render_frame(&self, values: &[f64], lambda: f64, x0: f64, y0: f64) -> Result<Vec<f64>> {
        let design = self.carrier.build(values, lambda)?;
        let input = (self.source)(x0, y0, lambda)?;
        let output = design
            .beamline()
            .trace(&input, |band, z, medium| {
                let cfg = AsmConfig { medium, ..self.asm };
                propagate(band, z, &cfg).map_err(|e| InverseError::Render {
                    reason: e.to_string(),
                })
            })
            .map_err(|e| InverseError::Render {
                reason: e.to_string(),
            })?;
        Ok(output.field().intensities())
    }

    /// Render the full observation for a parameter vector.
    pub fn render(&self, values: &[f64]) -> Result<Observation> {
        if self.bands.is_empty() || self.scenes.is_empty() {
            return Err(InverseError::Render {
                reason: "renderer needs at least one band and one scene".into(),
            });
        }
        if (self.derivatives.x || self.derivatives.y)
            && (!self.scene_step.is_finite() || self.scene_step <= 0.0)
        {
            return Err(InverseError::Render {
                reason: "scene_step must be positive and finite".into(),
            });
        }
        if self.derivatives.lambda && (!self.lambda_step.is_finite() || self.lambda_step <= 0.0) {
            return Err(InverseError::Render {
                reason: "lambda_step must be positive and finite".into(),
            });
        }

        let mut channels = vec![Channel::Intensity];
        if self.derivatives.x {
            channels.push(Channel::DxDyX);
        }
        if self.derivatives.y {
            channels.push(Channel::DxDyY);
        }
        if self.derivatives.lambda {
            channels.push(Channel::Dlambda);
        }

        // Validate geometry against the first frame.
        let first = self.render_frame(values, self.bands[0], self.scenes[0].0, self.scenes[0].1)?;
        let (dims, pitch) = {
            let b = (self.source)(self.scenes[0].0, self.scenes[0].1, self.bands[0])?;
            (b.field().dims(), b.field().pitch())
        };
        let n = dims.0 * dims.1;
        if first.len() != n {
            return Err(InverseError::Render {
                reason: "intensity grid mismatch".into(),
            });
        }

        let frames = self.bands.len() * self.scenes.len();
        let mut obs = Observation {
            channels: channels.clone(),
            bands: self.bands.clone(),
            scenes: self.scenes.clone(),
            dims,
            pitch,
            data: vec![0.0; channels.len() * frames * n],
            fd_steps: (self.scene_step, self.lambda_step),
        };
        obs.slice_mut(0, 0, 0).copy_from_slice(&first);

        for (bi, &lambda) in self.bands.iter().enumerate() {
            for (si, &(x0, y0)) in self.scenes.iter().enumerate() {
                let base = if bi == 0 && si == 0 {
                    first.clone()
                } else {
                    self.render_frame(values, lambda, x0, y0)?
                };
                Self::check_grid(&base, n, lambda, x0, y0)?;
                obs.slice_mut(0, bi, si).copy_from_slice(&base);

                if self.derivatives.x {
                    let hi = self.render_frame(values, lambda, x0 + self.scene_step, y0)?;
                    let lo = self.render_frame(values, lambda, x0 - self.scene_step, y0)?;
                    let d: Vec<f64> = hi
                        .iter()
                        .zip(&lo)
                        .map(|(h, l)| (h - l) / (2.0 * self.scene_step))
                        .collect();
                    obs.slice_mut(obs.channel_index(Channel::DxDyX).unwrap(), bi, si)
                        .copy_from_slice(&d);
                }
                if self.derivatives.y {
                    let hi = self.render_frame(values, lambda, x0, y0 + self.scene_step)?;
                    let lo = self.render_frame(values, lambda, x0, y0 - self.scene_step)?;
                    let d: Vec<f64> = hi
                        .iter()
                        .zip(&lo)
                        .map(|(h, l)| (h - l) / (2.0 * self.scene_step))
                        .collect();
                    obs.slice_mut(obs.channel_index(Channel::DxDyY).unwrap(), bi, si)
                        .copy_from_slice(&d);
                }
                if self.derivatives.lambda {
                    let hi = self.render_frame(values, lambda + self.lambda_step, x0, y0)?;
                    let lo = self.render_frame(values, lambda - self.lambda_step, x0, y0)?;
                    let d: Vec<f64> = hi
                        .iter()
                        .zip(&lo)
                        .map(|(h, l)| (h - l) / (2.0 * self.lambda_step))
                        .collect();
                    obs.slice_mut(obs.channel_index(Channel::Dlambda).unwrap(), bi, si)
                        .copy_from_slice(&d);
                }
            }
        }
        Ok(obs)
    }

    fn check_grid(frame: &[f64], n: usize, lambda: f64, x0: f64, y0: f64) -> Result<()> {
        if frame.len() != n || !frame.iter().all(|v| v.is_finite()) {
            return Err(InverseError::Render {
                reason: format!("grid mismatch or non-finite intensity at λ={lambda}, ({x0},{y0})"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carrier::Design;
    use crate::params::{Param, Params};
    use num_complex::Complex64;
    use optics_core::element::{ClearAperture, Material, PhaseMask};
    use optics_core::sources::{GaussianBeam, gaussian};

    const L: f64 = 500e-9;

    fn plane_wave_source() -> impl Fn(f64, f64, f64) -> Result<BandField> {
        |_x0, _y0, lambda| {
            BandField::plane_wave(0.0, 0.0, 1.0, 64, 64, 10e-6, 10e-6, lambda).map_err(|e| {
                InverseError::Render {
                    reason: e.to_string(),
                }
            })
        }
    }

    fn gaussian_source(w0: f64) -> impl Fn(f64, f64, f64) -> Result<BandField> {
        move |x0, y0, lambda| {
            let beam = GaussianBeam {
                w0,
                amplitude: 1.0,
                center: (x0, y0),
            };
            gaussian(&beam, 0.0, lambda, 64, 64, 10e-6, 10e-6).map_err(|e| InverseError::Render {
                reason: e.to_string(),
            })
        }
    }

    fn asm_no_bandlimit(band: &BandField, z: f64, medium: f64) -> Result<BandField> {
        let cfg = AsmConfig {
            medium,
            band_limited: false,
            strict: false,
            ..Default::default()
        };
        propagate(band, z, &cfg).map_err(|e| InverseError::Render {
            reason: e.to_string(),
        })
    }

    /// Uniform relief h in material n as an analytic carrier with the
    /// defocus/mask/defocus relay chain of the coded-aperture workflow.
    fn relay_carrier(d: f64, n: f64) -> Carrier {
        Carrier::analytic(
            Params::from_slots([Param::new("h", 0.0, 20e-6, 2e-6)]),
            move |v, lambda| {
                let phase = 2.0 * std::f64::consts::PI * (n - 1.0) * v[0] / lambda;
                let mask = PhaseMask::sampled(
                    vec![phase; 64 * 64],
                    (64, 64),
                    (10e-6, 10e-6),
                    None,
                    ClearAperture::Field,
                    (0.0, 0.0),
                )
                .map_err(|e| InverseError::Carrier {
                    reason: e.to_string(),
                })?;
                let mut design = Design::new();
                design.gap(-d, 1.0);
                design.push(mask, 0.0, 1.0);
                design.gap(d, 1.0);
                Ok(design)
            },
        )
    }

    fn flat_relief(h: f64, n: f64) -> Carrier {
        let material = Material::new(move |_l, _a| n, |_l, _a| 0.0);
        Carrier::grid_thickness(
            (32, 32),
            (10e-6, 10e-6),
            ClearAperture::Field,
            (0.0, 0.0),
            20e-6,
            h,
            material,
        )
        .unwrap()
    }

    #[test]
    fn defocus_mask_defocus_chain_is_pure_piston() {
        // Smooth (Gaussian) input through -d/mask/+d with uniform relief:
        // the chain is pure piston, so the detector-plane intensity equals
        // the input intensity exactly (collimation preserved). A truncated
        // plane wave is NOT used here: its sharp window edges carry
        // evanescent content that -z/+z filtering alters.
        let (d, n, h) = (5e-3, 1.5, 2e-6);
        let carrier = relay_carrier(d, n);
        let mut r = Renderer::new(&carrier, gaussian_source(80e-6));
        r.asm = AsmConfig {
            band_limited: false,
            strict: false,
            ..Default::default()
        };
        let obs = r.render(&[h]).unwrap();
        let ci = obs.channel_index(Channel::Intensity).unwrap();
        // w0 = 80 um truncates at amplitude ~1e-7 on this window: no
        // edge-diffraction ripple to confuse the piston identity.
        let beam = GaussianBeam {
            w0: 80e-6,
            amplitude: 1.0,
            center: (0.0, 0.0),
        };
        let input = gaussian(&beam, 0.0, L, 64, 64, 10e-6, 10e-6).unwrap();
        let i_in = input.field().intensities();
        let peak = i_in.iter().cloned().fold(0.0f64, f64::max);
        for (a, b) in obs.slice(ci, 0, 0).iter().zip(&i_in) {
            assert!((a - b).abs() < 1e-6 * peak, "piston changed I: {a} vs {b}");
        }
    }

    #[test]
    fn plane_wave_x_y_channels_are_exactly_zero() {
        // The plane-wave source closure ignores (x0, y0), so shifted
        // renders are bit-identical: dI/dx0 = dI/dy0 = 0 everywhere.
        let carrier = relay_carrier(5e-3, 1.5);
        let mut r = Renderer::new(&carrier, plane_wave_source());
        r.derivatives = Derivatives {
            x: true,
            y: true,
            lambda: false,
        };
        let obs = r.render(&[2e-6]).unwrap();
        for ch in [Channel::DxDyX, Channel::DxDyY] {
            let cix = obs.channel_index(ch).unwrap();
            for v in obs.slice(cix, 0, 0) {
                assert_eq!(*v, 0.0, "{ch:?} must be bit-exact zero");
            }
        }
    }

    #[test]
    fn round_trip_propagation_preserves_gaussian() {
        // A(+z) then A(-z) is the identity on propagating content.
        let mut design = Design::new();
        design.gap(2e-3, 1.0);
        design.gap(-2e-3, 1.0);
        let input = (gaussian_source(100e-6))(0.0, 0.0, L).unwrap();
        let out = design.beamline().trace(&input, asm_no_bandlimit).unwrap();
        let sum_in: f64 = input.field().samples().iter().map(|s| s.norm_sqr()).sum();
        let sum_out: f64 = out.field().samples().iter().map(|s| s.norm_sqr()).sum();
        assert!(
            (sum_out / sum_in - 1.0).abs() < 1e-6,
            "power {sum_out} vs {sum_in}"
        );
        let ip: Complex64 = input
            .field()
            .samples()
            .iter()
            .zip(out.field().samples())
            .map(|(a, b)| *a * b.conj())
            .sum();
        // Cauchy-Schwarz normalized overlap: exactly 1 iff out == input.
        assert!((ip.norm() / (sum_in * sum_out).sqrt() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dx_channel_matches_translation_invariance() {
        // Free-space gap of a centered Gaussian: dI/dx0 = -dI/dx on-grid.
        let carrier = Carrier::analytic(Params::new(), |_v, _l| {
            let mut d = Design::new();
            d.gap(3e-3, 1.0);
            Ok(d)
        });
        let mut r = Renderer::new(&carrier, gaussian_source(80e-6));
        r.derivatives.x = true;
        r.scene_step = 1e-6;
        r.asm = AsmConfig {
            band_limited: false,
            strict: false,
            ..Default::default()
        };
        let obs = r.render(&[]).unwrap();
        let ci = obs.channel_index(Channel::Intensity).unwrap();
        let cix = obs.channel_index(Channel::DxDyX).unwrap();
        let i = obs.slice(ci, 0, 0);
        let dx = obs.slice(cix, 0, 0);
        let (nx, ny) = obs.dims;
        let peak = i.iter().cloned().fold(0.0f64, f64::max);
        let scale = peak / 80e-6; // typical |dI/dx| near the core
        let mut worst = 0.0f64;
        let mut checked = 0usize;
        for iy in 1..ny - 1 {
            for ix in 2..nx - 2 {
                let k = iy * nx + ix;
                if i[k] < 0.02 * peak {
                    continue;
                }
                // 4th-order central stencil along x (k +/- 1); the FD
                // channel is 2nd-order in scene_step, so the comparison
                // must not add grid-scale truncation error of its own.
                let gx =
                    (i[k - 2] - 8.0 * i[k - 1] + 8.0 * i[k + 1] - i[k + 2]) / (12.0 * obs.pitch.0);
                let err = (dx[k] + gx).abs() / scale;
                worst = worst.max(err);
                checked += 1;
            }
        }
        assert!(checked > 100, "too few interior pixels: {checked}");
        assert!(worst < 0.02, "translation-invariance mismatch {worst}");
    }

    #[test]
    fn lambda_channel_vanishes_for_piston_only_system() {
        // Uniform relief, zero gaps: I is lambda-independent.
        let carrier = flat_relief(2e-6, 1.5);
        let mut r = Renderer::new(&carrier, plane_wave_source());
        r.bands = vec![450e-9, 500e-9, 550e-9];
        r.derivatives.lambda = true;
        let obs = r.render(&carrier.params().values()).unwrap();
        assert_eq!(obs.channels.len(), 2);
        let ci = obs.channel_index(Channel::Dlambda).unwrap();
        for bi in 0..obs.bands.len() {
            for v in obs.slice(ci, bi, 0) {
                assert!(v.abs() < 1e-3, "dI/dlambda should be ~0, got {v}");
            }
        }
    }

    #[test]
    fn lambda_channel_detects_chromatic_focus() {
        // Ideal diffractive lens f = f0 * lambda0 / lambda: focal length
        // scales with band, so the detector-plane spot (at f0) defocuses
        // away from lambda0 and dI/dlambda is strongly nonzero.
        let lambda0 = 500e-9;
        let f0 = 50e-3;
        let carrier = Carrier::analytic(
            Params::from_slots([Param::new("f0", 0.01, 1.0, f0)]),
            move |v, lambda| {
                let f = v[0] * lambda0 / lambda;
                let lens = optics_core::element::ThinLens::new(
                    move |_| f,
                    None,
                    ClearAperture::Circle(300e-6),
                    (0.0, 0.0),
                )
                .map_err(|e| InverseError::Carrier {
                    reason: e.to_string(),
                })?;
                let mut d = Design::new();
                d.push(lens, v[0], 1.0);
                Ok(d)
            },
        );
        let mut r = Renderer::new(&carrier, |_, _, lambda| {
            BandField::plane_wave(0.0, 0.0, 1.0, 128, 128, 10e-6, 10e-6, lambda).map_err(|e| {
                InverseError::Render {
                    reason: e.to_string(),
                }
            })
        });
        r.bands = vec![450e-9, 500e-9, 550e-9];
        r.derivatives.lambda = true;
        r.lambda_step = 5e-9;
        r.asm = AsmConfig {
            band_limited: false,
            strict: false,
            ..Default::default()
        };
        let obs = r.render(&[f0]).unwrap();
        let ci = obs.channel_index(Channel::Intensity).unwrap();
        let cdl = obs.channel_index(Channel::Dlambda).unwrap();
        let peak = obs.slice(ci, 1, 0).iter().cloned().fold(0.0f64, f64::max);
        let max_d = obs
            .slice(cdl, 1, 0)
            .iter()
            .cloned()
            .fold(0.0f64, |a, v| a.max(v.abs()));
        assert!(
            max_d > 1.0e5 * peak,
            "chromatic response missing: max|dI/dlambda| {max_d} vs peak {peak}"
        );
    }

    #[test]
    fn band_sum_matches_manual_incoherent_addition() {
        let carrier = flat_relief(2e-6, 1.5);
        let mut r = Renderer::new(&carrier, plane_wave_source());
        r.bands = vec![450e-9, 550e-9];
        let obs = r.render(&carrier.params().values()).unwrap();
        let sum = obs.band_sum(Channel::Intensity).unwrap();
        let ci = obs.channel_index(Channel::Intensity).unwrap();
        for (k, &s) in sum.iter().enumerate() {
            let want = obs.slice(ci, 0, 0)[k] + obs.slice(ci, 1, 0)[k];
            assert!((s - want).abs() < 1e-12);
        }
    }

    #[test]
    fn observation_layout_is_contiguous_frames() {
        let carrier = flat_relief(1e-6, 1.5);
        let mut r = Renderer::new(&carrier, plane_wave_source());
        r.bands = vec![450e-9, 550e-9];
        r.scenes = vec![(0.0, 0.0), (20e-6, 0.0)];
        r.derivatives = Derivatives {
            x: true,
            y: false,
            lambda: true,
        };
        let obs = r.render(&carrier.params().values()).unwrap();
        let c = obs.channels.len();
        let b = obs.bands.len();
        let s = obs.scenes.len();
        assert_eq!(obs.data.len(), c * b * s * obs.frame_len());
        assert_eq!(obs.channels[1], Channel::DxDyX);
        assert_eq!(obs.channels[2], Channel::Dlambda);
        assert_eq!(obs.channel_index(Channel::DxDyY), None);
        assert_eq!(obs.slice(0, 1, 1).len(), obs.frame_len());
    }

    #[test]
    fn renderer_rejects_empty_axes_and_bad_steps() {
        let carrier = flat_relief(1e-6, 1.5);
        let mut r = Renderer::new(&carrier, plane_wave_source());
        r.bands = vec![];
        assert!(r.render(&carrier.params().values()).is_err());
        let mut r = Renderer::new(&carrier, plane_wave_source());
        r.derivatives.x = true;
        r.scene_step = 0.0;
        assert!(r.render(&carrier.params().values()).is_err());
    }

    #[test]
    fn psf_rebuild_matches_analysis_consumption() {
        let carrier = flat_relief(0.0, 1.5);
        let r = Renderer::new(&carrier, plane_wave_source());
        let obs = r.render(&carrier.params().values()).unwrap();
        let psf = obs.psf(Channel::Intensity, 0, 0).unwrap();
        assert_eq!(psf.samples().len(), obs.frame_len());
    }
}
