//! Adjoint (reverse-mode) gradients through the render chain — task 9.9.
//!
//! ## Why no autodiff
//!
//! The forward model is a chain of exactly two primitive kinds, both **linear
//! operators on the complex field**: ASM propagation [`optics_core::propagation::asm`]
//! and thin elements (pointwise complex transmission,
//! [`optics_core::element::Element`]). Because every primitive is linear, its
//! adjoint (conjugate transpose) is another primitive of the same kind, so the
//! design gradient is obtained by pushing the merit's observation gradient
//! **backward through the forward primitives in reverse order**. No symbolic
//! or graph-based autodiff is needed. The full derivation, conventions, and
//! the GS error-signal relation are recorded in the repository ADR
//! `docs/adr-9.9-adjoint-gradients.md`.
//!
//! ## The three adjoint facts (ADR-9.9)
//!
//! - **ASM propagation**: $\mathcal{A}(z)^\dagger = \mathcal{A}(-z)$ —
//!   propagation by $-z$ with the *same* [`AsmConfig`]. The transfer phase
//!   conjugates, the evanescent and Matsushima masks are real and even in $z$,
//!   and the zero-pad / center-crop are each other's adjoints under the core's
//!   pinned FFT convention. [`asm_adjoint`] is that call;
//!   `asm_adjoint_inner_product_identity` pins the operator identity.
//! - **Thin element**: $T=\operatorname{diag}(t)$, so $T^\dagger$ is the
//!   pointwise **conjugate** multiply $\bar t\odot(\cdot)$. An aperture is the
//!   case $t=\Pi e^{i\varphi}$ with $\Pi\in\{0,1\}$ (blocked pixels pass no
//!   adjoint energy). The transmission $t$ is recovered without the element
//!   exposing it: `apply` is pointwise, so $t=\texttt{apply}(\mathbf 1)$ on a
//!   unit field of the plane's geometry ([`transmission`]).
//! - **Intensity seed**: $I_k=|U_k|^2$, so for real $M$ the detector-plane
//!   adjoint source is $w_k = (\partial M/\partial I_k)\,U_k$ and
//!   $dM = 2\,\Re\langle w, dU\rangle$.
//!
//! ## Phase-grid gradient
//!
//! With $\mu$ the adjoint field pushed back to the plane **after** element $j$
//! and $V=t\odot U$ the tape field after it, perturbing one lattice phase
//! injects $dV_p = i\,V_p\,d\varphi_p$, so
//!
//! $$\frac{\partial M}{\partial\varphi_p} = -2\,\Im\!\big(\bar\mu_p\,V_p\big).$$
//!
//! [`push_back`] returns that $dM/d\varphi$ per element plane for one
//! (band, scene) frame; [`grid_gradient`] drives it over the whole render
//! chain (1 forward render + one adjoint run per gradient-carrying frame) and
//! sums, giving the grid gradient the task names. The parameter→phase chain
//! rules (identity for `GridPhase`, the per-band $2\pi(n-1)/\lambda$ slope for
//! `GridThickness`, the closure's own Jacobian for analytic carriers) are
//! documented in the ADR and applied by the caller; see [`grid_gradient`] for
//! the band-independence requirement of the summed result.
//!
//! ## Scope and fallbacks
//!
//! - Merits must supply [`Merit::obs_gradient`]; those returning `None` keep
//!   the documented FD-on-observation fallback (task 9.7) — [`grid_gradient`]
//!   rejects them with [`InverseError::Adjoint`].
//! - Only the intensity channel (channel 0) is adjoint-supported. A seed with
//!   nonzero entries outside it (the ∂x₀/∂y₀/∂λ derivative channels) is
//!   rejected; their adjoint is a future extension (ADR consequences).
//! - The transmission probe assumes a thin element whose action is a pure
//!   pointwise complex multiply (all current core elements).

use num_complex::Complex64;

use optics_core::element::Element;
use optics_core::field::{BandField, Field};
use optics_core::propagation::asm::{AsmConfig, propagate};

use crate::Result;
use crate::carrier::Design;
use crate::error::InverseError;
use crate::merit::Merit;
use crate::observe::{Channel, Renderer};

/// Map a core construction/propagation failure onto the crate error type
/// (renderer precedent, `observe.rs`).
fn render_err(e: impl std::fmt::Display) -> InverseError {
    InverseError::Adjoint {
        reason: e.to_string(),
    }
}

/// The exact ASM adjoint: $\mathcal{A}(z)^\dagger = \mathcal{A}(-z)$ with the
/// same config (ADR-9.9). Back-propagates `band` by `z` — i.e. propagates by
/// `-z` — reproducing the zero-padding, band-limit, and medium of the forward
/// step (all even in $z$).
///
/// # Errors
/// [`InverseError::Adjoint`] if the underlying propagation fails (e.g. a
/// non-finite distance or an undersampled strict-mode step).
pub fn asm_adjoint(band: &BandField, z: f64, cfg: &AsmConfig) -> Result<BandField> {
    propagate(band, -z, cfg).map_err(render_err)
}

/// Pointwise complex transmission $t(x,y)$ of a thin element on `like`'s grid,
/// recovered by probing `apply` on a unit field: $t = \texttt{apply}(\mathbf 1)$
/// (apply is a pointwise multiply, so it returns $t\odot\mathbf 1 = t$). The
/// adjoint multiplies by $\bar t$.
///
/// # Errors
/// [`InverseError::Adjoint`] if `apply` fails (a stopped element whose
/// footprint exceeds the grid — the same failure the forward pass would hit).
pub fn transmission(element: &dyn Element, like: &BandField) -> Result<Vec<Complex64>> {
    let (nx, ny) = like.field().dims();
    let (dx, dy) = like.field().pitch();
    let probe = BandField::uniform(1.0, 0.0, nx, ny, dx, dy, like.lambda()).map_err(render_err)?;
    let out = element.apply(&probe).map_err(render_err)?;
    Ok(out.field().samples().to_vec())
}

/// $dM/d\varphi$ over one element plane's lattice samples (row-major, length
/// = frame). This is the adjoint grid gradient for a phase-only thin element;
/// the parameter→phase chain rule is applied by the caller (see the module
/// docs and ADR-9.9).
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneGradient {
    /// Index into the [`Design`]'s element list this gradient belongs to.
    pub element: usize,
    /// $dM/d\varphi_p$ per lattice sample.
    pub dphi: Vec<f64>,
}

/// One adjoint run for a single (band, scene) frame: trace `design` forward on
/// `input` (recording the tape), seed the detector with the observation
/// gradient `dmdi` $=\partial M/\partial I$, and push it back through the
/// forward primitives in reverse order.
///
/// The forward trace uses `asm` with each step's medium injected
/// (`AsmConfig { medium, ..asm }`), exactly matching
/// [`Renderer::render`](crate::observe::Renderer::render), so `dmdi` must be
/// the merit gradient of the observation this design/input produced.
///
/// Returns one [`PlaneGradient`] per distinct element that acted on the field,
/// sorted by element index. An element touched by multiple steps accumulates
/// all contributions.
///
/// # Errors
/// [`InverseError::WrongDimension`] if `dmdi.len()` is not the frame length;
/// [`InverseError::Adjoint`] for a bad element index or a failing
/// propagation/apply.
pub fn push_back(
    design: &Design,
    input: &BandField,
    asm: &AsmConfig,
    dmdi: &[f64],
) -> Result<Vec<PlaneGradient>> {
    let (nx, ny) = input.field().dims();
    let (dx, dy) = input.field().pitch();
    let n = nx * ny;
    if dmdi.len() != n {
        return Err(InverseError::WrongDimension {
            expected: n,
            actual: dmdi.len(),
        });
    }
    let lambda = input.lambda();

    // --- Forward tape: for each step, the element transmission and the field
    // immediately after it, plus the propagation that follows. ---
    struct Step {
        element: Option<usize>,
        t: Vec<Complex64>,
        v_after: Vec<Complex64>,
        z: f64,
        medium: f64,
    }
    let mut tape: Vec<Step> = Vec::with_capacity(design.len());
    let mut cur = input.clone();
    for st in design.steps() {
        let mut t = Vec::new();
        let mut v_after = Vec::new();
        if let Some(ei) = st.element {
            let element = design
                .elements()
                .get(ei)
                .ok_or_else(|| InverseError::Adjoint {
                    reason: format!("design step references missing element index {ei}"),
                })?;
            t = transmission(element.as_ref(), &cur)?;
            let after = element.apply(&cur).map_err(render_err)?;
            v_after = after.field().samples().to_vec();
            cur = after;
        }
        if st.z_after != 0.0 {
            let cfg = AsmConfig {
                medium: st.medium,
                ..*asm
            };
            cur = propagate(&cur, st.z_after, &cfg).map_err(render_err)?;
        }
        tape.push(Step {
            element: st.element,
            t,
            v_after,
            z: st.z_after,
            medium: st.medium,
        });
    }

    // --- Adjoint source at the detector: w_k = (dM/dI_k) · U_k. ---
    let det = cur.field().samples();
    let mut co_samples: Vec<Complex64> = (0..n).map(|k| dmdi[k] * det[k]).collect();

    // --- Backward sweep in reverse order: propagation adjoint (−z) then
    // element adjoint (conj(t)), accumulating dM/dφ at each element plane. ---
    let mut acc: Vec<Option<Vec<f64>>> = vec![None; design.elements().len().max(1)];
    for step in tape.iter().rev() {
        if step.z != 0.0 {
            let field = Field::new(co_samples, nx, ny, dx, dy).map_err(render_err)?;
            let band = BandField::new(lambda, field).map_err(render_err)?;
            co_samples = asm_adjoint(
                &band,
                step.z,
                &AsmConfig {
                    medium: step.medium,
                    ..*asm
                },
            )?
            .field()
            .samples()
            .to_vec();
        }
        if let Some(ei) = step.element {
            let mu = &co_samples;
            let slot = acc.get_mut(ei).ok_or_else(|| InverseError::Adjoint {
                reason: format!("element index {ei} out of range"),
            })?;
            let g = slot.get_or_insert_with(|| vec![0.0; n]);
            for p in 0..n {
                // dM/dφ_p += -2 Im( conj(μ_p) · V_p )
                g[p] += -2.0 * (mu[p].conj() * step.v_after[p]).im;
            }
            // Apply T†: co = conj(t) ⊙ μ.
            for p in 0..n {
                co_samples[p] *= step.t[p].conj();
            }
        }
    }

    let mut out: Vec<PlaneGradient> = acc
        .into_iter()
        .enumerate()
        .filter_map(|(ei, g)| g.map(|dphi| PlaneGradient { element: ei, dphi }))
        .collect();
    out.sort_by_key(|pg| pg.element);
    Ok(out)
}

/// The adjoint grid gradient over a full render chain: `dM/dφ` per element
/// plane, summed over every (band, scene) frame the merit reads.
///
/// Cost: one forward [`Renderer::render`] (for the observation and the merit's
/// `obs_gradient` seed) plus one [`push_back`] adjoint run per frame —
/// independent of the design dimensionality.
///
/// The summed result is exactly `dM/dθ` when the parameter→phase map is
/// band-independent and identity on the lattice samples — the `GridPhase`
/// carrier and analytic carriers whose closure passes its parameters straight
/// into a [`optics_core::element::PhaseMask::sampled`] DOE. For a
/// `GridThickness` carrier the phase law is band-dependent
/// ($\varphi_p=2\pi[n(\lambda)-1]h_p/\lambda$), so `dM/dh` requires weighting
/// each band's contribution by its slope; call [`push_back`] per band and sum
/// with the $2\pi(n(\lambda_b)-1)/\lambda_b$ factor (ADR-9.9).
///
/// # Errors
/// [`InverseError::Adjoint`] if the merit returns `None` from `obs_gradient`
/// (use FD on the observation instead) or if a seed is nonzero outside the
/// intensity channel; other errors as in [`Renderer::render`] / [`push_back`].
pub fn grid_gradient(
    renderer: &Renderer,
    values: &[f64],
    merit: &dyn Merit,
) -> Result<Vec<PlaneGradient>> {
    let obs = renderer.render(values)?;
    let seed = merit
        .obs_gradient(&obs)
        .ok_or_else(|| InverseError::Adjoint {
            reason: format!(
                "merit '{}' has no obs_gradient; use FD on the observation (task 9.7)",
                merit.name()
            ),
        })?;
    let ci = obs
        .channel_index(Channel::Intensity)
        .ok_or_else(|| InverseError::Adjoint {
            reason: "observation has no intensity channel".into(),
        })?;
    let frames = obs.bands.len() * obs.scenes.len();
    let n = obs.frame_len();

    // Seed must be confined to the intensity channel: derivative channels have
    // no adjoint path yet (ADR-9.9 consequences).
    let intensity_len = frames * n;
    if seed[ci * intensity_len..][..intensity_len]
        .iter()
        .all(|&v| v == 0.0)
    {
        // Whole merit is constant on intensity; gradient is zero.
        return Ok(Vec::new());
    }
    for (c, block) in seed.chunks(intensity_len).enumerate() {
        if c != ci && block.iter().any(|&v| v != 0.0) {
            return Err(InverseError::Adjoint {
                reason: format!("adjoint seed is nonzero on channel {c} (non-intensity)"),
            });
        }
    }

    let scenes = obs.scenes.len();
    let mut acc: Vec<Option<Vec<f64>>> = Vec::new();
    for (bi, &lambda) in renderer.bands.iter().enumerate() {
        let design = renderer.carrier.build(values, lambda)?;
        if acc.len() < design.elements().len() {
            acc.resize(design.elements().len(), None);
        }
        for (si, &(x0, y0)) in renderer.scenes.iter().enumerate() {
            let input = (renderer.source)(x0, y0, lambda)?;
            let start = (ci * frames + bi * scenes + si) * n;
            let dmdi = &seed[start..start + n];
            for pg in push_back(&design, &input, &renderer.asm, dmdi)? {
                let slot = acc
                    .get_mut(pg.element)
                    .expect("acc sized to design element count");
                match slot {
                    Some(g) => {
                        for (a, b) in g.iter_mut().zip(&pg.dphi) {
                            *a += b;
                        }
                    }
                    None => *slot = Some(pg.dphi),
                }
            }
        }
    }

    Ok(acc
        .into_iter()
        .enumerate()
        .filter_map(|(ei, g)| g.map(|dphi| PlaneGradient { element: ei, dphi }))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carrier::Carrier;
    use crate::fd;
    use crate::merit::{MseToTarget, Selector, StrehlMerit};
    use crate::params::Params;
    use optics_core::element::{ClearAperture, PhaseMask};
    use std::f64::consts::{PI, TAU};

    const L: f64 = 500e-9;

    fn asm_plain() -> AsmConfig {
        AsmConfig {
            band_limited: false,
            strict: false,
            ..Default::default()
        }
    }

    /// Deterministic pseudo-random complex field (chirp + gaussian envelope)
    /// so inner-product identities are non-degenerate.
    fn chirp_field(nx: usize, ny: usize, d: f64) -> BandField {
        let mut s = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * d;
                let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * d;
                let r2 = x * x + y * y;
                let env = (-(r2) / (2.0 * (60e-6f64).powi(2))).exp();
                let ph = 3.7 * x / 1e-3 + 2.1 * y / 1e-3 + 9.0e6 * r2;
                s.push(Complex64::new(env * ph.cos(), env * ph.sin()));
            }
        }
        BandField::new(L, Field::new(s, nx, ny, d, d).unwrap()).unwrap()
    }

    fn inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.conj() * *y)
            .sum::<Complex64>()
    }

    fn samples(band: &BandField) -> &[Complex64] {
        band.field().samples()
    }

    fn gaussian_intensity(n: usize, d: f64, w: f64) -> Vec<f64> {
        (0..n)
            .flat_map(|iy| {
                let y = (iy as f64 - (n as f64 - 1.0) / 2.0) * d;
                (0..n).map(move |ix| {
                    let x = (ix as f64 - (n as f64 - 1.0) / 2.0) * d;
                    (-(x * x + y * y) / (w * w)).exp()
                })
            })
            .collect()
    }

    // -------------------------------------------------------------------
    // Adjoint primitive identities (convention pins — agents.md §4)
    // -------------------------------------------------------------------

    #[test]
    fn asm_adjoint_inner_product_identity() {
        // <w, A(z) d> == <A(-z) w, d> — the full ASM operator adjoint,
        // including zero-pad / center-crop / band-limit under the pinned FFT
        // convention (ADR-9.9). Checked band-limited off and on.
        let (n, d, z) = (48, 10e-6, 1.5e-3);
        let dir = chirp_field(n, n, d);
        let w = chirp_field(n, n, d);
        for band_limited in [false, true] {
            let cfg = AsmConfig {
                band_limited,
                strict: false,
                ..Default::default()
            };
            let ad = propagate(&dir, z, &cfg).unwrap();
            let lhs = inner(samples(&w), samples(&ad));
            let aw = asm_adjoint(&w, z, &cfg).unwrap();
            let rhs = inner(samples(&aw), samples(&dir));
            let rel = (lhs - rhs).norm() / lhs.norm();
            assert!(rel < 1e-12, "band_limited={band_limited}: rel {rel}");
        }
    }

    #[test]
    fn element_transmission_probe_and_adjoint_identity() {
        // t = apply(1) recovers e^{iφ} inside the aperture and 0 outside;
        // <w, t⊙d> == <conj(t)⊙w, d> pins the element-adjoint convention.
        let (n, d) = (32, 10e-6);
        let phase: Vec<f64> = (0..n * n).map(|k| (k as f64 * 0.37).sin()).collect();
        let mask = PhaseMask::sampled(
            phase.clone(),
            (n, n),
            (d, d),
            None,
            ClearAperture::Circle(120e-6),
            (0.0, 0.0),
        )
        .unwrap();
        let like = chirp_field(n, n, d);
        let t = transmission(&mask, &like).unwrap();
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = like.field().coord(ix, iy);
                let k = iy * n + ix;
                let want = if x * x + y * y <= (120e-6f64).powi(2) {
                    Complex64::from_polar(1.0, phase[k])
                } else {
                    Complex64::ZERO
                };
                assert!((t[k] - want).norm() < 1e-12, "k {k}");
            }
        }
        let dir = chirp_field(n, n, d);
        let w = chirp_field(n, n, d);
        let td: Vec<Complex64> = samples(&dir).iter().zip(&t).map(|(u, tc)| tc * u).collect();
        let cw: Vec<Complex64> = samples(&w)
            .iter()
            .zip(&t)
            .map(|(u, tc)| tc.conj() * u)
            .collect();
        let lhs = inner(samples(&w), &td);
        let rhs = inner(&cw, samples(&dir));
        assert!((lhs - rhs).norm() / lhs.norm() < 1e-12);
    }

    // -------------------------------------------------------------------
    // Acceptance 1: adjoint grid gradient == FD on a coarse grid
    // -------------------------------------------------------------------

    /// Analytic carrier whose closure passes its parameters straight into a
    /// sampled DOE between two relay gaps: params == phase lattice, so
    /// dM/dparams == dM/dφ (the identity map the ADR names).
    fn doe_relay_carrier(params: Params, n: usize, d: f64, z: f64) -> Carrier {
        Carrier::analytic(params, move |v, _lambda| {
            let mask = PhaseMask::sampled(
                v.to_vec(),
                (n, n),
                (d, d),
                None,
                ClearAperture::Field,
                (0.0, 0.0),
            )
            .map_err(|e| InverseError::Carrier {
                reason: e.to_string(),
            })?;
            let mut design = Design::new();
            design.gap(z, 1.0); // input plane -> DOE
            design.push(mask, z, 1.0); // DOE -> detector
            Ok(design)
        })
    }

    #[test]
    fn grid_gradient_matches_fd_on_coarse_grid() {
        let (n, d, z) = (6, 30e-6, 1.2e-3);
        let values: Vec<f64> = (0..n * n).map(|k| (k as f64 * 1.3).sin() * 1.1).collect();
        let mut params = Params::uniform(n * n, -PI, PI, 0.0, "phi");
        params.set_values(&values).unwrap();
        let carrier = doe_relay_carrier(params, n, d, z);

        let merit = MseToTarget {
            target: gaussian_intensity(n, d, 60e-6),
            selector: Selector::intensity(None, None),
        };
        let mut renderer = Renderer::new(&carrier, move |_, _, lambda| {
            BandField::plane_wave(0.0, 0.0, 1.0, n, n, d, d, lambda).map_err(|e| {
                InverseError::Render {
                    reason: e.to_string(),
                }
            })
        });
        renderer.asm = asm_plain();

        let objective = |v: &[f64]| -> Result<f64> { Ok(merit.evaluate(&renderer.render(v)?)?) };
        let (g_fd, _) = fd::gradient(carrier.params(), &objective, &values).unwrap();

        let planes = grid_gradient(&renderer, &values, &merit).unwrap();
        assert_eq!(planes.len(), 1, "single DOE element expected");
        assert_eq!(planes[0].element, 0);
        let g_adj = &planes[0].dphi;
        assert_eq!(g_adj.len(), g_fd.len());

        // Global relative error over the whole grid (acceptance: < 1e-4).
        let num: f64 = g_fd
            .iter()
            .zip(g_adj)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let den: f64 = g_fd.iter().map(|a| a * a).sum::<f64>().sqrt();
        let rel = num / den;
        assert!(rel < 1e-4, "adjoint vs FD rel error {rel}");
        // No sign flips on well-conditioned components.
        for (a, b) in g_fd.iter().zip(g_adj) {
            if a.abs() > 1e-3 * den {
                assert!(a.signum() == b.signum(), "sign flip: fd {a} adj {b}");
            }
        }
    }

    #[test]
    fn grid_gradient_rejects_merit_without_obs_gradient() {
        let (n, d, z) = (4, 30e-6, 1e-3);
        let params = Params::uniform(n * n, -PI, PI, 0.0, "phi");
        let carrier = doe_relay_carrier(params, n, d, z);
        let values = vec![0.1; n * n];
        let merit = StrehlMerit {
            reference: vec![1.0; n * n],
            selector: Selector::intensity(None, None),
        };
        let renderer = Renderer::new(&carrier, move |_, _, lambda| {
            BandField::plane_wave(0.0, 0.0, 1.0, n, n, d, d, lambda).map_err(|e| {
                InverseError::Render {
                    reason: e.to_string(),
                }
            })
        });
        let err = grid_gradient(&renderer, &values, &merit).unwrap_err();
        assert!(matches!(err, InverseError::Adjoint { .. }), "{err}");
    }

    #[test]
    fn push_back_rejects_bad_seed_and_empty_design() {
        let (n, d) = (4, 30e-6);
        let input = chirp_field(n, n, d);
        let design = Design::new();
        // Wrong seed length.
        let err = push_back(&design, &input, &asm_plain(), &[0.0; 3]).unwrap_err();
        assert!(matches!(err, InverseError::WrongDimension { .. }));
        // No-element design: no planes, no error (merit independent of phase).
        let planes = push_back(&design, &input, &asm_plain(), &vec![0.0; n * n]).unwrap();
        assert!(planes.is_empty());
    }

    // -------------------------------------------------------------------
    // Acceptance 2: GS error signal == negative adjoint MSE gradient
    // -------------------------------------------------------------------

    #[test]
    fn gs_phase_error_is_adjoint_mse_negative_gradient() {
        // Textbook §5.5.1 (Eqs. 5.14–5.18) bidirectional loop, run a few
        // iterations to shrink the error, then confirm at the final iterate
        // the ADR-9.9 relation ∇φ M = -(2/N) A² ⊙ Δφ: the DOE-plane
        // phase error signal Δφ is the negative adjoint gradient of the
        // Eq. 5.18 amplitude MSE, preconditioned by the illumination
        // intensity A². Built from the ADR's own primitives (asm_adjoint,
        // the ADR seed, the ADR chain rule).
        let (n, d) = (32, 20e-6);
        let half_z = 1.5e-3;
        let npx = n * n;
        let asm = asm_plain();

        let input = {
            let mut s = Vec::with_capacity(npx);
            for iy in 0..n {
                for ix in 0..n {
                    let x = (ix as f64 - (n as f64 - 1.0) / 2.0) * d;
                    let y = (iy as f64 - (n as f64 - 1.0) / 2.0) * d;
                    s.push(Complex64::new(
                        (-(x * x + y * y) / (2.0 * (90e-6f64).powi(2))).exp(),
                        0.0,
                    ));
                }
            }
            BandField::new(L, Field::new(s, n, n, d, d).unwrap()).unwrap()
        };
        let arriving = propagate(&input, half_z, &asm).unwrap();
        let illum: Vec<f64> = samples(&arriving).iter().map(|s| s.norm()).collect();
        let target: Vec<f64> = {
            let g = gaussian_intensity(n, d, 120e-6);
            let peak = g.iter().cloned().fold(0.0f64, f64::max).sqrt();
            g.iter().map(|v| v.sqrt() / peak * 0.8).collect()
        };

        // Run 40 textbook GS iterations from a flat start; keep the LAST
        // iteration's Δφ and the phase it drove (φ before the update).
        let mut phi = vec![0.0f64; npx];
        let mut dphi = vec![0.0f64; npx];
        for _ in 0..40 {
            let mask: Vec<Complex64> = (0..npx)
                .map(|k| Complex64::from_polar(illum[k], phi[k]))
                .collect();
            let mask_band = BandField::new(L, Field::new(mask, n, n, d, d).unwrap()).unwrap();
            let u_t = propagate(&mask_band, half_z, &asm).unwrap();
            let proj: Vec<Complex64> = samples(&u_t)
                .iter()
                .enumerate()
                .map(|(k, s)| Complex64::from_polar(target[k], s.arg()))
                .collect();
            let proj_band = BandField::new(L, Field::new(proj, n, n, d, d).unwrap()).unwrap();
            let v = asm_adjoint(&proj_band, half_z, &asm).unwrap(); // A(-z/2)
            dphi = samples(&v)
                .iter()
                .zip(&phi)
                .map(|(s, p)| (s.arg() - p + PI).rem_euclid(TAU) - PI)
                .collect();
            for (p, dd) in phi.iter_mut().zip(&dphi) {
                *p += dd;
            }
        }
        // φ before the last update (the state whose Δφ we kept):
        let phi_prev: Vec<f64> = phi
            .iter()
            .zip(&dphi)
            .map(|(p, dd)| (p - dd + PI).rem_euclid(TAU) - PI)
            .collect();

        // Adjoint of the amplitude MSE at φ_prev, per ADR-9.9:
        // seed w = (1/N)(|U_t|-T)e^{iθ}; μ = A(z/2)† w; dM/dφ_p = -2 Im(μ̄_p m_p).
        let mask: Vec<Complex64> = (0..npx)
            .map(|k| Complex64::from_polar(illum[k], phi_prev[k]))
            .collect();
        let mask_band = BandField::new(L, Field::new(mask.clone(), n, n, d, d).unwrap()).unwrap();
        let u_t = propagate(&mask_band, half_z, &asm).unwrap();
        let w: Vec<Complex64> = samples(&u_t)
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let a = s.norm();
                if a > 1e-30 {
                    Complex64::new((a - target[k]) / npx as f64, 0.0) * (*s / a)
                } else {
                    Complex64::ZERO
                }
            })
            .collect();
        let w_band = BandField::new(L, Field::new(w, n, n, d, d).unwrap()).unwrap();
        let mu = asm_adjoint(&w_band, half_z, &asm).unwrap();
        let g: Vec<f64> = (0..npx)
            .map(|k| -2.0 * (samples(&mu)[k].conj() * mask[k]).im)
            .collect();

        // Sign/shape: -g must align with A²⊙Δφ (the GS update direction).
        let pred: Vec<f64> = (0..npx).map(|k| illum[k] * illum[k] * dphi[k]).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (mg, mp) = (mean(&g), mean(&pred));
        let mut cov = 0.0;
        let mut vg = 0.0;
        let mut vp = 0.0;
        for (a, b) in g.iter().zip(&pred) {
            cov += -(a - mg) * (b - mp);
            vg += (a - mg) * (a - mg);
            vp += (b - mp) * (b - mp);
        }
        let corr = cov / (vg.sqrt() * vp.sqrt());
        assert!(
            corr > 0.9,
            "GS error signal not aligned with -adjoint MSE gradient: corr {corr}"
        );

        // And the ADR's first-order identity ∇φ M = -(2/N) A² ⊙ Δφ holds in
        // magnitude: the least-squares fit of g against -A²⊙Δφ lands within
        // 25% of 2/N (A† = A(−z) exactly — no padding factor; the slack is
        // the O(Δφ²) linearization error and the dark-pixel wraparound).
        let scale = 2.0 / npx as f64;
        let ls = -g.iter().zip(&pred).map(|(a, b)| a * b).sum::<f64>()
            / pred.iter().map(|b| b * b).sum::<f64>();
        assert!(
            (ls / scale - 1.0).abs() < 0.25,
            "ADR first-order adjoint↔GS magnitude identity off: ls {ls} vs 2/N {scale} (corr {corr})"
        );
    }
}
