# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), and commits
follow [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Decisions (roadmap revision, 2026-10-12, user)

- Roadmap re-sequenced: **Phase 3** thin element library + Gaussian
  `sources` initializers; **Phase 4** Rayleigh–Sommerfeld (restored);
  **Phase 5** sources & output incl. visualization/export (restored);
  **Phase 6** element library expansion — thick lens (surfaces + internal
  propagation via `Extent::Thick`/`Material`) and further elements.
  **Design TBD (unscheduled)**: Zemax-style interactive front-end
  (separate project; this crate stays the headless compute core), inverse
  solving (weighted least-squares DOE, Gerchberg–Saxton retrieval), PSF
  visual-analysis suite.
- Phase-3 design choices locking future doors: `apply` stays strictly
  local (thick elements compose surface→propagate→surface); lens focal
  length is a dispersive closure $f(\lambda)$ mirroring `Material`;
  `PhaseMask` takes an analytic closure now, sampled-profile variant with
  Phase 5; engine reports stay `Copy` + public fields for later
  serialization; `sources` is a new top-level module returning
  `BandField`s (multiband-ready by construction).

### Decisions (Phase-3 task-b review, 2026-10-12, user)

- **Apertures unified under the ellipse**: no separate circular element —
  `EllipticalAperture` carries the circle as its axis-symmetric special case
  (`EllipticalAperture::circle(r, center)`; `radius() -> Option<f64>` is
  `Some` exactly when $a == b$; `ElementShape::Circle` reported in that
  case). `CircularAperture` removed (breaking; call sites migrated).
- **`OpticsError` generalized to footprint half-extents**: `InvalidAperture`
  and `OutsideField` now carry `(half_x, half_y)` instead of a
  circle-specific `radius` — one error taxonomy for all aperture shapes
  (breaking; variant usage was confined to element.rs unit tests).
  `Element::extent` for non-circular apertures returns the **conservative**
  larger half-extent $\max(a, b)$ for regime gating; per-axis geometry
  stays on the concrete accessors.
- **Thin lens carries its clear aperture** (task-c revision, user
  2026-10-12; **supersedes** the same-day "pure phase, no clear aperture"
  choice): the primary `ThinLens` form is the quadratic phase inside a
  circular clear aperture (hard-edge stop, boundary inclusive) so
  `Element::extent` describes the lens itself and regime gating uses the
  lens's own half-extent. The field-wide behavior (phase over the whole
  sampled plane) is kept as the `clear_radius = None` special case;
  taxonomy retains `ElementShape::Unbounded` / `Extent::Unbounded` for it.

### Added

- `sources` module (`src/sources.rs`, Phase 3 task a, decision D3): closed-form
  field initializers returning `BandField`s (multiband by construction).
  First initializer: `GaussianBeam` parameter record (`w0`, `amplitude`,
  `center`; SI meters) + `gaussian(&beam, z, lambda, nx, ny, dx, dy) ->
  Result<BandField, SourcesError>` sampling the paraxial fundamental-mode
  solution $u = A\frac{w_0}{w(z)} e^{-\rho^2/w(z)^2}
  e^{\,i(k\rho^2/2R(z))} e^{\,i(kz-\zeta(z))}$ with
  $z_R = \pi w_0^2/\lambda$, $w(z) = w_0\sqrt{1+(z/z_R)^2}$,
  $1/R(z) = z/(z^2+z_R^2)$ (finite at $z=0$), $\zeta = \arctan(z/z_R)$ —
  written in the crate's $e^{+ikz}$ convention ($q = z - i z_R$, conjugate of
  the usual $e^{-ikz}$ textbook form); the project textbook does not restate
  the $q(z)$ solution, so the convention is pinned by cross-checks against
  the crate's own engines.
- **Position is a system property, not a source property** (task-a review,
  user 2026-10-12): the beam record holds no longitudinal position — the
  signed waist-to-plane distance `z` is a sampling argument, as elements hold
  no position either; `beam_radius(z, lambda)` and `gouy_phase(z, lambda)`
  are explicit functions of the measurement plane. `SourcesError`
  (thiserror): `InvalidBeam` diagnostics + `InvalidPlane { z }` +
  transparent `FieldError` carry; convenience accessor `rayleigh_range`.
  Unit tests (6): waist plane equals the real Gaussian with analytic power
  $A^2\pi w_0^2/2$; on-axis amplitude/Gouy of the $q$-form; Fresnel-TF and
  ASM propagation of a waist beam reproduce the analytic initializer at $z$
  ($<10^{-3}$ Frobenius — sign-convention pin); $u(-z) = \overline{u(z)}$
  mirror symmetry; off-center waist translates by integer samples;
  beam/plane/wavelength/grid validation.
- `EllipticalAperture` (`src/element.rs`, Phase 3 task b): ideal hard-edge
  binary amplitude mask $t = 1$ iff $(x-x_c)^2/a^2 + (y-y_c)^2/b^2 \le 1$
  (semi-axes meters, boundary inclusive — same convention as the former
  circle); circle special case via `circle()`/`radius()`; `apply` uses the
  division-free axis-multiplied predicate
  $b^2\Delta x^2 + a^2\Delta y^2 \le a^2b^2$.
- `RectangularAperture` (`src/element.rs`, Phase 3 task b): hard-edge mask
  $t = 1$ iff $|x-x_c| \le a \land |y-y_c| \le b$ (half-widths meters,
  edges inclusive; square = $a == b$ special case); far field is the
  separable sinc² pattern already validated in `tests/phase2.rs`.
- Element unit tests (10): per-shape constructor validation, circle
  special-case reporting, bounding-box `OutsideField` rejection with
  diagnostics, boundary-inclusive mask geometry, power reduction = area
  ratio ($\pi ab$ and $4ab$ over $WH$, 3%), interior phase-ramp
  preservation, `dyn Element` dispatch incl. conservative extent.
- `ThinLens` (`src/element.rs`, Phase 3 task c, decision D1): ideal thin
  lens — quadratic phase screen with **dispersive focal closure**
  $t = \exp(-i\frac{2\pi}{\lambda}\frac{\rho^2}{2 f(\lambda)})$ inside a
  circular clear aperture (hard-edge stop, boundary inclusive), optical
  axis `center` (m); `clear_radius = None` gives the field-wide special
  case (no stop). Sign: converging $f>0$ = negative quadratic — matches
  the converging-curvature convention of `sources` ($e^{+ikz}$ frame) and
  the textbook lens/DOE-lens construction (Sec. 4.2.1, Eq. 4.13 zone radii
  $r_p^2 = 2fp\lambda$); chromatic law lives in the closure (Sec. 4.1.3:
  refractive $f$ shortens at blue; DOE-like $f \propto \lambda$ keeps the
  phase fixed). Constructors `new(closure, clear_radius, center)` +
  `fixed(f0, clear_radius, center)` sugar; `focal_at(lambda)`,
  `clear_radius() -> Option<f64>`; `worst_phase_step(band)` closed-form
  $O(1)$ sampling diagnostic (upper bound for the stopped form, exact
  field-wide; reported, not enforced — elements stay local).
  `kind = Lens`; stopped lens reports `shape = Circle` /
  `extent = Aperture(a)`, field-wide reports `Unbounded`/`Unbounded`;
  new errors `OpticsError::InvalidFocal { lambda, f }` (zero or non-finite
  closure result at `apply`) and `InvalidAxis { .. }`; `InvalidAperture` /
  `OutsideField` reused for the clear aperture.
- Thin-lens unit tests (8): Fourier property — Gaussian waist at the lens
  plane, propagate exactly $f$ (ASM + Fresnel convolution), focal width
  $\lambda f/(\pi w_0)$ and amplitude $w_0/w_0'$ to 1%; pure-phase profile
  $|t|=1$ with $-k\rho^2/2f$ wrapped-exact at sampled points; dispersive
  closure evaluated per band ($f$ linear in $\lambda$); diverging lens =
  conjugate phase; `InvalidFocal`/`InvalidAxis`/`InvalidAperture` +
  taxonomy for both forms; `worst_phase_step` closed form = finite-
  difference sweep maximum (field-wide); stopped lens masks outside the
  aperture (boundary inclusive, power = aperture area ratio, oversized
  footprint rejected).

### Changed

- `OpticsError` variants carry footprint half-extents (see Decisions);
  `CircularAperture` references in `lib.rs` quick-start, `regime`/
  `fraunhofer` docs, `docs/architecture.md`, `docs/physics.md`, and the
  integration tests migrated to `EllipticalAperture::circle`.

### Removed

- `CircularAperture` — superseded by `EllipticalAperture::circle` (task-b
  review decision, 2026-10-12); mask predicate, validation, and regime
  pairing preserved.

## [0.2.0] - 2026-10-12

Phase 2: Fresnel (TF + convolution) and Fraunhofer propagation, regime
guidance, off-axis modeling, review-driven sampling guards.

### Decisions (approved in planning check-ins, 2026-10-11, Phase 2)

- **FFT normalization migrated to NumPy convention** (Phase 2 task-1
  check-in): unnormalized forward, $1/N$ inverse — the circular convolution
  theorem becomes exact ($\mathcal{F}^{-1}\{\mathcal{F}\{u\}\cdot
  \mathcal{F}\{h\}\} = u \circledast h$), required by the convolution-form
  Fresnel propagator and PSF analysis. **Supersedes** the Phase-1 unitary
  scaling decision (see 0.1.0 above); ASM and transfer-form propagation are
  behaviorally unaffected (scalings cancel). Parseval/delta/plane-wave fft
  tests updated to the new convention; a new direct circular-convolution
  cross-check pins the theorem.
- **Phase 2 scope**: Fresnel transfer-function propagator + Fresnel
  convolution (impulse-response) propagator + Fraunhofer limit; regime
  guidance as **diagnostics only** (`fresnel_number`, `regime_report` — no
  auto-select wrapper); full analytic validation set (Gaussian beam $q(z)$,
  straight-edge Fresnel integrals, rectangular-aperture sinc², Airy via
  Fraunhofer, cross-model consistency).
- **Pre-commit check-in mandatory** (agents.md update, commit `be2cb30`):
  explicit approval before every commit; Conventional Commits enforced.
- **Off-axis modeling via `FraunhoferConfig::source_pos`** (review
  2026-10-11): the optical axis must not be implicitly tied to the grid
  center. Audit result: ASM and Fresnel (both forms) are translation-
  invariant — diagonal-in-frequency operators / kernels of the difference
  $(x - \xi)$ — so they need no parameter (documented in their module
  docs); Fraunhofer's prefactor chirp and frequency map ARE axis-centered,
  so `source_pos: Option<(f64, f64)>` (meters, field coordinates; `None` =
  origin) was added: an exact unit-modulus input ramp $e^{i2\pi(f_s \xi +
  g_s \eta)}$, $f_s = x_s/(\lambda z)$, re-centers the evaluated frequency
  grid on the axis (no interpolation), the chirp is taken about
  $(x_s, y_s)$, and the regime gate's conservative extent $N_F$ is measured
  from the axis to the farthest window edge.

### Added

- Fresnel **convolution (impulse-response) propagator**
  (`fresnel::propagate_convolution`, Phase 2 task 3): paraxial Green's
  function $h = \frac{e^{ikz}}{i\lambda z}\exp\!\big(i\pi\frac{x^2+y^2}{\lambda z}\big)$
  (textbook Eq. 2.24 kernel, obliquity factor reduced; Sec. 2.2.2 + Ch. 3:
  shift-invariant propagation = spatial convolution) sampled on the padded
  grid in DFT order (kernel origin at bin (0,0)) with the $\Delta x\Delta y$
  quadrature weight folded in; evaluated as
  $\text{ifft2}(\text{fft2}(u)\cdot\text{fft2}(h))$ — the first consumer of
  the exact convolution theorem from the task-1 NumPy scaling migration.
  Same embed/pad/extract pipeline, config, and error type as the TF form.
- **Reciprocal kernel-sampling guard**, expressed in the textbook direction
  (review 2026-10-11): worst adjacent-sample chirp step
  $\pi(n_{\text{pad}}-1)\Delta x^2/(\lambda|z|) < \pi$, i.e.
  $n_{\text{pad}}\Delta x^2 \lesssim \lambda z$ — textbook rule (2)
  $N > D^2/(Z\lambda)$ verbatim at the padded width. The convolution form is
  at home for **long $z$ / fine grids**, the TF form for **short $z$ /
  coarse grids** (negation of rule 2), crossing at critical sampling
  $n_{\text{pad}}\Delta x^2 = \lambda z$. (Corrects an inverted regime
  statement in the task-2/3 draft docs.) Densification cannot mitigate an
  undersampled kernel, so `strict` mode raises `AliasingRisk` immediately
  (no mitigation loop); the crossover feeds the task-5 regime guidance.
  `FresnelSamplingReport` now carries both criteria
  (`max_kernel_step_x/y`, `kernel_nyquist_ok`).
- **AliasingRisk reports boxed** (review 2026-10-11): `Box<SamplingReport>`
  in `asm::PropagationError` and `Box<FresnelSamplingReport>` in
  `FresnelError` — consistent API, small `Result::Err` arm (breaking for
  pattern-matchers that moved the payload; deref-transparent for field
  reads).
- **Remedy hints in report `Display`** (review 2026-10-11): undersampled TF
  → "larger field (raise padding/max_padding), coarsen pitch, or reduce
  |z|"; undersampled kernel → "padding worsens this — smaller kernel
  extent, coarsen pitch, or increase |z|".
- Convolution unit tests (5): delta input reproduces the sampled kernel in
  absolute amplitude ($\Delta^2/\lambda z$), phase, and flatness; guard
  complementarity demonstrated in one geometry (strict convolution OK where
  strict TF fails at `max_padding = 1.0`, TF recovered by densification);
  TF-vs-convolution agreement $< 10^{-6}$ at critical sampling on a compact
  Gaussian; $z = 0$ identity; shared validation rejection.
- Fraunhofer far-field propagator (`src/propagation/fraunhofer.rs`, public
  via `propagation::fraunhofer`, Phase 2 task 4):
  $U = \frac{e^{ikz}}{i\lambda z} e^{i\pi r^2/\lambda z}\,
  \widetilde{U}_0(x/\lambda z, y/\lambda z)$ (textbook Sec. 2.2, Eq. 2.19;
  assumption 9). The **only propagator that resamples**: output grid pitch
  $\lambda|z|/(n\Delta)$, extent $\lambda|z|/\Delta$, dims preserved;
  negative $z$ mirrors the pattern; $z = 0$ rejected (singular map).
  Exact bin-to-origin-centered-grid alignment via an offset DFT
  (unit-modulus pre/post phase factors implementing a $c=(n-1)/2$-bin
  shift) — no half-pixel error on even grids, no odd-$n$ restriction.
- **Fraunhofer regime gate = the Fresnel number**: $N_F = D^2/(\lambda|z|)$
  over the source window, with the dropped-quadratic phase bound
  $\pi N_F/4$ reported; `strict` mode refuses $N_F >$ `max_n_f`
  (default 1.0) with `RegimeViolation { report: Box<FraunhoferReport> }`
  and Display remedy hints (larger |z|, smaller window, or Fresnel/ASM);
  `FraunhoferConfig { strict, max_n_f }`.
- **Off-axis axis reference `FraunhoferConfig::source_pos`** (review):
  `Option<(f64, f64)>`, meters, field coordinates, `None` = grid origin.
  Frequency map $f = (X - x_s)/(\lambda z)$ realized exactly by a
  unit-modulus input ramp; prefactor chirp centered at $(x_s, y_s)$;
  $N_F$ extent measured from the axis (grows with offset);
  `FraunhoferReport` carries the resolved `source_pos`.
- Fraunhofer unit tests (10): delta-at-origin reproduces the analytic
  prefactor (flat $\Delta^2/\lambda z$ amplitude, exact quadratic phase,
  all samples); uniform window concentrates at DC with exact nulls and
  magnitude $A(n\Delta)^2/\lambda z$; far-field power equals input power to
  $10^{-9}$ (discrete Parseval through the offset DFT); regime gate +
  non-strict bypass + configurable threshold; $-z$ mirror symmetry;
  $z=0$/non-finite rejection; `None` equals explicit-origin axis; off-axis
  propagation conserves power (ramp is unit-modulus, bin map bijective);
  regime extent grows with axis offset.
- Fresnel transfer-function propagator (`src/propagation/fresnel.rs`, public
  via `propagation::fresnel`, Phase 2 task 2): paraxial form of the ASM
  transfer function, $H_F = e^{ikz}\exp\!\big(-i\pi\lambda z(f_x^2+f_y^2)\big)$
  (binomial expansion of the direction cosine; textbook Sec. 2.1–2.2), same
  FFT-diagonal structure and embed/pad/extract pipeline as ASM. API mirrors
  `asm`: `propagate(&BandField, z, &FresnelConfig)` and
  `sampling_report(&BandField, z, &FresnelConfig)`.
  `FresnelConfig { padding, strict, max_padding }` (defaults 2.0/true/8.0;
  no band-limit knob — the Matsushima mask is ASM-specific).
  `FresnelError` mirrors the propagation taxonomy with a
  `FresnelSamplingReport` payload (unification into one engine-level error
  type flagged as a later API decision).
- **Fresnel sampling guard = textbook Sec. 2.2 rules 2–3**: the worst
  adjacent-bin chirp step is closed-form and *exact* (separable quadratic:
  $\pi\lambda|z|\,\delta f\,(2f_{\text{nyq}} - \delta f)$ at the
  Nyquist-straddling bin pair); requiring it $< \pi$ is algebraically
  equivalent to $N > \lambda z/\Delta x^2$. Pinned by a brute-force
  sweep-equality unit test. Strict mode mitigates by densification
  (1 → 2 → 4 → 8) before `AliasingRisk`.
- **Paraxial validity diagnostic** on the report:
  `max_phase_error_vs_asm` (worst binomial-truncation error
  $k|z|\,(1 - s/2 - \sqrt{1-s})$ at the grid-corner bin vs the exact ASM
  root) with `paraxial_ok` flag (< 0.01 rad) — the Fresnel model has no
  evanescent cutoff, so validity is reported, not enforced.
- Fresnel unit tests (9): config defaults, $H_F$ vs $H_{ASM}$ low-frequency
  agreement at the $O(kz\,s^2/8)$ rate, closed-form step = exact sweep
  maximum, $z=0$ identity, plane-wave-bin quadratic phase law, exact
  $+z/-z$ round trip on the periodic grid ($|H_F| = 1$), strict-mode risk /
  bypass / mitigation ladder, paraxial flag vs grid angular extent, invalid
  distance rejection.
- Regime guidance (`src/propagation/regime.rs`, public via
  `propagation::regime`, Phase 2 task 5) — **diagnostics only**, per the
  approved decision (no auto-dispatch wrapper):
  `fresnel_number(a, lambda, z) -> f64` implements the regime criterion
  $N_F = a^2/(\lambda|z|)$ for a known aperture half-width (pairs with
  `CircularAperture::radius()`); `regime_report(band, z) -> RegimeReport`
  conservatively assesses a whole grid (window half-extent) and exposes
  $N_F$, far-field flag, paraxial angle ($\theta = h/|z| \le 0.1$), both
  Fresnel guards (`tf_sampled`, `kernel_sampled`), and an
  `Engine::{Asm, FresnelTransfer, FresnelConvolution, Fraunhofer}`
  recommendation via a documented ladder: non-paraxial → ASM (exact);
  $N_F \le 1$ → Fraunhofer (cheapest valid); else the Fresnel form whose
  guard holds; fallback ASM. Display prints all criteria.
  Note: the rule-2 kernel guard ($\lambda z > (n-1)d^2$) is weaker than
  far field ($N_F \le 1 \Leftrightarrow \lambda z \ge h^2$, a factor
  $\sim n/4$ larger) — far field implies kernel-sampled, not conversely;
  corrected from the task-3 draft.
- **`FraunhoferConfig::aperture_extent: Option<f64>`** (decision point,
  user choice 2026-10-11): the far-field condition is a property of the
  *source*, not the sampling window — a small aperture in a large window
  was being conservatively rejected by the window-extent gate. When given
  (positive finite half-width from the axis, e.g.
  `CircularAperture::radius()`), the regime gate and `FraunhoferReport`
  use it (`extent_used`, `extent_is_aperture`); `None`/degenerate values
  fall back to the window extent. Mirrored in `regime`: new
  `regime_report_with(band, z, aperture)`, with `RegimeReport` gaining
  `extent_used`/`extent_is_aperture`; `regime_report` delegates with
  `None` (no breaking change).
- **Fresnel-number convention unified to half-extent** $N_F = h^2/(\lambda
  |z|)$ across `fraunhofer` and `regime` (previously the Fraunhofer gate
  used full width $(2h)^2$, a factor-4 inconsistency with
  `fresnel_number`); dropped-quadratic bound now stated as $\varphi_{\max}
  = \pi N_F$ accordingly. Existing gate outcomes for the tested geometries
  are unchanged (thresholds in tests adjusted to the convention).
- Fraunhofer/regime tests for the new surface: declared aperture turns a
  rejected window geometry into an accepted one with exact $N_F$;
  degenerate declarations fall back; NaN-aperture regime equality; Display
  prints extent source.
- Regime unit tests (8): $N_F$ math + edge cases (signed $z$, $z=0 \to
  \infty$, invalid → NaN); far-field geometry → Fraunhofer; paraxial
  near-field → Fresnel transfer; large angle and $z=0$ → ASM; guard
  complementarity across the rule-2 crossover $\lambda z = (n-1)d^2$;
  window-vs-aperture conservatism; Display roundtrip.
- Phase-2 physical validation suite (`tests/phase2.rs` + shared
  `tests/common/mod.rs`, task 6) — five analytics-based integration tests,
  all passing:
  1. Gaussian beam $q(z)$ analytics (width $w(z)=w_0\sqrt{1+(z/z_R)^2}$,
     on-axis amplitude $w_0/w$, Gouy phase $-\arctan(z/z_R)$) for ASM and
     Fresnel TF to $10^{-3}$, plus ASM-vs-Fresnel cross-model consistency.
  2. Straight-edge diffraction vs the Fresnel-integral profile
     $I(u)=\tfrac12[(\tfrac12-C)^2+(\tfrac12-S)^2]$, $u=x\sqrt{2/\lambda z}$,
     to 2.5% of $I_0$ — $C,S$ by cumulative quadrature in-test.
  3. Rectangular aperture far field vs $\operatorname{sinc}^2$ via the
     Fraunhofer engine (with `aperture_extent` declared) to 3% of peak.
  4. Airy profile $[2J_1(v)/v]^2$ at $N_F = 0.05$ across three engines —
     Fraunhofer (3%), Fresnel **convolution** (2%), ASM (2%) — with the
     analytic averaged over each profile pixel (matched averaging;
     bin-center evaluation of the steep Airy core is biased).
  5. Regime-ladder smoke: `regime`'s recommended engine propagates each
     validated geometry cleanly.
  Validation surfaced two regime facts now asserted in the suite: fringe
  structure requires the grid to resolve angles $f \approx x/(\lambda z)$
  (edge-test pitch set by that criterion), and the Fresnel **TF** form is
  legitimately unusable at the Airy distance ($\lambda z \gg n\Delta^2$):
  the test asserts TF-guard-fails/kernel-guard-holds complementarity.
- Phase-2 documentation (task 7): `docs/physics.md` gains §6 Fresnel
  (both forms, reciprocal guards, critical sampling), §7 Fraunhofer (regime
  gate with half-extent convention, `source_pos`, `aperture_extent`, offset
  DFT, Parseval power), §8 regime-guidance ladder, and the validation table
  extended to eleven tests; `docs/architecture.md` updated (module map with
  all engines, uniform API shape, boxed-report error taxonomy, data flow
  with regime diagnostics, Phase-2 roadmap marked complete); crate docs in
  `lib.rs` list all engines and their safety surfaces.

### Changed

- `fresnel`: engine-boundary validation extracted into a shared private
  `validate()` used by both propagator forms (task 3; internal).
- `src/propagation/mod.rs`: `padded_len` hoisted to a shared `pub(crate)`
  helper of the propagation module (was private to `asm`) so all
  transfer-function engines use one padding policy; no behavior change.
- `src/propagation/fft.rs`: `fft2` is now unnormalized and `ifft2` divides by
  $N$ (was: $1/\sqrt{N}$ per axis in both directions). Propagator outputs are
  mathematically identical (factors cancel in transfer-function form);
  floating-point rounding differs at the $10^{-15}$ level. fft unit tests
  re-pinned to the new convention (Parseval $\times N$, flat-unit delta
  spectrum, single-bin plane wave $= N$) plus a new direct circular-
  convolution theorem test.

## [0.1.0] - 2026-10-11

Phase 1: field representation, ASM propagation, first element, physical
validation, documentation.

### Decisions (approved in planning check-ins, 2026-10-11)

- Phased roadmap; Phase 1 = field representation + ASM propagation.
- **f64 everywhere** (refactor existing f32 skeleton).
- FFT backend: **rustfft** (pure Rust, f64, arbitrary sizes).
- **Evolve the existing `Element` trait** to operate on fields.
- ASM boundaries: zero-padding (default 2x) + band-limited transfer function by
  default; `AsmConfig` extensible for future boundary-condition strategies
  (infrastructure only in Phase 1).
- Errors: plain enums with manual `Display`/`Error` impls (no `thiserror`).
  **Superseded 2026-10-11 (task 3.a check-in):** error enums now derive
  `Display`/`Error` via **thiserror** v2 — the extra dependency is accepted for
  API cleanliness; applies to all future error types (`PropagationError`,
  `OpticsError`).
- **FFT normalization: orthonormal (unitary) scaling for Phase 1** (check-in
  2026-10-11). Unitary scaling conserves total (sum) energy and is exact for
  ASM transfer-function multiplication. Flagged for Phase 2+: convolution-form
  propagators and PSF/impulse-response analysis make the convolution theorem
  carry a $\sqrt{N}$ factor under the unitary convention; revisit
  unnormalized-forward/$1/N$-inverse scaling or explicit factor handling then.
- **Wavelength decoupled from `Field`** (check-in 2026-10-11): `Field` is a pure
  spatial grid; `BandField { lambda, field }` is the unit of propagation and
  element interaction, so future incoherent multiband sources are collections
  of `BandField`s (intensities sum at detection). Phase-1 API follows:
  `propagate(&BandField, z, cfg)` and `Element::apply(&BandField)`.
- **Signed propagation distances** (task 4, resolving a plan conflict): the
  plan's `InvalidDistance (z ≤ 0)` contradicted its own round-trip test
  (propagate $+z$ then $-z$). Implemented: `InvalidDistance` fires only for
  non-finite $z$; $z = 0$ is the identity; negative $z$ back-propagates
  ($H(f, -z) = H(f, z)^*$).
- **Element interaction is local; regime judgment lives in propagation**
  (task 5 check-in, 2026-10-11): `Element::apply(&BandField) ->
  Result<BandField, OpticsError>` replaces `transform(&Wave, angle)`
  (breaking; oblique interaction deferred). Whether the pattern downstream of
  an aperture of radius $a$ is Fresnel or Fraunhofer at distance $z$ is judged
  by the **Fresnel number** $N_F = a^2 / (\lambda z)$ ($N_F \gg 1$ near field,
  $N_F \lesssim 1$ far field) — a propagation decision combining aperture,
  wavelength, and inter-element distance. ASM is regime-agnostic; Phase 2+
  Fresnel/Fraunhofer propagators must expose/consume $N_F$ (recorded in
  `element` module docs for regime guidance).

### Added

- `Field` (`src/field.rs`): complex scalar optical field $u(x, y)$ on a uniform,
  origin-centered, row-major grid; SI units (meters); wavelength-independent.
  Constructors `new`, `zeros`, `uniform`; accessors `dims`, `pitch`, `sample`/
  `sample_mut`, `coord`, `at` (nearest-neighbor in physical meters, `None`
  outside the grid extent), `amplitudes`, `phases`, `intensities`, `power`,
  `samples`; `FieldError` validation enum.
- `BandField` (`src/field.rs`): a `Field` bound to one wavelength. Constructors
  `new`, `uniform`, `plane_wave` (tilted plane wave via direction cosines
  $\alpha = \sin\theta_x$, $\beta = \sin\theta_y$ per textbook Eq. 3.5);
  accessors `lambda`, `field`/`field_mut`, `into_field`.
- Unit tests for field index mapping, coordinate centering, `at()` bounds,
  uniform-field power, plane-wave phase ramp, band/grid validation.
- `propagation` module (`src/propagation/`): module docs establish the
  `propagation → field` dependency direction (algorithms depend on data, never
  the reverse).
- Private FFT layer (`src/propagation/fft.rs`): orthonormal in-place `fft2` /
  `ifft2` over row-major buffers via **rustfft** (arbitrary sizes), frequency
  axes `fftfreq` (unshifted DFT bin order) and `fftfreq_shifted` (ascending),
  NumPy-convention `fftshift1d/2` and `ifftshift1d/2`. Note: rustfft is
  unnormalized in *both* directions (raw round-trip returns $N u$), so unitary
  scaling is $1/\sqrt{N}$ per axis per direction.
- FFT unit tests: round-trip identity (even/odd sizes incl. 1×N), Parseval
  energy conservation, delta → flat spectrum, plane-wave bin/sign convention,
  `fftfreq` and shift semantics vs NumPy.
- Dependencies: `rustfft = "6"`, `thiserror = "2"` (error derive).
- ASM propagator (`src/propagation/asm.rs`, public via `propagation::asm`):
  transfer function $H = \exp(i k z \sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2})$
  with evanescent cutoff (textbook Eqs. 3.5–3.13), zero-padding (default 2x,
  central window embedded/extracted), Matsushima & Shimobaba band-limited mask
  using the **padded** grid width in $f_{\lim} = 1/(\lambda\sqrt{(2z/L)^2+1})$.
  API: `propagate(&BandField, z, &AsmConfig) -> Result<BandField,
  PropagationError>` and `sampling_report(&BandField, z, &AsmConfig) ->
  SamplingReport`. `AsmConfig { padding, band_limited, strict, max_padding }`
  with Phase-1 defaults (2.0, true, true, 8.0); structurally extensible for
  future boundary strategies.
- **Aliasing guard redesigned as O(1) mitigation, not O(N²) detection**
  (check-in 2026-10-11): the worst adjacent-bin phase step of $H$ is bounded in
  closed form from candidate worst-case bin pairs (band edges + mask corner);
  in `strict` mode a violation first densifies the padded grid (doubling
  `padding` up to `max_padding`; `1.0` disables mitigation), and
  `AliasingRisk` is returned only when the cap cannot restore Nyquist adequacy.
  Prevention remains the default via band-limiting.
- `SamplingReport`: lambda, z, input dims/pitch, requested vs used padding,
  padded dims, per-axis band limits, closed-form max adjacent-bin
  transfer-function phase steps per axis, `nyquist_ok` (steps < π); manual
  `Display`. `PropagationError` (thiserror):
  `InvalidDistance`, `InvalidWavelength`, `EmptyGrid`, `NonFiniteSample`,
  `AliasingRisk { report }`.
- ASM unit tests: config defaults, padding math, band-limit behavior vs z,
  $z=0$ identity, exact on-axis plane-wave phase $e^{ikz}$ and $+z/-z$
  round-trip on the unpadded periodic grid, invalid-distance rejection, and
  strict-mode `AliasingRisk` when mitigation is disabled (`max_padding = 1.0`)
  and strict-mode automatic mitigation succeeding by densification
  (`padding 1 → 8`, band-limiting off).
- `Element` trait now operates on fields (task 5, `src/element.rs`):
  `apply(&self, band: &BandField) -> Result<BandField, OpticsError>` — a local
  transformation of the sampled plane (grid geometry and wavelength
  preserved); propagation and Fresnel/Fraunhofer regime judgment remain in
  `crate::propagation`. Module docs record the regime criterion
  $N_F = a^2/(\lambda z)$.
- `OpticsError` (thiserror, `src/element.rs`): `InvalidAperture` (degenerate
  element geometry) and `OutsideField { radius, center, half-extents }`
  (element footprint exceeds the field grid; carries diagnostics).
- `CircularAperture { radius, center }` (meters, `src/element.rs`): ideal
  hard-edge binary amplitude mask $t = 1$ inside the circle (boundary
  inclusive), $0$ outside; `new` validates geometry; implements `Element`
  (`shape` = `Circle`, `kind` = `Aperture`, `extent` = `Aperture(radius)`,
  placeholder non-absorbing `material` metadata unused by the Phase-1 mask).
- Element unit tests: constructor validation, `OutsideField` rejection with
  diagnostics, boundary-inclusive mask geometry, power reduction = area ratio
  ($\pi a^2 / WH$) for uniform input, phase-ramp preservation on tilted plane
  waves, `dyn Element` dispatch (trait-object safety), grid half-extent math.
- Physical validation suite (`tests/asm.rs`, task 6) — six integration tests
  of the ASM propagator against textbook analytics, all passing:
  1. Energy conservation: compact Gaussian, default config, relative power
     change $\le 10^{-9}$ (window half-extent = 8.5 sigma so crop/band-clip
     loss is negligible).
  2. Plane-wave invariance (`padding = 1.0`, periodic grid): intensity flat,
     global phase $= kz \pmod{2\pi}$ to $10^{-6}$.
  3. Tilted plane wave: exact-DFT-bin tilt ($\alpha = m\lambda/L_x$), phase
     ramp slope $k\alpha$ preserved after propagation to $10^{-6}$.
  4. Airy far field: hard-edge circular aperture ($a = 1$ mm) at $N_F = a^2/(\lambda z) = 0.05 \ll 1$
     (Fraunhofer regime per the task-5 decision); ring-averaged profile vs
     $[2J_1(v)/v]^2$ within 2% of peak to $r = 12.6$ mm, first zero at
     $1.22\lambda z/(2a)$ within 2%. $J_1$ computed by Simpson integration of
     its standard integral in-test (no special-functions dependency added).
  5. Sampling guard: closed-form step ~19 rad > $\pi$ scenario ->
     `AliasingRisk` with mitigation disabled, `strict = false` proceeds with
     flagged report, `max_padding = 8.0` succeeds via automatic densification
     (step $\propto 1/$padding; adequate exactly at 8).
  6. Round trip $+z/-z$: Gaussian band recovers to $10^{-6}$ relative
     Frobenius norm (default padding, band-limited).
- `docs/physics.md` (task 7): ASM derivation from the plane-wave basis
  (textbook Eqs. 2.9, 3.5–3.13), transfer function and evanescent cutoff,
  FFT conventions (unitary scaling, forward sign, unshifted bin order,
  convolution-$\sqrt{N}$ scope warning), zero-padding rationale, Matsushima
  band-limit formula (padded-width convention), $O(1)$ closed-form sampling
  guard and mitigation, Fresnel-number regime criterion
  $N_F = a^2/(\lambda z)$, validity regime/limitations, and the test-to-
  analytics mapping table.
- `docs/architecture.md` (task 7): module map and dependency direction
  (`propagation → field`, never the reverse), `Field`/`BandField` split with
  multiband rationale, element local-interaction contract, error taxonomy
  table, data flow, testing strategy, Phase 2–5 roadmap hooks.
- Crate-level docs in `src/lib.rs` with a runnable quick-start doctest
  (mask + propagate); unit-bearing docstrings completed for all public items
  of `common` (Angle radians newtype, `J`) and `wave` (Wavelength,
  Polarization Jones vector, Wave accessors).
- `CHANGELOG.md`.

### Changed

- `refactor:` widened all numeric types from f32 to f64 across `common`, `wave`,
  and `element` (`Angle` derefs to `f64`, `J: Complex64`, `Material` closures
  `(f64, f64) -> f64`, `Extent`/`ElementShape::Poly` to `f64`). No behavior
  change.
- **Breaking (task 5):** removed `Element::transform(&Wave, angle) ->
  Option<Wave>` in favor of `Element::apply(&BandField)` (see Added). The
  angle parameter (oblique interaction) is deferred; `Wave` is now documented
  as a scalar source/beam parameter record only, not an element operand.
