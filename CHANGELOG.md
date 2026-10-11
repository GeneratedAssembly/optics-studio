# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), and commits
follow [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

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
