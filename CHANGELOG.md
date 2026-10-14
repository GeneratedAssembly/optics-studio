# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), and commits
follow [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

- Phase-9 task 9.11 — custom merit frontend ADR **accepted**: pyo3+numpy
  behind feature `py` (Rust binary embeds the Python interpreter), layered
  gradients for user merits (optional user-supplied `obs_gradient`
  callable, automatic FD-on-obs fallback, shipped gradient check helper);
  expression DSL deferred to the Phase-10 UI ADR, extension-module
  packaging deferred to 0.2.x (`docs/adr-9.11-custom-merit-frontend.md`).
  User decision via the GEN-46 question card, 2026-10-14. Docs only —
  implementation is task 9.12. (GitHub #3 / GEN-46.)
- Phase-10 task 10.0 — UI framework ADR **accepted**: dioxus, pinned
  `=0.7.10`, web target (`wasm32-unknown-unknown`), CSR-only; leptos
  0.8.22 recorded as fallback (`docs/adr-10.0-ui-framework.md`). Ratifies
  the S1-D draft (`docs/ux/adr-10.0-draft.md`, now marked superseded)
  after a pin-time check: standalone edition-2024 crate with
  `dioxus = "=0.7.10", features = ["web"]` compiles clean for wasm32 on
  rustc 1.98 (dioxus MSRV 1.83). Draft Q1–Q3 resolved (pin, `dx` tooling,
  worker responsiveness model); Q4–Q7 deferred to tasks 10.2–10.4.
  `tasks/plan.md` decision 4 and Phase-10 section record the decision.
  Docs only — no UI code (10.1 adds `crates/studio`). (GitHub #8 / GEN-51.)
- Phase-10 UI/UX spec (`docs/ux/spec.md`) and framework ADR draft
  (`docs/ux/adr-10.0-draft.md`): personas, forward (`.zmx` → simulate →
  PSF/metrics) and inverse (carrier → merit → solver → review) workflows,
  10.1–10.3 screen inventory with panel→headless-API bindings verified
  against `cargo doc` of `optics-core`/`optics-zemax`/`optics-inverse`,
  rendering + sampling-warning constraints, and a gap register (no
  invented APIs). ADR compares dioxus vs leptos against the workspace
  constraints and recommends dioxus with open questions for the Phase-10
  check-in. Docs only — no UI code. (S1-D pre-work, task GEN-42.)
- `Material::sellmeier_book1(constant, terms)` (core): generalized Book-1
  form $n^2 = C_0 + \sum B\lambda^2/(\lambda^2 - C)$ — required by the
  H.H. Li (1980) CaF2 parameterization; `sellmeier` is the $C_0 = 1$ case.
- Example `examples/inverse_drism.rs`: SAGE IV DRISM telescope — spec-to-
  geometry inverse derivation (six-band meta-filter blaze + passbands vs
  FPA format), wave-optics verification of band landing positions, and
  off-band rejection; writes `target/drism_fpa.pgm`.

### Workspace

- Repository is now a cargo workspace: root package `optics-core` +
  `crates/optics-zemax` (path dependency). No `git mv`, history intact.

### Fixed

- `optics-zemax`: `agf` NM-line nd/Vd scan compiles on rustc ≥ 1.98
  (`Range::contains` received the `find` closure's `&&f64`; deref to
  `&f64`). Build-only — no behaviour change; surfaced when
  `optics-inverse` gained the `optics-zemax` dev-dependency (task 9.7).

### New crate: optics-zemax 0.1.0 (Phase 8)

- `zmx`: text-format `.zmx` parser (line-oriented, CRLF-tolerant):
  `MODE SEQ` required; `UNIT` length normalization to mm (MM/CM/M/IN);
  `NAME`/`GCAT`/`ENPD`/`WAVM`/`PWAV` header capture; per-surface
  `COMM/STOP/TYPE/CURV/CONI/PARA/DISZ(INFINITY)/GLAS/DIAM/MIRR`. Unknown
  keywords ignored (real files carry hundreds); structural failures
  typed (`UnsupportedMode`, `BadHeader`, `BadField`).
- `catalog`: built-in Sellmeier entries with provenance — CAF2 (H.H. Li,
  JPCRD 9, 161, 1980), F_SILICA (Malitson 1963), N-BK7 (SCHOTT);
  `index_at`, `material` (air = None). Published fits reproduce the
  Thorlabs ACA254-100-UV paraxial BFL to 0.15% at the design center
  wavelength (EFL ~3% off nominal 100 mm — Zemax's proprietary SCHOTTIR
  catalog residual; external `.AGF` import pending).
- `builder`: `ImportedSystem` — parse-time validation (STANDARD only,
  mirrors and PARA aspherics rejected with reason), dispersive
  `RefractiveSurface` + medium-gap `Beamline` plan (two-phase assembly,
  no self-reference), `paraxial_at` (matrix EFL/BFL), `trace`/`psf` and
  `trace_subaperture`/`psf_subaperture` (stop-down knob for small
  analysis grids), `TraceError<E>`. Glasses resolve at the traced band's
  wavelength — dispersive media stay consistent through the chain.
- Golden corpus: `tests/data/ACA254-100-UV.zmx` (user-provided Thorlabs
  prescription, text format); `import_doublet` integration test traces a
  UV beam through the imported doublet and verifies the imported image
  plane is the focal plane (on-axis intensity > 8x at ±20 mm defocus,
  > 30% power within 60 um of the spot centroid).
- Binary Zemax containers are out of import scope (verified 2026-10-14):
  the supplied `.zar` archive and the Edmund Optics 2019 `.ZMF`
  multi-config file (7432 part-number records, high-entropy payloads —
  not text, not binary-ZMX blocks, no zlib streams). Supported corpus =
  text `.zmx`; export from OpticStudio (File > Save As > Text) for
  batch designs.

## [optics-inverse 0.1.0] - 2026-10-14

Phase 9 Milestone A close-out — **first release of the `optics-inverse`
crate**: the inverse-design framework (bounded `Params`, carriers,
renderer + `Observation`, `Merit` seam, solvers), Gerchberg–Saxton DOE
design, finite-difference Adam, and adjoint grid gradients (no autodiff
graph). Physics documented in `docs/physics.md` §12–13, crate architecture
in `docs/architecture.md` §7, adjoint derivation in
`docs/adr-9.9-adjoint-gradients.md`. Milestone B (custom-merit frontend,
0.2.0) and Milestone C (`burn` surrogates, 0.3.0) follow per
`tasks/todo.md`.

### Added

- New workspace member `crates/optics-inverse` (`optics-inverse` 0.1.0):
  `InverseError` + crate docs scaffold (Phase-9 task 9.1), then the
  framework below built on top of the untouched core API.
- `Param`/`Params` — the bounded design-space vector all solvers see:
  closed `[lo, hi]` bounds, per-slot FD step (default 1e-4·span),
  validate-at-apply (never panics), clamp policy, seeded `randomize`.
  (Phase-9 task 9.2.)
- `Design`/`Step` (owned element stack borrowing into a core `Beamline`,
  zemax two-phase-plan pattern) and `Carrier` — `analytic` (closure
  `(values, λ) → Design`), `grid_phase` (sampled radians), and
  `grid_thickness` (relief meters, chromatic
  $\varphi = 2\pi(n(\lambda)-1)h/\lambda$ rebuilt per band). (Phase-9
  task 9.3.)
- `Observation` (numpy-ready flat `[channel][band][scene]
  [row][col]` stacks) + `Renderer` — incoherent multi-band, multi-scene
  render through the core ASM, with finite-difference derivative channels
  ∂I/∂x₀, ∂I/∂y₀, ∂I/∂λ (the Fisher/CRLB input). Goldens: defocus/mask/
  defocus piston identity, A(+z)∘A(−z) round-trip, ∂I/∂x₀ = −∂I/∂x
  translation invariance, chromatic-focus λ-channel. (Phase-9 task 9.4.)
- `Merit` trait (`evaluate` + `obs_gradient` adjoint seam) with built-ins
  `MseToTarget`, `EncircledEnergyMerit`, `SpotRmsMerit`, `StrehlMerit`,
  `MtfMerit` over a `Selector` (channel × band × scene, aggregation
  supported), plus `objective()` composing renderer+merit for solvers.
  Built-in values match core `analysis` on golden PSFs; analytic
  gradients FD-verified. (Phase-9 task 9.5.)
- Derivative-free solvers — adaptive-step random search and simulated
  annealing (auto-calibrated T0, geometric cooling), sharing a
  `Solver`/`Solution`/`StochasticConfig` API over the objective closure
  with bound-clamping prevention. (Phase-9 task 9.6.)
- `fd` — central-difference gradients over `Params` (2·dim renders per
  step; steps are each slot's `fd_step`, defaulting to 1e-4 of the bound
  span) and `Solver::Adam`: Adam descent with gradient-norm clipping and
  best-epoch snapshot, bound-clamped probes (one-sided slope at bounds),
  deterministic by default; for this variant `StochasticConfig::budget`
  counts epochs (`2·dim + 1` evaluations each) and `Solution::history`
  records one merit value per epoch like the 9.6 solvers. Golden: bend
  one curvature of the imported ACA254-100-UV doublet to minimize spot
  RMS — the zemax→inverse loop, with `optics-zemax` as a dev-dependency
  only (public API unchanged). (Phase-9 task 9.7.)
- `gs` — Gerchberg–Saxton phase retrieval / DOE design (textbook §5.5,
  `docs/physics.md` §12): a phase-only `GridPhase` carrier between input
  and target planes relayed by two ASM half-transforms, with energy
  projections at both planes (target amplitude $T$; DOE-plane
  illumination amplitude under the carrier aperture), exposed error
  signals (target-plane amplitude residual $|U_t|-T$ and DOE-plane phase
  residual $\Delta\varphi$) and per-iteration convergence diagnostics
  (merit, best-so-far, Eq. 5.18 amplitude MSE, error RMS). Adaptive
  iteration options `Dvr` (damped phase update, $\beta\in(0,2)$) and
  `Fpr` (relaxed Fourier-plane replacement, $\beta\in(0,1]$); stopping
  runs through the 9.5 `Merit` seam on each iteration's `Observation`.
  Goldens: top-hat shaper (flat interior, monotone best-so-far MSE) and
  recovery of a known phase-only lens from its target-plane intensity;
  FFT shift/normalization and phase-sign conventions pinned by dedicated
  tests. New `Carrier::grid_geom()` accessor for grid-domain solvers.
  (Phase-9 task 9.8.)
- `adjoint` — reverse-mode (adjoint) grid gradients, ADR first
  (`docs/adr-9.9-adjoint-gradients.md` records the math,
  `docs/physics.md` §13): the ASM adjoint is propagation by −z with the
  same config (exact under the pinned FFT convention —
  `asm_adjoint_inner_product_identity`), the thin-element adjoint is
  conjugate transmission recovered by probing `apply` on a unit field,
  and the aperture is a self-adjoint 0/1 mask — so `dM/dgrid` costs one
  forward render + one adjoint run per observation frame with no
  autodiff graph (`asm_adjoint`, `transmission`, `push_back`,
  `grid_gradient`). Acceptance: adjoint grid gradient matches FD on a
  coarse grid (global relative error < 1e-4, test
  `grid_gradient_matches_fd_on_coarse_grid`), and the GS phase error
  signal is the negative adjoint gradient of the Eq. 5.18 amplitude MSE,
  preconditioned by the illumination intensity (∇φM = −(2/N)A²⊙Δφ to
  first order — correlation > 0.9 pinned by
  `gs_phase_error_is_adjoint_mse_negative_gradient`).
  Merits without `obs_gradient` and non-intensity seed channels surface
  as the new `InverseError::Adjoint` (FD-on-observation remains the
  fallback). New `Design::elements()`/`Design::steps()` accessors.
  (Phase-9 task 9.9.)
- Example `crates/optics-inverse/examples/coded_doe.rs`: GS top-hat
  beam shaper — Gaussian illumination, power-scaled flat-top target,
  convergence trace, interior flatness report, and PGM dumps of the DOE
  phase and delivered target frame; rebuilds the design through
  `Carrier::build` to prove the returned phase is a legal parameter
  vector. (Phase-9 task 9.10.)

### Docs

- `docs/physics.md` §12 (Gerchberg–Saxton: relay geometry, Eqs.
  5.14–5.18 projections, DVR/FPR relaxations, power-reachability,
  conventions) and §13 (adjoint gradients: the three adjoint facts,
  phase-grid formula $\partial M/\partial\varphi_p = -2\,\Im(\bar\mu_p
  V_p)$, cost, GS error-signal relation, scope/fallbacks).
- `docs/architecture.md` §7: the `optics-inverse` crate — module map,
  one-way dependency on the core public API (`optics-zemax` dev-only),
  solver-loop data flow, and the 0.2.0/0.3.0 milestone seams.

## [0.7.0] - 2026-10-14

Phase 7: core forward-physics close-out — `Material::sellmeier`,
incoherent multiband summation (`sum_intensity`, `Psf` promoted to
`field`), PSF analysis metrics (EE/spot RMS/MTF/Strehl/Maréchal), Zernike
wavefront decomposition, and paraxial surface tilt. **The core crate is
feature-complete for forward modelling**; the workspace roadmap
(`optics-zemax`, `optics-inverse`, `optics-studio`) is tracked in
`tasks/plan.md`.

### Added

- `Material::sellmeier(terms)` (Phase 7 task 7.1): standard Sellmeier Book-1
  dispersion model $n^2 = 1 + \sum B_i \lambda_\mu^2/(\lambda_\mu^2 - C_i)$
  (catalog $(B_i, C_i)$ pairs, µm convention, zero absorption, angle
  ignored); NaN outside resonance-free bands surfaces as `InvalidMedium` at
  apply. N-BK7 golden tests: n_d/n_F/n_C within 5e-4 of catalog,
  F−C dispersion 0.00807, Abbe V_d ≈ 64.2, blue-stronger law.
- `field::sum_intensity(&[&BandField]) -> Result<Psf, FieldError>` (Phase 7
  task 7.2, decision D3 close-out): incoherent multiband intensity sum —
  identical-grid validation (`FieldError::GeometryMismatch` newly added),
  intensities not amplitudes, no cross terms.
- `analysis` module (Phase 7 task 7.3): detector-plane metrics over `Psf` —
  `encircled_energy` (radius-sorted cumulative curve with
  `value_at`/`radius_for_fraction` interpolation; Airy anchor
  EE(r1) = 0.84 validated through the Fraunhofer engine),
  `spot_rms_radius` (second moment; Gaussian anchor = w), `mtf` (|F{I}|/DC
  along +fx in DFT order; Gaussian anchor exp(-pi^2 w^2 f^2) < 2%),
  `strehl` (power-equalized peak ratio vs reference PSF), and
  `marechal_strehl` (exp(-(2 pi sigma)^2), pairs with Zernike RMS).
  `propagation::fft` module raised to `pub(crate)` for the MTF transform.
- Zernike wavefront decomposition (Phase 7 task 7.4, `analysis`):
  `ZernikeMode`/`zernike_modes` (explicit (n, m) ordering, not Noll),
  `zernike_radial` via the exact binomial-sum Jacobi $P_k^{(m,0)}$
  (golden-checked: $R_2^0 = 2\rho^2-1$, $R_3^1 = 3\rho^3-2\rho$,
  $R_4^0$, $R_4^2$), `zernike` with $\sqrt{n+1}$/$\sqrt{2(n+1)}$
  normalization (unit RMS over the disk), `zernike_fit` least-squares
  (Cholesky normal equations) returning coefficients (radians), residual
  RMS, and `residual_rms_waves` feeding `marechal_strehl`;
  `AnalysisError { InvalidPupil, DimensionMismatch, EmptyPupil, Singular }`.
  Tests: discrete orthonormality Gram ($\pm 2\cdot10^{-3}$, 512^2),
  synthetic 3-mode recovery to 1e-9, unmodeled spherical-aberration
  residual tracks 0.02 waves -> Strehl 0.984.
- `Psf` promoted from `beamline` to `field.rs` as the crate's real-valued
  intensity-grid data type (pure data + bookkeeping): constructors
  `Psf::new` (validating) and `Psf::from_band`; `beamline` re-exports
  `Psf` so the `optics_core::beamline::Psf` path stays live.

- Paraxial surface tilt (Phase 7 task 7.5, breaking):
  `RefractiveSurface::new` gains a `tilt: Option<(f64, f64)>` parameter
  (rad; adds $\tau_x x' + \tau_y y'$ to the surface height — prism action
  $\sin\theta = -(n_2-n_1)\tau$, validated bit-for-bit against `Wedge` at
  $\tan\alpha = \tau$ with zero-piston equivalence and a sampled piston
  check); `surface_z(x, y)` / `tilt()` accessors; the guard bound includes
  $|\tau|$. `ThickLens` deliberately takes NO tilt: a rigid-body tilt
  cancels identically in the thickness difference (documented; tilted
  thick elements compose tilted surfaces). `OpticsError::InvalidTilt`.

### Changed

- **Breaking (Phase 7 task 7.5):** `RefractiveSurface::new` signature
  gains the `tilt` parameter (position 4); `None` = untilted.
- **Crate renamed `optics-rs` → `optics-core`** (Phase 7 task 7.0): a crate
  named `core` collides with Rust's built-in core crate in extern-prelude
  paths; the `optics-` prefix is the locked naming rule for the workspace
  (`optics-core`, `optics-zemax`, `optics-inverse`, `optics-studio`).
  Repository directory name unchanged; no re-export shim (0.x).

## [0.6.0] - 2026-10-14

Phase 6: thick elements — `RefractiveSurface` with Zemax standard-surface
sag profiles (sphere/conic/even-asphere/analytic), medium-aware ASM
(`AsmConfig::medium`), `Beamline` per-step media (`push_medium`, `gap`)
composing the Zemax-faithful surface → glass → surface stack (thick-lens
Eq. 4.1 validated), `ThickLens` single-plane convenience (Eq. 4.17,
`Extent::Thick` + `Material` consumed, spherical-aberration anchors), and
the `Wedge` prism; hybrid refractive-DOE achromats validated.

### Decisions (approved in planning check-ins, 2026-10-14, Phase 6)

- **Zemax import as model driver** (user requirement): the primary thick-
  lens representation is the composition surface → medium step → surface
  (a prescription row maps 1:1 to `RefractiveSurface` + `Beamline` step);
  `ThickLens` is a demoted single-plane convenience (thin-lens law +
  exact-profile aberrations, no principal-plane separation). A future
  `zemax` parse layer needs no model changes.
- **Surface screens are scalar normal-incidence phase plates**: no Fresnel
  amplitude coefficients, no polarization; valid at small angles and
  moderate $|n-1|$ phase strengths. Documented limitation.
- **Mirrors and waveplates deferred** (planning decision 2, user-delegated):
  mirrors require an axis fold (retro-propagation breaks the single
  forward-$z$ model); waveplates require vector fields (polarization is
  locked out of scope).
- **Eq. 4.1 bracket re-verified against the rendered PDF page**: the
  (n−1) factor multiplies the whole bracket incl. the thickness term
  (text-extraction layout had misplaced it); matrix-optics BFL
  $\Phi = \Phi_1 + \Phi_2 - (t/n)\Phi_1\Phi_2$,
  BFL $= (1 - (t/n)\Phi_1)/\Phi$ is the validated anchor.
- **Beamline closure signature is `(band, z, medium)`** (breaking, task b):
  medium reaches the caller's engine per step; `push` keeps vacuum 1.0.

### Added

- `SurfaceProfile` + `RefractiveSurface` (Phase 6 task a): Zemax standard-
  surface sag $z(r) = cr^2/(1+\sqrt{1-(1+k)c^2r^2}) + \sum \alpha_i r^{2i}$
  (rational form, stable at the vertex; `Spherical`/`Conic`/`EvenAsphere`/
  `Analytic`), boundary screen $\varphi = k(n_1 - n_2)z(r)$ with dispersive
  `Material` front/back; `sag`, `vertex_power(lambda)`, `indices_at`,
  closed-form `worst_phase_step`; conic-domain rejection at construction
  (`InvalidSurface`), non-finite sag at apply (`InvalidSag`), bad media
  (`InvalidMedium`). 6 unit tests (golden sags incl. exact paraboloid,
  Eq. 4.21 plano-convex anchor, taxonomy, validation, guard brackets).
- Medium-aware ASM (Phase 6 task b, breaking): `AsmConfig::medium` (n),
  `SamplingReport.medium`, `PropagationError::InvalidMedium`; the whole
  vacuum machinery runs at $\lambda_n = \lambda/n$ (evanescent cutoff and
  Matsushima band-limit rise with n; transfer chirp smooths as
  $\lambda^2/n$ — documented and guarded). Slab piston anchor
  $e^{ink t}$ bit-exact. 4 unit tests + round trip.
- `Beamline::push_medium` / `gap(z, n)` (Phase 6 task b): per-step media,
  engine closure `(band, z, medium)`; thick-lens composition primitive.
  2 unit tests (per-step medium reaching the closure, gap semantics).
- `ThickLens` single-plane screen (Phase 6 task c):
  $\varphi = k(n-1)[t_0 - z_1(\rho) + z_2(\rho)]$ (Eq. 4.17 generalized),
  vertex piston + thin-lens law + exact-profile spherical aberration;
  `thickness_profile`, `power_at`/`focal_at`, `worst_phase_step`,
  first consumer of `Extent::Thick`; `InvalidThickness`. 2 unit tests
  (piston, exact geometric-sag phase pin, quantified $r^4$ SA residual
  0.025 rad @ rho=2 mm, zero-thickness degeneration, validation).
- `Wedge` prism (Phase 6 task c): linear OPD
  $t = t_0 - \tan\alpha\,(\mathbf{r}'\cdot\hat g)$, mandatory apex
  `Angle`, `deflection_sine(lambda)` $= -(n-1)\tan\alpha$,
  `worst_phase_step`, `ElementKind::Prism` in use; `InvalidWedge`. 1 unit
  test.
- `tests/phase6.rs` (3 integration tests, task d): surface → 5 mm glass
  gap → surface stack focuses at the Eq. 4.1 matrix BFL (exit wavefront
  curvature fit, within 1%, rejecting the thin-lens law by >50% of its
  2.16 mm offset; geometry calibrated against an independent numpy twin —
  small apertures put the fit in the truncation-diffraction ripple
  regime, documented); hybrid refractive+DOE achromat cancels first-order
  color (<1% power drift 500–600 nm vs >4% uncorrected, Sec. 4.4.1);
  wedge far-field centroid at $-(n-1)\tan\alpha\cdot z$ (<100 µm @ 2 m).
- Phase-6 documentation (task e): `physics.md` §9.7 (+§9.6 step shape,
  §10 screen-model limits, §11 rows 23–25, References Eqs. 4.1–4.3 /
  4.17–4.21 / Secs. 4.1.5, 4.4.1), `architecture.md` element/asm/beamline/
  error/testing/roadmap, `lib.rs` blurb.

### Changed

- **Breaking** (Phase 6 task b): `Beamline::trace`/`psf` engine closures
  take `(band, z, medium)`; steps are `(Option<&dyn Element>, z, medium)`.
  `AsmConfig`/`SamplingReport` gain a `medium` field (default 1.0 —
  non-breaking for `..Default::default()` users, struct-literal callers
  must add it).

## [0.5.0] - 2026-10-13

Phase 5: sources & output — Hermite-Gauss and Laguerre-Gauss mode
initializers, the point-source spherical wave, the sampled-`PhaseMask`
variant (D2 delivered), and `beamline` composition sugar with PSF
validation; visualization/export backends deferred to the front-end
project (library stays headless).

### Decisions (approved in planning check-ins, 2026-10-12/13, Phase 5)

- **Export/visualization backend deferred** (decision 1): the library stays
  headless — plotting, serialization (PNG/FITS/CSV), and spectrum display
  are front-end-project concerns. PSF work lands as **data** (intensity
  grids) only.
- **Ergonomics = minimal `Beamline` sugar, no `Propagator` trait**
  (decision 2): ordered `(element, z_after)` steps over borrowed
  `&dyn Element`; the engine is a caller-supplied closure
  `FnMut(&BandField, f64) -> Result<BandField, E>` — per-engine configs and
  the open engine-error unification stay out of the beamline API surface;
  zero spacings skip the engine call.
- **Mode normalization: orthonormal profiles** (decision 3): HG/LG mode
  power is $A^2$ (Siegman/Goodman convention), contrasting `GaussianBeam`'s
  peak-amplitude scale; `HG00 = LG00 = gaussian × sqrt(2/pi)/w0` pinned
  bit-exactly. Total order capped at 64 (`InvalidMode`).
- **Sampled PhaseMask: bilinear lattice, zero beyond span** (decision 4):
  center-origin element-frame lattice (same convention as `Field`),
  bilinear lookup, transparent (phase 0) outside the lattice extent within
  the aperture; $O(N)$ construction validation.
- **fft.rs reserved shift helpers REMOVED** (user-approved 2026-10-13):
  `fftfreq_shifted`, `fftshift1d/2`, `ifftshift1d/2` had no consumer since
  Phase 2 — every engine works in unshifted DFT order end-to-end, and the
  Phase-5 headless decision defers display centering to the front-end.
  Recoverable from git history if a real consumer appears.

### Added

- `sources` mode initializers (Phase 5 task a): `HermiteGaussBeam` /
  `hermite_gauss` and `LaguerreGaussBeam` / `laguerre_gauss` — paraxial
  mode families on the pinned $q$-form machinery ($w(z)$, $1/R(z)$, Gouy
  $(N+1)\zeta$; LG azimuthal $e^{i\ell\varphi}$ = OAM $+\ell\hbar$ along
  $+z$); physicists' Hermite / associated Laguerre polynomials by
  dependency-free three-term recurrences; shared `envelope()` helper.
  `SourcesError::InvalidMode { order, max }`, `MAX_MODE_ORDER = 64`.
  7 unit tests: $\mathrm{HG}_{00} = \mathrm{LG}_{00} = $ `gaussian` $\times
  \sqrt{2/\pi}/w_0$ (bit-exact, waist and off-waist), on-grid orthonormal
  Gram ($<10^{-3}$), $(-1)^m$ reflection parity (bit-exact), LG ring radius
  $w_0/\sqrt2$ + telescoping $2\pi\ell$ vortex winding (both signs),
  $(N{+}1)\zeta$ Gouy on axis (incl. the $H_2(0) = -2$ phase flip and
  HG11/LG01 on-axis nodes), eigenfunction propagation through Fresnel-TF
  and ASM ($<10^{-3}$, four modes), validation.
- `PhaseMask::sampled(values, dims, pitch, orientation, aperture, center)`
  (Phase 5 task b, decision D2 delivered): sampled phase lattice in the
  element frame — bilinear lookup (exact for linear profiles, reproduces
  the DOE-lens quadratic at lattice points), zero phase beyond the span,
  bit-exact rotation equivalence (pattern rotates with the stop),
  construction-time validation. Public `SampledPhase` (`at`, `dims`,
  `pitch`, `values`) + `PhaseMask::grid()`;
  `OpticsError::{InvalidGrid, InvalidGridPitch}`; non-finite samples reuse
  `InvalidPhase` at the lattice position. 4 unit tests.
- `beamline` module (Phase 5 task c): `Beamline<'a>` ordered
  `(element, z_after)` steps, `push` chaining, `trace` with caller-supplied
  engine closure, `psf` returning `Psf` — detector intensity grid with
  `samples/dims/pitch/coord/total_power/peak_bin/peak_position/centroid`;
  `BeamlineError<E> = Element(OpticsError) | Engine(E)`. Runnable doctest;
  6 unit tests incl. bit-for-bit equality with a hand-rolled chain and
  zero-spacing call-skipping.
- `sources::spherical(source, distance, amplitude, lambda, grid)` (Phase 5
  task d): diverging point-source wave $A e^{ikr}/r$ — the finite-
  conjugate PSF input; exact-$kr/r$ anchors, $+k\rho^2/2d$ diverging-sign
  near-axis expansion, ASM↔Fresnel-TF cross-engine agreement (the
  analytic-at-$d{+}z$ comparison is dominated by hard-window truncation
  diffraction — documented why it is the wrong statement);
  `SourcesError::InvalidSource`. 3 unit tests.
- `tests/phase5.rs` (3 integration tests, task b–d chain): PSF of a
  circular pupil through the beamline vs the $J_1$ Airy anchor (full
  profile <3%, central lobe <2%, first dark ring ±2 bins); focal shift of
  a tilted plane wave through a lens (centroid at $f\tan\theta$ within
  20 µm, on-axis reference at origin); off-axis point source images at the
  lens-equation location ($d_i = f d_o/(d_o-f)$, magnification $-d_i/d_o$,
  centroid within 20 µm, spot concentration peak > 10× mean).
- `examples/psf_image_convolve.rs` (task d): test-image blur via the
  normalized PSF kernel (wrap-order roll, 1e-6 support truncation +
  renormalization) — power preserved exactly, point-source centroids
  unmoved within the even-grid half-bin ambiguity, deltas demonstrably
  blurred.

### Removed

- `propagation::fft` reserved shift helpers (`fftfreq_shifted`,
  `fftshift1d`, `ifftshift1d`, `fftshift2`, `ifftshift2`, private `shift2`)
  and their three unit tests — no consumer from Phase 2 through Phase 5;
  display centering is a front-end concern (Phase-5 decision, approved
  2026-10-13). `fftfreq` (unshifted axis) remains, still consumed by the
  transfer-function engines.

## [0.4.0] - 2026-10-12

Phase 4: Rayleigh–Sommerfeld first-kind propagation engine (exact spherical
impulse response, convolution form, closed-form kernel guard), cross-model
consistency validation vs ASM/Fresnel with a direct-quadrature reference,
and Phase-4 documentation.

### Decisions (approved in planning check-ins, 2026-10-12, Phase 4)

- **Phase 4 scope — RS first kind only**: the exact Kirchhoff integral
  couples the boundary field with its unknown normal derivative; the
  antisymmetric (Dirichlet) Green's function of the **first kind** drops
  the derivative term and consumes *only sampled field values* — the data
  a `BandField` carries. The second kind (Neumann, needs $\partial u/\partial
  n$) is deliberately excluded: estimating it as $\approx -iku$ would
  reintroduce the inconsistent-boundary approximation RS theory avoids.
  The two differ only by the $O(1/kr)$ obliquity term under the book's
  assumptions (7)–(8), and both reduce to the same paraxial kernel.
- **RS ≡ ASM is not new physics** (textbook Sec. 2.2.2/3.2: RS propagation
  *is* the plane-wave spectrum): the engine's deliverable is a **third
  discretization** (spatially sampled spherical kernel) for cross-model
  validation against ASM (spectral transfer) and Fresnel (paraxial), plus
  the exact-spherical phase where the paraxial truncation matters.
- **Direct-integral form is test-only**: the $O(n^4)$ RS-I quadrature lives
  in `tests/common` as reference truth (dependency policy, like the $J_1$/
  Fresnel-integral quadrature); the library ships the FFT convolution form.
- **$z = 0$ rejected** (`InvalidDistance`): the RS-I impulse response tends
  to a $\delta$-distribution there — sampleable nowhere; use ASM for zero
  steps (deliberate divergence from the Fresnel-convolution identity
  special case).
- **Regime ladder unchanged**: RS carries the convolution form's
  kernel-sampling condition (fails at short $|z|$; ASM remains the exact
  fallback) — documented as a cross-validation engine, not a recommendation.
- **`fft.rs` shift helpers stay reserved** (Phase 5 export paths),
  `#[allow(dead_code)]` persists.

### Added

- `propagation::rayleigh_sommerfeld` module (Phase 4 task a):
  Rayleigh–Sommerfeld **first-kind** convolution engine — impulse response
  $h = \frac{|z|}{2\pi r^2}\big(\frac{1}{r} - i\,\mathrm{sgn}(z)k\big)
  e^{i\,\mathrm{sgn}(z)kr}$, $r = \sqrt{x^2+y^2+z^2}$ (textbook Eq. 2.8
  adapted to the crate's $e^{+ikz}$ frame; the book's $e^{-ikr}$ form is
  the conjugate = back-propagation kernel), sampled on the padded grid in
  DFT order with the $\Delta x\,\Delta y$ quadrature weight folded in;
  same embed/transform/extract pipeline as `fresnel::propagate_convolution`.
  Closed-form $O(1)$ kernel guard: spherical rim step
  $k(r_{hi} - r_{lo})$ at the Nyquist-straddling pair + obliquity bound
  $k\Delta/(1 + k^2z^2) < \pi$ — **always weaker than the paraxial Fresnel
  step** ($|x|/r \le 1$: the spherical phase gradient saturates where the
  chirp's grows without bound); `strict` raises `AliasingRisk` immediately
  (densification worsens kernel sampling — no mitigation loop).
  `RayleighSommerfeldConfig { padding, strict }` (defaults 2.0/true);
  `RayleighSommerfeldSamplingReport` (kernel steps +
  `max_phase_error_vs_fresnel`/`paraxial_ok` — the non-paraxial excess
  $k(r_c - |z| - \rho_c^2/2|z|)$ the Fresnel models drop, closed-form at
  the padded corner) with remedy-hint `Display`;
  `RayleighSommerfeldError` (`InvalidDistance` non-finite **or zero**,
  `InvalidWavelength`, `EmptyGrid`, `NonFiniteSample`,
  `AliasingRisk { report: Box<…> }`). Signed-$z$ mirror
  $h(\cdot,-z) = \overline{h(\cdot,z)}$ bit-exact by construction.
- RS unit tests (9, `src/propagation/rayleigh_sommerfeld.rs`): paraxial
  kernel reduction = Fresnel convolution kernel (relative $<10^{-6}$,
  small-angle geometry); conjugate back-propagation kernel (bit-exact);
  delta input = sampled kernel anchor ($10^{-12}$); **RS ≡ ASM engine
  agreement** on a smooth Gaussian at critical-sampling scale (100 kernel
  zones, rim step 2.0 rad, rel $<5\cdot10^{-2}$); guard closed-form
  brackets a brute-force unwrapped-phase sweep (≥ sweep, < 2× sweep);
  strict raises immediately and padding provably worsens the step;
  $z = 0$/inf/NaN rejection; round trip $<10^{-2}$ (sign-convention pin);
  config defaults.
- `tests/phase4.rs` (new integration suite, 5 tests, task b) with
  `tests/common::rs_direct` — $O(n^4)$ direct RS-I quadrature (linear
  convolution over a kernel offset table) as reference truth: FFT engine =
  direct linear sum up to guard-controlled wrap (rel $<10^{-2}$; geometry
  n=48, d=2 µm, z=2 mm — rim step 1.2 rad, ~9 kernel zones); **three
  discretizations agree** (RS↔ASM and direct↔ASM $<2\cdot10^{-2}$);
  **Fresnel diverges where its own report says** — aperture geometry
  (a=20 µm, n=32, d=4 µm, z=3 mm; corner binomial truncation ≈0.29 rad per
  `max_phase_error_vs_asm`): RS vs truth $<2\cdot10^{-2}$ while Fresnel-TF
  error exceeds $4\times$ the RS error (and stays $<0.3$); $+z/-z$ round
  trip $<3\cdot10^{-2}$ (measured kernel-window ripple ≈ $1/$zones — a
  conjugate-kernel sign error would give $O(1)$); $|H_{RS}| = 1$ power
  conservation on the Gaussian geometry ($<10^{-3}$). Calibration notes:
  no power statements on the aperture geometry — its Airy-tail crop past
  the window (~11%) is physical spreading, not an artifact.
- Phase-4 documentation (task c): `docs/physics.md` §6.3 (RS first kind —
  kernel, first-vs-second-kind rationale, PWS identification, paraxial
  reduction, guard), §8 ladder note, §10 validity entry, §11 rows 15–19 +
  RS unit-test paragraph, References (Sec. 2.1, Eq. 2.8, Sec. 2.2.2/3.2);
  `docs/architecture.md` engine/diagram/error-table/testing/roadmap;
  `src/lib.rs` engine blurb.

## [0.3.0] - 2026-10-12

Phase 3: thin element library — unified apertures (`EllipticalAperture`
with the circle special case, `RectangularAperture`), `ThinLens` with
dispersive $f(\lambda)$ closure, `PhaseMask` DOE interface, `BlazedGrating`
with Fraunhofer-engine order validation, uniform element orientation — plus
the `sources` module (Gaussian beam initializer).

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

### Decisions (Phase-3 task-d review, 2026-10-12, user)

- **`ClearAperture` enum for phase-element stops** ("we can't square a
  circle"): `Field | Circle(r) | Ellipse{a,b} | Rectangle{half_x,half_y}`
  shared by `ThinLens` and `PhaseMask`, replacing the `Option<f64>`
  circular-only stop (rectangular DOE windows are the motivating case).
  Hard-edge, boundary-inclusive predicates identical to the aperture
  elements; `shape()` maps to the taxonomy, `extent_radius()` gives the
  conservative larger half-side for regime gating, `footprint()` drives
  `OutsideField`. A predicate-closure stop (arbitrary polygons) is
  **deferred for later discussion**.
- **`PhaseMask` API**: closure takes **center-relative** coordinates so DOE
  designs are translation-invariant; `kind = Diffractive`;
  `InvalidPhase { x, y, phase }` reports the first non-finite closure value
  with position; no sampling guard for arbitrary profiles (caller-side
  Nyquist duty; structured elements self-validate analytically).
- **Thin lens carries its clear aperture** (task-c revision, user
  2026-10-12; **supersedes** the same-day "pure phase, no clear aperture"
  choice): the primary `ThinLens` form is the quadratic phase inside a
  circular clear aperture (hard-edge stop, boundary inclusive) so
  `Element::extent` describes the lens itself and regime gating uses the
  lens's own half-extent. The field-wide behavior (phase over the whole
  sampled plane) is kept as the `clear_radius = None` special case;
  taxonomy retains `ElementShape::Unbounded` / `Extent::Unbounded` for it.

### Decisions (Phase-3 task 3.e.1 review, 2026-10-12, user)

- **Uniform element orientation**: every element carries an orientation ψ
  as `Angle` (not f64, for specificity). Elements with a typical
  orientation (flat, normal to $z$, axes aligned) take `Option<Angle>` —
  `None` *is* the typical orientation (ψ = 0); the `BlazedGrating` vector
  direction is intrinsic design data, so it takes a mandatory `Angle`.
  Convention pinned in the element module docs: **positive ψ rotates
  counterclockwise about the optical axis $z$ viewed from downstream
  (looking back along $-z$ against the beam); equivalently clockwise
  viewed from upstream along $+z$** (right-hand rule). Pattern *and* stop
  live in the element frame $\mathbf{r}' = R(-\psi)(\mathbf{r} - \mathbf{c})$;
  mirrors and future elements adopt the same pattern. (Also audited:
  explicit `ClearAperture::Field` everywhere — no `Option` aperture ever
  implied field.)
- **`Angle` wraps implicitly** into $[0, 2\pi)$ at every constructor and
  arithmetic op (non-finite wraps to NaN for callers to reject);
  degree/radian conversions stay explicit (`from_degrees`, `degrees()`);
  `PartialEq` added — equal-modulo angles compare equal.
- **Rotated non-circular stops**: `OutsideField` carries the axis-aligned
  bounding-box half-extents $h_x|\cos\psi| + h_y|\sin\psi|$; circles are
  rotation-invariant; `PhaseMask`/`BlazedGrating` `phase_at` inputs and
  `InvalidPhase` positions are element-frame coordinates.

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
- `PhaseMask` (`src/element.rs`, Phase 3 task d, decision D2): the DOE
  interface — analytic closure $(x, y) \mapsto \varphi$ (center-relative
  meters in, radians out, so designs are translation-invariant) painted as
  $t = e^{i\varphi}$ with a [`ClearAperture`] stop following the task-c
  lens precedent (boundary-inclusive hard edge; `Field` acts over the
  whole sampled plane and reports `Unbounded` shape/extent).
  `kind = Diffractive`; `phase_at` closure probe; new
  `OpticsError::InvalidPhase { x, y, phase }` reports the first non-finite
  closure value with its center-relative position. No sampling guard: an
  arbitrary profile admits no closed-form step bound and $O(N^2)$
  warning-sweeps are against project guidelines — caller-side Nyquist duty
  documented; structured elements self-validate analytically (grating in
  task e). Sampled-profile variant deferred to Phase 5 per D2.
- Phase-mask unit tests (6): identity mask is bit-exact; linear ramp
  reproduces `BandField::plane_wave` tilt (blaze profile, Eq. 3.5
  direction cosines); **DOE-lens closure equals `ThinLens` sample-exactly**
  (textbook Sec. 4.2.1 construction); spiral/vortex $\ell\,\theta$ mask is
  unit-modulus with wrapped phase $\ell\,\mathrm{atan2}$ everywhere;
  non-finite closure rejected with position (row-major first offender);
  taxonomy, constructor validation, oversized-footprint rejection,
  `phase_at` probe.
- `ClearAperture` (`src/element.rs`, task-d review): shared stop geometry
  for phase elements (see Decisions). Unit tests (2 added): helper
  predicates/extent/shape/validation per variant; rectangular-stop mask
  window (inclusive corners, one-past-edge zeros, area-ratio power,
  asymmetric `OutsideField` diagnostics, rectangle taxonomy with
  conservative extent).
- `BlazedGrating` (`src/element.rs`, Phase 3 task e): sawtooth blaze
  $\varphi = \Phi\,((\mathbf{r}-\mathbf{c})\cdot\hat g/\Lambda \bmod 1)$
  with period $\Lambda$, depth $\Phi$ (rad/period), and grating-vector
  **orientation** $\psi$ from $+x$ (carried now; rotated-order validation
  deferred with its tests, task-e decision 2026-10-12). Kinematic
  identity: integer-$2\pi$ depths make $e^{i\varphi}$ EXACTLY a plane
  wave tilted at $f = m/\Lambda$ (mod-1 wrap invisible), so the far field
  is the pure order $m$ at $\sin\theta_m = m\lambda/\Lambda$ (textbook
  Eqs. 1.1/2.26/5.1; Sec. 2.4.2 blaze efficiency). Constructors
  `new(period, phase_depth, orientation, aperture, center)` +
  `kinoform(...)` ($\Phi = 2\pi$) sugar; `phase_at` probe;
  `worst_phase_step(band)` = $\Phi/\Lambda\cdot\max(|\cos\psi|\Delta x,
  |\sin\psi|\Delta y)$ ramp bound (for $\Phi=2\pi,\psi=0$: period-
  Nyquist $\Delta x < \Lambda/2$). `ClearAperture` stop (task-d form),
  `kind = Diffractive`; new `OpticsError::InvalidGrating { period, depth,
  orientation }`.
- Grating unit tests (5, `src/element.rs`): sawtooth mod-1 wrap +
  periodicity incl. negative coordinates; orientation rotates the grating
  vector ($\psi=\pi/2$ steers along $+y$); integer-$2\pi$ depth equals a
  tilted plane wave sample-exactly ($m=1,2$); hard-edge stop + oversized
  footprint rejection + constructor validation + taxonomy; closed-form
  worst step = finite-difference sweep maximum.
- `tests/phase3.rs` (new integration suite, Fraunhofer-engine grating
  validation; geometry gives exactly 100 output bins per order,
  $N_F = 0.87 \le 1$ via declared `aperture_extent`): kinoform blaze
  peaks at $x_1 = \lambda z/\Lambda$ (±2 bins) with >85% power in the
  order +1 lobe and <3% leakage to orders 0/-1 (Eq. 2.26 + Sec. 2.4.2);
  $4\pi$ depth steers order +2 to $x_2 = 2\lambda z/\Lambda$ (>80%);
  $\pi$ depth splits ~40/40 between orders 0 and +1 with order -1 weak,
  matching the sawtooth Fourier coefficients $|c_m|^2 = 4/(\pi^2(1-2m)^2)$
  (Eq. 5.1 positions). Shared-helper toolbox `tests/common/mod.rs`
  marked `allow(dead_code)` (targets use subsets).
- Element **orientation** (task 3.e.1): `EllipticalAperture::new`,
  `RectangularAperture::new`, `ThinLens::new`/`fixed`, `PhaseMask::new`
  take `Option<Angle>` (`None` = typical, normal to $z$);
  `BlazedGrating::new`/`kinoform` take a mandatory `Angle`; all expose
  `orientation() -> Angle`. Rotation helpers `to_element_frame` /
  `rotated_box` + `ClearAperture::{contains_at, footprint_at,
  axis_halves_at}`; new `OpticsError::InvalidOrientation { radians }`.
  `Angle` gained implicit $[0, 2\pi)$ wrapping, `PartialEq`, and unit
  tests (2); element orientation tests (5): default/validation across
  all constructors, rotated ellipse aperture (major axis follows ψ,
  circle invariance), rotated rectangle (axis swap), rotated-stop
  `OutsideField` bounding box, lens/grating stops rotating with the
  element, rotated grating vector via `apply`.

### Changed

- **Breaking (task 3.e.1)**: element constructor signatures gained the
  orientation argument; `BlazedGrating::orientation()` returns `Angle`
  (was `f64`); `PhaseMask`/`BlazedGrating` `phase_at` take element-frame
  coordinates (mask closure and grating ramp now evaluated in the rotated
  frame; `InvalidPhase` positions likewise); `Angle` constructors and
  arithmetic now wrap (values normalized to $[0, 2\pi)$ — trig
  unaffected, stored raw angles from `Angle::new`/`from_degrees`/`Add`/
  `Sub` change representation).
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
