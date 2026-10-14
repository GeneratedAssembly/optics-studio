# Task List — optics-rs Forward/Inverse Suite

Ordered per `tasks/plan.md`. Phase 7 is delivered (core 0.7.0, `b3f2aa1`);
Phase 8 is delivered (`optics-zemax` 0.1.0, `bacf5db`) — its former gate on
plan Open Questions 1/5 is resolved (plan.md Resolved Decisions 1/5/6);
Phases 9–10 are epics to be re-decomposed into S/M tasks at their own planning check-in
(agents.md §6). Every core-crate task follows the established discipline:
physics verified against the textbook PDF first, gates
(`cargo fmt --check && cargo clippy -q --all-targets -- -D warnings &&
cargo doc -q --no-deps && cargo test`), approval before each commit.

## Phase 7 — Core forward-physics close-out (optics-core → 0.7.0)

- [x] **Task 7.0: Rename crate optics-rs → optics-core** (delivered `ba005d6`)
  Package/crate name change (`core` is a reserved collision with the
  built-in core crate). Mechanical: `Cargo.toml` name, `use
  optics_rs::…` → `use optics_core::…` across tests/examples/doctests,
  intra-doc links, `lib.rs` title, README/agents.md references if any;
  CHANGELOG `### Changed` note (0.x rename, no re-export shim). Repo
  directory name stays `optics-rs/`. Acceptance: `cargo test` green with
  zero `optics_rs` references in tracked sources. Verification: gates;
  `grep -r optics_rs src tests examples` empty. Deps: none. Scope: S.

- [x] **Task 7.1: Sellmeier material model**
  Concrete dispersion mechanism for `Material`: `Material::sellmeier(coeffs)`
  with n²(λ) = 1 + Σ Bᵢλ²/(λ²−Cᵢ) (λ in µm, standard form), angle argument
  ignored (normal incidence). Acceptance: golden values for N-BK7
  (n_d ≈ 1.5168 @ 0.5876 µm ± melt spread, n_F − n_C ≈ 0.0081, V_d ≈ 64.2); non-finite/out-of-band
  behavior documented; unit tests. Verification: gates green; values match
  catalog to 4 decimals. Deps: none. Files: `src/element.rs`, tests. Scope: S.

- [x] **Task 7.2: Incoherent multiband summation (D3 close-out)**
  `field::sum_intensity(&[&BandField]) -> Result<Field, FieldError>`:
  identical-grid validation, intensity (not amplitude) accumulation,
  documented coherent-addition caveat. Acceptance: doubling one band doubles
  power; two bands show no cross term; mismatched grids rejected. Verification:
  unit tests + gates. Deps: none. Files: `src/field.rs`, tests. Scope: S.

- [x] **Task 7.3: PSF analysis metrics**
  New `analysis` module on `Psf`/`Field`: encircled-energy curve EE(r),
  Strehl estimate (peak vs same-energy diffraction-limited reference or
  Maréchal from wavefront RMS when phase available), MTF via |FT(PSF)|
  normalized at origin, spot RMS radius. Acceptance: Airy anchor — EE at
  first dark ring ≈ 0.84 (±0.02), MTF of a Gaussian PSF matches analytic
  Gaussian MTF < 2%, Strehl of unaberrated vs aberrated synthetic wavefront
  orders correctly. Verification: unit + one integration test on the
  phase5 PSF. Deps: 7.2 optional. Files: `src/analysis.rs` (new), lib.rs,
  docs. Scope: M.

- [x] **Task 7.4: Zernike wavefront decomposition**
  Radial polynomials by recurrence (dependency-free, crate convention),
  least-squares fit of a phase grid to Zernikes (explicit (n,m) mode list, self-documenting) (piston/tip/tilt
  optional removal), RMS residual + per-coefficient output. Acceptance:
  golden ring tests (defocus = √8(n²−n)/… convention documented),
  recovered coefficients of synthetic Zernike sums within 1e-3, orthogonality
  on-grid. Verification: unit tests + gates. Deps: none. Files:
  `src/analysis.rs`, tests. Scope: M.

- [x] **Task 7.5: Paraxial tilt/decenter on refractive surfaces (resolved: plan.md Decision 1)**
  Add small-angle tilt (τx, τy) to `RefractiveSurface`/`ThickLens`: sag
  acquires + τx·x′ − τy·y′ (linear term ⇒ wedge-like deflection
  (n₂−n₁)τ); decenter already exists via `center`. Acceptance: tilt τ
  reproduces wedge deflection law within paraxial tolerance; validation of
  finiteness; docs + physics.md §9.7 update. Resolved by plan.md Resolved
  Decision 1 (paraxial tilt in core now); delivered `d23e000`.
  Deps: none. Scope: M.

- [x] **Task 7.6: Core-complete freeze + release 0.7.0** (delivered `b3f2aa1`)
  Docs (physics/architecture/CHANGELOG), roadmap hook marks core
  forward-physics COMPLETE (remaining: mirrors/waveplates/oblique/vector
  stay out-of-scope-by-design), version cut per established pattern.
  Scope: S. Deps: 7.1–7.5.

### Checkpoint: Core complete (after 7.6)
- [x] Full gates green; 0.7.0 tagged in CHANGELOG
- [ ] Public API reviewed as the stable base for dependent crates
- [ ] Review with human before Phase 8 (workspace restructure) — overtaken
  by events: Phase 8 shipped (`bacf5db`) and Phase 9 is already running;
  kept open for the human path.

## Phase 8 — Zemax import (optics-zemax → 0.1.0) — delivered; former Open Q1/Q5 gate resolved (plan.md Resolved Decisions 1/5/6)

- [x] **Task 8.1: Workspace restructure** — root `Cargo.toml` gains
  `[workspace] members = [".", "crates/optics-zemax"]` (exclude-default for
  ui later), `crates/` scaffold, CI/gates still green at root. Scope: S.
- [x] **Task 8.2: .zmx line parser** — header (VERSION/LENSNAME/NSURF/WAVE),
  surface rows (RADII/THIC/GLASS/CONI/PARA even-asphere terms/semi-diam),
  strict unknown-token failure with line numbers; golden-file corpus
  (hand-built + user-supplied real prescriptions). Scope: M.
- [x] **Task 8.3: Glass catalog** — Sellmeier coefficient tables
  (SCHOTT/… formats) → `Material::sellmeier`; built-in subset fallback;
  golden n_d/n_F/n_C. Scope: M. Deps: 7.1, 8.2.
- [x] **Task 8.4: Prescription → system builder** — sign/convention
  conversion table (Zemax R/thickness/semi-diam → `RefractiveSurface` +
  `Beamline` medium steps), paraxial invariants (EFL/BFL vs Eq. 4.1) as
  unit checks. Scope: M. Deps: 8.2, 8.3.
- [x] **Task 8.5: Validation suite** — known doublet/triplet prescriptions:
  imported-system BFL/PSF vs published values (tolerances documented),
  round-trip vs manual construction; release 0.1.0 + README. Scope: M.

### Checkpoint: Import works (after 8.5)
- [ ] A real .zmx imports and its PSF is computed through the core
- [ ] Review with human before Phase 9

## Phase 9 — Inverse design (optics-inverse) — decomposed at 9.0 check-in 2026-10-14

Order locked (resolved decisions 2–3, 7–8): FD + derivative-free + GS first,
adjoint before 0.1.0; custom-merit Python/DSL frontend is its own milestone
(0.2.0); DL `burn` milestone stays in Phase 9 behind a feature (0.3.0).
Reference workflow the API must serve: hypercam `coded_mask.py` —
parameterized DOE relief → differentiable ASM render (with ∂I/∂x, ∂I/∂y,
∂I/∂λ channels) → estimation-theoretic merit (Fisher/CRLB trace) →
first-order descent. CRLB itself is a USER merit (decision 7), not a built-in.

- [x] **9.0 Planning check-in** — decomposition below; resolutions recorded
  in plan.md decisions 7–8. Scope: S.

### Milestone A — framework + solvers (optics-inverse 0.1.0)

- [x] **9.1 Crate scaffold** — `crates/optics-inverse` (package
  `optics-inverse`), workspace member, path-dep `optics-core`, `rand`
  (0.8, default-features minimal); `InverseError` (thiserror) in `error.rs`;
  lib.rs module skeleton + crate docs. Acceptance: `cargo test --workspace`
  green with empty crate. Scope: S. Deps: none.

- [x] **9.2 Parameter & design-space types** — `Param { name, lo, hi, init,
  fd_step }`; `Params` bounded vector: validation (non-finite, out-of-range
  → `InvalidParams` at apply, never panics — precedent: core InvalidMedium),
  clamp vs reject policy documented, `random(&mut rng)` init, FD step
  defaults (relative to span). Pure data, no solver machinery. Scope: S.
  Deps: 9.1.

- [x] **9.3 Carriers** — `Carrier` design objects: `Analytic` (params →
  closure builds owned `Beamline`/surfaces — the zemax `plan` two-phase
  pattern reused as shape, not type) and `GridPhase` (params → sampled
  phase grid → `PhaseMask::grid`; thickness form h(x,y) → phase
  2π(n(λ)−1)h/λ via `Material`, the notebook's chromatic carrier).
  Acceptance: round-trip grid → PhaseMask applies to BandField; analytic
  closure rebuilds a beamline deterministically. Scope: M. Deps: 9.2.

- [x] **9.4 Observation + renderer** — `Observation`: flat f64 intensity
  stack over (λ-bin, field-sample) + optional derivative channels
  (∂x, ∂y via analytic derivative source fields; ∂λ via central differences)
  + grid metadata; numpy-transferable layout (documented strides).
  `render(carrier, bands, fields) -> Observation` via core ASM +
  `field::sum_intensity` (incoherent). Golden: −δ/mask/+δ defocus chain
  reproduces pure-piston collimation; Gaussian-PSF ∂x channel matches
  analytic derivative < 1e-3 rel. Scope: M. Deps: 9.3.

- [x] **9.5 Merit trait + simple built-in FOMs** — `Merit {
  evaluate(&Observation) -> f64; obs_gradient(&Observation) ->
  Option<Vec<f64>> }` (None ⇒ FD-on-obs fallback; the seam decision 7's
  frontend will implement). Built-ins: MSE-to-target, EE-in-radius,
  Strehl-vs-reference, MTF@freq, spot RMS (wrap core `analysis`).
  Acceptance: built-ins agree with core analysis on golden PSFs; trait
  object usable as `Box<dyn Merit>`. Scope: M. Deps: 9.4.

- [x] **9.6 Derivative-free solvers** — `Solver` API
  `solve(problem, budget) -> Solution { params, value, history, best }`;
  random search w/ adaptive step, simulated annealing (auto-calibrated T0,
  geometric cooling). ES evaluated and dropped (premature bound-convergence
  without CMA; adaptive-step random walk covers the role — see solve.rs
  module docs). Golden: 1-param focus flattening (tune gap, maximize
  EE-in-radius), 2-param doublet curvatures. Scope: M. Deps: 9.5.

- [x] **9.7 FD gradients + Adam** — central differences over `Params`
  (2·dim renders), Adam w/ norm-clip + best-epoch snapshot (notebook
  discipline). Golden: adjust one curvature of the imported ACA254 doublet
  to minimize spot RMS — proves zemax→inverse loop (dev-dep optics-zemax).
  Scope: M. Deps: 9.6.

- [x] **9.8 Gerchberg–Saxton + adaptive GS** — textbook §5.5; DOE phase
  mask between input/target planes, two ASM half-transforms, energy
  projections, error signal + convergence diagnostics; DVR/FPR variants as
  iterations. Golden: top-hat shaper (flat target within aperture, MSE
  monotone), GS recovers a known phase-only lens. Scope: M. Deps: 9.4
  (9.5 for stop test).

- [x] **9.9 Adjoint gradients (ADR first)** — key fact: the adjoint of ASM
  propagation is propagation by −z (conjugate transfer function), of a
  thin element is pointwise conjugate multiply, of an aperture is masked
  multiply ⇒ adjoint = forward primitives in reverse order, NO autodiff
  graph. ADR records the math: ḡ = (∂M/∂obs)ᵀ pushed back through the
  render chain → dM/dgrid in 1 forward + |channels| adjoint runs.
  Acceptance: adjoint grid gradient matches FD-on-coarse-grid < 1e-4 rel
  (analytic carrier); GS error signal sign/shape matches adjoint of MSE.
  Scope: L. Deps: 9.8.

- [ ] **9.10 Release optics-inverse 0.1.0** — docs (physics.md GS + adjoint
  sections, architecture.md crate), CHANGELOG, example
  `examples/coded_doe.rs` (GS top-hat), version cut. Scope: S.
  Deps: 9.1–9.9.

### Milestone B — custom merit frontend (optics-inverse 0.2.0)

- [ ] **9.11 Frontend ADR** — pyo3+numpy embedding (feature `py`) vs
  expression DSL compiled to closures vs both; gradient strategy for user
  merits (user-supplied / autodiff-on-Python-side / FD-on-obs). Decision
  needs user input (they write merits in Python daily). Scope: M.
  Deps: 9.10.

- [ ] **9.12 Custom-merit implementation + CRLB worked example** — user
  merits over `Observation`; first worked example = Fisher/CRLB A-optimal
  trace (Poisson + gaussian weighting; event log-ratio variant) expressed
  as a user merit, optimized on a small DOE window via 9.9 adjoint.
  Validation: cross-check forward render against a trained thickness map
  exported from hypercam `coded_mask.py` (byte-level Observation compare at
  reduced scale). Scope: L. Deps: 9.11.

- [ ] **9.13 Release 0.2.0** + docs + example. Scope: S. Deps: 9.12.

### Milestone C — deep surrogates (optics-inverse 0.3.0, feature `dl`)

- [ ] **9.14 burn harness** — headless dataset generation (params →
  observations), surrogate forward + inverse network training; GPU allowed
  here, never in core. Scope: L. Deps: 9.13.
- [ ] **9.15 Release 0.3.0** + docs. Scope: S.

### Checkpoint: Inverse works (end of Phase 9)
- [ ] GS + one stochastic + one gradient solver recover a known DOE design
- [ ] Doublet-curvature golden recovers spot-size minimum (zemax→inverse)
- [ ] One user-defined CRLB merit optimizes a DOE window
- [ ] Review with human before Phase 10

## Phase 10 — UI (optics-studio) — EPIC, re-decompose at check-in

Web from the start (resolved decision 4).

- [x] **S1-D UI/UX spec + ADR draft (pre-work)** — `docs/ux/spec.md`
  (personas, forward/backward workflows, 10.1–10.3 screen inventory with
  panel→headless-API bindings verified against `cargo doc`, rendering +
  sampling-warning constraints, gap register) and
  `docs/ux/adr-10.0-draft.md` (dioxus vs leptos against workspace
  constraints; recommendation + open questions for the check-in). Feeds
  10.0. No UI code.

- [ ] **10.0 Framework ADR** — dioxus (recommended, full-Rust wasm) vs
  leptos; record in plan.
- [ ] **10.1 Shell** — prescription view/editor (import via optics-zemax),
  parameter sliders → live Beamline preview.
- [ ] **10.2 PSF/metrics viewport** — intensity render + EE/MTF/Strehl panel
  (analysis crate APIs only).
- [ ] **10.3 Inverse run controls** — start/stop solvers, convergence plot.
- [ ] **10.4 Release 0.1.0** (`optics-studio`).

### Checkpoint: Suite complete
- [ ] Import → simulate → analyze → inverse-design round trip in the UI
- [ ] Final review

## Verification commands (all phases)

- Core: `cargo test` at workspace root (excludes ui by default-members); `-p optics-core` once workspace exists
- Per-crate: `cargo test -p optics-zemax` / `-p optics-inverse`
- UI (manual): `cargo run -p optics-studio` (web target)
