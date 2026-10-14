# Task List — optics-rs Forward/Inverse Suite

Ordered per `tasks/plan.md`. Phase 7 tasks are implementation-ready;
Phase 8 tasks are concrete but gated on plan Open Questions 1/5; Phases 9–10
are epics to be re-decomposed into S/M tasks at their own planning check-in
(agents.md §6). Every core-crate task follows the established discipline:
physics verified against the textbook PDF first, gates
(`cargo fmt --check && cargo clippy -q --all-targets -- -D warnings &&
cargo doc -q --no-deps && cargo test`), approval before each commit.

## Phase 7 — Core forward-physics close-out (optics-core → 0.7.0)

- [ ] **Task 7.0: Rename crate optics-rs → optics-core**
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

- [ ] **Task 7.5: Paraxial tilt/decenter on refractive surfaces (OPEN Q1)**
  Add small-angle tilt (τx, τy) to `RefractiveSurface`/`ThickLens`: sag
  acquires + τx·x′ − τy·y′ (linear term ⇒ wedge-like deflection
  (n₂−n₁)τ); decenter already exists via `center`. Acceptance: tilt τ
  reproduces wedge deflection law within paraxial tolerance; validation of
  finiteness; docs + physics.md §9.7 update. GATE: only if Open Question 1
  resolves "core" — else move to Phase 8 mapping. Deps: none. Scope: M.

- [ ] **Task 7.6: Core-complete freeze + release 0.7.0**
  Docs (physics/architecture/CHANGELOG), roadmap hook marks core
  forward-physics COMPLETE (remaining: mirrors/waveplates/oblique/vector
  stay out-of-scope-by-design), version cut per established pattern.
  Scope: S. Deps: 7.1–7.5.

### Checkpoint: Core complete (after 7.6)
- [ ] Full gates green; 0.7.0 tagged in CHANGELOG
- [ ] Public API reviewed as the stable base for dependent crates
- [ ] Review with human before Phase 8 (workspace restructure)

## Phase 8 — Zemax import (optics-zemax → 0.1.0) — GATE: Open Q1/Q5

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

## Phase 9 — Inverse design (optics-inverse) — EPIC, re-decompose at check-in

Order locked (resolved decisions 2–3): FD + derivative-free + GS first,
adjoint later, autodiff only if required; DL stack = `burn`.

- [ ] **9.0 Planning check-in** — detailed task decomposition (scope,
  golden problems, burn feature-gating).
- [ ] **9.1 Parameterized systems** — param vector → `Beamline` builder
  closure trait; design problem = (params → system, FOM, penalties).
- [ ] **9.2 Gerchberg–Saxton phase retrieval** (textbook Sec. 5.5) + DVR/FPR
  variants for DOE masks; golden hologram problems.
- [ ] **9.3 Derivative-free solvers** — random walk, simulated annealing,
  evolution strategies; golden lens-shaping problems.
- [ ] **9.4 Finite-difference gradients + first-order solvers** (Adam /
  L-BFGS wrapper).
- [ ] **9.5 Adjoint-gradient milestone** — ADR first; backprop through the
  FFT chain; 2-run gradients.
- [ ] **9.6 Deep-learning milestone** — headless dataset generation harness,
  surrogate + inverse-network training on `burn` (GPU allowed here, not in
  core).
- [ ] **9.7 Release 0.1.0** + docs.

### Checkpoint: Inverse works (end of Phase 9)
- [ ] GS + one stochastic + one gradient solver recover a known DOE design
- [ ] Review with human before Phase 10

## Phase 10 — UI (optics-studio) — EPIC, re-decompose at check-in

Web from the start (resolved decision 4).

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
