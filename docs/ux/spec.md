# UI/UX Specification — Forward + Inverse Design Suite

Status: draft for the Phase 10 check-in (pre-work, task S1-D / GEN-42).
Consumers: `tasks/todo.md` 10.0–10.3 and the `optics-studio` crate once it
exists. This document specifies screens, flows, and API bindings only — it
contains no code and no new physics. Framework choice is fixed by
[`docs/adr-10.0-ui-framework.md`](../adr-10.0-ui-framework.md) (accepted:
dioxus 0.7.10; draft comparison in [`adr-10.0-draft.md`](adr-10.0-draft.md)).

Sources of truth: `tasks/plan.md` (Phase 10, resolved decision 4),
`tasks/todo.md` (10.1–10.4), `docs/physics.md` (models + textbook citations),
`docs/architecture.md` (crate layout). Every API named below was verified
against `cargo doc -p optics-core -p optics-zemax -p optics-inverse` at this
commit. Items that do not exist yet are listed in the **gap register** (§7),
never invented.

Revision note (task 10.3, 2026-10-14): re-verified against optics-inverse
0.1.0 — `Solver::Adam` and the 9.7–9.9 surfaces landed since the draft, so
§7 G-2/G-3/G-4 are marked resolved for the UI path; `Carrier::params_mut`
was added for the chunked continuation loop. The run console itself ships
in `crates/studio` (`run/` + `bridge.rs` + `ui/run_console.rs`).

---

## 1. Personas

**P1 — Wave-optics engineer (forward workflow).** Designs or receives a
prescription (vendor `.zmx`, e.g. the Thorlabs ACA254 corpus), imports it,
simulates the PSF at the detector, and reads image-quality metrics. Cares
about: import fidelity, engine/sampling correctness warnings, per-wavelength
and sub-aperture behavior. Speaks Zemax vocabulary (curvature in 1/mm,
conic, semi-diameter, DISZ, GLAS). Primary screens: shell + PSF viewport.

**P2 — DOE designer (inverse workflow).** Defines a parameterized design
(analytic element set, sampled phase grid, or relief-thickness grid), picks
or composes a merit function, runs a solver, and reviews the recovered
design. Cares about: parameter bounds, live preview while dragging sliders,
convergence trace, run/stop control, reproducibility (seeds). Primary
screens: shell (sliders + preview) + inverse run controls + PSF viewport.

**P3 — System lead / reviewer.** Compares candidate designs side by side,
checks metrics against a spec (EE at radius, MTF at frequency, Strehl), and
signs off. Cares about: honest diagnostics display (sampling warnings,
regime validity), exportable numbers, no hidden numerical magic. Reads the
same PSF viewport; writes no parameters.

Out of persona scope (by plan): ray-tracing users, polarization users,
coating design users, binary `.ZMF`/`.zar` catalog browsers (resolved
decision 6 — import scope is text `.zmx` + `.AGF` only).

---

## 2. Core workflows

### 2.1 Forward: import → simulate → analyze

```
[ .zmx file ] --ImportedSystem::load--> [ ImportedSystem ]
      |                                        |-- paraxial_at --> ParaxialReport (EFL/BFL)
      |                                        |-- system() --> ZmxSystem (surface table)
      v
[ BandField source ]  sources::spherical / BandField::plane_wave / sources::gaussian
      v
[ trace ]  ImportedSystem::psf | psf_subaperture   (engine closure: asm::propagate …)
      v
[ Psf ] --analysis--> EncircledEnergy, Mtf, strehl, spot_rms_radius, zernike_fit
      v
[ viewport ]  intensity render + metric panels + sampling banner
```

Steps and their backing APIs:

1. **Import.** `optics_zemax::ImportedSystem::load(path)` / `parse(text)`;
   `load_with(path, registry)` for `.AGF` catalogs
   (`optics_zemax::AgfCatalog::parse`, `Registry::builtin().with_agf(..)`).
   Rejections surface as `optics_zemax::ZemaxError`
   (`UnsupportedMode`, `UnsupportedSurface`, `BadField`, `Io`) — the UI shows
   the `Display` text verbatim in an import dialog; it must not swallow it.
2. **Prescription review.** `ImportedSystem::system() -> &ZmxSystem`
   exposes `name`, `catalog`, `enpd_mm`, `wavelengths_um`, `primary`, and
   `surfaces: Vec<ZmxSurface>` (all fields public: `index`, `kind`, `stop`,
   `comment`, `curvature_mm`, `conic`, `has_para`, `mirror`, `semi_diam_mm`,
   `thickness_mm`, `glass`). Paraxial sanity: `paraxial_at(lambda_um) ->
   ParaxialReport { efl_mm, bfl_mm, lambda_um }`.
3. **Source setup.** Per selected band: `optics_core::sources::spherical`
   (point source → PSF), `optics_core::field::BandField::plane_wave`
   (off-axis field point), or `optics_core::sources::gaussian` (beam
   input). Wavelengths come from `ZmxSystem::wavelengths_um` (µm — the UI
   converts to meters for core APIs; `optics_core::wave::Wavelength`
   round-trips nm).
4. **Simulate.** `ImportedSystem::psf(band, engine)` or
   `psf_subaperture(band, semi_diam_mm, engine)`; engine closure supplied by
   the engine picker (§3.4). Multi-wavelength incoherent sum: loop bands,
   `optics_core::field::sum_intensity(&[&BandField]) -> Psf`.
5. **Analyze.** `optics_core::analysis::{encircled_energy, mtf, strehl,
   spot_rms_radius, zernike_fit, marechal_strehl}` on the resulting `Psf`.
6. **Diagnostics.** Every propagation step can emit sampling reports
   (§5.4); the viewport shows them whenever `nyquist_ok`/`regime_ok` is
   false or an `AliasingRisk`/`RegimeViolation` error variant was returned.

### 2.2 Backward: define carrier → pick merit → run solver → review

```
[ Carrier ]  analytic(params, build) | grid_phase | grid_thickness
      |--> params(): &Params  (Param { name, lo, hi, value, fd_step })
      v
[ Renderer ]  new(carrier, source) ; pub fields: bands, scenes, asm,
              derivatives, scene_step, lambda_step
      v  render(values) --> Observation (flat f64 stacks, channels/bands/scenes)
[ Merit ]  MseToTarget | EncircledEnergyMerit | SpotRmsMerit | StrehlMerit |
           MtfMerit   (Selector: channel × band × scene)
      v  objective(&renderer, &merit) --> closure
[ solve ]  optics_inverse::solve(params, objective, solver, cfg, random_init)
      v
[ Solution { best, value, history, evaluations } ]
      v
[ review ]  carrier.build(best, λ) --> Design --> Beamline --> psf --> §2.1 viewport
```

Steps and their backing APIs:

1. **Define carrier.** `optics_inverse::Carrier::analytic(params, build)`
   with `build: Fn(&[f64], f64) -> Result<Design>` (`AnalyticBuilder`), or
   `Carrier::grid_phase(dims, pitch, aperture, center, lo, hi, init)` /
   `Carrier::grid_thickness(..., t_max, init, material)` taking
   `optics_core::element::ClearAperture` and `Material`. Design space reads
   back through `Carrier::params() -> &Params`, `dim()`, `slot_bounds()`.
2. **Pick merit.** Built-ins `MseToTarget`, `EncircledEnergyMerit { radius,
   selector }`, `SpotRmsMerit`, `StrehlMerit { reference, selector }`,
   `MtfMerit { freq, selector }` (all implement the `Merit` trait; lower is
   better). Frame selection via `Selector::intensity(band, scene)` or
   `Selector { channel, band, scene }`. Custom user merits (Python/DSL,
   CRLB class) are **not yet available** — gap G-5.
3. **Configure and run.** `StochasticConfig { budget, seed, initial_scale }`
   (default budget 2000, seed `Some(0)`), `Solver::RandomWalk { stagnation }`,
   `Solver::Annealing { t0, t_end_frac }`, or `Solver::Adam(AdamConfig)`
   (epoch budget). Renderer observation channels
   selectable via `Derivatives { x, y, lambda }` and `Channel::{Intensity,
   DxDyX, DxDyY, Dlambda}`. Run control (start/stop) is constrained by gap
   G-2 — the UI runs `solve` in chunks on a worker.
4. **Review.** `Solution.history` drives the convergence plot; `Solution.best`
   is applied via `Params::set_values` and previewed with
   `Carrier::build(values, lambda) -> Design` → `Design::beamline()` →
   `Beamline::psf(band, engine)`; `Solution.value`/`evaluations` reported.

---

## 3. Screen inventory and interaction flows (10.1–10.3)

Layout: single-window web app, three regions — left **prescription/editor
column**, center **viewport** (PSF + metric tabs), right/bottom **run
console** (inverse controls + convergence). One screen set serves both
workflows; the mode (forward/inverse) selects which panels are active.

### 3.1 Application shell (10.1)

| Panel | Content | Backing API |
|---|---|---|
| File panel | Import `.zmx`; load `.AGF` glass catalogs; recent files | `ImportedSystem::load` / `load_with`, `AgfCatalog::parse`, `Registry::builtin().with_agf(..)` |
| Prescription table | Surface list: index, type, stop, curvature (1/mm), conic, semi-diam (mm), thickness (mm), glass, comment | `ImportedSystem::system() -> &ZmxSystem`, `ZmxSurface` fields |
| Paraxial card | EFL / BFL at chosen band; file's own image distance as cross-check | `ImportedSystem::paraxial_at(lambda_um) -> ParaxialReport` |
| Wavelength selector | Checkboxes over `WAVM`, radio for `PWAV` primary | `ZmxSystem::wavelengths_um`, `primary`, `primary_um()`; `optics_core::wave::Wavelength` |
| Source picker | Point source / tilted plane wave / Gaussian beam, with grid spec (nx, ny, dx, dy) | `sources::spherical`, `BandField::plane_wave`, `sources::gaussian` + `GaussianBeam { w0, amplitude, center }` |
| Engine picker | Asm / Fresnel transfer / Fresnel convolution / Fraunhofer / Rayleigh–Sommerfeld + config knobs (padding, band_limited, strict, max_padding, medium) | `propagation::regime::Engine`, `AsmConfig`, `FresnelConfig`, `FraunhoferConfig`, `RayleighSommerfeldConfig`; auto-suggest via `regime::regime_report_with(band, z, aperture).recommended` |
| Beamline preview | Schematic strip of the element stack (apertures, surfaces, gaps, media) | `Beamline::len` + per-element `Element::shape/kind/material/extent`; for designs: `Design::beamline()` |

**Flow (import → live preview):** pick file → `load` → table + paraxial card
populate → choose band/source/engine → `psf` render into viewport →
diagnostics banner updates. Errors in the chain render as dismissible
toasts carrying the typed error's `Display` text.

### 3.2 Prescription editor + parameter sliders → live preview (10.1)

Two editing modes, both bound to real APIs:

- **Text mode (authoritative).** The `.zmx` source is shown in a code
  editor; edits validate by `ImportedSystem::parse(text)` on debounce
  (~250 ms). Parse errors display line-level `ZemaxError` text. This is the
  only path that can produce a *truly* re-importable prescription — see gap
  G-1 (no serializer, so table edits round-trip through text, not objects).
- **Table/parameter mode (live simulation only).** The UI keeps its own
  `Vec<ZmxSurface>` (clone of `system().surfaces`) and edits numeric fields
  (`curvature_mm`, `conic`, `semi_diam_mm`, `thickness_mm`, `glass` via a
  catalog combobox backed by `optics_zemax::catalog::BUILTIN` /
  `Registry::material(name)`). Edited surfaces rebuild a simulation beamline
  directly with core types: `RefractiveSurface::new` with
  `SurfaceProfile::Conic { radius, conic }` (note: `CURV` is 1/mm — the UI
  converts to meters-radius), `ClearAperture::Circle(semi_diam)`, gaps via
  `Beamline::gap(z, medium)` / `push_medium`, then `Beamline::psf(band,
  engine)`. The imported `ImportedSystem` itself is immutable (gap G-1);
  the UI marks the session "edited, re-parse to persist".

**Sliders.** For inverse designs, the parameter list is `Carrier::params()
-> &Params` → `Param { name, lo, hi, value, fd_step }`: one slider per slot
clamped to `[lo, hi]`, with `Params::set_values` on change and
`Params::validate_values` guarding programmatic writes. Grid carriers
(`grid_phase`, `grid_thickness`) expose thousands of slots — the UI shows a
**grid editor canvas** (paint/inspect cells, bounds from `slot_bounds()`)
rather than sliders, and always keeps a "global" scalar summary (mean,
min, max) computed client-side from `values()`.

**Live Beamline preview.** Each slider commit (throttled to ≥100 ms, or on
release) triggers `Carrier::build(values, lambda) -> Design` →
`Design::beamline()` → `Beamline::psf(band, engine)` and re-renders the
viewport (§3.3). Preview cost is bounded by the current grid size; the UI
offers a "preview grid" (coarse) vs "analysis grid" (full) band, both via
`BandField::uniform`/`sources::spherical` grid arguments.

### 3.3 PSF / metrics viewport (10.2)

| Panel | Content | Backing API |
|---|---|---|
| Intensity render | 2D image of the detector-plane PSF, colormap + scale controls | `Psf::samples` (`&[f64]`, row-major), `dims`, `pitch`, `coord`, `total_power`, `peak_bin`, `peak_position`, `centroid` |
| Cross-sections | Horizontal/vertical cuts through the peak or cursor position, plotted as line charts | client slicing of `Psf::samples` with `coord(ix, iy)` mapping |
| EE panel | Encircled-energy curve EE(r) + readouts `value_at(r)` / `radius_for_fraction(f)` (84% Airy radius, 50% median) | `analysis::encircled_energy(&Psf) -> EncircledEnergy`, `curve() -> (&[f64], &[f64])` |
| MTF panel | 1-D MTF along +fx, freqs to Nyquist, log-x option | `analysis::mtf(&Psf) -> Mtf { freqs, values }` |
| Strehl panel | Ratio vs a reference PSF; Maréchal estimate from Zernike residual | `analysis::strehl(measured, reference)`, `analysis::marechal_strehl(rms_waves)`, `analysis::zernike_fit(phase, nx, ny, dx, dy, pupil, n_max) -> WavefrontDecomp` + `residual_rms_waves()` |
| Spot panel | RMS spot radius, centroid excursion | `analysis::spot_rms_radius(&Psf)`, `Psf::centroid` |
| Wavefront panel | Zernike coefficient bars (modes up to `n_max` via `zernike_modes`), residual map | `analysis::zernike_fit`, `ZernikeMode { n, m }`, `WavefrontDecomp { modes, coefficients, residual_rms }` |
| Sampling banner | Diagnostics per last run (§5.4) | `SamplingReport`, `FresnelSamplingReport`, `RayleighSommerfeldSamplingReport`, `FraunhoferReport`, `RegimeReport`, error variants |
| Multi-band view | Per-band PSF tabs + incoherent sum view | loop `ImportedSystem::psf` per band; `field::sum_intensity` for the summed `Psf` |

**Strehl reference PSF.** No built-in reference generator exists (gap
G-6). Documented recipe the UI implements: build a diffraction-limited
stand-in with the same aperture and focal geometry —
`EllipticalAperture::circle` + `ThinLens::new` (or `fixed`) traced with the
identical engine/grid from `sources::spherical` — and pass it as
`reference` to `analysis::strehl`. The panel labels this "reference is
synthetic, same grid/engine".

**Interaction.** Cursor over the intensity image drives the cross-section
chart and a readout (`coord`, sample value, and log-scaled value). EE/MTF
panels share a "radius/frequency" scrubber so a design comparison keeps one
operating point. Metric panels are read-only; all mutation happens in the
shell/editor.

### 3.4 Inverse run controls (10.3)

| Control | Behavior | Backing API |
|---|---|---|
| Solver picker | Random walk (stagnation knob) / annealing (t0, t_end_frac) / Adam (lr_frac, betas, eps, clip; budget counts epochs) | `optics_inverse::Solver` variants |
| Budget/seed/scale | Numeric fields with defaults from `StochasticConfig::default()` | `StochasticConfig { budget, seed, initial_scale }` |
| Random-init toggle | Start from a uniform draw in bounds | `solve(..., random_init)` |
| Start / Stop | Chunked execution: repeated `solve` calls with `budget = chunk`, `random_init = false`, after `Params::set_values(current_best)`; Stop = do not launch the next chunk | `solve`, `Params::set_values`, `Solution::best` |
| Convergence plot | Merit-after-each-accepted-evaluation trace, log-y | `Solution::history`, `evaluations`, `value` |
| Progress readout | Evaluations consumed so far, chunk count, elapsed | client accounting (no server push — gap G-2) |
| Apply result | Load `best` into the sliders/grid editor and re-preview | `Params::set_values` → §3.2 live preview |
| Observation config | Bands list, scenes list (field points), derivative channels | `Renderer` pub fields `bands`, `scenes`, `derivatives: Derivatives { x, y, lambda }`, `scene_step`, `lambda_step`, `asm` |
| Merit editor | Pick built-in + parameters (radius / freq / reference frame / target frame) and `Selector` (channel × band × scene) | `EncircledEnergyMerit`, `MtfMerit`, `SpotRmsMerit`, `StrehlMerit`, `MseToTarget`, `Selector` |

**Flow (run → review):** configure carrier/merit/solver → Start → worker
loop runs chunks, appends `Solution.history` segments to the convergence
plot, keeps the latest `best` in a side buffer → Stop (or budget exhausted)
→ "Apply best" → viewport re-renders the recovered design → forward metrics
(§3.3) for sign-off. Because `solve` is synchronous with no progress
callback (gap G-2), chunk granularity is the UI's responsiveness knob;
chunk size must be ≥ ~50 evaluations to avoid overhead spam.

**Reproducibility.** Run card records: crate versions, solver + config
fields verbatim, seed, merit name (`Merit::name`), band/scene lists, and
final `value`/`evaluations`. `seed: Some(0)` default is surfaced so users
learn it is deterministic by default. On the wasm target the UI must always
pass `seed: Some(..)`: with `seed: None`, `solve` draws from
`rand::thread_rng()` (rand 0.8), which is not available on
wasm32-unknown-unknown and would fail the run.

### 3.5 Global interaction rules

- All numeric inputs show SI-internal values with engineering display units
  (mm/µm/deg) and explicit unit labels; conversions happen in the UI layer
  only (µm→m for core APIs; 1/mm→1/m curvature; degrees↔`Angle`).
- Long operations (import of large files, full-grid renders, solver chunks)
  run off the main thread; the viewport keeps the last good result visible
  with a "stale" badge rather than blanking.
- Errors are rendered from the typed enums' `Display` (`ZemaxError`,
  `OpticsError`, `FieldError`, `PropagationError`, `FresnelError`,
  `FraunhoferError`, `RayleighSommerfeldError`, `BeamlineError`,
  `AnalysisError`, `InverseError`) plus their carried diagnostics; no
  string-only error path.
- No physics formulas are evaluated in the UI crate beyond unit conversion
  and display normalization (plan constraint; see §6).

---

## 4. Rendering constraints (f64 grids → pixels)

- **Source data.** Everything on screen is an `f64` row-major grid:
  `Psf::samples()` (non-negative intensity), `Field::intensities()` /
  `amplitudes()` / `phases()` (complex-derived), `Observation.data` (flat
  channel stacks, `slice(ci, bi, si)`), `SampledPhase::values()` /
  `PhaseMask::phase_at` (radians), `SurfaceProfile::sag` (meters).
- **Colormaps.** Intensity-like: viridis or inferno (perceptually uniform,
  colorblind-safe), single-hue for print. Phase maps: cyclic colormap
  (hue wheel) because radians wrap at ±π; never a sequential ramp on phase.
  SIGNED data (cross-sections about a baseline, `∂I/∂x₀` derivative channels,
  Zernike residuals): diverging colormap centered at 0 (RdBu).
- **Normalization.** Default autoscale per frame to `[min, max]` of the
  *current* view; a "lock scale" mode freezes the mapping so two designs
  compare honestly. `Psf::total_power` and `Field::power` are shown next to
  any locked view so power drift is visible. NaN/inf samples must never
  reach the canvas: core types reject non-finite grids at construction
  (`FieldError::NonFiniteSample`, `OpticsError`), but the UI still asserts
  finiteness before mapping and shows a "non-finite sample" error badge if
  one ever appears.
- **Log scale.** Intensity toggle: `I_display = log10(max(I, ε)/max(I))`
  with ε a per-frame floor (default 1e-6 of peak) — needed to see PSF wings
  and grating order background (textbook Fig. 4.17: zero-order leakage
  shows as a broad pedestal invisible on linear scale). Log applies to the
  MTF x-axis option (cycles/m spans decades) and the convergence plot
  (merit value).
- **Cross-sections.** Peak row/column by default (`Psf::peak_bin`), cursor
  row/column on hover; plotted in physical coordinates from `Psf::coord(ix,
  iy)` (µm display for detector grids, m for field planes).
- **Downsampling.** Display never renders more samples than device pixels:
  box-average the grid to ≤2048² before mapping (analysis metrics always
  run on the full-resolution `Psf`, not the display copy). Grids larger than
  the viewport's budget render with an explicit "display downsampled ×k"
  note.
- **Line charts** (EE, MTF, cross-sections, convergence): sampled arrays
  are small (`curve()`, `Mtf.freqs/values`) — render as SVG/canvas polylines,
  no resampling; hover readouts interpolate linearly only for display,
  matching `EncircledEnergy::value_at` semantics (which itself linearly
  interpolates the curve).

---

## 5. Diagnostics display (sampling warnings per core diagnostics)

The core mitigation policy is *prevention first* (agents.md §4): engines
auto-densify in `strict` mode and only then fail. The UI must show what the
engine actually did, not just whether it succeeded.

### 5.1 Per-run report capture

- The engine picker's "run diagnostics" mode calls, before tracing:
  `asm::sampling_report(band, z, cfg)`, `fresnel::sampling_report`,
  `rayleigh_sommerfeld::sampling_report`, `fraunhofer::regime_report`, and
  `regime::regime_report_with(band, z, aperture)` for the recommendation
  line. These are pure diagnostics — nothing executes a propagation.
- After a trace, if it failed, the error variant itself carries the report:
  `PropagationError::AliasingRisk`, `FresnelError::AliasingRisk`,
  `RayleighSommerfeldError::AliasingRisk`, `FraunhoferError::RegimeViolation`
  — the banner renders the embedded report plus the variant's remedy hint
  (the `Display` impls already append "hint: … remedy: …" text; the UI
  shows it, and additionally offers one-click fixes listed below).

### 5.2 Banner rules

- **Green:** `SamplingReport.nyquist_ok == true` (and Fresnel/RS
  equivalents; `FraunhoferReport.regime_ok`, `RegimeReport.paraxial` where
  applicable). Show a compact line: grid, pitch, padding used, `max_phase_step_x/y`.
- **Yellow (mitigated):** `used_padding > requested_padding` — the engine
  densified automatically. Show both values and the final `padded_dims`;
  this is the visible cost of prevention and must never be hidden.
- **Red (rejected):** an `AliasingRisk`/`RegimeViolation` was returned —
  show the full report `Display` text, the hint, and one-click remedies that
  map to real knobs: raise `AsmConfig::max_padding` / `padding`, coarsen
  grid pitch, reduce `|z|`, or switch engine per `RegimeReport::recommended`
  (textbook Sec. 2.2 rule `N > D²/(Zλ)` cited in `docs/physics.md`).
- **Regime advisory:** when Fraunhofer is selected but
  `RegimeReport.far_field == false` (N_F > 1), or Fresnel engines are used
  with `paraxial == false` (θ > 0.1 rad), show a non-blocking advisory even
  if the run succeeded with `strict = false`.

### 5.3 Element-level warnings

`ThinLens::worst_phase_step`, `BlazedGrating::worst_phase_step`,
`RefractiveSurface::worst_phase_step`, `ThickLens::worst_phase_step`,
`Wedge::worst_phase_step` (each takes `&BandField`) feed the prescription
table a per-surface shading column: surfaces whose phase step between
adjacent samples approaches π are flagged before any trace is run.

### 5.4 Inverse-run diagnostics

`Renderer::render` failures surface as `InverseError::Render { reason }`
(engine text forwarded) — the run console shows it and pauses the chunk
loop rather than aborting the whole session. `Params::validate` /
`validate_values` errors (out-of-bounds writes) are shown inline on the
slider row.

---

## 6. Non-goals and constraints

- **No physics in the UI crate.** `optics-studio` may only call public APIs
  of `optics-core`, `optics-zemax`, `optics-inverse`; no propagation math,
  no metric formulas, no sampling criteria re-derived client-side (plan
  risk table: "UI scope creep into physics").
- **Opt-in workspace member.** `optics-studio` stays out of
  `workspace.default-members`; CI (`cargo test`, gates) never builds it
  (plan: architecture decisions). wasm build is a separate, explicitly
  invoked target.
- **Out of scope:** ray tracing / geometric optics (agents.md §1),
  polarization, coatings, non-sequential mode, coordinate breaks, binary
  `.ZMF`/`.zar` inputs (resolved decision 6), GPU compute in the browser.
- **f64 everywhere** (approved decision 2026-10-11); the UI may downsample
  for display but analysis always uses full-resolution core types.

---

## 7. Gap register (flag, do not invent)

| ID | Gap | Affected screens | Suggested resolution (not committed) |
|---|---|---|---|
| G-1 | No `.zmx` serializer and no constructor from an edited `ZmxSystem` (`ImportedSystem` fields private; only `parse`/`load` from text) | 10.1 editor persistence | Round-trip edits through the text editor (works today); a writer task if table-mode persistence is wanted |
| G-2 | `solve()` is synchronous — no progress callback, cancellation token, or streaming history | 10.3 start/stop | **Resolved for 10.3** by the chunked budget loop in `crates/studio/src/run/session.rs` (spec §3.4): `RunSession` runs one budget slice per `solve` call and carries `best` through `Carrier::params_mut` (new accessor). A native `solve_with_progress`/cancel API in `optics-inverse` would still replace the workaround. |
| G-3 | Solver zoo is partial: only `Solver::RandomWalk` + `Annealing` exist. Gerchberg–Saxton (todo 9.8), FD-Adam (9.7), adjoint (9.9) are not in the public surface yet | 10.3 solver picker | **Resolved:** `Solver::Adam` (9.7) is now in the public surface and the picker ships Random walk / Annealing / Adam. GS (9.8) and adjoint (9.9) reach the UI in a later task (they are grid-domain, not `Params`-domain, so they do not fit the `solve` chunk loop). |
| G-4 | No gradient-based solver entry point in `solve` (merit `obs_gradient` seam exists, unused by solvers) | 10.3 | **Resolved for the UI path:** `Solver::Adam` drives FD gradients through `solve`; merit `obs_gradient`/adjoint (`grid_gradient`, 9.9) remain a headless seam for a future grid solver. |
| G-5 | No custom-merit frontend (Python/DSL, CRLB-class) — planned `optics-inverse` 0.2.0 (todo 9.11/9.12) | 10.3 merit editor | Built-in merits + `Box<dyn Merit>` in-process only for now |
| G-6 | No reference-PSF generator API for Strehl (must be hand-composed per §3.3) | 10.2 Strehl panel | Candidate core helper `analysis::reference_psf(...)`; file as Phase-10+ request |
| G-7 | `GridSpec` fields are private — grid geometry cannot be read back from a `Carrier` | 10.1 grid editor | UI caches construction args; accessor on `Carrier`/`GridSpec` if state must survive reload |
| G-8 | No engine-agnostic trace helper: `Beamline::trace` takes a caller closure, so the UI assembles the `(band, z, medium)` → engine dispatch itself | 10.1/10.2 | Thin adapter in the UI crate (allowed: plumbing, not physics) |
| G-9 | No serialization format for a saved "studio session" (design + renderer + merit + config) | all | Out of scope for 10.1–10.3; note for 10.4 |

---

## 8. Traceability (acceptance check)

| Epic feature | Flow section | Named backing APIs |
|---|---|---|
| 10.1 prescription view/editor | §3.1, §3.2 | `ImportedSystem::{load, load_with, parse, system, paraxial_at, psf, psf_subaperture}`, `ZmxSystem`, `ZmxSurface`, `Registry`, `AgfCatalog`, `RefractiveSurface`, `SurfaceProfile`, `Beamline` |
| 10.1 parameter sliders → live Beamline preview | §3.2 | `Carrier::{analytic, grid_phase, grid_thickness, params, build}`, `Params`, `Param`, `Design`, `Beamline::{push_medium, gap, psf}`, `asm::propagate` |
| 10.2 intensity render | §3.3, §4 | `Psf::{samples, dims, pitch, coord, peak_bin, centroid, total_power}`, `sum_intensity` |
| 10.2 EE/MTF/Strehl panels | §3.3, §5 | `analysis::{encircled_energy, mtf, strehl, marechal_strehl, spot_rms_radius, zernike_fit, zernike_modes}` |
| 10.2 sampling-warning display | §5 | `asm::sampling_report`, `fresnel::sampling_report`, `rayleigh_sommerfeld::sampling_report`, `fraunhofer::regime_report`, `regime::{regime_report_with, Engine}`, error variants |
| 10.3 start/stop + convergence plot | §3.4 | `solve`, `Solver`, `StochasticConfig`, `Solution::history`, `objective`, `Renderer`, `Merit` built-ins, `Selector` |

Every table entry above resolves to a rustdoc page in the current
`cargo doc` output of the three headless crates; gaps are §7, not invented
APIs.
