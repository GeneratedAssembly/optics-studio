# Architecture — `optics-core`

Module map, data flow, trait specs, and error taxonomy for the Diffractive
Optics Engineering Suite (Phases 1–6: field representation, ASM/Fresnel/
Fraunhofer/Rayleigh–Sommerfeld propagation (ASM in vacuum or media) +
regime guidance, thin element library incl. refractive surfaces, thick
lenses, and wedges, `sources` initializers incl. HG/LG modes and the
point source, and `beamline` PSF composition).
The physics rationale lives in [`physics.md`](physics.md); development
protocol in [`../agents.md`](../agents.md). Section 7 documents the
`optics-inverse` workspace crate (Phase 9 Milestone A, released 0.1.0).

## 1. Dependency direction

```
                     common  (Angle, J)
                       ↑
        wave ─────────┤          (scalar source/beam records)
                      │
        field ────────┼          (pure data grids: Field, BandField)
                      │
        element ──────┼──→ field (local transformations of BandField)
                      │
        sources ──────┼──→ field (closed-form initializers → BandField)
                      │
        propagation ──┼──→ field (algorithms over BandField)
            ├── fft        (private layer: rustfft wrapper, conventions)
            ├── asm        (exact Angular Spectrum; vacuum or medium n)
            ├── fresnel    (paraxial: transfer function + convolution)
            ├── fraunhofer (far-field FT, regime-gated)
            ├── rayleigh_sommerfeld (exact RS-I spherical kernel, convolution)
            └── regime     (diagnostics: N_F, guards, engine recommendation)
                      │
        beamline ─────┴──→ element + propagation + field
                         (composition sugar: (element, z) steps, trace, PSF)
```

Hard rule (agents.md §4): **algorithms depend on data, never the reverse.**
`propagation` depends on `field`; `field` knows nothing about propagation.
Propagators are **free functions**, not methods; composition sugar lives in
`beamline` (Phase 5) — the loop a caller would write by hand, with the
engine supplied as a closure so per-engine configs and errors stay open.

## 2. Modules

| Module | Role | Public surface |
|---|---|---|
| `common` | Angle newtype (radians, derefs to f64), `J` imaginary unit | `Angle`, `J` |
| `wave` | Scalar source/beam parameter records (pre-sampling) | `Wavelength`, `Polarization`, `Wave` |
| `field` | Complex sample grids + real intensity grids; wavelength decoupled from geometry | `Field`, `BandField`, `Psf`, `sum_intensity`, `FieldError` |
| `element` | Local field transformations; element taxonomy | `Element` trait, `EllipticalAperture`, `RectangularAperture`, `ThinLens`, `PhaseMask`, `SampledPhase`, `BlazedGrating`, `RefractiveSurface`, `SurfaceProfile`, `ThickLens`, `Wedge`, `ClearAperture`, `OpticsError`, `ElementShape`, `ElementKind`, `Extent`, `Material` |
| `sources` | Closed-form field initializers → `BandField` | `GaussianBeam`, `gaussian`, `HermiteGaussBeam`, `hermite_gauss`, `LaguerreGaussBeam`, `laguerre_gauss`, `spherical`, `MAX_MODE_ORDER`, `SourcesError` |
| `propagation` | Wave-optics engines + regime diagnostics | `asm`, `fresnel`, `fraunhofer`, `rayleigh_sommerfeld`, `regime` (public); `fft` (private) |
| `beamline` | Composition sugar over the above | `Beamline`, `Psf`, `BeamlineError` |

### 2.1 `field` — the two-layer split

- **`Field`**: pure spatial grid — samples, `nx`/`ny`, pitch `dx`/`dy`,
  origin-centered coordinates, intensity/power bookkeeping. No wavelength.
- **`BandField { lambda, field }`**: the **unit of propagation and element
  interaction**. Phase effects (tilt, transfer functions, thin-lens masks) are
  wavelength-dependent, so λ binds here.

Rationale (approved decision 2026-10-11): future *incoherent multiband*
sources are collections of `BandField`s sharing grid geometry; their
intensities (not amplitudes) sum at detection. Coherent addition is valid only
between equal-λ bands on identical grids.

Key invariants enforced at construction (`FieldError`): nonempty grid, buffer
length `nx*ny`, positive finite pitch, positive finite λ, finite samples.

### 2.2 `element` — local interactions

`Element::apply(&BandField) -> Result<BandField, OpticsError>` multiplies a
transmission function onto the sampled plane; grid dimensions, pitch, and
wavelength are preserved. An element never resamples or propagates — the
Fresnel/Fraunhofer regime of the downstream pattern is judged by the Fresnel
number $N_F = a^2/(\lambda z)$ at propagation time (see `physics.md` §9), so
elements expose footprint half-extents (concrete accessors plus the
conservative `Element::extent`) for regime guidance. `OpticsError`
validates geometry in **footprint half-extents** shared by all shapes.

**Uniform orientation (task 3.e.1).** Every concrete element stores an
effective `Angle` ψ (CCW about $z$ viewed from downstream; module docs) and
evaluates pattern *and* stop in the element frame
$\mathbf{r}' = R(-\psi)(\mathbf{r} - \mathbf{c})$ — shared helpers
`to_element_frame` / `rotated_box` plus `ClearAperture::{contains_at,
footprint_at, axis_halves_at}`. Constructors take `Option<Angle>` where a
typical orientation exists (apertures, lens, mask — `None` = normal to $z$,
axes aligned) and a mandatory `Angle` for the `BlazedGrating` vector;
finiteness is validated via `resolve_orientation` (`InvalidOrientation`).
`Angle` wraps implicitly into $[0, 2\pi)$ and derives `PartialEq`;
non-finite wraps to NaN and is rejected by constructors. Mirrors (Phase 6)
must adopt the same pattern. Rotated non-circular stops report bounding-box
half-extents in `OutsideField`.

Concrete elements (physics: `physics.md` §9.1–9.4, §9.7):

- `EllipticalAperture` — hard-edge binary mask, semi-axes $(a, b)$, with the
  **circle as the axis-symmetric special case** (`circle()`,
  `radius() -> Option<f64>`; task-b decision 2026-10-12 superseded the former
  `CircularAperture`), and `RectangularAperture` — half-widths $(a, b)$.
- `ThinLens` — quadratic phase screen with **dispersive focal closure**
  $f(\lambda)$ (decision D1) inside a `ClearAperture` stop (task-c revision;
  `ClearAperture::Field` = field-wide special case reporting
  `Unbounded` shape/extent); `new(f, orientation, aperture, center)` +
  `fixed(f0, ...)` sugar; `focal_at(lambda)`; `worst_phase_step(band)`
  closed-form $O(1)$ sampling diagnostic (reported, not enforced).
- `PhaseMask` — the DOE interface (decision D2): analytic closure
  $(x', y') \mapsto \varphi$ in **center-relative element-frame**
  coordinates (translation-invariant designs), stop = `ClearAperture`
  (task-d shared enum: `Field | Circle | Ellipse | Rectangle` —
  "we can't square a circle"); `kind = Diffractive`; first non-finite
  closure value raises `InvalidPhase { x, y, phase }`; no sampling guard
  (caller-side Nyquist duty documented). **Sampled variant** (D2, Phase 5
  task b): `PhaseMask::sampled(values, dims, pitch, orientation, aperture,
  center)` — bilinear lookup on a center-origin element-frame lattice,
  zero phase beyond the lattice span, $O(N)$ construction-time validation
  (`InvalidGrid`, `InvalidGridPitch`, `InvalidPhase` at the lattice
  position); `grid() -> Option<&SampledPhase>` exposes the lattice.
  `phase_at` probes take element-frame coordinates.
- `BlazedGrating` — sawtooth phase $\varphi = \Phi(((\mathbf{r}-\mathbf{c})
  \cdot\hat g/\Lambda) \bmod 1)$ with period $\Lambda$, depth $\Phi$,
  mandatory vector orientation ψ, `ClearAperture` stop; `kinoform()` sugar
  ($\Phi = 2\pi$); `worst_phase_step` ramp bound for period-Nyquist checks.
- `RefractiveSurface` — the Zemax-import primitive (Phase 6 task a; paraxial tilt added task 7.5): curved
  boundary between `Material` front/back media as a local screen
  $\varphi = k(n_1 - n_2)z(r)$ with `SurfaceProfile` sag —
  `Spherical`/`Conic`/`EvenAsphere` (Zemax standard-surface rational form
  + even polynomial) or `Analytic` closure; `sag`, `vertex_power(lambda)`,
  `worst_phase_step`; construction rejects conic domains breached by the
  stop (`InvalidSurface`), runtime rejects non-finite sags (`InvalidSag`)
  and bad media (`InvalidMedium`).
- `ThickLens` — single-plane convenience screen $\varphi = k(n-1)t(\rho)$
  (physics.md §9.7): vertex piston Eq. 4.17, thin-lens law + exact-profile
  spherical aberration; `thickness_profile`, `power_at`/`focal_at`,
  `Extent::Thick { aperture, thickness }` (first consumer); the
  Zemax-faithful thick lens is composed as surface → medium `gap` →
  surface (reproduces Eq. 4.1 incl. the thickness term).
- `Wedge` — prism screen $t = t_0 - \tan\alpha\,(\mathbf{r}'\cdot\hat g)$
  with mandatory apex-direction `Angle` (grating pattern); deflects to
  $\sin\theta = -(n-1)\tan\alpha$ toward the base; `ElementKind::Prism`.

`shape`/`kind`/`material`/`extent` accessors are taxonomy metadata;
`material` is unused by the hard-edge/phase models (index-1 placeholders,
reserved for refractive/absorptive and thick models in Phase 6).

### 2.3 `sources` — field initializers

Closed-form paraxial solutions sampled onto a grid and bound to a
wavelength, returning `BandField`s (decision D3: top-level module,
multiband by construction — a source is a collection of `BandField`s).
All records are SI meters with **no longitudinal position** — the signed
waist-to-plane (or source-to-plane) distance is a sampling argument, per
task-a review "position is a system property". Initializers:

- `GaussianBeam` + `gaussian(&beam, z, lambda, grid)` — fundamental mode,
  $e^{+ikz}$ $q$-form ($q = z - iz_R$); `beam_radius(z, lambda)` /
  `gouy_phase(z, lambda)` are functions of the measurement plane
  (physics: `physics.md` §9.5).
- `HermiteGaussBeam { w0, m, n, amplitude, center }` + `hermite_gauss`,
  `LaguerreGaussBeam { w0, p, l, amplitude, center }` + `laguerre_gauss`
  (Phase 5 task a): orthonormal mode families on the same $q$-form
  machinery, Gouy $(N+1)\zeta$, LG winding $e^{i\ell\varphi}$ (OAM
  $+\ell\hbar$); physicists' Hermite / associated Laguerre by dependency-
  free recurrences; total order capped at `MAX_MODE_ORDER` (`InvalidMode`)
  (physics: §9.6).
- `spherical(source, distance, amplitude, lambda, grid)` (Phase 5 task d):
  diverging point-source wave $A e^{ikr}/r$ — the PSF input for finite-
  conjugate systems (physics: §9.6).

Sign conventions are pinned by cross-checks against the crate's own engines
(unit tests: modes and the spherical wave are Fresnel-TF/ASM eigenfunctions).
`SourcesError`: `InvalidBeam` / `InvalidPlane { z }` / `InvalidMode { order,
max }` / `InvalidSource { x, y, distance, amplitude }` diagnostics plus
transparent `FieldError` carry.

### 2.4 `propagation` — engines

- `fft` (private): NumPy-convention 2D transforms (unnormalized forward,
  $1/N$ inverse — exact circular convolution theorem) over row-major buffers via
  rustfft, plus the unshifted frequency axis. All conventions
  (sign, scaling, bin order) are documented in the module and pinned
  by dedicated unit tests *before* dependent code — per agents.md, these are
  the primary bug source. Every engine works in unshifted DFT bin order
  end-to-end; the reserved `fftshift` helpers were removed in Phase 5
  (no consumer ever materialized — display centering is a front-end concern).
- `asm` (public): exact transfer-function propagation with zero-padding and
  the Matsushima band-limit; closed-form $O(1)$ sampling guard with automatic
  densification in strict mode. Phase 6 adds `medium` (n): the whole
  machinery runs at $\lambda_n = \lambda/n$ (evanescent cutoff and band
  limit rise with n, transfer chirp smooths as $\lambda^2/n$), enabling
  in-glass gaps. `AsmConfig { padding, band_limited, strict, max_padding,
  medium }` is structurally extensible: future boundary-condition
  strategies arrive as new fields/variants without changing existing
  behavior.
- `fresnel` (public): paraxial pair sharing one report/config surface —
  `propagate` (transfer-function form, $H_F$ chirp, guard $n\Delta x^2 >
  \lambda z$, densification mitigation) and `propagate_convolution`
  (sampled impulse response, rule-2 guard $n\Delta x^2 \lesssim \lambda z$,
  no mitigation loop — densification worsens kernel sampling). The guards
  are reciprocal (complementary regimes, crossing at critical sampling).
  `FresnelConfig { padding, strict, max_padding }`;
  `FresnelSamplingReport` carries both guards plus the paraxial-validity
  diagnostic (`max_phase_error_vs_asm`, `paraxial_ok`).
- `fraunhofer` (public): far-field FT with its own output grid
  ($\Delta X = \lambda|z|/(n\Delta x)$ — the only resampling engine),
  exact offset-DFT bin alignment (all parities), Fresnel-number regime gate,
  and off-axis support. `FraunhoferConfig { strict, max_n_f, source_pos,
  aperture_extent }`; `regime_report(band, z, cfg)` exposes $N_F$, extent
  provenance, and the dropped-quadratic bound.
- `rayleigh_sommerfeld` (public, Phase 4): Rayleigh–Sommerfeld **first-kind**
  convolution engine — the exact spherical impulse response
  $h = \frac{|z|}{2\pi r^2}(\frac{1}{r} - i\,\mathrm{sgn}(z)k)e^{i\,\mathrm{sgn}(z)kr}$
  (textbook Eq. 2.8 in the crate's $e^{+ikz}$ frame) sampled on the padded
  grid in DFT order, same embed/transform/extract pipeline as the Fresnel
  convolution form. First kind only (Dirichlet Green's function needs just
  the sampled field; the second kind would need $\partial u/\partial n$ —
  deliberately excluded). Continuous operator equals ASM (RS *is* the
  plane-wave spectrum, Sec. 2.2.2/3.2): the engine is a third
  discretization for cross-validation, not new physics. Closed-form $O(1)$
  kernel guard (spherical rim step + obliquity bound; always weaker than
  the paraxial Fresnel step; no densification loop), `strict` raises
  immediately; $z = 0$ rejected (distributional kernel).
  `RayleighSommerfeldConfig { padding, strict }`;
  `RayleighSommerfeldSamplingReport` carries the kernel steps plus
  `max_phase_error_vs_fresnel`/`paraxial_ok` (the non-paraxial excess the
  Fresnel models drop).
- `regime` (public): **diagnostics only** (approved decision — no
  auto-dispatch). `fresnel_number(a, lambda, z)` and
  `regime_report(_with)(band, z[, aperture])` evaluate the criteria ladder
  and recommend an `Engine`; the caller always chooses.

Public API shape (uniform across engines):

```rust
pub fn propagate(band: &BandField, z: f64, cfg: &Cfg) -> Result<BandField, Err>;
pub fn sampling_report(band: &BandField, z: f64, cfg: &Cfg) -> Report;  // asm, fresnel
pub fn regime_report(band: &BandField, z: f64, cfg: &FraunhoferConfig) -> FraunhoferReport;
```

All engines are free functions; propagators depend on `field`, never the
reverse, and never on `element` (composition is caller-side — see §3).

### 2.5 `beamline` — composition sugar (Phase 5 task c)

`Beamline<'a>` is an ordered `Vec<(Option<&'a dyn Element>, f64, f64)>` of
`(element, z_after, medium)` steps — the loop a caller would write by
hand, not a new abstraction layer. `trace(&band, propagate_closure)`
applies each element and propagates nonzero spacings through the
**caller-supplied engine closure**
(`FnMut(&BandField, z, medium) -> Result<BandField, E>`,
$E$: `std::error::Error`); zero spacings skip the engine call entirely.
`push` = vacuum step; `push_medium` and `gap(z, medium)` (Phase 6) carry
refractive indices so thick lenses compose surface → glass → surface.
A closure rather than an engine enum: per-engine configs differ and the
engine-error unification remains an open decision — the beamline API is
stable as engines are added. `psf(&band, closure)` returns `Psf` (data type living in
`field`, re-exported here): the detector-plane intensity grid $I = |u|^2$
with `coord`, `total_power`, `peak_bin`/`peak_position`, and `centroid`
(sub-bin spot position); `field::sum_intensity` folds incoherent bands
into the same type.
Elements stay caller-owned borrows; the beamline never picks an engine,
guards, or resamples — `regime` stays the diagnostics authority.
`BeamlineError<E>` = `Element(OpticsError)` | `Engine(E)` (untouched).

## 3. Data flow

```
GaussianBeam / HG / LG / point source ──sources::*──→ BandField (grid + λ)
Wave (source params) ──────front-end────────→ Field ──bind λ──→ BandField
                                                        │
                 Beamline::trace/psf (element, z) steps │  (composition sugar)
                          Element::apply (mask, phase)  │  local, at a plane
                                                        ↓
                                            BandField (plane z0)
                                                        │
              regime::regime_report(band, z) ──(diagnostic)──┐
                                                        ↓     ↓
              asm::propagate | fresnel::propagate[_convolution] | fraunhofer::propagate | rayleigh_sommerfeld::propagate
                                                        │  free space z1−z0, guards/mitigation
                                                        ↓
                                            BandField (plane z1) ──→ … → detector
                                                        (intensities sum for multiband)
```

`apply` and `propagate` never call each other: elements are local
transmission-function multiplications; propagators are free space only.
Engine choice per segment is the caller's, optionally informed by `regime`.
Fraunhofer is the sole exception to the same-grid contract (its output
plane is physically a different sampling). Phase entry points:
`BandField::{uniform, plane_wave, new}` build fields;
`propagation::*::propagate` advances them; `Element::apply` transforms them.

## 4. Error taxonomy

Plain thiserror enums; variants carry diagnostics (agents.md §4).
Sampling/regime reports travel *boxed* inside the risk variants (review
decision 2026-10-11, consistent across engines) and are also available
proactively via the `*_report` functions.

| Type | Module | Variants |
|---|---|---|
| `FieldError` | `field` | `EmptyGrid`, `DimensionMismatch { len, expected }`, `NonPositivePitch { dx, dy }`, `NonPositiveWavelength(λ)`, `NonFiniteSample` |
| `OpticsError` | `element` | `InvalidAperture { half_x, half_y, center_x, center_y }` (footprint half-extents, all shapes), `OutsideField { …, grid_half_x, grid_half_y }` (bounding box for rotated stops), `InvalidAxis { center_x, center_y }`, `InvalidFocal { lambda, f }`, `InvalidPhase { x, y, phase }` (first non-finite closure value / sampled grid value, element-frame position), `InvalidGrating { period, depth, orientation }`, `InvalidOrientation { radians }`, `InvalidGrid { nx, ny, len }`, `InvalidGridPitch { dx, dy }`, `InvalidSurface { radius, conic, r }`, `InvalidMedium { n_front, n_back }`, `InvalidSag { r, sag }`, `InvalidThickness { thickness }`, `InvalidWedge { angle, base_thickness }`, `InvalidTilt { x, y }` |
| `SourcesError` | `sources` | `InvalidBeam { w0, amplitude, cx, cy }`, `InvalidPlane { z }`, `InvalidMode { order, max }`, `InvalidSource { x, y, distance, amplitude }`, `Field(#[from] FieldError)` (transparent grid/wavelength carry) |
| `PropagationError` | `propagation::asm` | `InvalidDistance(z)` (non-finite only — signed z is valid), `InvalidWavelength`, `InvalidMedium(n)` (Phase 6: `AsmConfig::medium` not finite/positive), `EmptyGrid`, `NonFiniteSample`, `AliasingRisk { report: Box<SamplingReport> }` |
| `FresnelError` | `propagation::fresnel` | same taxonomy; `AliasingRisk { report: Box<FresnelSamplingReport> }` (TF form: post-mitigation failure; convolution form: immediate kernel failure) |
| `FraunhoferError` | `propagation::fraunhofer` | `InvalidDistance(z)` (non-finite **or zero** — far-field map singular at z=0), `InvalidWavelength`, `EmptyGrid`, `NonFiniteSample`, `RegimeViolation { report: Box<FraunhoferReport> }` |
| `RayleighSommerfeldError` | `propagation::rayleigh_sommerfeld` | `InvalidDistance(z)` (non-finite **or zero** — RS-I kernel is a $\delta$-distribution at z=0), `InvalidWavelength`, `EmptyGrid`, `NonFiniteSample`, `AliasingRisk { report: Box<RayleighSommerfeldSamplingReport> }` (kernel undersampled; immediate — no densification) |
| `BeamlineError<E>` | `beamline` | `Element(OpticsError)` (apply failure mid-trace), `Engine(E)` (the caller's engine error, preserved untouched); generic over the engine error type |

`sampling_report` (asm, fresnel, rayleigh_sommerfeld) and `regime_report`
(fraunhofer, regime)
expose the criteria before propagation; `Display` implementations print all
diagnostics plus remedy hints. Unifying the four engine error enums into
one propagation-level type is a later API decision.

## 5. Testing strategy

- **Unit tests** beside each module: grid/index/coordinate mapping, FFT
  conventions (round-trip, Parseval $\times N$, bin/sign, exact
  circular-convolution theorem), ASM internals (padding math,
  band-limit vs z, identity, mitigation ladder), Fresnel guards (closed-form
  step = sweep maximum, complementarity, critical-sampling TF-vs-convolution
  agreement $<10^{-6}$, delta = sampled kernel), Fraunhofer (delta prefactor
  identity, DC concentration, Parseval power, regime gate, off-axis ramp),
  Rayleigh–Sommerfeld (paraxial kernel reduction $=$ Fresnel kernel,
  delta $=$ sampled kernel, bit-exact conjugate mirror, guard brackets a
  brute-force sweep, padding-worsens-kernel, RS $\equiv$ ASM on a smooth band),
  regime ladder outcomes, element mask geometry, thin-lens Fourier property +
  dispersive closure + DOE-lens-equals-thin-lens sample-exact, grating sawtooth
  wrap + integer-$2\pi$ depth = tilted plane wave, `ClearAperture` predicates,
  uniform orientation (defaults/validation/rotated stops/bounding box), `Angle`
  $[0,2\pi)$ wrapping, sampled `PhaseMask` (bilinear lattice exact on linear
  profiles, zero beyond span, rotation bit-exact, DOE-lens lattice $=$
  `ThinLens` at lattice points, validation taxonomy), Gaussian $q$-form
  cross-engine agreement + $u(-z)$ conjugate, HG/LG (orthonormal Gram,
  eigenfunction propagation, parity, ring + vortex winding, $(N{+}1)$ Gouy,
  $\mathrm{HG}_{00}$ anchor), spherical wave ($kr/r$ anchors, diverging sign,
  cross-engine), `Beamline` (bit-for-bit vs manual chain, zero-spacing skip,
  typed error arms, `Psf` intensity/geometry bookkeeping).
- **Physical validation** (`tests/asm.rs`, `tests/phase2.rs`, `tests/phase3.rs`,
  `tests/phase4.rs`, `tests/phase5.rs`, `tests/phase6.rs`;
  shared analytics in `tests/common/mod.rs` — dependency-free quadrature for
  $J_1$, Fresnel integrals, and the direct RS-I convolution integral): twenty-five
  analytics-based integration tests with
  stated tolerances — see the mapping table in `physics.md` §11. `tests/phase3.rs`
  validates grating orders through the Fraunhofer engine (kinoform 2π→+1,
  4π→+2, π→40/40 split). `tests/phase4.rs` validates the RS engine against the
  direct quadrature, ASM, and Fresnel (FFT$=$direct, three-discretizations
  agree, Fresnel diverges where its reported paraxial error says, round trip,
  power conservation). `tests/phase5.rs` validates the PSF pipeline
  (beamline-traced Airy of the pupil, focal shift $f\tan\theta$ of a tilted
  plane wave, geometric image of an off-axis point source);
  `examples/psf_image_convolve.rs` demonstrates test-image blur through a
  normalized PSF kernel (power-preserving, centroids unmoved).
- Gates: `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`,
  `cargo doc` (docstring/unit-documentation standard).

## 6. Roadmap hooks

- **Phase 2** (Fresnel/Fraunhofer) — **complete, released as 0.2.0**:
  NumPy-convention FFT scaling (exact convolution theorem), Fresnel TF +
  convolution propagators with reciprocal guards, Fraunhofer limit with
  regime gate, off-axis `source_pos`, declared `aperture_extent`, and
  diagnostics-only regime guidance.
- **Phase 3** (element library, thin) — **complete, released as 0.3.0**:
  `sources` Gaussian initializer; apertures
  unified into `EllipticalAperture` (circle special case) +
  `RectangularAperture` (breaking); `ThinLens` with dispersive $f(\lambda)$
  closure + `ClearAperture` stop; `PhaseMask` analytic-closure DOE interface
  (center-relative, shared `ClearAperture`); `BlazedGrating` with
  Fraunhofer-engine order validation; uniform element orientation with
  implicit-wrapping `Angle` (breaking).
- **Phase 4** (Rayleigh–Sommerfeld) — **complete, released as 0.4.0**:
  `rayleigh_sommerfeld::propagate` first-kind
  convolution engine (exact spherical kernel, closed-form guard, $z=0$
  rejected; second kind excluded — no $\partial u/\partial n$ data model),
  `tests/phase4.rs` cross-model consistency vs ASM/Fresnel with a
  test-only direct-quadrature reference (planned "direct-integral form"
  delivered as the validation truth, keeping the lib minimal); regime
  ladder unchanged (RS outside the recommendation — cross-validation
  engine).
- **Phase 5** (sources & output) — **complete, released as 0.5.0**:
  HG/LG mode initializers (orthonormal, $N \le 64$)
  and `spherical` point source in `sources`; sampled-`PhaseMask` variant
  (D2 delivered — bilinear element-frame lattice); `beamline` composition
  sugar (`Beamline`/`trace`/`psf`/`Psf`, caller-supplied engine closure —
  no `Propagator` trait refactor); PSF validation `tests/phase5.rs` +
  `examples/psf_image_convolve.rs`. **Deferred by decision:**
  visualization/export backend (front-end project — the library stays
  headless); the reserved `fft.rs` shift helpers were removed (no
  consumer ever materialized).
- **Phase 6** (element library expansion — thick elements) — **complete,
  released as 0.6.0**: `RefractiveSurface` with
  `SurfaceProfile` (Zemax standard-surface sag: sphere/conic/even-asphere/
  analytic), medium-aware ASM (`AsmConfig::medium`, $\lambda_n$
  substitution), `Beamline` per-step media (`push_medium`, `gap`) — the
  Zemax-faithful composition surface → glass → surface reproduces the
  thick-lens Eq. 4.1 back focal length (`tests/phase6.rs`); `ThickLens`
  single-plane convenience (Eq. 4.17, `Extent::Thick` + `Material` pair
  consumed, first spherical aberration anchors); `Wedge` prism
  ($\sin\theta = -(n-1)\tan\alpha$); hybrid refractive-DOE achromat
  validated (Sec. 4.4.1). **Deferred with documented reasons:** mirrors
  (axis fold breaks the single-forward-z model), waveplates (vector
  fields/polarization out of scope), oblique-incidence Fresnel
  coefficients (scalar screens only). Next: a `zemax` import layer
  (prescription parse → surfaces + medium steps) — no model changes
  needed, format mapping only.
- **Phase 7** (core forward-physics close-out) — **complete, released as
  0.7.0**: `Material::sellmeier` dispersion mechanism; incoherent multiband
  summation (`field::sum_intensity`, D3 closed — `Psf` now the shared
  intensity-grid data type in `field`); `analysis` metrics (encircled
  energy, spot RMS, MTF, Strehl, Maréchal); Zernike least-squares
  decomposition; paraxial surface tilt (breaking constructor change;
  `ThickLens` tilt-free by cancellation). **The core crate is now
  feature-complete for forward modelling** — remaining exclusions are
  by design (mirrors/waveplates need axis-fold/vector fields; oblique
  Fresnel coefficients; ray tracing per agents.md scope).
- **Roadmap (post-core, tracked in `tasks/plan.md` + `tasks/todo.md`)**:
  workspace of sibling crates — `optics-zemax` (prescription import:
  parse → `RefractiveSurface` + medium steps; delivered Phase 8),
  `optics-inverse` (GS/derivative-free/finite-difference first, adjoint
  later, `burn` deep surrogates last — **Milestone A complete, released
  as 0.1.0**, see §7; Milestone B custom-merit frontend and Milestone C
  surrogates follow), `optics-studio` (web UI, opt-in workspace
  member). Design choices below keep these open.

**Forward-compatibility design choices (decided at Phase-3 planning):**

1. `Element::apply` stays **strictly local** (same grid in/out) — thick
   elements are later *composed* as surface → `propagate` → surface, never
   a special case inside `apply`.
2. Focal length is a **closure $f(\lambda)$** mirroring `Material`'s
   design — dispersive and achromatic doubles need no API change.
3. Phase masks take an **analytic closure** $(x,y)\to$ phase; the
   sampled-profile variant (D2) arrived with Phase 5 for inverse-design
   output.
4. All engine reports are `Copy` with public fields — trivially
   serializable for the future front-end/export layer (serde is a Phase 5
   decision, not added now).
5. `sources` is a new top-level module; initializers return `BandField`,
   so multiband incoherent stacks (collections of `BandField`s) extend it
   without touching propagation or element code.

Out of scope (agents.md): ray tracing / geometric optics, vector field grids,
GPU/parallelism, dispersive propagators, interactive visualization *inside
this crate* (the front-end project owns the UI).

## 7. Workspace crate: `optics-inverse` (Phase 9 Milestone A — released 0.1.0)

Inverse design flips the forward model: a **design** is parameterized by
a bounded vector, a **renderer** produces an `Observation`, a **merit**
scores it, and a **solver** searches the parameter space. Dependency
direction stays one-way: `optics-inverse` depends on `optics-core` (path
dependency) and consumes the public API exactly as a front-end would;
`optics-zemax` is a dev-dependency only (the ACA254 Adam golden) and
never appears in the public surface.

| Module | Role | Public surface |
|---|---|---|
| `params` | The bounded design-space vector all solvers see (validate-at-apply, clamp, seeded `randomize`) | `Param`, `Params` |
| `carrier` | What a parameter vector *means*: closure → owned element stack (`Design`) borrowing into a core `Beamline` on demand | `Carrier` (`analytic` / `grid_phase` / `grid_thickness`), `Design`, `Step`, `GridSpec`, `GridGeom`, `AnalyticBuilder` |
| `observe` | Design → data: incoherent multi-band, multi-scene render through the core ASM, with FD derivative channels ∂I/∂x₀, ∂I/∂y₀, ∂I/∂λ | `Renderer`, `Observation` (flat `[channel][band][scene][row][col]`, numpy-ready), `Channel`, `Derivatives` |
| `merit` | Scoring seam with optional observation gradient (`obs_gradient`; `None` ⇒ FD-on-observation fallback) | `Merit`, `MseToTarget`, `EncircledEnergyMerit`, `SpotRmsMerit`, `StrehlMerit`, `MtfMerit`, `Selector`, `objective` |
| `solve` | Derivative-free solvers: adaptive-step random search, simulated annealing (auto-calibrated T0, geometric cooling) | `Solver`, `Solution`, `StochasticConfig` |
| `fd` | Central-difference parameter gradients (2·dim renders/step) + Adam descent (norm-clip, best-epoch snapshot) | `gradient`, `AdamConfig` |
| `gs` | Gerchberg–Saxton phase retrieval / DOE design (physics §12): two ASM half-transforms, energy projections, DVR/FPR variants, `Merit`-seam stopping | `GsProblem`, `GsConfig`, `GsVariant`, `GsInit`, `run`, `GsSolution`, `GsRecord` |
| `adjoint` | Reverse-mode grid gradients (physics §13, ADR-9.9): adjoints of the forward primitives in reverse order — no autodiff graph | `asm_adjoint`, `transmission`, `push_back`, `grid_gradient`, `PlaneGradient` |
| `error` | Crate-wide typed errors via thiserror — validation surfaces at apply time, never a panic | `InverseError` |

**Data flow (solver loop):** `Params` vector → `Carrier::build(values, λ)`
→ `Design` → `Renderer` (core ASM) → `Observation` → `Merit::evaluate` →
solver update. Gradient paths substitute `fd::gradient` (parameter-space
central differences) or `adjoint::grid_gradient` (one forward render +
one adjoint run per gradient-carrying frame). GS runs its own
bidirectional loop on the carrier lattice and returns a legal `GridPhase`
parameter vector — the same lattice the solver paths optimize.

**Milestone seams (0.2.0 / 0.3.0):** the `Merit` trait is the attachment
point for the custom-merit frontend (Milestone B), and the flat
numpy-ready `Observation` layout is the dataset interface for `burn`
surrogates (Milestone C, feature-gated). Example:
`crates/optics-inverse/examples/coded_doe.rs` (GS top-hat beam shaper).
