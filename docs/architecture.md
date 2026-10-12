# Architecture — `optics-rs`

Module map, data flow, trait specs, and error taxonomy for the Diffractive
Optics Engineering Suite (Phase 1: field representation + ASM propagation).
The physics rationale lives in [`physics.md`](physics.md); development
protocol in [`../agents.md`](../agents.md).

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
        propagation ──┴──→ field (algorithms over BandField)
            ├── fft        (private layer: rustfft wrapper, conventions)
            ├── asm        (exact Angular Spectrum)
            ├── fresnel    (paraxial: transfer function + convolution)
            ├── fraunhofer (far-field FT, regime-gated)
            └── regime     (diagnostics: N_F, guards, engine recommendation)
```

Hard rule (agents.md §4): **algorithms depend on data, never the reverse.**
`propagation` depends on `field`; `field` knows nothing about propagation.
Propagators are **free functions**, not methods; convenience sugar is a
Phase 5 decision point.

## 2. Modules

| Module | Role | Public surface |
|---|---|---|
| `common` | Angle newtype (radians, derefs to f64), `J` imaginary unit | `Angle`, `J` |
| `wave` | Scalar source/beam parameter records (pre-sampling) | `Wavelength`, `Polarization`, `Wave` |
| `field` | Complex sample grids; wavelength decoupled from geometry | `Field`, `BandField`, `FieldError` |
| `element` | Local field transformations; element taxonomy | `Element` trait, `CircularAperture`, `OpticsError`, `ElementShape`, `ElementKind`, `Extent`, `Material` |
| `propagation` | Wave-optics engines + regime diagnostics | `asm`, `fresnel`, `fraunhofer`, `regime` (public); `fft` (private) |

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
elements expose aperture geometry (`CircularAperture::radius`) for Phase 2+
regime guidance.

`CircularAperture` (first concrete element): ideal hard-edge binary mask,
boundary-inclusive; rejects footprints exceeding the grid (`OutsideField`).
`shape`/`kind`/`material`/`extent` accessors are taxonomy metadata;
`material` is unused by the Phase-1 mask (reserved for refractive/absorptive
models).

### 2.3 `propagation` — engines

- `fft` (private): NumPy-convention 2D transforms (unnormalized forward,
  $1/N$ inverse — exact circular convolution theorem) over row-major buffers via
  rustfft, frequency axes, NumPy-semantics shifts. All conventions
  (sign, scaling, bin order, shift) are documented in the module and pinned
  by dedicated unit tests *before* dependent code — per agents.md, these are
  the primary bug source.
- `asm` (public): exact transfer-function propagation with zero-padding and
  the Matsushima band-limit; closed-form $O(1)$ sampling guard with automatic
  densification in strict mode. `AsmConfig { padding, band_limited, strict,
  max_padding }` is structurally extensible: future boundary-condition
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

## 3. Data flow

```
Wave (source params) ──Phase 5──→ Field ──bind λ──→ BandField
                                                        │
                          Element::apply (mask, phase)  │  local, at a plane
                                                        ↓
                                            BandField (plane z0)
                                                        │
              regime::regime_report(band, z) ──(diagnostic)──┐
                                                        ↓     ↓
              asm::propagate | fresnel::propagate[_convolution] | fraunhofer::propagate
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
| `OpticsError` | `element` | `InvalidAperture { radius, center }`, `OutsideField { radius, center, half-extents }` |
| `PropagationError` | `propagation::asm` | `InvalidDistance(z)` (non-finite only — signed z is valid), `InvalidWavelength`, `EmptyGrid`, `NonFiniteSample`, `AliasingRisk { report: Box<SamplingReport> }` |
| `FresnelError` | `propagation::fresnel` | same taxonomy; `AliasingRisk { report: Box<FresnelSamplingReport> }` (TF form: post-mitigation failure; convolution form: immediate kernel failure) |
| `FraunhoferError` | `propagation::fraunhofer` | `InvalidDistance(z)` (non-finite **or zero** — far-field map singular at z=0), `InvalidWavelength`, `EmptyGrid`, `NonFiniteSample`, `RegimeViolation { report: Box<FraunhoferReport> }` |

`sampling_report` (asm, fresnel) and `regime_report` (fraunhofer, regime)
expose the criteria before propagation; `Display` implementations print all
diagnostics plus remedy hints. Unifying the three engine error enums into
one propagation-level type is a later API decision.

## 5. Testing strategy

- **Unit tests** beside each module: grid/index/coordinate mapping, FFT
  conventions (round-trip, Parseval $\times N$, bin/sign, shift semantics vs
  NumPy, exact circular-convolution theorem), ASM internals (padding math,
  band-limit vs z, identity, mitigation ladder), Fresnel guards (closed-form
  step = sweep maximum, complementarity, critical-sampling TF-vs-convolution
  agreement $<10^{-6}$, delta = sampled kernel), Fraunhofer (delta prefactor
  identity, DC concentration, Parseval power, regime gate, off-axis ramp),
  regime ladder outcomes, element mask geometry.
- **Physical validation** (`tests/asm.rs`, `tests/phase2.rs`; shared
  analytics in `tests/common/mod.rs` — dependency-free quadrature for
  $J_1$, Fresnel integrals): eleven analytics-based integration tests with
  stated tolerances — see the mapping table in `physics.md` §11.
- Gates: `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`,
  `cargo doc` (docstring/unit-documentation standard).

## 6. Roadmap hooks

- **Phase 2** (Fresnel/Fraunhofer) — **complete**: NumPy-convention FFT
  scaling (exact convolution theorem), Fresnel TF + convolution propagators
  with reciprocal guards, Fraunhofer limit with regime gate, off-axis
  `source_pos`, declared `aperture_extent`, and diagnostics-only regime
  guidance.
- **Phase 3** (element library): rectangular/elliptical apertures, thin lens
  (quadratic phase), general phase-mask/DOE interface, grating.
- **Phase 4** (Rayleigh–Sommerfeld): direct-integral/convolution form,
  cross-model consistency.
- **Phase 5** (sources & output): field initializers from `Wave`, Gaussian/
  multimode sources, visualization/export (backend = decision point), API
  ergonomics (convenience methods / `Propagator` trait).

Out of scope (agents.md): ray tracing / geometric optics, vector field grids,
GPU/parallelism, dispersive propagators.
