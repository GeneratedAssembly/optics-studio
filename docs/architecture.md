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
            ├── fft   (private layer: rustfft wrapper, conventions)
            └── asm   (public Angular Spectrum propagator)
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
| `propagation` | Wave-optics engines | `asm` (public); `fft` (private) |

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
number $N_F = a^2/(\lambda z)$ at propagation time (see `physics.md` §6), so
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
- `asm` (public): transfer-function propagation with zero-padding and the
  Matsushima band-limit; closed-form $O(1)$ sampling guard with automatic
  densification in strict mode. API:

  ```rust
  pub fn propagate(band: &BandField, z: f64, cfg: &AsmConfig)
      -> Result<BandField, PropagationError>;
  pub fn sampling_report(band: &BandField, z: f64, cfg: &AsmConfig)
      -> SamplingReport;
  ```

  `AsmConfig { padding, band_limited, strict, max_padding }` is structurally
  extensible: future boundary-condition strategies arrive as new
  fields/variants without changing existing behavior (Phase 1 implements only
  pad + band-limit, per the approved decision).

## 3. Data flow

```
Wave (source params) ──Phase 5──→ Field ──bind λ──→ BandField
                                                        │
                          Element::apply (mask, phase)  │  local
                                                        ↓
                                                     BandField ──→ propagate(band, z, cfg)
                                                                     │ sampling guard /
                                                                     │ mitigation
                                                                     ↓
                                              BandField (next plane) ──→ … → detector
                                                                     (intensities sum for multiband)
```

Phase 1 entry points: `BandField::{uniform, plane_wave, new}` build fields
directly; `propagation::asm::propagate` advances them; `Element::apply`
transforms them.

## 4. Error taxonomy

Plain thiserror enums; variants carry diagnostics (agents.md §4).

| Type | Module | Variants |
|---|---|---|
| `FieldError` | `field` | `EmptyGrid`, `DimensionMismatch { len, expected }`, `NonPositivePitch { dx, dy }`, `NonPositiveWavelength(λ)`, `NonFiniteSample` |
| `OpticsError` | `element` | `InvalidAperture { radius, center }`, `OutsideField { radius, center, half-extents }` |
| `PropagationError` | `propagation::asm` | `InvalidDistance(z)` (non-finite only — signed z is valid), `InvalidWavelength`, `EmptyGrid`, `NonFiniteSample`, `AliasingRisk { report: SamplingReport }` |

`SamplingReport` is the exposed sampling check (λ, z, grid, requested/used
padding, band limits, closed-form worst phase steps, `nyquist_ok`, `Display`).
It travels *inside* the error variant when mitigation fails, and is available
proactively via `sampling_report`.

## 5. Testing strategy

- **Unit tests** beside each module: grid/index/coordinate mapping, FFT
  conventions (round-trip, Parseval, bin/sign, shift semantics vs NumPy), ASM
  internals (padding math, band-limit vs z, identity, mitigation ladder),
  element mask geometry.
- **Physical validation** (`tests/asm.rs`): six analytics-based integration
  tests with stated tolerances — see the mapping table in `physics.md` §8.
- Gates: `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`,
  `cargo doc` (docstring/unit-documentation standard).

## 6. Roadmap hooks

- **Phase 2** (Fresnel/Fraunhofer): FFT normalization migrated to the
  NumPy convention (exact convolution theorem; decision 2026-10-11 — done in
  the task-1 commit); remaining: Fresnel TF + convolution propagators,
  Fraunhofer limit, $N_F$-based regime guidance.
- **Phase 3** (element library): rectangular/elliptical apertures, thin lens
  (quadratic phase), general phase-mask/DOE interface, grating.
- **Phase 4** (Rayleigh–Sommerfeld): direct-integral/convolution form,
  cross-model consistency.
- **Phase 5** (sources & output): field initializers from `Wave`, Gaussian/
  multimode sources, visualization/export (backend = decision point), API
  ergonomics (convenience methods / `Propagator` trait).

Out of scope (agents.md): ray tracing / geometric optics, vector field grids,
GPU/parallelism, dispersive propagators.
