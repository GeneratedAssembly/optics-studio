# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), and commits
follow [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Decisions (approved in planning check-ins, 2026-10-11)

- Phased roadmap; Phase 1 = field representation + ASM propagation.
- **f64 everywhere** (refactor existing f32 skeleton).
- FFT backend: **rustfft** (pure Rust, f64, arbitrary sizes).
- **Evolve the existing `Element` trait** to operate on fields.
- ASM boundaries: zero-padding (default 2x) + band-limited transfer function by
  default; `AsmConfig` extensible for future boundary-condition strategies
  (infrastructure only in Phase 1).
- Errors: plain enums with manual `Display`/`Error` impls (no `thiserror`).
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
- Dependency: `rustfft = "6"`.
- `CHANGELOG.md`.

### Changed

- `refactor:` widened all numeric types from f32 to f64 across `common`, `wave`,
  and `element` (`Angle` derefs to `f64`, `J: Complex64`, `Material` closures
  `(f64, f64) -> f64`, `Extent`/`ElementShape::Poly` to `f64`). No behavior
  change.
