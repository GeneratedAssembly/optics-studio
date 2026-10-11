# Physics — Wave-Optics Propagation in `optics-rs`

This document details the mathematical models, transfer functions, and
validity regimes implemented in Phase 1. The ultimate source of truth is
*Diffractive Optics: Design, Fabrication, and Test* (SPIE TT95, the textbook
PDF at the repository root); references below use its numbering.

Conventions: **SI units throughout** (meters, radians, cycles per meter),
**f64 precision**, scalar (single-band) fields. The unit of propagation is a
`BandField` — a spatial `Field` grid bound to one wavelength $\lambda$
(the wavelength-free `Field` keeps incoherent multiband sources possible:
collections of `BandField`s whose *intensities* sum at detection).

---

## 1. Field representation

A `Field` samples a complex scalar optical amplitude $u(x, y)$ on a uniform,
origin-centered grid:

$$x_{ix} = \left(ix - \frac{n_x - 1}{2}\right)\Delta x, \qquad
  y_{iy} = \left(iy - \frac{n_y - 1}{2}\right)\Delta y$$

row-major storage at flat index $iy \cdot n_x + ix$. The field amplitude is in
arbitrary coherent units with $|u|^2 \propto$ irradiance; the discrete power

$$P = \sum_{ix,\,iy} |u_{ix,iy}|^2 \, \Delta x \, \Delta y$$

is the energy-conservation invariant checked against propagation (watts if
$|u|^2$ is read in W/m²).

## 2. Plane-wave basis and the angular spectrum

Per the textbook (Eqs. 2.9, 3.5–3.13), any monochromatic field can be
decomposed into plane waves with direction cosines

$$\alpha^2 + \beta^2 + \gamma^2 = 1, \qquad
  u_{\alpha\beta} \propto \exp\!\Big(i \tfrac{2\pi}{\lambda}(\alpha x + \beta y + \gamma z)\Big).$$

Sampling on a grid with pitch $(\Delta x, \Delta y)$ maps spatial frequencies
$f_x, f_y$ (cycles/m) to direction cosines $\alpha = \lambda f_x$,
$\beta = \lambda f_y$; the DFT of the samples is (up to normalization) the
angular spectrum of the field.

## 3. Angular Spectrum Method (ASM)

Each spectral component accumulates only a phase over a distance $z$, so the
propagation is diagonal in frequency:

$$U(x, y, z) = \mathcal{F}^{-2}\!\Big\{\, H(f_x, f_y)\cdot
  \mathcal{F}^{2}\!\{U(x, y, 0)\}\,\Big\},$$

$$H(f_x, f_y) =
  \begin{cases}
    \exp\!\Big(i k z \sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2}\Big)
      & (\lambda f_x)^2 + (\lambda f_y)^2 \le 1 \\[4pt]
    0 & \text{(evanescent)}
  \end{cases},
  \qquad k = \frac{2\pi}{\lambda}.$$

**Exactness.** No paraxial (Fresnel) approximation is made: the square root is
the full direction cosine $\gamma$. The method is exact for the scalar
Helmholtz model within the sampled band. Negative $z$ back-propagates
($H(f,-z) = H(f,z)^*$); $z = 0$ is the identity.

**Evanescent cutoff.** Components with $(\lambda f_x)^2 + (\lambda f_y)^2 > 1$
have imaginary $\gamma$ and decay within $\sim\lambda$ of the plane; they are
dropped (set to 0 in $H$). Near-field imaging below the diffraction limit is
therefore not modeled — acceptable for engineering distances in the
diffractive-optics design workflow.

## 4. Discrete implementation

### 4.1 FFT conventions (primary bug source — pinned by unit tests)

- **Backend:** `rustfft` (pure Rust, f64, arbitrary sizes). rustfft is
  *unnormalized in both directions* (raw round-trip returns $N u$), unlike
  NumPy's $1/N$-inverse. We therefore apply unitary scaling $1/\sqrt{N}$ per
  axis per direction, making `fft2`/`ifft2` exact inverses and energy-
  transparent (Parseval holds numerically).
- **Sign:** forward kernel $e^{-i 2\pi k n/N}$ (rustfft/NumPy), so a discrete
  plane wave $e^{+i2\pi k_0 n/N}$ lands in bin $+k_0$.
- **Bin order:** transforms operate in standard (unshifted) DFT bin order;
  bin $k$ means frequency $f_k = k/(N\Delta)$ with wraparound above Nyquist.
  The ASM multiplication $H \cdot \mathcal{F}^2\{U\}$ is shift-agnostic, so no
  `fftshift` is needed; `fftfreq`/shift helpers exist for reporting and are
  tested against NumPy semantics.
- **Scope warning:** unitary scaling is *not* convolution-theorem-neutral
  ($\mathcal{F}\{x \otimes h\} = \sqrt{N}\,\mathcal{F}\{x\}\mathcal{F}\{h\}$).
  It is safe for the transfer-function form used here because the scalings
  cancel. Convolution-form propagators and PSF analysis (Phase 2+) must
  revisit this — recorded as a decision point in the roadmap and CHANGELOG.

### 4.2 Wraparound control: zero-padding

The DFT implements *circular* convolution: the periodic replicas of the
sampled window contaminate the physical window with wraparound. Phase 1
embeds the input into a zero-padded grid (default factor `padding = 2.0` per
axis), propagates there, and extracts the central window.

### 4.3 Band-limited ASM

Zero-padding alone does not stop aliasing of the transfer function's chirp at
large $z$. We apply the band-limited transfer function of
**Matsushima & Shimobaba, "Band-limited angular spectrum method for numerical
simulation of free-space propagation in close and far fields," Opt. Express
17(23), 19731–19741 (2009)**: additionally mask $H$ to

$$|f_x| \le f_{\lim,x} = \frac{1}{\lambda \sqrt{\left(\dfrac{2z}{L_x}\right)^2 + 1}},
  \qquad L_x = n_{x,\text{pad}}\, \Delta x,$$

(and symmetrically along $y$). Physically: components whose associated rays
shift more than the padded grid width over $z$ cannot be represented without
wraparound and are suppressed. **The formula uses the padded grid width
$L_{\text{pad}}$, not the original window** — this convention is asserted by
test 5 of `tests/asm.rs`. Band-limiting is on by default
(`AsmConfig::band_limited = true`).

## 5. Sampling verification (mandated automatic check)

Nyquist adequacy of the *transfer function itself* requires the phase step of
$H$ between adjacent frequency bins to stay below $\pi$ everywhere on the
masked grid. Per the project guideline to prefer closed-form correction over
diagnostics, the propagator computes a conservative **upper bound** of that
step in $O(1)$ from candidate worst-case bin pairs only — the band-edge
axis bins and the band-limit corner (where the phase gradient
$\partial_f \gamma \propto \lambda^2 f/\gamma$ is steepest) — never an
$O(N^2)$ sweep:

- `sampling_report(band, z, cfg)` exposes the diagnostics (`max_phase_step_x/y`,
  band limits, requested/used padding, `nyquist_ok`).
- In `strict` mode (default), a violation is **mitigated automatically**:
  the padding is doubled (1 → 2 → 4 → …) up to `max_padding` (default 8.0;
  `1.0` disables mitigation) until the criterion passes.
- `PropagationError::AliasingRisk { report }` is returned only when the cap
  cannot restore adequacy.

Note that with band-limiting active, the worst-case step is bounded by
$\approx \pi$ by construction ($k\lambda/2$ at the mask edge), which is
precisely why prevention-by-band-limiting is the default and the guard rarely
fires; the guard matters for `band_limited = false` runs.

## 6. Elements: local interaction, propagation decides regimes

An `Element::apply(&BandField)` is a **local** transformation of the sampled
plane (transmission function multiplication; grid geometry and $\lambda$
preserved). Diffraction — light bending around aperture edges into the
geometric shadow (textbook Sec. 1.4, Huygens construction) — is *not* produced
by the element; it emerges when the truncated field is propagated, because
the sharp edges inject high spatial frequencies whose plane-wave components
diverge.

**Regime judgment (Fresnel number).** Whether the pattern downstream of an
aperture of radius $a$ is Fresnel (near field) or Fraunhofer (far field) is
decided by the distance $z$ to the next element or detector through

$$N_F = \frac{a^2}{\lambda z}, \qquad
  N_F \gg 1\ \text{(Fresnel / near)}, \qquad
  N_F \lesssim 1\ \text{(Fraunhofer / far)}.$$

This is a *propagation* property combining aperture, wavelength, and
inter-element distance — Phase 1's ASM is exact in both regimes and needs no
$N_F$; Phase 2's Fresnel/Fraunhofer propagators must select and validate
regimes with it (recorded in the CHANGELOG decisions, 2026-10-11).

**Fraunhofer check (test 4).** At $N_F \ll 1$ the far field is the Fourier
transform of the aperture function (textbook Sec. 2.2), so a circular
aperture must produce the Airy profile

$$\frac{I(v)}{I(0)} = \left[\frac{2 J_1(v)}{v}\right]^2, \qquad
  v = k a \sin\theta \approx \frac{k a r}{z}, \qquad
  v_{\text{first zero}} = 3.8317 \Leftrightarrow r_1 = \frac{1.22\,\lambda z}{2a}.$$

## 7. Validity regime and limitations

- Scalar theory; polarization stays a `Wave`-level property (no vector field
  grids yet).
- Exact Helmholtz propagation **within the sampled band**: evanescent
  components dropped; transfer-function aliasing guarded as in §5.
- Wraparound controlled by padding + band-limiting; energy leaving the
  physical window (crop) is lost from the returned field — the validation
  suite sizes grids so this is $\le 10^{-9}$ relative.
- Elements are on-axis hard-edge/phase models for now; oblique interaction
  (incidence-angle dependence) deferred.
- Dispersion: $\lambda$ is fixed per `BandField`; dispersive materials enter
  through `Material` closures in later phases.

## 8. Verification mapping (`tests/asm.rs`)

| Test | Physics statement | Analytic anchor | Tolerance |
|---|---|---|---|
| 1 | Energy conservation | unitary FFT, $\lvert H\rvert = 1$ on band | rel. power $\le 10^{-9}$ |
| 2 | Plane-wave invariance | single spectral bin gains $e^{ikz}$ | $10^{-6}$ |
| 3 | Tilted plane wave | ramp slope $k\alpha$, $\alpha = \lambda f_x$ | $10^{-6}$ |
| 4 | Airy far field | FT of circular aperture, $N_F = 0.05$ | profile $\le 2\%$ of peak; $r_1$ $\le 2\%$ |
| 5 | Sampling guard | worst step $\approx 19$ rad $> \pi$; mitigation $\propto 1/$padding | exact variant match |
| 6 | Round trip $+z/-z$ | $H \cdot H^* = 1$ on pass band | rel. Frobenius $< 10^{-6}$ |

## References

1. *Diffractive Optics: Design, Fabrication, and Test* (workspace root PDF):
   Sec. 1.4 (Huygens, Fresnel zones), Sec. 2.2 (far field = FT of aperture),
   Eqs. 2.9, 3.5–3.13 (plane-wave basis, direction cosines).
2. N. Matsushima and T. Shimobaba, "Band-limited angular spectrum method for
   numerical simulation of free-space propagation in close and far fields,"
   *Optics Express* **17**(23), 19731–19741 (2009).
3. J. W. Goodman, *Introduction to Fourier Optics* — standard Airy form
   $[2J_1(v)/v]^2$ for the circular aperture transform (the textbook states
   the far-field-is-the-aperture-FT relation; the explicit $J_1$ form is the
   classical evaluation of that transform).
