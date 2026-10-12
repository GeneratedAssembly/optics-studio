# Physics — Wave-Optics Propagation in `optics-rs`

This document details the mathematical models, transfer functions, and
validity regimes implemented in Phases 1–2 (ASM exact; Fresnel transfer-
function and convolution forms; Fraunhofer far field; regime guidance).
The ultimate source of truth is
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
  NumPy's $1/N$-inverse. We therefore apply the **NumPy convention**: forward
  unnormalized, inverse scaled by $1/N$ — making `fft2`/`ifft2` exact
  inverses, Parseval $\sum|U|^2 = N\sum|u|^2$, and the circular convolution
  theorem *exact*: $\mathcal{F}^{-1}\{\mathcal{F}\{u\}\cdot\mathcal{F}\{h\}\}
  = u \circledast h$ with no stray factors. (Adopted at the Phase-2 check-in
  2026-10-11, superseding the Phase-1 unitary scaling; transfer-function
  propagation — ASM, Fresnel TF form — is unaffected because the scalings
  cancel in $\mathcal{F}^{-1}\{H\cdot\mathcal{F}\{u\}\}$.)
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

## 6. Fresnel propagation (Phase 2)

Two discrete forms of the paraxial propagator, related by the convolution
theorem and complementary in sampling. Expanding the ASM direction cosine
$\gamma = \sqrt{1-s} \approx 1 - s/2$ with $s = (\lambda f_x)^2 +
(\lambda f_y)^2$ gives the Fresnel transfer function

$$H_F(f_x, f_y; z) = e^{\,ikz}\,\exp\!\big(-i\pi\lambda z\,(f_x^2 + f_y^2)\big).$$

The binomial-truncation phase error at a bin is
$k|z|\big(1 - \tfrac{s}{2} - \sqrt{1-s}\big) \approx k|z|\,s^2/8$ — reported
(worst-case over the grid, closed form at the corner bin) as
`FresnelSamplingReport::max_phase_error_vs_asm` with a `paraxial_ok` flag at
0.01 rad. **Reported, not enforced**: unlike ASM there is no evanescent
cutoff in $H_F$; validity is a property of the geometry.

### 6.1 Transfer-function form (`fresnel::propagate`)

Same diagonal structure as ASM: $\mathcal{F}^{-1}\{H_F \cdot \mathcal{F}\{U\}\}$
on the zero-padded grid, central window extracted. The chirp phase step
between adjacent frequency bins is separable and its worst case is exact in
closed form:

$$\Delta\phi_x^{\max} = \pi\lambda|z|\,\delta f_x\,(2f_{\text{nyq},x} - \delta f_x),
  \qquad \delta f_x = \frac{1}{n_{x,\text{pad}}\Delta x},$$

requiring $\Delta\phi < \pi \Leftrightarrow n_{\text{pad}}\Delta x^2 >
\lambda|z|$: the **short-distance / coarse-pitch** side of the pair. Strict
mode mitigates by densification (step $\propto 1/\text{padding}$) before
`AliasingRisk`, as in ASM.

### 6.2 Convolution form (`fresnel::propagate_convolution`)

Parallel-plane propagation is linear and shift-invariant, so the field is
the convolution with the impulse response (Green's function — textbook
Sec. 2.2.2, Ch. 3; Eq. 2.24 kernel reduced to paraxial form):

$$h(x, y; z) = \frac{e^{ikz}}{i\lambda z}\,
  \exp\!\Big(i\pi\frac{x^2 + y^2}{\lambda z}\Big).$$

The kernel is sampled on the padded grid **in DFT order** (origin at bin
$(0,0)$; its $x^2$-symmetry makes the Nyquist-bin sign irrelevant) with the
$\Delta x \Delta y$ quadrature weight folded in, and evaluated as
$\text{ifft2}(\text{fft2}(u)\cdot\text{fft2}(h))$ — the *exact* discrete
circular convolution, which is why the NumPy FFT scaling (superseded
decision above) is required: under unitary scaling the theorem carries a
$\sqrt{N}$ factor. Here the kernel's chirp is sampled in *space*: the worst
adjacent-sample phase step is

$$\Delta\phi^{\text{kernel}}_x = \pi(n_{x,\text{pad}} - 1)\Delta x^2 / (\lambda|z|) < \pi
  \;\Leftrightarrow\; n_{\text{pad}}\Delta x^2 \lesssim \lambda|z|,$$

algebraically the textbook Sec. 2.2 rule (2) $N > D^2/(Z\lambda)$ at the
padded width — the **long-distance / fine-pitch** side. The two guards are
reciprocal (their worst steps multiply to $\approx \pi^2$), crossing at
critical sampling $n_{\text{pad}}\Delta x^2 = \lambda z$ where both equal
$\approx\pi$. Densification *worsens* kernel sampling, so strict mode raises
`AliasingRisk` immediately for this form; remedies are a smaller kernel
extent, coarser pitch, or longer $|z|$.

Both forms share the module-level validation, signed-$z$ semantics
($h(\cdot,-z) = h(\cdot,z)^*$, $|H_F| = |h|$-based round trips are exact),
and the embed/pad/extract pipeline.

## 7. Fraunhofer propagation (Phase 2)

When the observation distance is large compared with the source (textbook
assumption 9), the Fresnel quadratic *inside* the integral becomes
negligible and the field is the Fourier transform of the aperture function
(textbook Sec. 2.2, Eq. 2.19) wrapped in an axis-centered chirp:

$$U(x, y; z) = \frac{e^{ikz}}{i\lambda z}\,
  e^{\,i\pi((x - x_s)^2 + (y - y_s)^2)/(\lambda z)}\,
  \widetilde{U}_0\!\Big(\frac{x - x_s}{\lambda z}, \frac{y - y_s}{\lambda z}\Big).$$

This is the only propagator that **resamples**: the far-field plane has
pitch $\Delta X = \lambda|z|/(n_x \Delta x)$ and extent $\lambda|z|/\Delta x$
(the DFT frequency grid mapped through $x = \lambda z f$); output dims match
input dims. $z = 0$ is rejected (singular map); negative $z$ mirrors the
pattern.

- **Regime gate (Fresnel number, half-extent convention).** The dropped
  interior quadratic is bounded by $\varphi_{\max} = \pi r^2/(\lambda z) =
  \pi N_F$ at source half-extent $r$, with
  $$N_F = \frac{r^2}{\lambda|z|} \le \texttt{max\_n\_f}\ (\text{default } 1.0).$$
  $r$ is the declared `FraunhoferConfig::aperture_extent` when given (the
  physically relevant source size — e.g. `CircularAperture::radius()`),
  else the conservative axis-to-window-edge extent. The condition is a
  property of the *source*, not the sampling window; the declared-extent
  option (user decision 2026-10-11) prevents small apertures in large
  windows from being rejected. Strict mode raises
  `RegimeViolation { report }`; remedies: larger $|z|$, smaller source, or
  the Fresnel/ASM engines.
- **Off-axis (`source_pos`).** The chirp and frequency map are measured
  from the axis; `source_pos: Option<(f64, f64)>` decouples the axis from
  the grid center. Realized exactly by a unit-modulus input ramp
  $e^{i2\pi(f_s \xi + g_s \eta)}$, $f_s = x_s/(\lambda z)$ — no
  interpolation — plus the axis-centered chirp. ASM and Fresnel need no
  such parameter (translation-invariant operators).
- **Exact off-bin alignment.** Field sample positions are
  $(i - \frac{n-1}{2})\Delta$, half-integer multiples of the DFT bin
  spacing for even $n$; the transform is evaluated through a unit-modulus
  offset-DFT (pre/post phase factors implementing a $\frac{n-1}{2}$-bin
  shift), so output samples land at their exact physical frequencies for
  all parities.
- **Energy.** Through the offset DFT, Parseval makes the discrete far-field
  power $\sum|U|^2 \Delta X \Delta Y$ equal the input power exactly
  (verified at $10^{-9}$).

## 8. Regime guidance (Phase 2)

`propagation::regime` is **diagnostics only** (approved decision — no
auto-dispatch): `fresnel_number(a, lambda, z)` implements
$N_F = a^2/(\lambda|z|)$ for a known aperture half-width, and
`regime_report(_with)(band, z[, aperture])` evaluates, for a geometry:
$N_F$ (aperture-declared or conservative window), the far-field flag
($N_F \le 1$), the paraxial angle ($h/|z| \le 0.1$ rad), and both Fresnel
guards — recommending an engine by the documented ladder:

1. non-paraxial → **ASM** (the only exact model);
2. far field → **Fraunhofer** (valid, cheapest: one forward transform);
3. near field paraxial → **Fresnel TF** if its guard holds, else **Fresnel
   convolution** (guards are complementary, so one nearly always holds);
4. otherwise → ASM.

Note the guard hierarchy: the rule-2 kernel condition
($\lambda z > (n-1)\Delta^2$) is *weaker* than far field
($N_F \le 1 \Leftrightarrow \lambda z \ge r^2$ at window extent, a factor
$\sim n/4$ stronger) — far field implies kernel-sampled, not conversely.

## 9. Elements: local interaction, propagation decides regimes

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

## 10. Validity regime and limitations

- Scalar theory; polarization stays a `Wave`-level property (no vector field
  grids yet).
- Exact Helmholtz propagation **within the sampled band**: evanescent
  components dropped; transfer-function aliasing guarded as in §5.
- Fresnel forms (§6) add the paraxial restriction (truncation error
  reported); Fraunhofer (§7) additionally requires the far-field regime
  ($N_F \le$ threshold) — the engines enforce or report these per geometry.
- Wraparound controlled by padding + band-limiting; energy leaving the
  physical window (crop) is lost from the returned field — the validation
  suite sizes grids so this is $\le 10^{-9}$ relative.
- Elements are on-axis hard-edge/phase models for now; oblique interaction
  (incidence-angle dependence) deferred.
- Dispersion: $\lambda$ is fixed per `BandField`; dispersive materials enter
  through `Material` closures in later phases.

## 11. Verification mapping (`tests/asm.rs`, `tests/phase2.rs`)

| # | Test | Physics statement | Analytic anchor | Tolerance |
|---|---|---|---|---|
| 1 | Energy conservation | NumPy scalings cancel in ASM; $\lvert H\rvert = 1$ on band | discrete power invariant | rel. power $\le 10^{-9}$ |
| 2 | Plane-wave invariance | single spectral bin gains only a phase | $e^{ikz}$ (Eqs. 3.5–3.13) | $10^{-6}$ |
| 3 | Tilted plane wave | ramp slope unchanged by propagation | $k\alpha$, $\alpha = \lambda f_x$ | $10^{-6}$ |
| 4 | Airy far field (ASM) | FT of circular aperture at $N_F = 0.05$ | $[2J_1(v)/v]^2$; $r_1 = 1.22\lambda z/2a$ | profile $\le 2\%$ of peak; $r_1$ $\le 2\%$ |
| 5 | Sampling guard | undersampled chirp detected, mitigated, or raised | worst step $\approx 19$ rad $> \pi$; $\propto 1/$padding | exact variant match |
| 6 | Round trip $+z/-z$ | $H \cdot H^* = 1$ on pass band | identity | rel. Frobenius $< 10^{-6}$ |
| 7 | Gaussian beam $q(z)$ | ASM and Fresnel TF agree with the paraxial beam solution and each other | $w(z)$, $w_0/w$, Gouy $\arctan(z/z_R)$ | $10^{-3}$ |
| 8 | Straight edge | near-field edge diffraction (Fresnel zones) | $I(u)=\tfrac12[(\tfrac12-C)^2+(\tfrac12-S)^2]$ | $2.5\%$ of $I_0$ |
| 9 | Rectangular aperture | far field of a rect | $\operatorname{sinc}^2(\pi W x/\lambda z)$ | $3\%$ of peak |
| 10 | Airy three engines | regime-complementary engines vs one anchor | $[2J_1(v)/v]^2$ at $N_F = 0.05$; TF-guard-fails/kernel-holds asserted | $2\%$–$3\%$ |
| 11 | Regime ladder | recommendations are executable | engine choice per §8 ladder | propagates cleanly |

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
