# Physics — Wave-Optics Propagation and Inverse Design

This document details the mathematical models, transfer functions, and
validity regimes implemented in Phases 1–7 (ASM exact, in vacuum or
refractive media; Fresnel transfer-function and convolution forms;
Fraunhofer far field; Rayleigh–Sommerfeld first kind; regime guidance;
thin elements — apertures, dispersive thin lens, phase-mask DOE interface
(analytic and sampled), blazed grating — refractive surfaces and thick
lenses with Zemax standard-surface profiles, wedge prisms; and sources:
Gaussian/Hermite-Gauss/Laguerre-Gauss beams and the point-source
spherical wave, composed into PSF pipelines by `beamline`) and, from
Phase 9, the inverse-design physics of the `optics-inverse` crate —
Gerchberg–Saxton DOE design (§12) and adjoint gradients through the
render chain (§13).
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

## 6. Fresnel and Rayleigh–Sommerfeld propagation (Phases 2, 4)

Two discrete forms of the paraxial propagator (§§6.1–6.2), related by the
convolution theorem and complementary in sampling, plus the exact
spherical-kernel parent both reduce to (§6.3). Expanding the ASM direction
cosine $\gamma = \sqrt{1-s} \approx 1 - s/2$ with $s = (\lambda f_x)^2 +
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

### 6.3 Rayleigh–Sommerfeld first kind (Phase 4)

`rayleigh_sommerfeld::propagate` advances the field with the *exact*
spherical impulse response rather than its paraxial (Fresnel) reduction.
Textbook Sec. 2.1 derives the first-kind RS integral (Eq. 2.8); written in
this crate's $e^{+ikz}$ convention the impulse response is

$$h(x, y; z) = \frac{|z|}{2\pi r^2}\!\left(\frac{1}{r} - i\,\mathrm{sgn}(z)\,k\right)
   e^{\,i\,\mathrm{sgn}(z)\,kr}, \qquad
   r = \sqrt{x^2 + y^2 + z^2}, \qquad k = \frac{2\pi}{\lambda},$$

where the $\big(\tfrac{1}{r} - i\,\mathrm{sgn}(z)k\big)$ factor is the
**obliquity** carried by the Dirichlet Green's function. Propagation is the
spatial convolution $U_z = U_0 \circledast h$, evaluated on the padded grid
by the same embed/DFT-order-kernel/extract pipeline as §6.2.

**Why first kind, not second.** The exact Kirchhoff surface integral couples
the boundary field $u$ with its unknown normal derivative $\partial u/\partial
n$. The antisymmetric (first-kind) image Green's function $G_- = 0$ on the
plane drops the $\partial u/\partial n$ term, so the formula consumes *only
the sampled field values* — the data a `BandField` actually carries. The
second kind needs $\partial u/\partial n$; estimating it as $\approx -iku$
would reintroduce exactly the inconsistent-boundary approximation RS theory
exists to avoid. RS-II is therefore deliberately excluded.

**Not new physics — a third discretization.** With the first-kind kernel the
transfer function is exactly the ASM's $e^{ikz\gamma}$ (the book makes the
identification in Sec. 2.2.2/3.2: RS propagation *is* the plane-wave
spectrum). The engine's value is a spatial-sampled counterpart to ASM's
spectral one: RS $\leftrightarrow$ ASM agreement validates both discretizations,
and RS $\leftrightarrow$ Fresnel quantifies the paraxial truncation — the
Fresnel report's `max_phase_error_vs_asm` predicts exactly where the paraxial
model diverges from the RS truth (`tests/phase4.rs`).

**Paraxial reduction.** With $r \approx |z|$ and $1/r \ll k$ the kernel
becomes $\frac{e^{ikz}}{i\lambda z}\exp\!\big(i\pi\frac{x^2+y^2}{\lambda z}\big)$
— the §6.2 Fresnel kernel (textbook Eq. 2.24), pinned by the unit test
`kernel_reduces_to_fresnel_paraxially`.

**Kernel sampling (guard).** The kernel phase is $\phi = \mathrm{sgn}(z)kr -
\arctan(kr)$; along one axis the geometric term is convex in $|x|$, so the
worst adjacent-sample step is closed-form at the Nyquist-straddling pair
$x_{hi} = \tfrac{n_{\text{pad}}}{2}\Delta x$, $x_{lo} = x_{hi} - \Delta x$,
with the obliquity gradient added as a bound:

$$\Delta\phi_x^{\max} = k\big(r(x_{hi}) - r(x_{lo})\big) + \frac{k\,\Delta x}{1 + k^2 z^2} < \pi.$$

Because $|x|/r \le 1$, this step is **always smaller than the Fresnel
convolution step** of §6.2 (the paraxial flattening grows the phase gradient
without bound; the spherical $kr$ saturates at $k\sin\theta \le k$) — the RS
form tolerates wider padded grids than its paraxial sibling. Densification
again *worsens* kernel sampling, so `strict` mode raises `AliasingRisk`
immediately. $z = 0$ is **rejected** (the kernel is a $\delta$-distribution
there; use ASM for a zero step). Signed-$z$ mirror $h(\cdot,-z) =
\overline{h(\cdot,z)}$ is bit-exact by construction.

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
  physically relevant source size — e.g. `EllipticalAperture::radius()`),
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

The Phase-4 Rayleigh–Sommerfeld engine (§6.3) is **not** in the ladder: it
carries the convolution form's kernel-sampling condition (fails at short
$|z|$; ASM remains the exact fallback there), and its role is
cross-validation — a third discretization of the same physics, plus the
exact-spherical kernel where the paraxial truncation matters while the
geometry stays guard-passing.

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

### 9.1 Orientation convention (Phase 3, task 3.e.1)

Every element carries one in-plane rotation $\psi$ about the optical axis
$z$: **positive $\psi$ is counterclockwise viewed from downstream** —
looking back along $-z$ against the oncoming beam, $+x$ right, $+y$ up
(right-hand rule about $+z$, the propagation direction); viewed from
upstream along $+z$ the same rotation appears clockwise. Pattern *and*
stop are defined in the element frame

$$\mathbf{r}' = R(-\psi)(\mathbf{r} - \mathbf{c}), \qquad
  R(\psi) = \begin{pmatrix}\cos\psi & -\sin\psi\\ \sin\psi & \cos\psi\end{pmatrix},$$

so rotating the element rotates its transmission function together with
its clear aperture. Elements with a *typical* orientation (flat, normal to
$z$, intrinsic axes aligned with the lab axes) accept `Option<Angle>`
(`None` $\equiv \psi = 0$); the grating-vector direction is intrinsic
design data, so `BlazedGrating` takes a mandatory `Angle`. `Angle` wraps
implicitly into $[0, 2\pi)$; non-finite orientations are rejected by the
constructors (`InvalidOrientation`). Mirrors (not yet implemented) must
adopt the same pattern. Orientation is purely in-plane — oblique
incidence-angle dependence remains deferred.

### 9.2 Thin lens (Phase 3, task c)

An ideal thin lens is a pure phase screen, in the element frame,

$$t(x, y) = \exp\!\Big(-\,i\,\frac{2\pi}{\lambda}\,\frac{\rho^2}{2 f(\lambda)}\Big),
  \qquad \rho^2 = x'^2 + y'^2,$$

with converging $f > 0$ carrying the **negative quadratic** in this crate's
$e^{+ikz}$ forward-propagation frame — the same convention as the
converging curvature of `sources` (§9.5). Textbook construction (Sec. 4.2.1):
a plane wave must gain $2\pi$ of phase at the zone radii
$r_p^2 = 2fp\lambda$ (Eq. 4.13, from Eq. 4.12 dropping $(p\lambda)^2 \ll f^2$)
to focus at $f$ — exactly this quadratic; the wrapped kinoform version of the
profile is a `PhaseMask` closure (§9.3). The focal length is a **dispersive
closure $f(\lambda)$** (decision D1): refractive glass has power
$\propto (n-1)c$ with $n$ higher at blue, so $f$ shortens at blue
(Sec. 4.1.3); a diffractive surface keeps $f \propto \lambda$ at fixed
phase. Both are just closures — the lens API does not change. The
`ClearAperture` stop (task-c revision) bounds the footprint and drives
regime gating through the lens's own half-extent; `ClearAperture::Field`
reproduces the field-wide phase screen.

**Fourier property (validated).** A Gaussian waist $w_0$ at the lens plane,
acted on by the lens and propagated exactly $f$, focuses to the
transform-limited waist $w_0' = \lambda f/(\pi w_0)$ with on-axis amplitude
scaling $w_0/w_0'$ — verified to 1% through both ASM and Fresnel
convolution (`focal_plane_reproduces_fourier_property`).
`worst_phase_step(band)` reports $\frac{k\delta}{|f|}\big(r - \frac{\delta}{2}\big)$
($r$ = farthest acting coordinate along the measured axis) as a closed-form
$O(1)$ sampling diagnostic — reported, not enforced (elements stay local).

### 9.3 Phase mask — the DOE interface (Phase 3, task d)

`PhaseMask` paints an analytic closure $\varphi: (x', y') \mapsto$ radians
(**center-relative element-frame** coordinates, so DOE designs are
translation-invariant) as $t = e^{i\varphi}$ inside a shared
`ClearAperture` stop; `kind = Diffractive`. The quadratic closure is the
Sec. 4.2.1 DOE lens and equals `ThinLens` sample-exactly (validated).
There is **no engine-side sampling guard**: an arbitrary profile admits no
closed-form worst step, and $O(N^2)$ warning sweeps are against project
guidelines — the caller-side Nyquist duty is $|\nabla\varphi|\cdot\delta < \pi$.
The first non-finite closure value raises
`InvalidPhase { x, y, phase }` with its element-frame position. The
**sampled-profile variant** (Phase-3 decision D2, delivered Phase 5 task b)
is `PhaseMask::sampled(values, dims, pitch, orientation, aperture, center)`:
phase samples on a **center-origin rectangular lattice in the element
frame** (the same convention as `Field`, §1) with bilinear lookup, **zero
phase beyond the lattice span** (transparent within the aperture). Bilinear
interpolation is exact for a linear profile and reproduces the DOE-lens
quadratic at every lattice point (validated); the caller-side Nyquist duty
becomes resolving the profile on the lattice. Structural errors are
rejected at construction (`InvalidGrid`, `InvalidGridPitch`), and a
one-time $O(N)$ scan rejects non-finite samples (`InvalidPhase` at the
lattice position).

### 9.4 Blazed grating and diffraction orders (Phase 3, task e)

Sawtooth phase screen with period $\Lambda$, depth $\Phi$ (radians per
period), and grating-vector direction $\hat g = (\cos\psi, \sin\psi)$:

$$\varphi(x, y) = \Phi \cdot \Big(\frac{(\mathbf{r}-\mathbf{c})\cdot\hat g}{\Lambda} \bmod 1\Big).$$

Textbook Eq. 2.25 is exactly this linear phase of slope $\Phi/\Lambda$ in
one period (the depth-to-phase relation $\Phi = 2\pi(n-1)d/\lambda$ is
fabrication-side). Periodic structure $\Rightarrow$ discrete orders at the
grating angles

$$\Lambda \sin\theta_m = m\lambda \qquad \text{(Eqs. 1.1, 2.26, 5.1)},$$

with efficiencies the squared Fourier coefficients
$\eta_m = \operatorname{sinc}^2(\pi(k - m))$, $k = \Phi/(2\pi)$ (Eq. 2.27).
An **integer-$2\pi$ depth** ($k = m$ integer) puts *all* power into order
$m$ — the kinoform blaze (Sec. 2.4.2: $\eta = 1$ while depth $\ll$ period;
shadowing degrades it as depth approaches $\Lambda$, Sec. 2.5). On the grid
this is a kinematic identity, not an approximation: for $\Phi = 2\pi m$ the
mod-$1$ wrap is invisible and $e^{i\varphi}$ is **exactly** the tilted
plane wave $e^{i 2\pi m (x'\cos\psi + y'\sin\psi)/\Lambda}$, so the uniform-
illumination far field is the pure order $m$ at $\sin\theta_m = m\lambda/\Lambda$.

Validated through the Fraunhofer engine (`tests/phase3.rs`, §11 rows
12–14): $\Phi = 2\pi$ concentrates $>85\%$ in the order $+1$ lobe at
$x_1 = \lambda z/\Lambda$ with $<3\%$ leakage to orders $0/-1$;
$\Phi = 4\pi$ steers $>80\%$ into order $+2$ at $x_2 = 2\lambda z/\Lambda$;
$\Phi = \pi$ splits $\approx 40/40$ between orders $0$ and $+1$ with order
$-1$ weak — matching $|c_m|^2 = 4/(\pi^2(1-2m)^2)$.
`worst_phase_step(band)` $= \frac{\Phi}{\Lambda}\max(|\cos\psi|\Delta x,
|\sin\psi|\Delta y)$ (ramp bound; $\Phi = 2\pi$, $\psi = 0$ gives the
period-Nyquist $\Delta x < \Lambda/2$). Full rotated-order and binary-
amplitude coverage is deferred with the corresponding features (task-e
decision 2026-10-12).

### 9.5 Gaussian beam source (Phase 3, task a)

`sources::gaussian` samples the paraxial fundamental mode in the $q$-form
adapted to this crate's $e^{+ikz}$ convention (Siegman/Goodman; the project
textbook uses Gaussian beams only as beam-shaper input, Sec. 11.2, and does
not restate the $q(z)$ solution — hence the pin below):

$$u(\rho; z) = A\,\frac{w_0}{w(z)}\,e^{-\rho^2/w(z)^2}\,
  e^{\,i k\rho^2/(2R(z))}\,e^{\,i(kz - \zeta(z))},
  \qquad u \propto e^{\,ik\rho^2/(2q)}, \quad q(z) = z - i z_R,$$

(the conjugate of the usual $e^{-ikz}$-textbook $q = z + iz_R$) with
$z_R = \pi w_0^2/\lambda$, $w(z) = w_0\sqrt{1 + (z/z_R)^2}$,
$1/R(z) = z/(z^2 + z_R^2)$ (finite at $z = 0$), $\zeta(z) = \arctan(z/z_R)$.
$z$ is the **signed waist-to-sampling-plane distance** — a system
placement argument, not a stored beam property (task-a review): $z = 0$
samples the waist (pure real Gaussian), $z > 0$ diverging, $z < 0$
converging, and $u(\rho;-z) = \overline{u(\rho;z)}$; the converging
($z<0$) curvature is the same-sign convention as the converging thin lens
of §9.2. Sign conventions (Gouy, curvature) are the historical bug source:
they are pinned against the crate's own validated engines — Fresnel-TF and
ASM propagation of a waist beam reproduce the analytic initializer at $z$
(unit tests; the on-axis phase $kz - \zeta(z)$ was pinned by `tests/phase2.rs`
test 7).

### 9.6 Higher-order modes and the point source (Phase 5, tasks a, d)

`sources::hermite_gauss` / `sources::laguerre_gauss` sample the paraxial
$\mathrm{HG}_{mn}$ / $\mathrm{LG}_{p\ell}$ families on the same $q$-form
machinery (Siegman/Goodman — the project textbook does not cover them):

$$u_{mn} \propto \frac{H_m(\sqrt2 x/w)\,H_n(\sqrt2 y/w)}{w(z)}\,
  e^{-\rho^2/w^2}\,e^{\,ik\rho^2/2R}\,e^{\,i(kz - (m+n+1)\zeta)},$$

$$u_{p\ell} \propto \frac{(\sqrt2 r/w)^{|\ell|}\,L_p^{|\ell|}(2r^2/w^2)}{w(z)}\,
  e^{-r^2/w^2}\,e^{\,ikr^2/2R}\,e^{\,i\ell\varphi}\,
  e^{\,i(kz - (2p+|\ell|+1)\zeta)},$$

physicists' $H_m$ and associated $L_p^\alpha$ by three-term recurrences
(dependency-free). Conventions: **orthonormal profiles** (mode power $= A^2$,
contrasting `GaussianBeam`'s peak-amplitude scale; $\mathrm{HG}_{00} =
\mathrm{LG}_{00} = $ `gaussian` $\times\sqrt{2/\pi}/w_0$, pinned bit-exactly);
Gouy phase $(N+1)\zeta$ with total order $N$; LG azimuthal $e^{i\ell\varphi}$
carries OAM $+\ell\hbar$ per photon along $+z$ in the crate's $e^{+ikz}$
frame. The orders are capped at `MAX_MODE_ORDER = 64` (`InvalidMode`).
Convention pins: HG/LG at the waist propagate through **Fresnel-TF and ASM**
to the analytic mode at $z$ ($<10^{-3}$, the eigenfunction tests), plus
on-grid orthonormality, $(-1)^m$ reflection parity, the LG ring radius
$w_0/\sqrt2$, exact $2\pi\ell$ vortex winding, and the $(N{+}1)$ Gouy
tripling on axis (with the $H_2(0) = -2$ phase flip).

`sources::spherical` samples the diverging point-source wave
$u = A\,e^{ikr}/r$, $r = \sqrt{(x-x_s)^2 + (y-y_s)^2 + d^2}$ — the exact
Helmholtz solution at a plane distance $d$ downstream of the source
(Huygens secondaries, Sec. 1.4, as an initializer; $d \le 0$ rejected,
`InvalidSource`). Near the axis the phase is $kd + k\rho^2/2d + \cdots$:
**positive** curvature (diverging), the mirror of the converging lens.
Caveat: a spherical wave is nonzero at the window edge, so ASM/Fresnel
propagation of the *sampled* wave equals the analytic wave at $d+z$ only
up to hard-edge truncation diffraction ($\sim\sqrt{\lambda z}$ of spread
from the window boundary) — the cross-engine statement is ASM $\leftrightarrow$
Fresnel-TF on the same input (validated $<10^{-3}$); the absolute
convention pins are the exact-$kr/r$ anchors and the near-axis expansion.

**PSF composition** (`beamline`): tracing a point source or tilted plane
wave through `(element, z, medium)` steps and taking $I = |u|^2$ is the
coherent PSF — classical imaging anchors validated in `tests/phase5.rs`
(§11 rows 20–22): Airy of the pupil stop, focal shift $x = f\tan\theta$ of
a tilted plane wave, and the lens-equation image $d_i = f d_o/(d_o - f)$
with magnification $-d_i/d_o$ for an off-axis point.

### 9.7 Refractive surfaces, thick lenses, and media (Phase 6)

`RefractiveSurface` is the Zemax-import primitive: a curved boundary
between media $n_1$ (front) and $n_2$ (back), modeled as a local phase
screen at its vertex plane,

$$\varphi(\rho; \lambda) = \frac{2\pi}{\lambda}\,[n_1(\lambda) - n_2(\lambda)]\,z(\rho),$$

with $z(\rho)$ the surface sag in the **Zemax standard-surface form**
(rational conic + even polynomial, numerically stable at the vertex):

$$z(r) = \frac{c\,r^2}{1 + \sqrt{1 - (1+k)c^2 r^2}} + \sum_{i\ge2}\alpha_i r^{2i}, \qquad c = \frac{1}{R},$$

($k=0$ sphere, $k=-1$ paraboloid; the paraxial sag $r^2/2R$ is the book's
"sag equation", Sec. 4.2.2, Eq. 4.18). Sign convention: $R > 0$ bulges
toward $+z$ (direction of propagation); a convex surface into a denser
medium converges (the OPL argument gives the $n_1 - n_2$ sign; the
paraxial quadratic is $-k\rho^2\Phi/2$ with standard surface power
$\Phi = (n_2 - n_1)c$, Eqs. 4.3/4.21). Vertex piston is zero; the stop,
orientation, and `worst_phase_step` follow the element conventions
(§9.1–9.2). Construction rejects profiles whose conic domain is breached
by the stop (`InvalidSurface`); runtime rejects non-finite sags
(`InvalidSag`).

`ThickLens` is the single-plane convenience form: the glass thickness
profile $t(\rho) = t_0 - z_1(\rho) + z_2(\rho)$ painted as one screen,
$\varphi = k(n-1)t(\rho)$ — textbook Eq. 4.17 generalized to exact sag
profiles. The vertex term $k(n-1)t_0$ is the on-axis glass OPD (Eq. 4.17
itself); the paraxial quadratic is the thin-lens law
$1/f = (n-1)(c_1 - c_2)$ (Eq. 4.2); the $r^4$ sag terms give real
spherical aberration (marginal rays focus short, Sec. 4.1.5 — pinned
quantitatively: the paraxial-vs-exact phase residual at $\rho = 2$ mm is
0.025 rad in the unit test). **What the single-plane model cannot carry:**
principal-plane separation and the thickness term of the full lensmaker
Eq. 4.1 — those need propagation *inside* the glass, so the primary
thick-lens model is the composition

$$\text{surface}_1 \;\to\; \texttt{gap}(t_0, n) \;\to\; \text{surface}_2,$$

which reproduces Eq. 4.1 including the $t(n-1)c_1c_2/n$ term (validated
against the back-focal-length wavefront-curvature fit, §11 row 23 — the
measured BFL lands within 1% of the matrix-optics value, 5.4% away from
the thin-lens law). This is also the Zemax-faithful path: a prescription
row is exactly (sag, thickness, glass).

**Medium-aware ASM** (`AsmConfig::medium`, $n$): inside a medium of index
$n$ the wavenumber is $nk$ and the direction cosine of bin $f$ is
$\lambda f/n$, so the transfer function is the vacuum machinery with
$\lambda \to \lambda_n = \lambda/n$:
$H_n = \exp\!\big(i\,nkz\sqrt{1 - (\lambda f_x/n)^2 - (\lambda f_y/n)^2}\big)$;
the evanescent cutoff rises to $n/\lambda$ and the Matsushima band-limit
scales by $n$ (the transfer function gets *smoother* in frequency — the
paraxial chirp coefficient scales as $\lambda^2/n$). The slab anchor: a
normal-incidence plane wave through thickness $t$ picks up $e^{ink t}$ —
optical path $nt$, piston $k(n-1)t$ relative to vacuum (validated
bit-exactly).

`Wedge` (prism): two plane surfaces at apex angle $\alpha$, thickness
$t = t_0 - \tan\alpha\,(\mathbf{r}'\cdot\hat g)$ along the mandatory apex
direction $\hat g$ — a linear OPD ramp deflecting a plane wave to
$\sin\theta = -(n-1)\tan\alpha$ toward the base (exact as a plane-wave
tilt; validated in the far field, §11 row 25).

**Hybrid achromats** (Sec. 4.4.1): a refractive power (blue-stronger,
Sec. 4.1.3) paired with a wavelength-independent DOE phase screen whose
equivalent power goes $\Phi_d \propto \lambda$ (blue-weaker) cancels
first-order color at the design wavelength — same-sign powers allowed,
unlike the all-refractive doublet of Eq. 4.11 (validated: <1% residual
power drift over 500–600 nm vs >4% uncorrected, §11 row 24).

## 10. Validity regime and limitations

- Scalar theory; polarization stays a `Wave`-level property (no vector field
  grids yet).
- Exact Helmholtz propagation **within the sampled band**: evanescent
  components dropped; transfer-function aliasing guarded as in §5.
- Fresnel forms (§6) add the paraxial restriction (truncation error
  reported); Fraunhofer (§7) additionally requires the far-field regime
  ($N_F \le$ threshold) — the engines enforce or report these per geometry.
- Rayleigh–Sommerfeld first kind (§6.3) keeps the exact spherical phase but
  is still scalar theory under the book's assumptions (7)–(8): aperture and
  $|z|$ large compared with $\lambda$ (it is not a vector/Richards–Wolf
  model), and the sampled-kernel discretization carries its own window
  truncation ($\sim 1/\text{zones}$, zones $= R^2/(\lambda|z|)$ at the
  padded rim).
- Wraparound controlled by padding + band-limiting; energy leaving the
  physical window (crop) is lost from the returned field — the validation
  suite sizes grids so this is $\le 10^{-9}$ relative.
- Elements are hard-edge/phase models with centered placement and in-plane
  orientation (§9.1); oblique interaction (incidence-angle dependence)
  deferred.
- Dispersion: $\lambda$ is fixed per `BandField`; thin-lens chromaticity
  lives in the $f(\lambda)$ closure (§9.2); refractive screens consume
  `Material` $n(\lambda,\text{angle})$ closures (angle reserved), with
  `Material::sellmeier` the standard Book-1 dispersion mechanism (§9.7);
  the grating depth-to-phase relation (§9.4) is fabrication-side, not
  applied.
- Sources (§9.5–9.6) are paraxial modes ($N \le 64$ cap) and the scalar
  spherical wave; mode windows must cover $\sim\sqrt{N}\,w(z)$, and the
  spherical wave's hard window edge bounds any absolute-field comparison
  (§9.6 caveat). PSF work is headless: intensity grids only — plotting and
  serialization live in the front-end project (Phase-5 decision 1).
- Refractive surfaces (§9.7) are scalar normal-incidence phase screens:
  no Fresnel amplitude coefficients, no polarization, no angle-dependent
  transmission; valid for small angles and $|n-1| \ll 1$-scale phase
  strengths. Mirrors (need an axis fold) and waveplates (need vector
  fields) are deliberately deferred (Phase-6 decision).

## 11. Verification mapping (`tests/asm.rs`, `tests/phase2.rs`, `tests/phase3.rs`, `tests/phase4.rs`, `tests/phase5.rs`, `tests/phase6.rs`)

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
| 12 | Kinoform blaze (phase3) | $\Phi = 2\pi$ sawtooth $\to$ pure order $+1$ at $x_1 = \lambda z/\Lambda$ | Eqs. 1.1/2.26/5.1; Sec. 2.4.2, Eq. 2.27 $k=1$ | lobe $\pm 2$ bins; $>85\%$ in order, $<3\%$ leak to $0/-1$ |
| 13 | Double-depth blaze (phase3) | $\Phi = 4\pi \to$ order $+2$ at $x_2 = 2\lambda z/\Lambda$ | Eq. 2.27 $k=2$ | $>80\%$ in lobe |
| 14 | Half-wave sawtooth (phase3) | $\Phi = \pi \to \approx 40/40$ split orders $0/+1$, $-1$ weak | $|c_m|^2 = 4/(\pi^2(1-2m)^2)$ | each $0.30$–$0.55$; $f_{-1} < 0.10$ |
| 15 | FFT vs direct quadrature (phase4) | circular (engine) $=$ linear (sum) up to guard-controlled wrap | RS-I integral, Eq. 2.8 kernel | rel Frobenius $< 10^{-2}$ |
| 16 | Three discretizations agree (phase4) | spatial-kernel (FFT + direct) $=$ spectral-transfer (ASM) — RS *is* the PWS | Sec. 2.2.2/3.2 identification | pairwise $< 2\cdot10^{-2}$ |
| 17 | Fresnel diverges where reported (phase4) | paraxial truncation vs RS truth at aperture-rich corner bins | $k|z|s^2/8 \approx 0.29$ rad; report `max_phase_error_vs_asm` | RS $< 2\cdot10^{-2}$; Fresnel $> 4\times$ RS error, $< 0.3$ |
| 18 | Round trip $+z/-z$ (phase4) | conjugate spherical kernel inverts the operator | $h(\cdot,-z) = \overline{h(\cdot,z)}$ | rel $< 3\cdot10^{-2}$ ($\sim 1/$zones ripple) |
| 19 | Power conservation (phase4) | $\lvert H_{RS}\rvert = 1$ on band | discrete power invariant | rel drift $< 10^{-3}$ |
| 20 | PSF is Airy (phase5) | stopped pupil traced through `Beamline::psf` = circular-aperture far field | $[2J_1(v)/v]^2$; $r_1 = 1.22\lambda z/2a$ | full profile $< 3\%$, central lobe $< 2\%$, $r_1$ $\pm 2$ bins |
| 21 | Focal shift vs field angle (phase5) | tilted plane wave through lens focuses at $x = f\tan\theta$ | $1.22\lambda f/2a$ spot, centroid estimator | centroid $< 20\,\mu$m; on-axis $\to$ origin |
| 22 | Point source images geometrically (phase5) | off-axis spherical wave through lens $\to$ image at $d_i = fd_o/(d_o-f)$, $m = -d_i/d_o$ | lens equation + magnification | centroid $< 20\,\mu$m; peak $> 10\times$ mean |
| 23 | Thick stack = Eq. 4.1 BFL (phase6) | surface $\to$ medium gap $\to$ surface focuses at the full thick-lens back focal length | Eq. 4.1 (bracketed, incl. $t(n-1)c_1c_2/n$) + matrix BFL | fit within 1%; $>50\%$ of the 2.16 mm thin-law gap |
| 24 | Hybrid achromat (phase6) | refractive + fixed-DOE screen cancel first-order color | Sec. 4.4.1; $\Phi_d = -\lambda_0\,d\Phi_r/d\lambda$ | corrected $<1\%$ over 500–600 nm vs $>4\%$ uncorrected |
| 25 | Wedge deflects to base (phase6) | far-field centroid at $\sin\theta = -(n-1)\tan\alpha$ | prism law via plane-wave tilt | $< 10^{-4}$ m at $z = 2$ m |

Sources/mode physics additionally pinned by unit tests in `src/sources.rs`:
HG/LG eigenfunction propagation through Fresnel-TF and ASM ($<10^{-3}$),
on-grid orthonormality, $(-1)^m$ parity, LG ring $w_0/\sqrt2$ and $2\pi\ell$
vortex winding, $(N{+}1)$ Gouy on axis, $\mathrm{HG}_{00}=\mathrm{LG}_{00}=$
`gaussian`$\times\sqrt{2/\pi}/w_0$, spherical-wave $kr/r$ anchors and the
diverging $+k\rho^2/2d$ sign, and the `Beamline::psf` intensity/geometry
bookkeeping (`src/beamline.rs`).

Element/source physics additionally pinned by unit tests in `src/element.rs`
and `src/sources.rs`: thin-lens Fourier property ($w_0' = \lambda f/\pi w_0$,
1%, ASM + Fresnel convolution), DOE-lens closure = `ThinLens` sample-exact,
integer-$2\pi$ grating depth = tilted plane wave sample-exact, orientation
defaults/validation/rotated stops, `Angle` $[0,2\pi)$ wrapping, Gaussian
$q$-form cross-engine agreement ($<10^{-3}$) and $u(-z) = \overline{u(z)}$.
RS physics additionally pinned in `src/propagation/rayleigh_sommerfeld.rs`:
paraxial kernel reduction $=$ Fresnel kernel ($<10^{-6}$), bit-exact
conjugate mirror, delta input $=$ sampled kernel ($10^{-12}$), guard
closed-form brackets a brute-force sweep, RS $\equiv$ ASM on a smooth band.
Phase-6 additions: Zemax standard-sag golden values and the exact
geometric-sphere cross-check, surface power / plano-convex $f = R/(n-1)$
(Eq. 4.21), ThickLens vertex piston (Eq. 4.17) and the quantified $r^4$
spherical-aberration residual, conic-domain and media validation, slab
piston $e^{ink t}$ and band-limit/guard scaling with $n$, wedge ramp
gradient and deflection sign.
Inverse-design physics additionally pinned by unit tests in
`crates/optics-inverse/src/gs.rs` and `src/adjoint.rs`: GS convention
pins (ASM round-trip identity, half-transform composition, converging-
lens sign, blaze steering), the top-hat shaper golden (flat interior,
monotone best-so-far Eq. 5.18 MSE), recovery of a known phase-only lens
from its target-plane intensity, DVR/FPR convergence, the adjoint
inner-product identity, adjoint-vs-FD grid-gradient agreement (global
relative error $< 10^{-4}$), and the GS-error-signal / adjoint-gradient
correlation ($> 0.9$).

## 12. Gerchberg–Saxton phase retrieval / DOE design (`optics-inverse`, textbook §5.5)

The *inverse* problem: given coherent illumination $U_{\text{in}}$ at an
input plane and a target-plane amplitude $T(x, y)$, find a phase-only DOE
that delivers $T$. The DOE (a `GridPhase` carrier, §9.3 sampled variant)
sits at the midpoint of an input→target separation $z_{\text{total}} > 0$,
so the coherent relay is **two ASM half-transforms** (§3) with the element
between them:

$$U_t = \mathcal{A}(z/2)\big\{A\,\Pi\,e^{i\varphi}\big\}, \qquad
  A\,e^{i\psi} = \mathcal{A}(z/2)\{U_{\text{in}}\},$$

where $\Pi$ is the carrier's clear-aperture indicator and $A\,e^{i\psi}$
the illumination arriving at the DOE plane — computed once, since its
amplitude $A$ (masked by $\Pi$) is the DOE-plane energy constraint. Each
iteration (Eqs. 5.14–5.17):

1. **Forward half-transform** of the current iterate
   $A\,\Pi\,e^{i\varphi_k}$ to the target plane.
2. **Fourier-plane (target) projection** — Eq. 5.16: replace the
   amplitude, keep the phase, $U_t' = T\,U_t/|U_t|$; the `Fpr` variant
   relaxes the replacement,
   $|U_t| \to (1-\beta)\,|U_t| + \beta\,T$, $\beta \in (0, 1]$.
3. **Back-propagation** $\mathcal{A}(-z/2)$ to the DOE plane, giving $V$
   — the exact ASM adjoint on the passband (§13).
4. **DOE-plane projection** — re-impose the illumination amplitude and
   update only the phase,
   $\varphi_{k+1} = \varphi_k + \beta\,\Delta\varphi$ with
   $\Delta\varphi = \operatorname{wrap}_{[-\pi,\pi)}(\arg V - \varphi_k)$;
   the `Dvr` variant damps/over-relaxes with $\beta \in (0, 2)$.

`Classic` is the textbook algorithm ($\beta = 1$ in both projections).
The design merit is the Eq. 5.18 amplitude MSE
$M = \frac{1}{N}\sum(|U_t| - T)^2$; the exposed error signals are the
target-plane amplitude residual $|U_t| - T$ and the DOE-plane phase
residual $\Delta\varphi$ — the negative-gradient direction of $M$
preconditioned by the illumination intensity (§13). Convergence and
stopping run through the crate `Merit` seam: each iteration's
target-plane intensity is wrapped as a single-band/single-scene
`Observation` and scored, so any merit can terminate the loop.

**Reachability:** a phase-only DOE conserves power, so $T$ must be
scaled to the illumination ($T^2$ summed over the target support equals
$\sum|U_{\text{in}}|^2$); a target demanding more energy than the
illumination carries is unreachable by construction — the top-hat golden
scales its flat top this way.

**Conventions:** single band, fully coherent (multiband shaping is out of
scope); transmission $e^{+i\varphi}$ with the converging-lens sign
$\varphi(r) = -k r^2/(2f)$ focusing at $+z = f$; FFT scaling/sign/bin
order inherited from §4.1; the phase vector is encoded into the
carrier's slot bounds every iteration, so the iterate is always a legal
`GridPhase` parameter vector.

## 13. Adjoint gradients through the render chain (`optics-inverse`)

The forward model of an `optics-inverse` render is a chain of exactly two
primitive kinds, both **linear operators on the complex field**: ASM
propagation (§3) and thin elements (pointwise complex multiply, §9).
Because every primitive is linear, its adjoint (conjugate transpose) is
another primitive of the *same kind* — the design gradient is obtained by
pushing the merit's observation gradient **backward through the forward
primitives in reverse order**, with no symbolic or graph-based autodiff.
Full derivation, conventions, and rejected alternatives:
[`adr-9.9-adjoint-gradients.md`](adr-9.9-adjoint-gradients.md).

- **ASM propagation:**
  $\mathcal{A}(z)^\dagger = \mathcal{A}(-z)$ with the *same* config —
  the transfer phase conjugates, the evanescent and band-limit masks are
  real and even in $z$, and the zero-pad / center-crop are each other's
  adjoints under the §4.1 FFT convention (test
  `asm_adjoint_inner_product_identity`).
- **Thin element:** $T = \operatorname{diag}(t)$, so $T^\dagger$ is the
  pointwise **conjugate** multiply $\bar{t} \odot (\cdot)$. An aperture
  is the case $t = \Pi e^{i\varphi}$ with $\Pi \in \{0, 1\}$ — blocked
  pixels pass no adjoint energy, so the aperture is self-adjoint in the
  energy sense. The transmission is recovered without the element
  exposing it: `apply` is pointwise, so $t = \texttt{apply}(\mathbf{1})$
  on a unit field of the plane's geometry.
- **Intensity seed:** the observation is $I_k = |U_k|^2$, so for a real
  merit $M$ the detector-plane adjoint source is
  $w_k = (\partial M/\partial I_k)\,U_k$ and
  $dM = 2\,\Re\langle w, dU\rangle$.

With $\mu$ the adjoint field pushed back to the plane **after** element
$j$ and $V = t \odot U$ the tape field after it, perturbing one lattice
phase injects $dV_p = i\,V_p\,d\varphi_p$, hence

$$\frac{\partial M}{\partial \varphi_p} = -2\,\Im\!\big(\bar{\mu}_p\,V_p\big).$$

**Cost:** $dM/d\text{grid}$ takes one forward render plus one adjoint run
per gradient-carrying (band, scene) frame — independent of the grid
length (contrast: FD-on-parameters costs $2\cdot\text{dim}$ renders per
step, §12's GS needs no gradient at all).

**GS relation (acceptance link):** the GS phase error signal of §12 is
the negative adjoint gradient of the Eq. 5.18 amplitude MSE,
preconditioned by the illumination intensity,
$\nabla_\varphi M = -\frac{2}{N}\,A^2 \odot \Delta\varphi$ to first order
(test `gs_phase_error_is_adjoint_mse_negative_gradient`) — §12 and §13
are the same physics seen from the iteration side and the gradient side.

**Scope and fallbacks:** merits must supply an observation gradient
(`Merit::obs_gradient`); those that do not keep the FD-on-observation
fallback. Only the intensity channel is adjoint-supported — a seed with
nonzero entries in the $\partial I/\partial x_0$,
$\partial I/\partial y_0$, $\partial I/\partial\lambda$ derivative
channels is rejected (their adjoint is a future extension). The
transmission probe assumes a thin element whose action is a pure
pointwise complex multiply — all current core elements (§9).

## 14. Custom merits and the CRLB observation model (`optics-inverse`, task 9.12)

Per plan decision 7, estimation-theoretic merits are **user merits**, not
built-ins: their measurement-model parameters (photon budget, read noise,
event thresholds) are project-specific. The 0.2.0 frontend (ADR-9.11,
feature `py`) embeds CPython and scores a user function over the
`Observation` buffer viewed as a read-only float64 array shaped
$(C, B, S, N_y, N_x)$ — the §13 layout, so the adjoint path consumes the
user's $\partial M/\partial\text{data}$ directly.

The worked example (`examples/crlb_merit.py`, driven by
`examples/crlb_doe.rs`) is the A-optimal Cramér–Rao trace for
source-position estimation $\theta = (x_0, y_0)$. The Fisher information
is built from the observation itself, with **no extra physics**:

- **Scene-stencil derivatives.** With the renderer scenes placed at
  $(0,0), (\pm h, 0), (0, \pm h)$, the source-position derivatives are
  central differences of the intensity stacks,
  $\partial I/\partial x_0 \approx (I_{+x} - I_{-x})/2h$ — the same
  identity `observe.rs` uses for its FD derivative channels, evaluated
  inside the merit so the adjoint seed stays confined to the intensity
  channel (§13 scope).
- **Noise model.** Per pixel $p$, expected photo-electrons
  $E_p = \Phi\, I_p / \sum I + b$ (peak-normalized flux $\Phi$, background
  $b$); variance $V_p = E_p + \sigma_r^2$ adds gaussian read noise to the
  Poisson term. The Fisher matrix is the inverse-variance Gram sum

  $$\mathcal{F} = \sum_p \frac{1}{V_p}\, \nabla I_p\, \nabla I_p^{\mathsf{T}},
  \qquad M = \operatorname{tr}\!\big[(\mathcal{F} + r\,I)^{-1}\big]\,(s_i s_j),$$

  minimized ($r$ = ridge, $s$ = parameter scales; the trace is reported
  in $\mu m^2$). This is `crlb_from_images` in the hypercam
  `coded_mask.py` notebook, reduced to spatial parameters.
- **Event log-ratio variant.** For an event camera the measurement is
  $O = \log I_t - \log I_{t-\tau}$ (CodedEvents, Shah et al.); the
  delta-method variance is $\approx 2/E_p$, so the Jacobian becomes
  $\nabla \log I = \nabla I / I$ with weights $E_p/2$. The example runs
  this variant through the FD-on-obs fallback (`gradcheck::FdObsMerit`)
  — the ADR's layered-gradient story: user gradient → exact adjoint seed,
  no gradient → window-scale FD (validated against autograd in the
  example's development, cos ≈ 1.0).

**Gradient discipline.** User gradients are a correctness trap: the
frontend ships `gradcheck::assert_close`, comparing
`Merit::obs_gradient` against central differences on a pixel window
(the example validates the analytic CRLB gradient this way before
optimizing; FD steps on tail pixels must not drive intensities negative —
the event merit floors its log-ratio denominators for exactly that).

**Cross-check.** `tests/hypercam_forward.rs` renders a *trained* thickness
map exported from the notebook model
(`scripts/hypercam_forward_export.py`) through the Rust
`Renderer`/`PhaseMask`/`ThinLens` chain and compares the raw
`Observation` buffers element-for-element at reduced scale: agreement is
$\sim 3\cdot10^{-12}$ of peak — two independent implementations (torch
FFT vs rustfft) of the same ASM spec.

## References

1. *Diffractive Optics: Design, Fabrication, and Test* (workspace root PDF):
   Sec. 1.4 (Huygens, Fresnel zones), Eq. 1.1 (grating equation),
   Sec. 2.1 + Eqs. 2.1–2.8 (Maxwell $\to$ Helmholtz, assumptions (1)–(8),
   Rayleigh–Sommerfeld first-kind integral), Sec. 2.2 (far field = FT of
   aperture), Sec. 2.2.2 + Ch. 3 (RS $\leftrightarrow$ plane-wave-spectrum
   identification), Eqs. 2.9, 3.5–3.13 (plane-wave basis, direction
   cosines), Sec. 2.4 + Eqs. 2.25–2.29 (blazed grating,
   order angles and efficiencies), Sec. 2.5 (shadowing), Sec. 4.1.2 +
   Eqs. 4.1–4.3 (thick/thin lensmaker, surface power $\Phi=(n-1)c$ —
   Eq. 4.1's bracket re-verified against the rendered page), Sec. 4.1.3
   (chromatic aberration: refractive power $\propto (n-1)c$), Sec. 4.1.5
   (spherical aberration: LSA/TSA, ray-intercept curve), Sec. 4.2.1 +
   Eqs. 4.12–4.13 (diffractive lens = quadratic phase, zone radii
   $r_p^2 = 2fp\lambda$), Sec. 4.2.2 + Eqs. 4.17–4.21 (phase from sag
   profile, OPD $=(n-1)r^2/2R$), Sec. 4.4.1 (diffractive surfaces for
   color correction), Eq. 5.1 (grating equation in design context),
   Sec. 5.5 + Eqs. 5.14–5.18 (Gerchberg–Saxton iterative DOE design:
   energy projections at both planes, amplitude-error merit),
   Sec. 11.2 (Gaussian beams as beam-shaper input).
2. N. Matsushima and T. Shimobaba, "Band-limited angular spectrum method for
   numerical simulation of free-space propagation in close and far fields,"
   *Optics Express* **17**(23), 19731–19741 (2009).
3. J. W. Goodman, *Introduction to Fourier Optics* — standard Airy form
   $[2J_1(v)/v]^2$ for the circular aperture transform (the textbook states
   the far-field-is-the-aperture-FT relation; the explicit $J_1$ form is the
   classical evaluation of that transform).
4. A. E. Siegman, *Lasers*, and Goodman — paraxial Gaussian beam $q(z)$
   solution and the Hermite/Laguerre-Gauss mode families with their
   orthonormal normalizations (the project textbook does not restate
   them; the crate's $e^{+ikz}$ form $q = z - iz_R$ and $e^{i\ell\varphi}$
   OAM sign are pinned against its own validated engines, §9.5–9.6).
