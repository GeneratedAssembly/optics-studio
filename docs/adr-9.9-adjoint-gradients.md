# ADR-9.9: Adjoint gradients without an autodiff graph

Status: **accepted** — implements Phase-9 task 9.9 (`tasks/todo.md`) under
resolved decisions 2 and 8 (`tasks/plan.md`, user, 2026-10-14). Decision 2
locks the order FD → derivative-free → GS → adjoint, and allows an autodiff
re-port *only if* the adjoint proves insufficient. This record shows it is
sufficient for thin-element chains, so no autodiff dependency is added.
Companion code: `crates/optics-inverse/src/adjoint.rs`.

## Context

The design gradient $\partial M/\partial\theta$ drives first-order solvers.
Finite differences (task 9.7) cost $2\dim$ renders per gradient. For a
sampled DOE with an $n\times n$ lattice, $\dim=n^2$: a $128\times128$ grid
needs 32 768 renders per step. That is the wall decision 8 names ("grid-DOF
merits are FD-infeasible").

The forward model is a chain of two primitive kinds only:

1. **ASM propagation** by a signed distance $z$ through a medium $n$
   (`optics_core::propagation::asm::propagate`), and
2. **thin elements** — pointwise complex transmission $t(x,y)$ applied at a
   plane (`optics_core::element::Element::apply`; phase masks, thin lenses,
   apertures).

Both are linear operators on the complex field. Therefore the gradient can
be computed by pushing the merit's observation gradient *backward* through
the **adjoints** of these primitives, in reverse chain order. No symbolic or
graph-based autodiff is needed — one forward pass plus one backward pass per
observation frame.

## The math (recorded per the task: "ADR first")

### Conventions

Fields are complex vectors $U\in\mathbb{C}^{N}$ ($N=n_x n_y$ samples). The
inner product is Hermitian, $\langle a,b\rangle=\sum_k \bar a_k b_k$, and
the adjoint $L^\dagger$ of a linear operator $L$ is defined by
$\langle a, L b\rangle=\langle L^\dagger a, b\rangle$. Merits are real
($M\in\mathbb{R}$). The FFT scaling/sign/bin conventions are the core
`propagation::fft` module's (unnormalized forward, $1/N$ inverse,
$e^{-i2\pi kn/N}$ kernel, unshifted bins), pinned there and re-pinned here
by the adjoint identity tests.

### Forward chain, one (band, scene) frame

$$U_{\text{det}} \;=\; L_K\,T_K\,L_{K-1}\,T_{K-1}\cdots L_1\,T_1\,U_{\text{in}},$$

with $T_j=\mathrm{diag}(t_j)$ the thin-element transmission at plane $j$ and
$L_j=\mathcal{A}(z_j;n_j,\lambda)$ the ASM step after it. The observation
channel 0 is $I_k=|U_{\text{det},k}|^2$, and the merit supplies
$g_k=\partial M/\partial I_k$ through the 9.5 `Merit::obs_gradient` seam.

### Adjoint of the ASM step

The core ASM is $\mathcal{A}(z)=\mathrm{crop}\circ\mathcal{F}^{-1}\circ
\tilde H(z)\circ\mathcal{F}\circ\mathrm{pad}$, with
$\tilde H(z)=\Pi(f)\,e^{\,i n k z\sqrt{1-(\lambda f_x/n)^2-(\lambda f_y/n)^2}}$
on the propagating band ($\Pi$ = evanescent + Matsushima band-limit masks).
Using $\mathcal{F}^\dagger=N\mathcal{F}^{-1}_{\text{raw}}$ (the pinned
unnormalized-forward / $1/N$-inverse convention),
$(\mathcal{F}^{-1})^\dagger=\mathcal{F}_{\text{raw}}/N$,
$\mathrm{pad}^\dagger=\mathrm{crop}$, and $\tilde H(z)^\dagger=\overline{\tilde
H(z)}=\tilde H(-z)$ (the phase conjugates; both masks are real and the
band-limit $f_{\lim}\propto(2z/L)^2$ is even in $z$), every factor
transposes into the same recipe run with $-z$:

$$\boxed{\;\mathcal{A}(z)^\dagger=\mathcal{A}(-z)\;}$$

**exactly**, including zero-padding and window crop: the $1/N$ factors
transpose consistently and the centered embed/extract are each other's
adjoints. The strict-mode densification picks padding from grid geometry and
$|z|$ only, so the $-z$ call uses the same padded grid as the $+z$ call.
This is the same fact the GS loop already leans on (`gs.rs`:
back-propagation $\mathcal{A}(-z/2)$ is "the ASM adjoint on the passband"),
and the `asm_adjoint_inner_product_identity` test pins the full operator
identity $\langle w,\mathcal{A}(z)d\rangle=\langle\mathcal{A}(-z)w,d\rangle$
numerically.

### Adjoint of a thin element

$T=\mathrm{diag}(t)$ is a pointwise multiply, so $T^\dagger$ is the pointwise
**conjugate** multiply:

$$\lambda_{\text{before}} = \bar t \odot \mu_{\text{after}}.$$

An aperture is the special case $t=\Pi\,e^{i\varphi}$ with $\Pi\in\{0,1\}$:
$\bar t=\Pi\,e^{-i\varphi}$ — masked conjugate multiply; blocked pixels
($t=0$) pass no adjoint energy. The implementation does **not** need the
element to expose $t$: `apply` is pointwise, so probing
$t=\texttt{apply}(\mathbf{1})$ on a unit-amplitude field of the plane's grid
geometry recovers the exact transmission array (including aperture zeros,
center offsets, and orientation), and the adjoint applies $\bar t$.

### Merit-to-field seed

$I_k=\bar U_k U_k$, so for real $M$
$dM=\sum_k g_k(\bar U_k\,dU_k+U_k\,d\bar U_k)=2\,\Re\langle w,\,dU\rangle$
with the detector-plane adjoint source

$$w_k = g_k\, U_{\text{det},k}.$$

### Phase-grid gradient

Let $\mu$ be the adjoint field pushed back to the plane **after** element $j$
(all downstream adjoints applied), and let $U,V=t\odot U$ be the tape fields
before/after the element. Perturbing one lattice phase
$\varphi_p\to\varphi_p+d\varphi_p$ injects
$dV_p=i\,t_pU_p\,d\varphi_p=i\,V_p\,d\varphi_p$, so

$$\frac{\partial M}{\partial\varphi_p}
= 2\,\Re\!\big(\overline{\mu_p}\; i\, V_p\big)
= -2\,\Im\!\big(\overline{\mu_p}\; V_p\big).$$

That is the whole grid gradient: $dM/d\text{grid}$, one complex array per
element plane. Chain rules to physical DOFs:

- **`GridPhase` carriers** — parameters *are* the phase samples:
  $dM/dv_p=dM/d\varphi_p$.
- **`GridThickness` carriers** — $\varphi_p=\frac{2\pi}{\lambda}[n(\lambda)-1]h_p$,
  so $dM/dh_p=\sum_{\text{bands}}\frac{2\pi[n(\lambda_b)-1]}{\lambda_b}
  \left.\frac{\partial M}{\partial\varphi_p}\right|_{\lambda_b}$ — the
  per-band adjoint runs accumulate with the band's phase-law slope.
- **Analytic carriers** — the closure owns the parameter→phase map; the
  adjoint returns $dM/d\varphi_p$ on the element plane and the closure's
  (user-side) Jacobian finishes the chain. The acceptance test uses an
  analytic carrier whose map is the identity (params = lattice samples), so
  $dM/d\theta=dM/d\varphi$ directly and FD on the same coarse grid is the
  reference truth.

### Cost

Per gradient: **1 forward render** (the observation + merit value +
`obs_gradient` seed) plus **one adjoint run per gradient-carrying
observation frame** (each run = a forward tape of $O(N)$ FFTs + a backward
sweep of the same primitives in reverse order, with $\mathcal{A}(-z)$ and
$\bar t$). Frames whose seed slice is zero are skipped. The cost is
independent of the design dimensionality — $N$ FFTs, not $2n^2$ renders.

## The GS error-signal relation (acceptance 2)

Textbook §5.5.1 (Eqs. 5.14–5.18) runs the bidirectional loop; `gs.rs`
iterates it between the DOE plane and the target plane. At a GS iterate let

$$m=\Pi\,A\,e^{i\varphi}\quad(\text{GS DOE mask}),\qquad
U_t=\mathcal{A}(z/2)\{m\},\qquad \theta=\arg U_t,$$

and take the Eq. 5.18 amplitude MSE $M=\frac1N\sum_k(|U_{t,k}|-T_k)^2$. Its
observation seed is $g=\partial M/\partial I=(|U_t|-T)/(N|U_t|)$, so the
adjoint source at the target plane is
$w=\frac1N(|U_t|-T)\,e^{i\theta}$ and

$$\mu=\mathcal{A}(z/2)^\dagger w=\tfrac1N\mathcal{A}(-z/2)\{(|U_t|-T)e^{i\theta}\}.$$

The GS projection field decomposes exactly as
$V=\mathcal{A}(-z/2)\{T e^{i\theta}\}=m+\delta+\varepsilon$, where
$\delta=\mathcal{A}(-z/2)\{(T-|U_t|)e^{i\theta}\}=-N\mu$ (round-trip
identity on the passband) and $\varepsilon$ is the off-passband residual.
Then, to first order in $\delta/m$,

$$\Delta\varphi=\arg V-\varphi\approx \frac{\Im(\bar m\,\delta)}{|m|^2}
=-\frac{N}{2A^2}\,\frac{\partial M}{\partial\varphi}\qquad(\text{inside }\Pi),$$

i.e. **the GS phase error signal is the negative phase-gradient of the MSE,
preconditioned pointwise by the illumination intensity $A^2$**:
$\nabla_\varphi M = -\frac{2}{N}\,A^2\odot\Delta\varphi$. The update
$\varphi\mathrel{+}=\Delta\varphi$ is therefore a (preconditioned) gradient
descent step — which is what task 9.9 was asked to prove. The acceptance
test `gs_phase_error_is_adjoint_mse_negative_gradient` pins the sign/shape
correlation between $-\partial M/\partial\varphi$ (adjoint) and $A^2\Delta\varphi$
(GS) at a mid-loop iterate, and the first-order magnitude identity.

## Decision

Implement `crates/optics-inverse/src/adjoint.rs` with:

- `asm_adjoint(band, z, cfg)` — the exact ASM adjoint, $\mathcal{A}(-z)$;
- `transmission(element, like)` — unit-probe recovery of $t$;
- `push_back(design, input, dmdi, asm)` — one frame: forward tape + reverse
  adjoint sweep → `PlaneGradient` list (dM/dφ per element plane);
- `grid_gradient(renderer, values, merit)` — the render-chain driver:
  1 forward render + |frames| adjoint runs → `AdjointGradient` with
  per-band, per-plane `dphi` arrays (the chain rules above are one-line
  user-side sums, and `fd::gradient` remains the general fallback).

No new dependencies, no autodiff graph, no `optics-core` API change (the
`Design::steps`/`Design::element` accessors live in `optics-inverse`).

## Rejected alternatives

- **Autodiff re-port** (decision 2's contingency): rejected — the adjoint is
  complete for thin-element chains; autodiff would add a heavy dependency
  for zero capability gain here.
- **Complex-step FD**: still $O(\dim)$ runs; does not fix the grid-DOF wall.
- **Analytic per-element derivatives** ($\partial U_{\det}/\partial\varphi_p$
  column by column): same $O(\dim\cdot N\log N)$ cost as FD; the adjoint's
  reverse sweep gives all columns at once.
- **Exposing $t$ through the core `Element` trait**: unnecessary (unit probe
  recovers it) and it would break the core's minimal surface.

## Consequences

- `Merit::obs_gradient` (the 9.5 seam) is now load-bearing: merits without it
  keep the documented FD-on-observation fallback in the solvers.
- Grid-domain descent (Adam over `GridPhase`/`GridThickness` values) is
  feasible at production lattice sizes; task 9.12 (CRLB on a DOE window) can
  build on this.
- Derivative channels (∂x₀, ∂y₀, ∂λ) are not adjoint-supported yet: a seed
  with nonzero entries outside the intensity channel is rejected with
  `InverseError::Adjoint`. Their adjoint is a future extension (the FD
  stencil's transpose is trivial, but the λ channel differentiates the
  carrier rebuild itself, which the closure does not expose).
