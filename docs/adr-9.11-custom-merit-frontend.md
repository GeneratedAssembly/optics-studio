# ADR-9.11: Custom merit frontend (optics-inverse 0.2.0)

Status: **accepted** (user decision, 2026-10-14, via the GEN-46 question
card). Resolved decision 7 (`tasks/plan.md`) delegated this fork to the
user because they write merits in Python daily; task 9.11
(`tasks/todo.md`) and GitHub issue optics-studio#3 both marked the
decision as needing user input. The user answered all three questions
(§7) exactly as recommended: **Q1 = A (pyo3+numpy behind feature `py`),
Q2 = A1 (Rust binary embeds the Python interpreter), Q3 = layered
gradients (optional user-supplied `obs_gradient` callable, automatic
FD-on-obs fallback, shipped gradient check helper)**. The DSL is deferred
(not rejected) to the Phase-10 UI ADR; extension-module packaging is
deferred to 0.2.x. Task 9.12 implements §8.

Companion seam (already shipped in 0.1.0): `Merit` in
`crates/optics-inverse/src/merit.rs` and the numpy-ready `Observation`
layout in `crates/optics-inverse/src/observe.rs`. The chosen frontend
implements exactly that trait — nothing else in the solver stack changes
(decision 7's "plugs in without redesign" constraint holds).

## Context

The 0.1.0 crate scores observations through `Merit`:

- `evaluate(&Observation) -> f64` — every solver (derivative-free, FD/Adam,
  adjoint drivers) consumes only this for its search loop, so *any* user
  score is immediately optimizable once it implements the trait.
- `obs_gradient(&Observation) -> Option<Vec<f64>>` — $\partial M/\partial
  \text{data}$, the adjoint seed for task 9.9. `None` means "no analytic
  gradient available" and routes the user to FD fallbacks.

Milestone B (0.2.0) must let the **user** define merits — the CRLB/Fisher
class that decision 7 refuses to bake in ("depend heavily on parameters we
don't know a priori") — expressed "in Python or mathematical notation"
(user's words, Phase-9 check-in 2026-10-14). The reference workflow is
`hypercam/notebooks/coded_mask.py`: the merit is a numpy/torch function
over stacks shaped `(C, B, S, Ny, Nx)` — exactly `Observation.data`
reshaped (channel 0 = $I$, then $\partial_x,\partial_y,\partial_\lambda$),
zero-copy viewable.

### Cost model (the numbers that decide the gradient question)

Let $N = $ `obs.data.len()` $= C\cdot B\cdot S\cdot n_x n_y$ and
$\dim$ = design parameters. A **render** (multi-band ASM chain) dominates
all other costs; a merit call is $O(N)$ numpy.

| Gradient route | Renders | Merit calls | Exactness |
|---|---|---|---|
| FD on params (9.7) | $2\dim$ | $2\dim$ | $\epsilon^2$ |
| Adjoint (9.9) | $1 + \lvert\text{frames}\rvert$ | 1 (+ user grad) | exact |
| FD on obs | $1 + \lvert\text{frames}\rvert$ | $2N$ | $\epsilon^2$ |

FD on obs is render-cheap but call-heavy: a $128^2$ single-band single-
scene 4-channel observation has $N \approx 65\,536$, so $2N$ Python
round-trips per gradient step — fine for small windows ($32^2$ crops, as
the notebook actually uses), impractical for full frames unless the merit
is written batch-aware against a stacked perturbation array. For the
adjoint path the fallback is therefore a *window-scale* tool, not the
grid-scale answer (§6).

## Decision points

1. **Frontend vehicle:** (A) pyo3+numpy embedding behind cargo feature
   `py`; (B) expression DSL compiled to closures; (C) both.
   1b. **If A, host direction:** (A1) the Rust binary embeds the Python
   interpreter; (A2) Rust ships as a Python extension module (maturin),
   notebook-hosted.
2. **Gradient strategy for user merits:** (G1) user-supplied
   $\partial M/\partial\text{obs}$; (G2) autodiff on the Python side
   (jax/torch over the merit); (G3) FD on obs.

## Option analysis — frontend vehicle

### A. pyo3 + numpy (feature `py`)

Shape: `PyMerit` implements `Merit`. `evaluate` takes the GIL, presents
`Observation` as a read-only ndarray view (no copy) plus a small metadata
object (`channels`, `bands`, `scenes`, `dims`, `pitch`, `fd_steps`), calls
the user callable, expects a real scalar. An optional second callable (or
a `grad` attribute) supplies `obs_gradient`; absent ⇒ `None` ⇒ the 9.5
fallback semantics unchanged.

- **For:** full expressiveness on day one — the CRLB example (9.12) needs
  $3\times3$ Fisher matrices, ridge regularization, einsum reductions, an
  event-camera log-ratio variant; that is "just numpy", not a DSL feature
  request. The user's daily idiom (decision 7's rationale) ports
  near-verbatim from the notebook. `abi3` keeps the Python-version coupling
  loose; the feature gate keeps the core dependency-minimal (plan risk
  table: "pyo3 bloat / build friction → feature-gated `py`, workspace
  default-members keep CI light"). The 0.1.0 seams were designed for
  exactly this adapter.
- **Against:** embedded-interpreter lifecycle and GIL discipline (merit
  calls serialize under the GIL — harmless while solvers are
  single-threaded, must be documented before any parallel-render path);
  CI/build friction for the gated feature; a `py`-enabled binary needs a
  Python runtime; opaque errors surface as Python tracebacks (frontend
  must wrap them in `InverseError::Render`-style diagnostics).

1b sub-fork: **A1 (embed)** keeps the Rust-first story — `cargo run`,
config-driven solves, the solver stack stays the host. **A2 (extension
module)** flips the host: the notebook imports `optics_inverse`, passes
merits as callables, and drives solves from Python — closest to the
current daily workflow, but makes the crate a published Python package
(maturin, wheels, version pairing) and inverts the CLI story. Both reuse
the same `PyMerit` adapter; A2 adds only a `#[pymodule]` entry surface.
The plan text names "pyo3+numpy embedding (feature `py`)" — A1 as scoped,
A2 as a packaging follow-up.

### B. Expression DSL compiled to closures

Shape: parse a math-notation string over named frames — e.g.
`mean((I - T)^2)`, `tr(inv(F))` — into an AST, compile to a Rust closure,
derive $\partial M/\partial\text{obs}$ by reverse mode on the AST (exact,
free).

- **For:** zero new runtime deps; merits are serializable strings (config
  files, CLI, and eventually the Phase-10 web UI — a browser cannot embed
  a CPython interpreter, but it can edit an expression); sandboxed,
  deterministic, fast.
- **Against:** the CRLB class needs linear algebra (matrix inverse,
  ridge, weighted sums over pixel axes) — a DSL that covers it is a
  numpy subset, i.e. re-implementing numpy badly inside the crate; parser
  + typechecker + codegen + linalg + docs is a milestone of its own, and
  9.12 (the worked example) sits behind it. The user writes Python daily;
  "mathematical notation" is already expressible as Python expressions.

### C. Both

DSL as sugar for small serializable merits, pyo3 as the power path. Two
public surfaces, two test matrices, two gradient stories to document, in
a scope-L milestone whose acceptance target (9.12 CRLB) only exercises
the Python side.

## Option analysis — gradient strategy

- **G1 user-supplied gradient:** exact and $O(N)$, zero extra calls. For
  the CRLB merit the gradient is a small local closed form (the Fisher
  matrix is $3\times3$, so $\partial\,\mathrm{tr}(F^{-1})/\partial I_p$
  is a short einsum) — writable, but a correctness trap. The frontend
  should ship a check helper (`assert_close(fd_on_obs(M), grad, window)`
  on a small crop) so user gradients are validated cheaply, reusing the
  existing FD machinery.
- **G2 Python-side autodiff:** jax/torch over the merit gives exact
  gradients in one call — but it is a *user-environment* dependency, not a
  crate feature. Structurally it collapses into G1: the user passes
  `grad=lambda o: jax.grad(M)(o)` through the same optional-gradient
  seam. No crate-side autodiff of Python code is proposed.
- **G3 FD on obs:** already the shipped fallback (`obs_gradient() →
  None`); zero user burden; cost $2N$ merit calls per gradient step,
  practical at window scale, not at full-frame scale (see cost model).

The three are not exclusive; the real question is which is the *documented
default path* and which is the automatic fallback.

## 6. Decision (accepted as recommended)

- **Frontend: A, embedding direction A1** — pyo3+numpy behind feature
  `py`, Rust binary hosts the interpreter, `PyMerit` implements the 0.1.0
  trait against zero-copy ndarray views. Rationale: decision 7's user
  constraint (Python daily), the 9.12 CRLB example's expressiveness needs,
  and the plan's own tentative direction ("likely pyo3 primary, DSL sugar
  later").
- **DSL: deferred, not rejected.** Revisit as the serializable merit spec
  in the Phase-10 UI ADR (`docs/ux/adr-10.0-draft.md`), where a browser
  front-end creates the real need; by then the built-in FOMs + a small
  expression subset can be designed against concrete UI requirements.
- **A2 (extension module): deferred packaging follow-up** (0.2.x), same
  adapter, decided when the wheel story matters (it will matter for
  notebook users before long).
- **Gradient: layered.** First-class optional user gradient callable
  (covers G1 and, as a user-side recipe, G2) + automatic FD-on-obs
  fallback (G3, existing seam) + a shipped gradient check helper. No
  crate dependency on jax/torch.

## 7. Decision questions — answered 2026-10-14

1. **Frontend vehicle** — answered **A: pyo3+numpy behind feature `py`**
   (recommended option).
2. **Host direction** — answered **A1: Rust embeds Python** (recommended
   option). A2 (extension module) deferred to 0.2.x packaging work.
3. **Gradient strategy** — answered **layered** (recommended option):
   optional user gradient callable + FD-on-obs fallback + check helper.

## 8. Consequences

After acceptance, task 9.12 implements:

- `crates/optics-inverse/src/py.rs` behind feature `py` (+ `Cargo.toml`
  optional deps `pyo3` (abi3) and `numpy`); default features unchanged so
  CI stays Python-free;
- `PyMerit { call, grad: Option<call> }` implementing `Merit`, with Python
  exceptions wrapped into `InverseError`;
- an `Observation` ⇄ ndarray bridge (view, not copy; metadata object);
- the gradient check helper (FD-on-obs vs user gradient on a small crop);
- the CRLB worked example (9.12) written as a user merit, optimized via
  the 9.9 adjoint — it is the acceptance test of this ADR's choice;
- `docs/architecture.md` + `CHANGELOG.md` entries, feature `py` documented
  with its GIL/threading constraints.

Budget note (either way): pyo3/numpy add build time only under the
feature; no core API change; the `Box<dyn Merit>` object story is
untouched.

## Rejected alternatives

- **DSL as primary:** expressiveness gap for exactly the merit class this
  milestone is for, plus parser/codegen scope; deferred to the Phase-10
  UI ADR instead.
- **Both surfaces in 0.2.0:** two test matrices, neither exercised by the
  9.12 acceptance example.
- **A2 as the primary host:** inverts the Rust-first release story before
  the Python packaging pipeline (maturin/wheels) exists; deferred.
- **Crate-side autodiff of Python merits:** impossible without owning the
  merit graph; the G1/G2 seam plus FD fallback covers the need.
- **Rust autodiff re-port:** out of scope here and already rejected in
  ADR-9.9 (decision 2) — this ADR is about the *merit-side* gradient only.
