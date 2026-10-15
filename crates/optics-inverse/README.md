# optics-inverse — inverse design for `optics-core` systems

Inverse design flips the forward model: instead of choosing a surface and
simulating the image it makes, a **design** is parameterized by a bounded
vector, a **renderer** produces an `Observation` (intensity stacks over
wavelength/field, optionally with derivative channels), a **merit
function** scores that observation, and a **solver** searches the
parameter space.

Part of the [optics-studio workspace](../../README.md) (Phase 9,
`tasks/plan.md`; full design in the crate docs and
`docs/architecture.md` §7). Public API depends only on `optics-core`
(+ `thiserror`, `rand`, `num-complex`); `optics-zemax` is a
**dev-dependency only** for the golden import→solve tests, so the crate
API stays zemax-free.

## Milestones

| Version | Contents |
|---|---|
| 0.1.0 | Framework + solvers: bounded `Params`, carriers (analytic element parameters and sampled phase grids), the `Observation`/`Renderer` model, built-in merits (MSE-to-target, encircled energy, Strehl, MTF, spot RMS), derivative-free solvers (random walk, annealing, evolution strategies), finite-difference Adam, Gerchberg–Saxton phase retrieval, adjoint gradients (the adjoint of an ASM propagation is propagation by −z, so gradients need no autodiff). |
| 0.2.0 | Custom merits: `PyMerit` (feature `py`) embeds CPython via pyo3 and presents `Observation` as a read-only numpy array (ADR-9.11); `gradcheck` ships the FD-on-observation check helper and the `FdObsMerit` adjoint fallback. Estimation-theoretic merits (CRLB-class) are *user* merits — see `examples/crlb_doe.rs`. |
| 0.3.0 | Deep surrogates: `dl` (feature `dl`) — headless dataset generation from the renderer plus `burn` MLP surrogates in both directions (forward stand-in, inverse warm start for the solvers). |

## Module map

| Module | Role |
|---|---|
| [`params`](src/params.rs) | Fixed-length bounded parameter vectors — the design space every solver searches; optimizers never see design structure. |
| [`carrier`](src/carrier.rs) | What a parameter vector *means*: maps `Vec<f64>` (+ wavelength) to an owned `Design` element stack — analytic carriers (lens curvatures…) and `grid_phase` sampled DOE grids. |
| [`observe`](src/observe.rs) | `Renderer` → `Observation`: flat row-major `f64` intensity stacks with documented strides (zero-copy views for Rust or Python merits), optional derivative channels. |
| [`merit`](src/merit.rs) | Scalar scores over an `Observation`. Convention: **all merits are minimized** ("maximize EE" is `-EE(r)`). |
| [`solve`](src/solve.rs) | Solvers: random search with adaptive step, simulated annealing, evolution strategies, `Solver::Adam`. |
| [`fd`](src/fd.rs) | Central finite-difference gradients over `Params` + Adam config (task 9.7). |
| [`adjoint`](src/adjoint.rs) | Reverse-mode gradients through the render chain without autodiff (task 9.9, ADR-9.9). |
| [`gs`](src/gs.rs) | Gerchberg–Saxton phase retrieval / DOE design (textbook §5.5). |
| [`gradcheck`](src/gradcheck.rs) | Gradient-check helpers and the FD-on-observation fallback for user merits (ADR-9.11 §6). |
| [`py`](src/py.rs) | *feature `py`* — Python custom-merit frontend (pyo3 + numpy). |
| [`dl`](src/dl/mod.rs) | *feature `dl`* — burn deep-surrogate harness: dataset generation, forward/inverse MLPs, `dl::warm_start`. |
| [`error`](src/error.rs) | `InverseError`: validation surfaces at *apply* time, never a panic. |

## Features

| Feature | Pulls in | Notes |
|---|---|---|
| (default) | — | Framework + solvers; pure Rust, CPU only. |
| `py` | `pyo3`, `numpy` | Embeds CPython (ADR-9.11). Needs a Python 3 interpreter at build/run time. |
| `dl` | `burn`, `serde`, `serde_json` | Deep surrogates (task 9.14). Burns with the pure-Rust **Flex CPU** backend by default; GPU (wgpu/cuda) is allowed *here* — enable those burn features and pass the matching `Device` — but never in `optics-core`. `serde_json` uses `float_roundtrip` for bit-exact dataset persistence. |

## Usage sketch

```rust
use optics_inverse::{
    carrier::{AnalyticBuilder, Carrier},
    merit::{Merit, SpotRmsMerit},
    observe::Renderer,
    params::Params,
    solve::Solver,
};
// See the module docs for the full loop: build a Params box, pick a
// Carrier that renders it through optics-core, score the Observation
// with a Merit, and run a Solver. examples/coded_doe.rs is the shortest
// end-to-end read.
```

## Examples & tests

```sh
cargo run --release -p optics-inverse --example coded_doe
  # GS top-hat beam shaper: phase-only DOE on a coherent relay (task 9.8/9.10)

cargo run --release -p optics-inverse --features py --example crlb_doe
  # A-optimal CRLB merit defined in Python (examples/crlb_merit.py) — task 9.12

cargo run --release -p optics-inverse --features dl --example surrogate_doe
  # dataset generation → forward+inverse burn surrogates → warm-started Adam — task 9.14

cargo test -p optics-inverse                          # default features
cargo test -p optics-inverse --features py            # + py_merit
cargo test -p optics-inverse --features dl            # + dl_surrogate
```

Notable tests: [`aca254_adam.rs`](tests/aca254_adam.rs) closes the
zemax→inverse loop (import a real Thorlabs doublet, tune one curvature
with FD/Adam against spot RMS); [`hypercam_forward.rs`](tests/hypercam_forward.rs)
byte-compares our `Renderer` output against a reference forward render
exported from the hypercam `coded_mask.py` optical model (regenerate via
[`scripts/hypercam_forward_export.py`](scripts/hypercam_forward_export.py)).

## License

MIT OR Apache-2.0.
