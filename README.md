# optics-studio — Diffractive Optics Engineering Suite

A pure-Rust, wave-optics design suite for diffractive optical elements
(DOE): a scalar field propagation engine, Zemax prescription import,
inverse-design solvers, and a browser UI that ties them together.

The physics is **wave optics only** — Angular Spectrum (exact, vacuum or
refractive media), Fresnel (transfer-function and convolution forms),
Fraunhofer, and Rayleigh–Sommerfeld first kind. Ray tracing and
geometric-optics approximations are deliberately out of scope
([`agents.md`](agents.md) §1). Units are SI throughout (meters, radians,
cycles per meter) and precision is `f64` (approved decision 2026-10-11).

The ultimate physics reference is *Diffractive Optics: Design,
Fabrication, and Test* (SPIE TT95), kept at the repository root; the
models, formulas, and validity regimes are written up in
[`docs/physics.md`](docs/physics.md).

## Repository layout

```
optics-rs/                        # repo dir; cargo workspace root
├── Cargo.toml                    # workspace + ROOT PACKAGE = optics-core
├── src/                          # optics-core engine (root-as-core layout)
│   ├── common.rs                 #   Angle newtype, imaginary unit J
│   ├── wave.rs                   #   scalar source/beam parameter records
│   ├── field.rs                  #   Field / BandField / Psf data grids
│   ├── element.rs                #   Element trait + apertures, lens, DOE
│   │                             #   phase masks, gratings, refractive
│   │                             #   surfaces, thick lenses, wedges
│   ├── sources.rs                #   closed-form initializers (Gaussian,
│   │                             #   Hermite-Gauss, Laguerre-Gauss, point)
│   ├── propagation/              #   engines: asm, fresnel, fraunhofer,
│   │                             #   rayleigh_sommerfeld, regime, fft
│   ├── beamline.rs               #   (element, z) composition + PSF trace
│   └── analysis.rs               #   detector-plane metrics (EE, spot, MTF…)
├── tests/                        # optics-core integration tests (per phase)
├── examples/                     # optics-core examples (see below)
├── crates/
│   ├── optics-zemax/             # crate: optics-zemax — .zmx/.AGF import
│   ├── optics-inverse/           # crate: optics-inverse — inverse design
│   └── studio/                   # crate: optics-studio — dioxus web UI
├── docs/
│   ├── physics.md                # models, transfer functions, regimes
│   ├── architecture.md           # module map, data flow, error taxonomy
│   ├── ux/spec.md                # UI screens, flows, API bindings
│   └── adr-*.md                  # accepted architecture decision records
├── tasks/plan.md, tasks/todo.md  # approved roadmap + ordered task list
├── zmx-samples/                  # sample prescriptions for import demos
├── agents.md                     # AI-assistant development protocol
└── CHANGELOG.md                  # Keep-a-Changelog history
```

**Root-as-core layout:** the mature engine crate lives at the workspace
root (package `optics-core`), and the satellite crates sit under
`crates/*` with path dependencies on it. The core crate is named
`optics-core`, never `core` — `core` collides with Rust's built-in
extern prelude.

Dependency direction (build bottom-up):

```
optics-core ──┬── optics-zemax ──┐
              └── optics-inverse ┴── optics-studio (web UI)
```

Algorithms depend on data, never the reverse: `propagation` consumes
`field` types; elements are local transformations; `beamline` is
composition sugar. The UI crate contains **no physics** — it only calls
the headless crates' public APIs.

## Crates at a glance

| Crate (package) | Version | Role |
|---|---|---|
| [`optics-core`](src/lib.rs) (this root) | 0.7.0 | Wave-optics engine: `BandField` propagation through ASM / Fresnel / Fraunhofer / Rayleigh–Sommerfeld with sampling guards and regime advice; element library (apertures, thin lens, DOE phase masks, blazed gratings, refractive surfaces, thick lenses, wedges); `Beamline` PSF composition; detector-plane `analysis` metrics. |
| [`optics-zemax`](crates/optics-zemax/README.md) | 0.1.0 | Zemax prescription import: parses text `.zmx` files, resolves glasses through built-in Sellmeier catalogs (plus `.AGF` catalog import), and builds `optics-core` systems with paraxial EFL/BFL reports for validation. |
| [`optics-inverse`](crates/optics-inverse/README.md) | 0.3.0 | Inverse design: bounded parameter vectors, design carriers, the observation/merit model, solvers (GS phase retrieval, adjoint gradients, Adam, annealing, evolution strategies), optional Python custom merits (`py`) and burn deep surrogates (`dl`). |
| [`optics-studio`](crates/studio/README.md) (in `crates/studio/`) | 0.1.0 | Browser UI (dioxus 0.7.10, wasm32, CSR): prescription editor + live Beamline preview, PSF/metrics viewport (EE/MTF/Strehl), and an inverse-run console with a web-worker solver and convergence plot. |

`optics-studio` is an **opt-in** workspace member: the default gates and
CI never build dioxus (ADR-10.0). The other three are in
`default-members`.

## Quickstart

### Prerequisites

- Rust ≥ 1.85 (the workspace uses edition 2024; developed on 1.98).
- For the UI only: the dioxus CLI, pinned to `0.7.10`.

### Build and test the engine crates

```sh
cargo build                      # optics-core + optics-zemax + optics-inverse
cargo test                       # unit + integration + doc tests (default members)
```

### Start the UI

```sh
cargo binstall dioxus-cli --version 0.7.10 -y   # or: cargo install dioxus-cli --version 0.7.10
cd crates/studio
dx serve --platform web
```

Open <http://localhost:8080>. The app boots with demo data; to drive the
full loop, use the file picker to import a prescription — e.g. the
Thorlabs collimator in the repo at
`zmx-samples/ACA254-100-UV-Zemax-ZMX.zmx` — then:

1. **Import & edit** the prescription (text/table modes, surface sliders).
2. **Simulate** to produce a live PSF preview (engine and source pickers).
3. **Analyze** it in the viewport: encircled energy, MTF, Strehl, spot,
   cross-sections, sampling banner.
4. **Inverse-design** from the run console: pick a merit (captured PSF
   target, −Strehl, EE focus), start/stop the solver in its worker, and
   watch the convergence plot.

Native smoke test of the run loop (no browser):

```sh
cargo run -p optics-studio --example demo_run
```

The wasm leak guard must pass before merging UI changes:

```sh
cargo check -p optics-studio --target wasm32-unknown-unknown
```

### Examples

```sh
cargo run --release --example psf_image_convolve     # core: PSF-based image blur
cargo run --release --example inverse_drism          # core: SAGE IV DRISM spec→geometry
cargo run --release -p optics-inverse --example coded_doe              # GS top-hat shaper
cargo run --release -p optics-inverse --features py  --example crlb_doe       # CRLB merit (needs Python)
cargo run --release -p optics-inverse --features dl  --example surrogate_doe  # burn surrogates
```

## Development workflow

The repository is driven by an approved roadmap: `tasks/plan.md` (phases
7–10) and the ordered checklist in `tasks/todo.md`. AI assistants follow
[`agents.md`](agents.md). Every change must pass the gates before commit:

```sh
cargo fmt --check
cargo clippy -q --all-targets -- -D warnings
cargo doc -q --no-deps
cargo test
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/);
`CHANGELOG.md` is updated in the same commit. See
[`docs/architecture.md`](docs/architecture.md) for the module map and
error taxonomy, [`docs/physics.md`](docs/physics.md) for the models, and
the ADRs in [`docs/`](docs/) for framework/frontend decisions.

## License

MIT (see [`LICENSE`](LICENSE)); the satellite crates declare
`MIT OR Apache-2.0` in their manifests.
