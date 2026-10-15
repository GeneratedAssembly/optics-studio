# optics-studio (crates/studio)

Phase-10 web UI over the headless optics APIs — dioxus `=0.7.10`,
`wasm32-unknown-unknown`, client-side rendering only
([ADR-10.0](../../docs/adr-10.0-ui-framework.md); screens and bindings in
[docs/ux/spec.md](../../docs/ux/spec.md)).

## Layout (Phase-10 tasks)

- `viewmodel/` — pure DOM-free viewport model (task 10.2): colormaps,
  intensity mapping, metrics, charts, cross-sections, sampling banner,
  four-way engine dispatch, reference-PSF recipe, demo data.
- `run/` + `bridge/` — host-testable inverse-run core and the dedicated
  web-worker transport (task 10.3, ADR-10.0 Q3).
- `app/` — dioxus PSF/metrics viewport components (task 10.2, wasm).
- `shell/` — task-10.1 application shell: `prescription` (`.zmx`
  import state + edited-surface → simulation-beamline plan mirroring the
  `optics-zemax` builder + schematic strip model), `source` (source
  picker), `dispatch` (five-way engine picker incl. Rayleigh-Sommerfeld,
  G-8 adapter), `preview` (live-preview raster via `viewmodel::intensity`
  + BMP encode), `carrier` (demo inverse carrier); `shell/ui/` holds the
  dioxus editor column, live preview, and pickers (wasm only).
- `ui/` — the app root (`ui::App`): shell column + live preview left/
  center, setup panels right, 10.2 viewport, 10.3 run console.

No physics in this crate (spec §6): only public-API calls from
`optics-core` / `optics-zemax` / `optics-inverse`, unit conversion, and
display normalization.

## Boot note

The crate's single `#[wasm_bindgen(start)]` lives in `bridge.rs` and
dispatches by scope: page → `ui::launch()` (dioxus), worker → solver
handler registration. The dioxus loader skips `main()` when a `start`
export exists, so `main.rs` is a safety net only. Do not add a second
start hook.

## Build & run

The crate is an **opt-in workspace member** (see root
`default-members`): repo gates (`cargo fmt/clippy/doc/test` at the root)
never build dioxus; the native half is testable per-crate:

```sh
cargo test -p optics-studio          # native view-model + run-core tests
```

Dev loop (ADR-10.0 Q2 — `dx`, not trunk):

```sh
cargo binstall dioxus-cli --version 0.7.10 -y   # or cargo install
cd crates/studio
dx serve --platform web              # http://localhost:8080
```

Explicit wasm leak guard (must pass before merging UI changes):

```sh
cargo check -p optics-studio --target wasm32-unknown-unknown
cargo clippy -p optics-studio --target wasm32-unknown-unknown --all-targets -- -D warnings
```

Bundle for packaging (task 10.4): `dx build --platform web` →
`target/dx/optics-studio/*/web/public/`.

## Browser constraints

- `.zmx`/`.AGF` import goes through the file picker into
  `ImportedSystem::parse_with` / `AgfCatalog::parse` (no filesystem on
  wasm; `load*` paths serve native callers/tests).
- Solver runs pass `seed = Some(..)`; the `getrandom` `js` backend is
  enabled for wasm in `Cargo.toml` (ADR-10.0 Q6).
- Shell preview runs synchronously on a small grid (chunked
  single-thread responsiveness, ADR-10.0 Q3); solver work executes in
  the dedicated worker (`bridge.rs`).
