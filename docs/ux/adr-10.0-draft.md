# ADR-10.0 (draft): UI framework for `optics-studio` — dioxus vs leptos

Status: **superseded** — ratified as
[`docs/adr-10.0-ui-framework.md`](../adr-10.0-ui-framework.md) (task 10.0 /
GEN-51, 2026-10-14): dioxus accepted, pinned `=0.7.10`, leptos 0.8.22
recorded as fallback; Q1–Q3 resolved there. This draft is kept unchanged as
history of the comparison. Original status follows.

Original: Status: **draft** — pre-work for the Phase 10 check-in (task S1-D /
GEN-42). Resolved decision 4 (tasks/plan.md, 2026-10-14) fixed "web from
the start" and left the framework choice to this ADR; the plan records
dioxus as recommended, leptos as the alternative, user agnostic. This
document compares both against the workspace constraints and ends with a
recommendation plus open questions. No decision is binding until the
Phase 10 check-in ratifies it and `tasks/plan.md` records it.

Companion document: [`spec.md`](spec.md) — the UI/UX spec this framework
must be able to implement (screen flows, panel→API bindings, rendering
constraints).

## Context and constraints (from the plan, not negotiable)

1. **Opt-in workspace member.** `optics-studio` lives at
   `crates/studio/` and is excluded from `workspace.default-members` so
   `cargo test`/gates never build it (plan.md, architecture decisions).
   Whatever framework is chosen, its dependency tree stays inside this one
   crate; the headless crates (`optics-core`, `optics-zemax`,
   `optics-inverse`) gain no UI dependencies.
2. **wasm toolchain.** Target is `wasm32-unknown-unknown` (already
   installed on this machine). The headless crates are pure Rust and wasm
   compatible with one pinned caveat: `rand` 0.8's `thread_rng()` is
   unavailable on wasm, so solver runs must pass
   `StochasticConfig::seed = Some(..)` (spec §3.4).
3. **No physics in the UI crate.** The UI calls only the public APIs named
   in `spec.md`; colormaps, downsampling, and unit conversion are
   presentation code and belong to the UI crate (plan risk table: "UI
   scope creep into physics").
4. **Workload shape from the spec.** Interactive but compute-bound:
   throttled live previews (`Carrier::build` → `Beamline::psf`), chunked
   solver runs (`solve` in budget slices, gap G-2), f64 grid → canvas
   painting (up to ~2048² display samples), small SVG/canvas line charts
   (EE, MTF, convergence). No server, no multi-user state, no SEO — this
   is a local single-user tool.
5. **Rust edition 2024**, current stable toolchain; conventional-commit
   gates apply to the crate when opted in.

## Evaluation

| Driver | What it demands | dioxus | leptos |
|---|---|---|---|
| D1. Full-Rust wasm SPA | No JS framework glue; CSR-only | SPA-first design (`dioxus-web`); no SSR machinery to carry or configure away | SSR-first framework; CSR mode exists but the server/hydration half is dead weight for a local tool |
| D2. Dependency weight / opt-in | Heavy proc-macro tree must stay out of the default graph | Moderate tree (rsx macro, signals, web backend); desktop feature pulls native windowing deps (cargo features isolate them) | Moderate tree (reactive signals, `leptos_macro`, `wasm-bindgen` stack); server feature must be carefully excluded |
| D3. Reactivity for live preview | Slider → throttled re-render → viewport swap without full re-render storms | Fine via `Signal` + `use_effect`; coarser-by-default re-render is acceptable because heavy output goes to canvas, not DOM | Fine-grained signal graph is its headline strength; slightly better for many small DOM updates (metric readouts), marginal in practice here |
| D4. Canvas painting | Direct `web-sys`/`js-sys` access to `CanvasRenderingContext2D` / `ImageData`, bypassing the virtual DOM | Supported (elements + `mount` refs); ecosystem charts are irrelevant — we hand-roll colormaps per spec §4 | Equally supported; same conclusion — both are thin over `web-sys` |
| D5. Worker story | Chunked `solve` off the UI thread; progress via messages | No first-class worker RPC; use `wasm-bindgen` + web worker crate (`workerlib`/`gloo-worker`) — DIY either way | Same DIY story; no advantage |
| D6. File I/O | Browser file picker for `.zmx`; export PNG/CSV | `web-sys` File API works; desktop target would use native dialogs | `web-sys` File API works equally; web-only |
| D7. Maturity / churn | Pin an exact version at phase start; expect pre-1.0 breaking releases either way | Pre-1.0, active; `dx serve` dev loop is good | Pre-1.0, active; docs excellent; release cadence steady |
| D8. Edition/MSRV fit | Must compile under edition 2024 with the workspace resolver | Verify at pin time (open question Q1) | Verify at pin time (open question Q1) |
| D9. Desktop fallback (optional) | A native window build for offline use would please P1 users | Built-in second target (`dioxus-desktop`, wry) from the same code — real differentiator | Web-first; desktop would mean Tauri glue outside the framework |
| D10. Alignment with plan | Plan already names dioxus as recommended | Matches | Matches the "alternative" role |

Notes on things that do **not** differentiate: both give signals-based
state, both render to wasm without a JS framework, both need hand-rolled
chart/canvas code for our rendering constraints (spec §4), both are
pre-1.0 with pin-and-verify discipline, and neither solves solver
cancellation for us (that is gap G-2 in `optics-inverse`, an API problem,
not a framework problem).

## Recommendation

**Dioxus**, pinned to an exact version at Phase 10 start, web target
first. Reasons, in order of weight:

1. The app is a local CSR tool (D1): dioxus' SPA-first model fits with no
   SSR/hydration machinery to disable and audit, while leptos' core
   strengths (SSR, hydration, server functions) are all dead weight for
   this workload.
2. The desktop fallback (D9) is the only concrete capability difference
   that matters to the personas (engineers who run offline at the
   optical bench), and dioxus provides it from the same code.
3. It is what `tasks/plan.md` already recommends (D10), and the user is
   agnostic — switching costs nothing and keeps the plan coherent.

The recommendation is conditional on Q1–Q3 below; if the pin-time check
shows dioxus' current release cannot compile in this workspace (edition
2024 / resolver), leptos is the fallback and this ADR flips with the same
drivers minus D9.

## Open questions for the Phase 10 check-in

- **Q1 — Version pin.** Exact dioxus (or leptos) version + feature set;
  confirm it compiles under edition 2024 with the workspace resolver and
  does not leak deps into the default members.
- **Q2 — Build tooling.** `dx serve`/`dx bundle` vs `trunk`: which is the
  documented dev loop, and how does the opt-in CI wasm build invoke it
  (toolchain install, caching)?
- **Q3 — Worker strategy.** Web worker via `gloo-worker`/`workerlib` for
  chunked `solve` runs (no wasm threads — SharedArrayBuffer needs
  COOP/COEP headers a local file can't assume). Confirm chunked
  single-threaded preview + worker-solver is the accepted 10.3
  responsiveness model (spec §3.4).
- **Q4 — Canvas backend.** 2D canvas `ImageData` vs WebGL for PSF
  intensity rendering at 2048²; decide before 10.2 implementation.
- **Q5 — Packaging for 10.4.** Web-only release, or web + desktop
  (`dioxus-desktop`) artifacts? Affects file dialogs and export UX.
- **Q6 — Seed policy on wasm.** Keep the `seed: Some(..)` requirement
  (spec §3.4), or add a `getrandom`-backed feature to `optics-inverse` so
  `seed: None` works in the browser?
- **Q7 — Session state.** Global store vs per-panel signals, and whether
  a saved "studio session" file format (spec gap G-9) is in or out of
  10.4 scope.
