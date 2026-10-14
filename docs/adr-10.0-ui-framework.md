# ADR-10.0: UI framework for `optics-studio` — dioxus

Status: **accepted** — Phase 10 task 10.0 (GitHub #8, Paperclip GEN-51),
2026-10-14. Resolved decision 4 (`tasks/plan.md`, user, 2026-10-14) fixed
"web from the start" and delegated the framework choice to this ADR with the
user agnostic. This record ratifies the draft
[`docs/ux/adr-10.0-draft.md`](ux/adr-10.0-draft.md) (S1-D pre-work, GEN-42),
which stays in place as history. Companion document:
[`docs/ux/spec.md`](ux/spec.md) — the screens and bindings this framework
must implement.

## Decision

**Dioxus**, pinned to `=0.7.10`, web target (`wasm32-unknown-unknown`),
client-side rendering only. **Leptos 0.8.22** is the documented fallback.

The framework dependency tree lives entirely inside `crates/studio/`
(`optics-studio`). The headless crates (`optics-core`, `optics-zemax`,
`optics-inverse`) gain no UI dependencies, and the studio crate stays out of
`workspace.default-members` so `cargo test`/gates never build it
(`tasks/plan.md`, architecture decisions).

## Why dioxus (drivers from the draft, in order of weight)

1. **D1 — the app is a local CSR tool.** Dioxus is SPA-first: no
   SSR/hydration machinery to disable and audit. Leptos' core strengths
   (SSR, hydration, server functions) are dead weight for a single-user
   local tool with no server and no SEO.
2. **D9 — desktop fallback from the same code.** `dioxus-desktop` (wry)
   gives persona P1 (bench engineers working offline) a native-window
   build without Tauri glue. This is the only concrete capability
   difference that matters here.
3. **D10 — plan coherence.** `tasks/plan.md` already recommends dioxus;
   the user is agnostic, so confirming costs nothing.

Fine-grained reactivity (leptos' headline strength, D3) is marginal for this
workload because heavy output goes to canvas painting, not DOM diffing
(`docs/ux/spec.md` §4). Canvas, file-picker, and worker stories (D4–D6) are
equivalent across both frameworks — thin `web-sys` either way.

## Pin-time verification (resolves draft Q1)

Checked 2026-10-14 on this machine (rustc 1.98.0, cargo 1.98.0, stable):

- A standalone crate with `edition = "2024"` and
  `dioxus = { version = "=0.7.10", features = ["web"] }` passes
  `cargo check --target wasm32-unknown-unknown` with zero errors.
- Dioxus 0.7.10 declares MSRV 1.83 — below the workspace toolchain, so
  edition 2024 / resolver constraints are satisfied.
- 0.7.10 is the newest stable release; 0.8.0 is alpha (`0.8.0-alpha.2`) and
  is not a candidate. Leptos 0.8.22 (MSRV 1.88) is current stable for the
  fallback path.

The draft's flip condition ("dioxus cannot compile in this workspace") did
not trigger, so the recommendation stands. When task 10.1 adds
`crates/studio`, it must be a workspace member excluded from
`default-members`; re-run this check inside the workspace at that point as
the leak guard.

## Ratified answers to the draft's open questions

- **Q1 — Version pin:** `=0.7.10`, features `["web"]`. Upgrade only as a
  deliberate workspace commit with gates green.
- **Q2 — Build tooling:** `dx` (dioxus-cli) is the documented dev loop
  (`dx serve`, `dx build --platform web`); `trunk` is not used. Dioxus
  asset handling (manganis) and subsecond hot-reload integrate with `dx`.
  Any CI wasm job is opt-in and outside the default gates.
- **Q3 — Worker strategy:** accepted responsiveness model — chunked
  single-threaded preview plus solver runs in a dedicated web worker with
  progress messages (no wasm threads; `SharedArrayBuffer` needs COOP/COEP
  headers a local file cannot assume). Worker crate choice
  (`gloo-worker` vs `comlink`) is an implementation decision for task 10.3.
- **Q4 — Canvas backend:** deferred to task 10.2 (2D `ImageData` vs WebGL
  at 2048²).
- **Q5 — Packaging for 10.4:** deferred to task 10.4; web-only is the
  default, desktop artifact optional.
- **Q6 — Seed policy on wasm:** unchanged — solver runs pass
  `StochasticConfig::seed = Some(..)` (`docs/ux/spec.md` §3.4) until
  `optics-inverse` ships a `getrandom`-js feature.
- **Q7 — Session state:** deferred to 10.1+; a saved "studio session" file
  format (spec gap G-9) is out of scope for 10.4 unless the plan changes.

## Consequences

- Tasks 10.1–10.3 build `optics-studio` on dioxus 0.7.10 web per
  `docs/ux/spec.md`; no framework code in the headless crates.
- Pre-1.0 churn risk is held by the exact pin; breaking upgrades are their
  own commits.
- If a future dioxus release breaks the workspace build, leptos 0.8.22
  (CSR feature set) is the fallback and this ADR is superseded — same
  drivers minus D9.
