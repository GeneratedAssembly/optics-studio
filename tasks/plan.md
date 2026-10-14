# Implementation Plan: optics-rs — Forward Modelling / Inverse Design Suite

## Overview

The goal is a headless wave-optics **forward-modelling + inverse-design**
suite comparable in feature scope to Zemax OpticStudio, organized as a cargo
workspace of cooperating crates. The core physics crate (`optics-core`,
renamed from `optics-rs` at Phase 7 and released at 0.7.0)
already delivers coherent scalar propagation (ASM/Fresnel/Fraunhofer/RS-I),
thin and thick refractive/diffractive elements, apertures/gratings/wedges,
Gaussian/HG/LG/point sources, and PSF composition, all validated against the
project textbook (25 integration + 140 unit tests). This plan closes the
remaining **core forward-physics** gaps, then adds three separate crates:
**Zemax prescription import**, **inverse design** (least-squares,
Gerchberg–Saxton, stochastic, gradient/adjoint, and deep-learning surrogates),
and a **UI** that wraps the headless APIs.

The workspace is the spine: every later crate depends on the core's public
API and none of them feed back into it (the agents.md data/algorithm split,
extended across crates: core knows nothing about import, inverse, or UI).

## Repository / crate layout (target)

```
optics-rs/                      # repo dir; workspace root; ROOT PACKAGE = core
├── Cargo.toml       (crate: optics-core)      # renamed from optics-rs (Task 7.0)
├── src/, tests/, examples/                    # core untouched (no git mv)
├── crates/
│   ├── zemax/       (crate: optics-zemax)     # .zmx parse -> core system
│   ├── inverse/     (crate: optics-inverse)   # objectives + solvers
│   └── studio/      (crate: optics-studio)    # web UI (opt-in workspace)
├── tasks/plan.md, tasks/todo.md               # this plan + task list
└── .kilo/plans/*-handoff.md                   # per-phase session handoffs
```

Root-as-core avoids moving the mature crate; the workspace adds sibling
crates under `crates/*` with path deps on the root package.

**Naming rule (locked):** the core crate is `optics-core`, never `core` —
a crate named `core` collides with Rust's built-in `core` in extern-prelude
paths (`use core::...`) and breaks resolution. Directory names may still
read `core/` where convenient; crate/package names always carry the
`optics-` prefix.

Dependency direction (build bottom-up):
`core` → { `zemax` → core, `inverse` → core (+zemax for real systems) } → `ui` → all.

## Architecture decisions (already fixed)

- **Scalar, coherent-per-band + incoherent-across-band** wave model stays the
  core representation; polarization/vector fields remain out of scope
  (agents.md). Inverse design operates on this model.
- **Headless core.** No plotting/serialization in `core` (Phase-5 decision);
  the `ui` crate owns rendering; `zemax` owns I/O.
- **Workspace, not one crate.** Import/inverse/UI are separate crates with
  their own dependency budgets (the core stays minimal: rustfft, num-complex,
  thiserror). This is the user's explicit requirement.
- **`optics-studio` (web UI) excluded from `workspace.default-members`**
  (heavy wasm/toolchain deps) so `cargo test` in CI never builds it; it is
  opted into explicitly.
- **Conventional Commits + per-task approval** carry across all crates.

## Resolved decisions (user, 2026-10-14 — planning check-in)

1. **Tilt/decenter:** small paraxial tilt in the core NOW (Phase 7
   Task 7.5); full 3D coordinate breaks deferred — added only if the
   `zemax` import proves they are required.
2. **Inverse order locked:** finite-difference gradients, derivative-free
   (random walk / annealing / ES), and Gerchberg–Saxton first; adjoint as a
   later milestone; autodiff re-port ONLY if adjoint proves insufficient.
3. **Deep-learning stack: `burn`** (GPU allowed in `optics-inverse`; the
   core stays dependency-minimal). Training data generated with the
   headless forward model.
4. **UI: web from the start.** Framework choice (dioxus recommended for
   full-Rust + wasm; leptos alternative) is a Phase-10 ADR — user is
   agnostic.
6. **Binary Zemax files are OUT of import scope** (2026-10-14, from the
   Edmund Optics 2019 `.ZMF` inspection): 7432 part-number records,
   high-entropy payloads, no text/binary-ZMX/zlib structure — same class
   as the `.zar` archives. Supported corpus = text `.zmx` (the ACA254
   golden validates the pipeline end-to-end). For batch designs: export
   from OpticStudio with File > Save As > Text, or per-product `.zmx`
   downloads. External `.AGF` text catalogs remain the pending upgrade
   for exact catalog reproduction.
5. **Crate names locked:** core `optics-core` (`core` reserved by the
   language — never as a crate name), `optics-zemax`, `optics-inverse`,
   **`optics-studio`** (UI).
7. **Merit extensibility (Phase-9 check-in, user 2026-10-14):** simple
   common FOMs are built-ins; CRLB-class merits are NOT baked in — they
   "depend heavily on parameters we don't know a priori" and are expressed
   by the user as **custom merit functions in Python or mathematical
   notation**. That frontend is its own milestone (optics-inverse 0.2.0);
   the 0.1.0 framework must keep the `Merit` seam (trait + numpy-ready
   `Observation` layout) so it plugs in without redesign.
8. **Adjoint in Phase 9 before 0.1.0** (grid-DOF merits are FD-infeasible);
   DL `burn` milestone stays in Phase 9 (feature-gated, ships as 0.3.0).

## Phase 7 — Core forward-physics close-out (crate: optics-core → 0.7.0)

Close the three core gaps so the crate is *feature-complete for forward
modelling*, then freeze the API surface the other crates build on.

## Phase 8 — Zemax import (crate: optics-zemax → 0.1.0)

Parse `.zmx` prescriptions + glass catalogs into core systems; round-trip
validate. Depends on Phase 7 (Sellmeier, decenter/tilt).

## Phase 9 — Inverse design (crate: optics-inverse → 0.1.0)

Objective framework + solver zoo, from derivative-free to gradient/adjoint
to deep surrogates. Depends on Phase 7 core; Phase 8 for realistic systems.

## Phase 10 — UI (crate: optics-studio → 0.1.0)

Web front-end (framework ADR at phase start; dioxus recommended) over the
headless APIs. Depends on 7–9.

See `tasks/todo.md` for the ordered, sized task list with acceptance criteria
and checkpoints. Distant-phase tasks (9–10) are epic-level and will be
re-decomposed into S/M tasks at their own phase-planning check-in, per the
agents.md §6 protocol.

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Differentiable propagation for gradient inverse is expensive | High | Lead with adjoint/finite-difference + GS + stochastic; autodiff is optional |
| `.zmx` format is undocumented / varies by version | High | Pin to a known sample corpus; version-gated parser; golden-file tests |
| Workspace move breaks the mature core + git history | Med | Pure `git mv`, CI-green checkpoint before any feature work |
| DL surrogate needs large datasets / GPU | Med | `burn` locked; generate data headless; DL milestone last |
| UI scope creep into physics | Med | UI depends on core APIs only; no physics in the UI crate |
| Scalar model limits fidelity vs real Zemax ray tools | Low (by design) | Documented scope; wave-optics is the product, not ray tracing |
