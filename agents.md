# AGENTS.md — Development Guidelines for AI Assistants

## 1. Project Overview & Scope
This project is a dedicated **Diffractive Optics Engineering Suite**. 

### Core Architectural Mandate
- **Primary Engine Physics:** Wave optics propagation models (e.g., Angular Spectrum Method (ASM), Fresnel / Fraunhofer diffraction, Rayleigh-Sommerfeld).
- **Out of Scope (Initial Phase):** Ray tracing or geometric optics approximations. Do not attempt to introduce ray-based tools or hybrid approximations unless explicitly requested.

---

## 2. Interaction Protocol & Collaboration
You (the AI Agent) are operating as a **autonomous engineer executing an
approved roadmap** (protocol revision 2026-10-14, user directive: per-task
check-ins and per-commit approvals are retired in favour of proceeding as
planned). The approved plan is `tasks/plan.md` with the ordered task list in
`tasks/todo.md`.

### Mandatory Rules
1. **Plan-Gated Autonomy:** Build only what an approved plan task prescribes. New architectural directions outside the roadmap still require a user-approved plan first.
2. **Per-Task Execution:** Implement one task at a time from `tasks/todo.md`, in dependency order; tick the checkbox only when its acceptance criteria are met and gates pass.
3. **Design Points:** Resolve choices using the plan's resolved decisions and the documented conventions; pause and ask only when a genuine fork is *not* covered by the plan and cannot be settled by precedent.
4. **Incremental Delivery:** One logical conventional commit per task (see rule 5).
5. **Conventional Commits (mandatory):** Every commit message MUST follow the Conventional Commits specification (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`; `!` marker for breaking). No exceptions — commits without a conventional type prefix are non-conforming.
6. **Per-Commit Report (mandatory):** After each task commit, report the touched-file list and gate results in the session output (record, not a gate — no approval wait). Release cuts are included in the roadmap and do not require separate approval unless the plan changes.

---

## 3. Workflow Lifecycle

```
[ Take Next Task from tasks/todo.md ] ──> [ Verify physics against textbook ] ──> [ Implement + tests ] ──> [ Gates: fmt/clippy/doc/test ] ──> [ Commit (Conventional) + report touched files ] ──> [ Next task ]
```

1. **Step 1: Task context**
   - Re-read the task entry (acceptance criteria, verification, deps) and the
     relevant textbook section; specify mathematical formulations using
     Standard Markdown / LaTeX ($...$ or $$...$$) in docs and docstrings.
2. **Step 2: Implementation**
   - Write clean, typed code with clear error handling for edge cases (e.g.,
     aliasing detection, numerical precision limits).
3. **Step 3: Verification & Documentation**
   - Write unit/integration tests verifying physical accuracy (e.g.,
     conservation of energy/power, known analytical solutions for circular
     apertures). Update repository documentation and `CHANGELOG.md` in the
     same commit; tick the task in `tasks/todo.md`.
4. **Step 4: Gates & Commit**
   - `cargo fmt --check && cargo clippy -q --all-targets -- -D warnings &&
     cargo doc -q --no-deps && cargo test` must pass, then commit with a
     conventional message and report touched files.

---

## 4. Engineering & Scientific Standards

### Wave Optics & Physics
- **Sampling Verification:** Ensure propagation routines automatically check or expose spatial sampling criteria (e.g.,Nyquist-Shannon sampling limits for ASM phase transfer functions).
- **Prefer Prevention/Mitigation over Detection:** When a numerical guard exists, cheap closed-form criteria should first *correct* the situation (e.g., automatic grid densification up to a configured cap) rather than merely compute diagnostics to raise an error. Avoid expensive O(N²) sweeps used only to emit warnings.
- **Units:** Explicitly document unit expectations in docstrings (SI units by default: meters, radians, Hertz/wavelengths in meters).
- **Precision:** `f64` throughout (approved decision 2026-10-11).
- **Complex Fields:** Maintain explicit handling of amplitude and phase representations throughout field transformations.
- **Numerical Conventions:** FFT normalization, sign, bin order, and shift semantics are primary bug sources. State conventions in module docs and pin them with dedicated unit tests before dependent code is written.
- **Multiband Readiness:** Keep wavelength out of pure data types: `Field` is a spatial grid; `BandField` (grid + λ) is the unit of propagation/element interaction, so future incoherent multiband sources are collections of `BandField`s (intensities sum at detection).

### Code Quality & Structure
- **Modularity:** Separate physical propagation algorithms, optical element interfaces (e.g., phase masks, DOE surfaces), field initializers, and visualization/export utilities.
- **Data vs. Algorithms:** Keep data types pure (no algorithm machinery as methods); propagators are free functions in `propagation` that depend on field types, never the reverse. Composition sugar lives in `beamline` (delivered Phase 5).
- **Errors:** Plain enums deriving `Display`/`Error` via **thiserror** (revised decision 2026-10-11). Carry diagnostics (e.g., sampling reports) inside the variant.
- **Documentation Standards:** Every module, class, and function must include detailed docstrings (NumPy or Google style) explaining mathematical intent and physical inputs/outputs.
- **Type Safety:** Use strict type hints for all interfaces.

- **Physics Verification:** The ultimate source of truth is the diffractive_optics_design_fabrication_and_test.pdf text book located at the root of the workspace. Always verify any implementation that touches on a physics concept with the text in the
  aforementioned textbook. You may not assume that a "right answer" means that you have the physics modeled correctly.

---

## 5. Documentation Requirements

Maintain the following documentation structure alongside code updates:
- **`docs/physics.md`**: Detail the mathematical models, transfer functions, and validity regimes implemented.
- **`docs/architecture.md`**: Module interactions, class hierarchies, and API specs.
- **`CHANGELOG.md`**: Concise record of features added, APIs modified, and architectural decisions approved during check-ins. Update it in the same commit as the task it documents; record superseded decisions rather than deleting them.

---

## 6. How to Start a Session
At the start of any new session, read this file, then `tasks/plan.md` and
`tasks/todo.md` (plus the latest `.kilo/plans/*-handoff.md`), state the
current task in one line, and **proceed autonomously** under the §2
protocol — no approval to begin. Phase planning check-ins for phases not yet
in the roadmap remain collaborative.
