# AGENTS.md — Development Guidelines for AI Assistants

## 1. Project Overview & Scope
This project is a dedicated **Diffractive Optics Engineering Suite**. 

### Core Architectural Mandate
- **Primary Engine Physics:** Wave optics propagation models (e.g., Angular Spectrum Method (ASM), Fresnel / Fraunhofer diffraction, Rayleigh-Sommerfeld).
- **Out of Scope (Initial Phase):** Ray tracing or geometric optics approximations. Do not attempt to introduce ray-based tools or hybrid approximations unless explicitly requested.

---

## 2. Interaction Protocol & Collaboration
You (the AI Agent) are operating as a **collaborative partner**, not an autonomous builder. 

### Mandatory Rules
1. **No Autonomous Architecture/Major Code Runs:** Never build entire modules or write hundreds of lines of code without prior alignment on the plan.
2. **Step-by-Step Check-ins:** Propose a implementation step or architecture plan, wait for user review, and receive approval before implementing code. After approval, report back once task is complete with a list of files that were touched for the
   purposes of human review.
3. **Explicit Decision Points:** Whenever a design trade-off occurs (e.g., choice of FFT backend, boundary condition implementations, spatial sampling constraints, memory vs. performance trade-offs), pause and ask for the user's input.
4. **Incremental Delivery:** Break implementation into small, testable, well-documented pull requests or logical commits.
5. **Conventional Commits:** Every commit message follows the Conventional Commits specification (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).
6. **Task Cadence:** One logical commit per approved plan task; pause for review before committing and before starting the next task.

---

## 3. Workflow Lifecycle

```
[ Propose Strategy / Plan ] ──> [ Await Approval ] ──> [ Write Test / Code Step ] ──> [ Document & Review ]
```

1. **Step 1: Planning**
   - Outline the specific sub-task, algorithm choice, data structures, and mathematical formulations.
   - Specify mathematical formulas using Standard Markdown / LaTeX ($...\$$ or $\$...\$$).
2. **Step 2: Check-in**
   - Ask clarifying questions regarding sampling criteria, physical constraints, or engineering preferences.
3. **Step 3: Implementation**
   - Write clean, type-hinted code with clear error handling for edge cases (e.g., aliasing detection, numerical precision limits).
4. **Step 4: Verification & Documentation**
   - Write unit tests verifying physical accuracy (e.g., conservation of energy/power, known analytical solutions for circular apertures).
   - Update repository documentation.

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
- **Data vs. Algorithms:** Keep data types pure (no algorithm machinery as methods); propagators are free functions in `propagation` that depend on field types, never the reverse. Convenience method sugar is a Phase 5 decision point.
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
At the start of any new session or task, read this file, state your understanding of the current module objective, present a brief step-by-step plan for the immediate task, and **ask for approval to begin**.
