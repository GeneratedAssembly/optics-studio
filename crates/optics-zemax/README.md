# optics-zemax — Zemax prescription import for `optics-core`

Parses text-format Zemax prescriptions (`.zmx`, line-oriented ASCII,
CRLF-tolerant) into a surface/glass description, resolves glasses through
a Sellmeier catalog, and builds `optics_core` systems: per-surface
[`optics_core::element::RefractiveSurface`]s composed through an
[`optics_core::beamline::Beamline`] with medium steps (the Phase-6
composition model). Paraxial reports (EFL/BFL) let an imported
prescription be checked against its own data — an imported file's image
distance *is* the paraxial back focal length at the primary wavelength,
which the golden tests use as validation.

Part of the [optics-studio workspace](../../README.md) (Phase 8,
`tasks/plan.md`). Dependencies: `optics-core` (path) + `thiserror` only.

## What it reads

| Input | Module | Notes |
|---|---|---|
| `.zmx` prescriptions | [`src/zmx.rs`](src/zmx.rs) | Sequential-mode `STANDARD` surfaces, radius/thickness/semi-diameter, `GLAS` names, `WAVM`/`PWAV` wavelengths, circular apertures. Unknown lines are ignored; unsupported constructs are documented rejections. |
| `.AGF` glass catalogs | [`src/agf.rs`](src/agf.rs) | Text catalogs as shipped by HOYA/OHARA distributions; parsed into `AgfCatalog` and layered into the resolver. |
| Built-in catalog | [`src/catalog.rs`](src/catalog.rs) | Published Sellmeier Book-1 coefficients: `N-BK7`, `CAF2` (H.H. Li 1980), `F_SILICA` (Malitson 1963). Provenance is explicit: external fits may differ from Zemax's proprietary `GCAT` data (paraxial invariants typically agree to ~0.5% EFL / a few % BFL). |
| System builder | [`src/builder.rs`](src/builder.rs) | `ImportedSystem` = parsed system + registry; produces `optics-core` materials, paraxial reports, and traced PSFs. |

## Scope

In scope (tasks/plan.md Phase 8): sequential `STANDARD` surfaces,
circular apertures (semi-diameters), air/glass media from the catalogs,
wavelengths from `WAVM`/`PWAV`.

Out of scope — rejected with `ZemaxError`, not silently dropped: binary
`.zar` archives, coatings, aspheres beyond even terms, non-sequential
modes, coordinate breaks.

## Usage

```rust
// ACA254-100-UV lives at the repo root under zmx-samples/; the crate's
// tests/data copy is used here for a manifest-relative path.
let sys = optics_zemax::ImportedSystem::load(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/ACA254-100-UV.zmx",
))?;
assert_eq!(sys.system().surfaces.len(), 6);

// The file's own image distance (89.4896 mm) is the paraxial BFL at the
// design center wavelength (0.325 um) — golden validation.
let par = sys.paraxial_at(0.325)?;
assert!((par.bfl_mm - 89.4896).abs() < 0.2);

// Or go straight to a wave-optics PSF through the composed beamline:
// let psf = sys.psf::<AsmEngine>(...)?;  (see builder docs)
```

From the browser, `.zmx`/`.AGF` text reaches the crate through
`ImportedSystem::parse_with` / `AgfCatalog::parse` (the `load*`
path-based helpers serve native callers and tests — no filesystem on
wasm).

## Build & test

```sh
cargo test -p optics-zemax      # parser + golden import tests (tests/import_doublet.rs)
cargo doc -p optics-zemax --no-deps
```

Golden data lives in [`tests/data/`](tests/data): the Thorlabs
`ACA254-100-UV.zmx` prescription plus HOYA and OHARA `.AGF` catalogs.
Sample prescriptions for manual/UI import are at
[`zmx-samples/`](../../zmx-samples) in the repo root (note: the `.zar`
sample there is binary and intentionally rejected).

## License

MIT OR Apache-2.0.
