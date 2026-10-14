//! Engine dispatch adapter (gap G-8, docs/ux/spec.md §7).
//!
//! `Beamline::trace` takes a caller closure `(band, z, medium) -> engine`,
//! so the UI assembles the dispatch itself — thin plumbing, no physics:
//! this module only routes to the selected `optics_core::propagation`
//! engine and unifies the error taxonomy for the banner (§5.1).

use optics_core::field::BandField;
use optics_core::propagation::asm::{self, AsmConfig, PropagationError};
use optics_core::propagation::fraunhofer::{self, FraunhoferConfig, FraunhoferError};
use optics_core::propagation::fresnel::{self, FresnelConfig, FresnelError};
use optics_core::propagation::regime::Engine;

/// Per-engine configuration knobs (the shell's engine picker owns these;
/// the viewport only needs them to reproduce a run for a reference PSF).
#[derive(Clone, Debug)]
pub struct EngineSettings {
    /// Which engine to dispatch to.
    pub engine: Engine,
    /// ASM knobs ([`AsmConfig::default`] when unset).
    pub asm: AsmConfig,
    /// Fresnel knobs (both forms share this config).
    pub fresnel: FresnelConfig,
    /// Fraunhofer knobs.
    pub fraunhofer: FraunhoferConfig,
}

impl Default for EngineSettings {
    /// Robust default: [`Engine::Asm`] with each engine's own defaults —
    /// the regime ladder's fallback (core `regime` module docs).
    fn default() -> Self {
        Self {
            engine: Engine::Asm,
            asm: AsmConfig::default(),
            fresnel: FresnelConfig::default(),
            fraunhofer: FraunhoferConfig::default(),
        }
    }
}

/// Unified dispatch failure (spec §3.5: render from typed `Display`, no
/// string-only error path).
#[derive(Clone, Debug, thiserror::Error)]
pub enum EngineError {
    /// ASM failure (carries the [`PropagationError`] `Display`).
    #[error(transparent)]
    Asm(#[from] PropagationError),
    /// Fresnel failure (either form).
    #[error(transparent)]
    Fresnel(#[from] FresnelError),
    /// Fraunhofer failure.
    #[error(transparent)]
    Fraunhofer(#[from] FraunhoferError),
    /// The selected engine is paraxial-vacuum only and the step requested a
    /// medium other than `n = 1` (spec §3.1: `Beamline::push_medium` — the
    /// UI must not silently drop the index).
    #[error("engine {engine} does not model a medium (n = {medium} requested)")]
    MediumUnsupported {
        /// The engine that refused.
        engine: Engine,
        /// The requested refractive index.
        medium: f64,
    },
}

/// Propagates `band` a signed distance `z` (m) through a medium of index
/// `medium` using the configured engine.
///
/// Only [`Engine::Asm` models media][asm] (`AsmConfig::medium`); the
/// paraxial forms reject `medium != 1.0` with [`EngineError::MediumUnsupported`]
/// rather than pretending vacuum.
pub fn propagate(
    band: &BandField,
    z: f64,
    medium: f64,
    settings: &EngineSettings,
) -> Result<BandField, EngineError> {
    match settings.engine {
        Engine::Asm => {
            let mut cfg = settings.asm; // AsmConfig is Copy
            cfg.medium = medium;
            Ok(asm::propagate(band, z, &cfg)?)
        }
        Engine::FresnelTransfer => {
            paraxial_medium(settings.engine, medium)?;
            Ok(fresnel::propagate(band, z, &settings.fresnel)?)
        }
        Engine::FresnelConvolution => {
            paraxial_medium(settings.engine, medium)?;
            Ok(fresnel::propagate_convolution(band, z, &settings.fresnel)?)
        }
        Engine::Fraunhofer => {
            paraxial_medium(settings.engine, medium)?;
            Ok(fraunhofer::propagate(band, z, &settings.fraunhofer)?)
        }
    }
}

fn paraxial_medium(engine: Engine, medium: f64) -> Result<(), EngineError> {
    if (medium - 1.0).abs() > f64::EPSILON {
        Err(EngineError::MediumUnsupported { engine, medium })
    } else {
        Ok(())
    }
}

/// A [`beamline closure](optics_core::beamline::Beamline::trace) form of
/// [`propagate`] for the given settings.
pub fn closure(
    settings: EngineSettings,
) -> impl Fn(&BandField, f64, f64) -> Result<BandField, EngineError> {
    move |band, z, medium| propagate(band, z, medium, &settings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use optics_core::field::BandField;

    fn band() -> BandField {
        BandField::uniform(1.0, 0.0, 32, 32, 20e-6, 20e-6, 550e-9).unwrap()
    }

    #[test]
    fn asm_dispatch_runs() {
        let s = EngineSettings::default();
        let out = propagate(&band(), 0.001, 1.0, &s).expect("asm ok");
        assert_eq!(out.field().dims(), (32, 32));
    }

    #[test]
    fn asm_models_media_via_cfg() {
        let s = EngineSettings::default();
        // n = 1.5 reaches AsmConfig::medium (no refusal).
        let out = propagate(&band(), 0.001, 1.5, &s);
        assert!(out.is_ok());
    }

    #[test]
    fn paraxial_engines_refuse_media() {
        let s = EngineSettings {
            engine: Engine::FresnelTransfer,
            ..Default::default()
        };
        let e = propagate(&band(), 0.001, 1.5, &s).unwrap_err();
        assert!(matches!(
            e,
            EngineError::MediumUnsupported {
                engine: Engine::FresnelTransfer,
                medium: 1.5
            }
        ));
        // Vacuum passes the gate (engine math may still fail regime checks,
        // but not for the medium reason).
        let ok = propagate(&band(), 0.0, 1.0, &s);
        assert!(ok.is_ok());
    }

    #[test]
    fn closure_compiles_as_beamline_engine() {
        // The exact shape Beamline::trace expects.
        let f = closure(EngineSettings::default());
        let out = f(&band(), 0.0005, 1.0).expect("closure ok");
        assert_eq!(out.lambda(), 550e-9);
    }
}
