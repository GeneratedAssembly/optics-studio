//! Engine picker model + the G-8 dispatch adapter (docs/ux/spec.md §7).
//!
//! `Beamline::trace` takes a caller closure `(band, z, medium) -> engine`,
//! so the UI assembles the dispatch itself — thin plumbing, no physics:
//! this module only routes to the selected `optics_core::propagation`
//! engine, unifies the error taxonomy for the banner (spec §3.5), and
//! formats the per-run sampling diagnostics (spec §5.1–§5.2).

use optics_core::field::BandField;
use optics_core::propagation::asm::{self, AsmConfig, PropagationError};
use optics_core::propagation::fraunhofer::{self, FraunhoferConfig, FraunhoferError};
use optics_core::propagation::fresnel::{self, FresnelConfig, FresnelError};
use optics_core::propagation::rayleigh_sommerfeld::{
    self, RayleighSommerfeldConfig, RayleighSommerfeldError,
};
use optics_core::propagation::regime::{self, RegimeReport};

/// The engine picker's five choices (spec §3.1: Asm / Fresnel transfer /
/// Fresnel convolution / Fraunhofer / Rayleigh–Sommerfeld).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnginePick {
    /// Angular spectrum method.
    Asm,
    /// Fresnel transfer function.
    FresnelTransfer,
    /// Fresnel convolution form.
    FresnelConvolution,
    /// Fraunhofer far-field.
    Fraunhofer,
    /// Rayleigh–Sommerfeld first integral.
    RayleighSommerfeld,
}

impl EnginePick {
    /// All picker entries, in display order.
    pub const ALL: [EnginePick; 5] = [
        EnginePick::Asm,
        EnginePick::FresnelTransfer,
        EnginePick::FresnelConvolution,
        EnginePick::Fraunhofer,
        EnginePick::RayleighSommerfeld,
    ];

    /// Whether the engine models a propagation medium other than vacuum
    /// (`AsmConfig::medium` is only read by the ASM path).
    pub fn supports_medium(self) -> bool {
        matches!(self, EnginePick::Asm)
    }
}

impl std::fmt::Display for EnginePick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            EnginePick::Asm => "ASM",
            EnginePick::FresnelTransfer => "Fresnel (transfer)",
            EnginePick::FresnelConvolution => "Fresnel (convolution)",
            EnginePick::Fraunhofer => "Fraunhofer",
            EnginePick::RayleighSommerfeld => "Rayleigh–Sommerfeld",
        };
        f.write_str(name)
    }
}

/// Picker configuration: the selected engine plus every knob the spec's
/// engine panel exposes (padding, band-limited, strict, max_padding,
/// medium). Defaults follow the core `Default` impls.
#[derive(Clone, Copy, Debug)]
pub struct EngineSettings {
    /// Selected engine.
    pub pick: EnginePick,
    /// ASM knobs.
    pub asm: AsmConfig,
    /// Fresnel knobs (shared by both forms).
    pub fresnel: FresnelConfig,
    /// Fraunhofer knobs.
    pub fraunhofer: FraunhoferConfig,
    /// Rayleigh–Sommerfeld knobs.
    pub rs: RayleighSommerfeldConfig,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            pick: EnginePick::Asm,
            asm: AsmConfig::default(),
            fresnel: FresnelConfig::default(),
            fraunhofer: FraunhoferConfig::default(),
            rs: RayleighSommerfeldConfig::default(),
        }
    }
}

/// Unified dispatch failure (spec §3.5: every banner renders from a typed
/// `Display`; no string-only error path).
#[derive(Clone, Debug, thiserror::Error)]
pub enum DispatchError {
    /// ASM failure (carries the [`PropagationError`] `Display`, which
    /// appends the aliasing hint/remedy text).
    #[error(transparent)]
    Asm(#[from] PropagationError),
    /// Fresnel failure (either form).
    #[error(transparent)]
    Fresnel(#[from] FresnelError),
    /// Fraunhofer failure.
    #[error(transparent)]
    Fraunhofer(#[from] FraunhoferError),
    /// Rayleigh–Sommerfeld failure.
    #[error(transparent)]
    RayleighSommerfeld(#[from] RayleighSommerfeldError),
    /// The selected engine does not model media and the step requested
    /// `n != 1` — refuse loudly instead of silently propagating in vacuum.
    #[error("engine {pick} does not model a medium (n = {medium} requested)")]
    MediumUnsupported {
        /// The engine that refused.
        pick: EnginePick,
        /// The requested refractive index.
        medium: f64,
    },
}

/// Propagate `band` a distance `z` (m) through index `medium` with the
/// configured engine.
pub fn propagate(
    settings: &EngineSettings,
    band: &BandField,
    z: f64,
    medium: f64,
) -> Result<BandField, DispatchError> {
    match settings.pick {
        EnginePick::Asm => Ok(asm::propagate(
            band,
            z,
            &AsmConfig {
                medium,
                ..settings.asm
            },
        )?),
        EnginePick::FresnelTransfer => {
            require_vacuum(settings.pick, medium)?;
            Ok(fresnel::propagate(band, z, &settings.fresnel)?)
        }
        EnginePick::FresnelConvolution => {
            require_vacuum(settings.pick, medium)?;
            Ok(fresnel::propagate_convolution(band, z, &settings.fresnel)?)
        }
        EnginePick::Fraunhofer => {
            require_vacuum(settings.pick, medium)?;
            Ok(fraunhofer::propagate(band, z, &settings.fraunhofer)?)
        }
        EnginePick::RayleighSommerfeld => {
            require_vacuum(settings.pick, medium)?;
            Ok(rayleigh_sommerfeld::propagate(band, z, &settings.rs)?)
        }
    }
}

fn require_vacuum(pick: EnginePick, medium: f64) -> Result<(), DispatchError> {
    if (medium - 1.0).abs() <= f64::EPSILON {
        Ok(())
    } else {
        Err(DispatchError::MediumUnsupported { pick, medium })
    }
}

/// Sampling-banner severity (spec §5.2). Re-export of the shared
/// `viewmodel::banner` vocabulary — the shell and the 10.2 viewport speak
/// one banner type.
pub use crate::viewmodel::banner::{Banner, Severity as BannerLevel};

/// Whether the selected engine runs with its strict sampling guard on.
pub fn is_strict(settings: &EngineSettings) -> bool {
    match settings.pick {
        EnginePick::Asm => settings.asm.strict,
        EnginePick::FresnelTransfer | EnginePick::FresnelConvolution => settings.fresnel.strict,
        EnginePick::Fraunhofer => settings.fraunhofer.strict,
        EnginePick::RayleighSommerfeld => settings.rs.strict,
    }
}

/// Pure-diagnostics report for the last planned run (spec §5.1: call the
/// `*_report` functions before tracing; nothing executes a propagation).
pub fn run_banner(settings: &EngineSettings, band: &BandField, z: f64, medium: f64) -> Banner {
    match settings.pick {
        EnginePick::Asm => {
            let cfg = AsmConfig {
                medium,
                ..settings.asm
            };
            let r = asm::sampling_report(band, z, &cfg);
            if !r.nyquist_ok {
                Banner {
                    severity: BannerLevel::Rejected,
                    headline: "ASM sampling rejected (pre-run report)".into(),
                    detail: Some(r.to_string()),
                    remedies: crate::viewmodel::banner::standard_remedies(),
                }
            } else if r.used_padding > r.requested_padding + f64::EPSILON {
                Banner {
                    severity: BannerLevel::Mitigated,
                    headline: format!(
                        "engine densified: padding {} → {} (padded {:?})",
                        r.requested_padding, r.used_padding, r.padded_dims
                    ),
                    detail: Some(r.to_string()),
                    remedies: Vec::new(),
                }
            } else {
                Banner {
                    severity: BannerLevel::Ok,
                    headline: format!(
                        "ASM grid {:?} pitch ({:.3}, {:.3}) mm · padding {} · max phase step ({:.3}, {:.3}) rad",
                        r.input_dims,
                        r.pitch.0 * 1e3,
                        r.pitch.1 * 1e3,
                        r.used_padding,
                        r.max_phase_step_x,
                        r.max_phase_step_y,
                    ),
                    detail: None,
                    remedies: Vec::new(),
                }
            }
        }
        EnginePick::FresnelTransfer | EnginePick::FresnelConvolution => {
            let r = fresnel::sampling_report(band, z, &settings.fresnel);
            let ok = r.nyquist_ok && r.kernel_nyquist_ok && r.paraxial_ok;
            Banner {
                severity: if ok {
                    BannerLevel::Ok
                } else if r.used_padding > r.requested_padding + f64::EPSILON {
                    BannerLevel::Mitigated
                } else {
                    BannerLevel::Rejected
                },
                headline: format!(
                    "{} grid {:?} pitch ({:.3}, {:.3}) mm",
                    settings.pick,
                    r.input_dims,
                    r.pitch.0 * 1e3,
                    r.pitch.1 * 1e3
                ),
                detail: Some(r.to_string()),
                remedies: if ok {
                    Vec::new()
                } else {
                    crate::viewmodel::banner::standard_remedies()
                },
            }
        }
        EnginePick::Fraunhofer => {
            let r = fraunhofer::regime_report(band, z, &settings.fraunhofer);
            Banner {
                severity: if r.regime_ok {
                    BannerLevel::Ok
                } else {
                    BannerLevel::Rejected
                },
                headline: format!(
                    "Fraunhofer N_F ({:.2}, {:.2}) vs max {:.2}",
                    r.n_f_x, r.n_f_y, r.max_n_f
                ),
                detail: Some(r.to_string()),
                remedies: if r.regime_ok {
                    Vec::new()
                } else {
                    crate::viewmodel::banner::standard_remedies()
                },
            }
        }
        EnginePick::RayleighSommerfeld => {
            let r = rayleigh_sommerfeld::sampling_report(band, z, &settings.rs);
            let ok = r.kernel_nyquist_ok && r.paraxial_ok;
            Banner {
                severity: if ok {
                    BannerLevel::Ok
                } else {
                    BannerLevel::Rejected
                },
                headline: format!(
                    "Rayleigh–Sommerfeld grid {:?} pitch ({:.3}, {:.3}) mm",
                    r.input_dims,
                    r.pitch.0 * 1e3,
                    r.pitch.1 * 1e3
                ),
                detail: Some(r.to_string()),
                remedies: if ok {
                    Vec::new()
                } else {
                    crate::viewmodel::banner::standard_remedies()
                },
            }
        }
    }
}

/// Regime advisory (spec §5.2): non-blocking line when the *selected*
/// engine sits outside its regime even though the run succeeded.
pub fn regime_advisory(
    pick: EnginePick,
    band: &BandField,
    z: f64,
    aperture: Option<f64>,
) -> Option<String> {
    let r: RegimeReport = regime::regime_report_with(band, z, aperture);
    match pick {
        EnginePick::Fraunhofer if !r.far_field => Some(format!(
            "advisory: N_F = {:.1} > 1 — Fraunhofer selected but the grid is not far-field; recommended {}",
            r.n_f, r.recommended
        )),
        EnginePick::FresnelTransfer
        | EnginePick::FresnelConvolution
        | EnginePick::RayleighSommerfeld
            if !r.paraxial =>
        {
            Some(format!(
                "advisory: max angle {:.3} rad > 0.1 — paraxial engine selected outside the paraxial regime; recommended {}",
                r.max_angle, r.recommended
            ))
        }
        _ => None,
    }
}

/// Auto-suggest line for the picker (spec §3.1): the regime report's
/// recommendation mapped onto the picker's entries.
pub fn suggested(band: &BandField, z: f64, aperture: Option<f64>) -> EnginePick {
    match regime::regime_report_with(band, z, aperture).recommended {
        regime::Engine::Asm => EnginePick::Asm,
        regime::Engine::FresnelTransfer => EnginePick::FresnelTransfer,
        regime::Engine::FresnelConvolution => EnginePick::FresnelConvolution,
        regime::Engine::Fraunhofer => EnginePick::Fraunhofer,
    }
}
