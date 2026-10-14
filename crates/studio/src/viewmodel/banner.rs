//! Sampling-warning banner (docs/ux/spec.md §5.2 banner rules).
//!
//! The banner reports what the engine *did*, not just whether it succeeded,
//! and classifies a run into one of four severities:
//!
//! - **Green** ([`Severity::Ok`]) — Nyquist satisfied, no padding growth.
//! - **Yellow** ([`Severity::Mitigated`]) — the engine auto-densified
//!   (`used_padding > requested_padding`): the visible cost of prevention,
//!   which the spec says must never be hidden.
//! - **Red** ([`Severity::Rejected`]) — an `AliasingRisk` / `RegimeViolation`
//!   came back; the banner carries the report `Display` text plus remedy
//!   hints and one-click knob suggestions.
//! - **Advisory** ([`Severity::Advisory`]) — a non-blocking regime note
//!   (Fraunhofer chosen but `far_field == false`, or a Fresnel engine used
//!   with `paraxial == false`) even when the run succeeded.
//!
//! This module never runs a propagation: it consumes the core diagnostics
//! types the engine already produced (`SamplingReport`, `RegimeReport`, the
//! error variants) and maps them to display state.

use optics_core::propagation::asm::SamplingReport;
use optics_core::propagation::regime::{Engine, RegimeReport};

/// Banner severity (spec §5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// Green.
    Ok,
    /// Yellow — mitigated by densification.
    Mitigated,
    /// Red — rejected; carries the report + remedies.
    Rejected,
    /// Regime advisory — non-blocking.
    Advisory,
}

impl Severity {
    /// A stable CSS class name for the banner element.
    pub fn css_class(self) -> &'static str {
        match self {
            Severity::Ok => "banner-ok",
            Severity::Mitigated => "banner-mitigated",
            Severity::Rejected => "banner-rejected",
            Severity::Advisory => "banner-advisory",
        }
    }
}

/// A resolved banner: severity, the primary line, optional detail, and one
/// or more remedy hints (the engine `Display` impls already append the
/// "hint: … remedy: …" text — the banner shows it verbatim).
#[derive(Clone, Debug, PartialEq)]
pub struct Banner {
    /// Severity class.
    pub severity: Severity,
    /// Compact primary line (spec §5.2 green: grid, pitch, padding, phase
    /// step).
    pub headline: String,
    /// Full report text for red/advisory banners (the `Display` of the
    /// carried report).
    pub detail: Option<String>,
    /// Remedy lines mapped to real knobs (spec §5.2 red).
    pub remedies: Vec<String>,
}

/// Compact green/yellow headline from an ASM [`SamplingReport`].
///
/// Yellow when `strict`-mode densification grew the grid
/// (`used_padding > requested_padding`, spec §5.2); green otherwise.
pub fn from_asm(report: &SamplingReport) -> Banner {
    let mitigated = report.used_padding > report.requested_padding + f64::EPSILON;
    let sev = if mitigated {
        Severity::Mitigated
    } else if report.nyquist_ok {
        Severity::Ok
    } else {
        Severity::Rejected
    };
    let headline = format!(
        "grid {}x{} pitch {:.2}x{:.2} µm · padding {:.2}→{:.2} · Δφx={:.2} Δφy={:.2} rad · nyquist={}",
        report.input_dims.0,
        report.input_dims.1,
        report.pitch.0 * 1e6,
        report.pitch.1 * 1e6,
        report.requested_padding,
        report.used_padding,
        report.max_phase_step_x,
        report.max_phase_step_y,
        report.nyquist_ok,
    );
    let mut remedies = Vec::new();
    if mitigated {
        remedies.push(format!(
            "engine densified padding {:.2} → {:.2} (final padded {}x{}) — the visible cost of prevention",
            report.requested_padding, report.used_padding, report.padded_dims.0, report.padded_dims.1
        ));
    }
    if sev == Severity::Rejected {
        remedies.push(
            "raise AsmConfig padding / max_padding, coarsen pitch, reduce |z|, or enable band_limited"
                .into(),
        );
    }
    Banner {
        severity: sev,
        headline,
        detail: if sev == Severity::Rejected {
            Some(report.to_string())
        } else {
            None
        },
        remedies,
    }
}

/// Regime advisory (spec §5.2 "Regime advisory") for a chosen engine.
///
/// Non-blocking: returns [`Severity::Advisory`] with the reason when the
/// selected engine sits outside its justified regime, else [`None`] so the
/// caller keeps the run-success banner.
pub fn regime_advisory(engine: Engine, regime: &RegimeReport) -> Option<Banner> {
    let reason = match engine {
        Engine::Fraunhofer if !regime.far_field => Some(format!(
            "Fraunhofer selected but N_F={:.2} > 1 (far_field=false) — pattern is not in the far field",
            regime.n_f
        )),
        Engine::FresnelTransfer | Engine::FresnelConvolution if !regime.paraxial => Some(format!(
            "Fresnel engine used with θ_max={:.2} rad > 0.1 (paraxial=false) — chirp truncation may be invalid",
            regime.max_angle
        )),
        _ => None,
    }?;
    Some(Banner {
        severity: Severity::Advisory,
        headline: reason,
        detail: Some(regime.to_string()),
        remedies: vec![format!(
            "recommended engine for this geometry: {} (regime ladder)",
            regime.recommended
        )],
    })
}

/// Red banner from a rejected run's error text (the caller passes the typed
/// error's `Display`, which already carries the report + hint/remedy).
pub fn rejected(error_display: String, remedies: Vec<String>) -> Banner {
    Banner {
        severity: Severity::Rejected,
        headline: "run rejected — sampling/regime guard tripped".into(),
        detail: Some(error_display),
        remedies,
    }
}

/// Standard red-banner remedies (spec §5.2 one-click fixes).
pub fn standard_remedies() -> Vec<String> {
    vec![
        "raise AsmConfig::max_padding / padding".into(),
        "coarsen the grid pitch".into(),
        "reduce |z|".into(),
        "switch engine per RegimeReport::recommended".into(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(used: f64, req: f64, nyquist: bool) -> SamplingReport {
        SamplingReport {
            lambda: 550e-9,
            medium: 1.0,
            z: 0.05,
            input_dims: (256, 256),
            pitch: (10e-6, 10e-6),
            requested_padding: req,
            used_padding: used,
            padded_dims: ((256.0 * used) as usize, (256.0 * used) as usize),
            band_limit_fx: 1.0 / 550e-9,
            band_limit_fy: 1.0 / 550e-9,
            max_phase_step_x: 0.5,
            max_phase_step_y: 0.5,
            nyquist_ok: nyquist,
        }
    }

    #[test]
    fn green_when_ok_no_growth() {
        let b = from_asm(&report(2.0, 2.0, true));
        assert_eq!(b.severity, Severity::Ok);
        assert!(b.detail.is_none());
        assert!(b.remedies.is_empty());
    }

    #[test]
    fn yellow_when_mitigated() {
        let b = from_asm(&report(4.0, 2.0, true));
        assert_eq!(b.severity, Severity::Mitigated);
        assert!(
            b.remedies
                .iter()
                .any(|r| r.contains("densified padding 2.00 → 4.00"))
        );
    }

    #[test]
    fn red_when_rejected() {
        let b = from_asm(&report(2.0, 2.0, false));
        assert_eq!(b.severity, Severity::Rejected);
        assert!(b.detail.is_some());
        assert!(!b.remedies.is_empty());
    }

    #[test]
    fn css_classes_distinct() {
        assert_eq!(Severity::Ok.css_class(), "banner-ok");
        assert_eq!(Severity::Mitigated.css_class(), "banner-mitigated");
    }

    #[test]
    fn advisory_for_fraunhofer_near_field() {
        let regime = RegimeReport {
            z: 0.01,
            half_extent: (1e-3, 1e-3),
            extent_used: 1e-3,
            extent_is_aperture: true,
            n_f: 100.0,
            far_field: false,
            max_angle: 0.2,
            paraxial: false,
            tf_sampled: true,
            kernel_sampled: true,
            recommended: Engine::Asm,
        };
        let b = regime_advisory(Engine::Fraunhofer, &regime).expect("advisory");
        assert_eq!(b.severity, Severity::Advisory);
        assert!(b.headline.contains("far_field=false"));
        // ASM is always fine — no advisory.
        assert!(regime_advisory(Engine::Asm, &regime).is_none());
        // Fresnel non-paraxial warns.
        let b2 = regime_advisory(Engine::FresnelTransfer, &regime).expect("advisory");
        assert!(b2.headline.contains("paraxial=false"));
    }

    #[test]
    fn rejected_carries_display_and_remedies() {
        let b = rejected("AliasingRisk: ...".into(), standard_remedies());
        assert_eq!(b.severity, Severity::Rejected);
        assert_eq!(b.remedies.len(), 4);
    }
}
