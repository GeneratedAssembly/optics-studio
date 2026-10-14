//! Sampling banner component (spec §5.2).

use dioxus::prelude::*;

use crate::viewmodel::banner::{Banner, Severity};

/// The run-diagnostics banner: severity-colored line, full report text for
/// red/advisory, and the remedy list.
#[component]
pub fn BannerView(banner: Banner) -> Element {
    let color = match banner.severity {
        Severity::Ok => "#2e7d32",
        Severity::Mitigated => "#b8860b",
        Severity::Rejected => "#b71c1c",
        Severity::Advisory => "#1565c0",
    };
    let box_style = format!(
        "border-left: 6px solid {color}; padding: 6px 10px; margin: 6px 0; background: #1d1d27; border-radius: 4px;"
    );
    rsx! {
        div { class: banner.severity.css_class(), style: "{box_style}",
            div { style: "font-weight: 600;", "{headline_label(banner.severity)}" }
            div { style: "font-size: 13px;", "{banner.headline.clone()}" }
            if let Some(detail) = banner.detail.clone() {
                details {
                    summary { style: "cursor: pointer; font-size: 12px;", "full report" }
                    pre { style: "white-space: pre-wrap; font-size: 11px;", "{detail}" }
                }
            }
            for remedy in banner.remedies.clone() {
                div { style: "font-size: 12px; color: #ffb74d;", "· {remedy}" }
            }
        }
    }
}

fn headline_label(severity: Severity) -> &'static str {
    match severity {
        Severity::Ok => "Sampling: OK",
        Severity::Mitigated => "Sampling: mitigated (engine densified)",
        Severity::Rejected => "Sampling: rejected",
        Severity::Advisory => "Regime advisory",
    }
}
