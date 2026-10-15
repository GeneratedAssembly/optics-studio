//! Center viewport: live preview image, readouts, sampling banner, and the
//! Beamline schematic strip (spec §3.1–§3.2, §5).

use dioxus::prelude::*;

use super::strip::StripSvg;
use super::studio::{PreviewSource, Shell};

#[component]
pub fn Viewport() -> Element {
    let studio = use_context::<Shell>();
    let image = studio.image_url();
    let psf = studio.psf();
    let stale = studio.stale();
    let banner = studio.banner();
    let advisory = studio.advisory();
    let log = studio.log_scale();
    let down = studio.downsampled_by();
    let from = studio.psf_from();

    let readouts = psf.as_ref().map(|p| {
        let (nx, ny) = p.dims();
        let (dx, dy) = p.pitch();
        let (px, py) = p.peak_position();
        let (cx, cy) = p.centroid();
        let (dxu, dyu) = (dx * 1e6, dy * 1e6);
        let power = p.total_power();
        let (pxu, pyu) = (px * 1e6, py * 1e6);
        let (cxu, cyu) = (cx * 1e6, cy * 1e6);
        format!(
            "{nx}×{ny} · pitch ({dxu:.2}, {dyu:.2}) µm · power {power:.4} · peak ({pxu:.1}, {pyu:.1}) µm · centroid ({cxu:.1}, {cyu:.1}) µm"
        )
    });

    rsx! {
        div { class: "view-wrap",
            div { class: "row",
                h2 { "Live preview" }
                span { class: "small",
                    match from {
                        PreviewSource::Forward => "prescription → Beamline::psf",
                        PreviewSource::Carrier => "Carrier::build → Design::beamline → psf",
                    }
                }
                span { style: "margin-left:auto" }
                label { class: "small",
                    input {
                        r#type: "checkbox",
                        checked: log,
                        onchange: move |evt| {
                            let mut s = studio;
                            s.log_scale.set(evt.checked());
                            s.run_preview();
                        },
                    }
                    " log"
                }
                if stale {
                    span { class: "badge stale", "stale — last good frame" }
                }
            }
            StripSvg {}
            if let Some(url) = image {
                img { class: "psf-img", src: "{url}" }
            } else {
                div { class: "small", "no preview yet — load a sample or import a .zmx" }
            }
            if let Some(text) = readouts {
                div { class: "readouts", "{text}" }
            }
            if let Some(k) = down {
                div { class: "small", "display downsampled ×{k} (analysis runs on the full grid)" }
            }
            if let Some(b) = banner {
                div { class: "banner {b.severity.css_class()}",
                    "{b.headline}"
                    if let Some(d) = &b.detail {
                        div { class: "small", "{d}" }
                    }
                    for r in b.remedies.iter() {
                        div { class: "small", "→ {r}" }
                    }
                }
            }
            if let Some(a) = advisory {
                div { class: "banner banner-advisory", "{a}" }
            }
        }
    }
}

#[component]
pub fn Toasts() -> Element {
    let studio = use_context::<Shell>();
    let toasts = studio.toasts();
    rsx! {
        div { class: "toasts",
            for t in toasts.into_iter() {
                div { class: "toast", key: "{t.id}",
                    span { "{t.text.clone()}" }
                    button {
                        onclick: {
                            let id = t.id;
                            move |_| {
                                let mut s = studio;
                                s.dismiss_toast(id);
                            }
                        },
                        "×"
                    }
                }
            }
        }
    }
}
