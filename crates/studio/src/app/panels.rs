//! Metric tab column: EE / MTF / Strehl / Spot / Wavefront panels plus the
//! shared scrubber (spec §3.3).

use std::rc::Rc;

use dioxus::prelude::*;

use crate::viewmodel::charts::{Chart, PlotBox};
use crate::viewmodel::demo::StudioModel;
use crate::viewmodel::metrics::{DEFAULT_ZERNIKE_N_MAX, ee_max_radius, mtf_max_freq};
use crate::viewmodel::state::{PanelTab, ViewportState};

/// Chart pixel box for the metric panels.
const CHART_BOX: PlotBox = PlotBox::new(380.0, 150.0);

/// Right column: tabs + active panel.
#[component]
pub fn MetricColumn(mut view: Signal<ViewportState>) -> Element {
    let v = view.read().clone();
    rsx! {
        div { class: "metric-col", style: "display:flex; flex-direction:column; gap:8px; min-width:320px; flex:1;",
            div { class: "panel-tabs", style: "display:flex; gap:4px;",
                for tab in PanelTab::ALL {
                    button {
                        style: if tab == v.tab {
                            "background:#3a3a55;color:#fff;border:1px solid #5a5a80;border-radius:4px 4px 0 0;padding:4px 10px;font-size:13px;"
                        } else {
                            "background:#1c1c26;color:#9aa;border:1px solid #2c2c3c;border-radius:4px 4px 0 0;padding:4px 10px;font-size:13px;"
                        },
                        onclick: move |_| view.write().tab = tab,
                        "{tab.label()}"
                    }
                }
            }
            div { style: "border:1px solid #2c2c3c; border-radius:0 0 6px 6px; padding:10px; background:#181822;",
                match v.tab {
                    PanelTab::EncircledEnergy => rsx! { EePanel { view } },
                    PanelTab::Mtf => rsx! { MtfPanel { view } },
                    PanelTab::Strehl => rsx! { StrehlPanel { view } },
                    PanelTab::Spot => rsx! { SpotPanel { view } },
                    PanelTab::Wavefront => rsx! { WavefrontPanel { view } },
                }
            }
        }
    }
}

/// Shared scrubber: one normalized position drives both the EE radius and
/// the MTF frequency readouts (spec §3.3).
#[component]
pub fn Scrubber(mut view: Signal<ViewportState>) -> Element {
    let v = view.read().clone();
    rsx! {
        div { style: "display:flex; align-items:center; gap:8px; font-size:12px; margin-top:8px;",
            span { "scrub" }
            input {
                r#type: "range",
                min: "0",
                max: "1000",
                value: "{(v.scrub * 1000.0) as i64}",
                style: "flex:1;",
                oninput: move |evt| {
                    if let Ok(x) = evt.value().parse::<i64>() {
                        view.write().set_scrub(x as f64 / 1000.0);
                    }
                },
            }
            span { style: "color:#888;", "{(v.scrub * 100.0) as i64}%" }
        }
    }
}

#[component]
pub fn EePanel(mut view: Signal<ViewportState>) -> Element {
    let model = use_context::<Rc<StudioModel>>();
    let v = view.read().clone();
    let frame = model.active(v.active_frame);
    let (radii, fractions) = frame.metrics.ee.curve();
    let chart = Chart::encircled_energy(radii, fractions, CHART_BOX);
    let r_max = ee_max_radius(&frame.metrics.ee);
    let readout = frame.metrics.ee_readout(r_max * v.scrub);
    rsx! {
        div {
            PanelTitle { title: "Encircled energy EE(r)".to_string() }
            ChartView { chart, stroke: "#7ee787".to_string(), x_label: "radius (µm)".to_string(), y_label: "fraction".to_string() }
            div { style: "font-size:13px; line-height:1.6;",
                div { "EE at scrub radius ({readout.scrub_radius * 1e6:.1} µm): " b { "{readout.value_at_scrub:.4}" } }
                div { "r(84%) = " b { "{readout.r84 * 1e6:.2} µm" } " · r(50%) = " b { "{readout.r50 * 1e6:.2} µm" } }
                div { style: "color:#888; font-size:11px;", "Airy anchor: r(84%) ≈ first dark ring 1.22 λz/2a" }
            }
            Scrubber { view }
        }
    }
}

#[component]
pub fn MtfPanel(mut view: Signal<ViewportState>) -> Element {
    let model = use_context::<Rc<StudioModel>>();
    let v = view.read().clone();
    let frame = model.active(v.active_frame);
    let chart = Chart::mtf(
        &frame.metrics.mtf.freqs,
        &frame.metrics.mtf.values,
        CHART_BOX,
        v.mtf_log_x,
    );
    let f_max = mtf_max_freq(&frame.metrics.mtf);
    let f = f_max * v.scrub;
    let value = interp_pair(&frame.metrics.mtf.freqs, &frame.metrics.mtf.values, f);
    rsx! {
        div {
            PanelTitle { title: "MTF along +f_x (normalized at DC)".to_string() }
            label { style: "font-size:12px; color:#9aa;",
                input {
                    r#type: "checkbox",
                    checked: v.mtf_log_x,
                    onchange: move |_| view.with_mut(|s| s.mtf_log_x = !s.mtf_log_x),
                }
                " log-x"
            }
            ChartView { chart, stroke: "#6ab0ff".to_string(), x_label: "spatial freq (cyc/mm)".to_string(), y_label: "MTF".to_string() }
            div { style: "font-size:13px;",
                "MTF at " b { "{f * 1e-3:.2} cyc/mm" } " = " b { "{value:.4}" }
                div { style: "color:#888; font-size:11px;", "Nyquist {f_max * 1e-3:.2} cyc/mm (1 / 2Δx)" }
            }
            Scrubber { view }
        }
    }
}

#[component]
pub fn StrehlPanel(view: Signal<ViewportState>) -> Element {
    let model = use_context::<Rc<StudioModel>>();
    let v = view.read().clone();
    let frame = model.active(v.active_frame);
    match frame.metrics.strehl {
        Some(s) => rsx! {
            div {
                PanelTitle { title: "Strehl ratio (peak ratio at equal power)".to_string() }
                div { style: "font-size:34px; font-weight:700; color:{strehl_color(s)};", "{s:.4}" }
                div { style: "font-size:12px; color:#888; margin-top:6px;",
                    "reference is synthetic, same grid/engine (spec §3.3 — gap G-6: no core reference generator)"
                }
                if let Some(m) = frame.metrics.marechal_strehl {
                    div { style: "font-size:13px; margin-top:8px;",
                        "Maréchal estimate from Zernike residual: " b { "{m:.4}" }
                    }
                }
            }
        },
        None => rsx! {
            div {
                PanelTitle { title: "Strehl ratio".to_string() }
                div { style: "font-size:13px; color:#888;",
                    "This frame is the reference itself — pick a measured band tab to see the ratio."
                }
            }
        },
    }
}

#[component]
pub fn SpotPanel(view: Signal<ViewportState>) -> Element {
    let model = use_context::<Rc<StudioModel>>();
    let v = view.read().clone();
    let frame = model.active(v.active_frame);
    let (cx, cy) = frame.metrics.centroid;
    let (px, py) = frame.psf.peak_position();
    rsx! {
        div { style: "font-size:14px; line-height:1.9;",
            PanelTitle { title: "Spot".to_string() }
            div { "RMS spot radius: " b { "{frame.metrics.spot_rms * 1e6:.2} µm" } }
            div { "Centroid: " b { "({cx * 1e6:.2}, {cy * 1e6:.2}) µm" } }
            div { "Peak bin position: " b { "({px * 1e6:.2}, {py * 1e6:.2}) µm" } }
            div { "Total power: " b { "{frame.metrics.total_power:.4e}" } }
            div { style: "color:#888; font-size:11px;",
                "analysis::spot_rms_radius · Psf::centroid · Psf::peak_position"
            }
        }
    }
}

#[component]
pub fn WavefrontPanel(view: Signal<ViewportState>) -> Element {
    let model = use_context::<Rc<StudioModel>>();
    let v = view.read().clone();
    let frame = model.active(v.active_frame);
    let Some(wf) = &frame.metrics.wavefront else {
        return rsx! {
            div {
                PanelTitle { title: "Wavefront (Zernike)".to_string() }
                div { style: "font-size:13px; color:#888;", "No pupil phase supplied for this frame." }
            }
        };
    };
    let rms_waves = wf.residual_rms_waves();
    rsx! {
        div {
            PanelTitle { title: format!("Wavefront · Zernike n ≤ {DEFAULT_ZERNIKE_N_MAX}") }
            div { style: "font-size:13px; margin-bottom:6px;",
                "residual RMS: " b { "{rms_waves:.4} waves" }
                if let Some(m) = frame.metrics.marechal_strehl {
                    "  · Maréchal Strehl " b { "{m:.4}" }
                }
            }
            div { style: "display:grid; grid-template-columns:repeat(auto-fill, minmax(96px, 1fr)); gap:4px;",
                for (i , (n , m)) in StudioModel::zernike_labels(DEFAULT_ZERNIKE_N_MAX).into_iter().enumerate() {
                    CoefBar {
                        label: format!("Z{n}m{m}"),
                        value: wf.coefficients.get(i).copied().unwrap_or(0.0),
                        scale: wf.coefficients.iter().cloned().fold(0.0f64, f64::max),
                    }
                }
            }
            div { style: "color:#888; font-size:11px; margin-top:6px;",
                "coefficients in radians (phase ≈ Σ cₖ Zₖ) · analysis::zernike_fit"
            }
        }
    }
}

/// One Zernike coefficient bar (diverging: positive right, negative left).
#[component]
pub fn CoefBar(label: String, value: f64, scale: f64) -> Element {
    let frac = if scale > 0.0 {
        (value / scale).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let pct = (frac.abs() * 50.0) as i32;
    let color = if frac >= 0.0 { "#7ee787" } else { "#ff8a80" };
    let bar_style = if frac >= 0.0 {
        format!("position:absolute; left:50%; width:{pct}%; background:{color}; height:100%;")
    } else {
        format!("position:absolute; right:50%; width:{pct}%; background:{color}; height:100%;")
    };
    rsx! {
        div { style: "font-size:10px;",
            div { "{label}: {value:.2} rad" }
            div { style: "position:relative; height:8px; background:#22222e; border-radius:2px;",
                div { style: bar_style }
                div { style: "position:absolute; left:50%; top:0; width:1px; height:100%; background:#555;" }
            }
        }
    }
}

/// A chart with axis labels.
#[component]
pub fn ChartView(chart: Chart, stroke: String, x_label: String, y_label: String) -> Element {
    rsx! {
        div { style: "margin-bottom:6px;",
            div { style: "font-size:11px; color:#888;", "{y_label}" }
            svg {
                view_box: format!("0 0 {} {}", chart.box_.width, chart.box_.height),
                style: "width:100%; max-width:400px; height:150px; background:#101018; border:1px solid #2a2a35; border-radius:3px;",
                path { d: chart.path.clone(), fill: "none", stroke: stroke.clone(), stroke_width: "1.5" }
            }
            div { style: "font-size:11px; color:#888; text-align:right;", "{x_label}" }
        }
    }
}

#[component]
pub fn PanelTitle(title: String) -> Element {
    rsx! { div { style: "font-weight:600; font-size:14px; margin-bottom:6px;", "{title}" } }
}

fn strehl_color(s: f64) -> &'static str {
    if s >= 0.8 {
        "#7ee787"
    } else if s >= 0.5 {
        "#e3b341"
    } else {
        "#ff8a80"
    }
}

/// Linear interpolation for display readouts (matches `value_at` semantics,
/// spec §4 "hover readouts interpolate linearly only for display").
fn interp_pair(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    for i in 1..xs.len() {
        if xs[i] >= x {
            let t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
            return ys[i - 1] + t * (ys[i] - ys[i - 1]);
        }
    }
    ys[ys.len() - 1]
}
