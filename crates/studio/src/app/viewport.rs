//! Intensity column: multi-band tabs, the PSF canvas, display controls,
//! cross-sections, and the cursor readout (spec §3.3, §4).

use std::rc::Rc;

use dioxus::prelude::*;

use crate::app::canvas;
use crate::viewmodel::crosscut;
use crate::viewmodel::demo::StudioModel;
use crate::viewmodel::intensity::{RenderOptions, autoscale, render_intensity};
use crate::viewmodel::state::ViewportState;

/// CSS display edge of the canvas (backing store stays at grid resolution;
/// the box is upscaled with `pixelated` rendering).
const DISPLAY_PX: u32 = 440;

/// Left column of the viewport.
#[component]
pub fn IntensityColumn(mut view: Signal<ViewportState>) -> Element {
    let model = use_context::<Rc<StudioModel>>();
    let v = view.read().clone();
    let frame = model.active(v.active_frame);
    let (nx, ny) = frame.psf.dims();

    // Paint the canvas after every state change (spec §3.5: keep the last
    // good image; paint failure degrades to numbers-only).
    let paint_model = Rc::clone(&model);
    use_effect(move || {
        let v = view.read().clone();
        let frame = paint_model.active(v.active_frame);
        let opts = RenderOptions {
            colormap: v.colormap,
            log: v.log,
            lock: v.lock,
            ..Default::default()
        };
        if let Ok(img) = render_intensity(frame.psf.samples(), nx, ny, &opts) {
            canvas::paint("psf-canvas", &img);
        }
    });

    let (row, col) = match v.cursor {
        Some((ix, iy)) => (
            crosscut::row_cut(&frame.psf, iy),
            crosscut::column_cut(&frame.psf, ix),
        ),
        None => crosscut::through_peak(&frame.psf),
    };

    rsx! {
        div { class: "intensity-col", style: "display:flex; flex-direction:column; gap:8px;",
            div { class: "band-tabs", style: "display:flex; gap:4px; flex-wrap:wrap;",
                for (i , f) in model.frames.iter().enumerate() {
                    button {
                        style: tab_style(i == v.active_frame),
                        onclick: move |_| view.write().active_frame = i,
                        "{f.label.clone()}"
                    }
                }
            }
            canvas {
                id: "psf-canvas",
                width: "{nx}",
                height: "{ny}",
                style: "width:{DISPLAY_PX}px; height:{DISPLAY_PX}px; image-rendering:pixelated; border:1px solid #333; background:#000; max-width:100%;",
                onmousemove: move |evt| {
                    let p = evt.element_coordinates();
                    let ix = (p.x / DISPLAY_PX as f64 * nx as f64).clamp(0.0, nx as f64 - 1.0) as usize;
                    let iy = (p.y / DISPLAY_PX as f64 * ny as f64).clamp(0.0, ny as f64 - 1.0) as usize;
                    view.write().cursor = Some((ix, iy));
                },
                onmouseleave: move |_| view.write().cursor = None,
            }
            ControlsRow { view }
            CursorReadout { view }
            CrosscutsView { row: row.clone(), col: col.clone() }
        }
    }
}

/// Colormap / log / scale-lock controls.
#[component]
pub fn ControlsRow(mut view: Signal<ViewportState>) -> Element {
    let model = use_context::<Rc<StudioModel>>();
    let v = view.read().clone();
    let frame = model.active(v.active_frame);
    let (nx, ny) = frame.psf.dims();
    let current = {
        let (lo, hi) = autoscale(frame.psf.samples());
        crate::viewmodel::intensity::ScaleLock { lo, hi }
    };
    rsx! {
        div { class: "controls", style: "display:flex; gap:12px; align-items:center; flex-wrap:wrap; font-size:13px;",
            label {
                "colormap "
                select {
                    onchange: move |evt| {
                        let chosen = match evt.value().as_str() {
                            "inferno" => crate::viewmodel::colormap::Colormap::Inferno,
                            "rdbu" => crate::viewmodel::colormap::Colormap::RdBu,
                            _ => crate::viewmodel::colormap::Colormap::Viridis,
                        };
                        view.with_mut(|s| s.colormap = chosen);
                    },
                    option { value: "viridis", selected: v.colormap == crate::viewmodel::colormap::Colormap::Viridis, "viridis" }
                    option { value: "inferno", selected: v.colormap == crate::viewmodel::colormap::Colormap::Inferno, "inferno" }
                    option { value: "rdbu", selected: v.colormap == crate::viewmodel::colormap::Colormap::RdBu, "rdbu (diverging)" }
                }
            }
            label {
                input {
                    r#type: "checkbox",
                    checked: v.log,
                    onchange: move |_| view.with_mut(|s| s.log = !s.log),
                }
                " log I"
            }
            button {
                style: if v.lock.is_some() { lock_style() } else { "font-size:12px".to_string() },
                onclick: move |_| {
                    let cur = current;
                    view.write().toggle_lock(Some(cur));
                },
                "{lock_label(v.lock.is_some())}"
            }
            span { style: "color:#888;", "power {fmt_power(frame.metrics.total_power)} · frame {nx}×{ny}" }
        }
    }
}

/// Cursor sample readout: coordinates (µm), value, and log-scaled value
/// (spec §3.3 "Interaction").
#[component]
pub fn CursorReadout(view: Signal<ViewportState>) -> Element {
    let model = use_context::<Rc<StudioModel>>();
    let v = view.read();
    let Some((ix, iy)) = v.cursor else {
        return rsx! {
            div { style: "font-size:12px; color:#888;", "hover the image for cross-sections + readout" }
        };
    };
    let frame = model.active(v.active_frame);
    let (x, y) = frame.psf.coord(ix, iy);
    let value = frame.psf.samples()[iy * frame.psf.dims().0 + ix];
    let peak = frame.psf.samples().iter().cloned().fold(0.0f64, f64::max);
    let logv = if peak > 0.0 && value > 0.0 {
        format!("{:.1} dB", 10.0 * (value / peak).log10())
    } else {
        "-inf dB".into()
    };
    rsx! {
        div { style: "font-size:12px; color:#bbb;",
            "bin ({ix},{iy}) · x={x * 1e6:.1} µm · y={y * 1e6:.1} µm · I={value:.3e} ({logv})"
        }
    }
}

/// Horizontal + vertical cuts through the peak or cursor (spec §3.3).
#[component]
pub fn CrosscutsView(row: crosscut::Crosscut, col: crosscut::Crosscut) -> Element {
    rsx! {
        div { class: "crosscuts", style: "display:flex; gap:12px; flex-wrap:wrap;",
            MiniChart { title: format!("H-cut {}", row.axis_label), xs: row.xs.clone(), ys: row.values.clone() }
            MiniChart { title: format!("V-cut {}", col.axis_label), xs: col.xs.clone(), ys: col.values.clone() }
        }
    }
}

/// A small SVG polyline chart (cross-sections, no axis chrome).
#[component]
pub fn MiniChart(title: String, xs: Vec<f64>, ys: Vec<f64>) -> Element {
    let box_ = crate::viewmodel::charts::PlotBox::new(200.0, 80.0);
    let x_axis = crate::viewmodel::charts::Axis::linear(
        *xs.first().unwrap_or(&0.0),
        *xs.last().unwrap_or(&1.0),
    );
    let y_axis = crate::viewmodel::charts::Axis::linear(
        ys.iter().cloned().fold(f64::INFINITY, f64::min),
        ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let chart = crate::viewmodel::charts::Chart::from_points(&xs, &ys, x_axis, y_axis, box_);
    rsx! {
        div { style: "flex:1; min-width:180px;",
            div { style: "font-size:11px; color:#888;", "{title}" }
            svg {
                view_box: "0 0 200 80",
                style: "width:100%; height:80px; background:#101018; border:1px solid #2a2a35; border-radius:3px;",
                path { d: chart.path.clone(), fill: "none", stroke: "#6ab0ff", stroke_width: "1.2" }
            }
        }
    }
}

fn tab_style(active: bool) -> String {
    if active {
        "background:#3a3a55; color:#fff; border:1px solid #5a5a80; border-radius:4px; padding:3px 8px; font-size:12px;".into()
    } else {
        "background:#20202c; color:#aaa; border:1px solid #33334a; border-radius:4px; padding:3px 8px; font-size:12px;".into()
    }
}

fn lock_style() -> String {
    "font-size:12px; background:#33502f; color:#cfe8c8; border:1px solid #4a7044; border-radius:4px; padding:2px 8px;".into()
}

fn lock_label(locked: bool) -> &'static str {
    if locked { "unlock scale" } else { "lock scale" }
}

/// Power readout with engineering units.
pub fn fmt_power(p: f64) -> String {
    if p == 0.0 {
        "0".into()
    } else if p >= 1e-3 {
        format!("{:.4}", p)
    } else {
        format!("{:.3e}", p)
    }
}
