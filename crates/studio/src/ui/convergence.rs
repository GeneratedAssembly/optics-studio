//! The convergence plot (spec §3.4 "Convergence plot" row, §4 "Line
//! charts"): the merit-after-each-accepted-evaluation trace as an SVG
//! polyline, log-y by default (spec §4: log scale applies to the
//! convergence plot). Negative or zero merits (the `neg_*` built-ins)
//! cannot use log-y; the plot falls back to linear and says so.

use dioxus::prelude::*;

const W: f64 = 640.0;
const H: f64 = 240.0;
const PAD: f64 = 10.0;

/// Line chart of `history` (merit vs accepted evaluation index).
#[component]
pub fn ConvergencePlot(history: Vec<f64>, log_y: bool) -> Element {
    let n = history.len();
    if n < 2 {
        let message = if n == 0 {
            "run to see the convergence trace"
        } else {
            "waiting for the second accepted evaluation…"
        };
        return rsx! {
            div { class: "plot empty", "{message}" }
        };
    }
    let min = history.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let log_ok = log_y && min > 0.0 && max.is_finite();
    let mapped: Vec<f64> = if log_ok {
        history.iter().map(|v| v.log10()).collect()
    } else {
        history.clone()
    };
    let lo = mapped.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = mapped.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-12);
    let points: String = mapped
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let x = PAD + (i as f64 / (n - 1) as f64) * (W - 2.0 * PAD);
            let y = H - PAD - ((v - lo) / span) * (H - 2.0 * PAD);
            format!("{x:.1},{y:.1}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let scale_note = if log_ok {
        "log-y".to_string()
    } else if log_y {
        "log-y unavailable (merit ≤ 0) — linear".to_string()
    } else {
        "linear".to_string()
    };
    rsx! {
        div { class: "plot",
            svg {
                view_box: "0 0 {W} {H}",
                width: "100%",
                height: "auto",
                polyline {
                    points: "{points}",
                    fill: "none",
                    stroke: "#5ac8fa",
                    stroke_width: "1.5",
                }
            }
            div { class: "plot-meta",
                "n={n}  merit ∈ [{min:.4e}, {max:.4e}]  {scale_note}"
            }
        }
    }
}
