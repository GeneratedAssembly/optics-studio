//! Beamline schematic strip (spec §3.1 "Beamline preview"): an SVG of the
//! element stack — apertures, surfaces, gaps, media — read straight from
//! the [`StripModel`].

use dioxus::prelude::*;

use super::studio::Shell;
use crate::shell::prescription::StripNode;

/// Layout constants for the strip viewBox.
const VB_W: f64 = 720.0;
const VB_H: f64 = 150.0;
const PAD_L: f64 = 24.0;
const PAD_R: f64 = 24.0;
const AXIS_Y: f64 = 78.0;
const SEMI_PX_PER_MM: f64 = 2.2;

#[component]
pub fn StripSvg() -> Element {
    let studio = use_context::<Shell>();
    let strip = studio.strip();
    if strip.nodes.is_empty() {
        return rsx! {
            svg { class: "strip-svg", view_box: "0 0 720 150",
                text { x: "12", y: "80", class: "strip-empty", "beamline strip — import a .zmx" }
            }
        };
    }

    let track = strip.track_m.max(1e-9) * 1e3; // mm
    let x_right = VB_W - PAD_R;
    let meta_y = VB_H - 6.0;
    let usable = VB_W - PAD_L - PAD_R;
    let scale = usable / track; // px per mm
    let max_semi = strip.max_semi_mm.max(0.1);
    let semi_clamped = max_semi.min(AXIS_Y / SEMI_PX_PER_MM);

    let mut z_cursor_mm = 0.0f64;
    let mut glyphs = Vec::new();
    for node in &strip.nodes {
        match node {
            StripNode::Surface {
                index,
                semi_mm,
                front_n,
                back_n,
                curvature_per_mm,
                stop,
                comment,
                ..
            } => {
                let x = PAD_L + z_cursor_mm * scale;
                let half = (semi_mm.min(semi_clamped) * SEMI_PX_PER_MM).max(2.0);
                let bulge = (*curvature_per_mm * 400.0).clamp(-14.0, 14.0);
                glyphs.push(Glyph::Surface {
                    x,
                    half,
                    bulge,
                    stop: *stop,
                    refracts: (back_n - front_n).abs() > 1e-9,
                    label: format!("S{index}"),
                    comment: comment.clone(),
                });
            }
            StripNode::Gap { z_m, n } => {
                let x0 = PAD_L + z_cursor_mm * scale;
                z_cursor_mm += z_m * 1e3;
                let x1 = PAD_L + z_cursor_mm * scale;
                glyphs.push(Glyph::Gap {
                    x0,
                    x1,
                    glass: *n > 1.0001,
                    n: *n,
                    mm: z_m * 1e3,
                });
            }
        }
    }

    rsx! {
        svg { class: "strip-svg", view_box: "0 0 720 150",
            line { class: "axis", x1: "{PAD_L}", y1: "{AXIS_Y}", x2: "{x_right}", y2: "{AXIS_Y}" }
            for (i, g) in glyphs.iter().enumerate() {
                {render_glyph(g, i)}
            }
            text { class: "strip-meta", x: "{PAD_L}", y: "{meta_y}",
                "track {track:.2} mm · max semi {max_semi:.2} mm"
            }
        }
    }
}

enum Glyph {
    Surface {
        x: f64,
        half: f64,
        bulge: f64,
        stop: bool,
        refracts: bool,
        label: String,
        comment: String,
    },
    Gap {
        x0: f64,
        x1: f64,
        glass: bool,
        n: f64,
        mm: f64,
    },
}

fn render_glyph(g: &Glyph, key: usize) -> Element {
    match g {
        Glyph::Gap {
            x0,
            x1,
            glass,
            n,
            mm,
        } => {
            let w = (x1 - x0).max(0.0);
            let gap_top = AXIS_Y - 26.0;
            let tick_y = AXIS_Y + 40.0;
            let n_y = AXIS_Y - 30.0;
            let mm_label = format!("{mm:.1}");
            let n_label = format!("n {n:.3}");
            let mid = (x0 + x1) / 2.0;
            rsx! {
                g { key: "g{key}",
                    rect {
                        class: if *glass { "gap gap-glass" } else { "gap" },
                        x: "{x0}",
                        y: "{gap_top}",
                        width: "{w}",
                        height: "52.0",
                    }
                    if *mm > 0.1 {
                        text { class: "strip-tick", x: "{mid}", y: "{tick_y}",
                            "{mm_label}"
                        }
                    }
                    if *glass {
                        text { class: "strip-n", x: "{mid}", y: "{n_y}",
                            "{n_label}"
                        }
                    }
                }
            }
        }
        Glyph::Surface {
            x,
            half,
            bulge,
            stop,
            refracts,
            label,
            comment,
        } => {
            let top = AXIS_Y - half;
            let bot = AXIS_Y + half;
            let path = format!(
                "M {x:.1} {top:.1} Q {:.1} {AXIS_Y:.1} {x:.1} {bot:.1}",
                x + bulge
            );
            let label_text = label.clone();
            let comment_text = comment.clone();
            let label_y = AXIS_Y - half - 6.0;
            rsx! {
                g { key: "s{key}",
                    if !comment_text.is_empty() {
                        title { "{comment_text}" }
                    }
                    path {
                        class: if *stop { "surf surf-stop" } else if *refracts { "surf" } else { "surf surf-flat" },
                        d: "{path}",
                    }
                    text { class: "strip-label", x: "{x}", y: "{label_y}", "{label_text}" }
                }
            }
        }
    }
}
