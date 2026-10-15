//! 2D Layout renderer (GEN-57): an SVG cross-section of the element
//! stack in the spirit of Zemax's 2D Layout window — true conic surface
//! profiles, shaded glass, stop and image-plane markers, the paraxial
//! focus reference, and a scale bar. Mounted live inside the Lens Data
//! Editor tab, so the system redraws as elements are added.
//!
//! No traced rays here: the colored fan in the reference is a geometric
//! ray trace, out of scope per agents.md §1 / spec §6.

use dioxus::prelude::*;

use super::studio::Shell;

const VB_W: f64 = 720.0;
const VB_H: f64 = 280.0;
const PAD_L: f64 = 30.0;
const PAD_R: f64 = 30.0;
const AXIS_Y: f64 = 150.0;
/// Half-height budget for the cross-section (px), top and bottom.
const EDGE: f64 = 95.0;

#[component]
pub fn Layout2dSvg() -> Element {
    let studio = use_context::<Shell>();
    let model = studio.layout();
    if model.surfaces.is_empty() {
        return rsx! {
            svg { class: "layout2d-svg", view_box: "0 0 720 280",
                text { x: "12", y: "150", class: "strip-empty",
                    "2D layout — press New System or import a .zmx"
                }
            }
        };
    }

    let track = model.track_mm.max(1.0);
    let scale_z = (VB_W - PAD_L - PAD_R) / track;
    let semi = model.max_semi_mm.max(0.5);
    let scale_y = EDGE / semi;
    let zx = move |z: f64| PAD_L + z * scale_z;
    let yx = move |y: f64| AXIS_Y - y * scale_y;

    let mut parts: Vec<Element> = Vec::new();
    // Glass bodies: gap i sits after surface i, bounded by that surface's
    // semi-diameter (unbounded surfaces draw at the field edge).
    for (i, g) in model.gaps.iter().enumerate() {
        if g.n <= 1.0001 {
            continue;
        }
        let h = model
            .surfaces
            .get(i)
            .map(|s| if s.semi_mm > 0.0 { s.semi_mm } else { semi })
            .unwrap_or(semi);
        let x0 = zx(g.z0_mm);
        let x1 = zx(g.z1_mm).max(x0 + 0.5);
        let ytop = yx(h);
        parts.push(rsx! {
            rect {
                key: "glass{i}",
                class: "l2d-glass",
                x: "{x0}",
                y: "{ytop}",
                width: "{x1 - x0}",
                height: "{(h * scale_y * 2.0)}",
            }
        });
    }
    // Surface curves: sag profile from -semi to +semi.
    for (i, s) in model.surfaces.iter().enumerate() {
        let h = if s.semi_mm > 0.0 { s.semi_mm } else { semi };
        let mut d = String::new();
        for (k, (&y, &sag)) in s.y_mm.iter().zip(s.sag_mm.iter()).enumerate().rev() {
            let (px, py) = (zx(s.z_mm + sag), yx(y));
            if k == s.y_mm.len() - 1 {
                d.push_str(&format!("M {px:.2} {py:.2}"));
            } else {
                d.push_str(&format!(" L {px:.2} {py:.2}"));
            }
        }
        for (&y, &sag) in s.y_mm.iter().zip(s.sag_mm.iter()).skip(1) {
            let (px, py) = (zx(s.z_mm + sag), yx(-y));
            d.push_str(&format!(" L {px:.2} {py:.2}"));
        }
        let cls = if s.stop {
            "l2d-surf l2d-stop"
        } else {
            "l2d-surf"
        };
        let lx = zx(s.z_mm);
        let ly = yx(h) - 6.0;
        let label = format!("S{}", s.index);
        let comment = s.comment.clone();
        parts.push(rsx! {
            g { key: "surf{i}",
                if !comment.is_empty() {
                    title { "{comment}" }
                }
                path { class: "{cls}", d: "{d}" }
                text { class: "l2d-label", x: "{lx}", y: "{ly}", "{label}" }
            }
        });
    }
    // Paraxial focus reference: dashed cone from the first surface edge to
    // the paraxial image of an on-axis object at infinity.
    if let Some(bfl) = model.bfl_mm {
        let focus_z = model
            .surfaces
            .last()
            .map(|s| s.z_mm + bfl)
            .unwrap_or(model.track_mm);
        let (x0, ytop, ybot) = (zx(0.0), yx(semi), yx(-semi));
        let (xf, y0) = (zx(focus_z), AXIS_Y);
        parts.push(rsx! {
            g {
                line { class: "l2d-focus", x1: "{x0}", y1: "{ytop}", x2: "{xf}", y2: "{y0}" }
                line { class: "l2d-focus", x1: "{x0}", y1: "{ybot}", x2: "{xf}", y2: "{y0}" }
            }
        });
    }
    // Image plane.
    let xi = zx(model.track_mm);
    parts.push(
        rsx! {
            g {
                line { class: "l2d-image", x1: "{xi}", y1: "{yx(semi * 1.15)}", x2: "{xi}", y2: "{yx(-semi * 1.15)}" }
                text { class: "l2d-label", x: "{xi}", y: "{yx(semi * 1.15) - 5.0}", "IMAGE" }
            }
        },
    );
    // Scale bar: a 1/2/5·10^k mm span of at least ~60 px.
    let target_mm = 100.0 / scale_z;
    let pow = 10f64.powf(target_mm.max(1e-9).log10().floor());
    let bar_mm = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * pow)
        .find(|v| *v * scale_z >= 60.0)
        .unwrap_or(10.0 * pow);
    let bar_px = bar_mm * scale_z;
    let (bx0, by) = (PAD_L, VB_H - 18.0);
    let axis_right = VB_W - PAD_R;
    let bar_label = if bar_mm >= 1.0 {
        format!("{bar_mm:.0} mm")
    } else {
        format!("{bar_mm:.2} mm")
    };

    rsx! {
        svg { class: "layout2d-svg", view_box: "0 0 720 280",
            line { class: "axis", x1: "{PAD_L}", y1: "{AXIS_Y}", x2: "{axis_right}", y2: "{AXIS_Y}" }
            for p in parts {
                {p}
            }
            line { class: "l2d-bar", x1: "{bx0}", y1: "{by}", x2: "{bx0 + bar_px}", y2: "{by}" }
            line { class: "l2d-bar", x1: "{bx0}", y1: "{by - 4.0}", x2: "{bx0}", y2: "{by + 4.0}" }
            line { class: "l2d-bar", x1: "{bx0 + bar_px}", y1: "{by - 4.0}", x2: "{bx0 + bar_px}", y2: "{by + 4.0}" }
            text { class: "l2d-label", x: "{bx0 + bar_px + 8.0}", y: "{by + 4.0}", "{bar_label}" }
            text { class: "strip-meta", x: "{PAD_L}", y: "{VB_H - 2.0}",
                "2D Layout · track {model.track_mm:.2} mm · λ {model.lambda_um:.3} µm · dashed: paraxial focus (no ray trace — see scope note)"
            }
        }
    }
}
