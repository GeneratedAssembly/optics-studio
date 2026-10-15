//! Editor-column and setup panels (spec §3.1–§3.2).

use dioxus::prelude::*;
use optics_zemax::ImportedSystem;

use super::studio::Shell;
use super::studio::SurfaceField;
use crate::shell::dispatch::{EnginePick, EngineSettings};
use crate::shell::prescription::image_distance_mm;
use crate::shell::source::SourceSpec;

/// Number + label row committing on change (blur/enter).
fn num_field(
    label: &'static str,
    value: String,
    step: &'static str,
    commit: impl Fn(String) + 'static,
) -> Element {
    rsx! {
        div { class: "row",
            label { "{label}" }
            input {
                class: "num",
                r#type: "number",
                value: "{value}",
                step: "{step}",
                onchange: move |evt| commit(evt.value()),
            }
        }
    }
}

fn parse(raw: &str) -> Option<f64> {
    raw.trim().parse().ok()
}

/// Thickness cell label (mm or `INF`).
fn thick_of(s: &optics_zemax::ZmxSurface) -> String {
    s.thickness_mm
        .map(|t| format!("{t:.3}"))
        .unwrap_or_else(|| "INF".into())
}

#[component]
pub fn FilePanel() -> Element {
    let studio = use_context::<Shell>();
    let recent = studio.recent();
    rsx! {
        div { class: "panel",
            h2 { "File" }
            div { class: "row",
                button {
                    class: "btn",
                    onclick: move |_| {
                        let mut s = studio;
                        s.load_sample();
                    },
                    "Load sample"
                }
                label { class: "btn",
                    "Import .zmx"
                    input {
                        r#type: "file",
                        accept: ".zmx",
                        style: "display:none",
                        oninput: move |evt| {
                            for file in evt.files() {
                                let mut s = studio;
                                wasm_bindgen_futures::spawn_local(async move {
                                    match file.read_string().await {
                                        Ok(text) => s.set_document(&file.name(), &text),
                                        Err(e) => s.push_toast(format!("{}: {e}", file.name())),
                                    }
                                });
                            }
                        },
                    }
                }
                label { class: "btn",
                    "Load .AGF"
                    input {
                        r#type: "file",
                        accept: ".agf,.AGF",
                        style: "display:none",
                        oninput: move |evt| {
                            for file in evt.files() {
                                let mut s = studio;
                                wasm_bindgen_futures::spawn_local(async move {
                                    match file.read_string().await {
                                        Ok(text) => s.load_agf(&file.name(), &text),
                                        Err(e) => s.push_toast(format!("{}: {e}", file.name())),
                                    }
                                });
                            }
                        },
                    }
                }
            }
            if !recent.is_empty() {
                div { class: "recent-list",
                    for (i, (name, text)) in recent.iter().enumerate() {
                        button {
                            key: "{i}",
                            onclick: {
                                let name = name.clone();
                                let text = text.clone();
                                move |_| {
                                    let mut s = studio;
                                    s.set_document(&name, &text);
                                }
                            },
                            "{name}"
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn TextEditor() -> Element {
    let studio = use_context::<Shell>();
    let text = studio.zmx_text();
    let err = studio.parse_error();
    let edited = studio.edited();
    rsx! {
        div { class: "panel",
            h2 { "Prescription source (.zmx — authoritative)" }
            textarea {
                class: "editor",
                value: "{text}",
                oninput: move |evt| {
                    let mut s = studio;
                    s.zmx_text.set(evt.value());
                    s.schedule_parse(250);
                },
            }
            div { class: "row",
                if edited {
                    span { class: "badge edited", "edited, re-parse to persist" }
                }
                button {
                    class: "btn",
                    onclick: move |_| {
                        let mut s = studio;
                        s.parse_now();
                    },
                    "Re-parse"
                }
            }
            if let Some(e) = err {
                div { class: "err", "{e}" }
            }
        }
    }
}

fn paraxial_of(
    imported: &Option<ImportedSystem>,
    band: usize,
    surfaces: &[optics_zemax::ZmxSurface],
) -> Option<(f64, f64, Option<f64>)> {
    let sys = imported.as_ref()?;
    let lambda = *sys.system().wavelengths_um.get(band)?;
    let report = sys.paraxial_at(lambda).ok()?;
    Some((report.efl_mm, report.bfl_mm, image_distance_mm(surfaces)))
}

#[component]
pub fn ParaxialCard() -> Element {
    let studio = use_context::<Shell>();
    let imported = studio.imported();
    let band = studio.band();
    let surfaces = studio.surfaces();
    let card = paraxial_of(&imported, band, &surfaces);
    let lambda_label = imported
        .as_ref()
        .and_then(|s| s.system().wavelengths_um.get(band).copied())
        .map(|w| format!("{w:.4}"))
        .unwrap_or_default();
    rsx! {
        div { class: "panel",
            h2 { "Paraxial" }
            if let Some((efl, bfl, image)) = card {
                div { class: "row", "EFL {efl:.3} mm · BFL {bfl:.3} mm" }
                div { class: "small", "at λ {lambda_label} µm" }
                if let Some(d) = image {
                    div { class: "small", "file image distance: {d:.3} mm (cross-check)" }
                }
            } else {
                div { class: "small", "import a .zmx to populate" }
            }
        }
    }
}

#[component]
pub fn WavelengthPanel() -> Element {
    let studio = use_context::<Shell>();
    let imported = studio.imported();
    let band = studio.band();
    let enabled = studio.enabled();
    rsx! {
        div { class: "panel",
            h2 { "Wavelengths (WAVM / PWAV)" }
            if let Some(sys) = imported.as_ref() {
                for (i, w) in sys.system().wavelengths_um.iter().enumerate() {
                    if *w > 0.0 {
                        div { class: "wave-row", key: "{i}",
                            input {
                                r#type: "checkbox",
                                checked: enabled.get(i).copied().unwrap_or(true),
                                oninput: move |evt| {
                                    let mut s = studio;
                                    let mut v = s.enabled.peek().clone();
                                    if let Some(slot) = v.get_mut(i) {
                                        *slot = evt.checked();
                                    }
                                    s.enabled.set(v);
                                },
                            }
                            input {
                                r#type: "radio",
                                name: "band",
                                checked: band == i,
                                onchange: move |_| {
                                    let mut s = studio;
                                    s.band.set(i);
                                    s.run_preview();
                                },
                            }
                            span { "{w:.4} µm" }
                            if sys.system().primary == i {
                                span { class: "small", "primary" }
                            }
                        }
                    }
                }
                div { class: "small", "radio = active preview band · checkboxes = analysis bands (10.2)" }
            } else {
                div { class: "small", "—" }
            }
        }
    }
}

#[component]
pub fn SourcePanel() -> Element {
    let studio = use_context::<Shell>();
    let spec = studio.source();
    let set_spec = move |s: SourceSpec| {
        let mut st = studio;
        st.source.set(s);
        st.schedule_preview(0);
    };
    rsx! {
        div { class: "panel",
            h2 { "Source" }
            div { class: "row",
                for (name, kind) in [("Point", 0), ("Plane wave", 1), ("Gaussian", 2)] {
                    label {
                        input {
                            r#type: "radio",
                            name: "source",
                            checked: kind_of(&spec) == kind,
                            onchange: {
                                let grid = spec.grid();
                                move |_| {
                                    let s = match kind {
                                        0 => SourceSpec::Point {
                                            distance_m: 1.0,
                                            amplitude: 1.0,
                                            nx: grid.0,
                                            ny: grid.1,
                                            dx: grid.2,
                                            dy: grid.3,
                                        },
                                        1 => SourceSpec::PlaneWave {
                                            theta_x_deg: 0.0,
                                            theta_y_deg: 0.0,
                                            amplitude: 1.0,
                                            nx: grid.0,
                                            ny: grid.1,
                                            dx: grid.2,
                                            dy: grid.3,
                                        },
                                        _ => SourceSpec::Gaussian {
                                            w0: 1e-3,
                                            amplitude: 1.0,
                                            z_m: 0.0,
                                            nx: grid.0,
                                            ny: grid.1,
                                            dx: grid.2,
                                            dy: grid.3,
                                        },
                                    };
                                    set_spec(s);
                                }
                            },
                        }
                        " {name}"
                    }
                }
            }
            {grid_fields(studio, spec)}
            match spec {
                SourceSpec::Point { distance_m, amplitude, .. } => rsx! {
                    {num_field("dist (m)", format!("{distance_m}"), "0.1", move |raw| {
                        if let (Some(v), SourceSpec::Point { nx, ny, dx, dy, .. }) = (parse(&raw), read_spec(studio)) {
                            set_spec(SourceSpec::Point { distance_m: v, amplitude, nx, ny, dx, dy });
                        }
                    })}
                },
                SourceSpec::PlaneWave { theta_x_deg, theta_y_deg, .. } => rsx! {
                    {num_field("θx (deg)", format!("{theta_x_deg}"), "0.1", move |raw| {
                        if let (Some(v), SourceSpec::PlaneWave { theta_y_deg, amplitude, nx, ny, dx, dy, .. }) =
                            (parse(&raw), read_spec(studio))
                        {
                            set_spec(SourceSpec::PlaneWave { theta_x_deg: v, theta_y_deg, amplitude, nx, ny, dx, dy });
                        }
                    })}
                    {num_field("θy (deg)", format!("{theta_y_deg}"), "0.1", move |raw| {
                        if let (Some(v), SourceSpec::PlaneWave { theta_x_deg, amplitude, nx, ny, dx, dy, .. }) =
                            (parse(&raw), read_spec(studio))
                        {
                            set_spec(SourceSpec::PlaneWave { theta_x_deg, theta_y_deg: v, amplitude, nx, ny, dx, dy });
                        }
                    })}
                },
                SourceSpec::Gaussian { w0, z_m, .. } => rsx! {
                    {num_field("w0 (mm)", format!("{:.4}", w0 * 1e3), "0.1", move |raw| {
                        if let (Some(v), SourceSpec::Gaussian { amplitude, nx, ny, dx, dy, .. }) =
                            (parse(&raw), read_spec(studio))
                        {
                            set_spec(SourceSpec::Gaussian { w0: v * 1e-3, amplitude, z_m, nx, ny, dx, dy });
                        }
                    })}
                    {num_field("z (m)", format!("{z_m}"), "0.1", move |raw| {
                        if let (Some(v), SourceSpec::Gaussian { w0, amplitude, nx, ny, dx, dy, .. }) =
                            (parse(&raw), read_spec(studio))
                        {
                            set_spec(SourceSpec::Gaussian { w0, amplitude, z_m: v, nx, ny, dx, dy });
                        }
                    })}
                },
            }
        }
    }
}

fn kind_of(spec: &SourceSpec) -> u8 {
    match spec {
        SourceSpec::Point { .. } => 0,
        SourceSpec::PlaneWave { .. } => 1,
        SourceSpec::Gaussian { .. } => 2,
    }
}

fn read_spec(studio: Shell) -> SourceSpec {
    *studio.source.peek()
}

fn grid_fields(studio: Shell, spec: SourceSpec) -> Element {
    let (nx, ny, dx, dy) = spec.grid();
    let dx_mm = format!("{:.4}", dx * 1e3);
    let dy_mm = format!("{:.4}", dy * 1e3);
    rsx! {
        div { class: "row",
            label { "nx" }
            input { class:"num", r#type: "number", value: "{nx}", min: "8", max: "512", onchange: move |evt| {
                let mut s = studio;
                let spec = *s.source.peek();
                let (nx, ny, dx, dy) = spec.grid();
                s.source.set(with_grid(&spec, evt.value().parse().unwrap_or(nx), ny, dx, dy));
                s.schedule_preview(0);
            } }
            label { "ny" }
            input { class:"num", r#type: "number", value: "{ny}", min: "8", max: "512", onchange: move |evt| {
                let mut s = studio;
                let spec = *s.source.peek();
                let (nx, ny, dx, dy) = spec.grid();
                s.source.set(with_grid(&spec, nx, evt.value().parse().unwrap_or(ny), dx, dy));
                s.schedule_preview(0);
            } }
        }
        div { class: "row",
            label { "dx (mm)" }
            input { class:"num", r#type: "number", value: "{dx_mm}", onchange: move |evt| {
                let mut s = studio;
                let spec = *s.source.peek();
                let (nx, ny, dx, dy) = spec.grid();
                let v = evt.value().parse::<f64>().unwrap_or(dx * 1e3) * 1e-3;
                s.source.set(with_grid(&spec, nx, ny, v, dy));
                s.schedule_preview(0);
            } }
            label { "dy (mm)" }
            input { class:"num", r#type: "number", value: "{dy_mm}", onchange: move |evt| {
                let mut s = studio;
                let spec = *s.source.peek();
                let (nx, ny, dx, dy) = spec.grid();
                let v = evt.value().parse::<f64>().unwrap_or(dy * 1e3) * 1e-3;
                s.source.set(with_grid(&spec, nx, ny, dx, v));
                s.schedule_preview(0);
            } }
        }
    }
}

fn with_grid(spec: &SourceSpec, nx: usize, ny: usize, dx: f64, dy: f64) -> SourceSpec {
    match *spec {
        SourceSpec::Point {
            distance_m,
            amplitude,
            ..
        } => SourceSpec::Point {
            distance_m,
            amplitude,
            nx,
            ny,
            dx,
            dy,
        },
        SourceSpec::PlaneWave {
            theta_x_deg,
            theta_y_deg,
            amplitude,
            ..
        } => SourceSpec::PlaneWave {
            theta_x_deg,
            theta_y_deg,
            amplitude,
            nx,
            ny,
            dx,
            dy,
        },
        SourceSpec::Gaussian {
            w0, amplitude, z_m, ..
        } => SourceSpec::Gaussian {
            w0,
            amplitude,
            z_m,
            nx,
            ny,
            dx,
            dy,
        },
    }
}

#[component]
pub fn EnginePanel() -> Element {
    let studio = use_context::<Shell>();
    let eng = studio.engine();
    let sugg = studio.suggestion();
    rsx! {
        div { class: "panel",
            h2 { "Propagation engine" }
            div { class: "row",
                for pick in EnginePick::ALL {
                    label {
                        input {
                            r#type: "radio",
                            name: "engine",
                            checked: eng.pick == pick,
                            onchange: move |_| {
                                let mut s = studio;
                                let mut e = *s.engine.peek();
                                e.pick = pick;
                                s.engine.set(e);
                                s.run_preview();
                            },
                        }
                        " {pick}"
                    }
                }
            }
            if let Some(sug) = sugg {
                div { class: "row",
                    span { class: "small", "regime recommends: {sug}" }
                    button {
                        class: "btn",
                        onclick: move |_| {
                            let mut s = studio;
                            let mut e = *s.engine.peek();
                            e.pick = sug;
                            s.engine.set(e);
                            s.run_preview();
                        },
                        "switch"
                    }
                }
            }
            {engine_knobs(studio, eng)}
        }
    }
}

fn engine_knobs(studio: Shell, eng: EngineSettings) -> Element {
    rsx! {
        match eng.pick {
            EnginePick::Asm => rsx! {
                {num_field("padding", format!("{}", eng.asm.padding), "0.5", move |raw| {
                    if let Some(v) = parse(&raw) {
                        let mut s = studio;
                        let mut e = *s.engine.peek();
                        e.asm.padding = v;
                        s.engine.set(e);
                        s.schedule_preview(0);
                    }
                })}
                {num_field("max padding", format!("{}", eng.asm.max_padding), "0.5", move |raw| {
                    if let Some(v) = parse(&raw) {
                        let mut s = studio;
                        let mut e = *s.engine.peek();
                        e.asm.max_padding = v;
                        s.engine.set(e);
                        s.schedule_preview(0);
                    }
                })}
                {num_field("medium n", format!("{}", eng.asm.medium), "0.01", move |raw| {
                    if let Some(v) = parse(&raw) {
                        let mut s = studio;
                        let mut e = *s.engine.peek();
                        e.asm.medium = v;
                        s.engine.set(e);
                        s.schedule_preview(0);
                    }
                })}
                div { class: "row",
                    label { input { r#type: "checkbox", checked: eng.asm.band_limited,
                        onchange: move |evt| {
                            let mut s = studio;
                            let mut e = *s.engine.peek();
                            e.asm.band_limited = evt.checked();
                            s.engine.set(e);
                            s.schedule_preview(0);
                        } } " band-limited" }
                    label { input { r#type: "checkbox", checked: eng.asm.strict,
                        onchange: move |evt| {
                            let mut s = studio;
                            let mut e = *s.engine.peek();
                            e.asm.strict = evt.checked();
                            s.engine.set(e);
                            s.schedule_preview(0);
                        } } " strict" }
                }
            },
            EnginePick::FresnelTransfer | EnginePick::FresnelConvolution => rsx! {
                {num_field("padding", format!("{}", eng.fresnel.padding), "0.5", move |raw| {
                    if let Some(v) = parse(&raw) {
                        let mut s = studio;
                        let mut e = *s.engine.peek();
                        e.fresnel.padding = v;
                        s.engine.set(e);
                        s.schedule_preview(0);
                    }
                })}
                {num_field("max padding", format!("{}", eng.fresnel.max_padding), "0.5", move |raw| {
                    if let Some(v) = parse(&raw) {
                        let mut s = studio;
                        let mut e = *s.engine.peek();
                        e.fresnel.max_padding = v;
                        s.engine.set(e);
                        s.schedule_preview(0);
                    }
                })}
                div { class: "row",
                    label { input { r#type: "checkbox", checked: eng.fresnel.strict,
                        onchange: move |evt| {
                            let mut s = studio;
                            let mut e = *s.engine.peek();
                            e.fresnel.strict = evt.checked();
                            s.engine.set(e);
                            s.schedule_preview(0);
                        } } " strict" }
                }
            },
            EnginePick::Fraunhofer => rsx! {
                {num_field("max N_F", format!("{}", eng.fraunhofer.max_n_f), "0.5", move |raw| {
                    if let Some(v) = parse(&raw) {
                        let mut s = studio;
                        let mut e = *s.engine.peek();
                        e.fraunhofer.max_n_f = v;
                        s.engine.set(e);
                        s.schedule_preview(0);
                    }
                })}
                div { class: "row",
                    label { input { r#type: "checkbox", checked: eng.fraunhofer.strict,
                        onchange: move |evt| {
                            let mut s = studio;
                            let mut e = *s.engine.peek();
                            e.fraunhofer.strict = evt.checked();
                            s.engine.set(e);
                            s.schedule_preview(0);
                        } } " strict" }
                }
            },
            EnginePick::RayleighSommerfeld => rsx! {
                {num_field("padding", format!("{}", eng.rs.padding), "0.5", move |raw| {
                    if let Some(v) = parse(&raw) {
                        let mut s = studio;
                        let mut e = *s.engine.peek();
                        e.rs.padding = v;
                        s.engine.set(e);
                        s.schedule_preview(0);
                    }
                })}
                div { class: "row",
                    label { input { r#type: "checkbox", checked: eng.rs.strict,
                        onchange: move |evt| {
                            let mut s = studio;
                            let mut e = *s.engine.peek();
                            e.rs.strict = evt.checked();
                            s.engine.set(e);
                            s.schedule_preview(0);
                        } } " strict" }
                }
            },
        }
    }
}

#[component]
pub fn PrescriptionTable() -> Element {
    let studio = use_context::<Shell>();
    let surfaces = studio.surfaces();
    let phase = studio.phase_steps();
    let glasses = studio.glass_names();
    rsx! {
        div { class: "panel",
            h2 { "Prescription (table mode — live simulation only)" }
            if surfaces.is_empty() {
                div { class: "small", "import a .zmx to edit surfaces" }
            } else {
                table { class: "surf",
                    thead {
                        tr {
                            th { "idx" }
                            th { "CURV 1/mm" }
                            th { "conic" }
                            th { "semi (mm)" }
                            th { "thick (mm)" }
                            th { "glass" }
                            th { "comment" }
                        }
                    }
                    tbody {
                        for (i, s) in surfaces.iter().enumerate() {
                            tr {
                                key: "{i}",
                                class: if phase.iter().any(|(idx, v)| *idx == s.index && *v > 2.0) { "flagged" } else { "" },
                                td { "{s.index}" if s.stop { span { class: "stop-flag", " STOP" } } }
                                td {
                                    {surf_input(studio, i, SurfaceField::Curvature, format!("{:.6}", s.curvature_mm), "-0.1", "0.1", "0.0005")}
                                }
                                td {
                                    {surf_input(studio, i, SurfaceField::Conic, format!("{:.3}", s.conic), "-5", "5", "0.05")}
                                }
                                td {
                                    {surf_input(studio, i, SurfaceField::SemiDiam, format!("{:.3}", s.semi_diam_mm), "0", "50", "0.1")}
                                }
                                td {
                                    input {
                                        class: "num",
                                        r#type: "text",
                                        value: "{thick_of(s)}",
                                        onchange: move |evt| {
                                            let mut st = studio;
                                            st.edit_surface(i, SurfaceField::Thickness, &evt.value());
                                            st.schedule_preview(100);
                                        },
                                    }
                                }
                                td {
                                    select {
                                        value: "{s.glass.clone().unwrap_or_default()}",
                                        onchange: move |evt| {
                                            let mut st = studio;
                                            st.edit_surface(i, SurfaceField::Glass, &evt.value());
                                            st.schedule_preview(100);
                                        },
                                        option { value: "", "— (air)" }
                                        for name in glasses.iter() {
                                            option { value: "{name}", selected: s.glass.as_deref() == Some(name.as_str()), "{name}" }
                                        }
                                    }
                                }
                                td { class: "small", "{s.comment}" }
                            }
                        }
                    }
                }
                div { class: "small", "dark rows: worst per-surface phase step > 2 rad at the active band (spec §5.3)" }
            }
        }
    }
}

/// Numeric field + slider pair bound to one working-copy surface field.
fn surf_input(
    studio: Shell,
    i: usize,
    field: SurfaceField,
    value: String,
    min: &'static str,
    max: &'static str,
    step: &'static str,
) -> Element {
    rsx! {
        div {
            input {
                class: "num",
                r#type: "number",
                value: "{value}",
                step: "{step}",
                onchange: move |evt| {
                    let mut s = studio;
                    s.edit_surface(i, field, &evt.value());
                    s.schedule_preview(100);
                },
            }
            input {
                r#type: "range",
                min: "{min}",
                max: "{max}",
                step: "{step}",
                value: "{value}",
                oninput: move |evt| {
                    let mut s = studio;
                    if let Ok(v) = evt.value().parse::<f64>() {
                        s.slide_surface(i, field, v);
                        s.schedule_preview(100);
                    }
                },
            }
        }
    }
}

#[component]
pub fn CarrierPanel() -> Element {
    let studio = use_context::<Shell>();
    let carrier = studio.carrier();
    let values = studio.carrier_values();
    let err = studio.carrier_error();
    rsx! {
        div { class: "panel",
            h2 { "Carrier sliders — demo singlet (spec §3.2)" }
            div { class: "small", "Carrier::params() → one slider per slot, clamped [lo, hi]; each commit re-runs build → Design::beamline → Beamline::psf." }
            for (i, slot) in carrier.params().slots().iter().enumerate() {
                div { class: "slider-row", key: "{i}",
                    label { "{slot.name}" }
                    input {
                        r#type: "range",
                        min: "{slot.lo}",
                        max: "{slot.hi}",
                        step: "{(slot.hi - slot.lo) / 200.0}",
                        value: "{values.get(i).copied().unwrap_or(slot.value)}",
                        oninput: move |evt| {
                            let mut s = studio;
                            if let Ok(v) = evt.value().parse::<f64>() {
                                let mut vals = s.carrier_values.peek().clone();
                                if let Some(slot) = vals.get_mut(i) {
                                    *slot = v;
                                }
                                s.carrier_values.set(vals);
                                s.schedule_preview(100);
                            }
                        },
                        onchange: move |evt| {
                            let mut s = studio;
                            if let Ok(v) = evt.value().parse::<f64>() {
                                let mut vals = s.carrier_values.peek().clone();
                                if let Some(slot) = vals.get_mut(i) {
                                    *slot = v;
                                }
                                s.carrier_values.set(vals);
                                s.run_preview();
                            }
                        },
                    }
                    output { "{values.get(i).copied().unwrap_or(slot.value):.3}" }
                }
            }
            if let Some(e) = err {
                div { class: "err", "{e}" }
            }
        }
    }
}
