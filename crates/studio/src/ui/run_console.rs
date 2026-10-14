//! The inverse run console (spec §3.4): solver picker, budget/seed/scale,
//! random-init toggle, chunked Start/Stop, convergence plot, progress
//! readout, apply-result, observation config, and merit editor.
//!
//! The console owns the complete [`RunJob`] because the solver worker
//! reconstructs the problem from it across the worker boundary. Apply
//! result writes `Solution::best` into `RunJob::start_values` (the
//! continuation start); wiring the best vector into the shell's sliders /
//! grid editor is task 10.1's side of §3.4.

use std::cell::RefCell;
use std::rc::Rc;

use dioxus::prelude::*;

use crate::bridge::{RunBridge, RunOutput, schedule};
use crate::run::controller::{RunController, RunPhase};
use crate::run::job::{DesignSpec, MaterialSpec, MeritSpec, RunJob, SolverSpec, SourceSpec};

use super::convergence::ConvergencePlot;

/// Bridge + output sink, shared between the component and the worker
/// callback (the callback drives the next chunk).
#[derive(Default)]
struct Holder {
    bridge: Option<RunBridge>,
    sink: Option<Rc<dyn Fn(RunOutput)>>,
}

/// Parse helpers for the form fields (UI-layer unit conversion only,
/// spec §3.5: fields are labeled in engineering units, the job is SI).
fn f(raw: &str) -> Option<f64> {
    raw.trim().parse().ok()
}

fn usize_(raw: &str) -> Option<usize> {
    raw.trim().parse().ok()
}

fn u64_(raw: &str) -> Option<u64> {
    raw.trim().parse().ok()
}

fn parse_bands(raw: &str) -> Option<Vec<f64>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        out.push(part.trim().parse::<f64>().ok()? * 1e-6);
    }
    (!out.is_empty()).then_some(out)
}

fn parse_scenes(raw: &str) -> Option<Vec<(f64, f64)>> {
    let mut out = Vec::new();
    for part in raw.split(';') {
        let (x, y) = part.trim().split_once(',')?;
        out.push((
            x.trim().parse::<f64>().ok()? * 1e-6,
            y.trim().parse::<f64>().ok()? * 1e-6,
        ));
    }
    (!out.is_empty()).then_some(out)
}

fn fmt_bands(bands: &[f64]) -> String {
    bands
        .iter()
        .map(|b| format!("{:.2}", b * 1e6))
        .collect::<Vec<_>>()
        .join(", ")
}

fn fmt_scenes(scenes: &[(f64, f64)]) -> String {
    scenes
        .iter()
        .map(|(x, y)| format!("{:.1},{:.1}", x * 1e6, y * 1e6))
        .collect::<Vec<_>>()
        .join("; ")
}

/// The Rc'd output handler: fold reports into the controller and launch
/// the next chunk unless stopped/finished (spec §3.4 run flow).
fn make_sink(
    cell: Rc<RefCell<Holder>>,
    controller: Signal<RunController>,
) -> Rc<dyn Fn(RunOutput)> {
    Rc::new(move |out| {
        let mut controller = controller;
        let launch = match &out {
            RunOutput::Chunk(report) => controller.write().absorb(report),
            RunOutput::Failed(message) => {
                controller.write().fail(message.clone());
                false
            }
            RunOutput::Started => true,
        };
        if launch {
            let cell = Rc::clone(&cell);
            schedule(move || {
                let mut holder = cell.borrow_mut();
                if let Some(sink) = holder.sink.clone()
                    && let Some(bridge) = holder.bridge.as_mut()
                {
                    bridge.step(&sink);
                }
            });
        }
    })
}

/// A labelled text field; commits the raw text on change (blur/Enter).
#[component]
fn Num(label: String, initial: String, on_commit: EventHandler<String>) -> Element {
    let mut text = use_signal(move || initial.clone());
    rsx! {
        div { class: "field",
            label { "{label}" }
            input {
                value: "{text}",
                oninput: move |e| text.set(e.value()),
                onchange: move |e| on_commit.call(e.value()),
            }
        }
    }
}

/// A labelled checkbox.
#[component]
fn Toggle(label: String, current: bool, on_toggle: EventHandler<bool>) -> Element {
    let mut box_state = use_signal(move || current);
    rsx! {
        div { class: "field toggle",
            input {
                r#type: "checkbox",
                checked: box_state(),
                onchange: move |e| {
                    let v = e.checked();
                    box_state.set(v);
                    on_toggle.call(v);
                },
            }
            label { "{label}" }
        }
    }
}

/// A labelled dropdown with `(value, label)` options.
#[component]
fn Picker(
    label: String,
    current: String,
    options: Vec<(String, String)>,
    on_pick: EventHandler<String>,
) -> Element {
    let mut choice = use_signal(move || current.clone());
    rsx! {
        div { class: "field",
            label { "{label}" }
            select {
                onchange: move |e| {
                    choice.set(e.value());
                    on_pick.call(e.value());
                },
                for (value, text) in options {
                    option { value: "{value}", selected: choice() == value, "{text}" }
                }
            }
        }
    }
}

/// The run console screen (right/bottom region, spec §3).
#[component]
pub fn RunConsole() -> Element {
    let job = use_signal(RunJob::demo_ee_focus);
    let mut controller = use_signal(RunController::new);
    let log_y = use_signal(|| true);
    let cell = use_hook(|| Rc::new(RefCell::new(Holder::default()))).clone();

    let phase = controller.read().phase();
    let running = matches!(phase, RunPhase::Running | RunPhase::Stopping);
    let history = controller.read().history().to_vec();
    let value = controller.read().value();
    let evaluations = controller.read().total_evaluations();
    let chunks = controller.read().chunks_done();
    let error = controller.read().error().map(str::to_string);
    let best = controller.read().best().to_vec();
    let elapsed = controller.read().elapsed_ms(js_sys::Date::now());
    let transport = match cell.borrow().bridge.as_ref() {
        Some(bridge) if bridge.is_worker() => "web worker",
        Some(_) => "inline (fallback)",
        None => "not started",
    };
    let card = {
        let controller = controller.read();
        let job = job.read();
        controller.run_card(&job)
    };

    // --- Start / Stop / Apply --------------------------------------------
    let start_cell = cell.clone();
    let on_start = move |_| {
        let prepared = job.read().clone();
        controller.write().begin(js_sys::Date::now());
        let mut holder = start_cell.borrow_mut();
        if holder.bridge.is_none() {
            let sink = make_sink(Rc::clone(&start_cell), controller);
            holder.sink = Some(Rc::clone(&sink));
            holder.bridge = Some(RunBridge::spawn(sink));
        }
        let sink = Rc::clone(holder.sink.as_ref().unwrap());
        holder.bridge.as_mut().unwrap().start(prepared, &sink);
    };
    let on_stop = {
        let mut controller = controller;
        move |_| controller.write().request_stop()
    };
    let on_apply = {
        move |_| {
            let mut job = job;
            let best = controller.read().best().to_vec();
            if !best.is_empty() {
                job.write().start_values = Some(best);
            }
        }
    };
    let on_new_run = {
        let cell = cell.clone();
        let mut controller = controller;
        move |_| {
            controller.write().reset();
            if let Some(bridge) = cell.borrow_mut().bridge.as_mut() {
                bridge.reset();
            }
        }
    };

    // --- Design form ------------------------------------------------------
    let (dims, pitch, z_after, design_init) = {
        let j = job.read();
        match &j.design {
            DesignSpec::GridPhaseRelay {
                dims,
                pitch,
                z_after,
                init,
                ..
            }
            | DesignSpec::GridThicknessRelay {
                dims,
                pitch,
                z_after,
                init,
                ..
            } => (*dims, *pitch, *z_after, *init),
        }
    };
    let on_dims_x = {
        move |raw: String| {
            let mut job = job;
            if let Some(v) = usize_(&raw) {
                let mut j = job.write();
                match &mut j.design {
                    DesignSpec::GridPhaseRelay { dims, .. }
                    | DesignSpec::GridThicknessRelay { dims, .. } => dims.0 = v,
                }
            }
        }
    };
    let on_dims_y = {
        move |raw: String| {
            let mut job = job;
            if let Some(v) = usize_(&raw) {
                let mut j = job.write();
                match &mut j.design {
                    DesignSpec::GridPhaseRelay { dims, .. }
                    | DesignSpec::GridThicknessRelay { dims, .. } => dims.1 = v,
                }
            }
        }
    };
    let on_pitch = {
        move |raw: String| {
            let mut job = job;
            if let Some(v) = f(&raw) {
                let mut j = job.write();
                match &mut j.design {
                    DesignSpec::GridPhaseRelay { pitch, .. }
                    | DesignSpec::GridThicknessRelay { pitch, .. } => *pitch = (v * 1e-6, v * 1e-6),
                }
            }
        }
    };
    let on_z_after = {
        move |raw: String| {
            let mut job = job;
            if let Some(v) = f(&raw) {
                let mut j = job.write();
                match &mut j.design {
                    DesignSpec::GridPhaseRelay { z_after, .. }
                    | DesignSpec::GridThicknessRelay { z_after, .. } => *z_after = v * 1e-3,
                }
            }
        }
    };
    let on_lo = {
        move |raw: String| {
            let mut job = job;
            if let Some(v) = f(&raw) {
                let mut j = job.write();
                if let DesignSpec::GridPhaseRelay { lo, .. } = &mut j.design {
                    *lo = v;
                }
            }
        }
    };
    let on_hi = {
        move |raw: String| {
            let mut job = job;
            if let Some(v) = f(&raw) {
                let mut j = job.write();
                if let DesignSpec::GridPhaseRelay { hi, .. } = &mut j.design {
                    *hi = v;
                }
            }
        }
    };
    let on_init = {
        move |raw: String| {
            let mut job = job;
            if let Some(v) = f(&raw) {
                let mut j = job.write();
                match &mut j.design {
                    DesignSpec::GridPhaseRelay { init, .. }
                    | DesignSpec::GridThicknessRelay { init, .. } => *init = v,
                }
            }
        }
    };
    let on_t_max = {
        move |raw: String| {
            let mut job = job;
            if let Some(v) = f(&raw) {
                let mut j = job.write();
                if let DesignSpec::GridThicknessRelay { t_max, .. } = &mut j.design {
                    *t_max = v * 1e-6;
                }
            }
        }
    };
    let design_kind = match &job.read().design {
        DesignSpec::GridPhaseRelay { .. } => "phase",
        DesignSpec::GridThicknessRelay { .. } => "thickness",
    };
    let (t_max, material_name) = match &job.read().design {
        DesignSpec::GridThicknessRelay {
            t_max, material, ..
        } => (*t_max, material.name.clone()),
        _ => (1e-6, "silica".into()),
    };
    let on_material_name = {
        move |raw: String| {
            let mut job = job;
            let mut j = job.write();
            if let DesignSpec::GridThicknessRelay { material, .. } = &mut j.design {
                material.name = raw;
            }
        }
    };
    let on_design_kind = {
        move |kind: String| {
            let mut job = job;
            let mut j = job.write();
            let common = match &j.design {
                DesignSpec::GridPhaseRelay {
                    dims,
                    pitch,
                    aperture,
                    center,
                    z_after,
                    medium,
                    ..
                } => (*dims, *pitch, *aperture, *center, *z_after, *medium),
                DesignSpec::GridThicknessRelay {
                    dims,
                    pitch,
                    aperture,
                    center,
                    z_after,
                    medium,
                    ..
                } => (*dims, *pitch, *aperture, *center, *z_after, *medium),
            };
            let (dims, pitch, aperture, center, z_after, medium) = common;
            j.design = if kind == "thickness" {
                DesignSpec::GridThicknessRelay {
                    dims,
                    pitch,
                    aperture,
                    center,
                    t_max: 1e-6,
                    init: 0.0,
                    material: MaterialSpec {
                        name: "silica".to_string(),
                        constant: Some(1.45f64.powi(2)),
                        terms: Vec::new(),
                    },
                    z_after,
                    medium,
                }
            } else {
                DesignSpec::GridPhaseRelay {
                    dims,
                    pitch,
                    aperture,
                    center,
                    lo: -std::f64::consts::PI,
                    hi: std::f64::consts::PI,
                    init: 0.0,
                    z_after,
                    medium,
                }
            };
        }
    };

    // --- Source form ------------------------------------------------------
    let source = job.read().source;
    let on_source_kind = {
        move |kind: String| {
            let mut job = job;
            let mut j = job.write();
            j.source = match kind.as_str() {
                "gaussian" => SourceSpec::Gaussian {
                    w0: 150e-6,
                    amplitude: 1.0,
                    center: (0.0, 0.0),
                    z: 0.0,
                },
                "planewave" => SourceSpec::PlaneWave {
                    theta_x: 0.0,
                    theta_y: 0.0,
                    amplitude: 1.0,
                },
                _ => SourceSpec::Spherical {
                    distance: 50e-3,
                    amplitude: 1.0,
                },
            };
        }
    };
    let on_source_field = {
        move |(which, raw): (String, String)| {
            let mut job = job;
            let Some(v) = f(&raw) else { return };
            let mut j = job.write();
            match (which.as_str(), &mut j.source) {
                ("distance", SourceSpec::Spherical { distance, .. }) => *distance = v * 1e-3,
                ("amplitude", SourceSpec::Spherical { amplitude, .. }) => *amplitude = v,
                ("w0", SourceSpec::Gaussian { w0, .. }) => *w0 = v * 1e-6,
                ("z", SourceSpec::Gaussian { z, .. }) => *z = v * 1e-3,
                ("amplitude", SourceSpec::Gaussian { amplitude, .. }) => *amplitude = v,
                ("theta_x", SourceSpec::PlaneWave { theta_x, .. }) => *theta_x = v,
                ("theta_y", SourceSpec::PlaneWave { theta_y, .. }) => *theta_y = v,
                ("amplitude", SourceSpec::PlaneWave { amplitude, .. }) => *amplitude = v,
                _ => {}
            }
        }
    };

    // --- Observation form ---------------------------------------------------
    let bands_text = fmt_bands(&job.read().bands);
    let scenes_text = fmt_scenes(&job.read().scenes);
    let on_bands = {
        move |raw: String| {
            let mut job = job;
            if let Some(bands) = parse_bands(&raw) {
                job.write().bands = bands;
            }
        }
    };
    let on_scenes = {
        move |raw: String| {
            let mut job = job;
            if let Some(scenes) = parse_scenes(&raw) {
                job.write().scenes = scenes;
            }
        }
    };
    let grid = job.read().grid;
    let on_grid_field = {
        move |(which, raw): (String, String)| {
            let mut job = job;
            let mut j = job.write();
            match which.as_str() {
                "nx" => j.grid.nx = usize_(&raw).unwrap_or(j.grid.nx),
                "ny" => j.grid.ny = usize_(&raw).unwrap_or(j.grid.ny),
                "dx" => {
                    if let Some(v) = f(&raw) {
                        j.grid.dx = v * 1e-6;
                    }
                }
                "dy" => {
                    if let Some(v) = f(&raw) {
                        j.grid.dy = v * 1e-6;
                    }
                }
                "scene_step" => {
                    if let Some(v) = f(&raw) {
                        j.scene_step = v * 1e-6;
                    }
                }
                "lambda_step" => {
                    if let Some(v) = f(&raw) {
                        j.lambda_step = v * 1e-9;
                    }
                }
                _ => {}
            }
        }
    };
    let asm = job.read().asm;
    let on_asm_field = {
        move |(which, raw): (String, String)| {
            let mut job = job;
            if let Some(v) = f(&raw) {
                let mut j = job.write();
                match which.as_str() {
                    "padding" => j.asm.padding = v,
                    "max_padding" => j.asm.max_padding = v,
                    _ => {}
                }
            }
        }
    };
    let on_band_limited = {
        let mut job = job;
        move |v: bool| job.write().asm.band_limited = v
    };
    let on_strict = {
        let mut job = job;
        move |v: bool| job.write().asm.strict = v
    };
    let derivatives = job.read().derivatives;
    let on_derivative = {
        move |(which, v): (String, bool)| {
            let mut job = job;
            let mut j = job.write();
            match which.as_str() {
                "x" => j.derivatives.x = v,
                "y" => j.derivatives.y = v,
                "lambda" => j.derivatives.lambda = v,
                _ => {}
            }
        }
    };

    // --- Merit form ---------------------------------------------------------
    let (merit_label, ee_radius_um, mtf_freq_lpmm, selector) = {
        let j = job.read();
        match &j.merit {
            MeritSpec::EncircledEnergy { radius, selector } => (
                "neg_encircled_energy".to_string(),
                *radius * 1e6,
                0.0,
                *selector,
            ),
            MeritSpec::Mtf { freq, selector } => {
                ("neg_mtf".to_string(), 0.0, *freq / 1e3, *selector)
            }
            _ => (j.merit.label().to_string(), 0.0, 0.0, j.merit_selector()),
        }
    };
    let on_merit_kind = {
        move |kind: String| {
            let mut job = job;
            let mut j = job.write();
            let selector = j.merit_selector();
            j.merit = match kind.as_str() {
                "spot_rms" => MeritSpec::SpotRms { selector },
                "neg_mtf" => MeritSpec::Mtf {
                    freq: 50.0e3,
                    selector,
                },
                _ => MeritSpec::EncircledEnergy {
                    radius: 60e-6,
                    selector,
                },
            };
        }
    };
    let on_ee_radius = {
        move |raw: String| {
            let mut job = job;
            if let Some(v) = f(&raw) {
                let mut j = job.write();
                if let MeritSpec::EncircledEnergy { radius, .. } = &mut j.merit {
                    *radius = v * 1e-6;
                }
            }
        }
    };
    let on_mtf_freq = {
        move |raw: String| {
            let mut job = job;
            if let Some(v) = f(&raw) {
                let mut j = job.write();
                if let MeritSpec::Mtf { freq, .. } = &mut j.merit {
                    *freq = v * 1e3;
                }
            }
        }
    };
    let on_selector_field = {
        move |(which, raw): (String, String)| {
            let mut job = job;
            let mut j = job.write();
            let mut selector = j.merit_selector();
            match which.as_str() {
                "band" => {
                    selector.band = if raw.trim().is_empty() {
                        None
                    } else {
                        usize_(&raw)
                    }
                }
                "scene" => {
                    selector.scene = if raw.trim().is_empty() {
                        None
                    } else {
                        usize_(&raw)
                    }
                }
                _ => {}
            }
            j.set_merit_selector(selector);
        }
    };

    // --- Solver + run form ---------------------------------------------------
    let solver = job.read().solver;
    let on_solver_kind = {
        move |kind: String| {
            let mut job = job;
            let mut j = job.write();
            j.solver = match kind.as_str() {
                "random_walk" => SolverSpec::RandomWalk { stagnation: 100 },
                "adam" => SolverSpec::Adam {
                    lr_frac: 0.02,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    clip_norm: Some(1.0),
                },
                _ => SolverSpec::Annealing {
                    t0: None,
                    t_end_frac: 1e-3,
                },
            };
        }
    };
    let on_solver_field = {
        move |(which, raw): (String, String)| {
            let mut job = job;
            let mut j = job.write();
            match (which.as_str(), &mut j.solver) {
                ("stagnation", SolverSpec::RandomWalk { stagnation }) => {
                    *stagnation = usize_(&raw).unwrap_or(*stagnation)
                }
                ("t_end_frac", SolverSpec::Annealing { t_end_frac, .. }) => {
                    if let Some(v) = f(&raw) {
                        *t_end_frac = v;
                    }
                }
                ("t0", SolverSpec::Annealing { t0, .. }) => {
                    *t0 = if raw.trim().is_empty() { None } else { f(&raw) };
                }
                ("lr_frac", SolverSpec::Adam { lr_frac, .. }) => {
                    if let Some(v) = f(&raw) {
                        *lr_frac = v;
                    }
                }
                ("beta1", SolverSpec::Adam { beta1, .. }) => {
                    if let Some(v) = f(&raw) {
                        *beta1 = v;
                    }
                }
                ("beta2", SolverSpec::Adam { beta2, .. }) => {
                    if let Some(v) = f(&raw) {
                        *beta2 = v;
                    }
                }
                ("eps", SolverSpec::Adam { eps, .. }) => {
                    if let Some(v) = f(&raw) {
                        *eps = v;
                    }
                }
                ("clip_norm", SolverSpec::Adam { clip_norm, .. }) => {
                    *clip_norm = if raw.trim().is_empty() { None } else { f(&raw) };
                }
                _ => {}
            }
        }
    };
    let run_cfg = {
        let j = job.read();
        (
            j.budget,
            j.chunk,
            j.seed,
            j.initial_scale,
            j.random_init,
            j.start_values.is_some(),
        )
    };
    let on_run_field = {
        move |(which, raw): (String, String)| {
            let mut job = job;
            let mut j = job.write();
            match which.as_str() {
                "budget" => j.budget = usize_(&raw).unwrap_or(j.budget),
                "chunk" => j.chunk = usize_(&raw).unwrap_or(j.chunk),
                "seed" => j.seed = u64_(&raw).unwrap_or(j.seed),
                "initial_scale" => {
                    if let Some(v) = f(&raw) {
                        j.initial_scale = v;
                    }
                }
                _ => {}
            }
        }
    };
    let on_random_init = {
        let mut job = job;
        move |v: bool| job.write().random_init = v
    };
    let on_log_y = {
        let mut log_y = log_y;
        move |v: bool| log_y.set(v)
    };

    rsx! {
        div { class: "console",
            div { class: "config",
                h3 { "Design" }
                Picker {
                    label: "carrier".to_string(),
                    current: design_kind.to_string(),
                    options: vec![
                        ("phase".to_string(), "Phase grid (rad)".into()),
                        ("thickness".to_string(), "Thickness grid (m)".into()),
                    ],
                    on_pick: on_design_kind,
                }
                Num { label: "dims nx".to_string(), initial: dims.0.to_string(), on_commit: on_dims_x }
                Num { label: "dims ny".to_string(), initial: dims.1.to_string(), on_commit: on_dims_y }
                Num { label: "pitch (µm)".to_string(), initial: format!("{:.1}", pitch.0 * 1e6), on_commit: on_pitch }
                Num { label: "DOE→detector z (mm)".to_string(), initial: format!("{:.2}", z_after * 1e3), on_commit: on_z_after }
                if matches!(job.read().design, DesignSpec::GridPhaseRelay { .. }) {
                    Num { label: "phase lo (rad)".to_string(), initial: format!("{:.4}", -std::f64::consts::PI), on_commit: on_lo }
                    Num { label: "phase hi (rad)".to_string(), initial: format!("{:.4}", std::f64::consts::PI), on_commit: on_hi }
                }
                if matches!(job.read().design, DesignSpec::GridThicknessRelay { .. }) {
                    Num { label: "t_max (µm)".to_string(), initial: format!("{:.2}", t_max * 1e6), on_commit: on_t_max }
                    Num { label: "material name".to_string(), initial: material_name, on_commit: on_material_name }
                }
                Num { label: "init value".to_string(), initial: design_init.to_string(), on_commit: on_init }

                h3 { "Illumination" }
                Picker {
                    label: "source".to_string(),
                    current: match source {
                        SourceSpec::Spherical { .. } => "spherical".to_string(),
                        SourceSpec::Gaussian { .. } => "gaussian".to_string(),
                        SourceSpec::PlaneWave { .. } => "planewave".to_string(),
                    },
                    options: vec![
                        ("spherical".to_string(), "Spherical point source".into()),
                        ("gaussian".to_string(), "Gaussian beam".into()),
                        ("planewave".to_string(), "Plane wave".into()),
                    ],
                    on_pick: on_source_kind,
                }
                match source {
                    SourceSpec::Spherical { distance, amplitude } => rsx! {
                        Num { label: "distance (mm)".to_string(), initial: format!("{:.1}", distance * 1e3), on_commit: move |raw| on_source_field(("distance".to_string(), raw)) }
                        Num { label: "amplitude".to_string(), initial: format!("{amplitude}"), on_commit: move |raw| on_source_field(("amplitude".to_string(), raw)) }
                    },
                    SourceSpec::Gaussian { w0, amplitude, z, .. } => rsx! {
                        Num { label: "w0 (µm)".to_string(), initial: format!("{:.1}", w0 * 1e6), on_commit: move |raw| on_source_field(("w0".to_string(), raw)) }
                        Num { label: "waist→plane z (mm)".to_string(), initial: format!("{:.1}", z * 1e3), on_commit: move |raw| on_source_field(("z".to_string(), raw)) }
                        Num { label: "amplitude".to_string(), initial: format!("{amplitude}"), on_commit: move |raw| on_source_field(("amplitude".to_string(), raw)) }
                    },
                    SourceSpec::PlaneWave { theta_x, theta_y, amplitude } => rsx! {
                        Num { label: "tilt θx (rad)".to_string(), initial: format!("{theta_x}"), on_commit: move |raw| on_source_field(("theta_x".to_string(), raw)) }
                        Num { label: "tilt θy (rad)".to_string(), initial: format!("{theta_y}"), on_commit: move |raw| on_source_field(("theta_y".to_string(), raw)) }
                        Num { label: "amplitude".to_string(), initial: format!("{amplitude}"), on_commit: move |raw| on_source_field(("amplitude".to_string(), raw)) }
                    },
                }

                h3 { "Observation" }
                Num { label: "bands λ (µm, comma list)".to_string(), initial: bands_text, on_commit: on_bands }
                Num { label: "scenes x,y (µm, semicolon list)".to_string(), initial: scenes_text, on_commit: on_scenes }
                Num { label: "frame nx".to_string(), initial: grid.nx.to_string(), on_commit: move |raw| on_grid_field(("nx".to_string(), raw)) }
                Num { label: "frame ny".to_string(), initial: grid.ny.to_string(), on_commit: move |raw| on_grid_field(("ny".to_string(), raw)) }
                Num { label: "frame pitch dx (µm)".to_string(), initial: format!("{:.1}", grid.dx * 1e6), on_commit: move |raw| on_grid_field(("dx".to_string(), raw)) }
                Num { label: "frame pitch dy (µm)".to_string(), initial: format!("{:.1}", grid.dy * 1e6), on_commit: move |raw| on_grid_field(("dy".to_string(), raw)) }
                Num { label: "scene FD step (µm)".to_string(), initial: format!("{:.2}", job.read().scene_step * 1e6), on_commit: move |raw| on_grid_field(("scene_step".to_string(), raw)) }
                Num { label: "band FD step (nm)".to_string(), initial: format!("{:.2}", job.read().lambda_step * 1e9), on_commit: move |raw| on_grid_field(("lambda_step".to_string(), raw)) }
                Toggle { label: "∂I/∂x₀ channel".to_string(), current: derivatives.x, on_toggle: move |v| on_derivative(("x".to_string(), v)) }
                Toggle { label: "∂I/∂y₀ channel".to_string(), current: derivatives.y, on_toggle: move |v| on_derivative(("y".to_string(), v)) }
                Toggle { label: "∂I/∂λ channel".to_string(), current: derivatives.lambda, on_toggle: move |v| on_derivative(("lambda".to_string(), v)) }
                Num { label: "ASM padding".to_string(), initial: format!("{}", asm.padding), on_commit: move |raw| on_asm_field(("padding".to_string(), raw)) }
                Num { label: "ASM max padding".to_string(), initial: format!("{}", asm.max_padding), on_commit: move |raw| on_asm_field(("max_padding".to_string(), raw)) }
                Toggle { label: "band-limited ASM".to_string(), current: asm.band_limited, on_toggle: on_band_limited }
                Toggle { label: "strict (densify on aliasing)".to_string(), current: asm.strict, on_toggle: on_strict }

                h3 { "Merit" }
                Picker {
                    label: "built-in".to_string(),
                    current: merit_label.clone(),
                    options: vec![
                        ("neg_encircled_energy".to_string(), "−EE at radius".into()),
                        ("spot_rms".to_string(), "Spot RMS radius".into()),
                        ("neg_mtf".to_string(), "−MTF at frequency".into()),
                    ],
                    on_pick: on_merit_kind,
                }
                if merit_label == "neg_encircled_energy" {
                    Num { label: "EE radius (µm)".to_string(), initial: format!("{:.1}", ee_radius_um), on_commit: on_ee_radius }
                }
                if merit_label == "neg_mtf" {
                    Num { label: "MTF freq (lp/mm)".to_string(), initial: format!("{:.1}", mtf_freq_lpmm), on_commit: on_mtf_freq }
                }
                Num { label: "selector band (blank = sum)".to_string(), initial: selector.band.map(|b| b.to_string()).unwrap_or_default(), on_commit: move |raw| on_selector_field(("band".to_string(), raw)) }
                Num { label: "selector scene (blank = sum)".to_string(), initial: selector.scene.map(|s| s.to_string()).unwrap_or_default(), on_commit: move |raw| on_selector_field(("scene".to_string(), raw)) }
                p { class: "note", "MSE-to-target and Strehl merits need a captured reference frame — wired with the viewport (task 10.2)." }
            }

            div { class: "run",
                h3 { "Run controls" }
                Picker {
                    label: "solver".to_string(),
                    current: match solver {
                        SolverSpec::RandomWalk { .. } => "random_walk".to_string(),
                        SolverSpec::Annealing { .. } => "annealing".to_string(),
                        SolverSpec::Adam { .. } => "adam".to_string(),
                    },
                    options: vec![
                        ("random_walk".to_string(), "Random walk".into()),
                        ("annealing".to_string(), "Annealing".into()),
                        ("adam".to_string(), "Adam (FD gradient)".into()),
                    ],
                    on_pick: on_solver_kind,
                }
                match solver {
                    SolverSpec::RandomWalk { stagnation } => rsx! {
                        Num { label: "stagnation".to_string(), initial: stagnation.to_string(), on_commit: move |raw| on_solver_field(("stagnation".to_string(), raw)) }
                    },
                    SolverSpec::Annealing { t0, t_end_frac } => rsx! {
                        Num { label: "t0 (blank = auto)".to_string(), initial: t0.map(|t| t.to_string()).unwrap_or_default(), on_commit: move |raw| on_solver_field(("t0".to_string(), raw)) }
                        Num { label: "t_end_frac".to_string(), initial: format!("{t_end_frac}"), on_commit: move |raw| on_solver_field(("t_end_frac".to_string(), raw)) }
                    },
                    SolverSpec::Adam { lr_frac, beta1, beta2, eps, clip_norm } => rsx! {
                        Num { label: "lr_frac".to_string(), initial: format!("{lr_frac}"), on_commit: move |raw| on_solver_field(("lr_frac".to_string(), raw)) }
                        Num { label: "beta1".to_string(), initial: format!("{beta1}"), on_commit: move |raw| on_solver_field(("beta1".to_string(), raw)) }
                        Num { label: "beta2".to_string(), initial: format!("{beta2}"), on_commit: move |raw| on_solver_field(("beta2".to_string(), raw)) }
                        Num { label: "eps".to_string(), initial: format!("{eps}"), on_commit: move |raw| on_solver_field(("eps".to_string(), raw)) }
                        Num { label: "clip_norm (blank = none)".to_string(), initial: clip_norm.map(|c| c.to_string()).unwrap_or_default(), on_commit: move |raw| on_solver_field(("clip_norm".to_string(), raw)) }
                    },
                }
                Num { label: "budget (evals; Adam: epochs)".to_string(), initial: run_cfg.0.to_string(), on_commit: move |raw| on_run_field(("budget".to_string(), raw)) }
                Num { label: "chunk (≥ 50)".to_string(), initial: run_cfg.1.to_string(), on_commit: move |raw| on_run_field(("chunk".to_string(), raw)) }
                Num { label: "seed".to_string(), initial: run_cfg.2.to_string(), on_commit: move |raw| on_run_field(("seed".to_string(), raw)) }
                Num { label: "initial_scale".to_string(), initial: format!("{}", run_cfg.3), on_commit: move |raw| on_run_field(("initial_scale".to_string(), raw)) }
                Toggle { label: "random init (uniform in bounds)".to_string(), current: run_cfg.4, on_toggle: on_random_init }
                div { class: "buttons",
                    button { class: "start", disabled: running, onclick: on_start, "Start" }
                    button { class: "stop", disabled: !running, onclick: on_stop, "Stop" }
                    button { disabled: running || best.is_empty(), onclick: on_apply, "Apply best" }
                    button { disabled: running, onclick: on_new_run, "New run" }
                }
                p { class: "status",
                    span { class: "badge", "{phase:?}" }
                    "transport: {transport} · evals: {evaluations} · chunks: {chunks} · elapsed: {elapsed:.0} ms · best: {value:.4e}"
                }
                if run_cfg.5 {
                    p { class: "note", "start_values loaded: this run continues from the applied best." }
                }
                if let Some(error) = &error {
                    p { class: "error", "{error}" }
                }

                h3 { "Convergence" }
                Toggle { label: "log-y".to_string(), current: log_y(), on_toggle: on_log_y }
                ConvergencePlot { history: history.clone(), log_y: log_y() }

                details {
                    summary { "Run card (reproducibility, spec §3.4)" }
                    pre { "{card}" }
                }
            }
        }
    }
}
