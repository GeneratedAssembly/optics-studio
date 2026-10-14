//! The solver web worker bridge (ADR-10.0 Q3, task 10.3 decision).
//!
//! Worker crate choice: **gloo-worker 0.6** (actor model, bincode
//! messages). It fits the chunked protocol exactly — the worker owns the
//! [`RunSession`], the UI sends `Step` after each report, and Stop is
//! "stop sending" (spec §3.4). comlink was the alternative; its async-RPC
//! harness adds nothing over a synchronous chunk loop.
//!
//! Spawning: the worker is a blob that imports the app's own module
//! script (gloo-worker's shim), so the worker-side bootstrap registers the
//! solver worker only inside worker scopes ([`worker_bootstrap`]). The
//! module script path is discovered from the page's `<script src>` at
//! runtime, which keeps it bundler-agnostic. If no loadable worker can be
//! created (strict CSP, unusual bundle layout), the console falls back to
//! the **inline transport**: the same chunk protocol executed on the main
//! thread, yielding to the event loop between chunks via a zero timeout.
//! Responsiveness then degrades to chunk granularity — the knob the spec
//! already defines (gap G-2) — and the UI labels the active transport.
//!
//! Known runtime caveat (manual QA, task 10.4): if the worker module loads
//! but never registers (bundler/CSP quirk), a run stays `Running` with no
//! reports; Stop returns control and the transport badge shows which path
//! is live. The gates here are compile-only for wasm (ADR-10.0 Q2).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use gloo_worker::{HandlerId, Registrable, Spawnable, Worker, WorkerBridge, WorkerScope};
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::run::job::RunJob;
use crate::run::session::{ChunkReport, RunSession};

/// Main → worker messages.
#[derive(Debug, Serialize, Deserialize)]
pub enum RunInput {
    /// Validate and prepare a run (does not execute it).
    Start(Box<RunJob>),
    /// Execute the next chunk.
    Step,
    /// Drop any session state.
    Reset,
}

/// Worker → main messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunOutput {
    /// A session is ready (or state was reset).
    Started,
    /// One chunk completed.
    Chunk(ChunkReport),
    /// Validation / solver / session error (typed display text, spec §3.5).
    Failed(String),
}

/// The dedicated solver worker: owns the [`RunSession`] and runs one chunk
/// per `Step` so the worker's message loop stays responsive.
#[derive(Default)]
pub struct SolverWorker {
    session: Option<RunSession>,
}

impl Worker for SolverWorker {
    type Message = ();
    type Input = RunInput;
    type Output = RunOutput;

    fn create(_scope: &WorkerScope<Self>) -> Self {
        Self::default()
    }

    fn update(&mut self, _scope: &WorkerScope<Self>, _msg: ()) {}

    fn received(&mut self, scope: &WorkerScope<Self>, msg: RunInput, id: HandlerId) {
        let out = match msg {
            RunInput::Start(job) => match RunSession::start(*job) {
                Ok(session) => {
                    self.session = Some(session);
                    RunOutput::Started
                }
                Err(e) => {
                    self.session = None;
                    RunOutput::Failed(e)
                }
            },
            RunInput::Step => match self.session.as_mut() {
                None => RunOutput::Failed("solver worker has no active session".into()),
                Some(session) => match session.run_chunk() {
                    Ok(report) => RunOutput::Chunk(report),
                    Err(e) => RunOutput::Failed(e),
                },
            },
            RunInput::Reset => {
                self.session = None;
                RunOutput::Started
            }
        };
        scope.respond(id, out);
    }
}

/// Runs on every module init (page and worker); registers the worker
/// handler only inside a worker scope (no `document` on the global).
#[wasm_bindgen(start)]
pub fn worker_bootstrap() {
    if !in_page_context() {
        SolverWorker::registrar().register();
    }
}

/// True when the global object is a window (main thread).
fn in_page_context() -> bool {
    js_sys::Reflect::has(
        &js_sys::global(),
        &wasm_bindgen::JsValue::from_str("document"),
    )
    .unwrap_or(false)
}

/// The active chunk transport.
pub enum RunBridge {
    /// Dedicated web worker (preferred).
    Worker(WorkerBridge<SolverWorker>),
    /// Main-thread inline session with event-loop yielding (fallback).
    /// Boxed: `RunSession` owns the carrier, merit, and job (large).
    Inline(Option<Box<RunSession>>),
}

impl RunBridge {
    /// Create the transport: try a dedicated worker against the discovered
    /// module script, fall back to inline execution. `sink` receives every
    /// output from either transport.
    pub fn spawn(sink: Rc<dyn Fn(RunOutput)>) -> Self {
        if let Some(path) = module_script_path() {
            let cb_sink = Rc::clone(&sink);
            let spawned = catch_unwind(AssertUnwindSafe(move || {
                let mut spawner = SolverWorker::spawner();
                spawner.callback(move |out| cb_sink(out));
                spawner.spawn(&path)
            }));
            if let Ok(bridge) = spawned {
                return Self::Worker(bridge);
            }
        }
        Self::Inline(None)
    }

    /// True when the dedicated-worker transport is active.
    pub fn is_worker(&self) -> bool {
        matches!(self, Self::Worker(_))
    }

    /// Begin a run (validate + prepare).
    pub fn start(&mut self, job: RunJob, sink: &Rc<dyn Fn(RunOutput)>) {
        match self {
            Self::Worker(bridge) => bridge.send(RunInput::Start(Box::new(job))),
            Self::Inline(session) => match RunSession::start(job) {
                Ok(s) => {
                    *session = Some(Box::new(s));
                    sink(RunOutput::Started);
                }
                Err(e) => sink(RunOutput::Failed(e)),
            },
        }
    }

    /// Launch the next chunk.
    pub fn step(&mut self, sink: &Rc<dyn Fn(RunOutput)>) {
        match self {
            Self::Worker(bridge) => bridge.send(RunInput::Step),
            Self::Inline(session) => {
                let out = match session.as_mut() {
                    None => RunOutput::Failed("no active session".into()),
                    Some(s) => match s.run_chunk() {
                        Ok(report) => RunOutput::Chunk(report),
                        Err(e) => RunOutput::Failed(e),
                    },
                };
                sink(out);
            }
        }
    }

    /// Drop worker/session state.
    pub fn reset(&mut self) {
        match self {
            Self::Worker(bridge) => bridge.send(RunInput::Reset),
            Self::Inline(session) => *session = None,
        }
    }
}

/// The app's own module script `src`, used by gloo-worker's blob shim to
/// load the same wasm inside the worker. `None` disables the worker path.
fn module_script_path() -> Option<String> {
    let document = web_sys::window()?.document()?;
    let scripts = document.query_selector_all("script").ok()?;
    for index in 0..scripts.length() {
        let node = scripts.item(index)?;
        let element = node.dyn_ref::<web_sys::Element>()?;
        if let Some(src) = element.get_attribute("src")
            && src.ends_with(".js")
        {
            return Some(src);
        }
    }
    None
}

/// Schedule `run` as a macrotask (setTimeout 0) so the browser can paint
/// between chunks on the inline transport.
pub fn schedule<F: FnOnce() + 'static>(run: F) {
    gloo_timers::callback::Timeout::new(0, run).forget();
}
