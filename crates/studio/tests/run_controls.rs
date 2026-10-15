//! Host smoke tests for the run controls (task 10.3): the serializable
//! job builds real `optics-inverse` objects, the chunked session runs to
//! budget with continuation semantics, and the controller mirrors the
//! spec §3.4 start/stop flow.

use optics_studio::run::{ChunkReport, RunController, RunJob, RunPhase, RunSession};

fn small_job() -> RunJob {
    let mut job = RunJob::demo_ee_focus();
    job.design = optics_studio::run::job::DesignSpec::GridPhaseRelay {
        dims: (8, 8),
        pitch: (20e-6, 20e-6),
        aperture: optics_studio::run::job::ApertureSpec::Circle { radius: 150e-6 },
        center: (0.0, 0.0),
        lo: -std::f64::consts::PI,
        hi: std::f64::consts::PI,
        init: 0.0,
        z_after: 2e-3,
        medium: 1.0,
    };
    job.grid = optics_studio::run::job::GridUi {
        nx: 16,
        ny: 16,
        dx: 20e-6,
        dy: 20e-6,
    };
    job.budget = 150;
    job.chunk = 75;
    job.seed = 3;
    job
}

#[test]
fn demo_preset_validates() {
    let job = RunJob::demo_ee_focus();
    job.validate().expect("demo preset is valid");
    assert_eq!(job.dim(), 16 * 16);
    // Builders succeed on the preset (no render yet).
    let carrier = job.build_carrier().expect("carrier builds");
    assert_eq!(carrier.dim(), 16 * 16);
    let _renderer = job.build_renderer(&carrier);
    let _merit = job.build_merit();
}

#[test]
fn serde_roundtrip_preserves_the_job() {
    let job = RunJob::demo_ee_focus();
    let json = serde_json::to_string(&job).unwrap();
    let back: RunJob = serde_json::from_str(&json).unwrap();
    assert_eq!(job, back);
}

#[test]
fn controller_drives_chunked_start_stop() {
    let job = small_job();
    let mut session = RunSession::start(job.clone()).expect("session starts");
    let mut controller = RunController::new();
    controller.begin(0.0);

    let mut launched = true;
    let mut chunks = 0;
    while launched {
        let report = session.run_chunk().expect("chunk runs");
        launched = controller.absorb(&report);
        chunks += 1;
    }
    assert_eq!(controller.phase(), RunPhase::Finished);
    assert_eq!(chunks, 2, "150 units / 75 per chunk = 2 chunks");
    assert_eq!(controller.total_evaluations(), 150);
    // annealing: history records each Metropolis acceptance.
    assert!(controller.history().len() >= 2);
    assert!(controller.value().is_finite());

    // Apply result feeds the continuation start (spec §3.4).
    let mut continued = job.clone();
    controller.apply_best_to(&mut continued);
    assert_eq!(continued.start_values.as_deref(), Some(controller.best()));
}

#[test]
fn stop_prevents_the_next_chunk() {
    let mut job = small_job();
    job.budget = 300; // stop must truncate well before the budget ends
    let mut session = RunSession::start(job).expect("session starts");
    let mut controller = RunController::new();
    controller.begin(0.0);

    let report = session.run_chunk().unwrap();
    assert!(controller.absorb(&report));
    controller.request_stop();
    let report = session.run_chunk().unwrap();
    assert!(!controller.absorb(&report), "stopped: no next launch");
    assert_eq!(controller.phase(), RunPhase::Idle);
    assert_eq!(controller.chunks_done(), 2);
}

#[test]
fn validation_errors_surface_before_solve() {
    let mut job = small_job();
    job.chunk = 1;
    let mut controller = RunController::new();
    controller.begin(0.0);
    match RunSession::start(job) {
        Ok(_) => panic!("chunk below the 50 floor must be rejected"),
        Err(e) => controller.fail(e),
    }
    assert_eq!(controller.phase(), RunPhase::Failed);
    assert!(controller.error().unwrap().contains("chunk"));
}

#[test]
fn reports_are_serializable_for_the_worker() {
    let report = ChunkReport {
        chunk_index: 0,
        history: vec![1.0, 0.5],
        best: vec![0.25],
        value: 0.5,
        chunk_evaluations: 2,
        total_evaluations: 2,
        finished: false,
    };
    let bytes = bincode::serialize(&report).unwrap();
    let back: ChunkReport = bincode::deserialize(&bytes).unwrap();
    assert_eq!(report, back);
}

#[test]
fn captured_merit_validates_against_the_observation_grid() {
    // Task 10.4 round-trip seam: the viewport capture must match the
    // job's observation grid, else it is a typed error.
    let mut job = RunJob::demo_ee_focus();
    let expected = job.grid.nx * job.grid.ny;
    let err = job
        .set_captured_merit(
            optics_studio::run::job::CaptureKind::MseToTarget,
            vec![1.0; 4],
        )
        .expect_err("wrong length must fail");
    assert!(
        err.contains(&expected.to_string()),
        "error names the expected count: {err}"
    );

    let samples: Vec<f64> = (0..expected).map(|i| (i as f64) * 0.125).collect();
    job.set_captured_merit(
        optics_studio::run::job::CaptureKind::MseToTarget,
        samples.clone(),
    )
    .expect("grid-sized capture accepted");
    assert_eq!(job.merit.label(), "mse_to_target");
    assert_eq!(
        job.merit_selector().channel,
        optics_studio::run::job::ChannelSpec::Intensity
    );
    let bytes = bincode::serialize(&job).unwrap();
    let back: RunJob = bincode::deserialize(&bytes).unwrap();
    assert_eq!(job, back, "captured merit survives the worker round-trip");

    let mut bad = samples.clone();
    bad[3] = f64::NAN;
    job.set_captured_merit(optics_studio::run::job::CaptureKind::Strehl, samples)
        .expect("strehl capture accepted");
    assert_eq!(job.merit.label(), "neg_strehl");

    let err = job
        .set_captured_merit(optics_studio::run::job::CaptureKind::Strehl, bad)
        .expect_err("non-finite samples must fail");
    assert!(err.contains("non-finite"));
}
