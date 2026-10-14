//! Host smoke run of the task-10.3 chunked loop: drives the demo job
//! (`RunJob::demo_ee_focus`) exactly as the browser console does — validate,
//! chunk, absorb, continue — and prints the convergence summary.
//!
//! Run: `cargo run -p optics-studio --example demo_run`

use optics_studio::run::{RunController, RunJob, RunSession};

fn main() {
    let job = RunJob::demo_ee_focus();
    job.validate().expect("demo preset validates");
    let mut session = RunSession::start(job).expect("session starts");
    let mut controller = RunController::new();
    controller.begin(0.0);

    let t0 = std::time::Instant::now();
    let mut launch = true;
    while launch {
        let report = session.run_chunk().expect("chunk runs");
        launch = controller.absorb(&report);
    }
    println!(
        "phase {:?}: {} chunks, {} evaluations, best merit {:.4} in {:.1}s",
        controller.phase(),
        controller.chunks_done(),
        controller.total_evaluations(),
        controller.value(),
        t0.elapsed().as_secs_f64(),
    );
    println!(
        "trace: first {:.4} -> last {:.4} ({} points)",
        controller.history().first().copied().unwrap_or(f64::NAN),
        controller.history().last().copied().unwrap_or(f64::NAN),
        controller.history().len(),
    );
}
