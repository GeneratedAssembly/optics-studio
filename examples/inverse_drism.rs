use optics_core::beamline;
use optics_core::element::ThinLens;
use optics_core::field::{BandField, Field};
use optics_core::propagation::asm::{AsmConfig, propagate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let band = BandField::uniform(amplitude, phase, nx, ny, dx, dy, lambda);
    todo!();
}
