use base::core::GridSpacing;
use base::elements::{Aperture, ApertureKind, ThinLens};
use base::propagation::FreeSpace;
use base::sources::ContinuousWave;
use base::{CoherenceState, ComplexField, OpticalElement, Source, Spectrum};
use dfdx::prelude::*;

const NY: usize = 257;
const NX: usize = 257;

#[derive(Debug, Clone, Copy)]
struct FieldStats {
    total_intensity: f64,
    peak_intensity: f64,
    center_intensity: f64,
    illuminated_samples: usize,
}

fn summarize_field<S, D, T>(field: &ComplexField<S, f64, D, T>) -> FieldStats
where
    S: Shape<Concrete = [usize; 2]>,
    D: Storage<f64>,
    T: Tape<f64, D>,
{
    let [ny, nx] = field.shape().concrete();
    let center_index = (ny / 2) * nx + (nx / 2);
    let real = field.real().as_vec();
    let imag = field.imag().as_vec();

    let mut total_intensity = 0.0;
    let mut peak_intensity = 0.0;
    let mut center_intensity = 0.0;
    let mut illuminated_samples = 0;

    for (index, (real, imag)) in real.into_iter().zip(imag).enumerate() {
        let intensity = real * real + imag * imag;
        total_intensity += intensity;
        peak_intensity = f64::max(peak_intensity, intensity);

        if intensity > 1.0e-12 {
            illuminated_samples += 1;
        }

        if index == center_index {
            center_intensity = intensity;
        }
    }

    FieldStats {
        total_intensity,
        peak_intensity,
        center_intensity,
        illuminated_samples,
    }
}

fn print_stats(label: &str, stats: FieldStats) {
    println!(
        "{label}: total={:.3}, peak={:.3}, center={:.3}, illuminated_samples={}",
        stats.total_intensity,
        stats.peak_intensity,
        stats.center_intensity,
        stats.illuminated_samples
    );
}

fn main() {
    let device: Cpu = Default::default();
    let shape = (NY, NX);

    let wavelength = 560.0e-9;
    let focal_length = 50.0e-3;
    let aperture_diameter = 1.0e-3;
    let aperture_radius = aperture_diameter * 0.5;
    let spacing = GridSpacing::new(10.0e-6, 10.0e-6);

    let source = ContinuousWave::new(
        shape,
        spacing,
        Spectrum::Monochromatic(wavelength),
        CoherenceState::Coherent,
        1.0,
        0.0,
    );

    println!("coherent 4f circular aperture example");
    println!("source: {source}");
    println!(
        "wavelength={:.0} nm, focal_length={:.1} mm, aperture_diameter={:.1} mm, grid={}x{}, spacing={:.1} um",
        wavelength * 1.0e9,
        focal_length * 1.0e3,
        aperture_diameter * 1.0e3,
        NX,
        NY,
        spacing.dx * 1.0e6
    );

    let field = source.emit(&device);
    let input_stats = summarize_field(&field);
    print_stats("input plane", input_stats);

    let field =
        FreeSpace::paraxial_phase(&device, shape, spacing, focal_length, wavelength).apply(field);
    let field = ThinLens::new(&device, shape, spacing, focal_length, wavelength).apply(field);
    let field =
        FreeSpace::paraxial_phase(&device, shape, spacing, focal_length, wavelength).apply(field);

    let field = Aperture::new(
        &device,
        shape,
        spacing,
        ApertureKind::Circular {
            radius: aperture_radius,
        },
    )
    .apply(field);
    let aperture_stats = summarize_field(&field);
    print_stats("Fourier aperture plane", aperture_stats);

    let field =
        FreeSpace::paraxial_phase(&device, shape, spacing, focal_length, wavelength).apply(field);
    let field = ThinLens::new(&device, shape, spacing, focal_length, wavelength).apply(field);
    let field =
        FreeSpace::paraxial_phase(&device, shape, spacing, focal_length, wavelength).apply(field);

    let output_stats = summarize_field(&field);
    print_stats("image plane", output_stats);

    let throughput = output_stats.total_intensity / input_stats.total_intensity;
    println!("aperture throughput={:.3}%", throughput * 100.0);

    assert!(output_stats.total_intensity > 0.0);
    assert!(output_stats.total_intensity < input_stats.total_intensity);
}
