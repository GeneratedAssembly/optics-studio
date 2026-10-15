//! Task-10.1 shell view-model tests (native; no wasm, no DOM).

use optics_core::field::BandField;
use optics_core::propagation::asm::{self, AsmConfig};
use optics_studio::shell::carrier::demo_carrier;
use optics_studio::shell::dispatch::{self, DispatchError, EnginePick, EngineSettings};
use optics_studio::shell::prescription::{TableRow, build_plan, image_distance_mm};
use optics_studio::shell::preview::render_psf;
use optics_studio::shell::source::SourceSpec;
use optics_zemax::{ImportedSystem, Registry};

const ACA254: &str = include_str!("../../../zmx-samples/ACA254-100-UV-Zemax-ZMX.zmx");

fn imported() -> ImportedSystem {
    ImportedSystem::parse(ACA254).expect("sample parses")
}

fn band() -> BandField {
    BandField::uniform(1.0, 0.0, 32, 32, 2e-3, 2e-3, 500e-9).expect("band")
}

#[test]
fn plan_mirrors_the_zemax_builder_bit_for_bit() {
    // The UI's table-mode plan must reproduce the optics-zemax plan:
    // identical PSF for the same (un-edited) prescription.
    let sys = imported();
    let registry = Registry::builtin();
    // Same wavelength as the traced band: the builder resolves glasses at
    // `band.lambda()`, so the mirror test must use the identical value.
    let b = band();
    let lambda_um = b.lambda() * 1e6;
    let plan = build_plan(&sys.system().surfaces, &registry, lambda_um).unwrap();
    assert_eq!(plan.element_count(), 4, "ACA254 has four curved surfaces");

    let settings = EngineSettings {
        pick: EnginePick::Asm,
        ..EngineSettings::default()
    };
    let via_plan = plan.psf_with(&b, &settings).expect("plan psf");
    let via_import = sys
        .psf(&b, |band, z, m| {
            asm::propagate(
                band,
                z,
                &AsmConfig {
                    medium: m,
                    ..AsmConfig::default()
                },
            )
        })
        .expect("imported psf");
    assert_eq!(via_plan.dims(), via_import.dims());
    for (a, b) in via_plan.samples().iter().zip(via_import.samples()) {
        assert_eq!(a, b, "plan/builder PSF parity is the mapping contract");
    }
}

#[test]
fn strip_model_tracks_and_nodes() {
    let sys = imported();
    let registry = Registry::builtin();
    let plan = build_plan(&sys.system().surfaces, &registry, 0.325).unwrap();
    let strip = plan.strip();
    // Surfaces 1..=4 each contribute a Surface node + a Gap node.
    assert_eq!(strip.nodes.len(), 8);
    assert!(
        strip.track_m > 0.01 && strip.track_m < 0.2,
        "track {} m",
        strip.track_m
    );
    assert!(strip.max_semi_mm > 10.0, "semi-diam in mm expected");
    assert_eq!(strip.stop_index, Some(1), "SURF 1 carries STOP");
    assert!(image_distance_mm(&sys.system().surfaces).unwrap() > 0.0);
}

#[test]
fn edited_curvature_changes_the_psf() {
    // Live-preview contract: a table-mode edit must change the trace.
    let sys = imported();
    let registry = Registry::builtin();
    let mut surfaces = sys.system().surfaces.clone();
    surfaces[1].curvature_mm *= 0.9;
    let settings = EngineSettings::default();
    let b = band();
    let base = build_plan(&sys.system().surfaces, &registry, 0.325)
        .unwrap()
        .psf_with(&b, &settings)
        .unwrap();
    let edited = build_plan(&surfaces, &registry, 0.325)
        .unwrap()
        .psf_with(&b, &settings)
        .unwrap();
    let diff: f64 = base
        .samples()
        .iter()
        .zip(edited.samples())
        .map(|(x, y)| (x - y).abs())
        .sum();
    assert!(diff > 0.0, "10% curvature edit must alter the PSF");
}

#[test]
fn table_rows_format_units() {
    let sys = imported();
    let row = TableRow::of(&sys.system().surfaces[1]);
    assert_eq!(row.index, 1);
    assert!(row.stop);
    assert_eq!(row.glass, "CAF2");
    assert!(row.curvature.starts_with("0.03297"), "{}", row.curvature);
    // Object surface: no glass, INF thickness renders as `INF`.
    let obj = TableRow::of(&sys.system().surfaces[0]);
    assert_eq!(obj.glass, "—");
    assert_eq!(obj.thickness, "INF");
}

#[test]
fn all_source_kinds_build_bands() {
    let lambda = 550e-9;
    let point = SourceSpec::Point {
        distance_m: 1.0,
        amplitude: 1.0,
        nx: 16,
        ny: 16,
        dx: 1e-3,
        dy: 1e-3,
    };
    let plane = SourceSpec::PlaneWave {
        theta_x_deg: 0.5,
        theta_y_deg: -0.25,
        amplitude: 1.0,
        nx: 16,
        ny: 16,
        dx: 1e-3,
        dy: 1e-3,
    };
    let gauss = SourceSpec::Gaussian {
        w0: 1e-3,
        amplitude: 1.0,
        z_m: 0.0,
        nx: 16,
        ny: 16,
        dx: 1e-3,
        dy: 1e-3,
    };
    for spec in [point, plane, gauss] {
        let b = spec.make_band(lambda).unwrap();
        assert_eq!(b.field().dims(), (16, 16));
        assert_eq!(b.lambda(), lambda);
        assert!(b.field().power() > 0.0);
    }
    assert_eq!(point.grid(), (16, 16, 1e-3, 1e-3));
}

#[test]
fn dispatch_routes_engines_and_guard_media() {
    // Fine grid: every engine can run clean at z = 0.5 m.
    let b = BandField::uniform(1.0, 0.0, 64, 64, 20e-6, 20e-6, 500e-9).unwrap();
    // ASM models the medium.
    let asm_settings = EngineSettings::default();
    assert!(dispatch::propagate(&asm_settings, &b, 0.5, 1.5).is_ok());
    // Fresnel transfer refuses a non-vacuum step loudly (no silent drop).
    let fresnel = EngineSettings {
        pick: EnginePick::FresnelTransfer,
        ..EngineSettings::default()
    };
    let err = dispatch::propagate(&fresnel, &b, 0.5, 1.5).unwrap_err();
    assert!(
        matches!(err, DispatchError::MediumUnsupported { medium, .. } if (medium - 1.5).abs() < 1e-12)
    );
    assert!(dispatch::propagate(&fresnel, &b, 0.05, 1.0).is_ok());
    // Rayleigh-Sommerfeld runs in vacuum.
    let rs = EngineSettings {
        pick: EnginePick::RayleighSommerfeld,
        ..EngineSettings::default()
    };
    assert!(dispatch::propagate(&rs, &b, 0.5, 1.0).is_ok());
    // Fraunhofer needs its regime guard off for this tiny demo grid.
    let fraun = EngineSettings {
        pick: EnginePick::Fraunhofer,
        fraunhofer: optics_core::propagation::fraunhofer::FraunhoferConfig {
            strict: false,
            ..Default::default()
        },
        ..EngineSettings::default()
    };
    assert!(dispatch::propagate(&fraun, &b, 1.0, 1.0).is_ok());
}

#[test]
fn banner_severity_matches_reports() {
    let b = BandField::uniform(1.0, 0.0, 64, 64, 20e-6, 20e-6, 500e-9).unwrap();
    let asm_settings = EngineSettings::default();
    let green = dispatch::run_banner(&asm_settings, &b, 0.5, 1.0);
    assert_eq!(
        green.severity,
        dispatch::BannerLevel::Ok,
        "{}",
        green.headline
    );
    // Unband-limited transfer function at this z is aliasing-red.
    let nobl = EngineSettings {
        asm: optics_core::propagation::asm::AsmConfig {
            band_limited: false,
            ..Default::default()
        },
        ..EngineSettings::default()
    };
    let red = dispatch::run_banner(&nobl, &b, 0.5, 1.0);
    assert_eq!(
        red.severity,
        dispatch::BannerLevel::Rejected,
        "{}",
        red.headline
    );
    assert!(red.detail.is_some(), "full report rides the banner");
    assert!(
        !red.remedies.is_empty(),
        "remedies must surface: {:?}",
        red.remedies
    );
}

#[test]
fn regime_suggestion_maps_to_picker() {
    let b = band();
    let pick = dispatch::suggested(&b, 0.05, Some(4e-3));
    assert!(EnginePick::ALL.contains(&pick));
    let _ = dispatch::regime_advisory(EnginePick::Fraunhofer, &b, 0.05, Some(4e-3));
}

#[test]
fn demo_carrier_sliders_build_and_trace() {
    let carrier = demo_carrier();
    let params = carrier.params();
    assert_eq!(params.dim(), 5);
    for slot in params.slots() {
        assert!(
            slot.value >= slot.lo && slot.value <= slot.hi,
            "{}",
            slot.name
        );
    }
    let values = params.values();
    // Out-of-bounds programmatic writes are guarded (§3.2).
    let mut bad = values.clone();
    bad[3] = 1e6;
    assert!(params.validate_values(&bad).is_err());

    let design = carrier.build(&values, 550e-9).unwrap();
    assert_eq!(design.len(), 2);
    let bl = design.beamline();
    let b = SourceSpec::Point {
        distance_m: 1.0,
        amplitude: 1.0,
        nx: 32,
        ny: 32,
        dx: 1e-3,
        dy: 1e-3,
    }
    .make_band(550e-9)
    .unwrap();
    let psf = bl
        .psf(&b, |band, z, m| {
            asm::propagate(
                band,
                z,
                &AsmConfig {
                    medium: m,
                    ..AsmConfig::default()
                },
            )
        })
        .expect("demo carrier traces");
    assert!(psf.samples().iter().all(|v| v.is_finite() && *v >= 0.0));

    // Moving the aperture slider to its bounds still builds.
    let mut edge = values.clone();
    edge[4] = 2.0;
    assert!(carrier.build(&edge, 550e-9).is_ok());
}

#[test]
fn preview_mapping_autoscale_log_and_downsample() {
    // 8×8 ramp, peak at the last bin.
    let samples: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let psf = optics_core::field::Psf::new(samples, 8, 8, 10e-6, 10e-6).unwrap();

    let img = render_psf(&psf, 512, false).unwrap();
    assert_eq!((img.width, img.height), (8, 8));
    assert_eq!(img.bytes.len(), 8 * 8 * 4);
    assert_eq!(img.downsampled_by, None);
    let lum = |px: usize| -> u32 {
        let i = px * 4;
        img.bytes[i] as u32 + img.bytes[i + 1] as u32 + img.bytes[i + 2] as u32
    };
    assert!(lum(63) > lum(0), "peak bin must be brightest");
    assert!(lum(63) > 450, "viridis high end is bright, got {}", lum(63));

    // Log scale keeps wings visible: the lowest non-zero bin is brighter
    // than the linear-mapped floor bin.
    let log_img = render_psf(&psf, 512, true).unwrap();
    let log_lum = |px: usize| -> u32 {
        let i = px * 4;
        log_img.bytes[i] as u32 + log_img.bytes[i + 1] as u32 + log_img.bytes[i + 2] as u32
    };
    assert!(log_lum(1) > lum(1), "log lifts dim bins");

    // Downsampling above the canvas budget.
    let big: Vec<f64> = (0..1000 * 1000).map(|i| (i % 997) as f64).collect();
    let psf = optics_core::field::Psf::new(big, 1000, 1000, 1e-6, 1e-6).unwrap();
    let img = render_psf(&psf, 512, false).unwrap();
    assert_eq!(img.downsampled_by, Some(2));
    assert_eq!((img.width, img.height), (500, 500));

    // All-zero frame maps to the colormap floor without dividing by zero.
    let zero = optics_core::field::Psf::new(vec![0.0; 16], 4, 4, 1e-6, 1e-6).unwrap();
    assert!(render_psf(&zero, 512, true).is_ok());
}
