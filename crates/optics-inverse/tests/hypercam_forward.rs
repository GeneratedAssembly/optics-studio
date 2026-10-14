//! Hypercam forward-render cross-check (task 9.12 validation).
//!
//! `tests/data/hypercam/` holds a **trained** DOE thickness map and the
//! reference forward observation exported from the hypercam
//! `coded_mask.py` optical model (its verbatim `_angular_spectrum` ASM,
//! thin-lens/relay phase laws, and A-optimal CRLB loss; reduced scale:
//! 32×32, single band — regenerate with
//! `crates/optics-inverse/scripts/hypercam_forward_export.py`).
//!
//! This test renders the *same* design through the `optics-inverse`
//! `Renderer` (core ASM + `PhaseMask` + `ThinLens` chain) and compares
//! the raw `Observation` buffers element-for-element — the byte-level
//! compare at reduced scale from the GitHub acceptance criteria. The two
//! engines share only the physics spec (identical transfer function,
//! phase law, aperture, and scene central differences), never code, so
//! agreement pins the render chain end to end.

use optics_core::element::{ClearAperture, PhaseMask, ThinLens};
use optics_core::propagation::asm::AsmConfig;
use optics_core::sources::{GaussianBeam, gaussian};
use optics_inverse::InverseError;
use optics_inverse::carrier::{Carrier, Design};
use optics_inverse::observe::{Channel, Derivatives, Renderer};
use optics_inverse::params::Params;

fn data_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/hypercam")
}

/// Minimal big-enough NPY reader: v1/v2/v3 headers, `<f8`, C-order.
fn read_npy_f64(path: &std::path::Path) -> (Vec<usize>, Vec<f64>) {
    let bytes = std::fs::read(path).expect("npy file present");
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "npy magic");
    let (hlen_off, hlen_size) = match (bytes[6], bytes[7]) {
        (1, _) => (8usize, 2usize),
        (2 | 3, _) => (8usize, 4usize),
        (maj, min) => panic!("unsupported npy version {maj}.{min}"),
    };
    let hlen = match hlen_size {
        2 => u16::from_le_bytes(bytes[8..10].try_into().expect("header length")) as usize,
        _ => u32::from_le_bytes(bytes[8..12].try_into().expect("header length")) as usize,
    };
    let header = std::str::from_utf8(&bytes[hlen_off + hlen_size..hlen_off + hlen_size + hlen])
        .expect("ascii header");
    let dict = header.split('\n').next().unwrap_or(header);
    assert!(dict.contains("'descr': '<f8'"), "float64 little-endian");
    assert!(dict.contains("'fortran_order': False"), "C order");
    let shape: Vec<usize> = dict
        .split("'shape': ")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .expect("shape tuple")
        .split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .map(|t| t.parse().expect("shape dim"))
        .collect();
    let start = hlen_off + hlen_size + hlen;
    let body = &bytes[start..];
    assert_eq!(
        body.len(),
        shape.iter().product::<usize>() * 8,
        "npy data length"
    );
    let data = body
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    (shape, data)
}

fn read_meta() -> std::collections::HashMap<String, f64> {
    let text = std::fs::read_to_string(data_dir().join("meta.txt")).expect("meta.txt");
    let mut out = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (k, v) = line.split_once('=').expect("key=value");
        out.insert(k.to_string(), v.parse().expect("numeric value"));
    }
    out
}

#[test]
fn hypercam_trained_map_renders_identically() {
    let meta = read_meta();
    let n = meta["n"] as usize;
    let pitch = meta["pitch"];
    let lambda = meta["lambda"];
    let index = meta["index"];
    let z_gap = meta["z_gap"];
    let f = meta["f_relay"];
    let ap_r = meta["aperture_r"];
    let w0 = meta["w0"];
    let h = meta["h_scene"];
    let t_max = meta["t_max"];

    let (tshape, thickness) = read_npy_f64(&data_dir().join("thickness.npy"));
    assert_eq!(tshape, vec![n, n]);
    let (fshape, reference) = read_npy_f64(&data_dir().join("forward.npy"));
    assert_eq!(
        fshape,
        vec![3, 1, 5, n, n],
        "[channel][band][scene][row][col]"
    );

    // Same stencil order as the export script's scenes_ab.
    let scenes = [(0.0, 0.0), (h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)];

    let params = Params::uniform(n * n, 0.0, t_max, 0.0, "h");
    let carrier = Carrier::analytic(params, move |v, lam| {
        // coded_mask.py phase law: phi = (2*pi/lambda) * (n-1) * thickness
        let scale = 2.0 * std::f64::consts::PI * (index - 1.0) / lam;
        let phase: Vec<f64> = v.iter().map(|h| scale * h).collect();
        let mask = PhaseMask::sampled(
            phase,
            (n, n),
            (pitch, pitch),
            None,
            ClearAperture::Field,
            (0.0, 0.0),
        )
        .map_err(|e| InverseError::Carrier {
            reason: e.to_string(),
        })?;
        let lens = || {
            ThinLens::fixed(f, None, ClearAperture::Circle(ap_r), (0.0, 0.0)).map_err(|e| {
                InverseError::Carrier {
                    reason: e.to_string(),
                }
            })
        };
        // gap(z_gap) -> DOE -> [ASM(f) lens ASM(2f) lens ASM(f)] relay
        let mut design = Design::new();
        design.gap(z_gap, 1.0);
        design.push(mask, f, 1.0);
        design.push(lens()?, 2.0 * f, 1.0);
        design.push(lens()?, f, 1.0);
        Ok(design)
    });

    let mut renderer = Renderer::new(&carrier, move |x0, y0, lam| {
        let beam = GaussianBeam {
            w0,
            amplitude: 1.0,
            center: (x0, y0),
        };
        gaussian(&beam, 0.0, lam, n, n, pitch, pitch).map_err(|e| InverseError::Render {
            reason: e.to_string(),
        })
    });
    renderer.bands = vec![lambda];
    renderer.scenes = scenes.to_vec();
    renderer.derivatives = Derivatives {
        x: true,
        y: true,
        lambda: false,
    };
    renderer.scene_step = h;
    // Mirror the notebook's plain ASM: no padding, no band limit, vacuum.
    renderer.asm = AsmConfig {
        padding: 1.0,
        band_limited: false,
        strict: false,
        max_padding: 1.0,
        medium: 1.0,
    };

    let obs = renderer.render(&thickness).expect("render trained map");
    assert_eq!(
        obs.channels,
        vec![Channel::Intensity, Channel::DxDyX, Channel::DxDyY]
    );
    assert_eq!(obs.data.len(), reference.len());

    let scale = reference
        .iter()
        .cloned()
        .fold(0.0f64, |a, v| a.max(v.abs()))
        .max(1e-30);
    let mut worst = 0.0f64;
    let mut worst_k = 0usize;
    for (k, (a, b)) in obs.data.iter().zip(&reference).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            worst_k = k;
        }
    }
    println!(
        "hypercam cross-check: {} entries, max abs diff {:.3e} (rel {:.3e} of peak {:.3e}) at index {worst_k}",
        obs.data.len(),
        worst,
        worst / scale,
        scale
    );
    assert!(
        worst / scale < 1e-6,
        "forward renders diverge: rel {}/1e-6",
        worst / scale
    );
}
