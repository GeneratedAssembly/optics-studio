//! Intensity grid → RGBA8 buffer (docs/ux/spec.md §4).
//!
//! The display pipeline for the PSF image, in strict order:
//!
//! 1. **Assert finiteness** — every core type rejects non-finite samples at
//!    construction, but the UI re-checks before mapping and reports
//!    [`IntensityError::NonFinite`] rather than painting garbage (spec §4).
//! 2. **Downsample** — box-average to at most [`MAX_DISPLAY_EDGE`] samples
//!    per axis (analysis always runs on the full-resolution `Psf`, never
//!    this copy).
//! 3. **Normalize** — autoscale per frame to the *current view* min/max, or
//!    a [`ScaleLock`] frozen across frames so two designs compare honestly;
//!    the locked view must show `Psf::total_power` beside it (the panel
//!    does).
//! 4. **Log** (optional) — $I_{\text{display}} = \log_{10}\!\big(
//!    \max(I, \varepsilon) / \max(I) \big)$ with $\varepsilon$ a per-frame
//!    floor at `log_floor_ratio` of the peak (default 1e-6), which is what
//!    makes PSF wings and zero-order leakage pedestals visible.
//! 5. **Colormap** — [`Colormap::rgba`] on the resulting $[0, 1]$ parameter.

use crate::viewmodel::colormap::Colormap;

/// Hard display budget: never render more samples than this per axis
/// (docs/ux/spec.md §4 "box-average the grid to ≤2048²").
pub const MAX_DISPLAY_EDGE: usize = 2048;

/// Failures of the display mapping.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IntensityError {
    /// A sample was not finite — the canvas must not paint it.
    #[error("non-finite sample in {0} grid — refusing to paint (samples {1})")]
    NonFinite(String, usize),
    /// The declared grid size did not match the sample buffer.
    #[error("grid {0}x{1} does not match sample count {2}")]
    DimensionMismatch(usize, usize, usize),
}

/// A frozen $[\text{min}, \text{max}]$ normalization, shared between frames
/// for honest comparison (spec §4 "lock scale").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleLock {
    /// Lower bound mapped to $t = 0$.
    pub lo: f64,
    /// Upper bound mapped to $t = 1$.
    pub hi: f64,
}

/// Per-frame autoscale bounds of a sample buffer.
///
/// Returns `(min, max)`; an empty buffer yields `(0.0, 0.0)`.
pub fn autoscale(samples: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in samples {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if lo > hi { (0.0, 0.0) } else { (lo, hi) }
}

/// Box-average downsample of a row-major `nx`×`ny` grid to
/// `out_nx`×`out_ny` (each output cell is the mean of its source box).
///
/// An exact box partition: source dimension `n` maps to `out` boxes of
/// width `n / out` (integer division with the remainder spread over the
/// first boxes), so `out_n == n` is the identity and `out_n > n` fails
/// with [`IntensityError::DimensionMismatch`].
pub fn downsample_box(
    samples: &[f64],
    nx: usize,
    ny: usize,
    out_nx: usize,
    out_ny: usize,
) -> Result<Vec<f64>, IntensityError> {
    if samples.len() != nx * ny {
        return Err(IntensityError::DimensionMismatch(nx, ny, samples.len()));
    }
    if out_nx == 0 || out_ny == 0 {
        return Err(IntensityError::DimensionMismatch(nx, ny, 0));
    }
    if out_nx > nx || out_ny > ny {
        return Err(IntensityError::DimensionMismatch(nx, ny, out_nx * out_ny));
    }
    if out_nx == nx && out_ny == ny {
        return Ok(samples.to_vec());
    }
    let mut out = Vec::with_capacity(out_nx * out_ny);
    for oy in 0..out_ny {
        let y0 = oy * ny / out_ny;
        let y1 = ((oy + 1) * ny / out_ny).max(y0 + 1);
        for ox in 0..out_nx {
            let x0 = ox * nx / out_nx;
            let x1 = ((ox + 1) * nx / out_nx).max(x0 + 1);
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for sy in y0..y1 {
                let row = &samples[sy * nx + x0..sy * nx + x1];
                for &v in row {
                    sum += v;
                    cnt += 1;
                }
            }
            out.push(sum / cnt as f64);
        }
    }
    Ok(out)
}

/// The box factor a `nx`×`ny` grid is downsampled by for display
/// ($1$ = full resolution).
pub fn display_factor(nx: usize, ny: usize, budget: usize) -> usize {
    let longest = nx.max(ny);
    if budget == 0 || longest <= budget {
        1
    } else {
        longest.div_ceil(budget)
    }
}

/// Display-ready frame: downsampled, normalized, colormapped RGBA bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderedImage {
    /// Image width in pixels (downsampled `nx`).
    pub width: usize,
    /// Image height in pixels (downsampled `ny`).
    pub height: usize,
    /// Row-major RGBA8 bytes, length `width * height * 4`.
    pub rgba: Vec<u8>,
    /// Box-average factor applied (`1` = none); panels show a
    /// "display downsampled ×k" note when `> 1` (spec §4).
    pub downsample_factor: usize,
    /// The `[lo, hi]` actually used for this frame.
    pub scale: ScaleLock,
}

/// Options for [`render_intensity`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderOptions {
    /// Colormap family.
    pub colormap: Colormap,
    /// Log display toggle (spec §4 "Log scale").
    pub log: bool,
    /// Per-frame log floor as a fraction of the peak (default `1e-6`).
    pub log_floor_ratio: f64,
    /// Frozen normalization; `None` = autoscale per frame.
    pub lock: Option<ScaleLock>,
    /// Display budget per axis ([`MAX_DISPLAY_EDGE`] by default).
    pub budget: usize,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            colormap: Colormap::Viridis,
            log: false,
            log_floor_ratio: 1e-6,
            lock: None,
            budget: MAX_DISPLAY_EDGE,
        }
    }
}

/// Maps a non-negative intensity grid to an [`RenderedImage`].
///
/// `samples` is row-major `nx`×`ny` (e.g. `Psf::samples`). The function is
/// generic over the value semantics: intensity grids use a sequential map
/// ([`crate::viewmodel::colormap::Colormap::Viridis`]); signed grids
/// (residuals, derivative channels) pass a diverging map with `log = false`
/// and a symmetric lock `[-a, a]` centered at 0.
pub fn render_intensity(
    samples: &[f64],
    nx: usize,
    ny: usize,
    opts: &RenderOptions,
) -> Result<RenderedImage, IntensityError> {
    if samples.len() != nx * ny {
        return Err(IntensityError::DimensionMismatch(nx, ny, samples.len()));
    }
    if let Some(i) = samples.iter().position(|v| !v.is_finite()) {
        return Err(IntensityError::NonFinite("intensity".into(), i));
    }
    let factor = display_factor(nx, ny, opts.budget);
    let (onx, ony) = (nx / factor, ny / factor);
    let (onx, ony) = (onx.max(1), ony.max(1));
    let data = if factor > 1 {
        downsample_box(samples, nx, ny, onx, ony)?
    } else {
        samples.to_vec()
    };

    let scale = match opts.lock {
        Some(s) => s,
        None => {
            let (lo, hi) = autoscale(&data);
            ScaleLock { lo, hi }
        }
    };

    let mut rgba = vec![0u8; onx * ony * 4];
    if opts.log {
        // Log domain: t = log10(max(v, floor) / max), floor = ratio * peak.
        // Peak comes from the *current view* so the wings stay visible
        // under either autoscale or lock.
        let peak = scale.hi.max(f64::MIN_POSITIVE);
        let floor = (opts.log_floor_ratio.max(f64::MIN_POSITIVE) * peak).max(f64::MIN_POSITIVE);
        let lo_log = (floor.max(scale.lo).max(f64::MIN_POSITIVE)).log10();
        let hi_log = peak.max(floor).max(f64::MIN_POSITIVE).log10();
        let span = (hi_log - lo_log).max(f64::EPSILON);
        for (i, &v) in data.iter().enumerate() {
            let lv = v.max(floor).max(f64::MIN_POSITIVE).log10();
            let t = (lv - lo_log) / span;
            write_pixel(&mut rgba, i, opts.colormap.rgba(t));
        }
    } else {
        let span = (scale.hi - scale.lo).max(f64::EPSILON);
        for (i, &v) in data.iter().enumerate() {
            let t = (v - scale.lo) / span;
            write_pixel(&mut rgba, i, opts.colormap.rgba(t));
        }
    }

    Ok(RenderedImage {
        width: onx,
        height: ony,
        rgba,
        downsample_factor: factor,
        scale,
    })
}

fn write_pixel(rgba: &mut [u8], i: usize, px: [u8; 4]) {
    let o = i * 4;
    rgba[o..o + 4].copy_from_slice(&px);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize) -> Vec<f64> {
        (0..nx * ny).map(|i| i as f64).collect()
    }

    #[test]
    fn autoscale_matches_bounds_and_empty() {
        assert_eq!(autoscale(&[3.0, 1.0, 2.0]), (1.0, 3.0));
        assert_eq!(autoscale(&[]), (0.0, 0.0));
    }

    #[test]
    fn downsample_identity_and_halving() {
        let g = grid(4, 2); // values 0..8
        let id = downsample_box(&g, 4, 2, 4, 2).unwrap();
        assert_eq!(id, g);
        // 4x2 -> 2x1: box means of the 2x2 cells: (0+1+4+5)/4=2.5, (2+3+6+7)/4=4.5
        let half = downsample_box(&g, 4, 2, 2, 1).unwrap();
        assert_eq!(half, vec![2.5, 4.5]);
    }

    #[test]
    fn downsample_preserves_mean() {
        // Box averaging preserves the global mean (energy proxy).
        let g: Vec<f64> = (0..64).map(|i| (i as f64).sin().abs() + 0.1).collect();
        let d = downsample_box(&g, 8, 8, 4, 4).unwrap();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!((mean(&g) - mean(&d)).abs() < 1e-12);
    }

    #[test]
    fn downsample_rejects_upsample_and_mismatch() {
        let g = grid(2, 2);
        assert_eq!(
            downsample_box(&g, 2, 2, 3, 3),
            Err(IntensityError::DimensionMismatch(2, 2, 9))
        );
        assert_eq!(downsample_box(&g, 2, 2, 2, 2).map(|v| v.len()), Ok(4));
        assert!(downsample_box(&g, 2, 3, 1, 1).is_err());
    }

    #[test]
    fn display_factor_budgeting() {
        assert_eq!(display_factor(512, 512, MAX_DISPLAY_EDGE), 1);
        assert_eq!(display_factor(2049, 100, MAX_DISPLAY_EDGE), 2);
        assert_eq!(display_factor(4096, 4096, MAX_DISPLAY_EDGE), 2);
        assert_eq!(display_factor(4097, 1, MAX_DISPLAY_EDGE), 3);
    }

    #[test]
    fn render_rejects_nonfinite() {
        let mut g = vec![1.0; 4];
        g[2] = f64::NAN;
        let e = render_intensity(&g, 2, 2, &RenderOptions::default()).unwrap_err();
        assert!(matches!(e, IntensityError::NonFinite(_, 2)));
    }

    #[test]
    fn render_linear_endpoints_map_to_colormap_ends() {
        let g = vec![0.0, 5.0, 10.0];
        let img = render_intensity(&g, 3, 1, &RenderOptions::default()).unwrap();
        assert_eq!(img.downsample_factor, 1);
        assert_eq!(img.scale, ScaleLock { lo: 0.0, hi: 10.0 });
        assert_eq!(
            &img.rgba[0..4],
            &Colormap::Viridis.rgba(0.0),
            "min pixel at t=0"
        );
        assert_eq!(
            &img.rgba[8..12],
            &Colormap::Viridis.rgba(1.0),
            "max pixel at t=1"
        );
    }

    #[test]
    fn render_lock_uses_frozen_scale() {
        let g = vec![100.0, 101.0];
        let opts = RenderOptions {
            lock: Some(ScaleLock { lo: 0.0, hi: 200.0 }),
            ..Default::default()
        };
        let img = render_intensity(&g, 2, 1, &opts).unwrap();
        assert_eq!(img.scale, ScaleLock { lo: 0.0, hi: 200.0 });
        // 100/200 -> t = 0.5
        assert_eq!(&img.rgba[0..4], &Colormap::Viridis.rgba(0.5));
    }

    #[test]
    fn render_log_maps_peak_to_one_and_floor_to_zero() {
        let peak = 1e6;
        let g = vec![0.0, peak];
        let opts = RenderOptions {
            log: true,
            log_floor_ratio: 1e-6,
            ..Default::default()
        };
        let img = render_intensity(&g, 2, 1, &opts).unwrap();
        // Peak -> t = 1; the 0 sample sits at the floor -> t = 0.
        assert_eq!(&img.rgba[4..8], &Colormap::Viridis.rgba(1.0));
        assert_eq!(&img.rgba[0..4], &Colormap::Viridis.rgba(0.0));
    }

    #[test]
    fn render_downsamples_big_grid() {
        let g = vec![1.0; 3000 * 10];
        let img = render_intensity(&g, 3000, 10, &RenderOptions::default()).unwrap();
        assert_eq!(img.downsample_factor, 2);
        assert_eq!((img.width, img.height), (1500, 5));
        assert_eq!(img.rgba.len(), 1500 * 5 * 4);
    }
}
