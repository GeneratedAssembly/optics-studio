//! SVG line-chart paths for the EE, MTF, cross-section, and convergence
//! plots (docs/ux/spec.md §4 "Line charts").
//!
//! Sampled arrays are small (`EncircledEnergy::curve`, `Mtf.freqs/values`,
//! [`crate::viewmodel::crosscut::Crosscut`]), so the spec calls for
//! polylines with no resampling. This module turns `(xs, ys)` into an SVG
//! path string in a fixed pixel box, with axis ticks the panel can label.
//! Pure geometry — no metric math.

/// A chart plot box in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlotBox {
    /// Box width (px).
    pub width: f64,
    /// Box height (px).
    pub height: f64,
}

impl PlotBox {
    /// A box with a small margin for axis labels.
    pub const fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}

/// An axis range resolved for a chart, with optional log transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    /// Data value at the plot-box origin.
    pub min: f64,
    /// Data value at the far plot-box edge.
    pub max: f64,
    /// Log-scale this axis (MTF x, convergence y — spec §4).
    pub log: bool,
}

impl Axis {
    /// A linear axis spanning the data range with a small pad.
    pub fn linear(min: f64, max: f64) -> Self {
        let (lo, hi) = ordered(min, max);
        let pad = ((hi - lo) * 0.02).max(1e-12);
        Self {
            min: lo - pad,
            max: hi + pad,
            log: false,
        }
    }

    /// A log axis over positive data; non-positive values are dropped by
    /// [`Chart::from_points`] before mapping.
    pub fn log10(min: f64, max: f64) -> Self {
        let lo = min.max(f64::MIN_POSITIVE);
        let hi = max.max(lo * 10.0);
        Self {
            min: lo.log10(),
            max: hi.log10(),
            log: true,
        }
    }

    /// Maps a data value to normalized $[0, 1]$ within the axis.
    pub fn norm(&self, v: f64) -> f64 {
        let x = if self.log { v.log10() } else { v };
        let span = (self.max - self.min).max(f64::EPSILON);
        ((x - self.min) / span).clamp(0.0, 1.0)
    }
}

fn ordered(a: f64, b: f64) -> (f64, f64) {
    if a <= b { (a, b) } else { (b, a) }
}

/// A resolved polyline chart: SVG path data plus the pixel-space sample
/// points (for hover readouts, which linearly interpolate for display only).
#[derive(Clone, Debug, PartialEq)]
pub struct Chart {
    /// SVG path `d` attribute (a `M ... L ...` polyline).
    pub path: String,
    /// Plot box the path was drawn into.
    pub box_: PlotBox,
    /// X axis range.
    pub x_axis: Axis,
    /// Y axis range.
    pub y_axis: Axis,
    /// Number of plotted (finite, in-domain) points.
    pub points: usize,
}

impl Chart {
    /// Builds a polyline chart from paired data arrays.
    ///
    /// Samples whose `x` or `y` is non-finite are skipped (never reach the
    /// SVG). A `y` axis of $[0, 1]$ (fractions) is typical for EE/MTF; the
    /// caller chooses the axes. `flip_y` maps data-up to screen-down so the
    /// path renders right-side-up in SVG coordinates.
    pub fn from_points(xs: &[f64], ys: &[f64], x_axis: Axis, y_axis: Axis, box_: PlotBox) -> Self {
        let mut seg = String::new();
        let mut n = 0;
        let mut started = false;
        for (i, &x) in xs.iter().enumerate() {
            let Some(&y) = ys.get(i) else { break };
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            if (x_axis.log && x <= 0.0) || (y_axis.log && y <= 0.0) {
                continue;
            }
            let px = x_axis.norm(x) * box_.width;
            let py = (1.0 - y_axis.norm(y)) * box_.height; // flip_y
            seg.push_str(if started { " L " } else { "M " });
            seg.push_str(&format!("{px:.2},{py:.2}"));
            started = true;
            n += 1;
        }
        Chart {
            path: seg,
            box_,
            x_axis,
            y_axis,
            points: n,
        }
    }

    /// An EE chart: radius (m) on x, fraction $[0, 1]$ on y.
    pub fn encircled_energy(radii: &[f64], fractions: &[f64], box_: PlotBox) -> Self {
        let x = Axis::linear(
            radii.first().copied().unwrap_or(0.0),
            radii.last().copied().unwrap_or(1.0),
        );
        Chart::from_points(
            radii,
            fractions,
            x,
            Axis {
                min: 0.0,
                max: 1.0,
                log: false,
            },
            box_,
        )
    }

    /// An MTF chart: frequency (cycles/m) on x (optionally log), modulation
    /// $[0, 1]$ on y.
    pub fn mtf(freqs: &[f64], values: &[f64], box_: PlotBox, log_x: bool) -> Self {
        let fmin = freqs
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, b| if b > 0.0 && b < a { b } else { a });
        let fmax = freqs.last().copied().unwrap_or(1.0);
        let x = if log_x {
            Axis::log10(fmin, fmax)
        } else {
            Axis::linear(0.0, fmax)
        };
        Chart::from_points(
            freqs,
            values,
            x,
            Axis {
                min: 0.0,
                max: 1.0,
                log: false,
            },
            box_,
        )
    }

    /// Maps a pixel x back to normalized $[0, 1]$ for a hover readout
    /// (display interpolation only — matches `value_at` semantics).
    pub fn x_norm_at(&self, px: f64) -> f64 {
        (px / self.box_.width).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_linear_and_log_norm() {
        let a = Axis::linear(0.0, 10.0);
        assert!((a.norm(5.0) - 0.5).abs() < 0.02);
        let l = Axis::log10(1.0, 1000.0); // min=0, max=3 (log10)
        assert!((l.norm(10.0) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn chart_path_skips_nonfinite_and_counts() {
        let xs = [0.0, 1.0, f64::NAN, 3.0];
        let ys = [0.0, 0.5, 0.9, 1.0];
        let c = Chart::from_points(
            &xs,
            &ys,
            Axis::linear(0.0, 3.0),
            Axis::linear(0.0, 1.0),
            PlotBox::new(100.0, 50.0),
        );
        assert_eq!(c.points, 3);
        assert!(c.path.starts_with("M "));
        assert_eq!(c.path.matches('M').count(), 1);
        assert_eq!(c.path.matches('L').count(), 2);
    }

    #[test]
    fn ee_chart_y_axis_is_unit_fraction() {
        let radii = [0.0, 1.0, 2.0];
        let fr = [0.0, 0.5, 1.0];
        let c = Chart::encircled_energy(&radii, &fr, PlotBox::new(120.0, 80.0));
        assert_eq!(c.y_axis.min, 0.0);
        assert_eq!(c.y_axis.max, 1.0);
        assert_eq!(c.points, 3);
    }

    #[test]
    fn mtf_log_drops_nonpositive_x() {
        let freqs = [0.0, 100.0, 1000.0];
        let vals = [1.0, 0.5, 0.1];
        let c = Chart::mtf(&freqs, &vals, PlotBox::new(100.0, 100.0), true);
        // The f=0 sample is dropped on the log axis.
        assert_eq!(c.points, 2);
    }

    #[test]
    fn hover_norm_clamps() {
        let c = Chart::encircled_energy(&[0.0, 1.0], &[0.0, 1.0], PlotBox::new(100.0, 100.0));
        assert_eq!(c.x_norm_at(50.0), 0.5);
        assert_eq!(c.x_norm_at(-10.0), 0.0);
        assert_eq!(c.x_norm_at(1e6), 1.0);
    }
}
