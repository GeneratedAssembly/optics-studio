//! Cross-section cuts through a [`Psf`] (docs/ux/spec.md §3.3, §4).
//!
//! Client slicing of `Psf::samples` with the `Psf::coord` mapping to
//! physical coordinates — pure display plumbing, no metric recomputation.
//! Default cut is the peak row/column (`Psf::peak_bin`); the viewport
//! swaps in the cursor row/column on hover.

use optics_core::field::Psf;

/// One line cut: physical sample coordinates (meters) and values.
#[derive(Clone, Debug, PartialEq)]
pub struct Crosscut {
    /// Axis coordinates along the cut (m).
    pub xs: Vec<f64>,
    /// Sample values along the cut.
    pub values: Vec<f64>,
    /// Human axis label with display units ("x (µm)" for detector grids).
    pub axis_label: String,
}

impl Crosscut {
    /// Number of samples in the cut.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// True when the cut has no samples.
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }
}

/// Which axis a [`row_cut`] / [`column_cut`] runs along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutAxis {
    /// Constant `iy`, varying `ix` (horizontal line across the image).
    Row,
    /// Constant `ix`, varying `iy` (vertical line down the image).
    Column,
}

/// Horizontal cut (row `iy`) through `psf`.
pub fn row_cut(psf: &Psf, iy: usize) -> Crosscut {
    line_cut(psf, CutAxis::Row, iy)
}

/// Vertical cut (column `ix`) through `psf`.
pub fn column_cut(psf: &Psf, ix: usize) -> Crosscut {
    line_cut(psf, CutAxis::Column, ix)
}

/// The (horizontal, vertical) pair through the PSF peak bin — the default
/// cross-section view (spec §4 "Peak row/column by default").
pub fn through_peak(psf: &Psf) -> (Crosscut, Crosscut) {
    let (px, py) = psf.peak_bin();
    (row_cut(psf, py), column_cut(psf, px))
}

fn line_cut(psf: &Psf, axis: CutAxis, index: usize) -> Crosscut {
    let (nx, ny) = psf.dims();
    let samples = psf.samples();
    // A row cut varies ix at fixed iy; a column cut varies iy at fixed ix.
    // The fixed index is clamped to the grid so an out-of-range cursor is
    // snapped to the nearest line rather than panicking.
    let count = match axis {
        CutAxis::Row => nx,
        CutAxis::Column => ny,
    };
    let fixed = index.min(match axis {
        CutAxis::Row => ny - 1,
        CutAxis::Column => nx - 1,
    });
    let mut xs = Vec::with_capacity(count);
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let (ix, iy) = match axis {
            CutAxis::Row => (i.min(nx - 1), fixed),
            CutAxis::Column => (fixed, i.min(ny - 1)),
        };
        let (x, y) = psf.coord(ix, iy);
        xs.push(if axis == CutAxis::Row { x } else { y });
        values.push(samples[iy * nx + ix]);
    }
    Crosscut {
        xs,
        values,
        axis_label: format!("{} ({})", axis_name(axis), display_unit(psf)),
    }
}

fn axis_name(axis: CutAxis) -> &'static str {
    match axis {
        CutAxis::Row => "x",
        CutAxis::Column => "y",
    }
}

/// Display unit for a grid pitch: microns for detector-plane PSFs (the
/// viewport's usual case), meters for coarse field grids (spec §4).
fn display_unit(psf: &Psf) -> &'static str {
    let (dx, _dy) = psf.pitch();
    if dx < 1e-2 { "µm" } else { "m" }
}

/// Convert a coordinate along a cut to display units (µm for detector
/// grids, m otherwise) — UI-layer unit conversion only (spec §3.5).
pub fn to_display_units(psf: &Psf, value_m: f64) -> f64 {
    let (dx, _dy) = psf.pitch();
    if dx < 1e-2 { value_m * 1e6 } else { value_m }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_psf() -> Psf {
        // 4x2 grid, pitch 10 µm, values = row*10 + col.
        let samples: Vec<f64> = (0..8).map(|i| i as f64).collect();
        Psf::new(samples, 4, 2, 10e-6, 10e-6).unwrap()
    }

    #[test]
    fn row_cut_matches_samples_and_coords() {
        let psf = ramp_psf();
        let cut = row_cut(&psf, 1);
        assert_eq!(cut.len(), 4);
        assert_eq!(cut.values, vec![4.0, 5.0, 6.0, 7.0]);
        // coord() is origin-centered: x = (ix - (nx-1)/2) * dx
        assert!((cut.xs[0] - (-15e-6)).abs() < 1e-9);
        assert!((cut.xs[3] - 15e-6).abs() < 1e-9);
        assert_eq!(cut.axis_label, "x (µm)");
    }

    #[test]
    fn column_cut_matches_samples() {
        let psf = ramp_psf();
        let cut = column_cut(&psf, 2);
        assert_eq!(cut.values, vec![2.0, 6.0]);
        assert_eq!(cut.axis_label, "y (µm)");
    }

    #[test]
    fn through_peak_hits_max_row_and_column() {
        let psf = ramp_psf(); // max value 7 at (ix=3, iy=1)
        let (row, col) = through_peak(&psf);
        assert_eq!(row.values, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(col.values, vec![3.0, 7.0]);
    }

    #[test]
    fn out_of_range_index_clamps() {
        let psf = ramp_psf();
        let cut = row_cut(&psf, 99);
        assert_eq!(cut.values, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn display_unit_switches_on_pitch() {
        let fine = ramp_psf();
        assert_eq!(to_display_units(&fine, 1e-6), 1.0); // µm
        let coarse_samples = vec![1.0; 4];
        let coarse = Psf::new(coarse_samples, 2, 2, 0.1, 0.1).unwrap(); // 10 cm pitch
        assert_eq!(to_display_units(&coarse, 1.0), 1.0); // m
        assert_eq!(column_cut(&coarse, 0).axis_label, "y (m)");
    }
}
