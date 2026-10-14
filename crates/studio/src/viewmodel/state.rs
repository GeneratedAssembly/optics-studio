//! Viewport view state — the small, mutable UI state the panels read and
//! the controls write (docs/ux/spec.md §3.3 "Interaction").
//!
//! Kept separate from the heavy `Psf` data so the dioxus signals hold a
//! cheap, `Clone` struct; frames live in a memoized resource. No metric is
//! stored here — only which frame is active, the display options, the
//! cursor position, and the shared scrubber operating point.

use crate::viewmodel::colormap::Colormap;
use crate::viewmodel::intensity::ScaleLock;

/// The metric tab shown in the panel column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelTab {
    /// Encircled-energy curve + readouts.
    EncircledEnergy,
    /// 1-D MTF along +fx.
    Mtf,
    /// Strehl ratio vs the synthetic reference.
    Strehl,
    /// RMS spot radius + centroid.
    Spot,
    /// Zernike coefficient bars + residual.
    Wavefront,
}

impl PanelTab {
    /// Tab label for the header.
    pub fn label(self) -> &'static str {
        match self {
            PanelTab::EncircledEnergy => "EE",
            PanelTab::Mtf => "MTF",
            PanelTab::Strehl => "Strehl",
            PanelTab::Spot => "Spot",
            PanelTab::Wavefront => "Wavefront",
        }
    }

    /// Left-to-right tab order.
    pub const ALL: [PanelTab; 5] = [
        PanelTab::EncircledEnergy,
        PanelTab::Mtf,
        PanelTab::Strehl,
        PanelTab::Spot,
        PanelTab::Wavefront,
    ];
}

/// Mutable viewport state (everything the user can change without a re-run).
#[derive(Clone, Debug, PartialEq)]
pub struct ViewportState {
    /// Active frame index (multi-band tabs + incoherent-sum view).
    pub active_frame: usize,
    /// Active metric tab.
    pub tab: PanelTab,
    /// Intensity colormap.
    pub colormap: Colormap,
    /// Log intensity display toggle.
    pub log: bool,
    /// Frozen normalization across frames (`None` = autoscale).
    pub lock: Option<ScaleLock>,
    /// Cursor position in *full-resolution* grid bins `(ix, iy)`, if the
    /// pointer is over the image; drives the cross-section and readout.
    pub cursor: Option<(usize, usize)>,
    /// Shared radius/frequency scrubber, normalized $[0, 1]$.
    pub scrub: f64,
    /// Log-x option on the MTF panel (spec §3.3 "log-x option").
    pub mtf_log_x: bool,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            active_frame: 0,
            tab: PanelTab::EncircledEnergy,
            colormap: Colormap::Viridis,
            log: false,
            lock: None,
            cursor: None,
            scrub: 0.5,
            mtf_log_x: false,
        }
    }
}

impl ViewportState {
    /// Toggle autoscale vs lock. On first lock, freeze the current scale if
    /// one is provided; toggling off clears it.
    pub fn toggle_lock(&mut self, current: Option<ScaleLock>) {
        self.lock = match self.lock {
            Some(_) => None,
            None => current,
        };
    }

    /// Set the scrubber from a normalized position (clamped).
    pub fn set_scrub(&mut self, u: f64) {
        self.scrub = if u.is_finite() {
            u.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state() {
        let s = ViewportState::default();
        assert_eq!(s.active_frame, 0);
        assert_eq!(s.tab, PanelTab::EncircledEnergy);
        assert_eq!(s.colormap, Colormap::Viridis);
        assert!(!s.log);
        assert!(s.lock.is_none());
        assert_eq!(s.scrub, 0.5);
    }

    #[test]
    fn lock_toggle_freezes_and_clears() {
        let mut s = ViewportState::default();
        let sc = ScaleLock { lo: 0.0, hi: 1.0 };
        s.toggle_lock(Some(sc));
        assert_eq!(s.lock, Some(sc));
        s.toggle_lock(Some(ScaleLock { lo: 2.0, hi: 3.0 }));
        assert!(s.lock.is_none());
    }

    #[test]
    fn scrub_clamps() {
        let mut s = ViewportState::default();
        s.set_scrub(2.0);
        assert_eq!(s.scrub, 1.0);
        s.set_scrub(f64::NAN);
        assert_eq!(s.scrub, 0.0);
    }

    #[test]
    fn tabs_have_labels_and_order() {
        assert_eq!(PanelTab::ALL.len(), 5);
        assert_eq!(PanelTab::Strehl.label(), "Strehl");
    }
}
