//! Colormaps (docs/ux/spec.md §4).
//!
//! Pure display mapping: a scalar in $[0, 1]$ → RGBA8. No physics — the
//! value domain (normalized intensity, wrapped phase, signed residual) is
//! decided by [`crate::viewmodel::intensity`] and the panels.
//!
//! Families, per the spec:
//!
//! - **Sequential** (intensity-like): Viridis and Inferno — perceptually
//!   uniform and colorblind-safe.
//! - **Diverging** (signed data about 0): RdBu — cross-sections about a
//!   baseline, derivative channels, Zernike residuals.
//! - **Cyclic** (phase in radians, wraps at $\pm\pi$): a hue wheel. Never a
//!   sequential ramp on phase (the seam would read as a discontinuity).

/// A named colormap family selected by the viewport controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colormap {
    /// Perceptually-uniform sequential (default for intensity).
    Viridis,
    /// Perceptually-uniform sequential, warm (alternative for intensity).
    Inferno,
    /// Diverging red–blue, centered at 0 (signed data).
    RdBu,
    /// Cyclic hue wheel (wrapped phase).
    HueWheel,
}

impl Colormap {
    /// The class of scale, used by the mapping layer to decide how the
    /// $[0, 1]$ domain is derived from the raw value.
    pub fn family(self) -> ColormapFamily {
        match self {
            Colormap::Viridis | Colormap::Inferno => ColormapFamily::Sequential,
            Colormap::RdBu => ColormapFamily::Diverging,
            Colormap::HueWheel => ColormapFamily::Cyclic,
        }
    }

    /// Maps a normalized parameter $t \in [0, 1]$ to sRGB bytes (opaque).
    ///
    /// Inputs outside $[0, 1]$ are clamped; non-finite inputs map to the
    /// middle of the scale so a stray `NaN` never reaches the canvas as a
    /// random pixel (docs/ux/spec.md §4 "NaN/inf samples must never reach
    /// the canvas").
    pub fn rgba(self, t: f64) -> [u8; 4] {
        let t = if !t.is_finite() {
            0.5
        } else {
            t.clamp(0.0, 1.0)
        };
        let [r, g, b] = match self {
            Colormap::Viridis => viridis(t),
            Colormap::Inferno => inferno(t),
            Colormap::RdBu => rdbu(t),
            Colormap::HueWheel => hue_wheel(t),
        };
        [r, g, b, 255]
    }
}

/// Scale class of a [`Colormap`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColormapFamily {
    /// Monotonic ramp; domain is a normalized $[0, 1]$ intensity.
    Sequential,
    /// Two ramps meeting at a neutral 0; domain is a signed value about 0.
    Diverging,
    /// Wrapping hue; domain is a phase already wrapped to $[0, 1)$.
    Cyclic,
}

/// Viridis control points (matplotlib), linearly interpolated in sRGB.
///
/// A short sampled table is adequate for display: the panel is a preview,
/// and analysis never runs on the mapped bytes.
const VIRIDIS: &[[f64; 3]] = &[
    [0.267004, 0.005803, 0.329415],
    [0.282327, 0.140545, 0.452190],
    [0.253819, 0.265288, 0.532006],
    [0.206756, 0.371758, 0.553117],
    [0.163545, 0.471136, 0.558147],
    [0.127568, 0.567064, 0.550898],
    [0.134692, 0.660726, 0.514389],
    [0.266941, 0.748751, 0.440778],
    [0.477504, 0.821444, 0.318195],
    [0.741388, 0.868316, 0.149530],
    [0.993248, 0.906157, 0.143936],
];

/// Inferno control points (matplotlib).
const INFERNO: &[[f64; 3]] = &[
    [0.001462, 0.000466, 0.013866],
    [0.087325, 0.044155, 0.304328],
    [0.250202, 0.120143, 0.539916],
    [0.429769, 0.146793, 0.593597],
    [0.606260, 0.154232, 0.542829],
    [0.783825, 0.214383, 0.428390],
    [0.919394, 0.339218, 0.265359],
    [0.985936, 0.519352, 0.126676],
    [0.973991, 0.722417, 0.180285],
    [0.874322, 0.910919, 0.389700],
    [0.988362, 0.998364, 0.644924],
];

/// RdBu diverging anchors (ColorBrewer RdBu, 11-class), blue→white→red.
const RDBU: &[[f64; 3]] = &[
    [0.196078, 0.533333, 0.741176],
    [0.400000, 0.760784, 0.647059],
    [0.631373, 0.878431, 0.878431],
    [0.850980, 0.949020, 0.949020],
    [0.968627, 0.968627, 0.968627], // neutral center (0)
    [0.968627, 0.850980, 0.803922],
    [0.901961, 0.658824, 0.603922],
    [0.784314, 0.396078, 0.352941],
    [0.596078, 0.109804, 0.117647],
    [0.403922, 0.000000, 0.121569],
];

/// Linear interpolation across a sampled table in $[0, 1]$.
fn table_ramp(table: &[[f64; 3]], t: f64) -> [u8; 3] {
    let n = table.len() as f64 - 1.0;
    let x = (t * n).clamp(0.0, n);
    let i0 = x.floor() as usize;
    let i1 = (i0 + 1).min(table.len() - 1);
    let f = x - i0 as f64;
    let a = table[i0];
    let b = table[i1];
    [
        to_u8(a[0] + (b[0] - a[0]) * f),
        to_u8(a[1] + (b[1] - a[1]) * f),
        to_u8(a[2] + (b[2] - a[2]) * f),
    ]
}

fn viridis(t: f64) -> [u8; 3] {
    table_ramp(VIRIDIS, t)
}

fn inferno(t: f64) -> [u8; 3] {
    table_ramp(INFERNO, t)
}

fn rdbu(t: f64) -> [u8; 3] {
    table_ramp(RDBU, t)
}

/// Cyclic hue wheel: $t \in [0, 1)$ → hue at full saturation/value, so the
/// seam at $t = 0 \equiv 1$ matches (phase wraps cleanly).
fn hue_wheel(t: f64) -> [u8; 3] {
    let h = (t - (t / 1.0).floor()) * 6.0; // [0, 6)
    let x = 1.0 - ((h % 2.0) - 1.0).abs();
    let (r, g, b) = match h.floor() as i32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [to_u8(r), to_u8(g), to_u8(b)]
}

fn to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_match_spec() {
        assert_eq!(Colormap::Viridis.family(), ColormapFamily::Sequential);
        assert_eq!(Colormap::RdBu.family(), ColormapFamily::Diverging);
        assert_eq!(Colormap::HueWheel.family(), ColormapFamily::Cyclic);
    }

    #[test]
    fn clamps_out_of_range_and_nonfinite() {
        // Clamped ends equal the t=0 / t=1 samples.
        assert_eq!(Colormap::Viridis.rgba(-1.0), Colormap::Viridis.rgba(0.0));
        assert_eq!(Colormap::Viridis.rgba(2.0), Colormap::Viridis.rgba(1.0));
        // Non-finite falls back to the mid-scale, never a random byte.
        let mid = Colormap::Inferno.rgba(0.5);
        assert_eq!(Colormap::Inferno.rgba(f64::NAN), mid);
        assert_eq!(Colormap::Inferno.rgba(f64::INFINITY), mid);
    }

    #[test]
    fn always_opaque() {
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for map in [
                Colormap::Viridis,
                Colormap::Inferno,
                Colormap::RdBu,
                Colormap::HueWheel,
            ] {
                assert_eq!(map.rgba(t)[3], 255);
            }
        }
    }

    #[test]
    fn viridis_endpoints_distinct_and_monotonic_green() {
        // Viridis starts dark purple and ends yellow: the green channel
        // rises and the blue channel falls across the ramp.
        let lo = Colormap::Viridis.rgba(0.0);
        let hi = Colormap::Viridis.rgba(1.0);
        assert!(hi[1] > lo[1], "green should increase");
        assert!(hi[2] < lo[2], "blue should decrease");
    }

    #[test]
    fn rdbu_center_is_near_neutral() {
        let c = Colormap::RdBu.rgba(0.5);
        // Neutral center: channels close together (near-white).
        let spread = c[0].max(c[1]).max(c[2]) as i32 - c[0].min(c[1]).min(c[2]) as i32;
        assert!(spread < 40, "center should be near-neutral, got {c:?}");
    }

    #[test]
    fn hue_wheel_wraps_seamlessly() {
        // t=0 and t=1 are the same hue (the wrap point).
        assert_eq!(Colormap::HueWheel.rgba(0.0), Colormap::HueWheel.rgba(1.0));
    }
}
