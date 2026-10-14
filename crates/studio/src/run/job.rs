//! The serializable run job (spec `docs/ux/spec.md` §3.4).
//!
//! A [`RunJob`] fully describes one inverse-design run in plain data:
//! design carrier, illumination, observation grid, merit, solver, budget,
//! seed, and chunking. It is the message payload between the UI and the
//! solver web worker (ADR-10.0 Q3): closures cannot cross a worker
//! boundary, so the worker rebuilds the real `optics-inverse` objects from
//! this description with [`RunJob::build_carrier`],
//! [`RunJob::build_renderer`], and [`RunJob::build_merit`].
//!
//! Units are SI throughout the struct (meters, radians) exactly as in the
//! headless crates; engineering display units are a UI-layer concern
//! (spec §3.5). The job composes headless APIs only — it evaluates no
//! physics formulas beyond what `Carrier`/`Renderer`/`Merit` already do
//! (plan constraint, spec §6).

use serde::{Deserialize, Serialize};

use optics_core::element::{ClearAperture, Material};
use optics_core::field::BandField;
use optics_core::propagation::asm::AsmConfig;
use optics_core::sources::{GaussianBeam, gaussian, spherical};
use optics_inverse::{
    AdamConfig, Carrier, Channel, Derivatives, EncircledEnergyMerit, Merit, MseToTarget, MtfMerit,
    Params, Renderer, Selector, Solver, SpotRmsMerit, StrehlMerit,
};

/// Clear aperture of a grid carrier, mirroring
/// [`optics_core::element::ClearAperture`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ApertureSpec {
    /// No stop — the element acts on the whole sampled plane.
    Field,
    /// Circular stop of radius `radius` (m).
    Circle { radius: f64 },
    /// Elliptical stop, semi-axes in meters.
    Ellipse { a: f64, b: f64 },
}

impl ApertureSpec {
    fn to_core(self) -> ClearAperture {
        match self {
            Self::Field => ClearAperture::Field,
            Self::Circle { radius } => ClearAperture::Circle(radius),
            Self::Ellipse { a, b } => ClearAperture::Ellipse { a, b },
        }
    }

    fn validate(self) -> Result<(), String> {
        match self {
            Self::Field => Ok(()),
            Self::Circle { radius } if radius.is_finite() && radius > 0.0 => Ok(()),
            Self::Ellipse { a, b } if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 => {
                Ok(())
            }
            _ => Err("aperture dimensions must be positive and finite".into()),
        }
    }
}

/// Sellmeier material for thickness carriers, mirroring
/// [`Material::sellmeier`] / [`Material::sellmeier_book1`]. `constant` is
/// the Book-1 $C_0$ term ($n^2 = C_0 + \sum B\lambda^2/(\lambda^2 - C)$);
/// `None` uses the $C_0 = 1$ form. Terms are $(B, C)$ pairs with $B$
/// dimensionless and $C$ in m².
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialSpec {
    /// Display name (run card / labels only).
    pub name: String,
    /// Book-1 constant $C_0$ (`None` = 1).
    pub constant: Option<f64>,
    /// Sellmeier $(B, C)$ terms.
    pub terms: Vec<(f64, f64)>,
}

impl MaterialSpec {
    fn to_core(&self) -> Material {
        let terms: Vec<(f64, f64)> = self.terms.clone();
        match self.constant {
            Some(c) => Material::sellmeier_book1(c, terms),
            None => Material::sellmeier(terms),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if let Some(c) = self.constant
            && !(c.is_finite() && c > 0.0)
        {
            return Err("material Book-1 constant must be positive and finite".into());
        }
        if self
            .terms
            .iter()
            .any(|(b, c)| !b.is_finite() || !c.is_finite())
        {
            return Err("sellmeier terms must be finite".into());
        }
        Ok(())
    }
}

/// Design carrier: a sampled grid relayed to the detector.
///
/// The grid sits at the plane where the [`SourceSpec`] field is sampled;
/// the design then propagates `z_after` (m) through `medium` to the
/// detector, matching the renderer's single-plane convention (the grid
/// carriers' own zero trailing gap is filled here — spec §3.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DesignSpec {
    /// Sampled phase grid (radians per slot), band-independent.
    GridPhaseRelay {
        /// Lattice samples `(nx, ny)`.
        dims: (usize, usize),
        /// Lattice pitch (m).
        pitch: (f64, f64),
        /// Clear aperture.
        aperture: ApertureSpec,
        /// Grid center (m).
        center: (f64, f64),
        /// Per-slot phase bounds `[lo, hi]` (rad).
        lo: f64,
        /// Upper bound (rad).
        hi: f64,
        /// Initial value for every slot (rad).
        init: f64,
        /// DOE-to-detector propagation (m), > 0.
        z_after: f64,
        /// Propagation medium index.
        medium: f64,
    },
    /// Sampled relief-thickness grid (meters per slot) in a material.
    GridThicknessRelay {
        /// Lattice samples `(nx, ny)`.
        dims: (usize, usize),
        /// Lattice pitch (m).
        pitch: (f64, f64),
        /// Clear aperture.
        aperture: ApertureSpec,
        /// Grid center (m).
        center: (f64, f64),
        /// Maximum relief thickness `t_max` (m), > 0.
        t_max: f64,
        /// Initial thickness for every slot (m).
        init: f64,
        /// Dispersive material of the relief.
        material: MaterialSpec,
        /// DOE-to-detector propagation (m), > 0.
        z_after: f64,
        /// Propagation medium index.
        medium: f64,
    },
}

/// Illumination sampled at the design plane. Scenes shift the source
/// laterally: spherical by source position, Gaussian by beam center,
/// plane wave not scene-sensitive (tilt mapping is a shell decision, 10.1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SourceSpec {
    /// Point source at `(x0, y0, -distance)` sampled at the design plane.
    Spherical {
        /// Source-to-plane distance (m), > 0.
        distance: f64,
        /// Amplitude $A$ (meter units so $A/r$ is a field amplitude).
        amplitude: f64,
    },
    /// Gaussian beam sampled at distance `z` from its waist.
    Gaussian {
        /// Waist radius $w_0$ (m), > 0.
        w0: f64,
        /// Peak waist amplitude.
        amplitude: f64,
        /// Waist center offset (m), added to the scene position.
        center: (f64, f64),
        /// Waist-to-plane distance (m).
        z: f64,
    },
    /// Uniform tilted plane wave.
    PlaneWave {
        /// Tilt about x (rad).
        theta_x: f64,
        /// Tilt about y (rad).
        theta_y: f64,
        /// Field amplitude.
        amplitude: f64,
    },
}

/// Observation frame grid: the detector-plane sample lattice every merit
/// reads (`nx * ny` samples at pitch `dx, dy`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridUi {
    /// Columns.
    pub nx: usize,
    /// Rows.
    pub ny: usize,
    /// Column pitch (m).
    pub dx: f64,
    /// Row pitch (m).
    pub dy: f64,
}

/// Merit frame selector, mirroring [`Selector`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectorSpec {
    /// Channel to read.
    pub channel: ChannelSpec,
    /// Band index; `None` sums over bands.
    pub band: Option<usize>,
    /// Scene index; `None` sums over scenes.
    pub scene: Option<usize>,
}

/// Observation channel, mirroring [`Channel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelSpec {
    /// Detector-plane intensity.
    Intensity,
    /// `∂I/∂x₀` derivative channel.
    DxDyX,
    /// `∂I/∂y₀` derivative channel.
    DxDyY,
    /// `∂I/∂λ` derivative channel.
    Dlambda,
}

impl ChannelSpec {
    fn to_core(self) -> Channel {
        match self {
            Self::Intensity => Channel::Intensity,
            Self::DxDyX => Channel::DxDyX,
            Self::DxDyY => Channel::DxDyY,
            Self::Dlambda => Channel::Dlambda,
        }
    }
}

/// Built-in merit selection (spec §3.4 "Merit editor" row; custom merits
/// are gap G-5 and out of scope).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeritSpec {
    /// MSE of the selected intensity frame against a target frame.
    MseToTarget {
        /// Target samples, row-major, length `nx * ny`.
        target: Vec<f64>,
        /// Frame selector (intensity-like channel).
        selector: SelectorSpec,
    },
    /// Negative encircled energy at radius `radius` (m).
    EncircledEnergy {
        /// EE radius (m), > 0.
        radius: f64,
        /// Frame selector.
        selector: SelectorSpec,
    },
    /// Spot RMS radius (m).
    SpotRms {
        /// Frame selector.
        selector: SelectorSpec,
    },
    /// Negative Strehl ratio against a reference frame.
    Strehl {
        /// Reference intensity frame, length `nx * ny`.
        reference: Vec<f64>,
        /// Frame selector.
        selector: SelectorSpec,
    },
    /// Negative MTF at frequency `freq` (cycles/m).
    Mtf {
        /// Frequency (cycles/m), below Nyquist `1 / (2 dx)`.
        freq: f64,
        /// Frame selector.
        selector: SelectorSpec,
    },
}

impl MeritSpec {
    /// Merit label — matches `Merit::name` of the built object (run card).
    pub fn label(&self) -> &'static str {
        match self {
            Self::MseToTarget { .. } => "mse_to_target",
            Self::EncircledEnergy { .. } => "neg_encircled_energy",
            Self::SpotRms { .. } => "spot_rms",
            Self::Strehl { .. } => "neg_strehl",
            Self::Mtf { .. } => "neg_mtf",
        }
    }
}

/// Solver selection, mirroring [`Solver`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SolverSpec {
    /// Adaptive-step random walk, accept-only-improvements.
    RandomWalk {
        /// Fraction of budget without improvement before halving the step.
        stagnation: usize,
    },
    /// Metropolis simulated annealing, geometric cooling.
    Annealing {
        /// Initial temperature as a fraction of the first cost scale
        /// (`None` = auto-calibrate).
        t0: Option<f64>,
        /// Final temperature as a fraction of `t0`, in `(0, 1]`.
        t_end_frac: f64,
    },
    /// Central-difference Adam descent (`budget`/`chunk` count epochs).
    Adam {
        /// Learning rate as a fraction of each slot's bound span.
        lr_frac: f64,
        /// First-moment decay.
        beta1: f64,
        /// Second-moment decay.
        beta2: f64,
        /// Denominator floor.
        eps: f64,
        /// Gradient L2-norm clip.
        clip_norm: Option<f64>,
    },
}

/// ASM engine knobs exposed by the observation config panel.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AsmUi {
    /// Zero-padding factor per axis.
    pub padding: f64,
    /// Matsushima band-limit mask.
    pub band_limited: bool,
    /// Densify-on-aliasing mitigation.
    pub strict: bool,
    /// Densification cap for `strict`.
    pub max_padding: f64,
}

/// Derivative channels to render (each costs two extra renders).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DerivativesUi {
    /// `∂I/∂x₀`.
    pub x: bool,
    /// `∂I/∂y₀`.
    pub y: bool,
    /// `∂I/∂λ`.
    pub lambda: bool,
}

/// One fully-specified inverse run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunJob {
    /// Design carrier.
    pub design: DesignSpec,
    /// Illumination at the design plane.
    pub source: SourceSpec,
    /// Detector-plane frame grid.
    pub grid: GridUi,
    /// Band wavelengths (m).
    pub bands: Vec<f64>,
    /// Scene source positions `(x0, y0)` (m).
    pub scenes: Vec<(f64, f64)>,
    /// Derivative channels.
    pub derivatives: DerivativesUi,
    /// Source-position FD step (m).
    pub scene_step: f64,
    /// Band FD step (m).
    pub lambda_step: f64,
    /// ASM knobs.
    pub asm: AsmUi,
    /// Merit selection.
    pub merit: MeritSpec,
    /// Solver selection.
    pub solver: SolverSpec,
    /// Total budget in solver units — evaluations for the stochastic
    /// solvers, epochs for Adam (`optics_inverse::solve` semantics).
    pub budget: usize,
    /// RNG seed. Always `Some` on wasm: `seed = None` draws from
    /// `rand::thread_rng()`, which is unavailable on wasm32-unknown-unknown
    /// (spec §3.4, ADR-10.0 Q6).
    pub seed: u64,
    /// Initial move scale as a fraction of the bound span.
    pub initial_scale: f64,
    /// Solver units consumed per chunk — the responsiveness knob (spec
    /// §3.4, gap G-2). Must be ≥ 50.
    pub chunk: usize,
    /// First chunk starts from a uniform draw in bounds.
    pub random_init: bool,
    /// Optional continuation start vector (length `dim`) — "Apply best"
    /// feeds the previous run's `Solution::best` back in.
    pub start_values: Option<Vec<f64>>,
}

/// Minimum chunk size (spec §3.4: "chunk size must be ≥ ~50 evaluations
/// to avoid overhead spam").
pub const MIN_CHUNK: usize = 50;

impl RunJob {
    /// Number of design slots.
    pub fn dim(&self) -> usize {
        let (nx, ny) = match &self.design {
            DesignSpec::GridPhaseRelay { dims, .. }
            | DesignSpec::GridThicknessRelay { dims, .. } => *dims,
        };
        nx * ny
    }

    /// Bounds + geometry validation. Everything the builders would fail on
    /// at runtime is caught here first so the UI can show the message
    /// before a worker round trip.
    pub fn validate(&self) -> Result<(), String> {
        let (dims, pitch, aperture, center, z_after, medium) = match &self.design {
            DesignSpec::GridPhaseRelay {
                dims,
                pitch,
                aperture,
                center,
                lo,
                hi,
                init,
                z_after,
                medium,
            } => {
                if !(lo.is_finite() && hi.is_finite() && init.is_finite() && lo < hi) {
                    return Err("phase bounds must be finite with lo < hi".into());
                }
                if !(init >= lo && init <= hi) {
                    return Err("phase init must lie inside [lo, hi]".into());
                }
                (*dims, *pitch, *aperture, *center, *z_after, *medium)
            }
            DesignSpec::GridThicknessRelay {
                dims,
                pitch,
                aperture,
                center,
                t_max,
                init,
                material,
                z_after,
                medium,
            } => {
                material.validate()?;
                if !(t_max.is_finite() && *t_max > 0.0) {
                    return Err("t_max must be positive and finite".into());
                }
                if !(init.is_finite() && *init >= 0.0 && *init <= *t_max) {
                    return Err("thickness init must lie inside [0, t_max]".into());
                }
                (*dims, *pitch, *aperture, *center, *z_after, *medium)
            }
        };
        let (nx, ny) = dims;
        if nx == 0 || ny == 0 {
            return Err("design grid dims must be positive".into());
        }
        let (px, py) = pitch;
        if !(px.is_finite() && py.is_finite() && px > 0.0 && py > 0.0) {
            return Err("design pitch must be positive and finite".into());
        }
        aperture.validate()?;
        let (cx, cy) = center;
        if !cx.is_finite() || !cy.is_finite() {
            return Err("grid center must be finite".into());
        }
        if !(z_after.is_finite() && z_after > 0.0) {
            return Err("DOE-to-detector distance must be positive and finite".into());
        }
        if !(medium.is_finite() && medium > 0.0) {
            return Err("propagation medium must be positive and finite".into());
        }

        match self.source {
            SourceSpec::Spherical {
                distance,
                amplitude,
            } => {
                if !(distance.is_finite() && distance > 0.0 && amplitude.is_finite()) {
                    return Err("spherical source needs distance > 0 and finite amplitude".into());
                }
            }
            SourceSpec::Gaussian {
                w0,
                amplitude,
                center,
                z,
            } => {
                if !(w0.is_finite()
                    && w0 > 0.0
                    && amplitude.is_finite()
                    && z.is_finite()
                    && center.0.is_finite()
                    && center.1.is_finite())
                {
                    return Err("gaussian beam needs w0 > 0 and finite parameters".into());
                }
            }
            SourceSpec::PlaneWave {
                theta_x,
                theta_y,
                amplitude,
            } => {
                if !(theta_x.is_finite() && theta_y.is_finite() && amplitude.is_finite()) {
                    return Err("plane wave needs finite tilts and amplitude".into());
                }
            }
        }

        let GridUi { nx, ny, dx, dy } = self.grid;
        if nx == 0 || ny == 0 {
            return Err("observation grid must have positive sample counts".into());
        }
        if !(dx.is_finite() && dy.is_finite() && dx > 0.0 && dy > 0.0) {
            return Err("observation pitch must be positive and finite".into());
        }
        if self.bands.is_empty() || self.bands.iter().any(|l| !l.is_finite() || *l <= 0.0) {
            return Err("bands must be non-empty, positive, finite wavelengths (m)".into());
        }
        if self.scenes.is_empty()
            || self
                .scenes
                .iter()
                .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return Err("scenes must be non-empty with finite positions".into());
        }
        if !(self.scene_step.is_finite() && self.scene_step > 0.0) {
            return Err("scene FD step must be positive and finite".into());
        }
        if !(self.lambda_step.is_finite() && self.lambda_step > 0.0) {
            return Err("band FD step must be positive and finite".into());
        }
        let AsmUi {
            padding,
            strict: _,
            band_limited: _,
            max_padding,
        } = self.asm;
        if !(padding.is_finite() && padding >= 1.0 && max_padding.is_finite() && max_padding >= 1.0)
        {
            return Err("ASM padding must be finite and >= 1".into());
        }

        let frame_len = nx * ny;
        let selector_channel = self.merit_selector().channel;
        match &self.merit {
            MeritSpec::MseToTarget { target, .. }
            | MeritSpec::Strehl {
                reference: target, ..
            } => {
                if target.len() != frame_len {
                    return Err(format!(
                        "{} frame length {} != observation frame {frame_len}",
                        self.merit.label(),
                        target.len()
                    ));
                }
                if target.iter().any(|v| !v.is_finite()) {
                    return Err("merit frame samples must be finite".into());
                }
            }
            MeritSpec::EncircledEnergy { radius, .. } => {
                if !(radius.is_finite() && *radius > 0.0) {
                    return Err("EE radius must be positive and finite".into());
                }
            }
            MeritSpec::Mtf { freq, .. } => {
                if !freq.is_finite() || *freq < 0.0 {
                    return Err("MTF frequency must be finite and non-negative".into());
                }
            }
            MeritSpec::SpotRms { .. } => {}
        }
        if selector_channel != ChannelSpec::Intensity {
            let required = match selector_channel {
                ChannelSpec::DxDyX => self.derivatives.x,
                ChannelSpec::DxDyY => self.derivatives.y,
                ChannelSpec::Dlambda => self.derivatives.lambda,
                ChannelSpec::Intensity => true,
            };
            if !required {
                return Err(format!(
                    "merit reads {:?} but that derivative channel is not rendered",
                    selector_channel
                ));
            }
        }
        let sel = self.merit_selector();
        if sel.band.is_some_and(|b| b >= self.bands.len())
            || sel.scene.is_some_and(|s| s >= self.scenes.len())
        {
            return Err("merit selector band/scene index out of range".into());
        }

        match self.solver {
            SolverSpec::RandomWalk { stagnation } => {
                if stagnation == 0 {
                    return Err("random-walk stagnation must be > 0".into());
                }
            }
            SolverSpec::Annealing { t0, t_end_frac } => {
                if t0.is_some_and(|t| !(t.is_finite() && t > 0.0)) {
                    return Err("annealing t0 must be positive and finite".into());
                }
                if !(t_end_frac.is_finite() && t_end_frac > 0.0 && t_end_frac <= 1.0) {
                    return Err("annealing t_end_frac must lie in (0, 1]".into());
                }
            }
            SolverSpec::Adam {
                lr_frac,
                beta1,
                beta2,
                eps,
                clip_norm,
            } => {
                if !(lr_frac.is_finite() && lr_frac > 0.0) {
                    return Err("Adam lr_frac must be positive and finite".into());
                }
                if !(0.0 < beta1 && beta1 < 1.0 && 0.0 < beta2 && beta2 < 1.0) {
                    return Err("Adam betas must lie in (0, 1)".into());
                }
                if !(eps.is_finite() && eps >= 0.0) {
                    return Err("Adam eps must be finite and non-negative".into());
                }
                if clip_norm.is_some_and(|c| !(c.is_finite() && c > 0.0)) {
                    return Err("Adam clip_norm must be positive and finite".into());
                }
            }
        }
        if self.budget == 0 {
            return Err("budget must be > 0".into());
        }
        if self.chunk < MIN_CHUNK {
            return Err(format!(
                "chunk must be >= {MIN_CHUNK} (spec §3.4 overhead floor)"
            ));
        }
        if !(self.initial_scale.is_finite() && self.initial_scale > 0.0) {
            return Err("initial_scale must be positive and finite".into());
        }
        if let Some(v) = &self.start_values {
            if v.len() != self.dim() {
                return Err(format!(
                    "start_values length {} != design dim {}",
                    v.len(),
                    self.dim()
                ));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err("start_values must be finite".into());
            }
        }
        Ok(())
    }

    pub fn merit_selector(&self) -> SelectorSpec {
        match &self.merit {
            MeritSpec::MseToTarget { selector, .. }
            | MeritSpec::EncircledEnergy { selector, .. }
            | MeritSpec::SpotRms { selector }
            | MeritSpec::Strehl { selector, .. }
            | MeritSpec::Mtf { selector, .. } => *selector,
        }
    }

    /// Replace the current merit's frame selector (merit editor).
    pub fn set_merit_selector(&mut self, selector: SelectorSpec) {
        match &mut self.merit {
            MeritSpec::MseToTarget { selector: s, .. }
            | MeritSpec::EncircledEnergy { selector: s, .. }
            | MeritSpec::SpotRms { selector: s }
            | MeritSpec::Strehl { selector: s, .. }
            | MeritSpec::Mtf { selector: s, .. } => *s = selector,
        }
    }

    /// Build the carrier: the grid relay wrapped in an analytic carrier so
    /// the design propagates `z_after` to the detector (see
    /// [`DesignSpec`] docs). Bounds are identical to the plain grid
    /// carrier; `build` delegates to it, so no physics is re-derived here.
    pub fn build_carrier(&self) -> Result<Carrier, String> {
        let (inner, z_after, medium) = match &self.design {
            DesignSpec::GridPhaseRelay {
                dims,
                pitch,
                aperture,
                center,
                lo,
                hi,
                init,
                z_after,
                medium,
            } => (
                Carrier::grid_phase(*dims, *pitch, aperture.to_core(), *center, *lo, *hi, *init)
                    .map_err(|e| e.to_string())?,
                *z_after,
                *medium,
            ),
            DesignSpec::GridThicknessRelay {
                dims,
                pitch,
                aperture,
                center,
                t_max,
                init,
                material,
                z_after,
                medium,
            } => (
                Carrier::grid_thickness(
                    *dims,
                    *pitch,
                    aperture.to_core(),
                    *center,
                    *t_max,
                    *init,
                    material.to_core(),
                )
                .map_err(|e| e.to_string())?,
                *z_after,
                *medium,
            ),
        };
        let params = inner.params().clone();
        Ok(Carrier::analytic(params, move |values, lambda| {
            let mut design = inner.build(values, lambda)?;
            design.gap(z_after, medium);
            Ok(design)
        }))
    }

    /// The design space of the carrier (bounds, names) for display.
    pub fn build_params(&self) -> Result<Params, String> {
        Ok(self.build_carrier()?.params().clone())
    }

    /// Build the renderer over `carrier` with the observation config.
    pub fn build_renderer<'c>(&self, carrier: &'c Carrier) -> Renderer<'c> {
        let source = self.source;
        let grid = self.grid;
        let mut renderer = Renderer::new(carrier, move |x0, y0, lambda| {
            let GridUi { nx, ny, dx, dy } = grid;
            let field =
                match source {
                    SourceSpec::Spherical {
                        distance,
                        amplitude,
                    } => spherical((x0, y0), distance, amplitude, lambda, nx, ny, dx, dy).map_err(
                        |e| optics_inverse::InverseError::Render {
                            reason: e.to_string(),
                        },
                    ),
                    SourceSpec::Gaussian {
                        w0,
                        amplitude,
                        center,
                        z,
                    } => {
                        let beam = GaussianBeam {
                            w0,
                            amplitude,
                            center: (center.0 + x0, center.1 + y0),
                        };
                        gaussian(&beam, z, lambda, nx, ny, dx, dy).map_err(|e| {
                            optics_inverse::InverseError::Render {
                                reason: e.to_string(),
                            }
                        })
                    }
                    SourceSpec::PlaneWave {
                        theta_x,
                        theta_y,
                        amplitude,
                    } => BandField::plane_wave(theta_x, theta_y, amplitude, nx, ny, dx, dy, lambda)
                        .map_err(|e| optics_inverse::InverseError::Render {
                            reason: e.to_string(),
                        }),
                }?;
            Ok(field)
        });
        renderer.bands = self.bands.clone();
        renderer.scenes = self.scenes.clone();
        renderer.derivatives = Derivatives {
            x: self.derivatives.x,
            y: self.derivatives.y,
            lambda: self.derivatives.lambda,
        };
        renderer.scene_step = self.scene_step;
        renderer.lambda_step = self.lambda_step;
        renderer.asm = AsmConfig {
            padding: self.asm.padding,
            band_limited: self.asm.band_limited,
            strict: self.asm.strict,
            max_padding: self.asm.max_padding,
            medium: 1.0,
        };
        renderer
    }

    /// Build the merit object.
    pub fn build_merit(&self) -> Box<dyn Merit> {
        let sel = |s: SelectorSpec| Selector {
            channel: s.channel.to_core(),
            band: s.band,
            scene: s.scene,
        };
        match &self.merit {
            MeritSpec::MseToTarget { target, selector } => Box::new(MseToTarget {
                target: target.clone(),
                selector: sel(*selector),
            }),
            MeritSpec::EncircledEnergy { radius, selector } => Box::new(EncircledEnergyMerit {
                radius: *radius,
                selector: sel(*selector),
            }),
            MeritSpec::SpotRms { selector } => Box::new(SpotRmsMerit {
                selector: sel(*selector),
            }),
            MeritSpec::Strehl {
                reference,
                selector,
            } => Box::new(StrehlMerit {
                reference: reference.clone(),
                selector: sel(*selector),
            }),
            MeritSpec::Mtf { freq, selector } => Box::new(MtfMerit {
                freq: *freq,
                selector: sel(*selector),
            }),
        }
    }

    /// The solver enum for `optics_inverse::solve`.
    pub fn solver_enum(&self) -> Solver {
        match self.solver {
            SolverSpec::RandomWalk { stagnation } => Solver::RandomWalk { stagnation },
            SolverSpec::Annealing { t0, t_end_frac } => Solver::Annealing { t0, t_end_frac },
            SolverSpec::Adam {
                lr_frac,
                beta1,
                beta2,
                eps,
                clip_norm,
            } => Solver::Adam(AdamConfig {
                lr_frac,
                beta1,
                beta2,
                eps,
                clip_norm,
            }),
        }
    }

    /// Solver label for the run card.
    pub fn solver_label(&self) -> &'static str {
        match self.solver {
            SolverSpec::RandomWalk { .. } => "random_walk",
            SolverSpec::Annealing { .. } => "annealing",
            SolverSpec::Adam { .. } => "adam",
        }
    }

    /// Demo preset: phase-grid DOE focusing a spherical wave — negative
    /// encircled energy inside 60 µm at the detector plane. Small enough
    /// to run interactively in a browser (16×16 slots, 32×32 frames).
    pub fn demo_ee_focus() -> Self {
        Self {
            design: DesignSpec::GridPhaseRelay {
                dims: (16, 16),
                pitch: (20e-6, 20e-6),
                aperture: ApertureSpec::Circle { radius: 150e-6 },
                center: (0.0, 0.0),
                lo: -std::f64::consts::PI,
                hi: std::f64::consts::PI,
                init: 0.0,
                z_after: 2e-3,
                medium: 1.0,
            },
            source: SourceSpec::Spherical {
                distance: 50e-3,
                amplitude: 1.0,
            },
            grid: GridUi {
                nx: 32,
                ny: 32,
                dx: 20e-6,
                dy: 20e-6,
            },
            bands: vec![500e-9],
            scenes: vec![(0.0, 0.0)],
            derivatives: DerivativesUi::default(),
            scene_step: 1e-6,
            lambda_step: 1e-9,
            asm: AsmUi {
                padding: 2.0,
                band_limited: true,
                strict: false,
                max_padding: 4.0,
            },
            merit: MeritSpec::EncircledEnergy {
                radius: 60e-6,
                selector: SelectorSpec {
                    channel: ChannelSpec::Intensity,
                    band: None,
                    scene: None,
                },
            },
            solver: SolverSpec::Annealing {
                t0: None,
                t_end_frac: 1e-3,
            },
            budget: 600,
            seed: 0,
            initial_scale: 0.2,
            chunk: 60,
            random_init: false,
            start_values: None,
        }
    }
}
