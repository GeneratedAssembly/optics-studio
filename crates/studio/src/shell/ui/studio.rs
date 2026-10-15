//! Studio state: the signal bundle shared by every shell component.

use dioxus::prelude::*;
use optics_core::field::Psf;
use optics_inverse::Carrier;
use optics_zemax::{AgfCatalog, ImportedSystem, Registry};
use std::rc::Rc;

use crate::shell::carrier::demo_carrier;
use crate::shell::dispatch::{self, Banner, BannerLevel, EnginePick, EngineSettings};
use crate::shell::prescription::{StripModel, build_plan, strip_from_design};
use crate::shell::preview::{bmp_data_url, render_psf};
use crate::shell::source::SourceSpec;

/// Which workflow the shell is presenting (spec §3: mode selects active
/// panels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppMode {
    /// Forward: import → edit → simulate (task 10.1 editor).
    Forward,
    /// Inverse: carrier sliders → live preview (spec §3.2).
    Inverse,
}

/// Which pipeline produced the current preview frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewSource {
    /// Edited prescription → simulation beamline.
    Forward,
    /// Carrier values → `Design::beamline()`.
    Carrier,
}

/// A dismissible toast (spec §3.1: errors carry the typed error `Display`).
#[derive(Clone, Debug, PartialEq)]
pub struct Toast {
    /// Monotonic id for `key`.
    pub id: usize,
    /// Message text.
    pub text: String,
}

/// Canvas-side preview defaults.
const PREVIEW_MAX_SIDE: usize = 512;

/// The shell state bundle. All fields are `Signal`s (Copy), so `Shell`
/// itself is `Copy` and cheap to hand to every component via context.
#[derive(Clone, Copy)]
pub struct Shell {
    // — document —
    pub mode: Signal<AppMode>,
    pub doc_name: Signal<String>,
    pub zmx_text: Signal<String>,
    pub imported: Signal<Option<ImportedSystem>>,
    pub parse_error: Signal<Option<String>>,
    pub edited: Signal<bool>,
    pub registry: Signal<Registry>,
    pub glass_names: Signal<Vec<String>>,
    pub recent: Signal<Vec<(String, String)>>,

    // — prescription working copy (table mode) —
    pub surfaces: Signal<Vec<optics_zemax::ZmxSurface>>,

    // — pickers —
    pub band: Signal<usize>,
    pub enabled: Signal<Vec<bool>>,
    pub source: Signal<SourceSpec>,
    pub engine: Signal<EngineSettings>,

    // — preview —
    pub psf: Signal<Option<Psf>>,
    pub psf_from: Signal<PreviewSource>,
    pub strip: Signal<StripModel>,
    pub image_url: Signal<Option<String>>,
    pub downsampled_by: Signal<Option<usize>>,
    pub stale: Signal<bool>,
    pub banner: Signal<Option<Banner>>,
    pub advisory: Signal<Option<String>>,
    pub suggestion: Signal<Option<EnginePick>>,
    pub log_scale: Signal<bool>,
    pub phase_steps: Signal<Vec<(usize, f64)>>,

    // — carrier sliders (spec §3.2) —
    pub carrier: Signal<Rc<Carrier>>,
    pub carrier_values: Signal<Vec<f64>>,
    pub carrier_error: Signal<Option<String>>,

    // — plumbing —
    pub toasts: Signal<Vec<Toast>>,
    pub preview_gen: Signal<usize>,
    pub parse_gen: Signal<usize>,
}

impl Shell {
    /// Fresh session: no document, sensible preview defaults (small grid
    /// keeps the synchronous preview responsive per ADR-10.0 Q3).
    pub fn new() -> Shell {
        let carrier = demo_carrier();
        let values = carrier.params().values();
        Shell {
            mode: Signal::new(AppMode::Forward),
            doc_name: Signal::new(String::new()),
            zmx_text: Signal::new(String::new()),
            imported: Signal::new(None),
            parse_error: Signal::new(None),
            edited: Signal::new(false),
            registry: Signal::new(Registry::builtin()),
            glass_names: Signal::new(builtin_glass_names()),
            recent: Signal::new(Vec::new()),
            surfaces: Signal::new(Vec::new()),
            band: Signal::new(0),
            enabled: Signal::new(Vec::new()),
            source: Signal::new(SourceSpec::Point {
                distance_m: 1.0,
                amplitude: 1.0,
                nx: 64,
                ny: 64,
                dx: 0.5e-3,
                dy: 0.5e-3,
            }),
            engine: Signal::new(EngineSettings::default()),
            psf: Signal::new(None),
            psf_from: Signal::new(PreviewSource::Forward),
            strip: Signal::new(StripModel::default()),
            image_url: Signal::new(None),
            downsampled_by: Signal::new(None),
            stale: Signal::new(false),
            banner: Signal::new(None),
            advisory: Signal::new(None),
            suggestion: Signal::new(None),
            log_scale: Signal::new(false),
            phase_steps: Signal::new(Vec::new()),
            carrier: Signal::new(Rc::new(carrier)),
            carrier_values: Signal::new(values),
            carrier_error: Signal::new(None),
            toasts: Signal::new(Vec::new()),
            preview_gen: Signal::new(0),
            parse_gen: Signal::new(0),
        }
    }

    /// Active wavelength (µm) from the imported system, if any.
    pub fn active_lambda_um(&self) -> Option<f64> {
        let imported = self.imported.peek();
        let imported = imported.as_ref()?;
        let w = imported.system().wavelengths_um.as_slice();
        let i = *self.band.peek();
        w.get(i).copied().filter(|v| *v > 0.0)
    }

    /// Active wavelength with the 0.55 µm demo fallback (carrier mode).
    pub fn lambda_um_or_default(&self) -> f64 {
        self.active_lambda_um().unwrap_or(0.55)
    }

    /// Load `.zmx` text (file panel, sample button, recent entry).
    pub fn set_document(&mut self, name: &str, text: &str) {
        self.doc_name.set(name.to_string());
        self.zmx_text.set(text.to_string());
        self.parse_now();
        let mut recent = self.recent.write();
        recent.retain(|(n, _)| n != name);
        recent.insert(0, (name.to_string(), text.to_string()));
        if recent.len() > 6 {
            recent.truncate(6);
        }
    }

    /// Parse the current text against the current registry (text mode is
    /// authoritative, spec §3.2).
    pub fn parse_now(&mut self) {
        let text = self.zmx_text.peek().clone();
        let registry = self.registry.peek().clone();
        match ImportedSystem::parse_with(&text, registry) {
            Ok(sys) => {
                let wavelengths = sys.system().wavelengths_um.clone();
                let primary = sys.system().primary;
                let surfaces = sys.system().surfaces.clone();
                self.enabled
                    .set(wavelengths.iter().map(|w| *w > 0.0).collect());
                self.band
                    .set(primary.min(wavelengths.len().saturating_sub(1)));
                self.parse_error.set(None);
                self.edited.set(false);
                self.surfaces.set(surfaces);
                self.imported.set(Some(sys));
                self.refresh_forward();
            }
            Err(e) => self.parse_error.set(Some(e.to_string())),
        }
    }

    /// Load an AGF glass catalog and re-parse the open document.
    pub fn load_agf(&mut self, name: &str, text: &str) {
        match AgfCatalog::parse(text) {
            Ok(catalog) => {
                let mut names = self.glass_names.peek().clone();
                for glass in &catalog.glasses {
                    if !names.contains(&glass.name) {
                        names.push(glass.name.clone());
                    }
                }
                let registry = self.registry.peek().clone().with_agf(&catalog);
                self.glass_names.set(names);
                self.registry.set(registry);
                self.push_toast(format!("catalog {name}: {} glasses", catalog.glasses.len()));
                if !self.zmx_text.peek().is_empty() {
                    self.parse_now();
                }
            }
            Err(e) => self.push_toast(format!("AGF {name}: {e}")),
        }
    }

    /// Load the bundled ACA254 sample prescription.
    pub fn load_sample(&mut self) {
        const SAMPLE: &str = include_str!("../../../../../zmx-samples/ACA254-100-UV-Zemax-ZMX.zmx");
        self.set_document("ACA254-100-UV.zmx", SAMPLE);
    }

    /// Debounced preview re-run (spec §3.2: slider commits ≥100 ms; text
    /// validation ~250 ms).
    pub fn schedule_preview(&mut self, delay_ms: u32) {
        let epoch = *self.preview_gen.peek() + 1;
        self.preview_gen.set(epoch);
        let mut st = *self;
        wasm_bindgen_futures::spawn_local(async move {
            gloo_timers::future::TimeoutFuture::new(delay_ms).await;
            if *st.preview_gen.peek() == epoch {
                st.run_preview();
            }
        });
    }

    /// Debounced text-mode validation (spec §3.2: ~250 ms).
    pub fn schedule_parse(&mut self, delay_ms: u32) {
        let epoch = *self.parse_gen.peek() + 1;
        self.parse_gen.set(epoch);
        let mut st = *self;
        wasm_bindgen_futures::spawn_local(async move {
            gloo_timers::future::TimeoutFuture::new(delay_ms).await;
            if *st.parse_gen.peek() == epoch {
                st.parse_now();
            }
        });
    }

    /// Immediate preview re-run for the active mode.
    pub fn run_preview(&mut self) {
        let mode = *self.mode.peek();
        match mode {
            AppMode::Forward => self.refresh_forward(),
            AppMode::Inverse => self.refresh_carrier(),
        }
    }

    /// Forward pipeline: edited surfaces → simulation beamline → PSF.
    pub fn refresh_forward(&mut self) {
        let surfaces = self.surfaces.peek().clone();
        if surfaces.len() < 3 {
            self.clear_preview();
            return;
        }
        let registry = self.registry.peek().clone();
        let lambda_um = self.lambda_um_or_default();
        let plan = match build_plan(&surfaces, &registry, lambda_um) {
            Ok(plan) => plan,
            Err(e) => {
                self.push_toast(format!("plan: {e}"));
                self.stale.set(true);
                return;
            }
        };
        let lambda_m = lambda_um * 1e-6;
        let spec = *self.source.peek();
        let band = match spec.make_band(lambda_m) {
            Ok(band) => band,
            Err(e) => {
                self.push_toast(format!("source: {e}"));
                self.stale.set(true);
                return;
            }
        };
        let settings = *self.engine.peek();
        match plan.psf_with(&band, &settings) {
            Ok(psf) => {
                let (z, medium) = plan.last_step().unwrap_or((0.0, 1.0));
                let aperture = if plan.strip().max_semi_mm > 0.0 {
                    Some(plan.strip().max_semi_mm * 1e-3)
                } else {
                    None
                };
                self.publish(
                    psf,
                    PreviewSource::Forward,
                    plan.strip().clone(),
                    &settings,
                    &band,
                    z,
                    medium,
                    plan.phase_steps(&band),
                    aperture,
                );
            }
            Err(e) => {
                self.push_toast(format!("preview trace: {e}"));
                self.stale.set(true);
            }
        }
    }

    /// Carrier pipeline: slider values → `Design::beamline()` → PSF.
    pub fn refresh_carrier(&mut self) {
        let carrier = self.carrier.peek().clone();
        let values = self.carrier_values.peek().clone();
        if let Err(e) = carrier.params().validate_values(&values) {
            self.carrier_error.set(Some(e.to_string()));
        } else {
            self.carrier_error.set(None);
        }
        let lambda_um = self.lambda_um_or_default();
        let lambda_m = lambda_um * 1e-6;
        let design = match carrier.build(&values, lambda_m) {
            Ok(design) => design,
            Err(e) => {
                self.push_toast(format!("carrier build: {e}"));
                self.stale.set(true);
                return;
            }
        };
        let spec = *self.source.peek();
        let band = match spec.make_band(lambda_m) {
            Ok(band) => band,
            Err(e) => {
                self.push_toast(format!("source: {e}"));
                self.stale.set(true);
                return;
            }
        };
        let settings = *self.engine.peek();
        let strip = strip_from_design(&design, lambda_m);
        let aperture = if strip.max_semi_mm > 0.0 {
            Some(strip.max_semi_mm * 1e-3)
        } else {
            None
        };
        let bl = design.beamline();
        match bl.psf(&band, |b, z, m| dispatch::propagate(&settings, b, z, m)) {
            Ok(psf) => {
                let (z, medium) = strip
                    .nodes
                    .iter()
                    .rev()
                    .find_map(|n| match n {
                        crate::shell::prescription::StripNode::Gap { z_m, n } => Some((*z_m, *n)),
                        _ => None,
                    })
                    .unwrap_or((0.0, 1.0));
                self.publish(
                    psf,
                    PreviewSource::Carrier,
                    strip,
                    &settings,
                    &band,
                    z,
                    medium,
                    Vec::new(),
                    aperture,
                );
            }
            Err(e) => {
                self.push_toast(format!("carrier trace: {e}"));
                self.stale.set(true);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn publish(
        &mut self,
        psf: Psf,
        from: PreviewSource,
        strip: StripModel,
        settings: &EngineSettings,
        band: &optics_core::field::BandField,
        z: f64,
        medium: f64,
        phase_steps: Vec<(usize, f64)>,
        aperture: Option<f64>,
    ) {
        let pre = dispatch::run_banner(settings, band, z, medium);
        // A red pre-run report with a *successful* strict trace means the
        // engine densified automatically — the visible cost of prevention
        // (spec §5.2 yellow).
        let banner = if pre.severity == BannerLevel::Rejected && dispatch::is_strict(settings) {
            Banner {
                severity: BannerLevel::Mitigated,
                headline: "engine densified automatically to satisfy sampling".into(),
                detail: Some(format!(
                    "{} {}",
                    pre.headline,
                    pre.detail.unwrap_or_default()
                )),
                remedies: Vec::new(),
            }
        } else {
            pre
        };
        let advisory = dispatch::regime_advisory(settings.pick, band, z, aperture);
        let suggested = dispatch::suggested(band, z, aperture);
        let log = *self.log_scale.peek();
        let image = match render_psf(&psf, PREVIEW_MAX_SIDE, log) {
            Ok(img) => {
                self.downsampled_by.set(img.downsampled_by);
                Some(bmp_data_url(&img))
            }
            Err(e) => {
                self.push_toast(format!("display: {e}"));
                self.downsampled_by.set(None);
                None
            }
        };
        self.psf.set(Some(psf));
        self.psf_from.set(from);
        self.strip.set(strip);
        self.banner.set(Some(banner));
        self.advisory.set(advisory);
        self.suggestion.set(if suggested == settings.pick {
            None
        } else {
            Some(suggested)
        });
        self.phase_steps.set(phase_steps);
        self.image_url.set(image);
        self.stale.set(false);
    }

    fn clear_preview(&mut self) {
        self.psf.set(None);
        self.image_url.set(None);
        self.strip.set(StripModel::default());
        self.banner.set(None);
        self.advisory.set(None);
        self.suggestion.set(None);
        self.phase_steps.set(Vec::new());
        self.stale.set(true);
    }

    /// Push a dismissible toast (keeps the latest four).
    pub fn push_toast(&mut self, text: impl Into<String>) {
        let id = self.toasts.peek().len() + *self.preview_gen.peek();
        let mut toasts = self.toasts.write();
        toasts.push(Toast {
            id,
            text: text.into(),
        });
        if toasts.len() > 4 {
            let cut = toasts.len() - 4;
            toasts.drain(0..cut);
        }
    }

    /// Remove a toast by id.
    pub fn dismiss_toast(&mut self, id: usize) {
        self.toasts.write().retain(|t| t.id != id);
    }

    /// Set the working-copy surface field from a table edit.
    pub fn edit_surface(&mut self, index: usize, field: SurfaceField, raw: &str) {
        let parsed: Option<f64> = match field {
            SurfaceField::Glass => {
                let name = if raw.is_empty() || raw == "—" {
                    None
                } else {
                    Some(raw.to_string())
                };
                let mut surfaces = self.surfaces.write();
                if let Some(s) = surfaces.get_mut(index) {
                    s.glass = name;
                }
                self.edited.set(true);
                return;
            }
            SurfaceField::Curvature => raw.parse().ok(),
            SurfaceField::Conic => raw.parse().ok(),
            SurfaceField::SemiDiam => raw.parse().ok(),
            SurfaceField::Thickness => {
                if raw.trim().eq_ignore_ascii_case("inf") {
                    let mut surfaces = self.surfaces.write();
                    if let Some(s) = surfaces.get_mut(index) {
                        s.thickness_mm = None;
                    }
                    self.edited.set(true);
                    return;
                }
                raw.parse().ok()
            }
        };
        let Some(value) = parsed else { return };
        {
            let mut surfaces = self.surfaces.write();
            let Some(s) = surfaces.get_mut(index) else {
                return;
            };
            match field {
                SurfaceField::Curvature => s.curvature_mm = value,
                SurfaceField::Conic => s.conic = value,
                SurfaceField::SemiDiam => s.semi_diam_mm = value,
                SurfaceField::Thickness => s.thickness_mm = Some(value),
                SurfaceField::Glass => unreachable!("handled above"),
            }
        }
        self.edited.set(true);
    }

    /// Slider commit on a working-copy field.
    pub fn slide_surface(&mut self, index: usize, field: SurfaceField, value: f64) {
        self.edit_surface(index, field, &value.to_string());
    }
}

/// Editable numeric fields in table mode (spec §3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceField {
    /// `CURV` (1/mm).
    Curvature,
    /// Conic constant.
    Conic,
    /// Semi-diameter (mm).
    SemiDiam,
    /// Thickness (mm).
    Thickness,
    /// Glass name.
    Glass,
}

/// Built-in catalog names for the glass combobox.
pub fn builtin_glass_names() -> Vec<String> {
    optics_zemax::catalog::BUILTIN
        .iter()
        .map(|e| e.name.to_string())
        .collect()
}

impl Default for Shell {
    fn default() -> Self {
        Self::new()
    }
}

/// Callable getters mirroring the signal fields (`studio.doc_name()` etc.).
macro_rules! sig_getters {
    ($($name:ident -> $ty:ty),* $(,)?) => {
        impl Shell {
            $(
                #[doc = concat!("Subscribed read of the `", stringify!($name), "` signal.")]
                pub fn $name(&self) -> $ty {
                    (self.$name)()
                }
            )*
        }
    };
}

sig_getters! {
    mode -> AppMode,
    doc_name -> String,
    zmx_text -> String,
    imported -> Option<ImportedSystem>,
    parse_error -> Option<String>,
    edited -> bool,
    glass_names -> Vec<String>,
    recent -> Vec<(String, String)>,
    surfaces -> Vec<optics_zemax::ZmxSurface>,
    band -> usize,
    enabled -> Vec<bool>,
    source -> SourceSpec,
    engine -> EngineSettings,
    psf -> Option<Psf>,
    psf_from -> PreviewSource,
    strip -> StripModel,
    image_url -> Option<String>,
    downsampled_by -> Option<usize>,
    stale -> bool,
    banner -> Option<Banner>,
    advisory -> Option<String>,
    suggestion -> Option<EnginePick>,
    log_scale -> bool,
    phase_steps -> Vec<(usize, f64)>,
    carrier -> Rc<Carrier>,
    carrier_values -> Vec<f64>,
    carrier_error -> Option<String>,
    toasts -> Vec<Toast>,
}
