//! Field initializers: physical beam descriptions turned into [`BandField`]s.
//!
//! Sources are *data + pure sampling*: no propagation machinery lives here.
//! Each initializer evaluates a closed-form paraxial solution on the sample
//! grid and binds the result to a wavelength, so a multiband source is simply
//! a collection of `BandField`s built from the same beam record (approved
//! Phase-3 decision D3: `sources` is a top-level module returning
//! [`BandField`]s, multiband by construction).
//!
//! # Position is a system property, not a source property
//!
//! A [`GaussianBeam`] describes the beam *itself* (waist radius, amplitude,
//! transverse center); it holds **no longitudinal position**. Where the beam
//! is sampled along the optical axis — the signed distance `z` from the waist
//! to the sampling plane — is supplied at sampling time to [`gaussian`], just
//! as an element's plane placement is a system decision rather than a property
//! of the element. The beam radius and Gouy phase are therefore *functions of
//! `z`* ([`GaussianBeam::beam_radius`], [`GaussianBeam::gouy_phase`]): a
//! measurement of the beam at a particular plane, which matters when a system
//! places sources at distances other than the waist.
//!
//! # Gaussian beam convention
//!
//! The classical paraxial Gaussian beam (Siegman/Goodman; the project
//! textbook only uses Gaussian beams as inputs to beam shapers, Sec. 11.2,
//! and does not restate the $q(z)$ solution) is written here in the
//! $q$-form adapted to **this crate's $e^{+ikz}$ forward-propagation
//! convention** (ASM/Fresnel transfer functions are $e^{+ikz\sqrt{1-\lambda^2 f^2}}$;
//! the on-axis phase $kz - \zeta(z)$ of the propagated waist was pinned by
//! the Phase-2 validation suite, `tests/phase2.rs`):
//!
//! $$u(\rho; z) = A\,\frac{w_0}{w(z)}\,
//!   \exp\!\Big(-\frac{\rho^2}{w(z)^2}\Big)\,
//!   \exp\!\Big(i\,\frac{k\rho^2}{2R(z)}\Big)\,
//!   e^{\,i(kz - \zeta(z))}, \qquad \rho^2 = (x-x_c)^2 + (y-y_c)^2$$
//!
//! with
//!
//! $$z_R = \frac{\pi w_0^2}{\lambda}, \quad
//!   w(z) = w_0\sqrt{1 + (z/z_R)^2}, \quad
//!   \frac{1}{R(z)} = \frac{z}{z^2 + z_R^2}, \quad
//!   \zeta(z) = \arctan(z/z_R), \quad k = \frac{2\pi}{\lambda}.$$
//!
//! Equivalently $u \propto \exp\!\big(i k \rho^2 / (2q)\big)$ with
//! $q(z) = z - i z_R$ — the conjugate of the usual $e^{-ikz}$-textbook
//! $q = z + i z_R$. Here `z` is the **signed distance from the waist to the
//! sampling plane** (meters), a plane-placement argument to [`gaussian`] and
//! to the radius/Gouy accessors, not a stored beam field: $z = 0$ samples the
//! waist (pure real Gaussian), $z > 0$ a diverging beam, $z < 0$ a converging
//! one, and $u(\rho; -z) = \overline{u(\rho; z)}$ (pin test below). The
//! curvature $1/R$ form is used so $z = 0$ is exact, not a limit.
//!
//! Sign conventions (Gouy, curvature) are the historical bug source: they are
//! pinned by cross-checks against the crate's own engines — Fresnel-TF
//! propagation of a waist beam must reproduce `gaussian` at $z$ to
//! discretization accuracy, and ASM to paraxial accuracy
//! ([`gaussian`] unit tests).
//!
//! # Sampling guidance
//!
//! The window should cover the beam without truncation: half-extents of
//! several $w(z)$ keep the power loss below $e^{-2(\text{extent}/w)^2}$ per
//! side, and the pitch should resolve the waist, $\Delta x \lesssim w_0/10$,
//! or the sampled profile (and its far field) is biased. Higher-order modes
//! spread to $\sim\sqrt{N}\,w(z)$ — widen the window accordingly.
//!
//! # Higher-order modes (Phase 5)
//!
//! [`hermite_gauss`] and [`laguerre_gauss`] sample the paraxial
//! $\mathrm{HG}_{mn}$ / $\mathrm{LG}_{p\ell}$ families on the same $q$-form
//! machinery as [`gaussian`] (Siegman/Goodman; the project textbook does
//! not cover them). Shared conventions:
//!
//! - **Envelope**: $w(z)$, $1/R(z)$ as in the fundamental mode; the Gouy
//!   phase enters as $e^{-i(N+1)\zeta(z)}$ with mode order $N = m + n$
//!   (HG) or $N = 2p + |\ell|$ (LG) — the $(N{+}1)$-fold of the pinned
//!   fundamental convention, verified by the eigenfunction tests below.
//! - **LG azimuthal phase** $e^{i\ell\varphi}$: with the crate's $e^{+ikz}$
//!   spatial convention this carries orbital angular momentum $+\ell\hbar$
//!   per photon along $+z$.
//! - **Normalization**: higher-order modes are **orthonormal** —
//!   $\iint |u|^2\,dA = A^2$ at every plane — in contrast to
//!   [`GaussianBeam`], whose `amplitude` is the peak field value
//!   ($P = A^2\pi w_0^2/2$). Consequently $\mathrm{HG}_{00}$ and
//!   $\mathrm{LG}_{00}$ equal a [`GaussianBeam`] of the same $w_0$ and $A$
//!   up to the constant $\sqrt{2/\pi}/w_0$ (pinned by unit test); mode
//!   power, not peak, is the natural scale here.
//! - Total order is capped at [`MAX_MODE_ORDER`] (sampled profiles beyond
//!   that are not resolvable in practice; [`SourcesError::InvalidMode`]).

use std::f64::consts::PI;

use num_complex::Complex64;
use thiserror::Error;

use crate::field::{BandField, Field, FieldError};

/// Errors produced by field initializers.
///
/// Beam-parameter problems are reported as [`SourcesError::InvalidBeam`]; a
/// non-finite sampling-plane distance as [`SourcesError::InvalidPlane`]; grid
/// and wavelength validity is delegated to [`FieldError`] (carried
/// transparently).
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum SourcesError {
    /// A Gaussian beam parameter violated its invariant: `w0` must be
    /// positive and finite; `amplitude` and the center coordinates must be
    /// finite (SI: meters, arbitrary coherent amplitude units).
    #[error(
        "invalid Gaussian beam: w0 = {w0} m (must be positive and finite), \
         amplitude = {amplitude}, center = ({cx}, {cy}) m (all must be finite)"
    )]
    InvalidBeam {
        /// Waist radius in meters.
        w0: f64,
        /// Peak field amplitude at the waist.
        amplitude: f64,
        /// x center of the waist in meters (field coordinates).
        cx: f64,
        /// y center of the waist in meters (field coordinates).
        cy: f64,
    },
    /// The signed waist-to-sampling-plane distance `z` was not finite
    /// (meters). Position is a sampling argument, not a beam field, so it is
    /// validated separately from the beam.
    #[error("invalid sampling plane: z = {z} m (must be finite)")]
    InvalidPlane {
        /// Signed waist-to-plane distance in meters.
        z: f64,
    },
    /// The mode order exceeded [`MAX_MODE_ORDER`] (HG: $m + n$; LG:
    /// $2p + |\ell|$). High-order sampled profiles are not resolvable in
    /// practice and the normalization factorials lose meaning; the reported
    /// order is the total mode order.
    #[error("mode order {order} exceeds the supported maximum {max}")]
    InvalidMode {
        /// Total mode order attempted.
        order: u32,
        /// The cap ([`MAX_MODE_ORDER`]).
        max: u32,
    },
    /// Grid or wavelength validity is delegated to [`FieldError`] (carried
    /// transparently).
    #[error(transparent)]
    Field(#[from] FieldError),
}

/// Gaussian beam parameter record (SI units, meters).
///
/// A pure data record describing the paraxial fundamental-mode beam *as
/// such*: waist radius, peak amplitude, and transverse center. It carries no
/// longitudinal position — the waist-to-plane distance `z` is a system
/// placement argument passed to [`gaussian`] and to the radius/Gouy accessors
/// (see the module's "position is a system property" note). Pass a beam to
/// [`gaussian`] to sample a [`BandField`] at a chosen plane and wavelength;
/// one record can generate many bands — an incoherent multiband source is a
/// collection of the resulting `BandField`s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianBeam {
    /// Waist radius $w_0$ in meters (field $1/e$ radius at the waist);
    /// positive and finite.
    pub w0: f64,
    /// Peak field amplitude $A$ at the waist (arbitrary coherent units;
    /// $|u|^2$ is irradiance up to a constant). Finite.
    pub amplitude: f64,
    /// Waist center $(x_c, y_c)$ in meters, field coordinates relative to
    /// the grid origin. Finite.
    pub center: (f64, f64),
}

impl GaussianBeam {
    /// Checks the beam invariants (see [`GaussianBeam`] field docs).
    ///
    /// # Errors
    /// [`SourcesError::InvalidBeam`] if any parameter is non-finite or
    /// `w0 <= 0`.
    pub fn validate(&self) -> Result<(), SourcesError> {
        let (cx, cy) = self.center;
        if !self.w0.is_finite()
            || self.w0 <= 0.0
            || !self.amplitude.is_finite()
            || !cx.is_finite()
            || !cy.is_finite()
        {
            return Err(SourcesError::InvalidBeam {
                w0: self.w0,
                amplitude: self.amplitude,
                cx,
                cy,
            });
        }
        Ok(())
    }

    /// Rayleigh range $z_R = \pi w_0^2 / \lambda$ in meters for the given
    /// wavelength (meters). This is the beam's intrinsic length scale and is
    /// independent of the sampling plane.
    pub fn rayleigh_range(&self, lambda: f64) -> f64 {
        PI * self.w0 * self.w0 / lambda
    }

    /// Beam radius $w(z) = w_0 \sqrt{1 + (z/z_R)^2}$ in meters measured at
    /// the sampling plane a signed distance `z` (meters) from the waist, for
    /// wavelength `lambda` (meters).
    pub fn beam_radius(&self, z: f64, lambda: f64) -> f64 {
        let zr = self.rayleigh_range(lambda);
        self.w0 * (1.0 + (z / zr).powi(2)).sqrt()
    }

    /// Gouy phase $\zeta(z) = \arctan(z / z_R)$ in radians accumulated at the
    /// sampling plane a signed distance `z` (meters) from the waist, for
    /// wavelength `lambda` (meters).
    pub fn gouy_phase(&self, z: f64, lambda: f64) -> f64 {
        (z / self.rayleigh_range(lambda)).atan()
    }
}

/// Samples a [`GaussianBeam`] onto an `nx` x `ny` grid at the plane a signed
/// distance `z` from the waist, at wavelength `lambda`, returning the
/// [`BandField`] (module docs give the exact formula and sign conventions).
///
/// # Arguments
/// - `beam`: waist radius, amplitude, and transverse center (meters).
/// - `z`: signed waist-to-sampling-plane distance in meters ($0$ = waist,
///   $>0$ diverging, $<0$ converging); must be finite. This is a system
///   placement argument, not a beam property.
/// - `lambda`: wavelength in meters; positive and finite.
/// - `nx`, `ny`: sample counts along $x$, $y$; both > 0.
/// - `dx`, `dy`: sample pitch in meters; positive and finite.
///
/// # Errors
/// [`SourcesError::InvalidBeam`] for bad beam parameters, then
/// [`SourcesError::InvalidPlane`] for a non-finite `z`, then [`FieldError`]
/// (wrapped) for invalid wavelength, grid, or non-finite samples.
///
/// # Sampling
/// See the module-level guidance: resolve the waist (`dx <= w0/10`) and cover
/// several $w(z)$ per half-window, or the truncation/bias shows up in the
/// power and far field.
// The full grid specification (nx, ny, dx, dy) mirrors BandField constructors;
// a grid-spec struct is deferred until more initializers exist (Phase 5).
#[allow(clippy::too_many_arguments)]
pub fn gaussian(
    beam: &GaussianBeam,
    z: f64,
    lambda: f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Result<BandField, SourcesError> {
    beam.validate()?;
    if !z.is_finite() {
        return Err(SourcesError::InvalidPlane { z });
    }
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(FieldError::NonPositiveWavelength(lambda).into());
    }
    let z_r = beam.rayleigh_range(lambda);
    let w_z = beam.beam_radius(z, lambda);
    let k = 2.0 * PI / lambda;
    // 1/R(z) written as z/(z^2 + z_R^2): exact and finite at z = 0 (flat
    // wavefront), no division-by-infinity branch needed.
    let inv_r = z / (z * z + z_r * z_r);
    let carrier = k * z - beam.gouy_phase(z, lambda);
    let amp = beam.amplitude * beam.w0 / w_z;
    let mut samples = Vec::with_capacity(nx.saturating_mul(ny));
    for iy in 0..ny {
        let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dy - beam.center.1;
        for ix in 0..nx {
            let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx - beam.center.0;
            let rho2 = x * x + y * y;
            let envelope = amp * (-rho2 / (w_z * w_z)).exp();
            let phase = carrier + 0.5 * k * rho2 * inv_r;
            samples.push(envelope * (phase * Complex64::I).exp());
        }
    }
    let field = Field::new(samples, nx, ny, dx, dy)?;
    Ok(BandField::new(lambda, field)?)
}

/// Maximum total mode order accepted by [`hermite_gauss`] and
/// [`laguerre_gauss`] (HG: $m + n$; LG: $2p + |\ell|$).
pub const MAX_MODE_ORDER: u32 = 64;

/// Shared $q$-form envelope quantities for one sampling plane:
/// $(z_R, w(z), 1/R(z))$ — see the module's Gaussian-beam convention.
fn envelope(w0: f64, z: f64, lambda: f64) -> (f64, f64, f64) {
    let z_r = PI * w0 * w0 / lambda;
    let w_z = w0 * (1.0 + (z / z_r).powi(2)).sqrt();
    // 1/R(z) as z/(z^2 + z_R^2): exact and finite at z = 0.
    let inv_r = z / (z * z + z_r * z_r);
    (z_r, w_z, inv_r)
}

/// $n!$ as f64 (empty product 1 at $n = 0$); exact well past the order cap.
fn factorial(n: u32) -> f64 {
    (1..=n).fold(1.0, |acc, k| acc * k as f64)
}

/// Physicists' Hermite polynomial $H_m(x)$ via the three-term recurrence
/// $H_{k+1} = 2x H_k - 2k H_{k-1}$.
fn hermite(m: u32, x: f64) -> f64 {
    match m {
        0 => 1.0,
        1 => 2.0 * x,
        _ => {
            let (mut h_prev, mut h) = (1.0, 2.0 * x);
            for k in 1..m {
                let next = 2.0 * x * h - 2.0 * (k as f64) * h_prev;
                h_prev = h;
                h = next;
            }
            h
        }
    }
}

/// Associated Laguerre polynomial $L_p^{\alpha}(x)$ via the recurrence
/// $(k+1)L_{k+1}^{\alpha} = (2k + \alpha + 1 - x)L_k^{\alpha} - (k+\alpha)L_{k-1}^{\alpha}$.
fn laguerre(p: u32, alpha: u32, x: f64) -> f64 {
    let a = alpha as f64;
    match p {
        0 => 1.0,
        1 => 1.0 + a - x,
        _ => {
            let (mut l_prev, mut l) = (1.0, 1.0 + a - x);
            for k in 1..p {
                let kf = k as f64;
                let next = ((2.0 * kf + a + 1.0 - x) * l - (kf + a) * l_prev) / (kf + 1.0);
                l_prev = l;
                l = next;
            }
            l
        }
    }
}

/// Hermite-Gauss beam parameter record (SI units, meters).
///
/// Describes the paraxial $\mathrm{HG}_{mn}$ mode *as such*: waist radius,
/// amplitude scale, mode orders, and transverse center. Like
/// [`GaussianBeam`] it carries **no longitudinal position** — the signed
/// waist-to-plane distance `z` is a sampling argument to [`hermite_gauss`].
///
/// Normalization (module docs): the transverse profile is **orthonormal**,
/// $\iint |u|^2\,dA = A^2$ at every plane — `amplitude` is $\sqrt{P}$, not
/// the peak field. $\mathrm{HG}_{00}$ equals [`gaussian`] up to the
/// constant $\sqrt{2/\pi}/w_0$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HermiteGaussBeam {
    /// Waist radius $w_0$ in meters (fundamental-mode $1/e$ radius that
    /// sets the family's scale); positive and finite.
    pub w0: f64,
    /// Hermite order along $x$ ($\ge 0$, $m + n \le$ [`MAX_MODE_ORDER`]).
    pub m: u32,
    /// Hermite order along $y$ ($\ge 0$, $m + n \le$ [`MAX_MODE_ORDER`]).
    pub n: u32,
    /// Amplitude scale $A$ (coherent units; mode power is $A^2$). Finite.
    pub amplitude: f64,
    /// Waist center $(x_c, y_c)$ in meters, field coordinates relative to
    /// the grid origin. Finite.
    pub center: (f64, f64),
}

impl HermiteGaussBeam {
    /// Checks the beam invariants (parameters; the order cap is validated
    /// at sampling time as [`SourcesError::InvalidMode`]).
    ///
    /// # Errors
    /// [`SourcesError::InvalidBeam`] if any parameter is non-finite or
    /// `w0 <= 0`.
    pub fn validate(&self) -> Result<(), SourcesError> {
        let (cx, cy) = self.center;
        if !self.w0.is_finite()
            || self.w0 <= 0.0
            || !self.amplitude.is_finite()
            || !cx.is_finite()
            || !cy.is_finite()
        {
            return Err(SourcesError::InvalidBeam {
                w0: self.w0,
                amplitude: self.amplitude,
                cx,
                cy,
            });
        }
        Ok(())
    }

    /// Total mode order $N = m + n$ (Gouy phase $(N+1)\zeta$).
    pub fn order(&self) -> u32 {
        self.m + self.n
    }
}

/// Laguerre-Gauss beam parameter record (SI units, meters).
///
/// The paraxial $\mathrm{LG}_{p\ell}$ mode: waist radius, amplitude scale,
/// radial order $p \ge 0$, azimuthal order $\ell \in \mathbb{Z}$ (sign =
/// OAM handedness, module docs), and transverse center. No longitudinal
/// position — `z` is a sampling argument to [`laguerre_gauss`].
///
/// Normalization as [`HermiteGaussBeam`]: orthonormal profile, mode power
/// $A^2$; $\mathrm{LG}_{00} = \mathrm{HG}_{00}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaguerreGaussBeam {
    /// Waist radius $w_0$ in meters; positive and finite.
    pub w0: f64,
    /// Radial order $p \ge 0$ ($2p + |\ell| \le$ [`MAX_MODE_ORDER`]).
    pub p: u32,
    /// Azimuthal order $\ell$ (any integer; $2p + |\ell| \le$
    /// [`MAX_MODE_ORDER`]). $e^{i\ell\varphi}$ winding, OAM $+\ell\hbar$
    /// per photon along $+z$ (module docs).
    pub l: i32,
    /// Amplitude scale $A$ (coherent units; mode power is $A^2$). Finite.
    pub amplitude: f64,
    /// Waist center $(x_c, y_c)$ in meters, field coordinates relative to
    /// the grid origin. Finite.
    pub center: (f64, f64),
}

impl LaguerreGaussBeam {
    /// Checks the beam invariants (parameters; the order cap is validated
    /// at sampling time as [`SourcesError::InvalidMode`]).
    ///
    /// # Errors
    /// [`SourcesError::InvalidBeam`] if any parameter is non-finite or
    /// `w0 <= 0`.
    pub fn validate(&self) -> Result<(), SourcesError> {
        let (cx, cy) = self.center;
        if !self.w0.is_finite()
            || self.w0 <= 0.0
            || !self.amplitude.is_finite()
            || !cx.is_finite()
            || !cy.is_finite()
        {
            return Err(SourcesError::InvalidBeam {
                w0: self.w0,
                amplitude: self.amplitude,
                cx,
                cy,
            });
        }
        Ok(())
    }

    /// Total mode order $N = 2p + |\ell|$ (Gouy phase $(N+1)\zeta$).
    pub fn order(&self) -> u32 {
        2 * self.p + self.l.unsigned_abs()
    }
}

/// Samples a [`HermiteGaussBeam`] onto an `nx` x `ny` grid at the plane a
/// signed distance `z` from the waist, at wavelength `lambda`:
///
/// $$u_{mn}(x, y; z) = A\sqrt{\frac{2}{\pi\,2^{m+n}m!\,n!}}\,
///   \frac{H_m\!\big(\tfrac{\sqrt2\,x}{w}\big)H_n\!\big(\tfrac{\sqrt2\,y}{w}\big)}{w(z)}\,
///   e^{-\rho^2/w(z)^2}\,e^{\,ik\rho^2/2R(z)}\,e^{\,i(kz - (m+n+1)\zeta(z))},$$
///
/// $\rho^2 = (x-x_c)^2 + (y-y_c)^2$, $w/R/\zeta$ as in [`gaussian`]
/// (module docs for conventions; Siegman/Goodman normalization).
///
/// # Errors
/// [`SourcesError::InvalidBeam`], then [`SourcesError::InvalidPlane`],
/// then [`SourcesError::InvalidMode`] ($m + n >$ [`MAX_MODE_ORDER`]), then
/// [`FieldError`] (wrapped) for invalid wavelength/grid.
#[allow(clippy::too_many_arguments)]
pub fn hermite_gauss(
    beam: &HermiteGaussBeam,
    z: f64,
    lambda: f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Result<BandField, SourcesError> {
    beam.validate()?;
    if !z.is_finite() {
        return Err(SourcesError::InvalidPlane { z });
    }
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(FieldError::NonPositiveWavelength(lambda).into());
    }
    let order = beam.order();
    if order > MAX_MODE_ORDER {
        return Err(SourcesError::InvalidMode {
            order,
            max: MAX_MODE_ORDER,
        });
    }
    let (z_r, w_z, inv_r) = envelope(beam.w0, z, lambda);
    let k = 2.0 * PI / lambda;
    let gouy = (z / z_r).atan() * (order as f64 + 1.0);
    let carrier = k * z - gouy;
    let norm = beam.amplitude
        * (2.0 / (PI * 2f64.powi(order as i32) * factorial(beam.m) * factorial(beam.n))).sqrt()
        / w_z;
    let mut samples = Vec::with_capacity(nx.saturating_mul(ny));
    for iy in 0..ny {
        let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dy - beam.center.1;
        let hn = hermite_scaled(beam.n, y, w_z);
        for ix in 0..nx {
            let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx - beam.center.0;
            let hm = hermite_scaled(beam.m, x, w_z);
            let rho2 = x * x + y * y;
            let env = norm * hm * hn * (-rho2 / (w_z * w_z)).exp();
            let phase = carrier + 0.5 * k * rho2 * inv_r;
            samples.push(env * (phase * Complex64::I).exp());
        }
    }
    let field = Field::new(samples, nx, ny, dx, dy)?;
    Ok(BandField::new(lambda, field)?)
}

/// $H_m(\sqrt2\,x/w)$ — argument pre-scaled helper.
fn hermite_scaled(m: u32, x: f64, w: f64) -> f64 {
    hermite(m, std::f64::consts::SQRT_2 * x / w)
}

/// Samples a [`LaguerreGaussBeam`] onto an `nx` x `ny` grid at the plane a
/// signed distance `z` from the waist, at wavelength `lambda`:
///
/// $$u_{p\ell}(r, \varphi; z) = A\sqrt{\frac{2\,p!}{\pi\,(p+|\ell|)!}}\,
///   \frac{1}{w(z)}\Big(\frac{\sqrt2\,r}{w}\Big)^{|\ell|}
///   L_p^{|\ell|}\!\Big(\frac{2r^2}{w^2}\Big)\,
///   e^{-r^2/w^2}\,e^{\,ikr^2/2R}\,e^{\,i\ell\varphi}\,
///   e^{\,i(kz - (2p+|\ell|+1)\zeta)},$$
///
/// $r, \varphi$ center-relative polar coordinates (module docs for
/// conventions; $\mathrm{LG}_{00} = \mathrm{HG}_{00}$).
///
/// # Errors
/// [`SourcesError::InvalidBeam`], then [`SourcesError::InvalidPlane`],
/// then [`SourcesError::InvalidMode`] ($2p + |\ell| >$ [`MAX_MODE_ORDER`]),
/// then [`FieldError`] (wrapped) for invalid wavelength/grid.
#[allow(clippy::too_many_arguments)]
pub fn laguerre_gauss(
    beam: &LaguerreGaussBeam,
    z: f64,
    lambda: f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Result<BandField, SourcesError> {
    beam.validate()?;
    if !z.is_finite() {
        return Err(SourcesError::InvalidPlane { z });
    }
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(FieldError::NonPositiveWavelength(lambda).into());
    }
    let order = beam.order();
    if order > MAX_MODE_ORDER {
        return Err(SourcesError::InvalidMode {
            order,
            max: MAX_MODE_ORDER,
        });
    }
    let abs_l = beam.l.unsigned_abs();
    let (z_r, w_z, inv_r) = envelope(beam.w0, z, lambda);
    let k = 2.0 * PI / lambda;
    let gouy = (z / z_r).atan() * (order as f64 + 1.0);
    let carrier = k * z - gouy;
    let norm =
        beam.amplitude * (2.0 * factorial(beam.p) / (PI * factorial(beam.p + abs_l))).sqrt() / w_z;
    let mut samples = Vec::with_capacity(nx.saturating_mul(ny));
    for iy in 0..ny {
        let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * dy - beam.center.1;
        for ix in 0..nx {
            let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * dx - beam.center.0;
            let rho2 = x * x + y * y;
            let scaled = 2.0 * rho2 / (w_z * w_z);
            let radial = if abs_l == 0 {
                1.0
            } else {
                (std::f64::consts::SQRT_2 * rho2.sqrt() / w_z).powi(abs_l as i32)
            };
            let prof = norm * radial * laguerre(beam.p, abs_l, scaled) * (-scaled / 2.0).exp();
            let phi = y.atan2(x);
            let phase = carrier + 0.5 * k * rho2 * inv_r + (beam.l as f64) * phi;
            samples.push(prof * (phase * Complex64::I).exp());
        }
    }
    let field = Field::new(samples, nx, ny, dx, dy)?;
    Ok(BandField::new(lambda, field)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::{asm, fresnel};

    const LAMBDA: f64 = 500e-9;

    fn beam(w0: f64) -> GaussianBeam {
        GaussianBeam {
            w0,
            amplitude: 1.0,
            center: (0.0, 0.0),
        }
    }

    /// Relative Frobenius norm ||a - b|| / ||b|| over the sample buffers.
    fn rel_frobenius(a: &BandField, b: &BandField) -> f64 {
        let diff: f64 = a
            .field()
            .samples()
            .iter()
            .zip(b.field().samples())
            .map(|(x, y)| (x - y).norm_sqr())
            .sum();
        let norm: f64 = b.field().samples().iter().map(|y| y.norm_sqr()).sum();
        diff.sqrt() / norm.sqrt()
    }

    fn wrap(phase: f64) -> f64 {
        (phase + PI).rem_euclid(2.0 * PI) - PI
    }

    #[test]
    fn waist_plane_is_real_gaussian_with_analytic_power() {
        // w0 = 1 mm, 20 samples per w0, half-window = 5 w0: truncation loss
        // ~ e^{-50} per side, discretization error tiny.
        let w0 = 1e-3;
        let d = w0 / 20.0;
        let n = 201usize;
        let band = gaussian(&beam(w0), 0.0, LAMBDA, n, n, d, d).unwrap();
        let f = band.field();
        // Sample values match A exp(-rho^2/w0^2) exactly (imaginary part 0).
        for (ix, iy) in [(0usize, 0usize), (100, 100), (200, 200), (37, 160)] {
            let (x, y) = f.coord(ix, iy);
            let expected = (-((x * x + y * y) / (w0 * w0))).exp();
            let s = f.sample(ix, iy);
            assert!((s.im).abs() < 1e-15);
            assert!((s.re - expected).abs() < 1e-12);
        }
        // P = A^2 * pi w0^2 / 2 (integral of exp(-2 rho^2/w0^2)).
        let expected_power = PI * w0 * w0 / 2.0;
        let rel = (f.power() - expected_power).abs() / expected_power;
        assert!(rel < 1e-3, "power rel. error {rel:e}");
    }

    #[test]
    fn on_axis_amplitude_and_gouy_match_q_form() {
        let w0 = 0.5e-3;
        let z = 0.25;
        let n = 129usize;
        let d = w0 / 20.0;
        let src = GaussianBeam {
            w0,
            amplitude: 2.0,
            center: (0.0, 0.0),
        };
        let band = gaussian(&src, z, LAMBDA, n, n, d, d).unwrap();
        let c = band.field().sample(n / 2, n / 2); // odd grid: origin sample
        let z_r = src.rayleigh_range(LAMBDA);
        let w_z = src.beam_radius(z, LAMBDA);
        let amp_exp = src.amplitude * w0 / w_z;
        assert!((c.norm() - amp_exp).abs() / amp_exp < 1e-12);
        let phase_exp = 2.0 * PI * z / LAMBDA - (z / z_r).atan();
        assert!(wrap(c.arg() - phase_exp).abs() < 1e-9);
        // The accessors are pure functions of z: w(0) = w0, gouy(0) = 0.
        assert!((src.beam_radius(0.0, LAMBDA) - w0).abs() < 1e-15);
        assert_eq!(src.gouy_phase(0.0, LAMBDA), 0.0);
    }

    #[test]
    fn analytic_beam_matches_own_engines() {
        // Convention pin: propagate a waist beam by z with the engines and
        // compare to the analytic initializer sampled at z. Fresnel TF is
        // exact for the paraxial Gaussian (its eigenfunction) up to
        // discretization; ASM differs by the (tiny) non-paraxial remainder.
        let w0 = 0.15e-3;
        let z = 0.1;
        let n = 255usize;
        let d = 1e-5;
        let analytic = gaussian(&beam(w0), z, LAMBDA, n, n, d, d).unwrap();
        let waist_band = gaussian(&beam(w0), 0.0, LAMBDA, n, n, d, d).unwrap();

        let fr = fresnel::propagate(
            &waist_band,
            z,
            &fresnel::FresnelConfig {
                padding: 2.0,
                strict: false,
                ..Default::default()
            },
        )
        .unwrap();
        let asm_out = asm::propagate(&waist_band, z, &asm::AsmConfig::default()).unwrap();

        let rel_fr = rel_frobenius(&fr, &analytic);
        let rel_asm = rel_frobenius(&asm_out, &analytic);
        assert!(rel_fr < 1e-3, "Fresnel TF vs analytic: {rel_fr:e}");
        assert!(rel_asm < 1e-3, "ASM vs analytic: {rel_asm:e}");
    }

    #[test]
    fn negative_z_is_complex_conjugate() {
        // u(rho; -z) = conj(u(rho; z)): converging is the mirror of diverging.
        let w0 = 0.5e-3;
        let z = 0.2;
        let at = |zz: f64| gaussian(&beam(w0), zz, LAMBDA, 64, 64, 1e-5, 1e-5).unwrap();
        let pos = at(z);
        let neg = at(-z);
        for (p, n) in pos.field().samples().iter().zip(neg.field().samples()) {
            assert!((p.conj() - n).norm() < 1e-12);
        }
    }

    #[test]
    fn off_center_waist_translates_by_integer_samples() {
        let w0 = 1e-3;
        let d = w0 / 20.0;
        let n = 161usize;
        let on_axis = gaussian(&beam(w0), 0.0, LAMBDA, n, n, d, d).unwrap();
        let shifted = gaussian(
            &GaussianBeam {
                w0,
                amplitude: 1.0,
                center: (3.0 * d, -2.0 * d),
            },
            0.0,
            LAMBDA,
            n,
            n,
            d,
            d,
        )
        .unwrap();
        // Beam moved +3 bins in x, -2 in y: shifted(ix) = on_axis(ix-3),
        // since u_shifted(x) = u(x - x_c) translates the profile along +x_c.
        for iy in 5..n - 5 {
            for ix in 5..n - 5 {
                let sx = ix as isize - 3;
                let sy = iy as isize + 2;
                if sx >= 0 && sy >= 0 && (sx as usize) < n && (sy as usize) < n {
                    let a = shifted.field().sample(ix, iy);
                    let b = on_axis.field().sample(sx as usize, sy as usize);
                    assert!((a - b).norm() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn validation_rejects_bad_parameters() {
        let bad = [
            GaussianBeam {
                w0: 0.0,
                ..beam(1e-3)
            },
            GaussianBeam {
                w0: -1e-3,
                ..beam(1e-3)
            },
            GaussianBeam {
                w0: f64::NAN,
                ..beam(1e-3)
            },
            GaussianBeam {
                amplitude: f64::INFINITY,
                ..beam(1e-3)
            },
            GaussianBeam {
                center: (0.0, f64::INFINITY),
                ..beam(1e-3)
            },
        ];
        for b in &bad {
            assert!(matches!(
                gaussian(b, 0.0, LAMBDA, 8, 8, 1e-4, 1e-4),
                Err(SourcesError::InvalidBeam { .. })
            ));
        }
        // Non-finite sampling-plane distance is rejected separately (z is a
        // placement argument, validated after the beam).
        for z in [f64::NAN, f64::INFINITY] {
            assert!(matches!(
                gaussian(&beam(1e-3), z, LAMBDA, 8, 8, 1e-4, 1e-4),
                Err(SourcesError::InvalidPlane { .. })
            ));
        }
        // Wavelength and grid errors come through as Field(_).
        assert!(matches!(
            gaussian(&beam(1e-3), 0.0, 0.0, 8, 8, 1e-4, 1e-4),
            Err(SourcesError::Field(FieldError::NonPositiveWavelength(0.0)))
        ));
        assert!(matches!(
            gaussian(&beam(1e-3), 0.0, LAMBDA, 0, 8, 1e-4, 1e-4),
            Err(SourcesError::Field(FieldError::EmptyGrid))
        ));
        assert!(matches!(
            gaussian(&beam(1e-3), 0.0, LAMBDA, 8, 8, f64::NAN, 1e-4),
            Err(SourcesError::Field(FieldError::NonPositivePitch { .. }))
        ));
    }

    // --- Higher-order modes (Phase 5, task a). ---

    fn hg(w0: f64, m: u32, n: u32) -> HermiteGaussBeam {
        HermiteGaussBeam {
            w0,
            m,
            n,
            amplitude: 1.0,
            center: (0.0, 0.0),
        }
    }

    fn lg(w0: f64, p: u32, l: i32) -> LaguerreGaussBeam {
        LaguerreGaussBeam {
            w0,
            p,
            l,
            amplitude: 1.0,
            center: (0.0, 0.0),
        }
    }

    /// Discrete inner product <a|b> = sum conj(a) b * dx dy.
    fn inner(a: &BandField, b: &BandField) -> Complex64 {
        let (dx, dy) = a.field().pitch();
        let s: Complex64 = a
            .field()
            .samples()
            .iter()
            .zip(b.field().samples())
            .map(|(x, y)| x.conj() * *y)
            .sum();
        s * (dx * dy)
    }

    #[test]
    fn fundamentals_match_scaled_gaussian() {
        // HG00 = LG00 = gaussian * sqrt(2/pi)/w0 (orthonormal vs peak
        // convention, module docs) at waist and off-waist alike.
        let w0 = 0.5e-3;
        let d = w0 / 20.0;
        let n = 161usize;
        let ratio = (2.0 / PI).sqrt() / w0;
        for z in [0.0, 0.1] {
            let g = gaussian(&beam(w0), z, LAMBDA, n, n, d, d).unwrap();
            let h = hermite_gauss(&hg(w0, 0, 0), z, LAMBDA, n, n, d, d).unwrap();
            let l = laguerre_gauss(&lg(w0, 0, 0), z, LAMBDA, n, n, d, d).unwrap();
            for (gi, hi) in g.field().samples().iter().zip(h.field().samples()) {
                assert!((hi - gi * ratio).norm() < 1e-12);
            }
            for (gi, li) in g.field().samples().iter().zip(l.field().samples()) {
                assert!((li - gi * ratio).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn hg_modes_are_orthonormal() {
        // Half-window 6 w0, 20 bins per w0: truncation and discretization
        // both below 1e-3 for orders up to 2.
        let w0 = 0.5e-3;
        let d = w0 / 20.0;
        let n = 241usize;
        let modes = [
            hermite_gauss(&hg(w0, 0, 0), 0.0, LAMBDA, n, n, d, d).unwrap(),
            hermite_gauss(&hg(w0, 1, 0), 0.0, LAMBDA, n, n, d, d).unwrap(),
            hermite_gauss(&hg(w0, 0, 1), 0.0, LAMBDA, n, n, d, d).unwrap(),
            hermite_gauss(&hg(w0, 2, 0), 0.0, LAMBDA, n, n, d, d).unwrap(),
        ];
        for (i, a) in modes.iter().enumerate() {
            for (j, b) in modes.iter().enumerate() {
                let ip = inner(a, b);
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (ip.norm() - expect).abs() < 1e-3,
                    "<{i}|{j}> = {} vs {expect}",
                    ip.norm()
                );
            }
        }
        // Power is invariant under propagation (orthonormality at any z).
        let h10 = hermite_gauss(&hg(w0, 1, 0), 0.0, LAMBDA, n, n, d, d).unwrap();
        let h10z = hermite_gauss(&hg(w0, 1, 0), 0.2, LAMBDA, n, n, d, d).unwrap();
        assert!((h10z.field().power() - h10.field().power()).abs() / h10.field().power() < 1e-3);
    }

    #[test]
    fn hg_parity_follows_order() {
        // H_m(-x) = (-1)^m H_m(x): u_mn reflected in x picks up (-1)^m.
        let w0 = 0.5e-3;
        let d = w0 / 16.0;
        let n = 129usize; // odd grid: reflection about the center sample
        for m in 0..4u32 {
            let u = hermite_gauss(&hg(w0, m, 0), 0.0, LAMBDA, n, n, d, d).unwrap();
            let c = n / 2;
            for iy in 0..n {
                for ix in 0..n {
                    let a = u.field().sample(ix, iy);
                    let b = u.field().sample(2 * c - ix, iy);
                    let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                    assert!((a - b * sign).norm() < 1e-12, "m={m} at {ix},{iy}");
                }
            }
        }
    }

    #[test]
    fn lg_ring_radius_and_vortex_winding() {
        // LG_01 intensity peaks at r = w0/sqrt(2) (maximize r^2 e^{-2r^2/w^2}).
        let w0 = 1e-3;
        let d = w0 / 40.0;
        let n = 201usize;
        let u = laguerre_gauss(&lg(w0, 0, 1), 0.0, LAMBDA, n, n, d, d).unwrap();
        let mut r_peak = 0.0f64;
        let mut best = -1.0f64;
        for (k, s) in u.field().samples().iter().enumerate() {
            let iy = k / n;
            let ix = k % n;
            let (x, y) = (
                (ix as f64 - (n as f64 - 1.0) / 2.0) * d,
                (iy as f64 - (n as f64 - 1.0) / 2.0) * d,
            );
            let r = (x * x + y * y).sqrt();
            if s.norm_sqr() > best {
                best = s.norm_sqr();
                r_peak = r;
            }
        }
        let expect = w0 / (2.0f64).sqrt();
        assert!(
            (r_peak - expect).abs() < 2.0 * d,
            "ring radius {r_peak:e} vs {expect:e}"
        );

        // Phase winding: traversing a circle of radius 2 w0 once accumulates
        // 2*pi*l for both l = +1 and l = -1 (sign convention, module docs).
        for l in [1i32, -1] {
            let u = laguerre_gauss(&lg(w0, 0, l), 0.0, LAMBDA, n, n, d, d).unwrap();
            let r = 2.0 * w0;
            let steps = 512usize;
            let mut total = 0.0f64;
            let mut prev = None;
            for s in 0..=steps {
                let t = 2.0 * PI * s as f64 / steps as f64;
                let x = r * t.cos();
                let y = r * t.sin();
                let ix = (x / d + (n as f64 - 1.0) / 2.0).round() as usize;
                let iy = (y / d + (n as f64 - 1.0) / 2.0).round() as usize;
                let ph = u.field().sample(ix, iy).arg();
                if let Some(p) = prev {
                    total += wrap(ph - p);
                }
                prev = Some(ph);
            }
            assert!(
                (total - 2.0 * PI * l as f64).abs() < 1e-6,
                "winding for l={l}: {total}"
            );
        }
    }

    #[test]
    fn mode_gouy_order_scales_on_axis() {
        // On-axis (rho = 0) the curvature phase vanishes and the carrier is
        // k z - (N+1) zeta(z): N = 2 modes LG10 (L_1(0) = +1) and HG20
        // (H_2(0) = -2, an extra pi) all carry the tripling; HG11 (odd x
        // AND odd y) and LG01 (vortex) vanish on axis.
        let w0 = 0.5e-3;
        let z = 0.25;
        let d = w0 / 20.0;
        let n = 129usize;
        let z_r = PI * w0 * w0 / LAMBDA;
        let carrier = 2.0 * PI * z / LAMBDA - 3.0 * (z / z_r).atan();
        let lg10 = laguerre_gauss(&lg(w0, 1, 0), z, LAMBDA, n, n, d, d).unwrap();
        let c = lg10.field().sample(n / 2, n / 2);
        assert!(c.norm() > 1e-6, "LG10 on-axis should not vanish");
        assert!(
            wrap(c.arg() - carrier).abs() < 1e-9,
            "LG10 phase {}",
            c.arg()
        );
        let hg20 = hermite_gauss(&hg(w0, 2, 0), z, LAMBDA, n, n, d, d).unwrap();
        let c = hg20.field().sample(n / 2, n / 2);
        assert!(c.norm() > 1e-6, "HG20 on-axis should not vanish");
        assert!(
            wrap(c.arg() - carrier - PI).abs() < 1e-9,
            "HG20 phase {} (expect carrier + pi)",
            c.arg()
        );
        for u in [
            hermite_gauss(&hg(w0, 1, 1), z, LAMBDA, n, n, d, d).unwrap(),
            laguerre_gauss(&lg(w0, 0, 1), z, LAMBDA, n, n, d, d).unwrap(),
        ] {
            assert!(u.field().sample(n / 2, n / 2).norm() < 1e-12);
        }
    }

    #[test]
    fn modes_propagate_as_eigenfunctions() {
        // Convention pin (same role as the gaussian engine cross-check):
        // Fresnel TF and ASM propagation of a waist mode reproduce the
        // analytic mode at z — HG and LG are exact paraxial solutions, so
        // any Gouy/curvature/winding sign error shows up here.
        let w0 = 0.15e-3;
        let z = 0.1;
        let n = 255usize;
        let d = 1e-5;
        let cases: Vec<BandField> = vec![
            hermite_gauss(&hg(w0, 1, 0), 0.0, LAMBDA, n, n, d, d).unwrap(),
            hermite_gauss(&hg(w0, 2, 1), 0.0, LAMBDA, n, n, d, d).unwrap(),
            laguerre_gauss(&lg(w0, 0, 1), 0.0, LAMBDA, n, n, d, d).unwrap(),
            laguerre_gauss(&lg(w0, 1, -2), 0.0, LAMBDA, n, n, d, d).unwrap(),
        ];
        let analytic: Vec<BandField> = vec![
            hermite_gauss(&hg(w0, 1, 0), z, LAMBDA, n, n, d, d).unwrap(),
            hermite_gauss(&hg(w0, 2, 1), z, LAMBDA, n, n, d, d).unwrap(),
            laguerre_gauss(&lg(w0, 0, 1), z, LAMBDA, n, n, d, d).unwrap(),
            laguerre_gauss(&lg(w0, 1, -2), z, LAMBDA, n, n, d, d).unwrap(),
        ];
        let fcfg = fresnel::FresnelConfig {
            padding: 2.0,
            strict: false,
            ..Default::default()
        };
        for (waist, want) in cases.iter().zip(&analytic) {
            let fr = fresnel::propagate(waist, z, &fcfg).unwrap();
            let rel = rel_frobenius(&fr, want);
            assert!(rel < 1e-3, "Fresnel TF eigenfunction: {rel:e}");
            let asm_out = asm::propagate(waist, z, &asm::AsmConfig::default()).unwrap();
            let rel = rel_frobenius(&asm_out, want);
            assert!(rel < 1e-3, "ASM eigenfunction: {rel:e}");
        }
    }

    #[test]
    fn mode_validation() {
        // Beam parameters reuse InvalidBeam; orders cap at MAX_MODE_ORDER.
        let bad = HermiteGaussBeam {
            w0: 0.0,
            m: 1,
            n: 1,
            amplitude: 1.0,
            center: (0.0, 0.0),
        };
        assert!(matches!(
            hermite_gauss(&bad, 0.0, LAMBDA, 8, 8, 1e-4, 1e-4),
            Err(SourcesError::InvalidBeam { .. })
        ));
        assert!(matches!(
            hermite_gauss(&hg(1e-3, 40, 25), 0.0, LAMBDA, 8, 8, 1e-4, 1e-4),
            Err(SourcesError::InvalidMode { order: 65, max: 64 })
        ));
        assert!(matches!(
            laguerre_gauss(&lg(1e-3, 20, -25), 0.0, LAMBDA, 8, 8, 1e-4, 1e-4),
            Err(SourcesError::InvalidMode { order: 65, max: 64 })
        ));
        // Exactly at the cap: accepted.
        assert!(hermite_gauss(&hg(1e-3, 40, 24), 0.0, LAMBDA, 8, 8, 1e-4, 1e-4).is_ok());
        for z in [f64::NAN, f64::INFINITY] {
            assert!(matches!(
                laguerre_gauss(&lg(1e-3, 0, 1), z, LAMBDA, 8, 8, 1e-4, 1e-4),
                Err(SourcesError::InvalidPlane { .. })
            ));
        }
        assert!(matches!(
            hermite_gauss(&hg(1e-3, 1, 0), 0.0, -1.0, 8, 8, 1e-4, 1e-4),
            Err(SourcesError::Field(FieldError::NonPositiveWavelength(-1.0)))
        ));
    }
}
