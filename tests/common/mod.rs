//! Shared analytic helpers for the physical validation suites.
//!
//! Deliberately dependency-free (project decision: no special-functions
//! crate): classical functions are computed by quadrature inside tests,
//! with the same care as the propagators — SI units, f64.

#![allow(dead_code)] // Shared toolbox: individual targets use a subset.

use num_complex::Complex64;
use optics_rs::field::{BandField, Field};
use std::f64::consts::PI;

pub const LAMBDA: f64 = 500e-9;

/// Wrap a phase difference (rad) into $[-\pi, \pi)$.
pub fn wrap_phase(d: f64) -> f64 {
    (d + PI).rem_euclid(2.0 * PI) - PI
}

/// Gaussian band $u = \exp(-(x^2 + y^2) / 2\sigma^2)$ centered at `c`
/// (field coordinates, m) on an origin-centered grid.
pub fn gaussian_band(nx: usize, ny: usize, d: f64, sigma: f64, c: (f64, f64)) -> BandField {
    let mut samples = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        let y = (iy as f64 - (ny as f64 - 1.0) / 2.0) * d - c.1;
        for ix in 0..nx {
            let x = (ix as f64 - (nx as f64 - 1.0) / 2.0) * d - c.0;
            samples.push(Complex64::from(
                (-((x * x + y * y) / (2.0 * sigma * sigma))).exp(),
            ));
        }
    }
    BandField::new(LAMBDA, Field::new(samples, nx, ny, d, d).unwrap()).unwrap()
}

/// Bessel $J_1$ via Simpson integration of
/// $J_1(x) = \frac{1}{\pi} \int_0^\pi \cos(x \sin t - t)\, dt$.
pub fn bessel_j1(x: f64) -> f64 {
    let n = 20_000usize;
    let h = PI / n as f64;
    let mut sum = 0.0f64;
    for i in 0..=n {
        let t = i as f64 * h;
        let f = (x * t.sin() - t).cos();
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * f;
    }
    sum * h / 3.0 / PI
}

/// Normalized Airy intensity $[2 J_1(v)/v]^2$ with the analytic limit at 0.
pub fn airy(v: f64) -> f64 {
    if v.abs() < 1e-8 {
        1.0 - v * v / 3.0
    } else {
        let j = 2.0 * bessel_j1(v) / v;
        j * j
    }
}

/// Unnormalized sinc $\sin(x)/x$ (limit 1 at 0).
pub fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0 - x * x / 6.0
    } else {
        x.sin() / x
    }
}

/// Fresnel integrals $C(u) = \int_0^u \cos(\pi t^2/2)\,dt$,
/// $S(u) = \int_0^u \sin(\pi t^2/2)\,dt$ for $u \ge 0$, via one cumulative
/// trapezoidal table (shared by all lookups; odd extensions for $u < 0$).
pub struct FresnelTable {
    dt: f64,
    c: Vec<f64>,
    s: Vec<f64>,
}

impl FresnelTable {
    /// Build a table over $[0, u_{\max}]$ with step `dt`.
    pub fn new(u_max: f64, dt: f64) -> Self {
        let n = (u_max / dt).ceil() as usize;
        let mut c = Vec::with_capacity(n + 1);
        let mut s = Vec::with_capacity(n + 1);
        let (mut cc, mut ss) = (0.0f64, 0.0f64);
        c.push(0.0);
        s.push(0.0);
        let f = |t: f64| ((PI * t * t / 2.0).cos(), (PI * t * t / 2.0).sin());
        let mut prev = f(0.0);
        for i in 1..=n {
            let t = i as f64 * dt;
            let cur = f(t);
            cc += 0.5 * dt * (prev.0 + cur.0);
            ss += 0.5 * dt * (prev.1 + cur.1);
            c.push(cc);
            s.push(ss);
            prev = cur;
        }
        Self { dt, c, s }
    }

    fn at(v: &[f64], dt: f64, u: f64) -> f64 {
        let x = (u / dt).clamp(0.0, v.len() as f64 - 1.0);
        let i = x.floor() as usize;
        let frac = x - i as f64;
        let a = v[i];
        let b = v[(i + 1).min(v.len() - 1)];
        a + (b - a) * frac
    }

    /// $C(u)$ (odd extension).
    pub fn c(&self, u: f64) -> f64 {
        let a = FresnelTable::at(&self.c, self.dt, u.abs());
        if u < 0.0 { -a } else { a }
    }

    /// $S(u)$ (odd extension).
    pub fn s(&self, u: f64) -> f64 {
        let a = FresnelTable::at(&self.s, self.dt, u.abs());
        if u < 0.0 { -a } else { a }
    }
}
