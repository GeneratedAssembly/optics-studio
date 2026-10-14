//! Built-in glass catalog: published Sellmeier Book-1 coefficients
//! (n² = constant + Σ Bᵢλ²/(λ²−Cᵢ), λ in µm).
//!
//! Provenance is explicit because Zemax's proprietary catalogs (the
//! `GCAT` line names one) fit their own data: the built-in set reproduces
//! published room-temperature dispersion, and imported prescriptions'
//! paraxial invariants typically agree to ~0.5% EFL / a few % BFL —
//! external `.AGF` import is a pending follow-up (tasks/plan.md Phase 8).

use crate::agf::AgfCatalog;
use crate::error::ZemaxError;
use optics_core::element::Material;

/// One catalog entry.
#[derive(Clone, Copy, Debug)]
pub struct CatalogEntry {
    /// Catalog glass name (upper-case, as in `GLAS` lines).
    pub name: &'static str,
    /// Literature source of the coefficients.
    pub source: &'static str,
    /// Constant term of n².
    pub constant: f64,
    /// Sellmeier (B, C) pairs, C in µm².
    pub terms: [(f64, f64); 3],
}

/// Built-in entries (Schott-infrared relevant + common visible).
pub const BUILTIN: &[CatalogEntry] = &[
    CatalogEntry {
        name: "CAF2",
        source: "H.H. Li, J. Phys. Chem. Ref. Data 9, 161 (1980)",
        constant: 1.33973,
        terms: [
            (0.69913, 0.09374 * 0.09374),
            (0.11994, 21.18 * 21.18),
            (4.35181, 38.46 * 38.46),
        ],
    },
    CatalogEntry {
        name: "F_SILICA",
        source: "I.H. Malitson, Appl. Opt. 6, 1560 (1963)",
        constant: 1.0,
        terms: [
            (0.6961663, 0.0684043 * 0.0684043),
            (0.4079426, 0.1162414 * 0.1162414),
            (0.8974794, 9.896161 * 9.896161),
        ],
    },
    CatalogEntry {
        name: "N-BK7",
        source: "SCHOTT AG data sheet (Sellmeier)",
        constant: 1.0,
        terms: [
            (1.03961212, 0.00600069867),
            (0.232796349, 0.0200179144),
            (1.01041127, 103.560653),
        ],
    },
];

/// Active glass registry: built-in published entries plus optionally
/// loaded AGF catalogs (AGF entries take precedence on name collisions).
#[derive(Clone, Debug, Default)]
pub struct Registry {
    agf: Vec<crate::agf::AgfGlass>,
}

impl Registry {
    /// Built-in entries only.
    pub fn builtin() -> Self {
        Self::default()
    }

    /// Append/override with an AGF catalog's glasses.
    pub fn with_agf(mut self, catalog: &AgfCatalog) -> Self {
        for g in &catalog.glasses {
            self.agf.retain(|x| !x.name.eq_ignore_ascii_case(&g.name));
            self.agf.push(g.clone());
        }
        self
    }

    /// Resolve a name to a dispersive [`Material`] (AGF first, then
    /// built-in; AIR/empty = n 1).
    pub fn material(&self, name: Option<&str>) -> Result<Material, ZemaxError> {
        let up = name.unwrap_or("").trim().to_uppercase();
        if up.is_empty() || up == "AIR" || up == "VACUUM" {
            return Ok(Material::sellmeier_book1(1.0, []));
        }
        if let Some(g) = self.agf.iter().find(|g| g.name.to_uppercase() == up) {
            let g = g.clone();
            return Ok(Material::from_index_fn(move |lambda_m| {
                g.index_at(lambda_m * 1e6).unwrap_or(f64::NAN)
            }));
        }
        match lookup(&up)? {
            None => Ok(Material::sellmeier_book1(1.0, [])),
            Some(e) => Ok(Material::sellmeier_book1(e.constant, e.terms)),
        }
    }

    /// Index at `lambda_um` for any source (registry-wide).
    pub fn index_at(&self, name: Option<&str>, lambda_um: f64) -> Result<f64, ZemaxError> {
        let up = name.unwrap_or("").trim().to_uppercase();
        if up.is_empty() || up == "AIR" || up == "VACUUM" {
            return Ok(1.0);
        }
        if let Some(g) = self.agf.iter().find(|g| g.name.to_uppercase() == up) {
            return g.index_at(lambda_um);
        }
        match lookup(&up)? {
            None => Ok(1.0),
            Some(e) => Ok(index_at(e, lambda_um)),
        }
    }

    /// Validate all names used by a prescription resolve here.
    pub fn validate_glass(&self, name: Option<&str>) -> Result<(), ZemaxError> {
        self.index_at(name, 0.5876).map(|_| ())
    }
}

/// Refractive index at `lambda_um` (micrometers) for an entry.
pub fn index_at(entry: &CatalogEntry, lambda_um: f64) -> f64 {
    let l2 = lambda_um * lambda_um;
    let mut n2 = entry.constant;
    for (b, c) in entry.terms {
        n2 += b * l2 / (l2 - c);
    }
    n2.sqrt()
}

/// Look up an entry (case-insensitive). Air/vacuum/empty resolve to `None`
/// (no glass); unknown names error.
pub fn lookup(name: &str) -> Result<Option<&'static CatalogEntry>, ZemaxError> {
    let up = name.trim().to_uppercase();
    if up.is_empty() || up == "AIR" || up == "VACUUM" {
        return Ok(None);
    }
    BUILTIN
        .iter()
        .find(|e| e.name.eq_ignore_ascii_case(&up))
        .map(Some)
        .ok_or(ZemaxError::UnknownGlass(up))
}

/// `Material` for a catalog name (`None` names yield air: n = 1).
pub fn material(name: Option<&str>) -> Result<Material, ZemaxError> {
    match lookup(name.unwrap_or(""))? {
        None => Ok(Material::sellmeier_book1(1.0, [])),
        Some(e) => Ok(Material::sellmeier_book1(e.constant, e.terms)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_anchors() {
        let caf2 = lookup("CAF2").unwrap().unwrap();
        let sil = lookup("f_silica").unwrap().unwrap();
        let bk7 = lookup("N-BK7").unwrap().unwrap();
        // Well-known catalog points (±1e-3 tolerance covers fit variants).
        assert!((index_at(caf2, 0.5876) - 1.4338).abs() < 1e-3, "CaF2 nd");
        assert!((index_at(sil, 0.5876) - 1.4585).abs() < 1e-3, "silica nd");
        assert!((index_at(bk7, 0.5876) - 1.5168).abs() < 1e-3, "BK7 nd");
        // Blue-stronger law.
        assert!(index_at(caf2, 0.2825) > index_at(caf2, 0.55));
    }

    #[test]
    fn air_and_unknown() {
        assert!(lookup("AIR").unwrap().is_none());
        assert!(lookup("").unwrap().is_none());
        assert!(matches!(
            lookup("UNOBTAINIUM"),
            Err(ZemaxError::UnknownGlass(_))
        ));
        let m = material(Some("N-BK7")).unwrap();
        assert!((m.n_at(587.6e-9, 0.0) - 1.5168).abs() < 1e-3);
        let air = material(None).unwrap();
        assert!((air.n_at(500e-9, 0.0) - 1.0).abs() < 1e-12);
    }
}
