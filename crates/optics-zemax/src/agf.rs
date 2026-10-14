//! Zemax `.AGF` glass catalog import (text format).
//!
//! Layout (as written by HOYA/OHARA distributions — real files vary):
//!
//! - `NM <name> <model> ... <nd> <Vd> ...` — glass header. `nd`/`Vd` are
//!   located by value range (1.3..2.5 / 10..100) because vendors insert
//!   extra fields inconsistently (HOYA encodes `(nd-1)*1e6` before nd;
//!   OHARA sometimes a status flag).
//! - `CD <c0> ... <c9>` — dispersion coefficients (the line following NM):
//!   - model 1 (Schott): n² = c0 + c1·λ² + c2·λ⁻² + c3·λ⁻⁴ + c4·λ⁻⁶ + c5·λ⁻⁸
//!   - model 2 (Sellmeier): n² = 1 + Σᵢ Bᵢλ²/(λ²−Cᵢ), interleaved
//!     (B1,C1,B2,C2,B3,C3,B4,C4,B5,C5)
//!   - model 3 (Sellmeier1): n² = c0 + Σᵢ Bᵢλ²/(λ²−Cᵢ), interleaved after
//!     the constant (A0,B1,C1,B2,C2,B3,C3,B4,C4)
//!   - models 4/5 (power/polynomial): rejected.
//! - `GC/ED/TD/MD/OD/LD/IT/...` — comments and non-dispersion data:
//!   ignored.
//!
//! λ is in micrometers throughout (Zemax convention).

use crate::error::ZemaxError;

/// One parsed glass.
#[derive(Clone, Debug, PartialEq)]
pub struct AgfGlass {
    /// Glass name (`NM` field 2).
    pub name: String,
    /// Dispersion model (1 Schott, 2 Sellmeier, 3 Sellmeier1).
    pub model: u8,
    /// Declared d-line index (informational; the model is authoritative).
    pub nd: f64,
    /// Encoded `(nd - 1) * 1e6` field when present (vendor decimal fields
    /// contain typos: MC-TAF115's decimal disagrees with its own encoding).
    pub nd_encoded: Option<f64>,
    /// Declared Abbe number.
    pub vd: f64,
    /// `CD` coefficients.
    pub cd: [f64; 10],
}

impl AgfGlass {
    /// Refractive index at `lambda_um` (micrometers).
    ///
    /// # Errors
    /// [`ZemaxError::UnsupportedGlassModel`] for models 1/2/3 outside the
    /// CD data or models 4/5.
    pub fn index_at(&self, lambda_um: f64) -> Result<f64, ZemaxError> {
        let l2 = lambda_um * lambda_um;
        let n2 = match self.model {
            1 => {
                let mut v = self.cd[0] + self.cd[1] * l2;
                let mut p = 1.0 / l2;
                for k in 2..6 {
                    v += self.cd[k] * p;
                    p /= l2;
                }
                v
            }
            2 | 3 => {
                let start = if self.model == 3 { 1 } else { 0 };
                let mut v = if self.model == 3 { self.cd[0] } else { 1.0 };
                let mut i = start;
                while i + 1 < 10 {
                    let b = self.cd[i];
                    let c = self.cd[i + 1];
                    if b != 0.0 {
                        if l2 == c {
                            return Err(ZemaxError::BadGlass {
                                name: self.name.clone(),
                                reason: "wavelength on Sellmeier resonance".into(),
                            });
                        }
                        v += b * l2 / (l2 - c);
                    }
                    i += 2;
                }
                v
            }
            m => {
                return Err(ZemaxError::UnsupportedGlassModel {
                    name: self.name.clone(),
                    model: m,
                });
            }
        };
        Ok(n2.sqrt())
    }
}

/// A parsed catalog.
#[derive(Clone, Debug, Default)]
pub struct AgfCatalog {
    /// `CC` line text.
    pub comment: String,
    /// Glasses in file order.
    pub glasses: Vec<AgfGlass>,
}

impl AgfCatalog {
    /// Parse AGF text (CRLF-tolerant).
    pub fn parse(text: &str) -> Result<Self, ZemaxError> {
        let mut out = Self::default();
        let mut pending: Option<AgfGlass> = None;
        for raw in text.lines() {
            let line = raw.trim_end_matches('\r').trim_end();
            let mut it = line.split_whitespace();
            match it.next().unwrap_or("") {
                "CC" => out.comment = line[2..].trim().to_string(),
                "NM" => {
                    if let Some(g) = pending.take() {
                        out.glasses.push(g);
                    }
                    let name = it.next().unwrap_or("").to_string();
                    let model: u8 =
                        it.next()
                            .unwrap_or("0")
                            .parse()
                            .map_err(|_| ZemaxError::BadGlass {
                                name: name.clone(),
                                reason: "bad NM model".into(),
                            })?;
                    let rest: Vec<f64> = it.filter_map(|t| t.parse().ok()).collect();
                    let nd_i = rest.iter().position(|v| (1.3..2.5).contains(v));
                    let nd_encoded = rest
                        .iter()
                        .find(|v| (3.0e5..2.0e6).contains(*v))
                        .map(|v| 1.0 + v * 1e-6);
                    let (nd, vd) = match nd_i {
                        Some(i) if rest.len() > i + 1 => (rest[i], rest[i + 1]),
                        _ => {
                            return Err(ZemaxError::BadGlass {
                                name,
                                reason: "NM line missing nd/Vd".into(),
                            });
                        }
                    };
                    pending = Some(AgfGlass {
                        name,
                        model,
                        nd,
                        nd_encoded,
                        vd,
                        cd: [0.0; 10],
                    });
                }
                "CD" => {
                    let Some(g) = pending.as_mut() else { continue };
                    for (slot, v) in g.cd.iter_mut().zip(it.filter_map(|t| t.parse().ok())) {
                        *slot = v;
                    }
                }
                _ => {}
            }
        }
        if let Some(g) = pending {
            out.glasses.push(g);
        }
        Ok(out)
    }

    /// Case-insensitive lookup.
    pub fn find(&self, name: &str) -> Option<&AgfGlass> {
        self.glasses
            .iter()
            .find(|g| g.name.eq_ignore_ascii_case(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hoya() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/HOYA20260707.agf")
    }

    fn ohara() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/OHARA_260701.AGF")
    }

    #[test]
    fn anchors_from_vendor_lines() {
        let cat = AgfCatalog::parse(&std::fs::read_to_string(hoya()).unwrap()).unwrap();
        let fc5 = cat.find("FC5").unwrap();
        assert_eq!(fc5.model, 1);
        assert!((fc5.index_at(0.5876).unwrap() - 1.48749).abs() < 5e-4);
        let cat2 = AgfCatalog::parse(&std::fs::read_to_string(ohara()).unwrap()).unwrap();
        let pbm = cat2.find("PBM2R").unwrap();
        assert_eq!(pbm.model, 2);
        assert!((pbm.index_at(0.5876).unwrap() - 1.620040).abs() < 5e-4);
    }

    #[test]
    fn full_corpus_self_consistency() {
        // Every model-1/2/3 glass: the dispersion formula must reproduce
        // the NM-declared nd and Abbe Vd from the model itself.
        for path in [hoya(), ohara()] {
            let cat = AgfCatalog::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
            assert!(cat.glasses.len() > 400, "{}", cat.glasses.len());
            let mut checked = 0usize;
            for g in &cat.glasses {
                if g.model > 3 {
                    assert!(g.index_at(0.5876).is_err());
                    continue;
                }
                let nd = g.index_at(0.5876).unwrap();
                let nf = g.index_at(0.4861).unwrap();
                let ncc = g.index_at(0.6563).unwrap();
                let vd = (nd - 1.0) / (nf - ncc);
                // Vendor decimal nd fields contain typos; accept a match
                // against either declared value.
                let declared_ok =
                    (nd - g.nd).abs() < 1e-3 || g.nd_encoded.is_some_and(|e| (nd - e).abs() < 1e-3);
                assert!(
                    declared_ok && (vd - g.vd).abs() < 0.15,
                    "{}: nd {nd} vs {}/{} vd {vd} vs {}",
                    g.name,
                    g.nd,
                    g.nd_encoded.unwrap_or(f64::NAN),
                    g.vd
                );
                checked += 1;
            }
            assert!(checked > 380);
        }
    }
}
