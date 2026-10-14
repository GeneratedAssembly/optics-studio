//! Text-format `.zmx` parser (line-oriented ASCII, CRLF-tolerant).
//!
//! Grammar (subset — everything else is ignored, real prescriptions carry
//! hundreds of optional lines):
//!
//! - Header: `MODE SEQ` (required), `NAME <text>`, `UNIT <L> ...`
//!   (length unit of all numeric lens data: `MM`/`CM`/`IN`/`M`),
//!   `ENPD <val>` (entrance pupil diameter), `GCAT <text>` (glass catalog
//!   name), `WAVM <i> <lambda_um> <w>` (wavelengths, always µm),
//!   `PWAV <i>` (primary wavelength index, 1-based).
//! - Surface blocks: `SURF <i>` at column zero; fields indented:
//!   `COMM <text>`, `STOP`, `TYPE <kind> ...`, `CURV <c> ...` (curvature,
//!   1/lens-unit), `CONI <k>`, `PARA ...` (even-asphere terms — recorded,
//!   rejected by the builder), `DISZ <d>|INFINITY` (thickness to next
//!   surface), `GLAS <name> ...` (medium AFTER this surface),
//!   `DIAM <r> ...` (semi-diameter), `MIRR <m> ...` (`m == 1` = mirror,
//!   rejected).
//!
//! All lengths are normalized to **millimeters** (curvature to 1/mm);
//! wavelengths stay µm. Unknown keywords are ignored at both levels;
//! structural failures (non-SEQ mode, no surfaces, no wavelengths, bad
//! primary index) are errors.

use crate::error::ZemaxError;

/// One parsed surface.
#[derive(Clone, Debug, PartialEq)]
pub struct ZmxSurface {
    /// Surface index as declared (`SURF n`).
    pub index: usize,
    /// `TYPE` keyword (e.g. `STANDARD`).
    pub kind: String,
    /// `STOP` flag present.
    pub stop: bool,
    /// `COMM` text.
    pub comment: String,
    /// `CURV` first argument, 1/mm.
    pub curvature_mm: f64,
    /// `CONI` first argument (default 0).
    pub conic: f64,
    /// `PARA` terms present (builder rejects; parser records).
    pub has_para: bool,
    /// `MIRR` flag = 1 (mirror; builder rejects).
    pub mirror: bool,
    /// `DIAM` first argument (mm; 0 = unset).
    pub semi_diam_mm: f64,
    /// `DISZ` (mm); `None` for `INFINITY`.
    pub thickness_mm: Option<f64>,
    /// `GLAS` name (medium after this surface); `None` = air.
    pub glass: Option<String>,
}

/// A parsed sequential prescription.
#[derive(Clone, Debug, PartialEq)]
pub struct ZmxSystem {
    /// `NAME` text.
    pub name: String,
    /// `GCAT` catalog name.
    pub catalog: String,
    /// `ENPD` (mm).
    pub enpd_mm: f64,
    /// `WAVM` wavelengths (µm), in file order.
    pub wavelengths_um: Vec<f64>,
    /// `PWAV` primary index (0-based into `wavelengths_um`).
    pub primary: usize,
    /// Surfaces in file order (index 0 = object).
    pub surfaces: Vec<ZmxSurface>,
}

fn first_num(field: &'static str, surface: usize, tok: Option<&str>) -> Result<f64, ZemaxError> {
    let t = tok.unwrap_or("");
    t.parse::<f64>().map_err(|_| ZemaxError::BadField {
        surface,
        field,
        value: t.to_string(),
    })
}

impl ZmxSystem {
    /// Parse `.zmx` text.
    pub fn parse(text: &str) -> Result<Self, ZemaxError> {
        let mut sys = Self {
            name: String::new(),
            catalog: String::new(),
            enpd_mm: 0.0,
            wavelengths_um: Vec::new(),
            primary: 0,
            surfaces: Vec::new(),
        };
        let mut scale: f64 = 1.0; // lens unit -> mm
        let mut mode: Option<String> = None;
        let mut cur: Option<ZmxSurface> = None;

        for raw in text.lines() {
            let line = raw.trim_end_matches('\r');
            let indented = line.starts_with(' ') || line.starts_with('\t');
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut it = trimmed.split_whitespace();
            let kw = it.next().unwrap_or("");
            if indented {
                let Some(s) = cur.as_mut() else { continue };
                let mut tok = || it.next();
                match kw {
                    "COMM" => s.comment = trimmed[4..].trim().to_string(),
                    "STOP" => s.stop = true,
                    "TYPE" => s.kind = tok().unwrap_or("").to_string(),
                    "CURV" => s.curvature_mm = first_num("CURV", s.index, tok())? * 1.0 / scale,
                    "CONI" => s.conic = first_num("CONI", s.index, tok())?,
                    "PARA" => s.has_para = true,
                    "MIRR" => s.mirror = tok().is_some_and(|t| t == "1"),
                    "DIAM" => s.semi_diam_mm = first_num("DIAM", s.index, tok())? * scale,
                    "DISZ" => {
                        let t = tok().unwrap_or("");
                        s.thickness_mm = if t.eq_ignore_ascii_case("INFINITY") {
                            None
                        } else {
                            Some(first_num("DISZ", s.index, Some(t))? * scale)
                        };
                    }
                    "GLAS" => {
                        let g = tok().unwrap_or("");
                        if !g.is_empty() {
                            s.glass = Some(g.to_string());
                        }
                    }
                    _ => {}
                }
                continue;
            }
            match kw {
                "MODE" => mode = it.next().map(str::to_string),
                "NAME" => sys.name = trimmed[4..].trim().to_string(),
                "UNIT" => {
                    scale = match it.next().unwrap_or("MM") {
                        "MM" => 1.0,
                        "CM" => 10.0,
                        "M" => 1000.0,
                        "IN" => 25.4,
                        other => {
                            return Err(ZemaxError::BadHeader(format!(
                                "unknown length unit {other:?}"
                            )));
                        }
                    };
                }
                "ENPD" => {
                    sys.enpd_mm = first_num("ENPD", 0, it.next())? * scale;
                }
                "GCAT" => sys.catalog = trimmed[4..].trim().to_string(),
                "WAVM" => {
                    let idx: usize = it
                        .next()
                        .and_then(|t| t.parse().ok())
                        .ok_or_else(|| ZemaxError::BadHeader("WAVM index".into()))?;
                    let lam = first_num("WAVM", 0, it.next())?;
                    if idx == 0 {
                        return Err(ZemaxError::BadHeader("WAVM index is 1-based".into()));
                    }
                    if sys.wavelengths_um.len() < idx {
                        sys.wavelengths_um.resize(idx, 0.0);
                    }
                    sys.wavelengths_um[idx - 1] = lam;
                }
                "PWAV" => {
                    sys.primary = first_num("PWAV", 0, it.next())? as usize - 1;
                }
                "SURF" => {
                    if let Some(done) = cur.take() {
                        sys.surfaces.push(done);
                    }
                    let idx = it
                        .next()
                        .and_then(|t| t.parse::<usize>().ok())
                        .ok_or_else(|| ZemaxError::BadHeader("SURF index".into()))?;
                    cur = Some(ZmxSurface {
                        index: idx,
                        kind: "STANDARD".into(),
                        stop: false,
                        comment: String::new(),
                        curvature_mm: 0.0,
                        conic: 0.0,
                        has_para: false,
                        mirror: false,
                        semi_diam_mm: 0.0,
                        thickness_mm: None,
                        glass: None,
                    });
                }
                _ => {}
            }
        }
        if let Some(done) = cur.take() {
            sys.surfaces.push(done);
        }
        if mode.as_deref() != Some("SEQ") {
            return Err(ZemaxError::UnsupportedMode(
                mode.unwrap_or_else(|| "<missing>".into()),
            ));
        }
        if sys.surfaces.len() < 2 {
            return Err(ZemaxError::BadHeader("fewer than two surfaces".into()));
        }
        if sys.wavelengths_um.is_empty() || sys.primary >= sys.wavelengths_um.len() {
            return Err(ZemaxError::BadHeader(
                "missing wavelengths or primary".into(),
            ));
        }
        Ok(sys)
    }

    /// Primary wavelength (µm).
    pub fn primary_um(&self) -> f64 {
        self.wavelengths_um[self.primary]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/ACA254-100-UV.zmx")
    }

    fn text() -> String {
        std::fs::read_to_string(sample()).unwrap()
    }

    #[test]
    fn parses_aca254_golden_fields() {
        let sys = ZmxSystem::parse(&text()).unwrap();
        assert!(sys.name.starts_with("Air-Spaced Achromatic Doublet"));
        assert_eq!(sys.catalog, "SCHOTT INFRARED");
        assert!((sys.enpd_mm - 18.0).abs() < 1e-9);
        // Zemax pads the WAVM table with default 0.55 um slots.
        assert_eq!(&sys.wavelengths_um[..3], &[0.2825, 0.325, 0.3675]);
        assert_eq!(sys.wavelengths_um.len(), 24);
        assert_eq!(sys.primary, 0);
        assert_eq!(sys.surfaces.len(), 6);
        let s1 = &sys.surfaces[1];
        assert!(s1.stop);
        assert_eq!(s1.glass.as_deref(), Some("CAF2"));
        assert!((s1.curvature_mm - 3.2970656116056714e-2).abs() < 1e-15);
        assert!((s1.thickness_mm.unwrap() - 7.1).abs() < 1e-9);
        assert!((s1.semi_diam_mm - 11.125).abs() < 1e-9);
        let s2 = &sys.surfaces[2];
        assert_eq!(s2.glass, None);
        assert!((s2.thickness_mm.unwrap() - 0.25).abs() < 1e-12);
        assert_eq!(sys.surfaces[3].glass.as_deref(), Some("F_SILICA"));
        assert!((sys.surfaces[4].thickness_mm.unwrap() - 89.4896).abs() < 1e-3);
        assert_eq!(sys.surfaces[0].thickness_mm, None); // object at infinity
    }

    #[test]
    fn rejects_non_seq_and_garbage() {
        let bad = "MODE NSC\nSURF 0\nSURF 1\n";
        assert!(matches!(
            ZmxSystem::parse(bad),
            Err(ZemaxError::UnsupportedMode(_))
        ));
        let no_wav = "MODE SEQ\nSURF 0\nSURF 1\n";
        assert!(matches!(
            ZmxSystem::parse(no_wav),
            Err(ZemaxError::BadHeader(_))
        ));
    }

    #[test]
    fn unit_conversion_cm() {
        let t = "MODE SEQ\r\nUNIT CM M M CM CM CM\r\nENPD 2.0\r\nWAVM 1 0.55 1\r\nPWAV 1\r\nSURF 0\r\n  DISZ INFINITY\r\nSURF 1\r\n  CURV 0.1\r\n  DISZ 0.5\r\n  DIAM 1.0\r\nSURF 2\r\n";
        let sys = ZmxSystem::parse(t).unwrap();
        assert!((sys.enpd_mm - 20.0).abs() < 1e-12); // 2 cm
        assert!((sys.surfaces[1].curvature_mm - 0.01).abs() < 1e-15); // 0.1/cm
        assert!((sys.surfaces[1].thickness_mm.unwrap() - 5.0).abs() < 1e-12);
    }
}
