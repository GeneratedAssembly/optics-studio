//! Element-editing operations on the prescription working copy (GEN-57).
//!
//! The table model is a `Vec<ZmxSurface>` in file order: position 0 is the
//! object, the last position is the image plane, everything between is a
//! refracting surface the user owns. These pure functions insert, delete,
//! flag, and renumber that list, and write it back into `.zmx` text so
//! table edits persist into the authoritative source (the text mode was
//! previously a one-way door: "edited, re-parse to persist" had no way to
//! apply).
//!
//! No physics: this is data plumbing over `optics-zemax` types — the same
//! surfaces the parser produces and [`super::prescription::build_plan`]
//! consumes.

use optics_zemax::{ZmxSurface, ZmxSystem};

/// Reasons an element edit can be refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EditError {
    /// The object and image planes are structural: they are never
    /// inserted, deleted, or stop-flagged as elements.
    #[error("{0} cannot be edited as an element")]
    Boundary(&'static str),
    /// Position outside `1..=len - 1`.
    #[error("surface position {0} is out of range")]
    OutOfRange(usize),
}

/// Re-number `index` to match list position (file order = `SURF` order).
pub fn renumber(surfaces: &mut [ZmxSurface]) {
    for (i, s) in surfaces.iter_mut().enumerate() {
        s.index = i;
    }
}

/// A brand-new system: object at infinity + image plane, nothing in
/// between. Add elements before the image plane to build a lens.
pub fn blank_surfaces() -> Vec<ZmxSurface> {
    ZmxSystem::blank("untitled", 0.55).surfaces
}

/// Validate an element position: `1..=len - 1` for insertion (before the
/// image plane is allowed), `1..=len - 2` for deletion/stop (the image
/// plane itself is not an element).
fn check_pos(surfaces: &[ZmxSurface], at: usize, insertable: bool) -> Result<(), EditError> {
    if at == 0 {
        return Err(EditError::Boundary("object plane"));
    }
    let hi = surfaces.len().saturating_sub(1);
    if at > hi {
        return Err(EditError::OutOfRange(at));
    }
    if !insertable && at == hi {
        return Err(EditError::Boundary("image plane"));
    }
    Ok(())
}

/// Insert a flat air surface before position `at`; returns its position.
pub fn insert_surface(surfaces: &mut Vec<ZmxSurface>, at: usize) -> Result<usize, EditError> {
    check_pos(surfaces, at, true)?;
    surfaces.insert(at, ZmxSurface::flat(at, Some(0.0)));
    renumber(surfaces);
    Ok(at)
}

/// Insert a biconvex lens (two spherical surfaces, glass in between)
/// before position `at`; returns the front-surface position. `radius_mm`
/// is the vertex radius (positive = convex front); the back surface uses
/// the same magnitude with the sign Zemax expects for a symmetric lens.
pub fn add_lens(
    surfaces: &mut Vec<ZmxSurface>,
    at: usize,
    radius_mm: f64,
    thick_mm: f64,
    semi_mm: f64,
    glass: Option<String>,
) -> Result<usize, EditError> {
    check_pos(surfaces, at, true)?;
    let k = if radius_mm > 0.0 { 1.0 } else { -1.0 } / radius_mm.abs();
    let front = ZmxSurface {
        index: at,
        kind: "STANDARD".into(),
        stop: false,
        comment: "lens front".into(),
        curvature_mm: k,
        conic: 0.0,
        has_para: false,
        mirror: false,
        semi_diam_mm: semi_mm,
        thickness_mm: Some(thick_mm),
        glass: glass.clone(),
    };
    let back = ZmxSurface {
        index: at + 1,
        kind: "STANDARD".into(),
        stop: false,
        comment: "lens back".into(),
        curvature_mm: -k,
        conic: 0.0,
        has_para: false,
        mirror: false,
        semi_diam_mm: semi_mm,
        thickness_mm: Some(0.0),
        glass: None,
    };
    surfaces.splice(at..at, [front, back]);
    renumber(surfaces);
    Ok(at)
}

/// Insert a flat window (two parallel surfaces, glass in between) before
/// position `at`; returns the front-surface position.
pub fn add_window(
    surfaces: &mut Vec<ZmxSurface>,
    at: usize,
    thick_mm: f64,
    semi_mm: f64,
    glass: Option<String>,
) -> Result<usize, EditError> {
    check_pos(surfaces, at, true)?;
    let front = ZmxSurface {
        index: at,
        kind: "STANDARD".into(),
        stop: false,
        comment: "window front".into(),
        curvature_mm: 0.0,
        conic: 0.0,
        has_para: false,
        mirror: false,
        semi_diam_mm: semi_mm,
        thickness_mm: Some(thick_mm),
        glass,
    };
    let back = ZmxSurface {
        index: at + 1,
        kind: "STANDARD".into(),
        stop: false,
        comment: "window back".into(),
        curvature_mm: 0.0,
        conic: 0.0,
        has_para: false,
        mirror: false,
        semi_diam_mm: semi_mm,
        thickness_mm: Some(0.0),
        glass: None,
    };
    surfaces.splice(at..at, [front, back]);
    renumber(surfaces);
    Ok(at)
}

/// Delete the element at position `at` (object and image planes are
/// protected).
pub fn remove_surface(surfaces: &mut Vec<ZmxSurface>, at: usize) -> Result<(), EditError> {
    check_pos(surfaces, at, false)?;
    surfaces.remove(at);
    renumber(surfaces);
    Ok(())
}

/// Make position `at` the stop, clearing any previous flag.
pub fn set_stop(surfaces: &mut [ZmxSurface], at: usize) -> Result<(), EditError> {
    check_pos(surfaces, at, false)?;
    for (i, s) in surfaces.iter_mut().enumerate() {
        s.stop = i == at;
    }
    Ok(())
}

/// Write the edited surfaces back into `.zmx` text, keeping the header
/// (name, catalog, ENPD, wavelengths, primary) of the current system.
/// Surfaces beyond the header's glass names are validated by the next
/// parse, not here.
pub fn apply_to_text(system: &ZmxSystem, surfaces: &[ZmxSurface]) -> String {
    let mut out = system.clone();
    out.surfaces = surfaces.to_vec();
    renumber(&mut out.surfaces);
    out.to_zmx()
}

/// Default glass name for new elements: `N-BK7` when the registry has it,
/// otherwise the first catalog name.
pub fn default_glass(names: &[String]) -> Option<String> {
    names
        .iter()
        .find(|n| n.as_str() == "N-BK7")
        .or_else(|| names.first())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank() -> Vec<ZmxSurface> {
        blank_surfaces()
    }

    #[test]
    fn insert_and_lens_renumber() {
        let mut s = blank();
        let p = insert_surface(&mut s, 1).unwrap();
        assert_eq!(p, 1);
        assert_eq!(s.len(), 3);
        assert_eq!(s[1].index, 1);
        assert_eq!(s[2].index, 2);
        let p = add_lens(&mut s, 2, 50.0, 4.0, 10.0, Some("N-BK7".into())).unwrap();
        assert_eq!(p, 2);
        assert_eq!(s.len(), 5);
        assert!((s[2].curvature_mm - 0.02).abs() < 1e-12);
        assert!((s[3].curvature_mm + 0.02).abs() < 1e-12);
        assert_eq!(s[2].glass.as_deref(), Some("N-BK7"));
        assert_eq!(s[3].glass, None);
        for (i, x) in s.iter().enumerate() {
            assert_eq!(x.index, i);
        }
    }

    #[test]
    fn boundaries_are_protected() {
        let mut s = blank();
        assert_eq!(
            insert_surface(&mut s, 0),
            Err(EditError::Boundary("object plane"))
        );
        assert_eq!(
            remove_surface(&mut s, 0),
            Err(EditError::Boundary("object plane"))
        );
        assert_eq!(
            remove_surface(&mut s, 1),
            Err(EditError::Boundary("image plane"))
        );
        assert_eq!(set_stop(&mut s, 1), Err(EditError::Boundary("image plane")));
        assert_eq!(insert_surface(&mut s, 9), Err(EditError::OutOfRange(9)));
    }

    #[test]
    fn stop_moves_exclusively() {
        let mut s = blank();
        add_lens(&mut s, 1, 50.0, 4.0, 10.0, None).unwrap();
        set_stop(&mut s, 1).unwrap();
        assert!(s[1].stop && !s[2].stop);
        set_stop(&mut s, 2).unwrap();
        assert!(!s[1].stop && s[2].stop);
    }

    #[test]
    fn remove_shrinks_and_renumbers() {
        let mut s = blank();
        add_window(&mut s, 1, 3.0, 8.0, Some("F_SILICA".into())).unwrap();
        assert_eq!(s.len(), 4);
        remove_surface(&mut s, 2).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[1].glass.as_deref(), Some("F_SILICA"));
        for (i, x) in s.iter().enumerate() {
            assert_eq!(x.index, i);
        }
    }

    #[test]
    fn applied_text_reparses_to_the_same_model() {
        let mut s = blank();
        add_lens(&mut s, 1, 25.0, 5.0, 12.0, Some("N-BK7".into())).unwrap();
        set_stop(&mut s, 1).unwrap();
        let sys = ZmxSystem::blank("built", 0.55);
        let text = apply_to_text(&sys, &s);
        let back = ZmxSystem::parse(&text).unwrap();
        assert_eq!(back.surfaces, s);
        assert_eq!(back.name, "built");
    }

    #[test]
    fn default_glass_prefers_bk7() {
        let names: Vec<String> = ["F_SILICA", "N-BK7", "CAF2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(default_glass(&names).as_deref(), Some("N-BK7"));
        assert_eq!(default_glass(&names[..1]).as_deref(), Some("F_SILICA"));
        assert_eq!(default_glass(&[]), None);
    }
}
