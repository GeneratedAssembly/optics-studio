//! Import failures.

use thiserror::Error;

/// Errors of the `.zmx` parser, catalog resolution, and system builder.
#[derive(Debug, Error)]
pub enum ZemaxError {
    /// I/O failure reading a prescription file.
    #[error("reading {path}: {source}")]
    Io {
        /// Path that failed.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Only sequential-mode prescriptions import.
    #[error("MODE {0:?} unsupported (only SEQ)")]
    UnsupportedMode(String),
    /// A surface carries a non-importable type or mirror flag.
    #[error("SURF {surface} unsupported: {reason}")]
    UnsupportedSurface {
        /// Zero-based surface index.
        surface: usize,
        /// What made the surface non-importable.
        reason: String,
    },
    /// A required field is missing or malformed.
    #[error("SURF {surface}: {field} invalid ({value:?})")]
    BadField {
        /// Surface index.
        surface: usize,
        /// Field keyword.
        field: &'static str,
        /// Offending raw value.
        value: String,
    },
    /// Header-level structural problem (no surfaces, no wavelengths, bad
    /// primary index).
    #[error("prescription invalid: {0}")]
    BadHeader(String),
    /// Glass not present in the built-in catalog.
    #[error("glass {0:?} not in built-in catalog (external .AGF import pending)")]
    UnknownGlass(String),
    /// An element construction rejected the imported geometry.
    #[error("element build failed: {0}")]
    Element(#[from] optics_core::element::OpticsError),
}
