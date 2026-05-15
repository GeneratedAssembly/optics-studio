use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSpacing<E> {
    pub dx: E,
    pub dy: E,
}

impl<E> GridSpacing<E> {
    pub const fn new(dx: E, dy: E) -> Self {
        Self { dx, dy }
    }
}

impl<E: Display> Display for GridSpacing<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "dx={}, dy={}", self.dx, self.dy)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoherenceState {
    Coherent,
    SpatiallyIncoherent,
    TemporallyIncoherent,
    Incoherent,
}

impl Display for CoherenceState {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Coherent => write!(f, "coherent"),
            Self::SpatiallyIncoherent => write!(f, "spatially incoherent"),
            Self::TemporallyIncoherent => write!(f, "temporally incoherent"),
            Self::Incoherent => write!(f, "incoherent"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralSample<E> {
    pub wavelength: E,
    pub weight: E,
}

impl<E> SpectralSample<E> {
    pub const fn new(wavelength: E, weight: E) -> Self {
        Self { wavelength, weight }
    }
}

impl<E: Display> Display for SpectralSample<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{} @ weight {}", self.wavelength, self.weight)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Spectrum<E> {
    Monochromatic(E),
    Broadband(Vec<SpectralSample<E>>),
}

impl<E: Copy> Spectrum<E> {
    pub fn representative_wavelength(&self) -> Option<E> {
        match self {
            Self::Monochromatic(wavelength) => Some(*wavelength),
            Self::Broadband(samples) => samples.first().map(|sample| sample.wavelength),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Monochromatic(_) => 1,
            Self::Broadband(samples) => samples.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Broadband(samples) if samples.is_empty())
    }
}

impl<E: Display> Display for Spectrum<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Monochromatic(wavelength) => write!(f, "monochromatic({})", wavelength),
            Self::Broadband(samples) => write!(f, "broadband({} samples)", samples.len()),
        }
    }
}
