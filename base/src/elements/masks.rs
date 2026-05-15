use std::fmt::{Display, Formatter, Result as FmtResult};

use dfdx::prelude::*;

use crate::core::{Field, OpticalElement};
use crate::utils::scalar::{FloatDtype, float};

#[derive(Debug, Clone)]
pub struct AmplitudeMask<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub amplitude: Tensor<S, E, D, NoneTape>,
}

impl<S, E, D> AmplitudeMask<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub fn new(amplitude: Tensor<S, E, D, NoneTape>) -> Self {
        Self { amplitude }
    }
}

impl<S, E, D> Display for AmplitudeMask<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "AmplitudeMask(shape={:?})", self.amplitude.shape())
    }
}

#[derive(Debug, Clone)]
pub struct PhaseMask<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub phase: Tensor<S, E, D, NoneTape>,
}

impl<S, E, D> PhaseMask<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub fn new(phase: Tensor<S, E, D, NoneTape>) -> Self {
        Self { phase }
    }
}

impl<S, E, D> Display for PhaseMask<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "PhaseMask(shape={:?})", self.phase.shape())
    }
}

#[derive(Debug, Clone)]
pub struct TransmissionMask<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub amplitude: Tensor<S, E, D, NoneTape>,
    pub phase: Tensor<S, E, D, NoneTape>,
}

impl<S, E, D> TransmissionMask<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub fn new(amplitude: Tensor<S, E, D, NoneTape>, phase: Tensor<S, E, D, NoneTape>) -> Self {
        assert_eq!(amplitude.shape(), phase.shape());
        Self { amplitude, phase }
    }
}

impl<S, E, D> Display for TransmissionMask<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "TransmissionMask(shape={:?})", self.phase.shape())
    }
}

#[derive(Debug, Clone)]
pub struct BirefringentThicknessMask<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub thickness: Tensor<S, E, D, NoneTape>,
    pub ordinary_index: E,
    pub extraordinary_index: E,
    pub wavelength: E,
}

impl<S, E, D> BirefringentThicknessMask<S, E, D>
where
    S: Shape,
    E: Dtype,
    D: Storage<E>,
{
    pub fn new(
        thickness: Tensor<S, E, D, NoneTape>,
        ordinary_index: E,
        extraordinary_index: E,
        wavelength: E,
    ) -> Self {
        Self {
            thickness,
            ordinary_index,
            extraordinary_index,
            wavelength,
        }
    }
}

impl<S, E, D> Display for BirefringentThicknessMask<S, E, D>
where
    S: Shape,
    E: Dtype + Display,
    D: Storage<E>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "BirefringentThicknessMask(shape={:?}, no={}, ne={}, wavelength={})",
            self.thickness.shape(),
            self.ordinary_index,
            self.extraordinary_index,
            self.wavelength
        )
    }
}

pub(crate) fn apply_amplitude<F>(
    field: F,
    amplitude: &Tensor<F::Shape, F::Elem, F::Device, NoneTape>,
) -> F
where
    F: Field,
    F::Device: Device<F::Elem>,
{
    let mut components = field.into_components();
    let real = components.real * amplitude.retaped::<F::Tape>();
    let imag = components.imag * amplitude.retaped::<F::Tape>();
    components.real = real;
    components.imag = imag;
    F::from_components(components)
}

pub(crate) fn apply_phase<F>(field: F, phase: &Tensor<F::Shape, F::Elem, F::Device, NoneTape>) -> F
where
    F: Field,
    F::Device: Device<F::Elem>,
{
    let components = field.into_components();
    let real_for_imag = components.real.with_empty_tape();
    let imag_for_imag = components.imag.with_empty_tape();

    let cos_for_real = phase.retaped::<F::Tape>().cos();
    let sin_for_real = phase.retaped::<F::Tape>().sin();
    let cos_for_imag = phase.retaped::<F::Tape>().cos();
    let sin_for_imag = phase.retaped::<F::Tape>().sin();

    let real_part = components.real * cos_for_real - components.imag * sin_for_real;
    let imag_part = real_for_imag * sin_for_imag + imag_for_imag * cos_for_imag;

    F::from_components(crate::core::FieldComponents {
        real: real_part,
        imag: imag_part,
        spacing: components.spacing,
        spectrum: components.spectrum,
        coherence: components.coherence,
    })
}

pub(crate) fn apply_transmission<F>(
    field: F,
    amplitude: &Tensor<F::Shape, F::Elem, F::Device, NoneTape>,
    phase: &Tensor<F::Shape, F::Elem, F::Device, NoneTape>,
) -> F
where
    F: Field,
    F::Device: Device<F::Elem>,
{
    let components = field.into_components();
    let real_for_imag = components.real.with_empty_tape();
    let imag_for_imag = components.imag.with_empty_tape();

    let amp_cos_real = amplitude.retaped::<F::Tape>() * phase.retaped::<F::Tape>().cos();
    let amp_sin_real = amplitude.retaped::<F::Tape>() * phase.retaped::<F::Tape>().sin();
    let amp_cos_imag = amplitude.retaped::<F::Tape>() * phase.retaped::<F::Tape>().cos();
    let amp_sin_imag = amplitude.retaped::<F::Tape>() * phase.retaped::<F::Tape>().sin();

    let real_part = components.real * amp_cos_real - components.imag * amp_sin_real;
    let imag_part = real_for_imag * amp_sin_imag + imag_for_imag * amp_cos_imag;

    F::from_components(crate::core::FieldComponents {
        real: real_part,
        imag: imag_part,
        spacing: components.spacing,
        spectrum: components.spectrum,
        coherence: components.coherence,
    })
}

pub(crate) fn apply_thickness_phase<F>(
    field: F,
    thickness: &Tensor<F::Shape, F::Elem, F::Device, NoneTape>,
    phase_per_meter: F::Elem,
) -> F
where
    F: Field,
    F::Device: Device<F::Elem>,
{
    let components = field.into_components();
    let real_for_imag = components.real.with_empty_tape();
    let imag_for_imag = components.imag.with_empty_tape();

    let phase_cos_real = thickness.retaped::<F::Tape>() * phase_per_meter;
    let phase_sin_real = thickness.retaped::<F::Tape>() * phase_per_meter;
    let phase_cos_imag = thickness.retaped::<F::Tape>() * phase_per_meter;
    let phase_sin_imag = thickness.retaped::<F::Tape>() * phase_per_meter;

    let real_part = components.real * phase_cos_real.cos() - components.imag * phase_sin_real.sin();
    let imag_part = real_for_imag * phase_sin_imag.sin() + imag_for_imag * phase_cos_imag.cos();

    F::from_components(crate::core::FieldComponents {
        real: real_part,
        imag: imag_part,
        spacing: components.spacing,
        spectrum: components.spectrum,
        coherence: components.coherence,
    })
}

impl<F, S, E, D> OpticalElement<F> for AmplitudeMask<S, E, D>
where
    F: Field<Shape = S, Elem = E, Device = D>,
    S: Shape,
    E: Dtype + Display,
    D: Device<E>,
{
    fn apply(self, field: F) -> F {
        apply_amplitude(field, &self.amplitude)
    }
}

impl<F, S, E, D> OpticalElement<F> for PhaseMask<S, E, D>
where
    F: Field<Shape = S, Elem = E, Device = D>,
    S: Shape,
    E: Dtype + Display,
    D: Device<E>,
{
    fn apply(self, field: F) -> F {
        apply_phase(field, &self.phase)
    }
}

impl<F, S, E, D> OpticalElement<F> for TransmissionMask<S, E, D>
where
    F: Field<Shape = S, Elem = E, Device = D>,
    S: Shape,
    E: Dtype + Display,
    D: Device<E>,
{
    fn apply(self, field: F) -> F {
        apply_transmission(field, &self.amplitude, &self.phase)
    }
}

impl<F, S, E, D> OpticalElement<F> for BirefringentThicknessMask<S, E, D>
where
    F: Field<Shape = S, Elem = E, Device = D>,
    S: Shape,
    E: FloatDtype,
    D: Device<E>,
{
    fn apply(self, field: F) -> F {
        let two_pi = float::<E>(std::f64::consts::TAU);
        let delta_n = self.extraordinary_index - self.ordinary_index;
        let phase_per_meter = two_pi * delta_n / self.wavelength;
        apply_thickness_phase(field, &self.thickness, phase_per_meter)
    }
}
