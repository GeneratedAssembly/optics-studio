//! Shell live-preview raster plumbing: pixel mapping is delegated to the
//! shared `viewmodel::intensity` (task 10.2 — autoscale, log floor,
//! downsampling, colormaps); this module owns only the BMP encode that lets
//! the shell paint its live PSF into an ordinary `<img>` with no canvas
//! refs. Display plumbing only — no metric, criterion, or physics (spec §6).

use optics_core::field::Psf;

use crate::viewmodel::intensity::{RenderOptions, render_intensity};

/// A display-ready RGBA8 buffer for the shell preview `<img>`.
#[derive(Clone, Debug)]
pub struct RgbaImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// RGBA bytes, row-major.
    pub bytes: Vec<u8>,
    /// `Some(k)` when the source grid was box-averaged by factor `k`
    /// (spec §4 "display downsampled ×k" note).
    pub downsampled_by: Option<usize>,
    /// Peak of the current frame (for the power-drift readout).
    pub peak: f64,
}

/// Shell preview failures (typed, spec §3.5).
#[derive(Clone, Debug, thiserror::Error)]
pub enum PreviewError {
    /// Mapping failure surfaced by the shared intensity view model.
    #[error(transparent)]
    Intensity(#[from] crate::viewmodel::intensity::IntensityError),
}

/// Maps a PSF to RGBA through the shared intensity pipeline. `max_side`
/// bounds the canvas budget; `log` selects the log display scale.
pub fn render_psf(psf: &Psf, max_side: usize, log: bool) -> Result<RgbaImage, PreviewError> {
    let (nx, ny) = psf.dims();
    let opts = RenderOptions {
        log,
        budget: max_side,
        ..RenderOptions::default()
    };
    let img = render_intensity(psf.samples(), nx, ny, &opts)?;
    Ok(RgbaImage {
        width: img.width,
        height: img.height,
        bytes: img.rgba,
        downsampled_by: (img.downsample_factor > 1).then_some(img.downsample_factor),
        peak: img.scale.hi,
    })
}

/// Encodes an [`RgbaImage`] as an uncompressed 24-bit BMP (bottom-up rows,
/// 4-byte row padding) so the preview rides a `<img src="data:image/bmp,
/// base64,…">` attribute with no platform APIs.
pub fn bmp_bytes(img: &RgbaImage) -> Vec<u8> {
    let w = img.width as i32;
    let h = img.height as i32;
    let row_bytes = (img.width * 3 + 3) & !3;
    let pixel_size = row_bytes * img.height;
    let file_size = 54 + pixel_size;
    let mut out = Vec::with_capacity(file_size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(pixel_size as u32).to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for y in (0..img.height).rev() {
        for x in 0..img.width {
            let i = (y * img.width + x) * 4;
            out.extend_from_slice(&[img.bytes[i + 2], img.bytes[i + 1], img.bytes[i]]);
        }
        out.resize(out.len() + row_bytes - img.width * 3, 0);
    }
    out
}

/// The BMP as a `data:` URL ready for an `<img>` `src`.
pub fn bmp_data_url(img: &RgbaImage) -> String {
    use base64::Engine as _;
    format!(
        "data:image/bmp;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bmp_bytes(img))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: usize, h: usize) -> RgbaImage {
        let mut bytes = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            bytes.extend_from_slice(&[10, 20, 30, 255]);
        }
        RgbaImage {
            width: w,
            height: h,
            bytes,
            downsampled_by: None,
            peak: 1.0,
        }
    }

    #[test]
    fn bmp_header_and_padding() {
        let b = bmp_bytes(&img(3, 1));
        assert_eq!(&b[0..2], b"BM");
        assert_eq!(u32::from_le_bytes(b[2..6].try_into().unwrap()), 54 + 12);
        assert_eq!(i32::from_le_bytes(b[18..22].try_into().unwrap()), 3);
        assert_eq!(&b[54..57], &[30, 20, 10]);
        assert_eq!(&b[63..66], &[0, 0, 0]);
        assert_eq!(b.len(), 66);
    }

    #[test]
    fn render_psf_maps_a_ramp() {
        let samples: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let psf = Psf::new(samples, 8, 8, 10e-6, 10e-6).unwrap();
        let out = render_psf(&psf, 512, false).unwrap();
        assert_eq!((out.width, out.height), (8, 8));
        assert_eq!(out.bytes.len(), 8 * 8 * 4);
        assert_eq!(out.downsampled_by, None);
        let lum = |px: usize| -> u32 {
            let i = px * 4;
            out.bytes[i] as u32 + out.bytes[i + 1] as u32 + out.bytes[i + 2] as u32
        };
        assert!(lum(63) > lum(0), "peak bin brightest");
    }

    #[test]
    fn data_url_prefix() {
        let url = bmp_data_url(&img(2, 2));
        assert!(url.starts_with("data:image/bmp;base64,"));
    }
}
