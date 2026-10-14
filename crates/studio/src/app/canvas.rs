//! Canvas painting via `web-sys` 2D `ImageData` (ADR-10.0 Q4 resolved: 2D
//! ImageData, not WebGL — the ≤2048² display budget never needs GPU compute,
//! which is out of scope per spec §6).

use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

use crate::viewmodel::intensity::RenderedImage;

/// Looks up a canvas element by id.
pub fn canvas(id: &str) -> Option<HtmlCanvasElement> {
    web_sys::window()?
        .document()?
        .get_element_by_id(id)?
        .dyn_into::<HtmlCanvasElement>()
        .ok()
}

/// The 2D context of a canvas.
pub fn context(canvas: &HtmlCanvasElement) -> Option<CanvasRenderingContext2d> {
    canvas
        .get_context("2d")
        .ok()?
        .and_then(|v| v.dyn_into::<CanvasRenderingContext2d>().ok())
}

/// Paints an [`RenderedImage`] onto the canvas `id`, resizing the canvas
/// backing store to the image dimensions.
///
/// Silently no-ops when the element is absent (first paint before mount) or
/// the 2D context cannot be created; the panel layer shows the frame's
/// numeric readouts regardless, so a paint failure degrades to numbers-only.
pub fn paint(id: &str, img: &RenderedImage) {
    let Some(canvas) = canvas(id) else { return };
    canvas.set_width(img.width as u32);
    canvas.set_height(img.height as u32);
    let Some(ctx) = context(&canvas) else { return };
    let Ok(data) =
        ImageData::new_with_u8_clamped_array(wasm_bindgen::Clamped(&img.rgba), img.width as u32)
    else {
        return;
    };
    let _ = ctx.put_image_data(&data, 0.0, 0.0);
}
