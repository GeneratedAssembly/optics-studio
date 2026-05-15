use dfdx::prelude::*;

use crate::core::GridSpacing;
use crate::utils::scalar::{FloatDtype, float};

pub fn centered_grid_points<S, E>(shape: S, spacing: GridSpacing<E>) -> Vec<(E, E)>
where
    S: Shape<Concrete = [usize; 2]>,
    E: FloatDtype,
{
    let [ny, nx] = shape.concrete();
    let half_x = (nx as f64 - 1.0) * 0.5;
    let half_y = (ny as f64 - 1.0) * 0.5;
    let mut points = Vec::with_capacity(nx * ny);

    for iy in 0..ny {
        let y = (float::<E>(iy as f64) - float::<E>(half_y)) * spacing.dy;
        for ix in 0..nx {
            let x = (float::<E>(ix as f64) - float::<E>(half_x)) * spacing.dx;
            points.push((x, y));
        }
    }

    points
}

pub fn tensor_from_fn_2d<S, E, D, F>(
    device: &D,
    shape: S,
    spacing: GridSpacing<E>,
    f: F,
) -> Tensor<S, E, D>
where
    S: Shape<Concrete = [usize; 2]>,
    E: FloatDtype,
    D: TensorFromVec<E>,
    F: Fn(E, E) -> E,
{
    let values = centered_grid_points(shape, spacing)
        .into_iter()
        .map(|(x, y)| f(x, y))
        .collect();
    device.tensor((values, shape))
}
