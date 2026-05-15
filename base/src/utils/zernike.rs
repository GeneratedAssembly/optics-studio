use crate::utils::scalar::factorial;

pub fn radial_zernike(n: usize, m: isize, rho: f64) -> f64 {
    let m = m.unsigned_abs();
    if n < m || !(n - m).is_multiple_of(2) {
        return 0.0;
    }

    let mut sum = 0.0;
    for k in 0..=((n - m) / 2) {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let numerator = factorial(n - k);
        let denominator = factorial(k) * factorial((n + m) / 2 - k) * factorial((n - m) / 2 - k);
        sum += sign * numerator / denominator * rho.powi((n - 2 * k) as i32);
    }
    sum
}

pub fn zernike(n: usize, m: isize, rho: f64, theta: f64) -> f64 {
    let radial = radial_zernike(n, m, rho);
    match m.cmp(&0) {
        std::cmp::Ordering::Greater => radial * (m as f64 * theta).cos(),
        std::cmp::Ordering::Less => radial * ((-m) as f64 * theta).sin(),
        std::cmp::Ordering::Equal => radial,
    }
}
