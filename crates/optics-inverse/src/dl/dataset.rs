//! Headless dataset generation for the deep-surrogate harness (task 9.14).
//!
//! The forward model is the data factory: [`generate`] draws parameter
//! vectors uniformly from the carrier's declared bounds (plan decision 3 —
//! "training data generated with the headless forward model"), renders each
//! through the borrowed [`Renderer`], and reduces the resulting
//! [`Observation`] to a fixed-length feature vector with a [`FeatureSpec`].
//! No network code lives here: the dataset is plain `f64` and serializes to
//! JSON so the (expensive) render sweep runs once and training reuses it.
//!
//! ## Feature layout
//!
//! For each scene, the intensity channel is summed incoherently over bands
//! (`field::sum_intensity` semantics on stored stacks, matching
//! [`Observation::band_sum`] per scene), then block-averaged by
//! `downsample` into an `(ny/k) × (nx/k)` frame. Scenes concatenate in
//! render order, so the feature vector is `(scene, block_row, block_col)`
//! row-major. Grids must be exactly divisible by `downsample` (a harness
//! configuration error surfaces as [`crate::InverseError::Dl`], never a
//! silent crop).
//!
//! ## Normalization
//!
//! [`Norm`] is a per-axis min–max map fitted on a dataset slice; degenerate
//! axes (`hi == lo`) encode to `0.0`. Values outside the fitted range encode
//! outside `[0, 1]` — extrapolation is allowed and documented (surrogates
//! are interpolators; callers should keep predictions inside the box).

use std::path::Path;

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::error::InverseError;
use crate::observe::{Channel, Observation, Renderer};

/// Schema tag written into dataset JSON files (`optics-inverse.dl.v1`).
pub const DATASET_SCHEMA: &str = "optics-inverse.dl.v1";

/// How to reduce an [`Observation`] to a fixed-length feature vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSpec {
    /// Block-average factor `k` applied to each scene's band-summed
    /// intensity frame (each output cell averages a `k × k` block).
    pub downsample: usize,
}

impl FeatureSpec {
    /// Feature length for an observation of this geometry.
    ///
    /// # Errors
    /// [`InverseError::Dl`] if `downsample` is zero or does not divide the
    /// grid exactly.
    pub fn feature_len(&self, obs: &Observation) -> Result<usize> {
        self.check(obs)?;
        let (nx, ny) = obs.dims;
        Ok(obs.scenes.len() * (nx / self.downsample) * (ny / self.downsample))
    }

    /// Reduce `obs` to its feature vector (see module docs for layout).
    ///
    /// # Errors
    /// [`InverseError::Dl`] for the same geometry violations as
    /// [`FeatureSpec::feature_len`], or if the intensity channel is absent.
    pub fn extract(&self, obs: &Observation) -> Result<Vec<f64>> {
        self.check(obs)?;
        let ci = obs
            .channel_index(Channel::Intensity)
            .ok_or_else(|| InverseError::Dl {
                reason: "observation has no intensity channel".into(),
            })?;
        let k = self.downsample;
        let (nx, ny) = obs.dims;
        let (bx, by) = (nx / k, ny / k);
        let mut out = Vec::with_capacity(obs.scenes.len() * bx * by);
        for si in 0..obs.scenes.len() {
            // Incoherent band sum for this scene's intensity frame.
            let mut frame: Vec<f64> = obs.slice(ci, 0, si).to_vec();
            for bi in 1..obs.bands.len() {
                for (a, v) in frame.iter_mut().zip(obs.slice(ci, bi, si)) {
                    *a += v;
                }
            }
            // Block average, row-major (block_row, block_col).
            for jb in 0..by {
                for ib in 0..bx {
                    let mut acc = 0.0f64;
                    for dy in 0..k {
                        let row = (jb * k + dy) * nx;
                        for dx in 0..k {
                            acc += frame[row + ib * k + dx];
                        }
                    }
                    out.push(acc / (k * k) as f64);
                }
            }
        }
        Ok(out)
    }

    fn check(&self, obs: &Observation) -> Result<()> {
        if self.downsample == 0 {
            return Err(InverseError::Dl {
                reason: "downsample must be >= 1".into(),
            });
        }
        let (nx, ny) = obs.dims;
        if nx % self.downsample != 0 || ny % self.downsample != 0 {
            return Err(InverseError::Dl {
                reason: format!(
                    "grid {nx}×{ny} not divisible by downsample {}",
                    self.downsample
                ),
            });
        }
        if obs.channel_index(Channel::Intensity).is_none() {
            return Err(InverseError::Dl {
                reason: "observation has no intensity channel".into(),
            });
        }
        Ok(())
    }
}

/// One harness sample: the design parameters and the rendered features.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    /// Parameter vector (carrier order, raw units).
    pub params: Vec<f64>,
    /// Feature vector for [`FeatureSpec`] applied to the rendered
    /// observation (raw units).
    pub features: Vec<f64>,
}

/// A generated dataset: samples plus the shape metadata to validate them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    /// Format tag ([`DATASET_SCHEMA`]); `load_json` rejects anything else.
    pub schema: String,
    /// Parameter dimension (carrier `dim`).
    pub dim_params: usize,
    /// Feature dimension for the generating [`FeatureSpec`].
    pub dim_features: usize,
    /// The samples, in generation order.
    pub samples: Vec<Sample>,
}

impl Dataset {
    /// Sample count.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// `true` when empty (train will reject it).
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Write the dataset as pretty JSON.
    ///
    /// # Errors
    /// [`InverseError::Dl`] on I/O failure.
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self).map_err(|e| InverseError::Dl {
            reason: e.to_string(),
        })?;
        std::fs::write(path, text).map_err(|e| InverseError::Dl {
            reason: format!("write {}: {e}", path.display()),
        })
    }

    /// Read a dataset written by [`Dataset::save_json`], validating the
    /// schema tag and per-sample shapes.
    ///
    /// # Errors
    /// [`InverseError::Dl`] on I/O, parse, schema, or shape failure.
    pub fn load_json(path: &Path) -> Result<Dataset> {
        let text = std::fs::read_to_string(path).map_err(|e| InverseError::Dl {
            reason: format!("read {}: {e}", path.display()),
        })?;
        let ds: Dataset = serde_json::from_str(&text).map_err(|e| InverseError::Dl {
            reason: e.to_string(),
        })?;
        ds.validate()?;
        Ok(ds)
    }

    /// Validate schema tag and every sample's shapes.
    ///
    /// # Errors
    /// [`InverseError::Dl`] describing the first violation.
    pub fn validate(&self) -> Result<()> {
        if self.schema != DATASET_SCHEMA {
            return Err(InverseError::Dl {
                reason: format!(
                    "unknown dataset schema {:?}, expected {DATASET_SCHEMA}",
                    self.schema
                ),
            });
        }
        for (i, s) in self.samples.iter().enumerate() {
            if s.params.len() != self.dim_params {
                return Err(InverseError::Dl {
                    reason: format!(
                        "sample {i}: {} params, expected {}",
                        s.params.len(),
                        self.dim_params
                    ),
                });
            }
            if s.features.len() != self.dim_features {
                return Err(InverseError::Dl {
                    reason: format!(
                        "sample {i}: {} features, expected {}",
                        s.features.len(),
                        self.dim_features
                    ),
                });
            }
            if !s.params.iter().all(|v| v.is_finite()) || !s.features.iter().all(|v| v.is_finite())
            {
                return Err(InverseError::Dl {
                    reason: format!("sample {i} contains a non-finite value"),
                });
            }
        }
        Ok(())
    }

    /// Deterministic train/val split: shuffle indices with `seed`, take
    /// `val_fraction` of the samples (rounded down, at least one when the
    /// fraction is positive) into the validation set.
    ///
    /// # Errors
    /// [`InverseError::Dl`] if the fraction is outside `(0, 1)` — use
    /// [`Dataset::validate`] shapes before splitting.
    pub fn split(&self, val_fraction: f64, seed: u64) -> Result<(Dataset, Dataset)> {
        if !(0.0..1.0).contains(&val_fraction) {
            return Err(InverseError::Dl {
                reason: format!("val_fraction {val_fraction} not in [0, 1)"),
            });
        }
        let n = self.samples.len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        rand::seq::SliceRandom::shuffle(order.as_mut_slice(), &mut rng);
        let n_val = ((n as f64) * val_fraction).floor() as usize;
        let mk = |idx: &[usize]| Dataset {
            schema: self.schema.clone(),
            dim_params: self.dim_params,
            dim_features: self.dim_features,
            samples: idx.iter().map(|i| self.samples[*i].clone()).collect(),
        };
        Ok((mk(&order[n_val..]), mk(&order[..n_val])))
    }
}

/// Generate a dataset headlessly: uniform draws in the carrier's bound box,
/// rendered through `renderer`, reduced by `spec`.
///
/// Deterministic for a given `seed`. The renderer owns the carrier, so the
/// parameter space is exactly the one the solvers search.
///
/// # Errors
/// Propagates render errors and [`FeatureSpec::extract`] geometry errors.
pub fn generate(
    renderer: &Renderer<'_>,
    spec: &FeatureSpec,
    n_samples: usize,
    seed: u64,
) -> Result<Dataset> {
    let mut params = renderer.carrier.params().clone();
    let dim = params.dim();
    let mut samples = Vec::with_capacity(n_samples);
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..n_samples {
        params.randomize(&mut rng);
        let values = params.values();
        let obs = renderer.render(&values)?;
        let features = spec.extract(&obs)?;
        samples.push(Sample {
            params: values,
            features,
        });
    }
    let ds = Dataset {
        schema: DATASET_SCHEMA.into(),
        dim_params: dim,
        dim_features: samples
            .first()
            .map(|s| s.features.len())
            .unwrap_or_default(),
        samples,
    };
    ds.validate()?;
    Ok(ds)
}

/// Per-axis min–max normalization fitted on a dataset axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Norm {
    lo: Vec<f64>,
    hi: Vec<f64>,
}

impl Norm {
    /// Fit elementwise min/max over `rows` (all rows must have `dim`
    /// entries).
    ///
    /// # Errors
    /// [`InverseError::Dl`] on an empty iterator or a row-length mismatch.
    pub fn fit<'a>(mut rows: impl Iterator<Item = &'a [f64]>, dim: usize) -> Result<Norm> {
        let Some(first) = rows.next() else {
            return Err(InverseError::Dl {
                reason: "Norm::fit needs at least one row".into(),
            });
        };
        let mut lo = first.to_vec();
        let mut hi = first.to_vec();
        if lo.len() != dim {
            return Err(InverseError::Dl {
                reason: format!("Norm::fit row has {} entries, expected {dim}", lo.len()),
            });
        }
        for row in rows {
            if row.len() != dim {
                return Err(InverseError::Dl {
                    reason: format!("Norm::fit row has {} entries, expected {dim}", row.len()),
                });
            }
            for ((l, h), v) in lo.iter_mut().zip(hi.iter_mut()).zip(row) {
                *l = (*l).min(*v);
                *h = (*h).max(*v);
            }
        }
        Ok(Norm { lo, hi })
    }

    /// Dimension of this normalization.
    pub fn dim(&self) -> usize {
        self.lo.len()
    }

    /// Fit one shared min/max (replicated to every axis) over all values —
    /// the right normalization for intensity features, where per-axis
    /// min–max would amplify the sub-grid background to full scale and
    /// hand the network noise to fit.
    ///
    /// # Errors
    /// [`InverseError::Dl`] on an empty iterator or a row-length mismatch.
    pub fn fit_shared<'a>(mut rows: impl Iterator<Item = &'a [f64]>, dim: usize) -> Result<Norm> {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut n_rows = 0usize;
        for row in rows.by_ref() {
            if row.len() != dim {
                return Err(InverseError::Dl {
                    reason: format!(
                        "Norm::fit_shared row has {} entries, expected {dim}",
                        row.len()
                    ),
                });
            }
            n_rows += 1;
            for v in row {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        if n_rows == 0 {
            return Err(InverseError::Dl {
                reason: "Norm::fit_shared needs at least one row".into(),
            });
        }
        Ok(Norm {
            lo: vec![lo; dim],
            hi: vec![hi; dim],
        })
    }

    /// Map raw values to normalized space: `(x - lo) / (hi - lo)`, with
    /// degenerate axes (`hi == lo`) mapping to `0.0`. Values outside the
    /// fitted range map outside `[0, 1]` (documented extrapolation).
    ///
    /// # Errors
    /// [`InverseError::Dl`] on a length mismatch.
    pub fn encode(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.dim() {
            return Err(InverseError::Dl {
                reason: format!("encode: {} values, expected {}", x.len(), self.dim()),
            });
        }
        Ok(self
            .lo
            .iter()
            .zip(&self.hi)
            .zip(x)
            .map(|((l, h), v)| if h == l { 0.0 } else { (v - l) / (h - l) })
            .collect())
    }

    /// Inverse of [`Norm::encode`].
    pub fn decode(&self, y: &[f64]) -> Vec<f64> {
        self.lo
            .iter()
            .zip(&self.hi)
            .zip(y)
            .map(|((l, h), v)| l + v * (h - l))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(dims: (usize, usize), scenes: usize, bands: usize) -> Observation {
        let n = dims.0 * dims.1;
        let frames = bands * scenes;
        Observation {
            channels: vec![Channel::Intensity],
            bands: vec![500e-9; bands],
            scenes: vec![(0.0, 0.0); scenes],
            dims,
            pitch: (10e-6, 10e-6),
            data: (0..frames * n).map(|i| i as f64).collect(),
            fd_steps: (0.0, 0.0),
        }
    }

    #[test]
    fn extract_block_averages_band_summed_intensity() {
        // 4x4 grid, 2 bands, 1 scene, downsample 2: band sum per pixel is
        // frame[i] + frame[16+i]; block averages are exact.
        let o = obs((4, 4), 1, 2);
        let spec = FeatureSpec { downsample: 2 };
        let f = spec.extract(&o).unwrap();
        assert_eq!(f.len(), 4);
        let mut expect = Vec::new();
        for jb in 0..2 {
            for ib in 0..2 {
                let mut acc = 0.0;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let idx = (jb * 2 + dy) * 4 + ib * 2 + dx;
                        acc += idx as f64 + (16 + idx) as f64;
                    }
                }
                expect.push(acc / 4.0);
            }
        }
        assert_eq!(f, expect);
    }

    #[test]
    fn extract_rejects_indivisible_grid() {
        let o = obs((6, 6), 1, 1);
        let err = FeatureSpec { downsample: 4 }.extract(&o).unwrap_err();
        assert!(err.to_string().contains("not divisible"));
    }

    #[test]
    fn norm_encode_decode_roundtrip_and_degenerate_axis() {
        let rows = [vec![0.0, 5.0, 1.0], vec![2.0, 5.0, 3.0]];
        let norm = Norm::fit(rows.iter().map(Vec::as_slice), 3).unwrap();
        let enc = norm.encode(&[1.0, 5.0, 1.0]).unwrap();
        assert_eq!(enc, vec![0.5, 0.0, 0.0]); // axis 1 degenerate -> 0
        let dec = norm.decode(&enc);
        assert!((dec[0] - 1.0).abs() < 1e-12);
        assert!((dec[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn split_is_deterministic_and_disjoint() {
        let ds = Dataset {
            schema: DATASET_SCHEMA.into(),
            dim_params: 1,
            dim_features: 1,
            samples: (0..10)
                .map(|i| Sample {
                    params: vec![i as f64],
                    features: vec![i as f64],
                })
                .collect(),
        };
        let (a, b) = ds.split(0.3, 7).unwrap();
        let (a2, b2) = ds.split(0.3, 7).unwrap();
        assert_eq!(a, a2);
        assert_eq!(b, b2);
        assert_eq!(b.len(), 3);
        assert_eq!(a.len(), 7);
        let mut all: Vec<f64> = a
            .samples
            .iter()
            .chain(b.samples.iter())
            .map(|s| s.params[0])
            .collect();
        all.sort_unstable_by(|x, y| x.total_cmp(y));
        assert_eq!(all, (0..10).map(|i| i as f64).collect::<Vec<_>>());
    }

    #[test]
    fn json_roundtrip_preserves_samples() {
        let ds = Dataset {
            schema: DATASET_SCHEMA.into(),
            dim_params: 2,
            dim_features: 2,
            samples: vec![Sample {
                params: vec![0.12345678901234567, -1e300],
                features: vec![1e-300, 42.0],
            }],
        };
        let dir = std::env::temp_dir().join("optics-dl-dataset-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ds.json");
        ds.save_json(&path).unwrap();
        let back = Dataset::load_json(&path).unwrap();
        assert_eq!(ds, back);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn load_rejects_bad_schema() {
        let ds = Dataset {
            schema: "nope".into(),
            dim_params: 0,
            dim_features: 0,
            samples: vec![],
        };
        assert!(ds.validate().is_err());
    }
}
