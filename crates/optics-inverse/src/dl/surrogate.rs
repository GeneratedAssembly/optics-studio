//! burn network surrogates: train, evaluate, persist (task 9.14).
//!
//! A [`Surrogate`] is a fully-connected MLP on `burn` (plan decision 3)
//! mapping a normalized [`crate::dl::dataset::Dataset`] axis to the other:
//!
//! - [`Direction::Forward`] — params → features: a cheap stand-in for the
//!   ASM renderer, usable inside merit loops and studies that render
//!   millions of designs.
//! - [`Direction::Inverse`] — features → params: a warm start for the
//!   solvers ([`crate::fd`] Adam, [`crate::solve`] stochastic runs).
//!
//! ## Training discipline
//!
//! Same notebook discipline as task 9.7's Adam: mini-batch MSE loss,
//! best-validation-epoch snapshot, per-epoch history in [`TrainReport`].
//! Inputs and targets are min–max normalized on the *training* split only
//! (validation statistics must not leak into the encoder). All tensors are
//! `f32`; the dataset stays `f64` and is converted at the batch boundary.
//!
//! ## Backends
//!
//! [`train`] uses the pure-Rust Flex CPU backend
//! (`Device::flex()`), so gates and CI stay dependency-light. GPU is
//! allowed here — never in core — by enabling the matching `burn` Cargo
//! feature (`wgpu`, `cuda`, …) and calling [`train_on`] with that device
//! (e.g. `Device::cuda(0)`); the model code itself is backend-agnostic.
//!
//! ## Persistence
//!
//! [`Surrogate::save`] writes a directory with the weights
//! (`net.burnpack`, burn format) and the metadata (`meta.json`: layer
//! sizes, direction, normalization bounds) needed to rebuild and
//! [`Surrogate::load`] the model elsewhere.

use std::path::Path;

use burn::module::Module;
use burn::nn::activation::Relu;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::{Device, Tensor};
use burn::tensor::TensorData;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::dl::dataset::{Dataset, Norm, Sample};
use crate::error::InverseError;

/// Which dataset axis maps to which: forward predicts the observation
/// features from the parameters, inverse predicts parameters from features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// params → features (surrogate of the renderer).
    Forward,
    /// features → params (solver warm start).
    Inverse,
}

/// MLP + mini-batch training configuration.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Hidden layer widths (one `Relu` after each; the output layer is
    /// linear).
    pub hidden: Vec<usize>,
    /// Full epochs over the training split.
    pub epochs: usize,
    /// Adam learning rate on the normalized loss (typ. 1e-2 … 5e-2).
    pub lr: f64,
    /// Mini-batch size.
    pub batch: usize,
    /// Seed for model init, batch shuffling, and the split-independent
    /// stream (deterministic on the Flex backend).
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            hidden: vec![64, 64],
            epochs: 200,
            lr: 0.02,
            batch: 32,
            seed: 0x5EED_914D,
        }
    }
}

/// One epoch's losses (mean mini-batch MSE in normalized space).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EpochRecord {
    /// Training split loss (0-based epoch index is the position in
    /// [`TrainReport::history`]).
    pub train_loss: f64,
    /// Validation split loss (one full forward pass, no gradient).
    pub val_loss: f64,
}

/// Training outcome: history plus the best-val-epoch snapshot loss.
#[derive(Debug, Clone)]
pub struct TrainReport {
    /// Per-epoch train/val losses.
    pub history: Vec<EpochRecord>,
    /// Lowest validation loss over all epochs (the retained snapshot).
    pub best_val_loss: f64,
}

/// Serialized rebuild recipe for a [`Surrogate`] (stored as `meta.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SurrogateMeta {
    dims: Vec<usize>,
    direction: Direction,
    in_norm: Norm,
    out_norm: Norm,
}

/// The burn MLP: `dims.windows(2)` linear layers, `Relu` between them.
#[derive(burn::module::Module, Debug)]
struct Mlp {
    layers: Vec<Linear>,
    relu: Relu,
}

impl Mlp {
    fn new(dims: &[usize], device: &Device) -> Mlp {
        let layers = dims
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(device))
            .collect();
        Mlp {
            layers,
            relu: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mut h = x;
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h);
            if i < last {
                h = self.relu.forward(h);
            }
        }
        h
    }
}

/// A trained surrogate: MLP weights plus the input/output normalizations.
pub struct Surrogate {
    net: Mlp,
    in_norm: Norm,
    out_norm: Norm,
    direction: Direction,
    dims: Vec<usize>,
    device: Device,
}

impl std::fmt::Debug for Surrogate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surrogate")
            .field("dims", &self.dims)
            .field("direction", &self.direction)
            .finish_non_exhaustive()
    }
}

impl Surrogate {
    /// Which axis mapping this surrogate learned.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Layer sizes including input and output.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Predict one vector: `input` in raw units of this direction's input
    /// axis (params for [`Direction::Forward`], features for
    /// [`Direction::Inverse`]), output in raw units of the other axis.
    ///
    /// # Errors
    /// [`InverseError::Dl`] on input-length mismatch or a backend data
    /// conversion failure.
    pub fn predict(&self, input: &[f64]) -> Result<Vec<f64>> {
        let batch = self.predict_rows(&[input.to_vec()])?;
        Ok(batch.into_iter().next().expect("one row requested"))
    }

    /// Predict a batch of rows (same units as [`Surrogate::predict`]).
    ///
    /// # Errors
    /// As [`Surrogate::predict`]; every row must match the input length.
    pub fn predict_rows(&self, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let d_in = self.dims[0];
        let mut flat: Vec<f32> = Vec::with_capacity(inputs.len() * d_in);
        for row in inputs {
            let enc = self.in_norm.encode(row)?;
            flat.extend(enc.into_iter().map(|v| v as f32));
        }
        let x = Tensor::<2>::from_data(TensorData::new(flat, [inputs.len(), d_in]), &self.device);
        let y = self.net.forward(x);
        let data = y.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| InverseError::Dl {
            reason: e.to_string(),
        })?;
        let d_out = self.dims[self.dims.len() - 1];
        if data.num_elements() != inputs.len() * d_out {
            return Err(InverseError::Dl {
                reason: format!(
                    "network returned {}, expected {}",
                    data.num_elements(),
                    inputs.len() * d_out
                ),
            });
        }
        Ok(slice
            .chunks(d_out)
            .map(|chunk| {
                self.out_norm
                    .decode(&chunk.iter().map(|v| *v as f64).collect::<Vec<_>>())
            })
            .collect())
    }

    /// Save weights + metadata into `dir` (`net.burnpack`, `meta.json`).
    ///
    /// # Errors
    /// [`InverseError::Dl`] on any I/O or record failure.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| InverseError::Dl {
            reason: format!("create {}: {e}", dir.display()),
        })?;
        let net = self.net.clone();
        net.save_file(dir.join("net.burnpack"))
            .map_err(|e| InverseError::Dl {
                reason: format!("save net.burnpack: {e}"),
            })?;
        let meta = SurrogateMeta {
            dims: self.dims.clone(),
            direction: self.direction,
            in_norm: self.in_norm.clone(),
            out_norm: self.out_norm.clone(),
        };
        let text = serde_json::to_string_pretty(&meta).map_err(|e| InverseError::Dl {
            reason: e.to_string(),
        })?;
        std::fs::write(dir.join("meta.json"), text).map_err(|e| InverseError::Dl {
            reason: format!("write meta.json: {e}"),
        })
    }

    /// Rebuild a surrogate from a [`Surrogate::save`] directory, evaluating
    /// on the Flex CPU backend.
    ///
    /// # Errors
    /// [`InverseError::Dl`] on missing/corrupt metadata or a record that
    /// does not validate against the architecture.
    pub fn load(dir: &Path) -> Result<Surrogate> {
        Self::load_on(dir, &Device::flex())
    }

    /// [`Surrogate::load`] with an explicit evaluation device (GPU allowed
    /// here, never in core).
    ///
    /// # Errors
    /// As [`Surrogate::load`].
    pub fn load_on(dir: &Path, device: &Device) -> Result<Surrogate> {
        let text =
            std::fs::read_to_string(dir.join("meta.json")).map_err(|e| InverseError::Dl {
                reason: format!("read {}: {e}", dir.join("meta.json").display()),
            })?;
        let meta: SurrogateMeta = serde_json::from_str(&text).map_err(|e| InverseError::Dl {
            reason: e.to_string(),
        })?;
        let net = Mlp::new(&meta.dims, device)
            .try_load_file(dir.join("net.burnpack"))
            .map_err(|e| InverseError::Dl {
                reason: format!("load net.burnpack: {e}"),
            })?;
        Ok(Surrogate {
            net,
            in_norm: meta.in_norm,
            out_norm: meta.out_norm,
            direction: meta.direction,
            dims: meta.dims,
            device: device.clone(),
        })
    }
}

/// Train a surrogate on the Flex CPU backend (see [`train_on`]).
///
/// # Errors
/// [`InverseError::Dl`] for empty datasets, dimension mismatches, or
/// invalid configuration.
pub fn train(
    train_set: &Dataset,
    val_set: &Dataset,
    direction: Direction,
    cfg: &TrainConfig,
) -> Result<(Surrogate, TrainReport)> {
    train_on(train_set, val_set, direction, cfg, &Device::flex())
}

/// Train a surrogate on `device` (base device; autodiff is applied
/// internally). Best-validation-epoch weights are retained, matching the
/// notebook discipline of task 9.7.
///
/// # Errors
/// [`InverseError::Dl`] when either dataset is empty, when the chosen
/// direction's dimensions disagree between train and val, or when the
/// configuration is invalid (no hidden layers, zero epochs/batch).
pub fn train_on(
    train_set: &Dataset,
    val_set: &Dataset,
    direction: Direction,
    cfg: &TrainConfig,
    device: &Device,
) -> Result<(Surrogate, TrainReport)> {
    if train_set.is_empty() {
        return Err(InverseError::Dl {
            reason: "training split is empty".into(),
        });
    }
    if val_set.is_empty() {
        return Err(InverseError::Dl {
            reason: "validation split is empty".into(),
        });
    }
    if cfg.hidden.is_empty() || cfg.epochs == 0 || cfg.batch == 0 {
        return Err(InverseError::Dl {
            reason: format!(
                "invalid TrainConfig: hidden {:?}, epochs {}, batch {}",
                cfg.hidden, cfg.epochs, cfg.batch
            ),
        });
    }
    let (d_in, d_out) = match direction {
        Direction::Forward => (train_set.dim_params, train_set.dim_features),
        Direction::Inverse => (train_set.dim_features, train_set.dim_params),
    };
    let check = |ds: &Dataset, name: &str| -> Result<()> {
        let (i, o) = match direction {
            Direction::Forward => (ds.dim_params, ds.dim_features),
            Direction::Inverse => (ds.dim_features, ds.dim_params),
        };
        if i != d_in || o != d_out {
            return Err(InverseError::Dl {
                reason: format!("{name} split is {i}→{o}, expected {d_in}→{d_out}"),
            });
        }
        Ok(())
    };
    check(train_set, "training")?;
    check(val_set, "validation")?;

    // Axis selectors and normalization kinds. Params get per-axis min–max
    // (every slot spans its full range); features get one shared min–max
    // so background-level pixels are not amplified to full scale.
    type AxisSel = fn(&Sample) -> &[f64];
    fn params_of(s: &Sample) -> &[f64] {
        &s.params
    }
    fn features_of(s: &Sample) -> &[f64] {
        &s.features
    }
    let (in_of, out_of): (AxisSel, AxisSel) = match direction {
        Direction::Forward => (params_of, features_of),
        Direction::Inverse => (features_of, params_of),
    };

    // Normalization fitted on the training split only (no val leakage).
    let (in_norm, out_norm) = match direction {
        Direction::Forward => (
            Norm::fit(train_set.samples.iter().map(params_of), d_in)?,
            Norm::fit_shared(train_set.samples.iter().map(features_of), d_out)?,
        ),
        Direction::Inverse => (
            Norm::fit_shared(train_set.samples.iter().map(features_of), d_in)?,
            Norm::fit(train_set.samples.iter().map(params_of), d_out)?,
        ),
    };

    // Encode the whole dataset once (f64 -> normalized -> f32 rows).
    let encode = |s: &Sample| -> Result<(Vec<f32>, Vec<f32>)> {
        let x: Vec<f32> = in_norm
            .encode(in_of(s))?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let y: Vec<f32> = out_norm
            .encode(out_of(s))?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        Ok((x, y))
    };
    let mut train_rows = Vec::with_capacity(train_set.len());
    for s in &train_set.samples {
        train_rows.push(encode(s)?);
    }
    let mut val_rows = Vec::with_capacity(val_set.len());
    for s in &val_set.samples {
        val_rows.push(encode(s)?);
    }

    let mut dims = vec![d_in];
    dims.extend(cfg.hidden.iter().copied());
    dims.push(d_out);

    let train_device = device.clone().autodiff();
    train_device.seed(cfg.seed);
    let mut model = Mlp::new(&dims, &train_device);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = MseLoss::new();

    let mut best_model = model.clone();
    let mut best_val = f64::INFINITY;
    let mut history = Vec::with_capacity(cfg.epochs);

    for epoch in 0..cfg.epochs {
        let mut order: Vec<usize> = (0..train_rows.len()).collect();
        let mut rng = StdRng::seed_from_u64(cfg.seed ^ (epoch as u64).wrapping_add(0x9E37_79B9));
        rand::seq::SliceRandom::shuffle(order.as_mut_slice(), &mut rng);

        let mut epoch_loss = 0.0f64;
        let mut n_batches = 0usize;
        for chunk in order.chunks(cfg.batch) {
            let (x, y) = pack(&train_rows, chunk, d_in, d_out, &train_device);
            let pred = model.forward(x);
            let loss = loss_fn.forward(pred, y, Reduction::Mean);
            let data = loss.clone().inner().into_data();
            epoch_loss += data.as_slice::<f32>().map_err(|e| InverseError::Dl {
                reason: e.to_string(),
            })?[0] as f64;
            n_batches += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(cfg.lr, model, grads);
        }
        let train_loss = epoch_loss / n_batches as f64;

        // Validation: one full forward pass (graph built and dropped).
        let val_idx: Vec<usize> = (0..val_rows.len()).collect();
        let (x, y) = pack(&val_rows, &val_idx, d_in, d_out, &train_device);
        let val_data = loss_fn
            .forward(model.forward(x), y, Reduction::Mean)
            .inner()
            .into_data();
        let val_loss = val_data.as_slice::<f32>().map_err(|e| InverseError::Dl {
            reason: e.to_string(),
        })?[0] as f64;

        if val_loss < best_val {
            best_val = val_loss;
            best_model = model.clone();
        }
        history.push(EpochRecord {
            train_loss,
            val_loss,
        });
    }

    // Rebuild the best snapshot on a non-autodiff evaluation device.
    let eval_device = device.clone();
    let net = Mlp::new(&dims, &eval_device).load_record(best_model.into_record());
    Ok((
        Surrogate {
            net,
            in_norm,
            out_norm,
            direction,
            dims,
            device: eval_device,
        },
        TrainReport {
            history,
            best_val_loss: best_val,
        },
    ))
}

/// Pack rows `[i0, i1, ...]` of an encoded dataset into one batch tensor
/// pair (inputs, targets).
fn pack(
    rows: &[(Vec<f32>, Vec<f32>)],
    idx: &[usize],
    d_in: usize,
    d_out: usize,
    device: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let mut xf = Vec::with_capacity(idx.len() * d_in);
    let mut yf = Vec::with_capacity(idx.len() * d_out);
    for &i in idx {
        xf.extend_from_slice(&rows[i].0);
        yf.extend_from_slice(&rows[i].1);
    }
    let x = Tensor::<2>::from_data(TensorData::new(xf, [idx.len(), d_in]), device);
    let y = Tensor::<2>::from_data(TensorData::new(yf, [idx.len(), d_out]), device);
    (x, y)
}
