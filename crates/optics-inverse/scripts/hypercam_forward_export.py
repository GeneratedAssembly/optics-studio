#!/usr/bin/env python3
"""Export a trained thickness map + reference forward render for the
`optics-inverse` hypercam cross-check (task 9.12 validation).

Provenance: the optical model is copied VERBATIM from
`hypercam/src/hypercam/notebooks/coded_mask.py` (author: Rich Baird,
2026-10-12): `SineLayer`/`MaskThicknessMLP` (lines 134-155),
`_angular_spectrum` (lines 208-222), the 4f relay + thin-lens phase laws
(`generate_incoherent_relay_stack`, lines 262-300), and the A-optimal
CRLB loss (`crlb_from_images`, lines 435-482, reduced to spatial
parameters). Scale is reduced for a deterministic, seconds-long export:
32x32 grid, single band, 5-position source stencil instead of the
measured morphology stack, 80 training epochs, fixed 500 um DOE gap
(the notebook's DOE_GAP), seed 0.

Outputs (into --outdir):
  thickness.npy  trained mask thickness map (m), float64 (N, N)
  forward.npy    reference Observation.data, float64, layout
                 [channel][band][scene][row][col] with channels
                 [I, dI/dx0, dI/dy0]; the derivative channels are central
                 differences of the scene stack at step h_scene — the
                 exact definition crates/optics-inverse/src/observe.rs
                 uses for its DxDyX/DxDyY channels
  meta.txt       key=value design parameters for the Rust renderer

Run (needs torch, e.g. the hypercam notebooks venv):
  ~/code/hypercam/src/notebooks/.venv/bin/python \
      crates/optics-inverse/scripts/hypercam_forward_export.py
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Reduced-scale design point (mirrored in tests/hypercam_forward.rs)
# ---------------------------------------------------------------------------
N = 32                      # grid samples per side
PITCH = 20e-6               # sample pitch (m)
LAMBDA = 532e-9             # single band (m)
INDEX = 1.5                 # DOE material refractive index (n_lambda)
MAX_THICKNESS = 5e-6        # MLP output scale (m), notebook line 423
Z_GAP = 500e-6              # input plane -> DOE (m), notebook DOE_GAP
F_RELAY = 5e-3              # relay focal length (m)
APERTURE_R = 200e-6         # relay stop radius at the lens planes (m)
W0 = 150e-6                 # Gaussian input waist (m)
H_SCENE = 1e-6              # scene stencil half-step (m)
EPOCHS = 80
SEED = 0
PHOTONS = 1e5               # photons_per_bin (crlb_from_images default)
BACKGROUND = 1.0            # background_photons
RIDGE = 1e-8                # fisher ridge (crlb_from_images default)
SCALES = (50.0, 50.0)       # parameter_scales (spatial) — notebook default


# ---------------------------------------------------------------------------
# Verbatim from coded_mask.py lines 109-155 (imports trimmed)
# ---------------------------------------------------------------------------
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                             1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0,
                                             np.sqrt(6 / self.linear.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class MaskThicknessMLP(nn.Module):
    def __init__(self, in_features=2, hidden_features=128, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(in_features, hidden_features, is_first=True, omega_0=30),
            SineLayer(hidden_features, hidden_features, omega_0=30),
            nn.Linear(hidden_features, out_features),
            nn.Softplus() # Enforce non-negative physical thickness
        )

    def forward(self, coords):
        thickness = self.net(coords)
        return thickness


# ---------------------------------------------------------------------------
# Verbatim from coded_mask.py lines 208-222 (_angular_spectrum)
# ---------------------------------------------------------------------------
def _angular_spectrum(field, wavelength, distance, pixel_pitch):
    """Propagate a complex field on a fixed Cartesian grid."""
    h, w = field.shape[-2:]
    fy = torch.fft.fftfreq(h, d=pixel_pitch)
    fx = torch.fft.fftfreq(w, d=pixel_pitch)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    argument = 1.0 - (wavelength * fx) ** 2 - (wavelength * fy) ** 2
    propagating = argument.clamp_min(0.0)
    transfer = torch.exp(1j * (2 * torch.pi / wavelength) * distance * torch.sqrt(propagating))
    transfer = transfer * (argument >= 0).to(transfer.dtype)
    spectrum = torch.fft.fft2(field, norm="ortho")
    return torch.fft.ifft2(spectrum * transfer, norm="ortho")


# ---------------------------------------------------------------------------
# 4f relay, reduced from coded_mask.py lines 262-300 (single band, uniform
# aperture/coords; same phase laws and step order)
# ---------------------------------------------------------------------------
def _relay_stack(field, doe, wavelength):
    yy, xx = torch.meshgrid(
        (torch.arange(N) - (N - 1) / 2) * PITCH,
        (torch.arange(N) - (N - 1) / 2) * PITCH,
        indexing="ij",
    )
    r2 = xx.square() + yy.square()
    relay_aperture = (r2 <= APERTURE_R**2).to(field.dtype)
    lens_phase = -torch.pi * r2 / (wavelength * F_RELAY)

    def thin_lens(x):
        return x * relay_aperture * torch.exp(1j * lens_phase)

    x = _angular_spectrum(field, wavelength, Z_GAP, PITCH) * doe
    x = _angular_spectrum(x, wavelength, F_RELAY, PITCH)
    x = thin_lens(x)
    x = _angular_spectrum(x, wavelength, 2 * F_RELAY, PITCH)
    x = thin_lens(x)
    return _angular_spectrum(x, wavelength, F_RELAY, PITCH)


# ---------------------------------------------------------------------------
# A-optimal CRLB loss, reduced from crlb_from_images (lines 435-482):
# spatial parameters only, frame_stride=1, weights 1/(photons*I + b).
# ---------------------------------------------------------------------------
def crlb_loss(mean, dx, dy):
    scales = torch.tensor(SCALES, dtype=mean.dtype)
    jacobian = torch.stack((dx, dy), dim=-1) * scales
    expected = PHOTONS * mean + BACKGROUND
    weights = 1.0 / expected.clamp_min(1e-12)
    j = jacobian.reshape(-1, 2)
    w = weights.reshape(-1)
    fisher = torch.einsum("ni,nj,n->ij", j, j, w)
    identity = torch.eye(2, dtype=fisher.dtype)
    scaled_covariance = torch.linalg.solve(fisher + RIDGE * identity, identity)
    covariance = scaled_covariance * scales[:, None] * scales[None, :]
    return torch.diagonal(covariance).sum(), fisher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        default=str(__import__("pathlib").Path(__file__).resolve().parent.parent
                    / "tests" / "data" / "hypercam"),
    )
    args = ap.parse_args()
    outdir = __import__("pathlib").Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float64)

    yy, xx = torch.meshgrid(
        (torch.arange(N) - (N - 1) / 2) * PITCH,
        (torch.arange(N) - (N - 1) / 2) * PITCH,
        indexing="ij",
    )
    coords = torch.stack((xx, yy), dim=-1)
    flat_coords = coords.reshape(-1, 2)

    mlp = MaskThicknessMLP()
    raw_gap = None  # notebook's trainable gap fixed at DOE_GAP for the export
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Position lattice (a*h, b*h) for a, b in -2..2: the 5 exported scenes
    # plus the central-difference neighbours for their dx/dy channels —
    # the exact ∂I/∂x0 definition (scene central difference) the Rust
    # Renderer uses for its DxDyX/DxDyY channels.
    pos = [(a * H_SCENE, b * H_SCENE) for a in range(-2, 3) for b in range(-2, 3)]
    pidx = {(a, b): (a + 2) * 5 + (b + 2) for a in range(-2, 3) for b in range(-2, 3)}
    scenes_ab = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    def forward(thickness_map):
        wavelength = torch.tensor(LAMBDA)
        phase = (2 * torch.pi / wavelength) * (INDEX - 1.0) * thickness_map
        doe = torch.exp(1j * phase)
        fields = []
        for (x0, y0) in pos:
            amp = torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / W0**2)
            fields.append(_relay_stack(amp, doe, wavelength))
        images = torch.stack([f.abs().square() for f in fields])  # (25, H, W)
        dxs, dys = [], []
        for (a, b) in scenes_ab:
            dxs.append((images[pidx[(a + 1, b)]] - images[pidx[(a - 1, b)]]) / (2 * H_SCENE))
            dys.append((images[pidx[(a, b + 1)]] - images[pidx[(a, b - 1)]]) / (2 * H_SCENE))
        mean = torch.stack([images[pidx[ab]] for ab in scenes_ab])
        return mean, torch.stack(dxs), torch.stack(dys)

    best_loss = float("inf")
    best_thickness = None
    for epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        thickness_map = mlp(flat_coords).reshape(N, N) * MAX_THICKNESS
        mean, dx, dy = forward(thickness_map)
        loss, _fisher = crlb_loss(mean[0], dx[0], dy[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(mlp.parameters()) + ([raw_gap] if raw_gap else []),
                                       max_norm=10.0)
        optimizer.step()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_thickness = thickness_map.detach().clone()
        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch + 1:4d}/{EPOCHS}: A-optimal trace = {float(loss.detach()):.6g}")

    # Final reference render of the best trained map (no grad).
    with torch.no_grad():
        mean, dx, dy = forward(best_thickness)
    channel_stack = torch.stack((mean, dx, dy))
    # Layout [channel][band=1][scene][row][col] == (C, 1, S, N, N).
    forward_obs = channel_stack[:, None].contiguous().numpy().astype(np.float64)

    np.save(outdir / "thickness.npy", best_thickness.numpy().astype(np.float64))
    np.save(outdir / "forward.npy", forward_obs)
    (outdir / "meta.txt").write_text(
        "\n".join(
            [
                "# hypercam coded_mask.py cross-check design point",
                f"n={N}",
                f"pitch={PITCH!r}",
                f"lambda={LAMBDA!r}",
                f"index={INDEX!r}",
                f"z_gap={Z_GAP!r}",
                f"f_relay={F_RELAY!r}",
                f"aperture_r={APERTURE_R!r}",
                f"w0={W0!r}",
                f"h_scene={H_SCENE!r}",
                f"t_max={float(best_thickness.max()) * 1.05!r}",
                f"epochs={EPOCHS}",
                f"seed={SEED}",
                f"best_loss={best_loss!r}",
                "",
            ]
        )
    )
    print(f"wrote {outdir}/thickness.npy, forward.npy, meta.txt")


if __name__ == "__main__":
    main()
