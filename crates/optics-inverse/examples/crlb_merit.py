# CRLB A-optimal user merits — optics-inverse task 9.12 worked example
# (ADR-9.11 acceptance test; GitHub optics-studio#4 / Paperclip GEN-47).
#
# Conventions (mirror crates/optics-inverse/src/py.rs docs):
#   array : read-only float64 view of Observation.data,
#           shape (C, B, S, Ny, Nx), layout [channel][band][scene][row][col]
#   meta  : dict — channels, bands, scenes, dims, pitch, fd_steps, frame_len
#
# Scene stencil (S = 5, the driver's renderer.scenes order):
#   [(0,0), (+h,0), (-h,0), (0,+h), (0,-h)],  h = meta['fd_steps'][0]
# so dI/dx0 and dI/dy0 are central differences over SCENE frames of the
# intensity channel only — the adjoint seed stays confined to channel 0
# (ADR-9.9 scope: derivative channels have no adjoint path yet).
#
# Physics (hypercam coded_mask.py / crlb_from_images, reduced to the
# position-estimation class): per pixel p, expected photo-electrons
#   E_p = PHOTONS * I_p / sum(I) + BACKGROUND      (Poisson term)
#   Var_p = E_p + READ_VAR                          (+ gaussian read noise)
# Fisher information for theta = (x0, y0):
#   F = sum_p (1/Var_p) * grad I_p grad I_p^T       (2x2, summed over bands)
# A-optimal trace with ridge regularization, reported in um^2:
#   M = tr( (F + RIDGE*I)^-1 * outer(SCALES, SCALES) )
#
# merit_event implements the event-camera log-ratio variant (CodedEvents,
# Shah et al.): the measurement is O = log I_t - log I_{t-tau}, whose
# delta-method variance is ~2/E_p, so the Jacobian is grad log I and the
# weight is E_p/2. No analytic gradient is attached — the driver runs it
# through the FD-on-obs fallback (gradcheck::FdObsMerit) to demonstrate
# the layered gradient story.

import numpy as np

PHOTONS = 1.0e5      # peak-normalized photon budget per bin (counts)
BACKGROUND = 1.0     # dark/background counts
READ_VAR = 1.0       # gaussian read-noise variance, counts^2
RIDGE = 1e-12        # Fisher ridge (1/m^2 units)
SCALES = np.array([1e6, 1e6])   # theta scales (m -> um); trace in um^2


def _parts(array, meta):
    """(I0, gx, gy, w) for the stencil: intensity center frame, scene-FD
    derivatives, and the inverse-variance weights."""
    ci = meta["channels"].index("intensity")
    I = array[ci]                       # (B, S, Ny, Nx)
    h = meta["fd_steps"][0]
    I0 = I[:, 0]
    gx = (I[:, 1] - I[:, 2]) / (2.0 * h)
    gy = (I[:, 3] - I[:, 4]) / (2.0 * h)
    S = np.maximum(I0.sum(axis=(-2, -1), keepdims=True), 1e-300)
    E = PHOTONS * I0 / S + BACKGROUND
    return I0, gx, gy, E


def _jac(I0, gx, gy, event):
    if event:
        # FD-on-obs perturbs pixels by an absolute-scale step; tails can go
        # negative. Floor log-ratio denominators at a fraction of the frame
        # peak so the event Fisher stays finite under perturbation.
        floor = 1e-6 * np.maximum(I0.max(axis=(-2, -1), keepdims=True), 1e-300)
        eps = np.maximum(I0, floor)
        return np.stack((gx / eps, gy / eps), axis=-1)
    return np.stack((gx, gy), axis=-1)


def _counts_weight(E, event):
    # Poisson + gaussian read noise (standard) / delta-method log-ratio
    # variance 2/E for the event output (weight E/2). The floor keeps the
    # weights finite if FD perturbation drives E negative at pixel tails.
    if event:
        return np.maximum(E, 0.0) / 2.0
    return 1.0 / np.maximum(E + READ_VAR, 1e-12)


def _fisher_trace(J, w):
    F = np.einsum("...i,...j,...->...ij", J, J, w).reshape(-1, 2, 2).sum(axis=0)
    Cinv = np.linalg.inv(F + RIDGE * np.eye(2))
    return F, Cinv, float((np.diag(Cinv) * SCALES * SCALES).sum())


def merit(array, meta):
    """Poisson+gaussian A-optimal CRLB trace (um^2), minimized."""
    I0, gx, gy, E = _parts(array, meta)
    J = _jac(I0, gx, gy, event=False)
    w = _counts_weight(E, event=False)
    return _fisher_trace(J, w)[2]


def merit_event(array, meta):
    """Event-camera log-ratio variant of the same A-optimal trace."""
    I0, gx, gy, E = _parts(array, meta)
    J = _jac(I0, gx, gy, event=True)
    w = _counts_weight(E, event=True)
    return _fisher_trace(J, w)[2]


def merit_grad(array, meta):
    """Analytic dM/ddata for `merit` (validated against FD-on-obs by the
    driver with gradcheck::assert_close)."""
    ci = meta["channels"].index("intensity")
    h = meta["fd_steps"][0]
    I0, gx, gy, E = _parts(array, meta)
    J = _jac(I0, gx, gy, event=False)
    w = _counts_weight(E, event=False)
    _, Cinv, _ = _fisher_trace(J, w)

    # dM/dF = -(Cinv @ diag(s^2) @ Cinv)
    G = -(Cinv * (SCALES * SCALES)[:, None]) @ Cinv
    Q = np.einsum("ij,...i,...j->...", G, J, J)      # (B, Ny, Nx)
    U = np.einsum("ij,...j->...i", G, J)             # (B, Ny, Nx, 2)

    # dE_p/dI0_q = PHOTONS * (delta/S - I0_p / S^2)
    S = np.maximum(I0.sum(axis=(-2, -1), keepdims=True), 1e-300)
    qv = Q / (E + READ_VAR) ** 2
    dI0 = -PHOTONS * (qv / S - (qv * I0).sum(axis=(-2, -1), keepdims=True) / S**2)

    g = np.zeros(array.shape, dtype=np.float64)
    g[ci, :, 0] = dI0
    g[ci, :, 1] = w * U[..., 0] / h
    g[ci, :, 2] = -w * U[..., 0] / h
    g[ci, :, 3] = w * U[..., 1] / h
    g[ci, :, 4] = -w * U[..., 1] / h
    return g.ravel()
