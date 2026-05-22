# Template heuristic: standard CMA-ES rank-1 + rank-mu covariance update (baseline).
# Replace the body of `update_covariance` with the best rule found by EoH.

import numpy as np
def update_covariance(
    C: np.ndarray,
    p_c: np.ndarray,
    weights: np.ndarray,
    y_k: np.ndarray,
    c1: float,
    cmu: float,
    cc: float,
    hsig: float,
    n: int,
) -> np.ndarray:
    # Geodesic interpolation factor
    tau = 0.3 * (c1 + cmu)

    # Compute natural gradient direction (tangent vector in symmetric space)
    # Align evolution path: gradient of dot(p_c, inv(C) p_c)
    p_c_flat = p_c
    grad_path = -np.outer(p_c_flat, p_c_flat) / (np.dot(p_c_flat, np.linalg.solve(C, p_c_flat)) + 1e-20)

    # Align selected steps: gradient of sum w_i * y_i^T inv(C) y_i
    grad_mu = np.zeros((n, n))
    for i, w in enumerate(weights):
        yi = y_k[i]
        grad_mu += w * np.outer(yi, yi) / (np.dot(yi, np.linalg.solve(C, yi)) + 1e-20)
    grad_mu = -grad_mu

    # Combine gradients
    G = (c1 / (c1 + cmu + 1e-20)) * grad_path + (cmu / (c1 + cmu + 1e-20)) * grad_mu

    # Symmetrise tangent vector
    G = 0.5 * (G + G.T)

    # Geodesic step: C_new = C^{1/2} expm(tau * C^{-1/2} G C^{-1/2}) C^{1/2}
    try:
        C_sqrt = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        C += 1e-12 * np.eye(n)
        C_sqrt = np.linalg.cholesky(C)
    C_sqrt_inv = np.linalg.solve(C_sqrt, np.eye(n))
    M = C_sqrt_inv @ G @ C_sqrt_inv.T
    M = 0.5 * (M + M.T)
    # Matrix exponential
    eigvals, eigvecs = np.linalg.eigh(M)
    exp_diag = np.exp(tau * eigvals)
    expM = eigvecs @ np.diag(exp_diag) @ eigvecs.T
    C_new = C_sqrt.T @ expM @ C_sqrt
    C_new = 0.5 * (C_new + C_new.T)

    # Ensure positive definiteness
    evals_new = np.linalg.eigvalsh(C_new)
    if np.min(evals_new) < 1e-12:
        C_new += np.eye(n) * (1e-12 - np.min(evals_new))

    return C_new
    