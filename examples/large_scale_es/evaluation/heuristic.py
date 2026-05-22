# Baseline: standard sep-CMA-ES diagonal rank-1 + rank-mu update.
# Replace the body of `adapt_diagonal_cov` with the best rule found by EoH.

import numpy as np

def adapt_diagonal_cov(
    d: np.ndarray,
    p_c: np.ndarray,
    weights: np.ndarray,
    y_k: np.ndarray,
    c1: float,
    cmu: float,
    cc: float,
    hsig: float,
    n: int,
    generation: int,
    max_generations: int,
) -> np.ndarray:
    # Progress factor (0\u21921)
    t = generation / max_generations
    
    # --- Conservative multiplicative update (baseline) ---
    rank1_mul  = c1  * (p_c ** 2 + (1 - hsig) * cc * (2 - cc) * d)
    rankmu_mul = cmu * np.einsum('i,ij->j', weights, y_k ** 2)
    d_mul = (1 - c1 - cmu) * d + rank1_mul + rankmu_mul
    
    # --- Aggressive log-space update (quadratic path penalty) ---
    log_d = np.log(np.maximum(d, 1e-20))
    # Scale learning rates with progress (higher early, lower late)
    alpha = 0.8 * (1 - t) + 0.2
    c1_log = c1 * alpha
    cmu_log = cmu * alpha
    # Quadratic penalty on evolution path (discourages extreme directional growth)
    penalty = 0.5 * (p_c ** 2) / (d + 1e-20)
    rank1_log = c1_log * (p_c ** 2 / d + (1 - hsig) * cc * (2 - cc)) - penalty
    rankmu_log = cmu_log * np.einsum('i,ij->j', weights, y_k ** 2) / d
    log_d_new = log_d + rank1_log + rankmu_log
    d_log = np.exp(log_d_new)
    
    # --- Per-dimension mixture via relative entropy ---
    # Estimate KL divergence between current and candidate distributions (Gaussian approx.)
    kl_mul = 0.5 * (d / d_mul + d_mul / d - 2.0)
    kl_log = 0.5 * (d / d_log + d_log / d - 2.0)
    # Soft mixing weight: prefer update with smaller KL (more stable)
    eps = 1e-8
    w_mul = 1.0 / (kl_mul + eps)
    w_log = 1.0 / (kl_log + eps)
    w_total = w_mul + w_log
    w_mul /= w_total
    w_log /= w_total
    
    # Blend updates
    d_new = w_mul * d_mul + w_log * d_log
    
    # Final safeguard: prevent collapse or explosion
    d_new = np.clip(d_new, 1e-20, 1e10)
    return d_new
    