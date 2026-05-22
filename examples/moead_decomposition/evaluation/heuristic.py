# Template: Tchebycheff decomposition (baseline).
# Replace the body of `custom_decomposition` with the best operator found by EoH.

import numpy as np
import math

def custom_decomposition(F: np.ndarray,
                         weights: np.ndarray,
                         ideal_point: np.ndarray) -> np.ndarray:

    # Adaptive angle-penalized weighted sum
    n_solutions, n_objectives = F.shape
    
    # Normalize objectives relative to ideal point
    normalized = F - ideal_point
    
    # Weighted sum
    weighted_sum = np.sum(normalized * weights, axis=1)
    
    # Compute deviation angle penalty: for each solution, compute its normalized direction
    # and penalty based on distance from weight direction
    norm_F = np.linalg.norm(normalized, axis=1, keepdims=True)
    # Avoid division by zero
    norm_F_safe = np.where(norm_F < 1e-12, 1.0, norm_F)
    direction = normalized / norm_F_safe
    
    # Cosine similarity with weight direction
    cos_sim = np.sum(direction * weights, axis=1)
    # Angle penalty: 1 - cos^2 (sine squared)
    angle_penalty = 1.0 - cos_sim**2
    
    # Density factor: average cosine similarity between weight vectors (simplified)
    # For each weight, compute its maximum similarity to other weights in batch
    weight_norms = np.linalg.norm(weights, axis=1, keepdims=True)
    w_normalized = weights / np.where(weight_norms < 1e-12, 1.0, weight_norms)
    cos_matrix = np.dot(w_normalized, w_normalized.T)
    # Set diagonal to 0 to avoid self-similarity
    np.fill_diagonal(cos_matrix, 0)
    max_cos = np.max(cos_matrix, axis=1)
    density_factor = 1.0 + max_cos  # higher density gives higher penalty
    
    # Final score: weighted sum + angle penalty scaled by density and norm magnitude
    penalty = angle_penalty * density_factor * norm_F_safe.flatten()
    score = weighted_sum + penalty
    
    return score
    
