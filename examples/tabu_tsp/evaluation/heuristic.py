# Baseline heuristic: classic best non-tabu move with simple aspiration criterion.
# Replace the body of `score_moves` with the best function found by EoH.

import numpy as np


def score_moves(delta_costs: np.ndarray, is_tabu_mask: np.ndarray,
                best_cost: float, current_cost: float, tabu_ages: np.ndarray,
                iteration: int, max_iterations: int) -> np.ndarray:
    scores = np.full(len(delta_costs), -np.inf)
    non_tabu = ~is_tabu_mask
    progress = iteration / max_iterations

    # Exponential diversification bonus for tabu moves, scaled by age and progress
    tabu_diversification = np.where(is_tabu_mask, np.exp(tabu_ages / (max_iterations * 0.5)) * (0.5 + progress), 0.0)

    # Dynamic aspiration: allow tabu move if it beats best or if improving and tabu age > threshold that decreases over time
    aspiration_threshold = 0.4 * max_iterations * (1 - 0.5 * progress)
    aspiration_improve = current_cost + delta_costs < best_cost
    aspiration_tabu = (delta_costs < 0) & (tabu_ages > aspiration_threshold)
    aspiration = is_tabu_mask & (aspiration_improve | aspiration_tabu)

    # Base score: negative delta (higher for improvements) with Cauchy noise decaying over iterations
    noise_scale = 0.5 * (1 - progress)
    noise = np.random.standard_cauchy(len(delta_costs)) * noise_scale
    base_score = -delta_costs + noise

    # Non-tabu moves: base score
    scores[non_tabu] = base_score[non_tabu]

    # Allowed tabu moves: base score + diversification bonus
    scores[aspiration] = base_score[aspiration] + 10.0 * tabu_diversification[aspiration]

    # Early exploration: allow some worsening moves with probability based on delta magnitude
    if progress < 0.15:
        worsening = (delta_costs > 0) & non_tabu
        accept_prob = np.clip(1.0 / (1.0 + delta_costs[worsening] / (current_cost + 1e-8)), 0, 0.3)
        accept_mask = np.random.uniform(0, 1, size=worsening.sum()) < accept_prob
        scores[worsening] = np.where(accept_mask, scores[worsening] - 0.3 * delta_costs[worsening], -np.inf)

    return scores


