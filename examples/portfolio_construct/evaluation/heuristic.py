# example heuristic
# replace it with your own heuristic designed by EoH
import numpy as np


def score_assets(asset_returns: np.ndarray,
                 selected_indices: np.ndarray,
                 candidate_indices: np.ndarray) -> np.ndarray:
    """Score by individual Sharpe ratio penalised for correlation with already-selected assets."""
    scores = np.zeros(len(candidate_indices))
    for k, asset in enumerate(candidate_indices):
        r = asset_returns[asset]
        sharpe = r.mean() / (r.std() + 1e-8)

        # Diversification penalty: mean absolute correlation with selected assets
        if len(selected_indices) > 0:
            corrs = [
                abs(np.corrcoef(r, asset_returns[s])[0, 1])
                for s in selected_indices
            ]
            div_penalty = float(np.mean(corrs))
        else:
            div_penalty = 0.0

        scores[k] = sharpe - div_penalty
    return scores
