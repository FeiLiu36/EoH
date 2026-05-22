import importlib
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prob import _portfolio_sharpe


class Evaluation:
    def __init__(self, dataset, n_test: int, n_select: int = 5):
        self.instance_data = dataset[:n_test]
        self.n_assets = dataset[0].shape[0]
        self.n_select = n_select

    def _greedy_select(self, asset_returns: np.ndarray, score_fn) -> np.ndarray | None:
        candidates = list(range(self.n_assets))
        selected = []
        for _ in range(self.n_select):
            if not candidates:
                break
            candidate_arr = np.array(candidates, dtype=int)
            selected_arr = np.array(selected, dtype=int)
            scores = score_fn(asset_returns, selected_arr, candidate_arr)
            scores = np.asarray(scores, dtype=float).flatten()
            if len(scores) != len(candidates) or not np.all(np.isfinite(scores)):
                return None
            best = int(np.argmax(scores))
            selected.append(candidates[best])
            candidates.pop(best)
        return np.array(selected, dtype=int)

    def evaluate(self) -> float:
        mod = importlib.reload(importlib.import_module("heuristic"))
        sharpes = []
        for returns in self.instance_data:
            selected = self._greedy_select(returns, mod.score_assets)
            if selected is None or len(selected) < self.n_select:
                sharpes.append(0.0)
            else:
                sharpes.append(_portfolio_sharpe(returns, selected))
        return float(np.mean(sharpes))
