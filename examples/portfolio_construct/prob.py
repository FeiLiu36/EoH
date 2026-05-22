# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


def _portfolio_sharpe(asset_returns: np.ndarray, selected: np.ndarray) -> float:
    """Annualised Sharpe ratio of an equal-weighted portfolio."""
    port_ret = asset_returns[selected].mean(axis=0)
    mu = port_ret.mean()
    sigma = port_ret.std()
    if sigma < 1e-10:
        return 0.0
    return float(mu / sigma * np.sqrt(252))


class PortfolioConstruct(BaseProblem):
    """Portfolio construction via greedy asset selection.

    The LLM designs score_assets, called at each greedy step to rank
    the remaining candidate assets.  The asset with the highest score is
    added to the portfolio and the process repeats until n_select assets
    are chosen.

    Greedy loop:
      For step = 1 … n_select:
        1. Call score_assets(asset_returns, selected_indices, candidate_indices).
        2. Add the highest-scoring candidate to the portfolio.

    Fitness: mean negative annualised Sharpe ratio across training instances
             (lower = better → higher Sharpe ratio).
    """

    template_program = '''
def score_assets(asset_returns: np.ndarray,
                 selected_indices: np.ndarray,
                 candidate_indices: np.ndarray) -> np.ndarray:
    """Score candidate assets for greedy portfolio inclusion.

    Args:
        asset_returns:      (n_assets, n_periods) float array of historical
                            daily returns for every asset in the universe
        selected_indices:   integer array of already-selected asset indices
                            (empty on the first call)
        candidate_indices:  integer array of candidate asset indices to score
    Returns:
        scores: 1-D float array of length len(candidate_indices);
                higher score means more preferred for inclusion
    """
    # Default: rank by individual Sharpe ratio (return / volatility)
    scores = np.array([
        asset_returns[i].mean() / (asset_returns[i].std() + 1e-8)
        for i in candidate_indices
    ])
    return scores
'''

    task_description = (
        "Given historical daily asset returns for a universe of assets, "
        "design a scoring function that ranks candidate assets at each step "
        "of a greedy portfolio construction process. "
        "The score should balance expected return, volatility, and "
        "diversification (low correlation with already-selected assets). "
        "Assets are added one at a time by picking the highest-scoring "
        "candidate. The goal is to maximise the equal-weighted portfolio's "
        "annualised Sharpe ratio."
    )

    def __init__(self, n_assets: int = 20, n_select: int = 5,
                 n_periods: int = 252, n_instance: int = 5,
                 timeout: int = 40, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_assets = n_assets
        self.n_select = n_select
        self.n_periods = n_periods
        self.n_instance = n_instance
        self.instance_data = GetData(n_instance, n_assets, n_periods).generate_instances()

    def _greedy_select(self, asset_returns: np.ndarray, score_fn) -> np.ndarray | None:
        """Run the greedy selection loop and return selected asset indices."""
        candidates = list(range(self.n_assets))
        selected = []

        for _ in range(self.n_select):
            if not candidates:
                break
            candidate_arr = np.array(candidates, dtype=int)
            selected_arr = np.array(selected, dtype=int)

            try:
                scores = score_fn(asset_returns, selected_arr, candidate_arr)
            except Exception:
                return None

            scores = np.asarray(scores, dtype=float).flatten()
            if len(scores) != len(candidates) or not np.all(np.isfinite(scores)):
                return None

            best = int(np.argmax(scores))
            selected.append(candidates[best])
            candidates.pop(best)

        return np.array(selected, dtype=int)

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        neg_sharpes = []
        for returns in self.instance_data:
            selected = self._greedy_select(returns, callable_func)
            if selected is None or len(selected) < self.n_select:
                return None
            neg_sharpes.append(-_portfolio_sharpe(returns, selected))
        return float(np.mean(neg_sharpes))
