# Template: Lower Confidence Bound (LCB) acquisition — classic BO baseline.
# Replace the body of `acquisition` with the best function found by EoH.

import numpy as np


def acquisition(mu: np.ndarray, sigma: np.ndarray, f_best: float) -> np.ndarray:
    """Lower Confidence Bound (LCB, kappa=2) — standard BO baseline.

    Replace this with the best acquisition function discovered by EoH.
    """
    kappa = 2.0
    return -mu + kappa * sigma
