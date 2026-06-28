# First Fit baseline for online bin packing.
#
# First Fit assigns the current item to the FIRST (lowest-index) bin that can
# hold it. `bins` contains the remaining capacities of the feasible bins in
# ascending bin-index order, so we give the earliest bin the highest score and
# let the framework's argmax select it.
import numpy as np


def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score each feasible bin for First Fit.

    Args:
        item: size of the current item to assign.
        bins: remaining capacities of feasible bins (all >= item), in
              ascending bin-index order.
    Returns:
        scores: strictly decreasing with bin index, so the first feasible
                bin is always preferred.
    """
    return -np.arange(len(bins))
