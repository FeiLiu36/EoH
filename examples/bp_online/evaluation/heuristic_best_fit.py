# Best Fit baseline for online bin packing.
#
# Best Fit assigns the current item to the feasible bin that will be left with
# the SMALLEST remaining capacity after the item is placed (i.e. the tightest
# fit). The framework places the item in the bin with the highest score, so we
# score each feasible bin by the negative of its leftover capacity.
import numpy as np


def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score each feasible bin for Best Fit.

    Args:
        item: size of the current item to assign.
        bins: remaining capacities of feasible bins (all >= item).
    Returns:
        scores: higher for bins that leave the least space after placing item.
    """
    return -(bins - item)
