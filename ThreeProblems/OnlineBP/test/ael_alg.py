import numpy as np
def score(item, bins):
    diff = bins-item  # remaining capacity
    exp = np.exp(diff)  # exponent term
    sqrt = np.sqrt(diff)  # square root term
    ulti = 1-diff/bins  # utilization term
    comb = ulti * sqrt  # combination of utilization and square root 
    adjust = np.where(diff > (item * 3), comb + 0.8, comb + 0.3)
    # hybrid adjustment term to penalize large bins 
    hybrid_exp = bins / ((exp + 0.7) *exp)
    # hybrid score based on exponent term
    scores = hybrid_exp + adjust
    # sum of hybrid score and adjustment
    return scores