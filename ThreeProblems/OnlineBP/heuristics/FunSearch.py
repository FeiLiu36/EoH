import numpy as np

def score(item, bins):
  max_bin_cap = max(bins)
  score = (bins - max_bin_cap)**2 / item + bins**2 / (item**2)
  score += bins**2 / item**3
  score[bins > item] = -score[bins > item]
  score[1:] -= score[:-1]
  return score

def score(item, bins):
  max_bin_cap = max(bins)
  score = (bins - max_bin_cap)**2 / (item**2)
  score[bins > item] = -score[bins > item]
  score[1:] -= score[:-1]
  return score