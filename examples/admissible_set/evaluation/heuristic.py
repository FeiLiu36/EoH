def priority(el, n, w):
    """Baseline: score by sum of absolute element values normalised by dimension."""
    return sum(abs(x) for x in el) / n
