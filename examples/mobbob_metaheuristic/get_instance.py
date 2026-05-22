import numpy as np


# ---------------------------------------------------------------------------
# ZDT benchmark functions (Zitzler, Deb, Thiele, 2000)
# All functions: f(x: ndarray shape (dim,)) -> ndarray shape (2,), minimise both
# ---------------------------------------------------------------------------

def _zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1 — convex Pareto front: f2 = 1 - sqrt(f1), f1 in [0,1]."""
    f1 = float(x[0])
    g  = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = float(g * (1.0 - np.sqrt(f1 / g)))
    return np.array([f1, f2])


def _zdt2(x: np.ndarray) -> np.ndarray:
    """ZDT2 — concave Pareto front: f2 = 1 - f1^2, f1 in [0,1]."""
    f1 = float(x[0])
    g  = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = float(g * (1.0 - (f1 / g) ** 2))
    return np.array([f1, f2])


def _zdt3(x: np.ndarray) -> np.ndarray:
    """ZDT3 — discontinuous Pareto front with five disjoint segments."""
    f1 = float(x[0])
    g  = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = float(g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)))
    return np.array([f1, f2])


def _zdt4(x: np.ndarray) -> np.ndarray:
    """ZDT4 — many local optima; same Pareto shape as ZDT1."""
    f1 = float(x[0])
    n  = len(x)
    g  = 1.0 + 10.0 * (n - 1) + float(
        np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
    )
    f2 = float(g * (1.0 - np.sqrt(f1 / g)))
    return np.array([f1, f2])


# ---------------------------------------------------------------------------
# Pareto / hypervolume utilities (unchanged interface)
# ---------------------------------------------------------------------------

def pareto_front_2d(objectives: np.ndarray) -> np.ndarray:
    """Extract 2-objective Pareto front (both minimised).

    Sorts by f1 ascending; sweeps to keep points with strictly decreasing f2.
    Returns array of shape (k, 2), sorted by f1 ascending.
    """
    if len(objectives) == 0:
        return np.empty((0, 2))
    pts = np.array(objectives, dtype=float)
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]
    pareto, min_f2 = [], np.inf
    for p in pts:
        if p[1] < min_f2:
            pareto.append(p)
            min_f2 = p[1]
    return np.array(pareto)


def hypervolume_2d(objectives: np.ndarray, ref_pt: np.ndarray) -> float:
    """Hypervolume indicator for 2-objective minimisation.

    Computes the area of objective space dominated by the non-dominated front
    of `objectives` and also dominated by `ref_pt`.
    """
    pf = pareto_front_2d(objectives)
    if len(pf) == 0:
        return 0.0
    pf = pf[np.all(pf < ref_pt, axis=1)]
    if len(pf) == 0:
        return 0.0
    hv = 0.0
    for i, p in enumerate(pf):
        next_f1 = pf[i + 1, 0] if i + 1 < len(pf) else ref_pt[0]
        hv += (next_f1 - p[0]) * (ref_pt[1] - p[1])
    return float(hv)


# ---------------------------------------------------------------------------
# Instance factory
# ---------------------------------------------------------------------------

class GetData:
    """ZDT1–ZDT4 benchmark instances for multi-objective metaheuristic design.

    ZDT1: convex Pareto front,        x in [0, 1]^n
    ZDT2: concave Pareto front,       x in [0, 1]^n
    ZDT3: discontinuous Pareto front, x in [0, 1]^n
    ZDT4: many local optima,          x[0] in [0,1], x[1:] in [-5, 5]

    Bounds are stored as per-dimension arrays (lo, hi) each of shape (dim,).
    Reference point (1.1, 1.1) lies just outside all true Pareto fronts.
    """

    _PROBLEMS = [
        ('zdt1', _zdt1),
        ('zdt2', _zdt2),
        ('zdt3', _zdt3),
        ('zdt4', _zdt4),
    ]

    def get_instances(self, dim: int = 10, n_instances: int = 4) -> list[dict]:
        """Return ZDT benchmark instances (cycles through ZDT1–4 if n_instances > 4).

        Each entry is a dict with keys:
            name    – 'zdt1' / 'zdt2' / 'zdt3' / 'zdt4'
            func    – callable f(x: ndarray shape (dim,)) -> ndarray shape (2,)
            dim     – number of decision variables
            bounds  – (lo, hi) each np.ndarray of shape (dim,)
            ref_pt  – hypervolume reference point np.ndarray shape (2,)
            n_obj   – 2
        """
        ref_pt = np.array([1.1, 1.1])
        instances = []
        for i in range(n_instances):
            name, func = self._PROBLEMS[i % len(self._PROBLEMS)]
            if name == 'zdt4':
                lo = np.concatenate([[0.0], np.full(dim - 1, -5.0)])
                hi = np.concatenate([[1.0], np.full(dim - 1,  5.0)])
            else:
                lo = np.zeros(dim)
                hi = np.ones(dim)
            instances.append({
                'name':   name,
                'func':   func,
                'dim':    dim,
                'bounds': (lo, hi),
                'ref_pt': ref_pt,
                'n_obj':  2,
            })
        return instances
