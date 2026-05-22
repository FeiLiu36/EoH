# example heuristic
# replace it with your own heuristic designed by EoH
import numpy as np


def generate_mutation(current_solution: np.ndarray, sigma: float,
                      success_rate: float, n_dims: int,
                      iteration: int, max_evals: int) -> np.ndarray:
    """Adaptive mixed Gaussian–Cauchy noise.

    When the success rate is low (< 0.2, stagnating), inject heavy-tailed
    Cauchy noise alongside Gaussian noise to escape local optima.
    When the success rate is normal or high, use pure Gaussian (exploiting).
    """
    gaussian = np.random.normal(0.0, 1.0, n_dims)
    # Cauchy weight increases as success drops below target (0.2)
    cauchy_weight = float(np.clip(0.2 - success_rate, 0.0, 0.2)) / 0.2  # 0→1
    if cauchy_weight > 0:
        cauchy = np.random.standard_cauchy(n_dims)
        # clip heavy Cauchy outliers to stay numerically reasonable
        cauchy = np.clip(cauchy, -10.0, 10.0)
        noise = (1.0 - cauchy_weight) * gaussian + cauchy_weight * cauchy
    else:
        noise = gaussian
    return sigma * noise
