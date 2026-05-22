import numpy as np


class GetData:
    """Generate synthetic asset-return instances via a one-factor market model.

    Each asset's daily return is:
        r_i(t) = beta_i * f(t) + eps_i(t)
    where f(t) is the common market factor and eps_i(t) is idiosyncratic noise.
    This produces a realistic correlation structure while varying expected
    returns and volatilities across assets.
    """

    def __init__(self, n_instance: int, n_assets: int, n_periods: int = 252):
        self.n_instance = n_instance
        self.n_assets = n_assets
        self.n_periods = n_periods

    def generate_instances(self):
        """Return list of (n_assets, n_periods) return arrays."""
        np.random.seed(2024)
        instances = []
        for _ in range(self.n_instance):
            # Market factor: positive drift, realistic daily volatility
            market = np.random.normal(0.0003, 0.012, self.n_periods)

            # Per-asset parameters
            betas = np.random.uniform(0.4, 1.6, self.n_assets)
            mu_idio = np.random.uniform(-0.0002, 0.0008, self.n_assets)
            sigma_idio = np.random.uniform(0.005, 0.025, self.n_assets)

            idio = np.array([
                np.random.normal(mu_idio[i], sigma_idio[i], self.n_periods)
                for i in range(self.n_assets)
            ])
            returns = betas[:, np.newaxis] * market[np.newaxis, :] + idio
            instances.append(returns)
        return instances
