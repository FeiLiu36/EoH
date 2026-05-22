import numpy as np


class GetData:
    """Generate Stochastic Block Model graphs for GNN community-detection benchmarking.

    Each instance is a tuple of (adj_matrix, node_features, community_labels):
      - adj_matrix:    (n, n) binary symmetric adjacency matrix
      - node_features: (n, d) initial feature matrix — community signal buried in noise
      - labels:        (n,) integer community membership (0 or 1)

    Within-community edge probability p_in=0.4, cross-community p_out=0.1.
    Initial features: community signal ±1 corrupted by Gaussian noise (std=5).
    After good GNN aggregation, nodes in the same community should converge to
    similar representations.
    """

    def __init__(self, n_instance: int, n_nodes: int = 30, n_feat: int = 4,
                 p_in: float = 0.4, p_out: float = 0.1, noise_std: float = 5.0):
        self.n_instance = n_instance
        self.n_nodes = n_nodes
        self.n_feat = n_feat
        self.p_in = p_in
        self.p_out = p_out
        self.noise_std = noise_std

    def generate_instances(self):
        np.random.seed(2024)
        instances = []
        half = self.n_nodes // 2
        labels = np.array([0] * half + [1] * (self.n_nodes - half))

        for _ in range(self.n_instance):
            adj = np.zeros((self.n_nodes, self.n_nodes))
            for i in range(self.n_nodes):
                for j in range(i + 1, self.n_nodes):
                    p = self.p_in if labels[i] == labels[j] else self.p_out
                    if np.random.rand() < p:
                        adj[i, j] = adj[j, i] = 1.0

            signal = np.where(labels == 0, 1.0, -1.0)[:, np.newaxis]
            signal = np.tile(signal, (1, self.n_feat))
            noise = np.random.randn(self.n_nodes, self.n_feat) * self.noise_std
            node_features = signal + noise

            instances.append((adj, node_features.copy(), labels.copy()))

        return instances
