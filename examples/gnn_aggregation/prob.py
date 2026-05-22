# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem
from get_instance import GetData


class GNNAggregation(BaseProblem):
    """EoH task: design the neighborhood aggregation step of a GNN for node classification.

    The LLM designs `aggregate_neighbors`. A fixed 3-layer GNN harness applies
    this function iteratively on Stochastic Block Model graphs. Node features are
    then clustered with k-means (k=2) and evaluated against ground-truth community
    labels. Lower error (1 - accuracy) is better.

    Calibration (n_nodes=30, n_feat=4, n_layers=3, n_instance=20):
      - Identity (no aggregation): ~0.38
      - Mean aggregation (baseline): ~0.15
      - GCN symmetric normalisation: ~0.09
    """

    template_program = '''
def aggregate_neighbors(node_features: np.ndarray, adj_matrix: np.ndarray, iteration: int) -> np.ndarray:
    """Aggregate features from neighboring nodes for each node in a GNN layer.

    This is the core message-passing step. It updates every node's feature vector
    by collecting and combining information from its direct neighbors.

    Args:
        node_features: (n, d) float array — current feature matrix (n nodes, d dims)
        adj_matrix:    (n, n) float array — binary symmetric adjacency matrix
        iteration:     int — current GNN layer index (0-based)
    Returns:
        aggregated: (n, d) float array — updated node features after neighbor aggregation
    """
    degree = adj_matrix.sum(axis=1, keepdims=True)
    degree = np.where(degree == 0, 1.0, degree)
    return adj_matrix @ node_features / degree
'''

    task_description = (
        "Design a neighborhood aggregation function for a Graph Neural Network (GNN) "
        "applied to node classification on graphs with community structure. "
        "The GNN iteratively applies the aggregation function for 3 layers. "
        "Initial node features contain a weak community signal buried in heavy Gaussian noise. "
        "The aggregation should leverage the graph structure (Stochastic Block Model) "
        "to amplify the within-community signal and suppress noise, enabling k-means "
        "clustering of the final node representations to correctly recover the two communities. "
        "Classic strategies include mean aggregation, sum aggregation, max aggregation, "
        "degree-normalised aggregation (GCN-style: D^{-1/2}(A+I)D^{-1/2}), and "
        "attention-weighted aggregation. You may design adaptive rules that change "
        "behaviour across GNN layers (e.g. stronger self-weight in later layers to "
        "counter over-smoothing). The goal is to minimise the node classification "
        "error averaged over 20 Stochastic Block Model graph instances."
    )

    def __init__(self, n_nodes: int = 30, n_feat: int = 4, n_layers: int = 3,
                 n_instance: int = 20, timeout: int = 40, n_processes: int = 1):
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.n_nodes = n_nodes
        self.n_feat = n_feat
        self.n_layers = n_layers
        self.instance_data = GetData(n_instance, n_nodes, n_feat).generate_instances()

    def _classification_error(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Run k-means(k=2) and return error accounting for label permutation."""
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        pred = kmeans.fit_predict(features)
        acc = max(
            np.mean(pred == labels),
            np.mean(pred != labels),
        )
        return float(1.0 - acc)

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        errors = []
        for adj, node_features, labels in self.instance_data:
            feats = node_features.copy()
            for layer in range(self.n_layers):
                feats = callable_func(feats, adj, layer)
                if not isinstance(feats, np.ndarray) or feats.ndim < 2:
                    return None
                if feats.shape[0] != len(labels):
                    return None
                if not np.all(np.isfinite(feats)):
                    return None
            errors.append(self._classification_error(feats, labels))
        return float(np.mean(errors))
