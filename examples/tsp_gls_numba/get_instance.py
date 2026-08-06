import os
import pickle as pkl

import numpy as np


class GetData:
    """Loads pre-generated TSP instances with known optimal tour costs.

    The training set (TSPAEL64.pkl) holds 64 random 100-node Euclidean TSP
    instances together with their optimal tour costs, used to compute the
    optimality gap during evolution.
    """

    def __init__(self, n_instance, datafile=None):
        self.n_instance = n_instance
        if datafile is None:
            datafile = os.path.join(os.path.dirname(__file__),
                                    'TrainingData', 'TSPAEL64.pkl')
        self.datafile = datafile

    def load_instances(self):
        with open(self.datafile, 'rb') as f:
            data = pkl.load(f)
        coords = [np.asarray(c) for c in data['coordinate'][:self.n_instance]]
        instances = [np.asarray(d) for d in data['distance_matrix'][:self.n_instance]]
        opt_costs = list(data['cost'][:self.n_instance])
        return coords, instances, opt_costs
