# -------
# Evaluaiton code for EoH on Online Bin Packing
#--------
# More results may refer to
# Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, Qingfu Zhang
# "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model"
# ICML 2024, https://arxiv.org/abs/2401.02051.

import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from get_instance import GetData
from evaluation  import Evaluation

eva = Evaluation()

capacity_list = [100, 300, 500]
with open("results.txt", "w") as file:
    for capacity in capacity_list:
        getdate = GetData()
        instances, lb = getdate.get_instances(capacity)
        heuristic_module = importlib.import_module("heuristic")
        heuristic = importlib.reload(heuristic_module)

        for name, dataset in instances.items():
            avg_num_bins = -eva.evaluateGreedy(dataset, heuristic)
            excess = (avg_num_bins - lb[name]) / lb[name]

            result = f'{name}, {capacity}, Excess: {100 * excess:.2f}%'
            print(result)
            file.write(result + "\n")
