#!/usr/bin/env python
# coding: utf-8


import time

import numpy as np
import torch

from gls import gls

import importlib

def evaluateGLS(is_test,time_limit,iter_max,perturbation_moves,coords,dis_matrixs,instances_name,instances_scale,debug_mode,algorithm):

    filename = "results_TSPLib_"+algorithm+".txt"

    # Create a new file
    with open(filename, "w") as file:
        pass
        
    print(algorithm)
           
    for nins in range(len(dis_matrixs)):
        dis_matrix = dis_matrixs[nins]
        # neighbor = neighbors[nins]
        coord = coords[nins]
        t = time.time()

        init_tour = gls.nearest_neighbor(dis_matrix,0)

        init_cost = gls.tour_cost(dis_matrix, init_tour)
  

        best_tour, best_cost = gls.guided_local_search (coord,dis_matrix, init_tour, init_cost,
                                                                                 t + time_limit, iter_max,
                                                                                 perturbation_moves=perturbation_moves,
                                                                                 first_improvement=False,
                                                                                 algorithm = algorithm)

        with open(filename, "a") as f:
            f.write(f"File: {instances_name[nins]} ")
            f.write(f"Best Cost: {best_cost*instances_scale[nins]} ")
            f.write(f"Time Cost: {time.time() - t} \n")

        print(f"File: {instances_name[nins]} Best Cost: {best_cost*instances_scale[nins]} Time Cost: {time.time() - t}")

        # nins += 1

    return 
