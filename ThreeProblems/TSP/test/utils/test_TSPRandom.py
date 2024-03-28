#!/usr/bin/env python
# coding: utf-8


import time

import numpy as np
import torch

from gls import gls

import importlib

def evaluateGLS(is_test,time_limit,iter_max,perturbation_moves,coords,dis_matrixs,instances_name,instances_scale,opt_costs,debug_mode,algorithm):


    gaps = []
    n = len(coords[0])
    filename = "results_random_TSP"+str(n)+algorithm+".txt"

    # Create a new file
    with open(filename, "w") as file:
        pass
    start_time = time.time()
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

        gap = (best_cost / opt_costs[nins] - 1) * 100
        
        gaps.append(gap)

        print(f"File: {instances_name[nins]} Gap: {gap:.5f}% Time Cost: {(time.time() - t):.2f}")

        if is_test and nins%10==0: print(f"gap on instance {nins+1} is {gap:.5f}%, average gap is {np.mean(gaps):.5f}%")

        # nins += 1
    with open(filename, "a") as f:
        f.write(f"Everage gap: {np.mean(gaps):.5f}, Everage time:{time.time()-start_time:.1f} ")

    return 
