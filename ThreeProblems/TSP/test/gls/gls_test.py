import time

import networkx as nx
import numpy as np
from numba import jit
from ael import alg
from utils import utils
from . import operators
import random

#@jit(nopython=True) 
def nearest_neighbor_2End(dis_matrix, depot):
    tour = [depot]
    n = len(dis_matrix)
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i,j]) for j in nodes if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)

    tour.append(depot)
    route2End = np.zeros((n,2))
    route2End[0,0] = tour[-2]
    route2End[0,1] = tour[1]
    for i in range(1,n):
        route2End[tour[i],0] = tour[i-1]
        route2End[tour[i],1] = tour[i+1]
    return route2End

@jit(nopython=True) 
def local_search(init_tour, init_cost, D,N, first_improvement=False):
    cur_route, cur_cost = init_tour, init_cost
    # search_progress = []

    improved = True
    n = 0

    # print(cur_route)
    # input()
    while improved:

        improved = False
        # for operator in [operators.two_opt_a2a, operators.relocate_a2a]:

        delta, new_tour = operators.two_opt_a2a(cur_route, D,N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

            # search_progress.append({
            #     'time': time.time(),
            #     'cost': cur_cost
            # })
            # print(delta)


        delta, new_tour = operators.relocate_a2a(cur_route, D,N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

            # search_progress.append({
            #     'time': time.time(),
            #     'cost': cur_cost
            # })          
            # print(delta)
            # print(cur_route)
            # input()

        n += 1
    return cur_route, cur_cost

@jit(nopython=True)
def calculate_width(depot , center_of_gravity, node_i, node_j):

    # Calculate the perpendicular axis (axis connecting the depot and center of gravity)
    depot_x, depot_y = depot[0],depot[1]
    gravity_x, gravity_y = center_of_gravity[0],center_of_gravity[1]

    perpendicular_axis_x = depot_y - gravity_y
    perpendicular_axis_y = gravity_x - depot_x

    # Calculate the distance between nodes i and j along the perpendicular axis
    distance = abs((node_i[0] - node_j[0]) * perpendicular_axis_x + (node_i[1] - node_j[1]) * perpendicular_axis_y)

    return distance
@jit(nopython=True)
def find_indices(Route):
    Route = Route[:-1]
    n = len(Route)
    indices = []

    # Iterate over the range of IDs
    for i in range(n):
        index = Route.index(i)
        indices.append(index)


    return indices

@jit(nopython=True) 
def route2tour(route):
    s = 0
    tour=[]
    for i in range(len(route)):
        tour.append(route[s,1])
        s = route[s,1]    
    return tour

@jit(nopython=True) 
def tour2route(tour):
    n = len(tour)
    route2End = np.zeros((n,2))
    route2End[tour[0],0] = tour[-1]
    route2End[tour[0],1] = tour[1]
    for i in range(1,n-1):
        route2End[tour[i],0] = tour[i-1]
        route2End[tour[i],1] = tour[i+1] 
    route2End[tour[n-1],0] = tour[n-2]
    route2End[tour[n-1],1] = tour[0]
    return route2End


#@jit(nopython=True)
def guided_local_search(coords, edge_weight, nearest_indices,  init_tour, init_cost, t_lim,iter_max, perturbation_moves,
                        first_improvement,algorithm):
    

    cur_route, cur_cost = local_search(init_tour, init_cost, edge_weight,nearest_indices, first_improvement)
    best_route, best_cost = cur_route, cur_cost

    cur_route_local = cur_route

    length = len(edge_weight[0])

    if algorithm in ["LS"]:
        pass
    elif algorithm in ["KGLS_c","KGLS_r"]:

        center = np.mean(coords,axis=1)
        # center[0] = np.mean(coords[0,:])
        # center[1] = np.mean(coords[1,:])

        perturbation_moves = 10 # set the defaul for KGLS
        
        k = 0.1 * cur_cost / length   # initialize k according to 

        iter_i = 0
        edge_penalty = np.zeros((length,length))

        while  iter_i<iter_max  and time.time() < t_lim:
            #guide = guides[iter_i % len(guides)]  # option change guide ever iteration (as in KGLS)

            # perturbation
            moves = 0
            while moves < perturbation_moves:
                if algorithm == "KGLS_r" and iter_i%3 == 0:
                    # penalize edge
                    max_util = 0
                    max_util_e = None
                    for e in zip(cur_route[:-1], cur_route[1:]):
                        width = calculate_width(coords[0],center,coords[e[0]],coords[e[1]])
                        util = width / (1 + edge_penalty[e[0],e[1]])
                        if util > max_util or max_util_e is None:
                            max_util = util
                            max_util_e = e
                elif algorithm == "KGLS_r" and iter_i%3 == 1:
                    # penalize edge
                    max_util = 0
                    max_util_e = None
                    for e in zip(cur_route[:-1], cur_route[1:]):
                        width = calculate_width(coords[0],center,coords[e[0]],coords[e[1]])
                        util = (width + edge_weight[e[0],e[1]]) / (1 + edge_penalty[e[0],e[1]])
                        if util > max_util or max_util_e is None:
                            max_util = util
                            max_util_e = e
                else:
                    # penalize edge
                    max_util = 0
                    max_util_e = None
                    for e in zip(cur_route[:-1], cur_route[1:]):
                        util = edge_weight[e[0],e[1]] / (1 + edge_penalty[e[0],e[1]])
                        if util > max_util or max_util_e is None:
                            max_util = util
                            max_util_e = e                    
                

                edge_penalty[max_util_e[0],max_util_e[1]] += 1.
                edge_penalty[max_util_e[1],max_util_e[0]] += 1.

                edge_weight_guided = edge_weight + k * edge_penalty

                # apply operator to edge
                for n in max_util_e:
                    if n != 0:  # not the depot
                        i = cur_route.index(n)

                        for operator in [operators.two_opt_o2a, operators.relocate_o2a]:
                            moved = False

                            delta, new_tour = operator(cur_route, edge_weight_guided, i, first_improvement)
                            if delta < 0:
                                cur_cost = utils.tour_cost_2End(edge_weight, new_tour)
                                cur_route = new_tour
                                moved = True

                            moves += moved

            # optimisation
            cur_route, cur_cost = local_search(cur_route, cur_cost, edge_weight, first_improvement)
            cur_route_local = cur_route
   
            # search_progress += new_search_progress
            if cur_cost < best_cost:
                best_route, best_cost = cur_route, cur_cost
            #print(str(iter_i)+" current cost = ",cur_cost)
            iter_i += 1

    elif algorithm in ['GLS','EBGLS']:
        
        if algorithm == 'GLS':
            k = 0.1 * cur_cost / length   # initialize k according to 
        elif algorithm == 'EBGLS':
            k = 0.3 * cur_cost / length   # initialize k according to 
            w = 2
            tour_e = cur_route

        iter_i = 0
        edge_penalty = np.zeros((length,length))

        while iter_i<iter_max and time.time() < t_lim:

            max_util = 0
            max_util_e = None

            if algorithm == "EBGLS" and time.time() > t_lim/10:
                for e in zip(np.arange(length), cur_route[:,1]):
                    if e in tour_e:
                        util = edge_weight[e[0],e[1]] / (1 + edge_penalty[e[0],e[1]])
                    else:
                        util = w*edge_weight[e[0],e[1]] / (1 + edge_penalty[e[0],e[1]])
                    if util > max_util or max_util_e is None:
                        max_util = util
                        max_util_e = e
            elif algorithm == 'GLS':
                for e in zip(np.arange(length), cur_route[:,1]):
                    util = edge_weight[e[0],e[1]] / (1 + edge_penalty[e[0],e[1]])
                    if util > max_util or max_util_e is None:
                        max_util = util
                        max_util_e = e

            edge_penalty[max_util_e[0],max_util_e[1]] += 1.
            edge_penalty[max_util_e[1],max_util_e[0]] += 1.

            edge_weight_guided = edge_weight + k * edge_penalty

            # optimisation
            cur_route, cur_cost = local_search(cur_route, cur_cost, edge_weight_guided, first_improvement)

            cur_cost = utils.tour_cost_2End(edge_weight, cur_route)

            # search_progress += new_search_progress
            if cur_cost < best_cost:
                best_route, best_cost = cur_route, cur_cost
            print(str(iter_i)+" current cost = ",cur_cost)
            iter_i += 1

            if algorithm == "EBGLS" and iter_i % 50 == 0:
                tour_e = zip(best_route[:-1], best_route[1:])


    elif algorithm == "AELGLS":
        iter_i = 0
        edge_penalty = np.zeros((length,length))

        ruin_max = max(int(length/10),20)
        ruin_min = min(int(length/10),20)
        while iter_i < iter_max and time.time()<t_lim:

            for move in range(perturbation_moves):


                cur_tour, best_tour = route2tour(cur_route), route2tour(best_route)
                #partial_tour = alg.perturb_route(edge_weight, coords, np.array(cur_tour))
                print(len(coords))
                partial_tour = alg.perturb_route(edge_weight, coords, np.arange(len(coords)))
                
                # if not isinstance(partial_tour, list):
                #     partial_tour = partial_tour.tolist()
                # if partial_tour[0] != ruin_nodes[0]:
                #     partial_tour.insert(0,ruin_nodes[0])
                # if partial_tour[-1] != ruin_nodes[-1]:
                #     partial_tour.append(ruin_nodes[-1])
                
                if set(partial_tour) != set(cur_tour):
                    continue
                # if (location+ruin_size<length):
                #     new_tour = cur_tour[:location] + partial_tour +  cur_tour[location+ruin_size:]      
                # else:         
                #     new_tour = cur_tour[ruin_size-length+location:location] + partial_tour
                # #print(new_tour)
                # new_tour0= [int(element) for element in new_tour]
                partial_tour0 = [int(element) for element in partial_tour]
                cur_route = tour2route(partial_tour0).astype(int)
                   
            cur_route, cur_cost = local_search(cur_route, cur_cost, edge_weight, nearest_indices, first_improvement)
            cur_cost = utils.tour_cost_2End(edge_weight,cur_route)

            if cur_cost < best_cost:
                best_route, best_cost = cur_route, cur_cost
            print(str(iter_i)+" current cost = ",cur_cost)
            iter_i += 1

            if iter_i%50==0 or cur_cost>best_cost*1.01:
                cur_route = best_route

    else:

        print(f"the algorithm {algorithm} is not implemented !")

    return best_route, best_cost
