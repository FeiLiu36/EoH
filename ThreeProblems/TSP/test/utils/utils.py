
from numba import jit

def tour_to_edge_attribute(G, tour):
    in_tour = {}
    tour_edges = list(zip(tour[:-1], tour[1:]))
    for e in G.edges:
        in_tour[e] = e in tour_edges or tuple(reversed(e)) in tour_edges
    return in_tour

#@jit(nopython=True)
def tour_cost(dis_m, tour):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += dis_m[e]
    return c

@jit(nopython=True)
def tour_cost_2End(dis_m, tour2End):
    c=0
    s = 0
    e = tour2End[0,1]
    for i in range(tour2End.shape[0]):
        c += dis_m[s,e]
        s = e
        e = tour2End[s,1]
    return c


def is_equivalent_tour(tour_a, tour_b):
    if tour_a == tour_b[::-1]:
        return True
    if tour_a == tour_b:
        return True
    return False
