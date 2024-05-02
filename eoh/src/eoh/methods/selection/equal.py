import random
def parent_selection(population, m):
    parents = random.choices(population, k=m)
    return parents