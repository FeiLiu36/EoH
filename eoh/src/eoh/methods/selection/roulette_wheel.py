import random
def parent_selection(population, m):
    fitness_values = [1 / (fit['objective'] + 1e-6) for fit in population]
    fitness_sum = sum(fitness_values)
    probs = [fit / fitness_sum for fit in fitness_values]
    parents = random.choices(population, weights=probs, k=m)
    return parents