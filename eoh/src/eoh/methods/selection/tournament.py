
import random

def parent_selection(population, m):
    tournament_size = 2
    parents = []
    while len(parents) < m:
        tournament = random.sample(population, tournament_size)
        tournament_fitness = [fit['objective'] for fit in tournament]
        winner = tournament[tournament_fitness.index(min(tournament_fitness))]
        parents.append(winner)
    return parents