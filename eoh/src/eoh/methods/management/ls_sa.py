import math
import random

def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    return math.exp(((old_cost - new_cost)/old_cost) / temperature)

def population_management(population, new, temperature):
    current_best = population[0]
    
    if (new['objective'] != None)  and (len(population) == 0  or acceptance_probability(current_best['objective'], new['objective'], temperature) > random.random()):
        population[0] = new
        
    return