

def population_management(population, new, temperature):
    
    if (new['objective'] != None) and (len(population) == 0 or new['objective'] < population[0]['objective']):
        population[0] = new
    return