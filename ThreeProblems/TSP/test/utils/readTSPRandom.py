
import pickle as pkl

def read_instance_all(instances_path):

    # Open the pickle file in read mode
    with open(instances_path, 'rb') as file:
        # Load the data from the pickle file
        data = pkl.load(file)

    # Access the individual data elements
    coords = data['coordinate']
    optimal_tour = data['optimal_tour']
    instances = data['distance_matrix']
    opt_costs = data['cost']
    
    return coords,instances,opt_costs

