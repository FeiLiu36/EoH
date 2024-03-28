import pickle as pkl
import random

instances_path = 'TSP100.pkl'
output_path = 'random_instances.pkl'

# Open the pickle file in read mode
with open(instances_path, 'rb') as file:
    # Load the data from the pickle file
    data = pkl.load(file)

# Access the individual data elements
coords = data['coordinate']
optimal_tour = data['optimal_tour']
instances = data['distance_matrix']
opt_costs = data['cost']

# Randomly select 64 instances
random_instances = random.sample(range(len(opt_costs)), 64)

# Filter the data based on the selected instances
selected_coords = [coords[i] for i in random_instances]
selected_optimal_tour = [optimal_tour[i] for i in random_instances]
selected_instances = [instances[i] for i in random_instances]
selected_opt_costs = [opt_costs[i] for i in random_instances]

# Create a new data dictionary with the selected instances
selected_data = {
    'coordinate': selected_coords,
    'optimal_tour': selected_optimal_tour,
    'distance_matrix': selected_instances,
    'cost': selected_opt_costs
}

# Write the selected data to a new pickle file
with open(output_path, 'wb') as file:
    pkl.dump(selected_data, file)
