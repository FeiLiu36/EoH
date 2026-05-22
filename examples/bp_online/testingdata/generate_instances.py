import numpy as np
import pickle

def generate_weibull_dataset(num_instances, num_items, clipping_limit):
    dataset = {}

    for i in range(num_instances):
        items = []

        # Generate random samples from Weibull(45, 3) distribution
        samples = np.random.weibull(3, num_items) * 45

        # Clip the samples at the specified limit
        samples = np.clip(samples, 1, clipping_limit)

        # Round the item sizes to the nearest integer
        sizes = np.round(samples).astype(int)

        # Add the items to the instance
        for size in sizes:
            items.append(size)

        if num_items not in dataset:
            dataset[num_items] = []

        dataset[num_items].append(items)

    return dataset
    
def read_dataset_from_file(filename):
    with open(filename, 'rb') as file:
        dataset = pickle.load(file)

    transformed_dataset = {}

    for num_items, instances in dataset.items():
        transformed_dataset[f"Weibull {num_items}k"] = {}
        for instance_num, items in enumerate(instances, 1):
            instance_name = f"test_{instance_num}"
            instance_data = {
                "capacity": 100,
                "num_items": num_items,
                "items": items
            }
            transformed_dataset[f"Weibull {num_items}k"][instance_name] = instance_data

    return transformed_dataset

# Generate training dataset with 5 instances and 5,000 items
training_dataset = generate_weibull_dataset(5, 5000, 100)

# Generate validation dataset with 5 instances and 5,000 items
validation_dataset = generate_weibull_dataset(5, 5000, 100)

# Generate test dataset with 5 instances and 5,000 items
test_dataset_05k = generate_weibull_dataset(5, 1000, 100)


# Generate test dataset with 5 instances and 5,000 items
test_dataset_1k = generate_weibull_dataset(5, 2000, 100)

# Generate test dataset with 5 instances and 5,000 items
test_dataset_5k = generate_weibull_dataset(5, 5000, 100)

# Generate test dataset with 5 instances and 10,000 items
test_dataset_10k = generate_weibull_dataset(5, 10000, 100)

# Generate test dataset with 1 instance and 100,000 items
test_dataset_100k = generate_weibull_dataset(1, 100000, 100)

# Write datasets to pickle files
def write_dataset_to_file(dataset, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dataset, file)

write_dataset_to_file(training_dataset, 'training_dataset_5k.pkl')
#write_dataset_to_file(validation_dataset, 'validation_dataset.pkl')
#write_dataset_to_file(test_dataset_05k, 'test_dataset_1k.pkl')
#write_dataset_to_file(test_dataset_1k, 'test_dataset_2k.pkl')
#write_dataset_to_file(test_dataset_5k, 'test_dataset_5k.pkl')
#write_dataset_to_file(test_dataset_10k, 'test_dataset_10k.pkl')
#write_dataset_to_file(test_dataset_100k, 'test_dataset_100k.pkl')


#test_dataset_5k = read_dataset_from_file('test_dataset_5k.pkl')
#test_dataset_10k = read_dataset_from_file('test_dataset_10k.pkl')
#test_dataset_100k = read_dataset_from_file('test_dataset_100k.pkl')
print(test_dataset_5k)


