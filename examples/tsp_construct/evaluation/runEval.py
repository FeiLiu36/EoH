from evaluation import Evaluation
import pickle
import time

debug_mode = False
# problem_size = [10,20,50,100,200]
problem_size = [20,50,100]
n_test_ins = 100
print("Start evaluation...")
with open("results.txt", "w") as file:
    for size in problem_size:
        instance_file_name = './testingdata/instance_data_' + str(size)+ '.pkl'
        with open(instance_file_name, 'rb') as f:
            instance_dataset = pickle.load(f)

        eva = Evaluation(size,instance_dataset,n_test_ins,debug_mode)

        time_start = time.time()
        gap = eva.evaluate()

        result = (f"Average dis on {n_test_ins} instance with size {size} is: {gap:7.3f} timecost: {time.time()-time_start:7.3f}")
        print(result)
        file.write(result + "\n")
        


