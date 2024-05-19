import os
import numpy as np
def read_taillard_instances(path):
    tasks_val_list = [] 
    machines_val_list = [] 
    tasks_list = []
    names = []

    files = os.listdir(path)

    with open('ub_lb.txt', 'w') as f:

        for filename in files:
            file = open(os.path.join(path, filename), "r")

            while True:
                file.readline()
                line = file.readline().strip()
                if line == "":
                    break

                tasks_val, machines_val, _, ub, lb = map(int, line.split())
                tasks = np.zeros((tasks_val, machines_val))
                file.readline()
                for i in range(machines_val):
                    line = file.readline().strip()
                    tmp = list(map(int, line.split()))
                    for j in range(tasks_val):
                        tasks[j][i] = tmp[j]

                tasks_val_list.append(tasks_val)
                machines_val_list.append(machines_val)
                tasks_list.append(tasks)
                names.append(filename)

                print(f"{filename}, {tasks_val}, {machines_val}, {ub}, {lb}", file=f, flush=True)

            file.close()

    return


ins_path = '../data/Taillard/'
read_taillard_instances(ins_path)