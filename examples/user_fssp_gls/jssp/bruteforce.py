import numpy as np
from prettytable import PrettyTable


def permute(x, index=0):
    if index+1 >= len(x):
        yield x
    else:
        for p in permute(x, index+1):
            yield p
        for i in range(index+1,len(x)):
            x[index], x[i]=x[i], x[index]
            for p in permute(x,index+1):
                yield p
            x[index], x[i]=x[i], x[index]


def makespan(order, tasks, machines_val):
    times = []
    for i in range(0, machines_val):
        times.append(0)
    for j in order:
        times[0] += tasks[j][0]
        for k in range(1, machines_val):
            if times[k] < times[k-1]:
                times[k] = times[k-1]
            times[k] += tasks[j][k]
    return max(times)


def bruteforce(tasks, machines_val, tasks_val):
    figure = PrettyTable()
    figure.field_names = ["Sequence", "Makespan"]
    print("Starting bruteforce")
    t = []
    min_time = 1000
    for z in range(0, tasks_val):
        t.append(z)
    for p in permute(t):
        tmp = makespan(p, tasks, machines_val)
        figure.add_row([format(p), tmp ])
        if (tmp < min_time):
            min_time = tmp
            best_permute = format(p)
    print(figure)
    print("Min time:", min_time, "  for :", format(best_permute), "permutation")
    print("Bruteforce: DONE")


def read_data(filename):
    file = open(filename, "r")

    tasks_val, machines_val = file.readline().split()
    tasks_val = int(tasks_val)
    machines_val = int(machines_val)

    tasks = np.zeros((tasks_val,machines_val))
    for i in range(tasks_val):
        tmp = file.readline().split()
        for j in range(machines_val):
            tasks[i][j] = int(tmp[j])

    print("Number of tasks: ", tasks_val)
    print("Number of machines: ", machines_val)
    print("Tasks: \n", tasks)
    file.close()
    return tasks_val, machines_val, tasks

if __name__=="__main__":
    TWO_OR_THREE = 3  # Two or three machines algorithm??
    print("Starting program...")

    if (TWO_OR_THREE == 2):
        tasks_num, machines_num, tasks = read_data("data/flowshop_2machines.txt")
    if (TWO_OR_THREE == 3):
        tasks_num, machines_num, tasks = read_data("data/flowshop_3machines.txt")

    bruteforce(tasks, machines_num, tasks_num)
