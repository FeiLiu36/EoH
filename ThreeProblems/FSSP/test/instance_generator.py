import random

n = 50
num_instances = 64

for instance in range(num_instances):
    m = random.randint(2, 20)
    processing_times = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]
    file_name = f"{instance + 1}.txt"
    with open(file_name, 'w') as file:
        file.write(f"{n} {m}\n")
        for i in range(n):
            for j in range(m):
                file.write(f"{j} {processing_times[i][j]} ")
            file.write("\n")