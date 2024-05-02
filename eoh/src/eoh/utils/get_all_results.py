import json
import numpy as np

ng = 10
n = 10
n_start = 0

def code2file(algorithm,code,n,ng):
    with open("algorithm_"+str(ng)+"_"+str(n)+".txt", "w") as file:
    # Write the code to the file
        file.write(algorithm)
        file.write(code)        
    return

obj_list = np.zeros((ng,n))
for i in range(n_start,n_start+ng):
    ### Get result ###
    #Load JSON data from file
    with open("population_generation_"+str(i)+".json") as file:
        data = json.load(file)


    #Print each individual in the population
    na = 0
    for individual in data:
        code = individual['code']
        alg = individual['algorithm']
        obj = individual['objective']
        
        #code2file(alg,code,na,i)
        #print(obj)
        obj_list[i-n_start,na] = obj
        na +=1


import numpy as np
import matplotlib.pyplot as plt

# Set font family to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# Generate x-axis values for number of generations
generations = np.arange(1, obj_list.shape[0] + 1)
best_objective = np.min(obj_list, axis=1)
mean_objective = np.mean(obj_list, axis=1)

# Set figure size
plt.figure(figsize=(10, 6), dpi=80)

# Plot objective value vs. number of generations for all samples as scatter points
for i in generations:
    plt.scatter(i*np.ones(n), obj_list[i-1, :], color='tab:blue', alpha=0.6,s=200)
    

#plt.plot(generations, 0.663*np.ones_like(generations), color='k', linestyle=':', label='Human-GLS',linewidth=3.0)

#plt.plot(generations, 0.175*np.ones_like(generations), color='purple', linestyle='--', label='Human-EBGLS',linewidth=3.0)

#plt.plot(generations, 0.035*np.ones_like(generations), color='orange', linestyle='-.', label='Human-KGLS',linewidth=3.0)

# Plot mean and best objectives
#plt.plot(generations, mean_objective, label='Mean', color='orange')
plt.plot(generations, best_objective, label='AEL-GLS', color='r',linewidth=3.0)

# Set plot title and labels with enlarged font size
#plt.title('Objective Value vs. Number of Generations', fontsize=18)
plt.xlabel('Number of Generations', fontsize=18)
plt.ylabel('Average Makespan', fontsize=20)

# Set y-axis range
plt.ylim([3150, 3200])

# Add legend with enlarged font size
#plt.legend(fontsize=18)



# Add scatter legend with enlarged font size
plt.scatter([], [], color='tab:blue', alpha=0.6, label='Algorithms',s=200)  # Empty scatter plot for legend
#plt.legend(scatterpoints=1, frameon=False, labelspacing=1, fontsize=20)
plt.legend(scatterpoints=1, frameon=True, labelspacing=1, fontsize=20, fancybox=True, facecolor='gainsboro')
# Adjust ticks and grid
plt.xticks(np.arange(1, obj_list.shape[0] + 1, 2),fontsize=18)
plt.yticks(np.arange(3150, 3200, 10),fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.savefig('plot.png')   # Save the plot as a file
plt.savefig('plot.pdf') 
plt.show()
