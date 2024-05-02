test = True
if test:
    import sys
    import os
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    ROOT_PATH = os.path.join(ABS_PATH, "..", "..")
    sys.path.append(ROOT_PATH)  # This is for finding all the modules
    from aell.src.ael import ael
    from aell.src.ael.utils import createFolders
else:
    from aell import ael
    from aell.utils import createFolders



###  LLM settings  ###
api_endpoint = "oa.api2d.site"
api_key = "your key"
llm_model = "gpt-3.5-turbo-1106" 

### output path ###
output_path = "./"  # default folder for ael outputs
createFolders.create_folders(output_path)


load_data = {
    'use_seed' : False,
    'seed_path' : None,
    "use_pop": False,
    "pop_path": output_path + "/ael_results/pops/population_generation_0.json",
    "n_pop_initial": 0
}

### Experimental settings ###
pop_size = 10  # number of algorithms in each population, default = 10
n_pop = 20  # number of populations, default = 10
p1 = 1.0  # probability of crossover, default = 1.0
p2 = 0.5  # probability of mutation, default = 0.5
operators = ['e1','e2','m1','m2']  # evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
operator_weights = [1,1,1,1] # weights for operators, i.e., the probability of use the operator in each iteration , default = [1,1,1,1]

### Debug model ###
debug_mode = False  # if debug



# AEL
print(">>> Start AEL ")

algorithmEvolution = ael.AEL(
    api_endpoint,api_key,llm_model,pop_size,n_pop,operators,m,operator_weights,load_data,output_path,debug_mode
)

# run AEL
algorithmEvolution.run()

print("AEL successfully finished !")
