class GetPrompts():
    def __init__(self):
        self.prompt_task = "Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. \
You should create a heuristic for me to update the edge distance matrix."
        self.prompt_func_name = "update_edge_distance"
        self.prompt_func_inputs = ['edge_distance', 'local_opt_tour', 'edge_n_used']
        self.prompt_func_outputs = ['updated_edge_distance']
        self.prompt_inout_inf = "'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation."
        self.prompt_other_inf = "All are Numpy arrays."

    def get_task(self):
        return self.prompt_task
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf
    
#     def get_prompt_create(self):
#         prompt_content = "Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. \
# You should create a strategy for me to update the edge distance matrix. \
# Provide a description of the new strategy in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new strategy. The code is a Python function called 'update_edge_distance' that takes three inputs 'edge_distance', 'local_opt_tour', 'edge_n_used', and outputs the 'updated_edge_distance', \
# where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation. All are Numpy arrays. Pay attention to the format and do not give additional explanation."
#         return prompt_content
    

#     def get_prompt_crossover(self,indiv1,indiv2):
#         prompt_content = "Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. \
# I have two strategies with their codes to update the distance matrix. \
# The first strategy and the corresponding code are: \n\
# Strategy description: "+indiv1['algorithm']+"\n\
# Code:\n\
# "+indiv1['code']+"\n\
# The second strategy and the corresponding code are: \n\
# Strategy description: "+indiv2['algorithm']+"\n\
# Code:\n\
# "+indiv2['code']+"\n\
# Please help me create a new strategy that is totally different from them but can be motivated from them. \
# Provide a description of the new strategy in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new strategy. The code is a Python function called 'update_edge_distance' that takes three inputs 'edge_distance', 'local_opt_tour', 'edge_n_used', and outputs the 'updated_edge_distance', \
# where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation. All are Numpy arrays. Pay attention to the format and do not give additional explanation."
#         return prompt_content
    
#     def get_prompt_mutation(self,indiv1):
#         prompt_content = "Task: Given a set of nodes with their coordinates, \
# you need to find the shortest route that visits each node once and returns to the starting node. \
# The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. \
# I have a strategy with its code to select the next node in each step as follows. \
# Strategy description: "+indiv1['algorithm']+"\n\
# Code:\n\
# "+indiv1['code']+"\n\
# Please assist me in creating a modified version of the strategy provided. \
# Provide a description of the new strategy in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new strategy. The code is a Python function called 'update_edge_distance' that takes three inputs 'edge_distance', 'local_opt_tour', 'edge_n_used', and outputs the 'updated_edge_distance', \
# where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation. All are Numpy arrays. Pay attention to the format and do not give additional explanation."
#         return prompt_content
