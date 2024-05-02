
class GetPrompts():
    def __init__(self):
        self.prompt_task = (
            "Given an image 'org_img', its adversarial image 'best_adv_img', "
            "and a random normal noise 'std_normal_noise', "
            "you need to design an algorithm to combine them to search for a new adversarial example 'x_new'. "
            "'hyperparams' ranges from 0.5 to 1.5.  It gets larger when "
            "this algorithm outputs more adversarial examples, and vice versa. "
            "It can be used to control the step size of the search."
            "Operations you may use include: adding, subtracting, multiplying, dividing, "
            "dot product, and l2 norm computation. Design an novel algorithm with various search techniques. Your code "
            "should be able to run without further assistance. "
        )
        self.prompt_func_name = "draw_proposals"
        self.prompt_func_inputs = ["org_img","best_adv_img","std_normal_noise", "hyperparams"]
        self.prompt_func_outputs = ["x_new"]
        self.prompt_inout_inf = (
            "'org_img', 'best_adv_img', 'x_new', and 'std_normal_noise' are shaped as (3, img_height, img_width). "
            "The bound of images are [0, 1]. "
            "'std_normal_noise' contains random normal noises. "
            "'hyperparams' is a numpy array with shape (1,). "
        )
        self.prompt_other_inf = ("All inouts are numpy arrays.")

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
    
#     def get_task():
#         task = "Task: Given a set of nodes with their coordinates, \
# you need to find the shortest route that visits each node once and returns to the starting node. \
# The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. \
# You should create a totally new strategy for me (different from the heuristics in the literature) \
# to select the next node in each step, using information including the current node, destination node, unvisited nodes, and distances between them. \
# Provide a description of the new algorithm in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new algorithm. The code is a function called 'select_next_node' that takes inputs 'current_node', 'destination_node', 'unvisited_nodes', and 'distance_matrix', \
# and outputs the 'next_node', where 'current_node', 'destination_node', 'next_node', and 'unvisited_nodes' are node id. Be creative and do not give additional explanation."
#         return prompt_content
    

#     def get_crossover(indiv1,indiv2):
#         prompt_content = "Task: Given a set of nodes with their coordinates, \
# you need to find the shortest route that visits each node once and returns to the starting node. \
# The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. \
# I have two algorithms with their codes to select the next node in each step. \
# The first algorithm and the corresponding code are: \n\
# Algorithm description: "+indiv1['algorithm']+"\n\
# Code:\n\
# "+indiv1['code']+"\n\
# The second algorithm and the corresponding code are: \n\
# Algorithm description: "+indiv2['algorithm']+"\n\
# Code:\n\
# "+indiv2['code']+"\n\
# Please help me create a new algorithm that motivated by the given algorithms. \
# Provide a description of the new algorithm in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new algorithm. The code is a function called 'select_next_node' that takes inputs 'current_node', 'destination_node', 'unvisited_nodes', and 'distance_matrix', \
# and outputs the 'next_node', where 'current_node', 'destination_node', 'next_node', and 'unvisited_nodes' are node id. Be creative and do not give additional explanation."
#         return prompt_content
    
#     def get_mutation(indiv1):
#         prompt_content = "Task: Given a set of nodes with their coordinates, \
# you need to find the shortest route that visits each node once and returns to the starting node. \
# The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. \
# I have an algorithm with its code to select the next node in each step as follows. \
# Algorithm description: "+indiv1['algorithm']+"\n\
# Code:\n\
# "+indiv1['code']+"\n\
# Please assist me in creating a modified version of the algorithm provided. \
# Provide a description of the new algorithm in no more than two sentences. The description must be inside a brace. \
# Provide the Python code for the new algorithm. The code is a function called 'select_next_node' that takes inputs 'current_node', 'destination_node', 'unvisited_nodes', and 'distance_matrix', \
# and outputs the 'next_node', where 'current_node', 'destination_node', 'next_node', and 'unvisited_nodes' are node id. Be creative and do not give additional explanation."
#         return prompt_content
if __name__ == "__main__":
    getprompts = GetPrompts()
    print(getprompts.get_task())
    import numpy as np
    def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
        potentials = []
        for node in unvisited_nodes:
            distance_to_current = distance_matrix[current_node, node]
            distance_to_destination = distance_matrix[node, destination_node]
            potential = distance_to_destination / (distance_to_current + distance_to_destination)
            potentials.append(potential)
            next_node = unvisited_nodes[np.argmax(potentials)]
            return next_node