
class GetPrompts():
    def __init__(self):
        self.prompt_task = "Given a set of nodes with their coordinates, \
you need to find the shortest route that visits each node once and returns to the starting node. \
The task can be solved step-by-step by starting from the current node and iteratively choosing the next node. \
Help me design a novel algorithm that is different from the algorithms in literature to select the next node in each step."
        self.prompt_func_name = "select_next_node"
        self.prompt_func_inputs = ["current_node","destination_node","univisited_nodes","distance_matrix"]
        self.prompt_func_outputs = ["next_node"]
        self.prompt_inout_inf = "'current_node', 'destination_node', 'next_node', and 'unvisited_nodes' are node IDs. 'distance_matrix' is the distance matrix of nodes."
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

if __name__ == "__main__":
    getprompts = GetPrompts()
    print(getprompts.get_task())
