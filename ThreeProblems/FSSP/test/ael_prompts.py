
class GetPrompts():
    def __init__(self):
        self.prompt_task = "I have n jobs and m machines. Help me create a novel algorithm to update the execution time matrix and select the top jobs to perturb to avoid being trapped in the local optimum scheduling \
with the final goal of finding a scheduling with minimized makespan."
        self.prompt_func_name = "get_matrix_and_jobs"
        self.prompt_func_inputs = ["current_sequence","time_matrix","m","n"]
        self.prompt_func_outputs = ["new_matrix",'perturb_jobs']
        self.prompt_inout_inf = "The variable 'current_sequence' represents the current sequence of jobs. The variables 'm' and 'n' denote the number of machines and number of jobs, respectively. \
The variable 'time_matrix' is a matrix of size n*m that contains the execution time of each job on each machine. The output 'new_matrix' is the updated time matrix, and 'perturb_jobs' includes the top jobs to be perturbed."
        self.prompt_other_inf = "The matrix and job list are Numpy arrays."

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
