class GetPrompts():
    def __init__(self):
        self.prompt_task = "Given a black-box maximization optimization problem with unknown heterogenous cost of evaluation, \
suppose I have trained the surrogate and cost model based on the evaluated samples, \
you need to create a totally new utility (different from the utilities in the literature) \
that quantifies the benefit of the given unobserved test input and budget information in each iteration. "
        self.prompt_func_name = "utility"
        self.prompt_func_inputs = ["train_x","train_y","best_x","best_y","test_x","mean_test_y","std_test_y",
                                   "cost_ test_y","budget_used","budget_total"]
        self.prompt_func_outputs = ["utility_value"]
        self.prompt_inout_inf = "The meanings of above inputs are: evaluated historical inputs and function values, \
the best optimal solution and corresponding maximum function values so far, the unobserved test input, \
the predicted mean and std of the function value at the unobserved test input, the cost spent when observing the test input, \
the budget has been used and the total given budget, respectively. The output is the utility value. \
All the inputs and output are torch.tensor with dtype=torch.float64. \
The input sizes are (n_train,dim), (n_train,1), (1,dim), (1), (n_test, dim), (n_test), (n_test), (n_test), (1), (1), respectively. \
The output size is (n_test). Here n_train is the number of evaluated samples, dim is the dimension of input variables, n_test is the number of test points."
        self.prompt_other_inf = "You must make sure the size of returned output utility_value is (n_test), \
so pay attention to the sizes of new variables you created in the code. You can use any mathematical operation on the inputs, \
please try to be creative and make full use of the inputs information."

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