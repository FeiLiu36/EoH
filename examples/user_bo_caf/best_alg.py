import torch
import torch.distributions as tdist

def utility(train_x, train_y, best_x, best_y, test_x, mean_test_y, std_test_y, cost_test_y, budget_used, budget_total):

    # Calculate the expected improvement (EI) for each test input    
    with torch.no_grad():
        z = (mean_test_y - best_y) / torch.sqrt(torch.pow(std_test_y, 2) + torch.pow(train_y.std(), 2))        
        ei = (mean_test_y - best_y) * tdist.Normal(0, 1).cdf(z) + torch.sqrt(torch.pow(std_test_y, 2) + torch.pow(train_y.std(), 2)) * tdist.Normal(0, 1).log_prob(z).exp()
        
    # Calculate the reduction in uncertainty through mutual information    
    with torch.no_grad():        
        mi = torch.max(torch.tensor(0), (torch.log(torch.pow(std_test_y, 2) + torch.pow(train_y.std(), 2)) - torch.log(torch.pow(train_y.std(), 2))) / 2)    
    
    # Adjust the utility value based on the reduction in uncertainty, cost of evaluation, and remaining budget    
    utility_value = (ei * (1 - mi)) - torch.exp(-cost_test_y) * (budget_total - budget_used)    
    
    # Calculate the distance of the test input from the known solutions    
    distance_to_known = torch.cdist(test_x, train_x)    
    
    # Adjust the utility value based on the diversity and coverage of the unobserved test input space    
    diversity_coverage_factor = torch.mean(torch.min(distance_to_known, dim=1).values)    
    
    utility_value += diversity_coverage_factor    
    
    return utility_value
     