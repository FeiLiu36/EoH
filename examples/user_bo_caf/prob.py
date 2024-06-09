import numpy as np
import importlib
import pickle
import torch
from torch import Tensor
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from botorch.acquisition import AnalyticAcquisitionFunction, ExpectedImprovement
from botorch.utils.transforms import unnormalize, normalize
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from abc import ABC, abstractmethod
from botorch.models.transforms import Log
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.exceptions import OptimizationWarning
from prompts import GetPrompts
import types
import sys
import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)

from joblib import Parallel, delayed

device = torch.device("cpu")
dtype = torch.double

class CostModel(torch.nn.Module, ABC):
    """
    Simple abstract class for a cost model.
    """    
    
    @abstractmethod
    def forward(self, X):
        pass
    

class CostModelGP(CostModel):
    """
    A basic cost model that assumes the cost is positive.
    It models the log cost to guarantee positive cost predictions.
    """

    def __init__(self, X, Y_cost):
        assert torch.all(Y_cost > 0)
        super().__init__()
        gp = SingleTaskGP(train_X=X, train_Y=Y_cost, outcome_transform=Log())
        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll)
        self.gp = gp

    def forward(self, X):
        return torch.exp(self.gp(X).mean)

class CustomAF(AnalyticAcquisitionFunction):
    def __init__(self, alg, model, cost_model, 
                 train_x, train_Y, best_x, best_Y,
                 budget_used, budget_total):
        super().__init__(model=model)
        self.alg = alg
        self.model = model
        self.cost_model = cost_model
        self.train_x = train_x
        self.train_Y = train_Y
        self.best_x = best_x
        self.best_Y = best_Y
        self.budget_used = budget_used
        self.budget_total = budget_total

    @t_batch_mode_transform(expected_q=1)
    def forward(self, test_x: Tensor) -> Tensor:
        mean_test_y, std_test_y = self._mean_and_sigma(test_x)
        test_x_sqz = test_x.squeeze(dim=1)
        cost_test_y = self.cost_model(test_x_sqz)

        # compute the utility value
        utility_value = self.alg.utility(self.train_x, self.train_Y, 
                                self.best_x, self.best_Y, 
                                test_x_sqz, mean_test_y, std_test_y, 
                                cost_test_y, self.budget_used, self.budget_total)

        return utility_value

class ExpectedImprovementWithCost(AnalyticAcquisitionFunction):
    """
    This is the acquisition function EI(x) / c(x) ^ alpha, where alpha is a decay
    factor that reduces or increases the emphasis of the cost model c(x).
    """

    def __init__(self, model, cost_model, best_f, budget_init, budget_used, budget_total):
        super().__init__(model=model)
        self.model = model
        self.cost_model = cost_model
        self.ei = ExpectedImprovement(model=model, best_f=best_f)
        self.alpha = (budget_total-budget_used) / (budget_total-budget_init)

    def forward(self, X):
        return self.ei(X) / torch.pow(self.cost_model(X)[:, 0], self.alpha)


class Evaluation():
    def __init__(self) -> None:
        self.prompts = GetPrompts()
        
        with open('./instance/botorch_problem.pkl', 'rb') as f:
            self.instance_data = pickle.load(f)

        self.n_instance = len(self.instance_data)

    def eval_objective(self, real_problem,x):
        """unnormalize and evalaute a point with objective function"""
        obj = real_problem(unnormalize(x, real_problem.bounds))
        return obj.to(dtype=dtype, device=device)

    def eval_cost(self, real_problem,x):
        """evalaute a point with cost function"""
        x_optimal_unnorm = normalize(real_problem.optimizers, real_problem.bounds)
        cost = torch.exp(-torch.norm(x - x_optimal_unnorm, p=2, dim=1))

        return cost.to(dtype=dtype, device=device)

    def get_initial_points(self, dim, n_pts, seed=0):
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
        return X_init

    def EGO(self, code_string, cost_total, real_problem, random_seed):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # Create a new module object
            heuristic_module = types.ModuleType("heuristic_module")
            # Execute the code string in the new module's namespace
            exec(code_string, heuristic_module.__dict__)
            # Add the module to sys.modules so it can be imported
            sys.modules[heuristic_module.__name__] = heuristic_module
            alg = heuristic_module

            # DoE
            dim = real_problem.dim
            n_init = 2 * dim
            torch.manual_seed(random_seed)
            X_ei = self.get_initial_points(dim, n_init,seed=random_seed)
            Y_ei = self.eval_objective(real_problem,X_ei).unsqueeze(-1) # (n,1)
            Y_cost = self.eval_cost(real_problem,X_ei).unsqueeze(-1) # (n,1)
            
            # build initial obj model
            train_Y = (Y_ei - Y_ei.mean()) / Y_ei.std()
            model = SingleTaskGP(X_ei, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # build initial cost model
            cost_model = CostModelGP(X_ei, Y_cost)
            cost_init = (torch.cumsum(Y_cost[:, 0], dim=0))[-1].item()
            cost_used = (torch.cumsum(Y_cost[:, 0], dim=0))[-1].item()

            # optimize loop
            while cost_used < cost_total:
                best_Y=train_Y.max()[None]
                best_index = train_Y.argmax()
                best_x = X_ei[best_index,None]
                
                # choose af
                # EvolCAF
                ei = CustomAF(alg, model, cost_model, 
                    X_ei, train_Y, best_x, best_Y,
                    budget_used=cost_used, budget_total=cost_total)
                # EI-cool
                # ei = ExpectedImprovementWithCost(model,cost_model,best_Y,
                #     budget_init=cost_init, budget_used=cost_used, budget_total=cost_total)
                # EIpu
                # ei = ExpectedImprovementWithCost(model,cost_model,best_Y,
                #     budget_init=0, budget_used=0, budget_total=cost_total)
                # # EI
                # ei = ExpectedImprovement(model, best_f=train_Y.max())
                
                # optimize af
                candidate, acq_value = optimize_acqf( # candidate (1,dim)
                    ei,
                    bounds=torch.stack(
                        [
                            torch.zeros(dim, dtype=dtype, device=device),
                            torch.ones(dim, dtype=dtype, device=device),
                        ]
                    ),
                    q=1, # sequential
                    num_restarts=20,
                    raw_samples=100,
                )

                # evaluate candidates
                Y_next = self.eval_objective(real_problem,candidate).unsqueeze(-1) # (1,1)
                Y_cost_next = self.eval_cost(real_problem,candidate).unsqueeze(-1) # (1,1)

                # append data
                X_ei = torch.cat((X_ei, candidate), axis=0)
                Y_ei = torch.cat((Y_ei, Y_next), axis=0)
                Y_cost = torch.cat((Y_cost, Y_cost_next), axis=0)

                # update obj model
                train_Y = (Y_ei - Y_ei.mean()) / Y_ei.std()
                model = SingleTaskGP(X_ei, train_Y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

                # update cost model
                cost_model = CostModelGP(X_ei, Y_cost)
                cost_used = (torch.cumsum(Y_cost[:, 0], dim=0))[-1].item()

        best_value = torch.tensor(Y_ei.max().item())
        gap = (torch.abs(real_problem.optimal_value-best_value)).numpy()

        return gap
    
    def EvalEGO(self,code_string):
        N_TRIALS = 10
        #num_cores = multiprocessing.cpu_count()
        time_limit = 60 # maximum 60 seconds for BO loop when training, can be set much larger when testing other AFs
        gap_all = Parallel(n_jobs=20, timeout=time_limit)(
            delayed(self.EGO)(code_string, cost_total=30, 
                              real_problem=instance, random_seed=trial)
            for trial in range(1, N_TRIALS + 1)
            for instance in self.instance_data
        )
        gap_all = np.array(gap_all).reshape(N_TRIALS, len(self.instance_data))
        gap_mean = np.mean(gap_all).astype(np.float64) # scalar
        return gap_mean   
    
    def evaluate(self, code_string):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitness = self.EvalEGO(code_string)
                print("fitness:",fitness)
                return fitness
        except Exception as e:
            #print("Error:", str(e))
            return None

if __name__ == "__main__":
    import inspect
    interface_eval = Evaluation()
    heuristic_module = importlib.import_module("best_alg")
    eva = importlib.reload(heuristic_module)
    source_code = inspect.getsource(eva)
    print(source_code)
    fitness = interface_eval.evaluate(source_code)