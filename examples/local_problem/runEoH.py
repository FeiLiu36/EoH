from eoh import eoh
from eoh.utils.getParas import Paras
from prob import BPONLINE

# Parameter initilization #
paras = Paras() 

# Set your local problem
problem_local = BPONLINE()

# Set parameters #
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = problem_local, # Set local problem, else use default problems
                llm_api_endpoint = "api.chatanywhere.tech", # set your LLM endpoint
                llm_api_key = "sk-poiXyXBUmQQqWxGOHiy09XYLwFB6pHehJQhbV8DVA9HETdRt",   # set your key
                llm_model = "gpt-3.5-turbo",
                ec_pop_size = 4, # number of samples in each population
                ec_n_pop = 4,  # number of populations
                exp_n_proc = 4,  # multi-core parallel
                exp_debug_mode = False,
                eva_numba_decorator = True)

# initilization
evolution = eoh.EVOL(paras)

# run 
evolution.run()