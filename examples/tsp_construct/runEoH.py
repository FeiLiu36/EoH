from eoh import eoh
from eoh.utils.getParas import Paras

# Parameter initilization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = "tsp_construct", #['tsp_construct','bp_online']
                llm_api_endpoint = "XXX", # set your LLM endpoint
                llm_api_key = "XXX",   # set your key
                llm_model = "gpt-3.5-turbo",
                ec_pop_size = 4, # number of samples in each population
                ec_n_pop = 4,  # number of populations
                exp_n_proc = 4,  # multi-core parallel
                exp_debug_mode = False)

# initilization
evolution = eoh.EVOL(paras)

# run 
evolution.run()