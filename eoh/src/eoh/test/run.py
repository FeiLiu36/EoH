### Test Only ###
# Set system path
import sys
import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(ABS_PATH, "..", "..")
sys.path.append(ROOT_PATH)  # This is for finding all the modules
sys.path.append(ABS_PATH)
print(ABS_PATH)
from eoh import eoh
from eoh.utils.getParas import Paras
# from evol.utils.createReport import ReportCreator
# 

# Parameter initilization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "ael",    # ['ael','funsearch','reevo']
                problem = "bp_online", #['tsp_construct','bp_online','tsp_gls','fssp_gls']
                llm_api_endpoint = "XXX",
                llm_api_key = "XXX",   # set your key
                llm_model = "gpt-3.5-turbo-1106",
                ec_pop_size = 4,
                ec_n_pop = 2,
                exp_n_proc = 4,
                exp_debug_mode = False)
# AEL initilization
evolution = eoh.EVOL(paras)


# run AEL
evolution.run()

# Generate AEL Report
# RC = ReportCreator(paras)
# RC.generate_doc_report()




