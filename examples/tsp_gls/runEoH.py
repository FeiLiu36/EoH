import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import TSPGLS

if __name__ == "__main__":
    llm = LLMConfig(
        api_endpoint='xxx',
        api_key='xxx',
        model='xxx',
        timeout=150,
    )

    task = TSPGLS(n_inst_eva=3, time_limit=10.0, ite_max=1000,
                  perturbation_moves=1, timeout=120)

    eoh = EoH(
        llm=llm,
        problem=task,
        num_samplers=4,
        num_evaluators=4,
        pop_size=10,
        n_pop=100,
        operators=['e1', 'e2', 'm1', 'm2'],
        output_dir=os.path.dirname(__file__),
    )

    eoh.run()
