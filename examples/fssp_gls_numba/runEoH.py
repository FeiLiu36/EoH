import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import FSSPGLS

if __name__ == "__main__":
    llm = LLMConfig(
        api_endpoint='api.deepseek.com',
        api_key='sk-xxx',
        model='deepseek-chat',
        timeout=150,
    )

    task = FSSPGLS(n_inst_eva=3, time_max=30.0, iter_max=1000, timeout=120)

    eoh = EoH(
        llm=llm,
        problem=task,
        num_samplers=4,
        num_evaluators=4,
        pop_size=4,
        n_pop=20,
        operators=['e1', 'e2', 'm1', 'm2'],
        output_dir=os.path.dirname(__file__),
    )

    eoh.run()
