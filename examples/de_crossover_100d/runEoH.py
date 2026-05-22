import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import DECrossover100D

if __name__ == "__main__":
    llm = LLMConfig(
        api_endpoint='api.deepseek.com',
        api_key='sk-xxx',
        model='deepseek-chat',
        timeout=150,
    )

    task = DECrossover100D(
        pop_size=20,       # DE population size (training)
        max_evals=20000,   # function-evaluation budget per trial
        n_runs=3,          # independent DE runs per benchmark instance
        F=0.8,             # DE scale factor
        CR=0.9,            # base crossover rate passed to the evolved operator
        timeout=60,        # per-candidate evaluation timeout (seconds)
        n_processes=4,     # parallel candidate evaluations
    )

    eoh = EoH(
        llm=llm,
        problem=task,
        pop_size=4,
        n_pop=20,
        operators=['e1', 'e2', 'm1', 'm2'],
        output_dir=os.path.dirname(__file__),
    )

    eoh.run()
