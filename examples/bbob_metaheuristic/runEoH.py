import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import BbobMetaheuristic

if __name__ == "__main__":
    llm = LLMConfig(
        api_endpoint='api.deepseek.com',
        api_key='sk-xxx',
        model='deepseek-chat',
        timeout=150,
    )

    task = BbobMetaheuristic(
        dim=10,
        budget=1000,
        n_runs=3,
        timeout=60,
        n_processes=4,
    )

    eoh = EoH(
        llm=llm,
        problem=task,
        pop_size=4,
        n_pop=10,
        operators=['e1', 'e2', 'm1', 'm2'],
        output_dir=os.path.dirname(__file__),
    )

    eoh.run()
