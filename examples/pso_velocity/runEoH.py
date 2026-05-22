import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import PSOVelocityUpdate

if __name__ == "__main__":
    llm = LLMConfig(
        api_endpoint='api.deepseek.com',
        api_key='sk-xxx',
        model='deepseek-chat',
        timeout=150,
    )

    task = PSOVelocityUpdate(
        pop_size=30,
        max_iterations=200,
        n_runs=3,
        w=0.729,
        c1=1.494,
        c2=1.494,
        v_max_ratio=0.2,
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
