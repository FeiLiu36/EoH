import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import BPONLINE

if __name__ == "__main__":

    llm = LLMConfig(
        api_endpoint='xxx',
        api_key='xxx',
        model='xxx',
        timeout=150,
    )

    task = BPONLINE(capacity=100, timeout=40)

    eoh = EoH(
        llm=llm,
        problem=task,
        pop_size=10,
        n_pop=100,
        num_samplers=4,      # concurrent LLM-generation threads (I/O bound)
        num_evaluators=4,     # cap on concurrent eval subprocesses (CPU bound)
        operators=['e1', 'e2', 'm1', 'm2'],
        output_dir=os.path.dirname(__file__),
    )

    eoh.run()
