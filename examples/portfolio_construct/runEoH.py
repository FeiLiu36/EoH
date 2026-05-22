import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import PortfolioConstruct

if __name__ == "__main__":
    llm = LLMConfig(
        api_endpoint='api.deepseek.com',
        api_key='sk-xxx',
        model='deepseek-chat',
        timeout=150,
    )

    task = PortfolioConstruct(
        n_assets=20, n_select=5, n_periods=252,
        n_instance=5, timeout=40, n_processes=4,
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
