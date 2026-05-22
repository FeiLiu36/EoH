"""Run EoH to evolve an Atari Pong heuristic.

Prerequisites
-------------
1. Install the framework (from repo root):
       pip install -e eoh/

2. Install ALE / gymnasium:
       pip install "gymnasium[atari]" ale-py
       pip install autorom && AutoROM --accept-license

3. Set your LLM credentials below (or via environment variables).

Usage
-----
    python runEoH.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import ALEPONG

if __name__ == "__main__":

    llm = LLMConfig(
        api_endpoint='api.deepseek.com',
        api_key='sk-xxx',
        model='deepseek-chat',
        timeout=150,
    )

    # 10 episodes × up to 5000 steps each; 120 s timeout per evaluation
    task = ALEPONG(n_episodes=10, max_steps=5000, timeout=120, n_processes=8)

    eoh = EoH(
        llm=llm,
        problem=task,
        pop_size=8,
        n_pop=100,
        operators=['e1', 'e2', 'm1', 'm2'],
        output_dir=os.path.dirname(__file__),
    )

    print(
        "Note: EoH minimises the objective = −mean_episode_reward.\n"
        "  positive objective → agent is losing  (e.g. +6 means losing by 6 pts)\n"
        "  negative objective → agent is winning  (e.g. −8 means winning by 8 pts)\n"
    )
    eoh.run()
