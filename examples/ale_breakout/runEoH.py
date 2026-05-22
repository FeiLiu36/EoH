"""Run EoH to evolve an Atari Breakout heuristic.

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

Fitness interpretation
----------------------
EoH minimises the objective  =  −mean_episode_score.
  large negative objective  →  agent scores many points  (good)
  near-zero objective       →  agent barely scores
  e.g. objective = −120     →  agent averages 120 pts/episode
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import ALEBREAKOUT

if __name__ == "__main__":

    llm = LLMConfig(
        api_endpoint='api.deepseek.com',
        api_key='sk-xxx',
        model='deepseek-chat',
        timeout=150,
    )

    # 10 episodes × up to 5000 steps each; 120 s timeout per evaluation
    task = ALEBREAKOUT(n_episodes=10, max_steps=5000, timeout=120, n_processes=8)

    eoh = EoH(
        llm=llm,
        problem=task,
        pop_size=8,
        n_pop=50,
        operators=['e1', 'e2', 'm1', 'm2'],
        output_dir=os.path.dirname(__file__),
    )

    eoh.run()
