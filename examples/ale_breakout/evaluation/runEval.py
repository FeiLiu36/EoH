# -------
# Evaluation code for EoH on Atari Breakout (ALE)
# -------
# More results may refer to:
# Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang,
# Zhichao Lu, Qingfu Zhang.
# "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#  Using Large Language Model." ICML 2024, https://arxiv.org/abs/2401.02051.

import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from evaluation import Evaluation

N_EVAL_EPISODES = 20   # episodes per heuristic
MAX_STEPS       = 5000  # step cap per episode


def run(label: str, module_name: str, eva: Evaluation, results_file,
        visualize: bool = True):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)   # pick up any on-disk edits

    t0    = time.time()
    stats = eva.evaluate(mod.select_action)
    elapsed = time.time() - t0

    line = (
        f"{label:30s} | "
        f"mean: {stats['mean']:7.2f}  "
        f"std: {stats['std']:6.2f}  "
        f"best: {stats['best']:6.0f}  "
        f"worst: {stats['worst']:5.0f}  "
        f"time: {elapsed:.1f}s"
    )
    print(line)
    results_file.write(line + "\n")

    if visualize:
        slug = label.replace(' ', '_').replace('(', '').replace(')', '').strip('_')
        out_dir = os.path.dirname(os.path.abspath(results_file.name))
        eva.plot_scores(stats['scores'], label=label,
                        save_path=os.path.join(out_dir, f'scores_{slug}.png'))
        eva.render_episode(mod.select_action,
                           save_path=os.path.join(out_dir, f'gameplay_{slug}.gif'))


if __name__ == "__main__":
    eva = Evaluation(n_episodes=N_EVAL_EPISODES, max_steps=MAX_STEPS)

    with open(os.path.dirname(__file__) + "/results.txt", "w") as f:
        header = (
            f"Breakout evaluation — {N_EVAL_EPISODES} episodes, "
            f"max {MAX_STEPS} steps/episode\n"
            + "-" * 90
        )
        print(header)
        f.write(header + "\n")

        # Baseline heuristic (ball-tracking template)
        run("baseline (ball-tracking)", "heuristic", eva, f)

        # To benchmark your EoH-evolved heuristic, copy its select_action
        # function into heuristic.py and re-run this script.
