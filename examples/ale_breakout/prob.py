# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import BaseProblem

# Requirements: pip install "gymnasium[atari]" ale-py
# ROMs are bundled with ale-py >= 0.9 via AutoROM:
#   pip install autorom && AutoROM --accept-license


class ALEBREAKOUT(BaseProblem):
    """EoH problem: evolve an action-selection heuristic for Atari Breakout.

    The agent controls a paddle at the bottom of the screen, bouncing a ball
    to break bricks arranged in rows at the top. Each episode gives 5 lives;
    a life is lost when the ball passes below the paddle. The episode ends
    when all lives are lost or all bricks are cleared.

    Fitness is the negative mean episode score over n_episodes (lower = better,
    so higher brick-breaking score → lower fitness value).

    Breakout RAM layout (empirically verified, 128-byte array):
        obs[99]   ball x position   (0–160, left→right)
        obs[101]  ball y position   (0–210, top→bottom; 0 = ball not in play)
        obs[72]   paddle x position (55–191, left→right)

    Actions (4-action minimal set):
        0 = NOOP
        1 = FIRE  (launch ball when obs[101] == 0; no effect while ball is live)
        2 = RIGHT (move paddle right)
        3 = LEFT  (move paddle left)

    Score: accumulated brick-breaking points per episode.
        Brick rows (top→bottom): red 7, orange 7, yellow 4, green 4, aqua 1, blue 1 pts.
        Maximum possible score: 864 (clear all bricks with one ball + bonus ball).
    """

    template_program = '''
def select_action(obs: np.ndarray) -> int:
    """Select a Breakout action from the 128-byte RAM state.

    The agent must bounce the ball off the paddle to break bricks above.
    When the ball is not in play (obs[101] == 0), FIRE launches it.

    Args:
        obs: 128-element uint8 array (ALE RAM).
            obs[99]:  ball x  (0-160, left to right)
            obs[101]: ball y  (0-210, larger = lower; 0 = not in play)
            obs[72]:  paddle x center (55-191, left to right)
    Returns:
        action: 0 (NOOP), 1 (FIRE), 2 (RIGHT), or 3 (LEFT)
    """
    if obs[101] == 0:
        return 1  # FIRE to launch the ball

    ball_x   = obs[99]
    paddle_x = obs[72]

    if ball_x < paddle_x:
        return 3  # LEFT
    elif ball_x > paddle_x:
        return 2  # RIGHT
    return 0  # NOOP
'''

    task_description = (
        "Design a heuristic action-selection policy for an Atari Breakout agent. "
        "The agent controls a paddle to bounce a ball upward and break bricks. "
        "The agent receives a 128-byte RAM array each step. Key bytes: "
        "obs[99]=ball_x (0-160, left to right), "
        "obs[101]=ball_y (0-210, larger = lower; 0 means ball not in play), "
        "obs[72]=paddle_x (55-191, left to right). "
        "Valid actions: 0=NOOP, 1=FIRE (launch ball when ball_y=0), "
        "2=RIGHT, 3=LEFT. "
        "Maximise the total score per episode (points per brick row: "
        "red=7, orange=7, yellow=4, green=4, aqua=1, blue=1)."
    )

    # Actions 0–3 are the complete valid set; anything outside → NOOP
    _ACTION_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

    def __init__(
        self,
        n_episodes: int = 10,
        max_steps: int = 5000,
        timeout: int = 120,
        n_processes: int = 1,
    ):
        super().__init__(timeout=timeout, n_processes=n_processes)
        # Fail fast: verify the ALE environment is available before EoH starts.
        try:
            import ale_py          # noqa: F401  registers ALE envs with gymnasium
            import gymnasium as gym  # noqa: F401
        except ImportError:
            raise ImportError(
                'gymnasium and ale-py are required. Install with: '
                'pip install "gymnasium[atari]" ale-py && '
                'pip install autorom && AutoROM --accept-license'
            )
        self.n_episodes = n_episodes
        self.max_steps = max_steps

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        import ale_py          # noqa: F401  registers ALE envs with gymnasium
        import gymnasium as gym

        # Mirror the loop structure of evaluation/evaluation.py so that the
        # training fitness and the offline benchmark are computed identically.
        # One env is created per call and reset between episodes; fixed seeds
        # (0 … n_episodes-1) guarantee a fair comparison across programs.
        env = gym.make('ALE/Breakout-v5', obs_type='ram', render_mode=None)
        scores = []
        try:
            for ep in range(self.n_episodes):
                obs, _ = env.reset(seed=ep)
                episode_score = 0.0
                for _ in range(self.max_steps):
                    try:
                        action = self._ACTION_MAP.get(int(callable_func(obs)), 0)
                    except Exception:
                        return None   # invalid program — signal EoH to discard
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_score += reward
                    if terminated or truncated:
                        break
                scores.append(episode_score)
        finally:
            env.close()

        # EoH minimises; negate so that higher mean score → lower fitness
        return float(-np.mean(scores))
