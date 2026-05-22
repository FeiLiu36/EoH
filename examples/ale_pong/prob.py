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


class ALEPONG(BaseProblem):
    """EoH problem: evolve an action-selection heuristic for Atari Pong.

    The agent controls the right paddle (player) against the built-in CPU
    opponent (left paddle). Each episode runs until the game ends naturally
    (first side to 21 points) or max_steps is reached. Fitness is the
    negative average episode reward over n_episodes (lower = better, so
    higher game score = lower fitness value).

    Pong RAM layout (128-byte array, key positions):
        obs[49]  ball x position    (17–160, left→right)
        obs[54]  ball y position    (17–194, top→bottom)
        obs[51]  player paddle y    (right side, 34–194)
        obs[50]  CPU paddle y       (left side,  34–194)

    Actions:
        0 = NOOP  (stay still)
        2 = UP    (move paddle up)
        3 = DOWN  (move paddle down)

    Note: actions 4 (UP+FIRE) and 5 (DOWN+FIRE) are also accepted and mapped
    to UP and DOWN respectively (FIRE has no effect in Pong).
    """

    template_program = '''
def select_action(obs: np.ndarray) -> int:
    """Select a Pong action from the 128-byte RAM state.

    The player controls the right paddle. A point is scored each time
    the ball passes the opponent's paddle; a point is lost when the
    ball passes the player's own paddle.

    Args:
        obs: 128-element uint8 array (ALE RAM).
            obs[49]: ball x  (17-160, left to right)
            obs[54]: ball y  (17-194, larger = lower on screen)
            obs[51]: player (right) paddle y center (34-194)
            obs[50]: CPU (left) paddle y center     (34-194)
    Returns:
        action: 0 (NOOP), 2 (UP), or 3 (DOWN)
    """
    ball_y = obs[54]
    paddle_y = obs[51]
    if ball_y < paddle_y:
        return 2  # move UP
    elif ball_y > paddle_y:
        return 3  # move DOWN
    return 0  # NOOP
'''

    task_description = (
        "Design a heuristic action-selection policy for an Atari Pong agent "
        "controlling the right paddle. The agent receives a 128-byte RAM array "
        "each step. Key bytes: obs[49]=ball_x (17-160, left to right), "
        "obs[54]=ball_y (17-194, larger = lower on screen), "
        "obs[51]=player_paddle_y (34-194), obs[50]=CPU_paddle_y (34-194). "
        "Valid actions: 0=NOOP, 2=UP, 3=DOWN. "
        "Maximise the average per-episode score."
    )

    # actions 4/5 (UP+FIRE / DOWN+FIRE) map to UP/DOWN; FIRE has no effect in Pong
    _ACTION_MAP = {0: 0, 1: 0, 2: 2, 3: 3, 4: 2, 5: 3}

    def __init__(
        self,
        n_episodes: int = 10,
        max_steps: int = 2000,
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
        env = gym.make('ALE/Pong-v5', obs_type='ram', render_mode=None)
        scores = []
        try:
            for ep in range(self.n_episodes):
                obs, _ = env.reset(seed=ep)
                episode_reward = 0.0
                for _ in range(self.max_steps):
                    try:
                        action = self._ACTION_MAP.get(int(callable_func(obs)), 0)
                    except Exception:
                        return None   # invalid program — signal EoH to discard
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    if terminated or truncated:
                        break
                scores.append(episode_reward)
        finally:
            env.close()

        # EoH minimises; negate so that higher mean score → lower fitness
        return float(-np.mean(scores))
