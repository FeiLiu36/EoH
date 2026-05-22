# Placeholder heuristic — replace this with the best heuristic produced by EoH.
# Copy the evolved select_action function from the EoH results here, then run
# runEval.py to benchmark it against a longer set of evaluation episodes.
import numpy as np

def select_action(obs: np.ndarray) -> int:
    if obs[101] == 0:
        return 1

    ball_x = obs[99]
    ball_y = obs[101]
    paddle_x = obs[72]

    # Precomputed aim points for each ball position (tuned for common trajectories)
    # Mapping: (ball_x, ball_y) -> desired paddle_x offset from ball_x
    # Simple heuristic: when ball is low, aim slightly left of ball; when high, aim right
    if ball_y > 150:
        aim_offset = -8
    elif ball_y > 100:
        aim_offset = -3
    elif ball_y > 50:
        aim_offset = 5
    else:
        aim_offset = 12

    target_x = ball_x + aim_offset
    target_x = max(55, min(191, target_x))

    if target_x < paddle_x - 4:
        return 3
    elif target_x > paddle_x + 4:
        return 2
    return 0
    
