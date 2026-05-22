# Placeholder heuristic — replace this with the best heuristic produced by EoH.
# Copy the evolved select_action function from the EoH results here, then run
# runEval.py to benchmark it against a longer set of evaluation episodes.
import numpy as np
def select_action(obs: np.ndarray) -> int:
    ball_x = obs[49]
    ball_y = obs[54]
    paddle_y = obs[51]
    
    # Estimate vertical velocity from direction
    if ball_y > 100:
        vel_y = 4
    else:
        vel_y = -4
    
    # Predict number of frames until ball reaches right side
    dx = 160 - ball_x
    frames = max(1, dx // 5)
    
    # Predict y after bounces
    pred_y = ball_y + vel_y * frames
    if pred_y < 17:
        pred_y = 17 + (17 - pred_y)
    if pred_y > 194:
        pred_y = 194 - (pred_y - 194)
    pred_y = max(17, min(194, pred_y))
    
    # Dynamic offset based on horizontal distance
    if dx > 100:
        offset = 10
    elif dx > 50:
        offset = 7
    else:
        offset = 4
    
    if pred_y < paddle_y - offset:
        return 2
    elif pred_y > paddle_y + offset:
        return 3
    return 0
    
        

