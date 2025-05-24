import numpy as np
from constants import DESIRED_STATE_SIZE

def sanitize_full_state(full_state: np.ndarray, target_size: int = DESIRED_STATE_SIZE) -> np.ndarray:
    """
    Pulisce e ridimensiona full_state per i modelli AI:
    - Rimuove NaN, infiniti.
    - Applica padding o troncamento a target_size.
    - Clipping finale per valori estremi.
    """
    full_state = np.nan_to_num(full_state, nan=0.0, posinf=0.0, neginf=0.0)

    if full_state.shape[0] > target_size:
        full_state = full_state[:target_size]
    elif full_state.shape[0] < target_size:
        padding = np.zeros(target_size - full_state.shape[0])
        full_state = np.concatenate([full_state, padding])
    
    return np.clip(full_state, -1.0, 1.0)
