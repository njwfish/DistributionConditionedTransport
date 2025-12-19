import numpy as np

def get_train_predictor_bool(batch: dict):

    s_val = batch.get("source_idx")
    t_val = batch.get("target_idx")
    
    if s_val is None or t_val is None:
        return False
    
    # TODO: check whether this is unneccessarily convoluted.
    if np.isclose(t_val - s_val, 1):
        return True
    else:
        return False
