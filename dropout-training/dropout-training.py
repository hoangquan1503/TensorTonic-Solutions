import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    state = np.random.default_rng(rng)
    prob = np.asarray(x, dtype=float)
    prob = state.random(prob.shape)
    prob = np.array([np.where(pr < 1 - p, pr, 0) for pr in prob])
    pattern = np.array([np.where(pr != 0, 1 / (1 - p), pr) for pr in prob])
    x = x * pattern
    return x, pattern