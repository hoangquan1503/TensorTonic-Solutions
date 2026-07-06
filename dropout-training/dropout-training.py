import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    state = np.random.default_rng(rng)
    prob = np.asarray(x, dtype=int)
    mask = state.random(prob.shape) < 1-p
    scale = 1.0 / (1-p)
    pattern = mask * scale
    x = x * pattern
    return x, pattern