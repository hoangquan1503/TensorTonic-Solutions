import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.array(param, dtype=float)
    grad = np.array(grad, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1- beta2) * grad ** 2
    bias_corr_m = m / (1 - beta1**t)
    bias_corr_v = v / (1 - beta2**t)
    param = param - lr * bias_corr_m / (np.sqrt(bias_corr_v) + eps)
    return param, m, v