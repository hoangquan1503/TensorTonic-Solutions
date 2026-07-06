import numpy as np
import math 
def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len)
    pos = pos.reshape(-1,1)
    
    num_pairs = math.ceil(d_model / 2)
    
    freq = np.array([(1.0) / base**(2 * i / d_model) for i in range(num_pairs)])

    angle = pos * freq

    pe = np.full((seq_len, d_model), 0.0)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, :d_model // 2])

    return pe
                               
    