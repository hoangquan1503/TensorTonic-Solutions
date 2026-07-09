import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    x = np.asarray(points)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    expand = np.ones((x.shape[0], 1))
    x = np.hstack((x, expand))
    transform = T @ x.T
    transform = transform[:3, :].T
    if transform.shape[0] == 1:
        transform = np.squeeze(transform)
    return transform
    
    