import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    l = len(A)
    w = len(A[0])
    transpose = np.full((w, l), 0)
    for i in range(len(transpose)):
        for j in range(len(transpose[i])):
            transpose[i][j] = A[j][i]
    return transpose
