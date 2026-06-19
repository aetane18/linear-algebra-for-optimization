import numpy as np


def dot_product(v, w):
    """
    Compute the dot product of two vectors.

    Math:
    v · w = v1*w1 + v2*w2 + ... + vn*wn
    """
    v = np.array(v)
    w = np.array(w)

    if v.shape != w.shape:
        raise ValueError("Vectors must have the same shape")

    return np.dot(v, w)


def matrix_multiply(A, B):
    """
    Multiply two matrices A and B using NumPy.

    Rule:
    If A is m x n and B is n x p,
    then AB is m x p.
    """
    A = np.array(A)
    B = np.array(B)

    if A.shape[1] != B.shape[0]:
        raise ValueError("Number of columns in A must equal number of rows in B")

    return A @ B


def identity_matrix(n):
    """
    Create an n x n identity matrix.

    The identity matrix has 1s on the diagonal and 0s everywhere else.
    """
    return np.eye(n)


if __name__ == "__main__":
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    B = np.array([
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]
    ])

    v = np.array([1, 2, 3])
    w = np.array([9, 6, 3])

    print("Matrix A:")
    print(A)

    print("\nMatrix B:")
    print(B)

    print("\nDot product of v and w:")
    print(dot_product(v, w))

    print("\nA @ B:")
    print(matrix_multiply(A, B))

    print("\n3x3 Identity Matrix:")
    print(identity_matrix(3))