import numpy as np


def shape(A):
    """
    Return the shape of matrix A.
    """
    A = np.array(A)
    return A.shape


def matrix_add(A, B):
    """
    Add two matrices using NumPy.
    """
    A = np.array(A)
    B = np.array(B)

    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape")

    return A + B


def matrix_subtract(A, B):
    """
    Subtract matrix B from matrix A using NumPy.
    """
    A = np.array(A)
    B = np.array(B)

    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape")

    return A - B


def scalar_multiply(c, A):
    """
    Multiply matrix A by scalar c using NumPy.
    """
    A = np.array(A)
    return c * A


def transpose(A):
    """
    Return transpose of matrix A using NumPy.
    """
    A = np.array(A)
    return A.T


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

    print("Matrix A:")
    print(A)

    print("\nMatrix B:")
    print(B)

    print("\nShape of A:")
    print(shape(A))

    print("\nA + B:")
    print(matrix_add(A, B))

    print("\nA - B:")
    print(matrix_subtract(A, B))

    print("\n3A:")
    print(scalar_multiply(3, A))

    print("\nTranspose of A:")
    print(transpose(A))