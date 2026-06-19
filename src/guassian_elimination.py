import numpy as np


def forward_elimination(A, b):
    """
    Convert the system Ax = b into upper triangular form.

    Example:
    [a b c]        [a b c]
    [d e f]   ->   [0 e f]
    [g h i]        [0 0 i]

    This makes the system easier to solve using back substitution.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)

    for pivot_row in range(n):
        # Partial pivoting:
        # Find the row with the largest absolute value in the pivot column.
        max_row = pivot_row

        for row in range(pivot_row + 1, n):
            if abs(A[row][pivot_row]) > abs(A[max_row][pivot_row]):
                max_row = row

        # Swap current pivot row with max_row if needed.
        if max_row != pivot_row:
            A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
            b[[pivot_row, max_row]] = b[[max_row, pivot_row]]

        # Check for zero pivot.
        if A[pivot_row][pivot_row] == 0:
            raise ValueError("Matrix is singular or has no unique solution")

        # Eliminate values below the pivot.
        for row in range(pivot_row + 1, n):
            factor = A[row][pivot_row] / A[pivot_row][pivot_row]

            A[row] = A[row] - factor * A[pivot_row]
            b[row] = b[row] - factor * b[pivot_row]

    return A, b


def back_substitution(U, b):
    """
    Solve an upper triangular system Ux = b.

    Example:
    2x + y - z = 8
         3y + z = 9
              5z = -5

    Start from the bottom equation and move upward.
    """
    U = np.array(U, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)
    x = np.zeros(n)

    for row in range(n - 1, -1, -1):
        sum_known_terms = 0

        for col in range(row + 1, n):
            sum_known_terms += U[row][col] * x[col]

        x[row] = (b[row] - sum_known_terms) / U[row][row]

    return x


def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination.

    Steps:
    1. Forward elimination
    2. Back substitution
    """
    U, new_b = forward_elimination(A, b)
    x = back_substitution(U, new_b)

    return x


if __name__ == "__main__":
    A = [
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ]

    b = [8, -11, -3]

    solution = gaussian_elimination(A, b)

    print("Matrix A:")
    print(np.array(A))

    print("\nVector b:")
    print(np.array(b))

    print("\nSolution [x, y, z]:")
    print(solution)