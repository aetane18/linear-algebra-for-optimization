import numpy as np


def is_square(A):
    """
    Check if matrix A is square.

    A square matrix has the same number of rows and columns.
    Example: 3x3, 2x2, 4x4
    """
    A = np.array(A, dtype=float)

    rows, cols = A.shape

    return rows == cols


def is_symmetric(A):
    """
    Check if matrix A is symmetric.

    A matrix is symmetric if:
    A = A.T
    """
    A = np.array(A, dtype=float)

    if not is_square(A):
        return False

    return np.allclose(A, A.T)


def eigenvalues_of_matrix(A):
    """
    Return eigenvalues of matrix A.
    """
    A = np.array(A, dtype=float)

    return np.linalg.eigvals(A)


def is_positive_definite_eigen(A):
    """
    Check positive definiteness using eigenvalues.

    A symmetric matrix is positive definite if all eigenvalues are positive.
    """
    A = np.array(A, dtype=float)

    if not is_symmetric(A):
        return False

    eigenvalues = eigenvalues_of_matrix(A)

    return np.all(eigenvalues > 0)


def is_positive_definite_cholesky(A):
    """
    Check positive definiteness using Cholesky decomposition.

    If Cholesky decomposition works, the matrix is positive definite.
    If it fails, the matrix is not positive definite.
    """
    A = np.array(A, dtype=float)

    if not is_symmetric(A):
        return False

    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def quadratic_form(A, x):
    """
    Compute the quadratic form:

    x^T A x
    """
    A = np.array(A, dtype=float)
    x = np.array(x, dtype=float)

    return x.T @ A @ x


def positive_definite_report(A):
    """
    Return a simple report about whether matrix A is positive definite.
    """
    A = np.array(A, dtype=float)

    return {
        "is_square": is_square(A),
        "is_symmetric": is_symmetric(A),
        "eigenvalues": eigenvalues_of_matrix(A),
        "positive_definite_by_eigenvalues": is_positive_definite_eigen(A),
        "positive_definite_by_cholesky": is_positive_definite_cholesky(A)
    }


if __name__ == "__main__":
    # Positive definite example
    A_pd = np.array([
        [2, 0],
        [0, 3]
    ])

    # Not positive definite example
    A_not_pd = np.array([
        [1, 2],
        [2, 1]
    ])

    # Non-symmetric example
    A_non_symmetric = np.array([
        [1, 2],
        [3, 4]
    ])

    x = np.array([1, 2])

    print("Positive Definite Matrix A_pd:")
    print(A_pd)

    print("\nReport:")
    print(positive_definite_report(A_pd))

    print("\nQuadratic form x^T A x:")
    print(quadratic_form(A_pd, x))

    print("\nNot Positive Definite Matrix A_not_pd:")
    print(A_not_pd)

    print("\nReport:")
    print(positive_definite_report(A_not_pd))

    print("\nNon-Symmetric Matrix A_non_symmetric:")
    print(A_non_symmetric)

    print("\nReport:")
    print(positive_definite_report(A_non_symmetric))