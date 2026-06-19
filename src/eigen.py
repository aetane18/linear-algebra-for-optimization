import numpy as np


def eigen_decomposition(A):
    """
    Compute eigenvalues and eigenvectors of matrix A.

    Math:
    A v = lambda v

    Returns:
    eigenvalues, eigenvectors
    """
    A = np.array(A, dtype=float)

    eigenvalues, eigenvectors = np.linalg.eig(A)

    return eigenvalues, eigenvectors


def dominant_eigenvalue(A):
    """
    Return the dominant eigenvalue.

    Dominant eigenvalue means the eigenvalue with the largest absolute value.
    """
    eigenvalues, _ = eigen_decomposition(A)

    index = np.argmax(np.abs(eigenvalues))

    return eigenvalues[index]


def dominant_eigenvector(A):
    """
    Return the eigenvector corresponding to the dominant eigenvalue.
    """
    eigenvalues, eigenvectors = eigen_decomposition(A)

    index = np.argmax(np.abs(eigenvalues))

    return eigenvectors[:, index]


def power_iteration(A, num_iterations=1000, tolerance=1e-10):
    """
    Estimate the dominant eigenvalue and dominant eigenvector using power iteration.

    Power iteration idea:
    1. Start with a random vector.
    2. Repeatedly multiply by A.
    3. Normalize the vector.
    4. The vector approaches the dominant eigenvector.
    """
    A = np.array(A, dtype=float)

    rows, cols = A.shape

    if rows != cols:
        raise ValueError("Matrix must be square")

    # Start with a vector of ones
    b_k = np.ones(cols)

    for _ in range(num_iterations):
        # Multiply matrix by vector
        b_k_next = A @ b_k

        # Compute vector norm
        b_k_next_norm = np.linalg.norm(b_k_next)

        if b_k_next_norm == 0:
            raise ValueError("Power iteration failed because vector norm became zero")

        # Normalize vector
        b_k_next = b_k_next / b_k_next_norm

        # Stop if vector has stopped changing much
        if np.linalg.norm(b_k_next - b_k) < tolerance:
            b_k = b_k_next
            break

        b_k = b_k_next

    # Estimate eigenvalue using Rayleigh quotient
    eigenvalue = (b_k.T @ A @ b_k) / (b_k.T @ b_k)

    eigenvector = b_k

    return eigenvalue, eigenvector


def check_eigenpair(A, eigenvalue, eigenvector):
    """
    Check whether A v is approximately equal to lambda v.

    If A v ≈ lambda v, then the eigenpair is valid.
    """
    A = np.array(A, dtype=float)
    eigenvector = np.array(eigenvector, dtype=float)

    left_side = A @ eigenvector
    right_side = eigenvalue * eigenvector

    return np.allclose(left_side, right_side)


if __name__ == "__main__":
    A = np.array([
        [4, 1],
        [2, 3]
    ], dtype=float)

    eigenvalues, eigenvectors = eigen_decomposition(A)

    dom_value = dominant_eigenvalue(A)
    dom_vector = dominant_eigenvector(A)

    power_value, power_vector = power_iteration(A)

    print("Matrix A:")
    print(A)

    print("\nEigenvalues from NumPy:")
    print(eigenvalues)

    print("\nEigenvectors from NumPy:")
    print(eigenvectors)

    print("\nDominant eigenvalue from NumPy:")
    print(dom_value)

    print("\nDominant eigenvector from NumPy:")
    print(dom_vector)

    print("\nDominant eigenvalue from power iteration:")
    print(power_value)

    print("\nDominant eigenvector from power iteration:")
    print(power_vector)

    print("\nCheck NumPy dominant eigenpair:")
    print(check_eigenpair(A, dom_value, dom_vector))

    print("\nCheck power iteration eigenpair:")
    print(check_eigenpair(A, power_value, power_vector))