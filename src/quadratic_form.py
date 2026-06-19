import numpy as np
import matplotlib.pyplot as plt


def is_symmetric(A):
    """
    Check whether matrix A is symmetric.

    A matrix is symmetric if:
    A = A.T
    """
    A = np.array(A, dtype=float)

    return np.allclose(A, A.T)


def quadratic_form(A, x):
    """
    Compute the quadratic form:

    f(x) = x^T A x

    A must be a square matrix.
    x must be a vector with compatible dimension.
    """
    A = np.array(A, dtype=float)
    x = np.array(x, dtype=float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")

    if A.shape[1] != x.shape[0]:
        raise ValueError("Matrix and vector dimensions do not match")

    return x.T @ A @ x


def classify_quadratic_form(A):
    """
    Classify a quadratic form using eigenvalues.

    If all eigenvalues are positive:
        positive definite

    If all eigenvalues are negative:
        negative definite

    If eigenvalues include both positive and negative:
        indefinite

    If eigenvalues are nonnegative but at least one is zero:
        positive semidefinite

    If eigenvalues are nonpositive but at least one is zero:
        negative semidefinite
    """
    A = np.array(A, dtype=float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")

    if not is_symmetric(A):
        raise ValueError("Matrix A must be symmetric to classify quadratic form")

    eigenvalues = np.linalg.eigvalsh(A)

    if np.all(eigenvalues > 0):
        return "positive definite"

    elif np.all(eigenvalues < 0):
        return "negative definite"

    elif np.all(eigenvalues >= 0) and np.any(eigenvalues == 0):
        return "positive semidefinite"

    elif np.all(eigenvalues <= 0) and np.any(eigenvalues == 0):
        return "negative semidefinite"

    else:
        return "indefinite"


def generate_surface_data(A, x_min=-5, x_max=5, num_points=100):
    """
    Generate X, Y, Z values for plotting a 2D quadratic form.

    This only works for 2x2 matrices.

    For every point [x1, x2], compute:
    z = [x1, x2]^T A [x1, x2]
    """
    A = np.array(A, dtype=float)

    if A.shape != (2, 2):
        raise ValueError("Surface plotting only works for 2x2 matrices")

    x1_values = np.linspace(x_min, x_max, num_points)
    x2_values = np.linspace(x_min, x_max, num_points)

    X1, X2 = np.meshgrid(x1_values, x2_values)

    Z = np.zeros_like(X1)

    for i in range(num_points):
        for j in range(num_points):
            x = np.array([X1[i, j], X2[i, j]])
            Z[i, j] = quadratic_form(A, x)

    return X1, X2, Z


def plot_quadratic_form(A, title="Quadratic Form"):
    """
    Plot the quadratic form:

    f(x) = x^T A x

    This only works for 2x2 matrices.
    """
    A = np.array(A, dtype=float)

    X1, X2, Z = generate_surface_data(A)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X1, X2, Z, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x) = x^T A x")

    plt.show()


if __name__ == "__main__":
    # Positive definite: bowl shape
    A_positive = np.array([
        [2, 0],
        [0, 3]
    ])

    # Negative definite: upside-down bowl
    A_negative = np.array([
        [-2, 0],
        [0, -3]
    ])

    # Indefinite: saddle shape
    A_indefinite = np.array([
        [1, 0],
        [0, -1]
    ])

    x = np.array([1, 2])

    print("Positive definite matrix:")
    print(A_positive)
    print("Quadratic form value at x = [1, 2]:")
    print(quadratic_form(A_positive, x))
    print("Classification:")
    print(classify_quadratic_form(A_positive))

    print("\nNegative definite matrix:")
    print(A_negative)
    print("Quadratic form value at x = [1, 2]:")
    print(quadratic_form(A_negative, x))
    print("Classification:")
    print(classify_quadratic_form(A_negative))

    print("\nIndefinite matrix:")
    print(A_indefinite)
    print("Quadratic form value at x = [1, 2]:")
    print(quadratic_form(A_indefinite, x))
    print("Classification:")
    print(classify_quadratic_form(A_indefinite))

    # Uncomment one at a time to see plots
    # plot_quadratic_form(A_positive, "Positive Definite: Bowl Shape")
    # plot_quadratic_form(A_negative, "Negative Definite: Upside-Down Bowl")
    # plot_quadratic_form(A_indefinite, "Indefinite: Saddle Shape")