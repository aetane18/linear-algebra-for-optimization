import numpy as np


def center_data(X):
    """
    Center the data by subtracting the mean of each column.

    PCA should be done on centered data.
    """
    X = np.array(X, dtype=float)

    column_means = np.mean(X, axis=0)

    X_centered = X - column_means

    return X_centered, column_means


def covariance_matrix(X_centered):
    """
    Compute the covariance matrix.

    Covariance matrix shows how features vary together.
    """
    X_centered = np.array(X_centered, dtype=float)

    return np.cov(X_centered, rowvar=False)


def compute_eigen(cov_matrix):
    """
    Compute eigenvalues and eigenvectors of the covariance matrix.
    """
    cov_matrix = np.array(cov_matrix, dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    return eigenvalues, eigenvectors


def sort_eigenpairs(eigenvalues, eigenvectors):
    """
    Sort eigenvalues and eigenvectors from largest eigenvalue to smallest.

    Larger eigenvalue = more variance explained.
    """
    sorted_indices = np.argsort(eigenvalues)[::-1]

    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors


def explained_variance_ratio(eigenvalues):
    """
    Compute how much variance each principal component explains.
    """
    eigenvalues = np.array(eigenvalues, dtype=float)

    total_variance = np.sum(eigenvalues)

    return eigenvalues / total_variance


def project_data(X_centered, components):
    """
    Project centered data onto selected principal components.

    Formula:
    Z = X_centered @ components
    """
    X_centered = np.array(X_centered, dtype=float)
    components = np.array(components, dtype=float)

    return X_centered @ components


def pca(X, n_components=2):
    """
    Run PCA from scratch using NumPy.

    Steps:
    1. Center data
    2. Compute covariance matrix
    3. Compute eigenvalues/eigenvectors
    4. Sort eigenpairs
    5. Select top principal components
    6. Project data
    """
    X_centered, column_means = center_data(X)

    cov_matrix = covariance_matrix(X_centered)

    eigenvalues, eigenvectors = compute_eigen(cov_matrix)

    sorted_eigenvalues, sorted_eigenvectors = sort_eigenpairs(
        eigenvalues,
        eigenvectors
    )

    selected_components = sorted_eigenvectors[:, :n_components]

    X_projected = project_data(X_centered, selected_components)

    variance_ratio = explained_variance_ratio(sorted_eigenvalues)

    return {
        "X_centered": X_centered,
        "column_means": column_means,
        "covariance_matrix": cov_matrix,
        "eigenvalues": sorted_eigenvalues,
        "eigenvectors": sorted_eigenvectors,
        "components": selected_components,
        "X_projected": X_projected,
        "explained_variance_ratio": variance_ratio
    }


if __name__ == "__main__":
    X = np.array([
        [2.5, 2.4, 1.2],
        [0.5, 0.7, 0.3],
        [2.2, 2.9, 1.5],
        [1.9, 2.2, 1.1],
        [3.1, 3.0, 1.7],
        [2.3, 2.7, 1.4],
        [2.0, 1.6, 0.9],
        [1.0, 1.1, 0.5],
        [1.5, 1.6, 0.8],
        [1.1, 0.9, 0.4]
    ])

    result = pca(X, n_components=2)

    print("Original data:")
    print(X)

    print("\nColumn means:")
    print(result["column_means"])

    print("\nCentered data:")
    print(result["X_centered"])

    print("\nCovariance matrix:")
    print(result["covariance_matrix"])

    print("\nEigenvalues:")
    print(result["eigenvalues"])

    print("\nEigenvectors:")
    print(result["eigenvectors"])

    print("\nSelected principal components:")
    print(result["components"])

    print("\nProjected data:")
    print(result["X_projected"])

    print("\nExplained variance ratio:")
    print(result["explained_variance_ratio"])