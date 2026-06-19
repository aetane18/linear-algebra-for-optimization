import numpy as np


def add_intercept_column(x):
    """
    Add a column of 1s to the input feature vector.

    For simple linear regression:
    y = beta_0 + beta_1*x

    The column of 1s represents beta_0, the intercept.
    """
    x = np.array(x, dtype=float)

    return np.column_stack((np.ones(len(x)), x))


def least_squares_normal_equation(X, y):
    """
    Compute least squares coefficients using the normal equation.

    Formula:
    beta = (X^T X)^(-1) X^T y
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    return beta


def least_squares_stable(X, y):
    """
    Compute least squares coefficients using NumPy's stable solver.

    This is usually better in real-world work than manually using inverse.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    beta, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)

    return beta


def predict(X, beta):
    """
    Make predictions using:

    y_pred = X beta
    """
    X = np.array(X, dtype=float)
    beta = np.array(beta, dtype=float)

    return X @ beta


def residuals(y_true, y_pred):
    """
    Compute residuals.

    residual = actual value - predicted value
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return y_true - y_pred


def mean_squared_error(y_true, y_pred):
    """
    Compute mean squared error.

    MSE = average of squared residuals
    """
    errors = residuals(y_true, y_pred)

    return np.mean(errors ** 2)


if __name__ == "__main__":
    # Simple dataset
    x = np.array([1, 2, 3, 4, 5])

    y = np.array([2, 4, 5, 4, 5])

    # Build design matrix
    X = add_intercept_column(x)

    # Fit using normal equation
    beta_normal = least_squares_normal_equation(X, y)

    # Fit using NumPy stable solver
    beta_stable = least_squares_stable(X, y)

    # Predictions
    y_pred = predict(X, beta_normal)

    # Error
    mse = mean_squared_error(y, y_pred)

    print("x values:")
    print(x)

    print("\ny values:")
    print(y)

    print("\nDesign matrix X:")
    print(X)

    print("\nBeta from normal equation:")
    print(beta_normal)

    print("\nBeta from NumPy stable least squares:")
    print(beta_stable)

    print("\nPredicted y values:")
    print(y_pred)

    print("\nResiduals:")
    print(residuals(y, y_pred))

    print("\nMean Squared Error:")
    print(mse)

    print("\nRegression equation:")
    print(f"y = {beta_normal[0]:.4f} + {beta_normal[1]:.4f}x")