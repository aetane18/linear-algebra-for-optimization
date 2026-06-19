from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_positive_definite(matrix):
    """
    Positive definite check using eigenvalues.

    A matrix is positive definite if all eigenvalues are greater than 0.
    """
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0)


def main():
    # ---------------------------------------------------------
    # 1. Load California Housing Dataset
    # ---------------------------------------------------------
    housing = fetch_california_housing(as_frame=True)

    X = housing.data
    y = housing.target

    print("\nDataset Loaded Successfully")
    print("Feature Matrix Shape:", X.shape)
    print("Target Vector Shape:", y.shape)

    print("\nFirst 5 Rows of Features:")
    print(X.head())

    print("\nTarget Name:")
    print("MedHouseVal")

    # ---------------------------------------------------------
    # 2. Train/Test Split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------------------
    # 3. Standardize Features
    # ---------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # 4. Linear Regression using scikit-learn
    # ---------------------------------------------------------
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    linear_predictions = linear_model.predict(X_test_scaled)

    linear_mae = mean_absolute_error(y_test, linear_predictions)
    linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
    linear_r2 = r2_score(y_test, linear_predictions)

    print("\n==============================")
    print("Linear Regression Results")
    print("==============================")
    print("MAE:", linear_mae)
    print("RMSE:", linear_rmse)
    print("R2 Score:", linear_r2)

    # ---------------------------------------------------------
    # 5. Ridge Regression using scikit-learn
    # ---------------------------------------------------------
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    ridge_predictions = ridge_model.predict(X_test_scaled)

    ridge_mae = mean_absolute_error(y_test, ridge_predictions)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))
    ridge_r2 = r2_score(y_test, ridge_predictions)

    print("\n==============================")
    print("Ridge Regression Results")
    print("==============================")
    print("MAE:", ridge_mae)
    print("RMSE:", ridge_rmse)
    print("R2 Score:", ridge_r2)

    # ---------------------------------------------------------
    # 6. Linear Algebra Behind Regression
    # ---------------------------------------------------------
    XTX = X_train_scaled.T @ X_train_scaled
    XTy = X_train_scaled.T @ y_train.to_numpy()

    print("\n==============================")
    print("Matrix Operation Results")
    print("==============================")
    print("X.T @ X Shape:", XTX.shape)
    print("X.T @ y Shape:", XTy.shape)

    print("\nIs X.T @ X Positive Definite?")
    print(check_positive_definite(XTX))

    # ---------------------------------------------------------
    # 7. Covariance Matrix
    # ---------------------------------------------------------
    covariance_matrix = np.cov(X_train_scaled, rowvar=False)

    print("\n==============================")
    print("Covariance Matrix")
    print("==============================")
    print("Covariance Matrix Shape:", covariance_matrix.shape)

    print("\nIs Covariance Matrix Positive Definite?")
    print(check_positive_definite(covariance_matrix))

    # ---------------------------------------------------------
    # 8. Eigenvalues and Eigenvectors
    # ---------------------------------------------------------
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    explained_variance_from_eigenvalues = eigenvalues / np.sum(eigenvalues)

    print("\n==============================")
    print("Eigenvalue Analysis")
    print("==============================")
    print("Eigenvalues:")
    print(eigenvalues)

    print("\nExplained Variance Ratio from Eigenvalues:")
    print(explained_variance_from_eigenvalues)

    # ---------------------------------------------------------
    # 9. PCA using scikit-learn
    # ---------------------------------------------------------
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)

    print("\n==============================")
    print("PCA Results")
    print("==============================")
    print("Original Feature Count:", X_train_scaled.shape[1])
    print("Reduced Feature Count:", X_train_pca.shape[1])
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    print("Total Variance Explained:", np.sum(pca.explained_variance_ratio_))

    # ---------------------------------------------------------
    # 10. Feature Coefficients
    # ---------------------------------------------------------
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": linear_model.coef_
    })

    coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()

    coefficients = coefficients.sort_values(
        by="Absolute_Coefficient",
        ascending=False
    )

    print("\n==============================")
    print("Linear Regression Feature Coefficients")
    print("==============================")
    print(coefficients)

    coefficients.to_csv("linear_regression_coefficients.csv", index=False)

    # ---------------------------------------------------------
    # 11. Save Actual vs Predicted Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, linear_predictions, alpha=0.4)
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title("Actual vs Predicted Housing Prices")
    plt.grid(True)
    plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------
    # 12. Save PCA Projection Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=y_train,
        alpha=0.4
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection of California Housing Data")
    plt.colorbar(scatter, label="Median House Value")
    plt.grid(True)
    plt.savefig("pca_projection.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------
    # 13. Save Metrics
    # ---------------------------------------------------------
    metrics = {
        "linear_regression_mae": linear_mae,
        "linear_regression_rmse": linear_rmse,
        "linear_regression_r2": linear_r2,
        "ridge_regression_mae": ridge_mae,
        "ridge_regression_rmse": ridge_rmse,
        "ridge_regression_r2": ridge_r2,
        "pca_component_1_variance": pca.explained_variance_ratio_[0],
        "pca_component_2_variance": pca.explained_variance_ratio_[1],
        "pca_total_variance_explained": np.sum(pca.explained_variance_ratio_)
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("model_metrics.csv", index=False)

    print("\n==============================")
    print("Saved Output Files")
    print("==============================")
    print("actual_vs_predicted.png")
    print("pca_projection.png")
    print("linear_regression_coefficients.csv")
    print("model_metrics.csv")


if __name__ == "__main__":
    main()